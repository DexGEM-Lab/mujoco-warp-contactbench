#!/usr/bin/env python3
"""Annotate every refined RL Lance row with reference-object motion onset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

import numpy as np

from sim.manorl.reference_motion import (
    REFERENCE_MOTION_ANNOTATION_CONTRACT,
    ReferenceMotionAnnotation,
    ReferenceMotionConfig,
    annotate_rl_episode_row,
)

ANNOTATED_RL_LANCE_CONTRACT = "manorl.refined_rl_motion_annotated.v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _annotation_arrow_type(pa: Any) -> Any:
    return pa.struct(
        [
            pa.field("contract", pa.string(), nullable=False),
            pa.field("start_frame", pa.int64(), nullable=False),
            pa.field("end_frame", pa.int64(), nullable=False),
            pa.field("confidence", pa.string(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("motion_mode", pa.string(), nullable=False),
            pa.field("observed_fps", pa.float64(), nullable=False),
            pa.field("stable_start_frame", pa.int64()),
            pa.field("stable_end_frame", pa.int64()),
            pa.field("candidate_frame", pa.int64()),
            pa.field(
                "onset_translation_from_baseline_m", pa.float64(), nullable=False
            ),
            pa.field(
                "onset_rotation_from_baseline_rad", pa.float64(), nullable=False
            ),
            pa.field(
                "max_translation_from_baseline_m", pa.float64(), nullable=False
            ),
            pa.field(
                "max_rotation_from_baseline_rad", pa.float64(), nullable=False
            ),
        ]
    )


def annotated_schema(source_schema: Any, config: ReferenceMotionConfig) -> Any:
    """Extend only trajectory metadata; preserve every existing field contract."""

    import pyarrow as pa

    metadata_field = source_schema.field("trajectory_metadata")
    if not pa.types.is_struct(metadata_field.type):
        raise ValueError("trajectory_metadata must be a struct")
    existing = {field.name for field in metadata_field.type}
    conflicts = existing & {"trajectory_info", "reference_motion_annotation"}
    if conflicts:
        raise ValueError(
            "source already contains movement annotation fields: "
            + ", ".join(sorted(conflicts))
        )
    object_move = pa.struct(
        [
            pa.field("object_name", pa.string(), nullable=False),
            pa.field("start_frame", pa.int64(), nullable=False),
            pa.field("end_frame", pa.int64(), nullable=False),
        ]
    )
    trajectory_info = pa.struct(
        [pa.field("object_move", pa.list_(object_move), nullable=False)]
    )
    extended_metadata = pa.struct(
        [
            *list(metadata_field.type),
            pa.field("trajectory_info", trajectory_info, nullable=False),
            pa.field(
                "reference_motion_annotation",
                _annotation_arrow_type(pa),
                nullable=False,
            ),
        ]
    )
    fields = [
        (
            pa.field(
                field.name,
                extended_metadata,
                nullable=field.nullable,
                metadata=field.metadata,
            )
            if field.name == "trajectory_metadata"
            else field
        )
        for field in source_schema
    ]
    metadata = dict(source_schema.metadata or {})
    metadata[b"manorl:annotation_contract"] = ANNOTATED_RL_LANCE_CONTRACT.encode(
        "ascii"
    )
    metadata[b"manorl:reference_motion_contract"] = (
        REFERENCE_MOTION_ANNOTATION_CONTRACT.encode("ascii")
    )
    metadata[b"manorl:reference_motion_config"] = _canonical_bytes(
        config.to_dict()
    )
    return pa.schema(fields, metadata=metadata)


def _clean_annotated_row_for_source(
    row: Mapping[str, Any], *, source_metadata_fields: tuple[str, ...]
) -> dict[str, Any]:
    cleaned = dict(row)
    metadata = row.get("trajectory_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("annotated row lacks trajectory_metadata")
    cleaned["trajectory_metadata"] = {
        name: metadata.get(name) for name in source_metadata_fields
    }
    return cleaned


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _validate_output(
    source: Any,
    destination: Any,
    *,
    source_schema: Any,
    batch_size: int,
) -> dict[str, object]:
    """Prove row order and every original value survived annotation exactly."""

    import pyarrow as pa

    source_rows = int(source.count_rows())
    if int(destination.count_rows()) != source_rows:
        raise RuntimeError("annotated dataset row count differs from source")
    source_names = tuple(source_schema.names)
    metadata_fields = tuple(
        field.name for field in source_schema.field("trajectory_metadata").type
    )
    statuses: Counter[str] = Counter()
    uuid_counts: Counter[str] = Counter()
    checked = 0
    for start in range(0, source_rows, batch_size):
        stop = min(source_rows, start + batch_size)
        indices = list(range(start, stop))
        source_table = source.take(indices, columns=list(source_names))
        destination_table = destination.take(indices)
        destination_rows = destination_table.to_pylist()
        reconstructed = pa.Table.from_pylist(
            [
                _clean_annotated_row_for_source(
                    row, source_metadata_fields=metadata_fields
                )
                for row in destination_rows
            ],
            schema=source_schema,
        )
        if not source_table.combine_chunks().equals(reconstructed.combine_chunks()):
            raise RuntimeError(
                f"annotated dataset changed original values in rows [{start}, {stop})"
            )
        for row in destination_rows:
            index = row["index"]
            metadata = row["trajectory_metadata"]
            row_uuid = str(index["uuid"])
            uuid_counts[row_uuid] += 1
            movement = metadata["trajectory_info"]["object_move"]
            annotation = metadata["reference_motion_annotation"]
            if (
                len(movement) != 1
                or movement[0]["object_name"] != index["scene"]
                or movement[0]["start_frame"] != annotation["start_frame"]
                or movement[0]["end_frame"] != annotation["end_frame"]
                or annotation["contract"] != REFERENCE_MOTION_ANNOTATION_CONTRACT
                or not 0
                <= annotation["start_frame"]
                <= annotation["end_frame"]
                < int(metadata["total_frames"])
            ):
                raise RuntimeError(f"invalid movement annotation for UUID {row_uuid}")
            statuses[str(annotation["status"])] += 1
        checked += len(indices)
    if checked != source_rows:
        raise RuntimeError("annotated validation did not account for every source row")
    duplicate_counts = [count for count in uuid_counts.values() if count > 1]
    return {
        "validated_rows": checked,
        "distinct_uuids": len(uuid_counts),
        "duplicate_uuid_values": len(duplicate_counts),
        "duplicate_uuid_extra_rows": sum(count - 1 for count in duplicate_counts),
        "maximum_uuid_multiplicity": max(duplicate_counts, default=1),
        "status_counts": dict(sorted(statuses.items())),
        "original_columns_exact": True,
    }


def annotate_dataset(
    source_path: Path,
    output_path: Path,
    *,
    source_version: int,
    diagnostics_path: Path,
    manifest_path: Path,
    batch_size: int,
    data_storage_version: str,
    config: ReferenceMotionConfig,
) -> Path:
    """Write and validate a new movement-annotated Lance dataset atomically."""

    import lance
    import pyarrow as pa

    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    diagnostics_path = diagnostics_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    for path in (output_path, diagnostics_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to replace annotation artifact: {path}")
    source = lance.dataset(str(source_path), version=source_version)
    if int(source.version) != source_version:
        raise RuntimeError(
            f"source dataset version {source.version} != requested {source_version}"
        )
    source_schema = source.schema
    destination_schema = annotated_schema(source_schema, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary_output = output_path.parent / f".{output_path.name}.{token}.pending"
    temporary_diagnostics = diagnostics_path.with_name(
        f".{diagnostics_path.name}.{token}.pending"
    )
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{token}.pending"
    )

    summary_status: Counter[str] = Counter()
    summary_confidence: Counter[str] = Counter()
    summary_modes: Counter[str] = Counter()
    summary_fps: Counter[int] = Counter()
    starts: list[float] = []
    start_seconds: list[float] = []
    starts_by_pair: dict[str, list[float]] = defaultdict(list)
    row_index = 0

    def batches() -> Iterable[Any]:
        nonlocal row_index
        scanner = source.scanner(batch_size=batch_size, scan_in_order=True)
        with temporary_diagnostics.open("w", encoding="utf-8") as diagnostics:
            for source_batch in scanner.to_batches():
                output_rows: list[dict[str, Any]] = []
                for source_row in source_batch.to_pylist():
                    annotated_row, annotation = annotate_rl_episode_row(
                        source_row, config=config
                    )
                    index = annotated_row["index"]
                    pair = f"{index['scene']}:{str(index['action_code']).zfill(2)}"
                    record = {
                        "row_index": row_index,
                        "uuid": index["uuid"],
                        "seed_uuid": index.get("seed_uuid"),
                        "pair": pair,
                        **annotation.to_dict(),
                    }
                    diagnostics.write(
                        json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                    )
                    summary_status[annotation.status] += 1
                    summary_confidence[annotation.confidence] += 1
                    summary_modes[annotation.motion_mode] += 1
                    summary_fps[round(annotation.observed_fps)] += 1
                    starts.append(float(annotation.start_frame))
                    start_seconds.append(
                        float(annotation.start_frame / annotation.observed_fps)
                    )
                    starts_by_pair[pair].append(
                        float(annotation.start_frame / annotation.observed_fps)
                    )
                    output_rows.append(annotated_row)
                    row_index += 1
                yield pa.RecordBatch.from_pylist(
                    output_rows, schema=destination_schema
                )

    try:
        reader = pa.RecordBatchReader.from_batches(destination_schema, batches())
        lance.write_dataset(
            reader,
            str(temporary_output),
            mode="create",
            data_storage_version=data_storage_version,
            max_rows_per_group=max(1, min(batch_size, 1024)),
        )
        if row_index != int(source.count_rows()):
            raise RuntimeError(
                f"annotator wrote {row_index} of {source.count_rows()} source rows"
            )
        destination = lance.dataset(str(temporary_output), version=1)
        validation = _validate_output(
            source,
            destination,
            source_schema=source_schema,
            batch_size=batch_size,
        )
        manifest: dict[str, object] = {
            "contract": ANNOTATED_RL_LANCE_CONTRACT,
            "annotation_contract": REFERENCE_MOTION_ANNOTATION_CONTRACT,
            "source": {
                "path": str(source_path),
                "version": int(source.version),
                "rows": int(source.count_rows()),
                "schema_sha256": hashlib.sha256(
                    str(source_schema).encode("utf-8")
                ).hexdigest(),
            },
            "output": {
                "path": str(output_path),
                "version": int(destination.version),
                "rows": int(destination.count_rows()),
                "schema_sha256": hashlib.sha256(
                    str(destination.schema).encode("utf-8")
                ).hexdigest(),
                "data_storage_version": data_storage_version,
            },
            "config": config.to_dict(),
            "summary": {
                "status_counts": dict(sorted(summary_status.items())),
                "confidence_counts": dict(sorted(summary_confidence.items())),
                "motion_mode_counts": dict(sorted(summary_modes.items())),
                "observed_fps_counts": {
                    str(key): value for key, value in sorted(summary_fps.items())
                },
                "start_frame_quantiles": _quantiles(starts),
                "start_seconds_quantiles": _quantiles(start_seconds),
                "start_seconds_by_pair": {
                    key: {"rows": len(values), **_quantiles(values)}
                    for key, values in sorted(starts_by_pair.items())
                },
            },
            "validation": validation,
            "diagnostics": str(diagnostics_path),
        }
        manifest["manifest_basis_sha256"] = _digest(manifest)
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_output, output_path)
        os.replace(temporary_diagnostics, diagnostics_path)
        os.replace(temporary_manifest, manifest_path)
        return output_path
    except BaseException:
        failed = temporary_output.with_name(temporary_output.name + ".failed")
        if temporary_output.exists():
            try:
                os.replace(temporary_output, failed)
            except OSError:
                pass
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--data-storage-version", choices=("2.1", "2.2", "stable"), default="2.1"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_version < 1 or args.batch_size < 1:
        raise SystemExit("input-version and batch-size must be positive")
    output = args.output.expanduser().resolve()
    diagnostics = (
        args.diagnostics.expanduser().resolve()
        if args.diagnostics is not None
        else output.with_name(output.name + ".motion_annotations.jsonl")
    )
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output.with_name(output.name + ".motion_annotations.manifest.json")
    )
    result = annotate_dataset(
        args.input,
        output,
        source_version=args.input_version,
        diagnostics_path=diagnostics,
        manifest_path=manifest,
        batch_size=args.batch_size,
        data_storage_version=args.data_storage_version,
        config=ReferenceMotionConfig(),
    )
    values = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(result),
                "diagnostics": str(diagnostics),
                "manifest": str(manifest),
                "rows": values["output"]["rows"],
                "summary": values["summary"],
                "validation": values["validation"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
