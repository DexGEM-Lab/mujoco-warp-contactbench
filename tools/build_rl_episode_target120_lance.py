#!/usr/bin/env python3
"""Build an immutable 120 Hz refined RL Lance with measured state and actuator target.

Every annotated row is joined to its provenance-pinned source RL row. The join
must reproduce UUID, timestamp, measured hand state and object pose exactly
before `urdf_dof_target` is accepted. State, target and object pose are then
resampled together on the exact timestamp duration to 120 Hz; object motion is
re-annotated on that output clock.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from sim.manorl.reference_motion import ReferenceMotionConfig, detect_reference_motion
from sim.manorl.trajectory import (
    JOINT_DOF,
    RL_EPISODE_RESAMPLING_ID,
    RL_EPISODE_TARGET120_CONTRACT,
)

TARGET_FPS = 120
TARGET120_LANCE_CONTRACT = "manorl.refined_rl_episode_state_target_120hz.lance.v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _schema(source_schema: Any, config: ReferenceMotionConfig) -> Any:
    import pyarrow as pa

    metadata_field = source_schema.field("trajectory_metadata")
    metadata_names = {field.name for field in metadata_field.type}
    if "state_target_contract" in metadata_names:
        raise ValueError("source already has state_target_contract")
    metadata_type = pa.struct(
        [
            *list(metadata_field.type),
            pa.field("state_target_contract", pa.string(), nullable=False),
        ]
    )

    hands_field = source_schema.field("hands")
    if not pa.types.is_list(hands_field.type) or not pa.types.is_struct(
        hands_field.type.value_type
    ):
        raise ValueError("hands must be list<struct>")
    hand_type = hands_field.type.value_type
    if "urdf_dof_target" in {field.name for field in hand_type}:
        raise ValueError("source already has urdf_dof_target")
    dof_type = hand_type.field("urdf_dof").type
    target_hand_type = pa.struct(
        [*list(hand_type), pa.field("urdf_dof_target", dof_type, nullable=False)]
    )
    target_hands_type = pa.list_(target_hand_type)

    fields = []
    for field in source_schema:
        if field.name == "trajectory_metadata":
            fields.append(
                pa.field(
                    field.name, metadata_type, nullable=field.nullable,
                    metadata=field.metadata,
                )
            )
        elif field.name == "hands":
            fields.append(
                pa.field(
                    field.name, target_hands_type, nullable=field.nullable,
                    metadata=field.metadata,
                )
            )
        else:
            fields.append(field)
    metadata = dict(source_schema.metadata or {})
    metadata[b"manorl:target120_contract"] = TARGET120_LANCE_CONTRACT.encode("ascii")
    metadata[b"manorl:state_target_contract"] = RL_EPISODE_TARGET120_CONTRACT.encode("ascii")
    metadata[b"manorl:reference_resampling"] = RL_EPISODE_RESAMPLING_ID.encode("ascii")
    metadata[b"manorl:reference_motion_config"] = _canonical_bytes(config.to_dict())
    return pa.schema(fields, metadata=metadata)


def _interpolate_hand(values: np.ndarray, source_times: np.ndarray, query: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    if values.shape[1] != JOINT_DOF:
        raise ValueError("hand track must have 28 DoF")
    values[:, 3:6] = np.unwrap(values[:, 3:6], axis=0, period=2.0 * np.pi)
    return np.stack(
        [np.interp(query, source_times, values[:, index]) for index in range(JOINT_DOF)],
        axis=1,
    )


def _resample(
    timestamps: np.ndarray,
    state_q: np.ndarray,
    target_q: np.ndarray,
    object_pos: np.ndarray,
    object_rotvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    source_times = timestamps - timestamps[0]
    if np.any(np.diff(source_times) <= 0.0) or source_times[-1] <= 0.0:
        raise ValueError("source timestamps must be strictly increasing")
    control_dt = 1.0 / TARGET_FPS
    intervals = int(np.ceil(source_times[-1] / control_dt - 1e-12))
    control_times = np.arange(intervals + 1, dtype=np.float64) * control_dt
    query = np.minimum(control_times, source_times[-1])
    state_120 = _interpolate_hand(state_q, source_times, query)
    target_120 = _interpolate_hand(target_q, source_times, query)
    pos_120 = np.stack(
        [np.interp(query, source_times, object_pos[:, index]) for index in range(3)],
        axis=1,
    )
    quat_120 = Slerp(
        source_times, Rotation.from_rotvec(object_rotvec)
    )(query).as_quat()
    quat_120 /= np.linalg.norm(quat_120, axis=1, keepdims=True)
    rotvec_120 = Rotation.from_quat(quat_120).as_rotvec()
    return control_times, state_120, target_120, pos_120, rotvec_120


def _matching_hand(source_row: Mapping[str, Any], annotated_state: np.ndarray) -> Mapping[str, Any]:
    hands = source_row.get("hands")
    if not isinstance(hands, list):
        raise ValueError("source hands are missing")
    candidates = []
    for hand in hands:
        if not isinstance(hand, Mapping):
            continue
        state = np.asarray(hand.get("urdf_dof", ()), dtype=np.float64)
        if state.shape == annotated_state.shape and np.array_equal(state, annotated_state):
            candidates.append(hand)
    if len(candidates) != 1:
        raise ValueError(f"source row resolved {len(candidates)} measured hand tracks")
    return candidates[0]


def _matching_object(source_row: Mapping[str, Any], annotated: Mapping[str, Any]) -> Mapping[str, Any]:
    objects = source_row.get("objects")
    if not isinstance(objects, list):
        raise ValueError("source objects are missing")
    pos = np.asarray(annotated.get("pos", ()), dtype=np.float64)
    rot = np.asarray(annotated.get("rot_aa", ()), dtype=np.float64)
    candidates = []
    for value in objects:
        if not isinstance(value, Mapping):
            continue
        if np.array_equal(np.asarray(value.get("pos", ())), pos) and np.array_equal(
            np.asarray(value.get("rot_aa", ())), rot
        ):
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError(f"source row resolved {len(candidates)} object tracks")
    return candidates[0]


def _source_rows_for_batch(rows: list[dict[str, Any]], cache: dict[tuple[str, int], Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for output_index, row in enumerate(rows):
        provenance = row["provenance"]
        key = (
            str(provenance["source_rl_lance_path"]),
            int(provenance["source_rl_version"]),
        )
        grouped.setdefault(key, []).append(
            (output_index, int(provenance["source_rl_row"]))
        )
    resolved: list[dict[str, Any] | None] = [None] * len(rows)
    import lance

    for key, requests in grouped.items():
        dataset = cache.get(key)
        if dataset is None:
            dataset = lance.dataset(key[0], version=key[1])
            cache[key] = dataset
        indices = [source_index for _, source_index in requests]
        values = dataset.take(
            indices, columns=["index", "timestamp", "hands", "objects"]
        ).to_pylist()
        if len(values) != len(requests):
            raise RuntimeError("source Lance take omitted rows")
        for (output_index, _), value in zip(requests, values, strict=True):
            resolved[output_index] = value
    if any(value is None for value in resolved):
        raise RuntimeError("source join did not resolve every annotated row")
    return [value for value in resolved if value is not None]


def build(
    source_path: Path,
    output_path: Path,
    *,
    source_version: int,
    diagnostics_path: Path,
    manifest_path: Path,
    batch_size: int,
    data_storage_version: str,
) -> Path:
    import lance
    import pyarrow as pa

    for path in (output_path, diagnostics_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to replace artifact: {path}")
    source = lance.dataset(str(source_path), version=source_version)
    schema = _schema(source.schema, ReferenceMotionConfig())
    token = uuid4().hex
    temporary_output = output_path.parent / f".{output_path.name}.{token}.pending"
    temporary_diagnostics = diagnostics_path.with_name(
        f".{diagnostics_path.name}.{token}.pending"
    )
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{token}.pending"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_cache: dict[tuple[str, int], Any] = {}
    row_index = 0
    statuses: Counter[str] = Counter()
    source_fps: Counter[int] = Counter()
    output_frames = 0
    state_target_mae: list[float] = []

    def batches() -> Iterable[Any]:
        nonlocal row_index, output_frames
        scanner = source.scanner(batch_size=batch_size, scan_in_order=True)
        with temporary_diagnostics.open("w", encoding="utf-8") as diagnostics:
            for batch in scanner.to_batches():
                annotated_rows = batch.to_pylist()
                source_rows = _source_rows_for_batch(annotated_rows, source_cache)
                output_rows = []
                for annotated, source_row in zip(
                    annotated_rows, source_rows, strict=True
                ):
                    provenance = annotated["provenance"]
                    if str(source_row["index"]["uuid"]) != str(
                        provenance["source_rl_uuid"]
                    ) or str(source_row["index"]["uuid"]) != str(
                        annotated["index"]["uuid"]
                    ):
                        raise ValueError(f"row {row_index} source UUID mismatch")
                    annotated_timestamps = np.asarray(
                        annotated["timestamp"], dtype=np.float64
                    )
                    source_timestamps = np.asarray(
                        source_row["timestamp"], dtype=np.float64
                    )
                    if not np.array_equal(annotated_timestamps, source_timestamps):
                        raise ValueError(f"row {row_index} source timestamp mismatch")
                    if len(annotated["hands"]) != 1 or len(annotated["objects"]) != 1:
                        raise ValueError(f"row {row_index} is not one-hand/one-object")
                    annotated_hand = annotated["hands"][0]
                    state_q = np.asarray(
                        annotated_hand["urdf_dof"], dtype=np.float64
                    )
                    source_hand = _matching_hand(source_row, state_q)
                    target_q = np.asarray(
                        source_hand.get("urdf_dof_target", ()), dtype=np.float64
                    )
                    if target_q.shape != state_q.shape or not np.all(
                        np.isfinite(target_q)
                    ):
                        raise ValueError(f"row {row_index} target track is invalid")
                    annotated_object = annotated["objects"][0]
                    _matching_object(source_row, annotated_object)
                    object_pos = np.asarray(
                        annotated_object["pos"], dtype=np.float64
                    )
                    object_rot = np.asarray(
                        annotated_object["rot_aa"], dtype=np.float64
                    )
                    times, state, target, pos, rot = _resample(
                        annotated_timestamps, state_q, target_q,
                        object_pos, object_rot,
                    )
                    annotation = detect_reference_motion(
                        pos, rot, times, config=ReferenceMotionConfig()
                    )
                    metadata = dict(annotated["trajectory_metadata"])
                    metadata.update(
                        data_fps=TARGET_FPS,
                        total_frames=len(times),
                        trajectory_info={
                            "object_move": [
                                {
                                    "object_name": str(annotated["index"]["scene"]),
                                    "start_frame": annotation.start_frame,
                                    "end_frame": annotation.end_frame,
                                }
                            ]
                        },
                        reference_motion_annotation=annotation.to_dict(),
                        state_target_contract=RL_EPISODE_TARGET120_CONTRACT,
                    )
                    output_hand = dict(annotated_hand)
                    output_hand["urdf_dof"] = state.astype(np.float32).tolist()
                    output_hand["urdf_dof_target"] = target.astype(np.float32).tolist()
                    output_object = dict(annotated_object)
                    output_object["pos"] = pos.astype(np.float32).tolist()
                    output_object["rot_aa"] = rot.astype(np.float32).tolist()
                    output_row = dict(annotated)
                    output_row.update(
                        trajectory_metadata=metadata,
                        timestamp=times.tolist(),
                        hands=[output_hand],
                        objects=[output_object],
                    )
                    output_rows.append(output_row)
                    observed_fps = 1.0 / float(np.median(np.diff(source_timestamps)))
                    record = {
                        "row_index": row_index,
                        "uuid": annotated["index"]["uuid"],
                        "pair": f"{annotated['index']['scene']}:{str(annotated['index']['action_code']).zfill(2)}",
                        "source_path": provenance["source_rl_lance_path"],
                        "source_version": int(provenance["source_rl_version"]),
                        "source_row": int(provenance["source_rl_row"]),
                        "source_frames": len(source_timestamps),
                        "source_observed_fps": observed_fps,
                        "output_frames": len(times),
                        "state_target_mae": float(np.mean(np.abs(state - target))),
                        "state_target_max_abs": float(np.max(np.abs(state - target))),
                        "motion": annotation.to_dict(),
                    }
                    diagnostics.write(
                        json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                    )
                    statuses[annotation.status] += 1
                    source_fps[round(observed_fps)] += 1
                    output_frames += len(times)
                    state_target_mae.append(record["state_target_mae"])
                    row_index += 1
                yield pa.RecordBatch.from_pylist(output_rows, schema=schema)

    try:
        reader = pa.RecordBatchReader.from_batches(schema, batches())
        lance.write_dataset(
            reader, str(temporary_output), mode="create",
            data_storage_version=data_storage_version,
            max_rows_per_group=max(1, min(batch_size, 256)),
        )
        expected_rows = int(source.count_rows())
        if row_index != expected_rows:
            raise RuntimeError(f"wrote {row_index} of {expected_rows} rows")
        destination = lance.dataset(str(temporary_output), version=1)
        if int(destination.count_rows()) != expected_rows:
            raise RuntimeError("target120 output row count mismatch")
        # Validate identities, dual tracks and clock across every output row.
        checked = 0
        for batch in destination.scanner(
            columns=["index", "trajectory_metadata", "timestamp", "hands", "provenance"],
            batch_size=batch_size, scan_in_order=True,
        ).to_batches():
            for row in batch.to_pylist():
                metadata = row["trajectory_metadata"]
                timestamps = np.asarray(row["timestamp"], dtype=np.float64)
                hand = row["hands"][0]
                state = np.asarray(hand["urdf_dof"], dtype=np.float64)
                target = np.asarray(hand["urdf_dof_target"], dtype=np.float64)
                frames = int(metadata["total_frames"])
                if (
                    metadata["data_fps"] != TARGET_FPS
                    or metadata["state_target_contract"] != RL_EPISODE_TARGET120_CONTRACT
                    or timestamps.shape != (frames,)
                    or state.shape != (frames, JOINT_DOF)
                    or target.shape != (frames, JOINT_DOF)
                    or not np.allclose(np.diff(timestamps), 1.0 / TARGET_FPS, atol=1e-12, rtol=0)
                    or row["index"]["uuid"] != row["provenance"]["source_rl_uuid"]
                ):
                    raise RuntimeError(f"invalid target120 output row {checked}")
                checked += 1
        manifest = {
            "contract": TARGET120_LANCE_CONTRACT,
            "state_target_contract": RL_EPISODE_TARGET120_CONTRACT,
            "resampling_contract": RL_EPISODE_RESAMPLING_ID,
            "source": {
                "path": str(source_path),
                "version": int(source.version),
                "rows": expected_rows,
                "schema_sha256": hashlib.sha256(str(source.schema).encode()).hexdigest(),
            },
            "output": {
                "path": str(output_path),
                "version": int(destination.version),
                "rows": int(destination.count_rows()),
                "schema_sha256": hashlib.sha256(str(destination.schema).encode()).hexdigest(),
                "total_frames": output_frames,
                "data_storage_version": data_storage_version,
            },
            "validation": {
                "validated_rows": checked,
                "all_sources_joined_by_pinned_provenance": True,
                "source_state_and_object_exact_before_resampling": True,
                "output_clock_hz": TARGET_FPS,
                "dual_tracks_present": True,
            },
            "summary": {
                "motion_status_counts": dict(sorted(statuses.items())),
                "source_observed_fps_counts": {
                    str(key): value for key, value in sorted(source_fps.items())
                },
                "state_target_mae_mean": float(np.mean(state_target_mae)),
                "state_target_mae_max": float(np.max(state_target_mae)),
            },
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--data-storage-version", choices=("2.1", "2.2", "stable"), default="2.1"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    diagnostics = (
        args.diagnostics.expanduser().resolve()
        if args.diagnostics is not None
        else output.with_name(output.name + ".diagnostics.jsonl")
    )
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output.with_name(output.name + ".manifest.json")
    )
    result = build(
        args.input.expanduser().resolve(), output,
        source_version=args.input_version,
        diagnostics_path=diagnostics,
        manifest_path=manifest,
        batch_size=args.batch_size,
        data_storage_version=args.data_storage_version,
    )
    values = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps({"output": str(result), **values["validation"], "summary": values["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
