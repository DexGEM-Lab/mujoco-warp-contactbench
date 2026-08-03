#!/usr/bin/env python3
"""Stream a full v2.2 synthetic Lance dataset into compact replay/visual rows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import lance

from sim.manorl.lance_v2 import (
    SYNTHETIC_LANCE_COMPACT_V1_CONTRACT,
    SYNTHETIC_LANCE_V22_CONTRACT,
    build_compact_row,
    file_sha256,
    write_compact_lance_stream,
)


def _ccd_settings(metadata: Mapping[str, Any]) -> tuple[int, int]:
    runtime = metadata.get("runtime_config")
    environment = runtime.get("environment") if isinstance(runtime, dict) else None
    warp = environment.get("warp_ccd") if isinstance(environment, dict) else None
    if not isinstance(warp, dict):
        raise ValueError("full v2.2 checkpoint metadata lacks Warp CCD settings")
    iterations = warp.get("ccd_iterations", warp.get("warp_ccd_iterations"))
    contacts = warp.get("contacts_per_world", warp.get("warp_ccd_contacts_per_world"))
    if iterations is None or contacts is None:
        raise ValueError(
            "full v2.2 checkpoint metadata has incomplete Warp CCD settings"
        )
    return int(iterations), int(contacts)


def compact_dataset(
    input_path: Path,
    output_path: Path,
    *,
    input_version: int | None = None,
    replace: bool = False,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Project a full dataset with bounded host memory and external metadata."""

    metadata_sidecar = (
        output_path.parent / f"{output_path.name}.checkpoint-metadata.json"
    )
    manifest_path = output_path.parent / f"{output_path.name}.manifest.json"
    if output_path.exists():
        if not replace:
            raise FileExistsError(f"output already exists: {output_path}")
        shutil.rmtree(output_path)
        for stale in (metadata_sidecar, manifest_path):
            stale.unlink(missing_ok=True)
    source = (
        lance.dataset(str(input_path), version=input_version)
        if input_version is not None
        else lance.dataset(str(input_path))
    )
    source_metadata = {
        key.decode(): value.decode()
        for key, value in (source.schema.metadata or {}).items()
    }
    if source_metadata.get("schema_version") != SYNTHETIC_LANCE_V22_CONTRACT:
        raise ValueError("compact input must use the full corrected v2.2 schema")
    source_rows = int(source.count_rows())
    if source_rows < 1:
        raise ValueError("compact input cannot be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    building = output_path.parent / f".{output_path.name}.partial"
    if building.exists():
        raise FileExistsError(f"stale compact building path exists: {building}")

    metadata_by_hash: dict[str, dict[str, Any]] = {}
    source_reader = source.scanner(
        columns=[
            "index",
            "trajectory_metadata",
            "timestamp",
            "hands",
            "objects",
            "provenance",
        ],
        batch_size=batch_size,
    ).to_reader()
    seen = 0

    def rows():
        nonlocal seen
        for batch in source_reader:
            for row in batch.to_pylist():
                provenance = row.get("provenance") or {}
                raw_metadata = provenance.get("checkpoint_metadata_json")
                if not isinstance(raw_metadata, str) or not raw_metadata:
                    raise ValueError("full v2.2 row lacks checkpoint_metadata_json")
                checkpoint_metadata = json.loads(raw_metadata)
                if not isinstance(checkpoint_metadata, dict):
                    raise ValueError(
                        "checkpoint_metadata_json must decode to a mapping"
                    )
                checkpoint_hash = provenance.get("checkpoint_sha256")
                if (
                    not isinstance(checkpoint_hash, str)
                    or len(checkpoint_hash) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in checkpoint_hash
                    )
                ):
                    raise ValueError("full v2.2 row has an invalid checkpoint SHA256")
                metadata_hash = hashlib.sha256(
                    json.dumps(
                        checkpoint_metadata, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
                metadata_by_hash.setdefault(metadata_hash, checkpoint_metadata)
                if checkpoint_hash is None:
                    raise ValueError("full v2.2 row lacks checkpoint_sha256")
                iterations, contacts = _ccd_settings(checkpoint_metadata)
                seen += 1
                if seen % 100 == 0:
                    print(
                        json.dumps(
                            {"event": "compact_projection_progress", "rows": seen},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                yield build_compact_row(
                    row,
                    checkpoint_metadata=checkpoint_metadata,
                    warp_ccd_iterations=iterations,
                    warp_ccd_contacts_per_world=contacts,
                )

    write_compact_lance_stream(rows(), output=building, batch_size=batch_size)
    compact = lance.dataset(str(building), version=1)
    if (
        int(compact.count_rows()) != source_rows
        or compact.schema.metadata.get(b"schema_version")
        != SYNTHETIC_LANCE_COMPACT_V1_CONTRACT.encode()
    ):
        raise RuntimeError("compact projection row count or schema contract failed")

    metadata_partial = metadata_sidecar.with_suffix(
        metadata_sidecar.suffix + ".partial"
    )
    if metadata_sidecar.exists() or metadata_partial.exists():
        raise FileExistsError(f"metadata sidecar already exists: {metadata_sidecar}")
    metadata_partial.write_text(
        json.dumps(
            {
                "schema": "manorl.synthetic_checkpoint_metadata_catalog.v1",
                "source_contract": SYNTHETIC_LANCE_V22_CONTRACT,
                "entries": metadata_by_hash,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    # Publish the catalog before the dataset so a visible dataset never lacks
    # the metadata required to interpret its per-row metadata SHA256 values.
    metadata_partial.replace(metadata_sidecar)
    building.replace(output_path)
    manifest = {
        "schema": SYNTHETIC_LANCE_COMPACT_V1_CONTRACT,
        "output_format": "compact-replay-visual",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "dataset": str(input_path.resolve()),
            "version": source.version,
            "schema": SYNTHETIC_LANCE_V22_CONTRACT,
            "rows": source_rows,
        },
        "output": {
            "dataset": str(output_path.resolve()),
            "version": 1,
            "rows": source_rows,
            "schema": SYNTHETIC_LANCE_COMPACT_V1_CONTRACT,
        },
        "projection": {
            "kept_top_level": [
                "index",
                "trajectory_metadata",
                "timestamp",
                "hands",
                "objects",
                "provenance",
            ],
            "dropped_top_level": ["contact", "reference", "rollout"],
            "checkpoint_metadata_sidecar": str(metadata_sidecar.resolve()),
            "checkpoint_metadata_sha256": sorted(metadata_by_hash),
        },
        "validation": {
            "source_rows_seen": seen,
            "output_rows": int(compact.count_rows()),
            "metadata_sidecar_sha256": file_sha256(metadata_sidecar),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output": str(output_path),
        "manifest": str(manifest_path),
        "rows": source_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-version", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    if args.input_version is not None and args.input_version < 1:
        parser.error("input-version must be positive")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = compact_dataset(
        args.input,
        args.output,
        input_version=args.input_version,
        replace=args.replace,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
