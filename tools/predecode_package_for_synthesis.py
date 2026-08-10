#!/usr/bin/env python3
"""Build a predecoded synthesis bundle (pkl + manifest) from an MTP package.

The long-lived MuJoCo trainer and the synthetic exporter must never decode
Lance on hosts with native Lance/PyArrow instability. This tool runs on a
Lance-healthy host (the coordinator workstation), loads reference trajectories
from a content-addressed MTP package (JSON + mmap NPY, no Lance), reads only
the two lightweight lineage columns from the pinned Lance source, and writes a
predecoded bundle that the exporter consumes through ``--predecoded-manifest``
with zero Lance imports on the target host.

Output layout::

    <output>/
        manifest.json          dataset_path/dataset_version/hand_side/valid_records
        <identity>.pkl         pickled ReferenceTrajectory per identity

Usage::

    python tools/predecode_package_for_synthesis.py \
        --package <mtp-dir> \
        --output <predecoded-dir> \
        --pairs banana:01,cube1:02,... \
        [--dataset-path <explicit source path>] [--dataset-version <int>]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from sim.manorl.trajectory import ObjectActionPair, parse_trajectory_selector
from sim.manorl.trajectory_package import file_sha256, load_trajectory_package

DATASET_PATH_DEFAULT = (
    "/mnt/nas-222-projects/mocap_v2/lance_datasets/human_p1_guangguan/"
    "human_p1_guangguan_clean.lance"
)


def _parse_pairs(raw: str) -> list[ObjectActionPair]:
    parsed = parse_trajectory_selector(raw)
    if not parsed:
        raise SystemExit("--pairs must name at least one object:action pair")
    return list(parsed)


def _read_lineage(
    dataset_path: str,
    dataset_version: int,
    row_indices: list[int],
) -> dict[int, tuple[dict[str, object], dict[str, object]]]:
    """Read only the two lineage columns from the pinned Lance source."""

    import lance

    dataset = lance.dataset(dataset_path, version=dataset_version)
    rows = dataset.take(row_indices, columns=["index", "trajectory_metadata"]).to_pylist()
    if len(rows) != len(row_indices):
        raise RuntimeError("source Lance did not return every requested lineage row")
    return {
        row_index: (
            dict(row.get("index") or {}),
            dict(row.get("trajectory_metadata") or {}),
        )
        for row_index, row in zip(row_indices, rows, strict=True)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="MTP package directory")
    parser.add_argument("--output", type=Path, required=True, help="predecoded bundle directory")
    parser.add_argument("--pairs", required=True, help="comma-separated object:action pairs")
    parser.add_argument("--dataset-path", type=Path, default=Path(DATASET_PATH_DEFAULT))
    parser.add_argument("--dataset-version", type=int, default=295)
    args = parser.parse_args(argv)

    package = args.package.expanduser().resolve()
    if not package.is_dir():
        raise SystemExit(f"package directory is absent: {package}")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    requested = _parse_pairs(args.pairs)
    requested_canonical = {pair.canonical for pair in requested}
    catalog = load_trajectory_package(package, verify_hashes=True)
    package_manifest = catalog.manifest
    trajectory_package = {
        "schema": str(package_manifest.get("schema") or ""),
        "package_digest": str(package_manifest.get("package_digest") or ""),
        "manifest_sha256": catalog.manifest_sha256,
        "catalog_digest": str(package_manifest.get("catalog_digest") or ""),
    }
    if any(not value for value in trajectory_package.values()):
        raise SystemExit("package manifest omits trajectory-package identity fields")


    def pair_of(item: object) -> ObjectActionPair:
        fields = item.identity.identity.split("_")  # type: ignore[attr-defined]
        if len(fields) != 3:
            raise SystemExit(f"unexpected identity format: {item.identity.identity!r}")
        return ObjectActionPair(fields[0], fields[1])

    selected = [
        item for item in catalog.trajectories if pair_of(item).canonical in requested_canonical
    ]
    missing = sorted(
        requested_canonical - {pair_of(item).canonical for item in selected}
    )
    if missing:
        raise SystemExit(f"package omits requested pairs: {missing}")

    row_indices = sorted({item.identity.row_index for item in selected})
    lineage = _read_lineage(
        str(args.dataset_path.resolve()), args.dataset_version, row_indices
    )
    records: list[dict[str, object]] = []
    total_bytes = 0
    for item in sorted(selected, key=lambda t: t.identity.identity):
        identity = item.identity.identity
        pair = pair_of(item).canonical
        path = output / f"{identity}.pkl"
        with path.open("wb") as stream:
            pickle.dump(item, stream, protocol=pickle.HIGHEST_PROTOCOL)
        total_bytes += path.stat().st_size
        index, metadata = lineage[item.identity.row_index]
        records.append(
            {
                "pair": pair,
                "identity": identity,
                "pickle_sha256": file_sha256(path),
                "row_index": item.identity.row_index,
                "source_index": index,
                "trajectory_metadata": metadata,
            }
        )
    records.sort(key=lambda record: str(record["identity"]))
    manifest = {
        "dataset_path": str(args.dataset_path.resolve()),
        "dataset_version": int(args.dataset_version),
        "hand_side": "right",
        "trajectory_count": len(records),
        "trajectory_package": trajectory_package,
        "valid_records": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "trajectories": len(records),
                "pairs": sorted(requested_canonical),
                "pkl_bytes": total_bytes,
                "manifest_sha256": file_sha256(manifest_path),
                "dataset_path": str(args.dataset_path.resolve()),
                "dataset_version": args.dataset_version,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
