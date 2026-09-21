#!/usr/bin/env python3
"""Resolve only the UUID/object pairs exhausted by the initial layout search."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

BASE_MAGS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
EXTENDED_MAGS = [*BASE_MAGS, 0.18, 0.20, 0.25, 0.30]


def _worker_init(asset_manifest: str) -> None:
    os.environ["MANORL_ASSET_MANIFEST"] = asset_manifest
    import tools.compute_u1_exact800_background_offsets as solver

    solver.MAGS = EXTENDED_MAGS
    solver._worker_init()


def _solve(row: dict) -> dict:
    import tools.compute_u1_exact800_background_offsets as solver

    return solver._solve_row(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance", type=Path, required=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")

    import lance

    payload = json.loads(args.offsets.read_text())
    unresolved = [
        (action, uuid, object_name)
        for action, report in payload["report_by_action"].items()
        for uuid, object_name, _distance in report["unresolved_detail"]
    ]
    if not unresolved:
        raise ValueError("initial report has no unresolved pairs")
    unresolved_uuids = [uuid for _action, uuid, _object in unresolved]
    unresolved_object = {uuid: object_name for _action, uuid, object_name in unresolved}
    if len(unresolved_uuids) != len(set(unresolved_uuids)):
        raise ValueError("resolver expects at most one unresolved unit per UUID")

    dataset = lance.dataset(str(args.lance), version=args.version)
    index_rows = dataset.to_table(columns=["index", "trajectory_metadata"]).to_pylist()
    uuid_to_index = {row["index"]["uuid"]: index for index, row in enumerate(index_rows)}
    uuid_to_action = {
        row["index"]["uuid"]: row["trajectory_metadata"]["gesture"].split("-")[0]
        for row in index_rows
    }
    if not all(uuid in uuid_to_index for uuid in unresolved_uuids):
        raise ValueError("unresolved UUID is absent from Lance")
    rows = dataset.take(
        [uuid_to_index[uuid] for uuid in unresolved_uuids],
        columns=["hands", "objects", "trajectory_metadata", "index"],
    ).to_pylist()

    results: dict[str, dict] = {}
    workers = min(len(rows), max(1, mp.cpu_count() - 2))
    with mp.Pool(
        workers,
        initializer=_worker_init,
        initargs=(str(args.asset_manifest),),
    ) as pool:
        for result in pool.imap_unordered(_solve, rows, chunksize=1):
            uuid = result["uuid"]
            if result["unresolved"]:
                raise ValueError(f"extended search still unresolved: {uuid} {result['unresolved']}")
            results[uuid] = result
            print(json.dumps({"resolved": uuid, "offsets": result["offsets"]}), flush=True)
    if set(results) != set(unresolved_uuids):
        raise ValueError("extended result UUID coverage mismatch")

    per_row_offsets = dict(payload.get("per_row_offsets", {}))
    for uuid, result in results.items():
        previous = per_row_offsets.get(uuid, {})
        if any(result["offsets"].get(name) != delta for name, delta in previous.items()):
            raise ValueError(f"extended search changed an earlier offset: {uuid}")
        if unresolved_object[uuid] not in result["offsets"]:
            raise ValueError(f"extended search omitted resolved object: {uuid}")
        if result["offsets"]:
            per_row_offsets[uuid] = result["offsets"]
        else:
            per_row_offsets.pop(uuid, None)

    by_action: dict[str, list[dict[str, list[float]]]] = defaultdict(list)
    for uuid, offsets in per_row_offsets.items():
        by_action[uuid_to_action[uuid]].append(offsets)
    reports: dict[str, dict] = {}
    action_counts: dict[str, int] = defaultdict(int)
    for action in uuid_to_action.values():
        action_counts[action] += 1
    for action in sorted(payload["report_by_action"]):
        offset_rows = by_action[action]
        magnitudes = [
            float(np.linalg.norm(delta))
            for offsets in offset_rows
            for delta in offsets.values()
        ]
        reports[action] = {
            "rows": action_counts[action],
            "rows_needing_offset": len(offset_rows),
            "object_pairs_offset": sum(len(offsets) for offsets in offset_rows),
            "unresolved_pairs": 0,
            "unresolved_detail": [],
            "offset_mag_cm": {
                "min": round(min(magnitudes) * 100, 2) if magnitudes else None,
                "median": round(float(np.median(magnitudes)) * 100, 2) if magnitudes else None,
                "max": round(max(magnitudes) * 100, 2) if magnitudes else None,
            },
        }
    payload["report_by_action"] = reports
    payload["per_row_offsets"] = per_row_offsets
    payload["extended_resolution"] = {
        "source_offsets": args.offsets.name,
        "unresolved_uuid_count": len(unresolved_uuids),
        "directions": 16,
        "magnitudes_m": EXTENDED_MAGS,
        "resolved_uuids": unresolved_uuids,
    }
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({"resolved": len(results), "output": str(args.output)}))


if __name__ == "__main__":
    main()
