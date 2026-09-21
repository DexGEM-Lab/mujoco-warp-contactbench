#!/usr/bin/env python3
"""Reuse exact640 clearances and solve mesh-accurate action005 backgrounds."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

import tools.compute_u1_exact800_background_offsets as solver
from tools.export_u1_largepose800 import ACTIONS, EXPECTED_COUNTS

EXTENDED_MAGS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]


def _worker_init() -> None:
    solver.MAGS = EXTENDED_MAGS
    solver._worker_init()


def _action005_rows(dataset):
    for start in range(160, 320, 8):
        indices = list(range(start, min(start + 8, 320)))
        table = dataset.take(
            indices,
            columns=["hands", "objects", "trajectory_metadata", "index"],
        )
        for row in table.to_pylist():
            if row["trajectory_metadata"]["gesture"].split("-")[0] != "005":
                raise ValueError("exact800 action005 order changed")
            yield row


def _summarize(results: list[dict]) -> tuple[dict, dict]:
    offsets: dict[str, dict[str, list[float]]] = {}
    magnitudes = []
    unresolved = []
    object_pairs = 0
    for result in results:
        if result["action"] != "005":
            raise ValueError("solver returned a non-action005 row")
        if result["offsets"]:
            offsets[result["uuid"]] = result["offsets"]
            object_pairs += len(result["offsets"])
            for delta in result["offsets"].values():
                magnitudes.append(float(np.linalg.norm(delta)))
        for name, distance in result["unresolved"]:
            unresolved.append([result["uuid"], name, distance])
    report = {
        "rows": len(results),
        "rows_needing_offset": len(offsets),
        "object_pairs_offset": object_pairs,
        "unresolved_pairs": len(unresolved),
        "unresolved_detail": unresolved,
        "offset_mag_cm": {
            "min": round(min(magnitudes) * 100, 2) if magnitudes else None,
            "median": round(float(np.median(magnitudes)) * 100, 2)
            if magnitudes
            else None,
            "max": round(max(magnitudes) * 100, 2) if magnitudes else None,
        },
    }
    return report, offsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance", type=Path, required=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--source640-offsets", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import lance

    dataset = lance.dataset(str(args.lance), version=args.version)
    if dataset.count_rows() != 800:
        raise ValueError("exact800 row count")
    metadata = dataset.to_table(columns=["index", "trajectory_metadata"]).to_pylist()
    actions = [row["trajectory_metadata"]["gesture"].split("-")[0] for row in metadata]
    if Counter(actions) != Counter(EXPECTED_COUNTS):
        raise ValueError("exact800 action counts")
    if actions != [action for action in ACTIONS for _ in range(160)]:
        raise ValueError("exact800 action order")
    uuids = {row["index"]["uuid"] for row in metadata}
    source = json.loads(args.source640_offsets.read_text())
    source_reports = source["report_by_action"]
    expected_source_actions = set(ACTIONS) - {"005"}
    if set(source_reports) != expected_source_actions:
        raise ValueError("exact640 offset action coverage")
    if not all(
        report["rows"] == 160 and report["unresolved_pairs"] == 0
        for report in source_reports.values()
    ):
        raise ValueError("exact640 has unresolved clearances")
    old_offsets = source.get("per_row_offsets", {})
    if not set(old_offsets) <= uuids:
        raise ValueError("exact640 offsets contain unknown UUIDs")
    solver.ASSET_ROOT = str(args.asset_root.resolve(strict=True))
    solver.os.environ["MANORL_ASSET_MANIFEST"] = str(
        args.asset_manifest.resolve(strict=True)
    )
    workers = max(1, min(26, mp.cpu_count() - 2))
    with mp.Pool(workers, initializer=_worker_init) as pool:
        results = list(
            pool.imap_unordered(
                solver._solve_row,
                _action005_rows(dataset),
                chunksize=4,
            )
        )
    if len(results) != 160:
        raise ValueError("action005 clearance row count")
    action005_report, action005_offsets = _summarize(results)
    if action005_report["unresolved_pairs"]:
        raise ValueError(
            "action005 unresolved clearances: "
            + json.dumps(action005_report["unresolved_detail"][:10])
        )
    if set(old_offsets) & set(action005_offsets):
        raise ValueError("old/new offset UUID collision")
    payload = {
        "lance": args.lance.name,
        "lance_version": args.version,
        "hand_target_m": solver.HAND_TARGET,
        "object_target_m": solver.OBJ_TARGET,
        "canonical": solver.CANON,
        "report_by_action": {
            action: (
                action005_report if action == "005" else source_reports[action]
            )
            for action in ACTIONS
        },
        "per_row_offsets": {**old_offsets, **action005_offsets},
        "extended_resolution": {
            "source_exact640": str(args.source640_offsets.resolve()),
            "action005_candidate_magnitudes_m": EXTENDED_MAGS,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rows": 800,
                "action005": action005_report,
                "offset_rows": len(payload["per_row_offsets"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
