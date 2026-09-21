#!/usr/bin/env python3
"""Build the exact640 per-UUID nine-object visualization layout."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from tools.compute_u1_exact800_background_offsets import (
    ALL_NAMES,
    CANON,
    HAND_TARGET,
    OBJ_TARGET,
)

ACTIONS = ("003", "006", "007", "009")
EXPECTED_COUNTS = {action: 160 for action in ACTIONS}
CONTRACT = "manorl.u1-largepose640.per-trajectory-visual-layout.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance", type=Path, required=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import lance

    dataset = lance.dataset(str(args.lance), version=args.version)
    if dataset.count_rows() != 640:
        raise ValueError(f"expected 640 rows, found {dataset.count_rows()}")
    offsets_payload = json.loads(args.offsets.read_text())
    per_row_offsets = offsets_payload.get("per_row_offsets", {})
    per_row_omit = offsets_payload.get("per_row_omit", {})

    entries: list[dict] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    row_index = 0
    for batch in dataset.to_batches(
        batch_size=32,
        columns=["index", "trajectory_metadata", "objects"],
    ):
        for row in batch.to_pylist():
            metadata = row["trajectory_metadata"]
            action = metadata["gesture"].split("-")[0]
            if action not in ACTIONS:
                raise ValueError(f"unexpected action {action}")
            uuid = row["index"]["uuid"]
            if uuid in seen:
                raise ValueError(f"duplicate UUID {uuid}")
            seen.add(uuid)
            counts[action] += 1

            active_names = list(metadata["object_names"])
            if len(active_names) != len(row["objects"]):
                raise ValueError(f"object-name mismatch for {uuid}")
            row_offsets = per_row_offsets.get(uuid, {})
            omitted = per_row_omit.get(uuid, [])
            objects: dict[str, dict] = {}
            for name, obj in zip(active_names, row["objects"]):
                objects[name] = {
                    "role": "active",
                    "pos": [round(float(value), 7) for value in obj["pos"][0]],
                    "rot_aa": [round(float(value), 7) for value in obj["rot_aa"][0]],
                    "pose_source": "lance_frame0",
                }
            for name in ALL_NAMES:
                if name in active_names or name in omitted:
                    continue
                delta = row_offsets.get(name)
                pos = np.asarray(CANON[name], dtype=float)
                if delta is not None:
                    pos[:2] += np.asarray(delta, dtype=float)
                objects[name] = {
                    "role": "background",
                    "pos": [round(float(value), 7) for value in pos],
                    "offset_applied": delta,
                    "pose_source": (
                        "canonical_layout_plus_offset" if delta else "canonical_layout"
                    ),
                }
            expected_names = set(ALL_NAMES) - set(omitted)
            if set(objects) != expected_names:
                raise ValueError(f"incomplete visualization objects for {uuid}")
            entries.append(
                {
                    "row_index": row_index,
                    "uuid": uuid,
                    "seed_uuid": row["index"]["seed_uuid"],
                    "action": action,
                    "gesture": metadata["gesture"],
                    "active_objects": active_names,
                    "objects": objects,
                }
            )
            row_index += 1

    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"action counts mismatch: {dict(counts)}")
    unknown_offsets = set(per_row_offsets) - seen
    if unknown_offsets:
        raise ValueError(f"offsets contain unknown UUIDs: {len(unknown_offsets)}")

    payload = {
        "contract": CONTRACT,
        "description": (
            "Visualization-only nine-object layout for the exact640 strict-C1 U1 "
            "dataset. Active poses come from Lance frame0; background poses come "
            "from the canonical kitchen layout plus mesh-clearance offsets. "
            "Background objects are not part of recorded physics."
        ),
        "lance": args.lance.name,
        "lance_version": args.version,
        "total_trajectories": row_index,
        "counts_by_action": dict(sorted(counts.items())),
        "canonical_positions": CANON,
        "hand_clearance_m": HAND_TARGET,
        "object_clearance_m": OBJ_TARGET,
        "per_row_omit": per_row_omit,
        "trajectories": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": row_index, "counts": dict(counts), "output": str(args.output)}))


if __name__ == "__main__":
    main()
