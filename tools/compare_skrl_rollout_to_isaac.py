#!/usr/bin/env python3
"""Compare a MuJoCo skrl rollout with an IsaacGym synthetic reference JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco-rollout", type=Path, required=True)
    parser.add_argument("--isaac-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def _lift(records: list[dict[str, Any]]) -> dict[str, float | None]:
    positions = [record.get("object_position") for record in records]
    valid = [np.asarray(position, dtype=np.float64) for position in positions if position]
    if not valid:
        return {"initial_z": None, "max_z": None, "actual_lift_m": None}
    z = np.asarray([position[2] for position in valid], dtype=np.float64)
    return {
        "initial_z": float(z[0]),
        "max_z": float(z.max()),
        "actual_lift_m": float(z.max() - z[0]),
    }


def _load_mujoco_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"MuJoCo rollout {path} does not contain a records list")
    return records


def _load_isaac_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Isaac reference {path} does not contain a records list")
    return records


def compare(mujoco_path: Path, isaac_path: Path) -> dict[str, Any]:
    mujoco_records = _load_mujoco_records(mujoco_path)
    isaac_records = _load_isaac_records(isaac_path)
    mujoco_by_frame = {
        int(record["lance_frame"]): record
        for record in mujoco_records
        if record.get("lance_frame") is not None and record.get("object_position")
    }
    isaac_by_frame = {
        int(record["task_current_frame"]): record
        for record in isaac_records
        if record.get("task_current_frame") is not None and record.get("object_position")
    }
    shared_frames = sorted(set(mujoco_by_frame).intersection(isaac_by_frame))
    if not shared_frames:
        raise ValueError("MuJoCo and Isaac records have no shared Lance frames")

    object_deltas = []
    for frame in shared_frames:
        mujoco_pos = np.asarray(mujoco_by_frame[frame]["object_position"], dtype=np.float64)
        isaac_pos = np.asarray(isaac_by_frame[frame]["object_position"], dtype=np.float64)
        object_deltas.append(float(np.linalg.norm(mujoco_pos - isaac_pos)))

    mujoco_errors = [
        float(record["object_position_error_m"])
        for frame in shared_frames
        for record in [mujoco_by_frame[frame]]
        if record.get("object_position_error_m") is not None
    ]
    isaac_errors = [
        float(record["object_position_l2_error_m"])
        for frame in shared_frames
        for record in [isaac_by_frame[frame]]
        if record.get("object_position_l2_error_m") is not None
    ]
    mujoco_hand_errors = [
        float(record["hand_active_l2_error"])
        for frame in shared_frames
        for record in [mujoco_by_frame[frame]]
        if record.get("hand_active_l2_error") is not None
    ]
    isaac_hand_errors = [
        float(record["active_l2_error"])
        for frame in shared_frames
        for record in [isaac_by_frame[frame]]
        if record.get("active_l2_error") is not None
    ]
    target_positions = [
        record.get("target_object_position")
        for frame in shared_frames
        for record in [isaac_by_frame[frame]]
        if record.get("target_object_position")
    ]
    target_lift = None
    if target_positions:
        target_z = np.asarray([position[2] for position in target_positions], dtype=np.float64)
        target_lift = float(target_z.max() - target_z[0])

    return {
        "schema": "dexhandrl.skrl_vs_isaac_reference.v1",
        "mujoco_rollout": str(mujoco_path),
        "isaac_reference": str(isaac_path),
        "shared_lance_frame_range": [shared_frames[0], shared_frames[-1]],
        "shared_frame_count": len(shared_frames),
        "mujoco": {
            "object_position_error_m": _stats(mujoco_errors),
            "hand_active_l2_error": _stats(mujoco_hand_errors),
            "actual_lift": _lift([mujoco_by_frame[frame] for frame in shared_frames]),
        },
        "isaac": {
            "object_position_error_m": _stats(isaac_errors),
            "active_l2_error": _stats(isaac_hand_errors),
            "actual_lift": _lift([isaac_by_frame[frame] for frame in shared_frames]),
            "target_lift_m": target_lift,
        },
        "absolute_object_trajectory_delta_m": _stats(object_deltas),
    }


def main() -> int:
    args = _parser().parse_args()
    result = compare(args.mujoco_rollout, args.isaac_reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
