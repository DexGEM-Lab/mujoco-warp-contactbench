#!/usr/bin/env python3
"""Summarize real Lance frame-to-frame target increments for PD calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.dexhandrl.constants import DEFAULT_LANCE_PATH, DEXHAND021PRO_ACTIVE_DOF_NAMES
from sim.dexhandrl.lance_loader import load_trajectory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance", type=Path, default=DEFAULT_LANCE_PATH)
    parser.add_argument("--object", default="cube1")
    parser.add_argument("--action", default="01")
    parser.add_argument("--sequence-index", type=int, default=1)
    parser.add_argument("--uuid", default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dexhandrl_outputs/lance_row22_deltas.json"),
    )
    return parser


def _stats(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("cannot summarize an empty increment array")
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _per_dimension(deltas: np.ndarray, names: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    return {
        name: _stats(deltas[:, index])
        for index, name in enumerate(names)
    }


def analyze(args: argparse.Namespace) -> dict[str, object]:
    trajectory = load_trajectory(
        lance_path=args.lance,
        object_name=args.object,
        action=args.action,
        sequence_index=args.sequence_index,
        uuid=args.uuid,
    )
    start = trajectory.movement_start_frame if args.start_frame is None else int(args.start_frame)
    end = trajectory.movement_end_frame if args.end_frame is None else int(args.end_frame)
    if start < 0 or end <= start or end >= trajectory.frames:
        raise ValueError(
            f"increment range must satisfy 0 <= start < end < {trajectory.frames}; "
            f"got {start}..{end}"
        )

    active = np.asarray(trajectory.hand_dof[start : end + 1], dtype=np.float64)
    active_delta = np.abs(np.diff(active, axis=0))
    base_pos = active_delta[:, :3]
    base_rot = active_delta[:, 3:6]
    active_finger = active_delta[:, 6:]

    result: dict[str, object] = {
        "schema": "dexhandrl.lance_target_deltas.v1",
        "lance_path": str(trajectory.lance_path),
        "object": trajectory.object_name,
        "action": trajectory.action,
        "sequence_index": int(args.sequence_index),
        "row_id": int(trajectory.row_id),
        "uuid": trajectory.uuid,
        "frame_range": {"start": start, "end": end, "increment_count": int(end - start)},
        "active_target_order": list(DEXHAND021PRO_ACTIVE_DOF_NAMES),
        "base_position_m": _stats(np.linalg.norm(base_pos, axis=1)),
        "base_rotation_rad": _stats(np.linalg.norm(base_rot, axis=1)),
        "active_finger_rad": _stats(np.linalg.norm(active_finger, axis=1)),
        "base_position_by_dimension_m": _per_dimension(base_pos, ("tx_world", "ty_world", "tz_world")),
        "base_rotation_by_dimension_rad": _per_dimension(
            base_rot,
            ("euler_rx_world", "euler_ry_world", "euler_rz_world"),
        ),
        "active_finger_by_dimension_rad": _per_dimension(
            active_finger,
            DEXHAND021PRO_ACTIVE_DOF_NAMES[6:],
        ),
    }

    if trajectory.hand_finger_dof_full_raw is not None:
        raw = np.asarray(trajectory.hand_finger_dof_full_raw[start : end + 1], dtype=np.float64)
        raw_delta = np.abs(np.diff(raw, axis=0))
        result["raw_finger_rad"] = _stats(np.linalg.norm(raw_delta, axis=1))
        result["raw_finger_by_dimension_rad"] = _per_dimension(
            raw_delta,
            tuple(f"r_f_jiont_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)),
        )
    else:
        result["raw_finger_rad"] = None
        result["raw_finger_by_dimension_rad"] = None
    return result


def main() -> int:
    args = _parser().parse_args()
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "row_id": result["row_id"], "frame_range": result["frame_range"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
