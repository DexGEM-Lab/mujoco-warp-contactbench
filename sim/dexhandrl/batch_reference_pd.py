#!/usr/bin/env python3
"""Batch DexHandRL MuJoCo reference-PD parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.dexhandrl.constants import DEFAULT_ISAAC_SOURCE_ROOT, DEFAULT_LANCE_PATH, DEFAULT_OUTPUT_ROOT  # noqa: E402
from sim.dexhandrl.reference_pd_replay import replay  # noqa: E402


def _stats(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _load_isaac_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records")
    if not isinstance(records, list):
        raise RuntimeError(f"IsaacGym metrics file has no records list: {path}")
    return records


def _tracking_record(result: dict[str, Any], sequence_index: int) -> dict[str, Any]:
    tracking = result["object_tracking"]
    return {
        "sequence_index": int(sequence_index),
        "row_id": int(result["row_id"]),
        "uuid": result["uuid"],
        "object_name": result["object_name"],
        "action": result["action"],
        "start_frame": int(result["start_frame"]),
        "end_frame": int(result["end_frame"]),
        "movement_start_frame": int(result["movement_start_frame"]),
        "movement_end_frame": int(result["movement_end_frame"]),
        "actual_lift_m": float(tracking["actual_lift_m"]),
        "target_lift_m": float(tracking["target_lift_m"]),
        "max_error_m": float(tracking["max_error_m"]),
        "mean_error_m": float(tracking["mean_error_m"]),
        "first_error_gt_10cm_step": int(tracking["first_error_gt_10cm_step"]),
        "hand_object_contact_frames": int(result["hand_object_contact_frames"]),
        "hand_object_contact_count_max": int(result["hand_object_contact_count_max"]),
    }


def _add_isaac_comparison(
    record: dict[str, Any],
    isaac_records: list[dict[str, Any]],
    sequence_index: int,
) -> None:
    if sequence_index >= len(isaac_records):
        return
    isaac = isaac_records[sequence_index]
    record["isaacgym"] = {
        "actual_lift_m": float(isaac["actual_lift_m"]),
        "target_lift_m": float(isaac["target_lift_m"]),
        "max_error_m": float(isaac["max_error_m"]),
        "mean_error_m": float(isaac["mean_error_m"]),
        "mean_error_move_m": float(isaac["mean_error_move_m"]),
        "first_error_gt_10cm_step": int(isaac["first_error_gt_10cm_step"]),
    }
    record["delta_vs_isaacgym"] = {
        "actual_lift_m": record["actual_lift_m"] - float(isaac["actual_lift_m"]),
        "target_lift_m": record["target_lift_m"] - float(isaac["target_lift_m"]),
        "max_error_m": record["max_error_m"] - float(isaac["max_error_m"]),
        "mean_error_m": record["mean_error_m"] - float(isaac["mean_error_m"]),
        "first_error_gt_10cm_step": (
            record["first_error_gt_10cm_step"] - int(isaac["first_error_gt_10cm_step"])
        ),
    }


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    isaac_records = _load_isaac_records(args.isaacgym_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    sequence_indices = range(args.sequence_start, args.sequence_start + args.max_sequences)
    for sequence_index in sequence_indices:
        per_sequence_output = args.output_dir / f"{args.object}_{args.action}_seq{sequence_index:04d}.json"
        scene_copy = args.output_dir / f"{args.object}_{args.action}_seq{sequence_index:04d}_scene.xml"
        replay_args = argparse.Namespace(
            lance=args.lance,
            object=args.object,
            action=args.action,
            sequence_index=sequence_index,
            uuid=None,
            device=args.device,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            pre_contact_frames=args.pre_contact_frames,
            object_reset_z_offset=args.object_reset_z_offset,
            dt=args.dt,
            substeps=args.substeps,
            settle_frames=args.settle_frames,
            reference_joint_source=args.reference_joint_source,
            naconmax=args.naconmax,
            njmax=args.njmax,
            progress_interval=args.progress_interval,
            isaac_source_root=args.isaac_source_root,
            output=per_sequence_output,
            scene_copy=scene_copy,
            keep_trajectory=args.keep_trajectory,
        )
        print(
            f"[batch] sequence_index={sequence_index} output={per_sequence_output}",
            flush=True,
        )
        result = replay(replay_args)
        record = _tracking_record(result, sequence_index=sequence_index)
        _add_isaac_comparison(record, isaac_records, sequence_index=sequence_index)
        records.append(record)

    summary = {
        "schema": "dexhandrl.reference_pd_batch.v1",
        "backend": "mjx_warp",
        "device": args.device,
        "lance_path": str(args.lance),
        "object": args.object,
        "action": args.action,
        "sequence_start": int(args.sequence_start),
        "max_sequences": int(args.max_sequences),
        "max_frames": int(args.max_frames),
        "elapsed_s": time.perf_counter() - started,
        "control": {
            "reference_joint_source": args.reference_joint_source,
            "dt": args.dt,
            "substeps": args.substeps,
            "settle_frames": args.settle_frames,
            "pre_contact_frames": args.pre_contact_frames,
            "object_reset_z_offset": args.object_reset_z_offset,
        },
        "counts": {
            "target_lifted_5cm": sum(record["target_lift_m"] > 0.05 for record in records),
            "actual_lifted_5cm": sum(record["actual_lift_m"] > 0.05 for record in records),
            "actual_lifted_10cm": sum(record["actual_lift_m"] > 0.10 for record in records),
            "max_error_lt_10cm": sum(record["max_error_m"] < 0.10 for record in records),
            "had_error_gt_10cm": sum(record["first_error_gt_10cm_step"] >= 0 for record in records),
        },
        "stats": {
            "actual_lift_m": _stats([record["actual_lift_m"] for record in records]),
            "target_lift_m": _stats([record["target_lift_m"] for record in records]),
            "max_error_m": _stats([record["max_error_m"] for record in records]),
            "mean_error_m": _stats([record["mean_error_m"] for record in records]),
        },
        "records": records,
    }
    if isaac_records:
        summary["delta_vs_isaacgym_stats"] = {
            key: _stats([record["delta_vs_isaacgym"][key] for record in records])
            for key in (
                "actual_lift_m",
                "target_lift_m",
                "max_error_m",
                "mean_error_m",
                "first_error_gt_10cm_step",
            )
            if "delta_vs_isaacgym" in records[0]
        }

    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run batch MuJoCo reference-PD checks.")
    parser.add_argument("--lance", type=Path, default=DEFAULT_LANCE_PATH)
    parser.add_argument("--object", default="cube1")
    parser.add_argument("--action", default="01")
    parser.add_argument("--sequence-start", type=int, default=0)
    parser.add_argument("--max-sequences", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--pre-contact-frames", type=int, default=200)
    parser.add_argument("--object-reset-z-offset", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--settle-frames", type=int, default=0)
    parser.add_argument(
        "--reference-joint-source",
        choices=("active", "full_raw"),
        default="active",
    )
    parser.add_argument("--naconmax", type=int, default=8192)
    parser.add_argument("--njmax", type=int, default=8192)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--isaac-source-root", type=Path, default=DEFAULT_ISAAC_SOURCE_ROOT)
    parser.add_argument("--isaacgym-metrics", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "dexhandrl_reference_pd_batch",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "dexhandrl_reference_pd_batch_summary.json",
    )
    parser.add_argument("--keep-trajectory", action="store_true")
    args = parser.parse_args()
    summary = run_batch(args)
    print(
        json.dumps(
            {
                "summary": str(args.summary),
                "counts": summary["counts"],
                "stats": summary["stats"],
                "delta_vs_isaacgym_stats": summary.get("delta_vs_isaacgym_stats"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
