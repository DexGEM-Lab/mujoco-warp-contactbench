"""CLI for the accepted ManoRL cube1 reference replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.contracts import (
    EFFORT,
    FLOOR_TOP_Z,
    JOINT_ARMATURE,
    JOINT_FRICTIONLOSS,
    JOINT_NAMES,
    OBJECT_CLEARANCE,
    SOURCE_PHYSX_KD,
    SOURCE_PHYSX_KP,
    ServoConfig,
    WRIST_DAMPRATIO_GRID,
    WRIST_KP_GRID,
)
from sim.manorl.mjx_sim import run_reference_replay
from sim.manorl.trajectory import load_reference_trajectory


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _rmse_norm(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(error), axis=-1))))


def _orientation_error(
    actual_xyzw: np.ndarray, reference_xyzw: np.ndarray
) -> np.ndarray:
    relative = Rotation.from_quat(reference_xyzw).inv() * Rotation.from_quat(actual_xyzw)
    return relative.magnitude()


def summarize_trace(
    trajectory: Any,
    config: Any,
    trace: dict[str, np.ndarray],
    *,
    trace_path: Path,
) -> dict[str, Any]:
    translation_error = trace["hand_qpos"][:, :3] - trace["hand_reference"][:, :3]
    angular_error = (
        trace["hand_qpos"][:, 3:6]
        - trace["hand_reference"][:, 3:6]
        + np.pi
    ) % (2.0 * np.pi) - np.pi
    object_position_error = trace["object_pos"] - trace["object_reference_pos"]
    object_orientation_error = _orientation_error(
        trace["object_quat_xyzw"], trace["object_reference_quat_xyzw"]
    )
    numeric_arrays = [value for value in trace.values() if np.issubdtype(value.dtype, np.number)]
    no_nan = all(np.all(np.isfinite(value)) for value in numeric_arrays)
    forces = np.abs(trace["actuator_force_substeps"])
    wrist_saturated = np.isclose(forces[:, :, :6], EFFORT[:6], rtol=0, atol=1e-6)
    finger_saturated = np.isclose(forces[:, :, 6:], EFFORT[6:], rtol=0, atol=1e-6)
    valid_hand_object_distance = trace["hand_object_min_distance_valid"].astype(bool)
    identity = asdict(trajectory.identity)
    identity["loaded_dataset_version"] = trajectory.dataset_version
    identity["object_z_shift"] = trajectory.object_z_shift
    warning_count = int(np.max(trace["warning_count"]))
    contact_capacity_boundary = bool(np.any(trace["contact_capacity_saturated"]))
    constraint_capacity_boundary = bool(np.any(trace["constraint_capacity_saturated"]))
    max_abs_dof_velocity = float(np.max(np.abs(trace["hand_qvel"])))
    wrist_saturation_fraction = float(np.mean(wrist_saturated))
    stability_gates = {
        "finite": no_nan,
        "warning_free": warning_count == 0,
        "below_mjx_contact_capacity_boundary": not contact_capacity_boundary,
        "below_mjx_constraint_capacity_boundary": not constraint_capacity_boundary,
        "max_abs_dof_velocity_below_100": max_abs_dof_velocity < 100.0,
        "zero_wrist_effort_saturation": wrist_saturation_fraction == 0.0,
    }
    return {
        "schema": "manorl.mujoco.reference_replay.v1",
        "trajectory_identity": identity,
        "backend": config.backend,
        "device": config.device,
        "trace_path": str(trace_path),
        "trace_steps": int(len(trace["target_index"])),
        "metrics": {
            "hand_translation_rmse_m": _rmse_norm(translation_error),
            "hand_angular_rmse_rad": _rmse_norm(angular_error),
            "object_position_rmse_m": _rmse_norm(object_position_error),
            "object_position_max_m": float(
                np.max(np.linalg.norm(object_position_error, axis=1))
            ),
            "object_orientation_rmse_rad": float(
                np.sqrt(np.mean(np.square(object_orientation_error)))
            ),
            "object_orientation_max_rad": float(np.max(object_orientation_error)),
            "max_abs_dof_velocity": max_abs_dof_velocity,
            "wrist_effort_saturation_fraction": wrist_saturation_fraction,
            "finger_effort_saturation_fraction": float(np.mean(finger_saturated)),
            "max_hand_object_penetration_m": float(
                max(
                    0.0,
                    -np.min(trace["hand_object_min_distance"][valid_hand_object_distance]),
                )
                if np.any(valid_hand_object_distance)
                else 0.0
            ),
            "hand_object_distance_observed": bool(np.any(valid_hand_object_distance)),
            "warning_count": warning_count,
            "no_nan": no_nan,
            "any_contact": bool(np.any(trace["has_contact"])),
            # These equality predicates are conservative capacity-boundary
            # warnings; public MJX data does not prove that contacts were dropped.
            "contact_capacity_saturated": contact_capacity_boundary,
            "constraint_capacity_saturated": constraint_capacity_boundary,
        },
        "stability_gates": stability_gates,
        "diagnostic_passed": all(stability_gates.values()),
        "config": {
            **asdict(config),
            "joint_names": list(JOINT_NAMES),
            "controller": config.controller,
            "wrist_kp": config.wrist_kp,
            "wrist_dampratio": config.wrist_dampratio,
            "finger_kp": list(config.finger_kp),
            "finger_dampratio": config.finger_dampratio,
            "hand_contacts_enabled": config.hand_contacts_enabled,
            "source_physx_kp_not_applied": SOURCE_PHYSX_KP.tolist(),
            "source_physx_kd_not_applied": SOURCE_PHYSX_KD.tolist(),
            "effort": EFFORT.tolist(),
            "joint_frictionloss": JOINT_FRICTIONLOSS,
            "joint_armature": JOINT_ARMATURE,
            "floor_top_z": FLOOR_TOP_Z,
            "object_clearance": OBJECT_CLEARANCE,
            "target_schedule": "source-compatible command 0,0,1,...,789; two substeps; post references 0..790",
            "object_source_rotation": "axis-angle",
            "task_quaternion_order": "xyzw",
            "mujoco_freejoint_quaternion_order": "wxyz",
            "direct_qpos_writes": "reset_only",
            "hand_gravity": "gravcomp",
            "object_gravity": "active",
            "mujoco_version": _distribution_version("mujoco"),
            "mujoco_mjx_version": _distribution_version("mujoco-mjx"),
            "warp_version": _distribution_version("warp-lang"),
            "pylance_version": _distribution_version("pylance"),
        },
        "claims": {
            "isaac_parity": "not_evaluated_no_isaac_trace",
        },
    }


def _output_paths(output: Path) -> tuple[Path, Path]:
    if output.suffix not in {"", ".npz"}:
        raise ValueError("--output must be a path prefix or end in .npz")
    trace_path = output if output.suffix == ".npz" else output.with_suffix(".npz")
    return trace_path, trace_path.with_suffix(".json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mjx-warp", "mujoco-cpu"), default="mjx-warp")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/manorl/cube1_01_009_mjx_warp")
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Diagnostic prefix only; omit to replay all 791 accepted transitions.",
    )
    parser.add_argument(
        "--wrist-kp", type=float, choices=WRIST_KP_GRID, default=200.0
    )
    parser.add_argument(
        "--wrist-dampratio",
        type=float,
        choices=WRIST_DAMPRATIO_GRID,
        default=1.0,
    )
    parser.add_argument(
        "--hand-contacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable only hand collision geoms for the free-space diagnostic.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trajectory = load_reference_trajectory()
    servo = ServoConfig(
        wrist_kp=args.wrist_kp,
        wrist_dampratio=args.wrist_dampratio,
        hand_contacts_enabled=args.hand_contacts,
    )
    config, trace = run_reference_replay(
        trajectory,
        backend=args.backend,
        device=args.device,
        max_steps=args.max_steps,
        servo=servo,
    )
    trace_path, summary_path = _output_paths(args.output)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace_path, **trace)
    summary = summarize_trace(trajectory, config, trace, trace_path=trace_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["diagnostic_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
