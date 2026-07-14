#!/usr/bin/env python3
"""Fail-fast target-side verifier for a source MANOHand semantic fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim.manorl.abi import check_termination, process_residual_actions
from sim.manorl.observations import (
    CURRENT_SOURCE_COMPATIBILITY,
    ObservationState,
    PointCloudTemplate,
    build_observation,
)
from sim.manorl.rewards import RewardState, compute_rewards
from tools.isaacgym_reference_trace import validate_semantic_fixture_schema

ATOL = 5.0e-5


def _scalar(arrays: dict[str, np.ndarray], name: str, row: int) -> Any:
    return arrays[name][row].item()


def _row(arrays: dict[str, np.ndarray], name: str, row: int) -> np.ndarray:
    return np.asarray(arrays[name][row])


def _assert_close(name: str, actual: np.ndarray | float, expected: np.ndarray | float) -> float:
    difference = np.max(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)))
    if not np.isfinite(difference) or difference > ATOL:
        raise AssertionError(f"{name} max abs difference {difference:.6g} exceeds {ATOL:.1e}")
    return float(difference)


def _action_result(arrays: dict[str, np.ndarray], row: int) -> dict[str, float]:
    result = process_residual_actions(
        _row(arrays, "raw_action", row)[None, :],
        trajectory_steps=np.asarray([_scalar(arrays, "action_trajectory_step", row)], dtype=np.int64),
        cumulative_offset=_row(arrays, "action_cumulative_offset", row)[None, :],
        cumulative_joint_offset=_row(arrays, "action_cumulative_joint_offset", row)[None, :],
        mocap_targets=_row(arrays, "mocap_target", row)[None, :],
        joint_lower=_row(arrays, "mano_dof_lower", row),
        joint_upper=_row(arrays, "mano_dof_upper", row),
        active_joint_mask=_row(arrays, "active_joint_mask", row)[None, :],
        use_residual=np.ones(1, dtype=np.float64),
        early_phase_starts=np.asarray([_scalar(arrays, "early_phase_start", row)], dtype=np.int64),
    )
    return {
        "processed_action": _assert_close("processed_action", result.actions[0], _row(arrays, "processed_action", row)),
        "cumulative_offset": _assert_close("cumulative_offset", result.cumulative_offset[0], _row(arrays, "cumulative_offset", row)),
        "cumulative_joint_offset": _assert_close(
            "cumulative_joint_offset", result.cumulative_joint_offset[0], _row(arrays, "cumulative_joint_offset", row)
        ),
        "controller_target": _assert_close("controller_target", result.targets[0], _row(arrays, "controller_target", row)),
    }


def _observation_result(arrays: dict[str, np.ndarray], row: int) -> float:
    count = int(_scalar(arrays, "object_support_point_count", row))
    cap = int(_scalar(arrays, "table_clearance_support_point_cap", row))
    action_id = int(_scalar(arrays, "action_id", row))
    action_type = _row(arrays, "action_type", row)
    if not 1 <= action_id <= 50 or not np.array_equal(action_type, np.eye(50, dtype=np.float32)[action_id - 1]):
        raise AssertionError("fixture action_id and source action_type one-hot disagree")
    point_cloud = PointCloudTemplate(
        local_points=_row(arrays, "point_cloud_local_template", row),
        mode=CURRENT_SOURCE_COMPATIBILITY.point_template_mode,
        normalized=bool(_scalar(arrays, "point_cloud_normalized", row)),
        scale=_row(arrays, "point_cloud_scale", row),
    )
    state = ObservationState(
        mano_dof_pos=_row(arrays, "mano_dof_pos", row)[None, :],
        mano_dof_lower=_row(arrays, "mano_dof_lower", row),
        mano_dof_upper=_row(arrays, "mano_dof_upper", row),
        hand_position=_row(arrays, "hand_position", row)[None, :],
        hand_orientation_xyzw=_row(arrays, "hand_orientation_xyzw", row)[None, :],
        object_position=_row(arrays, "object_position", row)[None, :],
        object_orientation_xyzw=_row(arrays, "object_orientation_xyzw", row)[None, :],
        target_object_position=_row(arrays, "target_object_position", row)[None, :],
        target_object_orientation_xyzw=_row(arrays, "target_object_orientation_xyzw", row)[None, :],
        target_object_pos_next_5=_row(arrays, "target_object_pos_next_5", row)[None, :],
        cumulative_offset=_row(arrays, "cumulative_offset", row)[None, :],
        cumulative_joint_offset=_row(arrays, "cumulative_joint_offset", row)[None, :],
        point_cloud=point_cloud,
        object_geometry=_row(arrays, "object_geometry", row)[None, :],
        hand_keypoint_positions=_row(arrays, "hand_keypoint_positions", row)[None, :, :],
        fingertip_positions=_row(arrays, "fingertip_positions", row)[None, :, :],
        hand_keypoint_contact_forces=_row(arrays, "hand_keypoint_contact_forces", row)[None, :, :],
        expected_contact_mask=_row(arrays, "expected_contact_mask", row)[None, :],
        action_ids=np.asarray([action_id], dtype=np.int64),
        object_support_points=_row(arrays, "object_support_points", row)[:count],
        table_surface_height=float(_scalar(arrays, "table_surface_height", row)),
        table_clearance_support_point_cap=cap,
    )
    target = build_observation(state, compatibility=CURRENT_SOURCE_COMPATIBILITY)
    return _assert_close("raw_observation", target.raw[0], _row(arrays, "source_raw_observation", row))


def _reward_result(arrays: dict[str, np.ndarray], metadata: dict[str, Any], row: int) -> dict[str, float]:
    config = metadata["configuration"]
    termination = check_termination(
        object_position=_row(arrays, "object_position", row)[None, :],
        target_position=_row(arrays, "target_object_position", row)[None, :],
        progress=np.asarray([_scalar(arrays, "progress", row)], dtype=np.int64),
        trajectory_lengths=np.asarray([792], dtype=np.int64),
        early_mask=np.asarray([
            _scalar(arrays, "trajectory_step", row)
            < _scalar(arrays, "early_phase_start", row) + CURRENT_SOURCE_COMPATIBILITY.early_phase_steps
        ]),
        max_deviation_distance=float(config["maxDeviationDistance"]),
        deviation_penalty=float(config["deviationPenalty"]),
    )
    state = RewardState(
        object_position=_row(arrays, "object_position", row)[None, :],
        target_object_position=_row(arrays, "target_object_position", row)[None, :],
        object_orientation_xyzw=_row(arrays, "object_orientation_xyzw", row)[None, :],
        target_object_orientation_xyzw=_row(arrays, "target_object_orientation_xyzw", row)[None, :],
        cumulative_offset=_row(arrays, "cumulative_offset", row)[None, :],
        cumulative_joint_offset=_row(arrays, "cumulative_joint_offset", row)[None, :],
        active_joint_mask=_row(arrays, "active_joint_mask", row)[None, :],
        hand_keypoint_contact_forces=_row(arrays, "hand_keypoint_contact_forces", row)[None, :, :],
        expected_contact_mask=_row(arrays, "expected_contact_mask", row)[None, :],
        expected_contact_weights=_row(arrays, "expected_contact_weights", row)[None, :],
        object_contact_force=_row(arrays, "object_contact_force", row)[None, :],
        object_gravity_force=np.asarray([_scalar(arrays, "object_gravity_force", row)]),
        object_linear_velocity=_row(arrays, "object_linear_velocity", row)[None, :],
        trajectory_steps=np.asarray([_scalar(arrays, "trajectory_step", row)], dtype=np.int64),
        contact_start_frames=np.asarray([_scalar(arrays, "contact_start_frame", row)], dtype=np.int64),
        contact_end_frames=np.asarray([_scalar(arrays, "contact_end_frame", row)], dtype=np.int64),
        rotation_disabled_mask=np.asarray([_scalar(arrays, "rotation_disabled_mask", row)], dtype=bool),
        early_phase_starts=np.asarray([_scalar(arrays, "early_phase_start", row)], dtype=np.int64),
    )
    reward = compute_rewards(state, compatibility=CURRENT_SOURCE_COMPATIBILITY, termination=termination)
    names = {
        "reward_total": reward.total[0],
        "reward_distance_x": reward.distance_x[0],
        "reward_distance_y": reward.distance_y[0],
        "reward_distance_z": reward.distance_z[0],
        "reward_rotation": reward.rotation[0],
        "reward_action_penalty": reward.action_penalty[0],
        "reward_position_penalty": reward.position_penalty[0],
        "reward_joint_penalty": reward.joint_penalty[0],
        "reward_contact": reward.contact[0],
        "reward_object_stability": reward.object_stability[0],
        "reward_object_speed": reward.object_speed[0],
    }
    return {name: _assert_close(name, actual, _scalar(arrays, name, row)) for name, actual in names.items()}


def _verify_termination_discriminator(metadata: dict[str, Any]) -> dict[str, float]:
    discriminator = metadata["termination_discriminator"]
    threshold = float(discriminator["max_deviation_distance"])
    penalty = float(discriminator["deviation_penalty"])
    report: dict[str, float] = {}
    for name, source in discriminator["cases"].items():
        result = check_termination(
            object_position=np.asarray([[source["object_target_distance"], 0.0, 0.0]]),
            target_position=np.zeros((1, 3)),
            progress=np.asarray([source["progress"]], dtype=np.int64),
            trajectory_lengths=np.asarray([792], dtype=np.int64),
            early_mask=np.asarray([source["early_mask"]], dtype=bool),
            max_deviation_distance=threshold,
            deviation_penalty=penalty,
        )
        actual = (
            bool(result.reset[0]),
            bool(result.deviation_reset[0]),
            float(result.deviation_penalty[0]),
        )
        expected = (source["reset"], source["deviation_reset"], source["deviation_penalty"])
        if actual[:2] != expected[:2]:
            raise AssertionError(f"termination {name} masks differ: {actual[:2]} != {expected[:2]}")
        report[name] = _assert_close(
            f"termination.{name}.penalty", actual[2], float(expected[2])
        )
    return report


def verify(path: Path) -> dict[str, Any]:
    npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    json_path = npz_path.with_suffix(".json")
    with np.load(npz_path) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    validate_semantic_fixture_schema(metadata, arrays)
    config = metadata["configuration"]
    if config["useResidualActions"] is not True or config["earlyPhaseMocapSteps"] != 100:
        raise AssertionError("semantic fixture is not current-source residual configuration")
    results: dict[str, dict[str, float]] = {}
    for row in range(len(arrays["physical_call"])):
        key = f"{int(arrays['physical_call'][row])}:{int(arrays['sample_kind'][row])}"
        if arrays["sample_kind"][row] == 0:
            results[key] = {
                **{f"action.{name}": value for name, value in _action_result(arrays, row).items()},
                "observation.raw": _observation_result(arrays, row),
                **{f"reward.{name}": value for name, value in _reward_result(arrays, metadata, row).items()},
            }
        else:
            # Source resets after the final post-step; this record is a state
            # transition witness, not the reward/action output of call 790.
            results[key] = {"delayed_reset_observation.raw": _observation_result(arrays, row)}
    return {
        "fixture": str(npz_path),
        "verified_samples": len(results),
        "max_abs_by_sample": results,
        "termination_discriminator_max_abs": _verify_termination_discriminator(metadata),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = verify(args.fixture)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
