from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim.dexhandrl.batch_evaluate import (
    N8EvaluationConfig,
    _determinism_summary,
    _force_distribution,
    _per_world_summary,
    build_arg_parser,
)
from sim.dexhandrl.rewards import REWARD_COMPONENTS
from sim.dexhandrl.trajectory_batch import DEXHAND_ALL_ACTIONS


def test_n8_config_enforces_fixed_scale_and_mode_checkpoint_contract(tmp_path: Path) -> None:
    assert N8EvaluationConfig(output_dir=tmp_path).num_envs == 8
    with pytest.raises(ValueError, match="exactly 8"):
        N8EvaluationConfig(num_envs=7)
    with pytest.raises(ValueError, match="requires --checkpoint"):
        N8EvaluationConfig(mode="trained")
    with pytest.raises(ValueError, match="only in trained"):
        N8EvaluationConfig(mode="reference", checkpoint=tmp_path / "run.pt")


def test_force_distribution_preserves_zero_and_calibrated_threshold_evidence() -> None:
    values = np.asarray([0.0, 0.1, 0.2, 0.20001, 1.0, 5.0, 5.1, np.nan])
    result = _force_distribution(values, (0.2, 5.0))
    assert result["count"] == 8
    assert result["finite_count"] == 7
    assert result["positive_count"] == 6
    assert result["zero_count"] == 1
    assert result["thresholds"]["0.2"]["above_count"] == 4
    assert result["thresholds"]["5"]["above_count"] == 1


def test_duplicate_determinism_checks_qpos_qvel_actions_and_physics() -> None:
    steps, worlds = 3, 8
    trace = {
        "active": np.ones((steps, worlds), dtype=bool),
        "actions": np.zeros((steps, worlds, 22), dtype=np.float32),
        "reward": np.zeros((steps, worlds), dtype=np.float32),
        "object_position": np.zeros((steps, worlds, 3), dtype=np.float32),
        "object_rotation_error_deg": np.zeros((steps, worlds), dtype=np.float32),
        "contact_magnitudes": np.zeros((steps, worlds, 16), dtype=np.float32),
        "duplicate_qpos": np.zeros((steps, 2, 4), dtype=np.float32),
        "duplicate_qvel": np.zeros((steps, 2, 3), dtype=np.float32),
    }
    exact = _determinism_summary(trace)
    assert exact["pass_1e_6"]
    assert exact["bitwise_qpos_equal"] and exact["bitwise_qvel_equal"]
    trace["duplicate_qpos"][1, 1, 0] = 2e-6
    drift = _determinism_summary(trace)
    assert not drift["pass_1e_6"]
    assert drift["qpos_max_abs_diff"] == pytest.approx(2e-6)


def test_per_world_summary_excludes_duplicate_arrays_and_reports_physics() -> None:
    steps, worlds = 4, 8
    components = np.zeros((steps, worlds, len(REWARD_COMPONENTS)), dtype=np.float32)
    components[:, :, 0] = 1.0
    reward = components.sum(axis=2)
    trace = {
        "active": np.ones((steps, worlds), dtype=bool),
        "global_valid": np.ones((steps,), dtype=bool),
        "frame": np.tile(np.arange(2, 6, dtype=np.int32)[:, None], (1, worlds)),
        "actions": np.zeros((steps, worlds, 22), dtype=np.float32),
        "reward": reward,
        "task_reward": reward,
        "embodiment_reward": np.zeros((steps, worlds), dtype=np.float32),
        "components": components,
        "terminated": np.zeros((steps, worlds), dtype=bool),
        "truncated": np.zeros((steps, worlds), dtype=bool),
        "reason": np.zeros((steps, worlds), dtype=np.int32),
        "object_position_error": np.full((steps, worlds), .05, dtype=np.float32),
        "object_rotation_error_deg": np.full((steps, worlds), 3.0, dtype=np.float32),
        "object_speed": np.full((steps, worlds), .1, dtype=np.float32),
        "object_angular_speed": np.full((steps, worlds), .2, dtype=np.float32),
        "object_position": np.zeros((steps, worlds, 3), dtype=np.float32),
        "target_object_position": np.zeros((steps, worlds, 3), dtype=np.float32),
        "object_height_delta": np.asarray([0, .01, .02, .01], dtype=np.float32)[:, None] * np.ones((1, worlds)),
        "target_height_delta": np.asarray([0, .02, .02, .01], dtype=np.float32)[:, None] * np.ones((1, worlds)),
        "hand_active_l2_error": np.full((steps, worlds), .01, dtype=np.float32),
        "finger_active_mean_abs_error": np.full((steps, worlds), .02, dtype=np.float32),
        "finger_active_max_abs_error": np.full((steps, worlds), .03, dtype=np.float32),
        "contact_magnitudes": np.zeros((steps, worlds, 16), dtype=np.float32),
        "raw_contact": np.zeros((steps, worlds), dtype=np.float32),
        "contact_gate": np.zeros((steps, worlds), dtype=np.float32),
        "fingertip_penetration": np.zeros((steps, worlds), dtype=np.float32),
        "fingertip_penetration_max_depth_m": np.full((steps, worlds), .001, dtype=np.float32),
        "fingertip_penetration_mean_depth_m": np.full((steps, worlds), .0005, dtype=np.float32),
        "duplicate_qpos": np.zeros((steps, 2, 4), dtype=np.float32),
        "duplicate_qvel": np.zeros((steps, 2, 3), dtype=np.float32),
    }
    trace["truncated"][-1] = True
    trace["reason"][-1] = 1
    trace["contact_magnitudes"][:, :, :2] = .3
    batch = type("Batch", (), {
        "task_expected_contact_masks": np.tile(np.asarray([1, 1] + [0] * 14, dtype=np.float32), (worlds, 1)),
        "movement_start_local": np.full(worlds, 2, dtype=np.int32),
        "movement_end_local": np.full(worlds, 4, dtype=np.int32),
        "lance_contact_annotation": np.zeros((worlds, 6, 16), dtype=np.float32),
        "lengths": np.full(worlds, 6, dtype=np.int32),
        "actions": DEXHAND_ALL_ACTIONS + ("01",),
        "row_ids": np.arange(worlds),
        "uuids": tuple(f"u-{i}" for i in range(worlds)),
        "seed_uuids": tuple(f"s-{i}" for i in range(worlds)),
        "start_frames": np.zeros(worlds, dtype=np.int32),
    })()
    world = _per_world_summary(
        world=0,
        trace=trace,
        batch=batch,
        thresholds_N=(.2, 5.0),
        deviation_threshold_m=.1,
        early_phase_steps=1,
    )
    assert world["completed_full_horizon"]
    assert world["raw_environment_return"] == pytest.approx(4.0)
    assert world["component_sums"]["distance_x"] == pytest.approx(4.0)
    assert world["object_lift_m"]["max"] == pytest.approx(.02)
    assert world["contact"]["0.2N"]["window_expected_contact_fraction_mean"] == pytest.approx(1.0)
    assert world["contact"]["5N"]["any_contact_frames"] == 0
    assert world["fingertip_penetration_max_depth_m"]["max"] == pytest.approx(.001)
    assert world["all_transitions_finite"]


def test_simulator_gate_excludes_duplicate_variance() -> None:
    """Long-horizon duplicate bifurcation is evidence, not a PPO blocker."""

    source = __import__("pathlib").Path("sim/dexhandrl/batch_evaluate.py").read_text(
        encoding="utf-8"
    )
    gate_block = source.split("        gates = {", 1)[1].split(
        "        assignment_manifest", 1
    )[0]
    assert '"duplicate_determinism_1e_6"' in gate_block
    assert 'simulator_gate_names = (' in gate_block
    selected = gate_block.split("simulator_gate_names = (", 1)[1].split(")", 1)[0]
    assert "duplicate_determinism_1e_6" not in selected
    assert 'gates["physical_interpretation_ready"] = gates["simulator_contract_ready"]' in gate_block


def test_parser_defaults_to_headless_reference_n8() -> None:
    args = build_arg_parser().parse_args([])
    assert (args.mode, args.num_envs, args.headless, args.tile_envs) == ("reference", 8, True, 8)
    assert args.force_thresholds == (0.2, 5.0)
