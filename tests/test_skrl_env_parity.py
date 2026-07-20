from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import sim.dexhandrl.skrl_env as skrl_env_module
from sim.dexhandrl.parity import (
    add_termination_reward,
    apply_reward_weights,
    load_isaac_reward_contract,
    rotation_tracking_reward,
)
from sim.dexhandrl.skrl_env import DexHandRLMJXEnv
from sim.dexhandrl.skrl_env import DexHandRLMJXEnvConfig
from sim.dexhandrl.skrl_env import _decode_contact_force_world, _mask_contact_vectors
from sim.dexhandrl.skrl_env import _farthest_point_sample


REWARD_WEIGHTS = {
    "alive": 0.001,
    "object_pos_tracking_x": 0.05,
    "object_pos_tracking_y": 0.05,
    "object_pos_tracking_z": 1.0,
    "contact_quality": 1.5,
    "object_stability_velocity": 1.0,
    "object_rot_tracking": 1.0,
    "fingertip_inside_object_penalty": 1.5,
    "action_penalty_position": 0.075,
    "action_penalty_joint": 0.016,
    "termination_success": 0.0,
    "termination_failure_penalty": -200.0,
    "termination_timeout_penalty": 0.0,
}


def test_residual_action_masks_inactive_finger_joints(monkeypatch) -> None:
    task = DexHandRLMJXEnv.__new__(DexHandRLMJXEnv)
    task.cfg = SimpleNamespace(
        use_residual=True,
        early_phase_frames=0,
        post_contact_base_position_decay=(0.9, 0.9, 0.9),
        base_position_decay=(0.9, 0.9, 0.9),
        post_contact_joint_decay=1.0,
        joint_decay=0.9,
        base_position_scales=(0.002, 0.002, 0.003),
        base_position_max_offset=0.04,
        reference_joint_source="active",
    )
    task.start_frame = 0
    task.last_contact_frame = 100
    task.num_actions = 22
    task.base_position_offset = np.zeros(3, dtype=np.float32)
    task.joint_offset = np.zeros(16, dtype=np.float32)
    task.active_joint_mask = np.asarray([1.0, 0.0] + [1.0] * 14, dtype=np.float32)
    task.trajectory = object()
    task.route_joint_upper_limits = {}
    monkeypatch.setattr(
        skrl_env_module,
        "_reference_full_targets",
        lambda *args, **kwargs: np.zeros(26, dtype=np.float32),
    )

    corrected = task._apply_residual_action(
        np.ones(22, dtype=np.float32),
        target_frame=1,
        current_full=np.zeros(26, dtype=np.float32),
    )

    assert task.joint_offset[0] > 0.0
    assert task.joint_offset[1] == 0.0
    assert corrected[6] > 0.0
    assert corrected[7] == 0.0


def test_residual_action_uses_active_to_full_joint_mapping(monkeypatch) -> None:
    task = DexHandRLMJXEnv.__new__(DexHandRLMJXEnv)
    task.cfg = SimpleNamespace(
        use_residual=True,
        early_phase_frames=0,
        post_contact_base_position_decay=(0.9, 0.9, 0.9),
        base_position_decay=(0.9, 0.9, 0.9),
        post_contact_joint_decay=1.0,
        joint_decay=0.9,
        base_position_scales=(0.002, 0.002, 0.003),
        base_position_max_offset=0.04,
        reference_joint_source="active",
    )
    task.start_frame = 0
    task.last_contact_frame = 100
    task.num_actions = 22
    task.base_position_offset = np.zeros(3, dtype=np.float32)
    task.joint_offset = np.zeros(16, dtype=np.float32)
    task.active_joint_mask = np.zeros(16, dtype=np.float32)
    task.active_joint_mask[7] = 1.0
    task.trajectory = object()
    task.route_joint_upper_limits = {}
    monkeypatch.setattr(
        skrl_env_module,
        "_reference_full_targets",
        lambda *args, **kwargs: np.zeros(26, dtype=np.float32),
    )
    action = np.zeros(22, dtype=np.float32)
    action[13] = 1.0

    corrected = task._apply_residual_action(
        action,
        target_frame=1,
        current_full=np.zeros(26, dtype=np.float32),
    )

    assert corrected[13] == 0.0
    assert corrected[14] > 0.0


def test_routed_residual_preserves_current_distal_distribution(monkeypatch) -> None:
    task = DexHandRLMJXEnv.__new__(DexHandRLMJXEnv)
    task.cfg = SimpleNamespace(
        use_residual=True,
        early_phase_frames=0,
        post_contact_base_position_decay=(0.9, 0.9, 0.9),
        base_position_decay=(0.9, 0.9, 0.9),
        post_contact_joint_decay=1.0,
        joint_decay=0.9,
        base_position_scales=(0.002, 0.002, 0.003),
        base_position_max_offset=0.04,
        reference_joint_source="active",
    )
    task.start_frame = 0
    task.last_contact_frame = 100
    task.num_actions = 22
    task.base_position_offset = np.zeros(3, dtype=np.float32)
    task.joint_offset = np.zeros(16, dtype=np.float32)
    task.active_joint_mask = np.zeros(16, dtype=np.float32)
    task.active_joint_mask[6] = 1.0
    task.trajectory = object()
    task.route_joint_upper_limits = {}
    monkeypatch.setattr(
        skrl_env_module,
        "_reference_full_targets",
        lambda *args, **kwargs: np.zeros(26, dtype=np.float32),
    )
    action = np.zeros(22, dtype=np.float32)
    action[12] = 1.0
    current_full = np.zeros(26, dtype=np.float32)
    current_full[12] = 0.02
    current_full[13] = 0.04

    corrected = task._apply_residual_action(action, target_frame=1, current_full=current_full)

    assert np.isclose(corrected[12], 0.02)
    assert np.isclose(corrected[13], 0.04)


def test_fingertip_penalty_matches_isaac_depth_ramp() -> None:
    task = DexHandRLMJXEnv.__new__(DexHandRLMJXEnv)
    task.cfg = SimpleNamespace(
        fingertip_inside_object_min_depth=0.001,
        fingertip_inside_object_max_depth=0.003,
        max_fingertip_inside_object_penalty=0.4,
    )
    task.object_primitive_type = 0
    task.object_primitive_params = np.asarray([0.1, 0.1, 0.1], dtype=np.float32)
    task.active_fingertip_mask = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    task.active_fingertip_count = 1
    object_pos = np.zeros(3, dtype=np.float32)
    object_quat_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    fingertips = np.asarray(
        [
            [0.046, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    penalty = task._compute_fingertip_penalty(fingertips, object_pos, object_quat_xyzw)

    assert np.isclose(penalty, -0.4)


def test_mesh_fps_sampling_is_deterministic_and_covers_oversample() -> None:
    points = np.arange(30 * 3, dtype=np.float32).reshape(30, 3)

    sampled_a = _farthest_point_sample(points, num_points=6, seed=42)
    sampled_b = _farthest_point_sample(points, num_points=6, seed=42)

    assert sampled_a.shape == (6, 3)
    assert np.array_equal(sampled_a, sampled_b)
    assert np.array_equal(sampled_a[0], points[12])
    assert len({tuple(point) for point in sampled_a}) == 6


def test_pyramidal_contact_force_includes_tangential_components() -> None:
    force_world, normal_force = _decode_contact_force_world(
        frame=np.eye(3, dtype=np.float32),
        friction=np.asarray([0.5, 0.25, 0.0, 0.0, 0.0], dtype=np.float32),
        condim=3,
        efc_addresses=np.asarray([0, 1, 2, 3], dtype=np.int32),
        efc_force=np.asarray([2.0, 1.0, 4.0, 2.0], dtype=np.float32),
        pyramidal=True,
    )

    assert normal_force == pytest.approx(9.0)
    assert force_world == pytest.approx(np.asarray([9.0, 0.5, 0.5]))


def test_contact_observation_vectors_apply_per_action_mask() -> None:
    vectors = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    masked = _mask_contact_vectors(vectors, np.asarray([1.0, 0.0], dtype=np.float32))

    np.testing.assert_allclose(masked, [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])


@pytest.mark.parametrize("angle_deg", [0.0, 12.5, 20.0, 45.0, 73.0, 90.0, 120.0])
def test_rotation_reward_matches_isaac_piecewise_curve(angle_deg: float) -> None:
    if angle_deg <= 20.0:
        expected = 0.4 * (-0.00125 * angle_deg**2 + 1.0) - 0.1
    elif angle_deg <= 90.0:
        offset = angle_deg - 20.0
        expected = 0.4 * (-0.00003175 * offset**2 - 0.019206 * offset + 0.5) - 0.1
    else:
        expected = -0.5

    assert rotation_tracking_reward(angle_deg) == pytest.approx(expected, abs=1e-12)


def _make_reward_task() -> DexHandRLMJXEnv:
    task = DexHandRLMJXEnv.__new__(DexHandRLMJXEnv)
    task.cfg = SimpleNamespace(
        early_phase_frames=10,
        contact_force_threshold=5.0,
        max_contact_reward=0.4,
        max_object_stability_reward=0.4,
        object_stability_velocity_scale=0.1,
    )
    task.reward_weights = REWARD_WEIGHTS
    task.start_frame = 0
    task.last_contact_frame = 100
    task.expected_contact_mask = np.asarray([1.0, 1.0] + [0.0] * 14, dtype=np.float32)
    task.rotation_disabled = False
    task.base_position_offset = np.asarray([0.001, -0.002, 0.0], dtype=np.float32)
    task.joint_offset = np.asarray([0.01, -0.02] + [0.0] * 14, dtype=np.float32)
    task.active_joint_count = 2
    task._compute_fingertip_penalty = lambda **_: -0.1
    return task


def _reward_inputs() -> dict[str, object]:
    contact_vectors = np.zeros((16, 3), dtype=np.float32)
    contact_vectors[0, 0] = 6.0
    contact_vectors[1, 0] = 4.0
    return {
        "object_pos": np.zeros(3, dtype=np.float32),
        "object_quat_xyzw": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "object_lin_vel": np.zeros(3, dtype=np.float32),
        "target_pos": np.zeros(3, dtype=np.float32),
        "target_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "contact_summary": {"contact_force_vectors": contact_vectors},
        "fingertip_positions_world": np.zeros((5, 3), dtype=np.float32),
        "hand_pose": np.zeros(7, dtype=np.float32),
        "position_error": np.zeros(3, dtype=np.float32),
        "rotation_error_deg": 45.0,
    }


def test_reward_terms_match_isaac_raw_weighted_contract() -> None:
    task = _make_reward_task()

    terms = task._compute_reward_terms(target_frame=20, **_reward_inputs())

    assert terms["alive"] == 1.0
    assert terms["alive_weighted"] == pytest.approx(0.001)
    assert terms["contact_quality"] == pytest.approx(0.2)
    assert terms["contact_quality_weighted"] == pytest.approx(0.3)
    assert terms["object_pos_tracking_x"] == pytest.approx(0.2)
    assert terms["object_pos_tracking_x_weighted"] == pytest.approx(0.01)
    assert terms["object_pos_tracking_z_weighted"] == pytest.approx(0.2)
    assert terms["object_stability_velocity"] == 0.0
    assert terms["object_rot_tracking"] == pytest.approx(rotation_tracking_reward(45.0))
    assert terms["fingertip_inside_object_penalty_weighted"] == pytest.approx(-0.15)
    assert terms["action_penalty_position"] == pytest.approx(-0.3)
    assert terms["action_penalty_position_weighted"] == pytest.approx(-0.0225)
    assert terms["action_penalty_joint"] == pytest.approx(-1.2)
    assert terms["action_penalty_joint_weighted"] == pytest.approx(-0.0192)
    weighted_values = [
        value
        for name, value in terms.items()
        if name.endswith("_weighted")
    ]
    assert terms["total"] == pytest.approx(sum(weighted_values))


def test_reward_phase_gating_matches_isaac() -> None:
    task = _make_reward_task()

    early = task._compute_reward_terms(target_frame=5, **_reward_inputs())
    assert early["contact_quality"] == 0.0
    assert early["object_pos_tracking_z"] == 0.0
    assert early["object_stability_velocity"] == 0.0
    assert early["object_rot_tracking"] == 0.0
    assert early["alive_weighted"] == pytest.approx(0.001)
    assert early["fingertip_inside_object_penalty_weighted"] == pytest.approx(-0.15)
    assert early["action_penalty_joint_weighted"] == pytest.approx(-0.0192)

    post_contact = task._compute_reward_terms(target_frame=101, **_reward_inputs())
    assert post_contact["contact_quality"] == 0.0
    assert post_contact["object_stability_velocity"] == pytest.approx(0.4)
    assert post_contact["action_penalty_position"] == pytest.approx(-0.15)
    assert post_contact["action_penalty_joint"] == pytest.approx(-0.6)


def test_termination_reward_updates_logged_and_returned_total() -> None:
    components = apply_reward_weights({"alive": 1.0}, REWARD_WEIGHTS)

    added = add_termination_reward(
        components, "termination_failure_penalty", REWARD_WEIGHTS
    )

    assert added == -200.0
    assert components["termination_failure_penalty"] == 1.0
    assert components["termination_failure_penalty_weighted"] == -200.0
    assert components["total"] == pytest.approx(-199.999)


def test_default_reward_config_matches_canonical_isaac_yaml() -> None:
    cfg = DexHandRLMJXEnvConfig()
    contract = load_isaac_reward_contract(cfg.task_config_path)

    assert contract.reward_weights == REWARD_WEIGHTS | {
        "height_safety": 0.0,
        "finger_velocity": 0.0,
        "hand_velocity": 0.0,
        "hand_angular_velocity": 0.0,
        "joint_limit": 0.0,
        "finger_acceleration": 0.0,
        "hand_acceleration": 0.0,
        "hand_angular_acceleration": 0.0,
        "contact_stability": 0.0,
    }
    assert cfg.early_phase_frames == contract.early_phase_frames
    assert cfg.object_position_terminal_threshold == contract.object_position_terminal_threshold
    assert cfg.contact_force_threshold == contract.contact_force_threshold
    assert cfg.object_stability_velocity_scale == contract.object_stability_velocity_scale
    assert cfg.fingertip_inside_object_min_depth == contract.fingertip_inside_object_min_depth
    assert cfg.fingertip_inside_object_max_depth == contract.fingertip_inside_object_max_depth
