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
from sim.dexhandrl.skrl_env import (
    _decode_contact_force_world,
    _mask_contact_vectors,
    _normalize_active_positions,
    apply_active_target_post_filters,
)


def _set_active_target_filter_state(task) -> None:
    task.active_prev_targets = np.zeros(22, dtype=np.float32)
    task.active_max_deltas = np.full(22, 100.0, dtype=np.float32)
    task.active_lower = np.full(22, -100.0, dtype=np.float32)
    task.active_upper = np.full(22, 100.0, dtype=np.float32)
from sim.dexhandrl.skrl_env import _farthest_point_sample


def test_reset_keeps_physical_start_but_exposes_next_policy_frame() -> None:
    task = DexHandRLMJXEnv.__new__(DexHandRLMJXEnv)
    task.start_frame, task.end_frame = 61, 693
    task.base_position_offset = np.zeros(3, dtype=np.float32)
    task.joint_offset = np.zeros(16, dtype=np.float32)
    physical_frames, observed_frames = [], []

    def set_initial_state():
        physical_frames.append(task._current_frame)

    def build_observation(*, target_frame):
        observed_frames.append(target_frame)
        return np.zeros(461, dtype=np.float32), {"current_frame": target_frame}

    task._set_initial_state = set_initial_state
    task._build_observation = build_observation
    obs, info = task.reset()

    assert physical_frames == [61]
    assert observed_frames == [62]
    assert task._current_frame == 62
    assert task._episode_step_count == 0
    assert obs.shape == (461,)
    assert info["current_frame"] == 62


def test_default_reset_settles_one_control_frame_before_policy_frame() -> None:
    class Data:
        def __init__(self):
            self.qpos = np.zeros(1, dtype=np.float32)
            self.ctrl = np.zeros(1, dtype=np.float32)

        def replace(self, *, ctrl):
            self.ctrl = ctrl
            return self

    task = DexHandRLMJXEnv.__new__(DexHandRLMJXEnv)
    task.cfg = DexHandRLMJXEnvConfig()
    task.start_frame = 61
    task.trajectory = SimpleNamespace(
        object_pos=np.zeros((62, 3), dtype=np.float32),
        object_quat_xyzw=np.zeros((62, 4), dtype=np.float32),
    )
    task.route_joint_upper_limits = {}
    task.ids = SimpleNamespace(actuator_ids=(),)
    task.model = SimpleNamespace(nu=1)
    task.mx = object()
    task.jnp = SimpleNamespace(asarray=lambda value, dtype: value)
    task.mjx = SimpleNamespace(make_data=lambda *args, **kwargs: Data())
    task.jax = SimpleNamespace(block_until_ready=lambda value: value)
    task.forward = lambda mx, dx: dx
    step_calls = []
    task.step_fn = lambda mx, dx: (step_calls.append(1) or dx)

    original = {
        "reference": skrl_env_module._reference_full_targets,
        "hand": skrl_env_module.set_hand_qpos,
        "object": skrl_env_module.set_object_freejoint,
        "ctrl": skrl_env_module.make_ctrl_vector,
    }
    try:
        skrl_env_module._reference_full_targets = lambda *args, **kwargs: np.zeros(26, dtype=np.float32)
        skrl_env_module.set_hand_qpos = lambda _jnp, dx, _ids, _full: dx
        skrl_env_module.set_object_freejoint = lambda _jnp, dx, _ids, _pos, _quat: dx
        skrl_env_module.make_ctrl_vector = lambda _full, _ids, _nu: np.zeros(1, dtype=np.float32)
        task._set_initial_state()
    finally:
        skrl_env_module._reference_full_targets = original["reference"]
        skrl_env_module.set_hand_qpos = original["hand"]
        skrl_env_module.set_object_freejoint = original["object"]
        skrl_env_module.make_ctrl_vector = original["ctrl"]

    assert task.cfg.settle_frames == 1
    assert len(step_calls) == task.cfg.substeps == 10
    np.testing.assert_array_equal(task.active_prev_targets, np.zeros(22, dtype=np.float32))


def test_active_position_normalization_matches_unclipped_gym_affine_contract() -> None:
    lower = np.asarray([0.0, -2.0], dtype=np.float32)
    upper = np.asarray([2.0, 2.0], dtype=np.float32)

    normalized = _normalize_active_positions(np.asarray([-1.0, 3.0], dtype=np.float32), lower, upper)

    np.testing.assert_allclose(normalized, np.asarray([-2.0, 1.5], dtype=np.float32))


def test_active_target_post_filters_apply_velocity_before_position() -> None:
    previous = np.asarray([0.0, 0.95], dtype=np.float32)
    corrected = np.asarray([10.0, 10.0], dtype=np.float32)
    max_deltas = np.asarray([0.25, 0.25], dtype=np.float32)
    lower = np.asarray([-1.0, -1.0], dtype=np.float32)
    upper = np.asarray([1.0, 1.0], dtype=np.float32)

    applied = apply_active_target_post_filters(previous, corrected, max_deltas, lower, upper)

    # The second coordinate reaches 1.20 after the velocity clamp, then position-clamps.
    np.testing.assert_array_equal(applied, np.asarray([0.25, 1.0], dtype=np.float32))
    assert applied.dtype == np.float32


def test_active_target_limits_use_model_base_then_primary_finger_order() -> None:
    task = DexHandRLMJXEnv.__new__(DexHandRLMJXEnv)
    names = list(skrl_env_module.DEXHAND021PRO_FULL_DOF_NAMES[:6]) + ["finger_a", "finger_b"]
    task.active_control_primary_joint_names = names[6:]
    task.mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_JOINT=object()),
        mj_name2id=lambda _model, _kind, name: names.index(name),
    )
    task.model = SimpleNamespace(jnt_range=np.asarray([[index, index + 0.5] for index in range(len(names))], dtype=np.float32))
    task.active_finger_lower, task.active_finger_upper = task._build_active_finger_limits()

    lower, upper = task._build_active_target_limits()

    np.testing.assert_array_equal(lower, np.arange(8, dtype=np.float32))
    np.testing.assert_array_equal(upper, np.arange(8, dtype=np.float32) + 0.5)


def test_active_target_filter_recurrence_and_exact_max_deltas() -> None:
    task = DexHandRLMJXEnv.__new__(DexHandRLMJXEnv)
    task.cfg = DexHandRLMJXEnvConfig(dt=0.01)
    expected = np.asarray([0.03] * 3 + [0.1] * 3 + [0.02] * 16, dtype=np.float32)
    np.testing.assert_array_equal(task._build_active_max_deltas(), expected)

    lower = np.full(22, -10.0, dtype=np.float32)
    upper = np.full(22, 10.0, dtype=np.float32)
    first = apply_active_target_post_filters(np.zeros(22, dtype=np.float32), np.ones(22, dtype=np.float32), expected, lower, upper)
    second = apply_active_target_post_filters(first, np.ones(22, dtype=np.float32), expected, lower, upper)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected * 2.0)


@pytest.mark.parametrize("use_residual,early_phase_frames", [(False, 0), (True, 2), (True, 0)])
def test_all_action_branches_trace_applied_active_target(monkeypatch, use_residual, early_phase_frames) -> None:
    task = DexHandRLMJXEnv.__new__(DexHandRLMJXEnv)
    task.cfg = SimpleNamespace(
        use_residual=use_residual, early_phase_frames=early_phase_frames,
        post_contact_base_position_decay=(0.9, 0.9, 0.9), base_position_decay=(0.9, 0.9, 0.9),
        post_contact_joint_decay=1.0, joint_decay=0.9, base_position_scales=(0.002, 0.002, 0.003),
        base_position_max_offset=0.04, reference_joint_source="active",
    )
    task.start_frame, task.last_contact_frame, task.num_actions = 0, 100, 22
    task.base_position_offset = np.zeros(3, dtype=np.float32)
    task.joint_offset = np.zeros(16, dtype=np.float32)
    task.active_joint_mask = np.ones(16, dtype=np.float32)
    task.route_joint_upper_limits = {}
    task.trajectory = SimpleNamespace(hand_dof=np.zeros((2, 22), dtype=np.float32))
    _set_active_target_filter_state(task)
    task.active_max_deltas[:] = 0.02
    monkeypatch.setattr(skrl_env_module, "_reference_full_targets", lambda *args, **kwargs: np.ones(26, dtype=np.float32))

    task._apply_residual_action(np.ones(22, dtype=np.float32), target_frame=1, current_full=np.zeros(26, dtype=np.float32))

    trace = task._last_action_trace
    assert trace is not None
    assert set(("reference_active_target", "corrected_active_target", "applied_active_target", "full_target")) <= trace.keys()
    np.testing.assert_array_equal(task.active_prev_targets, trace["applied_active_target"])


def test_episode_time_uses_action_count_not_logical_frame() -> None:
    task = DexHandRLMJXEnv.__new__(DexHandRLMJXEnv)
    task.cfg = SimpleNamespace(dt=0.02)
    task.max_episode_time_s = 633 * task.cfg.dt
    task._episode_step_count = 0
    assert task._normalized_episode_time() == 0.0
    task._episode_step_count = 1
    assert task._normalized_episode_time() == pytest.approx(1 / 633)
    task._episode_step_count = 631
    assert task._normalized_episode_time() == pytest.approx(631 / 633)


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
    _set_active_target_filter_state(task)
    raw_reference = np.linspace(-0.2, 0.2, 22, dtype=np.float32)
    task.trajectory = SimpleNamespace(hand_dof=np.stack([raw_reference, raw_reference]))
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
    assert task._last_action_trace is not None
    assert task._last_action_trace["decision_frame"] == 1
    np.testing.assert_array_equal(task._last_action_trace["reference_active_target"], raw_reference)
    assert task._last_action_trace["corrected_active_target"].shape == (22,)
    assert task._last_action_trace["applied_active_target"].shape == (22,)
    np.testing.assert_array_equal(task._last_action_trace["full_target"], corrected)


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
    _set_active_target_filter_state(task)
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
    _set_active_target_filter_state(task)
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
