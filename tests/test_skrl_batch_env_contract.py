"""Dependency-light contract tests for the one-model DexHand MJX batch runtime."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim.dexhandrl.hand_mapping import active_to_full_dof, full_to_active_dof
from sim.dexhandrl.rewards import DexHandRewardConfig
from sim.dexhandrl.skrl_batch_env import (
    DexHandRLBatchConfig,
    conservative_batch_ccd_capacity,
    conservative_batch_constraint_capacity,
    conservative_batch_contact_capacity,
)
from sim.dexhandrl.trajectory_batch import DexHandTrajectoryBatch


def _batch(worlds: int = 1) -> DexHandTrajectoryBatch:
    frames = 3
    zeros = lambda *shape, dtype=np.float32: np.zeros(shape, dtype=dtype)
    return DexHandTrajectoryBatch(
        hand_active=zeros(worlds, frames, 22), hand_finger_dof_full_raw=None,
        full_raw_validity=np.zeros((worlds, frames), dtype=bool), object_pos=zeros(worlds, frames, 3),
        object_quat_xyzw=zeros(worlds, frames, 4), lance_contact_annotation=zeros(worlds, frames, 16),
        lengths=np.full(worlds, frames, dtype=np.int32), start_frames=np.zeros(worlds, dtype=np.int32),
        movement_start_local=np.zeros(worlds, dtype=np.int32), movement_end_local=np.ones(worlds, dtype=np.int32),
        row_ids=np.arange(worlds, dtype=np.int64), action_indices=np.zeros(worlds, dtype=np.int32),
        actions=("01",) * worlds, uuids=tuple(f"u-{index}" for index in range(worlds)),
        action_onehot50=zeros(worlds, 50), task_expected_contact_masks=zeros(worlds, 16),
        task_contact_body_names=((),) * worlds, active_joint_masks=zeros(worlds, 16),
        active_fingertip_masks=zeros(worlds, 5), rotation_disabled=np.zeros(worlds, dtype=bool),
    )


def test_render_snapshot_materializes_only_render_state_not_contact_telemetry() -> None:
    from sim.dexhandrl.skrl_batch_env import DexHandRLMJXBatchEnv

    class Data:
        qpos = np.ones((2, 3), dtype=np.float32)
        qvel = np.ones((2, 2), dtype=np.float32)
        ctrl = np.ones((2, 4), dtype=np.float32)
        @property
        def _impl(self):
            raise AssertionError("render snapshot must not access Warp contact telemetry")

    env = object.__new__(DexHandRLMJXBatchEnv)
    env.state = type("State", (), {"data": Data(), "frame": np.array([3, 4]), "episode_count": np.array([5, 6])})()
    snapshot = env.render_snapshot()
    np.testing.assert_array_equal(snapshot.qpos, Data.qpos)
    np.testing.assert_array_equal(snapshot.frame, [3, 4])
    np.testing.assert_array_equal(snapshot.episode_count, [5, 6])


def test_batch_config_requires_all_actions_and_active_reference() -> None:
    batch = _batch()
    with pytest.raises(ValueError, match="action='all'"):
        DexHandRLBatchConfig(trajectories=batch, action="01")
    with pytest.raises(ValueError, match="reference_joint_source='active'"):
        DexHandRLBatchConfig(trajectories=batch, reference_joint_source="full_raw")


def test_batch_capacity_floors_are_global_and_per_world() -> None:
    batch = _batch(7)
    assert conservative_batch_contact_capacity(7) == 960
    assert conservative_batch_ccd_capacity(7) == 512
    # efc__force is (world, capacity), so N must not inflate the second axis.
    assert [conservative_batch_constraint_capacity(worlds) for worlds in (1, 7, 1024)] == [512, 512, 512]
    with pytest.raises(ValueError, match="global 128-row-per-world floor"):
        DexHandRLBatchConfig(trajectories=batch, contact_capacity=895)
    with pytest.raises(ValueError, match="global 64-candidate-per-world floor"):
        DexHandRLBatchConfig(trajectories=batch, ccd_capacity=447)
    with pytest.raises(ValueError, match="must be positive"):
        DexHandRLBatchConfig(trajectories=batch, per_world_constraint_capacity=0)


def test_batch_capacity_sizing_matches_measured_256_world_high_water() -> None:
    """Preserve the Server2 sizing rationale without changing physics limits."""
    worlds = 256
    observed_global_contacts = 1377
    observed_max_constraints_per_world = 56
    measured_initial_nefc = 146
    observed_reset_collisions = 220

    contact_capacity = conservative_batch_contact_capacity(worlds)
    ccd_capacity = conservative_batch_ccd_capacity(worlds)
    constraint_capacity = conservative_batch_constraint_capacity(worlds)

    assert contact_capacity == 128 * worlds + 64 == 32832
    assert ccd_capacity == 64 * worlds + 64 == 16448
    assert contact_capacity / observed_global_contacts > 23
    assert constraint_capacity == 512
    assert constraint_capacity >= measured_initial_nefc + 64
    # The production runtime retains fail-closed 80% high-water guards.
    assert observed_global_contacts / contact_capacity < 0.8
    assert observed_reset_collisions / conservative_batch_ccd_capacity(7) < 0.8
    assert observed_max_constraints_per_world / constraint_capacity < 0.8


def test_reset_target_routes_extreme_active_reference_before_storing_previous() -> None:
    active = np.zeros((1, 22), dtype=np.float32)
    # The first routed active slot is index finger's joint3 + joint4 target.
    active[0, 6 + 6] = 100.0
    limits = {6: (0.25, 0.5), 9: (1.2, 1.2), 12: (1.2, 1.2), 15: (1.2, 1.2)}

    full = active_to_full_dof(active, route_joint_upper_limits=limits)
    previous = full_to_active_dof(full)

    np.testing.assert_allclose(previous[0, 6 + 6], 0.75)
    assert full[0, 12] == pytest.approx(0.25)
    assert full[0, 13] == pytest.approx(0.5)


def test_selected_reset_rebuilds_contacts_and_preserves_unselected_worlds() -> None:
    pytest.importorskip("mujoco")
    from sim.dexhandrl.skrl_batch_env import DexHandRLMJXBatchEnv

    worlds = 7
    batch = replace(
        _batch(worlds),
        object_quat_xyzw=np.tile(np.asarray([0., 0., 0., 1.], dtype=np.float32), (worlds, 3, 1)),
    )
    env = DexHandRLMJXBatchEnv(DexHandRLBatchConfig(trajectories=batch, device="cpu", settle_frames=0))
    try:
        clean = env.reset_device()
        env.step_device(np.zeros((worlds, 22), dtype=np.float32))
        before_zero = env.telemetry()
        env.reset_done_device(np.zeros(worlds, dtype=bool))
        after_zero = env.telemetry()
        np.testing.assert_array_equal(after_zero["frame"], before_zero["frame"])
        np.testing.assert_array_equal(after_zero["episode_count"], before_zero["episode_count"])
        np.testing.assert_array_equal(after_zero["qpos"], before_zero["qpos"])
        np.testing.assert_array_equal(after_zero["qvel"], before_zero["qvel"])
        before = np.asarray(env.state.data.qpos).copy()
        selected = np.asarray([True, False, True, False, False, False, False])
        mixed = env.reset_done_device(selected)
        after = np.asarray(env.state.data.qpos)

        np.testing.assert_allclose(np.asarray(mixed.observation)[selected], np.asarray(clean.observation)[selected])
        np.testing.assert_array_equal(after[~selected], before[~selected])
        assert np.all(np.asarray(mixed.valid))
    finally:
        env.close()


def test_reward_config_is_downstream_of_residual_off_physics() -> None:
    """Reward-only changes cannot alter control, state, or strict contacts."""

    pytest.importorskip("mujoco")
    from sim.dexhandrl.skrl_batch_env import DexHandRLMJXBatchEnv

    batch = replace(
        _batch(1),
        object_quat_xyzw=np.tile(
            np.asarray([0., 0., 0., 1.], dtype=np.float32), (1, 3, 1)
        ),
    )
    baseline_reward = DexHandRewardConfig(early_phase_steps=0)
    counterfactual_reward = replace(
        baseline_reward,
        distance_scales=(3.0, 4.0, 5.0),
        contact_force_threshold_N=99.0,
        max_deviation_distance_m=0.0,
        deviation_penalty=17.0,
        survival_reward=0.7,
        fingertip_penetration_weight=9.0,
    )
    env_a = DexHandRLMJXBatchEnv(DexHandRLBatchConfig(
        trajectories=batch, device="cpu", settle_frames=0,
        use_residual=False, reward=baseline_reward,
    ))
    env_b = DexHandRLMJXBatchEnv(DexHandRLBatchConfig(
        trajectories=batch, device="cpu", settle_frames=0,
        use_residual=False, reward=counterfactual_reward,
    ))
    try:
        action = np.ones((1, 22), dtype=np.float32)
        transition_a = env_a.step_device(action)
        transition_b = env_b.step_device(action)

        assert not np.array_equal(
            np.asarray(transition_a.reward), np.asarray(transition_b.reward)
        )
        np.testing.assert_array_equal(transition_a.ctrl, transition_b.ctrl)
        np.testing.assert_array_equal(env_a.state.data.qpos, env_b.state.data.qpos)
        np.testing.assert_array_equal(env_a.state.data.qvel, env_b.state.data.qvel)
        np.testing.assert_array_equal(
            transition_a.contact_magnitudes, transition_b.contact_magnitudes
        )
    finally:
        env_a.close()
        env_b.close()


def test_zero_reset_mask_is_device_identity_without_forward() -> None:
    source = Path("sim/dexhandrl/skrl_batch_env.py").read_text(encoding="utf-8")
    reset_mask = source.split("    def _reset_mask(", 1)[1].split("    def _reset_observation", 1)[0]
    # The device conditional makes the false branch an identity. Therefore a
    # zero reset mask neither invokes Warp forward nor changes batch state.
    assert "self.jax.lax.cond(self.jp.any(mask), reset_rows, lambda current: current, state)" in reset_mask
    assert "data = self._forward_world(data)" in reset_mask


def test_device_transition_source_has_no_host_conversion() -> None:
    source = Path("sim/dexhandrl/skrl_batch_env.py").read_text(encoding="utf-8")
    transition = source.split("    def _transition(", 1)[1].split("    def _replace_rows", 1)[0]
    assert "np.asarray" not in transition
    assert "jax.vmap" in source
    assert "reduce_dexhand_contacts" in source
    assert "data._impl" in source
    assert source.count("self.mjx.make_data(") == 1
    assert "data = self._forward_world(data)" in source
    assert "naccdmax=self.ccd_capacity" in source
    assert "requested_ccd_capacity" in source
    assert "actual_ccd_capacity" in source
    assert "ncollision_high_water" in source
    assert "requested_per_world_constraint_capacity" in source
    assert "actual_constraint_buffer_shape" in source
