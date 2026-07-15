from __future__ import annotations

import numpy as np
import pytest

from sim.manorl.assets import OBJECT_MESH
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.mjx_sim import MujocoCpuReplay
from sim.manorl.observations import (
    CHECKPOINT_SIDECAR_COMPATIBILITY,
    OBSERVATION_SLICES,
    quat_rotate_xyzw,
)
from sim.manorl.trajectory import load_reference_trajectory


@pytest.fixture(scope="module")
def trajectory():
    return load_reference_trajectory()


def _environment(trajectory, *, num_envs: int = 1, residual_enabled: bool = False) -> MujocoManoEnvironment:
    return MujocoManoEnvironment(
        trajectory,
        EnvironmentConfig(
            num_envs=num_envs,
            residual_enabled=residual_enabled,
            # Physical contact/state extraction is under test; this avoids
            # making an action-ABI fixture depend on a trajectory deviation.
            max_deviation_distance=1_000_000.0,
        ),
    )


def test_native_contact_frame_and_geom_sign_match_body_external_force(trajectory) -> None:
    """Controlled native fixture validates the world-frame/sign convention."""

    replay = MujocoCpuReplay(trajectory)
    replay.step(np.zeros(26, dtype=np.float64))
    replay.mujoco.mj_rnePostConstraint(replay.model, replay.data)
    for contact_id, contact in enumerate(replay.data.contact[: replay.data.ncon]):
        wrench = np.zeros(6, dtype=np.float64)
        replay.mujoco.mj_contactForce(replay.model, replay.data, contact_id, wrench)
        if np.linalg.norm(wrench[:3]) <= 1e-8:
            continue
        first_geom, second_geom = map(int, contact.geom)
        first_body = int(replay.model.geom_bodyid[first_geom])
        second_body = int(replay.model.geom_bodyid[second_geom])
        if first_body == 0 or second_body == 0:
            world_force = wrench[:3] @ contact.frame.reshape(3, 3)
            moving_body = second_body if first_body == 0 else first_body
            expected = world_force if first_body == 0 else -world_force
            np.testing.assert_allclose(
                replay.data.cfrc_ext[moving_body, 3:], expected, rtol=1e-10, atol=1e-10
            )
            return
    pytest.fail("expected a solved floor-hand contact in the real accepted replay state")


def test_producer_keypoint_order_fingertips_and_static_template(trajectory) -> None:
    env = _environment(trajectory)
    assert env.last_physical is not None
    snapshot = env.last_physical
    xpos = np.asarray(env.data.xpos, dtype=np.float64)[0]
    np.testing.assert_allclose(
        snapshot.hand_keypoint_positions[0], xpos[np.asarray(env.producer.keypoint_body_ids)]
    )
    source_quat = snapshot.hand_orientation_xyzw[0]
    assert np.isclose(np.linalg.norm(source_quat), 1.0)
    fingertip_keypoint_ids = [15, 3, 6, 9, 12]
    quaternions = np.asarray(env.data.xquat, dtype=np.float64)[0, np.asarray(env.producer.keypoint_body_ids)[fingertip_keypoint_ids]]
    quaternions_xyzw = quaternions[:, (1, 2, 3, 0)]
    offsets = np.asarray(
        (
            (-0.028633, -0.004191, 0.023667),
            (-0.027384, -0.000215, -0.000966),
            (-0.027549, 0.000501, -0.004523),
            (-0.026165, -0.000075, -0.007781),
            (-0.018526, -0.001581, -0.011018),
        )
    )
    expected_tip = snapshot.hand_keypoint_positions[0, fingertip_keypoint_ids] + quat_rotate_xyzw(quaternions_xyzw, offsets)
    np.testing.assert_allclose(snapshot.fingertip_positions[0], expected_tip, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(np.flatnonzero(env.expected_contact_mask[0]), [3, 15])
    np.testing.assert_array_equal(env.active_joint_mask[0], [True] * 8 + [False] * 12)

    trimesh = pytest.importorskip("trimesh")
    source_mesh = trimesh.load(OBJECT_MESH, force="mesh")
    source_mesh.apply_scale(0.001)
    source_points, _ = trimesh.sample.sample_surface(source_mesh, 64, seed=42)
    lower, upper = source_points.min(axis=0), source_points.max(axis=0)
    expected_template = (source_points - (upper + lower) / 2.0) / np.maximum((upper - lower) / 2.0, 1e-6)
    np.testing.assert_allclose(env._static_template.local_points, expected_template, rtol=0, atol=1e-12)


def test_mjx_contact_producer_feeds_source_order_observation_and_reward(trajectory) -> None:
    env = _environment(trajectory)
    observation, reward, reset, _ = env.step(np.zeros((1, 26), dtype=np.float64))
    assert env.last_physical is not None
    assert env.last_observation is not None
    assert env.last_reward is not None
    snapshot = env.last_physical
    assert snapshot.contact_count[0] > 0
    assert np.linalg.norm(snapshot.hand_keypoint_contact_forces[0, 3]) > 0.0
    np.testing.assert_allclose(
        env.last_observation.contacts.force_xyz, snapshot.hand_keypoint_contact_forces
    )
    np.testing.assert_allclose(
        env.last_observation.raw[:, OBSERVATION_SLICES["expected_contact_mask"]], env.expected_contact_mask
    )
    assert observation["obs"].shape == (1, 476)
    assert reward.shape == reset.shape == (1,)
    assert np.all(np.isfinite(reward))


def test_source_counter_schedule_terminal_observation_and_delayed_reset(trajectory) -> None:
    env = _environment(trajectory)
    zero = np.zeros((1, 26), dtype=np.float64)
    clipped_reference = np.clip(trajectory.q_ref, env.joint_lower, env.joint_upper)
    env.step(zero)
    np.testing.assert_allclose(env.last_controller_targets[0], clipped_reference[0])
    np.testing.assert_array_equal(env.progress, [1])
    np.testing.assert_array_equal(env.trajectory_steps, [0])
    env.step(zero)
    np.testing.assert_allclose(env.last_controller_targets[0], clipped_reference[0])
    np.testing.assert_array_equal(env.trajectory_steps, [1])
    np.testing.assert_allclose(
        env.last_observation.raw[0, OBSERVATION_SLICES["target_object_position"]], trajectory.object_pos[1]
    )
    env.step(zero)
    np.testing.assert_allclose(env.last_controller_targets[0], clipped_reference[1])

    # Source state immediately before the final control call: target 789 is
    # commanded, post-step target 790 is observed, then reset is flagged.
    env.progress[:] = 790
    env.trajectory_steps[:] = 789
    terminal_obs, _, done, terminal_extras = env.step(zero)
    np.testing.assert_array_equal(done, [True])
    np.testing.assert_array_equal(terminal_extras["time_outs"], [False])
    assert env.last_transition is not None
    np.testing.assert_array_equal(env.last_transition.command_reference_indices, [789])
    np.testing.assert_array_equal(env.last_transition.target_indices, [790])
    np.testing.assert_array_equal(env.last_transition.reset_applied, [False])
    np.testing.assert_array_equal(env.last_transition.termination.reset, [True])
    np.testing.assert_array_equal(env.progress, [791])
    np.testing.assert_array_equal(env.trajectory_steps, [790])
    np.testing.assert_allclose(
        terminal_obs["obs"][0, OBSERVATION_SLICES["target_object_position"]], trajectory.object_pos[790]
    )
    _, _, next_done, _ = env.step(zero)
    np.testing.assert_array_equal(next_done, [False])
    assert env.last_transition is not None
    np.testing.assert_array_equal(env.last_transition.reset_applied, [True])
    np.testing.assert_array_equal(env.last_transition.target_indices, [0])
    np.testing.assert_array_equal(env.progress, [0])
    np.testing.assert_array_equal(env.trajectory_steps, [0])
    assert env.last_physical is not None
    np.testing.assert_allclose(env.last_physical.object_position[0], trajectory.object_pos[0], atol=1e-7)


def test_object_point_cloud_world_uses_metric_template_and_object_pose(trajectory) -> None:
    env = _environment(trajectory)
    env.step(np.zeros((1, 26), dtype=np.float64))
    assert env.last_physical is not None
    world_cloud = env.object_point_cloud_world()
    template = env._point_template()
    local = np.asarray(template.local_points, dtype=np.float64) * np.asarray(template.scale, dtype=np.float64)
    expected = quat_rotate_xyzw(
        np.broadcast_to(env.last_physical.object_orientation_xyzw[:, None, :], (1, 64, 4)),
        np.broadcast_to(local, (1, 64, 3)),
    ) + env.last_physical.object_position[:, None, :]
    np.testing.assert_allclose(world_cloud, expected, rtol=0.0, atol=1e-12)
    assert env.last_observation is not None
    normalized_hand_relative = env.last_observation.raw[:, OBSERVATION_SLICES["object_point_cloud_raw"]].reshape(1, 64, 3)
    assert not np.allclose(world_cloud, normalized_hand_relative + env.last_physical.hand_position[:, None, :])


def test_transition_snapshot_preserves_action_reference_and_rerun_artifact(trajectory, tmp_path) -> None:
    from sim.manorl.rerun_recorder import ManoRerunRecorder

    env = _environment(trajectory)
    action = np.zeros((1, 26), dtype=np.float64)
    action[0, 1] = 0.25
    env.step(action)
    snapshot = env.last_transition
    assert snapshot is not None
    assert snapshot.control_call == 0
    np.testing.assert_allclose(snapshot.raw_actions, action)
    np.testing.assert_array_equal(snapshot.command_reference_indices, [0])
    np.testing.assert_allclose(snapshot.command_targets, trajectory.q_ref[[0]])
    np.testing.assert_allclose(snapshot.controller_targets, env.last_controller_targets)
    recorder = ManoRerunRecorder(env, tmp_path / "env0.rrd")
    recorder.record_transition()
    assert recorder.close().stat().st_size > 0


def test_rerun_finalizes_terminal_episode_before_delayed_reset(trajectory, tmp_path) -> None:
    from sim.manorl.rerun_recorder import ManoRerunRecorder

    env = _environment(trajectory)
    recorder = ManoRerunRecorder(env, tmp_path / "episodes.rrd")
    env.progress[:] = len(trajectory.q_ref) - 2
    env.trajectory_steps[:] = len(trajectory.q_ref) - 3
    env.step(np.zeros((1, 26), dtype=np.float64))
    assert env.last_transition is not None
    assert bool(env.last_transition.termination.reset[0])
    recorder.record_transition()
    assert recorder.active_path.exists()
    assert not recorder.output.exists()
    env.step(np.zeros((1, 26), dtype=np.float64))
    assert env.last_transition is not None
    assert bool(env.last_transition.reset_applied[0])
    recorder.record_transition()
    assert recorder.output.name == "episodes.rrd"
    assert recorder.output.stat().st_size > 0
    assert recorder.active_path.exists()
    recorder.close()
    assert recorder.output.stat().st_size > 0
    assert not recorder.active_path.exists()


def test_residual_core_masks_inactive_fingers_in_live_environment(trajectory) -> None:
    env = _environment(trajectory, residual_enabled=True)
    env.progress[:] = 101
    env.trajectory_steps[:] = 100
    env.step(np.ones((1, 26), dtype=np.float64))
    assert np.all(np.abs(env.cumulative_joint_offset[0, :8]) > 0.0)
    np.testing.assert_allclose(env.cumulative_joint_offset[0, 8:], 0.0)
    assert np.all(np.abs(env.last_controller_targets[0, :3] - trajectory.q_ref[100, :3]) > 0.0)


def test_two_world_cpu_vector_smoke_has_independent_equal_worlds(trajectory) -> None:
    env = _environment(trajectory, num_envs=2)
    zero = np.zeros((2, 26), dtype=np.float64)
    for _ in range(3):
        observation, reward, reset, extras = env.step(zero)
        assert observation["obs"].shape == (2, 476)
        assert reward.shape == reset.shape == extras["time_outs"].shape == (2,)
        np.testing.assert_allclose(observation["obs"][0], observation["obs"][1], rtol=0, atol=1e-10)
        np.testing.assert_allclose(reward[0], reward[1], rtol=0, atol=1e-10)
    assert env.last_physical is not None
    assert np.all(env.last_physical.contact_count > 0)
    np.testing.assert_allclose(
        env.last_physical.hand_keypoint_contact_forces[0], env.last_physical.hand_keypoint_contact_forces[1], rtol=0, atol=1e-10
    )


def test_dynamic_template_variant_preserves_raw_surface_coordinates(trajectory) -> None:
    env = MujocoManoEnvironment(
        trajectory,
        EnvironmentConfig(
            compatibility=CHECKPOINT_SIDECAR_COMPATIBILITY,
            residual_enabled=False,
            max_deviation_distance=1_000_000.0,
        ),
    )
    assert env._dynamic_templates is not None
    first = env._dynamic_templates.copy()
    assert np.all(np.abs(first).max(axis=(1, 2)) < 0.1)
    env.reset(np.asarray([0], dtype=np.int64))
    assert env._dynamic_templates is not None
    assert not np.array_equal(first, env._dynamic_templates)
