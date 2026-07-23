from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from sim.manorl.device_runtime import (
    DevicePhysicalFeatures,
    advance_device_task_counters,
    check_device_termination,
    compute_device_reward_28,
    build_device_observation_28,
    extract_mjx_physical_features,
    jax_to_torch_cuda,
    reduce_warp_contacts,
    reduce_warp_contacts_for_right_policy,
    torch_to_jax_cuda,
)
from sim.manorl.environment import (
    EnvironmentConfig,
    MujocoManoEnvironment,
    _decode_contact_forces,
    _expand_legacy_hand_dofs,
    recommended_warp_contact_capacity,
)
from sim.manorl.abi import check_termination
from sim.manorl.rewards import RewardConfig, RewardState, compute_rewards
from sim.manorl.observations import (
    CURRENT_SOURCE_COMPATIBILITY,
    SOURCE_ALIGNED_COMPATIBILITY,
    ObservationState,
    PointCloudTemplate,
    build_observation,
)
from sim.manorl.trajectory import load_reference_trajectory


def _fixture() -> dict[str, object]:
    rng = np.random.default_rng(124)
    batch, capacity, ngeom = 4, 17, 24
    geom = rng.integers(0, ngeom, size=(capacity, 2), dtype=np.int32)
    # Explicitly exercise both hand-object orders and padded rows.
    geom[:4] = np.asarray(((0, 20), (20, 1), (2, 20), (20, 3)), dtype=np.int32)
    return {
        "nacon": np.asarray([capacity - 1], dtype=np.int32),
        "nefc": np.full(batch, 31, dtype=np.int32),
        "geom": geom,
        "world": rng.integers(0, batch, size=capacity, dtype=np.int32),
        "dimension": np.full(capacity, 3, dtype=np.int32),
        "addresses": rng.integers(0, 31, size=(capacity, 4), dtype=np.int32),
        "friction": rng.normal(size=(capacity, 5)).astype(np.float32),
        "frame": rng.normal(size=(capacity, 3, 3)).astype(np.float32),
        "constraint_force": rng.normal(size=(batch, 64)).astype(np.float32),
        "ngeom": ngeom,
        "keypoint_geom_ids": tuple(range(16)),
        "object_geom_ids": (20,),
    }


def _jitted_reducer(fixture: dict[str, object], *, compute_dtype: str = "float32"):
    static = {
        key: fixture[key]
        for key in ("ngeom", "keypoint_geom_ids", "object_geom_ids")
    }
    static["compute_dtype"] = compute_dtype
    dynamic = {key: value for key, value in fixture.items() if key not in static}
    fn = jax.jit(lambda **values: reduce_warp_contacts(**values, **static))
    return fn(**dynamic)


def test_jitted_physical_feature_gather_normalizes_xyzw_and_keeps_only_required_fields() -> None:
    rng = np.random.default_rng(52)
    batch, nq, nv, nbody = 3, 40, 38, 26
    qpos = rng.normal(size=(batch, nq)).astype(np.float32)
    qvel = rng.normal(size=(batch, nv)).astype(np.float32)
    xpos = rng.normal(size=(batch, nbody, 3)).astype(np.float32)
    xquat = rng.normal(size=(batch, nbody, 4)).astype(np.float32)
    keypoint_ids = tuple(range(3, 19))
    tips = (15, 3, 6, 9, 12)
    offsets = rng.normal(size=(5, 3)).astype(np.float32)
    fn = jax.jit(lambda q, v, p, r: extract_mjx_physical_features(
        qpos=q, qvel=v, xpos=p, xquat=r,
        hand_qpos_start=2, hand_dof=28, object_body_id=22,
        object_qvel_address=30, keypoint_body_ids=keypoint_ids,
        fingertip_keypoint_ids=tips, fingertip_local_offsets=offsets,
    ))
    actual = fn(qpos, qvel, xpos, xquat)
    expected_xyzw = xquat[:, keypoint_ids][:, :, (1, 2, 3, 0)]
    expected_xyzw /= np.linalg.norm(expected_xyzw, axis=-1, keepdims=True)
    np.testing.assert_allclose(np.asarray(actual.mano_dof_pos), qpos[:, 2:30], atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual.hand_keypoint_positions), xpos[:, keypoint_ids], atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual.hand_keypoint_orientations_xyzw), expected_xyzw, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(actual.object_orientation_xyzw), axis=1), 1.0, atol=2e-6)
    assert bool(actual.valid)


def test_jitted_28dof_observation_matches_numpy_at_contact_thresholds_and_partial_rows() -> None:
    rng = np.random.default_rng(98)
    batch = 4
    mano = rng.uniform(-0.7, 0.7, size=(batch, 28)).astype(np.float32)
    lower, upper = np.full(28, -1.0, np.float32), np.full(28, 1.0, np.float32)
    object_quat = rng.normal(size=(batch, 4)).astype(np.float32)
    object_quat /= np.linalg.norm(object_quat, axis=1, keepdims=True)
    hand_quat = rng.normal(size=(batch, 4)).astype(np.float32)
    hand_quat /= np.linalg.norm(hand_quat, axis=1, keepdims=True)
    keypoint_quat = np.broadcast_to(hand_quat[:, None], (batch, 16, 4)).copy()
    object_pos = rng.normal(size=(batch, 3)).astype(np.float32)
    hand_pos = rng.normal(size=(batch, 3)).astype(np.float32)
    keypoints = rng.normal(size=(batch, 16, 3)).astype(np.float32)
    tips = rng.normal(size=(batch, 5, 3)).astype(np.float32)
    physical = DevicePhysicalFeatures(
        mano, hand_pos, hand_quat, keypoint_quat, object_pos, object_quat,
        rng.normal(size=(batch, 3)).astype(np.float32), keypoints, tips, jax.numpy.asarray(True),
    )
    forces = np.zeros((batch, 16, 3), dtype=np.float32)
    forces[:, 3, 0] = np.asarray([0.199999, 0.2, 0.200001, 10.0], dtype=np.float32)
    # Dynamic reset templates are per-environment; this pins that device
    # broadcasting does not accidentally reuse environment zero's cloud.
    points = rng.normal(size=(batch, 64, 3)).astype(np.float32)
    scale = rng.uniform(0.2, 0.4, size=(batch, 3)).astype(np.float32)
    support = rng.normal(size=(13, 3)).astype(np.float32)
    target_pos = rng.normal(size=(batch, 3)).astype(np.float32)
    target_quat = rng.normal(size=(batch, 4)).astype(np.float32)
    target_quat *= 2.5  # Device path must normalize target XYZW exactly as NumPy does for support points.
    expected = np.zeros((batch, 16), dtype=np.float32)
    expected[:, (3, 15)] = 1.0
    kwargs = dict(
        physical=physical, hand_keypoint_contact_forces=forces,
        target_object_position=target_pos, target_object_orientation_xyzw=target_quat,
        target_object_pos_next_5=rng.normal(size=(batch, 3)).astype(np.float32),
        cumulative_offset=rng.normal(size=(batch, 3)).astype(np.float32),
        cumulative_joint_offset=rng.normal(size=(batch, 22)).astype(np.float32),
        point_cloud_local=points, point_cloud_scale=scale,
        object_geometry=rng.normal(size=(batch, 12)).astype(np.float32),
        expected_contact_mask=expected, action_ids=np.asarray([1, 2, 49, 50], dtype=np.int32),
        object_support_points=support, table_surface_height=-0.001,
        mano_dof_lower=lower, mano_dof_upper=upper,
    )
    raw, valid = jax.jit(lambda: build_device_observation_28(**kwargs))()
    numpy_state = ObservationState(
        mano_dof_pos=mano, mano_dof_lower=lower, mano_dof_upper=upper,
        hand_position=hand_pos, hand_orientation_xyzw=hand_quat,
        object_position=object_pos, object_orientation_xyzw=object_quat,
        target_object_position=target_pos,
        target_object_orientation_xyzw=target_quat,
        target_object_pos_next_5=kwargs["target_object_pos_next_5"],
        cumulative_offset=kwargs["cumulative_offset"],
        cumulative_joint_offset=kwargs["cumulative_joint_offset"],
        point_cloud=PointCloudTemplate(points, mode="dynamic_reset", normalized=True, scale=scale),
        object_geometry=kwargs["object_geometry"], hand_keypoint_positions=keypoints,
        fingertip_positions=tips, hand_keypoint_contact_forces=forces,
        expected_contact_mask=expected, action_ids=kwargs["action_ids"],
        object_support_points=support, table_surface_height=-0.001,
    )
    expected_observation = build_observation(numpy_state, compatibility=SOURCE_ALIGNED_COMPATIBILITY)
    np.testing.assert_allclose(np.asarray(raw), expected_observation.raw, rtol=0, atol=2e-6)
    np.testing.assert_allclose(np.clip(np.asarray(raw), -5.0, 5.0), expected_observation.policy_input, rtol=0, atol=2e-6)
    assert bool(valid)
    # Match NumPy's strict 0.2N gate after its float32-to-float64 promotion.
    # 28D layout shifts the fixed 26D direction slice by four finger channels.
    assert np.linalg.norm(np.asarray(raw)[0, 416 + 3 * 3 : 416 + 3 * 4]) == 0.0
    assert np.linalg.norm(np.asarray(raw)[1, 416 + 3 * 3 : 416 + 3 * 4]) > 0.9


def test_builder_rejects_unordered_limits_and_nonfinite_table_height() -> None:
    """Mirror the host encoder's fail-closed limit/table validation."""
    rng = np.random.default_rng(11)
    batch = 2
    physical = DevicePhysicalFeatures(
        rng.normal(size=(batch, 28)).astype(np.float32), rng.normal(size=(batch, 3)).astype(np.float32),
        np.tile(np.asarray((0, 0, 0, 1), np.float32), (batch, 1)),
        np.tile(np.asarray((0, 0, 0, 1), np.float32), (batch, 16, 1)),
        rng.normal(size=(batch, 3)).astype(np.float32), np.tile(np.asarray((0, 0, 0, 1), np.float32), (batch, 1)),
        rng.normal(size=(batch, 3)).astype(np.float32), rng.normal(size=(batch, 16, 3)).astype(np.float32),
        rng.normal(size=(batch, 5, 3)).astype(np.float32), jax.numpy.asarray(True),
    )
    kwargs = dict(
        physical=physical, hand_keypoint_contact_forces=np.zeros((batch, 16, 3), np.float32),
        target_object_position=np.zeros((batch, 3), np.float32), target_object_orientation_xyzw=np.tile(np.asarray((0, 0, 0, 1), np.float32), (batch, 1)),
        target_object_pos_next_5=np.zeros((batch, 3), np.float32), cumulative_offset=np.zeros((batch, 3), np.float32),
        cumulative_joint_offset=np.zeros((batch, 22), np.float32), point_cloud_local=np.zeros((64, 3), np.float32),
        point_cloud_scale=np.ones(3, np.float32), object_geometry=np.zeros((batch, 12), np.float32),
        expected_contact_mask=np.zeros((batch, 16), np.float32), action_ids=np.ones(batch, np.int32),
        object_support_points=np.zeros((1, 3), np.float32), mano_dof_lower=np.zeros(28, np.float32),
        mano_dof_upper=np.ones(28, np.float32), table_surface_height=0.0,
    )
    invalid_limits = dict(kwargs, mano_dof_upper=np.zeros(28, np.float32))
    assert not bool(jax.jit(lambda: build_device_observation_28(**invalid_limits))()[1])
    invalid_table = dict(kwargs, table_surface_height=np.nan)
    assert not bool(jax.jit(lambda: build_device_observation_28(**invalid_table))()[1])


def test_physical_feature_offsets_preserve_left_handedness_flip() -> None:
    """The caller owns the left-side X flip exactly as materialize_state does."""
    qpos = np.zeros((1, 32), np.float32)
    qvel = np.zeros((1, 32), np.float32)
    xpos = np.zeros((1, 20, 3), np.float32)
    xquat = np.zeros((1, 20, 4), np.float32)
    xquat[..., 0] = 1.0  # identity in source wxyz order
    offsets = np.asarray(((0.1, 0.2, 0.3),) * 5, np.float32)
    common = dict(qpos=qpos, qvel=qvel, xpos=xpos, xquat=xquat, hand_qpos_start=0, hand_dof=28,
                  object_body_id=19, object_qvel_address=0, keypoint_body_ids=tuple(range(16)),
                  fingertip_keypoint_ids=(0, 1, 2, 3, 4))
    right = extract_mjx_physical_features(**common, fingertip_local_offsets=offsets)
    left_offsets = offsets.copy(); left_offsets[:, 0] *= -1
    left = extract_mjx_physical_features(**common, fingertip_local_offsets=left_offsets)
    np.testing.assert_allclose(np.asarray(right.fingertip_positions)[..., 1:], np.asarray(left.fingertip_positions)[..., 1:])
    np.testing.assert_allclose(np.asarray(right.fingertip_positions)[..., 0], -np.asarray(left.fingertip_positions)[..., 0])


def test_jitted_device_reward_and_termination_match_numpy_contract() -> None:
    rng = np.random.default_rng(401)
    batch = 7
    q = rng.normal(size=(batch, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    target_q = rng.normal(size=(batch, 4)); target_q /= np.linalg.norm(target_q, axis=1, keepdims=True)
    state = dict(
        object_position=rng.normal(size=(batch, 3)).astype(np.float32), target_object_position=rng.normal(size=(batch, 3)).astype(np.float32),
        object_orientation_xyzw=q.astype(np.float32), target_object_orientation_xyzw=target_q.astype(np.float32),
        cumulative_offset=rng.normal(size=(batch, 3)).astype(np.float32), cumulative_joint_offset=rng.normal(size=(batch, 22)).astype(np.float32),
        active_joint_mask=rng.integers(0, 2, size=(batch, 22), dtype=np.int8).astype(bool),
        hand_object_force_on_object_world_N=rng.normal(size=(batch, 16, 3)).astype(np.float32),
        expected_contact_mask=rng.integers(0, 2, size=(batch, 16)).astype(np.float32), expected_contact_weights=rng.uniform(0.1, 2, size=(batch, 16)).astype(np.float32),
        object_linear_velocity=rng.normal(size=(batch, 3)).astype(np.float32), trajectory_steps=np.asarray((0, 5, 20, 21, 89, 90, 120), np.int64),
        contact_start_frames=np.full(batch, 20, np.int64), contact_end_frames=np.full(batch, 90, np.int64),
        rotation_disabled_mask=np.asarray((False, True, False, False, False, False, False)), early_phase_starts=np.zeros(batch, np.int64),
    )
    progress = np.asarray((0, 1, 2, 3, 4, 5, 6), np.int64)
    lengths = np.asarray((10, 2, 10, 10, 10, 10, 7), np.int64)
    early = state["trajectory_steps"] < 30
    host_term = check_termination(object_position=state["object_position"], target_position=state["target_object_position"], progress=progress, trajectory_lengths=lengths, early_mask=early, max_deviation_distance=1.2, deviation_penalty=0.7)
    device_term = jax.jit(lambda: check_device_termination(object_position=state["object_position"], target_position=state["target_object_position"], progress=progress, trajectory_lengths=lengths, early_mask=early, max_deviation_distance=1.2, deviation_penalty=0.7))()
    np.testing.assert_array_equal(np.asarray(device_term.reset), host_term.reset)
    np.testing.assert_array_equal(np.asarray(device_term.deviation_reset), host_term.deviation_reset)
    np.testing.assert_array_equal(np.asarray(device_term.reason_code), host_term.reason_code)
    host = compute_rewards(RewardState(**state), compatibility=SOURCE_ALIGNED_COMPATIBILITY, termination=host_term, config=RewardConfig())
    device = jax.jit(lambda: compute_device_reward_28(**state, early_phase_steps=30, termination=device_term, config=RewardConfig()))()
    for field in (
        "total", "distance_x", "distance_y", "distance_z",
        "ungated_distance_x", "ungated_distance_y", "ungated_distance_z",
        "rotation", "position_penalty", "joint_penalty", "action_penalty",
        "raw_contact", "contact", "distance_gate", "object_stability",
        "object_speed", "survival", "deviation_penalty",
    ):
        np.testing.assert_allclose(np.asarray(getattr(device, field)), getattr(host, field), rtol=0, atol=2e-6)
    np.testing.assert_array_equal(np.asarray(device.early_phase), host.early_phase)


def test_device_reward_zero_stability_speed_and_invalid_inputs_fail_closed() -> None:
    """Host's non-positive speed guard and validation must survive JIT lowering."""
    batch = 2
    common = dict(
        object_position=np.zeros((batch, 3), np.float32), target_object_position=np.zeros((batch, 3), np.float32),
        object_orientation_xyzw=np.tile(np.asarray((0, 0, 0, 1), np.float32), (batch, 1)),
        target_object_orientation_xyzw=np.tile(np.asarray((0, 0, 0, 1), np.float32), (batch, 1)),
        cumulative_offset=np.zeros((batch, 3), np.float32), cumulative_joint_offset=np.zeros((batch, 22), np.float32),
        active_joint_mask=np.zeros((batch, 22), bool), hand_object_force_on_object_world_N=np.zeros((batch, 16, 3), np.float32),
        expected_contact_mask=np.zeros((batch, 16), np.float32), expected_contact_weights=np.ones((batch, 16), np.float32),
        object_linear_velocity=np.zeros((batch, 3), np.float32), trajectory_steps=np.full(batch, 2, np.int64),
        contact_start_frames=np.zeros(batch, np.int64), contact_end_frames=np.ones(batch, np.int64),
        rotation_disabled_mask=np.zeros(batch, bool), early_phase_starts=np.zeros(batch, np.int64), early_phase_steps=0,
    )
    term = check_device_termination(
        object_position=common["object_position"], target_position=common["target_object_position"],
        progress=np.zeros(batch, np.int64), trajectory_lengths=np.full(batch, 9, np.int64),
        early_mask=np.zeros(batch, bool), max_deviation_distance=1.0, deviation_penalty=0.0,
    )
    zero_speed = RewardConfig(object_stability_reference_speed=0.0)
    actual = jax.jit(lambda: compute_device_reward_28(**common, termination=term, config=zero_speed))()
    assert bool(actual.valid)
    np.testing.assert_array_equal(np.asarray(actual.object_stability), np.zeros(batch))
    invalid = dict(common, expected_contact_mask=np.full((batch, 16), 0.5, np.float32))
    rejected = jax.jit(lambda: compute_device_reward_28(**invalid, termination=term, config=zero_speed))()
    assert not bool(rejected.valid)
    invalid_velocity = dict(common, object_linear_velocity=np.full((batch, 3), np.nan, np.float32))
    assert not bool(jax.jit(lambda: compute_device_reward_28(**invalid_velocity, termination=term, config=zero_speed))().valid)
    invalid_term = check_device_termination(
        object_position=np.full((batch, 3), np.nan, np.float32), target_position=common["target_object_position"],
        progress=np.zeros(batch, np.int64), trajectory_lengths=np.full(batch, 9, np.int64),
        early_mask=np.zeros(batch, bool), max_deviation_distance=1.0, deviation_penalty=0.0,
    )
    assert not bool(invalid_term.valid)


def test_device_counters_match_partial_delayed_reset_order() -> None:
    progress = np.asarray((0, 3, 8), np.int64)
    steps = np.asarray((0, 3, 8), np.int64)
    returns = np.asarray((1.0, 2.0, 3.0), np.float32)
    pending = np.asarray((False, True, False))
    reward = np.asarray((0.1, 0.2, 0.3), np.float32)
    reset = np.asarray((False, False, True))
    actual = jax.jit(lambda: advance_device_task_counters(progress=progress, trajectory_steps=steps, episode_returns=returns, pending_reset=pending, reward_total=reward, next_reset=reset, control_call=np.asarray(9, np.int64)))()
    np.testing.assert_array_equal(np.asarray(actual.progress), np.asarray((1, 0, 9)))
    np.testing.assert_array_equal(np.asarray(actual.trajectory_steps), np.asarray((0, 0, 9)))
    np.testing.assert_allclose(np.asarray(actual.episode_returns), np.asarray((1.1, 0.2, 3.3)), atol=2e-6)
    np.testing.assert_array_equal(np.asarray(actual.reset_mask), reset)
    assert int(np.asarray(actual.control_call)) == 10


def test_device_transition_matches_numpy_across_delayed_partial_reset_cycles() -> None:
    """Pin reset-before-next-reward boundaries over several mixed worlds."""
    batch = 3
    progress = np.zeros(batch, np.int64)
    steps = np.zeros(batch, np.int64)
    returns = np.zeros(batch, np.float32)
    pending = np.zeros(batch, bool)
    call = np.asarray(0, np.int64)
    lengths = np.asarray((2, 5, 3), np.int64)
    for cycle, positions in enumerate((
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        ((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0),) * 3,
    )):
        pre_progress = progress + 1
        pre_steps = steps + 1
        pre_steps[progress == 0] = 0
        progress_after_reset = pre_progress.copy(); progress_after_reset[pending] = 0
        steps_after_reset = pre_steps.copy(); steps_after_reset[pending] = 0
        host = check_termination(object_position=np.asarray(positions), target_position=np.zeros((batch, 3)), progress=progress_after_reset, trajectory_lengths=lengths, early_mask=np.zeros(batch, bool), max_deviation_distance=1.0, deviation_penalty=0.25)
        device_term = jax.jit(lambda p=np.asarray(positions), q=progress_after_reset: check_device_termination(object_position=p, target_position=np.zeros((batch, 3)), progress=q, trajectory_lengths=lengths, early_mask=np.zeros(batch, bool), max_deviation_distance=1.0, deviation_penalty=0.25))()
        np.testing.assert_array_equal(np.asarray(device_term.reset), host.reset)
        np.testing.assert_array_equal(np.asarray(device_term.reason_code), host.reason_code)
        reward = np.full(batch, cycle + 0.125, np.float32)
        actual = jax.jit(lambda: advance_device_task_counters(progress=progress, trajectory_steps=steps, episode_returns=returns, pending_reset=pending, reward_total=reward, next_reset=device_term.reset, control_call=call))()
        expected_returns = returns.copy(); expected_returns[pending] = 0; expected_returns += reward
        np.testing.assert_array_equal(np.asarray(actual.progress), progress_after_reset)
        np.testing.assert_array_equal(np.asarray(actual.trajectory_steps), steps_after_reset)
        np.testing.assert_allclose(np.asarray(actual.episode_returns), expected_returns, atol=2e-6)
        progress, steps, returns, pending, call = (np.asarray(actual.progress), np.asarray(actual.trajectory_steps), np.asarray(actual.episode_returns), np.asarray(actual.reset_mask), np.asarray(actual.control_call))


@pytest.mark.skipif(
    os.environ.get("MANORL_RUN_DEVICE_TRANSITION_PARITY") != "1",
    reason="device-transition host oracle is opt-in and requires configured CUDA JAX/MJX",
)
def test_device_transition_matches_host_oracle_over_forced_reset_branches() -> None:
    """Exercise the actual environment seam before a remote throughput run.

    The reference targets force a terminal world, a post-window/contact-reward
    world, and a deviation world.  We compare physical state and references
    before policy outputs because output agreement alone can hide divergent
    simulator state.
    """
    if jax.default_backend() != "gpu":
        pytest.skip("configured CUDA JAX backend required")
    batch = 4
    legacy_trajectory = load_reference_trajectory()
    right_q_ref = _expand_legacy_hand_dofs(legacy_trajectory.q_ref)
    left_q_ref = right_q_ref + 1e-3
    trajectory = replace(
        legacy_trajectory,
        q_ref=right_q_ref,
        q_ref_by_side={"right": right_q_ref, "left": left_q_ref},
        hand_sides=("right", "left"),
        selected_hand_sides=("right",),
    )
    assert not np.array_equal(trajectory.q_ref_by_side["right"], trajectory.q_ref_by_side["left"])
    compatibility = replace(SOURCE_ALIGNED_COMPATIBILITY, early_phase_steps=0)
    reward_config = replace(RewardConfig(), contact_force_threshold=-1.0)
    common = dict(
        num_envs=batch, device="gpu", device_resident_controls=True,
        contact_capacity=recommended_warp_contact_capacity(batch, trajectory.hand_sides),
        capture_transition_diagnostics=False, compatibility=compatibility,
        point_sampling_backend="numpy_per_env", reward_config=reward_config,
        max_deviation_distance=0.05,
    )
    host = MujocoManoEnvironment(trajectory, EnvironmentConfig(**common))
    device = MujocoManoEnvironment(trajectory, EnvironmentConfig(**common, device_transition=True))
    for environment in (host, device):
        assert environment.model_action_dim == 56
        assert environment.action_dim == 28
        assert environment.observation_dim == 480
    # Mutate before the device branch retains its immutable tables.  This makes
    # each requested boundary deterministic rather than relying on a contact
    # realization from the random action trace.
    for environment in (host, device):
        environment.trajectory_lengths[:] = np.asarray((2, 80, 80, 80), dtype=np.int64)
        environment.contact_start_frames[:] = 0
        environment.contact_end_frames[:] = 0
        environment.expected_contact_mask[:] = 1.0
        environment.expected_contact_weights[:] = 1.0
        environment.reference_object_pos[2] += np.asarray((10.0, 0.0, 0.0))
    rng = np.random.default_rng(20260723)
    saw_terminal = saw_deviation = saw_contact = saw_post_window = False
    for step in range(64):
        if step in (16, 43):
            ids = np.asarray((1, 3), dtype=np.int64)
            host.reset(env_ids=ids)
            device.reset(env_ids=ids)
        actions = rng.uniform(-0.25, 0.25, size=(batch, host.action_dim))
        host_output, host_reward, host_done, host_extras = host.step(actions)
        device_transition = device.step_device(actions)
        assert device_transition.observation.shape == (batch, 480)
        assert device_transition.observation.dtype == jax.numpy.float32
        assert device_transition.reward.shape == (batch,)
        assert device_transition.reset.shape == (batch,)
        assert device_transition.reason_code.shape == (batch,)
        assert device_transition.deviation_reset.shape == (batch,)
        assert bool(np.asarray(device_transition.valid))
        # The public Gym method retains its NumPy ABI by materializing this
        # exact narrow egress; parity itself deliberately uses the device seam.
        device_output = {"obs": np.asarray(device_transition.observation, dtype=np.float32)}
        device_reward = np.asarray(device_transition.reward, dtype=np.float32)
        device_done = np.asarray(device_transition.reset, dtype=bool)
        device_extras = device._device_transition_extras()
        host_physical = host.producer.extract(host.data)
        device_physical = device.producer.extract(device.data)
        for field in ("object_position", "object_orientation_xyzw", "mano_dof_pos", "object_linear_velocity"):
            np.testing.assert_allclose(
                getattr(device_physical, field), getattr(host_physical, field), rtol=1e-4, atol=1e-5
            )
        np.testing.assert_allclose(np.asarray(device.data.qpos), np.asarray(host.data.qpos), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.asarray(device.data.qvel), np.asarray(host.data.qvel), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.asarray(device.data.ctrl), np.asarray(host.data.ctrl), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(device.data.qpos)[:, 28:56], np.asarray(host.data.qpos)[:, 28:56], rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(device.data.ctrl)[:, 28:56], np.asarray(host.data.ctrl)[:, 28:56], rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(device._target_indices(), host._target_indices(), rtol=0, atol=0)
        np.testing.assert_allclose(device.reference_object_pos, host.reference_object_pos, rtol=0, atol=0)
        np.testing.assert_array_equal(device_done, host_done)
        np.testing.assert_array_equal(device_extras["termination_reason_code"], host_extras["termination_reason_code"])
        observation_slices = host.observation_layout.slices
        for name, observation_slice in observation_slices.items():
            if name in {"contact_forces", "contact_force_directions"}:
                continue
            np.testing.assert_allclose(
                device_output["obs"][:, observation_slice],
                host_output["obs"][:, observation_slice],
                rtol=1e-4,
                atol=1e-5,
                err_msg=name,
            )
        contact_force_slice = observation_slices["contact_forces"]
        np.testing.assert_allclose(
            device_output["obs"][:, contact_force_slice],
            host_output["obs"][:, contact_force_slice],
            rtol=1e-4,
            atol=3e-4,
            err_msg="contact_forces",
        )
        direction_slice = observation_slices["contact_force_directions"]
        device_directions = device_output["obs"][:, direction_slice].reshape(batch, -1, 3)
        host_directions = host_output["obs"][:, direction_slice].reshape(batch, -1, 3)
        direction_gate = np.any(device_directions != 0.0, axis=-1)
        np.testing.assert_array_equal(direction_gate, np.any(host_directions != 0.0, axis=-1))
        np.testing.assert_allclose(
            device_directions[direction_gate],
            host_directions[direction_gate],
            rtol=1e-4,
            atol=2e-4,
            err_msg="contact_force_directions",
        )
        np.testing.assert_allclose(device_reward, host_reward, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(device.progress, host.progress)
        np.testing.assert_array_equal(device.trajectory_steps, host.trajectory_steps)
        np.testing.assert_array_equal(device.reset_mask, host.reset_mask)
        np.testing.assert_allclose(device.episode_returns, host.episode_returns, rtol=1e-4, atol=1e-5)
        assert device.last_reward is not None
        saw_terminal |= bool(np.any(device.last_termination.success))
        saw_deviation |= bool(np.any(device.last_termination.failure))
        saw_contact |= bool(np.any(device.last_reward.raw_contact > 0.0))
        saw_post_window |= bool(np.any(device.trajectory_steps > device.contact_end_frames))
    assert saw_terminal and saw_deviation and saw_contact and saw_post_window


def test_dual_contact_reducer_keeps_right_observation_and_aggregates_reward_force() -> None:
    fixture = _fixture()
    fixture["ngeom"] = 48
    fixture["keypoint_geom_ids"] = tuple(range(16))
    fixture["object_geom_ids"] = (40,)
    fixture["nacon"] = np.asarray([2], dtype=np.int32)
    fixture["geom"] = np.asarray(
        [(0, 40), (24, 40)] + [(0, 0)] * 15, dtype=np.int32
    )
    fixture["world"] = np.zeros(17, dtype=np.int32)
    fixture["addresses"] = np.zeros((17, 4), dtype=np.int32)
    fixture["friction"] = np.zeros((17, 5), dtype=np.float32)
    fixture["frame"] = np.broadcast_to(np.eye(3, dtype=np.float32), (17, 3, 3)).copy()
    constraint_force = np.zeros((4, 64), dtype=np.float32)
    constraint_force[0, 0] = 2.0
    fixture["constraint_force"] = constraint_force
    static = {
        key: fixture[key]
        for key in ("ngeom", "object_geom_ids")
    }
    dynamic = {
        key: value for key, value in fixture.items()
        if key not in static and key != "keypoint_geom_ids"
    }
    actual = jax.jit(lambda **values: reduce_warp_contacts_for_right_policy(
        **values,
        **static,
        right_keypoint_geom_ids=tuple(range(16)),
        left_keypoint_geom_ids=tuple(range(24, 40)),
    ))(**dynamic)
    right = _jitted_reducer(fixture)
    np.testing.assert_allclose(np.asarray(actual.keypoint_forces), np.asarray(right.keypoint_forces))
    np.testing.assert_allclose(np.asarray(actual.keypoint_forces)[0, 0], (-2.0, 0.0, 0.0))
    np.testing.assert_allclose(np.asarray(actual.hand_object_forces)[0, 0], (4.0, 0.0, 0.0))
    np.testing.assert_array_equal(np.asarray(actual.per_world_count), (2, 0, 0, 0))
    assert bool(actual.valid) is True


def test_jitted_contact_reduction_matches_numpy_decoder_and_masks_capacity() -> None:
    fixture = _fixture()
    actual = _jitted_reducer(fixture)
    count = int(np.asarray(fixture["nacon"])[0])
    expected = _decode_contact_forces(
        count=count,
        geom=np.asarray(fixture["geom"]),
        world=np.asarray(fixture["world"]),
        dimension=np.asarray(fixture["dimension"]),
        addresses=np.asarray(fixture["addresses"]),
        nefc=np.asarray(fixture["nefc"]),
        friction=np.asarray(fixture["friction"]),
        frame=np.asarray(fixture["frame"]),
        constraint_force=np.asarray(fixture["constraint_force"]),
        ngeom=int(fixture["ngeom"]),
        keypoint_geom_ids=fixture["keypoint_geom_ids"],
        object_geom_ids=set(fixture["object_geom_ids"]),
    )
    np.testing.assert_allclose(
        np.asarray(actual.keypoint_forces), expected[0][:, :16], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(actual.hand_object_forces), expected[1], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(actual.per_world_count), expected[2])
    assert bool(actual.valid) is True


@pytest.mark.parametrize(
    ("field", "index", "value"),
    (("dimension", 0, 4), ("world", 0, -1), ("geom", (0, 0), 99), ("addresses", (0, 0), 31)),
)
def test_contact_reduction_reports_malformed_live_rows_without_unsafe_gathers(
    field: str, index: object, value: int
) -> None:
    fixture = _fixture()
    changed = np.asarray(fixture[field]).copy()
    changed[index] = value  # type: ignore[index]
    fixture[field] = changed
    actual = _jitted_reducer(fixture)
    assert bool(actual.valid) is False
    assert np.isfinite(np.asarray(actual.keypoint_forces)).all()
    assert np.isfinite(np.asarray(actual.hand_object_forces)).all()


def test_contact_reduction_rejects_capacity_saturation() -> None:
    fixture = _fixture()
    fixture["nacon"] = np.asarray([len(np.asarray(fixture["world"]))], dtype=np.int32)
    actual = _jitted_reducer(fixture)
    assert bool(actual.valid) is False


def test_scalar_nefc_broadcast_matches_host_decoder() -> None:
    fixture = _fixture()
    fixture["nefc"] = np.asarray([31], dtype=np.int32)
    actual = _jitted_reducer(fixture)
    count = int(np.asarray(fixture["nacon"])[0])
    expected = _decode_contact_forces(
        count=count,
        geom=np.asarray(fixture["geom"]),
        world=np.asarray(fixture["world"]),
        dimension=np.asarray(fixture["dimension"]),
        addresses=np.asarray(fixture["addresses"]),
        nefc=np.full(4, 31, dtype=np.int64),
        friction=np.asarray(fixture["friction"]),
        frame=np.asarray(fixture["frame"]),
        constraint_force=np.asarray(fixture["constraint_force"]),
        ngeom=int(fixture["ngeom"]),
        keypoint_geom_ids=fixture["keypoint_geom_ids"],
        object_geom_ids=set(fixture["object_geom_ids"]),
    )
    np.testing.assert_allclose(np.asarray(actual.keypoint_forces), expected[0][:, :16], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual.hand_object_forces), expected[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual.per_world_count), expected[2])
    assert bool(actual.valid) is True


def test_nonfinite_live_floor_contact_fails_closed_like_host_decoder() -> None:
    fixture = _fixture()
    # Geom 21 is neither a hand keypoint nor the object geom. The row remains
    # live and therefore must fail before reduction hides it from outputs.
    fixture["geom"] = np.asarray(fixture["geom"]).copy()
    fixture["geom"][0] = (21, 22)
    fixture["constraint_force"] = np.asarray(fixture["constraint_force"]).copy()
    addresses = np.asarray(fixture["addresses"])
    fixture["constraint_force"][np.asarray(fixture["world"])[0], addresses[0, 0]] = np.nan
    actual = _jitted_reducer(fixture)
    assert bool(actual.valid) is False
    with pytest.raises(RuntimeError, match="non-finite world force"):
        _decode_contact_forces(
            count=int(np.asarray(fixture["nacon"])[0]),
            geom=np.asarray(fixture["geom"]),
            world=np.asarray(fixture["world"]),
            dimension=np.asarray(fixture["dimension"]),
            addresses=addresses,
            nefc=np.asarray(fixture["nefc"]),
            friction=np.asarray(fixture["friction"]),
            frame=np.asarray(fixture["frame"]),
            constraint_force=np.asarray(fixture["constraint_force"]),
            ngeom=int(fixture["ngeom"]),
            keypoint_geom_ids=fixture["keypoint_geom_ids"],
            object_geom_ids=set(fixture["object_geom_ids"]),
        )


def test_real_mjx_warp_private_buffers_match_host_decoder() -> None:
    """Pin the actual DataWarp field names/shapes and reduction parity."""

    if jax.config.x64_enabled:
        pytest.skip("MJX-Warp 3.10 FFI pins its model buffers to float32")
    try:
        jax.devices("cpu")
    except RuntimeError as error:
        pytest.skip(f"JAX CPU backend is unavailable: {error}")
    environment = MujocoManoEnvironment(
        load_reference_trajectory(),
        EnvironmentConfig(num_envs=2, device="cpu", max_deviation_distance=1_000_000.0),
    )
    buffers = environment.producer.materialize_contact_buffers(environment.data, batch=2)
    host_geometry, host_hand_object, host_counts = environment.producer.decode_contact_buffers(buffers)
    keypoint, hand_object, counts = environment.producer._device_decode_contact_buffers(
        environment.data, batch=2
    )
    assert buffers.count < buffers.capacity
    np.testing.assert_allclose(
        keypoint,
        host_geometry[:, environment.producer.keypoint_geom_ids],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(hand_object, host_hand_object, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(counts, host_counts)


def test_device_contact_decode_config_fails_closed_for_cpu_and_debug_snapshots() -> None:
    with pytest.raises(ValueError, match="requires device='gpu'"):
        EnvironmentConfig(device="cpu", device_contact_decode=True, capture_transition_diagnostics=False)
    with pytest.raises(ValueError, match="capture_transition_diagnostics=False"):
        EnvironmentConfig(device="gpu", device_contact_decode=True)


def test_device_transition_config_rejects_debug_cpu_and_host_controls() -> None:
    with pytest.raises(ValueError, match="device='gpu'"):
        EnvironmentConfig(device_transition=True, device_resident_controls=True, capture_transition_diagnostics=False)
    with pytest.raises(ValueError, match="device_resident_controls=True"):
        EnvironmentConfig(device="gpu", device_transition=True, capture_transition_diagnostics=False)
    with pytest.raises(ValueError, match="capture_transition_diagnostics=False"):
        EnvironmentConfig(device="gpu", device_transition=True, device_resident_controls=True)


def test_float64_reducer_matches_float64_host_decoder() -> None:
    if not jax.config.x64_enabled:
        pytest.skip("JAX x64 is disabled")
    fixture = _fixture()
    for key in ("friction", "frame", "constraint_force"):
        fixture[key] = np.asarray(fixture[key], dtype=np.float64)
    actual = _jitted_reducer(fixture, compute_dtype="float64")
    expected = _decode_contact_forces(
        count=int(np.asarray(fixture["nacon"])[0]),
        geom=np.asarray(fixture["geom"]),
        world=np.asarray(fixture["world"]),
        dimension=np.asarray(fixture["dimension"]),
        addresses=np.asarray(fixture["addresses"]),
        nefc=np.asarray(fixture["nefc"]),
        friction=np.asarray(fixture["friction"]),
        frame=np.asarray(fixture["frame"]),
        constraint_force=np.asarray(fixture["constraint_force"]),
        ngeom=int(fixture["ngeom"]),
        keypoint_geom_ids=fixture["keypoint_geom_ids"],
        object_geom_ids=set(fixture["object_geom_ids"]),
    )
    np.testing.assert_allclose(np.asarray(actual.keypoint_forces), expected[0][:, :16], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(actual.hand_object_forces), expected[1], rtol=1e-12, atol=1e-12)
    assert bool(actual.valid) is True


@pytest.mark.skipif(
    os.environ.get("MANORL_RUN_CUDA_INTEROP") != "1",
    reason="CUDA DLPack boundary is opt-in; focused CPU validation must not submit GPU work",
)
def test_cuda_dlpack_round_trip_retains_device_and_storage() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    jax_array = torch_to_jax_cuda(source)
    round_trip = jax_to_torch_cuda(jax_array)
    assert jax_array.device.platform in {"gpu", "cuda"}
    assert round_trip.is_cuda
    assert round_trip.device.index == source.device.index
    np.testing.assert_array_equal(round_trip.detach().cpu().numpy(), source.detach().cpu().numpy())
