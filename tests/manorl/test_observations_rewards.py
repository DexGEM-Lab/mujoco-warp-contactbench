from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim.manorl.abi import check_termination, early_phase_mask
from sim.manorl.observations import (
    CHECKPOINT_SIDECAR_COMPATIBILITY,
    CURRENT_SOURCE_COMPATIBILITY,
    OBSERVATION_KEYS,
    OBSERVATION_SLICES,
    POINT_COUNT,
    RAW_OBSERVATION_DIM,
    ObservationState,
    PointCloudTemplate,
    action_type_one_hot,
    build_observation,
    expected_contact_mask_from_keypoint_ids,
    extract_contact_features,
    geometry_encoding,
    object_category,
)
from sim.manorl.rewards import (
    REWARD_CONTRACT_ID,
    REWARD_HAND_OBJECT_THRESHOLD_N,
    RewardState,
    compute_rewards,
)


def _static_point_cloud() -> PointCloudTemplate:
    points = np.zeros((POINT_COUNT, 3), dtype=np.float64)
    points[0] = [0.1, 0.2, 0.3]
    points[-1] = [-0.4, 0.5, -0.6]
    return PointCloudTemplate(points, mode="static_seed_42")


def _observation_state(batch: int = 1) -> ObservationState:
    mano = np.tile(np.concatenate(([6.0, -6.0, 0.2, -0.3, 0.4, -0.5], np.linspace(0.0, 1.0, 20))), (batch, 1))
    lower = np.concatenate((np.full(6, -10.0), np.zeros(20)))
    upper = np.concatenate((np.full(6, 10.0), np.ones(20)))
    object_position = np.tile([[1.0, 2.0, 3.0]], (batch, 1))
    hand_position = np.tile([[10.0, 2.0, 3.0]], (batch, 1))
    keypoint_relative = np.arange(48, dtype=np.float64).reshape(1, 16, 3) / 10.0
    fingertip_relative = np.arange(15, dtype=np.float64).reshape(1, 5, 3) / 10.0 + [0.0, 0.0, 0.2]
    forces = np.zeros((batch, 16, 3), dtype=np.float64)
    forces[:, 3] = [3.0, 4.0, 0.0]
    forces[:, 15] = [0.0, 1.0, 0.0]
    return ObservationState(
        mano_dof_pos=mano,
        mano_dof_lower=lower,
        mano_dof_upper=upper,
        hand_position=hand_position,
        hand_orientation_xyzw=np.tile([[0.0, 0.0, 0.0, 1.0]], (batch, 1)),
        object_position=object_position,
        object_orientation_xyzw=np.tile([[0.0, 0.0, 0.0, 1.0]], (batch, 1)),
        target_object_position=np.tile([[4.0, 5.0, 6.0]], (batch, 1)),
        target_object_orientation_xyzw=np.tile([[0.0, 0.0, 1.0, 0.0]], (batch, 1)),
        target_object_pos_next_5=np.tile([[7.0, 8.0, 9.0]], (batch, 1)),
        cumulative_offset=np.tile([[0.01, -0.02, 0.03]], (batch, 1)),
        cumulative_joint_offset=np.tile(np.arange(20, dtype=np.float64)[None, :] / 100.0, (batch, 1)),
        point_cloud=_static_point_cloud(),
        object_geometry=np.tile(np.arange(12, dtype=np.float64)[None, :] / 10.0, (batch, 1)),
        hand_keypoint_positions=object_position[:, None, :] + np.broadcast_to(keypoint_relative, (batch, 16, 3)),
        fingertip_positions=object_position[:, None, :] + np.broadcast_to(fingertip_relative, (batch, 5, 3)),
        hand_keypoint_contact_forces=forces,
        expected_contact_mask=expected_contact_mask_from_keypoint_ids(np.tile([[3, 15]], (batch, 1)), batch),
        action_ids=np.ones(batch, dtype=np.int64),
        object_support_points=np.array([[0.0, 0.0, -0.2], [0.0, 0.0, 0.5]]),
        table_surface_height=-1.0,
    )


def _termination_for(state: RewardState, compatibility, *, trajectory_length: int = 1_000):
    early = early_phase_mask(
        state.trajectory_steps,
        starts=state.early_phase_starts,
        steps=compatibility.early_phase_steps,
    )
    return check_termination(
        object_position=state.object_position,
        target_position=state.target_object_position,
        progress=state.trajectory_steps,
        trajectory_lengths=np.full(len(state.trajectory_steps), trajectory_length, dtype=np.int64),
        early_mask=early,
    )


def _reward_state(batch: int = 1, *, steps: np.ndarray | None = None) -> RewardState:
    if steps is None:
        steps = np.full(batch, 105, dtype=np.int64)
    mask = expected_contact_mask_from_keypoint_ids(np.tile([[3, 15]], (batch, 1)), batch)
    forces = np.zeros((batch, 16, 3), dtype=np.float64)
    forces[:, 3] = [3.0, 0.0, 0.0]
    forces[:, 15] = [0.0, 3.0, 0.0]
    joint_offset = np.zeros((batch, 20), dtype=np.float64)
    joint_offset[:, :8] = 0.1
    active = np.zeros((batch, 20), dtype=bool)
    active[:, :8] = True
    return RewardState(
        object_position=np.zeros((batch, 3)),
        target_object_position=np.zeros((batch, 3)),
        object_orientation_xyzw=np.tile([[0.0, 0.0, 0.0, 1.0]], (batch, 1)),
        target_object_orientation_xyzw=np.tile([[0.0, 0.0, 0.0, 1.0]], (batch, 1)),
        cumulative_offset=np.tile([[0.01, 0.0, 0.0]], (batch, 1)),
        cumulative_joint_offset=joint_offset,
        active_joint_mask=active,
        hand_object_force_on_object_world_N=forces,
        expected_contact_mask=mask,
        expected_contact_weights=mask.copy(),
        object_linear_velocity=np.zeros((batch, 3)),
        trajectory_steps=steps,
        contact_start_frames=np.full(batch, 100, dtype=np.int64),
        contact_end_frames=np.full(batch, 110, dtype=np.int64),
        rotation_disabled_mask=np.zeros(batch, dtype=bool),
        early_phase_starts=np.zeros(batch, dtype=np.int64),
    )


def test_observation_has_all_named_source_slices_in_exact_order() -> None:
    state = _observation_state()
    result = build_observation(state, compatibility=CURRENT_SOURCE_COMPATIBILITY)
    raw = result.raw
    assert raw.shape == (1, RAW_OBSERVATION_DIM)
    assert tuple(OBSERVATION_SLICES) == OBSERVATION_KEYS
    assert [item.stop - item.start for item in OBSERVATION_SLICES.values()] == [6, 20, 4, 3, 4, 3, 15, 3, 4, 3, 3, 6, 192, 50, 12, 48, 16, 20, 48, 16]
    assert list(OBSERVATION_SLICES.values())[-1].stop == RAW_OBSERVATION_DIM
    expected = {
        "wrist_pos": state.mano_dof_pos[:, :6],
        "finger_pos": 2.0 * state.mano_dof_pos[:, 6:] - 1.0,
        "hand_orientation": state.hand_orientation_xyzw,
        "object_position": state.object_position,
        "object_orientation": state.object_orientation_xyzw,
        "hand_position": state.hand_position,
        "finger_tip_position": (state.fingertip_positions - state.object_position[:, None, :]).reshape(1, -1),
        "target_object_position": state.target_object_position,
        "target_object_orientation": state.target_object_orientation_xyzw,
        "cumulative_offset": state.cumulative_offset,
        "target_object_pos_next_5": state.target_object_pos_next_5,
        "table_clearance": np.array([[3.8, 4.0, 4.2, 4.4, 6.8, 7.0]]),
        "object_point_cloud_raw": np.concatenate((np.array([[-8.9, 0.2, 0.3]]), np.tile([[-9.0, 0.0, 0.0]], (62, 1)), np.array([[-9.4, 0.5, -0.6]])), axis=0).reshape(1, -1),
        "action_types": action_type_one_hot(state.action_ids, 1),
        "object_geometry": state.object_geometry,
        "hand_keypoints": (state.hand_keypoint_positions - state.object_position[:, None, :]).reshape(1, -1),
        "contact_forces": np.tanh(np.array([[0.0, 0.0, 0.0, 5.0] + [0.0] * 11 + [1.0]]) * 0.025),
        "cumulative_joint_offset": state.cumulative_joint_offset,
        "contact_force_directions": np.concatenate((np.zeros((3, 3)), np.array([[0.6, 0.8, 0.0]]), np.zeros((12, 3))), axis=0).reshape(1, -1),
        "expected_contact_mask": state.expected_contact_mask,
    }
    for key, values in expected.items():
        np.testing.assert_allclose(raw[:, OBSERVATION_SLICES[key]], values, err_msg=key)
    np.testing.assert_allclose(result.policy_input[:, :2], [[5.0, -5.0]])
    assert np.all(result.policy_input <= 5.0)
    assert np.all(result.policy_input >= -5.0)


def test_point_cloud_compatibility_variants_are_explicit_and_transform_source_frames() -> None:
    state = _observation_state(batch=2)
    current = build_observation(state, compatibility=CURRENT_SOURCE_COMPATIBILITY)
    np.testing.assert_allclose(current.raw[0, OBSERVATION_SLICES["object_point_cloud_raw"]][:3], [-8.9, 0.2, 0.3])
    assert (CURRENT_SOURCE_COMPATIBILITY.early_phase_steps, CURRENT_SOURCE_COMPATIBILITY.movement_pre_padding) == (100, 250)
    assert (CHECKPOINT_SIDECAR_COMPATIBILITY.early_phase_steps, CHECKPOINT_SIDECAR_COMPATIBILITY.movement_pre_padding) == (50, 200)
    dynamic_points = np.zeros((2, POINT_COUNT, 3))
    dynamic_points[0, 0] = [0.1, 0.0, 0.0]
    dynamic_points[1, 0] = [0.2, 0.0, 0.0]
    dynamic = replace(state, point_cloud=PointCloudTemplate(dynamic_points, mode="dynamic_reset"))
    historical = build_observation(dynamic, compatibility=CHECKPOINT_SIDECAR_COMPATIBILITY)
    cloud = historical.raw[:, OBSERVATION_SLICES["object_point_cloud_raw"]].reshape(2, POINT_COUNT, 3)
    np.testing.assert_allclose(cloud[:, 0], [[-8.9, 0.0, 0.0], [-8.8, 0.0, 0.0]])
    with pytest.raises(ValueError, match="template mode"):
        build_observation(state, compatibility=CHECKPOINT_SIDECAR_COMPATIBILITY)
    with pytest.raises(ValueError, match="template mode"):
        build_observation(dynamic, compatibility=CURRENT_SOURCE_COMPATIBILITY)
    normalized = replace(
        state,
        point_cloud=PointCloudTemplate(
            np.ones((POINT_COUNT, 3)), mode="static_seed_42", normalized=True,
            scale=np.array([0.2, 0.1, 0.5]),
        ),
    )
    normalized_cloud = build_observation(normalized, compatibility=CURRENT_SOURCE_COMPATIBILITY).raw[
        0, OBSERVATION_SLICES["object_point_cloud_raw"]
    ].reshape(POINT_COUNT, 3)
    np.testing.assert_allclose(normalized_cloud[0], [-44.0, 1.0, 1.0])


def test_contact_order_mask_and_geometry_encoding_match_source_rules() -> None:
    forces = np.zeros((1, 16, 3))
    forces[0, 3] = [0.0, 0.0, 2.0]
    forces[0, 15] = [0.0, -3.0, 0.0]
    contact = extract_contact_features(forces)
    np.testing.assert_array_equal(contact.direction[0, 3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(contact.direction[0, 15], [0.0, -1.0, 0.0])
    mask = expected_contact_mask_from_keypoint_ids(np.array([[3, 15]], dtype=np.int64), 1)
    np.testing.assert_array_equal(np.flatnonzero(mask[0]), [3, 15])
    assert [object_category(name) for name in ("cube1", "cuboid2H", "cylinder1", "sphere1", "banana")] == ["box", "box", "cylinder", "sphere", "irregular"]
    np.testing.assert_allclose(geometry_encoding(object_name="cube1", geometry_type="box", dimensions=np.array([0.1, 0.2, 0.3])), [0.5, 1.0, 1.0] + [0.0] * 9)
    np.testing.assert_allclose(geometry_encoding(object_name="cylinder2", geometry_type="box", dimensions=np.array([0.1, 0.2, 0.3])), [0.0] * 3 + [1.0, 1.0, 1.0] + [0.0] * 6)
    np.testing.assert_allclose(geometry_encoding(object_name="sphere1", geometry_type="box", dimensions=np.array([0.1, 0.2, 0.3])), [0.0] * 6 + [1.0, 1.0, 1.0] + [0.0] * 3)
    np.testing.assert_allclose(geometry_encoding(object_name="banana", geometry_type="box", dimensions=np.array([0.1, 0.2, 0.3])), [0.0] * 9 + [0.5, 1.0, 1.0])
    np.testing.assert_array_equal(geometry_encoding(object_name="cube1", geometry_type="unknown", dimensions=np.array([])), np.zeros(12))


def test_replay_derived_contact_features_preserve_isaac_keypoint_order() -> None:
    trace = Path(__file__).resolve().parents[2] / "outputs/manorl/cube1_01_009_isaacgym_20260714_recovered.npz"
    with np.load(trace) as data:
        indices = np.array([0, 1, 2, 300, 790])
        forces = data["contact_keypoint_force_xyz"][indices]
        source_magnitudes = data["contact_keypoint_force_magnitude"][indices]
        source_mask = data["expected_contact_mask"][indices]
    contact = extract_contact_features(forces)
    np.testing.assert_allclose(contact.magnitude, source_magnitudes, rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(source_mask, expected_contact_mask_from_keypoint_ids(np.tile([[3, 15]], (len(indices), 1)), len(indices)))
    np.testing.assert_allclose(contact.normalized_magnitude, np.tanh(source_magnitudes * 0.025), rtol=1e-6, atol=1e-7)


def test_observation_inputs_fail_fast_for_missing_shapes_and_nonfinite_values() -> None:
    state = _observation_state()
    with pytest.raises(TypeError, match="ObservationState"):
        build_observation(None, compatibility=CURRENT_SOURCE_COMPATIBILITY)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hand_keypoint_contact_forces"):
        build_observation(replace(state, hand_keypoint_contact_forces=np.zeros((1, 15, 3))), compatibility=CURRENT_SOURCE_COMPATIBILITY)
    with pytest.raises(ValueError, match="mano_dof_pos"):
        build_observation(replace(state, mano_dof_pos=np.full((1, 26), np.nan)), compatibility=CURRENT_SOURCE_COMPATIBILITY)
    with pytest.raises(ValueError, match="expected_contact_mask"):
        build_observation(replace(state, expected_contact_mask=np.full((1, 16), 0.5)), compatibility=CURRENT_SOURCE_COMPATIBILITY)
    with pytest.raises(ValueError, match="action_ids"):
        build_observation(replace(state, action_ids=np.array([0], dtype=np.int64)), compatibility=CURRENT_SOURCE_COMPATIBILITY)


def test_reward_terms_and_windows_match_source_equations() -> None:
    state = _reward_state()
    diagnostics = compute_rewards(
        state,
        compatibility=CURRENT_SOURCE_COMPATIBILITY,
        termination=_termination_for(state, CURRENT_SOURCE_COMPATIBILITY),
    )
    np.testing.assert_allclose(diagnostics.ungated_distance_x, [0.5])
    np.testing.assert_allclose(diagnostics.ungated_distance_y, [0.5])
    np.testing.assert_allclose(diagnostics.ungated_distance_z, [2.0])
    np.testing.assert_allclose([diagnostics.distance_x[0], diagnostics.distance_y[0], diagnostics.distance_z[0]], [0.2, 0.2, 0.8])
    np.testing.assert_allclose(diagnostics.rotation, [0.3])
    np.testing.assert_allclose(diagnostics.position_penalty, [-0.025])
    np.testing.assert_allclose(diagnostics.joint_penalty, [-0.068])
    np.testing.assert_allclose(diagnostics.action_penalty, [-0.093])
    np.testing.assert_allclose(diagnostics.raw_contact, [0.4])
    np.testing.assert_allclose(diagnostics.contact, [0.4])
    np.testing.assert_allclose(diagnostics.distance_gate, [0.4])
    np.testing.assert_allclose(diagnostics.object_stability, [0.0])
    np.testing.assert_allclose(diagnostics.survival, [0.001])
    np.testing.assert_allclose(diagnostics.total, [1.808])

    window_state = _reward_state(batch=3, steps=np.array([99, 105, 111], dtype=np.int64))
    window = compute_rewards(
        window_state,
        compatibility=CHECKPOINT_SIDECAR_COMPATIBILITY,
        termination=_termination_for(window_state, CHECKPOINT_SIDECAR_COMPATIBILITY),
    )
    np.testing.assert_allclose(window.contact, [0.0, 0.4, 0.0])
    np.testing.assert_allclose(window.distance_gate, [0.0, 0.4, 0.4])
    np.testing.assert_allclose(window.object_stability, [0.0, 0.0, 0.4])
    np.testing.assert_allclose(window.total, [0.208, 1.808, 1.808])


def test_reward_contact_is_proportional_strict_and_has_no_gravity_gate() -> None:
    state = _reward_state(batch=3)
    forces = np.zeros((3, 16, 3), dtype=np.float64)
    forces[1, 3] = [1.000001, 0.0, 0.0]
    forces[1, 15] = [1.0, 0.0, 0.0]
    forces[2, 3] = [1.000001, 0.0, 0.0]
    forces[2, 15] = [0.0, 1.000001, 0.0]
    state = replace(state, hand_object_force_on_object_world_N=forces)
    diagnostics = compute_rewards(
        state,
        compatibility=CURRENT_SOURCE_COMPATIBILITY,
        termination=_termination_for(state, CURRENT_SOURCE_COMPATIBILITY),
    )

    np.testing.assert_allclose(diagnostics.raw_contact, [0.0, 0.2, 0.4])
    np.testing.assert_allclose(diagnostics.contact, [0.0, 0.2, 0.4])
    np.testing.assert_allclose(diagnostics.distance_gate, [0.0, 0.2, 0.4])
    assert "object_contact_force" not in RewardState.__dataclass_fields__
    assert "object_gravity_force" not in RewardState.__dataclass_fields__
    assert "object_contact_gate" not in diagnostics.__dataclass_fields__
    assert REWARD_CONTRACT_ID == "target_hand_object_contact_v1"
    assert REWARD_HAND_OBJECT_THRESHOLD_N == 1.0


def test_reward_contact_preserves_nonuniform_weights_and_zero_expected_contacts() -> None:
    state = _reward_state(batch=2)
    weights = np.zeros((2, 16), dtype=np.float64)
    weights[0, 3] = 1.0
    weights[0, 15] = 3.0
    forces = np.full((2, 16, 3), 3.0, dtype=np.float64)
    forces[0, 15] = 0.0
    mask = state.expected_contact_mask.copy()
    mask[1] = 0.0
    state = replace(
        state,
        hand_object_force_on_object_world_N=forces,
        expected_contact_mask=mask,
        expected_contact_weights=weights,
    )
    diagnostics = compute_rewards(
        state,
        compatibility=CURRENT_SOURCE_COMPATIBILITY,
        termination=_termination_for(state, CURRENT_SOURCE_COMPATIBILITY),
    )

    np.testing.assert_allclose(diagnostics.raw_contact, [0.1, 0.0])
    np.testing.assert_allclose(diagnostics.contact, [0.1, 0.0])


def test_reward_compatibility_rotation_and_termination_interaction() -> None:
    state = _reward_state(steps=np.array([50], dtype=np.int64))
    current = compute_rewards(state, compatibility=CURRENT_SOURCE_COMPATIBILITY, termination=_termination_for(state, CURRENT_SOURCE_COMPATIBILITY))
    historical = compute_rewards(state, compatibility=CHECKPOINT_SIDECAR_COMPATIBILITY, termination=_termination_for(state, CHECKPOINT_SIDECAR_COMPATIBILITY))
    np.testing.assert_array_equal(current.early_phase, [True])
    np.testing.assert_array_equal(historical.early_phase, [False])
    np.testing.assert_allclose(current.total, current.action_penalty)
    assert historical.total[0] > current.total[0]

    angles = np.deg2rad(np.array([0.0, 20.0, 90.0, 91.0, 0.0]))
    rotation_state = _reward_state(batch=5, steps=np.full(5, 105, dtype=np.int64))
    rotation_state = replace(
        rotation_state,
        object_orientation_xyzw=np.column_stack((np.zeros(5), np.zeros(5), np.sin(angles / 2.0), np.cos(angles / 2.0))),
        rotation_disabled_mask=np.array([False, False, False, False, True]),
    )
    rotation = compute_rewards(
        rotation_state,
        compatibility=CURRENT_SOURCE_COMPATIBILITY,
        termination=_termination_for(rotation_state, CURRENT_SOURCE_COMPATIBILITY),
    )
    expected_90 = 0.4 * (-0.00003175 * 70.0**2 - 0.019206 * 70.0 + 0.5) - 0.1
    np.testing.assert_allclose(rotation.rotation, [0.3, 0.1, expected_90, -0.5, 0.0], atol=1e-10)

    deviation_state = replace(state, trajectory_steps=np.array([101], dtype=np.int64), object_position=np.array([[0.100001, 0.0, 0.0]]))
    termination = _termination_for(deviation_state, CURRENT_SOURCE_COMPATIBILITY)
    assert termination.deviation_reset[0]
    penalized = compute_rewards(deviation_state, compatibility=CURRENT_SOURCE_COMPATIBILITY, termination=termination)
    without_penalty = compute_rewards(
        deviation_state,
        compatibility=CURRENT_SOURCE_COMPATIBILITY,
        termination=check_termination(
            object_position=deviation_state.object_position,
            target_position=deviation_state.target_object_position,
            progress=deviation_state.trajectory_steps,
            trajectory_lengths=np.array([1_000]),
            early_mask=np.array([True]),
        ),
    )
    np.testing.assert_allclose(penalized.deviation_penalty, [-25.0])
    np.testing.assert_allclose(penalized.total, without_penalty.total - 25.0)


def test_reward_inputs_and_termination_result_fail_fast() -> None:
    state = _reward_state()
    termination = _termination_for(state, CURRENT_SOURCE_COMPATIBILITY)
    with pytest.raises(TypeError, match="RewardState"):
        compute_rewards(None, compatibility=CURRENT_SOURCE_COMPATIBILITY, termination=termination)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hand_object_force_on_object_world_N"):
        compute_rewards(
            replace(state, hand_object_force_on_object_world_N=np.zeros((1, 15, 3))),
            compatibility=CURRENT_SOURCE_COMPATIBILITY,
            termination=termination,
        )
    with pytest.raises(ValueError, match="expected_contact_weights"):
        compute_rewards(replace(state, expected_contact_weights=np.ones((1, 15))), compatibility=CURRENT_SOURCE_COMPATIBILITY, termination=termination)
    bad_termination = replace(termination, reset=np.array([False, False]))
    with pytest.raises(ValueError, match="termination result batch size"):
        compute_rewards(state, compatibility=CURRENT_SOURCE_COMPATIBILITY, termination=bad_termination)
