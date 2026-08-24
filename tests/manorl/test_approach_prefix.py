from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from sim.manorl.abi import (
    check_termination,
    early_phase_mask,
    process_residual_actions,
)
from sim.manorl.approach_prefix import (
    APPROACH_PREFIX_CONTRACT,
    RETREAT_SUFFIX_CONTRACT,
    ApproachPrefixConfig,
    RetreatSuffixConfig,
    augment_trajectory_with_approach_prefix,
    augment_trajectory_with_retreat_suffix,
    augmentation_stream_seed,
)
from sim.manorl.contracts import JOINT_DOF, TrajectoryIdentity
from sim.manorl.synthetic_parent import (
    ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
    AcceptedSyntheticParent,
)
from sim.manorl.trajectory import ReferenceTrajectory, TrajectoryBatch


def _trajectory(*, base_pre_padding: int = 60) -> ReferenceTrajectory:
    fps = 120
    frames = max(180, base_pre_padding + 80)
    times = np.arange(frames, dtype=np.float64) / fps
    q_ref = np.zeros((frames, JOINT_DOF), dtype=np.float64)
    q_ref[:, :3] = (
        np.asarray((0.10, -0.20, 0.09))[None]
        + times[:, None] * np.asarray((0.10, 0.04, -0.02))[None]
        + 0.5 * times[:, None] ** 2 * np.asarray((0.03, -0.02, 0.01))[None]
    )
    q_ref[:, 3:6] = (
        np.asarray((0.10, -0.30, 0.20))[None]
        + times[:, None] * np.asarray((0.04, -0.02, 0.03))[None]
    )
    q_ref[:, 6:] = np.linspace(0.1, 0.3, JOINT_DOF - 6)[None]
    object_pos = np.repeat(np.asarray(((0.0, -0.1, 0.01),)), frames, axis=0)
    object_quat = np.repeat(np.asarray(((0.0, 0.0, 0.0, 1.0),)), frames, axis=0)
    identity = TrajectoryIdentity(
        dataset_path="/fixture/source.lance",
        dataset_version=295,
        row_index=17,
        object_index=0,
        uuid="fixture-uuid",
        file_uuid="fixture-file",
        identity="iphone_01_0017",
        source_start=100,
        source_stop=100 + frames,
        movement_start_raw=100 + base_pre_padding,
        movement_end_raw=100 + base_pre_padding + 50,
    )
    return ReferenceTrajectory(
        identity=identity,
        dataset_version=295,
        source_indices=np.arange(100, 100 + frames, dtype=np.int64),
        timestamps=times,
        q_ref=q_ref,
        object_pos_raw=object_pos,
        object_pos=object_pos,
        object_quat_xyzw=object_quat,
        object_z_shift=0.0,
        hand_sides=("right",),
        q_ref_by_side={"right": q_ref},
        selected_hand_sides=("right",),
        reference_fps=fps,
        control_fps=fps,
        movement_start_step=base_pre_padding,
        movement_end_step=base_pre_padding + 50,
    )


def test_approach_prefix_is_seeded_positive_z_and_suffix_exact() -> None:
    source = _trajectory()
    augmented, sample = augment_trajectory_with_approach_prefix(source, seed=42)
    repeated, repeated_sample = augment_trajectory_with_approach_prefix(source, seed=42)
    changed, changed_sample = augment_trajectory_with_approach_prefix(source, seed=43)

    assert sample.contract == APPROACH_PREFIX_CONTRACT
    assert sample.source_identity == source.identity.identity
    assert sample.to_manifest() == repeated_sample.to_manifest()
    np.testing.assert_array_equal(augmented.q_ref, repeated.q_ref)
    assert sample.to_manifest() != changed_sample.to_manifest()
    assert not np.array_equal(augmented.q_ref[: sample.prefix_frames], changed.q_ref[: changed_sample.prefix_frames])

    assert 0.30 <= sample.start_xy_radius_m <= 0.70
    assert 0.08 <= sample.start_z_offset_m <= 0.30
    assert -30.0 <= sample.xy_offset_deg <= 30.0
    assert 100 <= sample.effective_pre_padding <= 360
    assert sample.effective_pre_padding == 60 + sample.prefix_frames
    start = np.asarray(sample.start_position_m)
    object_start = source.object_pos[0]
    np.testing.assert_allclose(
        np.linalg.norm((start - object_start)[:2]),
        sample.start_xy_radius_m,
        atol=1e-12,
    )
    assert start[2] - object_start[2] == pytest.approx(
        sample.start_z_offset_m, abs=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(start - object_start), sample.start_distance_m, atol=1e-12
    )
    base_azimuth = math.atan2(
        source.q_ref[0, 1] - object_start[1],
        source.q_ref[0, 0] - object_start[0],
    )
    sampled_azimuth = math.atan2(start[1] - object_start[1], start[0] - object_start[0])
    wrapped_offset = math.degrees((sampled_azimuth - base_azimuth + math.pi) % (2 * math.pi) - math.pi)
    assert wrapped_offset == pytest.approx(sample.xy_offset_deg, abs=1e-12)

    prefix = sample.prefix_frames
    np.testing.assert_array_equal(augmented.q_ref[prefix:], source.q_ref)
    np.testing.assert_array_equal(augmented.q_ref_by_side["right"][prefix:], source.q_ref)
    np.testing.assert_array_equal(augmented.source_indices[prefix:], source.source_indices)
    np.testing.assert_array_equal(augmented.timestamps[prefix:], source.timestamps)
    np.testing.assert_array_equal(augmented.object_pos[prefix:], source.object_pos)
    np.testing.assert_array_equal(augmented.object_quat_xyzw[prefix:], source.object_quat_xyzw)
    np.testing.assert_array_equal(
        augmented.q_ref[:prefix, 6:],
        np.repeat(source.q_ref[:1, 6:], prefix, axis=0),
    )
    assert augmented.movement_start_step == source.movement_start_step + prefix
    assert augmented.movement_end_step == source.movement_end_step + prefix
    assert augmented.augmentation_prefix_frames == prefix
    assert sample.splice_position_error_m == pytest.approx(0.0, abs=1e-14)
    assert sample.splice_velocity_error_m_s < 0.01
    assert sample.splice_acceleration_jump_m_s2 < 20.0


def test_approach_smooths_source_row_frame0_pose_into_pre60() -> None:
    source = _trajectory()
    raw_frame0_q_3_28 = np.asarray(source.q_ref[0, 3:]).copy()
    raw_frame0_q_3_28[:3] += np.asarray((0.20, -0.10, 0.15))
    raw_frame0_q_3_28[3:] += np.linspace(-0.25, 0.20, JOINT_DOF - 6)
    augmented, sample = augment_trajectory_with_approach_prefix(
        source,
        seed=42,
        start_q_ref_3_28=raw_frame0_q_3_28,
    )
    prefix = sample.prefix_frames
    assert sample.orientation_template == "source_row_frame0"
    np.testing.assert_allclose(
        augmented.q_ref[0, 3:28], raw_frame0_q_3_28, atol=1e-14
    )
    assert not np.allclose(
        augmented.q_ref[:prefix, 6:],
        augmented.q_ref[0, 6:][None],
    )
    np.testing.assert_array_equal(augmented.q_ref[prefix:], source.q_ref)
    dt = 1.0 / source.control_fps
    prefix_splice_velocity = (
        augmented.q_ref[prefix, 3:28] - augmented.q_ref[prefix - 1, 3:28]
    ) / dt
    source_velocity = (source.q_ref[1, 3:28] - source.q_ref[0, 3:28]) / dt
    np.testing.assert_allclose(prefix_splice_velocity, source_velocity, atol=1e-12)


def test_approach_rejects_invalid_source_row_frame0_pose() -> None:
    with pytest.raises(ValueError, match=r"q_ref\[3:28\]"):
        augment_trajectory_with_approach_prefix(
            _trajectory(), seed=42, start_q_ref_3_28=np.zeros(24)
        )


def test_near_approach_uses_independent_retreat_like_object_relative_sample() -> None:
    source = _trajectory()
    anchor = 100
    episode_seed = 49
    near_seed = augmentation_stream_seed(episode_seed, "near-approach")
    retreat_seed = episode_seed
    assert near_seed != retreat_seed

    near, near_sample = augment_trajectory_with_approach_prefix(
        source,
        seed=near_seed,
        config=ApproachPrefixConfig(mode="near"),
        near_anchor_reference_index=anchor,
    )
    retreat, retreat_sample = augment_trajectory_with_retreat_suffix(
        near,
        seed=retreat_seed,
        anchor_reference_index=anchor + near_sample.prefix_frames,
    )
    assert near_sample.approach_mode == "near"
    assert near_sample.seed == near_seed
    assert retreat_sample.seed == retreat_seed
    assert near_sample.direction_reference == (
        "final_retreat_distribution_mapped_to_initial_object"
    )
    near_relative = (
        np.asarray(near_sample.start_position_m) - source.object_pos[0]
    )
    retreat_relative = (
        np.asarray(retreat_sample.end_position_m) - retreat.object_pos[-1]
    )
    assert not np.allclose(near_relative, retreat_relative)
    assert near_sample.start_xy_radius_m == pytest.approx(
        np.linalg.norm(near_relative[:2]), abs=1e-12
    )
    assert near_sample.start_z_offset_m == pytest.approx(
        near_relative[2], abs=1e-12
    )


def test_near_approach_keeps_retreat_endpoint_world_z_after_object_lift() -> None:
    source = _trajectory()
    lifted_object = np.asarray(source.object_pos).copy()
    lifted_object[:, 2] = np.linspace(0.01, 0.21, len(lifted_object))
    source = replace(
        source,
        object_pos=lifted_object,
        object_pos_raw=lifted_object,
    )
    seed = augmentation_stream_seed(49, "near-approach")
    _, near_sample = augment_trajectory_with_approach_prefix(
        source,
        seed=seed,
        config=ApproachPrefixConfig(mode="near"),
        near_anchor_reference_index=100,
    )
    _, endpoint_sample = augment_trajectory_with_retreat_suffix(
        source,
        seed=seed,
        anchor_reference_index=100,
    )
    assert near_sample.start_position_m[2] == pytest.approx(
        endpoint_sample.end_position_m[2], abs=1e-12
    )
    assert near_sample.start_position_m[2] != pytest.approx(
        source.object_pos[0, 2]
        + endpoint_sample.end_position_m[2]
        - source.object_pos[-1, 2],
        abs=1e-6,
    )


def test_near_approach_allows_start_below_object_center_for_contact_validation() -> None:
    source = _trajectory()
    raised_object = np.asarray(source.object_pos).copy()
    raised_object[:, 2] = 0.20
    source = replace(
        source,
        object_pos=raised_object,
        object_pos_raw=raised_object,
    )
    _, sample = augment_trajectory_with_approach_prefix(
        source,
        seed=augmentation_stream_seed(49, "near-approach"),
        config=ApproachPrefixConfig(mode="near"),
        near_anchor_reference_index=100,
    )
    assert sample.start_z_offset_m < 0.0
    assert np.all(np.isfinite(sample.start_position_m))


def test_near_approach_rejects_far_only_distribution_overrides() -> None:
    with pytest.raises(ValueError, match="far-only"):
        ApproachPrefixConfig(mode="near", minimum_xy_radius_m=0.20)


def test_near_approach_requires_movement_end_anchor() -> None:
    with pytest.raises(ValueError, match="movement-end-anchored"):
        augment_trajectory_with_approach_prefix(
            _trajectory(),
            seed=42,
            config=ApproachPrefixConfig(mode="near"),
        )


def test_retreat_suffix_replaces_post_contact_tail_from_original_direction() -> None:
    source = _trajectory()
    anchor = 100
    augmented, sample = augment_trajectory_with_retreat_suffix(
        source, seed=42, anchor_reference_index=anchor
    )
    repeated, repeated_sample = augment_trajectory_with_retreat_suffix(
        source, seed=42, anchor_reference_index=anchor
    )
    changed, changed_sample = augment_trajectory_with_retreat_suffix(
        source, seed=43, anchor_reference_index=anchor
    )

    assert sample.contract == RETREAT_SUFFIX_CONTRACT
    assert sample.source_identity == source.identity.identity
    assert sample.anchor_reference_index == anchor
    assert sample.to_manifest() == repeated_sample.to_manifest()
    np.testing.assert_array_equal(augmented.q_ref, repeated.q_ref)
    assert sample.to_manifest() != changed_sample.to_manifest()
    assert not np.array_equal(
        augmented.q_ref[anchor + 1 :, :3],
        changed.q_ref[anchor + 1 :, :3],
    )

    assert 0.03 <= sample.extra_horizontal_offset_m <= 0.15
    assert 0.04 <= sample.extra_z_offset_m <= 0.10
    assert -30.0 <= sample.xy_offset_deg <= 30.0
    original_delta = source.q_ref[-1, :3] - source.q_ref[anchor, :3]
    original_horizontal = float(np.linalg.norm(original_delta[:2]))
    assert sample.original_horizontal_distance_m == pytest.approx(
        original_horizontal, abs=1e-12
    )
    assert sample.end_horizontal_distance_m == pytest.approx(
        original_horizontal + sample.extra_horizontal_offset_m, abs=1e-12
    )
    endpoint_delta = np.asarray(sample.end_position_m) - source.q_ref[anchor, :3]
    assert np.linalg.norm(endpoint_delta[:2]) == pytest.approx(
        sample.end_horizontal_distance_m, abs=1e-12
    )
    assert sample.original_z_displacement_m == pytest.approx(
        original_delta[2], abs=1e-12
    )
    assert sample.end_z_displacement_m == pytest.approx(
        original_delta[2] + sample.extra_z_offset_m, abs=1e-12
    )
    assert endpoint_delta[2] == pytest.approx(
        sample.end_z_displacement_m, abs=1e-12
    )
    assert np.asarray(sample.end_position_m)[2] - source.q_ref[-1, 2] == pytest.approx(
        sample.extra_z_offset_m, abs=1e-12
    )
    original_direction = math.atan2(original_delta[1], original_delta[0])
    sampled_direction = math.atan2(endpoint_delta[1], endpoint_delta[0])
    wrapped_offset = math.degrees(
        (sampled_direction - original_direction + math.pi) % (2 * math.pi) - math.pi
    )
    assert wrapped_offset == pytest.approx(sample.xy_offset_deg, abs=1e-12)

    assert sample.suffix_frames == len(source.q_ref) - anchor
    assert sample.replaced_tail_frames == len(source.q_ref) - anchor - 1
    assert sample.suffix_frames == augmented.augmentation_suffix_frames
    assert len(augmented.q_ref) == len(source.q_ref)
    np.testing.assert_array_equal(
        augmented.q_ref[: anchor + 1], source.q_ref[: anchor + 1]
    )
    np.testing.assert_array_equal(
        augmented.q_ref[anchor + 1 : anchor + 3, :3],
        source.q_ref[anchor + 1 : anchor + 3, :3],
    )
    np.testing.assert_array_equal(
        augmented.q_ref_by_side["right"][: anchor + 1],
        source.q_ref_by_side["right"][: anchor + 1],
    )
    # Only right-wrist XYZ in the replacement tail changes. Rotation, fingers,
    # source clock/mapping, and object reference stay bit-identical.
    np.testing.assert_array_equal(augmented.q_ref[:, 3:], source.q_ref[:, 3:])
    np.testing.assert_array_equal(augmented.source_indices, source.source_indices)
    np.testing.assert_array_equal(augmented.timestamps, source.timestamps)
    np.testing.assert_array_equal(augmented.object_pos, source.object_pos)
    np.testing.assert_array_equal(
        augmented.object_quat_xyzw, source.object_quat_xyzw
    )
    assert augmented.movement_start_step == source.movement_start_step
    assert augmented.movement_end_step == source.movement_end_step
    assert sample.splice_position_m == pytest.approx(
        tuple(source.q_ref[anchor, :3]), abs=1e-14
    )
    assert sample.splice_velocity_error_m_s == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        augmented.q_ref[-1, :3], sample.end_position_m, atol=1e-12
    )


def test_retreat_suffix_config_bounds() -> None:
    with pytest.raises(ValueError, match="extra-horizontal bounds"):
        RetreatSuffixConfig(
            minimum_extra_horizontal_m=0.20,
            maximum_extra_horizontal_m=0.10,
        )
    with pytest.raises(ValueError, match="Z-offset bounds"):
        RetreatSuffixConfig(minimum_z_offset_m=-0.01)


def test_approach_prefix_rejects_non_pre60_source() -> None:
    with pytest.raises(ValueError, match="requires base pre-padding 60"):
        augment_trajectory_with_approach_prefix(_trajectory(base_pre_padding=180), seed=42)


def test_approach_prefix_config_requires_strictly_positive_z_offset() -> None:
    with pytest.raises(ValueError, match="Z-offset bounds"):
        ApproachPrefixConfig(minimum_z_offset_m=0.0)


def _action_inputs(batch: int) -> dict[str, np.ndarray]:
    return {
        "raw_actions": np.ones((batch, JOINT_DOF), dtype=np.float64),
        "cumulative_offset": np.full((batch, 3), 0.01, dtype=np.float64),
        "cumulative_joint_offset": np.full((batch, JOINT_DOF - 6), 0.01, dtype=np.float64),
        "mocap_targets": np.zeros((batch, JOINT_DOF), dtype=np.float64),
        "joint_lower": np.full(JOINT_DOF, -10.0, dtype=np.float64),
        "joint_upper": np.full(JOINT_DOF, 10.0, dtype=np.float64),
        "active_joint_mask": np.ones((batch, JOINT_DOF - 6), dtype=bool),
    }


def test_per_row_prefix_gates_residual_then_opens_at_original_pre60_start() -> None:
    lengths = np.asarray((40, 100), dtype=np.int64)
    during = process_residual_actions(
        **_action_inputs(2),
        trajectory_steps=lengths - 1,
        early_phase_lengths=lengths,
        hold_early_exit_zero=np.zeros(2, dtype=bool),
    )
    np.testing.assert_array_equal(during.actions, 0.0)
    np.testing.assert_array_equal(during.cumulative_offset, 0.0)
    np.testing.assert_array_equal(during.cumulative_joint_offset, 0.0)

    boundary = process_residual_actions(
        **_action_inputs(2),
        trajectory_steps=lengths,
        early_phase_lengths=lengths,
        hold_early_exit_zero=np.zeros(2, dtype=bool),
    )
    assert np.all(boundary.actions > 0.0)
    assert np.all(boundary.cumulative_offset > 0.0)
    assert np.all(boundary.cumulative_joint_offset > 0.0)


def test_per_row_suffix_disables_actions_and_smoothly_decays_residual() -> None:
    lengths = np.asarray((500, 600), dtype=np.int64)
    suffix_lengths = np.asarray((20, 40), dtype=np.int64)
    before = process_residual_actions(
        **_action_inputs(2),
        trajectory_steps=lengths - suffix_lengths - 1,
        trajectory_lengths=lengths,
        final_decay_lengths=suffix_lengths,
    )
    assert np.all(before.actions > 0.0)
    assert np.all(before.cumulative_offset > 0.0)
    assert np.all(before.cumulative_joint_offset > 0.0)

    inputs = _action_inputs(2)
    boundary = process_residual_actions(
        **inputs,
        trajectory_steps=lengths - suffix_lengths,
        trajectory_lengths=lengths,
        final_decay_lengths=suffix_lengths,
    )
    np.testing.assert_array_equal(boundary.actions, 0.0)
    np.testing.assert_array_equal(
        boundary.cumulative_offset, inputs["cumulative_offset"]
    )
    np.testing.assert_array_equal(
        boundary.cumulative_joint_offset, inputs["cumulative_joint_offset"]
    )

    position = boundary.cumulative_offset
    joints = boundary.cumulative_joint_offset
    for step in range(1, int(suffix_lengths.max())):
        active_steps = np.minimum(
            lengths - suffix_lengths + step,
            lengths - 3,
        )
        result = process_residual_actions(
            **{
                **_action_inputs(2),
                "cumulative_offset": position,
                "cumulative_joint_offset": joints,
            },
            trajectory_steps=active_steps,
            trajectory_lengths=lengths,
            final_decay_lengths=suffix_lengths,
        )
        np.testing.assert_array_equal(result.actions, 0.0)
        assert np.all(np.abs(result.cumulative_offset) <= np.abs(position) + 1e-15)
        assert np.all(
            np.abs(result.cumulative_joint_offset) <= np.abs(joints) + 1e-15
        )
        position = result.cumulative_offset
        joints = result.cumulative_joint_offset
    np.testing.assert_allclose(position, 0.0, atol=1e-15)
    np.testing.assert_allclose(joints, 0.0, atol=1e-15)


def test_near_attempt_uses_movement_anchor_when_source_index_is_edge_held() -> None:
    from tools.export_manorl_synthetic_lance import _augment_attempt_trajectories

    source = _trajectory()
    anchor = int(source.movement_end_step) + 15
    source_indices = source.source_indices.copy()
    source_indices[anchor:] = source_indices[anchor]
    source = replace(source, source_indices=source_indices)
    parent = AcceptedSyntheticParent(
        contract=ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
        parent_dataset_path="/prior.lance",
        parent_dataset_version=40,
        parent_row_index=3,
        parent_row_uuid="row-uuid",
        parent_row_contract="synthetic_mano_target_replay_visual_v2_contact",
        source_identity=source.identity.identity,
        source_dataset_path=source.identity.dataset_path,
        source_dataset_version=source.identity.dataset_version,
        source_row_index=source.identity.row_index,
        checkpoint_sha256="a" * 64,
        checkpoint_update=1000,
        parent_seed=42,
        parent_episode_index=0,
        parent_generation_attempt=1,
        object_init_xy_offset_m=(0.0, 0.0),
        reference_fps=120,
        retreat_last_contact_state_index=120,
        retreat_anchor_state_index=115,
        retreat_anchor_source_frame_index=int(source_indices[anchor]),
        retreat_anchor_horizontal_distance_m=float(
            np.linalg.norm(source.q_ref[-1, :2] - source.q_ref[anchor, :2])
        ),
        retreat_anchor_offset_frames=15,
        parent_movement_end_state_index=100,
        source_row_frame0_right_q_ref_3_28=tuple(source.q_ref[0, 3:28]),
    )
    augmented, samples = _augment_attempt_trajectories(
        TrajectoryBatch((source,)),
        attempt_seed=42,
        config=ApproachPrefixConfig(mode="near"),
        accepted_parent=parent,
    )
    assert augmented.num_envs == 1
    assert samples[source.identity.identity].approach_mode == "near"


def test_attempt_batch_resamples_per_reset_seed_and_identity() -> None:
    from tools.export_manorl_synthetic_lance import _augment_attempt_trajectories

    batch = TrajectoryBatch((_trajectory(),))
    first, first_samples = _augment_attempt_trajectories(
        batch, attempt_seed=42, config=ApproachPrefixConfig()
    )
    repeated, repeated_samples = _augment_attempt_trajectories(
        batch, attempt_seed=42, config=ApproachPrefixConfig()
    )
    reset, reset_samples = _augment_attempt_trajectories(
        batch, attempt_seed=43, config=ApproachPrefixConfig()
    )
    np.testing.assert_array_equal(first.trajectories[0].q_ref, repeated.trajectories[0].q_ref)
    assert first_samples[batch.trajectories[0].identity.identity].to_manifest() == repeated_samples[batch.trajectories[0].identity.identity].to_manifest()
    assert reset_samples[batch.trajectories[0].identity.identity].to_manifest() != first_samples[batch.trajectories[0].identity.identity].to_manifest()
    assert not np.array_equal(first.trajectories[0].q_ref, reset.trajectories[0].q_ref)


def test_prefix_masks_deviation_only_before_original_pre60_start() -> None:
    lengths = np.asarray((40, 100), dtype=np.int64)
    positions = np.asarray(((0.2, 0.0, 0.0), (0.2, 0.0, 0.0)))
    targets = np.zeros((2, 3), dtype=np.float64)
    trajectory_lengths = np.full(2, 500, dtype=np.int64)

    during = check_termination(
        object_position=positions,
        target_position=targets,
        progress=np.ones(2, dtype=np.int64),
        trajectory_lengths=trajectory_lengths,
        early_mask=early_phase_mask(lengths - 1, steps=lengths),
    )
    np.testing.assert_array_equal(during.deviation_reset, False)

    boundary = check_termination(
        object_position=positions,
        target_position=targets,
        progress=np.ones(2, dtype=np.int64),
        trajectory_lengths=trajectory_lengths,
        early_mask=early_phase_mask(lengths, steps=lengths),
    )
    np.testing.assert_array_equal(boundary.deviation_reset, True)


def test_prefix_only_far_and_near_preserve_complete_original_reference_tail() -> None:
    source = _trajectory()
    parent_anchor = int(source.movement_end_step) + 15
    source_indices = source.source_indices.copy()
    source_indices[parent_anchor:] = source_indices[parent_anchor]
    source = replace(source, source_indices=source_indices)
    parent = AcceptedSyntheticParent(
        contract=ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
        parent_dataset_path="/prior.lance",
        parent_dataset_version=40,
        parent_row_index=3,
        parent_row_uuid="row-uuid",
        parent_row_contract="synthetic_mano_target_replay_visual_v2_contact",
        source_identity=source.identity.identity,
        source_dataset_path=source.identity.dataset_path,
        source_dataset_version=source.identity.dataset_version,
        source_row_index=source.identity.row_index,
        checkpoint_sha256="a" * 64,
        checkpoint_update=1000,
        parent_seed=42,
        parent_episode_index=0,
        parent_generation_attempt=1,
        object_init_xy_offset_m=(0.0, 0.0),
        reference_fps=120,
        retreat_last_contact_state_index=120,
        retreat_anchor_state_index=115,
        retreat_anchor_source_frame_index=int(source_indices[parent_anchor]),
        retreat_anchor_horizontal_distance_m=float(
            np.linalg.norm(
                source.q_ref[-1, :2] - source.q_ref[parent_anchor, :2]
            )
        ),
        retreat_anchor_offset_frames=15,
        parent_movement_end_state_index=100,
        source_row_frame0_right_q_ref_3_28=tuple(source.q_ref[0, 3:28]),
    )
    from tools.export_manorl_synthetic_lance import _augment_attempt_trajectories

    for mode in ("far", "near"):
        augmented, samples = _augment_attempt_trajectories(
            TrajectoryBatch((source,)),
            attempt_seed=42,
            config=ApproachPrefixConfig(mode=mode),
            accepted_parent=parent,
        )
        resolved = augmented.trajectories[0]
        prefix_frames = samples[source.identity.identity].prefix_frames
        assert resolved.augmentation_suffix_frames == 0
        np.testing.assert_array_equal(resolved.q_ref[prefix_frames:], source.q_ref)
        np.testing.assert_array_equal(
            resolved.source_indices[prefix_frames:], source.source_indices
        )
        np.testing.assert_array_equal(
            resolved.object_pos[prefix_frames:], source.object_pos
        )
        np.testing.assert_array_equal(
            resolved.object_quat_xyzw[prefix_frames:], source.object_quat_xyzw
        )
