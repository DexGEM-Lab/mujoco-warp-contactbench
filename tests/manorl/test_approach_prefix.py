from __future__ import annotations

import math

import numpy as np
import pytest

from sim.manorl.abi import (
    check_termination,
    early_phase_mask,
    process_residual_actions,
)
from sim.manorl.approach_prefix import (
    APPROACH_PREFIX_CONTRACT,
    ApproachPrefixConfig,
    augment_trajectory_with_approach_prefix,
)
from sim.manorl.contracts import JOINT_DOF, TrajectoryIdentity
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
