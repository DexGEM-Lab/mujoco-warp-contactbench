import numpy as np
import pytest

from sim.manorl.abi import (
    ResidualActionConfig,
    check_termination,
    early_phase_mask,
    process_residual_actions,
)


def _control_inputs(batch: int = 3) -> dict[str, np.ndarray]:
    return {
        "raw_actions": np.full((batch, 26), 2.0),
        "cumulative_offset": np.full((batch, 3), 0.02),
        "cumulative_joint_offset": np.full((batch, 20), 0.2),
        "mocap_targets": np.zeros((batch, 26)),
        "joint_lower": np.full(26, -10.0),
        "joint_upper": np.full(26, 10.0),
        "active_joint_mask": np.array([[True] * 8 + [False] * 12] * batch),
    }


def test_residual_control_clips_masks_and_uses_target_early_interval() -> None:
    config = ResidualActionConfig()
    assert config.position_scale == (0.003, 0.003, 0.003)
    assert config.gamma_xy == config.gamma_z == 0.9
    assert config.max_position_offset == 0.03
    values = _control_inputs()
    result = process_residual_actions(
        **values,
        trajectory_steps=np.array([0, 99, 100]),
    )
    np.testing.assert_array_equal(result.actions[0], 0.0)
    np.testing.assert_array_equal(result.actions[1], 0.0)
    np.testing.assert_array_equal(result.actions[2, 14:26], 0.0)
    np.testing.assert_allclose(result.actions[2, :14], 1.0)
    np.testing.assert_array_equal(result.cumulative_offset[:2], 0.0)
    np.testing.assert_allclose(result.cumulative_offset[2], 0.003)
    np.testing.assert_array_equal(result.cumulative_joint_offset[:2], 0.0)
    np.testing.assert_allclose(
        result.cumulative_joint_offset[2, :8],
        [0.1, 0.12, 0.044, 0.01, 0.024, 0.04, 0.06, 0.01],
    )
    np.testing.assert_array_equal(result.cumulative_joint_offset[2, 8:], 0.0)
    # Rotation is immediate, unlike the position and joint cumulative residuals.
    np.testing.assert_allclose(result.targets[2, 3:6], 0.00025)


def test_residual_control_accumulates_only_active_joints_and_clamps() -> None:
    values = _control_inputs(batch=1)
    values["raw_actions"][:] = 1.0
    values["cumulative_offset"][:] = 0.029
    values["cumulative_joint_offset"][:] = 0.0
    result = process_residual_actions(**values, trajectory_steps=np.array([101]))
    np.testing.assert_allclose(result.cumulative_offset, [[0.0291, 0.0291, 0.0291]])
    np.testing.assert_allclose(
        result.cumulative_joint_offset[0, :8], [0.1, 0.12, 0.044, 0.01, 0.024, 0.04, 0.06, 0.01]
    )
    np.testing.assert_array_equal(result.cumulative_joint_offset[0, 8:], 0.0)
    disabled = process_residual_actions(
        **values,
        trajectory_steps=np.array([101]),
        use_residual=np.array([0.0]),
    )
    np.testing.assert_array_equal(disabled.cumulative_offset, 0.0)
    np.testing.assert_array_equal(disabled.targets, 0.0)


def test_position_recurrence_reaches_the_target_cap_for_both_signs() -> None:
    values = _control_inputs(batch=1)
    values["cumulative_joint_offset"][:] = 0.0
    for sign in (-1.0, 1.0):
        values["raw_actions"][:] = sign
        offset = np.zeros((1, 3))
        for step in range(101, 501):
            result = process_residual_actions(
                **{**values, "cumulative_offset": offset}, trajectory_steps=np.array([step])
            )
            offset = result.cumulative_offset
        np.testing.assert_allclose(offset, np.full((1, 3), sign * 0.03), atol=1e-12)
        capped = process_residual_actions(
            **{**values, "cumulative_offset": np.full((1, 3), sign * 0.03)}, trajectory_steps=np.array([501])
        )
        np.testing.assert_allclose(capped.cumulative_offset, np.full((1, 3), sign * 0.03))


def test_early_phase_uses_per_environment_starts() -> None:
    np.testing.assert_array_equal(
        early_phase_mask(np.array([0, 99, 100, 105]), starts=np.array([0, 0, 0, 6])),
        [True, True, False, True],
    )
    with pytest.raises(ValueError, match="integer"):
        early_phase_mask(np.array([1.0]))


def test_termination_uses_strict_target_threshold_and_early_suppression() -> None:
    outcome = check_termination(
        object_position=np.array([[0.0, 0.0, 0.0], [0.100001, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        target_position=np.zeros((3, 3)),
        progress=np.array([4, 0, 9]),
        trajectory_lengths=np.array([5, 20, 10]),
        early_mask=np.array([False, True, False]),
    )
    np.testing.assert_array_equal(outcome.reset, [True, False, True])
    np.testing.assert_array_equal(outcome.deviation_reset, [False, False, False])
    np.testing.assert_array_equal(outcome.deviation_penalty, [0.0, 0.0, 0.0])
    strict_boundary = check_termination(
        object_position=np.array([[0.10, 0.0, 0.0], [0.100001, 0.0, 0.0], [0.100001, 0.0, 0.0]]),
        target_position=np.zeros((3, 3)),
        progress=np.zeros(3, dtype=np.int64),
        trajectory_lengths=np.full(3, 10, dtype=np.int64),
        early_mask=np.array([False, False, True]),
    )
    np.testing.assert_array_equal(strict_boundary.reset, [False, True, False])
    np.testing.assert_array_equal(strict_boundary.deviation_reset, [False, True, False])
    np.testing.assert_allclose(strict_boundary.deviation_penalty, [0.0, 0.0, 0.0])


def test_control_input_shapes_fail_fast() -> None:
    values = _control_inputs(batch=1)
    with pytest.raises(ValueError, match="batch size"):
        process_residual_actions(
            **{**values, "mocap_targets": np.zeros((2, 26))},
            trajectory_steps=np.array([101]),
        )
    with pytest.raises(ValueError, match="active_joint_mask"):
        process_residual_actions(
            **{**values, "active_joint_mask": np.ones((1, 19), dtype=bool)},
            trajectory_steps=np.array([101]),
        )
    with pytest.raises(ValueError, match="ordered"):
        process_residual_actions(
            **{**values, "joint_lower": np.ones(26), "joint_upper": np.zeros(26)},
            trajectory_steps=np.array([101]),
        )
    with pytest.raises(ValueError, match="batch size"):
        check_termination(
            object_position=np.zeros((1, 3)),
            target_position=np.zeros((2, 3)),
            progress=np.array([0]),
            trajectory_lengths=np.array([1]),
            early_mask=np.array([False]),
        )
