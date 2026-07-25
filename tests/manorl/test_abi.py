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
        "raw_actions": np.full((batch, 28), 2.0),
        "cumulative_offset": np.full((batch, 3), 0.02),
        "cumulative_joint_offset": np.full((batch, 22), 0.2),
        "mocap_targets": np.zeros((batch, 28)),
        "joint_lower": np.full(28, -10.0),
        "joint_upper": np.full(28, 10.0),
        "active_joint_mask": np.array([[True] * 10 + [False] * 12] * batch),
    }


def test_residual_control_clips_masks_and_uses_target_early_interval() -> None:
    config = ResidualActionConfig()
    assert config.position_scale == (0.002, 0.002, 0.002)
    assert config.gamma_xy == config.gamma_z == 0.9
    assert config.max_position_offset == (0.02, 0.02, 0.02)
    assert config.joint_scale == (
        0.02,
        0.02,
        0.02,
        0.02,
        0.01,
        0.005,
        *(0.01, 0.01, 0.015, 0.005) * 4,
    )
    assert config.max_joint_offset == (
        0.2,
        0.2,
        0.2,
        0.2,
        0.1,
        0.05,
        *(0.1, 0.1, 0.15, 0.1) * 4,
    )
    assert len(config.joint_scale) == len(config.max_joint_offset) == 22
    assert config.joint_scale_multiplier == 1.0
    assert config.joint_max_offset_multiplier == 1.0
    assert config.early_phase_steps == 30
    values = _control_inputs()
    result = process_residual_actions(
        **values,
        trajectory_steps=np.array([0, 29, 30]),
    )
    np.testing.assert_array_equal(result.actions[0], 0.0)
    np.testing.assert_array_equal(result.actions[1], 0.0)
    np.testing.assert_array_equal(result.actions[2, 16:28], 0.0)
    np.testing.assert_allclose(result.actions[2, :16], 1.0)
    np.testing.assert_array_equal(result.cumulative_offset[:2], 0.0)
    np.testing.assert_allclose(result.cumulative_offset[2], [0.002, 0.002, 0.002])
    np.testing.assert_array_equal(result.cumulative_joint_offset[:2], 0.0)
    np.testing.assert_allclose(
        result.cumulative_joint_offset[2, :10],
        [0.02, 0.02, 0.02, 0.02, 0.01, 0.005, 0.01, 0.01, 0.015, 0.005],
    )
    np.testing.assert_array_equal(result.cumulative_joint_offset[2, 10:], 0.0)
    # Rotation is immediate, unlike the position and joint cumulative residuals.
    np.testing.assert_allclose(result.targets[2, 3:6], 0.00025)


def test_joint_scale_and_cap_multipliers_apply_exactly_once() -> None:
    values = _control_inputs(batch=1)
    values["raw_actions"][:] = 1.0
    values["cumulative_offset"][:] = 0.0
    values["cumulative_joint_offset"][:] = 0.0
    values["active_joint_mask"][:] = True
    config = ResidualActionConfig(
        joint_scale_multiplier=1.5,
        joint_max_offset_multiplier=1.5,
    )
    base_scales = np.asarray(config.joint_scale)
    base_caps = np.asarray(config.max_joint_offset)

    first = process_residual_actions(
        **values,
        trajectory_steps=np.array([51]),
        config=config,
    )
    np.testing.assert_allclose(
        first.cumulative_joint_offset,
        1.5 * base_scales[None, :],
    )
    np.testing.assert_allclose(first.cumulative_offset, [[0.002, 0.002, 0.002]])

    capped = process_residual_actions(
        **{
            **values,
            "cumulative_joint_offset": 10.0 * base_caps[None, :],
        },
        trajectory_steps=np.array([52]),
        config=config,
    )
    np.testing.assert_allclose(
        capped.cumulative_joint_offset,
        1.5 * base_caps[None, :],
    )


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_joint_multipliers_must_be_finite_and_positive(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ResidualActionConfig(joint_scale_multiplier=value)
    with pytest.raises(ValueError, match="finite and positive"):
        ResidualActionConfig(joint_max_offset_multiplier=value)


def test_position_action_unit_maps_signed_xy_and_z_steps() -> None:
    values = _control_inputs(batch=2)
    values["raw_actions"][:] = 0.0
    values["raw_actions"][:, :3] = [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
    values["cumulative_offset"][:] = 0.0
    values["cumulative_joint_offset"][:] = 0.0
    result = process_residual_actions(**values, trajectory_steps=np.array([51, 51]))
    np.testing.assert_allclose(
        result.cumulative_offset,
        [[0.002, 0.002, 0.002], [-0.002, -0.002, -0.002]],
        rtol=0,
        atol=1e-12,
    )


def test_residual_control_accumulates_only_active_joints_and_clamps() -> None:
    values = _control_inputs(batch=1)
    values["raw_actions"][:] = 1.0
    values["cumulative_offset"][:] = [0.019, 0.019, 0.019]
    values["cumulative_joint_offset"][:] = 0.0
    result = process_residual_actions(**values, trajectory_steps=np.array([51]))
    np.testing.assert_allclose(result.cumulative_offset, [[0.0191, 0.0191, 0.0191]])
    np.testing.assert_allclose(
        result.cumulative_joint_offset[0, :10],
        [0.02, 0.02, 0.02, 0.02, 0.01, 0.005, 0.01, 0.01, 0.015, 0.005],
    )
    np.testing.assert_array_equal(result.cumulative_joint_offset[0, 10:], 0.0)
    disabled = process_residual_actions(
        **values,
        trajectory_steps=np.array([51]),
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
        for step in range(51, 451):
            result = process_residual_actions(
                **{**values, "cumulative_offset": offset}, trajectory_steps=np.array([step])
            )
            offset = result.cumulative_offset
        expected_cap = sign * np.array([[0.02, 0.02, 0.02]])
        np.testing.assert_allclose(offset, expected_cap, atol=1e-12)
        capped = process_residual_actions(
            **{**values, "cumulative_offset": expected_cap}, trajectory_steps=np.array([451])
        )
        np.testing.assert_allclose(capped.cumulative_offset, expected_cap)


def test_joint_recurrence_converges_within_configured_caps() -> None:
    values = _control_inputs(batch=1)
    values["cumulative_offset"][:] = 0.0
    values["active_joint_mask"][:] = True
    expected_scales = np.array(
        [0.02, 0.02, 0.02, 0.02, 0.01, 0.005]
        + [0.01, 0.01, 0.015, 0.005] * 4
    )
    expected_caps = np.array(
        [0.2, 0.2, 0.2, 0.2, 0.1, 0.05]
        + [0.1, 0.1, 0.15, 0.1] * 4
    )
    expected_steady_state = np.minimum(
        expected_caps,
        expected_scales / (1.0 - ResidualActionConfig().gamma_joints),
    )
    first_step = process_residual_actions(
        **{
            **values,
            "raw_actions": np.ones((1, 28)),
            "cumulative_joint_offset": np.zeros((1, 22)),
        },
        trajectory_steps=np.array([51]),
    )
    np.testing.assert_allclose(first_step.cumulative_joint_offset, expected_scales[None, :])
    np.testing.assert_allclose(first_step.cumulative_offset, [[0.002, 0.002, 0.002]])
    np.testing.assert_allclose(first_step.targets[:, 3:6], 0.00025)
    for sign in (-1.0, 1.0):
        values["raw_actions"][:] = sign
        offset = np.zeros((1, 22))
        for step in range(51, 451):
            result = process_residual_actions(
                **{**values, "cumulative_joint_offset": offset}, trajectory_steps=np.array([step])
            )
            offset = result.cumulative_joint_offset
        np.testing.assert_allclose(
            offset,
            sign * expected_steady_state[None, :],
            atol=1e-12,
        )
        capped = process_residual_actions(
            **{**values, "cumulative_joint_offset": sign * expected_caps[None, :]},
            trajectory_steps=np.array([451]),
        )
        expected_from_cap = np.minimum(
            expected_caps,
            ResidualActionConfig().gamma_joints * expected_caps + expected_scales,
        )
        np.testing.assert_allclose(
            capped.cumulative_joint_offset,
            sign * expected_from_cap[None, :],
        )


def test_early_phase_uses_per_environment_starts() -> None:
    np.testing.assert_array_equal(
        early_phase_mask(np.array([0, 29, 30, 35]), starts=np.array([0, 0, 0, 6])),
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
    with pytest.raises(ValueError, match="match the action finger width"):
        process_residual_actions(
            **{
                **values,
                "cumulative_joint_offset": np.zeros((1, 20)),
            },
            trajectory_steps=np.array([101]),
        )
    with pytest.raises(ValueError, match="batch size"):
        process_residual_actions(
            **{**values, "mocap_targets": np.zeros((2, 28))},
            trajectory_steps=np.array([101]),
        )
    with pytest.raises(ValueError, match="active_joint_mask"):
        process_residual_actions(
            **{**values, "active_joint_mask": np.ones((1, 21), dtype=bool)},
            trajectory_steps=np.array([101]),
        )
    with pytest.raises(ValueError, match="ordered"):
        process_residual_actions(
            **{**values, "joint_lower": np.ones(28), "joint_upper": np.zeros(28)},
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
