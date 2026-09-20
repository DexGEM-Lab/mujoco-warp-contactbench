import numpy as np
import pytest

from tools.fit_uniform_direct_parent import smooth_time, target_correction, wrap_rotations


def test_wrap_rotations_shortest_branch():
    error = np.zeros((2, 28))
    error[:, 3] = 2 * np.pi - 0.1
    wrapped = wrap_rotations(error)
    np.testing.assert_allclose(wrapped[:, 3], -0.1, atol=1e-12)


def test_smooth_time_preserves_constant():
    values = np.ones((8, 28)) * 0.25
    np.testing.assert_array_equal(smooth_time(values), values)


def test_target_correction_is_bounded_and_keeps_frame_zero():
    teacher = np.ones((6, 28)) * 10.0
    replay = np.zeros((6, 28))
    correction = target_correction(
        teacher, replay, translation_gain=1.0, rotation_gain=1.0, finger_gain=1.0
    )
    np.testing.assert_array_equal(correction[0], 0.0)
    assert np.max(np.abs(correction[1:, :3])) <= 0.03
    assert np.max(np.abs(correction[1:, 3:6])) <= np.deg2rad(5) + 1e-12
    assert np.max(np.abs(correction[1:, 6:])) <= np.deg2rad(4) + 1e-12


def test_target_correction_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        target_correction(
            np.zeros((3, 28)), np.zeros((4, 28)),
            translation_gain=1, rotation_gain=1, finger_gain=1,
        )
