import numpy as np
from tools.fit_action005_reference_contacts import BOUNDS, phase_weights, smooth


def test_bounded_smooth_reference_window_and_order():
    weights = phase_weights(1321)
    assert weights.shape == (1321, 28)
    assert np.all((weights >= 0) & (weights <= 1))
    assert not weights[:225].any()
    assert not weights[1050:].any()
    assert weights[260, 12] > 0 and weights[260, 6] == 0
    assert weights[1010, 6] == 0 and weights[1010, 12] > 0
    assert np.max(abs(np.diff(weights, axis=0))) < .028


def test_wrist_correction_bounds_are_norm_bounded():
    assert np.linalg.norm(BOUNDS[:3]) <= .015 + 1e-12
    assert np.linalg.norm(BOUNDS[3:6]) <= np.deg2rad(8) + 1e-12
    np.testing.assert_array_equal(BOUNDS[6:], 1.)


def test_smooth_clips_without_overshoot():
    np.testing.assert_array_equal(smooth(np.array([-1., 0., 1., 2.])), [0., 0., 1., 1.])
