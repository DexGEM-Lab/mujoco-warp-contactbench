from dataclasses import replace

import numpy as np
import pytest

from sim.manorl.abi import ResidualActionConfig
from tools.visualize_hand_residual_ranges import residual_bounds, sweep


def test_range_envelope_matches_decay_not_just_hard_cap():
    cfg = ResidualActionConfig()
    low, high, envelope = residual_bounds(np.zeros(22), np.full(22, -2.), np.full(22, 2.), cfg)
    scale = np.asarray(cfg.joint_scale) * cfg.joint_scale_multiplier
    caps = np.asarray(cfg.max_joint_offset) * cfg.joint_max_offset_multiplier
    offsets = np.zeros(22)
    for _ in range(400):
        offsets = np.clip(.9 * offsets + scale, -caps, caps)
    np.testing.assert_allclose(offsets, envelope, atol=1e-12)
    np.testing.assert_allclose(low, -high)
    assert envelope[2] == pytest.approx(.16)  # thumb twist
    for finger_dip in (9, 13, 17, 21):
        assert envelope[finger_dip] == pytest.approx(.1)
        assert caps[finger_dip] == pytest.approx(.2)


def test_model_limits_clip_residual_about_the_reference_pose():
    q = np.full(22, .02)
    low, high, _ = residual_bounds(q, np.zeros(22), np.full(22, .04))
    np.testing.assert_allclose(low, -.02)
    np.testing.assert_allclose(high, .02)
    with pytest.raises(ValueError, match="reference"):
        residual_bounds(np.full(22, 2.), np.zeros(22), np.ones(22))
    with pytest.raises(ValueError, match="decay"):
        residual_bounds(q, np.zeros(22), np.ones(22), replace(ResidualActionConfig(), gamma_joints=1.))


def test_visual_sweep_holds_both_endpoints_without_exceeding_bounds():
    assert sweep(0, -.1, .2) == 0
    assert sweep(1, -.1, .2) == 0
    assert sweep(.32, -.1, .2) == pytest.approx(-.1)
    assert sweep(.77, -.1, .2) == pytest.approx(.2)
    values = [sweep(p, -.1, .2) for p in np.linspace(0, 1, 96)]
    assert min(values) == pytest.approx(-.1)
    assert max(values) == pytest.approx(.2)
