"""Tests for large-pose four-action U1 augmentation plan and prefix construction."""
import numpy as np
import pytest

from tools.run_u1_largepose_campaign import (ACTIONS, PREFIX_FRAMES, RADII_MM,
                                             RESERVES, ROT_AXES, ROT_DEG, SEED,
                                             build_plan, make_target,
                                             prefix_envelope)


def test_plan_shape_and_coverage():
    slots = build_plan()
    assert len(slots) == 160
    assert {s['radius_mm'] for s in slots} == set(RADII_MM)
    from collections import Counter
    radius = Counter(s['radius_mm'] for s in slots)
    assert radius[50] == 54 and radius[100] == 53 and radius[150] == 53
    sector = Counter(s['azimuth_sector'] for s in slots)
    assert set(sector.values()) == {20}
    rot = Counter((s['rotation_axis'], s['rotation_sign']) for s in slots)
    assert len(rot) == 6
    assert sum(rot.values()) == 160
    assert sum(1 for s in slots if s['kind'] == 'base') == 144
    assert sum(1 for s in slots if s['kind'] == 'extra') == 16
    for s in slots:
        assert len(s['candidates']) == RESERVES


def test_plan_deterministic_and_unique_deltas():
    a = build_plan(SEED)
    b = build_plan(SEED)
    assert a == b
    deltas = set()
    for s in a:
        for c in s['candidates']:
            deltas.add(tuple(round(x, 12) for x in c['delta']))
    assert len(deltas) == 160 * RESERVES


def test_candidate_geometry_exact():
    slots = build_plan()
    for s in slots:
        for c in s['candidates']:
            d = np.asarray(c['delta'])
            np.testing.assert_allclose(np.linalg.norm(d[:3]), s['radius_mm'] / 1000.0, rtol=1e-12)
            assert d[2] > 0  # historical approach starts above, never through the floor
            rot = np.degrees(np.abs(d[3:]))
            assert np.count_nonzero(rot > 1e-9) == 1
            np.testing.assert_allclose(rot[rot > 1e-9][0], ROT_DEG, rtol=1e-12)


def test_prefix_envelope_endpoints():
    delta = np.array([0.05, 0.0, 0.0, np.deg2rad(30), 0, 0])
    env = prefix_envelope(PREFIX_FRAMES, delta)
    np.testing.assert_allclose(env[0], delta, rtol=1e-12)
    np.testing.assert_allclose(env[-1], 0.0, atol=1e-15)
    # make_target adds the historical endpoint-smooth 4cm vertical arc.
    base = np.zeros((2, 28), dtype=float)
    target = make_target(base, delta, PREFIX_FRAMES)
    z_residual = target[:PREFIX_FRAMES, 2] - env[:, 2]
    assert abs(z_residual[0]) < 1e-15 and abs(z_residual[-1]) < 1e-15
    np.testing.assert_allclose(z_residual.max(), .04, rtol=3e-4)
    # Quintic has zero slope at both endpoints; first finite difference is O(dt^2).
    slope0 = env[1] - env[0]
    slope1 = env[-1] - env[-2]
    assert np.linalg.norm(slope0) < 1e-4 and np.linalg.norm(slope1) < 1e-4


def test_make_target_suffix_byte_exact():
    base = np.arange(641 * 28, dtype=np.float32).reshape(641, 28) / 1000.0
    delta = np.array([0.05 / np.sqrt(3)] * 3 + [np.deg2rad(30), 0, 0])
    target = make_target(base, delta, PREFIX_FRAMES)
    assert target.shape == (PREFIX_FRAMES + 641, 28)
    assert target[PREFIX_FRAMES:].tobytes() == base.tobytes()
    np.testing.assert_array_equal(target[:, 6:], np.concatenate(
        [np.repeat(base[:1], PREFIX_FRAMES, axis=0), base], axis=0)[:, 6:])
