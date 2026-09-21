"""Qualification must fail before probing a different native contact history."""
import numpy as np
from tools.reconstruct_action005_native import compare_trace, compare_forces, FINGERS


def traces():
    return {k: np.zeros((410, 4)) for k in ('qpos', 'qvel', 'ctrl')}


def test_exact_and_tolerance_qualified_reconstruction():
    old = traces(); new = traces()
    assert compare_trace(new, old)['passed']
    new['qpos'][380, 0] = 5e-7
    new['qvel'][400, 0] = 5e-5
    result = compare_trace(new, old)
    assert result['passed'] and not result['bitwise']['qpos']
    assert result['bitwise']['ctrl']


def test_any_prefix_mismatch_blocks_even_matching_checkpoints():
    old = traces(); new = traces()
    new['qvel'][29, 0] = .001
    assert not compare_trace(new, old)['passed']


def test_controls_must_be_bitwise_equal():
    old = traces(); new = traces()
    new['ctrl'][1, 0] = 1e-12
    assert not compare_trace(new, old)['passed']


def test_nonfinite_state_is_rejected():
    old = traces(); new = traces()
    new['qpos'][20, 0] = np.nan
    assert not compare_trace(new, old)['passed']


def test_force_tolerance_does_not_allow_bearing_sequence_change():
    old = dict.fromkeys(FINGERS, 1.); new = old.copy()
    old['middle'] = .199; new['middle'] = .201
    assert not compare_forces(new, old)['passed']


def test_v4_frame380_force_mismatch_rejected():
    old = dict(zip(FINGERS, [9.36077595, 3.40321302, .86679018, 1.99815953, 2.08324146]))
    new = dict(zip(FINGERS, [9.42161560, 3.47321859, .85670680, 2.01934743, 2.04454613]))
    assert not compare_forces(new, old)['passed']


def test_v4_frame400_small_force_difference_qualified():
    old = dict(zip(FINGERS, [9.25044250, 3.50260711, .52521467, 1.68297362, 1.74467301]))
    new = dict(zip(FINGERS, [9.26052856, 3.50379324, .53020662, 1.68892050, 1.74053645]))
    assert compare_forces(new, old)['passed']
