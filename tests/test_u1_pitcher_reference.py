import json
from pathlib import Path

import numpy as np
import pytest

from tools.build_u1_pitcher_reference import hold_fingers, position_registration


def recipe():
    return json.loads((Path(__file__).parents[1]/'recipes/u1_pitcher_row82.json').read_text())


def test_grip_hold_preserves_wrist_and_prefix_then_releases_thumb_first():
    r=recipe()
    target=np.arange(r['frames'])[:,None]*np.ones((1,28))*.001
    out=hold_fingers(target,r)
    np.testing.assert_array_equal(out[:401],target[:401])
    np.testing.assert_array_equal(out[:,:6],target[:,:6])
    expected=target[400]+np.array(r['loop_preload_target_delta'])
    np.testing.assert_allclose(out[500,6:],expected[6:])
    np.testing.assert_array_equal(out[1168,6:12],target[1168,6:12])
    np.testing.assert_allclose(out[1168,12:],expected[12:])
    np.testing.assert_array_equal(out[1220:],target[1220:])


def test_registration_is_position_only_and_rejoins_reference():
    r=recipe();offset=position_registration(r['frames'],r)
    np.testing.assert_array_equal(offset[:441],np.zeros((441,3)))
    np.testing.assert_allclose(offset[600],r['pour_offset_m'])
    np.testing.assert_allclose(offset[1100],r['landing_offset_m'])
    np.testing.assert_array_equal(offset[1500:],np.zeros_like(offset[1500:]))
    np.testing.assert_array_equal(offset[:,2],np.zeros(r['frames']))
    assert np.max(np.abs(np.diff(offset,axis=0)))<.002


def test_rejects_invalid_preload_and_phase_order():
    r=recipe();r['loop_preload_target_delta'][0]=.1
    with pytest.raises(ValueError,match='wrist'):
        hold_fingers(np.zeros((2196,28)),r)
    r=recipe();r['return_ramp']=[500,100]
    with pytest.raises(ValueError,match='ordered'):
        position_registration(2196,r)
