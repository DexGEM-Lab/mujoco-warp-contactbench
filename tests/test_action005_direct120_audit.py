import numpy as np
from scipy.spatial.transform import Rotation as R
from tools.audit_action005_direct120 import motion_metrics
from tools.verify_action005_source_contacts import valid_witness, longest


def test_failed_zero_distance_is_not_contact():
    assert not valid_witness(0., [0, 0, 0, .01, 0, 0])
    assert not valid_witness(0., np.zeros(6))
    assert valid_witness(-.002, [0, 0, 0, -.002, 0, 0])
    assert valid_witness(.003, [0, 0, 0, .003, 0, 0])
    assert not valid_witness(.003, [0, 0, 0, np.nan, 0, 0])


def test_temporal_persistence():
    assert longest([0, 1, 1, 0, 1, 1, 1, 0]) == 3
    assert longest([0, 0]) == 0


def test_deep_return_semantics():
    p = np.array([[0, 0, 0], [0, 0, .1], [0, 0, .1], [0, 0, 0]])
    r = R.from_euler('x', np.array([0, 60, 130, 0])[:, None], degrees=True)
    metrics = motion_metrics(p, r, np.zeros_like(p), np.arange(4)/120)
    assert metrics['deep_interval'] == [2, 2]
    assert metrics['return_upright_frame'] == 3
    assert metrics['max_tilt_deg'] > 129
    assert metrics['final_position_from_start_cm'] == 0
