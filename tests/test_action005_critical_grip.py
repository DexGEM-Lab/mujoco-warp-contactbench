import numpy as np
from scipy.spatial.transform import Rotation as R

from tools.search_action005_critical_grip import (
    candidates, contact_wrench, critical_frames, map_grip,
    pyramid_utilisation, score, wrench_reserve,
)


def test_deterministic_independent_bounded_candidates():
    plan = candidates()
    assert plan == candidates() and len(plan) == 32
    assert len({tuple(c['finger_delta_rad']) for c in plan}) == 32
    for c in plan:
        assert np.linalg.norm(c['translation_m']) <= .020+1e-12
        assert np.linalg.norm(c['rotation_vector_rad']) <= np.deg2rad(15)+1e-12
        assert max(abs(np.array(c['finger_delta_rad']))) <= .3
    assert np.linalg.matrix_rank(np.array([c['finger_delta_rad'] for c in plan])) == 22


def test_mapping_preserves_full_object_relative_pose():
    grip = np.zeros(35); grip[28:31] = [1, 2, 3]
    grip[31:35] = R.from_euler('XYZ', [.3, .2, .1]).as_quat(scalar_first=True)
    grip[:3] = [1.02, 2.1, 3.04]; grip[3:6] = [.4, -.2, .8]
    dest = grip.copy(); dest[28:31] = [-1, -2, 1]
    dest[31:35] = R.from_euler('XYZ', [1.2, -.7, .8]).as_quat(scalar_first=True)
    mapped = map_grip(grip, dest, 28, candidates()[0])
    old = R.from_quat(grip[31:35], scalar_first=True)
    new = R.from_quat(dest[31:35], scalar_first=True)
    np.testing.assert_allclose(new.inv().apply(mapped[:3]-dest[28:31]), old.inv().apply(grip[:3]-grip[28:31]))
    np.testing.assert_allclose((new.inv()*R.from_euler('XYZ', mapped[3:6])).as_matrix(), (old.inv()*R.from_euler('XYZ', grip[3:6])).as_matrix())
    np.testing.assert_array_equal(mapped[28:], dest[28:])


def test_tilt_selection_retains_teacher_quaternion():
    q = np.zeros((6, 35))
    q[:, 31:35] = R.from_euler('XYZ', [[x, 0, 33] for x in [0, 50, 70, 90, 110, 122.5]], degrees=True).as_quat(scalar_first=True)
    selected = critical_frames(q, 28)
    assert [o['frame'] for o in selected] == [1, 2, 3, 4, 5]
    for o in selected:
        np.testing.assert_array_equal(o['quaternion_wxyz'], q[o['frame'], 31:35])


def test_contact_torque_sign_and_basis():
    basis = R.from_euler('z', 90, degrees=True).as_matrix()
    value = contact_wrench([1, 0, 0], basis, [1, 0, 0, 0, 0, 2], -1, np.zeros(3))
    np.testing.assert_allclose(value, [0, 1, 0, 0, 0, -1], atol=1e-12)
    assert pyramid_utilisation([2, .5, -.5, 0, 0, 0], [1]*5, 3) == .5


def test_wrench_reserve_for_opposed_pinch():
    contacts = []
    for sign in [-1, 1]:
        contacts.append(dict(finger='thumb', position=[-sign, 0, 0], basis=np.diag([sign, sign, 1]).tolist(),
                             local_wrench=[1, 0, 0, 0, 0, 0], dimension=3,
                             friction=[1]*5, object_sign=1))
    assert abs(wrench_reserve(contacts, np.zeros(3), [0, 0, -1])-1) < 1e-8
    assert wrench_reserve(contacts[:1], np.zeros(3), [0, 0, -1]) == -1


def test_strict_scoring_does_not_accept_short_hold_or_saturation():
    row = dict(five_bearing=True, drift_m=.001, drift_deg=1., speed_m_s=.0001,
               speed_deg_s=.1, saturated=False, support=False, friction_reserve=.1,
               wrench_reserve=.1, net_wrench=[0]*6)
    rows = [dict(row, frame=f) for f in range(1, 241)]
    assert score(rows, 'duration')['passed']
    assert not score(rows[:100], 'relative_drift')['passed']
    rows[0]['saturated'] = True
    assert not score(rows, 'duration')['passed']
    rows[0]['saturated'] = False
    rows[-1]['wrench_reserve'] = 0
    assert not score(rows, 'duration')['passed']
