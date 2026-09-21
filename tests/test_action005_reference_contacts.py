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
    assert np.linalg.norm(BOUNDS[:3]) <= .020 + 1e-12
    assert np.linalg.norm(BOUNDS[3:6]) <= np.deg2rad(15) + 1e-12
    np.testing.assert_array_equal(BOUNDS[6:], 1.)


def test_smooth_clips_without_overshoot():
    np.testing.assert_array_equal(smooth(np.array([-1., 0., 1., 2.])), [0., 0., 1., 1.])


def test_lower_pinky_is_leverage_not_axial_error():
    from tools.fit_action005_reference_contacts import topology_metrics
    points = np.array([[0, -.023301, .006], [0, .030509, .041357],
                       [0, .027905, -.004688], [0, .026892, -.024352],
                       [0, .027854, -.052017]])
    normals = np.array([[0, 1, 0]]+[[0, -1, 0]]*4)
    result = topology_metrics(points, normals)
    assert result['opposed_sides'] and result['axial_order']
    assert abs(result['index_pinky_span_m']-.093374) < 1e-10
    assert min(result['opposed_normal_cosines']) == 1


def test_missing_bearing_finger_cannot_pass_topology():
    from tools.fit_action005_reference_contacts import bearing_topology
    result = bearing_topology([])
    assert not result['complete'] and result['index_pinky_span_m'] == 0


def test_retention_rejects_gap_induced_lower_contact_loss():
    from tools.fit_action005_reference_contacts import retention_failures
    row = dict(topology=dict(complete=True, opposed_sides=True, axial_order=True,
                            opposed_normal_cosines=[1]*4, index_pinky_span_m=.093),
               bearing_normal_N=dict(ring=1., pinky=0.), max_penetration_m=.0006,
               drift_deg=1., relative_angular_acceleration_deg_s2=0.)
    assert retention_failures(row) == ['lower_finger_bearing']
    row['bearing_normal_N']['pinky'] = 1.
    assert retention_failures(row) == []


def test_missing_middle_preserves_measured_index_pinky_span():
    from tools.fit_action005_reference_contacts import bearing_topology
    contacts = [dict(finger=f, local_wrench=[1.], object_local_position=[0, y, z],
                     object_local_normal=[0, -np.sign(y), 0])
                for f, y, z in [('thumb', -.02, 0), ('index', .03, .041),
                                ('ring', .03, -.024), ('pinky', .03, -.052)]]
    result = bearing_topology(contacts)
    assert result['missing_fingers'] == ['middle']
    assert result['opposed_sides'] and not result['complete']
    assert abs(result['index_pinky_span_m']-.093) < 1e-10
    assert min(result['opposed_normal_cosines']) == 1.
