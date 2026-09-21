"""Fresh-basin qualification and nonlinear contact inverse rejection."""
import numpy as np
from tools.identify_action005_fresh import consistency, qualify_basin, normal_basis, SCALES


def test_linear_response_qualified():
    zero = np.ones(19); slope = np.arange(1, 20.)
    result = consistency(zero, *[zero+a*slope for a in (.004, -.004, .002, -.002)])
    assert result['passed']
    np.testing.assert_allclose(result['derivative_per_rad'], slope)


def test_scale_dependent_and_unilateral_responses_rejected():
    zero = np.zeros(19); slope = SCALES.copy()
    assert not consistency(zero, .008*slope, -.008*slope, .002*slope, -.002*slope)['passed']
    assert not consistency(zero, .004*slope, zero, .002*slope, zero)['passed']


def test_force_curvature_not_hidden_by_angular_response():
    zero = np.zeros(19); slope = np.zeros(19); slope[16:] = 10000
    values = [a*slope for a in (.004, -.004, .002, -.002)]
    values[0][2] = 1; values[2][2] = .5
    assert not consistency(zero, *values)['passed']


def test_nonfinite_consistency_rejected():
    zero = np.zeros(19); bad = zero.copy(); bad[0] = np.nan
    assert not consistency(zero, bad, zero, zero, zero)['passed']


def basin_rows():
    return [dict(frame=f, bearing_normal_N=dict(thumb=9., index=3., middle=m, ring=2., pinky=2.),
                 topology=dict(index_pinky_span_m=s, opposed_sides=True, opposed_normal_cosines=[.95]*4))
            for f, m, s in [(380, .85, .079), (400, .5, .0793), (409, .4, .0797), (418, 0., .0809)]]


def test_sub80mm_historical_basin_explicitly_reported():
    rows = basin_rows(); old = {r['frame']: r for r in rows}
    result = qualify_basin(rows, old)
    assert result['passed'] and not result['all_spans_ge80mm']


def test_span_collapse_and_missing_lower_contact_reject():
    rows = basin_rows(); old = {r['frame']: r for r in basin_rows()}
    rows[-1]['topology']['index_pinky_span_m'] = .060
    assert not qualify_basin(rows, old)['passed']
    rows = basin_rows(); rows[-1]['bearing_normal_N']['pinky'] = 0
    assert not qualify_basin(rows, old)['passed']


def test_normal_basis_is_unit_disjoint_and_opens_surface():
    def geometry(q):
        return np.array([q[6]+2*q[7], 0, -q[16], 0, 0]),
    basis = normal_basis(geometry, np.zeros(42))
    np.testing.assert_allclose(basis.T@basis, np.eye(2))
    assert basis[6, 0] > 0 and basis[16, 1] < 0
    assert np.count_nonzero(basis[:6]) == 0
