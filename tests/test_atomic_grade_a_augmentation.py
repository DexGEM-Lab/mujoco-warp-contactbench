from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from tools.pilot_atomic_grade_a_augmentation import (
    PREFIX, hold_extend, native_pair_forces, plan_deltas, prefix_target,
)


def test_confirmed_pilot_has27_parents_three_equal_radii_and_signed_axes():
    plan = plan_deltas()
    assert len(plan) == 27
    assert Counter(p['radius_m'] for p in plan) == {0.05:9, 0.10:9, 0.15:9}
    assert {p['sector'] for p in plan} == set(range(8))
    assert len({(p['rotation_axis'],p['rotation_sign']) for p in plan}) == 6
    for p in plan:
        d = np.asarray(p['delta'])
        assert np.linalg.norm(d[:3]) == pytest.approx(p['radius_m'])
        assert d[2] == pytest.approx(p['radius_m']*.5)
        assert np.count_nonzero(d[3:]) == 1
        assert np.max(np.abs(d[3:])) == pytest.approx(np.pi/6)
    assert plan == plan_deltas()


def test_physical_reference_prefix_does_not_edit_control_suffix_or_fingers():
    base = np.full((8,28),0.1,dtype=np.float32)
    teacher = base.copy()
    teacher[1,:6] += 2e-6
    delta = np.array([.08660254,0,.05,0,0,np.pi/6])
    child,initial,report = prefix_target(base,teacher,delta)
    assert child.shape == (PREFIX+8,28)
    assert child[PREFIX:].tobytes() == base.tobytes()
    np.testing.assert_array_equal(child[:PREFIX,6:],np.repeat(base[:1,6:],PREFIX,axis=0))
    np.testing.assert_allclose(initial[:6],teacher[0,:6]+delta,atol=1e-7)
    np.testing.assert_array_equal(initial[6:],teacher[0,6:])
    np.testing.assert_allclose(child[PREFIX-1,:6],2*teacher[0,:6]-teacher[1,:6],atol=1e-7)
    assert report['parent_suffix_sha256'] == report['child_suffix_sha256']
    assert report['reference_C1'] and not report['control_C1_exact']
    assert max(map(abs,report['control_splice_residual'])) > 1e-6


def test_stale_ancestor_reference_is_not_accepted_as_current_parent_start():
    base = np.zeros((8,28),dtype=np.float32)
    teacher = base.copy();teacher[:,0] += .03
    with pytest.raises(AssertionError):
        prefix_target(base,teacher,np.zeros(6))


def test_prefix_hold_reference_does_not_transform_objects():
    source = np.arange(24).reshape(8,3)
    child = hold_extend(source)
    np.testing.assert_array_equal(child[:PREFIX],np.repeat(source[:1],PREFIX,axis=0))
    np.testing.assert_array_equal(child[PREFIX:],source)


def test_native_contact_force_gather_uses_its_own_world_and_addresses():
    b = SimpleNamespace(count=3, world=np.array([0,1,0]),
                        addresses=np.array([[0,1,2,3],[1,2,3,4],[-1,-1,-1,-1]]),
                        dimension=np.array([3,3,3]),nefc=np.array([4,5]),
                        constraint_force=np.array([[1,2,3,4,0],[100,10,20,30,40]]))
    pyramid,valid=native_pair_forces(b)
    np.testing.assert_array_equal(pyramid[:2],[[1,2,3,4],[10,20,30,40]])
    np.testing.assert_array_equal(valid,[True,True,False])
    np.testing.assert_array_equal(pyramid[2],0)
