import copy
import numpy as np
import pytest
from tools.u1_placement_workspace import edit_window


def case():
    base = np.arange(20*28, dtype=float).reshape(20,28)/1000
    recipe = {'window':[4,14], 'knots':[
        {'frame':4,'target':base[4].tolist()},
        {'frame':9,'target':(base[9]+.1).tolist()},
        {'frame':14,'target':base[14].tolist()}]}
    return base, recipe


def test_exact_prefix_tail_and_knots_without_mutating_source():
    base, recipe = case()
    original = base.copy()
    out = edit_window(base, recipe)
    assert out[:5].tobytes() == base[:5].tobytes()
    assert out[14:].tobytes() == base[14:].tobytes()
    np.testing.assert_array_equal(out[9], recipe['knots'][1]['target'])
    np.testing.assert_array_equal(base, original)
    assert not np.shares_memory(out, base)


@pytest.mark.parametrize('fault', ['frame0', 'reversed', 'nonfinite', 'duplicate', 'outside', 'endpoint', 'fractional'])
def test_rejects_invalid_or_nonjoining_edits(fault):
    base, recipe = case()
    recipe = copy.deepcopy(recipe)
    if fault == 'frame0': recipe['window'][0] = 0
    if fault == 'reversed': recipe['window'] = [14,4]
    if fault == 'nonfinite': recipe['knots'][1]['target'][5] = float('nan')
    if fault == 'duplicate': recipe['knots'][1]['frame'] = 4
    if fault == 'outside': recipe['knots'][1]['frame'] = 15
    if fault == 'endpoint': recipe['knots'][0]['target'][0] += .01
    if fault == 'fractional': recipe['knots'][1]['frame'] = 9.5
    with pytest.raises(ValueError): edit_window(base, recipe)


def test_u1_rejects_default_ccd_iterations():
    from types import SimpleNamespace
    import mujoco as mj
    from sim.manorl.local_contact_repair import MODEL_FIELDS
    from tools.u1_placement_workspace import assert_u1
    model=SimpleNamespace(nu=28,na=0,opt=SimpleNamespace(
        timestep=1/480,cone=mj.mjtCone.mjCONE_PYRAMIDAL,impratio=1,ccd_iterations=16))
    for field in MODEL_FIELDS:
        setattr(model,field,np.zeros((28,10)) if field.startswith('actuator_') and field.endswith('prm') else np.zeros(1))
    model.actuator_dyntype=np.full(28,mj.mjtDyn.mjDYN_NONE)
    model.actuator_gaintype=np.full(28,mj.mjtGain.mjGAIN_FIXED)
    model.actuator_biastype=np.full(28,mj.mjtBias.mjBIAS_AFFINE)
    model.actuator_gainprm[:,0]=100
    model.actuator_biasprm[:,1]=-100
    inp=SimpleNamespace(hz=120,manifest={'native':{k:getattr(model,k).copy() for k in MODEL_FIELDS}})
    contract=assert_u1(model,inp)
    assert contract['ccd_iterations']==16
    assert contract['single_world_warp_capacity']=={'nconmax':1024,'nccdmax':256,'njmax':4096}
    model.opt.ccd_iterations=35
    with pytest.raises(AssertionError,match='CCD16'):
        assert_u1(model,inp)


def test_interpolation_has_no_overshoot():
    base, recipe = case()
    out = edit_window(base, recipe)
    for a,b in zip(recipe['knots'][:-1],recipe['knots'][1:]):
        lo = np.minimum(a['target'],b['target'])
        hi = np.maximum(a['target'],b['target'])
        assert np.all(out[a['frame']:b['frame']+1] >= lo-1e-15)
        assert np.all(out[a['frame']:b['frame']+1] <= hi+1e-15)
