from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from tools import replay_capture_no_policy as raw
from tools.u1_raw_source import load_raw_source


@pytest.fixture
def source(monkeypatch):
    from sim.manorl import assets
    monkeypatch.setattr(assets, 'object_collision_vertices', lambda name: np.zeros((1,3)))
    manifest={'hands':{'right':{'betas':[0.]*10}}}
    row={'index':{'operator':'cheyingtong','uuid':'identity','scene':'pitcherbase'},
         'trajectory_metadata':{'hand_names':['left','right'],'mano_hand_shapes':[[1.]*10,[0.]*10],
                                'total_frames':3,'object_names':['pitcherbase'],
                                'trajectory_info':{'object_move':[{'object_name':'pitcherbase'}]}},
         'hands':[{'urdf_dof':np.full((3,28),9).tolist()}, {'urdf_dof':np.zeros((3,28)).tolist()}],
         'objects':[{'pos':np.tile([1.,2.,3.],(3,1)).tolist(),'rot_aa':np.zeros((3,3)).tolist()}],
         'timestamp':[0.,1/120,2/120]}
    return row,manifest


def test_right_slot_exact_grounding(source):
    row,manifest=source; before=deepcopy(row)
    s=raw.source_arrays(row,manifest,120)
    np.testing.assert_array_equal(s['original_commands'],np.zeros((3,28)))
    expected=np.zeros((3,28)); expected[:,2]=s['shift']
    np.testing.assert_array_equal(s['commands'],expected)
    np.testing.assert_array_equal(s['positions'][:,:,2],3+s['shift'])
    assert row==before


@pytest.mark.parametrize('failure',['missing_right','duplicate_right','betas','shape','nonfinite','clock'])
def test_invalid_source_rejected(source,failure):
    row,manifest=source; meta=row['trajectory_metadata']
    if failure=='missing_right':meta['hand_names']=['left','left']
    if failure=='duplicate_right':meta['hand_names']=['right','right']
    if failure=='betas':meta['mano_hand_shapes'][1][0]=1
    if failure=='shape':row['hands'][1]['urdf_dof'][0].pop()
    if failure=='nonfinite':row['hands'][1]['urdf_dof'][0][0]=float('nan')
    if failure=='clock':row['timestamp'][-1]=1.
    with pytest.raises(ValueError):raw.source_arrays(row,manifest,120)


def test_adapter_preserves_explicit_identity_and_profile(source,monkeypatch,tmp_path):
    import lance
    from sim.manorl import assets
    from sim.manorl.local_contact_repair import MODEL_FIELDS
    import tools.u1_placement_workspace as workspace
    row,manifest=source
    ds=SimpleNamespace(version=4,take=Mock(return_value=SimpleNamespace(to_pylist=lambda:[row])))
    dataset=Mock(return_value=ds); monkeypatch.setattr(lance,'dataset',dataset)
    activate=Mock(return_value=manifest);monkeypatch.setattr(raw,'activate_hand_profile',activate)
    model=SimpleNamespace(nu=28,nq=35,nv=34,nbody=1,body=lambda i:SimpleNamespace(name='world'),
                          joint=lambda n:SimpleNamespace(qposadr=[28]),
                          opt=SimpleNamespace(ccd_iterations=35))
    for field in MODEL_FIELDS:setattr(model,field,np.zeros(1))
    monkeypatch.setattr(assets,'compile_unified_model',lambda **kwargs:(None,model))
    monkeypatch.setattr(assets,'asset_provenance',lambda:{'pinned':True})
    check=Mock();monkeypatch.setattr(workspace,'assert_u1',check)
    inp,m,target,teacher,provenance=load_raw_source(tmp_path/'data',4,82,tmp_path/'assets',tmp_path/'manifest')
    dataset.assert_called_once_with(str(tmp_path/'data'),version=4)
    ds.take.assert_called_once_with([82],columns=raw.COLUMNS)
    activate.assert_called_once_with('cheyingtong',tmp_path/'assets',tmp_path/'manifest')
    assert provenance['index']['uuid']=='identity' and provenance['row']==82 and provenance['dataset_version']==4
    assert provenance['hand_side']=='right' and inp.hz==120
    assert target.shape==(3,28) and teacher['qpos'].shape==(3,35)
    assert model.opt.ccd_iterations == 16  # never silently retain MuJoCo's default35
    np.testing.assert_array_equal(inp.initial['qpos'],teacher['qpos'][0])
    np.testing.assert_array_equal(target,teacher['qpos'][:,:28])
    check.assert_called_once_with(model,inp)
    ds.version=5
    with pytest.raises(ValueError,match='version mismatch'):
        load_raw_source(tmp_path/'data',4,82,tmp_path/'assets',tmp_path/'manifest')
