"""Formal compact replay keeps multiobject scenes and arrival controls explicit."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from tools.replay_formal_compact import arrival_target,decode_row,bin_entry,CONTRACT,metadata_digest


def row_fixture():
    n=5
    return dict(index=dict(uuid='new',seed_uuid='source',is_generated=True,operator='cheyingtong'),
        trajectory_metadata=dict(data_fps=120,total_frames=n,hand_names=['right'],hand_slots=['right','left'],
          mano_hand_shapes=[[0]*10],object_names=['egg_cup','bowl'],trajectory_info=dict(object_move=[dict(object_name='egg_cup',start_frame=1,end_frame=4)])),
        provenance=dict(contract=CONTRACT,control_fps=120,physics_fps=480,physics_substeps_per_control=4,source_identity='egg_cup_04_062'),
        timestamp=(np.arange(n)/120).tolist(),
        hands=[dict(hand_name='right',urdf_dof=np.zeros((n,28)).tolist(),urdf_dof_target=np.arange(n*28).reshape(n,28).tolist()),dict(hand_name=None)],
        objects=[dict(pos=np.tile([0,0,.1],(n,1)).tolist(),rot_aa=np.zeros((n,3)).tolist()),dict(pos=np.tile([.2,0,.1],(n,1)).tolist(),rot_aa=np.zeros((n,3)).tolist())],
        reference=dict(hand_urdf_dof=np.zeros((n,28)).tolist(),object_pos=np.tile([0,0,.1],(n,1)).tolist(),object_rot_aa=np.zeros((n,3)).tolist(),source_frame_index=list(range(n))),
        command_reference_index=[0,0,1,2])


def test_multiobject_names_and_arrival_control():
    row=row_fixture();before=deepcopy(row)
    info,arrays=decode_row(row)
    assert info['active']=='egg_cup' and info['names']==['egg_cup','bowl'] and info['action']==4
    assert arrays['recorded_object_pos'].shape==(5,2,3)
    np.testing.assert_array_equal(arrival_target(arrays['targets'],1),np.arange(28,56))
    np.testing.assert_array_equal(arrival_target(arrays['targets'],4),np.arange(112,140))
    assert row==before
    for i in (0,5):
        with pytest.raises(ValueError):arrival_target(arrays['targets'],i)


@pytest.mark.parametrize('kind',['clock','frames','names','active','nonfinite'])
def test_malformed_contract_rejected(kind):
    row=row_fixture()
    if kind=='clock':row['provenance']['physics_fps']=400
    if kind=='frames':row['trajectory_metadata']['total_frames']=6
    if kind=='names':row['trajectory_metadata']['object_names']=['egg_cup','egg_cup']
    if kind=='active':row['provenance']['source_identity']='bowl_04_062'
    if kind=='nonfinite':row['hands'][0]['urdf_dof_target'][2][1]=float('nan')
    with pytest.raises(ValueError):decode_row(row)


def test_bin_opening_uses_actual_world_rim():
    mesh=np.array([[x,y,z]for x in(-.1,.1)for y in(-.1,.1)for z in(0,.2)])
    state=SimpleNamespace(xpos=np.array([[[.0,.0,.1],[.0,.0,0.]]]),xquat=np.array([[[1.,0,0,0],[1.,0,0,0]]]))
    event=bin_entry(state,0,0,1,mesh)
    assert event['inside_opening'] and event['below_rim'] and event['above_bin_bottom']
    state.xpos[0,0,0]=.09
    assert not bin_entry(state,0,0,1,mesh)['inside_opening']
    state.xpos[0,0]=[0,0,.21]
    assert not bin_entry(state,0,0,1,mesh)['below_rim']


def test_metadata_hash_is_canonical():
    assert metadata_digest({'a':1,'b':2})==metadata_digest({'b':2,'a':1})
    assert metadata_digest({'a':1})!=metadata_digest({'a':2})
