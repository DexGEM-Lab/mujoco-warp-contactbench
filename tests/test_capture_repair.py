"""Data-edit invariants; physical lift evidence is in the versioned replay artifacts."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from replay_pose_edits import apply_replay_edits
from export_repaired_motion import export_motion
from replay_repaired_capture import load_command_track
from sim.manorl.contracts import TrajectoryIdentity
from sim.manorl.trajectory import ReferenceTrajectory


def reference():
    count=401
    q=np.zeros((count,28));q[:,2]=.2
    positions=np.tile([0.,0.,.1],(count,1))
    return ReferenceTrajectory(
        identity=TrajectoryIdentity('',5,49,0,'source-uuid','capture','bowl_09_050',0,count,100,300),
        dataset_version=5, source_indices=np.arange(count), timestamps=np.arange(count)/100.,
        q_ref=q, q_ref_by_side={'right':q},
        object_pos_raw=positions, object_pos=positions.copy(),
        object_quat_xyzw=np.tile([0.,0.,0.,1.],(count,1)), object_z_shift=0.,
        reference_fps=100,control_fps=100,movement_start_step=100,movement_end_step=300,
        scene_object_types=('bowl','cuboid1'),scene_object_initial_pos=np.array([[0.,0.,.1],[0.,0.,.02]]),
        scene_object_initial_quat_xyzw=np.tile([0.,0.,0.,1.],(2,1)),
    )


def test_object_edit_preserves_hand_passive_and_source():
    base=reference();old_q=base.q_ref.copy();old_pos=base.object_pos.copy()
    new=apply_replay_edits(base,{'translation':[.006,.012,0.]})
    np.testing.assert_array_equal(base.q_ref,old_q)
    np.testing.assert_array_equal(base.object_pos,old_pos)
    np.testing.assert_array_equal(new.q_ref,base.q_ref)
    np.testing.assert_allclose(new.object_pos-base.object_pos,np.tile([.006,.012,0.],(401,1)))
    np.testing.assert_array_equal(new.scene_object_initial_pos[1],base.scene_object_initial_pos[1])
    np.testing.assert_array_equal(new.source_indices,base.source_indices)


def test_grasp_and_stage_preserve_unmodified_prefix():
    base=reference()
    edited=apply_replay_edits(base,{
        'object_relative_grasp':{'translation':[.1,0.,.05],'rotation_xyzw':[0.,0.,0.,1.]},
        'grasp_approach_start_offset':-40,'grasp_approach_end_offset':-10,
        'finger_targets':{'j2_index_pip':.5},
        'stages':[{'start_step':250,'end_step':300,'hand_offsets':{'j1_thumb_ip':.2}}],
    })
    np.testing.assert_array_equal(edited.q_ref[:60],base.q_ref[:60])
    np.testing.assert_allclose(edited.q_ref[150,:3],[.1,0.,.15])
    assert edited.q_ref[150,14]==pytest.approx(.5)
    assert edited.q_ref[250,11]==0.
    assert edited.q_ref[300,11]==pytest.approx(.2)


def test_release_restores_hand_withdrawal_after_contact():
    base=reference()
    edited=apply_replay_edits(base,{
        'object_relative_grasp':{'translation':[.1,0.,.05],'rotation_xyzw':[0.,0.,0.,1.]},
        'finger_targets':{'j2_index_pip':.5},'release_at_end':True,
    })
    assert edited.q_ref[200,0]==pytest.approx(.1)
    np.testing.assert_array_equal(edited.q_ref[350:],base.q_ref[350:])


def test_retiming_preserves_geometric_path():
    base=reference();q=base.q_ref.copy();q[:,0]=np.linspace(0,.1,len(q))
    base=replace(base,q_ref=q,q_ref_by_side={'right':q})
    slow=apply_replay_edits(base,{'time_scale':3})
    assert len(slow.q_ref)==1201
    np.testing.assert_allclose(slow.q_ref[::3],base.q_ref)
    np.testing.assert_allclose(slow.object_pos[::3],base.object_pos)
    assert slow.movement_start_step==300 and slow.movement_end_step==900
    np.testing.assert_allclose(np.diff(slow.timestamps),.01)


def test_acquisition_has_open_approach_and_same_postclosure_commands():
    base=reference();edit={'finger_targets':{'j2_index_mcp_flex':1.},
        'object_relative_grasp':{'translation':[.1,0.,.05],'rotation_xyzw':[0.,0.,0.,1.]}}
    normal=apply_replay_edits(base,edit)
    acquire=apply_replay_edits(base,{**edit,'acquisition':{
        'start_step':10,'pregrasp_step':50,'arrive_step':80,'closed_step':120,
        'clearance_object_local':[.06,0.,0.],
        'open_offsets':{'j2_index_mcp_flex':-.6},
    }})
    np.testing.assert_array_equal(acquire.q_ref[:11],normal.q_ref[:11])
    assert acquire.q_ref[50,0]==pytest.approx(.16)
    assert acquire.q_ref[80,13]==pytest.approx(.4)
    np.testing.assert_array_equal(acquire.q_ref[120:],normal.q_ref[120:])


@pytest.mark.parametrize('edit',[
    {'translation':[float('nan'),0,0]},
    {'time_scale':.5},
    {'hand_offsets':{'ARTx':.1}},
    {'stages':[{'start_step':30,'end_step':20}]},
])
def test_invalid_edit_is_explicit(edit):
    with pytest.raises(ValueError):
        apply_replay_edits(reference(),edit)


def test_published_recipes_bind_command_assets():
    assets=Path(__file__).resolve().parents[1]/'sim/manorl/task_assets/replay_repairs'
    names=['guangxue_bowl03_row0_v1.json','guangxue_bottle05_row19_v2.json',
           'guangxue_pitcher06_row31_v1.json','guangxue_bowl07_row36_v1.json',
           'guangxue_bowl09_row49_v3.json']
    rows=[]
    for name in names:
        path=assets/name;recipe=json.loads(path.read_text())
        assert recipe['schema']=='direct_capture_repair.v1'
        assert recipe['source']['version']==5
        assert recipe['solver']=={'cone':'elliptic','impratio':100.}
        rows.append(recipe['source']['row'])
        if 'command_track' in recipe:
            command,mapping=load_command_track(path,recipe['command_track'],10000)
            assert len(command)>0 and len(mapping)==len(command)
    assert rows==[0,19,31,36,49]


def test_command_track_binds_bytes_and_frame_mapping(tmp_path):
    path=tmp_path/'commands.npz'
    np.savez(path,ctrl=np.ones((3,28)),reference_index=np.array([0,1,1]))
    record={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'frames':3}
    commands,indices=load_command_track(tmp_path/'patch.json',record,2)
    np.testing.assert_array_equal(commands,np.ones((3,28)))
    np.testing.assert_array_equal(indices,[0,1,1])
    with pytest.raises(ValueError,match='hash differs'):
        load_command_track(tmp_path/'patch.json',{**record,'sha256':'bad'},2)
    with pytest.raises(ValueError,match='inside the patch directory'):
        load_command_track(tmp_path/'patch.json',{**record,'path':'../outside.npz'},2)
    with pytest.raises(ValueError,match='valid source-reference indices'):
        load_command_track(tmp_path/'patch.json',record,1)


def test_separate_finger_and_wrist_release():
    base=reference()
    new=apply_replay_edits(base,{
        'object_relative_grasp':{'translation':[.1,0.,.05],'rotation_xyzw':[0.,0.,0.,1.]},
        'finger_targets':{'j2_index_pip':.5},'release_at_end':True,
        'release_start_step':280,'release_end_step':320,
        'wrist_release_start_step':330,'wrist_release_end_step':370,
        'stages':[{'start_step':100,'end_step':120,'wrist_translation_world':[0.,0.,.1]}],
    })
    assert new.q_ref[325,14]==0.
    assert new.q_ref[325,0]==pytest.approx(.1)
    assert new.q_ref[325,2]==pytest.approx(.25)
    assert new.q_ref[380,0]==0.


def test_motion_export_is_measured_generated_data(tmp_path):
    run=tmp_path/'run';run.mkdir()
    recipe={'source':{'uuid':'original','row':49},'active_object':'bowl'}
    (run/'manifest.json').write_text(json.dumps({'patch':recipe,'source_gesture':'009-bowl-hold'}))
    (run/'result.json').write_text(json.dumps({'complete':True}))
    positions=np.array([[[0.,0.,.1]],[[0.,0.,.2]]])
    np.savez(run/'trajectory.npz',qpos=np.zeros((2,35)),ctrl=np.ones((2,28)),
        scene_object_names=np.array(['bowl']),scene_object_pos=positions,
        scene_object_rot_aa=np.zeros((2,1,3)),time_seconds=[.01,.02],source_frame=[1,2])
    out=tmp_path/'motion.lance';export_motion(run,out)
    import lance
    ds=lance.dataset(out);row=ds.take([0]).to_pylist()[0]
    assert row['index']['is_generated'] is True
    assert ds.schema.metadata[b'contract']==b'direct_repaired_physical_motion.v1'
    np.testing.assert_array_equal(row['objects'][0]['pos'],positions[:,0])
    np.testing.assert_array_equal(row['hands'][0]['command_target_dof'],np.ones((2,28)))
    with pytest.raises(FileExistsError):
        export_motion(run,out)
