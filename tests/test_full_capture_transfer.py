"""CPU contract checks for source-conditioned contact template transfer."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from bowl_hold_template_transfer import _make_patch,_phase_stage
from prepare_normal_timing import plateaus


def trajectory(rotation,end=384):
    return SimpleNamespace(scene_object_initial_quat_xyzw=np.array([rotation.as_quat(),[0,0,0,1]]),
        identity=SimpleNamespace(object_index=0),movement_start_step=180,movement_end_step=end)


def test_local_grasp_transfer_preserves_initial_world_contact_side():
    rd=Rotation.from_euler('XYZ',[.1,.02,-2.]);rr=Rotation.from_euler('XYZ',[-.05,.04,1.])
    local=Rotation.from_euler('XYZ',[.4,.2,.8]);pos=np.array([.08,-.04,.1])
    donor={'source':{'row':49},'edit':{'object_relative_grasp':{'translation':pos.tolist(),'rotation_xyzw':local.as_quat().tolist()},'translation':[.006,.012,0],'finger_targets':{},'hand_offsets':{},'stages':[{'wrist_rotation_world':[.1,0,0],'pivot_from_object':[.07,0,0]}]}}
    receiver={'index':{'uuid':'new-source'},'trajectory_metadata':{'total_frames':500}}
    patch=_make_patch(donor,trajectory(rd),receiver,trajectory(rr,450))
    grasp=patch['edit']['object_relative_grasp']
    np.testing.assert_allclose(rr.apply(grasp['translation']),rd.apply(pos),atol=1e-12)
    relative=(rr*Rotation.from_quat(grasp['rotation_xyzw'])).inv()*(rd*local)
    assert relative.magnitude()<1e-12
    assert patch['source']['uuid']=='new-source'
    assert patch['transfer']['timing']['prefix_preserved_through_step']==100
    assert patch['edit']['grasp_approach_end_offset']==-20


def test_leveling_uses_receiver_lift_phase_and_fixed_hold_tail():
    identity=Rotation.identity()
    start,end=_phase_stage(trajectory(identity),trajectory(identity,450))
    assert start==round(180+(310-180)/(384-180)*(450-180))
    assert end==450+(500-384)


def test_wait_removal_preserves_moving_commands_and_initial_prefix():
    commands=np.zeros((1000,28))
    commands[:200,0]=np.arange(200)*.001
    commands[200:800,0]=commands[199,0]
    commands[800:,0]=commands[199,0]+np.arange(1,201)*.001
    kept,removed=plateaus(commands)
    np.testing.assert_array_equal(kept[:180],np.arange(180))
    for index in np.flatnonzero(np.max(abs(np.diff(commands,axis=0)),axis=1)>1e-8):
        assert index in kept and index+1 in kept
    assert len(removed)==1 and removed[0]['removed_frames']>500
    np.testing.assert_array_equal(commands[kept][0],commands[0])
    np.testing.assert_array_equal(commands[kept][-1],commands[-1])
    assert np.max(np.abs(np.diff(commands[kept],axis=0)))<=.0010000001


def test_short_settling_is_not_removed():
    commands=np.zeros((240,28));commands[:200,0]=np.arange(200)*.001
    commands[200:,0]=commands[199,0]
    kept,removed=plateaus(commands)
    assert len(kept)==len(commands) and removed==[]


def test_recorded_command_assets_are_finite_and_source_bound():
    root=Path(__file__).resolve().parents[1]/'patches'
    checked=0
    for path in root.rglob('*.json'):
        recipe=json.loads(path.read_text())
        if not isinstance(recipe,dict) or recipe.get('schema')!='direct_capture_repair.v1':continue
        assert recipe['source']['version']==5
        assert 0<=recipe['source']['row']<59 and recipe['source']['uuid']
        assert recipe['solver']=={'cone':'elliptic','impratio':100.}
        if 'command_track' in recipe:
            record=recipe['command_track'];asset=path.parent/record['path']
            assert hashlib.sha256(asset.read_bytes()).hexdigest()==record['sha256']
            with np.load(asset,allow_pickle=False) as track:
                assert track['ctrl'].shape==(record['frames'],28)
                assert np.isfinite(track['ctrl']).all()
                assert track['reference_index'].shape==(record['frames'],)
                assert np.issubdtype(track['reference_index'].dtype,np.integer)
                assert (np.diff(track['reference_index'])>=0).all()
        checked+=1
    assert checked>=5
