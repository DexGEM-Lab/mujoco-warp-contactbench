"""Opt-in canonical generated references retain full simulated coordinates."""
from copy import deepcopy
import json
import numpy as np
import pytest

from sim.manorl.trajectory import (
    TrajectorySelection, trajectory_from_lance_row, _candidate_from_metadata_row,
)
from sim.manorl.lance_v2 import SYNTHETIC_LANCE_CONTRACT


def generated_row():
    q = np.zeros((6,28)); q[:,0] = np.arange(6)*.02; q[:,2] = .25
    q[:,3] = [3., 3.1, -3.1, -3., -2.9, -2.8]
    return dict(index=dict(uuid='generated-1',seed_uuid='raw-1',scene='bowl',operator='cheyingtong',is_generated=True),
        trajectory_metadata=dict(total_frames=6,data_fps=120,gesture='009-hold',hand_names=['right'],
            hand_slots=['right','left'],mano_hand_shapes=[[0.]*10],object_names=['bowl','cuboid1'],
            raw_data_info={'id':35},trajectory_info={'object_move':[dict(object_name='bowl',start_frame=2,end_frame=4)]}),
        timestamp=(np.arange(6)/120).tolist(),hands=[dict(hand_name='right',urdf_dof=q.tolist()),
                                                   dict(hand_name=None,urdf_dof=[])],
        objects=[dict(pos=np.tile([.1,.2,.4],(6,1)).tolist(),rot_aa=np.zeros((6,3)).tolist()),
                 dict(pos=np.tile([.0,.2,.1],(6,1)).tolist(),rot_aa=np.zeros((6,3)).tolist())],
        provenance=dict(contract=SYNTHETIC_LANCE_CONTRACT,reference_fps=120,control_fps=120,
                        physics_fps=480,physics_substeps_per_control=4,
                        checkpoint_metadata_json=json.dumps({'frame_zero_integrated':False,'physical_hand_profile':{'manifest_sha256':'test-profile'}})))


def test_full_episode_keeps_first_hand_and_world_coordinates(monkeypatch):
    from sim.manorl import assets
    monkeypatch.setattr(assets,'EXPLICIT_ASSET_MANIFEST','test-manifest')
    monkeypatch.setattr(assets,'MANO_OPERATOR','cheyingtong')
    monkeypatch.setattr(assets,'_asset_manifest',lambda: {'hands':{'right':{'betas':[0.]*10}}})
    monkeypatch.setattr(assets,'asset_provenance',lambda: {'asset_manifest_sha256':'test-profile'})
    row=generated_row(); original=deepcopy(row)
    t=trajectory_from_lance_row(row,dataset_version=1,generated_reference=True,hand_side='right')
    np.testing.assert_array_equal(t.q_ref,row['hands'][0]['urdf_dof'])
    np.testing.assert_array_equal(t.object_pos,row['objects'][0]['pos'])
    assert t.object_z_shift==0 and len(t.q_ref)==6
    assert t.reference_fps==t.control_fps==120
    assert (t.movement_start_step,t.movement_end_step)==(2,4)
    assert t.scene_object_types==('bowl','cuboid1')
    assert t.identity.identity=='bowl_09_035'
    assert row==original


def test_default_discovery_excludes_generated_and_opt_in_reads_metadata_action():
    row=generated_row()
    default=TrajectorySelection(selector='all',pre_padding=0,post_padding=0)
    assert _candidate_from_metadata_row(row,row_index=0,selection=default) is None
    selected=TrajectorySelection(selector='all',pre_padding=0,post_padding=0,generated_reference=True)
    candidate=_candidate_from_metadata_row(row,row_index=0,selection=selected)
    assert candidate.pair.canonical=='bowl:09'
    assert candidate.identity=='bowl_09_035'
    row['index']['is_generated']=False
    assert _candidate_from_metadata_row(row,row_index=0,selection=selected) is None


@pytest.mark.parametrize('field,value', [('physics_fps',400),('control_fps',100),('contract','legacy')])
def test_generated_contract_mismatches_fail(field,value):
    row=generated_row();row['provenance'][field]=value
    with pytest.raises(ValueError,match='contract'):
        trajectory_from_lance_row(row,dataset_version=1,generated_reference=True)


def test_padding_and_integrated_frame_zero_rejected():
    with pytest.raises(ValueError,match='zero padding'):
        TrajectorySelection(generated_reference=True,pre_padding=1,post_padding=0)
    row=generated_row();row['provenance']['checkpoint_metadata_json']='{"frame_zero_integrated":true}'
    with pytest.raises(ValueError,match='prestep'):
        trajectory_from_lance_row(row,dataset_version=1,generated_reference=True)


def test_nonempty_inactive_fixed_slot_rejected():
    row=generated_row();row['hands'][1]['urdf_dof']=[[0]*28]
    with pytest.raises(ValueError,match='inactive'):
        trajectory_from_lance_row(row,dataset_version=1,generated_reference=True)


def test_generated_reference_requires_explicit_physical_profile(monkeypatch):
    from sim.manorl import assets
    monkeypatch.setattr(assets,'EXPLICIT_ASSET_MANIFEST','')
    with pytest.raises(ValueError,match='explicit physical hand'):
        trajectory_from_lance_row(generated_row(),dataset_version=1,generated_reference=True)


def test_compiler_selection_payload_roundtrip(tmp_path):
    from tools.compile_manorl_trajectory_package import _selection_payload
    from tools.compile_manorl_trajectory_worker import _selection
    selection=TrajectorySelection(selector='all',generated_reference=True,pre_padding=0,post_padding=0,
                                  reference_fps=120,expected_dataset_version=1)
    p=tmp_path/'selection.json';p.write_text(json.dumps(_selection_payload(selection)))
    assert _selection(p).generated_reference


def test_trainer_flag_reaches_budget_without_running_training(tmp_path,monkeypatch):
    import importlib.util,sys
    path=__import__('pathlib').Path(__file__).parents[2]/'tools/train_manorl_cube1.py'
    spec=importlib.util.spec_from_file_location('generated_reference_train_cli_test',path)
    tool=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tool;spec.loader.exec_module(tool)
    captured=[]
    monkeypatch.setattr(tool,'run',lambda output,budget:captured.append(budget) or {})
    assert tool.main(['--output',str(tmp_path/'out'),'--generated-reference','--reference-fps','120',
                      '--pre-padding','0','--post-padding','0'])==0
    assert captured[0].generated_reference and captured[0].pre_padding==captured[0].post_padding==0
    with pytest.raises(SystemExit):
        tool.main(['--output',str(tmp_path/'invalid'),'--generated-reference'])
