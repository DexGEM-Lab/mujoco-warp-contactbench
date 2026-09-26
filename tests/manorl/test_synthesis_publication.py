from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from sim.manorl.synthesis_campaign import QuotaLedger
from sim.manorl.synthesis_publication import publish_bundle, reconcile_shards
from sim.manorl.lance_v2 import SYNTHETIC_LANCE_CONTRACT, FORCE_DIRECTION_CONTRACT, build_compact_row


def compact_fixture(source_key,slot):
    n=12;q=[[0.]*28 for _ in range(n)];xyz=[[0.,0.,0.] for _ in range(n)]
    hand={'hand_name':'right','urdf_dof':q,'urdf_dof_target':q,'mano_global_pos':xyz,
          'mano_global_rot_aa':xyz,'mano_hand_pose':[[0.]*48 for _ in range(n)],
          'mano_joint_pos':[[[0.]*3 for _ in range(21)] for _ in range(n)]}
    full={'index':{'uuid':source_key+str(slot),'seed_uuid':source_key,'capMachine':'fixture',
                  'operator':'fixture','scene':'cube1','is_generated':True},
          'trajectory_metadata':{'data_fps':120,'total_frames':n,'gesture':'01','hand_names':['right'],
             'hand_slots':['right','left'],'object_names':['cube1'],'mano_hand_shapes':[[0.]*10],
             'trajectory_info':{'object_move':[{'object_name':'cube1','start_frame':2,'end_frame':8}]}},
          'timestamp':[i/120 for i in range(n)],'hands':[hand], 'objects':[{'pos':xyz,'rot_aa':xyz}],
          'contact':[[] for _ in range(n)],'reference':{'source_frame_index':list(range(n)),
             'hand_urdf_dof':q,'object_pos':xyz,'object_rot_aa':xyz},
          'rollout':{'command_reference_index':list(range(n-1)),'command_source_frame_index':list(range(n-1))},
          'provenance':{'contract':SYNTHETIC_LANCE_CONTRACT,'force_contract':FORCE_DIRECTION_CONTRACT,
             'reference_fps':120,'control_fps':120,'control_timestep_seconds':1/120,
             'physics_fps':480,'physics_timestep_seconds':1/480,'physics_substeps_per_control':4,
             'policy_mode':'deterministic_mean','checkpoint_path':'fixture.pt','checkpoint_sha256':'1'*64,
             'checkpoint_update':5000,'dataset_path':'fixture.lance','dataset_version':1,'row_index':0,
             'source_identity':'cube1_01_001','software_commit':'fixture','seed':42,'episode_index':slot,
             'generation_attempt':1,'augmentation_identity':'fixture'}}
    return build_compact_row(full,checkpoint_metadata={},warp_ccd_iterations=16,warp_ccd_contacts_per_world=16)


def test_atomic_publication_recovers_promoted_rows_before_releasing_claims(tmp_path,monkeypatch):
    import sim.manorl.synthesis_publication as publication
    pytest.importorskip('lance')
    ledger=QuotaLedger(tmp_path/'state.sqlite');ledger.initialize({},[{'key':'a','group_key':'cube1'}])
    claims=ledger.claim('dead-worker','cube1',2)
    def row_for(result,**kwargs):
        return compact_fixture(result['claim']['source']['key'],result['claim']['slot'])
    monkeypatch.setattr(publication,'_cpu_processor',lambda _:SimpleNamespace(object_name='cube1',compact_with_retreat=row_for))
    monkeypatch.setattr(publication.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    bundle={'bundle_id':'test-wave','output':str(tmp_path/'output'),
        'context':{'metadata':{},'physical_manifest':{'hand_operator':'fixture'},'software_commit':'fixture'},
        'results':[{'claim':c,'diagnostic':{'accepted':True}} for c in claims]}
    path=tmp_path/'wave.pkl'
    with path.open('wb') as f:pickle.dump(bundle,f)
    result=publish_bundle(path)
    assert result['accepted']==2 and Path(result['shard']).is_dir()
    assert ledger.summary()['accepted']==0  # publishing alone does not silently update quotas
    assert reconcile_shards(ledger,tmp_path/'output')==2
    assert reconcile_shards(ledger,tmp_path/'output')==0
    assert ledger.summary()['accepted']==2
    assert ledger.release_owner('dead-worker')==0
    assert len(ledger.claim('new-worker','cube1',5))==3
    ledger.close()


def test_recovery_rejects_changed_payload(tmp_path,monkeypatch):
    import json
    from sim.manorl.synthesis_campaign import CAMPAIGN_CONTRACT
    ledger=QuotaLedger(tmp_path/'state.sqlite');ledger.initialize({},[{'key':'a','group_key':'g'}])
    claim=ledger.claim('w','g',1)[0]
    shard=tmp_path/'output/shards/cube1/wave.lance';shard.mkdir(parents=True)
    (shard/'data').write_text('corrupted')
    (shard/'campaign-commit.json').write_text(json.dumps({'contract':CAMPAIGN_CONTRACT,
        'rows':[{'claim':claim,'diagnostic':{'accepted':True},'row_uuid':'a0'}],
        'payload_sha256':{'data':'0'*64}}))
    with pytest.raises(RuntimeError,match='hash mismatch'):reconcile_shards(ledger,tmp_path/'output')
    assert ledger.summary()['accepted']==0
    ledger.close()


def test_retreat_checks_added_overlap_not_existing_compliant_floor_contact():
    from sim.manorl.synthesis_runtime import added_penetration
    baseline={(0,4):-0.0015761033320828592}
    assert added_penetration(dict(baseline),baseline) is None
    assert added_penetration({(0,4):-0.0015},baseline) is None
    assert added_penetration({(0,16):-0.0000338614},{(0,16):-0.0000355081}) is None
    assert added_penetration({(0,4):-0.0017},baseline)['baseline_depth_m']==baseline[(0,4)]
    assert added_penetration({(0,16):-0.0001},baseline)['geom_pair']==[0,16]
