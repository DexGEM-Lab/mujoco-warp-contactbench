import json
from pathlib import Path

import pytest

from sim.manorl.synthesis_admission import (
    admitted_gpu_pool, append_prepared, claim_object, completed_job,
    eligible_gpus, execution_revision, register_existing, sha256,
)
from sim.manorl.synthesis_campaign import QuotaLedger, atomic_json


def test_registry_never_duplicates_an_object_between_hosts(tmp_path):
    kwargs=dict(obj='cuboid3',checkpoint_sha='a'*64,output='nas/host-one')
    assert claim_object(tmp_path,host='server1',campaign_id='s1',**kwargs)
    assert claim_object(tmp_path,host='server1',campaign_id='s1',**kwargs)
    assert not claim_object(tmp_path,host='server2',campaign_id='s2',**kwargs)
    with pytest.raises(ValueError,match='another checkpoint'):
        claim_object(tmp_path,obj='cuboid3',host='server1',campaign_id='s1',checkpoint_sha='b'*64,output='nas/host-one')


def test_completed_report_requires_finished_matching_checkpoint(tmp_path):
    report=tmp_path/'cuboid3.json';cp=tmp_path/'cuboid3.pt';meta=tmp_path/'cuboid3.pt.json'
    report.write_text(json.dumps({'schema':'manorl.cube1_fast_training.v1','actual':{'updates':5000},
        'budget':{'raw_transfer':True,'trajectory_packages':['package.mtp']}}))
    assert completed_job(report) is None
    cp.write_bytes(b'checkpoint')
    value={'checkpoint_file':'cuboid3.pt','runtime_config':{
        'training_progress':{'completed_updates':4999},'environment':{'pre_padding':80,'compatibility':{'early_phase_steps':80},
          'reference_fps':120,'physics_fps':480,'controlled_hand_sides':['right']},
        'trajectory_selection':{'resolved_pairs':[{'object':'cuboid3','action':'01'}]}}}
    meta.write_text(json.dumps(value));assert completed_job(report) is None
    value['runtime_config']['training_progress']['completed_updates']=5000
    meta.write_text(json.dumps(value));assert completed_job(report)['pairs']==['cuboid3:01']
    value['checkpoint_file']='other.pt';meta.write_text(json.dumps(value));assert completed_job(report) is None


def test_append_preserves_accepted_slots_and_is_idempotent(tmp_path):
    root=tmp_path/'campaign';root.mkdir();ledger=QuotaLedger(root/'quota.sqlite')
    config={'contract':'test','prefix':{'quota':5}}
    old={'key':'old','group_key':'cube1-x','object':'cube1','checkpoint_sha256':'a'*64}
    ledger.initialize(config,[old]);claim=ledger.claim('live-worker','cube1-x',1)[0]
    ledger.finish(claim,{'accepted':True},row_uuid='old-row',shard='old.lance')
    live=ledger.claim('live-worker','cube1-x',1)[0]
    prepared=tmp_path/'prepared';prepared.mkdir();blob=prepared/'source.pkl';blob.write_bytes(b'example')
    new={'key':'new','group_key':'cuboid3-y','object':'cuboid3','pickle':str(blob),'pickle_sha256':sha256(blob),'checkpoint_sha256':'b'*64}
    atomic_json(prepared/'config.json',config);atomic_json(prepared/'sources.json',[new]);atomic_json(prepared/'checkpoints.json',{})
    assert append_prepared(ledger,root,prepared,config)==1
    assert append_prepared(ledger,root,prepared,config)==0
    assert ledger.summary()['target']==10 and ledger.summary()['accepted']==1
    state=ledger.db.execute('SELECT state,token FROM slots WHERE source_key=? AND slot=?',('old',live['slot'])).fetchone()
    assert state['state']=='leased' and state['token']==live['token']
    register_existing(ledger,tmp_path/'registry',host='server1',campaign_id='s1',output='nas')
    with pytest.raises(RuntimeError,match='owned by another host'):
        register_existing(ledger,tmp_path/'registry',host='server2',campaign_id='s2',output='nas')
    ledger.close()


def test_gpu_admission_waits_for_completed_objects_and_bounds_pool(tmp_path):
    policy={'always':[2,3],'after_objects':{'0':['largeclamp'],'1':['cylinder5']}}
    assert eligible_gpus(policy,set())==[2,3]
    assert eligible_gpus(policy,{'largeclamp'})==[0,2,3]
    assert eligible_gpus(policy,{'largeclamp','cylinder5'})==[0,1,2,3]
    atomic_json(tmp_path/'gpu-pool.json',{'gpus':[2,3]})
    assert admitted_gpu_pool(tmp_path,[0,1,2,3])==[2,3]
    atomic_json(tmp_path/'gpu-pool.json',{'gpus':[4]})
    with pytest.raises(ValueError,match='exceeds'):admitted_gpu_pool(tmp_path,[0,1,2,3])


def test_actual_worker_source_revision_is_explicit(monkeypatch):
    monkeypatch.setenv('MANORL_SOFTWARE_COMMIT','f'*40)
    assert execution_revision()=='f'*40
    monkeypatch.setenv('MANORL_SOFTWARE_COMMIT','unknown')
    with pytest.raises(RuntimeError,match='immutable'):execution_revision()
