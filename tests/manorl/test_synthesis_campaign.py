from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.manorl.contracts import JOINT_DOF, TrajectoryIdentity
from sim.manorl.trajectory import ReferenceTrajectory
from sim.manorl.synthesis_campaign import PrefixConfig, QuotaLedger, augment_source_prefix, source_key


def trajectory():
    n=240
    q=np.zeros((n,28)); q[:,:3]=[0.1,-0.2,0.4]; q[:,6:]=0.2
    pos=np.tile([0.,0.,0.03],(n,1)); quat=np.tile([0.,0.,0.,1.],(n,1))
    return ReferenceTrajectory(
        identity=TrajectoryIdentity(dataset_path='/fixture/raw.lance',dataset_version=1,row_index=0,
            object_index=0,uuid='01234567-0123-4567-89ab-0123456789ab',file_uuid='raw',identity='cube1_01_001',
            source_start=0,source_stop=n,movement_start_raw=80,movement_end_raw=170),
        dataset_version=1,source_indices=np.arange(n),timestamps=np.arange(n)/120,
        q_ref=q,object_pos_raw=pos,object_pos=pos,object_quat_xyzw=quat,object_z_shift=0.,
        reference_fps=120,control_fps=120,movement_start_step=80,movement_end_step=170)


def test_prefix_stratification_and_unchanged_base():
    src=trajectory(); low=np.full(28,-10.); high=-low
    starts=[]; azimuths=[]
    for slot in range(5):
        aug,sample=augment_source_prefix(src,slot=slot,attempt=1,joint_lower=low,joint_upper=high)
        again,same=augment_source_prefix(src,slot=slot,attempt=1,joint_lower=low,joint_upper=high)
        assert sample==same
        np.testing.assert_array_equal(aug.q_ref,again.q_ref)
        p=aug.augmentation_prefix_frames
        np.testing.assert_array_equal(aug.q_ref[p:],src.q_ref)
        np.testing.assert_array_equal(aug.object_pos[p:],src.object_pos)
        np.testing.assert_array_equal(aug.source_indices[p:],src.source_indices)
        assert aug.augmentation_base_early_phase_steps==80
        assert sample['policy_enable_step']==p+80
        radius=np.linalg.norm(aug.q_ref[0,:3]-src.q_ref[0,:3])
        lo,hi=sample['radius_stratum_m']; assert lo<=radius<=hi
        assert np.linalg.norm(aug.q_ref[0,3:6]-src.q_ref[0,3:6])>0
        assert not np.array_equal(aug.q_ref[0,6:],src.q_ref[0,6:])
        np.testing.assert_allclose(aug.q_ref[p,:3]-aug.q_ref[p-1,:3],src.q_ref[1,:3]-src.q_ref[0,:3],atol=1e-10)
        starts.append(tuple(aug.q_ref[0]));azimuths.append(sample['azimuth_stratum'])
    assert len(set(starts))==5
    assert sorted(azimuths)==list(range(5))


def test_retry_stays_in_slot_and_source_namespaces_do_not_collide():
    src=trajectory();low=np.full(28,-10.);high=-low
    _,a=augment_source_prefix(src,slot=2,attempt=1,joint_lower=low,joint_upper=high)
    _,b=augment_source_prefix(src,slot=2,attempt=2,joint_lower=low,joint_upper=high)
    assert a['radius_stratum_m']==b['radius_stratum_m']
    assert a['seed']!=b['seed'] and a['start_q']!=b['start_q']
    other=replace(src,identity=replace(src.identity,dataset_path='/fixture/other.lance'))
    assert source_key(src)!=source_key(other)


def test_environment_preserves_trained_early_phase():
    from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
    src=trajectory();low=np.full(28,-10.);high=-low
    aug,_=augment_source_prefix(src,slot=0,attempt=1,joint_lower=low,joint_upper=high)
    env=object.__new__(MujocoManoEnvironment)
    env.trajectories=(aug,);env.config=EnvironmentConfig();env.hand_sides=('right',)
    env.model_hand_sides=('right',);env.primary_hand_side='right'
    env._build_reference_tables()
    assert env.early_phase_lengths[0]==aug.augmentation_prefix_frames+80
    assert env.deviation_enable_steps[0]==env.early_phase_lengths[0]
    legacy=replace(aug,augmentation_base_early_phase_steps=0)
    env.trajectories=(legacy,);env._build_reference_tables()
    assert env.early_phase_lengths[0]==aug.augmentation_prefix_frames


def test_ledger_counts_only_saved_acceptance_and_recovers(tmp_path):
    path=tmp_path/'quota.sqlite';a=QuotaLedger(path);b=QuotaLedger(path)
    config={'ratio':5};sources=[{'key':'a','group_key':'cube1','pair':'cube1:01'},{'key':'b','group_key':'cube1','pair':'cube1:01'}]
    a.initialize(config,sources)
    first=a.claim('worker-a','cube1',3);second=b.claim('worker-b','cube1',7)
    assert len(first)==3 and len(second)==7
    assert a.claim('worker-a','cube1',10)==[]
    assert a.summary()['accepted']==0 and not a.summary()['complete']
    a.finish(first[0],{'accepted':False})
    retry=b.claim('worker-b','cube1',1)[0]
    assert retry['attempt']==2 and retry['slot']==first[0]['slot']
    with pytest.raises(RuntimeError,match='stale'):
        a.finish(first[0],{},row_uuid='old',shard='old.lance')
    assert a.release_owner('worker-a')==2
    rest=b.claim('worker-b','cube1',10)
    claims=second+[retry]+rest
    for claim in claims:
        row_uuid=claim['source']['key']+str(claim['slot'])
        b.finish(claim,{'accepted':True},row_uuid=row_uuid,shard='verified.lance')
        b.finish(claim,{'accepted':True},row_uuid=row_uuid,shard='verified.lance')
    assert a.summary()['complete'] and a.summary()['accepted']==10
    assert a.summary()['pairs_met']==1 and a.summary()['target']==10
    assert a.claim('worker-a','cube1',10)==[]
    a.close();b.close()
    c=QuotaLedger(path);c.initialize(config,sources)
    assert c.summary()['complete']
    with pytest.raises(ValueError,match='configuration changed'):c.initialize({'ratio':10},sources)
    c.close()


def test_duplicate_output_uuid_cannot_fill_two_slots(tmp_path):
    import sqlite3
    a=QuotaLedger(tmp_path/'state.sqlite');a.initialize({},[{'key':'a','group_key':'g','pair':'obj:01'}])
    c=a.claim('w','g',2)
    a.finish(c[0],{},row_uuid='same',shard='a.lance')
    with pytest.raises(sqlite3.IntegrityError):a.finish(c[1],{},row_uuid='same',shard='a.lance')
    assert a.summary()['accepted']==1
    a.close()


def test_owner_lock_survives_gpu_process_descriptor_close(tmp_path):
    import subprocess
    import sys
    from sim.manorl.synthesis_campaign import acquire_owner_lock, owner_lock_busy
    handle=acquire_owner_lock(tmp_path,'worker-1')
    child=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.buffer.read(1)'],
        stdin=subprocess.PIPE,pass_fds=(handle.fileno(),))
    try:
        handle.close()
        assert owner_lock_busy(tmp_path,'worker-1')
        child.communicate(b'x',timeout=5)
        assert not owner_lock_busy(tmp_path,'worker-1')
    finally:
        if child.poll() is None:child.kill();child.wait()


def test_short_original_prephase_keeps_training_early80():
    src=replace(trajectory(),movement_start_step=0)
    low=np.full(28,-10.);high=-low
    augmented,sample=augment_source_prefix(src,slot=0,attempt=1,joint_lower=low,joint_upper=high)
    assert augmented.movement_start_step==augmented.augmentation_prefix_frames
    assert sample['policy_enable_step']==augmented.augmentation_prefix_frames+80


def test_unreachable_raw_finger_reference_joins_native_clipped_target():
    source=trajectory();q=np.array(source.q_ref,copy=True);q[:,14]=-0.18
    source=replace(source,q_ref=q,q_ref_by_side={'right':q})
    lower=np.full(28,-10.);upper=np.full(28,10.);lower[14]=0.;upper[14]=1.745
    augmented,sample=augment_source_prefix(source,slot=0,attempt=1,joint_lower=lower,joint_upper=upper)
    prefix=augmented.augmentation_prefix_frames
    assert np.all(augmented.q_ref[:prefix,14]>=0)
    assert augmented.q_ref[prefix-1,14]==pytest.approx(0)
    np.testing.assert_array_equal(augmented.q_ref[prefix:],source.q_ref)
    assert sample['finger_template']=='native_servo_clipped_source_frame0'
    assert 8 in sample['source_finger_limit_excess']


def test_watch_continues_until_quota_or_explicit_pause():
    from tools.watch_manorl_synthesis_campaign import should_continue
    assert should_continue({'complete':False},paused=False)
    assert not should_continue({'complete':True},paused=False)
    assert not should_continue({'complete':False},paused=True)


def test_stale_monitor_checks_current_assignments_not_retired_failures():
    import json
    from tools.watch_manorl_synthesis_campaign import active_worker_heartbeats, stale_worker_owners
    rows=[{'owner':'retired','heartbeat':100.,'details':json.dumps({'phase':'publication_backpressure'})},
          {'owner':'active','heartbeat':2995.,'details':json.dumps({'phase':'rollout','step':400})}]
    status={'workers':{'0':{'owner':'active'}}}
    current=active_worker_heartbeats(status,rows,3000.)
    assert set(current)=={'active'} and current['active']['heartbeat_age_s']==5.
    assert stale_worker_owners(current)==[]
    current=active_worker_heartbeats(status,rows,5000.)
    assert stale_worker_owners(current)==['active']
    missing=active_worker_heartbeats({'workers':{'0':{'owner':'missing'}}},rows,3000.)
    assert stale_worker_owners(missing)==['missing']


@pytest.fixture
def wave_ledger(tmp_path):
    ledger=QuotaLedger(tmp_path/'waves.sqlite')
    ledger.initialize({},[{'key':'a','group_key':'g','pair':'obj:01'},{'key':'b','group_key':'g','pair':'obj:01'}])
    yield ledger
    ledger.close()


def slot_ids(claims):
    return {(c['source']['key'],c['slot']) for c in claims}


def test_claim_wave_settles_previous_failures_before_padding(wave_ledger):
    from tools.run_manorl_synthesis_campaign import claim_wave
    pending=wave_ledger.claim('w','g',8)
    calls=[]
    def settle():
        calls.append(True)
        for claim in pending:wave_ledger.finish(claim,{'accepted':False})
    current=claim_wave(wave_ledger,'w','g',8,settle_pending=settle)
    assert calls==[True]
    assert len(current)==len(slot_ids(current))==8
    assert not ({c['token'] for c in current}&{c['token'] for c in pending})
    assert sorted(c['attempt'] for c in current)==[1,1,2,2,2,2,2,2]
    assert wave_ledger.summary()['states']=={'leased':8,'ready':2}


def test_claim_wave_keeps_publication_overlap_when_full(wave_ledger):
    from tools.run_manorl_synthesis_campaign import claim_wave
    pending=wave_ledger.claim('w','g',2)
    def must_not_settle():
        pytest.fail('a full batch must retain publication overlap')
    current=claim_wave(wave_ledger,'w','g',8,settle_pending=must_not_settle)
    assert len(current)==8
    assert slot_ids(current).isdisjoint(slot_ids(pending))
    assert wave_ledger.summary()['states']=={'leased':10}


def test_claim_wave_waits_when_all_slots_are_pending(wave_ledger):
    from tools.run_manorl_synthesis_campaign import claim_wave
    pending=wave_ledger.claim('w','g',10)
    def settle():
        for claim in pending:wave_ledger.finish(claim,{'accepted':False})
    current=claim_wave(wave_ledger,'w','g',8,settle_pending=settle)
    assert len(current)==len(slot_ids(current))==8
    assert all(c['attempt']==2 for c in current)


def test_claim_wave_never_reclaims_accepted_slots(wave_ledger):
    from tools.run_manorl_synthesis_campaign import claim_wave
    pending=wave_ledger.claim('w','g',8)
    def settle():
        for i,claim in enumerate(pending):
            if i<6:
                wave_ledger.finish(claim,{'accepted':True},row_uuid=f'row-{i}',shard='verified.lance')
            else:wave_ledger.finish(claim,{'accepted':False})
    current=claim_wave(wave_ledger,'w','g',8,settle_pending=settle)
    assert len(current)==len(slot_ids(current))==4
    assert slot_ids(current).isdisjoint(slot_ids(pending[:6]))
    assert wave_ledger.summary()['states']=={'accepted':6,'leased':4}


def test_claim_wave_returns_empty_after_pending_completes_quota(wave_ledger):
    from tools.run_manorl_synthesis_campaign import claim_wave
    pending=wave_ledger.claim('w','g',10)
    def settle():
        for i,claim in enumerate(pending):
            wave_ledger.finish(claim,{'accepted':True},row_uuid=f'row-{i}',shard='verified.lance')
    assert claim_wave(wave_ledger,'w','g',8,settle_pending=settle)==[]
    assert wave_ledger.summary()['complete']


def test_claim_wave_propagates_publication_failure_without_releasing(wave_ledger):
    from tools.run_manorl_synthesis_campaign import claim_wave
    pending=wave_ledger.claim('w','g',8)
    def fail():raise RuntimeError('publication failed')
    with pytest.raises(RuntimeError,match='publication failed'):
        claim_wave(wave_ledger,'w','g',8,settle_pending=fail)
    assert wave_ledger.summary()['accepted']==0
    assert wave_ledger.summary()['states']=={'leased':10}
    # Original tokens remain valid: the helper did not silently free the batch.
    wave_ledger.finish(pending[0],{'accepted':False})


def test_claim_wave_without_pending_respects_remaining_quota(wave_ledger):
    from tools.run_manorl_synthesis_campaign import claim_wave
    current=claim_wave(wave_ledger,'w','g',16)
    assert len(current)==len(slot_ids(current))==10


def test_group_scheduler_prefers_unassigned_groups():
    from tools.run_manorl_synthesis_campaign import select_worker_group
    groups=[{'group_key':'a','ready':9000},{'group_key':'b','ready':1}]
    assert select_worker_group(groups,{'a':1},{},{},256)=='b'


def test_group_scheduler_shares_last_large_group():
    from tools.run_manorl_synthesis_campaign import select_worker_group
    groups=[{'group_key':'cube1','ready':8912}]
    assert select_worker_group(groups,{'cube1':1},{},{},256)=='cube1'


def test_group_scheduler_respects_reserved_and_leased_only_work():
    from tools.run_manorl_synthesis_campaign import select_worker_group
    groups=[{'group_key':'a','ready':300}]
    assert select_worker_group(groups,{'a':1},{},{},256)=='a'
    assert select_worker_group(groups,{'a':2},{},{'a':256},256) is None
    assert select_worker_group([{'group_key':'a','ready':0}],{},{},{},256) is None
    # A genuinely small final pool can run once, but is not shared prematurely.
    assert select_worker_group([{'group_key':'a','ready':4}],{},{},{},256)=='a'
    assert select_worker_group([{'group_key':'a','ready':4}],{'a':1},{},{},256) is None


def test_group_scheduler_balances_sharing_and_never_uses_blocked_groups():
    from tools.run_manorl_synthesis_campaign import select_worker_group
    groups=[{'group_key':'a','ready':2048},{'group_key':'b','ready':2048}]
    assert select_worker_group(groups,{'a':2,'b':1},{},{},256)=='b'
    assert select_worker_group(groups,{'a':1,'b':1},{},{},256)=='a'
    assert select_worker_group(groups,{'a':1,'b':1},{'a':{},'b':{}},{},256) is None


def test_same_group_workers_claim_disjoint_nonexcluded_slots(tmp_path):
    a=QuotaLedger(tmp_path/'shared.sqlite');b=QuotaLedger(tmp_path/'shared.sqlite')
    try:
        a.initialize({},[{'key':'excluded','group_key':'g','pair':'obj:01'}])
        a.configure_success_filter(minimum_attempts=1)
        rejected=a.claim('w','g',1)[0];a.finish(rejected,{'accepted':False})
        a.initialize({},[{'key':'b','group_key':'g','pair':'obj:01'},{'key':'c','group_key':'g','pair':'obj:01'}])
        assert a.groups()[0]['ready']==15  # ten ready plus five reassignable obligations
        one=a.claim('gpu1','g',6);two=b.claim('gpu2','g',6)
        assert len(one)==6 and len(two)==6  # the short claim reassigns retired debt
        assert slot_ids(one).isdisjoint(slot_ids(two))
        assert all(c['source']['key']!='excluded' for c in one+two)
        assert a.groups()[0]['ready']==3  # only the reassignable debt remains
        for claim in one+two:
            key=claim['source']['key']
            a.finish(claim,{'accepted':True},row_uuid=f"{key}-{claim['slot']}",shard='verified.lance')
        assert a.summary()['accepted']==12 and a.summary()['skipped_slots']==5
        assert not a.summary()['complete']  # 3 retired obligations remain owed
        assert a.summary()['remaining']==3
        rest=a.claim('gpu1','g',3)
        assert len(rest)==3 and all(c['slot']>=5 for c in rest)  # extra donor variants
        for claim in rest:
            a.finish(claim,{'accepted':True},row_uuid=f"extra-{claim['source']['key']}-{claim['slot']}",shard='verified.lance')
        assert a.summary()['accepted']==15 and a.summary()['complete']
    finally:a.close();b.close()
