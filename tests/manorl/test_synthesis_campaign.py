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
    config={'ratio':5};sources=[{'key':'a','group_key':'cube1'},{'key':'b','group_key':'cube1'}]
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
    assert a.summary()['sources_complete']==2
    assert a.claim('worker-a','cube1',10)==[]
    a.close();b.close()
    c=QuotaLedger(path);c.initialize(config,sources)
    assert c.summary()['complete']
    with pytest.raises(ValueError,match='configuration changed'):c.initialize({'ratio':10},sources)
    c.close()


def test_duplicate_output_uuid_cannot_fill_two_slots(tmp_path):
    import sqlite3
    a=QuotaLedger(tmp_path/'state.sqlite');a.initialize({},[{'key':'a','group_key':'g'}])
    c=a.claim('w','g',2)
    a.finish(c[0],{},row_uuid='same',shard='a.lance')
    with pytest.raises(sqlite3.IntegrityError):a.finish(c[1],{},row_uuid='same',shard='a.lance')
    assert a.summary()['accepted']==1
    a.close()


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
