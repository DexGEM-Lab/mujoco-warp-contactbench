"""Per-world reference conditions, lengths, and reset ownership."""
from dataclasses import replace
from types import SimpleNamespace
from typing import NamedTuple
import jax
import jax.numpy as j
import numpy as np
import pytest
from sim.manorl.autonomy_v4 import ReferenceBankV4, REFERENCE_TIME_FIELDS, _gather, build_raw_observation, compute_reward, raw_observation_slices
from sim.manorl.autonomy_batch import BatchedAutonomyRuntime
from sim.manorl.autonomy_contracts import V4_REWARD_TERM_NAMES
from sim.manorl.autonomy_training import _telemetry_slices
from tests.manorl.test_autonomy_v4 import _cache, _state, _contact


def caches():
    a = _cache(25)
    b = replace(_cache(13), q_feasible=np.full((13,28),.1), object_origin=np.tile([.01,.02,.03],(13,1)), palm_origin=np.tile([.03,.02,.01],(13,1)))
    return a,b


def test_bank_one_exact_observation_reward_parity():
    cache=caches()[0]; bank=ReferenceBankV4([cache]); state=_state(3); contact=_contact(3)
    index=j.array([0,12,24]); action=j.full((3,28),.2); refs=j.zeros(3,j.int32)
    np.testing.assert_array_equal(build_raw_observation(state,contact,bank,index,action,refs),build_raw_observation(state,contact,cache,index,action))
    for actual,expected in zip(compute_reward(state,contact,bank,index,action,refs),compute_reward(state,contact,cache,index,action)):
        np.testing.assert_array_equal(actual,expected)


def test_each_reference_observation_and_own_terminal_length():
    a,b=caches(); bank=ReferenceBankV4([a,b]); refs=j.array([0,1,0]); index=j.array([12,12,24]); action=j.zeros((3,28))
    for field in REFERENCE_TIME_FIELDS:
        value=getattr(bank,field)
        if value is None: continue
        assert value.shape[:2]==(2,25)
    assert bank.q_lower.shape==(2,28)
    raw=build_raw_observation(_state(3),_contact(3),bank,index,action,refs)
    for row,cache in enumerate((a,b,a)):
        np.testing.assert_array_equal(raw[row:row+1],build_raw_observation(_state(),_contact(),cache,index[row:row+1],action[row:row+1]))
    reward=compute_reward(_state(3),_contact(3),bank,index,action,refs)
    np.testing.assert_array_equal(reward.reason,[0,1,1])


def test_device_gathers_no_host_copy():
    bank=ReferenceBankV4(caches()); refs=j.array([0,1,0]); index=j.array([2,4,6]); action=j.zeros((3,28)); state=_state(3); contact=_contact(3)
    fn=jax.jit(lambda i: (build_raw_observation(state,contact,bank,i,action,refs),compute_reward(state,contact,bank,i,action,refs)))
    fn(index)[0].block_until_ready()
    with jax.transfer_guard('disallow'):
        fn(index)[0].block_until_ready()


def test_three_env_two_reference_reset_isolation():
    from sim.manorl.environment import _build_masked_reset_data_fn
    class Data(NamedTuple):
        qpos:object; qvel:object; ctrl:object; global_contact:object
        def replace(self,**kw): return self._replace(**kw)
    bank=ReferenceBankV4(caches()); refs=j.array([0,1,0]); initial=bank.q_feasible[refs,0]
    r=object.__new__(BatchedAutonomyRuntime); r.jp=j; r.num_envs=3; r.env_ref=refs; r.cache=bank
    r._reset_ctrl=initial; r.indices=j.array([7,8,9]); r.previous_command=j.ones((3,28)); r.observation=j.zeros((3,957))
    r.data=Data(j.ones((3,28)),j.ones((3,28)),j.ones((3,28)),j.arange(5))
    r._reset_and_forward=_build_masked_reset_data_fn(jax=jax,jp=j,reset_qpos=initial,reset_ctrl=initial)
    r._refresh=lambda *a: None
    before=r.data; r.reset(j.array([False,True,False]))
    np.testing.assert_array_equal(r.data.qpos[1],bank.q_feasible[1,0]); np.testing.assert_array_equal(r.previous_command[1],bank.q_feasible[1,0])
    for row in (0,2):
        np.testing.assert_array_equal(r.data.qpos[row],before.qpos[row]); np.testing.assert_array_equal(r.data.qvel[row],before.qvel[row]); np.testing.assert_array_equal(r.data.ctrl[row],before.ctrl[row])
    np.testing.assert_array_equal(r.indices,[7,0,9]); np.testing.assert_array_equal(r.env_ref,refs); np.testing.assert_array_equal(r.data.global_contact,before.global_contact)


def test_bank_allows_action_id_variation_and_gathers_action_one_hot():
    a=_cache(25); b=replace(_cache(13), action_id=3)
    bank=ReferenceBankV4([a,b]); refs=j.array([0,1]); index=j.array([0,0]); action=j.zeros((2,28))
    raw=build_raw_observation(_state(2),_contact(2),bank,index,action,refs)
    sl=raw_observation_slices()["action_types"]
    assert int(np.argmax(np.asarray(raw[0,sl]))) == 1
    assert int(np.argmax(np.asarray(raw[1,sl]))) == 2


def test_bank_rejects_geometry_drift_and_empty():
    with pytest.raises(ValueError,match='empty'): ReferenceBankV4([])
    with pytest.raises(ValueError,match='object_radius'): ReferenceBankV4([_cache(),replace(_cache(),object_radius=_cache().object_radius*1.01)])

def test_multireference_motion_gate_uses_each_env_reference_speed():
    a=_cache(25)
    b=replace(_cache(25),object_v_com=np.tile([.10,0.,0.],(25,1)))
    bank=ReferenceBankV4([a,b]); refs=j.array([0,1]); state=_state(2); contact=_contact(2)
    reward=compute_reward(state,contact,bank,j.array([0,0]),j.zeros((2,28)),refs)
    # Same actual state and targets; only each selected reference's speed differs.
    np.testing.assert_allclose(np.asarray(reward.object_position),[.012,1.2],atol=1e-6)
    with pytest.raises(ValueError,match='object_radius'):
        ReferenceBankV4([a,replace(a,object_radius=a.object_radius*1.01)])


def test_cli_selects_only_all_40_train_references():
    from tools.train_manorl_autonomy import parse_args,_training_references
    rows=[SimpleNamespace(identity=SimpleNamespace(identity=f'cube2_02_{i}')) for i in range(50)]
    catalog=SimpleNamespace(trajectories=rows); split={'train_indices':list(range(39,-1,-1))}
    args=parse_args(['train','--all-train-references'])
    assert _training_references(args,catalog,rows[0],split)==rows[39::-1]
    args.all_train_references=False
    assert _training_references(args,catalog,rows[0],split) is rows[0]


def test_cli_selects_all_package_references_without_split_filter():
    from tools.train_manorl_autonomy import parse_args,_training_references
    rows=[SimpleNamespace(identity=SimpleNamespace(identity=f'cube2_02_{i}')) for i in range(50)]
    catalog=SimpleNamespace(trajectories=rows); split={'train_indices':list(range(40))}
    args=parse_args(['train','--all-references'])
    assert args.all_references and not args.all_train_references
    assert _training_references(args,catalog,rows[0],split)==rows


def test_all_reference_requires_shared_cube2_action():
    from tools.train_manorl_autonomy import parse_args,_training_references
    rows=[SimpleNamespace(identity=SimpleNamespace(identity='cube2_02_0')), SimpleNamespace(identity=SimpleNamespace(identity='banana_03_1'))]
    args=parse_args(['train','--all-references'])
    with pytest.raises(ValueError,match='one shared object type'):
        _training_references(args,SimpleNamespace(trajectories=rows),rows[0],{'train_indices':[0]})

def test_identity_split_fallback_for_new_package_is_deterministic():
    from sim.manorl.autonomy_training import identity_split
    rows=[SimpleNamespace(identity=SimpleNamespace(identity=f'cube1_01_{i}')) for i in range(10)]
    catalog=SimpleNamespace(trajectories=rows)
    a=identity_split(catalog,seed=0); b=identity_split(catalog,seed=0)
    assert a==b
    assert set(a['train_indices'])==set(range(10)) and a['validation_indices']==[] and a['test_indices']==[]

    import torch
    from sim.manorl.autonomy_training import BatchedAutonomyAdapter
    bank=ReferenceBankV4(caches()); refs=j.array([0,1,0]); index=j.array([6,6,12])
    physical=_state(3); contact=_contact(3); action=j.zeros((3,28))
    adapter=object.__new__(BatchedAutonomyAdapter); adapter.num_envs=3
    adapter.reward_names=V4_REWARD_TERM_NAMES
    adapter._telemetry_slices,adapter._telemetry_width=_telemetry_slices(adapter.reward_names)
    adapter.runtime=SimpleNamespace(jp=j,cache=bank,env_ref=refs,indices=index,
        last_physical=physical,last_contact=contact,last_reward=compute_reward(physical,contact,bank,index,action,refs))
    adapter._packed_telemetry_fn=adapter._pack_device_telemetry
    # Keep JAX values on device exactly as the GPU bridge does; no NumPy path.
    adapter._to_torch=lambda value:value
    snapshot=adapter.telemetry_snapshot(torch.zeros((3,28)))
    np.testing.assert_array_equal(snapshot['reference_progress'],[.25,.5,.5])
    np.testing.assert_allclose(snapshot['origin_lift_delta'],[0.,-.03,0.],atol=1e-7)
    np.testing.assert_allclose(snapshot['position_error_abs'],[[0.,0.,0.],[.01,.02,.03],[0.,0.,0.]],atol=1e-7)
