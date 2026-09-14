"""Per-world reference conditions, lengths, and reset ownership."""
from dataclasses import replace
from types import SimpleNamespace
from typing import NamedTuple
import jax
import jax.numpy as j
import numpy as np
import pytest
from sim.manorl.autonomy_v4 import ReferenceBankV4, REFERENCE_TIME_FIELDS, _gather, build_raw_observation, compute_reward
from sim.manorl.autonomy_batch import BatchedAutonomyRuntime
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
        assert getattr(bank,field).shape[:2]==(2,25)
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


def test_bank_rejects_geometry_or_action_drift_and_empty():
    with pytest.raises(ValueError,match='empty'): ReferenceBankV4([])
    with pytest.raises(ValueError,match='action_id'): ReferenceBankV4([_cache(),replace(_cache(),action_id=3)])


def test_cli_selects_only_all_40_train_references():
    from tools.train_manorl_autonomy import parse_args,_training_references
    rows=[SimpleNamespace(identity=SimpleNamespace(identity=f'cube2_02_{i}')) for i in range(50)]
    catalog=SimpleNamespace(trajectories=rows); split={'train_indices':list(range(39,-1,-1))}
    args=parse_args(['train','--all-train-references'])
    assert _training_references(args,catalog,rows[0],split)==rows[39::-1]
    args.all_train_references=False
    assert _training_references(args,catalog,rows[0],split) is rows[0]
