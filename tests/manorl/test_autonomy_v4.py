"""Independent numerical contract checks for the v4 Warp-only autonomy ABI."""
from __future__ import annotations
import numpy as np
import pytest
from sim.manorl.autonomy_contracts import RAW_OBSERVATION_DIM, ENCODED_OBSERVATION_DIM, raw_observation_slices, encoded_observation_slices, validate_v4_checkpoint_metadata, CHECKPOINT_FORMAT, OBSERVATION_CONTRACT_ID, REWARD_CONTRACT_ID, ACTION_CONTRACT_ID
from sim.manorl.autonomy_v4 import ReferenceCacheV4, V4Physical, V4Contact, anchor_delta_and_velocity, build_raw_observation, compute_reward, encode_observation, quat_rotate, rot6, shortest_angle, reduce_pyramidal_contacts_v4


def _cache(T=25):
    q=np.zeros((T,28)); quat=np.tile([0.,0.,0.,1.],(T,1)); ah=np.zeros((T,16,3)); ao=np.zeros_like(ah); gap=np.full((T,16),.003)
    h=ReferenceCacheV4(q,q,np.zeros((T,3)),quat,np.zeros(3),np.zeros((T,3)),quat,np.zeros((T,3)),np.zeros((T,3)),np.zeros((T,3)),np.zeros((T,3)),ah,ao,np.zeros_like(ah),gap,np.exp(-(gap/.01)**2),np.ones((T,16)),np.ones((T,16)),np.zeros((64,3)),np.array([.25,.25,.25]+[0.]*9),2,np.zeros(3),"test")
    return h

def _state(b=1):
    jax=pytest.importorskip("jax"); j=jax.numpy
    quat=j.tile(j.asarray([0.,0.,0.,1.]),(b,1)); rquat=j.tile(quat[:,None],(1,16,1)); zero=j.zeros((b,3)); region=j.zeros((b,16,3)); return V4Physical(q=j.zeros((b,28)),qdot=j.zeros((b,28)),command_error=j.zeros((b,28)),object_origin=zero,object_quat_xyzw=quat,object_com=zero,object_v_com=zero,object_w=zero,palm_origin=zero,palm_quat_xyzw=quat,palm_w=zero,region_origin=region,region_quat_xyzw=rquat,region_com=region,region_v_com=j.zeros((b,16,3)),region_w=j.zeros((b,16,3)),object_bottom=j.full((b,),.02),valid=j.ones((b,),bool))

def _contact(b=1):
    jax=pytest.importorskip("jax"); j=jax.numpy
    return V4Contact(j.zeros((b,3)),j.zeros((b,16,3)),j.zeros((b,16,3)),j.zeros((b,16),j.int32),j.zeros((b,16,3)),j.ones((b,),bool))

def test_dimensions_slices_and_encoded_layout():
    assert list(raw_observation_slices().values())[-1].stop==RAW_OBSERVATION_DIM==957
    assert list(encoded_observation_slices().values())[-1].stop==ENCODED_OBSERVATION_DIM==829
    jax=pytest.importorskip("jax"); raw=jax.numpy.zeros((2,957)); encoded=encode_observation(raw,jax.numpy.zeros((2,64)))
    assert encoded.shape==(2,829)

def test_xyzw_double_cover_and_rotation6_columns():
    jax=pytest.importorskip("jax"); j=jax.numpy
    q=j.asarray([[0.,0.,np.sqrt(.5),np.sqrt(.5)]]); np.testing.assert_allclose(np.asarray(rot6(q)),np.asarray(rot6(-q)),atol=1e-6); np.testing.assert_allclose(np.asarray(shortest_angle(q,-q)),0.,atol=1e-6)
    np.testing.assert_allclose(np.asarray(quat_rotate(q,j.asarray([[1.,0.,0.]]))),[[0.,1.,0.]],atol=1e-6)

def test_reference_anchor_error_is_zero_for_positive_gap():
    jax=pytest.importorskip("jax"); j=jax.numpy; cache=_cache(); state=_state(); raw=build_raw_observation(state,_contact(),cache,j.asarray([0]),j.zeros((1,28))); g=raw[:,raw_observation_slices()["autonomous_geometry"]].reshape(1,16,22)
    np.testing.assert_allclose(np.asarray(g[...,13:16]),0.,atol=1e-6); assert np.all(np.asarray(g[...,0])>0)

def test_analytic_rotating_anchor_velocity_matches_closed_form():
    jax=pytest.importorskip("jax"); j=jax.numpy; s=_state(); s=s._replace(object_w=j.asarray([[0.,0.,2.]]),region_origin=j.asarray([[[1.,0.,0.]]*16]))
    _,_,delta,velocity=anchor_delta_and_velocity(s,j.zeros((1,16,3)),j.zeros((1,16,3)))
    # stationary world hand and rotating object frame: -omega x delta
    np.testing.assert_allclose(np.asarray(delta[0,0]),[1,0,0],atol=1e-6); np.testing.assert_allclose(np.asarray(velocity[0,0]),[0,-2,0],atol=1e-6)

def test_contact_pair_order_all_object_and_torque():
    jax=pytest.importorskip("jax"); j=jax.numpy
    kw=dict(nacon=j.asarray(3),nefc=j.asarray([12]),dimension=j.asarray([3,3,3,3]),addresses=j.asarray([[0,1,2,3],[4,5,6,7],[8,9,10,11],[0,0,0,0]]),friction=j.ones((4,5)),frame=j.tile(j.eye(3),(4,1,1)),position=j.asarray([[0,1,0],[0,0,0],[0,0,0.],[0,0,0.]]),constraint_force=j.asarray([[.5,.5,0,0, 1,1,0,0, 1.5,1.5,0,0,0]]),ngeom=4,hand_geom_ids=[0]+[3]*15,object_geom_ids=[1],object_com=j.zeros((1,3)))
    # hand-object, object-hand, object-table: paired should sum +1 and -2; all additionally has -3 table contact.
    allf,pair,tau,count,valid=reduce_pyramidal_contacts_v4(geom=j.asarray([[0,1],[1,0],[1,2],[2,2]]),world=j.zeros(4,j.int32),**kw)
    np.testing.assert_allclose(np.asarray(pair[0,0]),[-1,0,0],atol=1e-6); np.testing.assert_allclose(np.asarray(allf[0]),[-4,0,0],atol=1e-6); np.testing.assert_allclose(np.asarray(tau[0,0]),[0,0,-1],atol=1e-6); assert bool(valid)

def test_reward_boundaries_clipped_action_and_reason_bits():
    jax=pytest.importorskip("jax"); j=jax.numpy; cache=_cache(); s=_state(); r=compute_reward(s,_contact(),cache,j.asarray([0]),j.full((1,28),2.))
    np.testing.assert_allclose(np.asarray(r.action),[-.002],atol=1e-7); np.testing.assert_allclose(np.asarray(r.object_position),[1.2],atol=1e-6); assert not bool(r.done[0])
    fallen=s._replace(object_bottom=j.asarray([-.06])); f=compute_reward(fallen,_contact(),cache,j.asarray([0]),j.zeros((1,28))); assert int(f.reason[0])&4 and np.isclose(np.asarray(f.severe)[0],-25)

def test_v4_checkpoint_rejects_legacy_metadata():
    validate_v4_checkpoint_metadata({"checkpoint_format":CHECKPOINT_FORMAT,"observation_contract":OBSERVATION_CONTRACT_ID,"reward_contract":REWARD_CONTRACT_ID,"action_contract":ACTION_CONTRACT_ID})
    with pytest.raises(ValueError): validate_v4_checkpoint_metadata({"checkpoint_format":"manorl.autonomy.ppo.v3.1"})
