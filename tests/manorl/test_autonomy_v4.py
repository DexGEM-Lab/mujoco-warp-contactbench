"""Independent numerical contract checks for the v4 Warp-only autonomy ABI."""
from __future__ import annotations
import numpy as np
import pytest
from sim.manorl.autonomy_contracts import RAW_OBSERVATION_DIM, ENCODED_OBSERVATION_DIM, raw_observation_slices, encoded_observation_slices, validate_v4_checkpoint_metadata, CHECKPOINT_FORMAT, OBSERVATION_CONTRACT_ID, REWARD_CONTRACT_ID, ACTION_CONTRACT_ID
from sim.manorl.autonomy_v4 import ReferenceCacheV4, V4Physical, V4Contact, anchor_delta_and_velocity, build_raw_observation, compute_reward, encode_observation, quat_rotate, rot6, shortest_angle, reduce_pyramidal_contacts_v4


def _cache(T=25):
    q=np.zeros((T,28)); quat=np.tile([0.,0.,0.,1.],(T,1)); ah=np.zeros((T,16,3)); ao=np.zeros_like(ah); gap=np.full((T,16),.003)
    h=ReferenceCacheV4(q,q,-np.ones(28),np.ones(28),np.zeros((T,3)),quat,np.zeros(3),np.zeros((T,3)),quat,np.zeros((T,3)),np.zeros((T,3)),np.zeros((T,3)),np.zeros((T,3)),ah,ao,np.zeros_like(ah),gap,np.exp(-(gap/.01)**2),np.ones((T,16)),np.ones((T,16)),np.full(T,.02),-.001,1/120,(T-1)/120,np.zeros((64,3)),np.array([.25,.25,.25]+[0.]*9),2,np.zeros(3),"test")
    return h

def _state(b=1):
    jax=pytest.importorskip("jax"); j=jax.numpy
    quat=j.tile(j.asarray([0.,0.,0.,1.]),(b,1)); rquat=j.tile(quat[:,None],(1,16,1)); zero=j.zeros((b,3)); region=j.zeros((b,16,3)); return V4Physical(q_raw=j.zeros((b,28)),q_normalized=j.zeros((b,28)),qdot=j.zeros((b,28)),command_error=j.zeros((b,28)),object_origin=zero,object_quat_xyzw=quat,object_com=zero,object_v_com=zero,object_w=zero,palm_origin=zero,palm_quat_xyzw=quat,palm_w=zero,region_origin=region,region_quat_xyzw=rquat,region_com=region,region_v_com=j.zeros((b,16,3)),region_w=j.zeros((b,16,3)),object_bottom=j.full((b,),.02),valid=j.ones((b,),bool))

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
    np.testing.assert_allclose(np.asarray(g[...,13:16]),0.,atol=1e-6)
    np.testing.assert_allclose(np.asarray(raw[:,119+61:119+83]),0.,atol=1e-6)
    assert np.all(np.asarray(g[...,0])>0)

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

def test_skewed_convex_mesh_inside_projects_to_surface_and_outside_is_positive():
    from sim.manorl.autonomy_v4 import signed_closest_convex_mesh
    jax=pytest.importorskip("jax"); j=jax.numpy
    # Skewed tetrahedron has no axis-aligned-box ambiguity.
    tri=j.asarray([[[0,0,0],[0,1,0],[1,0,0]],[[0,0,0],[1,0,0],[0,0,1]],[[0,0,0],[0,0,1],[0,1,0]],[[1,0,0],[0,1,0],[0,0,1]]],dtype=j.float32)
    point=j.asarray([[.15,.15,.15],[2.,2.,2.]])
    closest,signed,inside=signed_closest_convex_mesh(point,tri)
    assert bool(inside[0]) and float(signed[0]) < 0 and not bool(inside[1]) and float(signed[1]) > 0
    assert not np.allclose(np.asarray(closest[0]),np.asarray(point[0]))

def test_mesh_sampling_uses_compiled_geom_pose_and_has_surface_spread():
    from sim.manorl.assets import compile_model_metadata_only
    from sim.manorl.environment import MjxWarpPhysicalProducer
    from sim.manorl.autonomy_v4 import _mesh_body_geometry, _deterministic_surface_samples
    mujoco,model=compile_model_metadata_only(object_type="cube2",hand_side="right",physics_timestep=1/480)
    producer=MjxWarpPhysicalProducer(mujoco,model,object_type="cube2",hand_sides=("right",))
    assert len(producer.keypoint_geom_ids)==16
    assert all(int(model.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_MESH) for g in [*producer.keypoint_geom_ids,*producer.object_geom_ids])
    assert np.all(np.linalg.norm(model.geom_pos[producer.keypoint_geom_ids],axis=1)>0)
    for gid in producer.keypoint_geom_ids:
        vertices,faces=_mesh_body_geometry(model,gid); samples=_deterministic_surface_samples(vertices,faces,128)
        assert samples.shape==(128,3) and len(np.unique(np.round(samples,10),axis=0))==128

def test_reference_cache_is_warp_only_even_when_native_fk_oracle_is_disabled(monkeypatch):
    from sim.manorl import assets
    from sim.manorl.trajectory_package import load_trajectory_package
    from sim.manorl.autonomy_v4 import compile_reference_cache_v4
    monkeypatch.setattr(assets,"validate_static_fk",lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError("native FK called")))
    path="/home/jay/dexrobot/FromSSH/manoRL_mujoco/outputs/manorl/contact_conditioned_autonomy/cube2_02_v295_f120_pre180_post180"
    trajectory=next(t for t in load_trajectory_package(path).trajectories if t.identity.identity=="cube2_02_2833")
    cache=compile_reference_cache_v4(trajectory,device="cpu")
    assert cache.q_feasible.flags.writeable is False and cache.q_raw.flags.writeable is False
    assert len(np.unique(np.round(cache.points_object_local,10),axis=0))==64 and np.max(np.abs(cache.object_w))>0

def test_scaled_actual_reference_and_bottom_fields_are_named_physical_quantities():
    jax=pytest.importorskip("jax"); j=jax.numpy; cache=_cache(); cache=cache.__class__(**{**cache.__dict__,"reference_bottom":np.full(25,.03),"table_height":-.001,"control_timestep":1/120,"duration":24/120})
    state=_state(); state=state._replace(q_raw=j.asarray(np.r_[np.zeros(6),np.full(22,.3)][None]),q_normalized=j.asarray(np.r_[np.zeros(6),np.full(22,.25)][None]),object_v_com=j.asarray([[1.,0,0]]),object_w=j.asarray([[0,0,3.]]),palm_origin=j.asarray([[.1,0,0]]),palm_w=j.asarray([[0,0,3.]]),object_bottom=j.asarray([.02]))
    raw=build_raw_observation(state,_contact(),cache,j.asarray([0]),j.zeros((1,28))); actual=np.asarray(raw[0,:119]); ref=np.asarray(raw[0,119:228])
    np.testing.assert_allclose(actual[93:99],[1,0,0,0,0,1],atol=1e-6)
    np.testing.assert_allclose(actual[99:102],[1,0,0],atol=1e-6)
    np.testing.assert_allclose(actual[111:113],[.21,.31],atol=1e-6)
    np.testing.assert_allclose(ref[61:83],1.,atol=1e-6)
