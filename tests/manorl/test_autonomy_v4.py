"""Independent numerical contract checks for the v4 Warp-only autonomy ABI."""
from __future__ import annotations
import numpy as np
import pytest
from dataclasses import replace
from sim.manorl.autonomy_contracts import RAW_OBSERVATION_DIM, ENCODED_OBSERVATION_DIM, raw_observation_slices, encoded_observation_slices, validate_v4_checkpoint_metadata, CHECKPOINT_FORMAT, OBSERVATION_CONTRACT_ID, REWARD_CONTRACT_ID, ACTION_CONTRACT_ID
from sim.manorl.autonomy_v4 import ReferenceCacheV4, V4Physical, V4Contact, anchor_delta_and_velocity, build_raw_observation, compute_reward, encode_observation, motion_gate_weight, quat_rotate, relative_rot6, rot6, shortest_angle, reduce_pyramidal_contacts_v4


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


def test_relative_rot6_normalizes_operands_before_multiplication():
    jax=pytest.importorskip("jax"); j=jax.numpy
    huge=j.asarray([[0.,0.,0.,1.e30]])
    result=np.asarray(relative_rot6(huge,huge))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result,[[1.,0.,0.,0.,1.,0.]],atol=1e-6)

def test_reference_anchor_error_is_zero_for_positive_gap():
    jax=pytest.importorskip("jax"); j=jax.numpy; cache=_cache(); state=_state(); raw=build_raw_observation(state,_contact(),cache,j.asarray([0]),j.zeros((1,28))); g=raw[:,raw_observation_slices()["autonomous_geometry"]].reshape(1,16,22)
    np.testing.assert_allclose(np.asarray(g[...,13:16]),0.,atol=1e-6)
    np.testing.assert_allclose(np.asarray(raw[:,119+61:119+83]),0.,atol=1e-6)
    assert np.all(np.asarray(g[...,0])>0)

def test_reference_phase_and_future_validity_use_reference_relative_delta():
    jax=pytest.importorskip("jax"); j=jax.numpy
    cache=_cache(); values={**cache.__dict__}
    palm=np.asarray(cache.palm_origin).copy(); obj=np.asarray(cache.object_origin).copy()
    palm[0]=[.1,0,0]; obj[0]=[.2,0,0]; palm[6]=[.6,0,0]; obj[6]=[.3,0,0]
    values.update(palm_origin=palm,object_origin=obj)
    cache=cache.__class__(**values); raw=build_raw_observation(_state(),_contact(),cache,j.asarray([0]),j.zeros((1,28)))
    ref=np.asarray(raw[0,119:228]); future=np.asarray(raw[0,228:351])
    np.testing.assert_allclose(ref[-2:],[0.,1.],atol=1e-6)
    # h=6 translation is ref-relative future minus current reference-relative.
    np.testing.assert_allclose(future[22:25],[((.6-.3)-(.1-.2))/.1,0,0],atol=1e-6)
    late=build_raw_observation(_state(),_contact(),cache,j.asarray([20]),j.zeros((1,28)))
    assert np.asarray(late[0,228+40]) == 0. and np.asarray(late[0,228+41+40]) == 0.


def test_analytic_rotating_anchor_velocity_matches_closed_form():
    jax=pytest.importorskip("jax"); j=jax.numpy; s=_state(); s=s._replace(object_w=j.asarray([[0.,0.,2.]]),region_origin=j.asarray([[[1.,0.,0.]]*16]))
    _,_,delta,velocity=anchor_delta_and_velocity(s,j.zeros((1,16,3)),j.zeros((1,16,3)))
    # stationary world hand and rotating object frame: -omega x delta
    np.testing.assert_allclose(np.asarray(delta[0,0]),[1,0,0],atol=1e-6); np.testing.assert_allclose(np.asarray(velocity[0,0]),[0,-2,0],atol=1e-6)

def test_contact_pair_order_all_object_and_torque():
    jax=pytest.importorskip("jax"); j=jax.numpy
    kw=dict(nacon=j.asarray(3),nefc=j.asarray([12]),dimension=j.asarray([3,3,3,3]),addresses=j.asarray([[0,1,2,3],[4,5,6,7],[8,9,10,11],[0,0,0,0]]),friction=j.ones((4,5)),frame=j.tile(j.eye(3),(4,1,1)),position=j.asarray([[0,1,0],[0,0,0],[0,0,0.],[0,0,0.]]),constraint_force=j.asarray([[.5,.5,0,0, 1,1,0,0, 1.5,1.5,0,0,0]]),ngeom=4,hand_geom_ids=[0]+[3]*15,object_geom_ids=[1],object_com=j.zeros((1,3)))
    # hand-object, object-hand, object-table: paired should sum +1 and -2; all additionally has -3 table contact.
    allf,pair,tau,count,slip,valid=reduce_pyramidal_contacts_v4(geom=j.asarray([[0,1],[1,0],[1,2],[2,2]]),world=j.zeros(4,j.int32),**kw)
    np.testing.assert_allclose(np.asarray(pair[0,0]),[-1,0,0],atol=1e-6); np.testing.assert_allclose(np.asarray(allf[0]),[-4,0,0],atol=1e-6); np.testing.assert_allclose(np.asarray(tau[0,0]),[0,0,-1],atol=1e-6); assert bool(valid)

def test_contact_tangential_velocity_is_force_weighted_and_nonzero():
    jax=pytest.importorskip("jax"); j=jax.numpy
    kw=dict(nacon=j.asarray(1),nefc=j.asarray([4]),geom=j.asarray([[0,16],[0,0]]),world=j.asarray([0,0]),dimension=j.asarray([3,3]),addresses=j.asarray([[0,1,2,3],[-1,-1,-1,-1]]),friction=j.ones((2,5)),frame=j.tile(j.eye(3),(2,1,1)),position=j.asarray([[0.,0.,0.],[np.nan,np.nan,np.nan]]),constraint_force=j.asarray([[1.,1.,0.,0.,0.]]),ngeom=17,hand_geom_ids=list(range(16)),object_geom_ids=[16],object_com=j.zeros((1,3)),hand_com=j.zeros((1,16,3)),hand_v_com=j.tile(j.asarray([[[0.,3.,0.]]]),(1,16,1)),hand_w=j.zeros((1,16,3)),object_v_com=j.zeros((1,3)),object_w=j.zeros((1,3)))
    _,pair,_,count,slip,valid=reduce_pyramidal_contacts_v4(**kw)
    assert bool(valid) and int(count[0,0])==1
    np.testing.assert_allclose(np.asarray(pair[0,0]),[2,0,0],atol=1e-6)
    # frame[0] is normal x: y relative velocity is exactly tangential slip.
    np.testing.assert_allclose(np.asarray(slip[0,0]),[0,3,0],atol=1e-6)


def test_masked_reset_only_changes_dynamic_batched_fields():
    from sim.manorl.environment import _build_masked_reset_data_fn, recommended_warp_contact_capacity
    from typing import NamedTuple
    jax=pytest.importorskip("jax"); j=jax.numpy
    class Data(NamedTuple):
        qpos: object; qvel: object; ctrl: object; global_contact: object
        def replace(self, **kwargs): return self._replace(**kwargs)
    old=Data(j.arange(9,dtype=j.float32).reshape(3,3),j.ones((3,2)),j.full((3,2),4.),j.arange(8).reshape(4,2))
    reset=_build_masked_reset_data_fn(jax=jax,jp=j,reset_qpos=j.full((3,3),9.),reset_ctrl=j.full((3,2),8.))
    got=reset(old,j.asarray([False,True,False]))
    np.testing.assert_array_equal(np.asarray(got.qpos),[[0,1,2],[9,9,9],[6,7,8]])
    np.testing.assert_array_equal(np.asarray(got.qvel),[[1,1],[0,0],[1,1]])
    np.testing.assert_array_equal(np.asarray(got.global_contact),np.asarray(old.global_contact))
    # one global arena grows with B; it is not B independent 128-contact slots.
    assert recommended_warp_contact_capacity(3,("right",)) > recommended_warp_contact_capacity(1,("right",))


def test_contact_rejects_unsupported_cone_and_skips_unsolved_rows():
    jax=pytest.importorskip("jax"); j=jax.numpy
    kwargs=dict(nacon=j.asarray(1),nefc=j.asarray([1]),geom=j.asarray([[0,16],[0,0]]),world=j.asarray([0,0]),dimension=j.asarray([3,3]),addresses=j.asarray([[-1,-1,-1,-1],[0,0,0,0]]),friction=j.ones((2,5)),frame=j.tile(j.eye(3),(2,1,1)),position=j.asarray([[np.nan,np.nan,np.nan],[0,0,0.]]),constraint_force=j.zeros((1,2)),ngeom=17,hand_geom_ids=list(range(16)),object_geom_ids=[16],object_com=j.zeros((1,3)))
    _,pair,_,count,slip,valid=reduce_pyramidal_contacts_v4(**kwargs)
    assert bool(valid) and int(count.sum())==0 and np.isfinite(np.asarray(pair)).all()
    with pytest.raises(ValueError,match="pyramidal"):
        reduce_pyramidal_contacts_v4(**kwargs,cone="elliptic")

def test_reward_boundaries_clipped_action_and_reason_bits():
    jax=pytest.importorskip("jax"); j=jax.numpy; cache=_cache(); s=_state(); r=compute_reward(s,_contact(),cache,j.asarray([0]),j.full((1,28),2.))
    # Synthetic reference is stationary: object terms retain precisely 1%.
    np.testing.assert_allclose(np.asarray(r.action),[-.002],atol=1e-7); np.testing.assert_allclose(np.asarray(r.object_position),[.012],atol=1e-6)
    np.testing.assert_allclose(np.asarray(r.hand_relative),[.125],atol=1e-6); np.testing.assert_allclose(np.asarray(r.geometry),[1.2],atol=1e-6); assert not bool(r.done[0])
    fallen=s._replace(object_bottom=j.asarray([-.06])); f=compute_reward(fallen,_contact(),cache,j.asarray([0]),j.zeros((1,28))); assert int(f.reason[0])&4 and np.isclose(np.asarray(f.severe)[0],-75)


def test_reference_speed_motion_gate_is_exact_and_independent_of_actual_speed():
    jax=pytest.importorskip("jax"); j=jax.numpy
    # Endpoints, midpoint, and monotonicity of smoothstep 1% -> 100%.
    weights=np.asarray(motion_gate_weight(j.asarray([0., .01, .055, .10, .2])))
    np.testing.assert_allclose(weights,[.01,.01,.505,1.,1.],atol=1e-7)
    assert np.all(np.diff(weights)>=0)
    stationary=_cache(); state=_state(); action=j.zeros((1,28))
    r_still=compute_reward(state,_contact(),stationary,j.asarray([0]),action)
    linear=replace(stationary,object_v_com=np.tile([.10,0.,0.],(len(stationary.q_feasible),1)))
    r_linear=compute_reward(state,_contact(),linear,j.asarray([0]),action)
    angular=replace(stationary,object_w=np.tile([.055/stationary.object_radius,0.,0.],(len(stationary.q_feasible),1)))
    r_angular=compute_reward(state,_contact(),angular,j.asarray([0]),action)
    np.testing.assert_allclose(np.asarray(r_still.object_position),[.012],atol=1e-6)
    np.testing.assert_allclose(np.asarray(r_linear.object_position),[1.2],atol=1e-6)
    np.testing.assert_allclose(np.asarray(r_angular.object_position),[.606],atol=1e-6)
    # A changed actual speed changes the velocity-match term, but never the gate.
    fast_actual=state._replace(object_v_com=j.asarray([[100.,0.,0.]]),object_w=j.asarray([[100.,0.,0.]]))
    r_fast_actual=compute_reward(fast_actual,_contact(),linear,j.asarray([0]),action)
    np.testing.assert_allclose(np.asarray(r_fast_actual.object_position),np.asarray(r_linear.object_position),atol=1e-7)

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

def test_cpu_warp_contact_force_oracle_matches_dim3_decoder_without_mjdata():
    """Controlled real contact: Warp Data/forward and bundled helper only."""
    jax=pytest.importorskip("jax")
    warp=pytest.importorskip("warp"); warp.set_device("cpu")
    from mujoco.mjx.third_party import mujoco_warp
    from sim.manorl.assets import compile_model_metadata_only
    from sim.manorl.environment import MjxWarpPhysicalProducer
    mujoco,model=compile_model_metadata_only(object_type="cube2",hand_side="right",physics_timestep=1/480)
    producer=MjxWarpPhysicalProducer(mujoco,model,object_type="cube2",hand_sides=("right",))
    model_warp=mujoco_warp.put_model(model)
    data=mujoco_warp.make_data(model,nworld=1,naconmax=128,njmax=512)
    qpos=np.zeros(model.nq,np.float32)
    qpos[producer.object_qpos_address:producer.object_qpos_address+7]=[0,0,.05,1,0,0,0]
    warp.copy(data.qpos,warp.array(qpos[None],dtype=warp.float32)); mujoco_warp.forward(model_warp,data); warp.synchronize()
    nacon=int(data.nacon.numpy()[0]); assert nacon > 0
    allf,pair,torque,count,slip,valid=reduce_pyramidal_contacts_v4(
        nacon=np.asarray(nacon,np.int32),nefc=data.nefc.numpy(),geom=data.contact.geom.numpy(),world=data.contact.worldid.numpy(),dimension=data.contact.dim.numpy(),addresses=data.contact.efc_address.numpy(),friction=data.contact.friction.numpy(),frame=data.contact.frame.numpy(),position=data.contact.pos.numpy(),constraint_force=data.efc.force.numpy(),ngeom=model.ngeom,hand_geom_ids=producer.keypoint_geom_ids,object_geom_ids=tuple(producer.object_geom_ids),object_com=data.xipos.numpy()[:,producer.object_body_id],hand_com=np.zeros((1,16,3),np.float32),hand_v_com=np.zeros((1,16,3),np.float32),hand_w=np.zeros((1,16,3),np.float32),object_v_com=np.zeros((1,3),np.float32),object_w=np.zeros((1,3),np.float32),
    )
    ids=warp.array(np.arange(nacon,dtype=np.int32),dtype=warp.int32,device="cpu")
    oracle=warp.empty(nacon,dtype=warp.spatial_vectorf,device="cpu")
    mujoco_warp.contact_force(model_warp,data,ids,True,oracle); warp.synchronize()
    geom=data.contact.geom.numpy()[:nacon]; force=oracle.numpy()[:,:3]
    # The fixture has a real hand geom1 -> object geom2 force; compare its
    # world vector and lever moment to the JAX reduction exactly.
    k=next(i for i,(a,b) in enumerate(geom) if a in producer.keypoint_geom_ids and b in producer.object_geom_ids)
    region=list(producer.keypoint_geom_ids).index(int(geom[k,0]))
    np.testing.assert_allclose(np.asarray(pair[0,region]),force[k],rtol=2e-5,atol=2e-5)
    expected_tau=np.cross(data.contact.pos.numpy()[k]-data.xipos.numpy()[0,producer.object_body_id],force[k])
    np.testing.assert_allclose(np.asarray(torque[0,region]),expected_tau,rtol=2e-5,atol=2e-5)
    assert bool(valid) and int(count[0,region]) > 0 and np.linalg.norm(force[k]) > 0


def test_scaled_actual_reference_and_bottom_fields_are_named_physical_quantities():
    jax=pytest.importorskip("jax"); j=jax.numpy; cache=_cache(); cache=cache.__class__(**{**cache.__dict__,"reference_bottom":np.full(25,.03),"table_height":-.001,"control_timestep":1/120,"duration":24/120})
    state=_state(); state=state._replace(q_raw=j.asarray(np.r_[np.zeros(6),np.full(22,.3)][None]),q_normalized=j.asarray(np.r_[np.zeros(6),np.full(22,.25)][None]),object_v_com=j.asarray([[1.,0,0]]),object_w=j.asarray([[0,0,3.]]),palm_origin=j.asarray([[.1,0,0]]),palm_w=j.asarray([[0,0,3.]]),object_bottom=j.asarray([.02]))
    raw=build_raw_observation(state,_contact(),cache,j.asarray([0]),j.zeros((1,28))); actual=np.asarray(raw[0,:119]); ref=np.asarray(raw[0,119:228])
    np.testing.assert_allclose(actual[93:99],[1,0,0,0,0,1],atol=1e-6)
    np.testing.assert_allclose(actual[99:102],[1,0,0],atol=1e-6)
    np.testing.assert_allclose(actual[111:113],[.21,.31],atol=1e-6)
    np.testing.assert_allclose(ref[61:83],1.,atol=1e-6)
