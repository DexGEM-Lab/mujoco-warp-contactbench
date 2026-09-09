"""Warp-only cube2 contact-conditioned autonomy v4 primitives.

This module owns the observation and reward meaning.  Inputs are device arrays
from one forward-consistent MJX state; it has no native MuJoCo data calls.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import Any, NamedTuple
import numpy as np
from sim.manorl.autonomy_contracts import RAW_OBSERVATION_DIM, ENCODED_OBSERVATION_DIM, raw_observation_slices, encoded_observation_slices

REGIONS = 16
POINTS = 64
GEOMETRY_PER_REGION = 22

class V4Physical(NamedTuple):
    q: Any; qdot: Any; command_error: Any
    object_origin: Any; object_quat_xyzw: Any; object_com: Any; object_v_com: Any; object_w: Any
    palm_origin: Any; palm_quat_xyzw: Any; palm_w: Any
    region_origin: Any; region_quat_xyzw: Any; region_com: Any; region_v_com: Any; region_w: Any
    object_bottom: Any; valid: Any

class V4Contact(NamedTuple):
    object_all_force: Any; paired_force_on_object: Any; paired_torque_com: Any
    paired_count: Any; tangential_slip: Any; valid: Any

@dataclass(frozen=True)
class ReferenceCacheV4:
    """Immutable T-only cache. Identity is deliberately not a policy feature."""
    q_feasible: np.ndarray; q_raw: np.ndarray
    object_origin: np.ndarray; object_quat_xyzw: np.ndarray; object_com_local: np.ndarray
    palm_origin: np.ndarray; palm_quat_xyzw: np.ndarray
    object_v_com: np.ndarray; object_w: np.ndarray; palm_v: np.ndarray; palm_w: np.ndarray
    region_anchor_hand: np.ndarray; region_anchor_object: np.ndarray; delta_ref: np.ndarray
    signed_gap: np.ndarray; proximity: np.ndarray; confidence: np.ndarray; valid: np.ndarray
    points_object_local: np.ndarray; object_geometry: np.ndarray; action_id: int
    support_shift: np.ndarray; content_hash: str
    def __post_init__(self):
        t = len(self.q_feasible)
        expected = {"q_feasible":(t,28), "q_raw":(t,28), "object_origin":(t,3), "object_quat_xyzw":(t,4), "palm_origin":(t,3), "palm_quat_xyzw":(t,4), "object_v_com":(t,3), "object_w":(t,3), "palm_v":(t,3), "palm_w":(t,3), "region_anchor_hand":(t,16,3), "region_anchor_object":(t,16,3), "delta_ref":(t,16,3), "signed_gap":(t,16), "proximity":(t,16), "confidence":(t,16), "valid":(t,16), "points_object_local":(64,3), "object_geometry":(12,), "support_shift":(3,)}
        for name, shape in expected.items():
            value=np.asarray(getattr(self,name));
            if value.shape != shape or not np.all(np.isfinite(value)): raise ValueError(f"v4 cache {name} must be finite {shape}")
        if not 1 <= self.action_id <= 50: raise ValueError("action_id must be in [1,50]")

def compile_reference_cache_v4(trajectory, *, device: str = "cpu", object_type: str = "cube2", table_height: float = -0.001) -> ReferenceCacheV4:
    """Compile a cube2 cache with N=1 Warp FK and exact box collision distance.

    It intentionally creates MJX data directly: host ``MjModel`` supplies
    static geom metadata, but no ``MjData``, native forward, step, or distance
    query participates in cache production.  cube2's collision geom is a box,
    so closest points/signs are exact rather than visual/AABB fallback.
    """
    import jax, jax.numpy as j
    from mujoco import mjx
    from sim.manorl.assets import compile_model, object_collision_vertices
    from sim.manorl.autonomy import align_reference_trajectory
    from sim.manorl.environment import MjxWarpPhysicalProducer
    mujoco, model=compile_model(object_type=object_type,hand_side="right",physics_timestep=1/480)
    producer=MjxWarpPhysicalProducer(mujoco,model,object_type=object_type,hand_sides=("right",))
    dev=jax.devices(device)[0]; m=mjx.put_model(model,device=dev,impl="warp")
    q,obj,shift=align_reference_trajectory(trajectory,np.asarray(object_collision_vertices(object_type)),table_height)
    T=len(q); base=mjx.make_data(model,device=dev,impl="warp")
    qpos=np.broadcast_to(np.asarray(base.qpos),(T,model.nq)).copy(); qpos[:,:28]=q
    qpos[:,producer.object_qpos_address:producer.object_qpos_address+3]=obj
    qpos[:,producer.object_qpos_address+3:producer.object_qpos_address+7]=np.asarray(trajectory.object_quat_xyzw)[:,(3,0,1,2)]
    data=jax.vmap(lambda x: mjx.forward(m,base.replace(qpos=x)))(j.asarray(qpos))
    xpos=np.asarray(data.xpos); xquat=np.asarray(data.xquat)[...,(1,2,3,0)]
    body=np.asarray(producer.keypoint_body_ids); hand_origin=xpos[:,body]; hand_q=xquat[:,body]; object_origin=xpos[:,producer.object_body_id]; object_q=xquat[:,producer.object_body_id]
    # Deterministic 128 samples per actual collision primitive: cube2 is exact
    # box geometry. Region collision-geoms in this asset are capsule/box; the
    # local direction lattice gives surface samples without visual meshes.
    lattice=np.linspace(-1.,1.,8); grid=np.array(np.meshgrid(lattice,lattice,lattice,indexing='ij')).reshape(3,-1).T[:128]
    geom_size=np.asarray(model.geom_size)[np.asarray(producer.keypoint_geom_ids)]
    local=np.empty((16,128,3),np.float64)
    for r,size in enumerate(geom_size):
        axis=np.argmax(np.abs(grid),axis=1); sample=grid.copy(); sample[np.arange(128),axis]=np.sign(sample[np.arange(128),axis]); local[r]=sample*np.maximum(size[:3],1e-5)
    # Object box extent comes from the collision geom itself, never visuals.
    object_geom=next(iter(producer.object_geom_ids)); extent=np.asarray(model.geom_size[object_geom,:3],np.float64)
    world=hand_origin[:,:,None,:]+np.asarray(quat_rotate(j.asarray(hand_q[:,:,None,:]),j.asarray(local)[None]))
    objlocal=np.asarray(quat_unrotate(j.asarray(object_q[:,None,None,:]),j.asarray(world-object_origin[:,None,None,:])))
    outside=np.maximum(np.abs(objlocal)-extent,0.); unsigned=np.linalg.norm(outside,axis=-1); inside=np.all(np.abs(objlocal)<=extent,axis=-1); signed=np.where(inside,-np.min(extent-np.abs(objlocal),axis=-1),unsigned)
    chosen=np.argmin(np.abs(signed),axis=-1); rows=np.arange(T)[:,None]; regions=np.arange(16)[None,:]
    ah=local[None].repeat(T,axis=0)[rows,regions,chosen]
    sample_o=objlocal[rows,regions,chosen]; ao=np.clip(sample_o,-extent,extent)
    hand_anchor_world=hand_origin+np.asarray(quat_rotate(j.asarray(hand_q),j.asarray(ah)))
    object_anchor_world=object_origin[:,None]+np.asarray(quat_rotate(j.asarray(object_q[:,None,:]),j.asarray(ao)))
    delta=np.asarray(quat_unrotate(j.asarray(object_q[:,None,:]),j.asarray(hand_anchor_world-object_anchor_world)))
    gap=signed[rows,regions,chosen]; depth=np.maximum(-np.min(signed,axis=-1),0); fraction=np.mean(signed<-.001,axis=-1); prox=np.exp(-(np.maximum(gap,0)/.01)**2); conf=np.exp(-(np.maximum(depth-.001,0)/.003)**2)*(1-fraction); valid=(depth<=.01).astype(np.float64)
    dt=1/120; com_local=np.asarray(model.body_ipos[producer.object_body_id]); com=object_origin+np.asarray(quat_rotate(j.asarray(object_q),j.asarray(com_local)[None])); ov=np.gradient(com,dt,axis=0); ow=np.zeros_like(ov); palm=hand_origin[:,0]; pv=np.gradient(palm,dt,axis=0); pw=np.zeros_like(pv)
    raw=np.asarray(trajectory.q_ref); feasible=np.clip(raw,np.asarray(model.jnt_range[:28,0]),np.asarray(model.jnt_range[:28,1])); points=np.array([[sx*extent[0],sy*extent[1],sz*extent[2]] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)],float); points=np.resize(points,(64,3))
    geometry=np.zeros(12); geometry[:3]=np.clip(2*extent/.2,0,1)
    values=(feasible,raw,object_origin,object_q,com_local,palm,hand_q[:,0],ov,ow,pv,pw,ah,ao,delta,gap,prox,conf,valid,points,geometry,np.asarray(shift))
    digest=cache_hash(*[np.asarray(x) for x in values])
    return ReferenceCacheV4(feasible,raw,object_origin,object_q,com_local,palm,hand_q[:,0],ov,ow,pv,pw,ah,ao,delta,gap,prox,conf,valid,points,geometry,int(trajectory.identity.identity.split('_')[1]),np.asarray(shift),digest)

def cache_hash(*values: np.ndarray, kernel_version: str = "warp-collision-v4") -> str:
    h=hashlib.sha256(kernel_version.encode())
    for value in values: h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()

def quat_normalize(q):
    import jax.numpy as j
    return q / j.maximum(j.linalg.norm(q,axis=-1,keepdims=True), 1e-12)
def quat_conj(q):
    import jax.numpy as j
    return j.concatenate((-q[...,:3],q[...,3:]),axis=-1)
def quat_mul(a,b):
    import jax.numpy as j
    av,aw=a[...,:3],a[...,3:]; bv,bw=b[...,:3],b[...,3:]
    return j.concatenate((aw*bv+bw*av+j.cross(av,bv),aw*bw-j.sum(av*bv,axis=-1,keepdims=True)),axis=-1)
def quat_rotate(q,v):
    import jax.numpy as j
    q=quat_normalize(q); u=q[...,:3]; w=q[...,3:]
    return v*(2*w*w-1)+2*w*j.cross(u,v)+2*u*j.sum(u*v,axis=-1,keepdims=True)
def quat_unrotate(q,v): return quat_rotate(quat_conj(quat_normalize(q)),v)
def rot6(q):
    import jax.numpy as j
    ex=quat_rotate(q,j.broadcast_to(j.asarray([1.,0.,0.]),q.shape[:-1]+(3,))); ey=quat_rotate(q,j.broadcast_to(j.asarray([0.,1.,0.]),q.shape[:-1]+(3,)))
    return j.concatenate((ex,ey),axis=-1)
def shortest_angle(q, target):
    import jax.numpy as j
    return 2*j.arccos(j.clip(j.abs(j.sum(quat_normalize(q)*quat_normalize(target),axis=-1)),0.,1.))
def extract_v4_physical(*, qpos, qvel, xpos, xquat, xipos, subtree_com, cvel, body_rootid, hand_dof_address: int, object_body_id: int, palm_body_id: int, region_body_ids, object_bottom, lower, upper, previous_command, antiwindup):
    """Extract v4 state from a Warp-forwarded MJX Data without qvel shortcuts.

    ``cvel`` is COM-based rot:lin.  The explicit subtree shift preserves both
    origin and COM meanings even for off-centre inertial frames.
    """
    import jax.numpy as j
    ids=np.asarray(region_body_ids,dtype=np.int32)
    if ids.shape != (16,): raise ValueError("v4 requires exactly 16 source-order regions")
    qpos,qvel,xpos,xquat,xipos,subtree_com,cvel=map(j.asarray,(qpos,qvel,xpos,xquat,xipos,subtree_com,cvel))
    if any(x.ndim != n for x,n in ((qpos,2),(qvel,2),(xpos,3),(xquat,3),(xipos,3),(subtree_com,3),(cvel,3))): raise ValueError("v4 MJX state must be batched")
    if not (0 <= object_body_id < xpos.shape[1] and 0 <= palm_body_id < xpos.shape[1] and np.all((ids>=0)&(ids<xpos.shape[1]))): raise ValueError("v4 body metadata out of range")
    rootids=np.asarray(body_rootid,dtype=np.int32)
    if rootids.shape != (xpos.shape[1],) or np.any(rootids < 0) or np.any(rootids >= subtree_com.shape[1]): raise ValueError("body_rootid is incompatible with MJX state")
    def body_values(body):
        omega=cvel[:,body,:3]; linear=cvel[:,body,3:]; origin=xpos[:,body]; com=xipos[:,body]
        root_com=subtree_com[:,rootids[body]]
        v_origin=linear+j.cross(omega,origin-root_com); v_com=linear+j.cross(omega,com-root_com)
        return origin,quat_normalize(xquat[:,body][:,(1,2,3,0)]),com,v_com,omega
    oo,oq,oc,ov,ow=body_values(object_body_id); po,pq,pc,pv,pw=body_values(palm_body_id)
    ro,rq,rc,rv,rw=(xpos[:,ids],quat_normalize(xquat[:,ids][...,(1,2,3,0)]),xipos[:,ids],None,cvel[:,ids,:3])
    root=subtree_com[:,rootids[ids]]
    rv=cvel[:,ids,3:]+j.cross(rw,rc-root)
    lower,upper,previous,antiwindup=map(j.asarray,(lower,upper,previous_command,antiwindup))
    q=qpos[:,hand_dof_address:hand_dof_address+28]; qd=qvel[:,hand_dof_address:hand_dof_address+28]
    normalized=j.concatenate((q[:,:3],q[:,3:6]/j.pi,2*(q[:,6:]-lower[6:])/(upper[6:]-lower[6:])-1),axis=-1)
    qdot=qd/j.maximum(antiwindup,1e-6)
    error=(previous-q)/j.maximum(antiwindup,1e-6)
    finite=j.all(j.isfinite(j.concatenate((q,qd,oo,oc,ov,ow,po,pc,pv,pw),axis=-1)),axis=-1)
    return V4Physical(normalized,qdot,error,oo,oq,oc,ov,ow,po,pq,pw,ro,rq,rc,rv,rw,j.asarray(object_bottom),finite)

def point_velocity(origin, com, v_com, omega, point):
    import jax.numpy as j
    del origin # named input preserves origin/COM distinction at call sites
    return v_com+j.cross(omega,point-com)
def anchor_delta_and_velocity(physical: V4Physical, anchor_hand, anchor_object):
    import jax.numpy as j
    ph=physical.region_origin+quat_rotate(physical.region_quat_xyzw,anchor_hand)
    po=physical.object_origin[:,None]+quat_rotate(physical.object_quat_xyzw[:,None],anchor_object)
    delta=quat_unrotate(physical.object_quat_xyzw[:,None],ph-po)
    vh=point_velocity(physical.region_origin,physical.region_com,physical.region_v_com,physical.region_w,ph)
    vo=point_velocity(physical.object_origin[:,None],physical.object_com[:,None],physical.object_v_com[:,None],physical.object_w[:,None],po)
    omega_o=quat_unrotate(physical.object_quat_xyzw,physical.object_w)
    velocity=quat_unrotate(physical.object_quat_xyzw[:,None],vh-vo)-j.cross(omega_o[:,None],delta)
    return ph,po,delta,velocity

def _gather(cache: ReferenceCacheV4, index, offset=0):
    import jax.numpy as j
    i=j.minimum(j.asarray(index)+offset,len(cache.q_feasible)-1)
    return i

def reduce_pyramidal_contacts_v4(*, nacon, nefc, geom, world, dimension, addresses, friction, frame, position, constraint_force, ngeom: int, hand_geom_ids, object_geom_ids, object_com):
    """Fail-closed same-transition reducer for pinned pyramidal/condim3 Warp buffers.

    Contact force rows apply negatively to geom1 and positively to geom2.
    The returned paired channel is therefore the canonical hand→object force,
    while object_all includes every contact incident on object collision geoms.
    """
    import jax.numpy as j
    geom,world,dimension,addresses,friction,frame,position,constraint_force=map(j.asarray,(geom,world,dimension,addresses,friction,frame,position,constraint_force))
    capacity=geom.shape[0]; batch=constraint_force.shape[0]; slots=j.arange(capacity); count=j.asarray(nacon).reshape(())
    if geom.shape!=(capacity,2) or addresses.shape!=(capacity,4) or position.shape!=(capacity,3) or frame.shape!=(capacity,3,3): raise ValueError("pinned Warp contact ABI mismatch")
    if count.dtype.kind not in 'iu': raise ValueError("contact count must be integer")
    live=slots<count; safe_world=j.clip(world,0,batch-1); safe_addr=j.clip(addresses,0,constraint_force.shape[1]-1); safe_geom=j.clip(geom,0,ngeom-1)
    nefc=j.broadcast_to(j.asarray(nefc).reshape(-1),(batch,)) if j.asarray(nefc).size==1 else j.asarray(nefc).reshape(batch)
    valid=(count>=0)&(count<capacity)&j.all((nefc>=0)&(nefc<constraint_force.shape[1]))&j.all(j.where(live,(world>=0)&(world<batch)&(dimension==3)&j.all((geom>=0)&(geom<ngeom),axis=-1)&j.all((addresses>=0)&(addresses<nefc[safe_world,None]),axis=-1),True))
    lam=constraint_force[safe_world[:,None],safe_addr]; local=j.stack((j.sum(lam,axis=-1),(lam[:,0]-lam[:,1])*friction[:,0],(lam[:,2]-lam[:,3])*friction[:,1]),axis=-1); force=j.einsum('ni,nij->nj',local,frame)
    valid=valid&j.all(j.where(live[:,None],j.isfinite(force),True))&j.all(j.where(live[:,None],j.isfinite(position),True))
    force=j.where(live[:,None],force,0.); hand=np.asarray(hand_geom_ids,dtype=np.int32); obj=np.asarray(object_geom_ids,dtype=np.int32)
    lookup=j.full((ngeom,),-1,dtype=j.int32).at[j.asarray(hand)].set(j.arange(16,dtype=j.int32)); a,b=lookup[safe_geom[:,0]],lookup[safe_geom[:,1]]
    a_obj=j.any(safe_geom[:,0,None]==j.asarray(obj)[None],axis=-1); b_obj=j.any(safe_geom[:,1,None]==j.asarray(obj)[None],axis=-1)
    paired=j.zeros((batch,16,3),force.dtype); torque=j.zeros_like(paired); counts=j.zeros((batch,16),j.int32); all_force=j.zeros((batch,3),force.dtype)
    # object is geom2 → +force; geom1 → -force
    all_force=all_force.at[safe_world].add(j.where(b_obj[:,None],force,0.)+j.where(a_obj[:,None],-force,0.))
    first=live&(a>=0)&b_obj; second=live&(b>=0)&a_obj
    paired=paired.at[safe_world,j.maximum(a,0)].add(force*first[:,None]); paired=paired.at[safe_world,j.maximum(b,0)].add(-force*second[:,None])
    lever=position-object_com[safe_world]
    torque=torque.at[safe_world,j.maximum(a,0)].add(j.cross(lever,force)*first[:,None]); torque=torque.at[safe_world,j.maximum(b,0)].add(-j.cross(lever,force)*second[:,None])
    counts=counts.at[safe_world,j.maximum(a,0)].add(first.astype(j.int32)); counts=counts.at[safe_world,j.maximum(b,0)].add(second.astype(j.int32))
    valid=valid&j.all(j.isfinite(all_force))&j.all(j.isfinite(paired))&j.all(j.isfinite(torque))
    return all_force,paired,torque,counts,valid

def build_raw_observation(physical: V4Physical, contact: V4Contact, cache: ReferenceCacheV4, index, previous_command):
    """Build exact seven raw blocks, unclipped, from one same-time physical state."""
    import jax.numpy as j
    b=physical.q.shape[0]; i=_gather(cache,index); t=max(len(cache.q_feasible)-1,1)
    C=lambda x:j.asarray(x)
    oq=C(cache.object_quat_xyzw)[i]; op=C(cache.object_origin)[i]; pq=C(cache.palm_quat_xyzw)[i]; pp=C(cache.palm_origin)[i]
    rq=C(cache.q_feasible)[i]; av=quat_unrotate(physical.object_quat_xyzw,physical.object_v_com); aw=quat_unrotate(physical.object_quat_xyzw,physical.object_w)
    palm_rel=quat_unrotate(physical.object_quat_xyzw,physical.palm_origin-physical.object_origin)
    palm_relq=quat_mul(quat_conj(physical.object_quat_xyzw),physical.palm_quat_xyzw)
    ref_rel=quat_unrotate(oq,pp-op); ref_relq=quat_mul(quat_conj(oq),pq)
    ref_com_v=C(cache.object_v_com)[i]; ref_w=C(cache.object_w)[i]
    ref_obj6=rot6(oq); ref_palm6=rot6(pq)
    actual = j.concatenate((physical.q, physical.qdot, physical.command_error, physical.object_origin, rot6(physical.object_quat_xyzw), av,aw,palm_rel,rot6(palm_relq),quat_unrotate(physical.object_quat_xyzw,physical.palm_w),physical.object_bottom[:,None]/.1,C(cache.signed_gap)[i].min(axis=-1)[:,None]/.1,quat_unrotate(physical.object_quat_xyzw,j.broadcast_to(j.asarray([0.,0.,-1.]),(b,3))),j.tanh(quat_unrotate(physical.object_quat_xyzw,contact.object_all_force)/5.)),axis=-1)
    # translation live errors are intentionally actual-object-frame, exactly as v4 schema specifies.
    ref = j.concatenate((rq[:,6:],ref_rel,rot6(ref_relq),C(cache.palm_v)[i],C(cache.palm_w)[i],op,ref_obj6,ref_com_v,ref_w,quat_unrotate(physical.object_quat_xyzw,physical.palm_origin-pp),rot6(quat_mul(quat_conj(pq),physical.palm_quat_xyzw)),physical.q[:,6:]-rq[:,6:],quat_unrotate(physical.object_quat_xyzw,physical.object_origin-op),rot6(quat_mul(quat_conj(oq),physical.object_quat_xyzw)),quat_unrotate(physical.object_quat_xyzw,physical.object_v_com-ref_com_v),quat_unrotate(physical.object_quat_xyzw,physical.object_w-ref_w),palm_rel-ref_rel,rot6(quat_mul(quat_conj(ref_relq),palm_relq)),j.stack((j.asarray(index)/t,1-j.asarray(index)/t),axis=-1)),axis=-1)
    futures=[]
    for horizon in (6,12,24):
        fi=_gather(cache,index,horizon); fq=C(cache.q_feasible)[fi]; fop=C(cache.object_origin)[fi]; foq=C(cache.object_quat_xyzw)[fi]; fpp=C(cache.palm_origin)[fi]; fpq=C(cache.palm_quat_xyzw)[fi]
        frel=quat_unrotate(foq,fpp-fop); frelq=quat_mul(quat_conj(foq),fpq)
        valid=(j.asarray(index)+horizon <= len(cache.q_feasible)-1).astype(j.float32)
        futures.append(j.concatenate((fq[:,6:]-rq[:,6:],frel-ref_rel,rot6(quat_mul(quat_conj(ref_relq),frelq)),quat_unrotate(oq,fop-op),rot6(quat_mul(quat_conj(oq),foq)),valid[:,None]),axis=-1))
    _,_,delta,velocity=anchor_delta_and_velocity(physical,C(cache.region_anchor_hand)[i],C(cache.region_anchor_object)[i])
    error=delta-C(cache.delta_ref)[i]
    geom=j.concatenate((C(cache.signed_gap)[i][...,None]/.1,C(cache.proximity)[i][...,None],C(cache.confidence)[i][...,None],C(cache.valid)[i][...,None],C(cache.region_anchor_hand)[i]/.1,C(cache.region_anchor_object)[i]/.1,C(cache.delta_ref)[i]/.01,error/.01,j.tanh(quat_unrotate(physical.object_quat_xyzw[:,None],contact.paired_force_on_object)/5.),velocity/.1),axis=-1).reshape(b,352)
    action=j.eye(50,dtype=physical.q.dtype)[cache.action_id-1][None].repeat(b,axis=0)
    raw=j.concatenate((actual,ref,j.concatenate(futures,axis=-1),geom,C(cache.points_object_local).reshape(1,192).repeat(b,axis=0)/.1,action,C(cache.object_geometry)[None].repeat(b,axis=0)),axis=-1)
    if raw.shape[-1] != RAW_OBSERVATION_DIM: raise AssertionError(f"v4 raw ABI {raw.shape[-1]} != 957")
    return raw

def pointnet_encode(raw_points, *, weights):
    import jax.numpy as j
    # deterministic PointNet: weights are an explicit checkpoint/cache contract
    w1,b1,w2,b2=weights
    x=j.maximum(raw_points@w1+b1,0.); return j.max(j.maximum(x@w2+b2,0.),axis=1)
def encode_observation(raw, embedding):
    import jax.numpy as j
    rs=raw_observation_slices(); es=encoded_observation_slices()
    if raw.shape[-1]!=957 or embedding.shape != (raw.shape[0],64): raise ValueError("v4 raw/PointNet shapes are fixed")
    return j.concatenate((raw[:,rs['autonomous_actual']],raw[:,rs['autonomous_reference']],raw[:,rs['autonomous_future']],raw[:,rs['autonomous_geometry']],embedding,raw[:,rs['action_types']],raw[:,rs['object_geometry']]),axis=-1)

class V4Reward(NamedTuple):
    total: Any; object_position: Any; object_rotation: Any; object_velocity: Any; hand_relative: Any; fingers: Any; geometry: Any; action: Any; survival: Any; severe: Any; done: Any; reason: Any; valid: Any

def compute_reward(physical: V4Physical, contact: V4Contact, cache: ReferenceCacheV4, index, executed_action):
    import jax.numpy as j
    i=_gather(cache,index); C=lambda x:j.asarray(x); target_p=C(cache.object_origin)[i]; target_q=C(cache.object_quat_xyzw)[i]
    dp=physical.object_origin-target_p
    pos=.2*j.exp(-40*j.abs(dp[:,0]))+.2*j.exp(-40*j.abs(dp[:,1]))+.8*j.exp(-40*j.abs(dp[:,2]))
    deg=shortest_angle(physical.object_quat_xyzw,target_q)*180/j.pi; rotvalue=j.where(deg<=20,1-.00125*deg**2,j.where(deg<=90,-.00003175*(deg-20)**2-.019206*(deg-20)+.5,-1.)); rot=.4*rotvalue-.1
    vel=.1*j.exp(-j.sum(((physical.object_v_com-C(cache.object_v_com)[i])/.25)**2,axis=-1)-j.sum(((physical.object_w-C(cache.object_w)[i])/2.)**2,axis=-1))
    rel=quat_unrotate(physical.object_quat_xyzw,physical.palm_origin-physical.object_origin); rrel=quat_unrotate(target_q,C(cache.palm_origin)[i]-target_p); relq=quat_mul(quat_conj(physical.object_quat_xyzw),physical.palm_quat_xyzw); rrelq=quat_mul(quat_conj(target_q),C(cache.palm_quat_xyzw)[i]); hand=.5*j.exp(-j.sum(((rel-rrel)/.04)**2,axis=-1)-(shortest_angle(relq,rrelq)/.35)**2)
    fingers=.2*j.exp(-j.mean((physical.q[:,6:]-C(cache.q_feasible)[i][:,6:])**2,axis=-1)/.35**2)
    _,_,delta,_=anchor_delta_and_velocity(physical,C(cache.region_anchor_hand)[i],C(cache.region_anchor_object)[i]); e=delta-C(cache.delta_ref)[i]; weight=C(cache.proximity)[i]*C(cache.confidence)[i]*C(cache.valid)[i]; geometry=.2*j.sum(weight*j.exp(-j.sum(e*e,axis=-1)/.01**2),axis=-1)/j.maximum(j.sum(weight,axis=-1),1.)
    action=-.002*j.mean(j.clip(executed_action,-1,1)**2,axis=-1); survival=j.full_like(pos,.001)
    fallen=physical.object_bottom < -0.05; deviation=j.linalg.norm(dp,axis=-1)>.10; finite=j.isfinite(pos+rot+vel+hand+fingers+geometry+action)&physical.valid&contact.valid
    severe=j.where(finite,j.where(fallen|deviation,-25.,0.),-25.); reason=(j.asarray(index)>=len(cache.q_feasible)-1).astype(j.int32)|j.where(deviation,2,0)|j.where(fallen,4,0)|j.where(~finite,8,0)
    total=j.where(finite,pos+rot+vel+hand+fingers+geometry+action+survival+severe,-25.)
    done=(reason!=0)
    return V4Reward(total,pos,rot,vel,hand,fingers,geometry,action,survival,severe,done,reason,finite)
