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
# Exact resolved MANOHandAutonomousV1.yaml arrays. Keeping the source names
# together prevents qdot, servo error and command envelope drift.
DOF_RATE = np.asarray([.8,.8,1.2,3.,4.,3.,2.,3.,2.,4.,2.,4.,2.,4.,4.,4.,2.,4.,4.,4.,2.,4.,4.,4.,2.,4.,4.,4.], dtype=np.float32)
ANTIWINDUP_ERROR = np.asarray([.04,.02,.07,.3,.3,.3,.2,.35,.2,.35,.2,.35,.15,.35,.35,.35,.15,.35,.35,.35,.15,.35,.35,.35,.15,.35,.35,.35], dtype=np.float32)
WORLD_POSITION=1.; RELATIVE_POSITION=.1; JOINT_ERROR=.3; LINEAR_VELOCITY=1.; ANGULAR_VELOCITY=3.; CONTACT_DISTANCE=.01; CONTACT_VELOCITY=.1; FORCE_SCALE=5.

class V4Physical(NamedTuple):
    # Raw SI q is the only state allowed in control and reward.  q_normalized
    # exists solely for the observation ABI.
    q_raw: Any; q_normalized: Any; qdot: Any; command_error: Any
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
    q_feasible: np.ndarray; q_raw: np.ndarray; q_lower: np.ndarray; q_upper: np.ndarray
    object_origin: np.ndarray; object_quat_xyzw: np.ndarray; object_com_local: np.ndarray
    palm_origin: np.ndarray; palm_quat_xyzw: np.ndarray
    object_v_com: np.ndarray; object_w: np.ndarray; palm_v: np.ndarray; palm_w: np.ndarray
    region_anchor_hand: np.ndarray; region_anchor_object: np.ndarray; delta_ref: np.ndarray
    signed_gap: np.ndarray; proximity: np.ndarray; confidence: np.ndarray; valid: np.ndarray
    reference_bottom: np.ndarray
    table_height: float
    points_object_local: np.ndarray; object_geometry: np.ndarray; action_id: int
    support_shift: np.ndarray; content_hash: str
    def __post_init__(self):
        t = len(self.q_feasible)
        expected = {"q_feasible":(t,28), "q_raw":(t,28), "q_lower":(28,), "q_upper":(28,), "object_origin":(t,3), "object_quat_xyzw":(t,4), "palm_origin":(t,3), "palm_quat_xyzw":(t,4), "object_v_com":(t,3), "object_w":(t,3), "palm_v":(t,3), "palm_w":(t,3), "region_anchor_hand":(t,16,3), "region_anchor_object":(t,16,3), "delta_ref":(t,16,3), "signed_gap":(t,16), "proximity":(t,16), "confidence":(t,16), "valid":(t,16), "reference_bottom":(t,), "points_object_local":(64,3), "object_geometry":(12,), "support_shift":(3,)}
        for name, shape in expected.items():
            value=np.asarray(getattr(self,name));
            if value.shape != shape or not np.all(np.isfinite(value)): raise ValueError(f"v4 cache {name} must be finite {shape}")
        if not np.isfinite(self.table_height): raise ValueError("table height must be finite")
        if not 1 <= self.action_id <= 50: raise ValueError("action_id must be in [1,50]")

def _np_quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q=np.asarray(q,dtype=np.float64); q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    u,w=q[...,:3],q[...,3:]
    return v*(2*w*w-1)+2*w*np.cross(u,v)+2*u*np.sum(u*v,axis=-1,keepdims=True)

def _mesh_body_geometry(model, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one collision mesh in its owning body frame, applying geom pose once."""
    mesh_id=int(model.geom_dataid[geom_id])
    if mesh_id < 0: raise ValueError(f"collision geom {geom_id} has no mesh data")
    va,nv=int(model.mesh_vertadr[mesh_id]),int(model.mesh_vertnum[mesh_id])
    fa,nf=int(model.mesh_faceadr[mesh_id]),int(model.mesh_facenum[mesh_id])
    verts=np.asarray(model.mesh_vert[va:va+nv],np.float64).copy()
    faces=np.asarray(model.mesh_face[fa:fa+nf],np.int32).copy()
    if verts.shape != (nv,3) or faces.shape != (nf,3) or nf == 0: raise ValueError("invalid compiled collision mesh slice")
    geom_q=np.asarray(model.geom_quat[geom_id],np.float64)[[1,2,3,0]]
    return _np_quat_rotate(np.broadcast_to(geom_q,(nv,4)),verts)+np.asarray(model.geom_pos[geom_id],np.float64),faces

def _deterministic_surface_samples(vertices: np.ndarray, faces: np.ndarray, count: int) -> np.ndarray:
    """Area-spread low-discrepancy samples with no repeated corners."""
    tri=vertices[faces]; e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    areas=.5*np.linalg.norm(np.cross(e1,e2),axis=1)
    if count < 1 or not np.all(np.isfinite(areas)) or np.any(areas <= 0): raise ValueError("mesh triangles must have positive finite area")
    cdf=np.cumsum(areas); total=cdf[-1]
    # midpoint stratification selects faces by area; Van der Corput changes
    # barycentric position within each stratum deterministically.
    u=(np.arange(count,dtype=np.float64)+.5)/count
    idx=np.searchsorted(cdf,u*total,side='right')
    vdc=np.zeros(count); x=np.arange(count,dtype=np.uint64)+1; denom=2.
    while np.any(x):
        vdc+=(x&1)/denom; x>>=1; denom*=2.
    r=np.sqrt(u); return tri[idx,0]+(1-r)[:,None]*(tri[idx,1]-tri[idx,0])+ (r*(1-vdc))[:,None]*(tri[idx,2]-tri[idx,0])

def _closest_point_triangle_jax(points, triangles):
    """Exact Ericson closest-point regions for points×triangles, JAX-native."""
    import jax.numpy as j
    p=points[:,None,:]; a=triangles[None,:,0]; b=triangles[None,:,1]; c=triangles[None,:,2]
    ab,ac= b-a,c-a; ap=p-a; d1=j.sum(ab*ap,-1); d2=j.sum(ac*ap,-1)
    bp=p-b; d3=j.sum(ab*bp,-1); d4=j.sum(ac*bp,-1); cp=p-c; d5=j.sum(ab*cp,-1); d6=j.sum(ac*cp,-1)
    va=d3*d6-d5*d4; vb=d5*d2-d1*d6; vc=d1*d4-d3*d2
    face=a+ab*(vb/(va+vb+vc+1e-20))[...,None]+ac*(vc/(va+vb+vc+1e-20))[...,None]
    qab=a; qab=j.where(((d3>=0)&(d4<=d3))[...,None],b,qab)
    qab=j.where(((d6>=0)&(d5<=d6))[...,None],c,qab)
    edgeab=a+ab*(d1/(d1-d3+1e-20))[...,None]
    qab=j.where(((vc<=0)&(d1>=0)&(d3<=0))[...,None],edgeab,qab)
    edgeac=a+ac*(d2/(d2-d6+1e-20))[...,None]
    qab=j.where(((vb<=0)&(d2>=0)&(d6<=0))[...,None],edgeac,qab)
    bc=c-b; edgebc=b+bc*((d4-d3)/(d4-d3+d5-d6+1e-20))[...,None]
    qab=j.where(((va<=0)&((d4-d3)>=0)&((d5-d6)>=0))[...,None],edgebc,qab)
    vertex_or_edge=((d1<=0)&(d2<=0))|((d3>=0)&(d4<=d3))|((d6>=0)&(d5<=d6))|((vc<=0)&(d1>=0)&(d3<=0))|((vb<=0)&(d2>=0)&(d6<=0))|((va<=0)&((d4-d3)>=0)&((d5-d6)>=0))
    return j.where(vertex_or_edge[...,None],qab,face)

def signed_closest_convex_mesh(points, triangles):
    """JAX triangle closest points and convex halfspace sign; inside projects to surface."""
    import jax, jax.numpy as j
    triangles=j.asarray(triangles); points=j.asarray(points)
    def kernel(p, tri):
        closest=_closest_point_triangle_jax(p,tri); d2=j.sum((p[:,None]-closest)**2,axis=-1); which=j.argmin(d2,axis=1)
        cp=closest[j.arange(len(p)),which]; distance=j.sqrt(j.maximum(d2[j.arange(len(p)),which],0.))
        center=j.mean(tri.reshape(-1,3),axis=0); normal=j.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); normal=j.where((j.sum(normal*(center-tri[:,0]),axis=-1)>0)[:,None],-normal,normal)
        inside=j.all(j.sum((p[:,None]-tri[:,0])*normal[None],axis=-1)<=1e-8,axis=-1)
        return cp,j.where(inside,-distance,distance),inside
    return jax.jit(kernel)(points,triangles)

def _filtered_derivative(values: np.ndarray, dt: float) -> np.ndarray:
    from scipy.signal import savgol_filter
    if dt <= 0 or not np.isfinite(dt): raise ValueError("reference control timestep must be finite positive")
    derivative=np.gradient(values,dt,axis=0)
    return savgol_filter(derivative,7,2,axis=0,mode='interp') if len(values)>=7 else derivative

def _filtered_angular_velocity(quaternions: np.ndarray, dt: float) -> np.ndarray:
    q=np.asarray(quaternions,dtype=np.float64).copy()
    for i in range(1,len(q)):
        if np.dot(q[i-1],q[i]) < 0: q[i]*=-1
    dq=np.empty_like(q); dq[0]=(q[1]-q[0])/dt; dq[-1]=(q[-1]-q[-2])/dt; dq[1:-1]=(q[2:]-q[:-2])/(2*dt)
    # qdot = .5 omega_quat * q -> omega = 2 qdot * conj(q), vector part
    conj=q.copy(); conj[:,:3]*=-1
    omega=2*np.asarray(quat_mul(np.asarray(dq),np.asarray(conj)))[:,:3]
    return _filtered_derivative(np.cumsum(omega,axis=0)*dt,dt)

def compile_reference_cache_v4(trajectory, *, device: str = "cpu", object_type: str = "cube2", table_height: float = -0.001) -> ReferenceCacheV4:
    """Compile the cache through Warp FK and exact compiled collision meshes only."""
    import jax, jax.numpy as j
    from mujoco import mjx
    from sim.manorl.assets import compile_model_metadata_only, object_collision_vertices
    from sim.manorl.autonomy import align_reference_trajectory
    from sim.manorl.environment import MjxWarpPhysicalProducer
    from sim.manorl.observations import geometry_encoding
    mujoco,model=compile_model_metadata_only(object_type=object_type,hand_side="right",physics_timestep=1/480)
    producer=MjxWarpPhysicalProducer(mujoco,model,object_type=object_type,hand_sides=("right",))
    mesh_type=int(mujoco.mjtGeom.mjGEOM_MESH)
    hand_geoms=np.asarray(producer.keypoint_geom_ids,np.int32)
    if not np.all(np.asarray(model.geom_type)[hand_geoms]==mesh_type): raise ValueError("all v4 hand regions must be compiled mesh collision geoms")
    object_geom=next(iter(producer.object_geom_ids))
    if int(model.geom_type[object_geom]) != mesh_type: raise ValueError("cube2 collision geom must be a mesh")
    # Source collision triangles are already body-local; deliberately do not
    # apply object geom_pos/geom_quat a second time.
    object_triangles=np.asarray(object_collision_vertices(object_type),np.float64).reshape(-1,3,3)
    hand_samples=[]
    for gid in hand_geoms:
        vertices,faces=_mesh_body_geometry(model,int(gid)); hand_samples.append(_deterministic_surface_samples(vertices,faces,128))
    hand_samples=np.stack(hand_samples)
    # Collision triangles are body-local source triangles.  Flattening with
    # sequential faces preserves that exact mesh without applying geom pose.
    object_points=_deterministic_surface_samples(object_triangles.reshape(-1,3),np.arange(object_triangles.size//3,dtype=np.int32).reshape(-1,3),64)
    dev=jax.devices(device)[0]; m=mjx.put_model(model,device=dev,impl="warp")
    raw=np.asarray(trajectory.q_ref,dtype=np.float64).copy(); q_aligned,obj,shift=align_reference_trajectory(trajectory,object_triangles.reshape(-1,3),table_height)
    lower=np.asarray(model.jnt_range[:28,0]); upper=np.asarray(model.jnt_range[:28,1]); feasible=np.clip(q_aligned,lower,upper)
    raw[:, :3]+=shift
    T=len(feasible); base=mjx.make_data(model,device=dev,impl="warp"); qpos=np.broadcast_to(np.asarray(base.qpos),(T,model.nq)).copy(); qpos[:,:28]=feasible
    qpos[:,producer.object_qpos_address:producer.object_qpos_address+3]=obj; qpos[:,producer.object_qpos_address+3:producer.object_qpos_address+7]=np.asarray(trajectory.object_quat_xyzw)[:,(3,0,1,2)]
    # Sequential/chunked offline N=1 FK keeps reference compilation distinct
    # from runtime physics batching.
    data=jax.vmap(lambda x:mjx.forward(m,base.replace(qpos=x)))(j.asarray(qpos))
    xpos=np.asarray(data.xpos); xquat=np.asarray(data.xquat)[...,(1,2,3,0)]; body=np.asarray(producer.keypoint_body_ids)
    hand_origin=xpos[:,body]; hand_q=xquat[:,body]; object_origin=xpos[:,producer.object_body_id]; object_q=xquat[:,producer.object_body_id]
    hand_world=hand_origin[:,:,None]+_np_quat_rotate(hand_q[:,:,None],hand_samples[None])
    object_local=np.asarray(quat_unrotate(j.asarray(object_q[:,None,None]),j.asarray(hand_world-object_origin[:,None,None])))
    points=object_local.reshape(-1,3); closest,signed,_=signed_closest_convex_mesh(points,object_triangles); closest=np.asarray(closest).reshape(T,16,128,3); signed=np.asarray(signed).reshape(T,16,128)
    chosen=np.argmin(np.abs(signed),axis=-1); rows=np.arange(T)[:,None]; region=np.arange(16)[None,:]
    ah=hand_samples[None].repeat(T,axis=0)[rows,region,chosen]
    ao=closest[rows,region,chosen]
    hand_anchor_world=hand_origin+_np_quat_rotate(hand_q,ah)
    object_anchor_world=object_origin[:,None]+_np_quat_rotate(object_q[:,None],ao)
    delta=np.asarray(quat_unrotate(j.asarray(object_q[:,None]),j.asarray(hand_anchor_world-object_anchor_world)))
    gap=signed[rows,region,chosen]; depth=np.maximum(-np.min(signed,axis=-1),0); fraction=np.mean(signed<-.001,axis=-1); proximity=np.exp(-(np.maximum(gap,0)/.01)**2); confidence=np.exp(-(np.maximum(depth-.001,0)/.003)**2)*(1-fraction); valid=(depth<=.01).astype(np.float64)
    dt=1./float(trajectory.control_fps or trajectory.reference_fps or 120)
    com_local=np.asarray(model.body_ipos[producer.object_body_id]); com=object_origin+_np_quat_rotate(object_q,np.broadcast_to(com_local,(T,3))); ov=_filtered_derivative(com,dt); ow=_filtered_angular_velocity(object_q,dt); palm=hand_origin[:,0]; pv=_filtered_derivative(palm,dt); pw=_filtered_angular_velocity(hand_q[:,0],dt)
    object_bottom=np.min(object_origin[:,None,2]+_np_quat_rotate(object_q[:,None],object_triangles.reshape(-1,3))[:,:,2],axis=1)
    dimensions=np.ptp(object_triangles.reshape(-1,3),axis=0); geometry=geometry_encoding(object_name="cube2",geometry_type="box",dimensions=dimensions)
    values=(feasible,raw,lower,upper,obj,object_q,com_local,palm,hand_q[:,0],ov,ow,pv,pw,ah,ao,delta,gap,proximity,confidence,valid,object_bottom,np.asarray([table_height]),object_points,geometry,np.asarray(shift),np.asarray([dt]),np.asarray(model.geom_pos[hand_geoms]),np.asarray(model.geom_quat[hand_geoms]))
    digest=cache_hash(*[np.asarray(x) for x in values],kernel_version="warp-mesh-cache-v4.1")
    result=ReferenceCacheV4(feasible,raw,lower,upper,obj,object_q,com_local,palm,hand_q[:,0],ov,ow,pv,pw,ah,ao,delta,gap,proximity,confidence,valid,object_bottom,float(table_height),object_points,geometry,int(trajectory.identity.identity.split('_')[1]),np.asarray(shift),digest)
    for name in result.__dataclass_fields__:
        value=getattr(result,name)
        if isinstance(value,np.ndarray): value.setflags(write=False)
    return result

def cache_hash(*values: np.ndarray, kernel_version: str = "warp-mesh-cache-v4.1") -> str:
    """Hash cache version, parameters, shapes/dtypes and static asset geometry."""
    h=hashlib.sha256(kernel_version.encode())
    for value in values:
        array=np.ascontiguousarray(value)
        h.update(str((array.shape,array.dtype.str)).encode())
        h.update(array.tobytes())
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
def extract_v4_physical(*, qpos, qvel, xpos, xquat, xipos, subtree_com, cvel, body_rootid, hand_dof_address: int, object_body_id: int, palm_body_id: int, region_body_ids, object_bottom, lower, upper, previous_command, dof_rate=DOF_RATE, antiwindup=ANTIWINDUP_ERROR):
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
    lower,upper,previous,dof_rate,antiwindup=map(j.asarray,(lower,upper,previous_command,dof_rate,antiwindup))
    q_raw=qpos[:,hand_dof_address:hand_dof_address+28]; qd=qvel[:,hand_dof_address:hand_dof_address+28]
    normalized=j.concatenate((q_raw[:,:3]/WORLD_POSITION,q_raw[:,3:6]/j.pi,2*(q_raw[:,6:]-lower[6:])/(upper[6:]-lower[6:])-1),axis=-1)
    qdot=qd/j.maximum(dof_rate,1e-6)
    error=(previous-q_raw)/j.maximum(antiwindup,1e-6)
    all_state=j.concatenate((q_raw,qd,oo,oc,ov,ow,po,pc,pv,pw,ro.reshape(q_raw.shape[0],-1),rc.reshape(q_raw.shape[0],-1),rv.reshape(q_raw.shape[0],-1),rw.reshape(q_raw.shape[0],-1)),axis=-1)
    finite=j.all(j.isfinite(all_state),axis=-1)&j.all(j.isfinite(oq),axis=-1)&j.all(j.isfinite(rq),axis=(-1,-2))
    return V4Physical(q_raw,normalized,qdot,error,oo,oq,oc,ov,ow,po,pq,pw,ro,rq,rc,rv,rw,j.asarray(object_bottom),finite)

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
    b=physical.q_raw.shape[0]; i=_gather(cache,index); t=max(len(cache.q_feasible)-1,1)
    C=lambda x:j.asarray(x)
    oq=C(cache.object_quat_xyzw)[i]; op=C(cache.object_origin)[i]; pq=C(cache.palm_quat_xyzw)[i]; pp=C(cache.palm_origin)[i]
    rq=C(cache.q_feasible)[i]; rq_normalized=j.concatenate((rq[:,:3]/WORLD_POSITION,rq[:,3:6]/j.pi,2*(rq[:,6:]-C(cache.q_lower)[6:])/(C(cache.q_upper)[6:]-C(cache.q_lower)[6:])-1),axis=-1); av=quat_unrotate(physical.object_quat_xyzw,physical.object_v_com)/LINEAR_VELOCITY; aw=quat_unrotate(physical.object_quat_xyzw,physical.object_w)/ANGULAR_VELOCITY
    palm_rel=quat_unrotate(physical.object_quat_xyzw,physical.palm_origin-physical.object_origin)
    palm_relq=quat_mul(quat_conj(physical.object_quat_xyzw),physical.palm_quat_xyzw)
    ref_rel=quat_unrotate(oq,pp-op); ref_relq=quat_mul(quat_conj(oq),pq)
    ref_com_v=C(cache.object_v_com)[i]; ref_w=C(cache.object_w)[i]
    ref_obj6=rot6(oq); ref_palm6=rot6(pq)
    actual = j.concatenate((physical.q_normalized, physical.qdot, physical.command_error, physical.object_origin/WORLD_POSITION, rot6(physical.object_quat_xyzw), av,aw,palm_rel/RELATIVE_POSITION,rot6(palm_relq),quat_unrotate(physical.object_quat_xyzw,physical.palm_w)/ANGULAR_VELOCITY,(physical.object_bottom-cache.table_height)[:,None]/RELATIVE_POSITION,(C(cache.reference_bottom)[i]-cache.table_height)[:,None]/RELATIVE_POSITION,quat_unrotate(physical.object_quat_xyzw,j.broadcast_to(j.asarray([0.,0.,-1.]),(b,3))),j.tanh(quat_unrotate(physical.object_quat_xyzw,contact.object_all_force)/FORCE_SCALE)),axis=-1)
    # translation live errors are intentionally actual-object-frame, exactly as v4 schema specifies.
    ref = j.concatenate((rq_normalized[:,6:],ref_rel/RELATIVE_POSITION,rot6(ref_relq),quat_unrotate(oq,C(cache.palm_v)[i])/LINEAR_VELOCITY,quat_unrotate(oq,C(cache.palm_w)[i])/ANGULAR_VELOCITY,op/WORLD_POSITION,ref_obj6,ref_com_v/LINEAR_VELOCITY,ref_w/ANGULAR_VELOCITY,quat_unrotate(physical.object_quat_xyzw,physical.palm_origin-pp)/RELATIVE_POSITION,rot6(quat_mul(quat_conj(pq),physical.palm_quat_xyzw)),(physical.q_raw[:,6:]-rq[:,6:])/JOINT_ERROR,quat_unrotate(physical.object_quat_xyzw,physical.object_origin-op)/RELATIVE_POSITION,rot6(quat_mul(quat_conj(oq),physical.object_quat_xyzw)),quat_unrotate(physical.object_quat_xyzw,physical.object_v_com-ref_com_v)/LINEAR_VELOCITY,quat_unrotate(physical.object_quat_xyzw,physical.object_w-ref_w)/ANGULAR_VELOCITY,(palm_rel-ref_rel)/RELATIVE_POSITION,rot6(quat_mul(quat_conj(ref_relq),palm_relq)),j.stack((j.asarray(index)/t,1-j.asarray(index)/t),axis=-1)),axis=-1)
    futures=[]
    for horizon in (6,12,24):
        fi=_gather(cache,index,horizon); fq=C(cache.q_feasible)[fi]; fop=C(cache.object_origin)[fi]; foq=C(cache.object_quat_xyzw)[fi]; fpp=C(cache.palm_origin)[fi]; fpq=C(cache.palm_quat_xyzw)[fi]
        frel=quat_unrotate(foq,fpp-fop); frelq=quat_mul(quat_conj(foq),fpq)
        valid=(j.asarray(index)+horizon <= len(cache.q_feasible)-1).astype(j.float32)
        futures.append(j.concatenate(((fq[:,6:]-rq[:,6:])/JOINT_ERROR,quat_unrotate(oq,(fpp-fop)-(pp-op))/RELATIVE_POSITION,rot6(quat_mul(quat_conj(ref_relq),frelq)),quat_unrotate(oq,fop-op)/RELATIVE_POSITION,rot6(quat_mul(quat_conj(oq),foq)),valid[:,None]),axis=-1))
    _,_,delta,velocity=anchor_delta_and_velocity(physical,C(cache.region_anchor_hand)[i],C(cache.region_anchor_object)[i])
    error=delta-C(cache.delta_ref)[i]
    geom=j.concatenate((C(cache.signed_gap)[i][...,None]/CONTACT_DISTANCE,C(cache.proximity)[i][...,None],C(cache.confidence)[i][...,None],C(cache.valid)[i][...,None],C(cache.region_anchor_hand)[i]/RELATIVE_POSITION,C(cache.region_anchor_object)[i]/RELATIVE_POSITION,C(cache.delta_ref)[i]/CONTACT_DISTANCE,error/CONTACT_DISTANCE,j.tanh(quat_unrotate(physical.object_quat_xyzw[:,None],contact.paired_force_on_object)/FORCE_SCALE),velocity/CONTACT_VELOCITY),axis=-1).reshape(b,352)
    action=j.eye(50,dtype=physical.q_raw.dtype)[cache.action_id-1][None].repeat(b,axis=0)
    raw=j.concatenate((actual,ref,j.concatenate(futures,axis=-1),geom,C(cache.points_object_local).reshape(1,192).repeat(b,axis=0)/RELATIVE_POSITION,action,C(cache.object_geometry)[None].repeat(b,axis=0)),axis=-1)
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
    fingers=.2*j.exp(-j.mean((physical.q_raw[:,6:]-C(cache.q_feasible)[i][:,6:])**2,axis=-1)/.35**2)
    _,_,delta,_=anchor_delta_and_velocity(physical,C(cache.region_anchor_hand)[i],C(cache.region_anchor_object)[i]); e=delta-C(cache.delta_ref)[i]; weight=C(cache.proximity)[i]*C(cache.confidence)[i]*C(cache.valid)[i]; geometry=.2*j.sum(weight*j.exp(-j.sum(e*e,axis=-1)/.01**2),axis=-1)/j.maximum(j.sum(weight,axis=-1),1.)
    action=-.002*j.mean(j.clip(executed_action,-1,1)**2,axis=-1); survival=j.full_like(pos,.001)
    fallen=physical.object_bottom < cache.table_height-.05; deviation=j.linalg.norm(dp,axis=-1)>.10; finite=j.isfinite(pos+rot+vel+hand+fingers+geometry+action)&physical.valid&contact.valid
    severe=j.where(finite,j.where(fallen|deviation,-25.,0.),-25.); reason=(j.asarray(index)>=len(cache.q_feasible)-1).astype(j.int32)|j.where(deviation,2,0)|j.where(fallen,4,0)|j.where(~finite,8,0)
    total=j.where(finite,pos+rot+vel+hand+fingers+geometry+action+survival+severe,-25.)
    done=(reason!=0)
    return V4Reward(total,pos,rot,vel,hand,fingers,geometry,action,survival,severe,done,reason,finite)
