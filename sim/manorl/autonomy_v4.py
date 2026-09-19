"""Warp-only cube2 contact-conditioned autonomy v4 primitives.

This module owns the observation and reward meaning.  Inputs are device arrays
from one forward-consistent MJX state; it has no native MuJoCo data calls.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import Any, NamedTuple
import numpy as np
from sim.manorl.autonomy_contracts import (
    RAW_OBSERVATION_DIM,
    ENCODED_OBSERVATION_DIM,
    RAW_OBSERVATION_DIM_V5,
    RAW_OBSERVATION_DIM_V6,
    encoded_observation_slices,
    raw_observation_slices,
    raw_observation_slices_v5,
)

REGIONS = 16
POINTS = 64
GEOMETRY_PER_REGION = 22
# Exact resolved MANOHandAutonomousV1.yaml arrays. Keeping the source names
# together prevents qdot, servo error and command envelope drift.
DOF_RATE = np.asarray([.8,.8,1.2,3.,4.,3.,2.,3.,2.,4.,2.,4.,2.,4.,4.,4.,2.,4.,4.,4.,2.,4.,4.,4.,2.,4.,4.,4.], dtype=np.float32)
ANTIWINDUP_ERROR = np.asarray([.04,.02,.07,.3,.3,.3,.2,.35,.2,.35,.2,.35,.15,.35,.35,.35,.15,.35,.35,.35,.15,.35,.35,.35,.15,.35,.35,.35], dtype=np.float32)
WORLD_POSITION=1.; RELATIVE_POSITION=.1; JOINT_ERROR=.3; LINEAR_VELOCITY=1.; ANGULAR_VELOCITY=3.; CONTACT_DISTANCE=.01; CONTACT_VELOCITY=.1; FORCE_SCALE=5.

# Reward parameter set v4-motion-gated. The physical formulas are unchanged; these
# constants are the single source for both the runtime and telemetry metadata.
REWARD_MOTION_GATE_LOW=0.01       # reference effective speed where object tracking keeps 1%
REWARD_MOTION_GATE_HIGH=0.10      # full weight above this reference speed (m/s)
REWARD_MOTION_STATIC_WEIGHT=0.01  # object terms preserve 1% on a still reference
REWARD_MOTION_RADIUS=0.04330042412758056  # cube2 max collision-vertex radius about COM (m)
REWARD_HAND_RELATIVE_COEF=0.125   # palm-object relative pose
REWARD_GEOMETRY_COEF=1.2          # contact anchor correspondence
REWARD_SEVERE_PENALTY=75.0        # deviation / fall / non-finite, applied once

def motion_gate_weight(v_eff):
    """Smooth 1%->100% object-tracking weight from REFERENCE effective speed only."""
    import jax.numpy as j
    u=j.clip((j.asarray(v_eff)-REWARD_MOTION_GATE_LOW)/(REWARD_MOTION_GATE_HIGH-REWARD_MOTION_GATE_LOW),0.,1.)
    s=u*u*(3.-2.*u)
    return REWARD_MOTION_STATIC_WEIGHT+(1.-REWARD_MOTION_STATIC_WEIGHT)*s

def reward_parameters(object_radius: float = REWARD_MOTION_RADIUS) -> dict[str, object]:
    """JSON-safe reward provenance for training and frozen-evaluation artifacts."""
    return {"id": "manorl.autonomy.reward.v4.reference-speed-gated.contact-priority.v1",
            "object_tracking": {"motion_source": "reference_object_com_linear_and_angular_velocity",
                                "effective_speed": "sqrt(norm(v_ref_com)^2 + (object_radius_m * norm(omega_ref))^2)",
                                "object_radius_m": float(object_radius),
                                "smoothstep_low_m_per_s": REWARD_MOTION_GATE_LOW,
                                "smoothstep_high_m_per_s": REWARD_MOTION_GATE_HIGH,
                                "static_weight": REWARD_MOTION_STATIC_WEIGHT,
                                "moving_weight": 1.0,
                                "terms": ["object_position", "object_rotation", "object_velocity"]},
            "coefficients": {"hand_relative": REWARD_HAND_RELATIVE_COEF,
                             "fingers": 0.2, "geometry": REWARD_GEOMETRY_COEF,
                             "action": -0.002, "survival": 0.001,
                             "severe_failure": -REWARD_SEVERE_PENALTY},
            "severe_conditions": ["object_position_deviation_gt_0.10_m", "object_bottom_below_table_minus_0.05_m", "nonfinite_physics_or_contact"]}

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
    object_all_torque: Any = None

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
    control_timestep: float; duration: float
    points_object_local: np.ndarray; object_geometry: np.ndarray; action_id: int
    support_shift: np.ndarray; content_hash: str
    object_radius: float = REWARD_MOTION_RADIUS
    # v5 point-cloud state: static hand surface template (16 regions x 16 points,
    # body-local per region) and the per-frame reference hand cloud expressed in
    # the object's local frame. These feed the observation encoder only.
    hand_cloud_template: np.ndarray | None = None
    hand_cloud_reference: np.ndarray | None = None
    object_mass: float = 1.0
    gravity_world: np.ndarray | None = None
    def __post_init__(self):
        t = len(self.q_feasible)
        expected = {"q_feasible":(t,28), "q_raw":(t,28), "q_lower":(28,), "q_upper":(28,), "object_origin":(t,3), "object_quat_xyzw":(t,4), "palm_origin":(t,3), "palm_quat_xyzw":(t,4), "object_v_com":(t,3), "object_w":(t,3), "palm_v":(t,3), "palm_w":(t,3), "region_anchor_hand":(t,16,3), "region_anchor_object":(t,16,3), "delta_ref":(t,16,3), "signed_gap":(t,16), "proximity":(t,16), "confidence":(t,16), "valid":(t,16), "reference_bottom":(t,), "points_object_local":(64,3), "object_geometry":(12,), "support_shift":(3,)}
        for name, shape in expected.items():
            value=np.asarray(getattr(self,name));
            if value.shape != shape or not np.all(np.isfinite(value)): raise ValueError(f"v4 cache {name} must be finite {shape}")
        if not np.isfinite(self.table_height) or not np.isfinite(self.control_timestep) or self.control_timestep <= 0 or not np.isfinite(self.duration) or self.duration <= 0: raise ValueError("cache table/clock/duration must be finite")
        if not np.isfinite(self.object_radius) or self.object_radius <= 0: raise ValueError("cache object radius must be positive finite")
        if not 1 <= self.action_id <= 50: raise ValueError("action_id must be in [1,50]")
        if self.hand_cloud_template is not None:
            template=np.asarray(self.hand_cloud_template)
            if template.shape != (256,3) or not np.all(np.isfinite(template)): raise ValueError("v5 hand cloud template must be finite (256,3)")
        if self.hand_cloud_reference is not None:
            cloud=np.asarray(self.hand_cloud_reference)
            if cloud.shape != (len(self.q_feasible),256,3) or not np.all(np.isfinite(cloud)): raise ValueError("v5 hand cloud reference must be finite (T,256,3)")
        if not np.isfinite(self.object_mass) or self.object_mass <= 0:
            raise ValueError("object mass must be positive finite")
        if self.gravity_world is not None:
            gravity=np.asarray(self.gravity_world)
            if gravity.shape != (3,) or not np.all(np.isfinite(gravity)) or np.linalg.norm(gravity) <= 0:
                raise ValueError("gravity_world must be a finite nonzero 3-vector")

# Time-varying arrays are padded by their final row; gathers still clamp to
# each reference's own length. Shared geometry is validated, while action_id
# is retained per reference so one bank can represent multiple cube2 actions.
REFERENCE_TIME_FIELDS = (
    "q_feasible", "q_raw", "object_origin", "object_quat_xyzw", "palm_origin",
    "palm_quat_xyzw", "object_v_com", "object_w", "palm_v", "palm_w",
    "region_anchor_hand", "region_anchor_object", "delta_ref", "signed_gap",
    "proximity", "confidence", "valid", "reference_bottom", "hand_cloud_reference",
)

class ReferenceBankV4:
    """Device-resident [reference,time,...] cache for shared cube2 geometry."""
    def __init__(self, caches, *, device=None):
        import jax
        import jax.numpy as j
        caches = tuple(caches)
        if not caches:
            raise ValueError("reference bank must not be empty")
        shared = ("object_com_local", "object_radius", "table_height", "control_timestep",
                  "points_object_local", "object_geometry", "hand_cloud_template",
                  "object_mass", "gravity_world")
        for name in shared + ("q_lower", "q_upper"):
            if any(not np.array_equal(np.asarray(getattr(caches[0], name)) if getattr(caches[0], name) is not None else getattr(caches[0], name), np.asarray(getattr(c, name)) if getattr(c, name) is not None else getattr(c, name)) for c in caches[1:]):
                raise ValueError(f"reference bank requires shared {name}")
        if any(c.hand_cloud_reference is not None for c in caches) and not all(c.hand_cloud_reference is not None for c in caches):
            raise ValueError("reference bank requires hand cloud present on every reference or none")
        put = lambda x: jax.device_put(j.asarray(x), device)
        lengths = np.asarray([len(c.q_feasible) for c in caches], np.int32)
        self.lengths = put(lengths)
        self.max_length = int(lengths.max())
        self.num_references = len(caches)
        for name in REFERENCE_TIME_FIELDS:
            if all(getattr(c, name) is None for c in caches):
                setattr(self, name, None)
                continue
            padded = [np.concatenate((getattr(c, name), np.repeat(getattr(c, name)[-1:], self.max_length-n, axis=0))) for c,n in zip(caches,lengths)]
            setattr(self, name, put(np.stack(padded)))
        for name in ("q_lower", "q_upper", "duration", "support_shift"):
            setattr(self, name, put(np.stack([getattr(c,name) for c in caches])))
        self.action_id = put(np.asarray([c.action_id for c in caches], dtype=np.int32))
        for name in shared:
            value = getattr(caches[0], name)
            setattr(self, name, put(value) if isinstance(value,np.ndarray) else value)
        self.content_hash = hashlib.sha256("\n".join(c.content_hash for c in caches).encode()).hexdigest()


def reference_lengths(cache, env_ref=None):
    import jax.numpy as j
    return j.asarray(cache.lengths)[env_ref] if isinstance(cache, ReferenceBankV4) else len(cache.q_feasible)


def reference_duration(cache, env_ref=None):
    import jax.numpy as j
    return j.asarray(cache.duration)[env_ref] if isinstance(cache, ReferenceBankV4) else cache.duration

def _np_quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q=np.asarray(q,dtype=np.float64); q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    u,w=q[...,:3],q[...,3:]
    return v*(2*w*w-1)+2*w*np.cross(u,v)+2*u*np.sum(u*v,axis=-1,keepdims=True)

def align_reference_trajectory_v4(trajectory, vertices: np.ndarray, table_height: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure v4 support alignment; no legacy/native autonomy route is imported."""
    vertices=np.asarray(vertices,dtype=np.float64).reshape(-1,3)
    q=np.asarray(trajectory.q_ref,dtype=np.float64).copy()
    object_pos=np.asarray(trajectory.object_pos_raw,dtype=np.float64).copy()
    quat=np.asarray(trajectory.object_quat_xyzw[0],dtype=np.float64)
    if len(q) == 0 or object_pos.shape != (len(q),3) or quat.shape != (4,):
        raise ValueError("v4 reference alignment requires nonempty q/object/quaternion tables")
    if not np.all(np.isfinite(vertices)) or np.linalg.norm(quat) <= 1e-12:
        raise ValueError("v4 reference alignment received invalid collision geometry or quaternion")
    rotated=_np_quat_rotate(np.broadcast_to(quat,(len(vertices),4)),vertices)
    shift=np.array([0.,0.,float(table_height)-float(np.min(rotated[:,2]+object_pos[0,2]))])
    q[:,:3] += shift
    return q, object_pos + shift, shift


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

def _savgol(values: np.ndarray) -> np.ndarray:
    from scipy.signal import savgol_filter
    width=min(7, len(values) if len(values)%2 else len(values)-1)
    return savgol_filter(values,width,min(2,width-1),axis=0,mode='interp') if width >= 3 else values

def _filtered_derivative(values: np.ndarray, dt: float) -> np.ndarray:
    """Filtered derivative with a defined short-table contract."""
    values=np.asarray(values)
    if dt <= 0 or not np.isfinite(dt): raise ValueError("reference control timestep must be finite positive")
    if len(values) == 0: raise ValueError("reference derivative requires at least one frame")
    # numpy's edge_order=2 needs three samples; Savgol's width is odd and
    # bounded by both the table and the v4 seven-frame policy.
    derivative=np.gradient(values,dt,axis=0,edge_order=min(2,len(values)-1)) if len(values)>1 else np.zeros_like(values)
    return _savgol(derivative)

def _filtered_angular_velocity(quaternions: np.ndarray, dt: float) -> np.ndarray:
    """Source-consistent relative quaternion log rotations, then Savgol."""
    q=np.asarray(quaternions,dtype=np.float64).copy()
    for i in range(1,len(q)):
        if np.dot(q[i-1],q[i]) < 0: q[i]*=-1
    def log_delta(a,b):
        rel=np.asarray(quat_mul(np.asarray(b[None]),np.asarray(quat_conj(a[None]))))[0].copy()
        rel/=max(np.linalg.norm(rel),1e-12); angle=2*np.arctan2(np.linalg.norm(rel[:3]),abs(rel[3]))
        return np.zeros(3) if np.linalg.norm(rel[:3])<1e-12 else rel[:3]/np.linalg.norm(rel[:3])*angle
    interval=np.stack([log_delta(q[i],q[i+1])/dt for i in range(len(q)-1)])
    omega=np.empty((len(q),3)); omega[0]=interval[0]; omega[-1]=interval[-1]; omega[1:-1]=(interval[:-1]+interval[1:])/2
    return _savgol(omega)

def compile_reference_cache_v4(trajectory, *, device: str = "cpu", object_type: str = "cube2", table_height: float = -0.001) -> ReferenceCacheV4:
    """Compile the cache through Warp FK and exact compiled collision meshes only."""
    import jax, jax.numpy as j
    from mujoco import mjx
    from sim.manorl.assets import compile_model_metadata_only, object_collision_vertices
    from sim.manorl.environment import MjxWarpPhysicalProducer
    from sim.manorl.observations import geometry_encoding
    mujoco,model=compile_model_metadata_only(object_type=object_type,hand_side="right",physics_timestep=1/480)
    producer=MjxWarpPhysicalProducer(mujoco,model,object_type=object_type,hand_sides=("right",))
    mesh_type=int(mujoco.mjtGeom.mjGEOM_MESH)
    hand_geoms=np.asarray(producer.keypoint_geom_ids,np.int32)
    if not np.all(np.asarray(model.geom_type)[hand_geoms]==mesh_type): raise ValueError("all v4 hand regions must be compiled mesh collision geoms")
    object_geom=next(iter(producer.object_geom_ids))
    if int(model.geom_type[object_geom]) != mesh_type: raise ValueError(f"{object_type} collision geom must be a mesh")
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
    raw=np.asarray(trajectory.q_ref,dtype=np.float64).copy(); q_aligned,obj,shift=align_reference_trajectory_v4(trajectory,object_triangles.reshape(-1,3),table_height)
    lower=np.asarray(model.jnt_range[:28,0]); upper=np.asarray(model.jnt_range[:28,1]); feasible=np.clip(q_aligned,lower,upper)
    raw[:, :3]+=shift
    T=len(feasible)
    if T < 2: raise ValueError("v4 reference cache requires at least two frames")
    base=mjx.make_data(model,device=dev,impl="warp"); qpos=np.broadcast_to(np.asarray(base.qpos),(T,model.nq)).copy(); qpos[:,:28]=feasible
    qpos[:,producer.object_qpos_address:producer.object_qpos_address+3]=obj; qpos[:,producer.object_qpos_address+3:producer.object_qpos_address+7]=np.asarray(trajectory.object_quat_xyzw)[:,(3,0,1,2)]
    # Sequential/chunked offline N=1 FK keeps reference compilation distinct
    # from runtime physics batching.
    data=jax.vmap(lambda x:mjx.forward(m,base.replace(qpos=x)))(j.asarray(qpos))
    xpos=np.asarray(data.xpos); xquat=np.asarray(data.xquat)[...,(1,2,3,0)]; body=np.asarray(producer.keypoint_body_ids)
    hand_origin=xpos[:,body]; hand_q=xquat[:,body]; object_origin=xpos[:,producer.object_body_id]; object_q=xquat[:,producer.object_body_id]
    hand_world=hand_origin[:,:,None]+_np_quat_rotate(hand_q[:,:,None],hand_samples[None])
    object_local=np.asarray(quat_unrotate(j.asarray(object_q[:,None,None]),j.asarray(hand_world-object_origin[:,None,None])))
    # v5: 16 points per region (every 8th of the 128 samples), object-local frame.
    hand_cloud_template=hand_samples[:, ::8, :].reshape(256,3)
    hand_cloud_reference=object_local[:, :, ::8, :].reshape(T,256,3)
    points=object_local.reshape(-1,3); closest,signed,_=signed_closest_convex_mesh(points,object_triangles); closest=np.asarray(closest).reshape(T,16,128,3); signed=np.asarray(signed).reshape(T,16,128)
    chosen=np.argmin(np.abs(signed),axis=-1); rows=np.arange(T)[:,None]; region=np.arange(16)[None,:]
    ah=hand_samples[None].repeat(T,axis=0)[rows,region,chosen]
    ao=closest[rows,region,chosen]
    hand_anchor_world=hand_origin+_np_quat_rotate(hand_q,ah)
    object_anchor_world=object_origin[:,None]+_np_quat_rotate(object_q[:,None],ao)
    delta=np.asarray(quat_unrotate(j.asarray(object_q[:,None]),j.asarray(hand_anchor_world-object_anchor_world)))
    gap=signed[rows,region,chosen]; depth=np.maximum(-np.min(signed,axis=-1),0); fraction=np.mean(signed<-.001,axis=-1); proximity=np.exp(-(np.maximum(gap,0)/.01)**2); confidence=np.exp(-(np.maximum(depth-.001,0)/.003)**2)*(1-fraction); valid=(depth<=.01).astype(np.float64)
    fps=float(trajectory.control_fps or trajectory.reference_fps or 120)
    if fps != 120 or trajectory.timestamps.shape != (T,) or np.any(np.diff(trajectory.timestamps) < 0): raise ValueError("v4 requires monotonic 120 Hz reference clock")
    dt=1./fps; duration=float(trajectory.timestamps[-1]-trajectory.timestamps[0])
    if duration <= 0: duration=(T-1)*dt
    com_local=np.asarray(model.body_ipos[producer.object_body_id]); object_radius=float(np.max(np.linalg.norm(object_triangles.reshape(-1,3)-com_local,axis=-1))); com=object_origin+_np_quat_rotate(object_q,np.broadcast_to(com_local,(T,3))); ov=_filtered_derivative(com,dt); ow=_filtered_angular_velocity(object_q,dt); palm=hand_origin[:,0]; pv=_filtered_derivative(palm,dt); pw=_filtered_angular_velocity(hand_q[:,0],dt)
    object_bottom=np.min(object_origin[:,None,2]+_np_quat_rotate(object_q[:,None],object_triangles.reshape(-1,3))[:,:,2],axis=1)
    dimensions=np.ptp(object_triangles.reshape(-1,3),axis=0); geometry=geometry_encoding(object_name=object_type,geometry_type="box",dimensions=dimensions)
    object_mass=float(model.body_subtreemass[producer.object_body_id])
    gravity_world=np.asarray(model.opt.gravity,dtype=np.float64)
    values=(feasible,raw,lower,upper,obj,object_q,com_local,palm,hand_q[:,0],ov,ow,pv,pw,ah,ao,delta,gap,proximity,confidence,valid,object_bottom,np.asarray([table_height,dt,duration,object_radius,object_mass]),gravity_world,object_points,geometry,np.asarray(shift),np.asarray([dt]),np.asarray(model.geom_pos[hand_geoms]),np.asarray(model.geom_quat[hand_geoms]),hand_cloud_template,hand_cloud_reference)
    digest=cache_hash(*[np.asarray(x) for x in values],kernel_version="warp-mesh-cache-v4.2-motion-gated")
    result=ReferenceCacheV4(feasible,raw,lower,upper,obj,object_q,com_local,palm,hand_q[:,0],ov,ow,pv,pw,ah,ao,delta,gap,proximity,confidence,valid,object_bottom,float(table_height),dt,duration,object_points,geometry,int(trajectory.identity.identity.split('_')[1]),np.asarray(shift),digest,object_radius,hand_cloud_template,hand_cloud_reference,object_mass,gravity_world)
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
    all_state=j.concatenate((q_raw,qd,previous,oo,oc,ov,ow,po,pc,pv,pw,ro.reshape(q_raw.shape[0],-1),rc.reshape(q_raw.shape[0],-1),rv.reshape(q_raw.shape[0],-1),rw.reshape(q_raw.shape[0],-1)),axis=-1)
    raw_quat=xquat[:,np.r_[object_body_id,palm_body_id,ids]]
    quat_ok=j.all(j.isfinite(raw_quat),axis=(-1,-2))&j.all(j.linalg.norm(raw_quat,axis=-1)>1e-12,axis=-1)
    finite=j.all(j.isfinite(all_state),axis=-1)&j.all(j.isfinite(j.asarray(object_bottom)),axis=-1)&quat_ok
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

def _gather(cache: ReferenceCacheV4, index, offset=0, env_ref=None):
    import jax.numpy as j
    i=j.minimum(j.asarray(index)+offset,reference_lengths(cache, env_ref)-1)
    if isinstance(cache, ReferenceBankV4):
        if env_ref is None: raise ValueError("bank gather requires env_ref")
        return env_ref, i
    return i

def reduce_pyramidal_contacts_v4(*, nacon, nefc, geom, world, dimension, addresses, friction, frame, position, constraint_force, ngeom: int, hand_geom_ids, object_geom_ids, object_com, hand_com=None, hand_v_com=None, hand_w=None, object_v_com=None, object_w=None, cone: str = "pyramidal", return_object_torque: bool = False):
    """Reduce one *global* Warp contact arena into object and regional signals.

    Padded/unsolved (`efc_address=-1`) rows are erased before force, lever, or
    velocity arithmetic.  A solved row with malformed IDs, frame, friction or
    force is invalid rather than silently redirected by the safe gather.
    """
    import jax.numpy as j
    geom,world,dimension,addresses,friction,frame,position,constraint_force=map(j.asarray,(geom,world,dimension,addresses,friction,frame,position,constraint_force))
    capacity=geom.shape[0]; batch=constraint_force.shape[0]; slots=j.arange(capacity); count=j.asarray(nacon).reshape(())
    hand_host,obj_host=np.asarray(hand_geom_ids,dtype=np.int32),np.asarray(object_geom_ids,dtype=np.int32)
    if cone != "pyramidal": raise ValueError("v4 fast reducer supports pyramidal cone only")
    if hand_host.shape != (16,) or not len(obj_host) or np.intersect1d(hand_host,obj_host).size or (ngeom >= 16 and len(set(hand_host.tolist())) != 16):
        raise ValueError("v4 contact geom ownership is invalid")
    if geom.shape!=(capacity,2) or addresses.shape!=(capacity,4) or position.shape!=(capacity,3) or frame.shape!=(capacity,3,3) or friction.ndim != 2 or friction.shape[0] != capacity:
        raise ValueError("pinned Warp contact ABI mismatch")
    if count.dtype.kind not in 'iu': raise ValueError("global contact count must be integer")
    # Warp nacon is scalar for the arena, never B-wide.  Equality is overflow.
    live=slots<count; solved=live&(addresses[:,0]>=0)
    safe_world=j.clip(world,0,batch-1); safe_addr=j.clip(addresses,0,constraint_force.shape[1]-1); safe_geom=j.clip(geom,0,ngeom-1)
    nefc_array=j.asarray(nefc)
    nefc=j.broadcast_to(nefc_array.reshape(-1),(batch,)) if nefc_array.size==1 else nefc_array.reshape(batch)
    row_valid=(world>=0)&(world<batch)&(dimension==3)&j.all((geom>=0)&(geom<ngeom),axis=-1)&j.all((addresses>=0)&(addresses<nefc[safe_world,None]),axis=-1)&j.all(j.isfinite(friction[:,:2]),axis=-1)&j.all(j.isfinite(frame),axis=(-1,-2))&j.all(j.isfinite(position),axis=-1)
    valid=(count>=0)&(count<capacity)&j.all((nefc>=0)&(nefc<constraint_force.shape[1]))&j.all(j.where(solved,row_valid,True))
    lam=constraint_force[safe_world[:,None],safe_addr]
    local=j.stack((j.sum(lam,axis=-1),(lam[:,0]-lam[:,1])*friction[:,0],(lam[:,2]-lam[:,3])*friction[:,1]),axis=-1)
    force=j.einsum('ni,nij->nj',local,frame)
    valid=valid&j.all(j.where(solved[:,None],j.isfinite(lam),True))&j.all(j.where(solved[:,None],j.isfinite(force),True))
    # Mask before calculating levers or point velocity so padded NaNs cannot
    # enter a later reduction even transiently.
    force=j.where(solved[:,None],force,0.); position=j.where(solved[:,None],position,0.); frame=j.where(solved[:,None,None],frame,0.)
    lookup=j.full((ngeom,),-1,dtype=j.int32).at[j.asarray(hand_host)].set(j.arange(16,dtype=j.int32))
    a,b=lookup[safe_geom[:,0]],lookup[safe_geom[:,1]]
    a_obj=j.any(safe_geom[:,0,None]==j.asarray(obj_host)[None],axis=-1); b_obj=j.any(safe_geom[:,1,None]==j.asarray(obj_host)[None],axis=-1)
    first=solved&(a>=0)&b_obj; second=solved&(b>=0)&a_obj
    paired=j.zeros((batch,16,3),force.dtype); torque=j.zeros_like(paired); counts=j.zeros((batch,16),j.int32); all_force=j.zeros((batch,3),force.dtype); all_torque=j.zeros_like(all_force)
    all_force=all_force.at[safe_world].add(j.where(b_obj[:,None],force,0.)+j.where(a_obj[:,None],-force,0.))
    paired=paired.at[safe_world,j.maximum(a,0)].add(force*first[:,None]); paired=paired.at[safe_world,j.maximum(b,0)].add(-force*second[:,None])
    lever=position-object_com[safe_world]
    all_torque=all_torque.at[safe_world].add(
        j.where(b_obj[:,None],j.cross(lever,force),0.)
        + j.where(a_obj[:,None],-j.cross(lever,force),0.)
    )
    torque=torque.at[safe_world,j.maximum(a,0)].add(j.cross(lever,force)*first[:,None]); torque=torque.at[safe_world,j.maximum(b,0)].add(-j.cross(lever,force)*second[:,None])
    counts=counts.at[safe_world,j.maximum(a,0)].add(first.astype(j.int32)); counts=counts.at[safe_world,j.maximum(b,0)].add(second.astype(j.int32))
    slip=j.zeros((batch,16,3),force.dtype)
    if all(value is not None for value in (hand_com,hand_v_com,hand_w,object_v_com,object_w)):
        hand_com,hand_v_com,hand_w,object_v_com,object_w=map(j.asarray,(hand_com,hand_v_com,hand_w,object_v_com,object_w))
        region=j.maximum(j.where(first,a,j.where(second,b,0)),0)
        hc,hv,hw=hand_com[safe_world,region],hand_v_com[safe_world,region],hand_w[safe_world,region]
        ov,ow=object_v_com[safe_world],object_w[safe_world]
        v_hand=hv+j.cross(hw,position-hc)
        v_object=ov+j.cross(ow,position-object_com[safe_world])
        normal=frame[:,0,:]
        normal=normal/j.maximum(j.linalg.norm(normal,axis=-1,keepdims=True),1e-12)
        tangential=(v_hand-v_object)-j.sum((v_hand-v_object)*normal,axis=-1,keepdims=True)*normal
        pair=first|second; weight=j.linalg.norm(force,axis=-1)*pair
        weighted=weight[:,None]*tangential
        totals=j.zeros((batch,16,3),force.dtype).at[safe_world,region].add(weighted)
        denom=j.zeros((batch,16),force.dtype).at[safe_world,region].add(weight)
        slip=totals/j.maximum(denom[...,None],1e-12)
        valid=valid&j.all(j.where(pair[:,None],j.isfinite(tangential),True))
    valid=valid&j.all(j.isfinite(all_force))&j.all(j.isfinite(all_torque))&j.all(j.isfinite(paired))&j.all(j.isfinite(torque))&j.all(j.isfinite(slip))
    result=(all_force,paired,torque,counts,slip,valid)
    return result+(all_torque,) if return_object_torque else result

def build_raw_observation(physical: V4Physical, contact: V4Contact, cache: ReferenceCacheV4, index, previous_command, env_ref=None):
    """Build exact seven raw blocks, unclipped, from one same-time physical state."""
    import jax.numpy as j
    b=physical.q_raw.shape[0]; i=_gather(cache,index,env_ref=env_ref); t=j.maximum(reference_lengths(cache,env_ref)-1,1)
    C=lambda x:j.asarray(x)
    lower=C(cache.q_lower) if env_ref is None else C(cache.q_lower)[env_ref]
    upper=C(cache.q_upper) if env_ref is None else C(cache.q_upper)[env_ref]
    oq=C(cache.object_quat_xyzw)[i]; op=C(cache.object_origin)[i]; pq=C(cache.palm_quat_xyzw)[i]; pp=C(cache.palm_origin)[i]
    rq=C(cache.q_feasible)[i]; rq_normalized=j.concatenate((rq[:,:3]/WORLD_POSITION,rq[:,3:6]/j.pi,2*(rq[:,6:]-lower[...,6:])/(upper[...,6:]-lower[...,6:])-1),axis=-1); av=quat_unrotate(physical.object_quat_xyzw,physical.object_v_com)/LINEAR_VELOCITY; aw=quat_unrotate(physical.object_quat_xyzw,physical.object_w)/ANGULAR_VELOCITY
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
        fi=_gather(cache,index,horizon,env_ref); fq=C(cache.q_feasible)[fi]; fop=C(cache.object_origin)[fi]; foq=C(cache.object_quat_xyzw)[fi]; fpp=C(cache.palm_origin)[fi]; fpq=C(cache.palm_quat_xyzw)[fi]
        frel=quat_unrotate(foq,fpp-fop); frelq=quat_mul(quat_conj(foq),fpq)
        valid=((j.asarray(index)+horizon)*cache.control_timestep <= reference_duration(cache,env_ref)+1e-6).astype(j.float32)
        futures.append(j.concatenate(((fq[:,6:]-rq[:,6:])/JOINT_ERROR,(frel-ref_rel)/RELATIVE_POSITION,rot6(quat_mul(quat_conj(ref_relq),frelq)),quat_unrotate(oq,fop-op)/RELATIVE_POSITION,rot6(quat_mul(quat_conj(oq),foq)),valid[:,None]),axis=-1))
    _,_,delta,velocity=anchor_delta_and_velocity(physical,C(cache.region_anchor_hand)[i],C(cache.region_anchor_object)[i])
    error=delta-C(cache.delta_ref)[i]
    geom=j.concatenate((C(cache.signed_gap)[i][...,None]/CONTACT_DISTANCE,C(cache.proximity)[i][...,None],C(cache.confidence)[i][...,None],C(cache.valid)[i][...,None],C(cache.region_anchor_hand)[i]/RELATIVE_POSITION,C(cache.region_anchor_object)[i]/RELATIVE_POSITION,C(cache.delta_ref)[i]/CONTACT_DISTANCE,error/CONTACT_DISTANCE,j.tanh(quat_unrotate(physical.object_quat_xyzw[:,None],contact.paired_force_on_object)/FORCE_SCALE),velocity/CONTACT_VELOCITY),axis=-1).reshape(b,352)
    if isinstance(cache, ReferenceBankV4):
        action=j.eye(50,dtype=physical.q_raw.dtype)[j.asarray(cache.action_id)[env_ref]-1]
    else:
        action=j.eye(50,dtype=physical.q_raw.dtype)[cache.action_id-1][None].repeat(b,axis=0)
    raw=j.concatenate((actual,ref,j.concatenate(futures,axis=-1),geom,C(cache.points_object_local).reshape(1,192).repeat(b,axis=0)/RELATIVE_POSITION,action,C(cache.object_geometry)[None].repeat(b,axis=0)),axis=-1)
    if raw.shape[-1] != RAW_OBSERVATION_DIM: raise AssertionError(f"v4 raw ABI {raw.shape[-1]} != 957")
    return raw

HAND_CLOUD_POINTS = 256

def hand_cloud_actual(physical: V4Physical, template):
    """Actual hand surface cloud in the object's local frame from the live state."""
    import jax.numpy as j
    template=j.asarray(template).reshape(16,16,3)
    world=physical.region_origin[:,:,None]+quat_rotate(physical.region_quat_xyzw[:,:,None],template[None])
    cloud=quat_unrotate(physical.object_quat_xyzw[:,None,None],world-physical.object_origin[:,None,None])
    return cloud.reshape(physical.q_raw.shape[0],HAND_CLOUD_POINTS*3)

def build_raw_observation_v5(physical: V4Physical, contact: V4Contact, cache: ReferenceCacheV4, index, previous_command, env_ref=None):
    """v5 point-cloud observation: geometry correspondence via hand/object clouds.

    Hand-crafted anchor correspondence features are replaced by the hand cloud;
    only reference intent gating (confidence/valid) and the measured paired
    hand->object force remain as explicit numbers. Reward and teacher gating are
    computed from the cache directly and do not depend on this observation.
    """
    import jax.numpy as j
    b=physical.q_raw.shape[0]; i=_gather(cache,index,env_ref=env_ref); t=j.maximum(reference_lengths(cache,env_ref)-1,1)
    C=lambda x:j.asarray(x)
    lower=C(cache.q_lower) if env_ref is None else C(cache.q_lower)[env_ref]
    upper=C(cache.q_upper) if env_ref is None else C(cache.q_upper)[env_ref]
    oq=C(cache.object_quat_xyzw)[i]; op=C(cache.object_origin)[i]; pq=C(cache.palm_quat_xyzw)[i]; pp=C(cache.palm_origin)[i]
    rq=C(cache.q_feasible)[i]; rq_normalized=j.concatenate((rq[:,:3]/WORLD_POSITION,rq[:,3:6]/j.pi,2*(rq[:,6:]-lower[...,6:])/(upper[...,6:]-lower[...,6:])-1),axis=-1); av=quat_unrotate(physical.object_quat_xyzw,physical.object_v_com)/LINEAR_VELOCITY; aw=quat_unrotate(physical.object_quat_xyzw,physical.object_w)/ANGULAR_VELOCITY
    palm_rel=quat_unrotate(physical.object_quat_xyzw,physical.palm_origin-physical.object_origin)
    palm_relq=quat_mul(quat_conj(physical.object_quat_xyzw),physical.palm_quat_xyzw)
    ref_rel=quat_unrotate(oq,pp-op); ref_relq=quat_mul(quat_conj(oq),pq)
    ref_com_v=C(cache.object_v_com)[i]; ref_w=C(cache.object_w)[i]
    ref_obj6=rot6(oq); ref_palm6=rot6(pq)
    actual = j.concatenate((physical.q_normalized, physical.qdot, physical.command_error, physical.object_origin/WORLD_POSITION, rot6(physical.object_quat_xyzw), av,aw,palm_rel/RELATIVE_POSITION,rot6(palm_relq),quat_unrotate(physical.object_quat_xyzw,physical.palm_w)/ANGULAR_VELOCITY,(physical.object_bottom-cache.table_height)[:,None]/RELATIVE_POSITION,(C(cache.reference_bottom)[i]-cache.table_height)[:,None]/RELATIVE_POSITION,quat_unrotate(physical.object_quat_xyzw,j.broadcast_to(j.asarray([0.,0.,-1.]),(b,3))),j.tanh(quat_unrotate(physical.object_quat_xyzw,contact.object_all_force)/FORCE_SCALE)),axis=-1)
    ref = j.concatenate((rq_normalized[:,6:],ref_rel/RELATIVE_POSITION,rot6(ref_relq),quat_unrotate(oq,C(cache.palm_v)[i])/LINEAR_VELOCITY,quat_unrotate(oq,C(cache.palm_w)[i])/ANGULAR_VELOCITY,op/WORLD_POSITION,ref_obj6,ref_com_v/LINEAR_VELOCITY,ref_w/ANGULAR_VELOCITY,quat_unrotate(physical.object_quat_xyzw,physical.palm_origin-pp)/RELATIVE_POSITION,rot6(quat_mul(quat_conj(pq),physical.palm_quat_xyzw)),(physical.q_raw[:,6:]-rq[:,6:])/JOINT_ERROR,quat_unrotate(physical.object_quat_xyzw,physical.object_origin-op)/RELATIVE_POSITION,rot6(quat_mul(quat_conj(oq),physical.object_quat_xyzw)),quat_unrotate(physical.object_quat_xyzw,physical.object_v_com-ref_com_v)/LINEAR_VELOCITY,quat_unrotate(physical.object_quat_xyzw,physical.object_w-ref_w)/ANGULAR_VELOCITY,(palm_rel-ref_rel)/RELATIVE_POSITION,rot6(quat_mul(quat_conj(ref_relq),palm_relq)),j.stack((j.asarray(index)/t,1-j.asarray(index)/t),axis=-1)),axis=-1)
    futures=[]
    for horizon in (6,12,24):
        fi=_gather(cache,index,horizon,env_ref); fop=C(cache.object_origin)[fi]
        valid=((j.asarray(index)+horizon)*cache.control_timestep <= reference_duration(cache,env_ref)+1e-6).astype(j.float32)
        futures.append(j.concatenate((quat_unrotate(oq,fop-op)/RELATIVE_POSITION,valid[:,None]),axis=-1))
    intent=j.concatenate((C(cache.confidence)[i],C(cache.valid)[i]),axis=-1)
    force=j.tanh(quat_unrotate(physical.object_quat_xyzw[:,None],contact.paired_force_on_object)/FORCE_SCALE).reshape(b,48)
    contact_intent=j.concatenate((intent,force),axis=-1)
    hand=hand_cloud_actual(physical,C(cache.hand_cloud_template))/RELATIVE_POSITION
    if isinstance(cache, ReferenceBankV4):
        action=j.eye(50,dtype=physical.q_raw.dtype)[j.asarray(cache.action_id)[env_ref]-1]
    else:
        action=j.eye(50,dtype=physical.q_raw.dtype)[cache.action_id-1][None].repeat(b,axis=0)
    raw=j.concatenate((actual,ref,j.concatenate(futures,axis=-1),contact_intent,C(cache.points_object_local).reshape(1,192).repeat(b,axis=0)/RELATIVE_POSITION,hand,action,C(cache.object_geometry)[None].repeat(b,axis=0)),axis=-1)
    if raw.shape[-1] != RAW_OBSERVATION_DIM_V5: raise AssertionError(f"v5 raw ABI {raw.shape[-1]} != {RAW_OBSERVATION_DIM_V5}")
    return raw


def build_raw_observation_v6(physical: V4Physical, contact: V4Contact, cache: ReferenceCacheV4, index, previous_command, env_ref=None):
    """VoxMani-inspired v6 observation with region-bound contact and goal geometry.

    The environment, reward, action command, PPO and raw-Normal distribution are
    unchanged. This function only enriches the observation consumed by the
    v6 token model.
    """
    import jax.numpy as j

    if cache.hand_cloud_reference is None:
        raise ValueError("v6 requires the per-frame reference hand cloud")
    if contact.object_all_torque is None:
        raise ValueError("v6 requires total object contact torque")

    base=build_raw_observation_v5(physical,contact,cache,index,previous_command,env_ref)
    slices=raw_observation_slices_v5()
    batch=physical.q_raw.shape[0]
    gather=_gather(cache,index,env_ref=env_ref)
    C=lambda value:j.asarray(value)

    gravity=C(cache.gravity_world if cache.gravity_world is not None else np.asarray([0.,0.,-9.81]))
    force_scale=j.maximum(C(cache.object_mass)*j.linalg.norm(gravity),1e-6)
    torque_scale=j.maximum(force_scale*C(cache.object_radius),1e-6)
    object_q=physical.object_quat_xyzw

    paired_force=j.arcsinh(
        quat_unrotate(object_q[:,None],contact.paired_force_on_object)/force_scale
    )
    slip=quat_unrotate(object_q[:,None],contact.tangential_slip)/CONTACT_VELOCITY
    active=(contact.paired_count>0).astype(physical.q_raw.dtype)
    count=j.log1p(contact.paired_count.astype(physical.q_raw.dtype))
    region_contact=j.concatenate((
        C(cache.confidence)[gather][...,None],
        C(cache.valid)[gather][...,None],
        active[...,None],
        count[...,None],
        paired_force,
        slip,
    ),axis=-1).reshape(batch,160)

    hand_force=j.sum(contact.paired_force_on_object,axis=1)
    hand_torque=j.sum(contact.paired_torque_com,axis=1)
    other_force=contact.object_all_force-hand_force
    other_torque=contact.object_all_torque-hand_torque
    object_wrench=j.concatenate((
        quat_unrotate(object_q,j.broadcast_to(C(gravity)*C(cache.object_mass),(batch,3)))/force_scale,
        j.arcsinh(quat_unrotate(object_q,hand_force)/force_scale),
        j.arcsinh(quat_unrotate(object_q,hand_torque)/torque_scale),
        j.arcsinh(quat_unrotate(object_q,other_force)/force_scale),
        j.arcsinh(quat_unrotate(object_q,other_torque)/torque_scale),
    ),axis=-1)

    reference_hand=C(cache.hand_cloud_reference)[gather].reshape(batch,768)/RELATIVE_POSITION
    raw=j.concatenate((
        base[:,slices["autonomous_actual"]],
        base[:,slices["autonomous_reference"]],
        base[:,slices["autonomous_future"]],
        region_contact,
        object_wrench,
        base[:,slices["object_point_cloud_raw"]],
        base[:,slices["hand_point_cloud_raw"]],
        reference_hand,
        base[:,slices["action_types"]],
        base[:,slices["object_geometry"]],
    ),axis=-1)
    if raw.shape[-1] != RAW_OBSERVATION_DIM_V6:
        raise AssertionError(f"v6 raw ABI {raw.shape[-1]} != {RAW_OBSERVATION_DIM_V6}")
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

def compute_reward(physical: V4Physical, contact: V4Contact, cache: ReferenceCacheV4, index, executed_action, env_ref=None):
    import jax.numpy as j
    i=_gather(cache,index,env_ref=env_ref); C=lambda x:j.asarray(x); target_p=C(cache.object_origin)[i]; target_q=C(cache.object_quat_xyzw)[i]
    dp=physical.object_origin-target_p
    pos=.2*j.exp(-40*j.abs(dp[:,0]))+.2*j.exp(-40*j.abs(dp[:,1]))+.8*j.exp(-40*j.abs(dp[:,2]))
    deg=shortest_angle(physical.object_quat_xyzw,target_q)*180/j.pi; rotvalue=j.where(deg<=20,1-.00125*deg**2,j.where(deg<=90,-.00003175*(deg-20)**2-.019206*(deg-20)+.5,-1.)); rot=.4*rotvalue-.1
    vel=.1*j.exp(-j.sum(((physical.object_v_com-C(cache.object_v_com)[i])/.25)**2,axis=-1)-j.sum(((physical.object_w-C(cache.object_w)[i])/2.)**2,axis=-1))
    rel=quat_unrotate(physical.object_quat_xyzw,physical.palm_origin-physical.object_origin); rrel=quat_unrotate(target_q,C(cache.palm_origin)[i]-target_p); relq=quat_mul(quat_conj(physical.object_quat_xyzw),physical.palm_quat_xyzw); rrelq=quat_mul(quat_conj(target_q),C(cache.palm_quat_xyzw)[i]); hand=REWARD_HAND_RELATIVE_COEF*j.exp(-j.sum(((rel-rrel)/.04)**2,axis=-1)-(shortest_angle(relq,rrelq)/.35)**2)
    fingers=.2*j.exp(-j.mean((physical.q_raw[:,6:]-C(cache.q_feasible)[i][:,6:])**2,axis=-1)/.35**2)
    _,_,delta,_=anchor_delta_and_velocity(physical,C(cache.region_anchor_hand)[i],C(cache.region_anchor_object)[i]); e=delta-C(cache.delta_ref)[i]; weight=C(cache.proximity)[i]*C(cache.confidence)[i]*C(cache.valid)[i]; geometry=REWARD_GEOMETRY_COEF*j.sum(weight*j.exp(-j.sum(e*e,axis=-1)/.01**2),axis=-1)/j.maximum(j.sum(weight,axis=-1),1.)
    v_ref=C(cache.object_v_com)[i]; w_ref=C(cache.object_w)[i]
    v_eff=j.sqrt(j.sum(v_ref*v_ref,axis=-1)+(j.asarray(cache.object_radius)*j.linalg.norm(w_ref,axis=-1))**2)
    w_obj=motion_gate_weight(v_eff)
    pos=w_obj*pos; rot=w_obj*rot; vel=w_obj*vel
    action=-.002*j.mean(j.clip(executed_action,-1,1)**2,axis=-1); survival=j.full_like(pos,.001)
    fallen=physical.object_bottom < cache.table_height-.05; deviation=j.linalg.norm(dp,axis=-1)>.10; finite=j.isfinite(pos+rot+vel+hand+fingers+geometry+action)&physical.valid&contact.valid
    severe=j.where(finite,j.where(fallen|deviation,-REWARD_SEVERE_PENALTY,0.),-REWARD_SEVERE_PENALTY); reason=(j.asarray(index)>=reference_lengths(cache,env_ref)-1).astype(j.int32)|j.where(deviation,2,0)|j.where(fallen,4,0)|j.where(~finite,8,0)
    total=j.where(finite,pos+rot+vel+hand+fingers+geometry+action+survival+severe,-REWARD_SEVERE_PENALTY)
    done=(reason!=0)
    return V4Reward(total,pos,rot,vel,hand,fingers,geometry,action,survival,severe,done,reason,finite)
