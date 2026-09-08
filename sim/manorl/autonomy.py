"""Physical, demonstration-conditioned cube2 autonomy diagnostics (v2).

The reference conditions observations, intent and reward.  The actor's 28-D
output is converted to commands only through the measured-state, per-second
rate map in :mod:`autonomy_contracts`.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray
from sim.manorl.assets import compile_model, object_collision_vertices, object_runtime
from sim.manorl.autonomy_contracts import ACTION_DIM, OBSERVATION_DIM, observation_slices, rate_limited_command
from sim.manorl.contracts import FLOOR_TOP_Z, JOINT_DOF, KEYPOINT_NAMES, simulation_clock
from sim.manorl.environment import MjxWarpPhysicalProducer
from sim.manorl.trajectory import ReferenceTrajectory
PACKAGE_DEFAULT = Path("outputs/manorl/contact_conditioned_autonomy/cube2_02_v295_f120_pre180_post180")

def _quat_rotate(q: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    qv, qw = q[:3], q[3]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)

def source_surface_points(object_type: str = "cube2", count: int = 128, seed: int = 0) -> NDArray[np.float64]:
    if count < 1: raise ValueError("surface sample count must be positive")
    triangles = np.asarray(object_collision_vertices(object_type), dtype=float).reshape(-1, 3, 3)
    areas = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1) * .5
    if np.any(areas <= 0) or not np.all(np.isfinite(areas)): raise ValueError("collision mesh contains degenerate triangles")
    rng = np.random.default_rng(seed); ids = np.searchsorted(np.cumsum(areas), rng.random(count) * areas.sum())
    base = triangles[ids, 0]; u, v = rng.random((count,1)), rng.random((count,1)); over = u + v > 1
    u[over], v[over] = 1-u[over], 1-v[over]
    return base + u*(triangles[ids,1]-base) + v*(triangles[ids,2]-base)

def surface_intent(keypoints_world: NDArray[np.floating], object_position: NDArray[np.floating], object_quat_xyzw: NDArray[np.floating], surface_local: NDArray[np.floating]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    points, obj, quat, surface = map(lambda x: np.asarray(x, dtype=float), (keypoints_world, object_position, object_quat_xyzw, surface_local))
    if points.shape != (16,3) or obj.shape != (3,) or quat.shape != (4,) or surface.ndim != 2 or surface.shape[1] != 3: raise ValueError("surface intent shapes must be (16,3), (3,), (4,), (N,3)")
    world_surface = np.stack([_quat_rotate(quat, p) for p in surface]) + obj
    distances = np.linalg.norm(points[:,None,:] - world_surface[None,:,:], axis=-1); nearest = np.argmin(distances, axis=1)
    min_distance = distances[np.arange(16), nearest]; proximity = np.exp(-min_distance / .01)
    return proximity, surface[nearest].copy(), np.clip(proximity, 0, 1)

def shared_support_alignment(hand_position: NDArray[np.floating], object_position: NDArray[np.floating], surface_local: NDArray[np.floating], object_quat_xyzw: NDArray[np.floating] | None = None, table_height: float = FLOOR_TOP_Z) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Translate hand and object exactly once using actual rotated vertices."""
    hand, obj, surface = map(lambda x: np.asarray(x, dtype=float), (hand_position, object_position, surface_local)); quat = np.array([0.,0.,0.,1.]) if object_quat_xyzw is None else np.asarray(object_quat_xyzw, dtype=float)
    if hand.shape != (3,) or obj.shape != (3,) or surface.ndim != 2 or surface.shape[1] != 3 or quat.shape != (4,): raise ValueError("alignment expects hand/object vectors, quaternion and (N,3) surface")
    min_z = float(np.min(np.stack([_quat_rotate(quat, p) for p in surface])[:,2] + obj[2])); shift = np.array([0.,0.,float(table_height)-min_z])
    return hand + shift, obj + shift, shift

def align_reference_trajectory(trajectory: ReferenceTrajectory, vertices: NDArray[np.floating], table_height: float = FLOOR_TOP_Z) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Align raw hand/object references once, preserving all source arrays."""
    q_ref = np.asarray(trajectory.q_ref, dtype=float).copy()
    raw_object = np.asarray(trajectory.object_pos_raw, dtype=float)
    _, _, shift = shared_support_alignment(q_ref[0, :3], raw_object[0], vertices, trajectory.object_quat_xyzw[0], table_height)
    q_ref[:, :3] += shift
    object_pos = raw_object.copy() + shift
    return q_ref, object_pos, shift

def terminal_status(step_index: int, last_index: int, *, dropped: bool, diverged: bool, actual_peak_lift: float, target_peak_lift: float, path_error: float) -> tuple[bool, bool, str]:
    """Classify terminal cause separately from physical task success."""
    horizon = step_index >= last_index
    success = bool(horizon and actual_peak_lift >= max(0.0, target_peak_lift - 0.02) and path_error < 0.03)
    done = bool(horizon or dropped or diverged)
    phase = "drop" if dropped else "path_divergence" if diverged else "task_success" if success else "horizon_reached" if horizon else "running"
    return done, success, phase

def collision_surface_intent(mujoco: Any, model: Any, data: Any, hand_geom_ids: list[int], object_geom_ids: set[int], object_position: NDArray[np.floating], object_quat_xyzw: NDArray[np.floating]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Measure segment/object signed distance and MuJoCo witness endpoints."""
    obj, quat = np.asarray(object_position, dtype=float), np.asarray(object_quat_xyzw, dtype=float)
    distances, anchors, witnesses = np.full(16, np.inf), np.zeros((16,3)), np.zeros((16,6))
    for idx, hand_geom in enumerate(hand_geom_ids):
        best = np.inf; best_fromto = np.zeros(6)
        for object_geom in sorted(object_geom_ids):
            fromto = np.zeros(6, dtype=float)
            distance = float(mujoco.mj_geomDistance(model, data, int(hand_geom), int(object_geom), 1.0, fromto))
            if distance < best: best, best_fromto = distance, fromto.copy()
        distances[idx], witnesses[idx] = best, best_fromto
        anchors[idx] = _quat_rotate(np.array([-quat[0], -quat[1], -quat[2], quat[3]]), best_fromto[3:] - obj)
    proximity = np.exp(-np.maximum(distances, 0.0) / .01)
    confidence = np.where(distances <= 0.0, np.clip(-distances / .002, 0.0, 1.0), np.exp(-distances / .005))
    return proximity, anchors, confidence, witnesses

def dense_autonomous_reward(object_position: NDArray[np.floating], target_object_position: NDArray[np.floating], object_velocity: NDArray[np.floating], hand_object_relative: NDArray[np.floating], reference_hand_object_relative: NDArray[np.floating], proximity: NDArray[np.floating], previous_proximity: NDArray[np.floating], action: NDArray[np.floating], *, phase: float, measured_hand_object_force: NDArray[np.floating] | None = None, supporting_object_force: NDArray[np.floating] | None = None, relative_contact_motion: NDArray[np.floating] | None = None, release_active: bool = False, target_object_velocity: NDArray[np.floating] | None = None, reference_proximity: NDArray[np.floating] | None = None, anchors: NDArray[np.floating] | None = None, reference_anchors: NDArray[np.floating] | None = None, measured_q: NDArray[np.floating] | None = None, reference_q: NDArray[np.floating] | None = None, object_quat: NDArray[np.floating] | None = None, reference_object_quat: NDArray[np.floating] | None = None) -> dict[str,float]:
    obj,target,vel,rel,ref = map(lambda x: np.asarray(x,dtype=float),(object_position,target_object_position,object_velocity,hand_object_relative,reference_hand_object_relative)); prox,prev,act = np.asarray(proximity,dtype=float),np.asarray(previous_proximity,dtype=float),np.asarray(action,dtype=float)
    if any(x.shape != (3,) for x in (obj,target,vel,rel,ref)) or prox.shape != (16,) or prev.shape != (16,): raise ValueError("reward vectors have invalid shape")
    force = np.zeros((16,3)) if measured_hand_object_force is None else np.asarray(measured_hand_object_force,dtype=float).reshape(16,3); support = np.zeros(3) if supporting_object_force is None else np.asarray(supporting_object_force,dtype=float)
    slip = 0.0 if relative_contact_motion is None else float(np.linalg.norm(np.asarray(relative_contact_motion,dtype=float).reshape(16,3),axis=1).mean())
    force_norm = np.linalg.norm(force,axis=1); demo_prox = prox if reference_proximity is None else np.asarray(reference_proximity,dtype=float); contact = float(np.mean(np.minimum(force_norm/.2,1.0) * demo_prox))
    anchor_error = 0.0 if anchors is None or reference_anchors is None else float(np.mean(np.linalg.norm(np.asarray(anchors)-np.asarray(reference_anchors),axis=1)))
    finger_error = 0.0 if measured_q is None or reference_q is None else float(np.linalg.norm(np.asarray(measured_q)[6:]-np.asarray(reference_q)[6:]) / np.sqrt(22.0))
    orientation_error = 0.0 if object_quat is None or reference_object_quat is None else float(1.0-abs(np.dot(np.asarray(object_quat),np.asarray(reference_object_quat))))
    motion_error = float(np.linalg.norm(obj-target)); velocity_error = 0.0 if target_object_velocity is None else float(np.linalg.norm(vel-np.asarray(target_object_velocity,dtype=float)))
    return {
        "object_motion": float(np.exp(-20.0*motion_error)),
        "reference_hand_object_relationship": float(np.exp(-30.0*np.linalg.norm(rel-ref))),
        "contact_anchor_correspondence": float(np.mean(prox*demo_prox) * np.exp(-anchor_error/.01)),
        "measured_contact": contact,
        "finger_configuration": float(np.exp(-finger_error)),
        "object_orientation": float(np.exp(-8.0*orientation_error)),
        "slip_proxy": float(-.02*slip if np.any(force_norm>.02) else 0.0),
        "object_velocity_tracking": float(np.exp(-8.0*velocity_error)),
        "release": float(1.0-np.mean(prox) if release_active else 0.0),
        "action_smoothness": float(-.001*np.mean(np.square(act))),
    }

@dataclass
class AutonomousStep:
    observation: NDArray[np.float32]; reward: float; terms: dict[str,float]; done: bool; info: dict[str,Any]

class Cube2AutonomousMJX:
    """One-world MJX-Warp diagnostic environment using canonical contacts/clock."""
    def __init__(self, trajectory: ReferenceTrajectory, *, device: str="cpu", seed: int=0, near_contact: bool=False):
        if trajectory.dof_dim != JOINT_DOF or trajectory.identity.identity.split("_")[0] != "cube2": raise ValueError("autonomous milestone requires a 28-DoF cube2 trajectory")
        if device not in {"cpu","gpu"}: raise ValueError("device must be cpu or gpu")
        import jax
        from mujoco import mjx
        self.trajectory,self.device_name,self.near_contact,self.seed = trajectory,device,near_contact,int(seed)
        fps = trajectory.control_fps or trajectory.reference_fps or 120; self.clock = simulation_clock(fps)
        if fps != 120: raise ValueError("M2 diagnostic requires the canonical 120 Hz reference clock")
        self.mujoco,self.model = compile_model(object_type="cube2", hand_side="right", physics_timestep=self.clock.physics_timestep)
        if not np.isclose(self.model.opt.timestep,1/480): raise RuntimeError("compiled model is not 480 Hz physics")
        self.mjx,self.jax = mjx,jax; devices=jax.devices(device)
        if not devices: raise RuntimeError(f"no JAX {device} device available")
        self.device=devices[0]; self.mjx_model=mjx.put_model(self.model,device=self.device,impl="warp")
        if str(self.mjx_model.impl).lower().split(".")[-1] != "warp": raise RuntimeError("MJX-Warp implementation was not selected")
        self.step_fn = jax.jit(jax.vmap(lambda world: mjx.step(self.mjx_model,world)))
        self.surface = source_surface_points("cube2",128,seed); self.vertices=np.asarray(object_collision_vertices("cube2"),dtype=float)
        self.keypoint_body_ids=[self.mujoco.mj_name2id(self.model,self.mujoco.mjtObj.mjOBJ_BODY,name) for name in KEYPOINT_NAMES]
        if min(self.keypoint_body_ids)<0: raise ValueError("missing keypoint body")
        object_joint=self.mujoco.mj_name2id(self.model,self.mujoco.mjtObj.mjOBJ_JOINT,object_runtime("cube2").free_joint_name); self.object_qpos=int(self.model.jnt_qposadr[object_joint]); self.object_qvel=int(self.model.jnt_dofadr[object_joint])
        self.lower,self.upper=self.model.jnt_range[:JOINT_DOF,0].copy(),self.model.jnt_range[:JOINT_DOF,1].copy()
        self.rate_per_second=np.r_[np.full(3,.5),np.full(3,2.),np.full(22,4.)]
        self.max_tracking_error=np.r_[np.full(3,.02),np.full(3,.25),np.full(22,.35)]
        self.producer=MjxWarpPhysicalProducer(self.mujoco,self.model,object_type="cube2",hand_sides=("right",),primary_hand_side="right")
        self.aligned_q_ref, self.aligned_object_pos, self.support_shift = align_reference_trajectory(trajectory, self.vertices); self._raw_initial_obj = np.asarray(trajectory.object_pos_raw[0], dtype=float).copy()
        self.reference_keypoints,self.reference_proximity,self.reference_anchors,self.reference_confidence=self._reference_fk()
        mean_ref=self.reference_proximity.mean(axis=1); contact_indices=np.flatnonzero(mean_ref>.35); self.release_start=int(contact_indices[-1]+1) if len(contact_indices) else len(mean_ref)
        self.reset()
    def _reference_fk(self):
        rows=[]; proximities=[]; anchors=[]; confidences=[]
        object_geoms=self.producer.object_geom_ids
        for q,obj,quat in zip(self.aligned_q_ref,self.aligned_object_pos,self.trajectory.object_quat_xyzw):
            d=self.mujoco.MjData(self.model); self.mujoco.mj_resetData(self.model,d); d.qpos[:JOINT_DOF]=q; d.qpos[self.object_qpos:self.object_qpos+3]=obj; d.qpos[self.object_qpos+3:self.object_qpos+7]=quat[[3,0,1,2]]; self.mujoco.mj_forward(self.model,d); rows.append(np.asarray(d.xpos[self.keypoint_body_ids]).copy()); prox,anchor,confidence,_=collision_surface_intent(self.mujoco,self.model,d,self.producer.keypoint_geom_ids,object_geoms,obj,quat); proximities.append(prox); anchors.append(anchor); confidences.append(confidence)
        return np.asarray(rows),np.asarray(proximities),np.asarray(anchors),np.asarray(confidences)
    def reset(self):
        d=self.mujoco.MjData(self.model); self.mujoco.mj_resetData(self.model,d); q=self.aligned_q_ref[0].copy(); obj=self.aligned_object_pos[0].copy()
        if self.near_contact: q[:3] += (obj-q[:3])*.25
        d.qpos[:JOINT_DOF],d.qpos[self.object_qpos:self.object_qpos+3]=q,obj; d.qpos[self.object_qpos+3:self.object_qpos+7]=self.trajectory.object_quat_xyzw[0][[3,0,1,2]]; d.qvel[:]=0; d.ctrl[:JOINT_DOF]=q; self.mujoco.mj_forward(self.model,d)
        actual_bottom = float(np.min(np.stack([_quat_rotate(self.trajectory.object_quat_xyzw[0], p) for p in self.vertices])[:,2] + obj[2]))
        if not np.isclose(actual_bottom, FLOOR_TOP_Z, atol=1e-8): raise RuntimeError(f"aligned cube support is {actual_bottom:.9f}, expected FLOOR_TOP_Z={FLOOR_TOP_Z}")
        single=self.mjx.put_data(self.model,d,device=self.device,impl="warp",naconmax=256,njmax=512); self.data=self.jax.vmap(lambda _: single)(self.jax.numpy.arange(1)); self.previous_command=q.copy(); self.step_index=0; self.previous_proximity=np.zeros(16); self.previous_keypoints=np.asarray(d.xpos[self.keypoint_body_ids]).copy(); self.last_relative_motion=np.zeros((16,3)); self.max_object_z=float(obj[2]); self.target_peak_lift=float(np.max(self.aligned_object_pos[:,2])-self.aligned_object_pos[0,2]); self.last_physical=self.producer.extract(self.data); return self._observation()
    def _host(self):
        d=self.mjx.get_data(self.model,self.data); return d[0] if isinstance(d,list) else d
    def _state(self):
        d=self._host(); q=np.asarray(d.qpos[0,:JOINT_DOF] if np.asarray(d.qpos).ndim==2 else d.qpos[:JOINT_DOF]); qvel=np.asarray(d.qvel[0,:JOINT_DOF] if np.asarray(d.qvel).ndim==2 else d.qvel[:JOINT_DOF]); obj=np.asarray(d.qpos[0,self.object_qpos:self.object_qpos+3] if np.asarray(d.qpos).ndim==2 else d.qpos[self.object_qpos:self.object_qpos+3]); quat=np.asarray(d.xquat[0,self.mujoco.mj_name2id(self.model,self.mujoco.mjtObj.mjOBJ_BODY,object_runtime("cube2").body_name)] if np.asarray(d.xquat).ndim==3 else d.xquat[self.mujoco.mj_name2id(self.model,self.mujoco.mjtObj.mjOBJ_BODY,object_runtime("cube2").body_name)])[[1,2,3,0]]; key=np.asarray(d.xpos[0,self.keypoint_body_ids] if np.asarray(d.xpos).ndim==3 else d.xpos[self.keypoint_body_ids]); return d,q,qvel,obj,quat,key
    def _actual_collision(self, q: NDArray[np.floating], qvel: NDArray[np.floating], obj: NDArray[np.floating], quat: NDArray[np.floating]):
        d=self.mujoco.MjData(self.model); self.mujoco.mj_resetData(self.model,d); d.qpos[:JOINT_DOF]=q; d.qpos[self.object_qpos:self.object_qpos+3]=obj; d.qpos[self.object_qpos+3:self.object_qpos+7]=quat[[3,0,1,2]]; d.qvel[:JOINT_DOF]=qvel; self.mujoco.mj_forward(self.model,d); return collision_surface_intent(self.mujoco,self.model,d,self.producer.keypoint_geom_ids,self.producer.object_geom_ids,obj,quat)
    def _observation(self):
        _,q,qvel,obj,quat,key=self._state(); i=min(self.step_index,len(self.aligned_q_ref)-1); j=min(i+1,len(self.aligned_q_ref)-1); k=min(i+5,len(self.aligned_q_ref)-1); physical=self.last_physical; measured_force=np.asarray(physical.hand_object_force_on_object_world_N[0]); support=measured_force.sum(axis=0); prox,anchors,conf,witnesses=self._actual_collision(q,qvel,obj,quat); ref_prox,ref_anchors,ref_conf=self.reference_proximity[i],self.reference_anchors[i],self.reference_confidence[i]; s=observation_slices(); out=np.zeros(OBSERVATION_DIM); out[s["measured_qpos_normalized"]]=2*(q-self.lower)/(self.upper-self.lower)-1; out[s["measured_qvel"]]=np.clip(qvel,-20,20); out[s["object_position"]]=obj; out[s["object_linear_velocity"]]=np.asarray(physical.object_linear_velocity[0]); out[s["hand_object_relative"]]=key[0]-obj; out[s["reference_q_current"]]=self.aligned_q_ref[i]; out[s["reference_q_next"]]=self.aligned_q_ref[j]; out[s["reference_q_velocity"]]=(self.aligned_q_ref[j]-self.aligned_q_ref[i])*self.clock.policy_fps; out[s["reference_object_relative"]]=self.reference_keypoints[i,0]-self.aligned_object_pos[i]; out[s["reference_object_future_delta"]]=self.aligned_object_pos[k]-self.aligned_object_pos[i]; out[s["previous_command_normalized"]]=2*(self.previous_command-self.lower)/(self.upper-self.lower)-1; action_id=int(self.trajectory.identity.identity.split("_")[1]); out[s["action_identity_one_hot"]][action_id-1]=1; out[s["object_geometry"]]=[.05,.05,.05,0,0,0,0,0,0,0,0,0]; out[s["measured_keypoint_relative"]]=(key-obj).reshape(-1); out[s["surface_proximity"]]=prox; out[s["surface_anchor_local"]]=anchors.reshape(-1); out[s["contact_phase_confidence"]]=[i/max(1,len(self.aligned_q_ref)-1),float(np.mean(ref_conf))]; out[s["reference_surface_proximity"]]=ref_prox; out[s["reference_surface_anchor_local"]]=ref_anchors.reshape(-1); out[s["reference_contact_confidence"]]=ref_conf; out[s["measured_hand_object_force"]]=measured_force.reshape(-1); out[s["supporting_object_net_force"]]=support; out[s["relative_contact_motion"]]=self.last_relative_motion.reshape(-1); return np.clip(out,-5,5).astype(np.float32)
    def step(self,action):
        action=np.asarray(action,dtype=float); _,q_before,_,obj_before,quat_before,key_before=self._state(); command=rate_limited_command(self.previous_command,action,self.lower,self.upper,self.rate_per_second,measured_qpos=q_before,control_timestep=self.clock.control_timestep,max_tracking_error=self.max_tracking_error); self.data=self.data.replace(ctrl=self.jax.numpy.asarray(command[None,:]));
        for _ in range(self.clock.physics_substeps_per_control): self.data=self.step_fn(self.data)
        self.previous_command,self.step_index=command,self.step_index+1; physical=self.producer.extract(self.data); self.last_physical=physical; _,q,qvel,obj,quat,key=self._state(); prox,anchors,conf,witnesses=self._actual_collision(q,qvel,obj,quat); i=min(self.step_index,len(self.aligned_q_ref)-1); target=self.aligned_object_pos[i]; target_vel=(self.aligned_object_pos[i]-self.aligned_object_pos[max(0,i-1)])/self.clock.control_timestep; rel_motion=(key-key_before-(obj-obj_before))/self.clock.control_timestep; force=np.asarray(physical.hand_object_force_on_object_world_N[0]); rel_motion*= (np.linalg.norm(force,axis=1)>.02)[:,None]; terms=dense_autonomous_reward(obj,target,np.asarray(physical.object_linear_velocity[0]),key[0]-obj,self.reference_keypoints[i,0]-target,prox,self.previous_proximity,action,phase=i/max(1,len(self.aligned_q_ref)-1),measured_hand_object_force=force,supporting_object_force=force.sum(axis=0),relative_contact_motion=rel_motion,release_active=i>=self.release_start,target_object_velocity=target_vel,reference_proximity=self.reference_proximity[i],anchors=anchors,reference_anchors=self.reference_anchors[i],measured_q=q,reference_q=self.aligned_q_ref[i],object_quat=quat,reference_object_quat=self.trajectory.object_quat_xyzw[i]); self.previous_proximity,self.previous_keypoints,self.last_relative_motion=prox,key.copy(),rel_motion.copy(); self.max_object_z=max(self.max_object_z,float(obj[2])); dropped=bool(obj[2] < FLOOR_TOP_Z-.03); diverged=bool(i>0 and np.linalg.norm(obj-target)>.30); done,task_success,phase=terminal_status(self.step_index,len(self.aligned_q_ref)-1,dropped=dropped,diverged=diverged,actual_peak_lift=self.max_object_z-self.aligned_object_pos[0,2],target_peak_lift=self.target_peak_lift,path_error=float(np.linalg.norm(obj-target))); info={"command":command.copy(),"step":self.step_index,"identity":self.trajectory.identity.identity,"object_position":obj.copy(),"target_object_position":target.copy(),"surface_proximity":prox.copy(),"reference_surface_proximity":self.reference_proximity[i].copy(),"surface_anchor_local":anchors.copy(),"reference_surface_anchor_local":self.reference_anchors[i].copy(),"hand_object_force":force.copy(),"supporting_object_net_force":np.asarray(physical.hand_object_force_on_object_world_N[0]).sum(axis=0).copy(),"contact_count":int(np.asarray(physical.contact_count)[0]),"qpos":q.copy(),"qvel":qvel.copy(),"failure_phase":phase,"task_success":task_success,"target_peak_lift":self.target_peak_lift,"actual_peak_lift":float(self.max_object_z-self.aligned_object_pos[0,2]),"path_error":float(np.linalg.norm(obj-target)),"lift_error":float(abs(obj[2]-target[2])),"relative_contact_motion":rel_motion.copy()}; return AutonomousStep(self._observation(),float(sum(terms.values())),terms,done,info)
