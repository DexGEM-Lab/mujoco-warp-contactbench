"""MJX-Warp-only cube2 autonomy v4 runtime.

Legacy 538-D reference-witness code was removed because it used native MjData
and reinterpreted fields.  This module is the one production transition path.
"""
from __future__ import annotations
from typing import Any, NamedTuple
import numpy as np
from sim.manorl.autonomy_contracts import ACTION_DIM, AUTONOMY_VERSION, OBSERVATION_DIM
from sim.manorl.autonomy_v4 import (ReferenceCacheV4, V4Contact, compile_reference_cache_v4,
    extract_v4_physical, reduce_pyramidal_contacts_v4, build_raw_observation,
    compute_reward, DOF_RATE, ANTIWINDUP_ERROR)

V4_OBSERVATION_DIM=OBSERVATION_DIM
V3_OBSERVATION_DIM=V4_OBSERVATION_DIM # import-only migration alias, never 538

class AutonomyTransitionState(NamedTuple):
    data: Any; indices: Any; pending_reset: Any; previous_command: Any

class BatchedAutonomyRuntime:
    """Pinned cube2 v4 transition: 4 Warp steps then exactly one Warp forward."""
    def __init__(self, trajectory, *, num_envs: int=1, device: str="cpu", seed: int=0, **_: Any):
        if num_envs != 1: raise ValueError("v4 vertical slice is intentionally N=1")
        if device not in {"cpu","gpu"}: raise ValueError("device must be cpu or gpu")
        import jax, jax.numpy as j
        from mujoco import mjx
        from sim.manorl.assets import compile_model_metadata_only, object_collision_vertices
        from sim.manorl.environment import MjxWarpPhysicalProducer
        self.jax,self.jp,self.mjx=jax,j,mjx; self.num_envs=1; self.device_name=device; self.trajectory=trajectory
        self.mujoco,self.model=compile_model_metadata_only(object_type="cube2",hand_side="right",physics_timestep=1/480)
        self.device=jax.devices(device)[0]; self.mjx_model=mjx.put_model(self.model,device=self.device,impl="warp")
        if getattr(self.mjx_model,"_impl",None) is None: raise RuntimeError("v4 requires mjx.put_model(..., impl='warp')")
        self.producer=MjxWarpPhysicalProducer(self.mujoco,self.model,object_type="cube2",hand_sides=("right",))
        self.cache=compile_reference_cache_v4(trajectory,device=device)
        self.length=len(self.cache.q_feasible); self.lower=np.asarray(self.model.jnt_range[:28,0],np.float32); self.upper=np.asarray(self.model.jnt_range[:28,1],np.float32)
        self.rate=DOF_RATE.copy(); self.envelope=ANTIWINDUP_ERROR.copy()
        self.antiwindup=ANTIWINDUP_ERROR.copy()
        base=mjx.make_data(self.model,device=self.device,impl="warp",naconmax=128,njmax=512)
        qpos=np.asarray(base.qpos).copy(); qpos[:28]=self.cache.q_feasible[0]; qpos[self.producer.object_qpos_address:self.producer.object_qpos_address+3]=self.cache.object_origin[0]; qpos[self.producer.object_qpos_address+3:self.producer.object_qpos_address+7]=self.cache.object_quat_xyzw[0,(3,0,1,2)]
        self.initial=base.replace(qpos=j.asarray(qpos),ctrl=j.asarray(np.r_[self.cache.q_feasible[0],np.asarray(base.ctrl)[28:]].copy()))
        self.data=jax.vmap(lambda _: mjx.forward(self.mjx_model,self.initial))(j.arange(1)); self.indices=j.zeros((1,),j.int32); self.pending_reset=j.zeros((1,),bool); self.previous_command=j.asarray(self.cache.q_feasible[:1],j.float32)
        # Static collision vertices give exact cube2 bottom after the same forward.
        self.object_vertices=j.asarray(object_collision_vertices("cube2"),j.float32)
        # Runtime emits raw 957; trainable PointNet belongs to the next model slice.
        self._transition_fn=jax.jit(self._transition)
        self._refresh(False,j.zeros((1,28),j.float32))

    def _physical(self,data,previous):
        objq=data.xquat[:,self.producer.object_body_id][...,(1,2,3,0)]
        from sim.manorl.autonomy_v4 import quat_rotate
        bottom=self.jp.min(data.xpos[:,self.producer.object_body_id,None,2]+quat_rotate(objq[:,None],self.object_vertices)[:,:,2],axis=1)
        return extract_v4_physical(qpos=data.qpos,qvel=data.qvel,xpos=data.xpos,xquat=data.xquat,xipos=data.xipos,subtree_com=data.subtree_com,cvel=data.cvel,body_rootid=self.model.body_rootid,hand_dof_address=0,object_body_id=self.producer.object_body_id,palm_body_id=self.producer.keypoint_body_ids[0],region_body_ids=self.producer.keypoint_body_ids,object_bottom=bottom,lower=self.jp.asarray(self.lower),upper=self.jp.asarray(self.upper),previous_command=previous,dof_rate=self.jp.asarray(self.rate),antiwindup=self.jp.asarray(self.antiwindup))
    def _contact(self,data,physical):
        x=data._impl; required=("nacon","nefc","contact__geom","contact__worldid","contact__dim","contact__efc_address","contact__friction","contact__frame","contact__pos","efc__force")
        if any(not hasattr(x,n) for n in required): raise RuntimeError("pinned MJX-Warp contact ABI unavailable")
        allf,pair,torque,count,valid=reduce_pyramidal_contacts_v4(nacon=x.nacon,nefc=x.nefc,geom=x.contact__geom,world=x.contact__worldid,dimension=x.contact__dim,addresses=x.contact__efc_address,friction=x.contact__friction,frame=x.contact__frame,position=x.contact__pos,constraint_force=x.efc__force,ngeom=self.model.ngeom,hand_geom_ids=self.producer.keypoint_geom_ids,object_geom_ids=tuple(self.producer.object_geom_ids),object_com=physical.object_com)
        return V4Contact(allf,pair,torque,count,self.jp.zeros((1,16,3)),valid)
    def _observe(self,data,index,previous):
        physical=self._physical(data,previous); contact=self._contact(data,physical); raw=build_raw_observation(physical,contact,self.cache,index,previous); return physical,contact,raw,raw
    def _transition(self,data,index,previous,action,execute):
        import jax
        physical=self._physical(data,previous)
        command=previous+self.jp.clip(action,-1,1)*self.jp.asarray(self.rate)[None]/120
        command=self.jp.clip(self.jp.clip(command,physical.q_raw-self.jp.asarray(self.envelope),physical.q_raw+self.jp.asarray(self.envelope)),self.jp.asarray(self.lower),self.jp.asarray(self.upper))
        stepped=data.replace(ctrl=command)
        def advance(x):
            for _ in range(4): x=jax.vmap(lambda row:self.mjx.step(self.mjx_model,row))(x)
            return jax.vmap(lambda row:self.mjx.forward(self.mjx_model,row))(x)
        next_data=jax.lax.cond(execute,advance,lambda x:x,stepped)
        next_index=index+self.jp.asarray(execute,self.jp.int32); p,c,raw,obs=self._observe(next_data,next_index,command); reward=compute_reward(p,c,self.cache,next_index,self.jp.clip(action,-1,1)); valid=p.valid&c.valid&reward.valid
        return next_data,next_index,command,raw,obs,reward,valid,c
    def _refresh(self,execute,action):
        self.data,self.indices,self.previous_command,self.raw_observation,self.observation,self.last_reward,self.last_valid,self.last_contact=self._transition_fn(self.data,self.indices,self.previous_command,action,execute)
        self.last_done=self.last_reward.done; self.last_reason=self.last_reward.reason; self.last_command=self.previous_command
    def reset(self,mask=None):
        if mask is not None and not bool(np.asarray(mask)[0]): return self.observation
        self.data=self.jax.vmap(lambda _:self.mjx.forward(self.mjx_model,self.initial))(self.jp.arange(1)); self.indices=self.jp.zeros((1,),self.jp.int32); self.previous_command=self.jp.asarray(self.cache.q_feasible[:1],self.jp.float32); self._refresh(False,self.jp.zeros((1,28),self.jp.float32)); return self.observation
    def prepare_action(self):
        if bool(np.asarray(self.last_done)[0]): self.reset()
        return self.observation
    def step(self,actions):
        action=self.jp.asarray(actions)
        if action.shape!=(1,28): raise ValueError("v4 requires actions shape (1,28)")
        self._refresh(True,action)
        return self.observation,self.last_reward.total,self.last_done,{"valid":self.last_valid,"reason_code":self.last_reason,"command":self.last_command,"raw_observation":self.raw_observation,"contact":self.last_contact,"contract":AUTONOMY_VERSION}

# Formula-level API retained as deliberate public v4 names.
__all__=["ReferenceCacheV4","compile_reference_cache_v4","V4Contact","V4_OBSERVATION_DIM","V3_OBSERVATION_DIM","AutonomyTransitionState","BatchedAutonomyRuntime"]
