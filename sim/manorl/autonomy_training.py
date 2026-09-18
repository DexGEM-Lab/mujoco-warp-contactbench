"""v4 raw-observation policy integration; training entrypoints remain disabled.

PPO memory stores raw 957 observations. PointNet is an ordinary registered Torch
module and builds the 829 actor/value feature inside the model.
"""
from __future__ import annotations
import hashlib,json,math,random
from typing import Any
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from skrl.models.torch import Model
from skrl.models.torch.gaussian import GaussianMixin
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.memories.torch import RandomMemory
from sim.manorl.rl_games_ppo import RlGamesPPO
from skrl.agents.torch.ppo.ppo import PPO_CFG
from sim.manorl.autonomy_contracts import ACTION_DIM, OBSERVATION_DIM, RAW_OBSERVATION_DIM, ENCODED_OBSERVATION_DIM
from sim.manorl.model import PointNetEncoder
from sim.manorl.trajectory_package import TrajectoryCatalog
from sim.manorl.autonomy_v4 import _gather, reference_lengths

RESERVED_TRAIN_IDENTITIES=("cube2_02_2833","cube2_02_2835","cube2_02_2837")
SPLIT_CONTRACT_ID="manorl.autonomy.identity_split.v1"
TRAINING_CONTRACT_ID="manorl.autonomy.training.v4.disabled"
ACTOR_CRITIC_ARCHITECTURE_ID="manorl.autonomy.actor_critic.v4.pointnet"

TEACHER_FLEX_JOINTS = (7, 9, 10, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27)
TEACHER_SQUEEZE_RAD = 0.2
TEACHER_CONTACT_INTENT_THRESHOLD = 0.5


def teacher_anchor_metadata(beta: float = 0.0, passes: int = 2) -> dict[str, Any]:
    """Training supervision only; these targets never enter physical execution."""
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("teacher-anchor-beta must be finite and nonnegative")
    if beta > 0 and (type(passes) is not int or passes < 1):
        raise ValueError("teacher-anchor-passes must be a positive integer when beta > 0")
    return {"beta": beta, "passes": passes, "squeeze_rad": TEACHER_SQUEEZE_RAD,
            "gate": "current_reference_max_proximity_confidence_valid",
            "contact_intent_threshold": TEACHER_CONTACT_INTENT_THRESHOLD,
            "gate_comparison": ">=", "flex_joints": list(TEACHER_FLEX_JOINTS),
            "role": "training_supervision_only"}


def identity_split(catalog: TrajectoryCatalog, *, seed:int=0)->dict[str,Any]:
    identities=tuple(t.identity.identity for t in catalog.trajectories)
    if not identities:
        raise ValueError("trajectory catalog must contain at least one identity")
    reserved_names=[x for x in RESERVED_TRAIN_IDENTITIES if x in identities]
    if len(reserved_names)==len(RESERVED_TRAIN_IDENTITIES):
        reserved=[identities.index(x) for x in reserved_names]
    else:
        # New object/action packages do not contain the cube2 seed witnesses.
        # Keep a deterministic three-identity TRAIN anchor without inventing a
        # compatibility identity; all-reference training does not use this
        # split for selection, but provenance still needs a stable contract.
        reserved=list(range(min(3,len(identities))))
    rest=[i for i in range(len(identities)) if i not in reserved]
    perm=np.random.default_rng(seed).permutation(rest).tolist()
    payload={"contract":SPLIT_CONTRACT_ID,"seed":seed,"train_indices":reserved+perm[:37],"validation_indices":perm[37:42],"test_indices":perm[42:47]}
    payload["digest"]=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(); return payload

class BatchedAutonomyAdapter:
    """Raw-957 adapter. CUDA borrows tensors through DLPack; CPU is test-only."""
    def __init__(self,trajectory,*,num_envs=1,device="cpu",seed=0,**kwargs):
        from sim.manorl.autonomy_batch import BatchedAutonomyRuntime
        self.runtime=BatchedAutonomyRuntime(trajectory,num_envs=num_envs,device=device,seed=seed,**kwargs)
        self.num_envs=self.runtime.num_envs; self.device_name=device; self._device=torch.device("cuda" if device=="gpu" else "cpu")
        self.observation_dim=RAW_OBSERVATION_DIM; self.action_dim=ACTION_DIM
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(RAW_OBSERVATION_DIM,),dtype=np.float32)
        self.action_space=gym.spaces.Box(-1.,1.,shape=(ACTION_DIM,),dtype=np.float32)
    @property
    def device(self): return self._device
    def _to_torch(self,x):
        if self.device_name=="gpu":
            from sim.manorl.device_runtime import jax_to_torch_cuda
            return jax_to_torch_cuda(x)
        return torch.as_tensor(np.array(x,copy=True),dtype=torch.float32,device=self._device)
    def reset(self,*,mask=None):
        return self._to_torch(self.runtime.reset(mask)),{"num_envs":self.num_envs,"contract":"manorl.autonomy.v4"}
    def prepare_action(self): return self._to_torch(self.runtime.prepare_action())
    def teacher_actions(self, squeeze_rad=TEACHER_SQUEEZE_RAD, contact_intent_threshold=TEACHER_CONTACT_INTENT_THRESHOLD):
        """Analytical chase labels at the current pre-action state, never controls."""
        runtime = self.runtime; jp = runtime.jp
        index = _gather(runtime.cache, runtime.indices, 1, getattr(runtime, "env_ref", None))
        current = _gather(runtime.cache, runtime.indices, env_ref=getattr(runtime, "env_ref", None))
        intent = jp.max(jp.asarray(runtime.cache.proximity)[current] *
                        jp.asarray(runtime.cache.confidence)[current] *
                        jp.asarray(runtime.cache.valid)[current], axis=-1)
        squeeze = jp.zeros((ACTION_DIM,), dtype=jp.float32).at[jp.asarray(TEACHER_FLEX_JOINTS)].set(squeeze_rad)
        target = jp.asarray(runtime.cache.q_feasible)[index] + (intent >= contact_intent_threshold)[:, None] * squeeze
        target = jp.clip(target, jp.asarray(runtime.lower), jp.asarray(runtime.upper))
        actions = jp.clip((target - runtime.previous_command) /
                          (jp.asarray(runtime.rate) * runtime.cache.control_timestep), -1., 1.)
        return self._to_torch(actions)

    def compact_summary(self):
        """Legacy compact fields from already-cached post-transition physics.

        ``object_motion`` historically means the signed sum of world-origin Z,
        not a lift delta.  Do not call ``telemetry_snapshot`` here: collection
        already captures that richer snapshot once per transition.
        """
        jp=self.runtime.jp; physical=self.runtime.last_physical; contact=self.runtime.last_contact
        index=_gather(self.runtime.cache,self.runtime.indices,env_ref=getattr(self.runtime,"env_ref",None))
        target_object=jp.asarray(self.runtime.cache.object_origin)[index]
        return {"object_motion": self._to_torch(physical.object_origin[:,2]).sum(),
                "contact_force": self._to_torch(jp.linalg.norm(contact.paired_force_on_object,axis=-1).sum(axis=-1)).sum(),
                "path": self._to_torch(jp.linalg.norm(physical.object_origin-target_object,axis=-1)).sum()}

    def telemetry_snapshot(self, raw_actions):
        """Post-transition physical/reward facts, still resident on the runtime device."""
        jp=self.runtime.jp; physical=self.runtime.last_physical; contact=self.runtime.last_contact; reward=self.runtime.last_reward
        index=_gather(self.runtime.cache,self.runtime.indices,env_ref=getattr(self.runtime,"env_ref",None)); cache=self.runtime.cache
        target_object=jp.asarray(cache.object_origin)[index]; target_palm=jp.asarray(cache.palm_origin)[index]
        target_raw=jp.asarray(cache.q_raw)[index]; target_feasible=jp.asarray(cache.q_feasible)[index]
        paired_norm=jp.linalg.norm(contact.paired_force_on_object,axis=-1); loaded=paired_norm>.02
        if raw_actions.shape != (self.num_envs, ACTION_DIM):
            raise ValueError("v4 raw actions must be (num_envs,28)")
        # Raw Normal samples stay in Torch: jp.asarray(CUDA Tensor) would take
        # NumPy's host path. Only physical execution crosses JAX/Torch via DLPack.
        raw_action_abs=raw_actions.detach().abs()
        executed=torch.clamp(raw_actions.detach(),-1.,1.)
        terms=jp.stack((reward.object_position,reward.object_rotation,reward.object_velocity,reward.hand_relative,
                        reward.fingers,reward.geometry,reward.action,reward.survival,reward.severe),axis=1)
        snapshot={
            "reward_terms":terms, "reward_total":reward.total, "reason":reward.reason, "valid":reward.valid,
            "position_error_abs":jp.abs(physical.object_origin-target_object),
            "object_rotation_error_rad": 2*jp.arccos(jp.clip(jp.abs(jp.sum(physical.object_quat_xyzw*jp.asarray(cache.object_quat_xyzw)[index],axis=-1)),0.,1.)),
            "palm_position_error":jp.linalg.norm(physical.palm_origin-target_palm,axis=-1),
            "finger_raw_error":jp.sqrt(jp.mean((physical.q_raw[:,6:]-target_raw[:,6:])**2,axis=-1)),
            "finger_feasible_error":jp.sqrt(jp.mean((physical.q_raw[:,6:]-target_feasible[:,6:])**2,axis=-1)),
            "bottom_clearance":physical.object_bottom-cache.table_height,
            "reference_bottom_clearance":jp.asarray(cache.reference_bottom)[index]-cache.table_height,
            "origin_lift_delta":physical.object_origin[:,2]-jp.asarray(cache.object_origin)[_gather(cache,jp.zeros_like(self.runtime.indices),env_ref=getattr(self.runtime,"env_ref",None))][:,2],
            "paired_loaded":loaded, "paired_active":contact.paired_count>0,
            "paired_contact_count":contact.paired_count.sum(axis=-1), "paired_force_norm":paired_norm.sum(axis=-1),
            "object_all_force_norm":jp.linalg.norm(contact.object_all_force,axis=-1),
            "paired_torque_com_norm":jp.linalg.norm(contact.paired_torque_com,axis=-1),
            # Contact reduction owns a [B,16,3] tangential velocity; this L2
            # produces exactly one scalar slip speed per hand region.
            "tangential_slip":jp.linalg.norm(contact.tangential_slip,axis=-1),
            "airborne":physical.object_bottom > cache.table_height+.005,
            "action_raw_abs_sum":raw_action_abs.sum(dim=-1), "action_raw_abs_max":raw_action_abs.amax(dim=-1),
            "action_raw_abs_denominator":torch.full((self.num_envs,), ACTION_DIM, dtype=raw_actions.dtype, device=raw_actions.device),
            "action_executed_norm":torch.linalg.vector_norm(executed,dim=-1),
            "action_clipped":(raw_action_abs>1.).to(raw_actions.dtype).sum(dim=-1),
            "action_denominator":torch.full((self.num_envs,), ACTION_DIM, dtype=raw_actions.dtype, device=raw_actions.device),
            "command_envelope_utilization":jp.mean(jp.abs(physical.command_error),axis=-1),
            "antiwindup_active":jp.any(jp.abs(physical.command_error)>=1.,axis=-1),
            "reference_progress":jp.minimum(self.runtime.indices,reference_lengths(cache,getattr(self.runtime,"env_ref",None))-1)/jp.asarray(jp.maximum(1,reference_lengths(cache,getattr(self.runtime,"env_ref",None))-1),jp.float32),
        }
        return {name:value if isinstance(value,torch.Tensor) else self._to_torch(value) for name,value in snapshot.items()}
    def step(self,actions):
        # Preserve raw Gaussian actions for likelihood; physical boundary clips.
        if not isinstance(actions,torch.Tensor): actions=torch.as_tensor(actions,dtype=torch.float32,device=self._device)
        if actions.shape!=(self.num_envs,ACTION_DIM): raise ValueError("v4 actions must be (num_envs,28)")
        physical=torch.clamp(actions,-1.,1.)
        if self.device_name=="gpu":
            from sim.manorl.device_runtime import torch_to_jax_cuda
            jax_actions=torch_to_jax_cuda(physical)
        else: jax_actions=np.asarray(physical.detach(),np.float32)
        obs,reward,done,info=self.runtime.step(jax_actions)
        self._last_raw_actions = actions.detach()
        return self._to_torch(obs),self._to_torch(reward).reshape(self.num_envs,1),self._to_torch(done).reshape(self.num_envs,1).bool(),info

def actor_critic_architecture(*, separate_critic: bool = False) -> dict[str, Any]:
    return {"id": ACTOR_CRITIC_ARCHITECTURE_ID, "raw_observation_dim": RAW_OBSERVATION_DIM,
            "encoded_feature_dim": ENCODED_OBSERVATION_DIM,
            "pointnet": "3-64-128-256-max-256-64",
            "value_trunk": "separate" if separate_critic else "shared", "action_dim": ACTION_DIM}


class AutonomyActorCritic(GaussianMixin,DeterministicMixin,Model):
    """Every seven raw blocks feed both policy and value after PointNet."""
    def __init__(self,observation_space,action_space,device="cpu",*,separate_critic=False,clip_actions=False):
        Model.__init__(self,observation_space=observation_space,state_space=None,action_space=action_space,device=device)
        self.observation_dim=int(self.num_observations); self.action_dim=int(self.num_actions)
        if (self.observation_dim,self.action_dim)!=(RAW_OBSERVATION_DIM,ACTION_DIM): raise ValueError("v4 actor requires raw 957 and 28 actions")
        GaussianMixin.__init__(self,clip_actions=clip_actions,clip_mean_actions=False,clip_log_std=True,min_log_std=-5.,max_log_std=2.,reduction="sum",role="policy")
        DeterministicMixin.__init__(self,clip_actions=False,role="value")
        self.pointnet=PointNetEncoder(device)
        self.net=nn.Sequential(nn.Linear(ENCODED_OBSERVATION_DIM,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh()).to(device)
        self.value_net=nn.Sequential(nn.Linear(ENCODED_OBSERVATION_DIM,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh()).to(device) if separate_critic else None
        self.separate_critic=bool(separate_critic)
        self.mean=nn.Linear(128,ACTION_DIM).to(device); self.value=nn.Linear(128,1).to(device); self.log_std=nn.Parameter(torch.full((ACTION_DIM,),-1.,device=device))
    def _encoded(self,x):
        if x.ndim!=2 or x.shape[1]!=RAW_OBSERVATION_DIM: raise ValueError("v4 model requires (batch,957) raw observations")
        cloud=x[:,703:895].reshape(-1,64,3); embedding=self.pointnet(cloud)
        return torch.cat((x[:,:703],embedding,x[:,895:]),dim=-1)
    def checkpoint_architecture(self): return actor_critic_architecture(separate_critic=self.separate_critic)
    def act(self,inputs,role=""):
        if role=="policy": return GaussianMixin.act(self,inputs,role=role)
        if role=="value": return DeterministicMixin.act(self,inputs,role=role)
        raise ValueError("role must be policy or value")
    def compute(self,inputs,role=""):
        features=self._encoded(inputs["observations"])
        if role=="policy":
            mean=self.mean(self.net(features)); return mean,{"log_std":self.log_std.expand_as(mean)}
        if role=="value": return self.value((self.value_net or self.net)(features)),{}
        raise ValueError("role must be policy or value")

def validate_learning_rate(learning_rate: float) -> None:
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")

def v4_ppo_config(*, rollouts:int, learning_epochs:int, mini_batches:int, learning_rate:float=3e-4) -> dict[str, Any]:
    """Installed PPO defaults plus the unchanged v4 overrides, in one source."""
    validate_learning_rate(learning_rate)
    defaults=PPO_CFG()
    return {"rollouts":rollouts,"learning_epochs":learning_epochs,"mini_batches":mini_batches,
            "discount_factor":defaults.discount_factor,"gae_lambda":defaults.gae_lambda,
            "learning_rate":learning_rate,"ratio_clip":defaults.ratio_clip,"value_clip":defaults.value_clip,
            "entropy_loss_scale":.001,"value_loss_scale":.5,"learning_starts":defaults.learning_starts,
            "grad_norm_clip":defaults.grad_norm_clip,"time_limit_bootstrap":True,
            "experiment":{"write_interval":0,"checkpoint_interval":0},"mixed_precision":False}


def resolved_v4_ppo_config(agent) -> dict[str, Any]:
    """Report the instantiated agent rather than a parallel config literal."""
    cfg=agent.cfg; optimizer=agent.optimizer.param_groups[0]
    return {"discount_factor":cfg.discount_factor,"gae_lambda":cfg.gae_lambda,"learning_rate":optimizer["lr"],
            "ratio_clip":cfg.ratio_clip,"value_clip":cfg.value_clip,"entropy_loss_scale":cfg.entropy_loss_scale,
            "value_loss_scale":cfg.value_loss_scale,"mixed_precision":cfg.mixed_precision,
            "normalize_observations":cfg.observation_preprocessor is not None,
            "normalize_values":cfg.value_preprocessor is not None,
            # skrl 2.1.0 compute_gae standardizes advantages unconditionally.
            "normalize_advantages":True,
            "grad_norm_clip":cfg.grad_norm_clip,"optimizer":type(agent.optimizer).__name__,
            "adam_betas":list(optimizer["betas"]),"adam_eps":optimizer["eps"],
            "learning_starts":cfg.learning_starts,"time_limit_bootstrap":cfg.time_limit_bootstrap,
            "learning_epochs":cfg.learning_epochs,"mini_batches":cfg.mini_batches,"rollouts":cfg.rollouts}


def build_batched_runtime(adapter, *, rollouts:int, learning_epochs:int, mini_batches:int, device:str, separate_critic:bool=False, learning_rate:float=3e-4):
    """Canonical RlGamesPPO over raw-957 storage and the v4 PointNet model."""
    cfg=v4_ppo_config(rollouts=rollouts,learning_epochs=learning_epochs,mini_batches=mini_batches,learning_rate=learning_rate)
    memory=RandomMemory(memory_size=rollouts,num_envs=adapter.num_envs,device=device)
    model=AutonomyActorCritic(adapter.observation_space,adapter.action_space,device=device,separate_critic=separate_critic,clip_actions=False)
    agent=RlGamesPPO(models={"policy":model,"value":model},memory=memory,
                     observation_space=adapter.observation_space,state_space=None,
                     action_space=adapter.action_space,device=device,cfg=cfg)
    agent.init(); return model,agent

def seed_everything(seed:int): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
