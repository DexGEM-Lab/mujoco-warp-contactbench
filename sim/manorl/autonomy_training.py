"""v4 raw-observation policy integration; training entrypoints remain disabled.

PPO memory stores raw 957 observations. PointNet is an ordinary registered Torch
module and builds the 829 actor/value feature inside the model.
"""
from __future__ import annotations
import hashlib,json,random
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
from sim.manorl.autonomy_contracts import ACTION_DIM, OBSERVATION_DIM, RAW_OBSERVATION_DIM, ENCODED_OBSERVATION_DIM
from sim.manorl.model import PointNetEncoder
from sim.manorl.trajectory_package import TrajectoryCatalog

RESERVED_TRAIN_IDENTITIES=("cube2_02_2833","cube2_02_2835","cube2_02_2837")
SPLIT_CONTRACT_ID="manorl.autonomy.identity_split.v1"
TRAINING_CONTRACT_ID="manorl.autonomy.training.v4.disabled"
ACTOR_CRITIC_ARCHITECTURE_ID="manorl.autonomy.actor_critic.v4.pointnet"

def identity_split(catalog: TrajectoryCatalog, *, seed:int=0)->dict[str,Any]:
    identities=tuple(t.identity.identity for t in catalog.trajectories); missing=[x for x in RESERVED_TRAIN_IDENTITIES if x not in identities]
    if missing: raise ValueError(f"reserved TRAIN identities absent: {missing}")
    reserved=[identities.index(x) for x in RESERVED_TRAIN_IDENTITIES]; rest=[i for i in range(len(identities)) if i not in reserved]
    perm=np.random.default_rng(seed).permutation(rest).tolist(); payload={"contract":SPLIT_CONTRACT_ID,"seed":seed,"train_indices":reserved+perm[:37],"validation_indices":perm[37:42],"test_indices":perm[42:47]}
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
    def compact_summary(self):
        """Legacy compact fields retained for existing callers."""
        sample=self.telemetry_snapshot(self._last_raw_actions)
        return {"object_motion": sample["origin_lift_delta"].sum(),
                "contact_force": sample["paired_force_norm"].sum(),
                "path": torch.linalg.vector_norm(sample["position_error_abs"], dim=1).sum()}

    def telemetry_snapshot(self, raw_actions):
        """Post-transition physical/reward facts, still resident on the runtime device."""
        jp=self.runtime.jp; physical=self.runtime.last_physical; contact=self.runtime.last_contact; reward=self.runtime.last_reward
        index=jp.minimum(self.runtime.indices,self.runtime.length-1); cache=self.runtime.cache
        target_object=jp.asarray(cache.object_origin)[index]; target_palm=jp.asarray(cache.palm_origin)[index]
        target_raw=jp.asarray(cache.q_raw)[index]; target_feasible=jp.asarray(cache.q_feasible)[index]
        paired_norm=jp.linalg.norm(contact.paired_force_on_object,axis=-1); loaded=paired_norm>.02
        action=jp.asarray(raw_actions); executed=jp.clip(action,-1.,1.)
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
            "origin_lift_delta":physical.object_origin[:,2]-jp.asarray(cache.object_origin)[0,2],
            "paired_loaded":loaded, "paired_active":contact.paired_count>0,
            "paired_contact_count":contact.paired_count.sum(axis=-1), "paired_force_norm":paired_norm.sum(axis=-1),
            "object_all_force_norm":jp.linalg.norm(contact.object_all_force,axis=-1),
            "paired_torque_com_norm":jp.linalg.norm(contact.paired_torque_com,axis=-1),
            "tangential_slip":jp.linalg.norm(contact.tangential_slip,axis=-1),
            "airborne":physical.object_bottom > cache.table_height+.005,
            "action_raw_abs":jp.mean(jp.abs(action),axis=-1), "action_executed_norm":jp.linalg.norm(executed,axis=-1),
            "action_clipped":jp.mean((jp.abs(action)>1.).astype(jp.float32),axis=-1),
            "command_envelope_utilization":jp.mean(jp.abs(physical.command_error),axis=-1),
            "antiwindup_active":jp.any(jp.abs(physical.command_error)>=1.,axis=-1),
            "reference_progress":index/jp.asarray(max(1,self.runtime.length-1),jp.float32),
        }
        return {name:self._to_torch(value) for name,value in snapshot.items()}
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

def build_batched_runtime(adapter, *, rollouts:int, learning_epochs:int, mini_batches:int, device:str, separate_critic:bool=False):
    """Canonical RlGamesPPO over raw-957 storage and the v4 PointNet model."""
    memory=RandomMemory(memory_size=rollouts,num_envs=adapter.num_envs,device=device)
    model=AutonomyActorCritic(adapter.observation_space,adapter.action_space,device=device,separate_critic=separate_critic,clip_actions=False)
    cfg={"rollouts":rollouts,"learning_epochs":learning_epochs,"mini_batches":mini_batches,
         "discount_factor":.99,"gae_lambda":.95,"learning_rate":3e-4,"ratio_clip":.2,
         "value_clip":.2,"entropy_loss_scale":.001,"value_loss_scale":.5,"learning_starts":0,
         "time_limit_bootstrap":True,"experiment":{"write_interval":0,"checkpoint_interval":0},"mixed_precision":False}
    agent=RlGamesPPO(models={"policy":model,"value":model},memory=memory,
                     observation_space=adapter.observation_space,state_space=None,
                     action_space=adapter.action_space,device=device,cfg=cfg)
    agent.init(); return model,agent

def seed_everything(seed:int): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
