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
from sim.manorl.autonomy_contracts import (
    ACTION_CONTRACT_ID,
    ACTION_DIM,
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_V5,
    ENCODED_OBSERVATION_DIM,
    ENCODED_OBSERVATION_DIM_V5,
    OBSERVATION_CONTRACT_ID,
    OBSERVATION_CONTRACT_ID_V5,
    OBSERVATION_DIM,
    RAW_OBSERVATION_DIM,
    RAW_OBSERVATION_DIM_V5,
    RAW_OBSERVATION_DIM_V6,
    REWARD_CONTRACT_ID,
    V4_REWARD_TERM_NAMES,
)
from sim.manorl.autonomy_v6_model import (
    AutonomyActorCriticV6,
    actor_critic_architecture_v6,
)
from sim.manorl.autonomy_v5_intermediate_model import (
    AutonomyActorCriticV525,
    AutonomyActorCriticV55,
    AutonomyActorCriticV575,
    actor_critic_architecture_v525,
    actor_critic_architecture_v55,
    actor_critic_architecture_v575,
)
from sim.manorl.model import PointNetEncoder
from sim.manorl.trajectory_package import TrajectoryCatalog
from sim.manorl.autonomy_v4 import _gather, reference_lengths

RESERVED_TRAIN_IDENTITIES=("cube2_02_2833","cube2_02_2835","cube2_02_2837")
SPLIT_CONTRACT_ID="manorl.autonomy.identity_split.v1"
TRAINING_CONTRACT_ID="manorl.autonomy.training.v4.disabled"
ACTOR_CRITIC_ARCHITECTURE_ID="manorl.autonomy.actor_critic.v4.pointnet"
ACTOR_CRITIC_ARCHITECTURE_ID_V5="manorl.autonomy.actor_critic.v5.pointcloud"

TEACHER_FLEX_JOINTS = (7, 9, 10, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27)
TEACHER_SQUEEZE_RAD = 0.2
TEACHER_CONTACT_INTENT_THRESHOLD = 0.5

_DEVICE_TELEMETRY_WIDTHS = (
    ("reward_terms", None),
    ("reward_total", 1),
    ("reason", 1),
    ("valid", 1),
    ("position_error_abs", 3),
    ("object_rotation_error_rad", 1),
    ("palm_position_error", 1),
    ("finger_raw_error", 1),
    ("finger_feasible_error", 1),
    ("bottom_clearance", 1),
    ("reference_bottom_clearance", 1),
    ("origin_lift_delta", 1),
    ("paired_loaded", 16),
    ("paired_active", 16),
    ("paired_contact_count", 1),
    ("paired_force_norm", 1),
    ("object_all_force_norm", 1),
    ("paired_torque_com_norm", 1),
    ("tangential_slip", 16),
    ("airborne", 1),
    ("opposing_loaded", 1),
    ("positive_lift", 1),
    ("stable_airborne", 1),
    ("command_envelope_utilization", 1),
    ("antiwindup_active", 1),
    ("reference_progress", 1),
    ("object_world_z", 1),
)


def _telemetry_slices(reward_names):
    """Describe the immutable columns in the single device telemetry buffer."""
    result = {}
    offset = 0
    for name, configured_width in _DEVICE_TELEMETRY_WIDTHS:
        width = len(reward_names) if configured_width is None else configured_width
        result[name] = slice(offset, offset + width)
        offset += width
    return result, offset


def teacher_anchor_metadata(
    beta: float = 0.0,
    passes: int = 2,
    *,
    final_beta: float | None = None,
    schedule_updates: int | None = None,
) -> dict[str, Any]:
    """Training supervision only; these targets never enter physical execution."""
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("teacher-anchor-beta must be finite and nonnegative")
    resolved_final = beta if final_beta is None else final_beta
    if not math.isfinite(resolved_final) or resolved_final < 0:
        raise ValueError("teacher-anchor-final-beta must be finite and nonnegative")
    if max(beta, resolved_final) > 0 and (type(passes) is not int or passes < 1):
        raise ValueError("teacher-anchor-passes must be a positive integer when beta > 0")
    if schedule_updates is not None and (
        type(schedule_updates) is not int or schedule_updates < 1
    ):
        raise ValueError("teacher-anchor schedule updates must be a positive integer")
    result = {"beta": beta, "passes": passes, "squeeze_rad": TEACHER_SQUEEZE_RAD,
            "gate": "current_reference_max_proximity_confidence_valid",
            "contact_intent_threshold": TEACHER_CONTACT_INTENT_THRESHOLD,
            "gate_comparison": ">=", "flex_joints": list(TEACHER_FLEX_JOINTS),
            "role": "training_supervision_only"}
    if final_beta is not None:
        result["schedule"] = {
            "type": "linear_by_update",
            "start_beta": beta,
            "final_beta": resolved_final,
            "updates": schedule_updates,
        }
    return result


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
    def __init__(
        self, trajectory, *, num_envs=1, device="cpu", seed=0,
        policy_version="v4", reward_version="v4", **kwargs,
    ):
        from sim.manorl.autonomy_batch import BatchedAutonomyRuntime
        self.policy_version=policy_version
        self.runtime=BatchedAutonomyRuntime(
            trajectory,num_envs=num_envs,device=device,seed=seed,
            observation_version=policy_version,reward_version=reward_version,
            **kwargs
        )
        self.num_envs=self.runtime.num_envs; self.device_name=device; self._device=torch.device("cuda" if device=="gpu" else "cpu")
        self.observation_dim=self.runtime.observation_dim; self.action_dim=ACTION_DIM
        self.reward_names=self.runtime.reward_names
        self._telemetry_slices, self._telemetry_width = _telemetry_slices(
            self.reward_names
        )
        self._packed_telemetry_fn = self.runtime.jax.jit(
            self._pack_device_telemetry
        )
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(self.observation_dim,),dtype=np.float32)
        self.action_space=gym.spaces.Box(-1.,1.,shape=(ACTION_DIM,),dtype=np.float32)
    @property
    def device(self): return self._device
    def _to_torch(self,x):
        if self.device_name=="gpu":
            from sim.manorl.device_runtime import jax_to_torch_cuda
            return jax_to_torch_cuda(x)
        return torch.as_tensor(np.array(x,copy=True),dtype=torch.float32,device=self._device)
    def reset(self,*,mask=None):
        return self._to_torch(self.runtime.reset(mask)),{
            "num_envs":self.num_envs,
            "contract":self.runtime.observation_contract,
        }
    def prepare_action(self): return self._to_torch(self.runtime.prepare_action())
    def configure_curriculum_stage(self, stage: int) -> None:
        self.runtime.configure_curriculum_stage(stage)
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
        """Return post-transition facts through one packed JAX/Torch transfer."""
        if raw_actions.shape != (self.num_envs, ACTION_DIM):
            raise ValueError("v4 raw actions must be (num_envs,28)")
        packed = self._to_torch(
            self._packed_telemetry_fn(
                self.runtime.last_physical,
                self.runtime.last_contact,
                self.runtime.last_reward,
                self.runtime.indices,
            )
        )
        if packed.shape != (self.num_envs, self._telemetry_width):
            raise RuntimeError("packed telemetry width does not match its schema")
        snapshot = {}
        for name, columns in self._telemetry_slices.items():
            value = packed[:, columns]
            snapshot[name] = value[:, 0] if value.shape[1] == 1 else value
        # Raw Normal samples stay in Torch: jp.asarray(CUDA Tensor) would take
        # NumPy's host path. Only physical execution crosses JAX/Torch via DLPack.
        raw_action_abs=raw_actions.detach().abs()
        executed=torch.clamp(raw_actions.detach(),-1.,1.)
        snapshot.update({
            "action_raw_abs_sum":raw_action_abs.sum(dim=-1),
            "action_raw_abs_max":raw_action_abs.amax(dim=-1),
            "action_raw_abs_denominator":torch.full((self.num_envs,), ACTION_DIM, dtype=raw_actions.dtype, device=raw_actions.device),
            "action_executed_norm":torch.linalg.vector_norm(executed,dim=-1),
            "action_clipped":(raw_action_abs>1.).to(raw_actions.dtype).sum(dim=-1),
            "action_denominator":torch.full((self.num_envs,), ACTION_DIM, dtype=raw_actions.dtype, device=raw_actions.device),
        })
        return snapshot

    def _pack_device_telemetry(self, physical, contact, reward, indices):
        """Fuse immutable post-transition diagnostics into one float32 buffer."""
        jp=self.runtime.jp; cache=self.runtime.cache
        env_ref=getattr(self.runtime,"env_ref",None)
        index=_gather(cache,indices,env_ref=env_ref)
        target_object=jp.asarray(cache.object_origin)[index]
        target_palm=jp.asarray(cache.palm_origin)[index]
        target_raw=jp.asarray(cache.q_raw)[index]
        target_feasible=jp.asarray(cache.q_feasible)[index]
        paired_norm=jp.linalg.norm(contact.paired_force_on_object,axis=-1)
        loaded=paired_norm>.02
        opposing_loaded=(jp.any(loaded[:,:13],axis=-1)&jp.any(loaded[:,13:],axis=-1))
        positive_lift=opposing_loaded&(physical.object_v_com[:,2]>.005)
        stable_airborne=opposing_loaded&(physical.object_bottom>cache.table_height+.005)
        initial_index=_gather(cache,jp.zeros_like(indices),env_ref=env_ref)
        values = (
            jp.stack(tuple(getattr(reward, name) for name in self.reward_names), axis=1),
            reward.total[:,None],
            reward.reason[:,None],
            reward.valid[:,None],
            jp.abs(physical.object_origin-target_object),
            (2*jp.arccos(jp.clip(jp.abs(jp.sum(
                physical.object_quat_xyzw*jp.asarray(cache.object_quat_xyzw)[index],
                axis=-1,
            )),0.,1.)))[:,None],
            jp.linalg.norm(physical.palm_origin-target_palm,axis=-1)[:,None],
            jp.sqrt(jp.mean((physical.q_raw[:,6:]-target_raw[:,6:])**2,axis=-1))[:,None],
            jp.sqrt(jp.mean((physical.q_raw[:,6:]-target_feasible[:,6:])**2,axis=-1))[:,None],
            (physical.object_bottom-cache.table_height)[:,None],
            (jp.asarray(cache.reference_bottom)[index]-cache.table_height)[:,None],
            (physical.object_origin[:,2]-jp.asarray(cache.object_origin)[initial_index][:,2])[:,None],
            loaded,
            contact.paired_count>0,
            contact.paired_count.sum(axis=-1)[:,None],
            paired_norm.sum(axis=-1)[:,None],
            jp.linalg.norm(contact.object_all_force,axis=-1)[:,None],
            jp.linalg.norm(contact.paired_torque_com,axis=-1)[:,None],
            jp.linalg.norm(contact.tangential_slip,axis=-1),
            (physical.object_bottom>cache.table_height+.005)[:,None],
            getattr(reward,"opposition_loaded",opposing_loaded)[:,None],
            getattr(reward,"positive_lift",positive_lift)[:,None],
            getattr(reward,"stable_airborne",stable_airborne)[:,None],
            jp.mean(jp.abs(physical.command_error),axis=-1)[:,None],
            jp.any(jp.abs(physical.command_error)>=1.,axis=-1)[:,None],
            (
                jp.minimum(indices,reference_lengths(cache,env_ref)-1)
                / jp.asarray(jp.maximum(1,reference_lengths(cache,env_ref)-1),jp.float32)
            )[:,None],
            physical.object_origin[:,2,None],
        )
        return jp.concatenate(
            tuple(jp.asarray(value,dtype=jp.float32) for value in values), axis=1
        )
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
    def checkpoint_contracts(self):
        return {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "observation_contract": OBSERVATION_CONTRACT_ID,
            "reward_contract": getattr(
                self, "reward_contract_id", REWARD_CONTRACT_ID
            ),
            "action_contract": ACTION_CONTRACT_ID,
        }
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

def actor_critic_architecture_v5(*, separate_critic: bool = False) -> dict[str, Any]:
    return {"id": ACTOR_CRITIC_ARCHITECTURE_ID_V5, "raw_observation_dim": RAW_OBSERVATION_DIM_V5,
            "encoded_feature_dim": ENCODED_OBSERVATION_DIM_V5,
            "pointnet": "object 3-64-128-256-max-256-64; hand 3-64-128-256-max-256-64",
            "value_trunk": "separate" if separate_critic else "shared", "action_dim": ACTION_DIM}


class AutonomyActorCriticV5(GaussianMixin,DeterministicMixin,Model):
    """v5 point-cloud policy: object and hand clouds each get their own PointNet."""
    def __init__(self,observation_space,action_space,device="cpu",*,separate_critic=False,clip_actions=False):
        Model.__init__(self,observation_space=observation_space,state_space=None,action_space=action_space,device=device)
        self.observation_dim=int(self.num_observations); self.action_dim=int(self.num_actions)
        if (self.observation_dim,self.action_dim)!=(RAW_OBSERVATION_DIM_V5,ACTION_DIM): raise ValueError("v5 actor requires raw 1342 and 28 actions")
        GaussianMixin.__init__(self,clip_actions=clip_actions,clip_mean_actions=False,clip_log_std=True,min_log_std=-5.,max_log_std=2.,reduction="sum",role="policy")
        DeterministicMixin.__init__(self,clip_actions=False,role="value")
        self.object_pointnet=PointNetEncoder(device)
        self.hand_pointnet=PointNetEncoder(device,points=256)
        self.net=nn.Sequential(nn.Linear(ENCODED_OBSERVATION_DIM_V5,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh()).to(device)
        self.value_net=nn.Sequential(nn.Linear(ENCODED_OBSERVATION_DIM_V5,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh()).to(device) if separate_critic else None
        self.separate_critic=bool(separate_critic)
        self.mean=nn.Linear(128,ACTION_DIM).to(device); self.value=nn.Linear(128,1).to(device); self.log_std=nn.Parameter(torch.full((ACTION_DIM,),-1.,device=device))
    def _encoded(self,x):
        if x.ndim!=2 or x.shape[1]!=RAW_OBSERVATION_DIM_V5: raise ValueError("v5 model requires (batch,1342) raw observations")
        object_cloud=x[:,320:512].reshape(-1,64,3); hand_cloud=x[:,512:1280].reshape(-1,256,3)
        object_embedding=self.object_pointnet(object_cloud); hand_embedding=self.hand_pointnet(hand_cloud)
        return torch.cat((x[:,:320],object_embedding,hand_embedding,x[:,1280:]),dim=-1)
    def checkpoint_architecture(self): return actor_critic_architecture_v5(separate_critic=self.separate_critic)
    def checkpoint_contracts(self):
        return {
            "checkpoint_format": CHECKPOINT_FORMAT_V5,
            "observation_contract": OBSERVATION_CONTRACT_ID_V5,
            "reward_contract": getattr(
                self, "reward_contract_id", REWARD_CONTRACT_ID
            ),
            "action_contract": ACTION_CONTRACT_ID,
        }
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

def validate_target_kl(target_kl: float | None) -> None:
    if target_kl is not None and (
        not math.isfinite(target_kl) or target_kl <= 0
    ):
        raise ValueError("target-kl must be finite and positive when enabled")


def v4_ppo_config(*, rollouts:int, learning_epochs:int, mini_batches:int, learning_rate:float=3e-4, target_kl:float|None=None) -> dict[str, Any]:
    """Installed PPO defaults plus the unchanged v4 overrides, in one source."""
    validate_learning_rate(learning_rate)
    validate_target_kl(target_kl)
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
            "learning_epochs":cfg.learning_epochs,"mini_batches":cfg.mini_batches,"rollouts":cfg.rollouts,
            "target_kl":getattr(agent,"target_kl",None)}


def _bind_reward_contract(model, reward_contract_id: str):
    model.reward_contract_id = reward_contract_id
    return model


def build_batched_runtime(adapter, *, rollouts:int, learning_epochs:int, mini_batches:int, device:str, separate_critic:bool=False, learning_rate:float=3e-4, target_kl:float|None=None):
    """Canonical RlGamesPPO with a version-selected observation/model pair."""
    cfg=v4_ppo_config(rollouts=rollouts,learning_epochs=learning_epochs,mini_batches=mini_batches,learning_rate=learning_rate,target_kl=target_kl)
    memory=RandomMemory(memory_size=rollouts,num_envs=adapter.num_envs,device=device)
    policy_version=getattr(adapter,"policy_version","v4")
    if policy_version == "v4":
        model=AutonomyActorCritic(
            adapter.observation_space,adapter.action_space,device=device,
            separate_critic=separate_critic,clip_actions=False
        )
    elif policy_version == "v5":
        model=AutonomyActorCriticV5(
            adapter.observation_space,adapter.action_space,device=device,
            separate_critic=separate_critic,clip_actions=False
        )
    elif policy_version == "v5.25":
        model=AutonomyActorCriticV525(
            adapter.observation_space,adapter.action_space,device=device,
            separate_critic=separate_critic,clip_actions=False
        )
    elif policy_version == "v5.5":
        model=AutonomyActorCriticV55(
            adapter.observation_space,adapter.action_space,device=device,
            separate_critic=separate_critic,clip_actions=False
        )
    elif policy_version == "v5.75":
        model=AutonomyActorCriticV575(
            adapter.observation_space,adapter.action_space,device=device,
            separate_critic=separate_critic,clip_actions=False
        )
    elif policy_version == "v6":
        model=AutonomyActorCriticV6(
            adapter.observation_space,adapter.action_space,device=device,
            clip_actions=False
        )
    else:
        raise ValueError(f"unsupported policy version: {policy_version!r}")
    reward_contract_id = getattr(
        getattr(adapter, "runtime", None),
        "reward_contract_id",
        REWARD_CONTRACT_ID,
    )
    _bind_reward_contract(model, reward_contract_id)
    agent=RlGamesPPO(models={"policy":model,"value":model},memory=memory,
                     observation_space=adapter.observation_space,state_space=None,
                     action_space=adapter.action_space,device=device,cfg=cfg)
    agent.target_kl = target_kl
    agent.init(); return model,agent


def actor_critic_architecture_for_version(
    policy_version: str, *, separate_critic: bool = False
) -> dict[str, Any]:
    if policy_version == "v4":
        return actor_critic_architecture(separate_critic=separate_critic)
    if policy_version == "v5":
        return actor_critic_architecture_v5(separate_critic=separate_critic)
    if policy_version == "v5.25":
        return actor_critic_architecture_v525(separate_critic=separate_critic)
    if policy_version == "v5.5":
        return actor_critic_architecture_v55(separate_critic=separate_critic)
    if policy_version == "v5.75":
        return actor_critic_architecture_v575(separate_critic=separate_critic)
    if policy_version == "v6":
        return actor_critic_architecture_v6()
    raise ValueError(f"unsupported policy version: {policy_version!r}")


def model_for_version(
    policy_version: str,
    observation_space,
    action_space,
    *,
    device: str | torch.device,
    separate_critic: bool = False,
    reward_contract_id: str = REWARD_CONTRACT_ID,
):
    if policy_version == "v4":
        model = AutonomyActorCritic(
            observation_space, action_space, device=device,
            separate_critic=separate_critic
        )
    elif policy_version == "v5":
        model = AutonomyActorCriticV5(
            observation_space, action_space, device=device,
            separate_critic=separate_critic
        )
    elif policy_version == "v5.25":
        model = AutonomyActorCriticV525(
            observation_space, action_space, device=device,
            separate_critic=separate_critic
        )
    elif policy_version == "v5.5":
        model = AutonomyActorCriticV55(
            observation_space, action_space, device=device,
            separate_critic=separate_critic
        )
    elif policy_version == "v5.75":
        model = AutonomyActorCriticV575(
            observation_space, action_space, device=device,
            separate_critic=separate_critic
        )
    elif policy_version == "v6":
        model = AutonomyActorCriticV6(
            observation_space, action_space, device=device
        )
    else:
        raise ValueError(f"unsupported policy version: {policy_version!r}")
    return _bind_reward_contract(model, reward_contract_id)

def seed_everything(seed:int): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
