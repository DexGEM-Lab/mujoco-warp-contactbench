"""M3 formal PPO adapter for the v2 cube2 autonomy environment.

The physical backend remains the v2 MJX-Warp producer/clock. This module adds
only a Gymnasium boundary and canonical skrl PPO runtime; it does not duplicate
physics or implement a return estimator.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, random
from pathlib import Path
from typing import Any
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from skrl.agents.torch.ppo.ppo import compute_gae
from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
from skrl.memories.torch import RandomMemory
from skrl.utils.spaces.torch import tensorize_space, flatten_tensorized_space
from sim.manorl.autonomy import Cube2AutonomousMJX
from sim.manorl.autonomy_contracts import ACTION_CONTRACT_ID, ACTION_DIM, OBSERVATION_CONTRACT_ID, OBSERVATION_DIM, REWARD_CONTRACT_ID
from sim.manorl.rl_games_ppo import RlGamesPPO
from sim.manorl.trajectory_package import TrajectoryCatalog, load_trajectory_package

RESERVED_TRAIN_IDENTITIES = ("cube2_02_2833", "cube2_02_2835", "cube2_02_2837")
SPLIT_CONTRACT_ID = "manorl.autonomy.identity_split.v1"
TRAINING_CONTRACT_ID = "manorl.autonomy.training.v1"

def identity_split(catalog: TrajectoryCatalog, *, seed: int = 0) -> dict[str, Any]:
    identities = tuple(t.identity.identity for t in catalog.trajectories)
    missing = [x for x in RESERVED_TRAIN_IDENTITIES if x not in identities]
    if missing: raise ValueError(f"reserved TRAIN identities absent from package: {missing}")
    reserved = [identities.index(x) for x in RESERVED_TRAIN_IDENTITIES]
    remaining = [i for i in range(len(identities)) if i not in reserved]
    if len(identities) != 50 or len(remaining) != 47: raise ValueError("M3 split requires exactly 50 package identities")
    perm = np.random.default_rng(seed).permutation(remaining).tolist()
    train = reserved + perm[:37]; val, test = perm[37:42], perm[42:47]
    payload = {"contract": SPLIT_CONTRACT_ID, "seed": int(seed), "reserved_train": list(RESERVED_TRAIN_IDENTITIES), "train_indices": train, "validation_indices": val, "test_indices": test, "train_identities": [identities[i] for i in train], "validation_identities": [identities[i] for i in val], "test_identities": [identities[i] for i in test]}
    payload["digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload

def canonical_gae(rewards, terminated, truncated, values, last_values, *, discount_factor=.99, lambda_coefficient=.95, time_limit_bootstrap=True):
    """Thin named boundary to skrl's done/bootstrap implementation."""
    return compute_gae(rewards=rewards, terminated=terminated, truncated=truncated, values=values, last_values=last_values, discount_factor=discount_factor, lambda_coefficient=lambda_coefficient, time_limit_bootstrap=time_limit_bootstrap)

class BatchedAutonomyAdapter:
    """Torch policy boundary over the device-resident homogeneous runtime.

    On CUDA the only full-tensor crossings are DLPack borrows. Compact done
    and validity telemetry may be copied to host for episode bookkeeping.
    """
    def __init__(self, trajectory, *, num_envs=1, device="gpu", seed=0):
        from sim.manorl.autonomy_batch import BatchedAutonomyRuntime, V3_OBSERVATION_DIM
        self.runtime = BatchedAutonomyRuntime(trajectory, num_envs=num_envs, device=device, seed=seed)
        self.num_envs = int(num_envs)
        self.observation_dim = int(V3_OBSERVATION_DIM)
        self.action_dim = ACTION_DIM
        self._device = torch.device("cuda" if device == "gpu" else "cpu")
        self.device_name = device
        self.observation_space = gym.spaces.Box(-5., 5., shape=(self.observation_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., shape=(self.action_dim,), dtype=np.float32)
        self._pending = np.zeros(self.num_envs, dtype=bool)

    @property
    def device(self):
        return self._device

    def _to_torch(self, value):
        from sim.manorl.device_runtime import jax_to_torch_cuda
        if self.device_name == "gpu":
            return jax_to_torch_cuda(value)
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self._device)

    def reset(self, *, mask=None):
        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)
        observation = self.runtime.reset(mask)
        self._pending[:] = False
        return self._to_torch(observation), {"num_envs": self.num_envs, "contract": "manorl.autonomy.v3"}

    def step(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        if actions.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(f"actions must have shape ({self.num_envs}, 28)")
        from sim.manorl.device_runtime import torch_to_jax_cuda
        jax_actions = torch_to_jax_cuda(actions) if self.device_name == "gpu" else np.asarray(actions.detach().cpu(), dtype=np.float32)
        observation, reward, done, info = self.runtime.step(jax_actions)
        # Full observation/reward tensors remain device resident for CUDA.
        result = (self._to_torch(observation), self._to_torch(reward).reshape(self.num_envs), torch.as_tensor(np.asarray(done), dtype=torch.bool, device=self._device), info)
        self._pending = np.asarray(done, dtype=bool)
        return result


class AutonomyVectorEnv(gym.Env):
    """Honest N=1 Gymnasium environment with explicit terminal resets."""
    metadata = {"render_modes": []}
    def __init__(self, trajectory, *, device="cpu", seed=0, contact_conditioned=True):
        self.environment = Cube2AutonomousMJX(trajectory, device=device, seed=seed, contact_conditioned=contact_conditioned)
        self.trajectory = trajectory; self.contact_conditioned = bool(contact_conditioned); self.num_envs = 1; self._device = "cuda" if device == "gpu" else "cpu"
        self.observation_space = gym.spaces.Box(-5., 5., shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., shape=(ACTION_DIM,), dtype=np.float32)
        self._pending = False
    @property
    def device(self): return self._device
    def reset(self, *, seed=None, options=None):
        if options: raise ValueError("M3 adapter has no indexed reset for N=1")
        obs=self.environment.reset(); self._pending=False
        return obs, {"identity": self.trajectory.identity.identity, "seed": seed}
    def step(self, actions):
        a=np.asarray(actions,dtype=float)
        if a.shape != (ACTION_DIM,): raise ValueError("actions must have shape (28,)")
        if self._pending: raise RuntimeError("terminal transition requires explicit reset before the next action")
        result=self.environment.step(np.clip(a, -1., 1.)); phase=result.info["failure_phase"]
        terminated=bool(result.done)
        truncated=False
        self._pending=bool(result.done)
        info={k: v for k,v in result.info.items() if isinstance(v,(str,int,float,bool,np.ndarray,list,dict))}; info["terms"]=result.terms; info["success"]=bool(result.info["task_success"])
        return result.observation, np.float32(result.reward), terminated, truncated, info
    def close(self): return None
    def render(self): return None

from skrl.models.torch import Model
from skrl.models.torch.gaussian import GaussianMixin
from skrl.models.torch.deterministic import DeterministicMixin
class AutonomyActorCritic(GaussianMixin, DeterministicMixin, Model):
    """Small Gaussian actor/value model compatible with canonical skrl PPO.

    The input width comes from the live space, allowing the versioned v3
    witness features without changing PPO or the frozen v2 model ABI.
    """
    def __init__(self, observation_space, action_space, device="cpu"):
        Model.__init__(self, observation_space=observation_space, state_space=None, action_space=action_space, device=device)
        self.observation_dim = int(self.num_observations)
        self.action_dim = int(self.num_actions)
        if self.action_dim != ACTION_DIM or self.observation_dim < 1:
            raise ValueError("autonomy actor requires 28 actions and a positive observation width")
        GaussianMixin.__init__(self, clip_actions=False, clip_mean_actions=False, clip_log_std=True, min_log_std=-5., max_log_std=2., reduction="sum", role="policy")
        DeterministicMixin.__init__(self, clip_actions=False, role="value")
        self.net=nn.Sequential(nn.Linear(self.observation_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh()).to(device)
        self.mean=nn.Linear(128,self.action_dim).to(device); self.value=nn.Linear(128,1).to(device); self.log_std=nn.Parameter(torch.full((self.action_dim,),-1.,device=device))
    def compute(self, inputs, role=""):
        h=self.net(inputs["observations"])
        if role=="policy": return self.mean(h), {"log_std":self.log_std.expand_as(self.mean(h))}
        if role=="value": return self.value(h), {}
        raise ValueError("role must be policy or value")

def build_runtime(environment, *, rollouts=2, learning_epochs=1, device="cpu"):
    """Construct canonical RlGamesPPO with done-aware skrl GAE."""
    wrapper=GymnasiumWrapper(environment); memory=RandomMemory(memory_size=rollouts,num_envs=environment.num_envs,device=device); model=AutonomyActorCritic(wrapper.observation_space,wrapper.action_space,device=device)
    cfg={"rollouts":rollouts,"learning_epochs":learning_epochs,"mini_batches":1,"discount_factor":.99,"gae_lambda":.95,"learning_rate":3e-4,"ratio_clip":.2,"value_clip":.2,"entropy_loss_scale":.001,"value_loss_scale":.5,"learning_starts":0,"time_limit_bootstrap":True,"experiment":{"write_interval":0,"checkpoint_interval":0},"mixed_precision":False}
    agent=RlGamesPPO(models={"policy":model,"value":model},memory=memory,observation_space=wrapper.observation_space,state_space=None,action_space=wrapper.action_space,device=device,cfg=cfg); agent.init(); return wrapper,model,agent

def seed_everything(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
