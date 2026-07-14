"""Explicit skrl PPO runtime for the bounded ManoRL Gymnasium adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv
from sim.manorl.model import ManoActorCritic
from sim.manorl.normalization import PointCloudAwareRunningStandardScaler


@dataclass(frozen=True)
class ManoPPOConfig:
    """PPO objective settings resolved from the current MANOHand source config."""

    rollouts: int = 48
    minibatch_size: int = 1024
    learning_epochs: int = 3
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.001
    value_loss_scale: float = 4.0
    learning_rate: float = 3.0e-4
    kl_threshold: float = 0.016
    grad_norm_clip: float = 1.0
    reward_scale: float = 0.5
    time_limit_bootstrap: bool = True

    def __post_init__(self) -> None:
        if self.rollouts < 1 or self.minibatch_size < 1 or self.learning_epochs < 1:
            raise ValueError("PPO rollout, minibatch, and epoch counts must be positive")

    @classmethod
    def optimizer_smoke(cls) -> "ManoPPOConfig":
        """Minimal non-training configuration that forces one finite update."""

        return cls(rollouts=2, minibatch_size=2, learning_epochs=1, kl_threshold=0.0)

    def skrl_config(self, *, num_envs: int, device: str) -> dict[str, Any]:
        batch_size = self.rollouts * num_envs
        if batch_size % self.minibatch_size:
            raise ValueError(
                f"source minibatch_size={self.minibatch_size} must divide rollout batch {batch_size}"
            )
        return {
            "rollouts": self.rollouts,
            "learning_epochs": self.learning_epochs,
            "mini_batches": batch_size // self.minibatch_size,
            "discount_factor": self.discount_factor,
            "gae_lambda": self.gae_lambda,
            "learning_rate": self.learning_rate,
            "learning_rate_scheduler": KLAdaptiveLR,
            "learning_rate_scheduler_kwargs": {"kl_threshold": self.kl_threshold},
            "observation_preprocessor": PointCloudAwareRunningStandardScaler,
            "observation_preprocessor_kwargs": {"size": 476, "device": device},
            "value_preprocessor": RunningStandardScaler,
            "value_preprocessor_kwargs": {"size": 1, "device": device},
            "grad_norm_clip": self.grad_norm_clip,
            "ratio_clip": self.ratio_clip,
            "value_clip": self.value_clip,
            "entropy_loss_scale": self.entropy_loss_scale,
            "value_loss_scale": self.value_loss_scale,
            "time_limit_bootstrap": self.time_limit_bootstrap,
            "rewards_shaper": scale_rewards,
            # Target AMP execution is a device policy, not a PPO semantic. It
            # remains off for deterministic CPU smoke coverage.
            "mixed_precision": False,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        }


def scale_rewards(rewards: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
    """Preserve the source reward-shaper scale without hiding it in config."""

    del timestep, timesteps
    return rewards * 0.5


class ManoSkrlRuntime:
    """One shared model, source-normalizers, and skrl PPO over the vector adapter."""

    def __init__(self, environment: ManoGymnasiumVectorEnv, config: ManoPPOConfig = ManoPPOConfig()) -> None:
        if not isinstance(environment, ManoGymnasiumVectorEnv):
            raise TypeError("environment must be a ManoGymnasiumVectorEnv")
        if not isinstance(config, ManoPPOConfig):
            raise TypeError("config must be a ManoPPOConfig")
        self.gymnasium_env = environment
        self.config = config
        self.env = wrap_env(environment, wrapper="gymnasium", verbose=False)
        self.device = str(self.env.device)
        self.model = ManoActorCritic(
            self.env.observation_space, self.env.state_space, self.env.action_space, device=self.device
        )
        self.memory = RandomMemory(memory_size=config.rollouts, num_envs=environment.num_envs, device=self.device)
        self.agent = PPO(
            models={"policy": self.model, "value": self.model},
            memory=self.memory,
            observation_space=self.env.observation_space,
            state_space=self.env.state_space,
            action_space=self.env.action_space,
            device=self.device,
            cfg=config.skrl_config(num_envs=environment.num_envs, device=self.device),
        )
        self.agent.init()

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "ppo": asdict(self.config),
            "model_state_dict": self.model.state_dict_manifest(),
            "normalizer": "source_pointcloud_shared_xyz",
            "deterministic_policy_mode": "mean_clipped_to_action_space",
        }

    @torch.no_grad()
    def deterministic_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """Target-selected evaluation mode: normalized actor mean clipped to action bounds."""

        inputs = {
            "observations": self.agent._observation_preprocessor(observations),
            "states": None,
        }
        means, _ = self.model.compute(inputs, role="policy")
        return torch.clamp(means, min=-1.0, max=1.0)

    def deterministic_rollout(self, *, steps: int) -> dict[str, object]:
        """Run a bounded mean-policy rollout without training or checkpoint output."""

        if steps < 1:
            raise ValueError("steps must be positive")
        observations, _ = self.env.reset()
        rewards: list[torch.Tensor] = []
        for _ in range(steps):
            actions = self.deterministic_actions(observations)
            observations, reward, terminated, truncated, _ = self.env.step(actions)
            if not torch.isfinite(observations).all() or not torch.isfinite(reward).all():
                raise RuntimeError("non-finite ManoRL runtime rollout value")
            rewards.append(reward.detach().clone())
            if torch.any(terminated | truncated):
                # The physical source contract resets on the following step;
                # keep this bounded rollout on its current vector boundary.
                break
        return {
            "steps": len(rewards),
            "rewards": torch.cat(rewards, dim=0),
            "observations": observations.detach().clone(),
        }

    def one_update_smoke(self) -> bool:
        """Collect exactly one configured rollout and execute one PPO update."""

        if self.config.rollouts * self.gymnasium_env.num_envs < 2:
            raise ValueError("PPO optimizer smoke requires at least two rollout samples")
        self.agent.enable_training_mode(True)
        observations, _ = self.env.reset()
        before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        for timestep in range(self.config.rollouts):
            with torch.no_grad():
                actions, _ = self.agent.act(observations, None, timestep=timestep, timesteps=self.config.rollouts)
            next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
            self.agent.record_transition(
                observations=observations,
                states=None,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=None,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=self.config.rollouts,
            )
            self.agent.post_interaction(timestep=timestep + 1, timesteps=self.config.rollouts)
            observations = next_observations
        after = dict(self.model.named_parameters())
        if not all(torch.isfinite(parameter).all() for parameter in after.values()):
            raise RuntimeError("PPO update produced non-finite model parameters")
        return any(not torch.equal(before[name], parameter.detach()) for name, parameter in after.items())
