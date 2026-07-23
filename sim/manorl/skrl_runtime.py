"""Explicit skrl PPO runtime for the bounded ManoRL Gymnasium adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium
import numpy as np
import torch

from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
from skrl.memories.torch import RandomMemory
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)

from sim.manorl.abi import ENVIRONMENT_CONTRACT_ID
from sim.manorl.device_runtime import DeviceTransitionBatch, jax_to_torch_cuda
from sim.manorl.environment import PhaseTimings
from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv
from sim.manorl.model import ManoActorCritic
from sim.manorl.normalization import PointCloudAwareRunningStandardScaler, SourceRunningStandardScaler
from sim.manorl.observations import CONTACT_FORCE_THRESHOLD
from sim.manorl.rewards import PPO_REWARD_CONTRACT_ID, PPO_REWARD_SCALE, REWARD_CONTRACT_ID
from sim.manorl.rl_games_ppo import RlGamesAdaptiveLR, RlGamesPPO


def source_aligned_reward_shaper(
    rewards: torch.Tensor, timestep: int, timesteps: int
) -> torch.Tensor:
    """Apply the rl-games 0.5 reward shaper without changing environment telemetry."""

    del timestep, timesteps
    return rewards * PPO_REWARD_SCALE


@dataclass(frozen=True)
class ManoPPOConfig:
    """PPO objective settings resolved from the current MANOHand source config."""

    rollouts: int = 48
    minibatch_size: int = 4096
    learning_epochs: int = 3
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.001
    # rl-games applies 0.5 * critic_coef (4.0) to the critic MSE.
    value_loss_scale: float = 2.0
    learning_rate: float = 3.0e-4
    kl_threshold: float = 0.016
    bounds_loss_coef: float = 1.0e-4
    learning_starts: int = 0
    use_film: bool = True
    grad_norm_clip: float = 1.0
    time_limit_bootstrap: bool = True
    profile_phases: bool = False

    def __post_init__(self) -> None:
        if self.rollouts < 1 or self.minibatch_size < 1 or self.learning_epochs < 1:
            raise ValueError("PPO rollout, minibatch, and epoch counts must be positive")
        if self.learning_starts < 0:
            raise ValueError("PPO learning_starts must be non-negative")

    @classmethod
    def optimizer_smoke(cls) -> "ManoPPOConfig":
        """Minimal non-training configuration that forces one finite update."""

        return cls(rollouts=2, minibatch_size=2, learning_epochs=1)

    def skrl_config(
        self, *, num_envs: int, device: str, observation_size: int = 476
    ) -> dict[str, Any]:
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
            "learning_rate_scheduler": RlGamesAdaptiveLR,
            "learning_rate_scheduler_kwargs": {"kl_threshold": self.kl_threshold},
            "observation_preprocessor": PointCloudAwareRunningStandardScaler,
            "observation_preprocessor_kwargs": {
                "size": int(observation_size),
                "device": device,
            },
            "value_preprocessor": SourceRunningStandardScaler,
            "value_preprocessor_kwargs": {"size": 1, "epsilon": 1.0e-5, "device": device},
            "grad_norm_clip": self.grad_norm_clip,
            "ratio_clip": self.ratio_clip,
            "value_clip": self.value_clip,
            "entropy_loss_scale": self.entropy_loss_scale,
            "value_loss_scale": self.value_loss_scale,
            "learning_starts": self.learning_starts,
            "time_limit_bootstrap": self.time_limit_bootstrap,
            "rewards_shaper": source_aligned_reward_shaper,
            # Target AMP execution is a device policy, not a PPO semantic. It
            # remains off for deterministic CPU smoke coverage.
            "mixed_precision": False,
            "experiment": {"write_interval": 0, "checkpoint_interval": 0},
        }


class ResettableGymnasiumWrapper(GymnasiumWrapper):
    """skrl Gymnasium wrapper with an uncached indexed vector reset.

    skrl 2.1 caches the first vector reset behind ``_reset_once``.  That is
    useful for its normal autoreset path, but it silently returns the previous
    terminal observation when a caller needs to reset only completed worlds.
    This wrapper always forwards explicit ``options`` resets and refreshes the
    cached observation after the call.
    """

    def _tensorize_observation(self, observation: Any) -> torch.Tensor:
        return flatten_tensorized_space(
            tensorize_space(self.observation_space, observation, device=self.device)
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        env_ids: Any | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self._vectorized:
            if options is not None or env_ids is not None:
                raise ValueError("indexed reset options require a vectorized Gymnasium environment")
            observation, info = self._env.reset(seed=self._seed if seed is None else seed)
            self._seed = None
            return self._tensorize_observation(observation), info
        if env_ids is not None:
            if options is not None and "env_ids" in options:
                raise ValueError("env_ids must be provided either directly or in options, not both")
            options = {"env_ids": np.asarray(env_ids, dtype=np.int64)}
        selected_seed = self._seed if seed is None else seed
        partial = options is not None and "env_ids" in options
        # A subset reset must not consume or reseed the wrapper's startup seed.
        observation, info = self._env.reset(
            seed=None if partial else selected_seed,
            options=options,
        )
        tensor_observation = self._tensorize_observation(observation)
        self._observation = tensor_observation
        self._info = info
        self._reset_once = False
        if not partial:
            self._seed = None
        return tensor_observation, info

    def reset_done(self, done: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Reset done worlds and return their full vector observation batch."""

        mask = torch.as_tensor(done, dtype=torch.bool).reshape(-1)
        if mask.numel() != self.num_envs:
            raise ValueError(f"done must contain {self.num_envs} environments")
        env_ids = torch.nonzero(mask, as_tuple=False).reshape(-1).cpu().numpy().astype(np.int64)
        if env_ids.size == 0:
            return self._observation
        observation, _ = self.reset(options={"env_ids": env_ids})
        return observation


class ProfiledGymnasiumWrapper(ResettableGymnasiumWrapper):
    """Profile skrl's two host/device conversion boundaries without altering its default wrapper."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.phase_timings = PhaseTimings(enabled=True)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def phase_profile(self) -> dict[str, dict[str, float | int]]:
        return self.phase_timings.summary()

    def reset_phase_profile(self) -> None:
        self.phase_timings.reset()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        action_started = self.phase_timings.start("skrl_cuda_action_to_numpy", self._synchronize)
        actions = untensorize_space(
            self.action_space,
            unflatten_tensorized_space(self.action_space, actions),
            squeeze_batch_dimension=not self._vectorized,
        )
        if self._vectorized and isinstance(self.action_space, gymnasium.spaces.Discrete):
            actions = actions.flatten()
        self.phase_timings.stop("skrl_cuda_action_to_numpy", action_started, self._synchronize)

        observation, reward, terminated, truncated, info = self._env.step(actions)

        response_started = self.phase_timings.start("skrl_numpy_response_to_cuda", self._synchronize)
        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, observation, device=self.device)
        )
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool).view(self.num_envs, -1)
        self.phase_timings.stop("skrl_numpy_response_to_cuda", response_started, self._synchronize)

        if self._vectorized:
            self._observation = observation
            self._info = info
        return observation, reward, terminated, truncated, info


class DeviceTransitionGymnasiumWrapper(ResettableGymnasiumWrapper):
    """skrl wrapper for the narrow JAX CUDA policy-egress contract.

    Actions intentionally follow skrl's established Torch-to-NumPy path. The
    raw sampled action handed to PPO is never clipped or replaced: clipping is
    owned by ``ManoGymnasiumVectorEnv.step_device``. Reset/reset_done continue
    through the ordinary host Gymnasium path, the explicit residual boundary.
    """

    def __init__(self, env: ManoGymnasiumVectorEnv) -> None:
        if not env.environment.config.device_transition:
            raise ValueError("device wrapper requires device_transition=True")
        super().__init__(env)
        if self.device.type != "cuda":
            raise RuntimeError("device_transition requires a CUDA skrl wrapper")

    @staticmethod
    def _validate_transition(transition: DeviceTransitionBatch, num_envs: int) -> None:
        expected = (num_envs,)
        if transition.observation.shape != (num_envs, 480):
            raise RuntimeError("device transition observation must be (num_envs, 480)")
        if transition.reward.shape != expected or transition.reset.shape != expected:
            raise RuntimeError("device transition reward/reset must be (num_envs,)")
        if transition.reason_code.shape != expected or transition.deviation_reset.shape != expected:
            raise RuntimeError("device transition diagnostics must be (num_envs,)")
        if str(transition.observation.dtype) != "float32":
            raise RuntimeError("device transition observation must be float32")

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        # Do not clamp/copy ``actions``: PPO records this exact sampled tensor.
        numpy_actions = untensorize_space(
            self.action_space,
            unflatten_tensorized_space(self.action_space, actions),
            squeeze_batch_dimension=not self._vectorized,
        )
        if self._vectorized and isinstance(self.action_space, gymnasium.spaces.Discrete):
            numpy_actions = numpy_actions.flatten()
        transition, info = self._env.step_device(numpy_actions)
        self._validate_transition(transition, self.num_envs)
        observation = jax_to_torch_cuda(transition.observation)
        reward = jax_to_torch_cuda(transition.reward).to(dtype=torch.float32).view(self.num_envs, 1)
        terminated = jax_to_torch_cuda(transition.reset).to(dtype=torch.bool).view(self.num_envs, 1)
        truncated = torch.zeros_like(terminated)
        if observation.device != self.device or reward.device != self.device or terminated.device != self.device:
            raise RuntimeError("JAX-to-Torch DLPack transition changed CUDA device")
        if observation.dtype != torch.float32:
            raise RuntimeError("JAX-to-Torch DLPack observation must be float32")
        if self._vectorized:
            self._observation = observation
            self._info = info
        return observation, reward, terminated, truncated, info


class ManoSkrlRuntime:
    """One shared model, source-normalizers, and skrl PPO over the vector adapter."""

    def __init__(self, environment: ManoGymnasiumVectorEnv, config: ManoPPOConfig = ManoPPOConfig()) -> None:
        if not isinstance(environment, ManoGymnasiumVectorEnv):
            raise TypeError("environment must be a ManoGymnasiumVectorEnv")
        if not isinstance(config, ManoPPOConfig):
            raise TypeError("config must be a ManoPPOConfig")
        self.gymnasium_env = environment
        self.config = config
        self.env = (
            DeviceTransitionGymnasiumWrapper(environment)
            if environment.environment.config.device_transition
            else ProfiledGymnasiumWrapper(environment)
            if config.profile_phases
            else ResettableGymnasiumWrapper(environment)
        )
        self.device = str(self.env.device)
        self.model = ManoActorCritic(
            self.env.observation_space,
            self.env.state_space,
            self.env.action_space,
            device=self.device,
            use_film=config.use_film,
        )
        self.memory = RandomMemory(memory_size=config.rollouts, num_envs=environment.num_envs, device=self.device)
        self.agent = RlGamesPPO(
            models={"policy": self.model, "value": self.model},
            memory=self.memory,
            observation_space=self.env.observation_space,
            state_space=self.env.state_space,
            action_space=self.env.action_space,
            device=self.device,
            cfg=config.skrl_config(
                num_envs=environment.num_envs,
                device=self.device,
                observation_size=environment.observation_dim,
            ),
        )
        # skrl's PPO_CFG rejects source-only fields. The custom PPO update
        # reads this source contract through the agent config after init.
        self.agent.cfg.bounds_loss_coef = config.bounds_loss_coef
        self.agent.init()
        # Checkpoint loading receives the skrl agent rather than the physical
        # environment.  Attach the resolved hand/action signature explicitly
        # so equal-width right- and left-hand policies cannot be interchanged
        # silently.  This attribute is runtime-only; checkpoint persistence
        # remains owned by ``checkpoint_metadata`` below.
        self.agent.manorl_environment_signature = self._environment_signature()

    def conversion_phase_profile(self) -> dict[str, dict[str, float | int]]:
        profile = getattr(self.env, "phase_profile", None)
        return profile() if callable(profile) else {}

    def reset_conversion_phase_profile(self) -> None:
        reset_profile = getattr(self.env, "reset_phase_profile", None)
        if callable(reset_profile):
            reset_profile()

    def reset_done(self, observations: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Reset only terminal rows and merge the returned ``s0`` into observations."""

        mask = torch.as_tensor(done, device=observations.device, dtype=torch.bool).reshape(-1)
        if mask.numel() != self.gymnasium_env.num_envs:
            raise ValueError(f"done must contain {self.gymnasium_env.num_envs} environments")
        if not bool(mask.any()):
            return observations
        reset_observations = self.env.reset_done(mask)
        updated = observations.clone()
        row_ids = torch.nonzero(mask, as_tuple=False).reshape(-1).to(updated.device)
        updated[row_ids] = reset_observations.to(updated.device)[row_ids]
        return updated

    def _environment_signature(self) -> dict[str, object]:
        """Return the checkpoint-relevant resolved hand and tensor layout."""

        physical = self.gymnasium_env.environment
        available_sides = tuple(physical.hand_sides)
        controlled_sides = tuple(physical.hand_layout.controlled_sides)
        return {
            "requested_hand_side": physical.config.hand_side,
            "resolved_hand_side": (
                "both" if len(controlled_sides) == 2 else controlled_sides[0]
            ),
            "available_hand_sides": list(available_sides),
            "controlled_hand_sides": list(controlled_sides),
            "reference_following_hand_sides": list(
                physical.hand_layout.reference_sides
            ),
            "action_dim": int(physical.action_dim),
            "observation_dim": int(physical.observation_dim),
            "model_action_dim": int(physical.model_action_dim),
            "warp_ccd": physical.warp_ccd_metadata(),
        }

    def checkpoint_metadata(self) -> dict[str, object]:
        ppo_metadata = asdict(self.config)
        ppo_metadata.pop("use_film", None)
        physical = self.gymnasium_env.environment
        return {
            "reward_contract": REWARD_CONTRACT_ID,
            "ppo_reward_contract": PPO_REWARD_CONTRACT_ID,
            "ppo_reward_scale": PPO_REWARD_SCALE,
            "environment_contract": ENVIRONMENT_CONTRACT_ID,
            "environment": {
                **self._environment_signature(),
                "residual_enabled": physical.config.residual_enabled,
                "residual_action": asdict(physical.config.residual_action),
                "compatibility": asdict(physical.config.compatibility),
                "point_sampling_backend": physical.config.point_sampling_backend,
                "observation_contact_threshold_N": CONTACT_FORCE_THRESHOLD,
                "reward": asdict(physical.config.reward_config),
                "max_deviation_distance": physical.config.max_deviation_distance,
            },
            "ppo": ppo_metadata,
            "model": {"use_film": self.model.use_film},
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
        pending_done = torch.zeros(
            (self.gymnasium_env.num_envs, 1), device=self.device, dtype=torch.bool
        )
        for timestep in range(self.config.rollouts):
            if bool(pending_done.any()):
                observations = self.reset_done(observations, pending_done)
                pending_done.zero_()
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
            pending_done = terminated | truncated
        after = dict(self.model.named_parameters())
        if not all(torch.isfinite(parameter).all() for parameter in after.values()):
            raise RuntimeError("PPO update produced non-finite model parameters")
        return any(not torch.equal(before[name], parameter.detach()) for name, parameter in after.items())
