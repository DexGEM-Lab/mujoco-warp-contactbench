"""Gymnasium vector adapter for the bounded ManoRL MJX environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space
from numpy.typing import NDArray

from sim.manorl.environment import MujocoManoEnvironment

OBSERVATION_DIM = 476
ACTION_DIM = 26


class ManoGymnasiumVectorEnv(gym.vector.VectorEnv):
    """Expose :class:`MujocoManoEnvironment` with Gymnasium vector semantics.

    The physical environment deliberately returns its terminal observation and
    resets physical state on the following call, matching the source VecTask.
    This adapter declares Gymnasium's ``NEXT_STEP`` autoreset mode and preserves
    that terminal observation. ``time_outs`` becomes ``truncated`` so PPO can
    bootstrap it; deviation/trajectory failures become ``terminated``.
    """

    metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP, "render_modes": []}

    def __init__(self, environment: MujocoManoEnvironment) -> None:
        if not isinstance(environment, MujocoManoEnvironment):
            raise TypeError("environment must be a MujocoManoEnvironment")
        self.environment = environment
        self.num_envs = environment.config.num_envs
        self.single_observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(OBSERVATION_DIM,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.render_mode = None
        self._last_seed: int | None = None

    @property
    def device(self) -> str:
        """Torch-facing device selection; physical backend is separately MJX-Warp."""

        if self.environment.config.device != "gpu":
            return "cpu"
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        if options:
            unknown = set(options) - {"env_ids"}
            if unknown:
                raise ValueError(f"unsupported reset options: {sorted(unknown)}")
        super().reset(seed=seed)
        if seed is not None:
            self._last_seed = int(seed)
            # Dynamic source-compatible point templates are reset-local. Static
            # seed-42 templates remain unchanged, as prescribed by the source.
            self.environment.reseed_point_templates(self._last_seed)
        env_ids = None if options is None else options.get("env_ids")
        output = self.environment.reset(env_ids=env_ids)
        observation = np.asarray(output["obs"], dtype=np.float32)
        if observation.shape != (self.num_envs, OBSERVATION_DIM):
            raise RuntimeError("physical environment returned an invalid observation batch")
        return observation, {"seed": self._last_seed, "time_outs": np.zeros(self.num_envs, dtype=bool)}

    def step(
        self, actions: NDArray[object]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_], dict[str, Any]]:
        actions_array = np.asarray(actions, dtype=np.float64)
        if actions_array.shape != (self.num_envs, ACTION_DIM) or not np.all(np.isfinite(actions_array)):
            raise ValueError(f"actions must be finite ({self.num_envs}, {ACTION_DIM})")
        # The physical action processor clips at the source action bound. Reject
        # only non-finite/shape errors here so the source clipping remains owner.
        output, rewards, reset, extras = self.environment.step(actions_array)
        observation = np.asarray(output["obs"], dtype=np.float32)
        reward = np.asarray(rewards, dtype=np.float32)
        done = np.asarray(reset, dtype=bool)
        time_outs = np.asarray(extras["time_outs"], dtype=bool)
        if observation.shape != (self.num_envs, OBSERVATION_DIM) or reward.shape != (self.num_envs,):
            raise RuntimeError("physical environment returned an invalid Gymnasium batch")
        if time_outs.shape != (self.num_envs,) or np.any(time_outs & ~done):
            raise RuntimeError("time_outs must be a subset of reset signals")
        truncated = time_outs
        terminated = done & ~truncated
        infos: dict[str, Any] = {"time_outs": time_outs.copy()}
        if np.any(done):
            # The observation returned at done is source-terminal, not reset
            # state. Gymnasium consumers that need it can read this standard key.
            infos["final_observation"] = observation.copy()
            infos["_final_observation"] = done.copy()
        return observation, reward, terminated, truncated, infos

    def close_extras(self, **kwargs: Any) -> None:
        del kwargs

    def render(self) -> None:
        return None
