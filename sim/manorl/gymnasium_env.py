"""Gymnasium vector adapter for the bounded ManoRL MJX environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space
from numpy.typing import NDArray

from sim.manorl.contracts import JOINT_DOF
from sim.manorl.environment import MujocoManoEnvironment
from sim.manorl.observations import OBSERVATION_DIM_28

OBSERVATION_DIM = OBSERVATION_DIM_28
ACTION_DIM = JOINT_DOF


class ManoGymnasiumVectorEnv(gym.vector.VectorEnv):
    """Expose :class:`MujocoManoEnvironment` with Gymnasium vector semantics.

    The physical environment returns its terminal observation and supports an
    explicit indexed reset on the following policy boundary. ``NEXT_STEP`` is
    retained as the compatibility mode for callers that still use the legacy
    source loop. The adapter latches terminal IDs and consumes them with an
    indexed physical reset immediately before the next direct ``step`` action;
    explicit indexed/full ``reset`` calls clear the corresponding latch.
    ``time_outs`` becomes ``truncated``; deviation/trajectory ends are always
    ``terminated``.
    """

    metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP, "render_modes": []}

    def __init__(self, environment: MujocoManoEnvironment) -> None:
        if not isinstance(environment, MujocoManoEnvironment):
            raise TypeError("environment must be a MujocoManoEnvironment")
        self.environment = environment
        self.num_envs = environment.config.num_envs
        # Real environments expose resolved 28/56-wide layouts. The fallbacks
        # keep lightweight adapter test doubles on the single-hand MuJoCo ABI.
        self.observation_dim = int(
            getattr(environment, "observation_dim", OBSERVATION_DIM)
        )
        self.action_dim = int(getattr(environment, "action_dim", ACTION_DIM))
        self.single_observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(self.observation_dim,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.render_mode = None
        self._last_seed: int | None = None
        self._pending_reset = np.zeros(self.num_envs, dtype=bool)

    def _pending_reset_mask(self) -> NDArray[np.bool_]:
        pending = getattr(self, "_pending_reset", None)
        if pending is None or pending.shape != (self.num_envs,):
            pending = np.zeros(self.num_envs, dtype=bool)
            self._pending_reset = pending
        return pending

    def _normalize_reset_ids(self, env_ids: Any) -> NDArray[np.int64]:
        ids = np.asarray(env_ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("env_ids must be a one-dimensional integer array")
        ids = np.unique(ids.astype(np.int64, copy=False))
        if np.any(ids < 0) or np.any(ids >= self.num_envs):
            raise ValueError("env_ids contains an invalid environment index")
        return ids

    def _consume_pending_resets(self) -> None:
        pending = self._pending_reset_mask()
        env_ids = np.flatnonzero(pending).astype(np.int64)
        if env_ids.size == 0:
            return
        # Keep this at the adapter boundary so direct callers get s0 before
        # their next action even when they bypass the skrl runtime helper.
        self.environment.reset(env_ids=env_ids)
        pending[env_ids] = False

    @property
    def pending_terminal_env_ids(self) -> NDArray[np.int64]:
        """Return terminal worlds awaiting an explicit/direct-step reset."""

        return np.flatnonzero(self._pending_reset_mask()).astype(np.int64)

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
        raw_env_ids = None if options is None else options.get("env_ids")
        env_ids = None if raw_env_ids is None else self._normalize_reset_ids(raw_env_ids)
        output = self.environment.reset(env_ids=env_ids)
        observation = np.asarray(output["obs"], dtype=np.float32)
        observation_dim = int(getattr(self, "observation_dim", OBSERVATION_DIM))
        if observation.shape != (self.num_envs, observation_dim):
            raise RuntimeError("physical environment returned an invalid observation batch")
        pending = self._pending_reset_mask()
        if env_ids is None:
            pending[:] = False
        else:
            pending[env_ids] = False
        return observation, {
            "seed": self._last_seed,
            "time_outs": np.zeros(self.num_envs, dtype=bool),
            "reset_applied": np.ones(self.num_envs, dtype=bool)
            if env_ids is None
            else np.isin(np.arange(self.num_envs), env_ids),
        }

    def reset_done(self, done: NDArray[object]) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset only environments whose previous transition was terminal."""

        mask = np.asarray(done, dtype=bool).reshape(-1)
        if mask.shape != (self.num_envs,):
            raise ValueError(f"done must have shape ({self.num_envs},)")
        env_ids = np.flatnonzero(mask).astype(np.int64)
        if env_ids.size == 0:
            raise ValueError("reset_done requires at least one terminal environment")
        return self.reset(options={"env_ids": env_ids})

    def step(
        self, actions: NDArray[object]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_], dict[str, Any]]:
        actions_array = np.asarray(actions, dtype=np.float64)
        expected_dim = int(getattr(self, "action_dim", ACTION_DIM))
        if actions_array.shape != (self.num_envs, expected_dim) or not np.all(
            np.isfinite(actions_array)
        ):
            raise ValueError(f"actions must be finite ({self.num_envs}, {expected_dim})")
        self._consume_pending_resets()
        # Match IsaacGym VecTask: PPO keeps raw samples, while the environment
        # clips the normalized action at its boundary before physical processing.
        clipped_actions = np.clip(actions_array, -1.0, 1.0)
        output, rewards, reset, extras = self.environment.step(clipped_actions)
        observation = np.asarray(output["obs"], dtype=np.float32)
        reward = np.asarray(rewards, dtype=np.float32)
        done = np.asarray(reset, dtype=bool)
        time_outs = np.asarray(extras["time_outs"], dtype=bool)
        expected_observation_dim = getattr(
            self,
            "observation_dim",
            OBSERVATION_DIM,
        )
        if observation.shape != (self.num_envs, expected_observation_dim) or reward.shape != (self.num_envs,):
            raise RuntimeError("physical environment returned an invalid Gymnasium batch")
        if time_outs.shape != (self.num_envs,) or np.any(time_outs & ~done):
            raise RuntimeError("time_outs must be a subset of reset signals")
        truncated = time_outs
        terminated = done & ~truncated
        termination = self.environment.last_termination
        if termination is None:
            raise RuntimeError("physical environment omitted termination diagnostics")
        reason_code = np.asarray(termination.reason_code, dtype=np.int32)
        success = np.asarray(termination.success, dtype=bool)
        failure = np.asarray(termination.failure, dtype=bool)
        if (
            reason_code.shape != (self.num_envs,)
            or success.shape != (self.num_envs,)
            or failure.shape != (self.num_envs,)
        ):
            raise RuntimeError("termination diagnostics have an invalid vector shape")
        infos: dict[str, Any] = {
            "time_outs": time_outs.copy(),
            "termination_reason_code": reason_code.copy(),
            "termination_success": success.copy(),
            "termination_failure": failure.copy(),
            "trajectory_complete_reset_mask": success.copy(),
            "deviation_reset_mask": np.asarray(termination.deviation_reset, dtype=bool).copy(),
            "success": success.copy(),
            "failure": failure.copy(),
        }
        if np.any(done):
            # The observation returned at done is source-terminal, not reset
            # state. Gymnasium consumers that need it can read this standard key.
            infos["final_observation"] = observation.copy()
            infos["_final_observation"] = done.copy()
        self._pending_reset_mask()[:] = done
        return observation, reward, terminated, truncated, infos

    def close_extras(self, **kwargs: Any) -> None:
        del kwargs

    def render(self) -> None:
        return None
