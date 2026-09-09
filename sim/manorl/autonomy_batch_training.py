"""Direct CUDA rollout boundary for the v3 homogeneous autonomy runtime.

The runtime owns reset/physics; PPO owns GAE and optimisation.  This module
keeps their tensor boundary explicit so a terminal state is recorded before
only that row is reset for the next policy action.
"""
from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sim.manorl.autonomy_contracts import (
    ACTION_V3_CONTRACT_ID, CHECKPOINT_V3_FORMAT, OBSERVATION_V3_CONTRACT_ID,
    REWARD_V3_CONTRACT_ID, validate_v3_checkpoint_metadata,
)
from sim.manorl.autonomy_training import build_batched_runtime
from sim.manorl.autonomy_telemetry import latest_ppo_metrics


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_payload(*, model: torch.nn.Module, agent: Any, config: dict[str, Any], provenance: dict[str, Any], step: int) -> dict[str, Any]:
    return {
        "checkpoint_format": CHECKPOINT_V3_FORMAT,
        "observation_contract": OBSERVATION_V3_CONTRACT_ID,
        "reward_contract": REWARD_V3_CONTRACT_ID,
        "action_contract": ACTION_V3_CONTRACT_ID,
        "model": model.state_dict(), "optimizer": agent.optimizer.state_dict(),
        "global_policy_step": int(step), "config": config, "provenance": provenance,
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def load_frozen_v3(path: str | Path, model: torch.nn.Module, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    validate_v3_checkpoint_metadata(payload)
    if payload.get("action_contract") != ACTION_V3_CONTRACT_ID:
        raise ValueError("incompatible v3 autonomy action contract")
    model.load_state_dict(payload["model"])
    model.eval()
    return payload


def _compact_window(runtime: Any, reward_sum: torch.Tensor, done_count: torch.Tensor, valid: torch.Tensor, sample_seconds: float, optimize_seconds: float, transitions: int, update: int, episode_return: torch.Tensor, episode_length: torch.Tensor) -> dict[str, float]:
    """Transfer only update reductions, never per-env contact/info rows."""
    device_summary = {
        "reward_sum": reward_sum, "done_count": done_count, "valid": valid,
        "episode_return_sum": episode_return.sum(), "episode_length_sum": episode_length.sum(),
        **runtime.compact_summary(),
    }
    # DLPack is used until this compact, once-per-update scalar egress.
    host = {name: float(tensor.detach().cpu()) for name, tensor in device_summary.items()}
    # A real adapter stores telemetry in torch; fake adapters can return CPU tensors.
    count = max(transitions, 1)
    result = {
        "update": float(update), "transitions": float(transitions), "global_step": float(transitions),
        "reward_mean": host["reward_sum"] / count, "terminations": host["done_count"],
        "valid_rows": host["valid"], "performance/sampling_time": sample_seconds,
        "performance/optimizer_time": optimize_seconds,
        "performance/total_transitions_per_second": count / max(sample_seconds + optimize_seconds, 1e-12),
        "performance/sampling_transitions_per_second": count / max(sample_seconds, 1e-12),
        "physics/object_z_sum": host["object_motion"], "physics/contact_force_sum": host["contact_force"],
        "physics/path_error_sum": host["path"],
    }
    return result


def run_batched_ppo(adapter: Any, *, updates: int, rollouts: int, learning_epochs: int, mini_batches: int, checkpoint: str | Path | None = None, checkpoint_interval: int = 16, config: dict[str, Any] | None = None, provenance: dict[str, Any] | None = None) -> tuple[torch.nn.Module, Any, list[dict[str, float]]]:
    """Run PPO against an adapter exposing reset/step/prepare_action tensors.

    `step` returns terminal *next* observations. They are recorded before
    `prepare_action` mutates only completed rows. This is essential for GAE
    on mixed terminal/nonterminal batches.
    """
    if updates < 1 or rollouts < 1 or learning_epochs < 1 or mini_batches < 1:
        raise ValueError("updates, rollouts, learning_epochs and mini_batches must be positive")
    if rollouts * adapter.num_envs < mini_batches:
        raise ValueError("mini-batches cannot exceed rollout transitions")
    model, agent = build_batched_runtime(adapter, rollouts=rollouts, learning_epochs=learning_epochs, mini_batches=mini_batches, device=str(adapter.device))
    agent.enable_training_mode(True, apply_to_models=True)
    observations, _ = adapter.reset()
    device = observations.device
    episode_return = torch.zeros((adapter.num_envs, 1), device=device)
    episode_length = torch.zeros((adapter.num_envs, 1), device=device)
    rows: list[dict[str, float]] = []
    global_step = 0
    config, provenance = config or {}, provenance or {}
    for update in range(1, updates + 1):
        sample_started = time.perf_counter()
        reward_sum = torch.zeros((), device=device); done_count = torch.zeros((), device=device); valid_all = torch.ones((), device=device)
        for _ in range(rollouts):
            with torch.no_grad():
                actions, _ = agent.act(observations, None, timestep=global_step, timesteps=updates * rollouts)
                # GaussianMixin clips to action space. Store the exact bounded
                # command that crosses DLPack and enters the physical rate map.
                actions = torch.clamp(actions, -1.0, 1.0)
            terminal_next, rewards, terminated, info = adapter.step(actions)
            truncated = torch.zeros_like(terminated)
            if rewards.shape != (adapter.num_envs, 1) or terminated.shape != (adapter.num_envs, 1):
                raise RuntimeError("batched runtime must return rewards/dones shaped (B, 1)")
            agent.record_transition(observations=observations, states=None, actions=actions, rewards=rewards,
                                    next_observations=terminal_next, next_states=None, terminated=terminated,
                                    truncated=truncated, infos=info, timestep=global_step, timesteps=updates * rollouts)
            reward_sum += rewards.sum(); done_count += terminated.sum()
            episode_return += rewards; episode_length += 1
            # Preserve episode sums across rollout cuts. Completed rows are
            # reduced then zeroed on device; live neighbours retain their state.
            episode_return = torch.where(terminated, torch.zeros_like(episode_return), episode_return)
            episode_length = torch.where(terminated, torch.zeros_like(episode_length), episode_length)
            valid = adapter._to_torch(info["valid"]).reshape(adapter.num_envs, 1).to(torch.bool)
            valid_all = valid_all * valid.all().to(valid_all.dtype)
            # terminal_next remains in agent's current transition. prepare_action
            # is called only afterwards to obtain next actor observation.
            observations = adapter.prepare_action()
            global_step += 1
        if not bool(valid_all.detach().cpu()):
            raise RuntimeError("invalid physics/contact reduction in batched PPO update; checkpoint withheld")
        sampling_seconds = time.perf_counter() - sample_started
        optimize_started = time.perf_counter()
        # PPO's rollout counter is independent of public global policy steps;
        # call update directly so the learning schedule never restarts per update.
        agent.update(timestep=global_step, timesteps=updates * rollouts)
        optimizer_seconds = time.perf_counter() - optimize_started
        transitions = global_step * adapter.num_envs
        row = _compact_window(adapter, reward_sum, done_count, valid_all, sampling_seconds, optimizer_seconds, transitions, update, episode_return, episode_length)
        row.update(latest_ppo_metrics(agent)); rows.append(row)
        if checkpoint is not None and update % checkpoint_interval == 0:
            _atomic_torch_save(checkpoint_payload(model=model, agent=agent, config=config, provenance=provenance, step=transitions), Path(checkpoint))
    if checkpoint is not None:
        _atomic_torch_save(checkpoint_payload(model=model, agent=agent, config=config, provenance=provenance, step=global_step * adapter.num_envs), Path(checkpoint))
    return model, agent, rows
