"""v4 single-reference PPO loop.

This reuses the mature RlGamesPPO/GAE implementation. PPO memory owns raw 957-D
observations and raw Normal actions; the adapter clips only the physical action.
Terminal next observations are recorded before ``prepare_action`` resets exactly
``runtime.last_done`` rows.
"""
from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from sim.manorl.autonomy_contracts import (
    ACTION_CONTRACT_ID, CHECKPOINT_FORMAT, OBSERVATION_CONTRACT_ID,
    POLICY_SAMPLING_CONTRACT, REWARD_CONTRACT_ID, validate_v4_checkpoint_metadata,
)
from sim.manorl.autonomy_telemetry import latest_ppo_metrics
from sim.manorl.autonomy_v4_telemetry import V4TelemetryAccumulator
from sim.manorl.autonomy_training import build_batched_runtime


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_paths(value: Any, path: str) -> list[str]:
    if isinstance(value, torch.Tensor):
        return [path] if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()) else []
    if isinstance(value, dict):
        return [p for key, child in value.items() for p in _finite_paths(child, f"{path}.{key}")]
    if isinstance(value, (tuple, list)):
        return [p for i, child in enumerate(value) for p in _finite_paths(child, f"{path}[{i}]")]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return [path]
    return []
def _assert_finite(*, model: torch.nn.Module, agent: Any, row: dict[str, float], checkpoint: Path | None, update: int) -> None:
    bad = _finite_paths(dict(model.named_parameters()), "model") + _finite_paths(agent.optimizer.state, "optimizer")
    bad_metrics = [name for name, value in row.items() if not math.isfinite(float(value))]
    if not bad and not bad_metrics:
        return
    if checkpoint is not None:
        diagnostic = checkpoint.with_name(f"{checkpoint.stem}.nonfinite-update{update:06d}.json")
        diagnostic.write_text(json.dumps({"failure": "non-finite PPO update", "update": update,
            "model_or_optimizer": bad, "metrics": bad_metrics,
            "last_finite_checkpoint": str(checkpoint)}, indent=2) + "\n")
    raise RuntimeError(f"non-finite PPO update {update}: {bad + bad_metrics}")


def checkpoint_payload(*, model: torch.nn.Module, agent: Any, config: dict[str, Any],
                       provenance: dict[str, Any], policy_steps: int, environment_transitions: int) -> dict[str, Any]:
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "observation_contract": OBSERVATION_CONTRACT_ID,
        "reward_contract": REWARD_CONTRACT_ID,
        "action_contract": ACTION_CONTRACT_ID,
        "policy_sampling_contract": dict(POLICY_SAMPLING_CONTRACT),
        "model": model.state_dict(), "model_architecture": model.checkpoint_architecture(),
        "optimizer": agent.optimizer.state_dict(), "normalizer": None,
        "policy_steps": int(policy_steps), "environment_transitions": int(environment_transitions),
        "config": config, "provenance": provenance,
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(*, checkpoint: str | Path, model: torch.nn.Module, agent: Any,
                    config: dict[str, Any], provenance: dict[str, Any], policy_steps: int,
                    environment_transitions: int, update: int, periodic: bool) -> None:
    path = Path(checkpoint)
    payload = checkpoint_payload(model=model, agent=agent, config=config, provenance=provenance,
                                 policy_steps=policy_steps, environment_transitions=environment_transitions)
    if periodic:
        _atomic_torch_save(payload, path.with_name(f"{path.stem}.update{update:06d}{path.suffix}"))
    _atomic_torch_save(payload, path)
    if not periodic:
        _atomic_torch_save(payload, path.with_name(f"{path.stem}.final{path.suffix}"))


def _validate_provenance(actual: Any, expected: Any, path: str = "provenance") -> None:
    """Require every expected provenance leaf while allowing recorded diagnostics."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"checkpoint provenance mismatch for {path}")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"checkpoint provenance mismatch for {path}.{key}")
            _validate_provenance(actual[key], value, f"{path}.{key}")
    elif actual != expected:
        raise ValueError(f"checkpoint provenance mismatch for {path}")


_POLICY_STATE_PREFIXES = ("pointnet.", "net.", "mean.")


def _is_policy_state(name: str) -> bool:
    return name == "log_std" or name.startswith(_POLICY_STATE_PREFIXES)


def _warmstart_transfer_mode(source: Any, target: dict[str, Any]) -> str:
    if source == target:
        return "exact_model"
    if not isinstance(source, dict):
        raise ValueError("checkpoint/model architecture mismatch")
    source_policy = {key: value for key, value in source.items() if key != "value_trunk"}
    target_policy = {key: value for key, value in target.items() if key != "value_trunk"}
    if source.get("value_trunk") == "shared" and target.get("value_trunk") == "separate" and source_policy == target_policy:
        return "shared_policy_to_separate_critic"
    raise ValueError("checkpoint/model architecture mismatch")


def inspect_v4_warmstart(path: str | Path, target_architecture: dict[str, Any], *,
                         map_location: str | torch.device = "cpu",
                         expected_provenance: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    """Validate public metadata/provenance and resolve the only supported transfer mode."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    validate_v4_checkpoint_metadata(payload)
    mode = _warmstart_transfer_mode(payload.get("model_architecture"), target_architecture)
    _validate_provenance(payload.get("provenance", {}), expected_provenance or {})
    return payload, mode


def _load_v4_model_state(path: str | Path, model: torch.nn.Module, *,
                         map_location: str | torch.device,
                         expected_provenance: dict[str, Any] | None,
                         allow_shared_policy_transfer: bool = False) -> tuple[dict[str, Any], str]:
    payload, mode = inspect_v4_warmstart(path, model.checkpoint_architecture(),
                                         map_location=map_location,
                                         expected_provenance=expected_provenance)
    if mode == "exact_model":
        model.load_state_dict(payload["model"], strict=True)
        return payload, mode
    if not allow_shared_policy_transfer:
        raise ValueError("checkpoint/model architecture mismatch")

    source_state = payload.get("model")
    if not isinstance(source_state, dict):
        raise ValueError("checkpoint model state must be a mapping")
    target_state = model.state_dict()
    expected_source_keys = set(target_state) - {name for name in target_state if name.startswith("value_net.")}
    if set(source_state) != expected_source_keys:
        raise ValueError("shared warm-start model state does not match the declared architecture")
    for name, source_value in source_state.items():
        target_value = target_state[name]
        if (not isinstance(source_value, torch.Tensor) or source_value.shape != target_value.shape
                or source_value.dtype != target_value.dtype):
            raise ValueError(f"shared warm-start tensor mismatch for {name}")
    transferred = {name: value for name, value in source_state.items() if _is_policy_state(name)}
    if set(transferred) != {name for name in target_state if _is_policy_state(name)}:
        raise ValueError("shared warm-start policy state is incomplete")
    target_state.update(transferred)
    model.load_state_dict(target_state, strict=True)
    return payload, mode


def load_v4_warmstart(path: str | Path, model: torch.nn.Module, *,
                      map_location: str | torch.device = "cpu",
                      expected_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load exact v4 weights or shared policy weights into a separate-critic target."""
    payload, mode = _load_v4_model_state(path, model, map_location=map_location,
                                         expected_provenance=expected_provenance,
                                         allow_shared_policy_transfer=True)
    return {**payload, "warmstart_transfer_mode": mode}


def load_frozen_v4(path: str | Path, model: torch.nn.Module, *, map_location: str | torch.device = "cpu",
                   expected_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    payload, _ = _load_v4_model_state(path, model, map_location=map_location,
                                      expected_provenance=expected_provenance)
    model.eval()
    return payload


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_batched_ppo(adapter: Any, *, updates: int, rollouts: int, learning_epochs: int,
                    mini_batches: int, checkpoint: str | Path | None = None,
                    checkpoint_interval: int = 16, config: dict[str, Any] | None = None,
                    provenance: dict[str, Any] | None = None, separate_critic: bool = False,
                    warmstart: str | Path | None = None,
                    expected_warmstart_provenance: dict[str, Any] | None = None,
                    on_update: Callable[[dict[str, float]], None] | None = None) -> tuple[torch.nn.Module, Any, list[dict[str, float]]]:
    """Execute canonical PPO with finite-horizon terminal bootstrapping.

    ``terminated`` is the environment's physical/reward boundary. Rollout cuts
    retain it as false, so RlGamesPPO bootstraps from the recorded terminal-next
    observation; true terminations do not bootstrap and reset afterward.
    """
    if min(updates, rollouts, learning_epochs, mini_batches) < 1:
        raise ValueError("updates, rollouts, learning_epochs and mini_batches must be positive")
    if rollouts * adapter.num_envs < mini_batches:
        raise ValueError("mini-batches cannot exceed rollout transitions")
    model, agent = build_batched_runtime(adapter, rollouts=rollouts, learning_epochs=learning_epochs,
                                         mini_batches=mini_batches, device=str(adapter.device), separate_critic=separate_critic)
    config, provenance = dict(config or {}), dict(provenance or {})
    warmstart_checkpoint = str(Path(warmstart).expanduser().resolve()) if warmstart is not None else None
    transfer_mode = None
    # The PPO agent and its full actor/value Adam are always new. A warm-start
    # supplies model weights only; teacher optimizer/progress/RNG are ignored.
    if warmstart is not None:
        warmstart_payload = load_v4_warmstart(warmstart, model, map_location=adapter.device,
                                              expected_provenance=expected_warmstart_provenance)
        transfer_mode = warmstart_payload["warmstart_transfer_mode"]
    lineage = {"separate_critic": bool(separate_critic), "warmstart_checkpoint": warmstart_checkpoint,
               "warmstart_transfer_mode": transfer_mode,
               "mode": "ppo_warmstart" if warmstart_checkpoint else "ppo_from_scratch"}
    config.update(lineage); provenance.update(lineage)
    agent.enable_training_mode(True, apply_to_models=True)
    observations, _ = adapter.reset()
    episode_return = torch.zeros((adapter.num_envs, 1), device=adapter.device)
    episode_length = torch.zeros_like(episode_return)
    rows: list[dict[str, float]] = []
    policy_steps = 0
    telemetry = V4TelemetryAccumulator(adapter.num_envs, adapter.device) if hasattr(adapter, "telemetry_snapshot") else None
    for update in range(1, updates + 1):
        _sync(adapter.device); sampled = time.perf_counter()
        reward_sum = torch.zeros((), device=adapter.device); done_count = torch.zeros((), device=adapter.device)
        valid = torch.ones((), dtype=torch.bool, device=adapter.device)
        completed_return = torch.zeros((), device=adapter.device); completed_length = torch.zeros((), device=adapter.device)
        completed_count = torch.zeros((), device=adapter.device)
        summaries = {name: torch.zeros((), device=adapter.device) for name in ("object_motion", "contact_force", "path")}
        for _ in range(rollouts):
            with torch.no_grad():
                # This action is the unmodified Normal sample used by PPO likelihood.
                actions, _ = agent.act(observations, None, timestep=policy_steps, timesteps=updates * rollouts)
            terminal_next, rewards, terminated, info = adapter.step(actions)
            if rewards.shape != (adapter.num_envs, 1) or terminated.shape != (adapter.num_envs, 1):
                raise RuntimeError("adapter must return (B,1) reward and done")
            agent.record_transition(observations=observations, states=None, actions=actions, rewards=rewards,
                                    next_observations=terminal_next, next_states=None, terminated=terminated,
                                    truncated=torch.zeros_like(terminated), infos=info,
                                    timestep=policy_steps, timesteps=updates * rollouts)
            reward_sum += rewards.sum(); done_count += terminated.sum(); valid &= adapter._to_torch(info["valid"]).bool().all()
            episode_return += rewards; episode_length += 1
            completed_return += torch.where(terminated, episode_return, torch.zeros_like(episode_return)).sum()
            completed_length += torch.where(terminated, episode_length, torch.zeros_like(episode_length)).sum(); completed_count += terminated.sum()
            episode_return = torch.where(terminated, torch.zeros_like(episode_return), episode_return)
            episode_length = torch.where(terminated, torch.zeros_like(episode_length), episode_length)
            if telemetry is not None:
                # Captures cached t+1 reward/physical/contact facts before reset.
                telemetry.add(adapter.telemetry_snapshot(actions))
            for name, value in adapter.compact_summary().items(): summaries[name] += value
            # Must happen after record_transition: reset source is runtime.last_done.
            observations = adapter.prepare_action(); policy_steps += 1
        _sync(adapter.device); sample_seconds = time.perf_counter() - sampled
        if not bool(valid.cpu()): raise RuntimeError("invalid v4 physics/contact reduction; checkpoint withheld")
        optimized = time.perf_counter(); agent.update(timestep=policy_steps, timesteps=updates * rollouts); _sync(adapter.device)
        optimize_seconds = time.perf_counter() - optimized
        count = float(adapter.num_envs * rollouts); transitions = policy_steps * adapter.num_envs
        row = {"update": float(update), "policy_steps": float(policy_steps), "transitions": float(transitions),
               "window_transitions": count, "reward_mean": float(reward_sum.cpu()) / count,
               "terminations": float(done_count.cpu()), "valid": float(valid.cpu()),
               "performance/sampling_time": sample_seconds, "performance/optimizer_time": optimize_seconds,
               "physics/window_object_z_mean": float(summaries["object_motion"].cpu()) / count,
               "physics/window_contact_force_mean": float(summaries["contact_force"].cpu()) / count,
               "physics/window_path_error_mean": float(summaries["path"].cpu()) / count}
        if telemetry is not None:
            # Scalar egress occurs once/update; episode state inside telemetry spans rollout cuts.
            row.update(telemetry.reduce(update=update, transitions=int(transitions)))
        elif float(completed_count.cpu()):
            row.update({"episodes/completed_count": float(completed_count.cpu()),
                        "episodes/return_mean": float(completed_return.cpu() / completed_count.cpu()),
                        "episodes/length_mean": float(completed_length.cpu() / completed_count.cpu())})
        row.update(latest_ppo_metrics(agent)); _assert_finite(model=model, agent=agent, row=row, checkpoint=None if checkpoint is None else Path(checkpoint), update=update)
        rows.append(row)
        if checkpoint is not None and update % checkpoint_interval == 0:
            save_checkpoint(checkpoint=checkpoint, model=model, agent=agent, config=config, provenance=provenance,
                            policy_steps=policy_steps, environment_transitions=transitions, update=update, periodic=True)
        if on_update is not None: on_update(row)
    if checkpoint is not None:
        save_checkpoint(checkpoint=checkpoint, model=model, agent=agent, config=config, provenance=provenance,
                        policy_steps=policy_steps, environment_transitions=policy_steps * adapter.num_envs,
                        update=updates, periodic=False)
    return model, agent, rows
