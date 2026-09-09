"""Direct CUDA rollout boundary for the v3 homogeneous autonomy runtime.

The runtime owns reset/physics; PPO owns GAE and optimisation. This boundary
records physical terminal observations before masked reset and only reduces
compact device telemetry at PPO update boundaries.
"""
from __future__ import annotations

import hashlib
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
    ACTION_V3_CONTRACT_ID, CHECKPOINT_V3_FORMAT, OBSERVATION_V3_CONTRACT_ID,
    POLICY_SAMPLING_CONTRACT, REWARD_V3_CONTRACT_ID, validate_v3_checkpoint_metadata,
)
from sim.manorl.autonomy_training import build_batched_runtime
from sim.manorl.autonomy_telemetry import latest_ppo_metrics


_WARMSTART_COMPATIBILITY_KEYS = (
    "asset_pin", "package_digest", "manifest_sha256", "catalog_digest",
    "identity_split", "witness_digest", "clock", "v3_contract",
)


def _warmstart_compatibility_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Return the physical contracts which must agree for weights-only initialization."""
    return {key: provenance[key] for key in _WARMSTART_COMPATIBILITY_KEYS}


def _checkpoint_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_payload(*, model: torch.nn.Module, agent: Any, config: dict[str, Any], provenance: dict[str, Any], policy_steps: int, environment_transitions: int) -> dict[str, Any]:
    if not hasattr(model, "checkpoint_architecture"):
        raise TypeError("v3 checkpoint model must declare its architecture")
    cuda_rng = None
    if torch.cuda.is_available():
        cuda_rng = torch.cuda.get_rng_state_all()
    return {
        "checkpoint_format": CHECKPOINT_V3_FORMAT,
        "observation_contract": OBSERVATION_V3_CONTRACT_ID,
        "reward_contract": REWARD_V3_CONTRACT_ID,
        "action_contract": ACTION_V3_CONTRACT_ID,
        "policy_sampling_contract": dict(POLICY_SAMPLING_CONTRACT),
        "model": model.state_dict(), "model_architecture": model.checkpoint_architecture(),
        "optimizer": agent.optimizer.state_dict(),
        # Retain prior name/meaning for consumers that already interpret it as
        # environment transitions. The explicit fields remove that ambiguity.
        "global_policy_step": int(environment_transitions),
        "policy_steps": int(policy_steps), "environment_transitions": int(environment_transitions),
        "config": config, "provenance": provenance,
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(), "cuda_rng": cuda_rng,
    }


def _save_checkpoint(*, checkpoint: str | Path, model: torch.nn.Module, agent: Any, config: dict[str, Any], provenance: dict[str, Any], policy_steps: int, environment_transitions: int, update: int, periodic: bool) -> None:
    path = Path(checkpoint)
    payload = checkpoint_payload(model=model, agent=agent, config=config, provenance=provenance,
                                 policy_steps=policy_steps, environment_transitions=environment_transitions)
    if periodic:
        numbered = path.with_name(f"{path.stem}.update{update:06d}{path.suffix}")
        _atomic_torch_save(payload, numbered)
    _atomic_torch_save(payload, path)  # atomic latest
    if not periodic:
        final = path.with_name(f"{path.stem}.final{path.suffix}")
        _atomic_torch_save(payload, final)


def _finite_tensor_paths(value: Any, *, path: str) -> list[str]:
    """Return tensor/scalar paths which cannot safely feed another PPO update."""
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            return [path]
        return []
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in _finite_tensor_paths(child, path=f"{path}.{key}")]
    if isinstance(value, (list, tuple)):
        return [item for index, child in enumerate(value) for item in _finite_tensor_paths(child, path=f"{path}[{index}]")]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return [path]
    return []


def _nonfinite_update_diagnostic(*, model: torch.nn.Module, agent: Any, row: dict[str, float],
                                 checkpoint: str | Path | None, update: int,
                                 policy_steps: int, environment_transitions: int) -> Path | None:
    """Persist the exact failed state boundary; never replace the last finite save."""
    bad_paths = _finite_tensor_paths(dict(model.named_parameters()), path="model")
    bad_paths.extend(_finite_tensor_paths(agent.optimizer.state, path="optimizer"))
    bad_metrics = [name for name, value in row.items() if not math.isfinite(float(value))]
    if not bad_paths and not bad_metrics:
        return None
    diagnostic = {
        "failure": "non-finite PPO update",
        "update": update,
        "policy_steps": policy_steps,
        "environment_transitions": environment_transitions,
        "nonfinite_model_or_optimizer_paths": bad_paths,
        "nonfinite_metrics": bad_metrics,
        "last_finite_checkpoint": None if checkpoint is None else str(Path(checkpoint)),
        "policy_sampling_contract": POLICY_SAMPLING_CONTRACT,
    }
    if checkpoint is None:
        return None
    path = Path(checkpoint).with_name(f"{Path(checkpoint).stem}.nonfinite-update{update:06d}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")
    return path


def _raise_if_nonfinite_update(*, model: torch.nn.Module, agent: Any, row: dict[str, float],
                                checkpoint: str | Path | None, update: int,
                                policy_steps: int, environment_transitions: int) -> None:
    diagnostic = _nonfinite_update_diagnostic(
        model=model, agent=agent, row=row, checkpoint=checkpoint, update=update,
        policy_steps=policy_steps, environment_transitions=environment_transitions,
    )
    if diagnostic is not None:
        raise RuntimeError(f"non-finite PPO update {update}; diagnostic written to {diagnostic}; last finite checkpoint retained")
    # With no checkpoint destination there is nowhere durable to write the
    # requested diagnostic, but the failure still must surface synchronously.
    bad_paths = _finite_tensor_paths(dict(model.named_parameters()), path="model")
    bad_paths.extend(_finite_tensor_paths(agent.optimizer.state, path="optimizer"))
    bad_metrics = [name for name, value in row.items() if not math.isfinite(float(value))]
    if bad_paths or bad_metrics:
        raise RuntimeError(f"non-finite PPO update {update}: model/optimizer={bad_paths}, metrics={bad_metrics}")


def checkpoint_model_architecture(payload: dict[str, Any]) -> dict[str, Any]:
    """Read architecture before a frozen model is constructed.

    Pre-metadata v3 checkpoints are unambiguously the original shared-trunk
    layout when their state dict has no ``value_net`` keys. New writes always
    carry the explicit form below.
    """
    architecture = payload.get("model_architecture")
    if architecture is None:
        state = payload.get("model")
        if not isinstance(state, dict) or any(name.startswith("value_net.") for name in state):
            raise ValueError("legacy checkpoint has no reconstructible shared architecture")
        try:
            return {"id": "manorl.autonomy.actor_critic.v2",
                    "policy_trunk": [int(state["net.0.weight"].shape[1]), int(state["net.0.weight"].shape[0]), int(state["net.2.weight"].shape[0])],
                    "value_trunk": "shared", "action_dim": int(state["mean.weight"].shape[0])}
        except (KeyError, IndexError, AttributeError) as error:
            raise ValueError("legacy checkpoint has no reconstructible shared architecture") from error
    if not isinstance(architecture, dict):
        raise ValueError("checkpoint model architecture metadata is malformed")
    required = {"id", "policy_trunk", "value_trunk", "action_dim"}
    if set(architecture) != required:
        raise ValueError("checkpoint model architecture metadata is malformed")
    return architecture


def _load_weights_only_init(model: torch.nn.Module, state: dict[str, Any], source_architecture: dict[str, Any] | None) -> str:
    """Load old shared BC weights and explicitly copy its trunk into value_net."""
    if not hasattr(model, "checkpoint_architecture"):
        raise TypeError("weights-only initialization target must declare its architecture")
    target_architecture = model.checkpoint_architecture()
    if source_architecture is not None and source_architecture == target_architecture:
        model.load_state_dict(state, strict=True)
        return "architecture-matched weights-only initialization"
    if not getattr(model, "separate_critic", False):
        raise ValueError("weights-only architecture conversion only supports shared BC into a separate critic")
    expected_source = {**target_architecture, "value_trunk": "shared"}
    if source_architecture is not None and source_architecture != expected_source:
        raise ValueError("incompatible source/target model architecture for weights-only initialization")
    result = model.load_state_dict(state, strict=False)
    expected_missing = {name for name in model.state_dict() if name.startswith("value_net.")}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError("legacy BC weights must differ only by missing value_net parameters")
    model.initialize_value_net_from_actor()
    return "shared-BC-to-separate-critic value_net copied from loaded actor trunk"


def _resume_config_compatible(source: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject changes that alter the PPO/environment trajectory contract."""
    critical = ("num_envs", "rollouts", "learning_epochs", "mini_batches", "seed", "setting", "learning_rate")
    for key in critical:
        if key in source and key in current and source[key] != current[key]:
            raise ValueError(f"resume configuration mismatch for {key}")


def load_optimizer_resume_v3(path: str | Path, model: torch.nn.Module, agent: Any, *,
                             map_location: str | torch.device = "cpu",
                             expected_provenance: dict[str, Any], current_config: dict[str, Any],
                             rollouts: int) -> dict[str, Any]:
    """Restore a v3.1 shared-model PPO checkpoint at an update boundary.

    This is deliberately unlike weights-only initialization: Adam moments and
    CPU RNG streams are restored, while the physical runtime starts a new
    full-start episode because MJX state is not checkpointed.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    validate_v3_checkpoint_metadata(payload)
    if payload.get("action_contract") != ACTION_V3_CONTRACT_ID:
        raise ValueError("incompatible v3 autonomy action contract")
    actual = payload.get("provenance", {})
    for key, expected in expected_provenance.items():
        if actual.get(key) != expected:
            raise ValueError(f"checkpoint provenance mismatch for {key}")
    if checkpoint_model_architecture(payload) != model.checkpoint_architecture():
        raise ValueError("checkpoint/model architecture mismatch")
    if model.checkpoint_architecture()["value_trunk"] != "shared":
        raise ValueError("optimizer resume supports the fixed shared critic only")
    _resume_config_compatible(payload.get("config", {}), current_config)
    policy_steps = payload.get("policy_steps")
    transitions = payload.get("environment_transitions")
    if not isinstance(policy_steps, int) or policy_steps < 0 or policy_steps % rollouts:
        raise ValueError("resume checkpoint policy_steps must be divisible by rollouts")
    if not isinstance(transitions, int) or transitions != policy_steps * current_config["num_envs"]:
        raise ValueError("resume checkpoint has inconsistent environment transition counter")
    required = ("model", "optimizer", "torch_rng", "numpy_rng", "python_rng")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"optimizer resume checkpoint missing {', '.join(missing)}")
    model.load_state_dict(payload["model"], strict=True)
    agent.optimizer.load_state_dict(payload["optimizer"])
    # ``map_location=adapter.device`` places every serialized tensor on CUDA,
    # including CPU generator buffers. Torch and CUDA generator setters each
    # require CPU ByteTensors, while model and Adam tensors remain on device.
    torch.set_rng_state(payload["torch_rng"].detach().cpu())
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    cuda_rng = payload.get("cuda_rng")
    if cuda_rng is not None:
        if not torch.cuda.is_available():
            raise ValueError("resume checkpoint has CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_rng])
        payload["cuda_rng_restore"] = "restored"
    else:
        payload["cuda_rng_restore"] = "unavailable in source checkpoint; CUDA continuation is not bit-exact"
    payload["resume_update"] = policy_steps // rollouts
    payload["policy_sampling_restore"] = (
        "legacy clipped-action likelihood metadata absent; model and Adam restored exactly, "
        "future PPO rollouts store raw Normal samples and clip only physical execution"
        if payload.get("policy_sampling_contract") is None
        else "raw Normal sampling contract preserved across optimizer resume"
    )
    return payload


def load_frozen_v3(path: str | Path, model: torch.nn.Module, *, map_location: str | torch.device = "cpu", expected_provenance: dict[str, Any] | None = None, allow_weights_only_init_conversion: bool = False) -> dict[str, Any]:
    """Load a frozen v3 checkpoint only after contract/provenance agreement."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    validate_v3_checkpoint_metadata(payload)
    if payload.get("action_contract") != ACTION_V3_CONTRACT_ID:
        raise ValueError("incompatible v3 autonomy action contract")
    if expected_provenance:
        actual = payload.get("provenance", {})
        for key, expected in expected_provenance.items():
            if actual.get(key) != expected:
                raise ValueError(f"checkpoint provenance mismatch for {key}")
    source_architecture = checkpoint_model_architecture(payload)
    if allow_weights_only_init_conversion:
        lineage = _load_weights_only_init(model, payload["model"], source_architecture)
        payload["weights_only_init_conversion"] = lineage
    else:
        architecture = checkpoint_model_architecture(payload)
        if not hasattr(model, "checkpoint_architecture") or architecture != model.checkpoint_architecture():
            raise ValueError("checkpoint/model architecture mismatch")
        model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return payload


def _sync_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _compact_window(*, reward_sum: torch.Tensor, done_count: torch.Tensor, valid: torch.Tensor,
                    completed_return_sum: torch.Tensor, completed_length_sum: torch.Tensor,
                    completed_count: torch.Tensor, physics_sums: dict[str, torch.Tensor],
                    sample_seconds: float, optimize_seconds: float, window_count: int,
                    environment_transitions: int, policy_steps: int, update: int) -> dict[str, float]:
    """Transfer only update reductions, never per-env contact/info rows."""
    device_summary = {"reward_sum": reward_sum, "done_count": done_count, "valid": valid,
                      "completed_return_sum": completed_return_sum,
                      "completed_length_sum": completed_length_sum,
                      "completed_count": completed_count, **physics_sums}
    host = {name: float(tensor.detach().cpu()) for name, tensor in device_summary.items()}
    result = {
        "update": float(update), "transitions": float(environment_transitions),
        "global_step": float(environment_transitions), "policy_steps": float(policy_steps),
        "window_transitions": float(window_count), "reward_mean": host["reward_sum"] / window_count,
        "terminations": host["done_count"], "valid": host["valid"],
        "performance/sampling_time": sample_seconds, "performance/optimizer_time": optimize_seconds,
        "performance/total_transitions_per_second": window_count / max(sample_seconds + optimize_seconds, 1e-12),
        "performance/sampling_transitions_per_second": window_count / max(sample_seconds, 1e-12),
        "physics/window_object_z_mean": host["object_motion"] / window_count,
        "physics/window_contact_force_mean": host["contact_force"] / window_count,
        "physics/window_path_error_mean": host["path"] / window_count,
    }
    if host["completed_count"]:
        result["episodes/completed_count"] = host["completed_count"]
        result["episodes/return_mean"] = host["completed_return_sum"] / host["completed_count"]
        result["episodes/length_mean"] = host["completed_length_sum"] / host["completed_count"]
    return result


def run_batched_ppo(adapter: Any, *, updates: int, rollouts: int, learning_epochs: int,
                    mini_batches: int, checkpoint: str | Path | None = None,
                    checkpoint_interval: int = 16, config: dict[str, Any] | None = None,
                    provenance: dict[str, Any] | None = None,
                    init_checkpoint: str | Path | None = None,
                    resume_checkpoint: str | Path | None = None,
                    separate_critic: bool = False,
                    on_update: Callable[[dict[str, float]], None] | None = None) -> tuple[torch.nn.Module, Any, list[dict[str, float]]]:
    """Run canonical PPO against an adapter exposing direct device tensors.

    ``step`` returns physical terminal next observations. They are recorded for
    GAE before ``prepare_action`` resets only terminal rows for the next actor
    call. Update callback execution is synchronous, after optimizer/checkpoint
    work for that update and before sampling the next one.
    """
    if updates < 1 or rollouts < 1 or learning_epochs < 1 or mini_batches < 1:
        raise ValueError("updates, rollouts, learning_epochs and mini_batches must be positive")
    if init_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("init_checkpoint and resume_checkpoint are mutually exclusive")
    if rollouts * adapter.num_envs < mini_batches:
        raise ValueError("mini-batches cannot exceed rollout transitions")
    model, agent = build_batched_runtime(adapter, rollouts=rollouts, learning_epochs=learning_epochs,
                                         mini_batches=mini_batches, device=str(adapter.device),
                                         separate_critic=separate_critic)
    # This is a weights-only warm start. The fresh PPO instance owns a new
    # optimizer; explicitly clearing state prevents a future builder change
    # from silently turning --init-checkpoint into an optimizer resume.
    if init_checkpoint is not None:
        # Source revision is lineage rather than physical compatibility: only
        # the pinned assets, package/split/witness, clock, and v3 ABI gate a
        # weights-only initialization. PPO itself is always newly created.
        if provenance is None:
            raise ValueError("warm start requires physical-contract provenance")
        init_path = Path(init_checkpoint)
        init_payload = load_frozen_v3(
            init_path, model, map_location=adapter.device,
            expected_provenance=_warmstart_compatibility_provenance(provenance),
            allow_weights_only_init_conversion=True,
        )
        provenance = {**provenance,
                      "init_source_commit": init_payload.get("provenance", {}).get("source_commit"),
                      "init_checkpoint_sha256": _checkpoint_sha256(init_path),
                      "init_checkpoint_path": str(init_path),
                      "init_model_conversion": init_payload["weights_only_init_conversion"]}
        agent.optimizer.state.clear()
    config, provenance = config or {}, provenance or {}
    start_policy_steps = 0
    if resume_checkpoint is not None:
        if provenance is None:
            raise ValueError("optimizer resume requires physical-contract provenance")
        resume_path = Path(resume_checkpoint)
        if checkpoint is not None and resume_path.resolve() == Path(checkpoint).resolve():
            raise ValueError("resume checkpoint output must differ from its source checkpoint")
        payload = load_optimizer_resume_v3(
            resume_path, model, agent, map_location=adapter.device,
            expected_provenance=_warmstart_compatibility_provenance(provenance),
            current_config=config, rollouts=rollouts,
        )
        start_policy_steps = payload["policy_steps"]
        parent_provenance = payload.get("provenance", {})
        provenance = {**parent_provenance, **provenance,
            "parent_checkpoint_sha256": _checkpoint_sha256(resume_path),
            "parent_checkpoint_path": str(resume_path),
            "parent_source_commit": parent_provenance.get("source_commit"),
            "optimizer_resume": True,
            "resume_boundary": "optimizer/model continuation with full-start env reset",
            "cuda_rng_restore": payload["cuda_rng_restore"],
            "policy_sampling_restore": payload["policy_sampling_restore"],
            "policy_sampling_contract": dict(POLICY_SAMPLING_CONTRACT),
            "additional_budget": {"updates": updates, "policy_steps": updates * rollouts,
                                  "environment_transitions": updates * rollouts * adapter.num_envs},
        }
    agent.enable_training_mode(True, apply_to_models=True)
    observations, _ = adapter.reset()
    device = observations.device
    episode_return = torch.zeros((adapter.num_envs, 1), device=device)
    episode_length = torch.zeros((adapter.num_envs, 1), device=device)
    rows: list[dict[str, float]] = []
    policy_steps = start_policy_steps
    window_count = adapter.num_envs * rollouts
    for additional_update in range(1, updates + 1):
        update = start_policy_steps // rollouts + additional_update
        _sync_torch(device)
        sample_started = time.perf_counter()
        reward_sum = torch.zeros((), device=device); done_count = torch.zeros((), device=device)
        valid_all = torch.ones((), device=device, dtype=torch.bool)
        completed_return_sum = torch.zeros((), device=device); completed_length_sum = torch.zeros((), device=device)
        completed_count = torch.zeros((), device=device)
        physics_sums: dict[str, torch.Tensor] = {"object_motion": torch.zeros((), device=device),
                                                   "contact_force": torch.zeros((), device=device),
                                                   "path": torch.zeros((), device=device)}
        for _ in range(rollouts):
            with torch.no_grad():
                # Store the raw Normal sample with its own Gaussian likelihood.
                # ``adapter.step`` clips only its physical-execution copy.
                actions, _ = agent.act(observations, None, timestep=policy_steps, timesteps=start_policy_steps + updates * rollouts)
            terminal_next, rewards, terminated, info = adapter.step(actions)
            truncated = torch.zeros_like(terminated)
            if rewards.shape != (adapter.num_envs, 1) or terminated.shape != (adapter.num_envs, 1):
                raise RuntimeError("batched runtime must return rewards/dones shaped (B, 1)")
            agent.record_transition(observations=observations, states=None, actions=actions, rewards=rewards,
                                    next_observations=terminal_next, next_states=None, terminated=terminated,
                                    truncated=truncated, infos=info, timestep=policy_steps, timesteps=start_policy_steps + updates * rollouts)
            reward_sum += rewards.sum(); done_count += terminated.sum()
            episode_return += rewards; episode_length += 1
            # Reduce completed episodes before clearing exactly those rows.
            completed_return_sum += torch.where(terminated, episode_return, torch.zeros_like(episode_return)).sum()
            completed_length_sum += torch.where(terminated, episode_length, torch.zeros_like(episode_length)).sum()
            completed_count += terminated.sum()
            episode_return = torch.where(terminated, torch.zeros_like(episode_return), episode_return)
            episode_length = torch.where(terminated, torch.zeros_like(episode_length), episode_length)
            # Runtime validity is globally reduced today; accept either scalar
            # or future row-wise tensors and fail an entire PPO update on false.
            valid_all &= adapter._to_torch(info["valid"]).to(torch.bool).all()
            # These are current-transition device reductions, captured before
            # prepare_action can reset completed rows.
            for name, value in adapter.compact_summary().items():
                physics_sums[name] += value
            observations = adapter.prepare_action()
            policy_steps += 1
        _sync_torch(device)
        sampling_seconds = time.perf_counter() - sample_started
        if not bool(valid_all.detach().cpu()):
            raise RuntimeError("invalid physics/contact reduction in batched PPO update; checkpoint withheld")
        _sync_torch(device)
        optimize_started = time.perf_counter()
        agent.update(timestep=policy_steps, timesteps=start_policy_steps + updates * rollouts)
        _sync_torch(device)
        optimizer_seconds = time.perf_counter() - optimize_started
        environment_transitions = policy_steps * adapter.num_envs
        row = _compact_window(reward_sum=reward_sum, done_count=done_count, valid=valid_all,
                              completed_return_sum=completed_return_sum, completed_length_sum=completed_length_sum,
                              completed_count=completed_count, physics_sums=physics_sums,
                              sample_seconds=sampling_seconds, optimize_seconds=optimizer_seconds,
                              window_count=window_count, environment_transitions=environment_transitions,
                              policy_steps=policy_steps, update=update)
        # Reject non-finite optimizer/model state and native PPO telemetry
        # before callbacks or checkpoint writes can present this update as good.
        row.update(latest_ppo_metrics(agent))
        _raise_if_nonfinite_update(model=model, agent=agent, row=row, checkpoint=checkpoint,
                                   update=update, policy_steps=policy_steps,
                                   environment_transitions=environment_transitions)
        rows.append(row)
        if checkpoint is not None and update % checkpoint_interval == 0:
            _save_checkpoint(checkpoint=checkpoint, model=model, agent=agent, config=config, provenance=provenance,
                             policy_steps=policy_steps, environment_transitions=environment_transitions,
                             update=update, periodic=True)
        if on_update is not None:
            on_update(row)
    if checkpoint is not None:
        _save_checkpoint(checkpoint=checkpoint, model=model, agent=agent, config=config, provenance=provenance,
                         policy_steps=policy_steps, environment_transitions=policy_steps * adapter.num_envs,
                         update=start_policy_steps // rollouts + updates, periodic=False)
    return model, agent, rows
