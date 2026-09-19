"""v4 single-reference PPO loop.

This reuses the mature RlGamesPPO/GAE implementation. PPO memory owns raw 957-D
observations and raw Normal actions; the adapter clips only the physical action.
Terminal next observations are recorded before ``prepare_action`` resets exactly
``runtime.last_done`` rows.
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
    ACTION_CONTRACT_ID, CHECKPOINT_FORMAT, OBSERVATION_CONTRACT_ID,
    POLICY_SAMPLING_CONTRACT, REWARD_CONTRACT_ID, validate_v4_checkpoint_metadata,
)
from sim.manorl.autonomy_telemetry import latest_ppo_metrics
from sim.manorl.autonomy_v4_telemetry import V4TelemetryAccumulator
from sim.manorl.autonomy_training import build_batched_runtime, resolved_v4_ppo_config, validate_learning_rate, teacher_anchor_metadata


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
    contracts = (
        model.checkpoint_contracts()
        if hasattr(model, "checkpoint_contracts")
        else {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "observation_contract": OBSERVATION_CONTRACT_ID,
            "reward_contract": REWARD_CONTRACT_ID,
            "action_contract": ACTION_CONTRACT_ID,
        }
    )
    return {
        **contracts,
        "policy_sampling_contract": dict(POLICY_SAMPLING_CONTRACT),
        "model": model.state_dict(), "model_architecture": model.checkpoint_architecture(),
        "optimizer": agent.optimizer.state_dict(), "normalizer": None,
        "policy_steps": int(policy_steps), "environment_transitions": int(environment_transitions),
        "update": int(policy_steps) // config["rollouts"],
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


def _validate_checkpoint_contracts(
    payload: dict[str, Any], expected: dict[str, str]
) -> None:
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"incompatible checkpoint: {name} must be {value!r}"
            )
    if payload.get("policy_sampling_contract") not in (
        None,
        POLICY_SAMPLING_CONTRACT,
    ):
        raise ValueError("incompatible policy sampling contract")


def inspect_v4_warmstart(path: str | Path, target_architecture: dict[str, Any], *,
                         map_location: str | torch.device = "cpu",
                         expected_provenance: dict[str, Any] | None = None,
                         target_contracts: dict[str, str] | None = None) -> tuple[dict[str, Any], str]:
    """Validate public metadata/provenance and resolve the only supported transfer mode."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if target_contracts is None:
        validate_v4_checkpoint_metadata(payload)
    else:
        _validate_checkpoint_contracts(payload, target_contracts)
    mode = _warmstart_transfer_mode(payload.get("model_architecture"), target_architecture)
    _validate_provenance(payload.get("provenance", {}), expected_provenance or {})
    return payload, mode


def _load_v4_model_state(path: str | Path, model: torch.nn.Module, *,
                         map_location: str | torch.device,
                         expected_provenance: dict[str, Any] | None,
                         allow_shared_policy_transfer: bool = False) -> tuple[dict[str, Any], str]:
    payload, mode = inspect_v4_warmstart(path, model.checkpoint_architecture(),
                                         map_location=map_location,
                                         expected_provenance=expected_provenance,
                                         target_contracts=model.checkpoint_contracts())
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


# Only orchestration/budget and lineage may change on optimizer continuation.
_RESUME_MUTABLE_CONFIG = {
    "updates", "total_transitions", "checkpoint", "checkpoint_interval",
    "wandb", "wandb_project", "wandb_entity", "wandb_mode", "wandb_run_id",
    "source_commit", "warmstart", "warmstart_checkpoint", "warmstart_transfer_mode",
    "mode", "resume_checkpoint", "resume",
}
_RESUME_DIAGNOSTIC_PROVENANCE = {
    "source_commit", "cache_hash_recorded_not_compared", "separate_critic",
    "warmstart_checkpoint", "warmstart_transfer_mode", "mode", "resume_checkpoint", "resume",
}
_RESUME_REQUIRED_PROVENANCE = {
    "asset_pin", "package_digest", "manifest_sha256", "catalog_digest",
    "identity_split", "contracts", "identity", "clock",
}


def inspect_v4_resume(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, *,
                      expected_config: dict[str, Any], expected_provenance: dict[str, Any],
                      updates: int, cuda_device_count: int | None = None) -> dict[str, Any]:
    """Validate on CPU without changing the model, optimizer or global RNG.

    CUDA count is the number of *logical visible* devices at the training target.
    None permits CPU-only artifact inspection; a CUDA runtime must pass its count.
    Checkpoints are trusted local Torch artifacts, not untrusted pickle inputs.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_contracts(payload, model.checkpoint_contracts())
    if payload.get("policy_sampling_contract") != POLICY_SAMPLING_CONTRACT:
        raise ValueError("resume requires the raw Normal sampling contract")
    if payload.get("model_architecture") != model.checkpoint_architecture():
        raise ValueError("resume checkpoint/model architecture mismatch")
    if "normalizer" not in payload or payload["normalizer"] is not None:
        raise ValueError("resume supports only normalizer=null")
    source_config = payload.get("config", {})
    required_config = {"num_envs", "rollouts", "learning_epochs", "mini_batches", "learning_rate", "separate_critic", "seed", "device"}
    if not required_config <= source_config.keys() or not required_config <= expected_config.keys():
        raise ValueError("resume config is missing fixed training fields")
    # Absent anchor metadata in pre-feature checkpoints means the disabled path.
    def with_anchor_defaults(config):
        config = dict(config)
        config.setdefault("all_train_references", False)
        config.setdefault("all_references", False)
        config.setdefault("teacher_anchor_beta", 0.0)
        config.setdefault("teacher_anchor_passes", 2)
        config.setdefault("teacher_anchor", teacher_anchor_metadata())
        if config["teacher_anchor_beta"] == 0:
            config["teacher_anchor"] = teacher_anchor_metadata(0., config["teacher_anchor_passes"])
        return config
    source_config = with_anchor_defaults(source_config)
    expected_config = with_anchor_defaults(expected_config)
    fixed = lambda config: {k: v for k, v in config.items() if k not in _RESUME_MUTABLE_CONFIG}
    if fixed(source_config) != fixed(expected_config):
        changed = sorted(k for k in fixed(source_config).keys() | fixed(expected_config).keys()
                         if fixed(source_config).get(k) != fixed(expected_config).get(k))
        raise ValueError(f"resume config mismatch: {changed}")
    source_provenance = payload.get("provenance", {})
    if not _RESUME_REQUIRED_PROVENANCE <= source_provenance.keys() or not _RESUME_REQUIRED_PROVENANCE <= expected_provenance.keys():
        raise ValueError("resume provenance is missing physical/package fields")
    source_provenance = {"teacher_anchor": teacher_anchor_metadata(), **source_provenance}
    expected_provenance = {"teacher_anchor": teacher_anchor_metadata(), **expected_provenance}
    # A disabled anchor has no executed supervision recipe to preserve. Enabled
    # frame-gated checkpoints must reject intent-gated optimizer continuation.
    for provenance in (source_provenance, expected_provenance):
        if provenance["teacher_anchor"].get("beta") == 0:
            provenance["teacher_anchor"] = teacher_anchor_metadata(0., provenance["teacher_anchor"].get("passes", 2))
    physical = lambda p: {k: v for k, v in p.items() if k not in _RESUME_DIAGNOSTIC_PROVENANCE}
    if physical(source_provenance) != physical(expected_provenance):
        raise ValueError("resume physical/package provenance mismatch")
    policy_steps = payload.get("policy_steps")
    rollouts, num_envs = source_config["rollouts"], source_config["num_envs"]
    if type(rollouts) is not int or rollouts < 1 or type(num_envs) is not int or num_envs < 1:
        raise ValueError("resume rollout/num-envs must be positive integers")
    if type(policy_steps) is not int or policy_steps <= 0 or policy_steps % rollouts:
        raise ValueError("resume policy_steps must be a positive integer multiple of rollouts")
    transitions = payload.get("environment_transitions")
    if type(transitions) is not int or transitions != policy_steps * num_envs:
        raise ValueError("resume environment_transitions mismatch")
    if type(updates) is not int or updates <= policy_steps // rollouts:
        raise ValueError("resume total updates must be greater than completed updates")
    if expected_config.get("total_transitions") not in (None, updates * rollouts * num_envs):
        raise ValueError("resume total-transitions mismatch")
    state = payload.get("model")
    target = model.state_dict()
    if not isinstance(state, dict) or state.keys() != target.keys():
        raise ValueError("resume model state keys mismatch")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != target[name].shape or value.dtype != target[name].dtype:
            raise ValueError(f"resume model tensor mismatch: {name}")
    saved_optimizer = payload.get("optimizer")
    if not isinstance(saved_optimizer, dict) or not isinstance(saved_optimizer.get("state"), dict):
        raise ValueError("resume requires a complete PPO Adam optimizer")
    groups = saved_optimizer.get("param_groups", [])
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("resume optimizer group topology mismatch")
    ids = []
    for saved, live in zip(groups, optimizer.param_groups):
        if len(saved.get("params", [])) != len(live["params"]):
            raise ValueError("resume optimizer parameter count mismatch (partial teacher optimizer)")
        if {k: v for k, v in saved.items() if k != "params"} != {k: v for k, v in live.items() if k != "params"}:
            raise ValueError("resume optimizer hyperparameters/LR mismatch")
        ids.extend(saved["params"])
        for param_id, param in zip(saved["params"], live["params"]):
            adam = saved_optimizer["state"].get(param_id, {})
            required = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if saved["amsgrad"] else set())
            if adam.keys() != required:
                raise ValueError("resume requires complete Adam state for every PPO parameter")
            for key, tensor in adam.items():
                shape = torch.Size([]) if key == "step" else param.shape
                if not isinstance(tensor, torch.Tensor) or tensor.shape != shape:
                    raise ValueError(f"resume optimizer tensor shape mismatch: {param_id}.{key}")
                if key != "step" and tensor.dtype != param.dtype:
                    raise ValueError("resume optimizer tensor dtype mismatch")
            if adam["step"].item() <= 0 or adam["step"].item() % 1:
                raise ValueError("resume Adam step must be a positive integer")
    if len(set(ids)) != len(ids) or set(ids) != saved_optimizer["state"].keys():
        raise ValueError("resume optimizer state topology mismatch")
    if {id(p) for g in optimizer.param_groups for p in g["params"]} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("resume target optimizer must own every trainable model parameter")
    if _finite_paths(state, "model") or _finite_paths(saved_optimizer, "optimizer"):
        raise ValueError("resume non-finite model/optimizer state")
    try:
        torch.Generator(device="cpu").set_state(payload["torch_rng"])
        np.random.RandomState().set_state(payload["numpy_rng"])
        random.Random().setstate(payload["python_rng"])
        cuda_rng = payload["cuda_rng"]
        if cuda_rng is not None:
            if not isinstance(cuda_rng, list): raise ValueError("CUDA RNG must be a list")
            for tensor in cuda_rng:
                if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8 or tensor.ndim != 1 or tensor.numel() == 0:
                    raise ValueError("invalid CUDA RNG byte tensor")
        if cuda_device_count is not None:
            if cuda_device_count < 1 or not isinstance(cuda_rng, list) or len(cuda_rng) != cuda_device_count:
                raise ValueError("CUDA RNG logical device count mismatch")
            if torch.cuda.is_available():
                for index, tensor in enumerate(cuda_rng):
                    torch.Generator(device=f"cuda:{index}").set_state(tensor)
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"resume RNG state invalid: {error}") from error
    return payload


def resume_lineage(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"resume_checkpoint": str(Path(path).expanduser().resolve()), "mode": "ppo_resume",
            "resume": {"sha256": digest, "source_commit": payload["provenance"].get("source_commit"),
                       "previous_policy_steps": payload["policy_steps"],
                       "previous_environment_transitions": payload["environment_transitions"],
                       "previous_updates": payload["policy_steps"] // payload["config"]["rollouts"],
                       "physics_restart": "full_start_new_episodes",
                       "source_config": payload["config"], "source_provenance": payload["provenance"]}}


def restore_v4_rng(payload: dict[str, Any], *, device: torch.device) -> None:
    # Called only after model/Adam construction and fresh physical reset.
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(payload["cuda_rng"])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _apply_teacher_anchor(model, optimizer, pairs, *, beta: float, passes: int, mini_batches: int):
    """Separate mean-only Adam steps after PPO, covering every collected sample."""
    observations = torch.cat([observation for observation, _ in pairs])
    targets = torch.cat([teacher for _, teacher in pairs])
    count = observations.shape[0]
    mse_sum = torch.zeros((), device=observations.device)
    for _ in range(passes):
        indices = torch.randperm(count, device=observations.device)
        for batch in torch.tensor_split(indices, mini_batches):
            mean, _ = model.compute({"observations": observations[batch]}, role="policy")
            mse = torch.nn.functional.mse_loss(mean, targets[batch])
            # None gradients are essential: Adam must not move value-only/log_std
            # parameters using momentum left over from the preceding PPO update.
            optimizer.zero_grad(set_to_none=True)
            (beta * mse).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            mse_sum += mse.detach() * batch.numel()
    mse = float((mse_sum / (count * passes)).cpu())
    return {"teacher_anchor/mse": mse, "teacher_anchor/loss": beta * mse,
            "teacher_anchor/samples": float(count), "teacher_anchor/rollout_steps": float(len(pairs)),
            "teacher_anchor/optimizer_steps": float(passes * mini_batches)}


def run_batched_ppo(adapter: Any, *, updates: int, rollouts: int, learning_epochs: int,
                    mini_batches: int, learning_rate: float = 3e-4, checkpoint: str | Path | None = None,
                    checkpoint_interval: int = 16, config: dict[str, Any] | None = None,
                    provenance: dict[str, Any] | None = None, separate_critic: bool = False,
                    teacher_anchor_beta: float = 0.0, teacher_anchor_passes: int = 2,
                    warmstart: str | Path | None = None,
                    resume_checkpoint: str | Path | None = None,
                    expected_warmstart_provenance: dict[str, Any] | None = None,
                    on_update: Callable[[dict[str, float]], None] | None = None) -> tuple[torch.nn.Module, Any, list[dict[str, float]]]:
    """Execute canonical PPO with finite-horizon terminal bootstrapping.

    ``terminated`` is the environment's physical/reward boundary. Rollout cuts
    retain it as false, so RlGamesPPO bootstraps from the recorded terminal-next
    observation; true terminations do not bootstrap and reset afterward.
    """
    validate_learning_rate(learning_rate)
    anchor = teacher_anchor_metadata(teacher_anchor_beta, teacher_anchor_passes)
    if warmstart is not None and resume_checkpoint is not None:
        raise ValueError("warmstart and resume-checkpoint are mutually exclusive")
    if min(updates, rollouts, learning_epochs, mini_batches) < 1:
        raise ValueError("updates, rollouts, learning_epochs and mini_batches must be positive")
    if rollouts * adapter.num_envs < mini_batches:
        raise ValueError("mini-batches cannot exceed rollout transitions")
    model, agent = build_batched_runtime(adapter, rollouts=rollouts, learning_epochs=learning_epochs,
                                         mini_batches=mini_batches, device=str(adapter.device), separate_critic=separate_critic, learning_rate=learning_rate)
    config, provenance = dict(config or {}), dict(provenance or {})
    actual_device = "gpu" if adapter.device.type == "cuda" else "cpu"
    if resume_checkpoint is not None and config.get("device", actual_device) != actual_device:
        raise ValueError("resume runtime device differs from fixed config; use inspect_v4_resume for CPU diagnostics")
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
    config.update({"updates": updates, "rollouts": rollouts, "num_envs": adapter.num_envs,
                   "learning_epochs": learning_epochs, "mini_batches": mini_batches,
                   "learning_rate": learning_rate, "separate_critic": bool(separate_critic)})
    config.update({"teacher_anchor_beta": teacher_anchor_beta, "teacher_anchor_passes": teacher_anchor_passes,
                   "teacher_anchor": anchor})
    provenance["teacher_anchor"] = anchor
    config.setdefault("seed", 0)
    config.setdefault("device", "gpu" if adapter.device.type == "cuda" else "cpu")
    resumed = None
    if resume_checkpoint is not None:
        resumed = inspect_v4_resume(resume_checkpoint, model, agent.optimizer,
            expected_config=config, expected_provenance=provenance, updates=updates,
            cuda_device_count=torch.cuda.device_count() if adapter.device.type == "cuda" else None)
        model.load_state_dict(resumed["model"], strict=True)
        agent.optimizer.load_state_dict(resumed["optimizer"])
        lineage = {key: resumed["provenance"][key] for key in ("warmstart_checkpoint", "warmstart_transfer_mode")
                   if key in resumed["provenance"]}
        lineage.update(resume_lineage(resume_checkpoint, resumed))
    config.update(lineage); provenance.update(lineage)
    config["learning_rate"] = agent.optimizer.param_groups[0]["lr"]
    agent.enable_training_mode(True, apply_to_models=True)
    observations, _ = adapter.reset()
    episode_return = torch.zeros((adapter.num_envs, 1), device=adapter.device)
    episode_length = torch.zeros_like(episode_return)
    rows: list[dict[str, float]] = []
    policy_steps = resumed["policy_steps"] if resumed is not None else 0
    telemetry = V4TelemetryAccumulator(adapter.num_envs, adapter.device) if hasattr(adapter, "telemetry_snapshot") else None
    if resumed is not None:
        restore_v4_rng(resumed, device=adapter.device)
    for update in range(policy_steps // rollouts + 1, updates + 1):
        _sync(adapter.device); sampled = time.perf_counter()
        reward_sum = torch.zeros((), device=adapter.device); done_count = torch.zeros((), device=adapter.device)
        valid = torch.ones((), dtype=torch.bool, device=adapter.device)
        completed_return = torch.zeros((), device=adapter.device); completed_length = torch.zeros((), device=adapter.device)
        completed_count = torch.zeros((), device=adapter.device)
        summaries = {name: torch.zeros((), device=adapter.device) for name in ("object_motion", "contact_force", "path")}
        teacher_pairs = [] if teacher_anchor_beta > 0 else None
        for _ in range(rollouts):
            with torch.no_grad():
                if teacher_pairs is not None:
                    teacher_pairs.append((observations.detach().clone(), adapter.teacher_actions().detach().clone()))
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
        optimized = time.perf_counter(); agent.update(timestep=policy_steps, timesteps=updates * rollouts)
        anchor_metrics = {}
        if teacher_pairs is not None:
            anchor_metrics = _apply_teacher_anchor(model, agent.optimizer, teacher_pairs,
                beta=teacher_anchor_beta, passes=teacher_anchor_passes, mini_batches=mini_batches)
            del teacher_pairs
        _sync(adapter.device)
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
        # This derives the effective values from the instantiated PPO/Adam,
        # rather than restating library defaults in telemetry configuration.
        row.update({f"config/{name}": value for name, value in resolved_v4_ppo_config(agent).items()
                    if isinstance(value, (bool, float, int))})
        row.update(anchor_metrics)
        row.update({"config/teacher_anchor_beta": teacher_anchor_beta,
                    "config/teacher_anchor_passes": teacher_anchor_passes,
                    "config/teacher_squeeze_rad": anchor["squeeze_rad"],
                    "config/teacher_contact_intent_threshold": anchor["contact_intent_threshold"]})
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
