"""Explicit, same-device skrl PPO loop for the batched DexHand MJX runtime.

The Gym/SequentialTrainer path remains the single-action compatibility path.
This module owns the all-action path, where JAX owns physics and Torch owns PPO.
Only DLPack crosses their production rollout boundary.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class BatchPPOResult:
    updates: int
    transitions: int
    log_path: Path


class _Bridge(Protocol):
    def torch_to_jax(self, value: Any) -> Any: ...
    def jax_to_torch(self, value: Any) -> Any: ...


def _torch_device_ordinal(tensor: Any) -> int | None:
    device = tensor.device
    return device.index if device.type == "cuda" and device.index is not None else (0 if device.type == "cuda" else None)


def _jax_device_ordinal(value: Any) -> int | None:
    device = value.device if not callable(getattr(value, "device", None)) else value.device()
    platform = getattr(device, "platform", None)
    return getattr(device, "id", None) if platform in {"gpu", "cuda"} else None


class DLPackBridge:
    """Strict Torch↔JAX zero-copy bridge.

    CPU is deliberately available for test/smoke loops only. Production requires
    CUDA tensors and JAX GPU buffers on the same logical ordinal.
    """
    def __init__(self, *, production_gpu: bool):
        self.production_gpu = production_gpu

    def _check_torch(self, value: Any) -> None:
        import torch
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not value.is_contiguous():
            raise RuntimeError("DLPack rollout tensors must be contiguous float32 Torch tensors")
        if self.production_gpu and value.device.type != "cuda":
            raise RuntimeError("production batch rollout requires CUDA Torch tensors")

    def _check_jax(self, value: Any) -> None:
        if str(getattr(value, "dtype", "")) != "float32" or not bool(getattr(value, "flags", None) is None or value.flags.c_contiguous):
            # JAX arrays expose no universal contiguity property; DLPack producer
            # layout is nevertheless checked by the receiving Torch tensor below.
            if str(getattr(value, "dtype", "")) != "float32":
                raise RuntimeError("DLPack rollout arrays must be float32 JAX arrays")
        device = value.device if not callable(getattr(value, "device", None)) else value.device()
        if self.production_gpu and getattr(device, "platform", None) not in {"gpu", "cuda"}:
            raise RuntimeError("production batch rollout requires JAX GPU/CUDA arrays")

    def torch_to_jax(self, value: Any) -> Any:
        import jax
        import torch
        self._check_torch(value)
        result = jax.dlpack.from_dlpack(value)
        self._check_jax(result)
        if self.production_gpu and _torch_device_ordinal(value) != _jax_device_ordinal(result):
            raise RuntimeError("Torch and JAX CUDA logical device ordinals differ")
        return result

    def jax_to_torch(self, value: Any) -> Any:
        import torch
        self._check_jax(value)
        result = torch.utils.dlpack.from_dlpack(value)
        self._check_torch(result)
        if self.production_gpu and _torch_device_ordinal(result) != _jax_device_ordinal(value):
            raise RuntimeError("JAX and Torch CUDA logical device ordinals differ")
        return result


def _scalar(value: Any) -> float:
    return float(value.detach().item())


def _agent_parameters(agent: Any):
    """Yield every distinct parameter exposed by skrl's model containers."""
    seen_models: set[int] = set()
    candidates = [getattr(agent, "models", None), getattr(agent, "policy", None), getattr(agent, "value", None)]
    for candidate in candidates:
        values = candidate.values() if isinstance(candidate, dict) else (candidate if isinstance(candidate, (tuple, list)) else (candidate,))
        for model in values:
            if model is None or id(model) in seen_models:
                continue
            parameters = getattr(model, "parameters", None)
            if callable(parameters):
                seen_models.add(id(model))
                yield from parameters()


def _assert_finite_agent_parameters(agent: Any, *, update: int) -> None:
    import torch

    parameters = list(_agent_parameters(agent))
    if not parameters:
        raise RuntimeError("skrl agent exposes no model parameters for finite validation")
    invalid = [index for index, parameter in enumerate(parameters) if not bool(torch.isfinite(parameter).all().item())]
    if invalid:
        raise RuntimeError(f"non-finite agent model parameters after PPO update {update}: indices={invalid}")


def _capacity_guard(telemetry: dict[str, Any]) -> None:
    contact_capacity = telemetry.get("actual_contact_buffer_shape", (0, 0))
    if len(contact_capacity) and contact_capacity[0]:
        if float(telemetry.get("contact_high_water", 0)) / float(contact_capacity[0]) >= 0.8:
            raise RuntimeError("MJX contact buffer reached 80% of actual capacity")
    constraint_capacity = telemetry.get("actual_constraint_buffer_shape", (0, 0))
    if len(constraint_capacity) > 1 and constraint_capacity[1]:
        if float(telemetry.get("constraint_high_water", 0)) / float(constraint_capacity[1]) >= 0.8:
            raise RuntimeError("MJX constraint buffer reached 80% of actual capacity")


def _checkpoint(agent: Any, output_dir: Path, update: int, provenance: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"checkpoint-{update:06d}.pt"
    temporary = target.with_suffix(".tmp")
    save = getattr(agent, "save", None)
    if not callable(save):
        raise RuntimeError("skrl agent does not provide checkpoint save()")
    save(str(temporary))
    os.replace(temporary, target)
    last_temporary = output_dir / ".last.pt.tmp"
    shutil.copyfile(target, last_temporary)
    os.replace(last_temporary, output_dir / "last.pt")
    sidecar = target.with_suffix(".json")
    sidecar.write_text(json.dumps({"update": update, **provenance}, indent=2, sort_keys=True, default=str), encoding="utf-8")


def run_batch_ppo(env: Any, agent: Any, cfg: Any, *, model_config: Any, assignment: list[dict[str, Any]], bridge: _Bridge | None = None) -> BatchPPOResult:
    """Train exactly ``cfg.timesteps`` vector calls, resetting done rows next step.

    Terminal next observations are recorded before reset so skrl's time-limit
    bootstrap sees the terminal state. Host materialization is restricted to one
    update boundary for logging, capacity checks, and checkpointing.
    """
    import torch

    if int(cfg.timesteps) != int(cfg.updates) * int(cfg.rollouts):
        raise ValueError("batch PPO requires timesteps == updates * rollouts")
    bridge = bridge or DLPackBridge(production_gpu=cfg.device == "gpu")
    # The explicit device loop owns initialization. The single-action
    # SequentialTrainer remains responsible for initializing its own agent.
    init = getattr(agent, "init", None)
    if callable(init):
        init()
    enable_training = getattr(agent, "enable_training_mode", None)
    if callable(enable_training):
        enable_training(True)
    initial = env.reset_device()
    observations = bridge.jax_to_torch(initial.observation)
    pending_done = torch.zeros((env.num_envs,), dtype=torch.bool, device=observations.device)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "progress.jsonl"
    started = time.perf_counter()
    reward_sum = torch.zeros((), dtype=torch.float32, device=observations.device)
    done_sum = torch.zeros((), dtype=torch.float32, device=observations.device)
    timeout_sum = torch.zeros((), dtype=torch.float32, device=observations.device)
    finite = torch.ones((), dtype=torch.bool, device=observations.device)
    valid = torch.ones((), dtype=torch.bool, device=observations.device)
    completed_updates = 0

    with log_path.open("w", encoding="utf-8") as logs:
        for timestep in range(int(cfg.timesteps)):
            if timestep:
                # DLPack borrows pending_done storage. Keep that tensor alive
                # and immutable until reset outputs have crossed back to Torch;
                # the final assignment below then installs this step's done mask.
                reset_mask = pending_done.to(dtype=torch.float32)
                reset = env.reset_done_device(bridge.torch_to_jax(reset_mask))
                observations = bridge.jax_to_torch(reset.observation)
            agent.pre_interaction(timestep=timestep, timesteps=int(cfg.timesteps))
            with torch.no_grad():
                actions, _ = agent.act(observations, None, timestep=timestep, timesteps=int(cfg.timesteps))
            # Distribution samples may be strided views; DLPack rollout
            # ownership requires a contiguous action buffer.
            actions = actions.contiguous()
            transition = env.step_device(bridge.torch_to_jax(actions))
            next_observations = bridge.jax_to_torch(transition.observation)
            rewards = bridge.jax_to_torch(transition.reward)
            terminated = bridge.jax_to_torch(transition.terminated.astype("float32")).to(dtype=torch.bool)
            truncated = bridge.jax_to_torch(transition.truncated.astype("float32")).to(dtype=torch.bool)
            transition_valid = bridge.jax_to_torch(transition.valid.astype("float32")).to(dtype=torch.bool)
            done = terminated | truncated
            finite &= torch.isfinite(next_observations).all() & torch.isfinite(rewards).all()
            valid &= transition_valid.all()
            # This must precede the next-iteration reset: terminal observations
            # are the bootstrap observations for time-limit transitions.
            # skrl's vector memory stores scalar transition fields as (N, 1),
            # while the device runtime keeps its JAX scalar fields as (N,).
            rewards_for_agent = rewards.unsqueeze(-1)
            terminated_for_agent = terminated.unsqueeze(-1)
            truncated_for_agent = truncated.unsqueeze(-1)
            agent.record_transition(observations=observations, states=observations, actions=actions, rewards=rewards_for_agent,
                                    next_observations=next_observations, next_states=next_observations,
                                    terminated=terminated_for_agent, truncated=truncated_for_agent,
                                    infos={"time_outs": truncated_for_agent},
                                    timestep=timestep, timesteps=int(cfg.timesteps))
            agent.post_interaction(timestep=timestep + 1, timesteps=int(cfg.timesteps))
            reward_sum += rewards.sum()
            done_sum += done.sum()
            timeout_sum += truncated.sum()
            observations, pending_done = next_observations, done

            if (timestep + 1) % int(cfg.rollouts):
                continue
            completed_updates += 1
            transitions = (timestep + 1) * int(env.num_envs)
            finite_value, valid_value = bool(finite.item()), bool(valid.item())
            if not finite_value or not valid_value:
                raise RuntimeError(f"invalid batch transition at PPO update {completed_updates}: finite={finite_value}, valid={valid_value}")
            _assert_finite_agent_parameters(agent, update=completed_updates)
            telemetry = env.telemetry()
            _capacity_guard(telemetry)
            elapsed = max(time.perf_counter() - started, 1e-9)
            checkpoint_due = completed_updates % int(cfg.checkpoint_interval_updates) == 0 or completed_updates == int(cfg.updates)
            record = {"update": completed_updates, "transitions": transitions,
                      "reward_mean": _scalar(reward_sum / (int(cfg.rollouts) * int(env.num_envs))),
                      "done": _scalar(done_sum), "timeout": _scalar(timeout_sum), "elapsed": elapsed,
                      "steps_per_s": transitions / elapsed, "finite": finite_value, "transition.valid": valid_value,
                      "contact_high_water": int(telemetry.get("contact_high_water", 0)),
                      "constraint_high_water": int(telemetry.get("constraint_high_water", 0)),
                      "capacity": {key: telemetry[key] for key in telemetry if "capacity" in key or "buffer_shape" in key}}
            if completed_updates % int(cfg.log_interval_updates) == 0 or checkpoint_due:
                logs.write(json.dumps(record, sort_keys=True) + "\n")
                logs.flush()
            if checkpoint_due:
                _checkpoint(agent, output, completed_updates, {"progress": record, "config": asdict(cfg),
                            "physics": cfg.physics.record(), "assignment": assignment, "model_config": asdict(model_config)})
            reward_sum.zero_(); done_sum.zero_(); timeout_sum.zero_(); finite.fill_(True); valid.fill_(True)
    return BatchPPOResult(completed_updates, int(cfg.timesteps) * int(env.num_envs), log_path)
