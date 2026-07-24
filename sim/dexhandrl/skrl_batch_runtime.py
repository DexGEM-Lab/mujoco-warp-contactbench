"""Explicit, same-device PPO loop for the device-resident DexHand MJX runtime.

JAX owns simulation and Torch owns PPO. DLPack is the rollout boundary; episode
aggregation remains Torch-device resident until an update logging boundary.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

try:  # Keep lightweight runtime tests importable without the optional skrl extra.
    from skrl.resources.schedulers.torch import KLAdaptiveLR
except ImportError:  # pragma: no cover - production requires skrl
    class KLAdaptiveLR:  # type: ignore[no-redef]
        pass


REWARD_COMPONENTS = (
    "alive", "object_pos_tracking_x", "object_pos_tracking_y", "object_pos_tracking_z",
    "contact_quality", "object_stability_velocity", "object_rot_tracking",
    "fingertip_inside_object_penalty", "action_penalty_position", "action_penalty_joint",
)
GAMES_TO_TRACK = 100


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
    return getattr(device, "id", None) if getattr(device, "platform", None) in {"gpu", "cuda"} else None


class DLPackBridge:
    """Strict zero-copy Torch↔JAX bridge; CPU is test/smoke-only."""
    def __init__(self, *, production_gpu: bool): self.production_gpu = production_gpu

    def _check_torch(self, value: Any) -> None:
        import torch
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not value.is_contiguous():
            raise RuntimeError("DLPack rollout tensors must be contiguous float32 Torch tensors")
        if self.production_gpu and value.device.type != "cuda":
            raise RuntimeError("production batch rollout requires CUDA Torch tensors")

    def _check_jax(self, value: Any) -> None:
        if str(getattr(value, "dtype", "")) != "float32":
            raise RuntimeError("DLPack rollout arrays must be float32 JAX arrays")
        device = value.device if not callable(getattr(value, "device", None)) else value.device()
        if self.production_gpu and getattr(device, "platform", None) not in {"gpu", "cuda"}:
            raise RuntimeError("production batch rollout requires JAX GPU/CUDA arrays")

    def torch_to_jax(self, value: Any) -> Any:
        import jax
        self._check_torch(value); result = jax.dlpack.from_dlpack(value); self._check_jax(result)
        if self.production_gpu and _torch_device_ordinal(value) != _jax_device_ordinal(result):
            raise RuntimeError("Torch and JAX CUDA logical device ordinals differ")
        return result

    def jax_to_torch(self, value: Any) -> Any:
        import torch
        self._check_jax(value); result = torch.utils.dlpack.from_dlpack(value); self._check_torch(result)
        if self.production_gpu and _torch_device_ordinal(result) != _jax_device_ordinal(value):
            raise RuntimeError("JAX and Torch CUDA logical device ordinals differ")
        return result


def _scalar(value: Any) -> float: return float(value.detach().item())


def _agent_parameters(agent: Any):
    seen: set[int] = set()
    for candidate in (getattr(agent, "models", None), getattr(agent, "policy", None), getattr(agent, "value", None)):
        values = candidate.values() if isinstance(candidate, dict) else (candidate if isinstance(candidate, (tuple, list)) else (candidate,))
        for model in values:
            if model is not None and id(model) not in seen and callable(getattr(model, "parameters", None)):
                seen.add(id(model)); yield from model.parameters()


def _assert_finite_agent_parameters(agent: Any, *, update: int) -> None:
    import torch
    parameters = list(_agent_parameters(agent))
    if not parameters: raise RuntimeError("skrl agent exposes no model parameters for finite validation")
    invalid = [i for i, p in enumerate(parameters) if not bool(torch.isfinite(p).all().item())]
    if invalid: raise RuntimeError(f"non-finite agent model parameters after PPO update {update}: indices={invalid}")


def _capacity_guard(telemetry: dict[str, Any]) -> None:
    contact = telemetry.get("actual_contact_buffer_shape", (0, 0))
    if len(contact) and contact[0] and float(telemetry.get("contact_high_water", 0)) / float(contact[0]) >= .8:
        raise RuntimeError("MJX contact buffer reached 80% of actual capacity")
    ccd = int(telemetry.get("actual_ccd_capacity", 0))
    if ccd and float(telemetry.get("ncollision_high_water", 0)) / ccd >= .8: raise RuntimeError("MJX CCD buffer reached 80% of actual capacity")
    constraint = telemetry.get("actual_constraint_buffer_shape", (0, 0))
    if len(constraint) > 1 and constraint[1] and float(telemetry.get("constraint_high_water", 0)) / float(constraint[1]) >= .8:
        raise RuntimeError("MJX constraint buffer reached 80% of actual capacity")


def _checkpoint(agent: Any, output: Path, update: int, provenance: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True); target = output / f"checkpoint-{update:06d}.pt"; temporary = target.with_suffix(".tmp")
    save = getattr(agent, "save", None)
    if not callable(save): raise RuntimeError("skrl agent does not provide checkpoint save()")
    save(str(temporary)); os.replace(temporary, target)
    last_temporary = output / ".last.pt.tmp"; shutil.copyfile(target, last_temporary); os.replace(last_temporary, output / "last.pt")
    target.with_suffix(".json").write_text(json.dumps({"update": update, **provenance}, indent=2, sort_keys=True, default=str), encoding="utf-8")


class DexHandTelemetryKLAdaptiveLR(KLAdaptiveLR):
    """Unmodified KL scheduler that retains the inputs skrl used per update."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kl_inputs: list[float] = []

    def step(self, kl: Any = None, *, epoch: int | None = None) -> None:
        if kl is not None:
            self.kl_inputs.append(float(kl.detach().item()) if hasattr(kl, "detach") else float(kl))
        super().step(kl, epoch=epoch)

    def consume_kl_inputs(self) -> list[float]:
        values, self.kl_inputs = self.kl_inputs, []
        return values


def _fixed_gaussian_entropy(policy: Any) -> float | None:
    """Entropy of DexHand's fixed 22-D diagonal Gaussian, not its loss term."""
    import math
    log_std = getattr(policy, "log_std", None)
    if log_std is None or not getattr(policy, "cfg", None) or not bool(getattr(policy.cfg, "fixed_sigma", False)):
        return None
    return _scalar((log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))).sum())


def _consume_tracking(agent: Any, cfg: Any) -> dict[str, float]:
    """Translate standard skrl tracking without changing its PPO update."""
    data = getattr(agent, "tracking_data", None)
    if not isinstance(data, dict):
        return {}
    values = {key: float(value[-1]) for key, value in data.items() if value}
    data.clear()
    metrics: dict[str, float] = {}
    if "Loss / Policy loss" in values:
        metrics["losses/a_loss"] = values["Loss / Policy loss"]
    if "Loss / Value loss" in values:
        metrics["losses/c_loss"] = values["Loss / Value loss"] / float(cfg.value_loss_scale)
    metrics["losses/bounds_loss"] = 0.0
    entropy = _fixed_gaussian_entropy(getattr(agent, "policy", None))
    if entropy is not None:
        metrics["losses/entropy"] = entropy
    if "Learning / Learning rate" in values:
        metrics["info/last_lr"] = values["Learning / Learning rate"]
    scheduler = getattr(agent, "scheduler", None)
    consume_kl = getattr(scheduler, "consume_kl_inputs", None)
    if callable(consume_kl):
        kl_inputs = consume_kl()
        if kl_inputs:
            metrics["info/kl"] = sum(kl_inputs) / len(kl_inputs)
    metrics["info/lr_mul"] = 1.0
    metrics["info/e_clip"] = float(cfg.ratio_clip)
    return metrics


class _EpisodeTelemetry:
    """Device accumulation plus source-aligned completed-episode reductions."""

    def __init__(self, worlds: int, action_indices: Any, device: Any):
        import torch
        self.actions = torch.as_tensor(action_indices, dtype=torch.long, device=device)
        self.raw = torch.zeros(worlds, device=device)
        self.shaped = torch.zeros(worlds, device=device)
        self.length = torch.zeros(worlds, device=device)
        self.weighted = torch.zeros((worlds, len(REWARD_COMPONENTS)), device=device)
        self.entries: list[tuple[Any, Any, Any, Any, Any, Any, Any]] = []
        # These intentionally start empty on resume: simulator state is reset.
        self.rewards = deque(maxlen=GAMES_TO_TRACK)
        self.shaped_rewards = deque(maxlen=GAMES_TO_TRACK)
        self.lengths = deque(maxlen=GAMES_TO_TRACK)

    def step(self, raw: Any, shaped: Any, weighted: Any, done: Any, reason: Any) -> None:
        self.raw += raw
        self.shaped += shaped
        self.length += 1
        self.weighted += weighted
        self.entries.append((self.raw.clone(), self.shaped.clone(), self.length.clone(), self.weighted.clone(), reason.clone(), self.actions, done.clone()))
        keep = (~done).to(dtype=self.raw.dtype)
        self.raw *= keep
        self.shaped *= keep
        self.length *= keep
        self.weighted *= keep[:, None]

    def flush(self) -> dict[str, float]:
        import torch
        if not self.entries:
            return {}
        columns = tuple(torch.cat([entry[index] for entry in self.entries]) for index in range(7))
        self.entries.clear()
        raw, shaped, length, weighted, reason, actions, done = columns
        if not bool(done.any()):
            return {}
        raw, shaped, length, weighted, reason, actions = (value[done] for value in (raw, shaped, length, weighted, reason, actions))
        # The only device→host materialization happens at the update boundary.
        self.rewards.extend(raw.detach().cpu().tolist())
        self.shaped_rewards.extend(shaped.detach().cpu().tolist())
        self.lengths.extend(length.detach().cpu().tolist())
        metrics = {
            "rewards": sum(self.rewards) / len(self.rewards),
            "shaped_rewards": sum(self.shaped_rewards) / len(self.shaped_rewards),
            "episode_lengths": sum(self.lengths) / len(self.lengths),
            "terminal_reward_raw_weighted_gap": _scalar((raw - weighted.sum(dim=1)).mean()),
        }

        def add_sample_means(mask: Any, suffix: str) -> None:
            if not bool(mask.any()):
                return
            selected_raw, selected_weighted = raw[mask], weighted[mask]
            if suffix:
                metrics[f"rewards{suffix}"] = _scalar(selected_raw.mean())
            for index, component in enumerate(REWARD_COMPONENTS):
                metrics[f"{component}{suffix}"] = _scalar(selected_weighted[:, index].mean())

        add_sample_means(torch.ones_like(actions, dtype=torch.bool), "")
        add_sample_means(torch.ones_like(actions, dtype=torch.bool), "/object_cube1")
        action_names = ("01", "02", "03", "04", "09", "10", "18")
        for index, action_name in enumerate(action_names):
            add_sample_means(actions == index, f"/cube1_{action_name}")
        return metrics


def _write_schema(output: Path) -> None:
    (output / "metrics_schema.json").write_text(json.dumps({
        "schema": "dexhandrl.gym_tensorboard.v2", "phase_1_full_metrics_available": False,
        "provenance": "standard-skrl PPO with telemetry-only KL observation",
        "standard_skrl_analogs": {"a_loss": "Loss / Policy loss", "c_loss": "Loss / Value loss / value_loss_scale", "bounds_loss": 0, "entropy": "fixed policy.log_std Gaussian summed over 22", "info/kl": "KLAdaptiveLR.step inputs mean"},
        "reward_shaper": 1.0,
        "axes": {"iter": "global update", "step": "global transitions", "time": "cumulative elapsed seconds"},
        "reward_components": list(REWARD_COMPONENTS),
    }, indent=2, sort_keys=True), encoding="utf-8")


def _write_metrics(writer: Any, metrics: dict[str, float], *, update: int, transitions: int, elapsed: float) -> None:
    if writer is None:
        return
    core = ("rewards", "shaped_rewards", "episode_lengths")
    for key in core:
        if key in metrics:
            writer.add_scalar(f"{key}/iter", metrics[key], update)
            writer.add_scalar(f"{key}/step", metrics[key], transitions)
            writer.add_scalar(f"{key}/time", metrics[key], elapsed)
    for tag, value in metrics.items():
        if tag not in core:
            writer.add_scalar(tag, value, transitions if tag.startswith(("performance/", "info/", "losses/")) else update)
    writer.add_scalar("global_step", transitions, transitions)
    writer.flush()


def _resume_checkpoint(cfg: Any) -> None:
    start = int(getattr(cfg, "start_update", 0))
    if not start: return
    checkpoint = Path(cfg.checkpoint)
    sidecar = checkpoint.with_suffix(".json")
    if not sidecar.is_file(): raise ValueError("resume checkpoint sidecar is required")
    if int(json.loads(sidecar.read_text(encoding="utf-8")).get("update", -1)) != start:
        raise ValueError("resume checkpoint sidecar update does not equal start_update")


def run_batch_ppo(env: Any, agent: Any, cfg: Any, *, model_config: Any, assignment: list[dict[str, Any]], bridge: _Bridge | None = None) -> BatchPPOResult:
    """Run global updates ``start_update + 1 .. updates`` with exact global steps."""
    import torch
    if int(cfg.timesteps) != int(cfg.updates) * int(cfg.rollouts): raise ValueError("batch PPO requires timesteps == updates * rollouts")
    _resume_checkpoint(cfg); start_update = int(getattr(cfg, "start_update", 0)); total_updates = int(cfg.updates)
    bridge = bridge or DLPackBridge(production_gpu=cfg.device == "gpu")
    init = getattr(agent, "init", None)
    if callable(init):
        init()
    enable = getattr(agent, "enable_training_mode", None)
    if callable(enable): enable(True)
    initial = env.reset_device(); observations = bridge.jax_to_torch(initial.observation)
    pending = torch.zeros(env.num_envs, dtype=torch.bool, device=observations.device)
    action_indices = getattr(getattr(env, "batch", None), "action_indices", [0] * env.num_envs)
    episodes = _EpisodeTelemetry(env.num_envs, action_indices, observations.device)
    output = Path(cfg.output_dir); output.mkdir(parents=True, exist_ok=True); _write_schema(output)
    writer = None
    logdir = getattr(cfg, "tensorboard_logdir", None)
    if logdir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(logdir))
    log_path = output / "progress.jsonl"; mode = "a" if start_update else "w"; started = time.perf_counter()
    with log_path.open(mode, encoding="utf-8") as logs:
        for update in range(start_update + 1, total_updates + 1):
            step_time = 0.0
            play_start = time.perf_counter(); reward_sum = torch.zeros((), device=observations.device); done_sum = torch.zeros((), device=observations.device); timeout_sum = torch.zeros((), device=observations.device); finite = valid = True
            for local in range(int(cfg.rollouts)):
                global_timestep = ((update - 1) * int(cfg.rollouts)) + local
                if global_timestep:
                    reset = env.reset_done_device(bridge.torch_to_jax(pending.to(dtype=torch.float32)))
                    observations = bridge.jax_to_torch(reset.observation)
                agent.pre_interaction(timestep=global_timestep, timesteps=int(cfg.timesteps))
                with torch.no_grad(): actions, _ = agent.act(observations, None, timestep=global_timestep, timesteps=int(cfg.timesteps))
                step_started = time.perf_counter()
                transition = env.step_device(bridge.torch_to_jax(actions.contiguous()))
                next_obs = bridge.jax_to_torch(transition.observation)
                raw = bridge.jax_to_torch(transition.reward)
                rewards = raw.clone()
                terminated = bridge.jax_to_torch(transition.terminated.astype("float32")).bool()
                truncated = bridge.jax_to_torch(transition.truncated.astype("float32")).bool()
                done = terminated | truncated
                weighted_value = getattr(transition, "weighted", None)
                reason_value = getattr(transition, "reason", None)
                weighted = bridge.jax_to_torch(weighted_value) if weighted_value is not None else torch.zeros((env.num_envs, len(REWARD_COMPONENTS)), device=observations.device)
                reason = bridge.jax_to_torch(reason_value.astype("float32")).long() if reason_value is not None else torch.where(terminated, 1, torch.where(truncated, 2, 0))
                transition_valid = bridge.jax_to_torch(transition.valid.astype("float32")).bool()
                step_time += time.perf_counter() - step_started
                finite = finite & torch.isfinite(next_obs).all() & torch.isfinite(raw).all()
                valid = valid & transition_valid.all()
                agent.record_transition(observations=observations, states=observations, actions=actions, rewards=rewards.unsqueeze(-1), next_observations=next_obs, next_states=next_obs, terminated=terminated.unsqueeze(-1), truncated=truncated.unsqueeze(-1), infos={"time_outs": truncated.unsqueeze(-1)}, timestep=global_timestep, timesteps=int(cfg.timesteps))
                episodes.step(raw, rewards.squeeze(-1), weighted, done, reason)
                if local + 1 < int(cfg.rollouts):
                    agent.post_interaction(timestep=global_timestep + 1, timesteps=int(cfg.timesteps))
                reward_sum += raw.sum()
                done_sum += done.sum()
                timeout_sum += truncated.sum()
                observations, pending = next_obs, done
            play_time = time.perf_counter() - play_start
            update_start = time.perf_counter()
            agent.post_interaction(timestep=global_timestep + 1, timesteps=int(cfg.timesteps))
            finite_value, valid_value = bool(finite.item()), bool(valid.item())
            if not finite_value or not valid_value:
                raise RuntimeError(f"invalid batch transition at PPO update {update}: finite={finite_value}, valid={valid_value}")
            _assert_finite_agent_parameters(agent, update=update)
            telemetry = env.telemetry()
            _capacity_guard(telemetry)
            update_time = time.perf_counter() - update_start
            transitions = update * int(cfg.rollouts) * int(env.num_envs)
            elapsed = time.perf_counter() - started
            metrics = episodes.flush()
            metrics.update(_consume_tracking(agent, cfg))
            update_transitions = int(cfg.rollouts) * int(env.num_envs)
            metrics.update({"info/epochs": float(update), "health/contact_high_water": float(telemetry.get("contact_high_water", 0)), "health/ncollision_high_water": float(telemetry.get("ncollision_high_water", 0)), "health/constraint_high_water": float(telemetry.get("constraint_high_water", 0)), "performance/step_time": step_time, "performance/step_inference_time": play_time, "performance/rl_update_time": update_time, "performance/step_fps": update_transitions / max(step_time, 1e-9), "performance/step_inference_fps": update_transitions / max(play_time, 1e-9), "performance/step_inference_rl_update_fps": update_transitions / max(play_time + update_time, 1e-9)})
            record = {"update": update, "transitions": transitions, "global_step": transitions, "reward_mean": _scalar(reward_sum / (cfg.rollouts * env.num_envs)), "done": _scalar(done_sum), "timeout": _scalar(timeout_sum), "elapsed": elapsed, "finite": finite_value, "transition.valid": valid_value, "contact_high_water": int(telemetry.get("contact_high_water", 0)), "ncollision_high_water": int(telemetry.get("ncollision_high_water", 0)), "constraint_high_water": int(telemetry.get("constraint_high_water", 0)), "capacity": {key: telemetry[key] for key in telemetry if "capacity" in key or "buffer_shape" in key}, "metrics": metrics}
            checkpoint_due = update == start_update + 1 or update % int(cfg.checkpoint_interval_updates) == 0 or update == total_updates
            if update % int(cfg.log_interval_updates) == 0 or checkpoint_due: logs.write(json.dumps(record, sort_keys=True) + "\n"); logs.flush()
            _write_metrics(writer, metrics, update=update, transitions=transitions, elapsed=elapsed)
            if checkpoint_due: _checkpoint(agent, output, update, {"progress": record, "config": asdict(cfg), "physics": cfg.physics.record(), "assignment": assignment, "model_config": asdict(model_config), "resume_environment_reset": bool(start_update)})
    if writer is not None: writer.close()
    return BatchPPOResult(total_updates - start_update, (total_updates - start_update) * int(cfg.rollouts) * int(env.num_envs), log_path)
