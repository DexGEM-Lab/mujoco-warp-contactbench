"""Explicit, same-device PPO loop for the device-resident DexHand MJX runtime.

JAX owns simulation and Torch owns PPO. DLPack is the rollout boundary; episode
aggregation remains Torch-device resident until an update logging boundary.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


REWARD_COMPONENTS = (
    "alive", "x", "y", "z", "contact", "stability", "rotation", "fingertip",
    "action_position", "action_joint",
)


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


def _consume_tracking(agent: Any) -> dict[str, float]:
    """Consume skrl tracking exactly once per update; stale values cannot leak."""
    data = getattr(agent, "tracking_data", None)
    if not isinstance(data, dict): return {}
    values = {key: float(value[-1]) for key, value in data.items() if value}
    data.clear()
    aliases = {
        "losses/a_loss": "Loss / Policy loss", "losses/c_loss": "Gym / Critic loss",
        "losses/bounds_loss": "Loss / Bounds loss", "losses/entropy": "Gym / Raw entropy",
        "info/last_lr": "Learning / Learning rate", "info/kl": "Learning / Exact KL mean",
    }
    return {tag: values[source] for tag, source in aliases.items() if source in values}


class _EpisodeTelemetry:
    """Per-world raw/shaped/component totals and update-window group reductions."""
    def __init__(self, worlds: int, action_indices: Any, device: Any):
        import torch
        self.worlds, self.device = worlds, device
        self.actions = torch.as_tensor(action_indices, dtype=torch.long, device=device)
        self.raw = torch.zeros(worlds, device=device); self.shaped = torch.zeros(worlds, device=device)
        self.length = torch.zeros(worlds, device=device); self.weighted = torch.zeros((worlds, 10), device=device)
        self.entries: list[tuple[Any, Any, Any, Any, Any, Any, Any]] = []

    def step(self, raw: Any, shaped: Any, weighted: Any, done: Any, reason: Any) -> None:
        self.raw += raw; self.shaped += shaped; self.length += 1; self.weighted += weighted
        # Retain every row on device; selecting completed rows only at the
        # update boundary avoids a per-step GPU→host synchronization.
        self.entries.append((self.raw.clone(), self.shaped.clone(), self.length.clone(), self.weighted.clone(), reason.clone(), self.actions.clone(), done.clone()))
        keep = (~done).to(dtype=self.raw.dtype)
        self.raw *= keep; self.shaped *= keep; self.length *= keep; self.weighted *= keep[:, None]

    def flush(self) -> dict[str, float]:
        import torch
        if not self.entries: return {}
        raw, shaped, length, weighted, reason, actions, done = (torch.cat([entry[i] for entry in self.entries]) for i in range(7))
        self.entries.clear()
        if not bool(done.any()): return {}
        raw, shaped, length, weighted, reason, actions = (value[done] for value in (raw, shaped, length, weighted, reason, actions))
        result: dict[str, float] = {}
        def groups(mask: Any, suffix: str) -> None:
            if not bool(mask.any()): return
            count = mask.sum().float(); lengths = length[mask]
            # Episode totals and PerGroup's episode-total/length instant values.
            for index, component in enumerate(REWARD_COMPONENTS):
                totals = weighted[mask, index]
                result[f"{component}{suffix}"] = _scalar(totals.mean())
                result[f"{component}_instant{suffix}"] = _scalar((totals / lengths).mean())
            result[f"rewards{suffix}"] = _scalar(raw[mask].mean())
            result[f"shaped_rewards{suffix}"] = _scalar(shaped[mask].mean())
            result[f"episode_lengths{suffix}"] = _scalar(lengths.mean())
        groups(torch.ones_like(actions, dtype=torch.bool), "")
        groups(torch.ones_like(actions, dtype=torch.bool), "/object_cube1")
        for action in range(7):
            mask = actions == action; label = f"cube1_{('01','02','03','04','09','10','18')[action]}"
            groups(mask, f"/{label}")
            if bool(mask.any()):
                result[f"reward_by_pair/{label}/total_reward"] = _scalar(raw[mask].mean())
                result[f"reward_by_pair_instant/{label}/total_reward"] = _scalar((raw[mask] / length[mask]).mean())
        total = float(len(actions))
        for code, name in ((1, "failure"), (0, "success"), (2, "timeout")):
            result[f"training/termination_rates/{name}"] = _scalar((reason == code).sum().float() / total)
        return result


def _write_schema(output: Path) -> None:
    (output / "metrics_schema.json").write_text(json.dumps({
        "schema": "dexhandrl.gym_tensorboard.v1", "phase_1_full_metrics_available": False,
        "provenance": "source-owned device runtime; no fabricated historical values",
        "axes": {"iter": "global update", "step": "global transitions", "time": "cumulative elapsed seconds"},
        "reward_components": list(REWARD_COMPONENTS),
    }, indent=2, sort_keys=True), encoding="utf-8")


def _write_metrics(writer: Any, metrics: dict[str, float], *, update: int, transitions: int, elapsed: float) -> None:
    if writer is None: return
    for tag, value in metrics.items():
        if tag.startswith(("performance/", "info/", "losses/")): writer.add_scalar(tag, value, transitions)
        else: writer.add_scalar(tag, value, update)
    for key in ("rewards", "shaped_rewards", "episode_lengths"):
        if key in metrics:
            writer.add_scalar(f"{key}/iter", metrics[key], update); writer.add_scalar(f"{key}/step", metrics[key], transitions); writer.add_scalar(f"{key}/time", metrics[key], elapsed)
    writer.add_scalar("global_step", transitions, transitions); writer.flush()


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
        try: init(trainer_cfg={"timesteps": int(cfg.timesteps)})
        except TypeError: init()  # test doubles and older skrl only
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
                boundary = time.perf_counter(); transition = env.step_device(bridge.torch_to_jax(actions.contiguous())); step_time += time.perf_counter() - boundary
                next_obs = bridge.jax_to_torch(transition.observation); raw = bridge.jax_to_torch(transition.reward); rewards = raw.clone()
                terminated = bridge.jax_to_torch(transition.terminated.astype("float32")).bool(); truncated = bridge.jax_to_torch(transition.truncated.astype("float32")).bool(); done = terminated | truncated
                weighted_value = getattr(transition, "weighted", None)
                reason_value = getattr(transition, "reason", None)
                weighted = bridge.jax_to_torch(weighted_value) if weighted_value is not None else torch.zeros((env.num_envs, 10), device=observations.device)
                reason = bridge.jax_to_torch(reason_value.astype("float32")).long() if reason_value is not None else torch.where(terminated, 1, torch.where(truncated, 2, 0))
                transition_valid = bridge.jax_to_torch(transition.valid.astype("float32")).bool(); finite = finite and bool(torch.isfinite(next_obs).all() and torch.isfinite(raw).all()); valid = valid and bool(transition_valid.all())
                agent.record_transition(observations=observations, states=observations, actions=actions, rewards=rewards.unsqueeze(-1), next_observations=next_obs, next_states=next_obs, terminated=terminated.unsqueeze(-1), truncated=truncated.unsqueeze(-1), infos={"time_outs": truncated.unsqueeze(-1)}, timestep=global_timestep, timesteps=int(cfg.timesteps))
                # record_transition may in-place add timeout bootstrap; that is shaped reward.
                episodes.step(raw, rewards.squeeze(-1), weighted, done, reason)
                agent.post_interaction(timestep=global_timestep + 1, timesteps=int(cfg.timesteps)); reward_sum += raw.sum(); done_sum += done.sum(); timeout_sum += truncated.sum(); observations, pending = next_obs, done
            play_time = time.perf_counter() - play_start; update_start = time.perf_counter()
            if not finite or not valid: raise RuntimeError(f"invalid batch transition at PPO update {update}: finite={finite}, valid={valid}")
            _assert_finite_agent_parameters(agent, update=update); telemetry = env.telemetry(); _capacity_guard(telemetry); update_time = time.perf_counter() - update_start
            transitions = update * int(cfg.rollouts) * int(env.num_envs); elapsed = time.perf_counter() - started
            metrics = episodes.flush(); metrics.update(_consume_tracking(agent))
            lr_mul = metrics.get("info/last_lr", float(cfg.learning_rate)) / float(cfg.learning_rate)
            metrics.update({"info/lr_mul": lr_mul, "info/e_clip": float(cfg.ratio_clip) * lr_mul, "info/epochs": float(update), "health/contact_high_water": float(telemetry.get("contact_high_water", 0)), "health/ncollision_high_water": float(telemetry.get("ncollision_high_water", 0)), "health/constraint_high_water": float(telemetry.get("constraint_high_water", 0)), "performance/step_time": step_time, "performance/step_inference_time": play_time, "performance/rl_update_time": update_time, "performance/step_fps": transitions / max(step_time, 1e-9), "performance/step_inference_fps": (int(cfg.rollouts) * env.num_envs) / max(play_time, 1e-9), "performance/step_inference_rl_update_fps": (int(cfg.rollouts) * env.num_envs) / max(play_time + update_time, 1e-9)})
            record = {"update": update, "transitions": transitions, "global_step": transitions, "reward_mean": _scalar(reward_sum / (cfg.rollouts * env.num_envs)), "done": _scalar(done_sum), "timeout": _scalar(timeout_sum), "elapsed": elapsed, "finite": finite, "transition.valid": valid, "contact_high_water": int(telemetry.get("contact_high_water", 0)), "ncollision_high_water": int(telemetry.get("ncollision_high_water", 0)), "constraint_high_water": int(telemetry.get("constraint_high_water", 0)), "capacity": {key: telemetry[key] for key in telemetry if "capacity" in key or "buffer_shape" in key}, "metrics": metrics}
            checkpoint_due = update == start_update + 1 or update % int(cfg.checkpoint_interval_updates) == 0 or update == total_updates
            if update % int(cfg.log_interval_updates) == 0 or checkpoint_due: logs.write(json.dumps(record, sort_keys=True) + "\n"); logs.flush()
            _write_metrics(writer, metrics, update=update, transitions=transitions, elapsed=elapsed)
            if checkpoint_due: _checkpoint(agent, output, update, {"progress": record, "config": asdict(cfg), "physics": cfg.physics.record(), "assignment": assignment, "model_config": asdict(model_config), "resume_environment_reset": bool(start_update)})
    if writer is not None: writer.close()
    return BatchPPOResult(total_updates - start_update, (total_updates - start_update) * int(cfg.rollouts) * int(env.num_envs), log_path)
