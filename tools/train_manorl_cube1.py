#!/usr/bin/env python3
"""Bounded CUDA PPO training for a selected object/gesture Lance trajectory batch."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal
from uuid import uuid4

import numpy as np
import torch

from sim.manorl.checkpoint import load_skrl_checkpoint, save_skrl_checkpoint
from sim.manorl.cli import parse_cli_bool
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv
from sim.manorl.rerun_recorder import ManoRerunRecorder
from sim.manorl.rewards import PPO_REWARD_CONTRACT_ID, PPO_REWARD_SCALE, REWARD_CONTRACT_ID
from sim.manorl.skrl_runtime import ManoPPOConfig, ManoSkrlRuntime
from sim.manorl.trajectory import TrajectorySelection, load_assigned_trajectory_batch

WARP_BROADPHASE_CONTACTS_PER_WORLD = 31
WARP_CONTACT_CAPACITY_MARGIN = 64
DEFAULT_WANDB_TAGS = ("manorl", "mujoco", "skrl")


@dataclass(frozen=True)
class WandbOptions:
    enabled: bool = False
    project: str = "one_policy"
    group: str = "s02"
    entity: str = ""
    name: str | None = None
    tags: tuple[str, ...] = DEFAULT_WANDB_TAGS


@dataclass(frozen=True)
class TrainingBudget:
    num_envs: int = 64
    updates: int = 64
    wall_clock_seconds: float | None = None
    seed: int = 42
    rerun_output: str | None = None
    rerun_env_id: int = 0
    rerun_stride: int = 1
    object_type: str = "cube1"
    gesture: str = "01"
    residual_enabled: bool = True
    terminal: bool = True
    wandb: WandbOptions = WandbOptions()
    checkpoint_interval_updates: int | None = None

    @property
    def transitions(self) -> int:
        return self.num_envs * ManoPPOConfig().rollouts * self.updates


@dataclass(frozen=True)
class EvaluationResult:
    mode: str
    calls: int
    return_mean: float
    reward_mean: float
    action_abs_mean: float
    final_object_target_distance: float
    max_object_target_distance: float
    contact_reward_mean: float
    reset_seen: bool
    timeout_seen: bool
    completed_horizon: bool
    rewards_by_call: list[float]
    object_target_distance_by_call: list[float]


def _wandb_run_name(output: Path, budget: TrainingBudget) -> str:
    return budget.wandb.name or f"{output.name}-{budget.object_type}-{budget.gesture}"


def _parse_wandb_tags(values: list[str]) -> tuple[str, ...]:
    tags = tuple(tag.strip() for value in values for tag in value.split(",") if tag.strip())
    return tags or DEFAULT_WANDB_TAGS


def _wandb_config(
    *,
    budget: TrainingBudget,
    ppo_config: ManoPPOConfig,
    trajectory_assignments: list[dict[str, object]],
    device: dict[str, object],
) -> dict[str, object]:
    config = {
        "training_budget": {**asdict(budget), "planned_transitions": budget.transitions},
        "ppo_config": asdict(ppo_config),
        "reward": {
            "environment_contract": REWARD_CONTRACT_ID,
            "ppo_contract": PPO_REWARD_CONTRACT_ID,
            "ppo_scale": PPO_REWARD_SCALE,
            "isaacgym_ppo_scale": 0.5,
        },
        "trajectory_assignments": trajectory_assignments,
        "device": device,
    }
    try:
        return json.loads(json.dumps(config, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("W&B config must be JSON-serializable") from exc


@contextmanager
def _wandb_run(
    *, output: Path, budget: TrainingBudget, config: dict[str, object]
) -> Iterator[tuple[Any | None, Any | None]]:
    if not budget.wandb.enabled:
        yield None, None
        return
    wandb_dir = output.parent
    wandb_dir.mkdir(parents=True, exist_ok=True)
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B is enabled but the wandb package is unavailable") from exc
    try:
        run = wandb.init(
            project=budget.wandb.project,
            entity=budget.wandb.entity or None,
            group=budget.wandb.group,
            name=_wandb_run_name(output, budget),
            tags=list(budget.wandb.tags),
            config=config,
            dir=str(wandb_dir),
        )
    except Exception as exc:
        raise RuntimeError("W&B initialization failed") from exc
    if run is None:
        raise RuntimeError("W&B initialization returned no run")
    try:
        yield run, wandb
    except BaseException as body_error:
        try:
            run.finish(exit_code=1)
        except BaseException as cleanup_error:
            body_error.add_note(f"W&B cleanup after training failure also failed: {cleanup_error!r}")
        raise
    else:
        try:
            run.finish()
        except BaseException as cleanup_error:
            raise RuntimeError("W&B cleanup after successful training failed") from cleanup_error


def _log_wandb_update(run: Any, update: dict[str, float]) -> None:
    transitions = int(update["environment_transitions"])
    run.log(
        {
            "global_step": transitions,
            "transitions": transitions,
            "update": int(update["update"]),
            "reward_mean": update["reward_mean"],
            "action_abs_mean": update["action_abs_mean"],
            "reset_count": update["reset_count"],
            "elapsed_seconds": update["elapsed_seconds"],
        },
        step=transitions,
    )


def _evaluation_metrics(result: EvaluationResult) -> dict[str, object]:
    return {
        f"evaluation/{result.mode}/{key}": value
        for key, value in asdict(result).items()
        if key not in {"rewards_by_call", "object_target_distance_by_call", "mode"}
    }


def _log_wandb_evaluations(run: Any, results: list[EvaluationResult], *, transitions: int) -> None:
    metrics: dict[str, object] = {"global_step": transitions, "transitions": transitions}
    for result in results:
        metrics.update(_evaluation_metrics(result))
    run.log(metrics, step=transitions)
    run.summary.update(metrics)


def _log_wandb_artifacts(run: Any, wandb: Any, *, output: Path, paths: list[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot log missing W&B artifacts: {', '.join(str(path) for path in missing)}")
    artifact = wandb.Artifact(name=f"{output.name}-artifacts", type="manorl-training")
    for path in paths:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)


def _assert_cuda_runtime() -> None:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA-capable target Torch is required for fast ManoRL training")


def _evaluate(runtime: ManoSkrlRuntime, mode: Literal["zero", "policy"]) -> EvaluationResult:
    environment = runtime.gymnasium_env.environment
    # ``deterministic_actions`` only selects the Gaussian mean. PointNet/FiLM
    # still contains BatchNorm, so evaluation must also switch skrl modules to
    # eval mode or the reported policy changes with vector batch statistics.
    runtime.agent.enable_models_training_mode(False)
    observations, _ = runtime.env.reset()
    returns = torch.zeros((environment.config.num_envs, 1), device=runtime.device)
    action_abs: list[float] = []
    contacts: list[float] = []
    reward_means: list[float] = []
    distance_means: list[float] = []
    reset_seen = False
    timeout_seen = False
    for call in range(791):
        if mode == "zero":
            actions = torch.zeros((environment.config.num_envs, 26), device=runtime.device)
        else:
            actions = runtime.deterministic_actions(observations)
        observations, rewards, terminated, truncated, info = runtime.env.step(actions)
        if not torch.isfinite(rewards).all() or not torch.isfinite(observations).all():
            raise RuntimeError(f"non-finite {mode} evaluation value at call {call}")
        returns += rewards
        action_abs.append(float(actions.abs().mean().item()))
        reward_means.append(float(rewards.mean().item()))
        physical = environment.last_physical
        diagnostics = environment.last_reward
        if physical is None or diagnostics is None:
            raise RuntimeError("environment omitted physical/reward diagnostics during evaluation")
        indices = environment.trajectory_steps
        target = environment.trajectory.object_pos[indices]
        distances = np.linalg.norm(physical.object_position - target, axis=1)
        distance_means.append(float(distances.mean()))
        contacts.append(float(diagnostics.contact.mean()))
        done = terminated | truncated
        reset_seen |= bool(done.any().item())
        timeout_seen |= bool(truncated.any().item())
        if bool(done.any().item()):
            return EvaluationResult(
                mode=mode,
                calls=call + 1,
                return_mean=float(returns.mean().item()),
                reward_mean=float(np.mean(reward_means)),
                action_abs_mean=float(np.mean(action_abs)),
                final_object_target_distance=float(distances.mean()),
                max_object_target_distance=float(np.max(distance_means)),
                contact_reward_mean=float(np.mean(contacts)),
                reset_seen=reset_seen,
                timeout_seen=timeout_seen,
                completed_horizon=call == 790 and reset_seen,
                rewards_by_call=reward_means,
                object_target_distance_by_call=distance_means,
            )
    raise RuntimeError(f"{mode} evaluation did not terminate at the source horizon")


def _train(
    runtime: ManoSkrlRuntime,
    budget: TrainingBudget,
    recorder: ManoRerunRecorder | None = None,
    on_update: Callable[[dict[str, float]], None] | None = None,
) -> tuple[list[dict[str, float]], int, float]:
    environment = runtime.gymnasium_env.environment
    config = runtime.config
    runtime.agent.enable_training_mode(True)
    observations, _ = runtime.env.reset()
    started = time.monotonic()
    updates: list[dict[str, float]] = []
    global_timestep = 0
    for update in range(budget.updates):
        if (
            budget.wall_clock_seconds is not None
            and time.monotonic() - started >= budget.wall_clock_seconds
        ):
            break
        rewards: list[torch.Tensor] = []
        action_magnitudes: list[torch.Tensor] = []
        reset_count = 0
        for _ in range(config.rollouts):
            with torch.no_grad():
                actions, _ = runtime.agent.act(
                    observations, None, timestep=global_timestep, timesteps=budget.transitions
                )
            next_observations, reward, terminated, truncated, infos = runtime.env.step(actions)
            if recorder is not None and global_timestep % budget.rerun_stride == 0:
                recorder.record_transition()
            if not torch.isfinite(next_observations).all() or not torch.isfinite(reward).all():
                raise RuntimeError(f"non-finite rollout value at global timestep {global_timestep}")
            runtime.agent.record_transition(
                observations=observations,
                states=None,
                actions=actions,
                rewards=reward,
                next_observations=next_observations,
                next_states=None,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=global_timestep,
                timesteps=budget.transitions,
            )
            runtime.agent.post_interaction(timestep=global_timestep + 1, timesteps=budget.transitions)
            observations = next_observations
            rewards.append(reward.detach())
            action_magnitudes.append(actions.detach().abs())
            reset_count += int((terminated | truncated).sum().item())
            global_timestep += 1
        if not all(torch.isfinite(parameter).all() for parameter in runtime.model.parameters()):
            raise RuntimeError(f"PPO update {update} produced non-finite model parameters")
        update_metrics = {
            "update": float(update + 1),
            "environment_transitions": float((update + 1) * config.rollouts * environment.config.num_envs),
            "reward_mean": float(torch.cat(rewards).mean().item()),
            "action_abs_mean": float(torch.cat(action_magnitudes).mean().item()),
            "reset_count": float(reset_count),
            "elapsed_seconds": time.monotonic() - started,
        }
        updates.append(update_metrics)
        if on_update is not None:
            on_update(update_metrics)
    return updates, global_timestep * environment.config.num_envs, time.monotonic() - started


def _numbered_checkpoint_path(output: Path, completed_updates: int) -> Path:
    return output / f"checkpoint-{completed_updates:06d}.pt"


def _last_checkpoint_path(output: Path) -> Path:
    return output / "last.pt"


def _checkpoint_runtime_config(runtime: ManoSkrlRuntime, update: dict[str, float]) -> dict[str, object]:
    return {
        **runtime.checkpoint_metadata(),
        "training_progress": {
            "completed_updates": int(update["update"]),
            "environment_transitions": int(update["environment_transitions"]),
        },
    }


def _checkpoint_sidecar_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".json")


def _temporary_checkpoint_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f".{checkpoint.name}.{uuid4().hex}.tmp")


def _save_checkpoint_atomically(
    agent: Any, checkpoint: Path, *, runtime_config: dict[str, object]
) -> Path:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = _temporary_checkpoint_path(checkpoint)
    temporary_sidecar = _checkpoint_sidecar_path(temporary_checkpoint)
    sidecar = _checkpoint_sidecar_path(checkpoint)
    try:
        save_skrl_checkpoint(agent, temporary_checkpoint, runtime_config=runtime_config)
        metadata = json.loads(temporary_sidecar.read_text(encoding="utf-8"))
        metadata["checkpoint_file"] = checkpoint.name
        temporary_sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_checkpoint, checkpoint)
        os.replace(temporary_sidecar, sidecar)
    finally:
        temporary_checkpoint.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
    return checkpoint


def _last_sidecar_content(checkpoint: Path, last_checkpoint: Path) -> bytes:
    metadata = json.loads(_checkpoint_sidecar_path(checkpoint).read_text(encoding="utf-8"))
    runtime_config = dict(metadata["runtime_config"])
    runtime_config.pop("training_progress", None)
    metadata["checkpoint_file"] = last_checkpoint.name
    metadata["runtime_config"] = runtime_config
    metadata["progress_metadata"] = "immutable numbered and final checkpoint sidecars"
    return (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _update_last_checkpoint(output: Path, checkpoint: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    last_checkpoint = _last_checkpoint_path(output)
    last_sidecar = _checkpoint_sidecar_path(last_checkpoint)
    sidecar_content = _last_sidecar_content(checkpoint, last_checkpoint)
    if last_sidecar.exists():
        if not last_sidecar.is_file() or last_sidecar.read_bytes() != sidecar_content:
            raise ValueError(f"last checkpoint compatibility sidecar does not match: {last_sidecar}")
    else:
        if last_checkpoint.exists():
            raise FileExistsError(f"last checkpoint exists without compatibility sidecar: {last_checkpoint}")
        temporary_sidecar = _temporary_checkpoint_path(last_sidecar)
        try:
            temporary_sidecar.write_bytes(sidecar_content)
            os.replace(temporary_sidecar, last_sidecar)
        finally:
            temporary_sidecar.unlink(missing_ok=True)
    temporary_checkpoint = _temporary_checkpoint_path(last_checkpoint)
    try:
        shutil.copyfile(checkpoint, temporary_checkpoint)
        os.replace(temporary_checkpoint, last_checkpoint)
    finally:
        temporary_checkpoint.unlink(missing_ok=True)
    return last_checkpoint


def _save_periodic_checkpoint(
    runtime: ManoSkrlRuntime, output: Path, update: dict[str, float]
) -> Path:
    checkpoint = _numbered_checkpoint_path(output, int(update["update"]))
    sidecar = _checkpoint_sidecar_path(checkpoint)
    if checkpoint.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to replace periodic checkpoint artifact: {checkpoint}")
    return _save_checkpoint_atomically(
        runtime.agent, checkpoint, runtime_config=_checkpoint_runtime_config(runtime, update)
    )


def _maybe_save_periodic_checkpoint(
    runtime: ManoSkrlRuntime,
    output: Path,
    checkpoint_interval_updates: int | None,
    update: dict[str, float],
) -> Path | None:
    completed_updates = int(update["update"])
    if checkpoint_interval_updates is None or completed_updates % checkpoint_interval_updates != 0:
        return None
    checkpoint = _save_periodic_checkpoint(runtime, output, update)
    _update_last_checkpoint(output, checkpoint)
    return checkpoint


def run(output: Path, budget: TrainingBudget) -> dict[str, Any]:
    _assert_cuda_runtime()
    if output.suffix:
        raise ValueError("--output must be a prefix without a suffix")
    output = output.resolve()
    checkpoint = output.with_suffix(".pt")
    last_checkpoint = _last_checkpoint_path(output)
    metrics_path = output.with_suffix(".json")
    trace_path = output.with_suffix(".eval.npz")
    rerun_path = Path(budget.rerun_output).resolve() if budget.rerun_output else None
    artifacts = (
        checkpoint,
        _checkpoint_sidecar_path(checkpoint),
        metrics_path,
        trace_path,
    )
    rerun_existing = [] if rerun_path is None else [
        rerun_path,
        rerun_path.with_name(f".{rerun_path.stem}.active{rerun_path.suffix or '.rrd'}"),
    ]
    if (
        output.exists()
        or any(path.exists() for path in artifacts)
        or any(path.exists() for path in rerun_existing)
    ):
        raise FileExistsError("refusing to replace an existing training artifact prefix")
    torch.manual_seed(budget.seed)
    np.random.seed(budget.seed)
    torch.cuda.manual_seed_all(budget.seed)

    contact_capacity = max(
        128,
        WARP_BROADPHASE_CONTACTS_PER_WORLD * budget.num_envs + WARP_CONTACT_CAPACITY_MARGIN,
    )
    selection = TrajectorySelection(object_type=budget.object_type, gesture=budget.gesture)
    trajectories = load_assigned_trajectory_batch(selection, num_envs=budget.num_envs)
    physical = MujocoManoEnvironment(
        trajectories,
        EnvironmentConfig(
            num_envs=budget.num_envs,
            device="gpu",
            residual_enabled=budget.residual_enabled,
            max_deviation_distance=0.1 if budget.terminal else 1_000_000.0,
            contact_capacity=contact_capacity,
        ),
    )
    ppo_config = ManoPPOConfig()
    runtime = ManoSkrlRuntime(ManoGymnasiumVectorEnv(physical), ppo_config)
    if runtime.device != "cuda":
        raise RuntimeError(f"skrl runtime must train on CUDA, got {runtime.device!r}")
    trajectory_assignments = [
        {
            "env_id": env_id,
            "identity": item.identity.identity,
            "row_index": item.identity.row_index,
            "uuid": item.identity.uuid,
            "source_slice": [item.identity.source_start, item.identity.source_stop],
        }
        for env_id, item in enumerate(trajectories.trajectories)
    ]
    device = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "skrl": runtime.device,
        "jax": physical.jax.default_backend(),
    }
    wandb_config = _wandb_config(
        budget=budget,
        ppo_config=ppo_config,
        trajectory_assignments=trajectory_assignments,
        device=device,
    )
    with _wandb_run(output=output, budget=budget, config=wandb_config) as (wandb_run, wandb):
        # The first two 48-step rollouts are entirely source-defined pure mocap.
        # Updating PPO on their zero reward signal moves the shared actor/critic
        # representation before the policy has any controllable consequence.
        runtime.agent.cfg.learning_starts = physical.contact_start_frame
        zero_baseline = _evaluate(runtime, "zero")
        untrained = _evaluate(runtime, "policy")
        if wandb_run is not None:
            _log_wandb_evaluations(wandb_run, [zero_baseline, untrained], transitions=0)
        recorder = (
            ManoRerunRecorder(physical, rerun_path, env_id=budget.rerun_env_id)
            if rerun_path is not None
            else None
        )
        periodic_checkpoints: list[Path] = []

        def on_update(update: dict[str, float]) -> None:
            periodic_checkpoint = _maybe_save_periodic_checkpoint(
                runtime, output, budget.checkpoint_interval_updates, update
            )
            if periodic_checkpoint is not None:
                periodic_checkpoints.append(periodic_checkpoint)
            if wandb_run is not None:
                _log_wandb_update(wandb_run, update)

        try:
            updates, transitions, elapsed = _train(
                runtime,
                budget,
                recorder,
                on_update=on_update,
            )
        finally:
            published_rerun = None if recorder is None else recorder.close()
        rerun_artifact = None if published_rerun is None else str(published_rerun)
        final_update = {
            "update": float(len(updates)),
            "environment_transitions": float(transitions),
        }
        _save_checkpoint_atomically(
            runtime.agent, checkpoint, runtime_config=_checkpoint_runtime_config(runtime, final_update)
        )
        _update_last_checkpoint(output, checkpoint)
        # Evaluate exactly what a user will later load. skrl preprocessor/module
        # state may differ in-process after PPO training, so a fresh native load is
        # the reproducibility boundary rather than an implementation detail.
        evaluation_physical = MujocoManoEnvironment(
            trajectories,
            EnvironmentConfig(
                num_envs=budget.num_envs,
                device="gpu",
                residual_enabled=budget.residual_enabled,
                max_deviation_distance=0.1 if budget.terminal else 1_000_000.0,
                contact_capacity=contact_capacity,
            ),
        )
        evaluation_runtime = ManoSkrlRuntime(ManoGymnasiumVectorEnv(evaluation_physical), ppo_config)
        load_skrl_checkpoint(evaluation_runtime.agent, checkpoint)
        trained = _evaluate(evaluation_runtime, "policy")
        np.savez_compressed(
            trace_path,
            zero_reward=np.asarray(zero_baseline.rewards_by_call, dtype=np.float32),
            untrained_reward=np.asarray(untrained.rewards_by_call, dtype=np.float32),
            trained_reward=np.asarray(trained.rewards_by_call, dtype=np.float32),
            zero_object_target_distance=np.asarray(zero_baseline.object_target_distance_by_call, dtype=np.float32),
            untrained_object_target_distance=np.asarray(untrained.object_target_distance_by_call, dtype=np.float32),
            trained_object_target_distance=np.asarray(trained.object_target_distance_by_call, dtype=np.float32),
        )
        result = {
            "schema": "manorl.cube1_fast_training.v1",
            "trajectory_selection": {
                "object": selection.object_type,
                "gesture": selection.action_id,
                "assignments": trajectory_assignments,
            },
            "checkpoint_conversion": "out_of_scope",
            "initialization": {
                "actor_mean": "source_default",
                "initial_log_std": -0.99,
                "ppo_learning_rate": ppo_config.learning_rate,
            },
            "reward": {
                "environment_contract": REWARD_CONTRACT_ID,
                "ppo_contract": PPO_REWARD_CONTRACT_ID,
                "ppo_scale": PPO_REWARD_SCALE,
                "isaacgym_ppo_scale": 0.5,
            },
            "learning_starts": runtime.agent.cfg.learning_starts,
            "budget": {
                **asdict(budget),
                "planned_transitions": budget.transitions,
                "warp_contact_capacity": contact_capacity,
            },
            "actual": {"transitions": transitions, "elapsed_seconds": elapsed, "updates": len(updates)},
            "device": device,
            "baseline": asdict(zero_baseline),
            "untrained": asdict(untrained),
            "trained": asdict(trained),
            "acceptance": {
                "trained_completed_horizon": trained.completed_horizon,
                "trained_calls_not_before_zero_reference": trained.calls >= zero_baseline.calls,
                "trained_return_exceeds_untrained": trained.return_mean > untrained.return_mean,
                "accepted": (
                    trained.calls >= zero_baseline.calls
                    and trained.return_mean > untrained.return_mean
                ),
            },
            "updates": updates,
            "artifacts": {
                "checkpoint": str(checkpoint),
                "last_checkpoint": str(last_checkpoint),
                "periodic_checkpoints": [str(path) for path in periodic_checkpoints],
                "evaluation_trace": str(trace_path),
                "rerun": rerun_artifact,
            },
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if wandb_run is not None:
            _log_wandb_evaluations(wandb_run, [trained], transitions=transitions)
            acceptance_metrics = {
                "global_step": transitions,
                "transitions": transitions,
                **{f"acceptance/{key}": value for key, value in result["acceptance"].items()},
            }
            wandb_run.log(acceptance_metrics, step=transitions)
            wandb_run.summary.update(acceptance_metrics)
            artifact_paths = [
                checkpoint,
                _checkpoint_sidecar_path(checkpoint),
                last_checkpoint,
                _checkpoint_sidecar_path(last_checkpoint),
                metrics_path,
                trace_path,
            ]
            for periodic_checkpoint in periodic_checkpoints:
                artifact_paths.extend(
                    [periodic_checkpoint, _checkpoint_sidecar_path(periodic_checkpoint)]
                )
            if published_rerun is not None:
                artifact_paths.append(published_rerun)
            _log_wandb_artifacts(wandb_run, wandb, output=output, paths=artifact_paths)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--checkpoint-interval-updates", type=int)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--wall-clock-seconds", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rerun-output", type=Path, help="optional .rrd transition recording for one training env")
    parser.add_argument("--rerun-env-id", type=int, default=0)
    parser.add_argument("--rerun-stride", type=int, default=1)
    parser.add_argument("--object", dest="object_type", default="cube1")
    parser.add_argument("--gesture", default="01")
    parser.add_argument("--use_residual", type=parse_cli_bool, default=True, metavar="{true,false}")
    parser.add_argument("--terminal", type=parse_cli_bool, default=True, metavar="{true,false}")
    parser.add_argument("--wandb", type=parse_cli_bool, default=False, metavar="{true,false}")
    parser.add_argument("--wandb-project", default="one_policy")
    parser.add_argument("--wandb-group", default="s02")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-tags", action="append", default=[], metavar="TAG[,TAG...]")
    args = parser.parse_args(argv)
    if args.updates < 1 or args.num_envs < 1 or args.rerun_stride < 1:
        parser.error("updates, num-envs, and rerun-stride must be positive")
    if args.checkpoint_interval_updates is not None and args.checkpoint_interval_updates < 1:
        parser.error("checkpoint-interval-updates must be positive when provided")
    if args.wall_clock_seconds is not None and (
        not math.isfinite(args.wall_clock_seconds) or args.wall_clock_seconds <= 0
    ):
        parser.error("wall-clock-seconds must be a finite positive value when provided")
    if not 0 <= args.rerun_env_id < args.num_envs:
        parser.error("rerun-env-id must be within num-envs")
    result = run(
        args.output,
        TrainingBudget(
            args.num_envs,
            args.updates,
            args.wall_clock_seconds,
            args.seed,
            str(args.rerun_output.resolve()) if args.rerun_output is not None else None,
            args.rerun_env_id,
            args.rerun_stride,
            args.object_type,
            args.gesture,
            args.use_residual,
            args.terminal,
            WandbOptions(
                enabled=args.wandb,
                project=args.wandb_project,
                group=args.wandb_group,
                entity=args.wandb_entity,
                name=args.wandb_name,
                tags=_parse_wandb_tags(args.wandb_tags),
            ),
            args.checkpoint_interval_updates,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
