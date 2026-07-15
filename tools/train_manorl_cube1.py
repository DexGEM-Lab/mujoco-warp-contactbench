#!/usr/bin/env python3
"""Bounded CUDA PPO training for the single accepted cube1_01_009 trajectory."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from sim.manorl.checkpoint import load_skrl_checkpoint, save_skrl_checkpoint
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv
from sim.manorl.skrl_runtime import ManoPPOConfig, ManoSkrlRuntime
from sim.manorl.trajectory import load_reference_trajectory

WARP_BROADPHASE_CONTACTS_PER_WORLD = 31
WARP_CONTACT_CAPACITY_MARGIN = 64
RESIDUAL_SAFE_INITIAL_LOG_STD = -5.0

@dataclass(frozen=True)
class TrainingBudget:
    num_envs: int = 64
    updates: int = 64
    wall_clock_seconds: float = 20.0 * 60.0
    seed: int = 42
    residual_safe_warm_start: bool = True

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
                completed_horizon=call == 790 and timeout_seen,
                rewards_by_call=reward_means,
                object_target_distance_by_call=distance_means,
            )
    raise RuntimeError(f"{mode} evaluation did not terminate at the source horizon")


def _train(runtime: ManoSkrlRuntime, budget: TrainingBudget) -> tuple[list[dict[str, float]], int, float]:
    environment = runtime.gymnasium_env.environment
    config = runtime.config
    runtime.agent.enable_training_mode(True)
    observations, _ = runtime.env.reset()
    started = time.monotonic()
    updates: list[dict[str, float]] = []
    global_timestep = 0
    for update in range(budget.updates):
        if time.monotonic() - started >= budget.wall_clock_seconds:
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
        updates.append({
            "update": float(update + 1),
            "environment_transitions": float((update + 1) * config.rollouts * environment.config.num_envs),
            "reward_mean": float(torch.cat(rewards).mean().item()),
            "action_abs_mean": float(torch.cat(action_magnitudes).mean().item()),
            "reset_count": float(reset_count),
            "elapsed_seconds": time.monotonic() - started,
        })
    return updates, global_timestep * environment.config.num_envs, time.monotonic() - started


def run(output: Path, budget: TrainingBudget) -> dict[str, Any]:
    _assert_cuda_runtime()
    if output.suffix:
        raise ValueError("--output must be a prefix without a suffix")
    output = output.resolve()
    checkpoint = output.with_suffix(".pt")
    metrics_path = output.with_suffix(".json")
    trace_path = output.with_suffix(".eval.npz")
    if any(path.exists() for path in (checkpoint, metrics_path, trace_path)):
        raise FileExistsError("refusing to replace an existing training artifact prefix")
    torch.manual_seed(budget.seed)
    np.random.seed(budget.seed)
    torch.cuda.manual_seed_all(budget.seed)

    contact_capacity = max(
        128,
        WARP_BROADPHASE_CONTACTS_PER_WORLD * budget.num_envs + WARP_CONTACT_CAPACITY_MARGIN,
    )
    physical = MujocoManoEnvironment(
        load_reference_trajectory(),
        EnvironmentConfig(
            num_envs=budget.num_envs,
            device="gpu",
            residual_enabled=True,
            max_deviation_distance=0.1,
            contact_capacity=contact_capacity,
        ),
    )
    ppo_config = ManoPPOConfig()
    runtime = ManoSkrlRuntime(ManoGymnasiumVectorEnv(physical), ppo_config)
    if runtime.device != "cuda":
        raise RuntimeError(f"skrl runtime must train on CUDA, got {runtime.device!r}")
    # The first two 48-step rollouts are entirely source-defined pure mocap.
    # Updating PPO on their zero reward signal moves the shared actor/critic
    # representation before the policy has any controllable consequence.
    runtime.agent.cfg.learning_starts = physical.contact_start_frame
    if budget.residual_safe_warm_start:
        with torch.no_grad():
            runtime.model.actor_head.weight.zero_()
            runtime.model.actor_head.bias.zero_()
            runtime.model.log_std.fill_(RESIDUAL_SAFE_INITIAL_LOG_STD)
    zero_baseline = _evaluate(runtime, "zero")
    untrained = _evaluate(runtime, "policy")
    updates, transitions, elapsed = _train(runtime, budget)
    save_skrl_checkpoint(runtime.agent, checkpoint, runtime_config=runtime.checkpoint_metadata())
    # Evaluate exactly what a user will later load. skrl preprocessor/module
    # state may differ in-process after PPO training, so a fresh native load is
    # the reproducibility boundary rather than an implementation detail.
    evaluation_physical = MujocoManoEnvironment(
        load_reference_trajectory(),
        EnvironmentConfig(
            num_envs=budget.num_envs,
            device="gpu",
            residual_enabled=True,
            max_deviation_distance=0.1,
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
        "trajectory": "cube1_01_009",
        "checkpoint_conversion": "out_of_scope",
        "initialization": {
            "residual_safe_warm_start": budget.residual_safe_warm_start,
            "actor_mean": "zero" if budget.residual_safe_warm_start else "source_default",
            "initial_log_std": RESIDUAL_SAFE_INITIAL_LOG_STD if budget.residual_safe_warm_start else -0.99,
            "ppo_learning_rate": ppo_config.learning_rate,
        },
        "learning_starts": runtime.agent.cfg.learning_starts,
        "budget": {
            **asdict(budget),
            "planned_transitions": budget.transitions,
            "warp_contact_capacity": contact_capacity,
        },
        "actual": {"transitions": transitions, "elapsed_seconds": elapsed, "updates": len(updates)},
        "device": {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "skrl": runtime.device, "jax": physical.jax.default_backend()},
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
        "artifacts": {"checkpoint": str(checkpoint), "evaluation_trace": str(trace_path)},
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--wall-clock-seconds", type=float, default=20.0 * 60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-initialization",
        action="store_true",
        help="disable the zero-residual/-5 log-std warm start used by the fast single-trajectory protocol",
    )
    args = parser.parse_args(argv)
    if args.updates < 1 or args.num_envs < 1 or args.wall_clock_seconds <= 0:
        parser.error("updates, num-envs, and wall-clock-seconds must be positive")
    result = run(
        args.output,
        TrainingBudget(
            args.num_envs,
            args.updates,
            args.wall_clock_seconds,
            args.seed,
            residual_safe_warm_start=not args.source_initialization,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
