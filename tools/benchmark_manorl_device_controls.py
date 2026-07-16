#!/usr/bin/env python3
"""Measure the four-way ManoRL environment control/diagnostic ablation.

Example (use an idle GPU only):
  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
    /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
    tools/benchmark_manorl_device_controls.py --device gpu --num-envs 2048 \
    --warmup-steps 8 --steps 48 --repeats 3 --profile-phases
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.trajectory import TrajectorySelection, load_assigned_trajectory_batch


def _mode_name(*, device_resident_controls: bool, capture_transition_diagnostics: bool) -> str:
    return (
        f"{'device' if device_resident_controls else 'legacy'}_controls_"
        f"{'diagnostics' if capture_transition_diagnostics else 'no_diagnostics'}"
    )


def _run_mode(
    args: argparse.Namespace,
    *,
    device_resident_controls: bool,
    capture_transition_diagnostics: bool,
    repeat: int,
    order_index: int,
) -> dict[str, Any]:
    contact_capacity = max(128, 31 * args.num_envs + 64)
    trajectories = load_assigned_trajectory_batch(
        TrajectorySelection(object_type="cube1", gesture="01"), num_envs=args.num_envs
    )
    environment = MujocoManoEnvironment(
        trajectories,
        EnvironmentConfig(
            num_envs=args.num_envs,
            device=args.device,
            contact_capacity=contact_capacity,
            device_resident_controls=device_resident_controls,
            capture_transition_diagnostics=capture_transition_diagnostics,
            profile_phases=args.profile_phases,
        ),
    )
    actions = np.linspace(-0.75, 0.75, args.num_envs * 26, dtype=np.float64).reshape(args.num_envs, 26)
    for _ in range(args.warmup_steps):
        environment.step(actions)
    environment.jax.block_until_ready(environment.data.qpos)
    environment.phase_timings.reset()

    started = time.perf_counter()
    rewards: list[float] = []
    resets = 0
    for _ in range(args.steps):
        _, reward, reset, _ = environment.step(actions)
        rewards.append(float(reward.mean()))
        resets += int(reset.sum())
    environment.jax.block_until_ready(environment.data.qpos)
    elapsed = time.perf_counter() - started
    return {
        "mode": _mode_name(
            device_resident_controls=device_resident_controls,
            capture_transition_diagnostics=capture_transition_diagnostics,
        ),
        "device_resident_controls": device_resident_controls,
        "capture_transition_diagnostics": capture_transition_diagnostics,
        "repeat": repeat,
        "order_index": order_index,
        "device": str(environment.device),
        "steps": args.steps,
        "num_envs": args.num_envs,
        "elapsed_seconds": elapsed,
        "transitions_per_second": args.steps * args.num_envs / elapsed,
        "reward_mean": float(np.mean(rewards)),
        "reset_count": resets,
        "transition_diagnostics": environment.last_transition is not None,
        "phase_profile": environment.phase_profile() if args.profile_phases else {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=42)
    parser.add_argument("--profile-phases", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 1 or args.warmup_steps < 0 or args.steps < 1 or args.repeats < 1:
        parser.error("num-envs, steps, and repeats must be positive; warmup-steps must be non-negative")
    modes = [
        (False, True),
        (False, False),
        (True, True),
        (True, False),
    ]
    order_rng = random.Random(args.order_seed)
    results: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        order = list(modes)
        order_rng.shuffle(order)
        for order_index, (device_resident_controls, capture_transition_diagnostics) in enumerate(order):
            results.append(
                _run_mode(
                    args,
                    device_resident_controls=device_resident_controls,
                    capture_transition_diagnostics=capture_transition_diagnostics,
                    repeat=repeat,
                    order_index=order_index,
                )
            )
    print(
        json.dumps(
            {
                "schema": "manorl.device_controls_ablation.v2",
                "config": {
                    "device": args.device,
                    "num_envs": args.num_envs,
                    "warmup_steps": args.warmup_steps,
                    "steps": args.steps,
                    "repeats": args.repeats,
                    "order_seed": args.order_seed,
                    "profile_phases": args.profile_phases,
                },
                "results": results,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
