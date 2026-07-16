#!/usr/bin/env python3
"""Compare legacy and device-resident ManoRL control/reset stepping.

Example (use an idle GPU only):
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  tools/benchmark_manorl_device_controls.py --device gpu --num-envs 64 --warmup-steps 4 --steps 24
"""

from __future__ import annotations

import argparse
import json
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


def _run_mode(args: argparse.Namespace, *, device_resident_controls: bool) -> dict[str, Any]:
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
            capture_transition_diagnostics=not device_resident_controls,
        ),
    )
    actions = np.linspace(-0.75, 0.75, args.num_envs * 26, dtype=np.float64).reshape(args.num_envs, 26)
    for _ in range(args.warmup_steps):
        environment.step(actions)
    environment.jax.block_until_ready(environment.data.qpos)

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
        "mode": "device_resident_controls" if device_resident_controls else "legacy_controls",
        "device": str(environment.device),
        "steps": args.steps,
        "num_envs": args.num_envs,
        "elapsed_seconds": elapsed,
        "transitions_per_second": args.steps * args.num_envs / elapsed,
        "reward_mean": float(np.mean(rewards)),
        "reset_count": resets,
        "transition_diagnostics": environment.last_transition is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    if args.num_envs < 1 or args.warmup_steps < 0 or args.steps < 1:
        parser.error("num-envs and steps must be positive; warmup-steps must be non-negative")
    results = [_run_mode(args, device_resident_controls=False), _run_mode(args, device_resident_controls=True)]
    print(json.dumps({"schema": "manorl.device_controls_benchmark.v1", "results": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
