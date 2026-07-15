"""Record actual residual-off ManoRL environment transitions to a Rerun .rrd file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.rerun_recorder import ManoRerunRecorder
from sim.manorl.trajectory import TrajectoryBatch, load_cube1_action_01_batch10, load_reference_trajectory


def _trajectory(name: str):
    if name == "accepted":
        return load_reference_trajectory()
    if name == "accepted-cube1-action-01-batch10":
        return load_cube1_action_01_batch10()
    raise ValueError(f"unsupported trajectory {name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--trajectory", choices=("accepted", "accepted-cube1-action-01-batch10"), default="accepted")
    parser.add_argument("--training-termination", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.num_envs < 1:
        parser.error("steps and num-envs must be positive")
    trajectory = _trajectory(args.trajectory)
    if isinstance(trajectory, TrajectoryBatch) and args.num_envs != trajectory.num_envs:
        parser.error(f"{args.trajectory} requires --num-envs {trajectory.num_envs}")
    environment = MujocoManoEnvironment(
        trajectory,
        EnvironmentConfig(
            device=args.device,
            num_envs=args.num_envs,
            residual_enabled=False,
            max_deviation_distance=0.1 if args.training_termination else 1_000_000.0,
            contact_capacity=max(128, 31 * args.num_envs + 64),
        ),
    )
    recorder = ManoRerunRecorder(environment, args.output, env_id=args.env_id)
    actions = np.zeros((args.num_envs, 26), dtype=np.float64)
    for _ in range(args.steps):
        environment.step(actions)
        recorder.record_transition()
    print(recorder.close())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
