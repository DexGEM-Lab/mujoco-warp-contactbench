"""Record actual ManoRL transitions for a deterministic Lance trajectory assignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.rerun_recorder import ManoRerunRecorder
from sim.manorl.trajectory import TrajectorySelection, load_assigned_trajectory_batch


def _assignment_payload(trajectory_batch) -> list[dict[str, object]]:
    return [
        {
            "env_id": env_id,
            "identity": trajectory.identity.identity,
            "row_index": trajectory.identity.row_index,
            "uuid": trajectory.identity.uuid,
            "source_slice": [trajectory.identity.source_start, trajectory.identity.source_stop],
        }
        for env_id, trajectory in enumerate(trajectory_batch.trajectories)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="stable latest-env0 .rrd output, atomically replaced at episode boundaries")
    parser.add_argument("--object", dest="object_type", default="cube1")
    parser.add_argument("--gesture", default="01")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--training-termination", action="store_true")
    parser.add_argument("--no-viewer", action="store_true", help="do not launch the Rerun GUI after recording")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.num_envs < 1:
        parser.error("steps and num-envs must be positive")
    selection = TrajectorySelection(object_type=args.object_type, gesture=args.gesture)
    trajectories = load_assigned_trajectory_batch(selection, num_envs=args.num_envs)
    print(json.dumps({"selection": {"object": selection.object_type, "gesture": selection.action_id}, "assignments": _assignment_payload(trajectories)}, indent=2))
    environment = MujocoManoEnvironment(
        trajectories,
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
    artifact = recorder.close()
    print(json.dumps({"rerun_artifact": str(artifact)}, indent=2))
    if not args.no_viewer:
        viewer = Path(sys.executable).with_name("rerun")
        if not viewer.exists():
            raise RuntimeError(f"Rerun viewer executable is absent: {viewer}")
        subprocess.Popen([str(viewer), str(artifact)], stdin=subprocess.DEVNULL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
