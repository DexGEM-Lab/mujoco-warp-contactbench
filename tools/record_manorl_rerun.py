"""Record actual ManoRL transitions for a deterministic Lance trajectory assignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from sim.manorl.abi import TARGET_MAX_DEVIATION_DISTANCE
from sim.manorl.cli import parse_cli_bool
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.rerun_recorder import ManoRerunRecorder
from sim.manorl.trajectory import TrajectorySelection, load_assigned_trajectory_batch
from sim.manorl.view_environment import (
    _build_checkpoint_stepper,
    _checkpoint_use_film,
    _inference_ppo_config,
    _validate_checkpoint_path,
)


class _StochasticCheckpointStepper:
    """Sample the loaded Gaussian policy, matching the training action path."""

    def __init__(self, environment: MujocoManoEnvironment, checkpoint: Path) -> None:
        import torch
        from sim.manorl.checkpoint import load_skrl_checkpoint_for_inference
        from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv
        from sim.manorl.skrl_runtime import ManoSkrlRuntime

        self._torch = torch
        adapter = ManoGymnasiumVectorEnv(environment)
        self._runtime = ManoSkrlRuntime(
            adapter,
            _inference_ppo_config(
                environment.config.num_envs,
                use_film=_checkpoint_use_film(checkpoint),
            ),
        )
        load_skrl_checkpoint_for_inference(self._runtime.agent, checkpoint)
        self._runtime.agent.enable_training_mode(True)
        self._observations, _ = self._runtime.env.reset()

    def step(self):
        with self._torch.no_grad():
            actions, _ = self._runtime.agent.act(
                self._observations,
                None,
                timestep=0,
                timesteps=1,
            )
        self._observations, rewards, terminated, truncated, info = self._runtime.env.step(actions)
        resets = terminated | truncated
        return self._observations, rewards.detach().cpu().numpy().reshape(-1), resets.detach().cpu().numpy().reshape(-1), info


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
    parser.add_argument("--use_residual", type=parse_cli_bool, default=True, metavar="{true,false}")
    parser.add_argument("--terminal", type=parse_cli_bool, default=True, metavar="{true,false}")
    parser.add_argument("--checkpoint", type=Path, help="optional native ManoRL checkpoint for deterministic actions")
    parser.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="sample the checkpoint Gaussian policy instead of using its deterministic mean",
    )
    parser.add_argument("--archive-dir", type=Path, help="preserve completed episodes above the threshold")
    parser.add_argument("--archive-threshold", type=float)
    parser.add_argument("--archive-following", type=int, default=5)
    parser.add_argument("--open-rerun", action="store_true", help="open the published .rrd after recording")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.num_envs < 1:
        parser.error("steps and num-envs must be positive")
    if args.archive_following < 0:
        parser.error("archive-following must be non-negative")
    if (args.archive_dir is None) != (args.archive_threshold is None):
        parser.error("archive-dir and archive-threshold must be supplied together")
    if args.checkpoint is not None and not args.use_residual:
        parser.error("checkpoint actions require --use_residual true")
    if args.stochastic_policy and args.checkpoint is None:
        parser.error("stochastic-policy requires --checkpoint")
    checkpoint = None if args.checkpoint is None else _validate_checkpoint_path(args.checkpoint)
    selection = TrajectorySelection(object_type=args.object_type, gesture=args.gesture)
    trajectories = load_assigned_trajectory_batch(selection, num_envs=args.num_envs)
    print(json.dumps({"selection": {"object": selection.object_type, "gesture": selection.action_id}, "assignments": _assignment_payload(trajectories)}, indent=2))
    environment = MujocoManoEnvironment(
        trajectories,
        EnvironmentConfig(
            device=args.device,
            num_envs=args.num_envs,
            residual_enabled=args.use_residual,
            max_deviation_distance=TARGET_MAX_DEVIATION_DISTANCE if args.terminal else 1_000_000.0,
            contact_capacity=max(128, 31 * args.num_envs + 64),
        ),
    )
    recorder_kwargs: dict[str, object] = {"env_id": args.env_id}
    if args.archive_dir is not None or args.archive_threshold is not None:
        recorder_kwargs.update(
            archive_dir=args.archive_dir,
            archive_threshold=args.archive_threshold,
            archive_following=args.archive_following,
        )
    recorder = ManoRerunRecorder(environment, args.output, **recorder_kwargs)
    actions = np.zeros((args.num_envs, 26), dtype=np.float64)
    if checkpoint is None:
        stepper = None
    elif args.stochastic_policy:
        stepper = _StochasticCheckpointStepper(environment, checkpoint)
    else:
        stepper = _build_checkpoint_stepper(environment, checkpoint)
    try:
        for _ in range(args.steps):
            if stepper is None:
                environment.step(actions)
            else:
                stepper.step()
            recorder.record_transition()
    finally:
        artifact = recorder.close()
    print(json.dumps({"rerun_artifact": None if artifact is None else str(artifact)}, indent=2))
    if args.open_rerun:
        if artifact is None:
            print("No reset-complete Rerun episode was published; Rerun viewer was not opened.")
            return 0
        viewer = Path(sys.executable).with_name("rerun")
        if not viewer.exists():
            raise RuntimeError(f"Rerun viewer executable is absent: {viewer}")
        subprocess.Popen([str(viewer), str(artifact)], stdin=subprocess.DEVNULL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
