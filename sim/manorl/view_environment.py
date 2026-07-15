"""Interactively test the actual residual-off cube1 MJX-Warp environment.

This uses :class:`MujocoManoEnvironment`, not the narrow reference-replay
helper. The viewer mirrors its one-world MJX state into native ``MjData`` only
for rendering; observations, rewards, termination, delayed reset, and action
processing all remain on the production environment path.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from sim.manorl.contracts import CONTROL_TIMESTEP
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.trajectory import load_generated_cube1_row_507, load_reference_trajectory


def _telemetry(environment: MujocoManoEnvironment, reward: float, reset: bool) -> str:
    call = int(environment.progress[0] - 1)
    command_index = max(call - 1, 0)
    post_index = int(environment.trajectory_steps[0])
    command = np.array2string(
        environment.last_controller_targets[0],
        precision=3,
        suppress_small=True,
        max_line_width=240,
    )
    return (
        f"call={call:03d}/{len(environment.trajectory.q_ref) - 2} command_ref={command_index:03d} "
        f"post_ref={post_index:03d} source_ref={environment.trajectory.source_indices[post_index]:04d} "
        f"reward={reward:.4f} reset={reset} ctrl={command}"
    )


def _require_graphical_session() -> None:
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    raise RuntimeError(
        "MuJoCo viewer needs an X11 or Wayland graphical session; set DISPLAY or run from a desktop terminal"
    )


def view_environment(
    *,
    device: str,
    speed: float,
    loop: bool,
    print_every: int,
    training_termination: bool,
    trajectory_name: str,
    num_envs: int,
) -> None:
    """Run a batched production environment and render its first world."""

    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if print_every < 1:
        raise ValueError("print_every must be positive")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    _require_graphical_session()

    import mujoco
    import mujoco.viewer

    if trajectory_name == "accepted":
        trajectory = load_reference_trajectory()
    elif trajectory_name == "generated-cube1-row-507":
        trajectory = load_generated_cube1_row_507()
    else:
        raise ValueError(f"unsupported viewer trajectory {trajectory_name!r}")
    max_deviation_distance = 0.1 if training_termination else 1_000_000.0
    contact_capacity = max(128, 31 * num_envs + 64)
    environment = MujocoManoEnvironment(
        trajectory,
        EnvironmentConfig(
            device=device,
            num_envs=num_envs,
            residual_enabled=False,
            max_deviation_distance=max_deviation_distance,
            contact_capacity=contact_capacity,
        ),
    )
    if environment.config.residual_enabled:
        raise RuntimeError("visual environment test must run with residual actions disabled")
    render_data = environment.host_data()
    zero_action = np.zeros((num_envs, 26), dtype=np.float64)
    sleep_seconds = CONTROL_TIMESTEP / speed

    print(
        "Testing MujocoManoEnvironment with residual_enabled=False and zero residual action "
        f"(trajectory={trajectory.identity.identity}, frames={len(trajectory.q_ref)}, "
        f"envs={num_envs}, maxDeviationDistance={max_deviation_distance:g}). "
        "The viewer renders env 0; all configured environments execute the same batched path."
    )
    with mujoco.viewer.launch_passive(
        environment.model, render_data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        viewer.sync()
        while viewer.is_running():
            started = time.perf_counter()
            _, rewards, resets, _ = environment.step(zero_action)
            mujoco.mj_copyData(render_data, environment.model, environment.host_data())
            viewer.sync()

            call = int(environment.progress[0] - 1)
            if call % print_every == 0 or bool(resets[0]):
                print(_telemetry(environment, float(rewards[0]), bool(resets[0])), flush=True)
            if bool(resets[0]) and not loop:
                return
            time.sleep(max(0.0, sleep_seconds - (time.perf_counter() - started)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.25,
        help="Simulation speed multiplier; 0.25 makes a source control trajectory easy to inspect.",
    )
    parser.add_argument(
        "--trajectory",
        choices=("accepted", "generated-cube1-row-507"),
        default="accepted",
        help="explicit versioned Lance trajectory contract to render",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="batched environments to execute; viewer renders env 0",
    )
    parser.add_argument(
        "--training-termination",
        action="store_true",
        help="use the current training deviation threshold (0.1 m) instead of formal 791-call replay termination",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue through the environment's source-compatible delayed reset after terminal.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print the applied 26D source command every N control calls.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    view_environment(
        device=args.device,
        speed=args.speed,
        loop=args.loop,
        print_every=args.print_every,
        training_termination=args.training_termination,
        trajectory_name=args.trajectory,
        num_envs=args.num_envs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
