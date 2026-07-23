"""Native passive viewer for one production DexHandRL MJX environment.

The viewer is an observer: environment ``step`` owns every physics transition;
this module mirrors the resulting MJX state into native ``MjData`` for display.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Iterator

import numpy as np

from sim.dexhandrl.constants import DEFAULT_ISAAC_SOURCE_ROOT, DEFAULT_LANCE_PATH
from sim.dexhandrl.contracts import DexHandPhysicsConfig, resolve_physics_config


_TERMINAL_THRESHOLD = 0.1
_DISABLED_TERMINAL_THRESHOLD = 1_000_000.0


def _parse_cli_bool(value: str) -> bool:
    """Parse the explicit true/false CLI contract used by ManoRL viewers."""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


class _StoreExplicit(argparse.Action):
    """Store a CLI value and retain whether the canonical flag was supplied."""

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, value: Any, option_string: str | None = None) -> None:
        setattr(namespace, self.dest, value)
        setattr(namespace, f"{self.dest}_explicit", True)


@dataclass(frozen=True)
class ViewerOptions:
    use_residual: bool
    terminal_threshold: float
    speed: float
    sleep_enabled: bool


def _resolve_viewer_options(args: argparse.Namespace) -> ViewerOptions:
    """Normalize canonical viewer flags and their deprecated compatibility aliases once."""
    if args.use_residual_explicit and args.legacy_use_residual is not None:
        raise ValueError("--use_residual cannot be combined with --use-residual/--no-use-residual")
    if args.terminal_explicit and args.object_position_terminal_threshold is not None:
        raise ValueError("--terminal cannot be combined with --object-position-terminal-threshold")
    if args.speed_explicit and args.realtime is not None:
        raise ValueError("--speed cannot be combined with --realtime/--no-realtime")

    use_residual = args.use_residual if args.legacy_use_residual is None else args.legacy_use_residual
    terminal_threshold = (
        args.object_position_terminal_threshold
        if args.object_position_terminal_threshold is not None
        else _TERMINAL_THRESHOLD if args.terminal else _DISABLED_TERMINAL_THRESHOLD
    )
    if args.realtime is True:
        speed, sleep_enabled = 1.0, True
    elif args.realtime is False:
        speed, sleep_enabled = args.speed, False
    else:
        speed, sleep_enabled = args.speed, True
    if speed <= 0.0:
        raise ValueError("--speed must be positive")
    return ViewerOptions(
        use_residual=use_residual,
        terminal_threshold=terminal_threshold,
        speed=speed,
        sleep_enabled=sleep_enabled,
    )


def _sleep_duration(*, control_dt: float, speed: float, compute_time: float) -> float:
    """Return the remaining wall time for one control step at ``speed``."""
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    return max(0.0, control_dt / speed - compute_time)


def _require_graphical_display(*, headless: bool) -> None:
    if headless:
        raise ValueError("the standalone native DexHand viewer is inherently graphical; --headless is unsupported")
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("the native DexHand viewer requires DISPLAY to name a graphical display server")


_VIEWER_SHUTDOWN_TIMEOUT_S = 5.0
_VIEWER_SHUTDOWN_POLL_S = 0.01


def _wait_for_passive_viewer_shutdown(
    handle: Any,
    *,
    timeout_s: float = _VIEWER_SHUTDOWN_TIMEOUT_S,
    clock: Any = time.monotonic,
    sleeper: Any = time.sleep,
) -> None:
    """Wait for MuJoCo's passive-viewer thread to release its simulator.

    ``Handle.close()`` only asks the viewer thread to stop.  The private
    ``_sim`` weakref becomes empty after that thread has left GLFW/GLX, which
    must happen before Python's GLFW atexit handler unloads the library.
    """
    sim_ref = getattr(handle, "_sim", None)
    if not callable(sim_ref):
        return

    deadline = clock() + timeout_s
    while sim_ref() is not None:
        remaining = deadline - clock()
        if remaining <= 0:
            raise RuntimeError(
                f"MuJoCo passive viewer did not stop within {timeout_s:g}s after close()"
            )
        sleeper(min(_VIEWER_SHUTDOWN_POLL_S, remaining))


def mirror_mjx_state_to_mjdata(data: Any, state: Any) -> None:
    """Copy the render-relevant dynamic state without advancing simulation."""
    for name in ("qpos", "qvel", "ctrl"):
        # Older trace callers passed a minimal qpos/qvel fixture. Production
        # DexHand state always supplies ctrl and time, which are mirrored too.
        if not hasattr(state, name) or not hasattr(data, name):
            continue
        destination = getattr(data, name)
        source = np.asarray(getattr(state, name))
        if destination.shape != source.shape:
            raise ValueError(f"native viewer {name} shape mismatch: {source.shape} -> {destination.shape}")
        np.copyto(destination, source)
    if hasattr(state, "time") and hasattr(data, "time"):
        data.time = float(np.asarray(state.time))


class PassiveDexHandViewer:
    """Native passive observer with a camera centered on the tracked object."""

    def __init__(self, mujoco: Any, model: Any, data: Any, handle: Any, object_body_id: int) -> None:
        self.mujoco, self.model, self.data, self.handle = mujoco, model, data, handle
        self.object_body_id = object_body_id
        self._closed = False
        self._camera_configured = False

    @classmethod
    def create(cls, env: Any) -> "PassiveDexHandViewer":
        _require_graphical_display(headless=False)
        import mujoco.viewer

        data = env.mujoco.MjData(env.model)
        return cls(env.mujoco, env.model, data, mujoco.viewer.launch_passive(env.model, data), env.object_body_id)

    def sync(self, source: Any) -> bool:
        """Mirror ``source`` and redraw; this method never calls a step primitive."""
        if self._closed:
            return False
        if not self.handle.is_running():
            self.close()
            return False
        with self.handle.lock():
            copier = getattr(source, "copy_state_to_mjdata", None)
            if callable(copier):
                copier(self.data)
            else:  # Compatibility for trace callers that pass MJX data directly.
                mirror_mjx_state_to_mjdata(self.data, source)
            self.mujoco.mj_forward(self.model, self.data)
            if not self._camera_configured:
                camera = getattr(self.handle, "cam", None)
                if camera is not None:
                    camera.lookat[:] = self.data.xpos[self.object_body_id]
                    camera.distance, camera.azimuth, camera.elevation = 0.6, 90.0, -25.0
                self._camera_configured = True
        self.handle.sync()
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.handle.close()
        _wait_for_passive_viewer_shutdown(self.handle)


@contextmanager
def passive_dexhand_viewer(env: Any, *, headless: bool) -> Iterator[PassiveDexHandViewer | None]:
    if headless:
        yield None
        return
    viewer = PassiveDexHandViewer.create(env)
    try:
        yield viewer
    finally:
        viewer.close()


class ZeroActionStepper:
    """Production stepper that supplies zero 22D policy actions each control step."""

    def __init__(self, environment: Any, *, loop: bool) -> None:
        self.environment = environment
        self.loop = loop
        self.actions = np.zeros((environment.num_actions,), dtype=np.float32)
        self.finished = False
        self.environment.reset()

    def step(self) -> tuple[Any, float, bool, bool, dict[str, Any]] | None:
        if self.finished:
            return None
        transition = self.environment.step(self.actions)
        if transition[2] or transition[3]:
            if self.loop:
                self.environment.reset()
            else:
                self.finished = True
        return transition


def _physics_from_args(args: argparse.Namespace) -> DexHandPhysicsConfig:
    return resolve_physics_config(
        dt=args.dt, substeps=args.substeps,
        base_pos_kp=args.base_pos_kp, base_pos_kv=args.base_pos_kv, base_pos_force=args.base_pos_force,
        base_rot_kp=args.base_rot_kp, base_rot_kv=args.base_rot_kv, base_rot_force=args.base_rot_force,
        finger_kp=args.finger_kp, finger_kv=args.finger_kv, finger_force=args.finger_force,
        hand_friction=args.hand_friction, object_friction=args.object_friction,
        ground_friction=args.ground_friction, hand_self_collision_mode=args.hand_self_collision_mode,
        collision=args.collision, naconmax=args.naconmax, njmax=args.njmax,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(
        use_residual_explicit=False,
        terminal_explicit=False,
        speed_explicit=False,
    )
    parser.add_argument("--object-name", "--object", dest="object_name", default="cube1")
    parser.add_argument("--action", "--gesture", dest="action", default="01")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--sequence-index", type=int, default=0)
    selector.add_argument("--uuid")
    parser.add_argument("--lance-path", type=Path, default=DEFAULT_LANCE_PATH)
    parser.add_argument("--isaac-source-root", type=Path, default=DEFAULT_ISAAC_SOURCE_ROOT)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--use_residual", type=_parse_cli_bool, default=True, action=_StoreExplicit,
        metavar="{true,false}", help="enable 22D residual action processing (default: true)",
    )
    parser.add_argument(
        "--use-residual", action=argparse.BooleanOptionalAction, dest="legacy_use_residual",
        default=None, help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--terminal", type=_parse_cli_bool, default=True, action=_StoreExplicit,
        metavar="{true,false}", help="use the 0.10 m object-position reset threshold (default: true)",
    )
    parser.add_argument(
        "--object-position-terminal-threshold", type=float, default=None, help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--speed", type=float, default=0.25, action=_StoreExplicit,
        help="simulation speed multiplier (default: 0.25)",
    )
    parser.add_argument("--checkpoint", type=Path, help="unsupported here: this viewer currently runs native zero actions")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False, help=argparse.SUPPRESS)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--substeps", type=int)
    for name in ("base-pos-kp", "base-pos-kv", "base-pos-force", "base-rot-kp", "base-rot-kv", "base-rot-force", "finger-kp", "finger-kv", "finger-force"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--hand-friction")
    parser.add_argument("--object-friction")
    parser.add_argument("--ground-friction")
    parser.add_argument("--hand-self-collision-mode", choices=("none", "disable_within_finger", "all_disabled"))
    parser.add_argument("--collision", choices=("none", "disable_within_finger", "all_disabled"))
    parser.add_argument("--naconmax", type=int)
    parser.add_argument("--njmax", type=int)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.checkpoint is not None:
        raise ValueError("--checkpoint is not supported by the native zero-action viewer")
    options = _resolve_viewer_options(args)
    _require_graphical_display(headless=args.headless)
    from sim.dexhandrl.skrl_env import DexHandRLMJXEnv, DexHandRLMJXEnvConfig

    physics = _physics_from_args(args)
    env = DexHandRLMJXEnv(DexHandRLMJXEnvConfig(
        lance_path=args.lance_path, isaac_source_root=args.isaac_source_root,
        object_name=args.object_name, action=args.action, sequence_index=args.sequence_index,
        uuid=args.uuid, device=args.device, use_residual=options.use_residual,
        object_position_terminal_threshold=options.terminal_threshold, physics=physics,
    ))
    try:
        stepper = ZeroActionStepper(env, loop=args.loop)
        label = "reference targets only" if not options.use_residual else "zero residual actions (no checkpoint)"
        print(
            f"DexHandRL viewer: {label}; action_dim={env.num_actions}; control_dt={physics.control_dt:g}s; "
            f"speed={options.speed:g}; use_residual={options.use_residual}; "
            f"terminal_threshold={options.terminal_threshold:g}"
        )
        steps = 0
        with passive_dexhand_viewer(env, headless=False) as viewer:
            assert viewer is not None
            while viewer.sync(env):
                if args.max_steps is not None and steps >= args.max_steps:
                    break
                started = time.monotonic()
                if stepper.step() is None:
                    break
                steps += 1
                if options.sleep_enabled:
                    time.sleep(_sleep_duration(
                        control_dt=physics.control_dt,
                        speed=options.speed,
                        compute_time=time.monotonic() - started,
                    ))
        return steps
    finally:
        env.close()


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
