"""Graphical viewer for one production DexHandRL environment on either backend.

The viewer is an observer: environment ``step`` owns every physics transition.
For MJX-Warp it mirrors state into native ``MjData`` for display; for explicit
CPU-only native MuJoCo it renders the live native state without stepping it.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Any, Iterator

import numpy as np

from sim.dexhandrl.asset_paths import contact_bodies_path, task_config_path
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


def _tile_layout(count: int, *, width: int, height: int) -> list[tuple[int, int, int, int]]:
    if count < 1:
        raise ValueError("tile_envs must be positive")
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    tile_width, tile_height = width // columns, height // rows
    return [
        ((index % columns) * tile_width, height - (index // columns + 1) * tile_height, tile_width, tile_height)
        for index in range(count)
    ]


class TiledDexHandRenderer:
    """Local GLFW/MuJoCo observer for a batch runtime; it never owns physics."""

    def __init__(self, environment: Any, *, tile_envs: int) -> None:
        if not 1 <= tile_envs <= environment.num_envs:
            raise ValueError("tile_envs must be within [1, num_envs]")
        _require_graphical_display(headless=False)
        import glfw

        self.environment, self.tile_envs = environment, tile_envs
        self.glfw, self.mujoco = glfw, environment.mujoco
        self.width, self.height = 1600, 900
        self.window: Any | None = None
        self._initialized = False
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed for tiled DexHand rendering")
        self._initialized = True
        try:
            self.window = glfw.create_window(self.width, self.height, f"DexHandRL batch viewer ({tile_envs} worlds)", None, None)
            if self.window is None:
                raise RuntimeError("could not create GLFW window for tiled DexHand rendering")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            mujoco = self.mujoco
            self.data = [mujoco.MjData(environment.model) for _ in range(tile_envs)]
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.camera)
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._reset_camera()
            self.option = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.option)
            self.scene = mujoco.MjvScene(environment.model, maxgeom=10_000)
            self.context = mujoco.MjrContext(environment.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.viewports = _tile_layout(tile_envs, width=self.width, height=self.height)
            self._install_controls()
        except BaseException:
            self.close()
            raise

    def _reset_camera(self) -> None:
        self.camera.azimuth, self.camera.elevation, self.camera.distance = 135.0, -20.0, 1.2
        self.camera.lookat[:] = (0.25, 0.0, 0.1)

    def _install_controls(self) -> None:
        glfw, mujoco, window = self.glfw, self.mujoco, self.window
        mouse = {"left": False, "middle": False, "right": False, "x": 0.0, "y": 0.0}
        def button(_: Any, which: int, action: int, __: int) -> None:
            if which == glfw.MOUSE_BUTTON_LEFT: mouse["left"] = action == glfw.PRESS
            elif which == glfw.MOUSE_BUTTON_MIDDLE: mouse["middle"] = action == glfw.PRESS
            elif which == glfw.MOUSE_BUTTON_RIGHT: mouse["right"] = action == glfw.PRESS
            mouse["x"], mouse["y"] = glfw.get_cursor_pos(window)
        def cursor(_: Any, x: float, y: float) -> None:
            w, h = glfw.get_window_size(window); dx, dy = (x - mouse["x"]) / max(w, 1), (y - mouse["y"]) / max(h, 1)
            mouse["x"], mouse["y"] = x, y
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V if mouse["left"] else (mujoco.mjtMouse.mjMOUSE_MOVE_V if mouse["middle"] else (mujoco.mjtMouse.mjMOUSE_MOVE_H if mouse["right"] else None))
            if action is not None: mujoco.mjv_moveCamera(self.environment.model, action, dx, -dy, self.scene, self.camera)
        def scroll(_: Any, __: float, offset: float) -> None:
            mujoco.mjv_moveCamera(self.environment.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * offset, self.scene, self.camera)
        def key(_: Any, key: int, __: int, action: int, ___: int) -> None:
            if action == glfw.PRESS and key == glfw.KEY_R: self._reset_camera()
            elif action == glfw.PRESS and key == glfw.KEY_ESCAPE: glfw.set_window_should_close(window, True)
        glfw.set_mouse_button_callback(window, button); glfw.set_cursor_pos_callback(window, cursor)
        glfw.set_scroll_callback(window, scroll); glfw.set_key_callback(window, key)

    def render(self, snapshot: Any) -> bool:
        """Mirror a minimal snapshot; no simulation step primitive is reachable here."""
        if self.window is None or self.glfw.window_should_close(self.window): return False
        glfw, mujoco = self.glfw, self.mujoco
        width, height = glfw.get_framebuffer_size(self.window)
        if (width, height) != (self.width, self.height):
            self.width, self.height = width, height; self.viewports = _tile_layout(self.tile_envs, width=width, height=height)
        mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, width, height), 0.055, 0.085, 0.12, 1.0)
        for env_id, (x, y, tile_width, tile_height) in enumerate(self.viewports):
            state = type("RenderState", (), {name: getattr(snapshot, name)[env_id] for name in ("qpos", "qvel", "ctrl")})()
            mirror_mjx_state_to_mjdata(self.data[env_id], state)
            mujoco.mj_forward(self.environment.model, self.data[env_id])
            mujoco.mjv_updateScene(self.environment.model, self.data[env_id], self.option, None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            viewport = mujoco.MjrRect(x, y, tile_width, tile_height)
            mujoco.mjr_render(viewport, self.scene, self.context)
            action = self.environment.batch.actions[env_id]
            row_id = int(self.environment.batch.row_ids[env_id])
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                f"world {env_id}  cube1/{action}",
                f"row {row_id}  frame {int(snapshot.frame[env_id])}  episode {int(snapshot.episode_count[env_id])}",
                self.context,
            )
        glfw.swap_buffers(self.window); glfw.poll_events()
        return not glfw.window_should_close(self.window)

    def close(self) -> None:
        window, initialized = self.window, self._initialized
        self.window, self._initialized = None, False
        try:
            if window is not None: self.glfw.destroy_window(window)
        finally:
            if initialized: self.glfw.terminate()


class BatchZeroActionStepper:
    """Zero 22D residual actions; reset terminal worlds on device before next step."""

    def __init__(self, environment: Any) -> None:
        self.environment = environment
        self.actions = np.zeros((environment.num_envs, environment.num_actions), dtype=np.float32)
        self.pending_done = None
        self.environment.reset_device()

    def step(self) -> None:
        if self.pending_done is not None:
            self.environment.reset_done_device(self.pending_done)
        transition = self.environment.step_device(self.actions)
        self.pending_done = transition.terminated | transition.truncated


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


def _validate_batch_options(args: argparse.Namespace) -> None:
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    if args.tile_envs < 1 or args.tile_envs > args.num_envs:
        raise ValueError("--tile-envs must be within [1, num-envs]")
    if args.render_stride < 1:
        raise ValueError("--render-stride must be positive")
    if args.num_envs > 1:
        if args.runtime_backend != "mjx-warp":
            raise ValueError("multi-env viewing requires --runtime-backend mjx-warp")
        if args.object_name != "cube1":
            raise ValueError("multi-env viewing requires --object cube1")
        if args.action not in {"01", "all"}:
            raise ValueError("multi-env viewing requires --gesture 01 or all")
        if args.action == "all" and args.num_envs < 7:
            raise ValueError("all-action viewing requires at least 7 environments")


def _resolve_batched_assignments(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Resolve stable single-action or seven-action viewer worlds."""
    from sim.dexhandrl.lance_loader import list_trajectories
    from sim.dexhandrl.trajectory_batch import resolve_balanced_action_assignments

    rows = list_trajectories(args.lance_path, limit=None)
    if args.action == "all":
        if args.uuid is not None:
            raise RuntimeError("--uuid cannot be combined with --gesture all")
        return resolve_balanced_action_assignments(
            rows,
            object_name="cube1",
            num_envs=args.num_envs,
            sequence_index=args.sequence_index,
            duplicate_first_for_extra=args.num_envs == 8,
        )

    selected = [
        row
        for row in rows
        if row["object_name"] == "cube1" and row["action"] == "01"
    ]
    selected.sort(key=lambda row: (int(row["row_id"]), str(row["uuid"])))
    if not selected:
        raise RuntimeError(f"no trajectories found for cube1/01 in {args.lance_path}")
    if args.uuid is not None:
        matched = [
            (index, row)
            for index, row in enumerate(selected)
            if row["uuid"] == args.uuid
        ]
        if not matched:
            raise RuntimeError(f"trajectory uuid {args.uuid!r} not found for cube1/01")
        index, row = matched[0]
        return [{**row, "sequence_index": index} for _ in range(args.num_envs)]
    start = args.sequence_index % len(selected)
    return [
        {
            **selected[(start + env_id) % len(selected)],
            "sequence_index": (start + env_id) % len(selected),
        }
        for env_id in range(args.num_envs)
    ]


def _run_batched(args: argparse.Namespace, options: ViewerOptions, physics: DexHandPhysicsConfig) -> int:
    from sim.dexhandrl.skrl_batch_env import DexHandRLBatchConfig, DexHandRLMJXBatchEnv
    from sim.dexhandrl.trajectory_batch import load_trajectory_batch
    root = Path(args.isaac_source_root)
    assignments = _resolve_batched_assignments(args)
    trajectories = load_trajectory_batch(
        args.lance_path, assignments, pre_contact_frames=200, post_contact_frames=250,
        contact_bodies_file=contact_bodies_path(root),
        task_config_path=task_config_path(root),
    )
    from dataclasses import replace
    from sim.dexhandrl.rewards import SOURCE_ALIGNED_REWARD_CONFIG

    reward = replace(
        SOURCE_ALIGNED_REWARD_CONFIG,
        max_deviation_distance_m=options.terminal_threshold,
    )
    env = DexHandRLMJXBatchEnv(DexHandRLBatchConfig(
        trajectories=trajectories,
        action="all",
        device=args.device,
        object_name="cube1",
        physics=physics,
        use_residual=options.use_residual,
        reward=reward,
        isaac_source_root=root,
    ))
    try:
        assignments_label = ",".join(env.batch.actions)
        print(
            f"DexHandRL tiled viewer: backend=mjx-warp device={args.device} "
            f"worlds={env.num_envs} tiles={args.tile_envs} actions={assignments_label}; "
            f"zero 22D residual actions use_residual={options.use_residual}",
            flush=True,
        )
        stepper, steps = BatchZeroActionStepper(env), 0
        renderer = TiledDexHandRenderer(env, tile_envs=args.tile_envs)
        try:
            while True:
                started = time.monotonic()
                stepper.step()
                steps += 1
                if steps % args.render_stride == 0 and not renderer.render(env.render_snapshot()):
                    break
                if args.max_steps is not None and steps >= args.max_steps:
                    break
                if options.sleep_enabled:
                    time.sleep(_sleep_duration(control_dt=physics.control_dt, speed=options.speed, compute_time=time.monotonic() - started))
        finally:
            renderer.close()
        return steps
    finally:
        env.close()


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
    parser.add_argument("--runtime-backend", choices=("mujoco-native", "mjx-warp"), default="mujoco-native")
    parser.add_argument("--num-envs", type=int, default=1, help="number of genuine MJX-Warp worlds (default: 1)")
    parser.add_argument("--tile-envs", type=int, default=1, help="number of worlds rendered as tiles (default: 1)")
    parser.add_argument("--render-stride", type=int, default=1, help="render every N completed batch transitions (default: 1)")
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
    if args.runtime_backend == "mujoco-native" and args.device != "cpu":
        raise ValueError("--runtime-backend mujoco-native requires --device cpu")
    _validate_batch_options(args)
    options = _resolve_viewer_options(args)
    _require_graphical_display(headless=args.headless)
    physics = _physics_from_args(args)
    if args.num_envs > 1:
        return _run_batched(args, options, physics)
    from sim.dexhandrl.skrl_env import DexHandRLMJXEnv, DexHandRLMJXEnvConfig

    env = DexHandRLMJXEnv(DexHandRLMJXEnvConfig(
        lance_path=args.lance_path, isaac_source_root=args.isaac_source_root,
        object_name=args.object_name, action=args.action, sequence_index=args.sequence_index,
        uuid=args.uuid, device=args.device, runtime_backend=args.runtime_backend, use_residual=options.use_residual,
        object_position_terminal_threshold=options.terminal_threshold, physics=physics,
    ))
    try:
        stepper = ZeroActionStepper(env, loop=args.loop)
        label = "reference targets only" if not options.use_residual else "zero residual actions (no checkpoint)"
        print(
            f"DexHandRL viewer: backend={env.cfg.runtime_backend}; {label}; action_dim={env.num_actions}; control_dt={physics.control_dt:g}s; "
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
