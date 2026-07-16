"""Interactively test the actual cube1 MJX-Warp environment.

This uses :class:`MujocoManoEnvironment`, not the narrow reference-replay
helper. The default viewer mirrors one MJX state into native ``MjData``;
the tiled mode renders several independently batched states in one window.
Observations, rewards, termination, delayed reset, and action processing all
remain on the production environment path.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from sim.manorl.cli import parse_cli_bool
from sim.manorl.contracts import CONTROL_TIMESTEP
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.rerun_recorder import ManoRerunRecorder
from sim.manorl.trajectory import (
    TrajectoryBatch,
    TrajectorySelection,
    load_assigned_trajectory_batch,
    load_cube1_action_01_batch10,
    load_generated_cube1_row_507,
    load_reference_trajectory,
)

if TYPE_CHECKING:
    from sim.manorl.skrl_runtime import ManoPPOConfig, ManoSkrlRuntime


class ViewerStepper(Protocol):
    """Advance one production environment control step for a viewer frame."""

    def step(self) -> tuple[Any, np.ndarray, np.ndarray, Any]: ...


class _ZeroActionStepper:
    """Preserve the reference-command viewer path with a zero residual batch."""

    def __init__(self, environment: MujocoManoEnvironment) -> None:
        self._environment = environment
        self._actions = np.zeros((environment.config.num_envs, 26), dtype=np.float64)

    def step(self) -> tuple[Any, np.ndarray, np.ndarray, Any]:
        return self._environment.step(self._actions)


class _CheckpointPolicyStepper:
    """Advance the wrapped environment with deterministic native PPO means."""

    def __init__(self, runtime: ManoSkrlRuntime, observations: Any) -> None:
        self._runtime = runtime
        self._observations = observations

    def step(self) -> tuple[Any, np.ndarray, np.ndarray, Any]:
        actions = self._runtime.deterministic_actions(self._observations)
        observations, rewards, terminated, truncated, info = self._runtime.env.step(actions)
        self._observations = observations
        rewards_array = rewards.detach().cpu().numpy().reshape(-1)
        resets = (terminated | truncated).detach().cpu().numpy().reshape(-1)
        return observations, rewards_array, resets, info


def _inference_ppo_config(num_envs: int) -> ManoPPOConfig:
    """Create a non-training PPO shell valid for any positive vector batch size."""

    from sim.manorl.skrl_runtime import ManoPPOConfig

    return ManoPPOConfig(rollouts=1, minibatch_size=num_envs, learning_epochs=1)


def _validate_checkpoint_path(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if not sidecar.is_file():
        raise ValueError(f"native ManoRL checkpoint sidecar is required: {sidecar}")
    return checkpoint


def _build_checkpoint_stepper(environment: MujocoManoEnvironment, checkpoint: Path) -> ViewerStepper:
    """Load the native policy and reset through its vector wrapper before rendering."""

    from sim.manorl.checkpoint import load_skrl_checkpoint_for_inference
    from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv
    from sim.manorl.skrl_runtime import ManoSkrlRuntime

    adapter = ManoGymnasiumVectorEnv(environment)
    runtime = ManoSkrlRuntime(adapter, _inference_ppo_config(environment.config.num_envs))
    load_skrl_checkpoint_for_inference(runtime.agent, checkpoint)
    runtime.agent.enable_training_mode(False)
    runtime.model.eval()
    observations, _ = runtime.env.reset()
    return _CheckpointPolicyStepper(runtime, observations)


def _telemetry(environment: MujocoManoEnvironment, env_id: int, reward: float, reset: bool) -> str:
    call = int(environment.progress[env_id] - 1)
    command_index = max(call - 1, 0)
    post_index = int(environment.trajectory_steps[env_id])
    command = np.array2string(
        environment.last_controller_targets[env_id],
        precision=3,
        suppress_small=True,
        max_line_width=240,
    )
    return (
        f"env={env_id} trajectory={environment.trajectories[env_id].identity.identity} "
        f"call={call:03d}/{environment.trajectory_lengths[env_id] - 2} command_ref={command_index:03d} "
        f"post_ref={post_index:03d} source_ref={environment.reference_source_indices[env_id, post_index]:04d} "
        f"reward={reward:.4f} reset={reset} ctrl={command}"
    )


def _require_graphical_session() -> None:
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    raise RuntimeError(
        "MuJoCo viewer needs an X11 or Wayland graphical session; set DISPLAY or run from a desktop terminal"
    )


def _tile_layout(count: int, *, width: int = 1600, height: int = 900) -> list[tuple[int, int, int, int]]:
    """Return lower-left OpenGL viewports for a compact tiled batch render."""

    if count < 1 or count > 100:
        raise ValueError("tile count must be in [1, 100]")
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    tile_width = width // columns
    tile_height = height // rows
    return [
        ((index % columns) * tile_width, height - ((index // columns) + 1) * tile_height, tile_width, tile_height)
        for index in range(count)
    ]


def _reset_tiled_camera(camera: object) -> None:
    """Set a stable free camera shared by every batch-rendered viewport."""

    camera.azimuth = 135.0
    camera.elevation = -20.0
    camera.distance = 1.2
    camera.lookat[:] = (0.25, 0.0, 0.1)


def _configure_tiled_visuals(model: object) -> None:
    """Apply renderer-only contrast and lighting for the compact tile view."""

    model.vis.rgba.fog[:] = (0.055, 0.085, 0.12, 1.0)
    model.vis.rgba.haze[:] = (0.32, 0.44, 0.56, 1.0)
    model.vis.headlight.ambient[:] = (0.22, 0.24, 0.28)
    model.vis.headlight.diffuse[:] = (0.62, 0.66, 0.72)
    model.vis.headlight.specular[:] = (0.28, 0.30, 0.34)


def _install_tiled_controls(
    *, glfw: Any, mujoco: Any, window: object, model: object, camera: object, scene: object
) -> None:
    """Install the shared tiled-view camera controls on one GLFW window."""

    mouse_state = {"left": False, "middle": False, "right": False, "x": 0.0, "y": 0.0}

    def mouse_button_callback(_: object, button: int, action: int, __: int) -> None:
        pressed = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse_state["left"] = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            mouse_state["middle"] = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            mouse_state["right"] = pressed
        mouse_state["x"], mouse_state["y"] = glfw.get_cursor_pos(window)

    def cursor_pos_callback(_: object, x: float, y: float) -> None:
        window_width, window_height = glfw.get_window_size(window)
        dx = (x - mouse_state["x"]) / max(window_width, 1)
        dy = (y - mouse_state["y"]) / max(window_height, 1)
        mouse_state["x"], mouse_state["y"] = x, y
        if mouse_state["left"]:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif mouse_state["middle"]:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif mouse_state["right"]:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        else:
            return
        mujoco.mjv_moveCamera(model, action, dx, -dy, scene, camera)

    def scroll_callback(_: object, __: float, yoffset: float) -> None:
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, camera)

    def key_callback(_: object, key: int, __: int, action: int, ___: int) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R:
            _reset_tiled_camera(camera)

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)


class TrainingViewer:
    """Tiled post-step renderer for PPO training; it never advances simulation."""

    def __init__(
        self, environment: MujocoManoEnvironment, *, tile_envs: int, stride: int = 1, quiet: bool = False
    ) -> None:
        if not 1 <= tile_envs <= environment.config.num_envs:
            raise ValueError("tile_envs must be within the configured batch")
        if stride < 1:
            raise ValueError("stride must be positive")
        _require_graphical_session()
        import glfw
        import mujoco

        self._environment, self._tile_envs, self._stride = environment, tile_envs, stride
        self._glfw, self._mujoco, self._frames, self.close_requested = glfw, mujoco, 0, False
        self._window: object | None = None
        self._glfw_initialized = False
        self._width, self._height = 1600, 900
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed for tiled MuJoCo rendering")
        self._glfw_initialized = True
        try:
            self._window = glfw.create_window(
                self._width, self._height, f"ManoRL training ({tile_envs} envs)", None, None
            )
            if self._window is None:
                raise RuntimeError("could not create GLFW window for tiled MuJoCo rendering")
            glfw.make_context_current(self._window)
            glfw.swap_interval(1)
            _configure_tiled_visuals(environment.model)
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._camera)
            self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            _reset_tiled_camera(self._camera)
            self._option = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self._option)
            self._scene = mujoco.MjvScene(environment.model, maxgeom=10_000)
            self._context = mujoco.MjrContext(environment.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self._viewports = _tile_layout(tile_envs, width=self._width, height=self._height)
            _install_tiled_controls(
                glfw=glfw, mujoco=mujoco, window=self._window, model=environment.model,
                camera=self._camera, scene=self._scene,
            )
        except BaseException as setup_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                setup_error.add_note(f"TrainingViewer cleanup also failed: {cleanup_error!r}")
            raise
        if not quiet:
            print("Training viewer controls: left-drag rotate | right-drag pan horizontal | middle-drag pan vertical | wheel zoom | R reset | Esc close", flush=True)

    def render(self) -> None:
        """Mirror and render current states only; training owns all step calls."""
        if self.close_requested:
            return
        self._frames += 1
        if self._frames % self._stride:
            return
        glfw, mujoco = self._glfw, self._mujoco
        if glfw.window_should_close(self._window):
            self.close_requested = True
            return
        host_data = self._environment.host_data_batch()
        width, height = glfw.get_framebuffer_size(self._window)
        if (width, height) != (self._width, self._height):
            self._width, self._height = width, height
            self._viewports = _tile_layout(self._tile_envs, width=width, height=height)
        mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, width, height), 0.055, 0.085, 0.12, 1.0)
        for env_id, (x, y, tile_width, tile_height) in enumerate(self._viewports):
            mujoco.mjv_updateScene(self._environment.model, host_data[env_id], self._option, None, self._camera, mujoco.mjtCatBit.mjCAT_ALL, self._scene)
            viewport = mujoco.MjrRect(x, y, tile_width, tile_height)
            mujoco.mjr_render(viewport, self._scene, self._context)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, f"env {env_id}", self._environment.trajectories[env_id].identity.identity, self._context)
        glfw.swap_buffers(self._window)
        glfw.poll_events()
        self.close_requested = glfw.window_should_close(self._window)

    def observe(self) -> None:
        """Observer adapter used by the trainer after its completed environment step."""
        self.render()

    def close(self) -> None:
        """Release any initialized GLFW resources exactly once."""

        window = self._window
        glfw_initialized = self._glfw_initialized
        self._window = None
        self._glfw_initialized = False
        try:
            if window is not None:
                self._glfw.destroy_window(window)
        finally:
            if glfw_initialized:
                self._glfw.terminate()


def _view_tiled(
    environment: MujocoManoEnvironment,
    *,
    stepper: ViewerStepper,
    tile_envs: int,
    speed: float,
    loop: bool,
    print_every: int,
    recorder: ManoRerunRecorder | None,
) -> None:
    """Render selected batched worlds in one GLFW/MuJoCo window without altering physics."""

    import glfw
    import mujoco

    width, height = 1600, 900
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed for tiled MuJoCo rendering")
    window = glfw.create_window(width, height, f"ManoRL batch viewer ({tile_envs} envs)", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("could not create GLFW window for tiled MuJoCo rendering")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    _configure_tiled_visuals(environment.model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    _reset_tiled_camera(camera)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    scene = mujoco.MjvScene(environment.model, maxgeom=10_000)
    context = mujoco.MjrContext(environment.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    sleep_seconds = CONTROL_TIMESTEP / speed
    viewports = _tile_layout(tile_envs, width=width, height=height)
    _install_tiled_controls(
        glfw=glfw, mujoco=mujoco, window=window, model=environment.model, camera=camera, scene=scene
    )
    print("Tiled controls: left-drag rotate | right-drag pan horizontal | middle-drag pan vertical | wheel zoom | R reset | Esc close")
    try:
        while not glfw.window_should_close(window):
            started = time.perf_counter()
            _, rewards, resets, _ = stepper.step()
            if recorder is not None:
                recorder.record_transition()
            host_data = environment.host_data_batch()
            framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)
            if (framebuffer_width, framebuffer_height) != (width, height):
                width, height = framebuffer_width, framebuffer_height
                viewports = _tile_layout(tile_envs, width=width, height=height)
            mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, width, height), 0.055, 0.085, 0.12, 1.0)
            for env_id, (x, y, tile_width, tile_height) in enumerate(viewports):
                mujoco.mjv_updateScene(
                    environment.model,
                    host_data[env_id],
                    option,
                    None,
                    camera,
                    mujoco.mjtCatBit.mjCAT_ALL,
                    scene,
                )
                viewport = mujoco.MjrRect(x, y, tile_width, tile_height)
                mujoco.mjr_render(viewport, scene, context)
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    viewport,
                    f"env {env_id}",
                    environment.trajectories[env_id].identity.identity,
                    context,
                )
            glfw.swap_buffers(window)
            glfw.poll_events()
            call = int(environment.progress[0] - 1)
            if call % print_every == 0 or bool(np.any(resets[:tile_envs])):
                print(
                    " | ".join(
                        _telemetry(environment, env_id, float(rewards[env_id]), bool(resets[env_id]))
                        for env_id in range(tile_envs)
                    ),
                    flush=True,
                )
            if bool(np.any(resets[:tile_envs])) and not loop:
                return
            time.sleep(max(0.0, sleep_seconds - (time.perf_counter() - started)))
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


def _view_single(
    environment: MujocoManoEnvironment,
    *,
    stepper: ViewerStepper,
    render_env: int,
    speed: float,
    loop: bool,
    print_every: int,
    recorder: ManoRerunRecorder | None,
) -> None:
    """Render one world while advancing the same action boundary as tiled mode."""

    import mujoco
    import mujoco.viewer

    render_data = environment.host_data(render_env)
    sleep_seconds = CONTROL_TIMESTEP / speed
    with mujoco.viewer.launch_passive(
        environment.model, render_data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        viewer.sync()
        while viewer.is_running():
            started = time.perf_counter()
            _, rewards, resets, _ = stepper.step()
            if recorder is not None:
                recorder.record_transition()
            mujoco.mj_copyData(render_data, environment.model, environment.host_data(render_env))
            viewer.sync()

            call = int(environment.progress[render_env] - 1)
            if call % print_every == 0 or bool(resets[render_env]):
                print(
                    _telemetry(environment, render_env, float(rewards[render_env]), bool(resets[render_env])),
                    flush=True,
                )
            if bool(resets[render_env]) and not loop:
                return
            time.sleep(max(0.0, sleep_seconds - (time.perf_counter() - started)))


def _close_rerun_recorder(recorder: ManoRerunRecorder) -> Path | None:
    artifact = recorder.close()
    if artifact is None:
        print("No reset-complete Rerun episode was published.", flush=True)
    else:
        print(f"Rerun artifact: {artifact}", flush=True)
    return artifact


def view_environment(
    *,
    device: str,
    speed: float,
    loop: bool,
    print_every: int,
    terminal: bool,
    trajectory_name: str,
    num_envs: int,
    render_env: int,
    tile_envs: int,
    object_type: str | None,
    gesture: str | None,
    rerun_output: Path | None,
    use_residual: bool,
    checkpoint: Path | None,
) -> None:
    """Run a batched production environment and render its first world."""

    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if print_every < 1:
        raise ValueError("print_every must be positive")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if not 0 <= render_env < num_envs:
        raise ValueError("render_env must be within the configured batch")
    if not 1 <= tile_envs <= num_envs:
        raise ValueError("tile_envs must be within the configured batch")
    if checkpoint is not None and not use_residual:
        raise ValueError("--checkpoint requires --use_residual true so policy actions reach the controller")
    checkpoint = None if checkpoint is None else _validate_checkpoint_path(checkpoint)
    _require_graphical_session()

    if (object_type is None) != (gesture is None):
        raise ValueError("--object and --gesture must be supplied together")
    if object_type is not None:
        trajectory = load_assigned_trajectory_batch(
            TrajectorySelection(object_type=object_type, gesture=gesture), num_envs=num_envs
        )
        trajectory_label = f"object={object_type}, gesture={gesture.zfill(2)}"
    elif trajectory_name == "accepted":
        trajectory = load_reference_trajectory()
        trajectory_label = trajectory_name
    elif trajectory_name == "generated-cube1-row-507":
        trajectory = load_generated_cube1_row_507()
        trajectory_label = trajectory_name
    elif trajectory_name == "accepted-cube1-action-01-batch10":
        trajectory = load_cube1_action_01_batch10()
        trajectory_label = trajectory_name
    else:
        raise ValueError(f"unsupported viewer trajectory {trajectory_name!r}")
    if isinstance(trajectory, TrajectoryBatch) and trajectory.num_envs != num_envs:
        raise ValueError(
            f"{trajectory_name} requires --num-envs {trajectory.num_envs}, got {num_envs}"
        )
    max_deviation_distance = 0.1 if terminal else 1_000_000.0
    contact_capacity = max(128, 31 * num_envs + 64)
    environment = MujocoManoEnvironment(
        trajectory,
        EnvironmentConfig(
            device=device,
            num_envs=num_envs,
            residual_enabled=use_residual,
            max_deviation_distance=max_deviation_distance,
            contact_capacity=contact_capacity,
        ),
    )
    recorder = None if rerun_output is None else ManoRerunRecorder(environment, rerun_output, env_id=0)
    stepper: ViewerStepper
    if checkpoint is None:
        stepper = _ZeroActionStepper(environment)
        action_source = "zero 26D action"
    else:
        stepper = _build_checkpoint_stepper(environment, checkpoint)
        action_source = f"checkpoint deterministic policy ({checkpoint})"

    print(
        f"Testing MujocoManoEnvironment with use_residual={use_residual}, terminal={terminal}, and {action_source} "
        f"(trajectory={trajectory_label}, envs={num_envs}, "
        f"maxDeviationDistance={max_deviation_distance:g}). "
        f"The viewer renders env {render_env}; all configured environments execute the same batched path."
    )
    if tile_envs > 1:
        try:
            _view_tiled(
                environment,
                stepper=stepper,
                tile_envs=tile_envs,
                speed=speed,
                loop=loop,
                print_every=print_every,
                recorder=recorder,
            )
        finally:
            if recorder is not None:
                _close_rerun_recorder(recorder)
        return
    try:
        _view_single(
            environment,
            stepper=stepper,
            render_env=render_env,
            speed=speed,
            loop=loop,
            print_every=print_every,
            recorder=recorder,
        )
    finally:
        if recorder is not None:
            _close_rerun_recorder(recorder)


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
        choices=("accepted", "generated-cube1-row-507", "accepted-cube1-action-01-batch10"),
        default="accepted",
        help="explicit versioned Lance trajectory contract to render",
    )
    parser.add_argument("--object", dest="object_type", help="Lance object selector; requires --gesture")
    parser.add_argument("--gesture", help="Lance two-digit action selector; requires --object")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="native ManoRL skrl checkpoint; runs deterministic mean policy actions",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="batched environments to execute; viewer renders env 0",
    )
    parser.add_argument("--render-env", type=int, default=0, help="batched env index mirrored into the viewer")
    parser.add_argument(
        "--tile-envs",
        type=int,
        default=1,
        help="render the first N batched worlds as tiles in one MuJoCo GLFW window",
    )
    parser.add_argument(
        "--use_residual",
        type=parse_cli_bool,
        default=True,
        metavar="{true,false}",
        help="enable 26D residual action processing (default: true)",
    )
    parser.add_argument(
        "--terminal",
        type=parse_cli_bool,
        default=True,
        metavar="{true,false}",
        help="use the 0.1 m training deviation threshold (default: true)",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue through the environment's source-compatible delayed reset after terminal.",
    )
    parser.add_argument(
        "--rerun-output",
        type=Path,
        help="optionally record latest env-0 Rerun data while the MuJoCo viewer runs",
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
        terminal=args.terminal,
        trajectory_name=args.trajectory,
        num_envs=args.num_envs,
        render_env=args.render_env,
        tile_envs=args.tile_envs,
        object_type=args.object_type,
        gesture=args.gesture,
        rerun_output=args.rerun_output,
        use_residual=args.use_residual,
        checkpoint=args.checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
