"""Interactively test the actual cube1 MJX-Warp environment.

This uses :class:`MujocoManoEnvironment`, not the narrow reference-replay
helper. The default viewer mirrors one MJX state into native ``MjData``;
the tiled mode renders several independently batched states in one window.
Observations, rewards, termination, indexed episode reset, and action processing
all remain on the production environment path.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
import json
import math
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from sim.manorl.abi import ResidualActionConfig, TARGET_MAX_DEVIATION_DISTANCE
from sim.manorl.approach_prefix import (
    ApproachPrefixConfig,
    augment_trajectory_with_approach_prefix,
    augment_trajectory_with_retreat_suffix,
    augmentation_stream_seed,
)
from sim.manorl.assets import (
    COLLISION_GEOM_GROUP,
    compile_model,
    compile_unified_model,
)
from sim.manorl.cli import parse_cli_bool
from sim.manorl.contracts import JOINT_DOF
from sim.manorl.environment import (
    EnvironmentConfig,
    MujocoManoEnvironment,
    recommended_warp_contact_capacity,
)
from sim.manorl.observations import SOURCE_ALIGNED_COMPATIBILITY
from sim.manorl.rerun_recorder import ManoRerunRecorder
from sim.manorl.synthetic_parent import (
    AcceptedSyntheticParent,
    load_accepted_synthetic_parent,
)
from sim.manorl.trajectory import (
    DEFAULT_POST_PADDING,
    DEFAULT_PRE_PADDING,
    DEFAULT_REFERENCE_FPS,
    SUPPORTED_REFERENCE_FPS,
    ReferenceTrajectory,
    TrajectoryBatch,
    TrajectorySelection,
    load_assigned_trajectory_batch,
    load_cube1_action_01_batch10,
    load_generated_cube1_row_507,
    load_reference_trajectory,
    parse_trajectory_selector,
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
        action_dim = int(getattr(environment, "action_dim", JOINT_DOF))
        self._actions = np.zeros((environment.config.num_envs, action_dim), dtype=np.float64)
        self._pending_done = np.zeros(environment.config.num_envs, dtype=bool)

    def step(self) -> tuple[Any, np.ndarray, np.ndarray, Any]:
        if np.any(self._pending_done):
            self._environment.reset(
                env_ids=np.flatnonzero(self._pending_done).astype(np.int64)
            )
            self._pending_done[:] = False
        transition = self._environment.step(self._actions)
        self._pending_done = np.asarray(transition[2], dtype=bool).copy()
        return transition


def _reset_runtime_done(runtime: Any, observations: Any, done: Any) -> Any:
    """Reset terminal rows through a runtime or its wrapped vector env."""

    if not bool(done.any()):
        return observations
    runtime_reset_done = getattr(runtime, "reset_done", None)
    if callable(runtime_reset_done):
        return runtime_reset_done(observations, done)
    mask = done.reshape(-1)
    row_ids = mask.nonzero(as_tuple=False).reshape(-1)
    env_ids = row_ids.detach().cpu().numpy().astype(np.int64)
    try:
        reset_observations, _ = runtime.env.reset(options={"env_ids": env_ids})
    except TypeError:
        reset_observations, _ = runtime.env.reset()
    if not hasattr(reset_observations, "to"):
        reset_observations = observations.new_tensor(reset_observations)
    updated = observations.clone()
    device_row_ids = row_ids.to(updated.device)
    updated[device_row_ids] = reset_observations.to(updated.device)[device_row_ids]
    return updated


class _CheckpointPolicyStepper:
    """Advance with deterministic PPO means after optional reference-only phases.

    ``policy_enable_steps`` opens policy inference at the first live step of
    each world (approach-prefix gate); ``policy_disable_steps`` closes it at
    the movement-end+15 retreat anchor so the deformed tail never calls policy;
    its entry residual is discharged without new action accumulation.
    """

    def __init__(
        self,
        runtime: ManoSkrlRuntime,
        observations: Any,
        *,
        policy_enable_steps: np.ndarray | None = None,
        policy_disable_steps: np.ndarray | None = None,
    ) -> None:
        self._runtime = runtime
        self._observations = observations
        self._pending_done = None
        self._policy_enable_steps = (
            None
            if policy_enable_steps is None
            else np.asarray(policy_enable_steps, dtype=np.int64)
        )
        self._policy_disable_steps = (
            None
            if policy_disable_steps is None
            else np.asarray(policy_disable_steps, dtype=np.int64)
        )
        if self._policy_enable_steps is not None or self._policy_disable_steps is not None:
            environment = runtime.gymnasium_env.environment
            expected = (environment.config.num_envs,)
            if self._policy_enable_steps is not None and self._policy_enable_steps.shape != expected:
                raise ValueError(
                    "policy_enable_steps must contain one step per environment"
                )
            if self._policy_disable_steps is not None and self._policy_disable_steps.shape != expected:
                raise ValueError(
                    "policy_disable_steps must contain one step per environment"
                )
            if (
                self._policy_enable_steps is not None
                and self._policy_disable_steps is not None
                and np.any(self._policy_disable_steps < self._policy_enable_steps)
            ):
                raise ValueError(
                    "policy_disable_steps must not precede policy_enable_steps"
                )

    def step(self) -> tuple[Any, np.ndarray, np.ndarray, Any]:
        if self._pending_done is not None and bool(self._pending_done.any()):
            self._observations = _reset_runtime_done(
                self._runtime,
                self._observations,
                self._pending_done,
            )
            self._pending_done = None
        if self._policy_enable_steps is None and self._policy_disable_steps is None:
            actions = self._runtime.deterministic_actions(self._observations)
        else:
            physical = self._runtime.gymnasium_env.environment
            steps = np.asarray(physical.trajectory_steps, dtype=np.int64)
            enabled = np.ones(physical.config.num_envs, dtype=bool)
            if self._policy_enable_steps is not None:
                enabled &= steps >= self._policy_enable_steps
            if self._policy_disable_steps is not None:
                enabled &= steps < self._policy_disable_steps
            action_dim = int(getattr(physical, "action_dim", JOINT_DOF))
            if np.all(enabled):
                actions = self._runtime.deterministic_actions(self._observations)
            elif not np.any(enabled):
                actions = self._observations.new_zeros(
                    (physical.config.num_envs, action_dim)
                )
            else:
                actions = self._observations.new_zeros(
                    (physical.config.num_envs, action_dim)
                )
                enabled_rows = np.flatnonzero(enabled)
                policy_actions = self._runtime.deterministic_actions(
                    self._observations[enabled_rows]
                )
                actions[enabled_rows] = policy_actions
        observations, rewards, terminated, truncated, info = self._runtime.env.step(actions)
        self._observations = observations
        rewards_array = rewards.detach().cpu().numpy().reshape(-1)
        self._pending_done = terminated | truncated
        resets = self._pending_done.detach().cpu().numpy().reshape(-1)
        return observations, rewards_array, resets, info


def _inference_ppo_config(num_envs: int, *, use_film: bool = True) -> ManoPPOConfig:
    """Create a non-training PPO shell valid for any positive vector batch size."""

    from sim.manorl.skrl_runtime import ManoPPOConfig

    return ManoPPOConfig(
        rollouts=1,
        minibatch_size=num_envs,
        learning_epochs=1,
        use_film=use_film,
    )


@dataclass(frozen=True)
class _CheckpointEnvironmentOptions:
    residual_action: ResidualActionConfig = ResidualActionConfig()
    reference_fps: int | None = None
    control_fps: int = DEFAULT_REFERENCE_FPS
    pre_padding: int = DEFAULT_PRE_PADDING
    post_padding: int = DEFAULT_POST_PADDING
    warp_ccd_iterations: int | None = None
    warp_ccd_contacts_per_world: int | None = None


def _resolve_reference_fps(
    requested: int | None,
    *,
    checkpoint_options: _CheckpointEnvironmentOptions,
    has_checkpoint: bool,
) -> int | None:
    """Resolve a requested source clock without silently changing a checkpoint ABI."""

    if requested is not None and (
        not isinstance(requested, int)
        or isinstance(requested, bool)
        or requested not in SUPPORTED_REFERENCE_FPS
    ):
        raise ValueError(f"reference_fps must be one of {SUPPORTED_REFERENCE_FPS}")
    if not has_checkpoint:
        return DEFAULT_REFERENCE_FPS if requested is None else requested
    checkpoint_fps = checkpoint_options.reference_fps
    if checkpoint_fps is None:
        if requested is not None:
            raise ValueError(
                "checkpoint predates the reference_fps contract; an explicit 100/120 Hz "
                "override would change its training clock"
            )
        return None
    if requested is not None and requested != checkpoint_fps:
        raise ValueError(
            f"reference_fps {requested} conflicts with checkpoint reference_fps "
            f"{checkpoint_fps}"
        )
    return checkpoint_fps


def _resolve_padding(
    requested: int | None,
    *,
    checkpoint_value: int,
    has_checkpoint: bool,
    name: str,
) -> int:
    """Resolve an explicit padding override without changing checkpoint semantics."""

    if requested is not None and (
        not isinstance(requested, int)
        or isinstance(requested, bool)
        or requested < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    if requested is None:
        return checkpoint_value
    if has_checkpoint and requested != checkpoint_value:
        raise ValueError(
            f"{name} {requested} conflicts with checkpoint {name} {checkpoint_value}"
        )
    return requested


def _checkpoint_environment_options(checkpoint: Path) -> _CheckpointEnvironmentOptions:
    """Restore trajectory, clocks, action, and per-world CCD semantics."""

    from sim.manorl.checkpoint import (
        CheckpointFormatError,
        checkpoint_post_padding,
        checkpoint_pre_padding,
        checkpoint_runtime_metadata,
        checkpoint_simulation_clock,
    )

    metadata = checkpoint_runtime_metadata(checkpoint)
    checkpoint_clock = checkpoint_simulation_clock(metadata)
    runtime_config = metadata.get("runtime_config")
    environment = (
        runtime_config.get("environment")
        if isinstance(runtime_config, dict)
        else None
    )
    if not isinstance(environment, dict):
        return _CheckpointEnvironmentOptions(
            control_fps=checkpoint_clock.policy_fps,
            pre_padding=checkpoint_pre_padding(metadata),
            post_padding=checkpoint_post_padding(metadata),
        )
    pre_padding = checkpoint_pre_padding(metadata)
    post_padding = checkpoint_post_padding(metadata)

    residual_values = environment.get("residual_action")
    if residual_values is None:
        residual_action = ResidualActionConfig()
    elif isinstance(residual_values, dict):
        allowed = {item.name for item in fields(ResidualActionConfig)}
        unknown = sorted(set(residual_values) - allowed)
        if unknown:
            raise CheckpointFormatError(
                f"checkpoint residual_action has unsupported fields: {unknown}"
            )
        values = dict(residual_values)
        for name in (
            "position_scale",
            "max_position_offset",
            "joint_scale",
            "max_joint_offset",
        ):
            if name in values:
                values[name] = tuple(values[name])
        try:
            residual_action = ResidualActionConfig(**values)
        except (TypeError, ValueError) as exc:
            raise CheckpointFormatError(
                f"checkpoint residual_action is invalid: {exc}"
            ) from exc
    else:
        raise CheckpointFormatError("checkpoint residual_action must be a mapping")

    reference_fps = environment.get("reference_fps")
    if reference_fps is not None and (
        not isinstance(reference_fps, int)
        or isinstance(reference_fps, bool)
        or reference_fps not in SUPPORTED_REFERENCE_FPS
    ):
        raise CheckpointFormatError(
            f"checkpoint reference_fps must be one of {SUPPORTED_REFERENCE_FPS}"
        )
    warp_ccd = environment.get("warp_ccd")
    if warp_ccd is None:
        warp_ccd = {}
    if not isinstance(warp_ccd, dict):
        raise CheckpointFormatError("checkpoint warp_ccd must be a mapping")
    return _CheckpointEnvironmentOptions(
        residual_action=residual_action,
        reference_fps=reference_fps,
        control_fps=checkpoint_clock.policy_fps,
        pre_padding=pre_padding,
        post_padding=post_padding,
        warp_ccd_iterations=warp_ccd.get("ccd_iterations"),
        warp_ccd_contacts_per_world=warp_ccd.get("contacts_per_world"),
    )


def _checkpoint_use_film(checkpoint: Path) -> bool:
    """Resolve the model variant recorded by a checkpoint, defaulting to FiLM."""

    from sim.manorl.checkpoint import CheckpointFormatError, checkpoint_runtime_metadata

    metadata = checkpoint_runtime_metadata(checkpoint)
    runtime_config = metadata.get("runtime_config")
    if not isinstance(runtime_config, dict):
        return True
    model = runtime_config.get("model")
    if not isinstance(model, dict) or "use_film" not in model:
        return True
    use_film = model["use_film"]
    if not isinstance(use_film, bool):
        raise CheckpointFormatError("checkpoint model use_film must be a boolean")
    return use_film


def _validate_checkpoint_path(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if not sidecar.is_file():
        raise ValueError(f"native ManoRL checkpoint sidecar is required: {sidecar}")
    return checkpoint


def _build_checkpoint_stepper(
    environment: MujocoManoEnvironment,
    checkpoint: Path,
    *,
    policy_transfer: bool = False,
    policy_enable_steps: np.ndarray | None = None,
    policy_disable_steps: np.ndarray | None = None,
) -> ViewerStepper:
    """Load the native policy and reset through its vector wrapper before rendering.

    ``policy_transfer`` is reserved for synthesis references whose padding or
    assets intentionally differ from the checkpoint. Ordinary viewer inference
    remains strict by default.
    """

    from sim.manorl.checkpoint import (
        load_skrl_checkpoint_for_inference,
        load_skrl_checkpoint_for_policy_transfer_inference,
    )
    from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv
    from sim.manorl.skrl_runtime import ManoSkrlRuntime

    adapter = ManoGymnasiumVectorEnv(environment)
    runtime = ManoSkrlRuntime(
        adapter,
        _inference_ppo_config(
            environment.config.num_envs,
            use_film=_checkpoint_use_film(checkpoint),
        ),
    )
    loader = (
        load_skrl_checkpoint_for_policy_transfer_inference
        if policy_transfer
        else load_skrl_checkpoint_for_inference
    )
    loader(runtime.agent, checkpoint)
    runtime.agent.enable_training_mode(False)
    runtime.model.eval()
    observations, _ = runtime.env.reset()
    return _CheckpointPolicyStepper(
        runtime,
        observations,
        policy_enable_steps=policy_enable_steps,
        policy_disable_steps=policy_disable_steps,
    )


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
    termination = environment.last_termination
    reason = 0 if termination is None else int(termination.reason_code[env_id])
    return (
        f"env={env_id} trajectory={environment.trajectories[env_id].identity.identity} "
        f"call={call:03d}/{environment.trajectory_lengths[env_id] - 2} command_ref={command_index:03d} "
        f"post_ref={post_index:03d} source_ref={environment.reference_source_indices[env_id, post_index]:04d} "
        f"reward={reward:.4f} reset={reset} termination_reason={reason} ctrl={command}"
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


def _model_names(
    mujoco: Any, model: Any, kind: Any, count: int
) -> tuple[str, ...]:
    return tuple(mujoco.mj_id2name(model, kind, index) or "" for index in range(count))


def _validate_native_viewer_abi(
    mujoco: Any, physics_model: Any, viewer_model: Any
) -> None:
    """Require identical dynamic state topology before mirroring into a visual model."""

    count_fields = ("nq", "nv", "nu", "na", "nbody", "njnt", "nmocap")
    mismatches = [
        f"{name}={getattr(physics_model, name)}/{getattr(viewer_model, name)}"
        for name in count_fields
        if getattr(physics_model, name) != getattr(viewer_model, name)
    ]
    name_specs = (
        (mujoco.mjtObj.mjOBJ_BODY, "nbody"),
        (mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"),
    )
    for kind, count_field in name_specs:
        count = getattr(physics_model, count_field)
        if _model_names(mujoco, physics_model, kind, count) != _model_names(
            mujoco, viewer_model, kind, count
        ):
            mismatches.append(f"{count_field} names differ")
    topology_fields = (
        "body_parentid",
        "jnt_bodyid",
        "jnt_qposadr",
        "jnt_dofadr",
        "actuator_trnid",
    )
    for name in topology_fields:
        if not np.array_equal(
            getattr(physics_model, name), getattr(viewer_model, name)
        ):
            mismatches.append(f"{name} differs")
    if mismatches:
        raise ValueError(
            "native viewer model changed the physics-state ABI: " + ", ".join(mismatches)
        )


def _compile_native_viewer_model(
    environment: MujocoManoEnvironment,
) -> tuple[Any, Any]:
    """Compile a visual-only native model without changing the MJX physics model."""

    if environment.is_heterogeneous:
        raise ValueError(
            "native visual viewing requires a homogeneous or unified object model"
        )
    # The viewer must mirror the exact hand topology used by the physics
    # model.  In particular, a left-only or bimanual environment cannot fall
    # back to the historical right-hand scene just because its visuals are
    # native-only.
    hand_side = getattr(environment, "model_hand_side", "right")
    if getattr(environment, "_unified_object_batch", False):
        mujoco, model = compile_unified_model(
            environment.config.servo,
            object_types=environment._unified_object_types,
            visual_meshes=True,
            hand_side=hand_side,
            physics_timestep=environment.config.physics_timestep,
        )
    else:
        mujoco, model = compile_model(
            environment.config.servo,
            object_type=environment.object_type,
            visual_meshes=True,
            hand_side=hand_side,
            physics_timestep=environment.config.physics_timestep,
        )
    _validate_native_viewer_abi(mujoco, environment.model, model)
    return mujoco, model


def _mirror_native_viewer_data(
    mujoco: Any,
    viewer_model: Any,
    viewer_data: Any,
    physics_data: Any,
) -> None:
    """Copy dynamic coordinates needed for rendering, then derive native transforms."""

    viewer_data.time = physics_data.time
    for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
        destination = getattr(viewer_data, name)
        source = getattr(physics_data, name)
        if destination.shape != source.shape:
            raise ValueError(
                f"native viewer data field {name} changed shape: "
                f"{source.shape} -> {destination.shape}"
            )
        destination[:] = source
    mujoco.mj_forward(viewer_model, viewer_data)


def _viewer_lock(viewer: object) -> Any:
    """Return the passive viewer lock, with a no-op fallback for test doubles."""

    lock = getattr(viewer, "lock", None)
    return lock() if callable(lock) else nullcontext()


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
        mujoco, viewer_model = _compile_native_viewer_model(environment)

        self._environment, self._tile_envs, self._stride = environment, tile_envs, stride
        self._viewer_model = viewer_model
        self._glfw, self._mujoco, self._frames, self.close_requested = glfw, mujoco, 0, False
        self._viewer_data = [mujoco.MjData(viewer_model) for _ in range(tile_envs)]
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
            _configure_tiled_visuals(viewer_model)
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._camera)
            self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            _reset_tiled_camera(self._camera)
            self._option = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self._option)
            self._option.geomgroup[COLLISION_GEOM_GROUP] = 0
            self._scene = mujoco.MjvScene(viewer_model, maxgeom=10_000)
            self._context = mujoco.MjrContext(viewer_model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self._viewports = _tile_layout(tile_envs, width=self._width, height=self._height)
            _install_tiled_controls(
                glfw=glfw, mujoco=mujoco, window=self._window, model=viewer_model,
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
            _mirror_native_viewer_data(
                mujoco,
                self._viewer_model,
                self._viewer_data[env_id],
                host_data[env_id],
            )
            mujoco.mjv_updateScene(
                self._viewer_model,
                self._viewer_data[env_id],
                self._option,
                None,
                self._camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                self._scene,
            )
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
    mujoco, viewer_model = _compile_native_viewer_model(environment)

    width, height = 1600, 900
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed for tiled MuJoCo rendering")
    window = glfw.create_window(width, height, f"ManoRL batch viewer ({tile_envs} envs)", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("could not create GLFW window for tiled MuJoCo rendering")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    _configure_tiled_visuals(viewer_model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    _reset_tiled_camera(camera)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[COLLISION_GEOM_GROUP] = 0
    scene = mujoco.MjvScene(viewer_model, maxgeom=10_000)
    context = mujoco.MjrContext(viewer_model, mujoco.mjtFontScale.mjFONTSCALE_150)
    viewer_data = [mujoco.MjData(viewer_model) for _ in range(tile_envs)]
    sleep_seconds = environment.config.control_timestep / speed
    viewports = _tile_layout(tile_envs, width=width, height=height)
    _install_tiled_controls(
        glfw=glfw,
        mujoco=mujoco,
        window=window,
        model=viewer_model,
        camera=camera,
        scene=scene,
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
                _mirror_native_viewer_data(
                    mujoco,
                    viewer_model,
                    viewer_data[env_id],
                    host_data[env_id],
                )
                mujoco.mjv_updateScene(
                    viewer_model,
                    viewer_data[env_id],
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

    from mujoco import viewer as mujoco_viewer

    mujoco, viewer_model = _compile_native_viewer_model(environment)
    render_data = mujoco.MjData(viewer_model)
    # This write occurs before launch_passive starts its UI thread, so no viewer
    # lock is needed yet. All subsequent writes to viewer-owned data are locked.
    _mirror_native_viewer_data(
        mujoco,
        viewer_model,
        render_data,
        environment.host_data(render_env),
    )
    sleep_seconds = environment.config.control_timestep / speed
    with mujoco_viewer.launch_passive(
        viewer_model, render_data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        with _viewer_lock(viewer):
            viewer.opt.geomgroup[COLLISION_GEOM_GROUP] = 0
            viewer.sync()
        while viewer.is_running():
            started = time.perf_counter()
            _, rewards, resets, _ = stepper.step()
            if recorder is not None:
                recorder.record_transition()
            physics_data = environment.host_data(render_env)
            with _viewer_lock(viewer):
                _mirror_native_viewer_data(
                    mujoco,
                    viewer_model,
                    render_data,
                    physics_data,
                )
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
        print("No terminal-complete Rerun episode was published.", flush=True)
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
    pairs: str | None = None,
    rerun_output: Path | None,
    use_residual: bool,
    checkpoint: Path | None,
    dataset_path: Path | None = None,
    dataset_version: int | None = None,
    reference_fps: int | None = None,
    pre_padding: int | None = None,
    post_padding: int | None = None,
    hand_side: str = "auto",
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
    checkpoint_options = (
        _CheckpointEnvironmentOptions()
        if checkpoint is None
        else _checkpoint_environment_options(checkpoint)
    )
    resolved_pre_padding = _resolve_padding(
        pre_padding,
        checkpoint_value=checkpoint_options.pre_padding,
        has_checkpoint=checkpoint is not None,
        name="pre_padding",
    )
    resolved_post_padding = _resolve_padding(
        post_padding,
        checkpoint_value=checkpoint_options.post_padding,
        has_checkpoint=checkpoint is not None,
        name="post_padding",
    )
    if device == "gpu":
        import torch

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    if (object_type is None) != (gesture is None):
        raise ValueError("--object and --gesture must be supplied together")
    if pairs is not None and (object_type is not None or gesture is not None):
        raise ValueError("--pairs is mutually exclusive with --object/--gesture")
    uses_dataset_selection = (
        dataset_path is not None or object_type is not None or pairs is not None
    )
    resolved_reference_fps = _resolve_reference_fps(
        reference_fps,
        checkpoint_options=checkpoint_options,
        has_checkpoint=checkpoint is not None,
    )
    resolved_control_fps = checkpoint_options.control_fps
    if checkpoint is None and uses_dataset_selection:
        # Dataset-only viewing has no checkpoint clock to preserve. Keep the
        # public control clock aligned with an explicit 100/120 Hz reference
        # selection instead of retaining the default 120 Hz checkpoint shell.
        resolved_control_fps = resolved_reference_fps
    elif not uses_dataset_selection and reference_fps is None and checkpoint is None:
        resolved_reference_fps = None
        resolved_control_fps = 200
    if resolved_reference_fps is not None and not uses_dataset_selection:
        raise ValueError(
            "reference_fps requires a dataset trajectory selected with --object/--gesture "
            "or --dataset-path"
        )
    _require_graphical_session()
    if uses_dataset_selection:
        if pairs is not None:
            selected_pairs = list(parse_trajectory_selector(pairs))
            if not selected_pairs:
                raise ValueError("--pairs must name at least one object:action pair")
            trajectory = load_assigned_trajectory_batch(
                TrajectorySelection(
                    selector=",".join(pair.canonical for pair in selected_pairs),
                    dataset_path=(
                        dataset_path
                        if dataset_path is not None
                        else TrajectorySelection().dataset_path
                    ),
                    expected_dataset_version=dataset_version,
                    pre_padding=resolved_pre_padding,
                    post_padding=resolved_post_padding,
                    hand_side=hand_side,
                    reference_fps=resolved_reference_fps,
                    control_fps=resolved_control_fps,
                ),
                num_envs=num_envs,
            )
            trajectory_label = "pairs=" + ",".join(pair.canonical for pair in selected_pairs)
        else:
            selected_object = object_type or "banana"
            selected_gesture = gesture or "01"
            trajectory = load_assigned_trajectory_batch(
                TrajectorySelection(
                    object_type=selected_object,
                    gesture=selected_gesture,
                    dataset_path=(dataset_path if dataset_path is not None else TrajectorySelection().dataset_path),
                    expected_dataset_version=dataset_version,
                    pre_padding=resolved_pre_padding,
                    post_padding=resolved_post_padding,
                    hand_side=hand_side,
                    reference_fps=resolved_reference_fps,
                    control_fps=resolved_control_fps,
                ),
                num_envs=num_envs,
            )
            trajectory_label = f"object={selected_object}, gesture={selected_gesture}, hand_side={hand_side}"
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
    max_deviation_distance = TARGET_MAX_DEVIATION_DISTANCE if terminal else 1_000_000.0
    contact_capacity = recommended_warp_contact_capacity(
        num_envs, getattr(trajectory, "hand_sides", ("right",))
    )
    environment = MujocoManoEnvironment(
        trajectory,
        EnvironmentConfig(
            device=device,
            num_envs=num_envs,
            residual_enabled=use_residual,
            residual_action=checkpoint_options.residual_action,
            compatibility=replace(
                SOURCE_ALIGNED_COMPATIBILITY,
                movement_pre_padding=resolved_pre_padding,
            ),
            max_deviation_distance=max_deviation_distance,
            contact_capacity=contact_capacity,
            reference_fps=resolved_reference_fps,
            control_fps=resolved_control_fps,
            post_padding=resolved_post_padding,
            warp_ccd_iterations=checkpoint_options.warp_ccd_iterations,
            warp_ccd_contacts_per_world=checkpoint_options.warp_ccd_contacts_per_world,
            hand_side=hand_side,
        ),
    )
    recorder = None if rerun_output is None else ManoRerunRecorder(environment, rerun_output, env_id=0)
    stepper: ViewerStepper
    if checkpoint is None:
        stepper = _ZeroActionStepper(environment)
        action_source = f"zero {getattr(environment, 'action_dim', JOINT_DOF)}D action"
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


def _reinstall_approach_prefix_reference(
    environment: MujocoManoEnvironment,
    stepper: ViewerStepper,
    trajectory: ReferenceTrajectory,
    *,
    seed: int,
    policy_disable_steps: np.ndarray | None = None,
) -> None:
    """Replace the single world's reference and reset the runtime in place.

    The MJX-Warp physics model, Warp buffers, and checkpoint policy runtime
    stay alive across resets; only the host reference tables, the reset qpos,
    and the seeded observation change.  This is what makes the accepted-parent
    viewer a true one-environment loop instead of rebuilding GPU runtimes per
    seed (which previously exhausted Warp's allocator).
    """

    if environment.config.num_envs != 1 or environment.is_heterogeneous:
        raise RuntimeError(
            "approach-prefix reference reset requires one homogeneous world"
        )
    if trajectory.identity.identity != environment.trajectory.identity.identity:
        raise RuntimeError("reset trajectory changed the accepted source identity")
    if trajectory.hand_sides != environment.trajectory.hand_sides:
        raise RuntimeError("reset trajectory changed hand topology")
    if trajectory.reference_fps != environment.config.reference_fps:
        raise RuntimeError("reset trajectory changed reference clock")

    environment.trajectory = trajectory
    environment.trajectories = (trajectory,)
    environment._build_reference_tables()
    environment._reset_qpos = environment._initial_qpos()
    runtime = getattr(stepper, "_runtime", None)
    if runtime is None:
        raise RuntimeError("approach-prefix stepper must expose its skrl runtime")
    observations, _ = runtime.env.reset(seed=seed)
    stepper._observations = observations
    stepper._pending_done = None
    stepper._policy_enable_steps = environment.early_phase_lengths.copy()
    resolved_disable_steps = policy_disable_steps
    if resolved_disable_steps is None and getattr(
        stepper, "_policy_disable_steps", None
    ) is not None:
        resolved_disable_steps = (
            environment.trajectory_lengths
            - environment.augmentation_suffix_frames
        )
    if resolved_disable_steps is not None:
        stepper._policy_disable_steps = np.asarray(
            resolved_disable_steps, dtype=np.int64
        ).copy()


def view_approach_prefix_episodes(
    *,
    checkpoint: Path,
    accepted_parent: Path | AcceptedSyntheticParent,
    predecode_dir: Path,
    start_seed: int = 49,
    speed: float = 1.0,
    print_every: int = 20,
    max_episodes: int | None = None,
    retreat_suffix: bool = False,
    approach_mode: str = "far",
) -> None:
    """View accepted-parent approach-prefix augmentation episodes in one window.

    One active MJX-Warp environment and one checkpoint policy runtime are
    reused across every reset.  Each terminal advances ``start_seed`` and
    reinstalls the next seeded approach-prefixed reference derived from the
    same accepted parent, so every episode starts from its own frame 0 with a
    freshly sampled hand approach while the original pre60 reference and the
    deterministic checkpoint policy continue unchanged.
    """

    if approach_mode not in ("far", "near"):
        raise ValueError("approach_mode must be 'far' or 'near'")
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if print_every < 1:
        raise ValueError("print_every must be positive")
    if not isinstance(start_seed, int) or isinstance(start_seed, bool) or start_seed < 0:
        raise ValueError("start_seed must be a non-negative integer")
    if max_episodes is not None and (
        not isinstance(max_episodes, int)
        or isinstance(max_episodes, bool)
        or max_episodes < 1
    ):
        raise ValueError("max_episodes must be a positive integer or None")
    _require_graphical_session()

    parent = (
        load_accepted_synthetic_parent(accepted_parent)
        if isinstance(accepted_parent, Path)
        else accepted_parent
    )
    predecode_dir = Path(predecode_dir)
    manifest_path = predecode_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        str(record.get("identity")): record
        for record in manifest.get("valid_records", [])
    }
    if parent.source_identity not in records:
        raise LookupError(
            f"predecoded manifest omits accepted identity {parent.source_identity}"
        )
    import pickle

    with (predecode_dir / f"{parent.source_identity}.pkl").open("rb") as stream:
        source = pickle.load(stream)
    if source.reference_fps != parent.reference_fps:
        raise ValueError("predecoded source clock differs from the accepted parent")
    if parent.source_row_frame0_right_q_ref_3_28 is None:
        raise ValueError(
            "accepted parent lacks source-row frame0 right q_ref[3:28]"
        )

    checkpoint = _validate_checkpoint_path(checkpoint)
    options = _checkpoint_environment_options(checkpoint)

    anchor_source = parent.retreat_anchor_source_frame_index
    if anchor_source is None:
        raise ValueError(
            "accepted parent lacks movement-end+15 retreat anchor provenance"
        )
    source_anchor_index = (
        int(source.movement_end_step) + parent.retreat_anchor_offset_frames
    )
    if (
        not 0 <= source_anchor_index < len(source.q_ref)
        or int(source.source_indices[source_anchor_index]) != int(anchor_source)
    ):
        raise ValueError(
            "base movement-end+offset anchor disagrees with accepted-parent source frame"
        )

    def sample(seed: int):
        approach_seed = (
            seed
            if approach_mode == "far"
            else augmentation_stream_seed(seed, "near-approach")
        )
        retreat_seed = seed
        trajectory, prefix_sample = augment_trajectory_with_approach_prefix(
            source,
            seed=approach_seed,
            config=ApproachPrefixConfig(mode=approach_mode),
            near_anchor_reference_index=(
                source_anchor_index if approach_mode == "near" else None
            ),
            start_q_ref_3_28=parent.source_row_frame0_right_q_ref_3_28,
        )
        suffix_sample = None
        if retreat_suffix:
            anchor_index = (
                int(trajectory.movement_end_step)
                + parent.retreat_anchor_offset_frames
            )
            if (
                not 0 <= anchor_index < len(trajectory.q_ref)
                or int(trajectory.source_indices[anchor_index]) != int(anchor_source)
            ):
                raise ValueError(
                    "augmented movement-end+offset anchor disagrees with accepted parent"
                )
            trajectory, suffix_sample = augment_trajectory_with_retreat_suffix(
                trajectory,
                seed=retreat_seed,
                anchor_reference_index=anchor_index,
            )
        return (
            trajectory,
            prefix_sample,
            suffix_sample,
            approach_seed,
            retreat_seed,
        )

    seed = start_seed
    (
        trajectory,
        prefix_sample,
        suffix_sample,
        approach_seed,
        retreat_seed,
    ) = sample(seed)
    batch = TrajectoryBatch((trajectory,))
    environment = MujocoManoEnvironment(
        batch,
        EnvironmentConfig(
            num_envs=1,
            device="gpu",
            residual_enabled=True,
            residual_action=options.residual_action,
            compatibility=replace(
                SOURCE_ALIGNED_COMPATIBILITY,
                movement_pre_padding=60,
            ),
            max_deviation_distance=0.10,
            contact_capacity=recommended_warp_contact_capacity(1, batch.hand_sides),
            reference_fps=120,
            control_fps=120,
            post_padding=250,
            warp_ccd_iterations=options.warp_ccd_iterations,
            warp_ccd_contacts_per_world=options.warp_ccd_contacts_per_world,
            hand_side="right",
            object_init_xy_offsets_m=(parent.object_init_xy_offset_m,),
        ),
    )
    stepper = _build_checkpoint_stepper(
        environment,
        checkpoint,
        policy_transfer=True,
        policy_enable_steps=environment.early_phase_lengths,
        policy_disable_steps=(
            environment.trajectory_lengths - environment.augmentation_suffix_frames
        ),
    )

    import mujoco
    from mujoco import viewer as mujoco_viewer

    mujoco_module, viewer_model = _compile_native_viewer_model(environment)
    viewer_data = mujoco_module.MjData(viewer_model)
    _mirror_native_viewer_data(
        mujoco_module, viewer_model, viewer_data, environment.host_data(0)
    )
    print("ACCEPTED_PARENT", json.dumps(parent.to_dict(), sort_keys=True), flush=True)
    print(
        "RUNTIME",
        json.dumps(
            {"active_envs": 1, "runtime_reused_across_resets": True},
            sort_keys=True,
        ),
        flush=True,
    )

    def print_reset(current_seed: int) -> None:
        reset_info: dict[str, object] = {
            "seed": current_seed,
            "approach_mode": approach_mode,
            "approach_seed": approach_seed,
            "retreat_seed": retreat_seed,
            "progress": int(environment.progress[0]),
            "trajectory_step": int(environment.trajectory_steps[0]),
            "start_xyz_m": [
                round(value, 6) for value in prefix_sample.start_position_m
            ],
            "xy_radius_m": round(prefix_sample.start_xy_radius_m, 6),
            "z_offset_m": round(prefix_sample.start_z_offset_m, 6),
            "distance_3d_m": round(prefix_sample.start_distance_m, 6),
            "xy_angle_deg": round(prefix_sample.xy_offset_deg, 3),
            "prefix_frames": prefix_sample.prefix_frames,
            "effective_pre": prefix_sample.effective_pre_padding,
            "object_xy_offset_m": list(parent.object_init_xy_offset_m),
        }
        if suffix_sample is not None:
            reset_info.update(
                {
                    "suffix_anchor_reference_index": (
                        suffix_sample.anchor_reference_index
                    ),
                    "suffix_frames": suffix_sample.suffix_frames,
                    "suffix_end_xyz_m": [
                        round(value, 6) for value in suffix_sample.end_position_m
                    ],
                    "suffix_original_horizontal_m": round(
                        suffix_sample.original_horizontal_distance_m, 6
                    ),
                    "suffix_extra_horizontal_m": round(
                        suffix_sample.extra_horizontal_offset_m, 6
                    ),
                    "suffix_end_horizontal_m": round(
                        suffix_sample.end_horizontal_distance_m, 6
                    ),
                    "suffix_original_dz_m": round(
                        suffix_sample.original_z_displacement_m, 6
                    ),
                    "suffix_extra_z_m": round(
                        suffix_sample.extra_z_offset_m, 6
                    ),
                    "suffix_end_dz_m": round(
                        suffix_sample.end_z_displacement_m, 6
                    ),
                    "suffix_original_direction_deg": round(
                        suffix_sample.original_direction_deg, 3
                    ),
                    "suffix_angle_offset_deg": round(
                        suffix_sample.xy_offset_deg, 3
                    ),
                }
            )
        print("RESET", json.dumps(reset_info, sort_keys=True), flush=True)

    print_reset(seed)
    episodes = 0
    with mujoco_viewer.launch_passive(
        viewer_model, viewer_data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        with _viewer_lock(viewer):
            viewer.opt.geomgroup[3] = 0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -22.0
            viewer.cam.distance = 1.45
            viewer.cam.lookat[:] = (0.18, -0.12, 0.12)
            viewer.sync()
        while viewer.is_running():
            started = time.perf_counter()
            _, rewards, resets, _ = stepper.step()
            with _viewer_lock(viewer):
                _mirror_native_viewer_data(
                    mujoco_module,
                    viewer_model,
                    viewer_data,
                    environment.host_data(0),
                )
                viewer.sync()
            if int(environment.progress[0]) % print_every == 0:
                print(
                    "FRAME",
                    json.dumps(
                        {
                            "seed": seed,
                            "progress": int(environment.progress[0]),
                            "trajectory_step": int(environment.trajectory_steps[0]),
                            "prefix_frames": prefix_sample.prefix_frames,
                            "suffix_frames": (
                                0 if suffix_sample is None else suffix_sample.suffix_frames
                            ),
                            "policy_enabled": bool(
                                environment.trajectory_steps[0]
                                >= prefix_sample.prefix_frames
                                and environment.trajectory_steps[0]
                                < environment.trajectory_lengths[0]
                                - environment.augmentation_suffix_frames[0]
                            ),
                            "reward": float(rewards[0]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if bool(resets[0]):
                reason = int(environment.last_termination.reason_code[0])
                print(
                    "TERMINAL",
                    json.dumps(
                        {"seed": seed, "progress": int(environment.progress[0]), "reason": reason},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                episodes += 1
                if max_episodes is not None and episodes >= max_episodes:
                    print("STOP", json.dumps({"episodes": episodes}), flush=True)
                    break
                seed += 1
                (
                    trajectory,
                    prefix_sample,
                    suffix_sample,
                    approach_seed,
                    retreat_seed,
                ) = sample(seed)
                _reinstall_approach_prefix_reference(
                    environment,
                    stepper,
                    trajectory,
                    seed=seed,
                )
                print_reset(seed)
                with _viewer_lock(viewer):
                    _mirror_native_viewer_data(
                        mujoco_module,
                        viewer_model,
                        viewer_data,
                        environment.host_data(0),
                    )
                    viewer.sync()
            time.sleep(max(0.0, 1.0 / (120.0 * speed) - (time.perf_counter() - started)))


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
        "--pairs",
        help="comma-separated object:action pairs to tile in one window (alternative to --object/--gesture)",
    )
    parser.add_argument("--dataset-path", type=Path, help="optional Lance dataset for modern hand-side rows")
    parser.add_argument(
        "--dataset-version",
        type=int,
        help="open this exact historical Lance version instead of the latest version",
    )
    parser.add_argument(
        "--reference-fps",
        type=int,
        choices=SUPPORTED_REFERENCE_FPS,
        help=(
            "source trajectory clock; omitted checkpoint inference restores the sidecar "
            "value, while dataset-only viewing defaults to 120 Hz"
        ),
    )
    parser.add_argument(
        "--pre-padding",
        type=int,
        help=f"trajectory frames before movement start (default: {DEFAULT_PRE_PADDING})",
    )
    parser.add_argument(
        "--post-padding",
        type=int,
        help=f"trajectory frames after movement end (default: {DEFAULT_POST_PADDING})",
    )
    parser.add_argument(
        "--hand-side",
        choices=("auto", "both", "right", "left"),
        default="auto",
        help="hands to control; explicit side makes the other hand follow reference",
    )
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
        help="enable residual action processing (default: true)",
    )
    parser.add_argument(
        "--terminal",
        type=parse_cli_bool,
        default=True,
        metavar="{true,false}",
        help="use the 0.10 m target training deviation threshold (default: true)",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue through explicit indexed episode resets after terminal.",
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
        help="Print the applied source command every N control calls.",
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
        pairs=args.pairs,
        dataset_path=args.dataset_path,
        dataset_version=args.dataset_version,
        reference_fps=args.reference_fps,
        pre_padding=args.pre_padding,
        post_padding=args.post_padding,
        hand_side=args.hand_side,
        rerun_output=args.rerun_output,
        use_residual=args.use_residual,
        checkpoint=args.checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
