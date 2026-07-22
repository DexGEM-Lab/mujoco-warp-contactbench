"""One-world residual-off reference replay for MuJoCo CPU and MJX-Warp."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sim.manorl.assets import compile_model, object_runtime
from sim.manorl.contracts import (
    CONTROL_STEP_COUNT,
    FINGER_SERVO_DAMPRATIO,
    FINGER_SERVO_KP,
    JOINT_NAMES,
    JOINT_DOF,
    LEGACY_JOINT_NAMES,
    PHYSICS_SUBSTEPS_PER_TARGET,
    PHYSICS_TIMESTEP,
    ServoConfig,
)
from sim.manorl.trajectory import ReferenceTrajectory, wxyz_to_xyzw, xyzw_to_wxyz

CONTACT_CAPACITY = 128
CONSTRAINT_CAPACITY = 512


@dataclass(frozen=True)
class ReplayConfig:
    backend: str
    device: str
    controller: str
    wrist_kp: float
    wrist_dampratio: float
    finger_kp: tuple[float, ...]
    finger_dampratio: float
    hand_contacts_enabled: bool
    physics_timestep: float = PHYSICS_TIMESTEP
    physics_substeps_per_target: int = PHYSICS_SUBSTEPS_PER_TARGET
    contact_capacity: int = CONTACT_CAPACITY
    constraint_capacity: int = CONSTRAINT_CAPACITY
    residual_enabled: bool = False


@dataclass(frozen=True)
class StepTrace:
    target_index: int
    reference_index: int
    source_reference_index: int
    sim_time: float
    q_target: NDArray[np.float64]
    hand_qpos: NDArray[np.float64]
    hand_qvel: NDArray[np.float64]
    actuator_force_substeps: NDArray[np.float64]
    hand_reference: NDArray[np.float64]
    object_pos: NDArray[np.float64]
    object_quat_xyzw: NDArray[np.float64]
    object_reference_pos_raw: NDArray[np.float64]
    object_reference_pos: NDArray[np.float64]
    object_reference_quat_xyzw: NDArray[np.float64]
    contact_count: int
    has_contact: bool
    hand_object_contact_count: int
    hand_object_min_distance: float
    hand_object_min_distance_valid: bool
    warning_count: int
    contact_capacity_saturated: bool
    constraint_capacity_saturated: bool


def command_target(
    q_reference: NDArray[np.floating[Any]],
    q_current: NDArray[np.floating[Any]],
    joint_lower: NDArray[np.floating[Any]],
    joint_upper: NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Choose nearest wrist Euler coordinates, then clamp the final command."""

    q_ref = np.asarray(q_reference, dtype=np.float64)
    current = np.asarray(q_current, dtype=np.float64)
    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    if q_ref.ndim != 1 or q_ref.shape[0] not in (26, 28, 52, 56):
        raise ValueError("q reference must have shape (26,), (28,), (52,), or (56,)")
    if any(values.shape != q_ref.shape for values in (current, lower, upper)):
        raise ValueError("q reference/current/limits must have matching shapes")
    if not all(np.all(np.isfinite(values)) for values in (q_ref, current, lower, upper)):
        raise ValueError("q reference/current/limits must be finite")
    target = q_ref.copy()
    per_hand = target.shape[0] if target.shape[0] < 52 else target.shape[0] // 2
    for start in range(0, target.shape[0], per_hand):
        wrist = slice(start + 3, start + 6)
        delta = (target[wrist] - current[wrist] + np.pi) % (2.0 * np.pi) - np.pi
        target[wrist] = current[wrist] + delta
    return np.clip(target, lower, upper)


def source_counter_indices(
    replay_step: int,
    *,
    control_step_count: int = CONTROL_STEP_COUNT,
) -> tuple[int, int]:
    """Map a physical call to source command/post indices from mano_hand.py counters.

    ``CONTROL_STEP_COUNT`` remains the accepted legacy trajectory contract.
    Modern Lance rows can have a different number of frames, so replay owners
    pass their resolved ``len(q_ref) - 1`` horizon explicitly.
    """

    if control_step_count < 1:
        raise ValueError("control_step_count must be positive")
    if not 0 <= replay_step < control_step_count:
        raise ValueError(f"replay_step must be in [0, {control_step_count - 1}]")
    return max(replay_step - 1, 0), replay_step


def _expand_legacy_hand_dofs(values: NDArray[object]) -> NDArray[np.float64]:
    """Embed a legacy 26-wide reference in the pinned 28-wide model order."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] not in (len(LEGACY_JOINT_NAMES), JOINT_DOF):
        raise ValueError(
            f"hand references must end in {len(LEGACY_JOINT_NAMES)} or {JOINT_DOF} DOFs"
        )
    if array.shape[-1] == JOINT_DOF:
        return array.copy()
    output = np.zeros((*array.shape[:-1], JOINT_DOF), dtype=np.float64)
    legacy_index = {name: index for index, name in enumerate(LEGACY_JOINT_NAMES)}
    for index, name in enumerate(JOINT_NAMES):
        # The old ``j1_thumb_mcp`` coordinate is the revised MCP flex axis;
        # the newly explicit CMC twist and MCP abduction start at zero.
        source_name = "j1_thumb_mcp" if name == "j1_thumb_mcp_flex" else name
        if source_name in legacy_index:
            output[..., index] = array[..., legacy_index[source_name]]
    return output


def _validate_action(action: NDArray[np.floating[Any]], *, dof: int) -> None:
    values = np.asarray(action)
    if values.shape != (dof,) or not np.all(np.isfinite(values)):
        raise ValueError(f"residual-off action must be finite with shape ({dof},)")


def _joint_limits(
    mujoco: Any,
    model: Any,
    *,
    hand_sides: tuple[str, ...],
    per_hand_dof: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read limits in the compiled right-then-left actuator order."""

    limits = []
    dual = len(hand_sides) > 1
    model_sides = tuple(side for side in ("right", "left") if side in hand_sides)
    for side in model_sides:
        prefix = f"{side}_" if dual else ""
        for name in JOINT_NAMES[:per_hand_dof]:
            joint_name = f"{prefix}{name}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"compiled joint is absent: {joint_name}")
            limits.append(model.jnt_range[joint_id].copy())
    array = np.asarray(limits, dtype=np.float64)
    return array[:, 0], array[:, 1]


class _ReplayBase:
    config: ReplayConfig

    def __init__(
        self, trajectory: ReferenceTrajectory, servo: ServoConfig = ServoConfig()
    ) -> None:
        self.trajectory = trajectory
        self.servo = servo
        parts = trajectory.identity.identity.split("_")
        if len(parts) != 3 or not parts[1].isdigit():
            raise ValueError("trajectory identity must be object_action_sequence")
        self.object_type = parts[0]
        runtime = object_runtime(self.object_type)
        # The pinned runtime URDF is 28-DoF.  A legacy trajectory is expanded
        # only at this replay boundary so its on-disk/reference ABI remains
        # 26-wide while native qpos/ctrl and traces are model-shaped.
        hand_side = (
            "both"
            if len(trajectory.hand_sides) == 2
            else trajectory.hand_sides[0]
        )
        self.mujoco, self.model = compile_model(
            servo, object_type=self.object_type, hand_side=hand_side
        )
        self.model_dof = int(self.model.nu)
        self.per_hand_dof = self.model_dof // len(trajectory.hand_sides)
        if self.per_hand_dof != JOINT_DOF:
            raise ValueError(
                f"compiled replay model has unsupported per-hand actuator width {self.per_hand_dof}"
            )
        if self.model_dof != self.per_hand_dof * len(trajectory.hand_sides):
            raise ValueError("compiled replay actuator width is not divisible by hand count")
        # Keep ``hand_dof`` as the total native qpos/ctrl width for callers of
        # the historical replay object; one-hand scenes therefore retain the
        # old value while bimanual scenes expose all 56 model coordinates.
        self.hand_dof = self.model_dof
        self.reference_q = {
            side: _expand_legacy_hand_dofs(trajectory.q_ref_for(side))[:, : self.per_hand_dof]
            for side in trajectory.hand_sides
        }
        self.reference_q_model = np.concatenate(
            [self.reference_q[side] for side in ("right", "left") if side in self.reference_q],
            axis=1,
        )
        if self.reference_q_model.shape[1] != self.model_dof:
            raise ValueError(
                "trajectory hand references do not match compiled replay actuator width"
            )
        model_sides = tuple(side for side in ("right", "left") if side in trajectory.hand_sides)
        self.joint_lower, self.joint_upper = _joint_limits(
            self.mujoco,
            self.model,
            hand_sides=model_sides,
            per_hand_dof=self.per_hand_dof,
        )
        self.action_dim = trajectory.action_layout.action_dim
        self.control_step_count = len(trajectory.q_ref) - 1
        object_joint = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, runtime.free_joint_name
        )
        if object_joint < 0:
            raise ValueError("compiled cube free joint is absent")
        self.object_qpos_address = int(self.model.jnt_qposadr[object_joint])
        object_body = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, runtime.body_name
        )
        self.object_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == object_body
        }
        self.hand_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) not in (0, object_body)
        }
        expected_hand_geoms = 16 * len(trajectory.hand_sides)
        if (
            len(self.object_geom_ids) != runtime.collision_geom_count
            or len(self.hand_geom_ids) != expected_hand_geoms
        ):
            raise ValueError("compiled hand/object geom partition is inconsistent")
        self.replay_step = 0

    def _config(self, *, backend: str, device: str) -> ReplayConfig:
        return ReplayConfig(
            backend=backend,
            device=device,
            controller="mujoco_native_position",
            wrist_kp=float(self.servo.wrist_kp),
            wrist_dampratio=float(self.servo.wrist_dampratio),
            finger_kp=tuple(float(value) for value in FINGER_SERVO_KP),
            finger_dampratio=float(FINGER_SERVO_DAMPRATIO),
            hand_contacts_enabled=self.servo.hand_contacts_enabled,
        )

    def _reset_host_data(self) -> Any:
        data = self.mujoco.MjData(self.model)
        self.mujoco.mj_resetData(self.model, data)
        data.qpos[: self.hand_dof] = self.reference_q_model[0]
        address = self.object_qpos_address
        data.qpos[address : address + 3] = self.trajectory.object_pos[0]
        data.qpos[address + 3 : address + 7] = xyzw_to_wxyz(
            self.trajectory.object_quat_xyzw[0]
        )
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, data)
        self.replay_step = 0
        return data

    def _build_trace(
        self,
        host_data: Any,
        target_index: int,
        reference_index: int,
        q_target: NDArray[np.float64],
        actuator_forces: list[NDArray[np.float64]],
    ) -> StepTrace:
        address = self.object_qpos_address
        object_quat_xyzw = wxyz_to_xyzw(host_data.qpos[address + 3 : address + 7]).copy()
        object_quat_xyzw /= np.linalg.norm(object_quat_xyzw)
        contact_count = int(host_data.ncon)
        hand_object_distances: list[float] = []
        for contact in host_data.contact[:contact_count]:
            first, second = (int(contact.geom[0]), int(contact.geom[1]))
            is_hand_object = (
                first in self.hand_geom_ids and second in self.object_geom_ids
            ) or (
                second in self.hand_geom_ids and first in self.object_geom_ids
            )
            if is_hand_object:
                hand_object_distances.append(float(contact.dist))
        hand_object_min_distance_valid = bool(hand_object_distances)
        # A separate validity bit makes zero an explicit finite sentinel rather
        # than silently treating absence of hand-object contact as zero gap.
        hand_object_min_distance = (
            min(hand_object_distances) if hand_object_distances else 0.0
        )
        return StepTrace(
            target_index=target_index,
            reference_index=reference_index,
            source_reference_index=int(self.trajectory.source_indices[reference_index]),
            sim_time=float(host_data.time),
            q_target=q_target.copy(),
            hand_qpos=host_data.qpos[: self.hand_dof].copy(),
            hand_qvel=host_data.qvel[: self.hand_dof].copy(),
            actuator_force_substeps=np.stack(actuator_forces),
            hand_reference=self.reference_q_model[reference_index].copy(),
            object_pos=host_data.qpos[address : address + 3].copy(),
            object_quat_xyzw=object_quat_xyzw,
            object_reference_pos_raw=self.trajectory.object_pos_raw[reference_index].copy(),
            object_reference_pos=self.trajectory.object_pos[reference_index].copy(),
            object_reference_quat_xyzw=self.trajectory.object_quat_xyzw[reference_index].copy(),
            contact_count=contact_count,
            has_contact=contact_count > 0,
            hand_object_contact_count=len(hand_object_distances),
            hand_object_min_distance=hand_object_min_distance,
            hand_object_min_distance_valid=hand_object_min_distance_valid,
            warning_count=sum(int(warning.number) for warning in host_data.warning),
            contact_capacity_saturated=(
                self.config.backend == "mjx-warp"
                and contact_count >= self.config.contact_capacity
            ),
            constraint_capacity_saturated=(
                self.config.backend == "mjx-warp"
                and int(host_data.nefc) >= self.config.constraint_capacity
            ),
        )


class MujocoCpuReplay(_ReplayBase):
    """Thin CPU backend used to diagnose model and controller behavior."""

    def __init__(
        self, trajectory: ReferenceTrajectory, servo: ServoConfig = ServoConfig()
    ) -> None:
        super().__init__(trajectory, servo)
        self.config = self._config(backend="mujoco-cpu", device="cpu")
        self.data = self._reset_host_data()

    def reset(self) -> None:
        self.data = self._reset_host_data()

    def step(self, action: NDArray[np.floating[Any]]) -> StepTrace:
        _validate_action(action, dof=self.action_dim)
        if self.replay_step >= self.control_step_count:
            raise StopIteration("source-compatible replay terminated before final slice reference")
        target_index, reference_index = source_counter_indices(
            self.replay_step,
            control_step_count=self.control_step_count,
        )
        actuator_forces: list[NDArray[np.float64]] = []
        q_target = command_target(
            self.reference_q_model[target_index],
            self.data.qpos[: self.hand_dof],
            self.joint_lower,
            self.joint_upper,
        )
        for _ in range(PHYSICS_SUBSTEPS_PER_TARGET):
            self.data.ctrl[:] = q_target
            self.mujoco.mj_step(self.model, self.data)
            actuator_forces.append(self.data.actuator_force.copy())
        trace = self._build_trace(
            self.data, target_index, reference_index, q_target, actuator_forces
        )
        self.replay_step += 1
        return trace


class MjxWarpReplay(_ReplayBase):
    """One-world MJX replay using the public Warp implementation selector."""

    def __init__(
        self,
        trajectory: ReferenceTrajectory,
        *,
        device: str = "cpu",
        servo: ServoConfig = ServoConfig(),
    ) -> None:
        super().__init__(trajectory, servo)
        if device not in {"cpu", "gpu"}:
            raise ValueError(f"unsupported JAX device selector: {device}")
        try:
            import jax
            from mujoco import mjx
        except ImportError as exc:
            raise RuntimeError("jax and mujoco-mjx are required for MJX-Warp replay") from exc
        devices = jax.devices(device)
        if not devices:
            raise RuntimeError(f"no JAX {device} device is available")
        self.jax = jax
        self.mjx = mjx
        self.device = devices[0]
        self.config = self._config(backend="mjx-warp", device=device)
        self.mjx_model = mjx.put_model(self.model, device=self.device, impl="warp")
        if str(self.mjx_model.impl).lower().split(".")[-1] != "warp":
            raise RuntimeError(f"MJX did not select the Warp implementation: {self.mjx_model.impl}")
        self._step_fn = jax.jit(mjx.step)
        self.reset()

    def reset(self) -> None:
        host_data = self._reset_host_data()
        self.data = self.mjx.put_data(
            self.model,
            host_data,
            device=self.device,
            impl="warp",
            naconmax=self.config.contact_capacity,
            njmax=self.config.constraint_capacity,
        )

    def _host_data(self) -> Any:
        host_data = self.mjx.get_data(self.model, self.data)
        if isinstance(host_data, list):
            if len(host_data) != 1:
                raise RuntimeError(f"expected one MJX world, got {len(host_data)}")
            return host_data[0]
        return host_data

    def step(self, action: NDArray[np.floating[Any]]) -> StepTrace:
        _validate_action(action, dof=self.action_dim)
        if self.replay_step >= self.control_step_count:
            raise StopIteration("source-compatible replay terminated before final slice reference")
        target_index, reference_index = source_counter_indices(
            self.replay_step,
            control_step_count=self.control_step_count,
        )
        actuator_forces: list[NDArray[np.float64]] = []
        qpos = np.asarray(self.data.qpos[: self.hand_dof], dtype=np.float64)
        q_target = command_target(
            self.reference_q_model[target_index], qpos, self.joint_lower, self.joint_upper
        )
        for _ in range(PHYSICS_SUBSTEPS_PER_TARGET):
            self.data = self.data.replace(ctrl=self.jax.numpy.asarray(q_target))
            self.data = self._step_fn(self.mjx_model, self.data)
            actuator_forces.append(self._host_data().actuator_force.copy())
        trace = self._build_trace(
            self._host_data(), target_index, reference_index, q_target, actuator_forces
        )
        self.replay_step += 1
        return trace


def create_replay(
    trajectory: ReferenceTrajectory,
    *,
    backend: str,
    device: str = "cpu",
    servo: ServoConfig = ServoConfig(),
) -> MujocoCpuReplay | MjxWarpReplay:
    if backend == "mjx-warp":
        return MjxWarpReplay(trajectory, device=device, servo=servo)
    if backend == "mujoco-cpu":
        if device != "cpu":
            raise ValueError("mujoco-cpu only supports --device cpu")
        return MujocoCpuReplay(trajectory, servo)
    raise ValueError(f"unsupported replay backend: {backend}")


def run_reference_replay(
    trajectory: ReferenceTrajectory,
    *,
    backend: str = "mjx-warp",
    device: str = "cpu",
    max_steps: int | None = None,
    servo: ServoConfig = ServoConfig(),
) -> tuple[ReplayConfig, dict[str, NDArray[Any]]]:
    """Run source counters: command 0, 0, 1, ...; compare reference 0, 1, 2, ...."""

    replay = create_replay(trajectory, backend=backend, device=device, servo=servo)
    step_count = replay.control_step_count if max_steps is None else int(max_steps)
    if not 1 <= step_count <= replay.control_step_count:
        raise ValueError(f"max_steps must be in [1, {replay.control_step_count}]")
    # Residual-off replay still validates the caller-facing action ABI (26 for
    # a legacy reference, 28/56 for revised one-/two-hand references), while
    # the native model/trace remains expanded to its compiled width.
    zero_action = np.zeros(replay.action_dim, dtype=np.float64)
    records = [replay.step(zero_action) for _ in range(step_count)]
    trace: dict[str, NDArray[Any]] = {}
    for field in fields(StepTrace):
        values = [getattr(record, field.name) for record in records]
        trace[field.name] = np.stack(values) if isinstance(values[0], np.ndarray) else np.asarray(values)
    return replay.config, trace
