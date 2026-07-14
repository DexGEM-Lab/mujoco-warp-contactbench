"""One-world residual-off reference replay for MuJoCo CPU and MJX-Warp."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sim.manorl.assets import compile_model
from sim.manorl.contracts import (
    CONTROL_STEP_COUNT,
    FINGER_SERVO_DAMPRATIO,
    FINGER_SERVO_KP,
    JOINT_NAMES,
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
    if any(values.shape != (26,) for values in (q_ref, current, lower, upper)):
        raise ValueError("q reference/current/limits must each have shape (26,)")
    if not all(np.all(np.isfinite(values)) for values in (q_ref, current, lower, upper)):
        raise ValueError("q reference/current/limits must be finite")
    target = q_ref.copy()
    delta = (target[3:6] - current[3:6] + np.pi) % (2.0 * np.pi) - np.pi
    target[3:6] = current[3:6] + delta
    return np.clip(target, lower, upper)


def source_counter_indices(replay_step: int) -> tuple[int, int]:
    """Map a physical call to source command/post indices from mano_hand.py counters."""

    if not 0 <= replay_step < CONTROL_STEP_COUNT:
        raise ValueError(f"replay_step must be in [0, {CONTROL_STEP_COUNT - 1}]")
    return max(replay_step - 1, 0), replay_step


def _validate_action(action: NDArray[np.floating[Any]]) -> None:
    values = np.asarray(action)
    if values.shape != (26,) or not np.all(np.isfinite(values)):
        raise ValueError("residual-off action must be finite with shape (26,)")


def _joint_limits(mujoco: Any, model: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    limits = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"compiled joint is absent: {name}")
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
        self.mujoco, self.model = compile_model(servo)
        self.joint_lower, self.joint_upper = _joint_limits(self.mujoco, self.model)
        object_joint = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "powerdrill_free"
        )
        if object_joint < 0:
            raise ValueError("compiled powerdrill free joint is absent")
        self.object_qpos_address = int(self.model.jnt_qposadr[object_joint])
        object_body = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, "powerdrill"
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
        if len(self.object_geom_ids) != 5 or len(self.hand_geom_ids) != 16:
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
        data.qpos[:26] = self.trajectory.q_ref[0]
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
            hand_qpos=host_data.qpos[:26].copy(),
            hand_qvel=host_data.qvel[:26].copy(),
            actuator_force_substeps=np.stack(actuator_forces),
            hand_reference=self.trajectory.q_ref[reference_index].copy(),
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
        _validate_action(action)
        if self.replay_step >= CONTROL_STEP_COUNT:
            raise StopIteration("source-compatible replay terminated before final slice reference")
        target_index, reference_index = source_counter_indices(self.replay_step)
        actuator_forces: list[NDArray[np.float64]] = []
        q_target = command_target(
            self.trajectory.q_ref[target_index],
            self.data.qpos[:26],
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
        _validate_action(action)
        if self.replay_step >= CONTROL_STEP_COUNT:
            raise StopIteration("source-compatible replay terminated before final slice reference")
        target_index, reference_index = source_counter_indices(self.replay_step)
        actuator_forces: list[NDArray[np.float64]] = []
        qpos = np.asarray(self.data.qpos[:26], dtype=np.float64)
        q_target = command_target(
            self.trajectory.q_ref[target_index], qpos, self.joint_lower, self.joint_upper
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

    step_count = CONTROL_STEP_COUNT if max_steps is None else int(max_steps)
    if not 1 <= step_count <= CONTROL_STEP_COUNT:
        raise ValueError(f"max_steps must be in [1, {CONTROL_STEP_COUNT}]")
    replay = create_replay(trajectory, backend=backend, device=device, servo=servo)
    zero_action = np.zeros(26, dtype=np.float64)
    records = [replay.step(zero_action) for _ in range(step_count)]
    trace: dict[str, NDArray[Any]] = {}
    for field in fields(StepTrace):
        values = [getattr(record, field.name) for record in records]
        trace[field.name] = np.stack(values) if isinstance(values[0], np.ndarray) else np.asarray(values)
    return replay.config, trace
