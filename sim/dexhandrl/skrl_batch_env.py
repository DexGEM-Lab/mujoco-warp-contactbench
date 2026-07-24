"""One-model, device-resident MJX-Warp batch runtime for DexHandRL.

This module is deliberately separate from the Gym adapter.  A batch owns one
compiled ``cube1`` model and a vmapped ``mjx.Data`` PyTree; all per-world state,
reference gathering, contact decoding, reward, and observation construction stay
in JAX through :meth:`step_device` and :meth:`reset_done_device`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from sim.dexhandrl.constants import (
    ACTION_DIM_021PRO,
    DEFAULT_ISAAC_SOURCE_ROOT,
    DEXHAND021PRO_CONTACT_BODY_NAMES,
    DEXHAND021PRO_FINGERTIP_BODY_NAMES,
    DEXHAND021PRO_ACTUATOR_NAMES,
    POINT_CLOUD_DIM_021PRO,
)
from sim.dexhandrl.contracts import DexHandPhysicsConfig
from sim.dexhandrl.device_contacts import reduce_dexhand_contacts
from sim.dexhandrl.device_task import (
    apply_device_control,
    build_device_observation,
    compute_device_reward_done,
    fingertip_inside_object_penalty,
    gather_trajectory,
    route_active_to_full,
    route_full_to_active,
)
from sim.dexhandrl.hand_mapping import (
    DEXHAND021PRO_FULL_DOF_NAMES,
    active_to_full_dof,
    dexhand021pro_route_joint_upper_limits,
    full_to_active_dof,
    resolve_mujoco_ids,
)
from sim.dexhandrl.obs_layout import OBS_DIM_021PRO
from sim.dexhandrl.parity import encode_geometry_12d, geometry_from_isaac_urdf, load_isaac_reward_contract
from sim.dexhandrl.scene import build_dexhandrl_scene_xml, validate_dexhandrl_compiled_model
from sim.dexhandrl.skrl_env import _sample_object_point_cloud
from sim.dexhandrl.trajectory_batch import DexHandTrajectoryBatch


@dataclass(frozen=True)
class DexHandRLBatchConfig:
    """Static contract for a homogeneous cube1 MJX-Warp batch."""

    trajectories: DexHandTrajectoryBatch
    action: str = "all"
    device: str = "cpu"
    object_name: str = "cube1"
    physics: DexHandPhysicsConfig = DexHandPhysicsConfig()
    reference_joint_source: str = "active"
    settle_frames: int = 1
    object_reset_z_offset: float = 0.01
    use_residual: bool = True
    early_phase_frames: int = 130
    object_position_terminal_threshold: float = 0.1
    termination_failure_penalty: float = -200.0
    termination_timeout_penalty: float = 0.0
    contact_force_threshold: float = 5.0
    max_contact_reward: float = 0.4
    max_object_stability_reward: float = 0.4
    object_stability_velocity_scale: float = 0.1
    fingertip_inside_object_min_depth: float = 0.001
    fingertip_inside_object_max_depth: float = 0.003
    max_fingertip_inside_object_penalty: float = 0.4
    base_position_decay: tuple[float, float, float] = (0.9, 0.9, 0.9)
    post_contact_base_position_decay: tuple[float, float, float] = (0.9, 0.9, 0.9)
    joint_decay: float = 0.9
    post_contact_joint_decay: float = 1.0
    base_position_scales: tuple[float, float, float] = (0.002, 0.002, 0.003)
    base_position_max_offset: float = 0.04
    policy_base_lin_velocity_limit: float = 3.0
    policy_base_ang_velocity_limit: float = 10.0
    policy_finger_velocity_limit: float = 2.0
    contact_capacity: int | None = None
    ccd_capacity: int | None = None
    per_world_constraint_capacity: int | None = None
    isaac_source_root: Path = DEFAULT_ISAAC_SOURCE_ROOT
    task_config_path: Path | None = None

    def __post_init__(self) -> None:
        if self.action != "all":
            raise ValueError("batched DexHandRL requires action='all'")
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'")
        if self.object_name != "cube1":
            raise ValueError("the one-model batch contract currently compiles cube1 only")
        if self.reference_joint_source != "active":
            raise ValueError("batched DexHandRL requires reference_joint_source='active'")
        if self.trajectories.num_envs < 1:
            raise ValueError("trajectory batch must contain at least one world")
        if self.physics.substeps != 10:
            raise ValueError("batched DexHandRL pins exactly 10 MJX steps per control transition")
        if self.settle_frames < 0:
            raise ValueError("settle_frames must be non-negative")
        if self.contact_capacity is not None and self.contact_capacity < 128 * self.trajectories.num_envs:
            raise ValueError("contact_capacity is below the conservative global 128-row-per-world floor")
        if self.ccd_capacity is not None and self.ccd_capacity < 64 * self.trajectories.num_envs:
            raise ValueError("ccd_capacity is below the conservative global 64-candidate-per-world floor")
        if self.per_world_constraint_capacity is not None and self.per_world_constraint_capacity < 1:
            raise ValueError("per_world_constraint_capacity must be positive")


# Keep the launcher spelling discoverable without importing the training CLI.
SkrlBatchEnvConfig = DexHandRLBatchConfig


class DeviceBatchTransition(NamedTuple):
    observation: Any
    reward: Any
    terminated: Any
    truncated: Any
    reason: Any
    weighted: Any
    reference_active: Any
    corrected_active: Any
    applied_active: Any
    full_target: Any
    ctrl: Any
    valid: Any


class _BatchState(NamedTuple):
    data: Any
    frame: Any
    episode_count: Any
    previous_active: Any
    base_offset: Any
    joint_offset: Any


@dataclass(frozen=True)
class BatchRenderSnapshot:
    """Small host-owned state needed to mirror batched worlds for rendering.

    This intentionally excludes MJX-Warp's global contact and constraint
    buffers.  Rendering needs generalized coordinates and controls only;
    asking it for telemetry would synchronize and copy capacity-sized buffers.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    frame: np.ndarray
    episode_count: np.ndarray


def conservative_batch_contact_capacity(num_envs: int) -> int:
    """Global Warp contact capacity: preregistered 128 rows per world + 64.

    This is an allocation bound, not a physics parameter. Server2's 256-world
    run peaked at 1,377 global contacts (about 5.4/world), leaving roughly 24x
    headroom at this floor while avoiding the previous graph-sized excess.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    return 128 * num_envs + 64


def conservative_batch_ccd_capacity(num_envs: int) -> int:
    """Global Warp CCD/EPA scratch capacity: 64 candidates per world + 64.

    ``naccdmax`` bounds collision-convex EPA scratch independently of the
    global contact table. Server2's direct seven-world reset observed 220
    collision candidates (about 31.4/world), so this retains a twofold reset
    bound without allocating contact-sized scratch.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    return 64 * num_envs + 64


def conservative_batch_constraint_capacity(num_envs: int) -> int:
    """Per-world Warp constraint capacity: 512, independent of batch size.

    ``efc__force`` is shaped ``(world, capacity)``. Construction also measures
    the native initial state and rejects any capacity below ``nefc + 64``;
    capacity guards still fail at 80% utilization during rollout.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    return 512


class DexHandRLMJXBatchEnv:
    """A genuine vmapped MJX-Warp DexHandRL environment.

    The public device methods never materialize NumPy values.  ``telemetry`` is
    the explicit host boundary for callers which need logging snapshots.
    """

    num_actions = ACTION_DIM_021PRO
    num_observations = OBS_DIM_021PRO

    def __init__(self, cfg: DexHandRLBatchConfig):
        self.cfg = cfg
        self.batch = cfg.trajectories
        self.num_envs = self.batch.num_envs
        self.contact_capacity = cfg.contact_capacity or conservative_batch_contact_capacity(self.num_envs)
        self.ccd_capacity = cfg.ccd_capacity or conservative_batch_ccd_capacity(self.num_envs)
        self.per_world_constraint_capacity = (
            cfg.per_world_constraint_capacity or conservative_batch_constraint_capacity(self.num_envs)
        )
        if cfg.device == "cpu":
            import os
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        import jax.numpy as jp
        import mujoco
        from mujoco import mjx

        self.jax, self.jp, self.mujoco, self.mjx = jax, jp, mujoco, mjx
        platform = "cpu" if cfg.device == "cpu" else "gpu"
        devices = jax.devices(platform)
        if not devices:
            raise RuntimeError(f"no JAX {platform} device is available")
        self.device = devices[0]
        root = Path(cfg.isaac_source_root)
        task_config = Path(cfg.task_config_path or root / "dexhand_env/cfg/task/Dexhand021proReconstruction.yaml")
        self.reward_contract = load_isaac_reward_contract(task_config)
        self.reward_weights = np.asarray([
            self.reward_contract.reward_weights[name] for name in (
                "alive", "object_pos_tracking_x", "object_pos_tracking_y", "object_pos_tracking_z",
                "contact_quality", "object_stability_velocity", "object_rot_tracking",
                "fingertip_inside_object_penalty", "action_penalty_position", "action_penalty_joint",
            )
        ], dtype=np.float32)
        scene_path = Path(__import__("tempfile").mkdtemp(prefix="dexhandrl_mjx_batch_")) / "cube1.xml"
        self.scene_path = build_dexhandrl_scene_xml(scene_path, object_name="cube1", isaac_source_root=root, physics=cfg.physics)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        validate_dexhandrl_compiled_model(mujoco, self.model, cfg.physics, object_name="cube1")
        self.ids = resolve_mujoco_ids(mujoco, self.model, object_name="cube1")
        self.mjx_model = mjx.put_model(self.model, device=self.device, impl="warp")
        if str(self.mjx_model.impl).lower().split(".")[-1] != "warp":
            raise RuntimeError(f"MJX did not select Warp implementation: {self.mjx_model.impl}")
        self._step_world = jax.jit(jax.vmap(lambda world: mjx.step(self.mjx_model, world)))
        self._forward_world = jax.jit(jax.vmap(lambda world: mjx.forward(self.mjx_model, world)))
        self._build_static_contract(root)
        self._build_reset_state()
        self._transition_fn = jax.jit(self._transition)
        self._reset_mask_fn = jax.jit(self._reset_mask)
        self._reset_observation_fn = jax.jit(self._reset_observation)
        self.state = self._initial_state

    def _assert_warp_abi(self, data: Any) -> None:
        """Fail closed when the one allocated Warp data has no contact ABI."""
        # MJX's public ``Data.impl`` is an enum; Warp arrays live in the
        # explicit private implementation payload used by the pinned ABI.
        impl = data._impl
        required = ("contact__geom", "contact__worldid", "contact__dim", "contact__efc_address", "contact__friction", "contact__frame")
        missing = [name for name in required if not hasattr(impl, name)]
        required_scalars = ("nacon", "nefc", "efc__force", "ncollision", "naccdmax")
        if missing or not all(hasattr(impl, name) for name in required_scalars):
            raise RuntimeError(f"MJX-Warp contact ABI is incompatible; missing={missing}")
        if tuple(impl.ncollision.shape) != (1,):
            raise RuntimeError(f"MJX-Warp ncollision ABI is incompatible: shape={impl.ncollision.shape}")
        if int(impl.naccdmax) != self.ccd_capacity:
            raise RuntimeError(
                f"MJX-Warp naccdmax ABI is incompatible: {impl.naccdmax} != requested {self.ccd_capacity}"
            )

    def _build_static_contract(self, root: Path) -> None:
        mujoco, model = self.mujoco, self.model
        self.qpos_ids = np.asarray([self.ids.joint_qpos_addr[name] for name in DEXHAND021PRO_FULL_DOF_NAMES], dtype=np.int32)
        self.dof_ids = np.asarray([self.ids.joint_dof_addr[name] for name in DEXHAND021PRO_FULL_DOF_NAMES], dtype=np.int32)
        self.actuator_ids = np.asarray([self.ids.actuator_ids[name] for name in DEXHAND021PRO_ACTUATOR_NAMES], dtype=np.int32)
        self.route_joint_upper_limits = dexhand021pro_route_joint_upper_limits(mujoco, model)
        self.route_upper = np.asarray([self.route_joint_upper_limits[index] for index in (6, 9, 12, 15)], dtype=np.float32)
        contact_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in DEXHAND021PRO_CONTACT_BODY_NAMES]
        tip_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in DEXHAND021PRO_FINGERTIP_BODY_NAMES]
        if min(contact_ids + tip_ids) < 0:
            raise RuntimeError("compiled cube1 model is missing DexHand contact/fingertip bodies")
        self.contact_body_ids, self.fingertip_body_ids = np.asarray(contact_ids, dtype=np.int32), np.asarray(tip_ids, dtype=np.int32)
        self.hand_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_base"))
        if self.hand_body_id < 0:
            self.hand_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "RFH1"))
        self.object_body_id = self.ids.object_body_id
        body_to_contact = {int(body): index for index, body in enumerate(contact_ids)}
        self.geom_to_contact_body = np.asarray([body_to_contact.get(int(body), -1) for body in model.geom_bodyid], dtype=np.int32)
        self.object_geom_flags = np.asarray(model.geom_bodyid == self.object_body_id, dtype=bool)
        if not np.any(self.object_geom_flags):
            raise RuntimeError("cube1 scene has no object-owned geom")
        active_lower, active_upper = self._active_limits()
        geometry = geometry_from_isaac_urdf("cube1", isaac_source_root=root)
        kind = {"box": 0, "cylinder": 1, "sphere": 2}.get(str(geometry["type"]), 3)
        size = np.asarray(geometry["size"], dtype=np.float32)
        primitive = size if kind == 0 else (np.asarray([size[0], size[1], 0], np.float32) if kind == 1 else np.asarray([size[0], 0, 0], np.float32))
        cloud = _sample_object_point_cloud(geometry, object_name="cube1", isaac_source_root=root, num_points=POINT_CLOUD_DIM_021PRO // 3, seed=42)
        max_delta = np.asarray(
            [self.cfg.physics.control_dt * self.cfg.policy_base_lin_velocity_limit] * 3
            + [self.cfg.physics.control_dt * self.cfg.policy_base_ang_velocity_limit] * 3
            + [self.cfg.physics.control_dt * self.cfg.policy_finger_velocity_limit] * 16,
            dtype=np.float32,
        )
        # Immutable reference/task tables are placed exactly once.
        put = lambda x: self.jax.device_put(self.jp.asarray(x), self.device)
        self._tables = {
            "hand": put(self.batch.hand_active), "object_pos": put(self.batch.object_pos), "object_quat": put(self.batch.object_quat_xyzw),
            "lengths": put(self.batch.lengths), "last_contact": put(self.batch.movement_end_local), "onehot": put(self.batch.action_onehot50),
            "expected": put(self.batch.task_expected_contact_masks), "joint_mask": put(self.batch.active_joint_masks),
            "tip_mask": put(self.batch.active_fingertip_masks), "rotation_disabled": put(self.batch.rotation_disabled),
            "geometry": put(np.broadcast_to(encode_geometry_12d(geometry), (self.num_envs, 12))), "primitive_type": put(np.full(self.num_envs, kind, np.int32)),
            "primitive": put(np.broadcast_to(primitive, (self.num_envs, 3))), "cloud": put(cloud), "qpos_ids": put(self.qpos_ids),
            "dof_ids": put(self.dof_ids), "actuator_ids": put(self.actuator_ids), "route_upper": put(self.route_upper),
            "active_lower": put(active_lower), "active_upper": put(active_upper), "finger_lower": put(active_lower[6:]), "finger_upper": put(active_upper[6:]),
            "max_delta": put(max_delta), "contact_body_ids": put(self.contact_body_ids), "tip_ids": put(self.fingertip_body_ids),
            "reward_weights": put(self.reward_weights), "geom_body": put(self.geom_to_contact_body), "geom_object": put(self.object_geom_flags),
        }

    def _active_limits(self) -> tuple[np.ndarray, np.ndarray]:
        base = []
        for name in DEXHAND021PRO_FULL_DOF_NAMES[:6]:
            jid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            base.append(self.model.jnt_range[jid].astype(np.float32))
        # Scalar authority normalizes its 16 active finger slots from the
        # corresponding primary MuJoCo joints, including each coupled joint3.
        primary = (6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24)
        fingers = []
        for index in primary:
            jid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, DEXHAND021PRO_FULL_DOF_NAMES[index])
            fingers.append(self.model.jnt_range[jid].astype(np.float32))
        arr = np.asarray(base + fingers, dtype=np.float32)
        return arr[:, 0], arr[:, 1]

    def _build_reset_state(self) -> None:
        # This is the only make_data call.  Vmap creates independent dynamic
        # worlds while preserving one shared MJX-Warp model/contact ABI.
        # Scalar reset routes each active reference through the model's true
        # distal limits and then stores the resulting active target.  Starting
        # from raw active data would make the first velocity-clamp recurrence
        # disagree whenever a coupled target clips.
        full0 = active_to_full_dof(
            self.batch.hand_active[:, 0], route_joint_upper_limits=self.route_joint_upper_limits,
        ).astype(np.float32)
        reset_active = full_to_active_dof(full0).astype(np.float32)
        qpos = np.broadcast_to(
            np.asarray(self.mujoco.MjData(self.model).qpos), (self.num_envs, self.model.nq),
        ).copy()
        qpos[:, self.qpos_ids] = full0
        qadr = self.ids.object_free_qpos_addr
        qpos[:, qadr:qadr + 3] = self.batch.object_pos[:, 0] + np.asarray([0, 0, self.cfg.object_reset_z_offset], np.float32)
        qpos[:, qadr + 3:qadr + 7] = self.batch.object_quat_xyzw[:, 0][:, [3, 0, 1, 2]]
        self._host_initial_nefc = self._host_initial_constraint_high_water(full0, qpos)
        self._minimum_per_world_constraint_capacity = self._host_initial_nefc + 64
        if self.per_world_constraint_capacity < self._minimum_per_world_constraint_capacity:
            raise ValueError(
                "per_world_constraint_capacity is below the initial host nefc + 64 margin "
                f"({self.per_world_constraint_capacity} < {self._minimum_per_world_constraint_capacity})"
            )
        single = self.mjx.make_data(
            self.model, device=self.device, impl="warp", naconmax=self.contact_capacity,
            naccdmax=self.ccd_capacity, njmax=self.per_world_constraint_capacity,
        )
        self._assert_warp_abi(single)
        template = self.jax.vmap(lambda _: single)(self.jp.arange(self.num_envs, dtype=self.jp.int32))
        initial_active = self.jp.asarray(reset_active)
        qvel = np.zeros((self.num_envs, single.qvel.shape[0]), dtype=np.float32)
        ctrl = np.zeros((self.num_envs, self.model.nu), dtype=np.float32)
        ctrl[:, self.actuator_ids] = full0
        data = template.replace(qpos=self.jp.asarray(qpos), qvel=self.jp.asarray(qvel), ctrl=self.jp.asarray(ctrl))
        data = self._forward_world(data)
        for _ in range(self.cfg.settle_frames * 10):
            data = self._step_world(data)
        # Cached settled rows are the source of truth for indexed resets.
        self._settled_data = data
        self._reset_previous_active = initial_active
        previous = initial_active
        self._initial_state = _BatchState(
            data=data, frame=self.jp.ones((self.num_envs,), dtype=self.jp.int32),
            episode_count=self.jp.zeros((self.num_envs,), dtype=self.jp.int32), previous_active=previous,
            base_offset=self.jp.zeros((self.num_envs, 3), dtype=self.jp.float32),
            joint_offset=self.jp.zeros((self.num_envs, 16), dtype=self.jp.float32),
        )

    def _host_initial_constraint_high_water(self, full0: np.ndarray, qpos: np.ndarray) -> int:
        """Measure native initial-state constraints before accepting an override."""
        maximum = 0
        for world, full in enumerate(full0):
            data = self.mujoco.MjData(self.model)
            data.qpos[:] = qpos[world]
            data.ctrl[self.actuator_ids] = full
            self.mujoco.mj_forward(self.model, data)
            maximum = max(maximum, int(data.nefc))
        return maximum

    def _features(self, data: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        t = self._tables
        current_full = data.qpos[:, t["qpos_ids"]]
        hand_pos = data.xpos[:, self.hand_body_id]
        hand_wxyz = data.xquat[:, self.hand_body_id]
        hand_xyzw = self.jp.concatenate((hand_wxyz[:, 1:], hand_wxyz[:, :1]), axis=1)
        object_pos = data.xpos[:, self.object_body_id]
        object_wxyz = data.xquat[:, self.object_body_id]
        object_xyzw = self.jp.concatenate((object_wxyz[:, 1:], object_wxyz[:, :1]), axis=1)
        fingertips = data.xpos[:, t["tip_ids"]]
        contact_bodies = data.xpos[:, t["contact_body_ids"]]
        obj_vel = data.qvel[:, self.ids.object_free_dof_addr:self.ids.object_free_dof_addr + 3]
        return current_full, hand_pos, hand_xyzw, object_pos, object_xyzw, obj_vel, fingertips, contact_bodies

    def _contacts(self, data: Any) -> Any:
        impl = data._impl
        return reduce_dexhand_contacts(
            nacon=impl.nacon, nefc=impl.nefc, contact_geom=impl.contact__geom, contact_worldid=impl.contact__worldid,
            contact_dim=impl.contact__dim, contact_efc_address=impl.contact__efc_address, contact_friction=impl.contact__friction,
            contact_frame=impl.contact__frame, efc_force=impl.efc__force, ngeom=self.model.ngeom,
            geom_to_contact_body=self.geom_to_contact_body, object_geom_flags=self.object_geom_flags,
        )

    def _observe_state(self, state: _BatchState) -> tuple[Any, Any]:
        t = self._tables
        current, hp, hq, op, oq, ov, tips, bodies = self._features(state.data)
        contacts = self._contacts(state.data)
        target = gather_trajectory(values=t["object_pos"], frame=state.frame, lengths=t["lengths"])
        target_quat = gather_trajectory(values=t["object_quat"], frame=state.frame, lengths=t["lengths"])
        target_next = gather_trajectory(values=t["object_pos"], frame=state.frame + 5, lengths=t["lengths"])
        episode_time = self.jp.clip(state.episode_count.astype(self.jp.float32) * self.cfg.physics.control_dt / self.jp.maximum(t["lengths"].astype(self.jp.float32) * self.cfg.physics.control_dt, 1e-6), 0., 1.)
        obs, valid = build_device_observation(
            action_onehot=t["onehot"], geometry=t["geometry"], current_full=current, active_finger_lower=t["finger_lower"], active_finger_upper=t["finger_upper"],
            hand_position=hp, hand_quat_xyzw=hq, fingertip_positions=tips, contact_body_positions=bodies, episode_time=episode_time,
            object_position=op, object_quat_xyzw=oq, object_linear_velocity=ov, target_position=target, target_quat_xyzw=target_quat,
            target_next_5=target_next, expected_contact_mask=t["expected"], hand_contact_force=contacts.hand, object_contact_force=contacts.object,
            base_offset=state.base_offset, joint_offset=state.joint_offset, point_cloud_local=t["cloud"],
        )
        return obs.astype(self.jp.float32), valid & contacts.valid

    def _transition(self, state: _BatchState, actions: Any) -> tuple[_BatchState, DeviceBatchTransition]:
        t = self._tables
        actions = self.jp.asarray(actions, dtype=self.jp.float32)
        reference = gather_trajectory(values=t["hand"], frame=state.frame, lengths=t["lengths"])
        current, _, _, op0, oq0, _, tips0, _ = self._features(state.data)
        control = apply_device_control(
            action=actions, reference_active=reference, current_full=current, previous_active=state.previous_active,
            base_offset=state.base_offset, joint_offset=state.joint_offset, active_joint_mask=t["joint_mask"], frame=state.frame,
            start_frame=self.jp.zeros_like(state.frame), last_contact_frame=t["last_contact"], use_residual=self.cfg.use_residual,
            early_phase_frames=self.cfg.early_phase_frames, base_decay=self.cfg.base_position_decay, post_contact_base_decay=self.cfg.post_contact_base_position_decay,
            joint_decay=self.cfg.joint_decay, post_contact_joint_decay=self.cfg.post_contact_joint_decay, base_scales=self.cfg.base_position_scales,
            base_max_offset=self.cfg.base_position_max_offset, active_max_deltas=t["max_delta"], active_lower=t["active_lower"], active_upper=t["active_upper"],
            route_upper=t["route_upper"], actuator_ids=t["actuator_ids"], nu=self.model.nu,
        )
        data = self._forward_world(state.data.replace(ctrl=control.ctrl))
        for _ in range(10):
            data = self._step_world(data)
        next_state = _BatchState(data=data, frame=self.jp.minimum(state.frame + 1, t["lengths"] - 1), episode_count=state.episode_count + 1,
            previous_active=control.applied_active, base_offset=control.base_offset, joint_offset=control.joint_offset)
        obs, obs_valid = self._observe_state(next_state)
        _, _, _, op, oq, ov, tips, _ = self._features(data)
        contacts = self._contacts(data)
        target = gather_trajectory(values=t["object_pos"], frame=next_state.frame, lengths=t["lengths"])
        target_quat = gather_trajectory(values=t["object_quat"], frame=next_state.frame, lengths=t["lengths"])
        penalty = fingertip_inside_object_penalty(fingertips_world=tips, object_position=op, object_quat_xyzw=oq, primitive_type=t["primitive_type"], primitive_params=t["primitive"], active_fingertip_mask=t["tip_mask"], min_depth=self.cfg.fingertip_inside_object_min_depth, max_depth=self.cfg.fingertip_inside_object_max_depth, max_penalty=self.cfg.max_fingertip_inside_object_penalty)
        reward = compute_device_reward_done(
            frame=next_state.frame, start_frame=self.jp.zeros_like(next_state.frame), last_contact_frame=t["last_contact"], lengths=t["lengths"],
            object_position=op, target_position=target, object_quat_xyzw=oq, target_quat_xyzw=target_quat, object_linear_velocity=ov,
            hand_contact_force=contacts.hand, expected_contact_mask=t["expected"], base_offset=next_state.base_offset, joint_offset=next_state.joint_offset,
            active_joint_mask=t["joint_mask"], fingertip_inside_penalty=penalty, rotation_disabled=t["rotation_disabled"], early_phase_frames=self.cfg.early_phase_frames,
            contact_force_threshold=self.cfg.contact_force_threshold, max_contact_reward=self.cfg.max_contact_reward, max_stability_reward=self.cfg.max_object_stability_reward,
            stability_scale=self.cfg.object_stability_velocity_scale, object_terminal_threshold=self.cfg.object_position_terminal_threshold,
            reward_weights=t["reward_weights"], termination_failure_reward=self.cfg.termination_failure_penalty, termination_timeout_reward=self.cfg.termination_timeout_penalty,
        )
        valid = control.valid & obs_valid & contacts.valid & reward.valid & self.jp.all(self.jp.isfinite(actions))
        reason = self.jp.where(reward.terminated, 1, self.jp.where(reward.truncated, 2, 0)).astype(self.jp.int32)
        return next_state, DeviceBatchTransition(obs, reward.total.astype(self.jp.float32), reward.terminated, reward.truncated, reason,
            reward.weighted.astype(self.jp.float32), control.reference_active, control.corrected_active, control.applied_active, control.full_target, control.ctrl, valid)

    def _replace_rows(self, current: Any, cached: Any, mask: Any) -> Any:
        if not hasattr(current, "ndim") or current.ndim == 0 or current.shape[0] != self.num_envs:
            return current
        shape = (self.num_envs,) + (1,) * (current.ndim - 1)
        return self.jp.where(mask.reshape(shape), cached, current)

    def _reset_mask(self, state: _BatchState, mask: Any) -> _BatchState:
        mask = self.jp.asarray(mask, dtype=bool)

        def reset_rows(current: _BatchState) -> _BatchState:
            data = self.jax.tree_util.tree_map(
                lambda value, cached: self._replace_rows(value, cached, mask), current.data, self._settled_data
            )
            # Global Warp contact leaves are not row-owned. Rebuild derived
            # fields only when at least one world was restored.
            data = self._forward_world(data)
            reset_previous = self._reset_previous_active
            return _BatchState(
                data=data,
                frame=self.jp.where(mask, 1, current.frame),
                episode_count=self.jp.where(mask, 0, current.episode_count),
                previous_active=self.jp.where(mask[:, None], reset_previous, current.previous_active),
                base_offset=self.jp.where(mask[:, None], 0., current.base_offset),
                joint_offset=self.jp.where(mask[:, None], 0., current.joint_offset),
            )

        # Keep this predicate on device: a host `any()` would synchronize every
        # rollout step, while the identity branch preserves the complete state.
        return self.jax.lax.cond(self.jp.any(mask), reset_rows, lambda current: current, state)

    def _reset_observation(self, state: _BatchState) -> DeviceBatchTransition:
        obs, valid = self._observe_state(state)
        zeros = self.jp.zeros((self.num_envs,), dtype=self.jp.float32)
        return DeviceBatchTransition(obs, zeros, self.jp.zeros((self.num_envs,), dtype=bool), self.jp.zeros((self.num_envs,), dtype=bool), self.jp.zeros((self.num_envs,), dtype=self.jp.int32),
            self.jp.zeros((self.num_envs, 10), dtype=self.jp.float32), state.previous_active, state.previous_active, state.previous_active, self.jp.zeros((self.num_envs, 26), dtype=self.jp.float32), self.jp.zeros((self.num_envs, self.model.nu), dtype=self.jp.float32), valid)

    def reset_device(self) -> DeviceBatchTransition:
        self.state = self._initial_state
        return self._reset_observation_fn(self.state)

    def step_device(self, actions: Any) -> DeviceBatchTransition:
        self.state, transition = self._transition_fn(self.state, actions)
        return transition

    def reset_done_device(self, mask: Any) -> DeviceBatchTransition:
        self.state = self._reset_mask_fn(self.state, mask)
        return self._reset_observation_fn(self.state)

    def render_snapshot(self) -> BatchRenderSnapshot:
        """Materialize only render-relevant per-world state on the host.

        Deliberately do not inspect ``data._impl``: its contact/constraint
        arrays scale with batch capacity and are logging-only telemetry.
        """
        data = self.state.data
        return BatchRenderSnapshot(
            qpos=np.asarray(data.qpos), qvel=np.asarray(data.qvel), ctrl=np.asarray(data.ctrl),
            frame=np.asarray(self.state.frame), episode_count=np.asarray(self.state.episode_count),
        )

    def telemetry(self) -> dict[str, Any]:
        """Opt-in host snapshot of requested/actual capacities and high water.

        Callers must judge saturation against the actual Warp buffer shapes;
        the runtime rejects either contact or constraint utilization at 80%.
        """
        impl = self.state.data._impl
        nefc = np.asarray(impl.nefc)
        return {
            "frame": np.asarray(self.state.frame), "episode_count": np.asarray(self.state.episode_count),
            "qpos": np.asarray(self.state.data.qpos), "qvel": np.asarray(self.state.data.qvel),
            "requested_global_contact_capacity": self.contact_capacity,
            "requested_ccd_capacity": self.ccd_capacity,
            "requested_per_world_constraint_capacity": self.per_world_constraint_capacity,
            "actual_contact_buffer_shape": tuple(np.asarray(impl.contact__geom).shape),
            "actual_ccd_capacity": int(impl.naccdmax),
            "actual_ncollision_shape": tuple(np.asarray(impl.ncollision).shape),
            "actual_constraint_buffer_shape": tuple(np.asarray(impl.efc__force).shape),
            "contact_high_water": int(np.asarray(impl.nacon).max(initial=0)),
            "ncollision_high_water": int(np.asarray(impl.ncollision).max(initial=0)),
            "constraint_high_water_per_world": nefc,
            "constraint_high_water": int(nefc.max(initial=0)),
            "host_initial_nefc": self._host_initial_nefc,
            "minimum_per_world_constraint_capacity": self._minimum_per_world_constraint_capacity,
        }

    def close(self) -> None:
        self.state = None
