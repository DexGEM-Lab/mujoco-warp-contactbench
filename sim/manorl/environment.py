"""Bounded MJX-Warp ManoRL physical producer and vector environment.

The source ABI consumes world-frame rigid-body contact forces.  MJX-Warp keeps
its contact records in a global per-batch buffer, so this module decodes the
pyramidal contact constraints from the pinned 3.10.0 Warp layout instead of
using ``mjx.get_data``.  The latter is unsuitable here because its host contact
array retains capacity-padding entries after the solved records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sim.manorl.abi import (
    ResidualActionConfig,
    TerminationResult,
    check_termination,
    early_phase_mask,
    process_residual_actions,
)
from sim.manorl.assets import ASSET_ROOT, compile_model, object_collision_vertices
from sim.manorl.contracts import (
    FLOOR_TOP_Z,
    KEYPOINT_NAMES,
    OBJECT_BODY_NAME,
    OBJECT_FREE_JOINT_NAME,
    PHYSICS_SUBSTEPS_PER_TARGET,
    ServoConfig,
)
from sim.manorl.mjx_sim import CONTACT_CAPACITY, CONSTRAINT_CAPACITY, command_target
from sim.manorl.observations import (
    CURRENT_SOURCE_COMPATIBILITY,
    POINT_COUNT,
    ObservationCompatibility,
    ObservationResult,
    ObservationState,
    PointCloudTemplate,
    build_observation,
    expected_contact_mask_from_keypoint_ids,
    geometry_encoding,
    quat_rotate_xyzw,
)
from sim.manorl.rewards import RewardDiagnostics, RewardState, compute_rewards
from sim.manorl.trajectory import ReferenceTrajectory, TrajectoryBatch, wxyz_to_xyzw, xyzw_to_wxyz

_FINGERTIP_NAMES = ("thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip")
_FINGERTIP_LOCAL_OFFSETS = np.asarray(
    (
        (-0.028633, -0.004191, 0.023667),
        (-0.027384, -0.000215, -0.000966),
        (-0.027549, 0.000501, -0.004523),
        (-0.026165, -0.000075, -0.007781),
        (-0.018526, -0.001581, -0.011018),
    ),
    dtype=np.float64,
)
_SOURCE_GRASP_MAPPING = ASSET_ROOT / "cube1" / "grasp_mapping.yaml"


@dataclass(frozen=True)
class EnvironmentConfig:
    """Resolved bounded environment settings for the accepted cube trajectory."""

    num_envs: int = 1
    device: str = "cpu"
    servo: ServoConfig = ServoConfig()
    compatibility: ObservationCompatibility = CURRENT_SOURCE_COMPATIBILITY
    residual_enabled: bool = True
    residual_action: ResidualActionConfig = ResidualActionConfig()
    max_deviation_distance: float = 0.10
    deviation_penalty: float = 25.0
    episode_length: int = 600
    contact_capacity: int = CONTACT_CAPACITY
    constraint_capacity: int = CONSTRAINT_CAPACITY
    point_seed: int = 42

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'")
        if not isinstance(self.compatibility, ObservationCompatibility):
            raise TypeError("compatibility must be an ObservationCompatibility")
        if not isinstance(self.servo, ServoConfig):
            raise TypeError("servo must be a ServoConfig")
        if not isinstance(self.residual_enabled, bool):
            raise TypeError("residual_enabled must be bool")
        if self.max_deviation_distance < 0 or self.deviation_penalty < 0:
            raise ValueError("termination thresholds must be non-negative")
        if self.episode_length < 1:
            raise ValueError("episode_length must be positive")
        if self.contact_capacity < 1 or self.constraint_capacity < 1:
            raise ValueError("MJX contact and constraint capacities must be positive")
        if self.compatibility.point_template_mode == "static_seed_42" and self.point_seed != 42:
            raise ValueError("static_seed_42 compatibility requires point_seed=42")


@dataclass(frozen=True)
class PhysicalSnapshot:
    """Source-order physical values extracted from one batched MJX state."""

    mano_dof_pos: NDArray[np.float64]
    hand_position: NDArray[np.float64]
    hand_orientation_xyzw: NDArray[np.float64]
    object_position: NDArray[np.float64]
    object_orientation_xyzw: NDArray[np.float64]
    object_linear_velocity: NDArray[np.float64]
    hand_keypoint_positions: NDArray[np.float64]
    fingertip_positions: NDArray[np.float64]
    hand_keypoint_contact_forces: NDArray[np.float64]
    object_contact_force: NDArray[np.float64]
    contact_count: NDArray[np.int64]


@dataclass(frozen=True)
class TransitionSnapshot:
    """Immutable evidence for one complete control transition, including delayed reset state."""

    control_call: int
    raw_actions: NDArray[np.float64]
    command_reference_indices: NDArray[np.int64]
    command_targets: NDArray[np.float64]
    processed_targets: NDArray[np.float64]
    controller_targets: NDArray[np.float64]
    reset_applied: NDArray[np.bool_]
    progress: NDArray[np.int64]
    trajectory_steps: NDArray[np.int64]
    target_indices: NDArray[np.int64]
    physical: PhysicalSnapshot
    observation: ObservationResult
    reward: RewardDiagnostics
    termination: TerminationResult


def _normalized_xyzw(quaternions_wxyz: NDArray[object]) -> NDArray[np.float64]:
    quaternions = wxyz_to_xyzw(np.asarray(quaternions_wxyz, dtype=np.float64))
    norm = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(~np.isfinite(quaternions)) or np.any(norm <= 1e-12):
        raise ValueError("MJX produced an invalid world quaternion")
    return quaternions / norm


def _source_surface_points(seed: int) -> NDArray[np.float64]:
    """Reproduce trimesh.sample.sample_surface(mesh, 64, seed=seed) for cube1."""

    triangles = object_collision_vertices().reshape(-1, 3, 3)
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    areas = np.linalg.norm(np.cross(edge_a, edge_b), axis=1) * 0.5
    if np.any(areas <= 0.0) or not np.all(np.isfinite(areas)):
        raise ValueError("cube surface triangles must have positive finite area")
    generator = np.random.default_rng(seed)
    face_indices = np.searchsorted(np.cumsum(areas), generator.random(POINT_COUNT) * areas.sum())
    origins = triangles[face_indices, 0]
    vectors = triangles[face_indices, 1:] - origins[:, None, :]
    lengths = generator.random((POINT_COUNT, 2, 1))
    lengths[lengths.sum(axis=1).reshape(-1) > 1.0] -= 1.0
    return origins + (vectors * np.abs(lengths)).sum(axis=1)


def _source_surface_template(seed: int) -> PointCloudTemplate:
    points = _source_surface_points(seed)
    lower, upper = points.min(axis=0), points.max(axis=0)
    scale = np.maximum((upper - lower) / 2.0, 1e-6)
    normalized = (points - (upper + lower) / 2.0) / scale
    return PointCloudTemplate(normalized, mode="static_seed_42", normalized=True, scale=scale)


def _dynamic_surface_template(generator: np.random.Generator) -> NDArray[np.float64]:
    """Use the source mesh sampler with the caller's explicit reset RNG."""

    seed = int(generator.integers(0, np.iinfo(np.int64).max))
    return _source_surface_points(seed)


def _load_expected_keypoint_ids() -> NDArray[np.int64]:
    # This accepted migration has exactly one source-pinned mapping.  Avoid a
    # parser dependency because the approved local runtime lacks PyYAML even
    # though it is present in the lock; a future multi-object environment must
    # replace this exact-content guard with a validated YAML loader.
    expected = 'cube1:\n  "01": [thumb3, index3]\n'
    if _SOURCE_GRASP_MAPPING.read_text(encoding="utf-8") != expected:
        raise ValueError("the pinned cube1 action-01 grasp mapping changed")
    aliases = ["thumb3", "index3"]
    alias_map = {
        "thumb3": "thumb_ip",
        "index3": "index_dip",
        "middle3": "middle_dip",
        "ring3": "ring_dip",
        "pinky3": "pinky_dip",
    }
    names = [alias_map.get(str(alias)) for alias in aliases]
    if any(name is None for name in names):
        raise ValueError("grasp mapping contains an unsupported source keypoint alias")
    return np.asarray([KEYPOINT_NAMES.index(name) for name in names], dtype=np.int64)


def _active_joint_mask(expected_mask: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Mirror FingerMaskManager: only fingers with expected keypoints are active."""

    finger_ranges = {
        "thumb": slice(0, 4),
        "index": slice(4, 8),
        "middle": slice(8, 12),
        "ring": slice(12, 16),
        "pinky": slice(16, 20),
    }
    batch = len(expected_mask)
    active = np.zeros((batch, 20), dtype=bool)
    for keypoint_index, keypoint_name in enumerate(KEYPOINT_NAMES):
        finger = keypoint_name.split("_", maxsplit=1)[0]
        if finger in finger_ranges:
            active[expected_mask[:, keypoint_index] > 0.5, finger_ranges[finger]] = True
    return active


class MjxWarpPhysicalProducer:
    """Convert pinned MJX-Warp state into source-order physical fields.

    Contact decoding relies on the installed 3.10.0 Warp data layout.  The
    source code's ``smooth.py`` converts a local contact wrench with
    ``force.reshape((-1, 3)) @ frame`` and applies it negatively to ``geom1``
    and positively to ``geom2``.  This producer uses that exact orientation and
    sign convention, then aggregates only the declared source keypoint bodies.
    """

    def __init__(self, mujoco: Any, model: Any) -> None:
        self.mujoco = mujoco
        self.model = model
        self.object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
        object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_FREE_JOINT_NAME)
        if self.object_body_id < 0 or object_joint_id < 0:
            raise ValueError("compiled cube body/free joint is absent")
        self.object_qpos_address = int(model.jnt_qposadr[object_joint_id])
        self.object_qvel_address = int(model.jnt_dofadr[object_joint_id])
        self.keypoint_geom_ids: list[int] = []
        self.keypoint_body_ids: list[int] = []
        self.geom_to_keypoint: dict[int, int] = {}
        for keypoint_index, keypoint_name in enumerate(KEYPOINT_NAMES):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{keypoint_name}_collision")
            if geom_id < 0:
                raise ValueError(f"compiled source keypoint geom is absent: {keypoint_name}")
            self.keypoint_geom_ids.append(geom_id)
            body_id = int(model.geom_bodyid[geom_id])
            self.keypoint_body_ids.append(body_id)
            self.geom_to_keypoint[geom_id] = keypoint_index
        if len(set(self.keypoint_geom_ids)) != len(KEYPOINT_NAMES):
            raise ValueError("source keypoint geom mapping is not one-to-one")
        self.fingertip_body_ids = np.asarray(
            [self.keypoint_body_ids[KEYPOINT_NAMES.index(name)] for name in _FINGERTIP_NAMES], dtype=np.int64
        )
        self.object_geom_ids = {
            geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == self.object_body_id
        }
        if len(self.object_geom_ids) != 1:
            raise ValueError("the bounded cube contract requires exactly one object collision geom")

    def _contact_arrays(
        self, data: Any, batch: int
    ) -> tuple[
        int,
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        impl = data._impl
        required = (
            "nacon",
            "nefc",
            "contact__geom",
            "contact__worldid",
            "contact__dim",
            "contact__efc_address",
            "contact__friction",
            "contact__frame",
            "efc__force",
        )
        missing = [name for name in required if not hasattr(impl, name)]
        if missing:
            raise RuntimeError(
                "the installed MJX-Warp contact ABI is incompatible with this producer: "
                + ", ".join(missing)
            )
        nacon = np.asarray(impl.nacon, dtype=np.int64).reshape(-1)
        if nacon.shape != (1,):
            raise RuntimeError(f"MJX-Warp global contact count must have shape (1,), got {nacon.shape}")
        count = int(nacon[0])
        nefc_values = np.asarray(impl.nefc, dtype=np.int64).reshape(-1)
        if nefc_values.shape not in {(1,), (batch,)}:
            raise RuntimeError(
                "MJX-Warp constraint count must be scalar or one value per world, "
                f"got {nefc_values.shape}"
            )
        geom = np.asarray(impl.contact__geom, dtype=np.int64)
        world = np.asarray(impl.contact__worldid, dtype=np.int64)
        dimension = np.asarray(impl.contact__dim, dtype=np.int64)
        addresses = np.asarray(impl.contact__efc_address, dtype=np.int64)
        friction = np.asarray(impl.contact__friction, dtype=np.float64)
        frame = np.asarray(impl.contact__frame, dtype=np.float64)
        forces = np.asarray(impl.efc__force, dtype=np.float64)
        capacity = len(geom)
        if not 0 <= count <= capacity:
            raise RuntimeError(f"MJX-Warp contact count {count} exceeds allocated capacity {capacity}")
        if count == capacity:
            raise RuntimeError("MJX-Warp contact capacity saturated; contact aggregation is invalid")
        expected_shapes = (
            geom.shape == (capacity, 2),
            world.shape == (capacity,),
            dimension.shape == (capacity,),
            addresses.shape == (capacity, 4),
            friction.shape == (capacity, 5),
            frame.shape == (capacity, 3, 3),
            forces.ndim == 2 and forces.shape[0] == batch,
        )
        if not all(expected_shapes):
            raise RuntimeError("MJX-Warp private contact buffer shapes differ from the pinned ABI")
        if np.any(nefc_values >= forces.shape[1]):
            raise RuntimeError("MJX-Warp constraint capacity saturated; contact forces are invalid")
        return count, geom, world, dimension, addresses, friction, frame, forces

    def extract(self, data: Any) -> PhysicalSnapshot:
        qpos = np.asarray(data.qpos, dtype=np.float64)
        qvel = np.asarray(data.qvel, dtype=np.float64)
        xpos = np.asarray(data.xpos, dtype=np.float64)
        xquat = np.asarray(data.xquat, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self.model.nq or qvel.shape != (len(qpos), self.model.nv):
            raise ValueError("producer requires batched MJX qpos/qvel state")
        batch = len(qpos)
        if xpos.shape != (batch, self.model.nbody, 3) or xquat.shape != (batch, self.model.nbody, 4):
            raise RuntimeError("MJX world-body transform shapes differ from the compiled model")
        keypoint_bodies = np.asarray(self.keypoint_body_ids, dtype=np.int64)
        keypoints = xpos[:, keypoint_bodies].copy()
        keypoint_quats = _normalized_xyzw(xquat[:, keypoint_bodies])
        fingertip_indices = np.asarray([KEYPOINT_NAMES.index(name) for name in _FINGERTIP_NAMES], dtype=np.int64)
        fingertips = keypoints[:, fingertip_indices] + quat_rotate_xyzw(
            np.broadcast_to(keypoint_quats[:, fingertip_indices], (batch, len(_FINGERTIP_NAMES), 4)),
            np.broadcast_to(_FINGERTIP_LOCAL_OFFSETS, (batch, len(_FINGERTIP_NAMES), 3)),
        )
        forces = np.zeros((batch, len(KEYPOINT_NAMES), 3), dtype=np.float64)
        object_force = np.zeros((batch, 3), dtype=np.float64)
        count, geom, world, dimension, addresses, friction, frame, constraint_force = self._contact_arrays(data, batch)
        per_world_count = np.zeros(batch, dtype=np.int64)
        for contact_id in range(count):
            world_id = int(world[contact_id])
            if not 0 <= world_id < batch:
                raise RuntimeError(f"contact {contact_id} references invalid MJX world {world_id}")
            if int(dimension[contact_id]) != 3:
                raise RuntimeError("the bounded MANO scene requires condim=3 contacts")
            address = addresses[contact_id]
            if np.any(address < 0) or np.any(address >= constraint_force.shape[1]):
                raise RuntimeError(f"contact {contact_id} has an invalid pyramidal constraint address")
            pyramid = constraint_force[world_id, address]
            local_force = np.asarray(
                (
                    pyramid.sum(),
                    (pyramid[0] - pyramid[1]) * friction[contact_id, 0],
                    (pyramid[2] - pyramid[3]) * friction[contact_id, 1],
                ),
                dtype=np.float64,
            )
            world_force = local_force @ frame[contact_id]
            first, second = (int(geom[contact_id, 0]), int(geom[contact_id, 1]))
            if first in self.geom_to_keypoint:
                forces[world_id, self.geom_to_keypoint[first]] -= world_force
            if second in self.geom_to_keypoint:
                forces[world_id, self.geom_to_keypoint[second]] += world_force
            if first in self.object_geom_ids:
                object_force[world_id] -= world_force
            if second in self.object_geom_ids:
                object_force[world_id] += world_force
            per_world_count[world_id] += 1
        if not np.all(np.isfinite(forces)) or not np.all(np.isfinite(object_force)):
            raise RuntimeError("MJX contact decoder produced a non-finite world force")
        return PhysicalSnapshot(
            mano_dof_pos=qpos[:, :26].copy(),
            hand_position=xpos[:, self.keypoint_body_ids[0]].copy(),
            hand_orientation_xyzw=_normalized_xyzw(xquat[:, self.keypoint_body_ids[0]]),
            object_position=xpos[:, self.object_body_id].copy(),
            object_orientation_xyzw=_normalized_xyzw(xquat[:, self.object_body_id]),
            object_linear_velocity=qvel[:, self.object_qvel_address : self.object_qvel_address + 3].copy(),
            hand_keypoint_positions=keypoints,
            fingertip_positions=fingertips,
            hand_keypoint_contact_forces=forces,
            object_contact_force=object_force,
            contact_count=per_world_count,
        )


class MujocoManoEnvironment:
    """A bounded MJX-Warp vector environment for the accepted ManoRL trajectory.

    This is intentionally an environment ABI slice, not an RL adapter.  It
    returns source-shaped observation/reward/reset values and omits privileged
    state, model, checkpoint, and training concerns.
    """

    def __init__(
        self, trajectory: ReferenceTrajectory | TrajectoryBatch, config: EnvironmentConfig = EnvironmentConfig()
    ) -> None:
        if not isinstance(trajectory, (ReferenceTrajectory, TrajectoryBatch)):
            raise TypeError("trajectory must be a ReferenceTrajectory or TrajectoryBatch")
        if not isinstance(config, EnvironmentConfig):
            raise TypeError("config must be an EnvironmentConfig")
        trajectories = (
            (trajectory,) * config.num_envs
            if isinstance(trajectory, ReferenceTrajectory)
            else trajectory.trajectories
        )
        if len(trajectories) != config.num_envs:
            raise ValueError("trajectory batch size must equal config.num_envs")
        if any(len(item.q_ref) < 2 for item in trajectories):
            raise ValueError("environment requires at least two source references per environment")
        try:
            import jax
            from mujoco import mjx
        except ImportError as exc:
            raise RuntimeError("jax and mujoco-mjx are required for the MJX environment") from exc
        # ``gpu`` is the repository's public device mode. JAX 0.10 treats
        # that alias as CUDA plus ROCm candidates; this CUDA/NVIDIA runtime
        # must request its concrete platform to avoid an unavailable ROCm
        # backend becoming a startup failure.
        jax_platform = "cuda" if config.device == "gpu" else "cpu"
        devices = jax.devices(jax_platform)
        if not devices:
            raise RuntimeError(f"no JAX {jax_platform} device is available")
        # ``trajectory`` remains the first assignment for legacy single-world
        # diagnostics. Production gathers below always use the per-env tables.
        self.trajectory = trajectories[0]
        self.trajectories = tuple(trajectories)
        self.config = config
        self.jax = jax
        self.jp = jax.numpy
        self.mjx = mjx
        self.device = devices[0]
        self.mujoco, self.model = compile_model(config.servo)
        minimum_contact_capacity = 31 * config.num_envs
        if config.contact_capacity < minimum_contact_capacity:
            raise ValueError(
                f"contact_capacity {config.contact_capacity} is below {minimum_contact_capacity} required for {config.num_envs} cube1 worlds"
            )
        self.producer = MjxWarpPhysicalProducer(self.mujoco, self.model)
        self.mjx_model = mjx.put_model(self.model, device=self.device, impl="warp")
        if str(self.mjx_model.impl).lower().split(".")[-1] != "warp":
            raise RuntimeError(f"MJX did not select Warp: {self.mjx_model.impl}")
        self.joint_lower = self.model.jnt_range[:26, 0].astype(np.float64, copy=True)
        self.joint_upper = self.model.jnt_range[:26, 1].astype(np.float64, copy=True)
        self._step_fn = jax.jit(jax.vmap(lambda world: mjx.step(self.mjx_model, world)))
        self._forward_fn = jax.jit(jax.vmap(lambda world: mjx.forward(self.mjx_model, world)))
        self._build_reference_tables()
        self._reset_qpos = self._initial_qpos()
        # Warp contact implementation metadata is static for the whole batch.
        # Replicate one capacity-configured world, then replace only dynamic
        # per-world qpos/qvel/ctrl below during reset.
        single_data = mjx.put_data(
            self.model,
            self._initial_host_data(0),
            device=self.device,
            impl="warp",
            naconmax=config.contact_capacity,
            njmax=config.constraint_capacity,
        )
        batch_index = jax.device_put(self.jp.arange(config.num_envs), self.device)
        self.data = jax.vmap(lambda _: single_data)(batch_index)
        self.expected_keypoint_ids = _load_expected_keypoint_ids()
        self.expected_contact_mask = expected_contact_mask_from_keypoint_ids(
            np.broadcast_to(self.expected_keypoint_ids, (config.num_envs, len(self.expected_keypoint_ids))),
            config.num_envs,
        )
        self.expected_contact_weights = self.expected_contact_mask.copy()
        self.active_joint_mask = _active_joint_mask(self.expected_contact_mask)
        self.action_ids = np.ones(config.num_envs, dtype=np.int64)
        self.object_geometry = geometry_encoding(
            object_name="cube1",
            geometry_type="box",
            dimensions=np.ptp(object_collision_vertices(), axis=0),
        )
        self.object_support_points = object_collision_vertices().copy()
        self.object_gravity_force = float(self.model.body_mass[self.producer.object_body_id] * abs(self.model.opt.gravity[2]))
        self._point_rngs = [np.random.default_rng(config.point_seed + index) for index in range(config.num_envs)]
        self._static_template = _source_surface_template(42)
        self._dynamic_templates: NDArray[np.float64] | None = None
        self.progress = np.zeros(config.num_envs, dtype=np.int64)
        self.trajectory_steps = np.zeros(config.num_envs, dtype=np.int64)
        self.cumulative_offset = np.zeros((config.num_envs, 3), dtype=np.float64)
        self.cumulative_joint_offset = np.zeros((config.num_envs, 20), dtype=np.float64)
        self.reset_mask = np.zeros(config.num_envs, dtype=bool)
        self.episode_returns = np.zeros(config.num_envs, dtype=np.float64)
        self.last_physical: PhysicalSnapshot | None = None
        self.last_observation: ObservationResult | None = None
        self.last_reward: RewardDiagnostics | None = None
        self.last_termination: TerminationResult | None = None
        self.last_controller_targets = np.zeros((config.num_envs, 26), dtype=np.float64)
        self.control_call = 0
        self.last_transition: TransitionSnapshot | None = None
        self.reset()

    def _build_reference_tables(self) -> None:
        self.trajectory_lengths = np.asarray([len(item.q_ref) for item in self.trajectories], dtype=np.int64)
        max_length = int(self.trajectory_lengths.max())
        def pad(name: str) -> NDArray[np.float64]:
            values = [getattr(item, name) for item in self.trajectories]
            return np.stack(
                [
                    np.pad(value, ((0, max_length - len(value)), (0, 0)), mode="edge")
                    for value in values
                ]
            )
        self.reference_q = pad("q_ref")
        self.reference_object_pos = pad("object_pos")
        self.reference_object_quat_xyzw = pad("object_quat_xyzw")
        self.reference_source_indices = np.stack(
            [np.pad(item.source_indices, (0, max_length - len(item.source_indices)), mode="edge") for item in self.trajectories]
        )
        self.contact_start_frames = np.asarray(
            [item.identity.movement_start_raw - item.source_indices[0] for item in self.trajectories], dtype=np.int64
        )
        self.contact_end_frames = np.asarray(
            [item.identity.movement_end_raw - item.source_indices[0] for item in self.trajectories], dtype=np.int64
        )
        if np.any(self.contact_start_frames < 0) or np.any(self.contact_end_frames >= self.trajectory_lengths):
            raise ValueError("trajectory movement window does not map into its reference slice")
        # Compatibility diagnostics still expose scalar values for a single/shared trajectory.
        self.contact_start_frame = int(self.contact_start_frames[0])
        self.contact_end_frame = int(self.contact_end_frames[0])

    def _initial_qpos(self) -> NDArray[np.float64]:
        qpos = np.zeros((self.config.num_envs, self.model.nq), dtype=np.float64)
        qpos[:, :26] = self.reference_q[:, 0]
        address = self.producer.object_qpos_address
        qpos[:, address : address + 3] = self.reference_object_pos[:, 0]
        qpos[:, address + 3 : address + 7] = xyzw_to_wxyz(self.reference_object_quat_xyzw[:, 0])
        return qpos

    def _initial_host_data(self, env_id: int) -> Any:
        data = self.mujoco.MjData(self.model)
        self.mujoco.mj_resetData(self.model, data)
        data.qpos[:] = self._reset_qpos[env_id]
        data.qvel[:] = 0.0
        data.ctrl[:] = self.reference_q[env_id, 0]
        self.mujoco.mj_forward(self.model, data)
        return data

    def _set_dynamic_templates(self, env_ids: NDArray[np.int64]) -> None:
        if self.config.compatibility.point_template_mode != "dynamic_reset":
            return
        if self._dynamic_templates is None:
            self._dynamic_templates = np.zeros((self.config.num_envs, POINT_COUNT, 3), dtype=np.float64)
        for env_id in env_ids:
            self._dynamic_templates[env_id] = _dynamic_surface_template(self._point_rngs[int(env_id)])

    def reseed_point_templates(self, seed: int) -> None:
        """Set deterministic reset-local RNGs without changing static templates."""

        if not isinstance(seed, (int, np.integer)):
            raise TypeError("point-template seed must be an integer")
        self._point_rngs = [
            np.random.default_rng(int(seed) + index) for index in range(self.config.num_envs)
        ]

    def host_data(self, env_id: int = 0) -> Any:
        """Mirror one MJX-Warp world into native data for rendering only.

        Contact extraction deliberately never uses this host representation:
        the production contact decoder reads the Warp capacity buffers directly.
        """

        if not 0 <= env_id < self.config.num_envs:
            raise IndexError(f"env_id must be in [0, {self.config.num_envs - 1}]")
        return self.host_data_batch()[env_id]

    def host_data_batch(self) -> list[Any]:
        """Transfer the complete batch once for tiled rendering diagnostics."""

        host_data = self.mjx.get_data(self.model, self.data)
        if not isinstance(host_data, list) or len(host_data) != self.config.num_envs:
            raise RuntimeError("batched MJX environment did not produce one host state per world")
        return host_data

    def _point_template(self) -> PointCloudTemplate:
        if self.config.compatibility.point_template_mode == "static_seed_42":
            return self._static_template
        if self._dynamic_templates is None:
            raise RuntimeError("dynamic point templates were not initialized")
        return PointCloudTemplate(self._dynamic_templates.copy(), mode="dynamic_reset")

    def _reset_indices(self, env_ids: NDArray[np.int64]) -> None:
        if len(env_ids) == 0:
            return
        qpos = np.asarray(self.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(self.data.qvel, dtype=np.float64).copy()
        ctrl = np.asarray(self.data.ctrl, dtype=np.float64).copy()
        qpos[env_ids] = self._reset_qpos[env_ids]
        qvel[env_ids] = 0.0
        ctrl[env_ids] = self.reference_q[env_ids, 0]
        self.data = self.data.replace(
            qpos=self.jax.device_put(self.jp.asarray(qpos), self.device),
            qvel=self.jax.device_put(self.jp.asarray(qvel), self.device),
            ctrl=self.jax.device_put(self.jp.asarray(ctrl), self.device),
        )
        self.data = self._forward_fn(self.data)
        self.progress[env_ids] = 0
        self.trajectory_steps[env_ids] = 0
        self.cumulative_offset[env_ids] = 0.0
        self.cumulative_joint_offset[env_ids] = 0.0
        self.reset_mask[env_ids] = False
        self.episode_returns[env_ids] = 0.0
        self._set_dynamic_templates(env_ids)

    def _target_indices(self) -> NDArray[np.int64]:
        return np.minimum(np.maximum(self.trajectory_steps, 0), self.trajectory_lengths - 1)

    def _reference_gather(self, table: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
        return table[np.arange(self.config.num_envs), indices]

    def _build_observation(self, physical: PhysicalSnapshot) -> ObservationResult:
        indices = self._target_indices()
        next_indices = np.minimum(indices + 5, self.trajectory_lengths - 1)
        state = ObservationState(
            mano_dof_pos=physical.mano_dof_pos,
            mano_dof_lower=self.joint_lower,
            mano_dof_upper=self.joint_upper,
            hand_position=physical.hand_position,
            hand_orientation_xyzw=physical.hand_orientation_xyzw,
            object_position=physical.object_position,
            object_orientation_xyzw=physical.object_orientation_xyzw,
            target_object_position=self._reference_gather(self.reference_object_pos, indices),
            target_object_orientation_xyzw=self._reference_gather(self.reference_object_quat_xyzw, indices),
            target_object_pos_next_5=self._reference_gather(self.reference_object_pos, next_indices),
            cumulative_offset=self.cumulative_offset,
            cumulative_joint_offset=self.cumulative_joint_offset,
            point_cloud=self._point_template(),
            object_geometry=np.broadcast_to(self.object_geometry, (self.config.num_envs, 12)),
            hand_keypoint_positions=physical.hand_keypoint_positions,
            fingertip_positions=physical.fingertip_positions,
            hand_keypoint_contact_forces=physical.hand_keypoint_contact_forces,
            expected_contact_mask=self.expected_contact_mask,
            action_ids=self.action_ids,
            object_support_points=self.object_support_points,
            table_surface_height=FLOOR_TOP_Z,
        )
        return build_observation(state, compatibility=self.config.compatibility)

    def _reward_state(self, physical: PhysicalSnapshot) -> RewardState:
        indices = self._target_indices()
        batch = self.config.num_envs
        return RewardState(
            object_position=physical.object_position,
            target_object_position=self._reference_gather(self.reference_object_pos, indices),
            object_orientation_xyzw=physical.object_orientation_xyzw,
            target_object_orientation_xyzw=self._reference_gather(self.reference_object_quat_xyzw, indices),
            cumulative_offset=self.cumulative_offset,
            cumulative_joint_offset=self.cumulative_joint_offset,
            active_joint_mask=self.active_joint_mask,
            hand_keypoint_contact_forces=physical.hand_keypoint_contact_forces,
            expected_contact_mask=self.expected_contact_mask,
            expected_contact_weights=self.expected_contact_weights,
            object_contact_force=physical.object_contact_force,
            object_gravity_force=np.full(batch, self.object_gravity_force, dtype=np.float64),
            object_linear_velocity=physical.object_linear_velocity,
            trajectory_steps=self.trajectory_steps.copy(),
            contact_start_frames=self.contact_start_frames.copy(),
            contact_end_frames=self.contact_end_frames.copy(),
            rotation_disabled_mask=np.zeros(batch, dtype=bool),
            early_phase_starts=np.zeros(batch, dtype=np.int64),
        )

    def _refresh_output(self) -> ObservationResult:
        self.last_physical = self.producer.extract(self.data)
        self.last_observation = self._build_observation(self.last_physical)
        return self.last_observation

    def reset(self, env_ids: NDArray[object] | None = None) -> dict[str, NDArray[np.float64]]:
        if env_ids is None:
            indices = np.arange(self.config.num_envs, dtype=np.int64)
        else:
            indices = np.asarray(env_ids)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("env_ids must be a one-dimensional integer array")
            indices = indices.astype(np.int64, copy=False)
            if np.any(indices < 0) or np.any(indices >= self.config.num_envs):
                raise ValueError("env_ids contains an invalid environment index")
            indices = np.unique(indices)
        self._reset_indices(indices)
        observation = self._refresh_output()
        return {"obs": observation.policy_input.copy()}

    def step(
        self, raw_actions: NDArray[object]
    ) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64], NDArray[np.bool_], dict[str, NDArray[np.bool_]]]:
        actions = np.asarray(raw_actions, dtype=np.float64)
        if actions.shape != (self.config.num_envs, 26) or not np.all(np.isfinite(actions)):
            raise ValueError(f"raw_actions must be finite ({self.config.num_envs}, 26)")
        mocap_indices = self._target_indices()
        mocap_targets = self._reference_gather(self.reference_q, mocap_indices)
        action_result = process_residual_actions(
            actions,
            trajectory_steps=self.trajectory_steps,
            cumulative_offset=self.cumulative_offset,
            cumulative_joint_offset=self.cumulative_joint_offset,
            mocap_targets=mocap_targets,
            joint_lower=self.joint_lower,
            joint_upper=self.joint_upper,
            active_joint_mask=self.active_joint_mask,
            use_residual=np.full(self.config.num_envs, self.config.residual_enabled, dtype=np.float64),
            config=self.config.residual_action,
        )
        current_qpos = np.asarray(self.data.qpos, dtype=np.float64)[:, :26]
        controller_targets = np.stack(
            [
                command_target(action_result.targets[index], current_qpos[index], self.joint_lower, self.joint_upper)
                for index in range(self.config.num_envs)
            ]
        )
        self.cumulative_offset = action_result.cumulative_offset
        self.cumulative_joint_offset = action_result.cumulative_joint_offset
        self.last_controller_targets = controller_targets.copy()
        self.trajectory_steps += 1
        self.trajectory_steps[self.progress == 0] = 0
        self.data = self.data.replace(ctrl=self.jax.device_put(self.jp.asarray(controller_targets), self.device))
        for _ in range(PHYSICS_SUBSTEPS_PER_TARGET):
            self.data = self._step_fn(self.data)
        self.progress += 1
        pending_reset = self.reset_mask.copy()
        if np.any(pending_reset):
            self._reset_indices(np.flatnonzero(pending_reset).astype(np.int64))
        physical = self.producer.extract(self.data)
        early = early_phase_mask(
            self.trajectory_steps,
            starts=np.zeros(self.config.num_envs, dtype=np.int64),
            steps=self.config.compatibility.early_phase_steps,
        )
        termination = check_termination(
            object_position=physical.object_position,
            target_position=self._reference_gather(self.reference_object_pos, self._target_indices()),
            progress=self.progress,
            trajectory_lengths=self.trajectory_lengths.copy(),
            early_mask=early,
            max_deviation_distance=self.config.max_deviation_distance,
            deviation_penalty=self.config.deviation_penalty,
        )
        reward = compute_rewards(
            self._reward_state(physical),
            compatibility=self.config.compatibility,
            termination=termination,
        )
        observation = self._build_observation(physical)
        self.reset_mask = termination.reset.copy()
        self.episode_returns += reward.total
        self.last_physical = physical
        self.last_observation = observation
        self.last_termination = termination
        self.last_reward = reward
        self.last_transition = TransitionSnapshot(
            control_call=self.control_call,
            raw_actions=actions.copy(),
            command_reference_indices=mocap_indices.copy(),
            command_targets=mocap_targets.copy(),
            processed_targets=action_result.targets.copy(),
            controller_targets=controller_targets.copy(),
            reset_applied=pending_reset.copy(),
            progress=self.progress.copy(),
            trajectory_steps=self.trajectory_steps.copy(),
            target_indices=self._target_indices().copy(),
            physical=physical,
            observation=observation,
            reward=reward,
            termination=termination,
        )
        self.control_call += 1
        # ``episode_length`` is a rollout/statistics setting in this source
        # slice, not an independent physics horizon. A source trajectory end
        # or deviation is therefore a terminal transition, never a timeout.
        timeouts = np.zeros(self.config.num_envs, dtype=bool)
        return (
            {"obs": observation.policy_input.copy()},
            reward.total.copy(),
            termination.reset.copy(),
            {"time_outs": timeouts.copy()},
        )
