"""Bounded MJX-Warp ManoRL physical producer and vector environment.

The source ABI consumes world-frame rigid-body contact forces.  MJX-Warp keeps
its contact records in a global per-batch buffer, so this module decodes the
pyramidal contact constraints from the pinned 3.10.0 Warp layout instead of
using ``mjx.get_data``.  The latter is unsuitable here because its host contact
array retains capacity-padding entries after the solved records.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
import inspect
import time
import warnings
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from sim.manorl.abi import (
    ResidualActionConfig,
    TARGET_MAX_DEVIATION_DISTANCE,
    TerminationResult,
    check_termination,
    early_phase_mask,
    process_residual_actions,
)
from sim.manorl.device_runtime import (
    advance_device_task_counters,
    build_device_observation_28,
    check_device_termination,
    compute_device_reward_28,
    extract_mjx_physical_features,
    reduce_warp_contacts,
    reduce_warp_contacts_for_right_policy,
)
from sim.manorl.assets import (
    compile_model,
    compile_unified_model,
    object_collision_vertices,
    object_runtime,
)
from sim.manorl.contracts import (
    ACTION_SIDE_ORDER,
    JOINT_DOF,
    LEGACY_JOINT_NAMES,
    JOINT_NAMES,
    FLOOR_TOP_Z,
    KEYPOINT_NAMES,
    OBJECT_TYPE,
    PHYSICS_SUBSTEPS_PER_TARGET,
    ServoConfig,
    canonical_hand_sides,
    normalize_hand_side,
)
from sim.manorl.hand_layout import HandActionLayout
from sim.manorl.mjx_sim import CONTACT_CAPACITY, CONSTRAINT_CAPACITY, command_target
from sim.manorl.observations import (
    SOURCE_ALIGNED_COMPATIBILITY,
    POINT_COUNT,
    ObservationCompatibility,
    ObservationResult,
    ObservationState,
    PointCloudTemplate,
    build_observation,
    geometry_encoding,
    quat_rotate_xyzw,
    observation_layout,
    reduce_support_points,
)
from sim.manorl.rewards import RewardConfig, RewardDiagnostics, RewardState, compute_rewards
from sim.manorl.trajectory import (
    ReferenceTrajectory,
    TrajectoryBatch,
    resolve_hand_selection,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

_FINGERTIP_NAMES = ("thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip")
POINT_SAMPLING_NUMPY_PER_ENV = "numpy_per_env"
POINT_SAMPLING_TORCH_CUDA_GLOBAL = "torch_cuda_global"
POINT_SAMPLING_AUTO = "auto"
WARP_CONTACTS_PER_HAND_PER_WORLD = 64
WARP_CONTACT_CAPACITY_MARGIN = 64
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
_CUBE1_GRASP_ALIASES = {
    "01": ("thumb3", "index3"),
    "02": ("thumb3", "index3", "middle3"),
    "03": ("thumb3", "index3", "middle3", "ring3"),
    "04": ("thumb3", "index3", "middle3", "ring3", "pinky3"),
    "07": ("thumb3", "index3", "middle3", "ring3"),
    "09": ("thumb3", "thumb2", "index3", "index2", "middle3", "middle2", "ring3", "ring2", "pinky3", "pinky2"),
    "10": ("thumb3", "thumb2", "index3", "index2", "middle3", "middle2", "ring3", "ring2"),
    "18": ("thumb3", "index3"),
}


def minimum_warp_contact_capacity(num_envs: int, hand_sides: object) -> int:
    """Return the conservative batch contact floor for the compiled hands.

    Capacity belongs to the full MJX-Warp batch, not one world.  The revised
    left hand can produce more contacts than the historical 31-contact right
    hand bound, and a forced policy side does not remove the reference-following
    hand from physics.  Scale from every available/compiled hand accordingly.
    """

    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    hand_count = len(canonical_hand_sides(hand_sides))
    return WARP_CONTACTS_PER_HAND_PER_WORLD * hand_count * num_envs


def recommended_warp_contact_capacity(num_envs: int, hand_sides: object) -> int:
    """Add allocation headroom to the per-hand contact capacity floor."""

    return max(
        CONTACT_CAPACITY,
        minimum_warp_contact_capacity(num_envs, hand_sides)
        + WARP_CONTACT_CAPACITY_MARGIN,
    )


@dataclass(frozen=True)
class EnvironmentConfig:
    """Resolved bounded environment settings for the accepted cube trajectory."""

    num_envs: int = 1
    device: str = "cpu"
    servo: ServoConfig = ServoConfig()
    compatibility: ObservationCompatibility = SOURCE_ALIGNED_COMPATIBILITY
    residual_enabled: bool = True
    residual_action: ResidualActionConfig = ResidualActionConfig()
    max_deviation_distance: float = TARGET_MAX_DEVIATION_DISTANCE
    deviation_penalty: float = 0.0
    reward_config: RewardConfig = RewardConfig()
    episode_length: int = 600
    contact_capacity: int = CONTACT_CAPACITY
    constraint_capacity: int = CONSTRAINT_CAPACITY
    point_seed: int = 42
    point_sampling_backend: str = POINT_SAMPLING_AUTO
    device_resident_controls: bool = False
    # This removes only the private contact-buffer host transfer. It is not a
    # device-resident rollout mode: state, observation, reward, and Gymnasium
    # remain on their existing host contracts.
    device_contact_decode: bool = False
    # Narrow end-to-end training transition. It intentionally has a separate
    # opt-in from contact decoding because it changes the live step boundary.
    # It retains no full PhysicalSnapshot, so evaluation remains explicitly
    # unsupported until its diagnostics contract is adapted.
    device_transition: bool = False
    capture_transition_diagnostics: bool = True
    profile_phases: bool = False
    unified_object_batch: bool = False
    # Experimental Warp-only CCD allocation. Both values are opt-in so the
    # established put_data capacity contract remains the default.
    # warp_ccd_contacts_per_world scales only GJK scratch, never naconmax.
    warp_ccd_iterations: int | None = None
    warp_ccd_contacts_per_world: int | None = None
    hand_side: str = "auto"

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
        if not isinstance(self.reward_config, RewardConfig):
            raise TypeError("reward_config must be a RewardConfig")
        if self.max_deviation_distance < 0 or self.deviation_penalty < 0:
            raise ValueError("termination thresholds must be non-negative")
        if self.episode_length < 1:
            raise ValueError("episode_length must be positive")
        if self.contact_capacity < 1 or self.constraint_capacity < 1:
            raise ValueError("MJX contact and constraint capacities must be positive")
        if self.compatibility.point_template_mode == "static_seed_42" and self.point_seed != 42:
            raise ValueError("static_seed_42 compatibility requires point_seed=42")
        if self.point_sampling_backend == POINT_SAMPLING_AUTO:
            object.__setattr__(
                self,
                "point_sampling_backend",
                POINT_SAMPLING_TORCH_CUDA_GLOBAL if self.device == "gpu" else POINT_SAMPLING_NUMPY_PER_ENV,
            )
        if self.point_sampling_backend not in {
            POINT_SAMPLING_NUMPY_PER_ENV,
            POINT_SAMPLING_TORCH_CUDA_GLOBAL,
        }:
            raise ValueError("point_sampling_backend must be auto, numpy_per_env, or torch_cuda_global")
        if (
            self.point_sampling_backend == POINT_SAMPLING_TORCH_CUDA_GLOBAL
            and self.compatibility.point_template_mode != "dynamic_reset"
        ):
            raise ValueError("torch_cuda_global point sampling requires dynamic_reset compatibility")
        if not isinstance(self.device_resident_controls, bool):
            raise TypeError("device_resident_controls must be bool")
        if not isinstance(self.device_contact_decode, bool):
            raise TypeError("device_contact_decode must be bool")
        if not isinstance(self.device_transition, bool):
            raise TypeError("device_transition must be bool")
        if self.device_transition and self.device != "gpu":
            raise ValueError("device_transition requires device='gpu'")
        if self.device_transition and not self.device_resident_controls:
            raise ValueError("device_transition requires device_resident_controls=True")
        if self.device_transition and self.capture_transition_diagnostics:
            raise ValueError("device_transition requires capture_transition_diagnostics=False")
        if self.device_transition and self.profile_phases:
            raise ValueError("device_transition requires profile_phases=False")
        if self.device_contact_decode and self.device != "gpu":
            raise ValueError("device_contact_decode requires device='gpu'")
        if self.device_contact_decode and self.capture_transition_diagnostics:
            raise ValueError(
                "device_contact_decode requires capture_transition_diagnostics=False; "
                "full geometry contact snapshots remain the explicit debug path"
            )
        if self.device_contact_decode and self.profile_phases:
            raise ValueError(
                "device_contact_decode cannot collect host contact phase metadata; "
                "use the explicit debug/profile path instead"
            )
        if not isinstance(self.capture_transition_diagnostics, bool):
            raise TypeError("capture_transition_diagnostics must be bool")
        if not isinstance(self.profile_phases, bool):
            raise TypeError("profile_phases must be bool")
        if not isinstance(self.unified_object_batch, bool):
            raise TypeError("unified_object_batch must be bool")
        for name, value in (
            ("warp_ccd_iterations", self.warp_ccd_iterations),
            ("warp_ccd_contacts_per_world", self.warp_ccd_contacts_per_world),
        ):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be a positive integer when provided")
        if self.warp_ccd_explicit and self.unified_object_batch:
            raise ValueError("explicit Warp CCD capacity does not support unified object batches")
        normalize_hand_side(self.hand_side)

    @property
    def warp_ccd_explicit(self) -> bool:
        return self.warp_ccd_iterations is not None or self.warp_ccd_contacts_per_world is not None


@dataclass
class PhaseTimings:
    """Opt-in synchronized phase timings for one environment instance."""

    enabled: bool = False
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def start(self, name: str, synchronize: Any | None = None) -> float | None:
        if not self.enabled:
            return None
        if synchronize is not None:
            synchronize()
        return time.perf_counter()

    def stop(self, name: str, started: float | None, synchronize: Any | None = None) -> None:
        if started is None:
            return
        if synchronize is not None:
            synchronize()
        self.totals[name] = self.totals.get(name, 0.0) + (time.perf_counter() - started)
        self.counts[name] = self.counts.get(name, 0) + 1

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "total_seconds": total,
                "calls": self.counts.get(name, 0),
                "mean_seconds": total / self.counts[name] if self.counts.get(name, 0) else 0.0,
            }
            for name, total in self.totals.items()
        }


@dataclass(frozen=True)
class MaterializedState:
    qpos: NDArray[np.float64]
    qvel: NDArray[np.float64]
    xpos: NDArray[np.float64]
    xquat: NDArray[np.float64]
    keypoints: NDArray[np.float64]
    keypoint_quats: NDArray[np.float64]
    fingertips: NDArray[np.float64]


@dataclass(frozen=True)
class MaterializedContactBuffers:
    count: int
    capacity: int
    geom: NDArray[np.int64]
    world: NDArray[np.int64]
    dimension: NDArray[np.int64]
    addresses: NDArray[np.int64]
    nefc: NDArray[np.int64]
    friction: NDArray[np.float64]
    frame: NDArray[np.float64]
    constraint_force: NDArray[np.float64]
    raw_metadata: dict[str, dict[str, object]]
    host_metadata: dict[str, dict[str, object]]


@dataclass(frozen=True)
class PhysicalSnapshot:
    """Source-order physical values extracted from one batched MJX state."""

    mano_dof_pos: NDArray[np.float64]
    hand_position: NDArray[np.float64]
    hand_orientation_xyzw: NDArray[np.float64]
    hand_keypoint_orientations_xyzw: NDArray[np.float64]
    object_position: NDArray[np.float64]
    object_orientation_xyzw: NDArray[np.float64]
    object_linear_velocity: NDArray[np.float64]
    hand_keypoint_positions: NDArray[np.float64]
    fingertip_positions: NDArray[np.float64]
    hand_keypoint_contact_forces: NDArray[np.float64]
    # Full geometry diagnostics are intentionally unavailable in the opt-in
    # device-contact-decode path. The default/debug producer still fills both.
    object_contact_force: NDArray[np.float64] | None
    geom_contact_force_world_N: NDArray[np.float64] | None
    hand_object_force_on_object_world_N: NDArray[np.float64]
    contact_count: NDArray[np.int64]


@dataclass(frozen=True)
class TransitionSnapshot:
    """Immutable evidence for one transition, including compatibility-reset state."""

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
    episode_return: NDArray[np.float64]
    termination: TerminationResult


def _array_metadata(value: Any) -> dict[str, object]:
    shape = tuple(int(item) for item in value.shape)
    dtype = np.dtype(value.dtype)
    return {
        "dtype": str(dtype),
        "shape": list(shape),
        "bytes": int(np.prod(shape, dtype=np.int64)) * dtype.itemsize,
    }


def _aggregate_geometry_contact_forces(
    *,
    count: int,
    geom: NDArray[np.int64],
    world: NDArray[np.int64],
    dimension: NDArray[np.int64],
    addresses: NDArray[np.int64],
    nefc: NDArray[np.int64],
    friction: NDArray[np.float64],
    frame: NDArray[np.float64],
    constraint_force: NDArray[np.float64],
    ngeom: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Decode the final solved pyramidal contact forces into world geom rows."""

    batch = len(nefc)
    if ngeom < 1 or constraint_force.ndim != 2 or constraint_force.shape[0] != batch:
        raise ValueError("contact aggregation received incompatible geometry or constraint-force shapes")
    forces = np.zeros((batch, ngeom, 3), dtype=np.float64)
    per_world_count = np.zeros(batch, dtype=np.int64)
    for contact_id in range(count):
        world_id = int(world[contact_id])
        if not 0 <= world_id < batch:
            raise RuntimeError(f"contact {contact_id} references invalid MJX world {world_id}")
        if int(dimension[contact_id]) != 3:
            raise RuntimeError("the bounded MANO scene requires condim=3 contacts")
        address = addresses[contact_id]
        if np.any(address < 0) or np.any(address >= nefc[world_id]):
            raise RuntimeError(
                f"contact {contact_id} has a pyramidal constraint address outside world {world_id}'s solved range"
            )
        first_geom, second_geom = map(int, geom[contact_id])
        if not 0 <= first_geom < ngeom or not 0 <= second_geom < ngeom:
            raise RuntimeError(f"contact {contact_id} references an invalid MuJoCo geom")
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
        forces[world_id, first_geom] -= world_force
        forces[world_id, second_geom] += world_force
        per_world_count[world_id] += 1
    if not np.all(np.isfinite(forces)):
        raise RuntimeError("MJX contact decoder produced a non-finite world force")
    return forces, per_world_count


def _decode_contact_forces_reference(
    *,
    count: int,
    geom: NDArray[np.int64],
    world: NDArray[np.int64],
    dimension: NDArray[np.int64],
    addresses: NDArray[np.int64],
    nefc: NDArray[np.int64],
    friction: NDArray[np.float64],
    frame: NDArray[np.float64],
    constraint_force: NDArray[np.float64],
    ngeom: int,
    keypoint_geom_ids: Sequence[int],
    object_geom_ids: set[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Decode solved contact rows into all-geom and hand-on-object force tensors."""

    batch = len(nefc)
    if ngeom < 1 or constraint_force.ndim != 2 or constraint_force.shape[0] != batch:
        raise ValueError("contact decoder received incompatible geometry or constraint-force shapes")
    if len(keypoint_geom_ids) != len(KEYPOINT_NAMES) or len(set(keypoint_geom_ids)) != len(KEYPOINT_NAMES):
        raise ValueError("contact decoder requires one source-order collision geom per keypoint")
    keypoint_geom_id_set = set(keypoint_geom_ids)
    if any(not 0 <= geom_id < ngeom for geom_id in keypoint_geom_id_set | object_geom_ids):
        raise ValueError("contact decoder received an invalid keypoint or object geom")
    if keypoint_geom_id_set & object_geom_ids:
        raise ValueError("contact decoder requires disjoint hand and object collision geoms")
    geom_to_keypoint = {geom_id: index for index, geom_id in enumerate(keypoint_geom_ids)}
    geometry_forces = np.zeros((batch, ngeom, 3), dtype=np.float64)
    hand_object_forces = np.zeros((batch, len(KEYPOINT_NAMES), 3), dtype=np.float64)
    per_world_count = np.zeros(batch, dtype=np.int64)
    for contact_id in range(count):
        world_id = int(world[contact_id])
        if not 0 <= world_id < batch:
            raise RuntimeError(f"contact {contact_id} references invalid MJX world {world_id}")
        if int(dimension[contact_id]) != 3:
            raise RuntimeError("the bounded MANO scene requires condim=3 contacts")
        address = addresses[contact_id]
        if np.any(address < 0) or np.any(address >= nefc[world_id]):
            raise RuntimeError(
                f"contact {contact_id} has a pyramidal constraint address outside world {world_id}'s solved range"
            )
        first_geom, second_geom = map(int, geom[contact_id])
        if not 0 <= first_geom < ngeom or not 0 <= second_geom < ngeom:
            raise RuntimeError(f"contact {contact_id} references an invalid MuJoCo geom")
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
        geometry_forces[world_id, first_geom] -= world_force
        geometry_forces[world_id, second_geom] += world_force
        first_keypoint = geom_to_keypoint.get(first_geom)
        second_keypoint = geom_to_keypoint.get(second_geom)
        if first_keypoint is not None and second_keypoint is None and second_geom in object_geom_ids:
            hand_object_forces[world_id, first_keypoint] += world_force
        elif second_keypoint is not None and first_keypoint is None and first_geom in object_geom_ids:
            hand_object_forces[world_id, second_keypoint] -= world_force
        per_world_count[world_id] += 1
    if not np.all(np.isfinite(geometry_forces)) or not np.all(np.isfinite(hand_object_forces)):
        raise RuntimeError("MJX contact decoder produced a non-finite world force")
    return geometry_forces, hand_object_forces, per_world_count


def _decode_contact_forces(
    *,
    count: int,
    geom: NDArray[np.int64],
    world: NDArray[np.int64],
    dimension: NDArray[np.int64],
    addresses: NDArray[np.int64],
    nefc: NDArray[np.int64],
    friction: NDArray[np.float64],
    frame: NDArray[np.float64],
    constraint_force: NDArray[np.float64],
    ngeom: int,
    keypoint_geom_ids: Sequence[int],
    object_geom_ids: set[int],
    active_object_geom_ids: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Vectorized contact decoding with the loop implementation retained for tests."""

    batch = len(nefc)
    if count < 0:
        raise ValueError("contact decoder requires a non-negative contact count")
    if ngeom < 1 or constraint_force.ndim != 2 or constraint_force.shape[0] != batch:
        raise ValueError("contact decoder received incompatible geometry or constraint-force shapes")
    if len(keypoint_geom_ids) != len(KEYPOINT_NAMES) or len(set(keypoint_geom_ids)) != len(KEYPOINT_NAMES):
        raise ValueError("contact decoder requires one source-order collision geom per keypoint")
    keypoint_geom_id_set = set(keypoint_geom_ids)
    if any(not 0 <= geom_id < ngeom for geom_id in keypoint_geom_id_set | object_geom_ids):
        raise ValueError("contact decoder received an invalid keypoint or object geom")
    if keypoint_geom_id_set & object_geom_ids:
        raise ValueError("contact decoder requires disjoint hand and object collision geoms")
    if active_object_geom_ids is not None:
        active_object_geom_ids = np.asarray(active_object_geom_ids, dtype=np.int64)
        if active_object_geom_ids.ndim not in (1, 2) or active_object_geom_ids.shape[0] != batch:
            raise ValueError(
                "active object geom ids must have one entry or a padded set of entries per MJX world"
            )
        valid_active = active_object_geom_ids[active_object_geom_ids >= 0]
        if np.any(valid_active >= ngeom):
            raise ValueError("active object geom ids contain an invalid MuJoCo geom")
        if not np.all(np.isin(valid_active, np.asarray(sorted(object_geom_ids), dtype=np.int64))):
            raise ValueError("active object geom ids must belong to the unified object geom set")

    if np.asarray(geom).ndim != 2 or np.asarray(geom).shape[0] < count or np.asarray(geom).shape[1] != 2:
        raise RuntimeError("contact decoder received an invalid geom-pair buffer shape")
    if np.asarray(world).ndim != 1 or len(world) < count:
        raise RuntimeError("contact decoder received an invalid world buffer shape")
    if np.asarray(dimension).ndim != 1 or len(dimension) < count:
        raise RuntimeError("contact decoder received an invalid dimension buffer shape")
    if np.asarray(addresses).ndim != 2 or np.asarray(addresses).shape[0] < count or np.asarray(addresses).shape[1] != 4:
        raise RuntimeError("contact decoder received an invalid address buffer shape")
    if np.asarray(friction).ndim != 2 or np.asarray(friction).shape[0] < count or np.asarray(friction).shape[1] < 2:
        raise RuntimeError("contact decoder received an invalid friction buffer shape")
    frame_array = np.asarray(frame)
    if frame_array.ndim != 3 or frame_array.shape[0] < count or frame_array.shape[1:] != (3, 3):
        raise RuntimeError("contact decoder received an invalid frame buffer shape")

    active_world = np.asarray(world[:count], dtype=np.int64)
    active_dimension = np.asarray(dimension[:count], dtype=np.int64)
    active_addresses = np.asarray(addresses[:count], dtype=np.int64)
    active_geom = np.asarray(geom[:count], dtype=np.int64)
    if np.any(active_world < 0) or np.any(active_world >= batch):
        invalid = int(np.flatnonzero((active_world < 0) | (active_world >= batch))[0])
        raise RuntimeError(f"contact {invalid} references invalid MJX world {int(active_world[invalid])}")
    if np.any(active_dimension != 3):
        raise RuntimeError("the bounded MANO scene requires condim=3 contacts")
    if np.any(active_addresses < 0) or np.any(active_addresses >= nefc[active_world, None]):
        invalid = int(np.flatnonzero(
            np.any((active_addresses < 0) | (active_addresses >= nefc[active_world, None]), axis=1)
        )[0])
        raise RuntimeError(
            f"contact {invalid} has a pyramidal constraint address outside world "
            f"{int(active_world[invalid])}'s solved range"
        )
    if active_geom.shape != (count, 2):
        raise RuntimeError("contact decoder received an invalid geom-pair buffer shape")
    if np.any(active_geom < 0) or np.any(active_geom >= ngeom):
        invalid = int(np.flatnonzero(
            np.any((active_geom < 0) | (active_geom >= ngeom), axis=1)
        )[0]) if count else 0
        raise RuntimeError(f"contact {invalid} references an invalid MuJoCo geom")

    geometry_forces = np.zeros((batch, ngeom, 3), dtype=np.float64)
    hand_object_forces = np.zeros((batch, len(KEYPOINT_NAMES), 3), dtype=np.float64)
    if count:
        pyramid = constraint_force[active_world[:, None], active_addresses]
        active_friction = np.asarray(friction[:count], dtype=np.float64)
        local_force = np.column_stack(
            (
                pyramid.sum(axis=1),
                (pyramid[:, 0] - pyramid[:, 1]) * active_friction[:, 0],
                (pyramid[:, 2] - pyramid[:, 3]) * active_friction[:, 1],
            )
        )
        world_force = np.einsum("ni,nij->nj", local_force, np.asarray(frame[:count], dtype=np.float64))
        np.add.at(geometry_forces, (active_world, active_geom[:, 0]), -world_force)
        np.add.at(geometry_forces, (active_world, active_geom[:, 1]), world_force)

        keypoint_lookup = np.full(ngeom, -1, dtype=np.int64)
        keypoint_lookup[np.asarray(keypoint_geom_ids, dtype=np.int64)] = np.arange(len(KEYPOINT_NAMES))
        first_keypoint = keypoint_lookup[active_geom[:, 0]]
        second_keypoint = keypoint_lookup[active_geom[:, 1]]
        if active_object_geom_ids is None:
            object_geom_array = np.asarray(sorted(object_geom_ids), dtype=np.int64)
            first_object = np.isin(active_geom[:, 1], object_geom_array)
            second_object = np.isin(active_geom[:, 0], object_geom_array)
        else:
            world_object_geom = active_object_geom_ids[active_world]
            if world_object_geom.ndim == 1:
                first_object = active_geom[:, 1] == world_object_geom
                second_object = active_geom[:, 0] == world_object_geom
            else:
                valid_object_geom = world_object_geom >= 0
                first_object = np.any(
                    (active_geom[:, 1, None] == world_object_geom) & valid_object_geom,
                    axis=1,
                )
                second_object = np.any(
                    (active_geom[:, 0, None] == world_object_geom) & valid_object_geom,
                    axis=1,
                )
        first_hand = (first_keypoint >= 0) & (second_keypoint < 0) & first_object
        second_hand = (second_keypoint >= 0) & (first_keypoint < 0) & second_object
        np.add.at(
            hand_object_forces,
            (active_world[first_hand], first_keypoint[first_hand]),
            world_force[first_hand],
        )
        np.add.at(
            hand_object_forces,
            (active_world[second_hand], second_keypoint[second_hand]),
            -world_force[second_hand],
        )

    per_world_count = np.bincount(active_world, minlength=batch).astype(np.int64, copy=False)
    if not np.all(np.isfinite(geometry_forces)) or not np.all(np.isfinite(hand_object_forces)):
        raise RuntimeError("MJX contact decoder produced a non-finite world force")
    return geometry_forces, hand_object_forces, per_world_count


def _normalized_xyzw(quaternions_wxyz: NDArray[object]) -> NDArray[np.float64]:
    quaternions = wxyz_to_xyzw(np.asarray(quaternions_wxyz, dtype=np.float64))
    norm = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(~np.isfinite(quaternions)) or np.any(norm <= 1e-12):
        raise ValueError("MJX produced an invalid world quaternion")
    return quaternions / norm


def _source_surface_points(
    seed: int, object_type: str = OBJECT_TYPE
) -> NDArray[np.float64]:
    """Reproduce trimesh surface sampling for one registered object."""

    triangles = object_collision_vertices(object_type).reshape(-1, 3, 3)
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


def _source_surface_template(
    seed: int, object_type: str = OBJECT_TYPE
) -> PointCloudTemplate:
    points = _source_surface_points(seed, object_type)
    lower, upper = points.min(axis=0), points.max(axis=0)
    scale = np.maximum((upper - lower) / 2.0, 1e-6)
    normalized = (points - (upper + lower) / 2.0) / scale
    return PointCloudTemplate(normalized, mode="static_seed_42", normalized=True, scale=scale)


def _dynamic_surface_template(
    generator: np.random.Generator, object_type: str = OBJECT_TYPE
) -> NDArray[np.float64]:
    """Use the source mesh sampler with the caller's explicit reset RNG."""

    seed = int(generator.integers(0, np.iinfo(np.int64).max))
    return _source_surface_points(seed, object_type)


def _torch_global_surface_templates(
    batch_size: int, object_type: str = OBJECT_TYPE
) -> NDArray[np.float64]:
    """Run the source CUDA sampler against PyTorch's global CUDA RNG."""

    if batch_size < 1:
        raise ValueError("dynamic point sampling batch must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch_cuda_global point sampling requires CUDA")
    device = torch.device("cuda")
    triangles = torch.as_tensor(
        object_collision_vertices(object_type).reshape(-1, 3, 3).copy(),
        dtype=torch.float32,
        device=device,
    )
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    areas = torch.linalg.norm(torch.cross(edge_1, edge_2, dim=1), dim=1) * 0.5
    area_cdf = torch.cumsum(areas / areas.sum(), dim=0)
    area_cdf[-1] = 1.0
    flat_count = batch_size * POINT_COUNT
    face_indices = torch.searchsorted(
        area_cdf,
        torch.rand(flat_count, device=device),
        right=False,
    )
    selected = triangles[face_indices].reshape(batch_size, POINT_COUNT, 3, 3)
    barycentric = torch.rand(batch_size, POINT_COUNT, 2, device=device)
    sqrt_u = torch.sqrt(barycentric[..., 0:1])
    v = barycentric[..., 1:2]
    points = (
        (1.0 - sqrt_u) * selected[:, :, 0]
        + sqrt_u * (1.0 - v) * selected[:, :, 1]
        + sqrt_u * v * selected[:, :, 2]
    )
    return points.detach().cpu().numpy().astype(np.float64, copy=False)


def _expected_keypoint_ids(object_type: str, action_id: str) -> NDArray[np.int64]:
    runtime = object_runtime(object_type)
    import yaml

    payload = yaml.safe_load(runtime.grasp_mapping_path.read_text(encoding="utf-8"))
    mappings = payload.get(object_type) if isinstance(payload, dict) else None
    if not isinstance(mappings, dict):
        raise ValueError(f"invalid grasp mapping asset for {object_type!r}")
    aliases = mappings.get(action_id)
    if aliases is None:
        raise ValueError(f"no source grasp mapping for object={object_type!r}, gesture={action_id!r}")
    if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
        raise ValueError(f"invalid source grasp aliases for {object_type!r}/{action_id!r}")
    alias_map = {
        "palm": "palm",
        "thumb1": "thumb_cmc",
        "thumb2": "thumb_mcp", "thumb3": "thumb_ip",
        "index1": "index_mcp",
        "index2": "index_pip", "index3": "index_dip",
        "middle1": "middle_mcp",
        "middle2": "middle_pip", "middle3": "middle_dip",
        "ring1": "ring_mcp",
        "ring2": "ring_pip", "ring3": "ring_dip",
        "pinky1": "pinky_mcp",
        "pinky2": "pinky_pip", "pinky3": "pinky_dip",
    }
    names = [alias_map.get(alias) for alias in aliases]
    if any(name is None for name in names):
        raise ValueError(f"unsupported source keypoint alias in {aliases!r}")
    return np.asarray([KEYPOINT_NAMES.index(name) for name in names], dtype=np.int64)


def _expand_legacy_hand_dofs(values: NDArray[object]) -> NDArray[np.float64]:
    """Embed legacy 26D references in the revised 28D joint order."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] not in (26, JOINT_DOF):
        raise ValueError("hand references must end in 26 or 28 DOFs")
    if array.shape[-1] == JOINT_DOF:
        return array.copy()
    output = np.zeros((*array.shape[:-1], JOINT_DOF), dtype=np.float64)
    legacy_index = {name: index for index, name in enumerate(LEGACY_JOINT_NAMES)}
    for index, name in enumerate(JOINT_NAMES):
        source_name = "j1_thumb_mcp" if name == "j1_thumb_mcp_flex" else name
        if source_name in legacy_index:
            output[..., index] = array[..., legacy_index[source_name]]
    return output


def _active_joint_mask(
    expected_mask: NDArray[np.float64], *, finger_dof: int = JOINT_DOF - 6
) -> NDArray[np.bool_]:
    """Mirror FingerMaskManager: only fingers with expected keypoints are active."""

    if finger_dof == 22:
        finger_ranges = {
            "thumb": slice(0, 6),
            "index": slice(6, 10),
            "middle": slice(10, 14),
            "ring": slice(14, 18),
            "pinky": slice(18, 22),
        }
    elif finger_dof == 20:
        finger_ranges = {
            "thumb": slice(0, 4),
            "index": slice(4, 8),
            "middle": slice(8, 12),
            "ring": slice(12, 16),
            "pinky": slice(16, 20),
        }
    else:
        raise ValueError("finger_dof must be 20 or 22")
    batch = len(expected_mask)
    active = np.zeros((batch, finger_dof), dtype=bool)
    for keypoint_index, keypoint_name in enumerate(KEYPOINT_NAMES):
        finger = keypoint_name.split("_", maxsplit=1)[0]
        if finger in finger_ranges:
            active[expected_mask[:, keypoint_index] > 0.5, finger_ranges[finger]] = True
    return active


def _model_hand_side_order(hand_sides: Sequence[str]) -> tuple[str, ...]:
    """Return hand sides in the compiled model/action slot order.

    Dataset metadata is canonicalized independently (left before right), while
    the MJCF builder and action ABI deliberately use right-before-left.  Keep
    this conversion at the environment boundary so every qpos/ctrl/reference
    table uses the same order without relying on Lance list order.
    """

    available = set(canonical_hand_sides(hand_sides))
    return tuple(side for side in ACTION_SIDE_ORDER if side in available)


class MjxWarpPhysicalProducer:
    """Convert pinned MJX-Warp state into source-order physical fields.

    Contact decoding relies on the installed 3.10.0 Warp data layout.  The
    source code's ``smooth.py`` converts a local contact wrench with
    ``force.reshape((-1, 3)) @ frame`` and applies it negatively to ``geom1``
    and positively to ``geom2``.  This producer uses that exact orientation and
    sign convention, then aggregates only the declared source keypoint bodies.
    """

    def __init__(
        self,
        mujoco: Any,
        model: Any,
        *,
        object_type: str = OBJECT_TYPE,
        hand_sides: Sequence[str] = ("right",),
        primary_hand_side: str = "right",
    ) -> None:
        self.mujoco = mujoco
        self.model = model
        self.hand_sides = canonical_hand_sides(hand_sides)
        self.primary_hand_side = normalize_hand_side(
            primary_hand_side, allow_auto=False, allow_both=False
        )
        if self.primary_hand_side not in self.hand_sides:
            raise ValueError("primary hand side is absent from compiled model")
        self.per_hand_dof = int(model.nu) // len(self.hand_sides)
        if (
            self.per_hand_dof != JOINT_DOF
            or model.nu != self.per_hand_dof * len(self.hand_sides)
        ):
            raise ValueError("compiled MANO actuator width does not match hand sides")
        self.hand_dof = self.per_hand_dof
        self.hand_qpos_slices = {
            side: slice(index * self.per_hand_dof, (index + 1) * self.per_hand_dof)
            for index, side in enumerate(
                tuple(side for side in ("right", "left") if side in self.hand_sides)
            )
        }
        runtime = object_runtime(object_type)
        self.object_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, runtime.body_name
        )
        object_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, runtime.free_joint_name
        )
        if self.object_body_id < 0 or object_joint_id < 0:
            raise ValueError(f"compiled {object_type} body/free joint is absent")
        self.object_qpos_address = int(model.jnt_qposadr[object_joint_id])
        self.object_qvel_address = int(model.jnt_dofadr[object_joint_id])
        dual = len(self.hand_sides) > 1
        self.keypoint_geom_ids_by_side: dict[str, list[int]] = {}
        self.keypoint_body_ids_by_side: dict[str, list[int]] = {}
        self.geom_to_keypoint: dict[int, int] = {}
        for side in self.hand_sides:
            prefix = f"{side}_" if dual else ""
            geom_ids: list[int] = []
            body_ids: list[int] = []
            for keypoint_index, keypoint_name in enumerate(KEYPOINT_NAMES):
                geom_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"{prefix}{keypoint_name}_collision",
                )
                if geom_id < 0:
                    raise ValueError(
                        f"compiled {side} source keypoint geom is absent: {keypoint_name}"
                    )
                geom_ids.append(geom_id)
                body_ids.append(int(model.geom_bodyid[geom_id]))
                self.geom_to_keypoint[geom_id] = keypoint_index
            self.keypoint_geom_ids_by_side[side] = geom_ids
            self.keypoint_body_ids_by_side[side] = body_ids
        all_hand_geom_ids = [
            geom_id for values in self.keypoint_geom_ids_by_side.values() for geom_id in values
        ]
        if len(set(all_hand_geom_ids)) != len(KEYPOINT_NAMES) * len(self.hand_sides):
            raise ValueError("source keypoint geom mapping is not one-to-one")
        self.keypoint_geom_ids = self.keypoint_geom_ids_by_side[self.primary_hand_side]
        self.keypoint_body_ids = self.keypoint_body_ids_by_side[self.primary_hand_side]
        self.fingertip_body_ids = np.asarray(
            [self.keypoint_body_ids[KEYPOINT_NAMES.index(name)] for name in _FINGERTIP_NAMES], dtype=np.int64
        )
        self.object_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == self.object_body_id
            and (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            ).endswith("_collision")
        }
        if len(self.object_geom_ids) != runtime.collision_geom_count:
            raise ValueError(
                f"the {object_type} runtime requires {runtime.collision_geom_count} object collision geoms"
            )
        if self.object_geom_ids & set(all_hand_geom_ids):
            raise ValueError("source hand and object collision geoms must be disjoint")
        # Unified superset scenes replace these scalar fields with one active
        # entry per world.  The default homogeneous path remains unchanged.
        self.active_object_body_ids: NDArray[np.int64] | None = None
        self.active_object_qvel_addresses: NDArray[np.int64] | None = None
        self.active_object_geom_ids: NDArray[np.int64] | None = None
        self._device_contact_decoder: Any | None = None
        self.reset_profile()

    def reset_profile(self) -> None:
        self._profile_nacon: list[int] = []
        self._profile_capacity: int | None = None
        self._profile_raw_metadata: dict[str, dict[str, object]] = {}
        self._profile_host_metadata: dict[str, dict[str, object]] = {}

    def contact_profile(self) -> dict[str, object]:
        if not self._profile_nacon:
            return {}
        counts = np.asarray(self._profile_nacon, dtype=np.float64)
        return {
            "nacon_per_step": list(self._profile_nacon),
            "nacon": {
                "calls": len(self._profile_nacon),
                "mean": float(counts.mean()),
                "p50": float(np.percentile(counts, 50)),
                "p95": float(np.percentile(counts, 95)),
                "max": int(counts.max()),
            },
            "capacity": self._profile_capacity,
            "raw_buffers": self._profile_raw_metadata,
            "raw_buffer_bytes": sum(int(item["bytes"]) for item in self._profile_raw_metadata.values()),
            "host_materialization": self._profile_host_metadata,
            "host_materialization_bytes": sum(
                int(item["bytes"]) for item in self._profile_host_metadata.values()
            ),
        }

    def materialize_contact_buffers(
        self, data: Any, batch: int, *, record_profile: bool = False
    ) -> MaterializedContactBuffers:
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
        raw_values = {name: getattr(impl, name) for name in required}
        raw_metadata = (
            {name: _array_metadata(value) for name, value in raw_values.items()}
            if record_profile
            else {}
        )
        nacon = np.asarray(raw_values["nacon"], dtype=np.int64).reshape(-1)
        if nacon.shape != (1,):
            raise RuntimeError(f"MJX-Warp global contact count must have shape (1,), got {nacon.shape}")
        count = int(nacon[0])
        nefc_values = np.asarray(raw_values["nefc"], dtype=np.int64).reshape(-1)
        if nefc_values.shape not in {(1,), (batch,)}:
            raise RuntimeError(
                "MJX-Warp constraint count must be scalar or one value per world, "
                f"got {nefc_values.shape}"
            )
        geom = np.asarray(raw_values["contact__geom"], dtype=np.int64)
        world = np.asarray(raw_values["contact__worldid"], dtype=np.int64)
        dimension = np.asarray(raw_values["contact__dim"], dtype=np.int64)
        addresses = np.asarray(raw_values["contact__efc_address"], dtype=np.int64)
        friction = np.asarray(raw_values["contact__friction"], dtype=np.float64)
        frame = np.asarray(raw_values["contact__frame"], dtype=np.float64)
        forces = np.asarray(raw_values["efc__force"], dtype=np.float64)
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
        if np.any(nefc_values < 0) or np.any(nefc_values >= forces.shape[1]):
            raise RuntimeError("MJX-Warp constraint capacity saturated; contact forces are invalid")
        normalized_nefc = (
            np.full(batch, int(nefc_values[0]), dtype=np.int64)
            if nefc_values.shape == (1,)
            else nefc_values
        )
        host_metadata = {}
        if record_profile:
            host_values = {
                "nacon": nacon,
                "nefc": nefc_values,
                "contact__geom": geom,
                "contact__worldid": world,
                "contact__dim": dimension,
                "contact__efc_address": addresses,
                "contact__friction": friction,
                "contact__frame": frame,
                "efc__force": forces,
            }
            host_metadata = {name: _array_metadata(value) for name, value in host_values.items()}
        if record_profile:
            self._profile_nacon.append(count)
            self._profile_capacity = capacity
            self._profile_raw_metadata = raw_metadata
            self._profile_host_metadata = host_metadata
        return MaterializedContactBuffers(
            count=count,
            capacity=capacity,
            geom=geom,
            world=world,
            dimension=dimension,
            addresses=addresses,
            nefc=normalized_nefc,
            friction=friction,
            frame=frame,
            constraint_force=forces,
            raw_metadata=raw_metadata,
            host_metadata=host_metadata,
        )

    def materialize_state(self, data: Any) -> MaterializedState:
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
        offsets = _FINGERTIP_LOCAL_OFFSETS.copy()
        if self.primary_hand_side == "left":
            offsets[:, 0] *= -1.0
        fingertips = keypoints[:, fingertip_indices] + quat_rotate_xyzw(
            np.broadcast_to(keypoint_quats[:, fingertip_indices], (batch, len(_FINGERTIP_NAMES), 4)),
            np.broadcast_to(offsets, (batch, len(_FINGERTIP_NAMES), 3)),
        )
        return MaterializedState(qpos, qvel, xpos, xquat, keypoints, keypoint_quats, fingertips)

    def decode_contact_buffers(
        self, buffers: MaterializedContactBuffers
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        decoder = _decode_contact_forces
        decoder_parameters: set[str] | None
        try:
            decoder_parameters = set(inspect.signature(decoder).parameters)
        except (TypeError, ValueError):  # pragma: no cover - opaque test/backend callable
            decoder_parameters = None
        decoded = [
            decoder(
                count=buffers.count,
                geom=buffers.geom,
                world=buffers.world,
                dimension=buffers.dimension,
                addresses=buffers.addresses,
                nefc=buffers.nefc,
                friction=buffers.friction,
                frame=buffers.frame,
                constraint_force=buffers.constraint_force,
                ngeom=self.model.ngeom,
                keypoint_geom_ids=self.keypoint_geom_ids_by_side[side],
                object_geom_ids=self.object_geom_ids,
                **(
                    {"active_object_geom_ids": self.active_object_geom_ids}
                    if decoder_parameters is None
                    or "active_object_geom_ids" in decoder_parameters
                    else {}
                ),
            )
            for side in self.hand_sides
        ]
        geometry_forces = decoded[0][0]
        hand_forces = np.sum([item[1] for item in decoded], axis=0)
        return geometry_forces, hand_forces, decoded[0][2]

    def device_contact_reduction(self, data: Any) -> Any:
        """Return compact JAX contact reductions without a host buffer copy."""

        if set(self.hand_sides) not in ({"right"}, {"right", "left"}) or self.active_object_geom_ids is not None:
            raise RuntimeError(
                "device contact reduction supports homogeneous right-policy worlds with "
                "a right-only or right-left compiled model; per-world object sets require "
                "the host/debug path"
            )
        impl = data._impl
        required = (
            "nacon", "nefc", "contact__geom", "contact__worldid", "contact__dim",
            "contact__efc_address", "contact__friction", "contact__frame", "efc__force",
        )
        missing = [name for name in required if not hasattr(impl, name)]
        if missing:
            raise RuntimeError(
                "the installed MJX-Warp contact ABI is incompatible with device contact reduction: "
                + ", ".join(missing)
            )
        if self._device_contact_decoder is None:
            import jax

            if len(self.hand_sides) == 1:
                self._device_contact_decoder = jax.jit(
                    lambda nacon, nefc, geom, world, dimension, addresses, friction, frame, force:
                    reduce_warp_contacts(
                        nacon=nacon, nefc=nefc, geom=geom, world=world, dimension=dimension,
                        addresses=addresses, friction=friction, frame=frame, constraint_force=force,
                        ngeom=self.model.ngeom, keypoint_geom_ids=tuple(self.keypoint_geom_ids),
                        object_geom_ids=tuple(sorted(self.object_geom_ids)), compute_dtype="float32",
                    )
                )
            else:
                self._device_contact_decoder = jax.jit(
                    lambda nacon, nefc, geom, world, dimension, addresses, friction, frame, force:
                    reduce_warp_contacts_for_right_policy(
                        nacon=nacon, nefc=nefc, geom=geom, world=world, dimension=dimension,
                        addresses=addresses, friction=friction, frame=frame, constraint_force=force,
                        ngeom=self.model.ngeom,
                        right_keypoint_geom_ids=tuple(self.keypoint_geom_ids_by_side["right"]),
                        left_keypoint_geom_ids=tuple(self.keypoint_geom_ids_by_side["left"]),
                        object_geom_ids=tuple(sorted(self.object_geom_ids)), compute_dtype="float32",
                    )
                )
        return self._device_contact_decoder(
            impl.nacon, impl.nefc, impl.contact__geom, impl.contact__worldid,
            impl.contact__dim, impl.contact__efc_address, impl.contact__friction,
            impl.contact__frame, impl.efc__force,
        )

    def _device_decode_contact_buffers(
        self, data: Any, batch: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Decode homogeneous single-hand contacts on JAX, then copy reductions only.

        This deliberately leaves full geometry forces unavailable. Callers that
        need Rerun or a transition snapshot use the existing host decoder.
        """

        reduction = self.device_contact_reduction(data)
        # This scalar is the only per-step validation transfer. It must be
        # observed before consuming the small reduced arrays, matching the
        # host decoder's same-transition fail-closed capacity/finite checks.
        if not bool(np.asarray(reduction.valid)):
            raise RuntimeError(
                "MJX-Warp device contact reduction rejected a live contact, capacity, "
                "or non-finite force; refusing to continue with reduced contacts"
            )
        forces = np.asarray(reduction.keypoint_forces, dtype=np.float64)
        hand_forces = np.asarray(reduction.hand_object_forces, dtype=np.float64)
        counts = np.asarray(reduction.per_world_count, dtype=np.int64)
        if forces.shape != (batch, len(KEYPOINT_NAMES), 3) or hand_forces.shape != forces.shape or counts.shape != (batch,):
            raise RuntimeError("MJX-Warp device contact reduction returned invalid reduced shapes")
        return forces, hand_forces, counts

    def extract(
        self,
        data: Any,
        *,
        timings: PhaseTimings | None = None,
        synchronize: Any | None = None,
        record_profile: bool = False,
        device_contact_decode: bool = False,
    ) -> PhysicalSnapshot:
        if device_contact_decode and record_profile:
            raise RuntimeError("device_contact_decode cannot materialize host contact profiling buffers")
        state_started = timings.start("state_materialization", synchronize) if timings is not None else None
        state = self.materialize_state(data)
        if timings is not None:
            timings.stop("state_materialization", state_started, synchronize)
        if device_contact_decode:
            decode_started = timings.start("device_contact_decode", synchronize) if timings is not None else None
            forces, hand_object_forces, per_world_count = self._device_decode_contact_buffers(data, len(state.qpos))
            if timings is not None:
                timings.stop("device_contact_decode", decode_started, synchronize)
            geometry_forces = None
        else:
            contact_started = (
                timings.start("contact_buffer_materialization", synchronize) if timings is not None else None
            )
            buffers = self.materialize_contact_buffers(
                data, len(state.qpos), record_profile=record_profile
            )
            if timings is not None:
                timings.stop("contact_buffer_materialization", contact_started, synchronize)
            decode_started = timings.start("python_contact_decode", synchronize) if timings is not None else None
            geometry_forces, hand_object_forces, per_world_count = self.decode_contact_buffers(buffers)
            # Keypoint contact observations retain the primary-hand ABI. The
            # hand-object field is aggregated across sides for reward.
            forces = geometry_forces[:, self.keypoint_geom_ids]
        if self.active_object_geom_ids is None:
            object_force = (
                None
                if geometry_forces is None
                else geometry_forces[:, sorted(self.object_geom_ids)].sum(axis=1)
            )
            object_position = state.xpos[:, self.object_body_id]
            object_orientation = state.xquat[:, self.object_body_id]
            object_linear_velocity = state.qvel[:, self.object_qvel_address : self.object_qvel_address + 3]
        else:
            world_ids = np.arange(len(state.qpos), dtype=np.int64)
            active_geom_ids = np.asarray(self.active_object_geom_ids, dtype=np.int64)
            if active_geom_ids.ndim == 1:
                object_force = geometry_forces[world_ids, active_geom_ids]
            elif active_geom_ids.ndim == 2:
                object_force = np.zeros((len(world_ids), 3), dtype=np.float64)
                for piece_index in range(active_geom_ids.shape[1]):
                    piece_ids = active_geom_ids[:, piece_index]
                    valid = piece_ids >= 0
                    if np.any(valid):
                        object_force[valid] += geometry_forces[world_ids[valid], piece_ids[valid]]
            else:  # pragma: no cover - guarded by setter/decoder validation
                raise RuntimeError("unified active object geom ids have an invalid rank")
            if self.active_object_body_ids is None or self.active_object_qvel_addresses is None:
                raise RuntimeError("unified producer active object state is incomplete")
            object_position = state.xpos[world_ids, self.active_object_body_ids]
            object_orientation = state.xquat[world_ids, self.active_object_body_ids]
            object_linear_velocity = np.stack(
                [
                    state.qvel[index, address : address + 3]
                    for index, address in enumerate(self.active_object_qvel_addresses)
                ]
            )
        if timings is not None and not device_contact_decode:
            timings.stop("python_contact_decode", decode_started, synchronize)
        return PhysicalSnapshot(
            mano_dof_pos=state.qpos[:, self.hand_qpos_slices[self.primary_hand_side]].copy(),
            hand_position=state.xpos[:, self.keypoint_body_ids[0]].copy(),
            hand_orientation_xyzw=_normalized_xyzw(state.xquat[:, self.keypoint_body_ids[0]]),
            hand_keypoint_orientations_xyzw=state.keypoint_quats,
            object_position=object_position.copy(),
            object_orientation_xyzw=_normalized_xyzw(object_orientation),
            object_linear_velocity=object_linear_velocity.copy(),
            hand_keypoint_positions=state.keypoints,
            fingertip_positions=state.fingertips,
            hand_keypoint_contact_forces=forces,
            object_contact_force=None if object_force is None else object_force.copy(),
            geom_contact_force_world_N=geometry_forces,
            hand_object_force_on_object_world_N=hand_object_forces,
            contact_count=per_world_count,
        )


class UnifiedMjxWarpPhysicalProducer(MjxWarpPhysicalProducer):
    """Extract source fields for one active object per unified MJX world."""

    def __init__(
        self,
        mujoco: Any,
        model: Any,
        *,
        object_types: Sequence[str],
        hand_sides: Sequence[str] = ("right",),
        primary_hand_side: str = "right",
    ) -> None:
        names = tuple(dict.fromkeys(object_types))
        if not names:
            raise ValueError("unified producer requires at least one object type")
        super().__init__(
            mujoco,
            model,
            object_type=names[0],
            hand_sides=hand_sides,
            primary_hand_side=primary_hand_side,
        )
        self.object_types = names
        self.object_body_ids_by_type = np.asarray(
            [
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    object_runtime(name).body_name,
                )
                for name in names
            ],
            dtype=np.int64,
        )
        self.object_qvel_addresses_by_type = np.asarray(
            [
                model.jnt_dofadr[
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        object_runtime(name).free_joint_name,
                    )
                ]
                for name in names
            ],
            dtype=np.int64,
        )
        piece_rows: list[tuple[int, ...]] = []
        for object_type, body_id in zip(names, self.object_body_ids_by_type, strict=True):
            runtime = object_runtime(object_type)
            pieces = tuple(
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == int(body_id)
                and (
                    self.mujoco.mj_id2name(
                        model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    or ""
                ).endswith("_collision")
            )
            if len(pieces) != runtime.collision_geom_count:
                raise ValueError(
                    f"unified {object_type} requires {runtime.collision_geom_count} collision geoms, "
                    f"found {len(pieces)}"
                )
            piece_rows.append(pieces)
        max_pieces = max(len(row) for row in piece_rows)
        self.object_geom_ids_by_type = np.full(
            (len(piece_rows), max_pieces), -1, dtype=np.int64
        )
        for row_index, pieces in enumerate(piece_rows):
            self.object_geom_ids_by_type[row_index, : len(pieces)] = pieces
        if np.any(self.object_body_ids_by_type < 0) or np.any(
            self.object_qvel_addresses_by_type < 0
        ):
            raise ValueError("unified object body or free-joint mapping is incomplete")
        self.object_geom_ids = {
            int(value)
            for value in self.object_geom_ids_by_type.reshape(-1)
            if int(value) >= 0
        }

    def set_active_objects(self, object_indices: NDArray[np.int64]) -> None:
        indices = np.asarray(object_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self.object_types)):
            raise ValueError("active object indices must be a valid one-dimensional batch")
        self.active_object_body_ids = self.object_body_ids_by_type[indices]
        self.active_object_qvel_addresses = self.object_qvel_addresses_by_type[indices]
        self.active_object_geom_ids = self.object_geom_ids_by_type[indices]


def _scatter_routed_value(
    routed: list[tuple[NDArray[np.int64], Any]], total: int
) -> Any:
    """Scatter matching route-local arrays/dataclasses into global env order."""

    non_null = [(indices, value) for indices, value in routed if value is not None]
    if not non_null:
        return None
    if len(non_null) != len(routed):
        raise RuntimeError("heterogeneous routes produced inconsistent optional diagnostics")
    sample = non_null[0][1]
    if is_dataclass(sample):
        cls = type(sample)
        return cls(
            **{
                item.name: _scatter_routed_value(
                    [(indices, getattr(value, item.name)) for indices, value in non_null],
                    total,
                )
                for item in fields(sample)
            }
        )
    if isinstance(sample, np.ndarray):
        if sample.ndim == 0:
            return sample.copy()
        result = np.empty((total, *sample.shape[1:]), dtype=sample.dtype)
        for indices, value in non_null:
            array = np.asarray(value)
            if array.shape[0] != len(indices) or array.shape[1:] != sample.shape[1:]:
                raise RuntimeError("heterogeneous routes produced incompatible array shapes")
            result[indices] = array
        return result
    if all(value == sample for _, value in non_null[1:]):
        return sample
    raise RuntimeError("heterogeneous routes produced incompatible scalar diagnostics")


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
        side_sets = {tuple(item.hand_sides) for item in self.trajectories}
        dof_dims = {item.dof_dim for item in self.trajectories}
        if len(side_sets) != 1 or len(dof_dims) != 1:
            raise ValueError("all vector trajectories must share hand sides and DOF width")
        self.hand_sides = _model_hand_side_order(next(iter(side_sets)))
        if config.warp_ccd_explicit and config.device != "gpu":
            raise ValueError("explicit Warp CCD capacity requires device='gpu'")
        if config.device_transition and (
            self.hand_sides not in {("right",), ("right", "left")} or next(iter(dof_dims)) != JOINT_DOF
        ):
            raise ValueError("device_transition supports a right-policy 28-DoF model with optional passive left hand")
        if config.device_contact_decode and len(self.hand_sides) != 1:
            raise ValueError(
                "device_contact_decode supports exactly one compiled hand; "
                "bimanual contact aggregation remains on the debug path"
            )
        # Lance metadata order is not a control-order contract.  MuJoCo XML
        # and policy/model controls always use right then left.
        self.model_hand_sides = tuple(
            side for side in ACTION_SIDE_ORDER if side in self.hand_sides
        )
        if config.hand_side == "auto":
            selected_sets = {tuple(item.selected_hand_sides) for item in self.trajectories}
            if len(selected_sets) != 1:
                raise ValueError("all vector trajectories must share one hand selection")
            controlled_sides = next(iter(selected_sets))
        else:
            controlled_sides = resolve_hand_selection(self.hand_sides, config.hand_side)
        self.hand_layout = HandActionLayout(
            self.hand_sides,
            controlled_sides,
            dof_per_hand=JOINT_DOF,
        )
        self.action_dim = self.hand_layout.action_dim
        self.hand_dof = JOINT_DOF
        self.finger_dof = self.hand_dof - 6
        self.primary_hand_side = (
            "right" if "right" in self.hand_layout.controlled_sides
            else self.hand_layout.controlled_sides[0]
        )
        self.model_hand_side = "both" if len(self.hand_sides) == 2 else self.hand_sides[0]
        self.model_action_dim = len(self.hand_sides) * self.hand_dof
        self.observation_layout = observation_layout(
            self.hand_dof,
            cumulative_joint_dim=self.hand_layout.cumulative_dim,
        )
        self.observation_dim = self.observation_layout.dimension
        identity_parts = [item.identity.identity.split("_") for item in self.trajectories]
        if any(len(parts) != 3 or not parts[1].isdigit() for parts in identity_parts):
            raise ValueError("each trajectory identity must be object_action_sequence")
        object_types = {parts[0] for parts in identity_parts}
        if config.device_transition and (
            self.hand_layout.controlled_sides != ("right",)
            or len(object_types) != 1
            or self.action_dim != JOINT_DOF
            or self.observation_dim != 480
        ):
            raise ValueError(
                "device_transition supports a homogeneous right-policy batch with 28D actions and 480D observations"
            )
        if config.device_contact_decode and len(object_types) != 1:
            raise ValueError(
                "device_contact_decode supports a homogeneous object batch; "
                "unified and heterogeneous object sets remain on the debug path"
            )
        if len(object_types) != 1:
            if config.warp_ccd_explicit:
                raise ValueError(
                    "explicit Warp CCD capacity supports homogeneous object batches only; "
                    "unified and heterogeneous routes are intentionally unsupported"
                )
            if config.unified_object_batch:
                self._initialize_unified_batch(
                    trajectories, identity_parts, object_types, config
                )
                return
            self._initialize_heterogeneous_router(
                trajectories, identity_parts, object_types, config
            )
            return
        self.object_type = next(iter(object_types))
        self.object_types = tuple(self.object_type for _ in trajectories)
        self.action_ids = np.asarray([int(parts[1]) for parts in identity_parts], dtype=np.int64)
        object_runtime(self.object_type)
        self._object_routes: dict[str, tuple[NDArray[np.int64], MujocoManoEnvironment]] = {}
        self.config = config
        self.jax = jax
        self.jp = jax.numpy
        self.phase_timings = PhaseTimings(enabled=config.profile_phases)
        self.mjx = mjx
        self.device = devices[0]
        self.mujoco, self.model = compile_model(
            config.servo,
            object_type=self.object_type,
            hand_side=self.model_hand_side,
        )
        minimum_contact_capacity = minimum_warp_contact_capacity(
            config.num_envs, self.hand_sides
        )
        if config.contact_capacity < minimum_contact_capacity:
            raise ValueError(
                f"contact_capacity {config.contact_capacity} is below {minimum_contact_capacity} "
                f"required for {config.num_envs} {self.object_type} worlds"
            )
        self.producer = MjxWarpPhysicalProducer(
            self.mujoco,
            self.model,
            object_type=self.object_type,
            hand_sides=self.hand_sides,
            primary_hand_side=self.primary_hand_side,
        )
        if config.warp_ccd_iterations is not None:
            self.model.opt.ccd_iterations = config.warp_ccd_iterations
        self.mjx_model = mjx.put_model(self.model, device=self.device, impl="warp")
        if str(self.mjx_model.impl).lower().split(".")[-1] != "warp":
            raise RuntimeError(f"MJX did not select Warp: {self.mjx_model.impl}")
        # Controls are laid out in canonical right-then-left model order;
        # observations and policy actions may expose only the selected side.
        self.model_joint_dof = self.model_action_dim
        self.joint_lower = self.model.jnt_range[: self.model_joint_dof, 0].astype(
            np.float64, copy=True
        )
        self.joint_upper = self.model.jnt_range[: self.model_joint_dof, 1].astype(
            np.float64, copy=True
        )
        self._step_fn = jax.jit(jax.vmap(lambda world: mjx.step(self.mjx_model, world)))
        self._forward_fn = jax.jit(jax.vmap(lambda world: mjx.forward(self.mjx_model, world)))
        self._build_reference_tables()
        self._reset_qpos = self._initial_qpos()
        # Warp contact implementation metadata is static for the whole batch.
        # Replicate one capacity-configured world, then replace only dynamic
        # per-world qpos/qvel/ctrl below during reset.
        initial_host_data = self._initial_host_data(0)
        # Revised 28-DoF hands add joint-limit/contact constraints.  The
        # historical default ``constraint_capacity`` was sufficient for the
        # 26-DoF pose but can be smaller than the initial reference state's
        # solved ``nefc``. Grow the Warp buffers from the native probe while
        # retaining user-provided larger capacities.
        warp_contact_capacity = max(config.contact_capacity, int(initial_host_data.ncon) + 1)
        warp_constraint_capacity = max(config.constraint_capacity, int(initial_host_data.nefc) + 1)
        self.warp_ccd_naccdmax = (
            None if config.warp_ccd_contacts_per_world is None
            else config.warp_ccd_contacts_per_world * config.num_envs
        )
        if self.warp_ccd_naccdmax is None:
            single_data = mjx.put_data(
                self.model, initial_host_data, device=self.device, impl="warp",
                naconmax=warp_contact_capacity, njmax=warp_constraint_capacity,
            )
        else:
            # Public make_data allocates independent GJK scratch. The normal
            # reset below writes qpos/qvel/ctrl then runs forward.
            single_data = mjx.make_data(
                self.model, device=self.device, impl="warp",
                naconmax=warp_contact_capacity, naccdmax=self.warp_ccd_naccdmax,
                njmax=warp_constraint_capacity,
            )
        batch_index = jax.device_put(self.jp.arange(config.num_envs), self.device)
        self.data = jax.vmap(lambda _: single_data)(batch_index)
        self._configure_warp_ccd_overflow_guard()
        self._reset_qpos_device = jax.device_put(self._reset_qpos, self.device)
        self._reset_ctrl_device = jax.device_put(self.reference_q_model[:, 0], self.device)
        self._joint_lower_device = jax.device_put(self.joint_lower, self.device)
        self._joint_upper_device = jax.device_put(self.joint_upper, self.device)
        self._controller_targets_fn = jax.jit(self._device_controller_targets)
        self.expected_keypoint_ids = tuple(
            _expected_keypoint_ids(self.object_type, parts[1]) for parts in identity_parts
        )
        self.expected_contact_mask = np.zeros((config.num_envs, len(KEYPOINT_NAMES)), dtype=np.float64)
        for env_id, keypoint_ids in enumerate(self.expected_keypoint_ids):
            self.expected_contact_mask[env_id, keypoint_ids] = 1.0
        self.expected_contact_weights = self.expected_contact_mask.copy()
        self.active_joint_mask_by_side = _active_joint_mask(
            self.expected_contact_mask, finger_dof=self.finger_dof
        )
        self.active_joint_mask = np.tile(
            self.active_joint_mask_by_side,
            (1, len(self.hand_layout.controlled_sides)),
        )
        self.object_geometry = geometry_encoding(
            object_name=self.object_type,
            geometry_type=object_runtime(self.object_type).geometry_type,
            dimensions=np.ptp(object_collision_vertices(self.object_type), axis=0),
        )
        self.object_support_points = reduce_support_points(
            object_collision_vertices(self.object_type)
        ).copy()
        self.object_gravity_world_force = (
            np.asarray(self.model.opt.gravity, dtype=np.float64)
            * float(self.model.body_subtreemass[self.producer.object_body_id])
        )
        self.object_gravity_force = float(np.linalg.norm(self.object_gravity_world_force))
        self._point_rngs = [np.random.default_rng(config.point_seed + index) for index in range(config.num_envs)]
        if config.point_sampling_backend == POINT_SAMPLING_TORCH_CUDA_GLOBAL:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("torch_cuda_global point sampling requires CUDA")
            # Source task construction samples one random-force probability per
            # environment before policy construction and the first point reset.
            torch.rand(config.num_envs, device="cuda")
        self._static_template = _source_surface_template(42, self.object_type)
        self._dynamic_templates: NDArray[np.float64] | None = None
        self.progress = np.zeros(config.num_envs, dtype=np.int64)
        self.trajectory_steps = np.zeros(config.num_envs, dtype=np.int64)
        self.cumulative_offset = np.zeros(
            (config.num_envs, 3 * len(self.hand_layout.controlled_sides)),
            dtype=np.float64,
        )
        self.cumulative_joint_offset = np.zeros(
            (config.num_envs, self.hand_layout.cumulative_dim), dtype=np.float64
        )
        self.cumulative_offset_by_side = {
            side: np.zeros((config.num_envs, 3), dtype=np.float64)
            for side in self.hand_layout.controlled_sides
        }
        self.reset_mask = np.zeros(config.num_envs, dtype=bool)
        self.episode_returns = np.zeros(config.num_envs, dtype=np.float64)
        self.last_physical: PhysicalSnapshot | None = None
        self.last_observation: ObservationResult | None = None
        self.last_reward: RewardDiagnostics | None = None
        self.last_termination: TerminationResult | None = None
        self.last_controller_targets: NDArray[np.float64] | None = np.zeros(
            (config.num_envs, self.model_action_dim), dtype=np.float64
        )
        self.control_call = 0
        self.last_transition: TransitionSnapshot | None = None
        self._initializing_point_templates = True
        self.reset()
        self._initializing_point_templates = False
        self.reset_phase_profile()

    def _initialize_unified_batch(
        self,
        trajectories: Sequence[ReferenceTrajectory],
        identity_parts: list[list[str]],
        object_types: set[str],
        config: EnvironmentConfig,
    ) -> None:
        """Build one fixed-topology MJX model for all mixed-object worlds."""

        names = tuple(sorted(object_types))
        for object_type in names:
            object_runtime(object_type)
        self.config = config
        self.object_type = "unified"
        self.object_types = tuple(parts[0] for parts in identity_parts)
        self.action_ids = np.asarray([int(parts[1]) for parts in identity_parts], dtype=np.int64)
        self._object_routes = {}
        self._unified_object_batch = True
        self._unified_object_types = names
        type_to_index = {name: index for index, name in enumerate(names)}
        self._unified_object_indices = np.asarray(
            [type_to_index[name] for name in self.object_types], dtype=np.int64
        )
        try:
            import jax
            from mujoco import mjx
        except ImportError as exc:
            raise RuntimeError("jax and mujoco-mjx are required for the unified environment") from exc
        jax_platform = "cuda" if config.device == "gpu" else "cpu"
        devices = jax.devices(jax_platform)
        if not devices:
            raise RuntimeError(f"no JAX {jax_platform} device is available")
        self.jax = jax
        self.jp = jax.numpy
        self.device = devices[0]
        self.mjx = mjx
        self.phase_timings = PhaseTimings(enabled=config.profile_phases)
        self.mujoco, self.model = compile_unified_model(
            config.servo,
            object_types=names,
            hand_side=self.model_hand_side,
        )
        minimum_contact_capacity = minimum_warp_contact_capacity(
            config.num_envs, self.hand_sides
        )
        if config.contact_capacity < minimum_contact_capacity:
            raise ValueError(
                f"contact_capacity {config.contact_capacity} is below {minimum_contact_capacity} "
                f"required for {config.num_envs} unified worlds"
            )
        self.producer = UnifiedMjxWarpPhysicalProducer(
            self.mujoco,
            self.model,
            object_types=names,
            hand_sides=self.hand_sides,
            primary_hand_side=self.primary_hand_side,
        )
        self.producer.set_active_objects(self._unified_object_indices)
        self._unified_qpos_addresses = np.asarray(
            [
                self.model.jnt_qposadr[
                    self.mujoco.mj_name2id(
                        self.model,
                        self.mujoco.mjtObj.mjOBJ_JOINT,
                        object_runtime(object_type).free_joint_name,
                    )
                ]
                for object_type in names
            ],
            dtype=np.int64,
        )
        self.mjx_model = mjx.put_model(self.model, device=self.device, impl="warp")
        if str(self.mjx_model.impl).lower().split(".")[-1] != "warp":
            raise RuntimeError(f"MJX did not select Warp: {self.mjx_model.impl}")
        self.model_joint_dof = self.model_action_dim
        self.joint_lower = self.model.jnt_range[: self.model_joint_dof, 0].astype(
            np.float64, copy=True
        )
        self.joint_upper = self.model.jnt_range[: self.model_joint_dof, 1].astype(
            np.float64, copy=True
        )
        self._step_fn = jax.jit(jax.vmap(lambda world: mjx.step(self.mjx_model, world)))
        self._forward_fn = jax.jit(jax.vmap(lambda world: mjx.forward(self.mjx_model, world)))
        self._build_reference_tables()
        self._reset_qpos = self._initial_qpos()
        initial_host_data = self._initial_host_data(0)
        warp_contact_capacity = max(config.contact_capacity, int(initial_host_data.ncon) + 1)
        warp_constraint_capacity = max(config.constraint_capacity, int(initial_host_data.nefc) + 1)
        single_data = mjx.put_data(
            self.model,
            initial_host_data,
            device=self.device,
            impl="warp",
            naconmax=warp_contact_capacity,
            njmax=warp_constraint_capacity,
        )
        batch_index = jax.device_put(self.jp.arange(config.num_envs), self.device)
        self.data = jax.vmap(lambda _: single_data)(batch_index)
        self._reset_qpos_device = jax.device_put(self._reset_qpos, self.device)
        self._reset_ctrl_device = jax.device_put(self.reference_q_model[:, 0], self.device)
        self._joint_lower_device = jax.device_put(self.joint_lower, self.device)
        self._joint_upper_device = jax.device_put(self.joint_upper, self.device)
        self._controller_targets_fn = jax.jit(self._device_controller_targets)
        self.expected_keypoint_ids = tuple(
            _expected_keypoint_ids(parts[0], parts[1]) for parts in identity_parts
        )
        self.expected_contact_mask = np.zeros((config.num_envs, len(KEYPOINT_NAMES)), dtype=np.float64)
        for env_id, keypoint_ids in enumerate(self.expected_keypoint_ids):
            self.expected_contact_mask[env_id, keypoint_ids] = 1.0
        self.expected_contact_weights = self.expected_contact_mask.copy()
        self.active_joint_mask_by_side = _active_joint_mask(
            self.expected_contact_mask, finger_dof=self.finger_dof
        )
        self.active_joint_mask = np.tile(
            self.active_joint_mask_by_side,
            (1, len(self.hand_layout.controlled_sides)),
        )
        self.object_geometry = np.stack(
            [
                geometry_encoding(
                    object_name=object_type,
                    geometry_type=object_runtime(object_type).geometry_type,
                    dimensions=np.ptp(object_collision_vertices(object_type), axis=0),
                )
                for object_type in self.object_types
            ]
        )
        self.object_support_points = tuple(
            reduce_support_points(object_collision_vertices(object_type)).copy()
            for object_type in self.object_types
        )
        self.object_gravity_world_force = np.stack(
            [
                np.asarray(self.model.opt.gravity, dtype=np.float64)
                * float(
                    self.model.body_subtreemass[
                        self.producer.object_body_ids_by_type[type_to_index[object_type]]
                    ]
                )
                for object_type in self.object_types
            ]
        )
        self.object_gravity_force = np.linalg.norm(self.object_gravity_world_force, axis=1)
        self._static_templates = tuple(
            _source_surface_template(42, object_type) for object_type in self.object_types
        )
        self._static_template = self._static_templates[0]
        self._unified_static_point_template = PointCloudTemplate(
            np.stack([template.local_points for template in self._static_templates]),
            mode="static_seed_42",
            normalized=True,
            scale=np.stack([template.scale for template in self._static_templates]),
        )
        self._dynamic_templates = None
        self._point_rngs = [
            np.random.default_rng(config.point_seed + index) for index in range(config.num_envs)
        ]
        if config.point_sampling_backend == POINT_SAMPLING_TORCH_CUDA_GLOBAL:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("torch_cuda_global point sampling requires CUDA")
            torch.rand(config.num_envs, device="cuda")
        self.progress = np.zeros(config.num_envs, dtype=np.int64)
        self.trajectory_steps = np.zeros(config.num_envs, dtype=np.int64)
        self.cumulative_offset = np.zeros(
            (config.num_envs, 3 * len(self.hand_layout.controlled_sides)),
            dtype=np.float64,
        )
        self.cumulative_joint_offset = np.zeros(
            (config.num_envs, self.hand_layout.cumulative_dim), dtype=np.float64
        )
        self.cumulative_offset_by_side = {
            side: np.zeros((config.num_envs, 3), dtype=np.float64)
            for side in self.hand_layout.controlled_sides
        }
        self.reset_mask = np.zeros(config.num_envs, dtype=bool)
        self.episode_returns = np.zeros(config.num_envs, dtype=np.float64)
        self.last_physical: PhysicalSnapshot | None = None
        self.last_observation: ObservationResult | None = None
        self.last_reward: RewardDiagnostics | None = None
        self.last_termination: TerminationResult | None = None
        self.last_controller_targets: NDArray[np.float64] | None = np.zeros(
            (config.num_envs, self.model_action_dim), dtype=np.float64
        )
        self.control_call = 0
        self.last_transition: TransitionSnapshot | None = None
        self._initializing_point_templates = True
        self.reset()
        self._initializing_point_templates = False
        self.reset_phase_profile()

    def _initialize_heterogeneous_router(
        self,
        trajectories: Sequence[ReferenceTrajectory],
        identity_parts: list[list[str]],
        object_types: set[str],
        config: EnvironmentConfig,
    ) -> None:
        """Build one homogeneous MJX sub-batch per object and preserve env order."""

        for object_type in sorted(object_types):
            object_runtime(object_type)
        self.config = config
        self.object_type = "heterogeneous"
        self.object_types = tuple(parts[0] for parts in identity_parts)
        self.action_ids = np.asarray([int(parts[1]) for parts in identity_parts], dtype=np.int64)
        self._object_routes: dict[
            str, tuple[NDArray[np.int64], MujocoManoEnvironment]
        ] = {}
        for object_type in sorted(object_types):
            env_ids = np.asarray(
                [index for index, value in enumerate(self.object_types) if value == object_type],
                dtype=np.int64,
            )
            route_config = replace(
                config,
                num_envs=len(env_ids),
                contact_capacity=recommended_warp_contact_capacity(
                    len(env_ids), self.hand_sides
                ),
            )
            route = MujocoManoEnvironment(
                TrajectoryBatch(tuple(trajectories[int(index)] for index in env_ids)),
                route_config,
            )
            self._object_routes[object_type] = (env_ids, route)

        first_route = next(iter(self._object_routes.values()))[1]
        self.jax = first_route.jax
        self.jp = first_route.jp
        self.device = first_route.device
        self.joint_lower = first_route.joint_lower.copy()
        self.joint_upper = first_route.joint_upper.copy()
        for _, route in self._object_routes.values():
            if not np.array_equal(route.joint_lower, self.joint_lower) or not np.array_equal(
                route.joint_upper, self.joint_upper
            ):
                raise RuntimeError("heterogeneous object routes changed the MANO joint limits")

        self._build_reference_tables()
        keypoint_ids: list[NDArray[np.int64] | None] = [None] * config.num_envs
        for env_ids, route in self._object_routes.values():
            for local_id, global_id in enumerate(env_ids):
                keypoint_ids[int(global_id)] = route.expected_keypoint_ids[local_id]
        if any(value is None for value in keypoint_ids):
            raise RuntimeError("heterogeneous keypoint routing omitted an environment")
        self.expected_keypoint_ids = tuple(value for value in keypoint_ids if value is not None)
        self.expected_contact_mask = _scatter_routed_value(
            [(env_ids, route.expected_contact_mask) for env_ids, route in self._object_routes.values()],
            config.num_envs,
        )
        self.expected_contact_weights = _scatter_routed_value(
            [(env_ids, route.expected_contact_weights) for env_ids, route in self._object_routes.values()],
            config.num_envs,
        )
        self.active_joint_mask = _scatter_routed_value(
            [(env_ids, route.active_joint_mask) for env_ids, route in self._object_routes.values()],
            config.num_envs,
        )
        self.object_geometry = _scatter_routed_value(
            [
                (env_ids, np.broadcast_to(route.object_geometry, (len(env_ids), 12)).copy())
                for env_ids, route in self._object_routes.values()
            ],
            config.num_envs,
        )
        self.object_support_points = tuple(
            reduce_support_points(object_collision_vertices(object_type)).copy()
            for object_type in self.object_types
        )
        self.object_gravity_world_force = _scatter_routed_value(
            [
                (
                    env_ids,
                    np.broadcast_to(route.object_gravity_world_force, (len(env_ids), 3)).copy(),
                )
                for env_ids, route in self._object_routes.values()
            ],
            config.num_envs,
        )
        self.object_gravity_force = np.linalg.norm(self.object_gravity_world_force, axis=1)
        self._sync_heterogeneous_state()
        self.reset_phase_profile()

    @property
    def is_heterogeneous(self) -> bool:
        return bool(self._object_routes)

    def normalize_actions(self, raw_actions: NDArray[object]) -> NDArray[np.float64]:
        """Validate a live 28/56-wide MuJoCo action batch."""

        values = np.asarray(raw_actions, dtype=np.float64)
        expected = (self.config.num_envs, self.action_dim)
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ValueError(f"raw_actions must be finite ({expected[0]}, {expected[1]})")
        return values

    def _sync_heterogeneous_state(self) -> None:
        routes = list(self._object_routes.values())
        total = self.config.num_envs
        for name in (
            "progress",
            "trajectory_steps",
            "cumulative_offset",
            "cumulative_joint_offset",
            "reset_mask",
            "episode_returns",
        ):
            setattr(
                self,
                name,
                _scatter_routed_value(
                    [(env_ids, getattr(route, name)) for env_ids, route in routes], total
                ),
            )
        self.last_physical = _scatter_routed_value(
            [(env_ids, route.last_physical) for env_ids, route in routes], total
        )
        self.last_observation = _scatter_routed_value(
            [(env_ids, route.last_observation) for env_ids, route in routes], total
        )
        self.last_reward = _scatter_routed_value(
            [(env_ids, route.last_reward) for env_ids, route in routes], total
        )
        self.last_termination = _scatter_routed_value(
            [(env_ids, route.last_termination) for env_ids, route in routes], total
        )
        self.last_transition = _scatter_routed_value(
            [(env_ids, route.last_transition) for env_ids, route in routes], total
        )
        controller_values = [route.last_controller_targets for _, route in routes]
        self.last_controller_targets = (
            None
            if any(value is None for value in controller_values)
            else _scatter_routed_value(
                [(env_ids, route.last_controller_targets) for env_ids, route in routes],
                total,
            )
        )
        self.control_call = max(route.control_call for _, route in routes)

    def _heterogeneous_reset(
        self, env_ids: NDArray[object] | None
    ) -> dict[str, NDArray[np.float64]]:
        if env_ids is None:
            selected = np.arange(self.config.num_envs, dtype=np.int64)
        else:
            selected = np.asarray(env_ids)
            if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
                raise ValueError("env_ids must be a one-dimensional integer array")
            selected = np.unique(selected.astype(np.int64, copy=False))
            if np.any(selected < 0) or np.any(selected >= self.config.num_envs):
                raise ValueError("env_ids contains an invalid environment index")
        for route_env_ids, route in self._object_routes.values():
            local_ids = np.flatnonzero(np.isin(route_env_ids, selected)).astype(np.int64)
            if len(local_ids):
                route.reset(env_ids=None if env_ids is None else local_ids)
        self._sync_heterogeneous_state()
        assert self.last_observation is not None
        return {"obs": self.last_observation.policy_input.copy()}

    def _heterogeneous_step(
        self, raw_actions: NDArray[object]
    ) -> tuple[
        dict[str, NDArray[np.float64]],
        NDArray[np.float64],
        NDArray[np.bool_],
        dict[str, NDArray[Any]],
    ]:
        actions = self.normalize_actions(raw_actions)
        routed_outputs = []
        for env_ids, route in self._object_routes.values():
            observation, rewards, resets, extras = route.step(actions[env_ids])
            routed_outputs.append((env_ids, observation, rewards, resets, extras))
        self._sync_heterogeneous_state()
        observations = _scatter_routed_value(
            [(env_ids, output["obs"]) for env_ids, output, _, _, _ in routed_outputs],
            self.config.num_envs,
        )
        rewards = _scatter_routed_value(
            [(env_ids, values) for env_ids, _, values, _, _ in routed_outputs],
            self.config.num_envs,
        )
        resets = _scatter_routed_value(
            [(env_ids, values) for env_ids, _, _, values, _ in routed_outputs],
            self.config.num_envs,
        )
        extra_keys = tuple(routed_outputs[0][4])
        if any(tuple(item[4]) != extra_keys for item in routed_outputs[1:]):
            raise RuntimeError("heterogeneous routes returned incompatible step extras")
        extras = {
            key: _scatter_routed_value(
                [(env_ids, item_extras[key]) for env_ids, _, _, _, item_extras in routed_outputs],
                self.config.num_envs,
            )
            for key in extra_keys
        }
        return {"obs": observations}, rewards, resets, extras

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
        self.reference_q_by_side = {
            side: np.stack(
                [
                    np.pad(
                        _expand_legacy_hand_dofs(item.q_ref_for(side)),
                        ((0, max_length - len(item.q_ref)), (0, 0)),
                        mode="edge",
                    )
                    for item in self.trajectories
                ]
            )
            for side in self.hand_sides
        }
        self.reference_q = self.reference_q_by_side[self.primary_hand_side]
        self.reference_q_model = np.concatenate(
            [self.reference_q_by_side[side] for side in self.model_hand_sides], axis=-1
        )
        self.reference_object_pos = pad("object_pos")
        self.reference_object_quat_xyzw = pad("object_quat_xyzw")
        self.reference_source_indices = np.stack(
            [np.pad(item.source_indices, (0, max_length - len(item.source_indices)), mode="edge") for item in self.trajectories]
        )
        self.contact_start_frames = np.asarray(
            [
                item.identity.movement_start_raw - int(item.source_indices[0])
                for item in self.trajectories
            ],
            dtype=np.int64,
        )
        # ``object_move.end_frame`` is inclusive in both the source metadata
        # and reward window.  Modern decoding converts it to an exclusive
        # Python slice stop while retaining this raw inclusive identity value.
        self.contact_end_frames = np.asarray(
            [
                item.identity.movement_end_raw - int(item.source_indices[0])
                for item in self.trajectories
            ],
            dtype=np.int64,
        )
        if np.any(self.contact_start_frames < 0) or np.any(
            self.contact_end_frames >= self.trajectory_lengths
        ):
            raise ValueError("trajectory movement window does not map into its reference slice")
        # Compatibility diagnostics still expose scalar values for a single/shared trajectory.
        self.contact_start_frame = int(self.contact_start_frames[0])
        self.contact_end_frame = int(self.contact_end_frames[0])

    def _initial_qpos(self) -> NDArray[np.float64]:
        qpos = np.zeros((self.config.num_envs, self.model.nq), dtype=np.float64)
        for side_index, side in enumerate(self.model_hand_sides):
            start = side_index * self.hand_dof
            qpos[:, start : start + self.hand_dof] = self.reference_q_by_side[side][:, 0]
        if getattr(self, "_unified_object_batch", False):
            # Inactive bodies retain native gravity but start high enough that
            # a bounded episode cannot reach the floor or hand workspace.
            for object_index, address in enumerate(self._unified_qpos_addresses):
                qpos[:, address + 3] = 1.0
                qpos[:, address : address + 3] = (
                    1000.0 + 10.0 * object_index,
                    0.0,
                    1000.0,
                )
            active_addresses = self._unified_qpos_addresses[self._unified_object_indices]
            for env_id, address in enumerate(active_addresses):
                qpos[env_id, address : address + 3] = self.reference_object_pos[env_id, 0]
                qpos[env_id, address + 3 : address + 7] = xyzw_to_wxyz(
                    self.reference_object_quat_xyzw[env_id, 0]
                )
            return qpos
        address = self.producer.object_qpos_address
        qpos[:, address : address + 3] = self.reference_object_pos[:, 0]
        qpos[:, address + 3 : address + 7] = xyzw_to_wxyz(self.reference_object_quat_xyzw[:, 0])
        return qpos

    def _initial_host_data(self, env_id: int) -> Any:
        data = self.mujoco.MjData(self.model)
        self.mujoco.mj_resetData(self.model, data)
        data.qpos[:] = self._reset_qpos[env_id]
        data.qvel[:] = 0.0
        data.ctrl[:] = self.reference_q_model[env_id, 0]
        data.qfrc_applied[:] = 0.0
        self.mujoco.mj_forward(self.model, data)
        return data

    def _set_dynamic_templates(self, env_ids: NDArray[np.int64]) -> None:
        if self.config.compatibility.point_template_mode != "dynamic_reset":
            return
        if self._dynamic_templates is None:
            self._dynamic_templates = np.zeros((self.config.num_envs, POINT_COUNT, 3), dtype=np.float64)
        if self.config.point_sampling_backend == POINT_SAMPLING_TORCH_CUDA_GLOBAL:
            for object_type in (
                sorted(set(self.object_types[index] for index in env_ids))
                if getattr(self, "_unified_object_batch", False)
                else (self.object_type,)
            ):
                selected = np.asarray(
                    [
                        int(index)
                        for index in env_ids
                        if not getattr(self, "_unified_object_batch", False)
                        or self.object_types[int(index)] == object_type
                    ],
                    dtype=np.int64,
                )
                if len(selected) == 0:
                    continue
                if self._initializing_point_templates:
                    self._dynamic_templates[selected] = np.stack(
                        [_source_surface_points(42, object_type) for _ in selected]
                    )
                else:
                    self._dynamic_templates[selected] = _torch_global_surface_templates(
                        len(selected), object_type
                    )
            return
        for env_id in env_ids:
            object_type = (
                self.object_types[int(env_id)]
                if getattr(self, "_unified_object_batch", False)
                else self.object_type
            )
            self._dynamic_templates[env_id] = _dynamic_surface_template(
                self._point_rngs[int(env_id)], object_type
            )

    def reseed_point_templates(self, seed: int) -> None:
        """Set deterministic reset-local RNGs without changing static templates."""

        if not isinstance(seed, (int, np.integer)):
            raise TypeError("point-template seed must be an integer")
        if self.is_heterogeneous:
            if self.config.point_sampling_backend == POINT_SAMPLING_TORCH_CUDA_GLOBAL:
                import torch

                torch.cuda.manual_seed_all(int(seed))
            else:
                for env_ids, route in self._object_routes.values():
                    route._point_rngs = [
                        np.random.default_rng(int(seed) + int(global_id))
                        for global_id in env_ids
                    ]
            return
        if self.config.point_sampling_backend == POINT_SAMPLING_TORCH_CUDA_GLOBAL:
            import torch

            torch.cuda.manual_seed_all(int(seed))
            return
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
        if self.is_heterogeneous:
            for env_ids, route in self._object_routes.values():
                matches = np.flatnonzero(env_ids == env_id)
                if len(matches):
                    return route.host_data(int(matches[0]))
            raise RuntimeError("heterogeneous host-data route is absent")
        return self.host_data_batch()[env_id]

    def host_data_batch(self) -> list[Any]:
        """Transfer the complete batch once for tiled rendering diagnostics."""

        if self.is_heterogeneous:
            result: list[Any | None] = [None] * self.config.num_envs
            for env_ids, route in self._object_routes.values():
                for local_id, value in enumerate(route.host_data_batch()):
                    result[int(env_ids[local_id])] = value
            if any(value is None for value in result):
                raise RuntimeError("heterogeneous host-data routing omitted a world")
            return [value for value in result if value is not None]
        host_data = self.mjx.get_data(self.model, self.data)
        if not isinstance(host_data, list) or len(host_data) != self.config.num_envs:
            raise RuntimeError("batched MJX environment did not produce one host state per world")
        return host_data

    def _point_template(self) -> PointCloudTemplate:
        if self.config.compatibility.point_template_mode == "static_seed_42":
            if getattr(self, "_unified_object_batch", False):
                return PointCloudTemplate(
                    self._unified_static_point_template.local_points[
                        self._unified_object_indices
                    ],
                    mode="static_seed_42",
                    normalized=True,
                    scale=self._unified_static_point_template.scale[
                        self._unified_object_indices
                    ],
                )
            return self._static_template
        if self._dynamic_templates is None:
            raise RuntimeError("dynamic point templates were not initialized")
        return PointCloudTemplate(self._dynamic_templates.copy(), mode="dynamic_reset")

    def object_point_cloud_local(self) -> NDArray[np.float64]:
        """Return the metric 64-point template in every object's local frame."""

        if self.is_heterogeneous:
            return _scatter_routed_value(
                [
                    (env_ids, route.object_point_cloud_local())
                    for env_ids, route in self._object_routes.values()
                ],
                self.config.num_envs,
            )
        template = self._point_template()
        points = np.asarray(template.local_points, dtype=np.float64)
        batch = self.config.num_envs
        if points.shape == (POINT_COUNT, 3):
            points = np.broadcast_to(points, (batch, POINT_COUNT, 3))
        elif points.shape != (batch, POINT_COUNT, 3):
            raise ValueError("object point template has invalid batch shape")
        if template.normalized:
            if template.scale is None:
                raise ValueError("normalized object point template lacks scale")
            scale = np.asarray(template.scale, dtype=np.float64)
            if scale.shape == (3,):
                scale = np.broadcast_to(scale, (batch, 3))
            if scale.shape != (batch, 3):
                raise ValueError("normalized object point template has invalid scale shape")
            points = points * scale[:, None, :]
        return points.copy()

    def object_point_cloud_world(self, physical: PhysicalSnapshot | None = None) -> NDArray[np.float64]:
        """Return the actual 64-point object template in each world's coordinates."""

        if self.is_heterogeneous:
            resolved = self.last_physical if physical is None else physical
            if resolved is None:
                raise RuntimeError("object point cloud requires a resolved physical snapshot")
            points = self.object_point_cloud_local()
            quaternions = np.broadcast_to(
                resolved.object_orientation_xyzw[:, None, :],
                (self.config.num_envs, POINT_COUNT, 4),
            )
            return quat_rotate_xyzw(quaternions, points) + resolved.object_position[:, None, :]
        resolved = self.last_physical if physical is None else physical
        if resolved is None:
            raise RuntimeError("object point cloud requires a resolved physical snapshot")
        points = self.object_point_cloud_local()
        quaternions = np.broadcast_to(resolved.object_orientation_xyzw[:, None, :], (self.config.num_envs, POINT_COUNT, 4))
        return quat_rotate_xyzw(quaternions, points) + resolved.object_position[:, None, :]

    def _reset_indices(self, env_ids: NDArray[np.int64]) -> None:
        if len(env_ids) == 0:
            return
        writes_started = self._phase_start("reset_indexed_writes")
        if self.config.device_resident_controls:
            device_ids = self.jax.device_put(env_ids, self.device)
            self.data = self.data.replace(
                qpos=self.data.qpos.at[device_ids].set(self._reset_qpos_device[device_ids]),
                qvel=self.data.qvel.at[device_ids].set(self.jp.zeros_like(self.data.qvel[device_ids])),
                ctrl=self.data.ctrl.at[device_ids].set(self._reset_ctrl_device[device_ids]),
            )
        else:
            qpos = np.asarray(self.data.qpos, dtype=np.float64).copy()
            qvel = np.asarray(self.data.qvel, dtype=np.float64).copy()
            ctrl = np.asarray(self.data.ctrl, dtype=np.float64).copy()
            qpos[env_ids] = self._reset_qpos[env_ids]
            qvel[env_ids] = 0.0
            ctrl[env_ids] = self.reference_q_model[env_ids, 0]
            self.data = self.data.replace(
                qpos=self.jax.device_put(self.jp.asarray(qpos), self.device),
                qvel=self.jax.device_put(self.jp.asarray(qvel), self.device),
                ctrl=self.jax.device_put(self.jp.asarray(ctrl), self.device),
            )
        self._phase_stop("reset_indexed_writes", writes_started)
        forward_started = self._phase_start("reset_full_batch_forward")
        self.data = self._forward_fn(self.data)
        self._phase_stop("reset_full_batch_forward", forward_started)
        self.progress[env_ids] = 0
        self.trajectory_steps[env_ids] = 0
        self.cumulative_offset[env_ids] = 0.0
        self.cumulative_joint_offset[env_ids] = 0.0
        for values in self.cumulative_offset_by_side.values():
            values[env_ids] = 0.0
        self.reset_mask[env_ids] = False
        self.episode_returns[env_ids] = 0.0
        self._set_dynamic_templates(env_ids)

    def _profile_sync(self) -> None:
        if self.phase_timings.enabled:
            self.jax.block_until_ready(self.data.qpos)

    def _phase_start(self, name: str) -> float | None:
        return self.phase_timings.start(name, self._profile_sync if self.phase_timings.enabled else None)

    def _phase_stop(self, name: str, started: float | None) -> None:
        self.phase_timings.stop(name, started, self._profile_sync if self.phase_timings.enabled else None)

    def phase_profile(self) -> dict[str, dict[str, float | int]]:
        """Return accumulated phase timings without changing production behavior."""

        if self.is_heterogeneous:
            return {
                object_type: route.phase_profile()
                for object_type, (_, route) in self._object_routes.items()
            }
        return self.phase_timings.summary()

    def contact_profile(self) -> dict[str, object]:
        """Return opt-in contact materialization metadata and per-step nacon samples."""

        if self.is_heterogeneous:
            return {
                object_type: route.contact_profile()
                for object_type, (_, route) in self._object_routes.items()
            }
        return self.producer.contact_profile() if self.phase_timings.enabled else {}

    def reset_phase_profile(self) -> None:
        if self.is_heterogeneous:
            for _, route in self._object_routes.values():
                route.reset_phase_profile()
            return
        self.phase_timings.reset()
        self.producer.reset_profile()

    def _configure_warp_ccd_overflow_guard(self) -> None:
        self._warp_ccd_overflow_guard_available = False
        self._warp_ccd_overflow_guard_limitation: str | None = None
        if self.warp_ccd_naccdmax is None:
            return
        impl = self.data._impl
        if hasattr(impl, "naccd") and hasattr(impl, "naccdmax"):
            self._warp_ccd_overflow_guard_available = True
            return
        self._warp_ccd_overflow_guard_limitation = (
            "Pinned MJX-Warp DataWarp exposes naccdmax but no live naccd count; "
            "same-step CCD-overflow detection is unavailable for this experimental allocation."
        )
        warnings.warn(self._warp_ccd_overflow_guard_limitation, RuntimeWarning, stacklevel=2)

    def _check_warp_ccd_overflow(self) -> None:
        if not self._warp_ccd_overflow_guard_available:
            return
        count = np.asarray(self.data._impl.naccd, dtype=np.int64).reshape(-1)
        if count.shape != (1,) or not 0 <= int(count[0]) < self.warp_ccd_naccdmax:
            raise RuntimeError(
                "MJX-Warp CCD capacity saturated or ABI-incompatible on this physics substep; "
                "refusing to continue with truncated convex contacts"
            )

    def warp_ccd_metadata(self) -> dict[str, object]:
        naccdmax = getattr(self, "warp_ccd_naccdmax", None)
        guard_available = getattr(self, "_warp_ccd_overflow_guard_available", False)
        return {
            "ccd_iterations": self.config.warp_ccd_iterations,
            "contacts_per_world": self.config.warp_ccd_contacts_per_world,
            "naccdmax": naccdmax,
            "overflow_guard": "available" if guard_available else (
                "unavailable" if naccdmax is not None else "not_requested"
            ),
            "overflow_guard_limitation": getattr(self, "_warp_ccd_overflow_guard_limitation", None),
        }

    def _target_indices(self) -> NDArray[np.int64]:
        return np.minimum(np.maximum(self.trajectory_steps, 0), self.trajectory_lengths - 1)

    def _reference_gather(self, table: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
        return table[np.arange(self.config.num_envs), indices]

    def _reference_model_gather(self, indices: NDArray[np.int64]) -> NDArray[np.float64]:
        """Gather all compiled-hand references in canonical model order."""

        return np.concatenate(
            [
                self.reference_q_by_side[side][np.arange(self.config.num_envs), indices]
                for side in self.model_hand_sides
            ],
            axis=1,
        )

    def _combined_cumulative_offset(self) -> NDArray[np.float64]:
        """Expose every controlled hand's XYZ residual in action-slot order."""

        if not hasattr(self, "hand_layout") or not hasattr(self, "cumulative_offset_by_side"):
            return np.asarray(self.cumulative_offset, dtype=np.float64)
        values = [
            self.cumulative_offset_by_side[side]
            for side in self.hand_layout.controlled_sides
        ]
        if not values:
            return np.zeros((self.config.num_envs, 0), dtype=np.float64)
        return np.concatenate(values, axis=1)

    def _device_controller_targets(self, targets: Any, current_qpos: Any) -> Any:
        """Apply the source's nearest-Euler controller mapping without host state copies."""

        resolved = targets
        for side_index in range(len(self.hand_sides)):
            start = side_index * self.hand_dof
            wrist = slice(start + 3, start + 6)
            wrist_delta = (
                targets[:, wrist] - current_qpos[:, wrist] + np.pi
            ) % (2.0 * np.pi) - np.pi
            resolved = resolved.at[:, wrist].set(current_qpos[:, wrist] + wrist_delta)
        return self.jp.clip(resolved, self._joint_lower_device, self._joint_upper_device)

    def _device_transition_outputs(
        self,
        *,
        pending_reset: NDArray[np.bool_],
        prior_progress: NDArray[np.int64],
        prior_steps: NDArray[np.int64],
        prior_returns: NDArray[np.float64],
        prior_control_call: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], dict[str, NDArray[Any]]]:
        """Complete the narrow post-physics transition without host state extraction.

        The preceding delayed reset has already replaced physics state and reset
        host-owned templates/residuals. Every gather below therefore consumes
        the post-reset counters and references; only final policy/reward/done
        arrays plus compact reason/counter telemetry cross back to NumPy.
        """

        indices = self._target_indices()
        next_indices = np.minimum(indices + 5, self.trajectory_lengths - 1)
        # Reference tables are immutable for this homogeneous environment.
        # Keep their sole device copy across transitions; reset changes indices,
        # never the source trajectories.
        if not hasattr(self, "_device_reference_object_pos"):
            self._device_reference_object_pos = self.jax.device_put(self.reference_object_pos, self.device)
            self._device_reference_object_quat = self.jax.device_put(self.reference_object_quat_xyzw, self.device)
            self._device_trajectory_lengths = self.jax.device_put(self.trajectory_lengths, self.device)
            self._device_contact_start_frames = self.jax.device_put(self.contact_start_frames, self.device)
            self._device_contact_end_frames = self.jax.device_put(self.contact_end_frames, self.device)
            self._device_expected_contact_mask = self.jax.device_put(self.expected_contact_mask, self.device)
            self._device_expected_contact_weights = self.jax.device_put(self.expected_contact_weights, self.device)
            self._device_active_joint_mask = self.jax.device_put(self.active_joint_mask, self.device)
            self._device_action_ids = self.jax.device_put(self.action_ids, self.device)
            self._device_object_geometry = self.jax.device_put(
                np.broadcast_to(self.object_geometry, (self.config.num_envs, 12)), self.device
            )
            self._device_object_support_points = self.jax.device_put(self.object_support_points, self.device)
        reference_pos = self._device_reference_object_pos
        reference_quat = self._device_reference_object_quat
        device_indices = self.jax.device_put(indices, self.device)
        device_next_indices = self.jax.device_put(next_indices, self.device)
        world = self.jp.arange(self.config.num_envs)
        physical = extract_mjx_physical_features(
            qpos=self.data.qpos, qvel=self.data.qvel, xpos=self.data.xpos, xquat=self.data.xquat,
            hand_qpos_start=self.producer.hand_qpos_slices["right"].start,
            hand_dof=JOINT_DOF, object_body_id=self.producer.object_body_id,
            object_qvel_address=self.producer.object_qvel_address,
            keypoint_body_ids=tuple(self.producer.keypoint_body_ids),
            fingertip_keypoint_ids=tuple(KEYPOINT_NAMES.index(name) for name in _FINGERTIP_NAMES),
            fingertip_local_offsets=_FINGERTIP_LOCAL_OFFSETS,
        )
        contacts = self.producer.device_contact_reduction(self.data)
        target_pos = reference_pos[world, device_indices]
        target_quat = reference_quat[world, device_indices]
        termination = check_device_termination(
            object_position=physical.object_position, target_position=target_pos,
            progress=self.jax.device_put(self.progress, self.device),
            trajectory_lengths=self._device_trajectory_lengths,
            early_mask=self.jax.device_put(early_phase_mask(
                self.trajectory_steps, starts=np.zeros(self.config.num_envs, dtype=np.int64),
                steps=self.config.compatibility.early_phase_steps,
            ), self.device),
            max_deviation_distance=self.config.max_deviation_distance,
            deviation_penalty=self.config.deviation_penalty,
        )
        reward = compute_device_reward_28(
            object_position=physical.object_position, target_object_position=target_pos,
            object_orientation_xyzw=physical.object_orientation_xyzw,
            target_object_orientation_xyzw=target_quat,
            cumulative_offset=self.jax.device_put(self.cumulative_offset, self.device),
            cumulative_joint_offset=self.jax.device_put(self.cumulative_joint_offset, self.device),
            active_joint_mask=self._device_active_joint_mask,
            hand_object_force_on_object_world_N=contacts.hand_object_forces,
            expected_contact_mask=self._device_expected_contact_mask,
            expected_contact_weights=self._device_expected_contact_weights,
            object_linear_velocity=physical.object_linear_velocity,
            trajectory_steps=self.jax.device_put(self.trajectory_steps, self.device),
            contact_start_frames=self._device_contact_start_frames,
            contact_end_frames=self._device_contact_end_frames,
            rotation_disabled_mask=self.jp.zeros(self.config.num_envs, dtype=bool),
            early_phase_starts=self.jp.zeros(self.config.num_envs, dtype=np.int64),
            early_phase_steps=self.config.compatibility.early_phase_steps,
            termination=termination, config=self.config.reward_config,
        )
        template = self._point_template()
        point_cloud = np.asarray(template.local_points, dtype=np.float64)
        point_scale = (
            np.asarray(template.scale, dtype=np.float64)
            if template.normalized and template.scale is not None
            else np.ones(3, dtype=np.float64)
        )
        raw_observation, observation_valid = build_device_observation_28(
            physical=physical, hand_keypoint_contact_forces=contacts.keypoint_forces,
            target_object_position=target_pos, target_object_orientation_xyzw=target_quat,
            target_object_pos_next_5=reference_pos[world, device_next_indices],
            cumulative_offset=self.jax.device_put(self.cumulative_offset, self.device),
            cumulative_joint_offset=self.jax.device_put(self.cumulative_joint_offset, self.device),
            point_cloud_local=self.jax.device_put(point_cloud, self.device),
            point_cloud_scale=self.jax.device_put(point_scale, self.device),
            object_geometry=self._device_object_geometry,
            expected_contact_mask=self._device_expected_contact_mask,
            action_ids=self._device_action_ids,
            object_support_points=self._device_object_support_points,
            table_surface_height=FLOOR_TOP_Z,
            mano_dof_lower=self._joint_lower_device[:JOINT_DOF], mano_dof_upper=self._joint_upper_device[:JOINT_DOF],
        )
        counters = advance_device_task_counters(
            progress=self.jax.device_put(prior_progress, self.device),
            trajectory_steps=self.jax.device_put(prior_steps, self.device),
            episode_returns=self.jax.device_put(prior_returns, self.device),
            pending_reset=self.jax.device_put(pending_reset, self.device), reward_total=reward.total,
            next_reset=termination.reset, control_call=self.jp.asarray(prior_control_call, dtype=np.int64),
        )
        valid = physical.valid & contacts.valid & termination.valid & reward.valid & observation_valid
        if not bool(np.asarray(valid)):
            raise RuntimeError("device_transition rejected non-finite or invalid transition inputs")
        observation = np.asarray(raw_observation, dtype=np.float64)
        reward_total = np.asarray(reward.total, dtype=np.float64)
        reset = np.asarray(termination.reset, dtype=bool)
        # JAX-to-NumPy conversion can yield a read-only view.  These compact
        # counters cross back into host-owned task state and are subsequently
        # updated by indexed delayed resets, so retain writable ownership.
        self.progress = np.asarray(counters.progress, dtype=np.int64).copy()
        self.trajectory_steps = np.asarray(counters.trajectory_steps, dtype=np.int64).copy()
        self.episode_returns = np.asarray(counters.episode_returns, dtype=np.float64).copy()
        self.reset_mask = np.asarray(counters.reset_mask, dtype=bool).copy()
        if not all(array.flags.writeable for array in (
            self.progress, self.trajectory_steps, self.episode_returns, self.reset_mask,
        )):
            raise RuntimeError("device transition counters must be writable host arrays")
        self.control_call = int(np.asarray(counters.control_call))
        self.last_physical = None
        self.last_observation = None
        # These compact vectors already cross the Gym boundary.  Materialize
        # the complete diagnostics contract from kernel outputs so training
        # telemetry observes the reward actually used for this transition.
        self.last_reward = RewardDiagnostics(
            total=reward_total.copy(),
            distance_x=np.asarray(reward.distance_x, dtype=np.float64),
            distance_y=np.asarray(reward.distance_y, dtype=np.float64),
            distance_z=np.asarray(reward.distance_z, dtype=np.float64),
            ungated_distance_x=np.asarray(reward.ungated_distance_x, dtype=np.float64),
            ungated_distance_y=np.asarray(reward.ungated_distance_y, dtype=np.float64),
            ungated_distance_z=np.asarray(reward.ungated_distance_z, dtype=np.float64),
            rotation=np.asarray(reward.rotation, dtype=np.float64),
            position_penalty=np.asarray(reward.position_penalty, dtype=np.float64),
            joint_penalty=np.asarray(reward.joint_penalty, dtype=np.float64),
            action_penalty=np.asarray(reward.action_penalty, dtype=np.float64),
            raw_contact=np.asarray(reward.raw_contact, dtype=np.float64),
            contact=np.asarray(reward.contact, dtype=np.float64),
            distance_gate=np.asarray(reward.distance_gate, dtype=np.float64),
            object_stability=np.asarray(reward.object_stability, dtype=np.float64),
            object_speed=np.asarray(reward.object_speed, dtype=np.float64),
            survival=np.asarray(reward.survival, dtype=np.float64),
            early_phase=np.asarray(reward.early_phase, dtype=bool),
            deviation_penalty=np.asarray(termination.deviation_penalty, dtype=np.float64),
        )
        self.last_termination = TerminationResult(
            reset=reset.copy(),
            deviation_reset=np.asarray(termination.deviation_reset, dtype=bool),
            deviation_penalty=np.asarray(termination.deviation_penalty, dtype=np.float64),
        )
        self.last_transition = None
        extras = {
            "time_outs": np.zeros(self.config.num_envs, dtype=bool),
            "termination_reason_code": self.last_termination.reason_code.copy(),
            "termination_success": self.last_termination.success.copy(),
            "termination_failure": self.last_termination.failure.copy(),
            "trajectory_complete_reset_mask": self.last_termination.success.copy(),
            "deviation_reset_mask": self.last_termination.deviation_reset.copy(),
        }
        return np.clip(observation, -5.0, 5.0), reward_total, reset, extras

    def _build_observation(self, physical: PhysicalSnapshot) -> ObservationResult:
        indices = self._target_indices()
        next_indices = np.minimum(indices + 5, self.trajectory_lengths - 1)
        primary_index = self.model_hand_sides.index(self.primary_hand_side)
        primary_slice = slice(
            primary_index * self.hand_dof,
            (primary_index + 1) * self.hand_dof,
        )
        state = ObservationState(
            mano_dof_pos=physical.mano_dof_pos,
            mano_dof_lower=self.joint_lower[primary_slice],
            mano_dof_upper=self.joint_upper[primary_slice],
            hand_position=physical.hand_position,
            hand_orientation_xyzw=physical.hand_orientation_xyzw,
            object_position=physical.object_position,
            object_orientation_xyzw=physical.object_orientation_xyzw,
            target_object_position=self._reference_gather(self.reference_object_pos, indices),
            target_object_orientation_xyzw=self._reference_gather(self.reference_object_quat_xyzw, indices),
            target_object_pos_next_5=self._reference_gather(self.reference_object_pos, next_indices),
            cumulative_offset=self._combined_cumulative_offset(),
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
            cumulative_offset=self._combined_cumulative_offset(),
            cumulative_joint_offset=self.cumulative_joint_offset,
            active_joint_mask=self.active_joint_mask,
            hand_object_force_on_object_world_N=physical.hand_object_force_on_object_world_N,
            expected_contact_mask=self.expected_contact_mask,
            expected_contact_weights=self.expected_contact_weights,
            object_linear_velocity=physical.object_linear_velocity,
            trajectory_steps=self.trajectory_steps.copy(),
            contact_start_frames=self.contact_start_frames.copy(),
            contact_end_frames=self.contact_end_frames.copy(),
            rotation_disabled_mask=np.zeros(batch, dtype=bool),
            early_phase_starts=np.zeros(batch, dtype=np.int64),
        )

    def _refresh_output(self) -> ObservationResult:
        self.last_physical = self.producer.extract(
            self.data,
            timings=self.phase_timings if self.phase_timings.enabled else None,
            synchronize=self._profile_sync if self.phase_timings.enabled else None,
            record_profile=self.phase_timings.enabled,
            device_contact_decode=self.config.device_contact_decode,
        )
        self.last_observation = self._build_observation(self.last_physical)
        return self.last_observation

    def reset(self, env_ids: NDArray[object] | None = None) -> dict[str, NDArray[np.float64]]:
        if self.is_heterogeneous:
            return self._heterogeneous_reset(env_ids)
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
    ) -> tuple[
        dict[str, NDArray[np.float64]],
        NDArray[np.float64],
        NDArray[np.bool_],
        dict[str, NDArray[Any]],
    ]:
        if self.is_heterogeneous:
            return self._heterogeneous_step(raw_actions)
        materialization_phase = self._phase_start("action_numpy_materialization")
        actions = self.normalize_actions(raw_actions)
        self._phase_stop("action_numpy_materialization", materialization_phase)
        action_phase = self._phase_start("action_conversion_processing")
        mocap_indices = self._target_indices()
        actions_by_side = self.hand_layout.split(actions)
        action_results: dict[str, Any] = {}
        for side in self.hand_layout.controlled_sides:
            cumulative_slice = self.hand_layout.cumulative_slice(side)
            model_index = self.model_hand_sides.index(side)
            model_slice = slice(
                model_index * self.hand_dof,
                (model_index + 1) * self.hand_dof,
            )
            side_result = process_residual_actions(
                actions_by_side[side],
                trajectory_steps=self.trajectory_steps,
                cumulative_offset=self.cumulative_offset_by_side[side],
                cumulative_joint_offset=self.cumulative_joint_offset[:, cumulative_slice],
                mocap_targets=self._reference_gather(
                    self.reference_q_by_side[side], mocap_indices
                ),
                joint_lower=self.joint_lower[model_slice],
                joint_upper=self.joint_upper[model_slice],
                active_joint_mask=self.active_joint_mask_by_side,
                use_residual=np.full(
                    self.config.num_envs,
                    self.config.residual_enabled,
                    dtype=np.float64,
                ),
                config=self.config.residual_action,
            )
            action_results[side] = side_result
            self.cumulative_offset_by_side[side] = side_result.cumulative_offset
            self.cumulative_joint_offset[:, cumulative_slice] = side_result.cumulative_joint_offset
        self.cumulative_offset = self._combined_cumulative_offset()
        mocap_targets = self._reference_model_gather(mocap_indices)
        processed_targets = np.concatenate(
            [
                action_results[side].targets
                if side in action_results
                else self._reference_gather(self.reference_q_by_side[side], mocap_indices)
                for side in self.model_hand_sides
            ],
            axis=1,
        )
        self._phase_stop("action_conversion_processing", action_phase)
        controller_phase = self._phase_start("controller_target_work")
        if self.config.device_resident_controls:
            controller_targets_device = self._controller_targets_fn(
                self.jax.device_put(processed_targets, self.device),
                self.data.qpos[:, : self.model_action_dim],
            )
            controller_targets = (
                np.asarray(controller_targets_device, dtype=np.float64)
                if self.config.capture_transition_diagnostics
                else None
            )
        else:
            current_qpos = np.asarray(self.data.qpos, dtype=np.float64)[:, : self.model_action_dim]
            controller_targets = np.stack(
                [
                    np.concatenate(
                        [
                            command_target(
                                processed_targets[index, model_index * self.hand_dof : (model_index + 1) * self.hand_dof],
                                current_qpos[index, model_index * self.hand_dof : (model_index + 1) * self.hand_dof],
                                self.joint_lower[model_index * self.hand_dof : (model_index + 1) * self.hand_dof],
                                self.joint_upper[model_index * self.hand_dof : (model_index + 1) * self.hand_dof],
                            )
                            for model_index, _ in enumerate(self.model_hand_sides)
                        ]
                    )
                    for index in range(self.config.num_envs)
                ]
            )
            controller_targets_device = self.jax.device_put(self.jp.asarray(controller_targets), self.device)
        self.last_controller_targets = None if controller_targets is None else controller_targets.copy()
        # Preserve pre-step values for the device counter primitive. The host
        # fields below still drive controller/reset ordering until its result
        # is committed at the narrow transition boundary.
        prior_progress = self.progress.copy()
        prior_steps = self.trajectory_steps.copy()
        prior_returns = self.episode_returns.copy()
        prior_control_call = self.control_call
        self.trajectory_steps += 1
        self.trajectory_steps[self.progress == 0] = 0
        self._phase_stop("controller_target_work", controller_phase)
        self.data = self.data.replace(ctrl=controller_targets_device)
        physics_phase = self._phase_start("mjx_physics")
        for _ in range(PHYSICS_SUBSTEPS_PER_TARGET):
            self.data = self._step_fn(self.data)
            self._check_warp_ccd_overflow()
        self._phase_stop("mjx_physics", physics_phase)
        self.progress += 1
        pending_reset = self.reset_mask.copy()
        reset_phase = self._phase_start("delayed_reset_application")
        if np.any(pending_reset):
            self._reset_indices(np.flatnonzero(pending_reset).astype(np.int64))
        self._phase_stop("delayed_reset_application", reset_phase)
        if self.config.device_transition:
            observation, reward_total, reset, extras = self._device_transition_outputs(
                pending_reset=pending_reset,
                prior_progress=prior_progress,
                prior_steps=prior_steps,
                prior_returns=prior_returns,
                prior_control_call=prior_control_call,
            )
            return {"obs": observation}, reward_total, reset, extras
        extraction_phase = self._phase_start("state_contact_extraction")
        physical = self.producer.extract(
            self.data,
            timings=self.phase_timings if self.phase_timings.enabled else None,
            synchronize=self._profile_sync if self.phase_timings.enabled else None,
            record_profile=self.phase_timings.enabled,
            device_contact_decode=self.config.device_contact_decode,
        )
        self._phase_stop("state_contact_extraction", extraction_phase)
        early = early_phase_mask(
            self.trajectory_steps,
            starts=np.zeros(self.config.num_envs, dtype=np.int64),
            steps=self.config.compatibility.early_phase_steps,
        )
        termination_phase = self._phase_start("termination")
        termination = check_termination(
            object_position=physical.object_position,
            target_position=self._reference_gather(self.reference_object_pos, self._target_indices()),
            progress=self.progress,
            trajectory_lengths=self.trajectory_lengths.copy(),
            early_mask=early,
            max_deviation_distance=self.config.max_deviation_distance,
            deviation_penalty=self.config.deviation_penalty,
        )
        self._phase_stop("termination", termination_phase)
        reward_phase = self._phase_start("reward")
        reward = compute_rewards(
            self._reward_state(physical),
            compatibility=self.config.compatibility,
            termination=termination,
            config=self.config.reward_config,
        )
        self._phase_stop("reward", reward_phase)
        observation_phase = self._phase_start("observation")
        observation = self._build_observation(physical)
        self._phase_stop("observation", observation_phase)
        self.reset_mask = termination.reset.copy()
        self.episode_returns += reward.total
        self.last_physical = physical
        self.last_observation = observation
        self.last_termination = termination
        self.last_reward = reward
        recording_phase = self._phase_start("transition_recording")
        if self.config.capture_transition_diagnostics:
            if controller_targets is None:
                raise RuntimeError("transition diagnostics require host controller targets")
            self.last_transition = TransitionSnapshot(
                control_call=self.control_call,
                raw_actions=actions.copy(),
                command_reference_indices=mocap_indices.copy(),
                command_targets=mocap_targets.copy(),
                processed_targets=processed_targets.copy(),
                controller_targets=controller_targets.copy(),
                reset_applied=pending_reset.copy(),
                progress=self.progress.copy(),
                trajectory_steps=self.trajectory_steps.copy(),
                target_indices=self._target_indices().copy(),
                physical=physical,
                observation=observation,
                reward=reward,
                episode_return=self.episode_returns.copy(),
                termination=termination,
            )
        else:
            self.last_transition = None
        self._phase_stop("transition_recording", recording_phase)
        self.control_call += 1
        # ``episode_length`` is a rollout/statistics setting in this source
        # slice, not an independent physics horizon. A source trajectory end
        # or deviation is therefore a terminal transition, never a timeout.
        timeouts = np.zeros(self.config.num_envs, dtype=bool)
        return (
            {"obs": observation.policy_input.copy()},
            reward.total.copy(),
            termination.reset.copy(),
            {
                "time_outs": timeouts.copy(),
                "termination_reason_code": termination.reason_code.copy(),
                "termination_success": termination.success.copy(),
                "termination_failure": termination.failure.copy(),
                "trajectory_complete_reset_mask": termination.success.copy(),
                "deviation_reset_mask": termination.deviation_reset.copy(),
            },
        )
