"""Synthesis-only human-like approach-prefix augmentation.

The transform prepends an immutable human-like wrist approach to an existing reference
trajectory.  The original reference (including its pre-padding, movement
window, and post-padding) is copied unchanged after the prefix.  Sampling is
fully determined by ``seed`` and the trajectory identity so fresh synthesis
attempts produce fresh starts without mutable environment references.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from sim.manorl.trajectory import ReferenceTrajectory

APPROACH_PREFIX_CONTRACT: Final = (
    "synthetic_approach_prefix_far_or_retreat_xy_near_world_z_c1_v4"
)
APPROACH_MODES: Final = ("far", "near")
RETREAT_SUFFIX_CONTRACT: Final = (
    "synthetic_parent_movement_end_plus15_retreat_tail_xy_extra_z_extra_discrete_c2_v4"
)


def _readonly(values: object, *, dtype: np.dtype[object] | type = np.float64) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ApproachPrefixConfig:
    """Bounded synthesis-only approach sampling contract."""

    mode: str = "far"
    required_base_pre_padding: int = 60
    minimum_xy_radius_m: float = 0.30
    maximum_xy_radius_m: float = 0.70
    maximum_xy_offset_deg: float = 30.0
    minimum_z_offset_m: float = 0.08
    maximum_z_offset_m: float = 0.30
    minimum_total_pre_padding: int = 100
    maximum_total_pre_padding: int = 360
    nominal_speed_m_s: float = 0.30
    nominal_angular_speed_deg_s: float = 90.0
    duration_jitter_fraction: float = 0.10
    vertical_arc_height_m: float = 0.04

    def __post_init__(self) -> None:
        if self.mode not in APPROACH_MODES:
            raise ValueError(f"approach-prefix mode must be one of {APPROACH_MODES}")
        if self.mode == "near" and (
            self.minimum_xy_radius_m,
            self.maximum_xy_radius_m,
            self.maximum_xy_offset_deg,
            self.minimum_z_offset_m,
            self.maximum_z_offset_m,
        ) != (0.30, 0.70, 30.0, 0.08, 0.30):
            raise ValueError(
                "near approach does not accept far-only XY/Z distribution overrides"
            )
        finite = (
            self.minimum_xy_radius_m,
            self.maximum_xy_radius_m,
            self.maximum_xy_offset_deg,
            self.minimum_z_offset_m,
            self.maximum_z_offset_m,
            self.nominal_speed_m_s,
            self.nominal_angular_speed_deg_s,
            self.duration_jitter_fraction,
            self.vertical_arc_height_m,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("approach-prefix parameters must be finite")
        if (
            not isinstance(self.required_base_pre_padding, int)
            or isinstance(self.required_base_pre_padding, bool)
            or self.required_base_pre_padding < 1
        ):
            raise ValueError("required base pre-padding must be a positive integer")
        if not 0.0 < self.minimum_xy_radius_m <= self.maximum_xy_radius_m:
            raise ValueError(
                "approach-prefix XY-radius bounds must be positive and ordered"
            )
        if not 0.0 <= self.maximum_xy_offset_deg <= 180.0:
            raise ValueError("approach-prefix XY angle must be in [0, 180] degrees")
        if not 0.0 < self.minimum_z_offset_m <= self.maximum_z_offset_m:
            raise ValueError(
                "approach-prefix Z-offset bounds must be positive and ordered"
            )
        if (
            not isinstance(self.minimum_total_pre_padding, int)
            or isinstance(self.minimum_total_pre_padding, bool)
            or not isinstance(self.maximum_total_pre_padding, int)
            or isinstance(self.maximum_total_pre_padding, bool)
            or self.minimum_total_pre_padding < 1
            or self.maximum_total_pre_padding < self.minimum_total_pre_padding
        ):
            raise ValueError("total pre-padding bounds must be ordered positive integers")
        if self.nominal_speed_m_s <= 0.0 or self.nominal_angular_speed_deg_s <= 0.0:
            raise ValueError("approach-prefix nominal speeds must be positive")
        if not 0.0 <= self.duration_jitter_fraction <= 0.5:
            raise ValueError("approach-prefix duration jitter must be in [0, 0.5]")
        if not 0.0 <= self.vertical_arc_height_m <= 0.10:
            raise ValueError("approach-prefix vertical arc height must be in [0, 0.10] m")


@dataclass(frozen=True)
class RetreatSuffixConfig:
    """Randomized replacement of the post-movement retreat tail."""

    minimum_extra_horizontal_m: float = 0.03
    maximum_extra_horizontal_m: float = 0.15
    maximum_xy_offset_deg: float = 30.0
    minimum_z_offset_m: float = 0.04
    maximum_z_offset_m: float = 0.10

    def __post_init__(self) -> None:
        finite = (
            self.minimum_extra_horizontal_m,
            self.maximum_extra_horizontal_m,
            self.maximum_xy_offset_deg,
            self.minimum_z_offset_m,
            self.maximum_z_offset_m,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("retreat-suffix parameters must be finite")
        if not 0.0 <= self.minimum_extra_horizontal_m <= self.maximum_extra_horizontal_m:
            raise ValueError(
                "retreat-suffix extra-horizontal bounds must be non-negative and ordered"
            )
        if not 0.0 <= self.maximum_xy_offset_deg <= 180.0:
            raise ValueError("retreat-suffix XY angle must be in [0, 180] degrees")
        if not 0.0 <= self.minimum_z_offset_m <= self.maximum_z_offset_m:
            raise ValueError(
                "retreat-suffix Z-offset bounds must be non-negative and ordered"
            )


@dataclass(frozen=True)
class RetreatSuffixSample:
    """One resolved replacement of the original post-contact retreat tail."""

    contract: str
    source_identity: str
    seed: int
    anchor_reference_index: int
    suffix_frames: int
    replaced_tail_frames: int
    original_horizontal_distance_m: float
    extra_horizontal_offset_m: float
    end_horizontal_distance_m: float
    original_z_displacement_m: float
    extra_z_offset_m: float
    end_z_displacement_m: float
    end_distance_m: float
    original_direction_deg: float
    xy_offset_deg: float
    end_position_m: tuple[float, float, float]
    splice_position_m: tuple[float, float, float]
    original_end_position_m: tuple[float, float, float]
    splice_velocity_error_m_s: float
    splice_acceleration_jump_m_s2: float

    def to_manifest(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "source_identity": self.source_identity,
            "seed": self.seed,
            "anchor_reference_index": self.anchor_reference_index,
            "suffix_frames": self.suffix_frames,
            "replaced_tail_frames": self.replaced_tail_frames,
            "original_horizontal_distance_m": self.original_horizontal_distance_m,
            "extra_horizontal_offset_m": self.extra_horizontal_offset_m,
            "end_horizontal_distance_m": self.end_horizontal_distance_m,
            "original_z_displacement_m": self.original_z_displacement_m,
            "extra_z_offset_m": self.extra_z_offset_m,
            "end_z_displacement_m": self.end_z_displacement_m,
            "end_distance_m": self.end_distance_m,
            "original_direction_deg": self.original_direction_deg,
            "xy_offset_deg": self.xy_offset_deg,
            "end_position_m": list(self.end_position_m),
            "splice_position_m": list(self.splice_position_m),
            "original_end_position_m": list(self.original_end_position_m),
            "splice_velocity_error_m_s": self.splice_velocity_error_m_s,
            "splice_acceleration_jump_m_s2": self.splice_acceleration_jump_m_s2,
        }


@dataclass(frozen=True)
class _RetreatEndpointSample:
    original_horizontal_distance_m: float
    extra_horizontal_offset_m: float
    end_horizontal_distance_m: float
    original_z_displacement_m: float
    extra_z_offset_m: float
    end_z_displacement_m: float
    original_direction_deg: float
    xy_offset_deg: float
    end_position_m: tuple[float, float, float]


@dataclass(frozen=True)
class ApproachPrefixSample:
    """One resolved per-trajectory approach-prefix sample."""

    contract: str
    source_identity: str
    seed: int
    approach_mode: str
    direction_reference: str
    base_pre_padding: int
    prefix_frames: int
    effective_pre_padding: int
    start_xy_radius_m: float
    start_z_offset_m: float
    start_distance_m: float
    xy_offset_deg: float
    orientation_template: str
    start_position_m: tuple[float, float, float]
    splice_position_m: tuple[float, float, float]
    splice_position_error_m: float
    splice_velocity_error_m_s: float
    splice_acceleration_jump_m_s2: float

    def to_manifest(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "source_identity": self.source_identity,
            "seed": self.seed,
            "approach_mode": self.approach_mode,
            "direction_reference": self.direction_reference,
            "base_pre_padding": self.base_pre_padding,
            "prefix_frames": self.prefix_frames,
            "effective_pre_padding": self.effective_pre_padding,
            "start_xy_radius_m": self.start_xy_radius_m,
            "start_z_offset_m": self.start_z_offset_m,
            "start_distance_m": self.start_distance_m,
            "xy_offset_deg": self.xy_offset_deg,
            "orientation_template": self.orientation_template,
            "start_position_m": list(self.start_position_m),
            "splice_position_m": list(self.splice_position_m),
            "splice_position_error_m": self.splice_position_error_m,
            "splice_velocity_error_m_s": self.splice_velocity_error_m_s,
            "splice_acceleration_jump_m_s2": self.splice_acceleration_jump_m_s2,
        }


def _identity_seed(seed: int, identity: str) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("approach-prefix seed must be a non-negative integer")
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def augmentation_stream_seed(seed: int, stream: str) -> int:
    """Derive one explicit deterministic child seed from an episode seed."""

    if not isinstance(stream, str) or not stream:
        raise ValueError("augmentation stream name must be non-empty")
    return _identity_seed(seed, f"manorl-augmentation-stream:{stream}")


def _endpoint_derivatives(values: NDArray[np.float64], dt: float) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < 3:
        raise ValueError("approach-prefix splice requires at least three source frames")
    velocity = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * dt)
    acceleration = (values[0] - 2.0 * values[1] + values[2]) / (dt * dt)
    return velocity, acceleration


def _quintic_prefix(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    end_velocity: NDArray[np.float64],
    end_acceleration: NDArray[np.float64],
    *,
    frames: int,
    dt: float,
    start_velocity: NDArray[np.float64] | None = None,
    start_acceleration: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Sample before an excluded endpoint with zero start velocity/acceleration.

    ``start_velocity``/``start_acceleration`` default to zero (the approach
    prefix contract); the retreat suffix passes the source tail velocity so
    the sampled velocity stays continuous across the append splice.
    """

    if frames < 1:
        raise ValueError("approach-prefix frames must be positive")
    duration = frames * dt
    c0 = np.asarray(start, dtype=np.float64)
    c1 = (
        np.zeros_like(c0)
        if start_velocity is None
        else np.asarray(start_velocity, dtype=np.float64)
    )
    c2 = (
        np.zeros_like(c0)
        if start_acceleration is None
        else 0.5 * np.asarray(start_acceleration, dtype=np.float64)
    )
    displacement = np.asarray(end, dtype=np.float64) - c0
    # Solve the final three coefficients independently for every coordinate.
    matrix = np.asarray(
        (
            (duration**3, duration**4, duration**5),
            (3.0 * duration**2, 4.0 * duration**3, 5.0 * duration**4),
            (6.0 * duration, 12.0 * duration**2, 20.0 * duration**3),
        ),
        dtype=np.float64,
    )
    rhs = np.stack(
        (
            displacement - c1 * duration - c2 * duration**2,
            np.asarray(end_velocity, dtype=np.float64)
            - c1
            - 2.0 * c2 * duration,
            np.asarray(end_acceleration, dtype=np.float64) - 2.0 * c2,
        ),
        axis=0,
    )
    c3, c4, c5 = np.linalg.solve(matrix, rhs)
    times = np.arange(frames, dtype=np.float64)[:, None] * dt
    return c0 + c1 * times + c2 * times**2 + c3 * times**3 + c4 * times**4 + c5 * times**5


def _discrete_c1_prefix(
    start: NDArray[np.float64],
    source_first_two: NDArray[np.float64],
    *,
    frames: int,
    dt: float,
    vertical_arc_height: float = 0.0,
) -> NDArray[np.float64]:
    """Generate a prefix with exact sampled velocity continuity at source frame0.

    The final prefix state is ``2*x0-x1``; therefore its interval into source
    ``x0`` is exactly the source ``x0->x1`` interval. Raw capture acceleration
    is deliberately not extrapolated backward: finite-difference noise there
    produced non-human spikes across the full catalog audit.
    """

    source = np.asarray(source_first_two, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != 2:
        raise ValueError("discrete C1 splice requires exactly two source frames")
    if frames < 2:
        raise ValueError("discrete C1 prefix requires at least two frames")
    x_minus_one = 2.0 * source[0] - source[1]
    velocity = (source[1] - source[0]) / dt
    core_frames = frames - 1
    core = _quintic_prefix(
        start,
        x_minus_one,
        velocity,
        np.zeros_like(velocity),
        frames=core_frames,
        dt=dt,
    )
    if vertical_arc_height:
        phase = np.arange(core_frames, dtype=np.float64) / float(core_frames)
        core[:, 2] += (
            vertical_arc_height
            * 64.0
            * phase**3
            * (1.0 - phase) ** 3
        )
    return np.concatenate((core, x_minus_one[None]), axis=0)


def _deform_retreat_tail(
    source_tail: NDArray[np.float64],
    end: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Preserve the source retreat while smoothly moving its endpoint.

    ``source_tail`` begins at anchor+1. The deformation weight is zero for its
    first two frames and one for its last three frames. Consequently the added
    displacement has zero discrete velocity/acceleration at both boundaries:
    the original release/retreat dynamics are untouched at the anchor, and the
    transformed trajectory inherits the source's stationary post-padding end.
    A quintic smoothstep distributes the endpoint displacement between them.
    """

    source = np.asarray(source_tail, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or len(source) < 6:
        raise ValueError("retreat-tail deformation requires at least six XYZ frames")
    weights = np.zeros(len(source), dtype=np.float64)
    phase = np.linspace(0.0, 1.0, len(source) - 3, dtype=np.float64)
    weights[1:-2] = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    weights[-2:] = 1.0
    endpoint_displacement = np.asarray(end, dtype=np.float64) - source[-1]
    return source + weights[:, None] * endpoint_displacement[None]


def _sample_retreat_endpoint(
    trajectory: ReferenceTrajectory,
    *,
    seed: int,
    anchor_reference_index: int,
    config: RetreatSuffixConfig,
) -> _RetreatEndpointSample:
    anchor = int(anchor_reference_index)
    if not 2 <= anchor < len(trajectory.q_ref) - 3:
        raise ValueError(
            "retreat anchor must leave two source frames before and three tail frames after"
        )
    anchor_position = np.asarray(trajectory.q_ref[anchor, :3], dtype=np.float64)
    original_end = np.asarray(trajectory.q_ref[-1, :3], dtype=np.float64)
    original_delta = original_end - anchor_position
    original_horizontal = float(np.linalg.norm(original_delta[:2]))
    if original_horizontal <= 1e-9:
        raise ValueError("original retreat tail has no nonzero horizontal direction")
    original_direction = math.atan2(
        float(original_delta[1]), float(original_delta[0])
    )
    rng = np.random.default_rng(
        _identity_seed(seed, f"{trajectory.identity.identity}|retreat")
    )
    extra_horizontal = float(
        rng.uniform(
            config.minimum_extra_horizontal_m,
            config.maximum_extra_horizontal_m,
        )
    )
    xy_offset_deg = float(
        rng.uniform(-config.maximum_xy_offset_deg, config.maximum_xy_offset_deg)
    )
    extra_z_offset = float(
        rng.uniform(config.minimum_z_offset_m, config.maximum_z_offset_m)
    )
    end_horizontal = original_horizontal + extra_horizontal
    end_z_displacement = float(original_delta[2] + extra_z_offset)
    sampled_direction = original_direction + math.radians(xy_offset_deg)
    end_position = anchor_position + np.asarray(
        (
            end_horizontal * math.cos(sampled_direction),
            end_horizontal * math.sin(sampled_direction),
            end_z_displacement,
        ),
        dtype=np.float64,
    )
    return _RetreatEndpointSample(
        original_horizontal_distance_m=original_horizontal,
        extra_horizontal_offset_m=extra_horizontal,
        end_horizontal_distance_m=end_horizontal,
        original_z_displacement_m=float(original_delta[2]),
        extra_z_offset_m=extra_z_offset,
        end_z_displacement_m=end_z_displacement,
        original_direction_deg=math.degrees(original_direction),
        xy_offset_deg=xy_offset_deg,
        end_position_m=tuple(float(value) for value in end_position),
    )


def _yaw_from_quaternion_xyzw(quaternion: NDArray[np.float64]) -> float:
    return float(Rotation.from_quat(np.asarray(quaternion)).as_euler("XYZ")[2])


def _map_endpoint_to_initial_object(
    trajectory: ReferenceTrajectory,
    endpoint: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map endpoint XY from the final object to the initial object pose.

    Horizontal offsets rotate by the object's yaw change. The sampled endpoint
    keeps its absolute world Z: vertically translating a final object relative
    offset to a table-supported initial object can otherwise put the hand below
    the floor after lift/place trajectories.
    """

    initial_object = np.asarray(trajectory.object_pos[0], dtype=np.float64)
    final_object = np.asarray(trajectory.object_pos[-1], dtype=np.float64)
    relative = np.asarray(endpoint, dtype=np.float64) - final_object
    initial_yaw = _yaw_from_quaternion_xyzw(trajectory.object_quat_xyzw[0])
    final_yaw = _yaw_from_quaternion_xyzw(trajectory.object_quat_xyzw[-1])
    yaw_delta = initial_yaw - final_yaw
    cosine = math.cos(yaw_delta)
    sine = math.sin(yaw_delta)
    mapped_xy = np.asarray(
        (
            cosine * relative[0] - sine * relative[1],
            sine * relative[0] + cosine * relative[1],
        ),
        dtype=np.float64,
    )
    return np.asarray(
        (
            initial_object[0] + mapped_xy[0],
            initial_object[1] + mapped_xy[1],
            endpoint[2],
        ),
        dtype=np.float64,
    )


def augment_trajectory_with_approach_prefix(
    trajectory: ReferenceTrajectory,
    *,
    seed: int,
    config: ApproachPrefixConfig = ApproachPrefixConfig(),
    near_anchor_reference_index: int | None = None,
    near_endpoint_config: RetreatSuffixConfig = RetreatSuffixConfig(),
    start_q_ref_3_28: NDArray[np.float64] | tuple[float, ...] | None = None,
) -> tuple[ReferenceTrajectory, ApproachPrefixSample]:
    """Prepend one seeded far or retreat-relative near approach."""

    if not isinstance(trajectory, ReferenceTrajectory):
        raise TypeError("trajectory must be a ReferenceTrajectory")
    if trajectory.reference_fps is None or trajectory.control_fps is None:
        raise ValueError("approach-prefix augmentation requires an explicit public clock")
    if trajectory.reference_fps != trajectory.control_fps:
        raise ValueError("approach-prefix augmentation requires coupled reference/control clocks")
    base_pre = trajectory.movement_start_step
    if base_pre is None or base_pre < 1:
        raise ValueError("approach-prefix augmentation requires an explicit positive movement_start_step")
    if base_pre != config.required_base_pre_padding:
        raise ValueError(
            "approach-prefix augmentation requires base pre-padding "
            f"{config.required_base_pre_padding}, got {base_pre}"
        )
    minimum_prefix = max(1, config.minimum_total_pre_padding - base_pre)
    maximum_prefix = config.maximum_total_pre_padding - base_pre
    if maximum_prefix < minimum_prefix:
        raise ValueError("approach-prefix total pre-padding bounds exclude the source pre-padding")

    rng = np.random.default_rng(_identity_seed(seed, trajectory.identity.identity))
    object_position = np.asarray(trajectory.object_pos[0], dtype=np.float64)
    splice_position = np.asarray(trajectory.q_ref[0, :3], dtype=np.float64)
    if config.mode == "far":
        xy_radius = float(
            rng.uniform(config.minimum_xy_radius_m, config.maximum_xy_radius_m)
        )
        xy_offset_deg = float(
            rng.uniform(-config.maximum_xy_offset_deg, config.maximum_xy_offset_deg)
        )
        z_offset = float(
            rng.uniform(config.minimum_z_offset_m, config.maximum_z_offset_m)
        )
        base_azimuth = math.atan2(
            float(splice_position[1] - object_position[1]),
            float(splice_position[0] - object_position[0]),
        )
        azimuth = base_azimuth + math.radians(xy_offset_deg)
        start_position = object_position + np.asarray(
            (
                xy_radius * math.cos(azimuth),
                xy_radius * math.sin(azimuth),
                z_offset,
            ),
            dtype=np.float64,
        )
        direction_reference = "initial_object_to_source_hand"
    else:
        if near_anchor_reference_index is None:
            raise ValueError(
                "near approach requires a movement-end-anchored reference index"
            )
        endpoint = _sample_retreat_endpoint(
            trajectory,
            seed=seed,
            anchor_reference_index=near_anchor_reference_index,
            config=near_endpoint_config,
        )
        start_position = _map_endpoint_to_initial_object(
            trajectory, np.asarray(endpoint.end_position_m, dtype=np.float64)
        )
        relative_start = start_position - object_position
        xy_radius = float(np.linalg.norm(relative_start[:2]))
        z_offset = float(relative_start[2])
        xy_offset_deg = endpoint.xy_offset_deg
        direction_reference = "final_retreat_distribution_mapped_to_initial_object"
    if not np.all(np.isfinite(start_position)):
        raise RuntimeError("approach sampling produced a non-finite start position")
    if config.mode == "far" and start_position[2] <= object_position[2]:
        raise RuntimeError("far approach sampling produced a non-positive elevation")

    resolved_start_q = np.asarray(
        (
            trajectory.q_ref[0, 3:28]
            if start_q_ref_3_28 is None
            else start_q_ref_3_28
        ),
        dtype=np.float64,
    )
    if resolved_start_q.shape != (25,) or not np.all(np.isfinite(resolved_start_q)):
        raise ValueError("approach start q_ref[3:28] must be 25 finite values")
    template = (
        "pre60_frame0_fallback"
        if start_q_ref_3_28 is None
        else "source_row_frame0"
    )
    start_euler = resolved_start_q[:3].copy()
    pre60_euler = np.asarray(trajectory.q_ref[0, 3:6], dtype=np.float64)
    if np.any(np.abs(start_euler - pre60_euler) > math.pi):
        raise ValueError(
            "source-row frame0 wrist Euler branch differs from pre60 by more than pi"
        )
    start_rotation = Rotation.from_euler("XYZ", start_euler)
    splice_rotation = Rotation.from_euler("XYZ", trajectory.q_ref[0, 3:6])
    rotation_distance_deg = math.degrees(float((start_rotation.inv() * splice_rotation).magnitude()))
    translation_distance = float(np.linalg.norm(start_position - splice_position))
    joint_distance_deg = math.degrees(
        float(
            np.max(
                np.abs(
                    resolved_start_q[3:]
                    - np.asarray(trajectory.q_ref[0, 6:], dtype=np.float64)
                )
            )
        )
    )
    duration_s = max(
        translation_distance / config.nominal_speed_m_s,
        rotation_distance_deg / config.nominal_angular_speed_deg_s,
        joint_distance_deg / config.nominal_angular_speed_deg_s,
    )
    duration_s *= float(
        rng.uniform(
            1.0 - config.duration_jitter_fraction,
            1.0 + config.duration_jitter_fraction,
        )
    )
    prefix_frames = int(np.clip(round(duration_s * trajectory.control_fps), minimum_prefix, maximum_prefix))
    dt = 1.0 / float(trajectory.control_fps)

    prefix_position = _discrete_c1_prefix(
        start_position,
        np.asarray(trajectory.q_ref[:2, :3], dtype=np.float64),
        frames=prefix_frames,
        dt=dt,
        vertical_arc_height=config.vertical_arc_height_m,
    )

    source_euler = np.unwrap(
        np.asarray(trajectory.q_ref[:3, 3:6], dtype=np.float64), axis=0
    )
    prefix_euler = _discrete_c1_prefix(
        start_euler,
        source_euler[:2],
        frames=prefix_frames,
        dt=dt,
    )
    prefix_joints = _discrete_c1_prefix(
        resolved_start_q[3:],
        np.asarray(trajectory.q_ref[:2, 6:], dtype=np.float64),
        frames=prefix_frames,
        dt=dt,
    )

    prefix_by_side: dict[str, np.ndarray] = {}
    for side, source in trajectory.q_ref_by_side.items():
        source_values = np.asarray(source, dtype=np.float64)
        prefix = np.repeat(source_values[:1], prefix_frames, axis=0)
        if side == "right":
            prefix[:, :3] = prefix_position
            prefix[:, 3:6] = prefix_euler
            prefix[:, 6:] = prefix_joints
        prefix_by_side[side] = _readonly(np.concatenate((prefix, source_values), axis=0))
    primary = "right" if "right" in trajectory.selected_hand_sides else trajectory.selected_hand_sides[0]

    prefix_source_indices = np.repeat(trajectory.source_indices[:1], prefix_frames)
    source_indices = _readonly(
        np.concatenate((prefix_source_indices, trajectory.source_indices)), dtype=np.int64
    )
    prefix_timestamps = (
        float(trajectory.timestamps[0])
        - dt * np.arange(prefix_frames, 0, -1, dtype=np.float64)
    )
    timestamps = _readonly(
        np.concatenate((prefix_timestamps, np.asarray(trajectory.timestamps)))
    )
    object_pos_raw = _readonly(
        np.concatenate(
            (np.repeat(trajectory.object_pos_raw[:1], prefix_frames, axis=0), trajectory.object_pos_raw),
            axis=0,
        )
    )
    object_pos = _readonly(
        np.concatenate(
            (np.repeat(trajectory.object_pos[:1], prefix_frames, axis=0), trajectory.object_pos),
            axis=0,
        )
    )
    object_quat = _readonly(
        np.concatenate(
            (
                np.repeat(trajectory.object_quat_xyzw[:1], prefix_frames, axis=0),
                trajectory.object_quat_xyzw,
            ),
            axis=0,
        )
    )
    augmented = replace(
        trajectory,
        source_indices=source_indices,
        timestamps=timestamps,
        q_ref=prefix_by_side[primary],
        q_ref_by_side=prefix_by_side,
        object_pos_raw=object_pos_raw,
        object_pos=object_pos,
        object_quat_xyzw=object_quat,
        movement_start_step=int(trajectory.movement_start_step) + prefix_frames,
        movement_end_step=int(trajectory.movement_end_step) + prefix_frames,
        augmentation_prefix_frames=prefix_frames,
    )

    augmented_position = np.asarray(augmented.q_ref, dtype=np.float64)[:, :3]
    splice_index = prefix_frames
    splice_velocity = (
        augmented_position[splice_index] - augmented_position[splice_index - 1]
    ) / dt
    source_velocity = (
        np.asarray(trajectory.q_ref[1, :3]) - np.asarray(trajectory.q_ref[0, :3])
    ) / dt
    prefix_acceleration = (
        augmented_position[splice_index - 2]
        - 2.0 * augmented_position[splice_index - 1]
        + augmented_position[splice_index]
    ) / (dt * dt)
    source_acceleration = (
        np.asarray(trajectory.q_ref[0, :3])
        - 2.0 * np.asarray(trajectory.q_ref[1, :3])
        + np.asarray(trajectory.q_ref[2, :3])
    ) / (dt * dt)
    sample = ApproachPrefixSample(
        contract=APPROACH_PREFIX_CONTRACT,
        source_identity=trajectory.identity.identity,
        seed=seed,
        approach_mode=config.mode,
        direction_reference=direction_reference,
        base_pre_padding=base_pre,
        prefix_frames=prefix_frames,
        effective_pre_padding=base_pre + prefix_frames,
        start_xy_radius_m=xy_radius,
        start_z_offset_m=z_offset,
        start_distance_m=float(math.hypot(xy_radius, z_offset)),
        xy_offset_deg=xy_offset_deg,
        orientation_template=template,
        start_position_m=tuple(float(value) for value in start_position),
        splice_position_m=tuple(float(value) for value in splice_position),
        splice_position_error_m=float(
            np.linalg.norm(augmented_position[splice_index] - splice_position)
        ),
        splice_velocity_error_m_s=float(np.linalg.norm(splice_velocity - source_velocity)),
        splice_acceleration_jump_m_s2=float(
            np.linalg.norm(prefix_acceleration - source_acceleration)
        ),
    )
    return augmented, sample


def augment_trajectory_with_retreat_suffix(
    trajectory: ReferenceTrajectory,
    *,
    seed: int,
    anchor_reference_index: int,
    config: RetreatSuffixConfig = RetreatSuffixConfig(),
) -> tuple[ReferenceTrajectory, RetreatSuffixSample]:
    """Replace the tail after the parent's movement-end+15 anchor.

    Frames through ``anchor_reference_index`` remain bit-identical. The base XY
    retreat direction is measured from that anchor to the original final wrist
    position. The sampled end travels the original horizontal distance plus an
    extra 3-15 cm, rotated +/-30 degrees around that measured direction, and
    ends 4-10 cm above the original final wrist position. The original tail
    length, timestamps, object reference, wrist orientation, and finger
    sequence remain unchanged; only right-wrist XYZ after the anchor is
    regenerated. Policy/residual gates use ``augmentation_suffix_frames`` to
    make this replaced tail policy-free while smoothly discharging the entry
    residual. Contact observations qualify the accepted parent but do not
    choose the anchor.
    """

    if not isinstance(trajectory, ReferenceTrajectory):
        raise TypeError("trajectory must be a ReferenceTrajectory")
    if trajectory.reference_fps is None or trajectory.control_fps is None:
        raise ValueError("retreat-suffix augmentation requires an explicit public clock")
    if trajectory.reference_fps != trajectory.control_fps:
        raise ValueError(
            "retreat-suffix augmentation requires coupled reference/control clocks"
        )
    if (
        not isinstance(anchor_reference_index, int)
        or isinstance(anchor_reference_index, bool)
        or not 2 <= anchor_reference_index < len(trajectory.q_ref) - 3
    ):
        raise ValueError(
            "retreat anchor must leave two source frames before and three tail frames after"
        )

    anchor = int(anchor_reference_index)
    replaced_tail_frames = len(trajectory.q_ref) - anchor - 1
    # Environment final-zero gating is expressed as ``T - window``. Include
    # the anchor transition itself so policy/residual closes at movement-end+15,
    # while only states after the immutable anchor are geometrically replaced.
    suffix_frames = len(trajectory.q_ref) - anchor
    anchor_position = np.asarray(trajectory.q_ref[anchor, :3], dtype=np.float64)
    original_end = np.asarray(trajectory.q_ref[-1, :3], dtype=np.float64)
    endpoint = _sample_retreat_endpoint(
        trajectory,
        seed=seed,
        anchor_reference_index=anchor,
        config=config,
    )
    end_position = np.asarray(endpoint.end_position_m, dtype=np.float64)

    dt = 1.0 / float(trajectory.control_fps)
    retreat_position = _deform_retreat_tail(
        np.asarray(trajectory.q_ref[anchor + 1 :, :3], dtype=np.float64),
        end_position,
    )

    suffix_by_side: dict[str, np.ndarray] = {}
    for side, source in trajectory.q_ref_by_side.items():
        source_values = np.asarray(source, dtype=np.float64).copy()
        if side == "right":
            source_values[anchor + 1 :, :3] = retreat_position
        suffix_by_side[side] = _readonly(source_values)
    primary = (
        "right"
        if "right" in trajectory.selected_hand_sides
        else trajectory.selected_hand_sides[0]
    )
    augmented = replace(
        trajectory,
        q_ref=suffix_by_side[primary],
        q_ref_by_side=suffix_by_side,
        augmentation_suffix_frames=suffix_frames,
    )

    augmented_position = np.asarray(augmented.q_ref, dtype=np.float64)[:, :3]
    splice_velocity = (
        augmented_position[anchor + 1] - augmented_position[anchor]
    ) / dt
    source_velocity = (
        np.asarray(trajectory.q_ref[anchor + 1, :3])
        - np.asarray(trajectory.q_ref[anchor, :3])
    ) / dt
    suffix_acceleration = (
        augmented_position[anchor + 2]
        - 2.0 * augmented_position[anchor + 1]
        + augmented_position[anchor]
    ) / (dt * dt)
    source_acceleration = (
        np.asarray(trajectory.q_ref[anchor + 2, :3])
        - 2.0 * np.asarray(trajectory.q_ref[anchor + 1, :3])
        + np.asarray(trajectory.q_ref[anchor, :3])
    ) / (dt * dt)
    sample = RetreatSuffixSample(
        contract=RETREAT_SUFFIX_CONTRACT,
        source_identity=trajectory.identity.identity,
        seed=seed,
        anchor_reference_index=anchor,
        suffix_frames=suffix_frames,
        replaced_tail_frames=replaced_tail_frames,
        original_horizontal_distance_m=endpoint.original_horizontal_distance_m,
        extra_horizontal_offset_m=endpoint.extra_horizontal_offset_m,
        end_horizontal_distance_m=endpoint.end_horizontal_distance_m,
        original_z_displacement_m=endpoint.original_z_displacement_m,
        extra_z_offset_m=endpoint.extra_z_offset_m,
        end_z_displacement_m=endpoint.end_z_displacement_m,
        end_distance_m=float(
            math.hypot(
                endpoint.end_horizontal_distance_m,
                endpoint.end_z_displacement_m,
            )
        ),
        original_direction_deg=endpoint.original_direction_deg,
        xy_offset_deg=endpoint.xy_offset_deg,
        end_position_m=tuple(float(value) for value in end_position),
        splice_position_m=tuple(float(value) for value in anchor_position),
        original_end_position_m=tuple(float(value) for value in original_end),
        splice_velocity_error_m_s=float(
            np.linalg.norm(splice_velocity - source_velocity)
        ),
        splice_acceleration_jump_m_s2=float(
            np.linalg.norm(suffix_acceleration - source_acceleration)
        ),
    )
    return augmented, sample
