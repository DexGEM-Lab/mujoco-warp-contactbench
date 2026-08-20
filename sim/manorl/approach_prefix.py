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

APPROACH_PREFIX_CONTRACT: Final = "synthetic_approach_prefix_xy_annulus_z_band_c1_v2"
APPROACH_ORIENTATION_TEMPLATES: Final = ("source_neutral", "object_facing", "top_oblique")


def _readonly(values: object, *, dtype: np.dtype[object] | type = np.float64) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ApproachPrefixConfig:
    """Bounded synthesis-only approach sampling contract."""

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
    maximum_orientation_perturbation_deg: float = 12.0
    vertical_arc_height_m: float = 0.04

    def __post_init__(self) -> None:
        finite = (
            self.minimum_xy_radius_m,
            self.maximum_xy_radius_m,
            self.maximum_xy_offset_deg,
            self.minimum_z_offset_m,
            self.maximum_z_offset_m,
            self.nominal_speed_m_s,
            self.nominal_angular_speed_deg_s,
            self.duration_jitter_fraction,
            self.maximum_orientation_perturbation_deg,
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
        if not 0.0 <= self.maximum_orientation_perturbation_deg <= 45.0:
            raise ValueError("approach-prefix orientation perturbation must be in [0, 45] degrees")
        if not 0.0 <= self.vertical_arc_height_m <= 0.10:
            raise ValueError("approach-prefix vertical arc height must be in [0, 0.10] m")


@dataclass(frozen=True)
class ApproachPrefixSample:
    """One resolved per-trajectory approach-prefix sample."""

    contract: str
    source_identity: str
    seed: int
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
) -> NDArray[np.float64]:
    """Sample before an excluded endpoint with zero start velocity/acceleration."""

    if frames < 1:
        raise ValueError("approach-prefix frames must be positive")
    duration = frames * dt
    c0 = np.asarray(start, dtype=np.float64)
    c1 = np.zeros_like(c0)
    c2 = np.zeros_like(c0)
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
    rhs = np.stack((displacement, end_velocity, end_acceleration), axis=0)
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


def _source_grasp_style(trajectory: ReferenceTrajectory) -> str:
    start = int(trajectory.movement_start_step or 0)
    end = int(trajectory.movement_end_step or start)
    grasp_index = min(max(start + (end - start) // 3, 0), len(trajectory.q_ref) - 1)
    hand_rotation = Rotation.from_euler("XYZ", trajectory.q_ref[grasp_index, 3:6])
    # The local +Z axis is a stable orientation feature even though exact palm
    # normal semantics are asset-specific.  Classification only selects a small
    # perturbation template; every sample still converges to the source pose.
    z_axis = hand_rotation.apply((0.0, 0.0, 1.0))
    if abs(float(z_axis[2])) >= 0.75:
        return "top_oblique"
    if abs(float(z_axis[2])) <= 0.30:
        return "object_facing"
    return "source_neutral"


def _start_orientation_euler(
    trajectory: ReferenceTrajectory,
    *,
    template: str,
    xy_offset_deg: float,
    rng: np.random.Generator,
    maximum_perturbation_deg: float,
) -> NDArray[np.float64]:
    """Sample a bounded source-relative intrinsic-XYZ approach pose.

    The grasp-family template is selected from the source trajectory; offsets
    remain deliberately small so augmentation changes the human approach style
    without replacing the solved grasp convention.
    """

    splice_euler = np.asarray(trajectory.q_ref[0, 3:6], dtype=np.float64)
    if template == "source_neutral":
        centre_deg = np.zeros(3, dtype=np.float64)
    elif template == "object_facing":
        centre_deg = np.asarray((0.0, 0.0, np.clip(0.30 * xy_offset_deg, -8.0, 8.0)))
    elif template == "top_oblique":
        centre_deg = np.asarray((0.0, -8.0, np.clip(0.20 * xy_offset_deg, -6.0, 6.0)))
    else:
        raise ValueError(f"unsupported approach orientation template: {template}")
    jitter_limit = min(6.0, maximum_perturbation_deg)
    offset_deg = centre_deg + rng.uniform(-jitter_limit, jitter_limit, size=3)
    norm = float(np.linalg.norm(offset_deg))
    if norm > maximum_perturbation_deg > 0.0:
        offset_deg *= maximum_perturbation_deg / norm
    return splice_euler + np.deg2rad(offset_deg)


def augment_trajectory_with_approach_prefix(
    trajectory: ReferenceTrajectory,
    *,
    seed: int,
    config: ApproachPrefixConfig = ApproachPrefixConfig(),
) -> tuple[ReferenceTrajectory, ApproachPrefixSample]:
    """Prepend one seeded, positive-Z, human-like approach to ``trajectory``."""

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
    xy_radius = float(
        rng.uniform(config.minimum_xy_radius_m, config.maximum_xy_radius_m)
    )
    xy_offset_deg = float(
        rng.uniform(-config.maximum_xy_offset_deg, config.maximum_xy_offset_deg)
    )
    z_offset = float(
        rng.uniform(config.minimum_z_offset_m, config.maximum_z_offset_m)
    )
    object_position = np.asarray(trajectory.object_pos[0], dtype=np.float64)
    splice_position = np.asarray(trajectory.q_ref[0, :3], dtype=np.float64)
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
    if start_position[2] <= object_position[2]:
        raise RuntimeError("positive-Z approach sampling produced a non-positive elevation")

    template = _source_grasp_style(trajectory)
    start_euler = _start_orientation_euler(
        trajectory,
        template=template,
        xy_offset_deg=xy_offset_deg,
        rng=rng,
        maximum_perturbation_deg=config.maximum_orientation_perturbation_deg,
    )
    start_rotation = Rotation.from_euler("XYZ", start_euler)
    splice_rotation = Rotation.from_euler("XYZ", trajectory.q_ref[0, 3:6])
    rotation_distance_deg = math.degrees(float((start_rotation.inv() * splice_rotation).magnitude()))
    translation_distance = float(np.linalg.norm(start_position - splice_position))
    duration_s = max(
        translation_distance / config.nominal_speed_m_s,
        rotation_distance_deg / config.nominal_angular_speed_deg_s,
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

    prefix_by_side: dict[str, np.ndarray] = {}
    for side, source in trajectory.q_ref_by_side.items():
        source_values = np.asarray(source, dtype=np.float64)
        prefix = np.repeat(source_values[:1], prefix_frames, axis=0)
        if side == "right":
            prefix[:, :3] = prefix_position
            prefix[:, 3:6] = prefix_euler
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
