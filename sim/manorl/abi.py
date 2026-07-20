"""Deterministic source-compatible training control and termination core.

This module deliberately contains no simulator calls. It is the narrow part of
the ManoRL environment ABI whose behavior is fully specified by source action
and termination code, so future MuJoCo batching cannot silently change it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray


SOURCE_ALIGNED_JOINT_SCALE: Final = tuple(
    [0.10, 0.12, 0.044, 0.01] + [0.024, 0.04, 0.06, 0.01] * 4
)
SOURCE_ALIGNED_JOINT_CAP: Final = tuple(
    [1.0, 1.2, 0.44, 0.1] + [0.24, 0.4, 0.6, 0.1] * 4
)

# This binds the source-aligned production observation/control mapping and
# terminal reset separately from reward contracts.
ENVIRONMENT_CONTRACT_ID: Final = (
    "source_aligned_film_dynamic_residual_gym_authority_early50_pre250_"
    "observation_contact_0p2n_deviation_0p10_v2"
)
TARGET_MAX_DEVIATION_DISTANCE: Final[float] = 0.10

# Termination reason values are part of the host-side ABI.  Keep ``0`` for an
# in-progress transition so callers can safely persist the code for every
# sample, including non-terminal samples.
TERMINATION_REASON_NONE: Final[int] = 0
TERMINATION_REASON_SUCCESS: Final[int] = 1
TERMINATION_REASON_DEVIATION: Final[int] = 2
TERMINATION_REASON_FAILURE: Final[int] = TERMINATION_REASON_DEVIATION

@dataclass(frozen=True)
class ResidualActionConfig:
    """Source-aligned production residual mapping over the normalized action Box."""

    gamma_xy: float = 0.9
    gamma_z: float = 0.9
    gamma_joints: float = 0.9
    position_scale: tuple[float, float, float] = (0.005, 0.005, 0.005)
    rotation_scale: float = 0.01
    max_position_offset: tuple[float, float, float] = (0.05, 0.05, 0.05)
    joint_scale: tuple[float, ...] = SOURCE_ALIGNED_JOINT_SCALE
    max_joint_offset: tuple[float, ...] = SOURCE_ALIGNED_JOINT_CAP
    early_phase_steps: int = 50


SOURCE_ALIGNED_RESIDUAL_ACTION: Final = ResidualActionConfig()
CHECKPOINT_SIDECAR_RESIDUAL_ACTION: Final = SOURCE_ALIGNED_RESIDUAL_ACTION


@dataclass(frozen=True)
class ResidualActionResult:
    """Post-mask actions, residual state, and source-compatible DOF targets."""

    actions: NDArray[np.float64]
    cumulative_offset: NDArray[np.float64]
    cumulative_joint_offset: NDArray[np.float64]
    targets: NDArray[np.float64]


@dataclass(frozen=True)
class TerminationResult:
    """Source task reset conditions, reason masks, and penalty.

    Reason data is exposed through derived properties so the original
    three-field dataclass and its ``dataclasses.replace`` behavior stay intact.
    """

    reset: NDArray[np.bool_]
    deviation_reset: NDArray[np.bool_]
    deviation_penalty: NDArray[np.float64]

    @property
    def success(self) -> NDArray[np.bool_]:
        """Trajectory completion without a simultaneous deviation failure."""

        return np.asarray(self.reset, dtype=bool) & ~np.asarray(self.deviation_reset, dtype=bool)

    @property
    def failure(self) -> NDArray[np.bool_]:
        """Position-deviation failure mask."""

        return np.asarray(self.reset, dtype=bool) & np.asarray(self.deviation_reset, dtype=bool)

    @property
    def reason_code(self) -> NDArray[np.int32]:
        """Return 0=ongoing, 1=success, 2=deviation failure."""

        return np.where(
            self.failure,
            TERMINATION_REASON_FAILURE,
            np.where(self.success, TERMINATION_REASON_SUCCESS, TERMINATION_REASON_NONE),
        ).astype(np.int32)

    @property
    def success_mask(self) -> NDArray[np.bool_]:
        """Alias used by telemetry and vectorized consumers."""

        return self.success

    @property
    def failure_mask(self) -> NDArray[np.bool_]:
        """Alias used by telemetry and vectorized consumers."""

        return self.failure

    @property
    def termination_reason_code(self) -> NDArray[np.int32]:
        """Stable name matching the source trace artifact field."""

        return self.reason_code

    @property
    def trajectory_complete_reset_mask(self) -> NDArray[np.bool_]:
        """Source-compatible success mask name."""

        return self.success

    @property
    def deviation_reset_mask(self) -> NDArray[np.bool_]:
        """Source-compatible deviation mask name."""

        return np.asarray(self.deviation_reset, dtype=bool)


def _as_batch(name: str, values: NDArray[object], width: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite (batch, {width}) array")
    return array


def _as_vector(name: str, values: NDArray[object], batch: int) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.shape != (batch,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be an integer ({batch},) array")
    return array.astype(np.int64, copy=False)


def early_phase_mask(
    trajectory_steps: NDArray[object],
    *,
    starts: NDArray[object] | None = None,
    steps: int = 50,
) -> NDArray[np.bool_]:
    """Return the source half-open early pure-mocap interval."""

    step_values = np.asarray(trajectory_steps)
    if step_values.ndim != 1 or not np.issubdtype(step_values.dtype, np.integer):
        raise ValueError("trajectory_steps must be a one-dimensional integer array")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if starts is None:
        start_values = np.zeros_like(step_values)
    else:
        start_values = _as_vector("starts", starts, len(step_values))
    return (step_values >= start_values) & (step_values < start_values + steps)


def process_residual_actions(
    raw_actions: NDArray[object],
    *,
    trajectory_steps: NDArray[object],
    cumulative_offset: NDArray[object],
    cumulative_joint_offset: NDArray[object],
    mocap_targets: NDArray[object],
    joint_lower: NDArray[object],
    joint_upper: NDArray[object],
    active_joint_mask: NDArray[object],
    use_residual: NDArray[object] | None = None,
    early_phase_starts: NDArray[object] | None = None,
    config: ResidualActionConfig = ResidualActionConfig(),
) -> ResidualActionResult:
    """Apply the source 26D action transformation without mutating inputs."""

    actions = _as_batch("raw_actions", raw_actions, 26)
    batch = actions.shape[0]
    steps = _as_vector("trajectory_steps", trajectory_steps, batch)
    position_offset = _as_batch("cumulative_offset", cumulative_offset, 3)
    joint_offset = _as_batch("cumulative_joint_offset", cumulative_joint_offset, 20)
    targets = _as_batch("mocap_targets", mocap_targets, 26)
    for name, values in (
        ("cumulative_offset", position_offset),
        ("cumulative_joint_offset", joint_offset),
        ("mocap_targets", targets),
    ):
        if values.shape[0] != batch:
            raise ValueError(f"{name} batch size must match raw_actions")
    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    if lower.shape != (26,) or upper.shape != (26,) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower > upper):
        raise ValueError("joint limits must be finite ordered (26,) arrays")
    active = np.asarray(active_joint_mask, dtype=bool)
    if active.shape != (batch, 20):
        raise ValueError("active_joint_mask must have shape (batch, 20)")
    if use_residual is None:
        residual = np.ones(batch, dtype=bool)
    else:
        residual_values = np.asarray(use_residual, dtype=np.float64)
        if residual_values.shape != (batch,) or not np.all(np.isfinite(residual_values)):
            raise ValueError("use_residual must be a finite (batch,) array")
        residual = residual_values > 0.5

    processed = np.clip(actions, -1.0, 1.0).copy()
    processed[:, 6:26] = np.where(active, processed[:, 6:26], 0.0)
    old_joint_offset = np.where(active, joint_offset, 0.0)
    early = early_phase_mask(
        steps, starts=early_phase_starts, steps=config.early_phase_steps
    )
    processed[early] = 0.0
    starts = (
        np.zeros(batch, dtype=np.int64)
        if early_phase_starts is None
        else _as_vector("early_phase_starts", early_phase_starts, batch)
    )
    exit_early = steps == starts + config.early_phase_steps
    zero_offset = ~residual | early | exit_early
    accumulate = (steps != 0) & ~early & residual

    position_scale = np.asarray(config.position_scale, dtype=np.float64)
    joint_scale = np.asarray(config.joint_scale, dtype=np.float64)
    joint_limit = np.asarray(config.max_joint_offset, dtype=np.float64)
    if position_scale.shape != (3,) or joint_scale.shape != (20,) or joint_limit.shape != (20,):
        raise ValueError("residual action scales and limits must have 3D position and 20D joint shapes")
    if not np.all(np.isfinite(position_scale)) or not np.all(np.isfinite(joint_scale)) or not np.all(np.isfinite(joint_limit)):
        raise ValueError("residual action scales and limits must be finite")
    scaled_position = processed[:, 0:3] * position_scale
    scaled_joints = processed[:, 6:26] * joint_scale
    next_position = position_offset.copy()
    next_joint = old_joint_offset.copy()
    next_position[zero_offset] = 0.0
    next_joint[zero_offset] = 0.0
    next_position[accumulate, :2] = (
        config.gamma_xy * next_position[accumulate, :2]
        + scaled_position[accumulate, :2]
    )
    next_position[accumulate, 2] = (
        config.gamma_z * next_position[accumulate, 2]
        + scaled_position[accumulate, 2]
    )
    next_joint[accumulate] = (
        config.gamma_joints * next_joint[accumulate] + scaled_joints[accumulate]
    )
    position_limit = np.asarray(config.max_position_offset, dtype=np.float64)
    next_position = np.clip(next_position, -position_limit, position_limit)
    next_joint = np.clip(next_joint, -joint_limit, joint_limit)

    residual_targets = np.zeros_like(targets)
    residual_targets[:, 0:3] = next_position
    residual_targets[:, 3:6] = 0.025 * processed[:, 3:6] * config.rotation_scale
    residual_targets[:, 6:26] = next_joint
    final_targets = np.clip(
        targets + residual[:, None] * residual_targets, lower, upper
    )
    return ResidualActionResult(processed, next_position, next_joint, final_targets)


def check_termination(
    *,
    object_position: NDArray[object],
    target_position: NDArray[object],
    progress: NDArray[object],
    trajectory_lengths: NDArray[object],
    early_mask: NDArray[object],
    max_deviation_distance: float = TARGET_MAX_DEVIATION_DISTANCE,
    deviation_penalty: float = 0.0,
) -> TerminationResult:
    """Evaluate exactly the task's trajectory-complete/deviation predicate."""

    object_values = _as_batch("object_position", object_position, 3)
    target_values = _as_batch("target_position", target_position, 3)
    batch = object_values.shape[0]
    if target_values.shape[0] != batch:
        raise ValueError("target_position batch size must match object_position")
    progress_values = _as_vector("progress", progress, batch)
    length_values = _as_vector("trajectory_lengths", trajectory_lengths, batch)
    early_values = np.asarray(early_mask, dtype=bool)
    if early_values.shape != (batch,):
        raise ValueError("early_mask must have shape (batch,)")
    if max_deviation_distance < 0 or deviation_penalty < 0:
        raise ValueError("termination distances and penalty must be non-negative")
    complete = progress_values >= length_values - 1
    deviation = (
        np.linalg.norm(object_values - target_values, axis=1)
        > max_deviation_distance
    )
    deviation &= ~early_values
    return TerminationResult(
        reset=complete | deviation,
        deviation_reset=deviation,
        deviation_penalty=np.where(deviation, -deviation_penalty, 0.0),
    )
