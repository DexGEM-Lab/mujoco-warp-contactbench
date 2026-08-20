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


# 28-DoF finger order is thumb(6) followed by four four-axis fingers.  This is
# the MANO actionScaling order: CMC abd/flex/twist, MCP flex/abd, IP, then MCP
# abd/flex, PIP, DIP for index through pinky.
SOURCE_ALIGNED_JOINT_SCALE: Final = tuple(
    [0.02, 0.02, 0.008, 0.02, 0.01, 0.005]
    + [0.01, 0.0025, 0.015, 0.005] * 4
)
SOURCE_ALIGNED_JOINT_CAP: Final = tuple(
    [0.2, 0.2, 0.08, 0.2, 0.1, 0.05]
    + [0.1, 0.025, 0.15, 0.1] * 4
)
DEFAULT_EARLY_PHASE_STEPS: Final[int] = 30

# This binds the production observation/control mapping and terminal reset
# separately from reward contracts. Legacy v4 checkpoints remain readable
# because their sidecars record the exact residual-action mapping.
LEGACY_ENVIRONMENT_CONTRACT_IDS: Final = frozenset(
    {
        "mujoco_28dof_hand_side_film_dynamic_residual_early30_pre100_"
        "action2mm_max20mm_observation_contact_0p2n_deviation_0p10_v4",
        "mujoco_28dof_hand_side_film_dynamic_residual_early30_pre100_"
        "action3mm_max30mm_joint2x_observation_contact_0p2n_deviation_0p10_v5",
        "mujoco_28dof_hand_side_film_dynamic_residual_early30_pre100_"
        "action3mm_max30mm_joint2x_thumbtwist0p008_fingermcpflex0p0025_"
        "observation_contact_0p2n_deviation_0p10_v6",
        "mujoco_28dof_hand_side_film_dynamic_residual_early30_pre100_"
        "action3mm_max30mm_joint2x_thumbtwist0p008_fingermcpflex0p0025_"
        "reference_fps_100_or_120_unwrap_linear_slerp_control200_"
        "observation_contact_0p2n_deviation_0p10_v7",
    }
)
ENVIRONMENT_CONTRACT_ID: Final = (
    "mujoco_28dof_hand_side_film_dynamic_residual_early30_pre180_"
    "action3mm_max30mm_joint2x_thumbtwist0p008_fingermcpflex0p0025_"
    "source_policy_fps_100_or_120_physics4x_edgehold_padding_unwrap_linear_slerp_"
    "observation_contact_0p2n_deviation_0p10_v8"
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
    position_scale: tuple[float, float, float] = (0.003, 0.003, 0.003)
    rotation_scale: float = 0.01
    max_position_offset: tuple[float, float, float] = (0.03, 0.03, 0.03)
    joint_scale: tuple[float, ...] = SOURCE_ALIGNED_JOINT_SCALE
    max_joint_offset: tuple[float, ...] = SOURCE_ALIGNED_JOINT_CAP
    joint_scale_multiplier: float = 2.0
    joint_max_offset_multiplier: float = 2.0
    early_phase_steps: int = DEFAULT_EARLY_PHASE_STEPS

    def __post_init__(self) -> None:
        for name, value in (
            ("joint_scale_multiplier", self.joint_scale_multiplier),
            ("joint_max_offset_multiplier", self.joint_max_offset_multiplier),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


SOURCE_ALIGNED_RESIDUAL_ACTION: Final = ResidualActionConfig()


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
    steps: int | NDArray[object] = DEFAULT_EARLY_PHASE_STEPS,
) -> NDArray[np.bool_]:
    """Return per-row half-open pure-reference intervals.

    Production trajectories use the established scalar early-phase length.
    Synthesis-only augmented trajectories may supply one prefix length per row,
    allowing each fresh attempt to open residual and deviation semantics exactly
    when its generated prefix reaches the unchanged source reference.
    """

    step_values = np.asarray(trajectory_steps)
    if step_values.ndim != 1 or not np.issubdtype(step_values.dtype, np.integer):
        raise ValueError("trajectory_steps must be a one-dimensional integer array")
    if starts is None:
        start_values = np.zeros_like(step_values)
    else:
        start_values = _as_vector("starts", starts, len(step_values))
    raw_steps = np.asarray(steps)
    if raw_steps.ndim == 0:
        if not np.issubdtype(raw_steps.dtype, np.integer):
            raise ValueError("steps must be an integer or integer (batch,) array")
        step_lengths = np.full(len(step_values), int(raw_steps), dtype=np.int64)
    else:
        step_lengths = _as_vector("steps", raw_steps, len(step_values))
    if np.any(step_lengths < 0):
        raise ValueError("steps must be non-negative")
    return (step_values >= start_values) & (step_values < start_values + step_lengths)


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
    early_phase_lengths: NDArray[object] | None = None,
    hold_early_exit_zero: NDArray[object] | None = None,
    config: ResidualActionConfig = ResidualActionConfig(),
) -> ResidualActionResult:
    """Apply the residual transformation for a 28-DoF MuJoCo hand.

    The 26-DoF branch remains read-only fixture support for historical source
    traces. Live environment and Gymnasium action boundaries accept only the
    current 28/56-wide MuJoCo layout.
    """

    actions_array = np.asarray(raw_actions, dtype=np.float64)
    if actions_array.ndim != 2 or actions_array.shape[1] not in (26, 28):
        raise ValueError("raw_actions must be a finite (batch, 26) or (batch, 28) array")
    dof_dim = int(actions_array.shape[1])
    actions = _as_batch("raw_actions", actions_array, dof_dim)
    finger_dim = dof_dim - 6
    batch = actions.shape[0]
    steps = _as_vector("trajectory_steps", trajectory_steps, batch)
    position_offset = _as_batch("cumulative_offset", cumulative_offset, 3)
    joint_values = np.asarray(cumulative_joint_offset, dtype=np.float64)
    if joint_values.ndim != 2 or joint_values.shape[0] != batch or joint_values.shape[1] not in (20, 22):
        raise ValueError("cumulative_joint_offset must have shape (batch, 20) or (batch, 22)")
    cumulative_dim = int(joint_values.shape[1])
    if cumulative_dim != finger_dim:
        raise ValueError(
            "cumulative_joint_offset must match the action finger width"
        )
    joint_offset = _as_batch("cumulative_joint_offset", joint_values, cumulative_dim)
    targets = _as_batch("mocap_targets", mocap_targets, dof_dim)
    for name, values in (
        ("cumulative_offset", position_offset),
        ("cumulative_joint_offset", joint_offset),
        ("mocap_targets", targets),
    ):
        if values.shape[0] != batch:
            raise ValueError(f"{name} batch size must match raw_actions")
    lower = np.asarray(joint_lower, dtype=np.float64)
    upper = np.asarray(joint_upper, dtype=np.float64)
    if lower.shape != (dof_dim,) or upper.shape != (dof_dim,) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower > upper):
        raise ValueError(f"joint limits must be finite ordered ({dof_dim},) arrays")
    active = np.asarray(active_joint_mask, dtype=bool)
    if active.shape != (batch, cumulative_dim):
        raise ValueError(f"active_joint_mask must have shape (batch, {cumulative_dim})")
    if use_residual is None:
        residual = np.ones(batch, dtype=bool)
    else:
        residual_values = np.asarray(use_residual, dtype=np.float64)
        if residual_values.shape != (batch,) or not np.all(np.isfinite(residual_values)):
            raise ValueError("use_residual must be a finite (batch,) array")
        residual = residual_values > 0.5

    processed = np.clip(actions, -1.0, 1.0).copy()
    processed[:, 6 : 6 + cumulative_dim] = np.where(
        active, processed[:, 6 : 6 + cumulative_dim], 0.0
    )
    old_joint_offset = np.where(active, joint_offset, 0.0)
    early_lengths = (
        config.early_phase_steps
        if early_phase_lengths is None
        else _as_vector("early_phase_lengths", early_phase_lengths, batch)
    )
    early = early_phase_mask(
        steps, starts=early_phase_starts, steps=early_lengths
    )
    processed[early] = 0.0
    starts = (
        np.zeros(batch, dtype=np.int64)
        if early_phase_starts is None
        else _as_vector("early_phase_starts", early_phase_starts, batch)
    )
    if hold_early_exit_zero is None:
        hold_exit = np.ones(batch, dtype=bool)
    else:
        hold_values = np.asarray(hold_early_exit_zero)
        if hold_values.shape != (batch,):
            raise ValueError("hold_early_exit_zero must have shape (batch,)")
        hold_exit = hold_values.astype(bool, copy=False)
    exit_early = (steps == starts + early_lengths) & hold_exit
    zero_offset = ~residual | early | exit_early
    accumulate = (steps != 0) & ~early & residual

    position_scale = np.asarray(config.position_scale, dtype=np.float64)
    joint_scale = (
        np.asarray(config.joint_scale, dtype=np.float64)
        * config.joint_scale_multiplier
    )
    joint_limit = (
        np.asarray(config.max_joint_offset, dtype=np.float64)
        * config.joint_max_offset_multiplier
    )
    if len(config.joint_scale) == 22 and cumulative_dim == 20:
        # A 28-DoF default config may be used by a legacy 26-DoF replay;
        # drop CMC twist and MCP abduction while retaining the named legacy
        # CMC-abduction, CMC-flexion, MCP-flexion, thumb-IP ordering.
        joint_scale = np.asarray(
            (joint_scale[0], joint_scale[1], joint_scale[3], joint_scale[5])
            + tuple(joint_scale[6:]),
            dtype=np.float64,
        )
        joint_limit = np.asarray(
            (joint_limit[0], joint_limit[1], joint_limit[3], joint_limit[5])
            + tuple(joint_limit[6:]),
            dtype=np.float64,
        )
    effective_joint_dim = cumulative_dim
    if position_scale.shape != (3,) or joint_scale.shape != (effective_joint_dim,) or joint_limit.shape != (effective_joint_dim,):
        raise ValueError("residual action scales and limits must be finite vectors")
    if len(joint_scale) < cumulative_dim or len(joint_limit) < cumulative_dim:
        raise ValueError("residual action scales and limits are shorter than cumulative state")
    if not np.all(np.isfinite(position_scale)) or not np.all(np.isfinite(joint_scale)) or not np.all(np.isfinite(joint_limit)):
        raise ValueError("residual action scales and limits must be finite")
    scaled_position = processed[:, 0:3] * position_scale
    scaled_joints = processed[:, 6 : 6 + cumulative_dim] * joint_scale[:cumulative_dim]
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
    next_joint = np.clip(next_joint, -joint_limit[:cumulative_dim], joint_limit[:cumulative_dim])

    residual_targets = np.zeros_like(targets)
    residual_targets[:, 0:3] = next_position
    residual_targets[:, 3:6] = 0.025 * processed[:, 3:6] * config.rotation_scale
    residual_targets[:, 6 : 6 + cumulative_dim] = next_joint
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
