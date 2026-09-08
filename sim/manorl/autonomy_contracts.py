"""Explicit contracts for the autonomous cube2 control path.

This module is deliberately separate from the residual ManoRL ABI.  The
residual environment and its checkpoints keep their historical semantics;
autonomous checkpoints carry these identifiers and fail closed on mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

AUTONOMY_VERSION: Final = "manorl.autonomy.v1"
OBSERVATION_CONTRACT_ID: Final = "manorl.autonomy.observation.v1"
ACTION_CONTRACT_ID: Final = "manorl.autonomy.action.v1"
REWARD_CONTRACT_ID: Final = "manorl.autonomy.reward.v1"
CHECKPOINT_FORMAT: Final = "manorl.autonomy.ppo.v1"

# Fields are serialized in this order.  Keeping the names next to the slices
# makes accidental reference/action coupling visible during review.
OBSERVATION_FIELDS: Final[tuple[tuple[str, int], ...]] = (
    ("measured_qpos_normalized", 28),
    ("measured_qvel", 28),
    ("object_position", 3),
    ("object_linear_velocity", 3),
    ("hand_object_relative", 3),
    ("reference_q_current", 28),
    ("reference_q_next", 28),
    ("reference_q_velocity", 28),
    ("reference_object_relative", 3),
    ("reference_object_future_delta", 3),
    ("previous_command_normalized", 28),
    ("action_identity_one_hot", 50),
    ("object_geometry", 12),
    ("measured_keypoint_relative", 48),
    ("surface_proximity", 16),
    ("surface_anchor_local", 48),
    ("contact_phase_confidence", 2),
)
OBSERVATION_DIM: Final[int] = sum(width for _, width in OBSERVATION_FIELDS)
ACTION_DIM: Final[int] = 28


@dataclass(frozen=True)
class AutonomousActionContract:
    """Reference-independent action semantics."""

    version: str = ACTION_CONTRACT_ID
    dof: int = ACTION_DIM
    normalized_range: tuple[float, float] = (-1.0, 1.0)
    command_mode: str = "rate_limited_measured_state"
    all_dofs_policy_owned_from_step0: bool = True
    reference_enters_command_map: bool = False

    def __post_init__(self) -> None:
        if self.version != ACTION_CONTRACT_ID or self.dof != ACTION_DIM:
            raise ValueError("unsupported autonomous action contract")
        if self.command_mode != "rate_limited_measured_state":
            raise ValueError("autonomous actions must use the measured-state rate map")
        if not self.all_dofs_policy_owned_from_step0:
            raise ValueError("all 28 DOFs must be policy-owned from step zero")
        if self.reference_enters_command_map:
            raise ValueError("reference data cannot enter the autonomous command map")


@dataclass(frozen=True)
class AutonomousObservationContract:
    version: str = OBSERVATION_CONTRACT_ID
    dimension: int = OBSERVATION_DIM
    fields: tuple[tuple[str, int], ...] = OBSERVATION_FIELDS
    includes_surface_intent: bool = True
    includes_actual_velocity: bool = True
    includes_previous_command: bool = True
    includes_action_identity: bool = True

    def __post_init__(self) -> None:
        if self.version != OBSERVATION_CONTRACT_ID:
            raise ValueError("unsupported autonomous observation contract")
        if self.dimension != sum(width for _, width in self.fields):
            raise ValueError("autonomous observation dimension does not match fields")
        if self.dimension != OBSERVATION_DIM:
            raise ValueError("autonomous observation dimension drifted")
        if not (self.includes_surface_intent and self.includes_actual_velocity):
            raise ValueError("surface intent and actual velocity are mandatory")
        if not (self.includes_previous_command and self.includes_action_identity):
            raise ValueError("command history and action identity are mandatory")


@dataclass(frozen=True)
class AutonomousRewardContract:
    version: str = REWARD_CONTRACT_ID
    dense_terms: tuple[str, ...] = (
        "object_motion",
        "reference_hand_object_relationship",
        "surface_proximity_contact",
        "stability",
        "release",
    )
    pre_grasp_dense: bool = True
    object_move_gate: bool = False

    def __post_init__(self) -> None:
        if self.version != REWARD_CONTRACT_ID or not self.pre_grasp_dense:
            raise ValueError("autonomous reward contract requires dense pre-grasp terms")
        if self.object_move_gate:
            raise ValueError("autonomous translation reward cannot be object_move gated")


ACTION_CONTRACT = AutonomousActionContract()
OBSERVATION_CONTRACT = AutonomousObservationContract()
REWARD_CONTRACT = AutonomousRewardContract()


def rate_limited_command(
    previous_command: NDArray[np.floating],
    action: NDArray[np.floating],
    lower: NDArray[np.floating],
    upper: NDArray[np.floating],
    rate: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Map normalized policy action to a bounded servo target.

    The only state entering this map is the measured previous command and
    physical actuator limits.  Reference tensors are intentionally absent from
    the signature, making the no-hidden-reference invariant mechanically
    testable.
    """

    previous = np.asarray(previous_command, dtype=np.float64)
    normalized = np.asarray(action, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    delta_limit = np.asarray(rate, dtype=np.float64)
    if any(value.shape != (ACTION_DIM,) for value in (previous, normalized, lo, hi, delta_limit)):
        raise ValueError("autonomous command vectors must all have shape (28,)")
    if not all(np.all(np.isfinite(value)) for value in (previous, normalized, lo, hi, delta_limit)):
        raise ValueError("autonomous command vectors must be finite")
    if np.any(hi <= lo) or np.any(delta_limit <= 0.0):
        raise ValueError("physical command limits/rates must be ordered and positive")
    if np.any(normalized < -1.0) or np.any(normalized > 1.0):
        raise ValueError("policy action must be normalized to [-1, 1]")
    return np.clip(previous + normalized * delta_limit, lo, hi)


def observation_slices() -> dict[str, slice]:
    start = 0
    result: dict[str, slice] = {}
    for name, width in OBSERVATION_FIELDS:
        result[name] = slice(start, start + width)
        start += width
    return result
