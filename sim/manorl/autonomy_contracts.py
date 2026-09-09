"""Versioned contracts for the physical autonomous cube2 diagnostic path."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Final
import numpy as np
from numpy.typing import NDArray

AUTONOMY_VERSION: Final = "manorl.autonomy.v2"
OBSERVATION_CONTRACT_ID: Final = "manorl.autonomy.observation.v2"
ACTION_CONTRACT_ID: Final = "manorl.autonomy.action.v2"
REWARD_CONTRACT_ID: Final = "manorl.autonomy.reward.v2"
CHECKPOINT_FORMAT: Final = "manorl.autonomy.ppo.v2"
# v3 is a separate ABI: fixed offline collision witnesses are part of the
# observation/reward contract and cannot be loaded with a v2 normalizer.
# v3.1 corrects the phase coordinate for unique (T, 28) reference tables.
# Width and every non-phase observation field are unchanged, but checkpoints
# trained with v3's saturated phase must not share this normalizer ABI.
AUTONOMY_V3_VERSION: Final = "manorl.autonomy.v3.1"
OBSERVATION_V3_CONTRACT_ID: Final = "manorl.autonomy.observation.v3.1"
ACTION_V3_CONTRACT_ID: Final = ACTION_CONTRACT_ID  # bit-equivalent measured-state map
REWARD_V3_CONTRACT_ID: Final = "manorl.autonomy.reward.v3"
CHECKPOINT_V3_FORMAT: Final = "manorl.autonomy.ppo.v3.1"
LEGACY_PHASE_BUG_OBSERVATION_CONTRACT_ID: Final = "manorl.autonomy.observation.v3"
LEGACY_PHASE_BUG_CHECKPOINT_FORMAT: Final = "manorl.autonomy.ppo.v3"
WITNESS_TABLE_CONTRACT_ID: Final = "manorl.autonomy.reference_witness.v1"
OBSERVATION_FIELDS: Final[tuple[tuple[str, int], ...]] = (
    ("measured_qpos_normalized", 28), ("measured_qvel", 28),
    ("object_position", 3), ("object_linear_velocity", 3),
    ("hand_object_relative", 3), ("reference_q_current", 28),
    ("reference_q_next", 28), ("reference_q_velocity", 28),
    ("reference_object_relative", 3), ("reference_object_future_delta", 3),
    ("previous_command_normalized", 28), ("action_identity_one_hot", 50),
    ("object_geometry", 12), ("measured_keypoint_relative", 48),
    ("surface_proximity", 16), ("surface_anchor_local", 48),
    ("contact_phase_confidence", 2), ("reference_surface_proximity", 16),
    ("reference_surface_anchor_local", 48), ("reference_contact_confidence", 16),
    ("measured_hand_object_force", 48), ("supporting_object_net_force", 3),
    ("relative_contact_motion", 48),
)
OBSERVATION_DIM: Final[int] = sum(width for _, width in OBSERVATION_FIELDS)
ACTION_DIM: Final[int] = 28

@dataclass(frozen=True)
class AutonomousActionContract:
    version: str = ACTION_CONTRACT_ID
    dof: int = ACTION_DIM
    normalized_range: tuple[float, float] = (-1.0, 1.0)
    command_mode: str = "rate_limited_measured_state"
    rate_units: str = "actuator_units_per_second"
    all_dofs_policy_owned_from_step0: bool = True
    reference_enters_command_map: bool = False
    def __post_init__(self) -> None:
        if self.version != ACTION_CONTRACT_ID or self.dof != ACTION_DIM:
            raise ValueError("unsupported autonomous action contract")
        if self.command_mode != "rate_limited_measured_state" or self.rate_units != "actuator_units_per_second":
            raise ValueError("autonomous actions require measured-state per-second rate semantics")
        if not self.all_dofs_policy_owned_from_step0 or self.reference_enters_command_map:
            raise ValueError("all DOFs must be policy-owned and references excluded from command map")

@dataclass(frozen=True)
class AutonomousObservationContract:
    version: str = OBSERVATION_CONTRACT_ID
    dimension: int = OBSERVATION_DIM
    fields: tuple[tuple[str, int], ...] = OBSERVATION_FIELDS
    includes_surface_intent: bool = True
    includes_reference_surface_intent: bool = True
    includes_actual_velocity: bool = True
    includes_previous_command: bool = True
    includes_action_identity: bool = True
    includes_contact_wrench: bool = True
    def __post_init__(self) -> None:
        if self.version != OBSERVATION_CONTRACT_ID or self.dimension != OBSERVATION_DIM or self.dimension != sum(w for _, w in self.fields):
            raise ValueError("autonomous observation contract drifted")
        if not all((self.includes_surface_intent, self.includes_reference_surface_intent, self.includes_actual_velocity, self.includes_previous_command, self.includes_action_identity, self.includes_contact_wrench)):
            raise ValueError("surface intent, references, velocity, command, identity and force fields are mandatory")

@dataclass(frozen=True)
class AutonomousRewardContract:
    version: str = REWARD_CONTRACT_ID
    dense_terms: tuple[str, ...] = ("object_motion", "reference_hand_object_relationship", "contact_anchor_correspondence", "measured_contact", "slip_proxy", "release", "action_smoothness", "finger_configuration", "object_orientation")
    pre_grasp_dense: bool = True
    object_move_gate: bool = False
    stability_is_slip_relative: bool = True
    def __post_init__(self) -> None:
        if self.version != REWARD_CONTRACT_ID or not self.pre_grasp_dense or self.object_move_gate or not self.stability_is_slip_relative:
            raise ValueError("autonomous reward must be additive, pre-grasp dense and slip-relative")

ACTION_CONTRACT = AutonomousActionContract()
OBSERVATION_CONTRACT = AutonomousObservationContract()
REWARD_CONTRACT = AutonomousRewardContract()


def validate_v3_checkpoint_metadata(metadata: dict[str, object]) -> None:
    """Accept only the phase-corrected v3.1 normalizer/checkpoint ABI.

    The v3 observation phase used the 28-DOF width as its denominator for
    unique reference tables. Loading its normalizer would silently preserve the
    saturated coordinate, so it requires an explicit legacy reader instead.
    """
    if metadata.get("checkpoint_format") == CHECKPOINT_FORMAT or metadata.get("observation_contract") == OBSERVATION_CONTRACT_ID:
        raise ValueError("v2 autonomy checkpoint/normalizer requires explicit migration before v3.1 loading")
    if (metadata.get("checkpoint_format") == LEGACY_PHASE_BUG_CHECKPOINT_FORMAT
            or metadata.get("observation_contract") == LEGACY_PHASE_BUG_OBSERVATION_CONTRACT_ID):
        raise ValueError("v3 phase-bug checkpoint/normalizer requires an explicit legacy reader")
    required = {"checkpoint_format": CHECKPOINT_V3_FORMAT, "observation_contract": OBSERVATION_V3_CONTRACT_ID, "reward_contract": REWARD_V3_CONTRACT_ID}
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(f"v3.1 checkpoint metadata {name} must be {expected!r}")

def rate_limited_command(previous_command: NDArray[np.floating], action: NDArray[np.floating], lower: NDArray[np.floating], upper: NDArray[np.floating], rate_per_second: NDArray[np.floating], *, measured_qpos: NDArray[np.floating] | None = None, control_timestep: float = 1.0 / 120.0, max_tracking_error: NDArray[np.floating] | None = None) -> NDArray[np.float64]:
    """Reference-independent rate map in actuator-units/second.

    ``max_tracking_error`` is a physical servo envelope around measured qpos,
    not a reference-relative clamp; load-induced tracking error therefore stays
    visible to the controller while integrated target wind-up is bounded.
    """
    previous, normalized, lo, hi, rate = map(lambda x: np.asarray(x, dtype=np.float64), (previous_command, action, lower, upper, rate_per_second))
    if any(x.shape != (ACTION_DIM,) for x in (previous, normalized, lo, hi, rate)):
        raise ValueError("autonomous command vectors must all have shape (28,)")
    if not all(np.all(np.isfinite(x)) for x in (previous, normalized, lo, hi, rate)) or control_timestep <= 0 or not np.isfinite(control_timestep):
        raise ValueError("autonomous command vectors and timestep must be finite")
    if np.any(hi <= lo) or np.any(rate <= 0) or np.any(normalized < -1) or np.any(normalized > 1):
        raise ValueError("physical limits/rates must be ordered and action normalized")
    command = previous + normalized * rate * float(control_timestep)
    if measured_qpos is not None:
        measured = np.asarray(measured_qpos, dtype=np.float64)
        bound = np.asarray(max_tracking_error if max_tracking_error is not None else np.maximum(.25 * (hi - lo), rate * control_timestep * 4.0), dtype=np.float64)
        if measured.shape != (ACTION_DIM,) or bound.shape != (ACTION_DIM,) or not np.all(np.isfinite(measured)) or np.any(bound <= 0):
            raise ValueError("measured state and tracking envelope must be finite (28,)")
        command = np.clip(command, measured - bound, measured + bound)
    return np.clip(command, lo, hi)

def observation_slices() -> dict[str, slice]:
    start, result = 0, {}
    for name, width in OBSERVATION_FIELDS:
        result[name] = slice(start, start + width); start += width
    return result
