"""Authoritative contracts for contact-conditioned autonomy v4.

The v4 ABI intentionally rejects every v2/v3 checkpoint: those observations
reused names for different physical quantities and cannot be normalized safely.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Final
import numpy as np
from numpy.typing import NDArray

AUTONOMY_VERSION: Final = "manorl.autonomy.v4"
OBSERVATION_CONTRACT_ID: Final = "manorl.autonomy.observation.v4"
ACTION_CONTRACT_ID: Final = "manorl.autonomy.action.v4"
REWARD_CONTRACT_ID: Final = "manorl.autonomy.reward.v4"
V10_REWARD_CONTRACT_ID: Final = (
    "manorl.autonomy.reward.v10.gated_tracking_airborne_hold"
)
REWARD_CONTRACT_BY_VERSION: Final[dict[str, str]] = {
    "v4": REWARD_CONTRACT_ID,
    "v10": V10_REWARD_CONTRACT_ID,
}
SUPPORTED_REWARD_CONTRACT_IDS: Final = frozenset(
    REWARD_CONTRACT_BY_VERSION.values()
)
V4_REWARD_TERM_NAMES: Final = (
    "object_position", "object_rotation", "object_velocity", "hand_relative",
    "fingers", "geometry", "action", "survival", "severe",
)
V10_REWARD_TERM_NAMES: Final = (
    "object_position", "object_rotation", "object_velocity", "hand_relative",
    "fingers", "geometry", "contact", "thumb_contact", "opposing_contact",
    "lift_progress", "lift_velocity", "hold_contact", "contact_loss",
    "falling", "lateral_slip", "action", "survival", "severe",
)
CHECKPOINT_FORMAT: Final = "manorl.autonomy.ppo.v4"
CACHE_CONTRACT_ID: Final = "manorl.autonomy.reference_cache.v4"
# The old name remains an import-only alias; it names the v4 cache and never
# makes the old native witness compiler valid.
WITNESS_TABLE_CONTRACT_ID: Final = CACHE_CONTRACT_ID
POLICY_SAMPLING_CONTRACT_ID: Final = "manorl.autonomy.policy_sampling.raw_normal.v1"
POLICY_SAMPLING_CONTRACT: Final[dict[str, str]] = {
    "id": POLICY_SAMPLING_CONTRACT_ID,
    "ppo_taken_actions": "raw_normal_sample",
    "physical_action": "clip(raw_normal_sample, -1, 1)",
    "evaluation_action": "clip(policy_mean, -1, 1)",
}
ACTION_DIM: Final = 28
RAW_OBSERVATION_DIM: Final = 957
ENCODED_OBSERVATION_DIM: Final = 829
# v5 point-cloud observation ABI: hand and object surface clouds carry the
# geometry correspondence; only intent gating and load transfer stay numeric.
OBSERVATION_CONTRACT_ID_V5: Final = "manorl.autonomy.observation.v5.pointcloud"
CHECKPOINT_FORMAT_V5: Final = "manorl.autonomy.ppo.v5"
RAW_OBSERVATION_DIM_V5: Final = 1342
ENCODED_OBSERVATION_DIM_V5: Final = 510
# v5.25 preserves the v5 flat ABI while changing only the current-hand
# encoder from one global PointNet to 16 identity-preserving region tokens.
OBSERVATION_CONTRACT_ID_V525: Final = (
    "manorl.autonomy.observation.v5.25.region-hand"
)
CHECKPOINT_FORMAT_V525: Final = "manorl.autonomy.ppo.v5.25"
RAW_OBSERVATION_DIM_V525: Final = RAW_OBSERVATION_DIM_V5
ENCODED_OBSERVATION_DIM_V525: Final = ENCODED_OBSERVATION_DIM_V5
# v5.5 adds one object-frame pose goal per hand region and one hand-only
# cross-attention layer while preserving the global object PointNet and MLP.
OBSERVATION_CONTRACT_ID_V55: Final = (
    "manorl.autonomy.observation.v5.5.region-hand-goal"
)
CHECKPOINT_FORMAT_V55: Final = "manorl.autonomy.ppo.v5.5"
RAW_OBSERVATION_DIM_V55: Final = 1486
ENCODED_OBSERVATION_DIM_V55: Final = ENCODED_OBSERVATION_DIM_V5
# v5.75 adds region-bound dynamic contact/slip and an object wrench without
# adopting the complete v6 object-patch / dual-fusion-tower architecture.
OBSERVATION_CONTRACT_ID_V575: Final = (
    "manorl.autonomy.observation.v5.75.contact-wrench"
)
CHECKPOINT_FORMAT_V575: Final = "manorl.autonomy.ppo.v5.75"
RAW_OBSERVATION_DIM_V575: Final = 1581
ENCODED_OBSERVATION_DIM_V575: Final = 525
# v6 keeps the v4 MDP/PPO/action semantics while adopting the useful parts
# of VoxMani v0.5's observation and token-fusion design.
OBSERVATION_CONTRACT_ID_V6: Final = "manorl.autonomy.observation.v6.region-token-cross-attention"
CHECKPOINT_FORMAT_V6: Final = "manorl.autonomy.ppo.v6"
RAW_OBSERVATION_DIM_V6: Final = 2205
MODEL_DIM_V6: Final = 128
CURRENT_TOKENS_V6: Final = 34
GOAL_TOKENS_V6: Final = 16
# Environments and PPO memory own raw 957. The registered Torch PointNet in
# the actor converts only the cloud to the 829 model feature internally.
OBSERVATION_DIM: Final = RAW_OBSERVATION_DIM
RAW_OBSERVATION_FIELDS: Final[tuple[tuple[str, int], ...]] = (
    ("autonomous_actual", 119), ("autonomous_reference", 109),
    ("autonomous_future", 123), ("autonomous_geometry", 352),
    ("object_point_cloud_raw", 192), ("action_types", 50),
    ("object_geometry", 12),
)
ENCODED_OBSERVATION_FIELDS: Final[tuple[tuple[str, int], ...]] = (
    ("autonomous_actual", 119), ("autonomous_reference", 109),
    ("autonomous_future", 123), ("autonomous_geometry", 352),
    ("object_pointnet", 64), ("action_types", 50), ("object_geometry", 12),
)
RAW_OBSERVATION_FIELDS_V5: Final[tuple[tuple[str, int], ...]] = (
    ("autonomous_actual", 119), ("autonomous_reference", 109),
    ("autonomous_future", 12), ("contact_intent", 80),
    ("object_point_cloud_raw", 192), ("hand_point_cloud_raw", 768),
    ("action_types", 50), ("object_geometry", 12),
)
RAW_OBSERVATION_FIELDS_V55: Final[tuple[tuple[str, int], ...]] = (
    ("autonomous_actual", 119),
    ("autonomous_reference", 109),
    ("autonomous_future", 12),
    ("contact_intent", 80),
    ("hand_region_goal_pose", 144),
    ("object_point_cloud_raw", 192),
    ("hand_point_cloud_raw", 768),
    ("action_types", 50),
    ("object_geometry", 12),
)
RAW_OBSERVATION_FIELDS_V575: Final[tuple[tuple[str, int], ...]] = (
    ("autonomous_actual", 119),
    ("autonomous_reference", 109),
    ("autonomous_future", 12),
    ("contact_intent", 80),
    ("hand_region_goal_pose", 144),
    ("hand_region_dynamic_contact", 80),
    ("object_wrench", 15),
    ("object_point_cloud_raw", 192),
    ("hand_point_cloud_raw", 768),
    ("action_types", 50),
    ("object_geometry", 12),
)
ENCODED_OBSERVATION_FIELDS_V5: Final[tuple[tuple[str, int], ...]] = (
    ("autonomous_actual", 119), ("autonomous_reference", 109),
    ("autonomous_future", 12), ("contact_intent", 80),
    ("object_pointnet", 64), ("hand_pointnet", 64),
    ("action_types", 50), ("object_geometry", 12),
)
RAW_OBSERVATION_FIELDS_V6: Final[tuple[tuple[str, int], ...]] = (
    ("autonomous_actual", 119),
    ("autonomous_reference", 109),
    ("autonomous_future", 12),
    # 16 regions x [confidence, valid, active, count, force xyz, slip xyz].
    ("hand_region_contact", 160),
    # gravity, hand force/torque and non-hand force/torque, all in object frame.
    ("object_wrench", 15),
    ("object_point_cloud_raw", 192),
    ("hand_point_cloud_actual_raw", 768),
    ("hand_point_cloud_reference_raw", 768),
    ("action_types", 50),
    ("object_geometry", 12),
)

def _slices(fields: tuple[tuple[str, int], ...]) -> dict[str, slice]:
    at = 0; result = {}
    for name, width in fields:
        result[name] = slice(at, at + width); at += width
    return result

def raw_observation_slices() -> dict[str, slice]: return _slices(RAW_OBSERVATION_FIELDS)
def encoded_observation_slices() -> dict[str, slice]: return _slices(ENCODED_OBSERVATION_FIELDS)
def observation_slices() -> dict[str, slice]: return raw_observation_slices()
def raw_observation_slices_v5() -> dict[str, slice]: return _slices(RAW_OBSERVATION_FIELDS_V5)
def encoded_observation_slices_v5() -> dict[str, slice]: return _slices(ENCODED_OBSERVATION_FIELDS_V5)
def raw_observation_slices_v55() -> dict[str, slice]: return _slices(RAW_OBSERVATION_FIELDS_V55)
def raw_observation_slices_v575() -> dict[str, slice]: return _slices(RAW_OBSERVATION_FIELDS_V575)
def raw_observation_slices_v6() -> dict[str, slice]: return _slices(RAW_OBSERVATION_FIELDS_V6)

@dataclass(frozen=True)
class AutonomousActionContract:
    version: str = ACTION_CONTRACT_ID
    dof: int = ACTION_DIM
    command_mode: str = "rate_limited_measured_state"
    reference_enters_command_map: bool = False
    def __post_init__(self) -> None:
        if self.version != ACTION_CONTRACT_ID or self.dof != ACTION_DIM or self.command_mode != "rate_limited_measured_state" or self.reference_enters_command_map:
            raise ValueError("v4 commands require 28 policy-owned measured-state rate control")

@dataclass(frozen=True)
class AutonomousObservationContract:
    version: str = OBSERVATION_CONTRACT_ID
    raw_dimension: int = RAW_OBSERVATION_DIM
    encoded_dimension: int = ENCODED_OBSERVATION_DIM
    pointnet_input_points: int = 64
    pointnet_embedding: int = 64
    quaternion: str = "xyzw; R columns ex, ey"
    def __post_init__(self) -> None:
        if (self.version != OBSERVATION_CONTRACT_ID or self.raw_dimension != sum(x[1] for x in RAW_OBSERVATION_FIELDS)
                or self.encoded_dimension != sum(x[1] for x in ENCODED_OBSERVATION_FIELDS)
                or (self.pointnet_input_points, self.pointnet_embedding) != (64, 64)):
            raise ValueError("v4 seven-block observation ABI drifted")

@dataclass(frozen=True)
class AutonomousRewardContract:
    version: str = REWARD_CONTRACT_ID
    post_action: bool = True
    force_magnitude_reward: bool = False
    def __post_init__(self) -> None:
        if self.version != REWARD_CONTRACT_ID or not self.post_action or self.force_magnitude_reward:
            raise ValueError("v4 reward is post-action and never rewards force magnitude")

ACTION_CONTRACT = AutonomousActionContract()
OBSERVATION_CONTRACT = AutonomousObservationContract()
REWARD_CONTRACT = AutonomousRewardContract()


def resolve_reward_contract(value: str) -> tuple[str, str]:
    """Return the canonical short reward version and contract identifier."""
    if value in REWARD_CONTRACT_BY_VERSION:
        return value, REWARD_CONTRACT_BY_VERSION[value]
    for version, contract_id in REWARD_CONTRACT_BY_VERSION.items():
        if value == contract_id:
            return version, contract_id
    choices = ", ".join(REWARD_CONTRACT_BY_VERSION)
    raise ValueError(f"unsupported autonomy reward {value!r}; choose {choices}")


@dataclass(frozen=True)
class AutonomousObservationContractV5:
    version: str = OBSERVATION_CONTRACT_ID_V5
    raw_dimension: int = RAW_OBSERVATION_DIM_V5
    encoded_dimension: int = ENCODED_OBSERVATION_DIM_V5
    object_points: int = 64
    hand_points: int = 256
    pointnet_embedding: int = 64

    def __post_init__(self) -> None:
        if (
            self.version != OBSERVATION_CONTRACT_ID_V5
            or self.raw_dimension != sum(width for _, width in RAW_OBSERVATION_FIELDS_V5)
            or self.encoded_dimension != sum(width for _, width in ENCODED_OBSERVATION_FIELDS_V5)
            or (self.object_points, self.hand_points, self.pointnet_embedding)
            != (64, 256, 64)
        ):
            raise ValueError("v5 point-cloud observation ABI drifted")


OBSERVATION_CONTRACT_V5 = AutonomousObservationContractV5()


@dataclass(frozen=True)
class AutonomousObservationContractV525:
    version: str = OBSERVATION_CONTRACT_ID_V525
    raw_dimension: int = RAW_OBSERVATION_DIM_V525
    encoded_dimension: int = ENCODED_OBSERVATION_DIM_V525
    hand_regions: int = 16
    points_per_region: int = 16
    region_embedding: int = 64

    def __post_init__(self) -> None:
        if (
            self.version != OBSERVATION_CONTRACT_ID_V525
            or self.raw_dimension != RAW_OBSERVATION_DIM_V5
            or self.encoded_dimension != ENCODED_OBSERVATION_DIM_V5
            or (self.hand_regions, self.points_per_region, self.region_embedding)
            != (16, 16, 64)
        ):
            raise ValueError("v5.25 region-hand observation ABI drifted")


OBSERVATION_CONTRACT_V525 = AutonomousObservationContractV525()


@dataclass(frozen=True)
class AutonomousObservationContractV55:
    version: str = OBSERVATION_CONTRACT_ID_V55
    raw_dimension: int = RAW_OBSERVATION_DIM_V55
    encoded_dimension: int = ENCODED_OBSERVATION_DIM_V55
    hand_regions: int = 16
    goal_pose_per_region: int = 9
    cross_attention_layers: int = 1

    def __post_init__(self) -> None:
        if (
            self.version != OBSERVATION_CONTRACT_ID_V55
            or self.raw_dimension != sum(width for _, width in RAW_OBSERVATION_FIELDS_V55)
            or self.encoded_dimension != ENCODED_OBSERVATION_DIM_V5
            or (self.hand_regions, self.goal_pose_per_region)
            != (16, 9)
            or self.cross_attention_layers != 1
        ):
            raise ValueError("v5.5 region-goal observation ABI drifted")


OBSERVATION_CONTRACT_V55 = AutonomousObservationContractV55()


@dataclass(frozen=True)
class AutonomousObservationContractV575:
    version: str = OBSERVATION_CONTRACT_ID_V575
    raw_dimension: int = RAW_OBSERVATION_DIM_V575
    encoded_dimension: int = ENCODED_OBSERVATION_DIM_V575
    hand_regions: int = 16
    dynamic_contact_per_region: int = 5
    object_wrench_dimension: int = 15

    def __post_init__(self) -> None:
        if (
            self.version != OBSERVATION_CONTRACT_ID_V575
            or self.raw_dimension != sum(width for _, width in RAW_OBSERVATION_FIELDS_V575)
            or self.encoded_dimension != ENCODED_OBSERVATION_DIM_V575
            or (self.hand_regions, self.dynamic_contact_per_region)
            != (16, 5)
            or self.object_wrench_dimension != 15
        ):
            raise ValueError("v5.75 contact-wrench observation ABI drifted")


OBSERVATION_CONTRACT_V575 = AutonomousObservationContractV575()


@dataclass(frozen=True)
class AutonomousObservationContractV6:
    version: str = OBSERVATION_CONTRACT_ID_V6
    raw_dimension: int = RAW_OBSERVATION_DIM_V6
    model_dimension: int = MODEL_DIM_V6
    object_patches: int = 16
    hand_regions: int = 16
    points_per_object_patch: int = 4
    points_per_hand_region: int = 16
    current_tokens: int = CURRENT_TOKENS_V6
    goal_tokens: int = GOAL_TOKENS_V6

    def __post_init__(self) -> None:
        if (
            self.version != OBSERVATION_CONTRACT_ID_V6
            or self.raw_dimension != sum(width for _, width in RAW_OBSERVATION_FIELDS_V6)
            or self.model_dimension != 128
            or (self.object_patches, self.hand_regions) != (16, 16)
            or (self.points_per_object_patch, self.points_per_hand_region) != (4, 16)
            or (self.current_tokens, self.goal_tokens) != (34, 16)
        ):
            raise ValueError("v6 token observation ABI drifted")


OBSERVATION_CONTRACT_V6 = AutonomousObservationContractV6()

def _validate_checkpoint_metadata(
    metadata: dict[str, object],
    *,
    checkpoint_format: str,
    observation_contract: str,
    version: str,
) -> None:
    required = {
        "checkpoint_format": checkpoint_format,
        "observation_contract": observation_contract,
        "action_contract": ACTION_CONTRACT_ID,
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(f"incompatible checkpoint: {name} must be {expected!r}")
    if metadata.get("reward_contract") not in SUPPORTED_REWARD_CONTRACT_IDS:
        raise ValueError("incompatible checkpoint: unsupported reward_contract")
    if metadata.get("policy_sampling_contract") not in (None, POLICY_SAMPLING_CONTRACT):
        raise ValueError(f"incompatible {version} policy sampling contract")


def validate_v4_checkpoint_metadata(metadata: dict[str, object]) -> None:
    _validate_checkpoint_metadata(
        metadata,
        checkpoint_format=CHECKPOINT_FORMAT,
        observation_contract=OBSERVATION_CONTRACT_ID,
        version="v4",
    )


def validate_v5_checkpoint_metadata(metadata: dict[str, object]) -> None:
    _validate_checkpoint_metadata(
        metadata,
        checkpoint_format=CHECKPOINT_FORMAT_V5,
        observation_contract=OBSERVATION_CONTRACT_ID_V5,
        version="v5",
    )


def validate_v525_checkpoint_metadata(metadata: dict[str, object]) -> None:
    _validate_checkpoint_metadata(
        metadata,
        checkpoint_format=CHECKPOINT_FORMAT_V525,
        observation_contract=OBSERVATION_CONTRACT_ID_V525,
        version="v5.25",
    )


def validate_v55_checkpoint_metadata(metadata: dict[str, object]) -> None:
    _validate_checkpoint_metadata(
        metadata,
        checkpoint_format=CHECKPOINT_FORMAT_V55,
        observation_contract=OBSERVATION_CONTRACT_ID_V55,
        version="v5.5",
    )


def validate_v575_checkpoint_metadata(metadata: dict[str, object]) -> None:
    _validate_checkpoint_metadata(
        metadata,
        checkpoint_format=CHECKPOINT_FORMAT_V575,
        observation_contract=OBSERVATION_CONTRACT_ID_V575,
        version="v5.75",
    )


def validate_v6_checkpoint_metadata(metadata: dict[str, object]) -> None:
    _validate_checkpoint_metadata(
        metadata,
        checkpoint_format=CHECKPOINT_FORMAT_V6,
        observation_contract=OBSERVATION_CONTRACT_ID_V6,
        version="v6",
    )

def rate_limited_command(previous_command: NDArray[np.floating], action: NDArray[np.floating], lower: NDArray[np.floating], upper: NDArray[np.floating], rate_per_second: NDArray[np.floating], *, measured_qpos: NDArray[np.floating] | None = None, control_timestep: float = 1 / 120, max_tracking_error: NDArray[np.floating] | None = None) -> NDArray[np.float64]:
    previous, raw, lo, hi, rate = (np.asarray(x, dtype=np.float64) for x in (previous_command, action, lower, upper, rate_per_second))
    if any(x.shape != (ACTION_DIM,) for x in (previous, raw, lo, hi, rate)) or not np.all(np.isfinite(np.stack((previous, raw, lo, hi, rate)))) or np.any(rate <= 0) or np.any(hi <= lo):
        raise ValueError("v4 command requires finite 28D ordered limits and rates")
    delta = np.clip(raw, -1., 1.) * rate * control_timestep
    if measured_qpos is not None:
        measured = np.asarray(measured_qpos, dtype=np.float64)
        margin = np.asarray(max_tracking_error if max_tracking_error is not None else rate * control_timestep * 4., dtype=np.float64)
        if measured.shape != (ACTION_DIM,) or margin.shape != (ACTION_DIM,) or np.any(margin <= 0): raise ValueError("invalid measured-state envelope")
        error = previous - measured
        # Preserve load-bearing servo error; only block further outward motion.
        delta = np.where((np.abs(error) >= margin) & (error * delta > 0), 0., delta)
    return np.clip(previous + delta, lo, hi)
