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

def _slices(fields: tuple[tuple[str, int], ...]) -> dict[str, slice]:
    at = 0; result = {}
    for name, width in fields:
        result[name] = slice(at, at + width); at += width
    return result

def raw_observation_slices() -> dict[str, slice]: return _slices(RAW_OBSERVATION_FIELDS)
def encoded_observation_slices() -> dict[str, slice]: return _slices(ENCODED_OBSERVATION_FIELDS)
def observation_slices() -> dict[str, slice]: return raw_observation_slices()

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

def validate_v4_checkpoint_metadata(metadata: dict[str, object]) -> None:
    required = {"checkpoint_format": CHECKPOINT_FORMAT, "observation_contract": OBSERVATION_CONTRACT_ID,
                "reward_contract": REWARD_CONTRACT_ID, "action_contract": ACTION_CONTRACT_ID}
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(f"incompatible checkpoint: {name} must be {expected!r}")
    if metadata.get("policy_sampling_contract") not in (None, POLICY_SAMPLING_CONTRACT):
        raise ValueError("incompatible v4 policy sampling contract")

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
