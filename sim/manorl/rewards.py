"""ManoRL reward equations over resolved environment state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from sim.manorl.observations import (
    CONTACT_FORCE_THRESHOLD,
    ObservationCompatibility,
    _batch_vector,
    _finite_batch,
    _finite_tensor,
)
from sim.manorl.abi import TerminationResult


REWARD_CONTRACT_ID: Final = "target_hand_object_contact_no_action_deviation_penalty_v3"
REWARD_HAND_OBJECT_THRESHOLD_N: Final[float] = 1.0
# PPO consumes the environment reward directly, so its contract is distinct
# because the PPO reward scale is tracked separately from the environment contract.
PPO_REWARD_CONTRACT_ID: Final = "target_hand_object_contact_no_action_deviation_penalty_v3_raw_ppo_reward_1x_v3"
PPO_REWARD_SCALE: Final[float] = 1.0


@dataclass(frozen=True)
class RewardConfig:
    distance_scales: tuple[float, float, float] = (0.5, 0.5, 2.0)
    distance_decay: float = 40.0
    rotation_scale: float = 0.4
    rotation_base_penalty: float = -0.1
    rotation_segment_1_threshold_deg: float = 20.0
    rotation_segment_2_threshold_deg: float = 90.0
    rotation_segment_1_coeff: float = -0.00125
    rotation_segment_2_coeff: float = -0.00003175
    rotation_segment_2_linear_coeff: float = -0.019206
    rotation_segment_2_constant: float = 0.5
    rotation_segment_3_value: float = -1.0
    action_penalty_scale: float = 0.0
    position_penalty_weight: float = 0.025
    joint_penalty_weight: float = 0.0085
    position_penalty_scale: float = 100.0
    joint_penalty_scale: float = 10.0
    reference_joint_count: float = 8.0
    max_contact_reward: float = 0.4
    direct_contact_reward_scale: float = 3.0
    contact_force_threshold: float = REWARD_HAND_OBJECT_THRESHOLD_N
    max_object_stability_reward: float = 0.4
    object_stability_reference_speed: float = 0.1
    survival_reward: float = 0.001


@dataclass(frozen=True)
class RewardState:
    object_position: NDArray[np.float64]
    target_object_position: NDArray[np.float64]
    object_orientation_xyzw: NDArray[np.float64]
    target_object_orientation_xyzw: NDArray[np.float64]
    cumulative_offset: NDArray[np.float64]
    cumulative_joint_offset: NDArray[np.float64]
    active_joint_mask: NDArray[np.bool_]
    hand_object_force_on_object_world_N: NDArray[np.float64]
    expected_contact_mask: NDArray[np.float64]
    expected_contact_weights: NDArray[np.float64]
    object_linear_velocity: NDArray[np.float64]
    trajectory_steps: NDArray[np.int64]
    contact_start_frames: NDArray[np.int64]
    contact_end_frames: NDArray[np.int64]
    rotation_disabled_mask: NDArray[np.bool_]
    early_phase_starts: NDArray[np.int64] | None = None


@dataclass(frozen=True)
class RewardDiagnostics:
    total: NDArray[np.float64]
    distance_x: NDArray[np.float64]
    distance_y: NDArray[np.float64]
    distance_z: NDArray[np.float64]
    ungated_distance_x: NDArray[np.float64]
    ungated_distance_y: NDArray[np.float64]
    ungated_distance_z: NDArray[np.float64]
    rotation: NDArray[np.float64]
    position_penalty: NDArray[np.float64]
    joint_penalty: NDArray[np.float64]
    action_penalty: NDArray[np.float64]
    raw_contact: NDArray[np.float64]
    contact: NDArray[np.float64]
    distance_gate: NDArray[np.float64]
    object_stability: NDArray[np.float64]
    object_speed: NDArray[np.float64]
    survival: NDArray[np.float64]
    early_phase: NDArray[np.bool_]
    deviation_penalty: NDArray[np.float64]


def _quat_diff_degrees(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Match source quat_mul(a, conjugate(b)) then 2*asin(||xyz||)."""

    ax, ay, az, aw = (a[:, i] for i in range(4))
    bx, by, bz, bw = (-b[:, 0], -b[:, 1], -b[:, 2], b[:, 3])
    product_xyz = np.stack((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), axis=1)
    radians = 2.0 * np.arcsin(np.minimum(np.linalg.norm(product_xyz, axis=1), 1.0))
    return radians * 180.0 / 3.14159265359


def compute_rewards(
    state: RewardState,
    *,
    compatibility: ObservationCompatibility,
    termination: TerminationResult,
    config: RewardConfig = RewardConfig(),
) -> RewardDiagnostics:
    """Compute reward terms plus the ABI termination contribution (default zero, configurable)."""

    if not isinstance(state, RewardState):
        raise TypeError("state must be a RewardState")
    if not isinstance(compatibility, ObservationCompatibility):
        raise TypeError("compatibility must be an explicit ObservationCompatibility")
    if not isinstance(termination, TerminationResult):
        raise TypeError("termination must be the result of sim.manorl.abi.check_termination")
    object_position = _finite_batch("object_position", state.object_position, 3)
    batch = len(object_position)
    target_position = _finite_batch("target_object_position", state.target_object_position, 3)
    object_orientation = _finite_batch("object_orientation_xyzw", state.object_orientation_xyzw, 4)
    target_orientation = _finite_batch("target_object_orientation_xyzw", state.target_object_orientation_xyzw, 4)
    cumulative_offset = _finite_batch("cumulative_offset", state.cumulative_offset, 3)
    cumulative_joint = _finite_batch("cumulative_joint_offset", state.cumulative_joint_offset, 20)
    hand_object_forces = _finite_tensor(
        "hand_object_force_on_object_world_N", state.hand_object_force_on_object_world_N, (16, 3), batch
    )
    expected_mask = _finite_batch("expected_contact_mask", state.expected_contact_mask, 16)
    expected_weights = _finite_batch("expected_contact_weights", state.expected_contact_weights, 16)
    object_velocity = _finite_batch("object_linear_velocity", state.object_linear_velocity, 3)
    active = np.asarray(state.active_joint_mask, dtype=bool)
    rotation_disabled = np.asarray(state.rotation_disabled_mask, dtype=bool)
    if active.shape != (batch, 20) or rotation_disabled.shape != (batch,):
        raise ValueError("active_joint_mask and rotation_disabled_mask have invalid shape")
    for name, values in (
        ("target_object_position", target_position), ("object_orientation_xyzw", object_orientation),
        ("target_object_orientation_xyzw", target_orientation), ("cumulative_offset", cumulative_offset),
        ("cumulative_joint_offset", cumulative_joint), ("expected_contact_mask", expected_mask),
        ("expected_contact_weights", expected_weights), ("object_linear_velocity", object_velocity),
    ):
        if len(values) != batch:
            raise ValueError(f"{name} batch size must match object_position")
    if np.any((expected_mask != 0.0) & (expected_mask != 1.0)) or np.any(expected_weights < 0.0):
        raise ValueError("expected contact masks must be binary and weights non-negative")
    deviation_penalty = _batch_vector("termination.deviation_penalty", termination.deviation_penalty, batch)
    deviation_reset = np.asarray(termination.deviation_reset, dtype=bool)
    reset = np.asarray(termination.reset, dtype=bool)
    if deviation_reset.shape != (batch,) or reset.shape != (batch,):
        raise ValueError("termination result batch size must match object_position")
    if np.any(deviation_reset & ~reset):
        raise ValueError("a deviation termination must also set the reset mask")
    steps = np.asarray(state.trajectory_steps)
    starts = np.asarray(state.contact_start_frames)
    ends = np.asarray(state.contact_end_frames)
    if any(array.shape != (batch,) or not np.issubdtype(array.dtype, np.integer) for array in (steps, starts, ends)):
        raise ValueError("trajectory and contact-window fields must be integer (batch,) arrays")
    if state.early_phase_starts is None:
        early_starts = np.zeros(batch, dtype=np.int64)
    else:
        early_starts = np.asarray(state.early_phase_starts)
        if early_starts.shape != (batch,) or not np.issubdtype(early_starts.dtype, np.integer):
            raise ValueError("early_phase_starts must be an integer (batch,) array")
    early_phase = (steps >= early_starts) & (steps < early_starts + compatibility.early_phase_steps)
    distances = np.abs(object_position - target_position)
    ungated_x = config.distance_scales[0] * np.exp(-config.distance_decay * distances[:, 0])
    ungated_y = config.distance_scales[1] * np.exp(-config.distance_decay * distances[:, 1])
    ungated_z = config.distance_scales[2] * np.exp(-config.distance_decay * distances[:, 2])
    rotation_deg = _quat_diff_degrees(object_orientation, target_orientation)
    segment_1 = config.rotation_segment_1_coeff * rotation_deg**2 + 1.0
    offset = rotation_deg - config.rotation_segment_1_threshold_deg
    segment_2 = config.rotation_segment_2_coeff * offset**2 + config.rotation_segment_2_linear_coeff * offset + config.rotation_segment_2_constant
    rotation_value = np.where(
        rotation_deg <= config.rotation_segment_1_threshold_deg,
        segment_1,
        np.where(rotation_deg <= config.rotation_segment_2_threshold_deg, segment_2, config.rotation_segment_3_value),
    )
    rotation = np.where(rotation_disabled, 0.0, config.rotation_scale * rotation_value + config.rotation_base_penalty)
    position_base = np.sum(np.abs(cumulative_offset * config.position_penalty_scale), axis=1)
    joint_base = np.sum(np.where(active, np.abs(cumulative_joint * config.joint_penalty_scale), 0.0), axis=1)
    joint_base = joint_base / np.maximum(active.sum(axis=1), 1.0) * config.reference_joint_count
    position_penalty = -config.action_penalty_scale * config.position_penalty_weight * position_base
    joint_penalty = -config.action_penalty_scale * config.joint_penalty_weight * joint_base
    action_penalty = position_penalty + joint_penalty
    contact_magnitudes = np.linalg.norm(hand_object_forces, axis=-1)
    weighted_expected = np.sum(expected_mask * expected_weights, axis=1)
    weighted_correct = np.sum(
        (contact_magnitudes > config.contact_force_threshold) * expected_mask * expected_weights, axis=1
    )
    raw_contact = np.divide(
        weighted_correct, weighted_expected, out=np.zeros(batch, dtype=np.float64), where=weighted_expected > 0
    ) * config.max_contact_reward
    within_window = (steps >= starts) & (steps <= ends)
    windowed_contact = np.where(within_window, raw_contact, 0.0)
    contact = windowed_contact * config.direct_contact_reward_scale
    valid_window = ends >= starts
    distance_gate = np.where(within_window & valid_window, windowed_contact, 0.0)
    distance_gate = np.where((steps > ends) & valid_window, config.max_contact_reward, distance_gate)
    distance_x, distance_y, distance_z = ungated_x * distance_gate, ungated_y * distance_gate, ungated_z * distance_gate
    object_speed = np.linalg.norm(object_velocity, axis=1)
    if config.object_stability_reference_speed <= 0:
        stability_base = np.zeros(batch, dtype=np.float64)
    else:
        stability_base = config.max_object_stability_reward * np.exp(-(object_speed / config.object_stability_reference_speed) ** 2)
    object_stability = np.where(valid_window & (steps > ends), stability_base, 0.0)
    survival = np.full(batch, config.survival_reward, dtype=np.float64)
    non_early_total = distance_x + distance_y + distance_z + rotation + action_penalty + contact + object_stability + survival
    total = np.where(early_phase, action_penalty, non_early_total) + deviation_penalty
    if not np.all(np.isfinite(total)):
        raise ValueError("computed reward is non-finite")
    return RewardDiagnostics(
        total=total, distance_x=distance_x, distance_y=distance_y, distance_z=distance_z,
        ungated_distance_x=ungated_x, ungated_distance_y=ungated_y, ungated_distance_z=ungated_z,
        rotation=rotation, position_penalty=position_penalty, joint_penalty=joint_penalty,
        action_penalty=action_penalty, raw_contact=raw_contact, contact=contact,
        distance_gate=distance_gate,
        object_stability=object_stability, object_speed=object_speed, survival=survival,
        early_phase=early_phase, deviation_penalty=deviation_penalty,
    )
