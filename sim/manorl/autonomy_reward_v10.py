"""Reward v10 for the current ManoRL v4 MDP and v5-v6 policy families.

The environment, observation, action, PPO and network contracts remain
unchanged. Reward v10 keeps the reference-speed smoothstep tracking already
implemented by reward v4 and adds the validated reward-v8 contact/lift terms
plus dense airborne hold, contact-loss and falling feedback.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, NamedTuple

from sim.manorl.autonomy_contracts import (
    V10_REWARD_CONTRACT_ID,
    V10_REWARD_TERM_NAMES,
)
from sim.manorl.autonomy_v4 import _gather, compute_reward as compute_reward_v4


THUMB_REGION_INDICES = (13, 14, 15)
NONTHUMB_REGION_INDICES = tuple(range(13))


@dataclass(frozen=True)
class RewardV10Config:
    contract_id: str = V10_REWARD_CONTRACT_ID
    loaded_contact_force_n: float = 0.02
    contact_max: float = 2.0
    thumb_contact_max: float = 3.0
    static_contact_scale: float = 1.0
    moving_contact_scale: float = 2.0
    contact_motion_speed_low_mps: float = 0.01
    contact_motion_speed_high_mps: float = 0.15
    contact_angular_speed_radius_m: float = 0.05
    nonthumb_quality_saturation: float = 1.0 / 13.0
    thumb_quality_saturation: float = 1.0 / 3.0
    opposing_contact_max: float = 2.0
    lift_phase_clearance_m: float = 0.05
    lift_deficit_normalizer_m: float = 0.10
    lift_deficit_penalty_max: float = 16.0
    lift_contact_success_max: float = 8.0
    reference_upward_speed_mps: float = 0.20
    actual_upward_speed_mps: float = 0.20
    actual_downward_speed_mps: float = 0.10
    upward_velocity_reward_max: float = 8.0
    downward_velocity_penalty_max: float = 4.0
    lateral_speed_mps: float = 0.10
    lateral_slip_penalty_max: float = 4.0
    hold_reference_clearance_m: float = 0.03
    airborne_clearance_m: float = 0.003
    hold_contact_reward_max: float = 12.0
    contact_loss_penalty_max: float = 20.0
    falling_speed_mps: float = 0.20
    falling_penalty_max: float = 12.0
    severe_penalty: float = -400.0

    def __post_init__(self) -> None:
        positive = tuple(
            value
            for name, value in self.__dict__.items()
            if name not in {"contract_id", "severe_penalty"}
        )
        if not all(math.isfinite(value) for value in (*positive, self.severe_penalty)):
            raise ValueError("reward v10 coefficients must be finite")
        if self.contract_id != V10_REWARD_CONTRACT_ID:
            raise ValueError("reward v10 contract identifier drifted")
        if min(positive) <= 0:
            raise ValueError("reward v10 coefficients must be positive")
        if self.contact_motion_speed_low_mps >= self.contact_motion_speed_high_mps:
            raise ValueError("reward v10 contact-motion bounds must be ordered")
        if self.static_contact_scale > self.moving_contact_scale:
            raise ValueError("reward v10 contact scales must be ordered")
        if self.severe_penalty >= 0:
            raise ValueError("reward v10 severe penalty must be negative")


DEFAULT_REWARD_V10_CONFIG = RewardV10Config()
REWARD_NAMES = V10_REWARD_TERM_NAMES


class V10Reward(NamedTuple):
    total: Any
    object_position: Any
    object_rotation: Any
    object_velocity: Any
    hand_relative: Any
    fingers: Any
    geometry: Any
    contact: Any
    thumb_contact: Any
    opposing_contact: Any
    lift_progress: Any
    lift_velocity: Any
    hold_contact: Any
    contact_loss: Any
    falling: Any
    lateral_slip: Any
    action: Any
    survival: Any
    severe: Any
    done: Any
    reason: Any
    valid: Any
    contact_scale: Any
    reference_motion_speed: Any
    hold_phase: Any


def _weighted_loaded_fraction(weight, loaded, *, jp):
    numerator = jp.sum(weight * loaded, axis=-1)
    denominator = jp.sum(weight, axis=-1)
    return jp.where(denominator > 0, numerator / denominator, 0.0)


def _contact_qualities(contact, cache, gathered, config, *, jp):
    intent_weight = (
        jp.asarray(cache.proximity)[gathered]
        * jp.asarray(cache.confidence)[gathered]
        * jp.asarray(cache.valid)[gathered]
    )
    loaded = (
        jp.linalg.norm(contact.paired_force_on_object, axis=-1)
        >= config.loaded_contact_force_n
    ).astype(intent_weight.dtype)
    all_quality = _weighted_loaded_fraction(intent_weight, loaded, jp=jp)
    nonthumb_quality = _weighted_loaded_fraction(
        intent_weight[:, NONTHUMB_REGION_INDICES],
        loaded[:, NONTHUMB_REGION_INDICES],
        jp=jp,
    )
    thumb_quality = _weighted_loaded_fraction(
        intent_weight[:, THUMB_REGION_INDICES],
        loaded[:, THUMB_REGION_INDICES],
        jp=jp,
    )
    nonthumb_presence = jp.clip(
        nonthumb_quality / config.nonthumb_quality_saturation, 0.0, 1.0,
    )
    thumb_presence = jp.clip(
        thumb_quality / config.thumb_quality_saturation, 0.0, 1.0,
    )
    opposition_quality = jp.minimum(nonthumb_presence, thumb_presence)
    contact_activity = jp.maximum(nonthumb_presence, thumb_presence)
    return all_quality, thumb_quality, opposition_quality, contact_activity


def reward_parameters_v10(
    config: RewardV10Config = DEFAULT_REWARD_V10_CONFIG,
) -> dict[str, object]:
    """Return the complete JSON-safe reward-v10 checkpoint provenance."""
    return {
        "id": config.contract_id,
        "tracking": {
            "source": "reward_v4_reference_speed_smoothstep",
            "static_weight": 0.01,
            "moving_weight": 1.0,
            "smoothstep_low_m_per_s": 0.01,
            "smoothstep_high_m_per_s": 0.10,
            "hand_relative": 0.125,
            "geometry": 1.2,
        },
        "coefficients": {
            name: value
            for name, value in config.__dict__.items()
            if name != "contract_id"
        },
        "thumb_regions": list(THUMB_REGION_INDICES),
        "nonthumb_regions": list(NONTHUMB_REGION_INDICES),
    }


def compute_reward(
    physical,
    contact,
    cache,
    index,
    executed_action,
    env_ref=None,
    *,
    config: RewardV10Config = DEFAULT_REWARD_V10_CONFIG,
) -> V10Reward:
    """Compute reward v10 without changing the surrounding MDP or PPO."""
    import jax.numpy as jp

    base = compute_reward_v4(
        physical, contact, cache, index, executed_action, env_ref,
    )
    gathered = _gather(cache, index, env_ref=env_ref)
    reference_linear = jp.asarray(cache.object_v_com)[gathered]
    reference_angular = jp.asarray(cache.object_w)[gathered]
    reference_motion_speed = (
        jp.linalg.norm(reference_linear, axis=-1)
        + config.contact_angular_speed_radius_m
        * jp.linalg.norm(reference_angular, axis=-1)
    )
    contact_alpha = jp.clip(
        (
            reference_motion_speed - config.contact_motion_speed_low_mps
        )
        / (
            config.contact_motion_speed_high_mps
            - config.contact_motion_speed_low_mps
        ),
        0.0,
        1.0,
    )
    contact_scale = (
        config.static_contact_scale
        + contact_alpha
        * (config.moving_contact_scale - config.static_contact_scale)
    )
    (
        contact_quality,
        thumb_quality,
        opposition_quality,
        contact_activity,
    ) = _contact_qualities(
        contact, cache, gathered, config, jp=jp,
    )
    contact_reward = config.contact_max * contact_scale * contact_quality
    thumb_contact = (
        config.thumb_contact_max * contact_scale * thumb_quality
    )

    reference_clearance = jp.maximum(
        jp.asarray(cache.reference_bottom)[gathered] - cache.table_height,
        0.0,
    )
    actual_clearance = physical.object_bottom - cache.table_height
    lift_phase = jp.clip(
        reference_clearance / config.lift_phase_clearance_m, 0.0, 1.0,
    )
    lift_deficit = jp.clip(
        (reference_clearance - actual_clearance)
        / config.lift_deficit_normalizer_m,
        0.0,
        1.0,
    )
    actual_progress = jp.clip(
        actual_clearance / jp.maximum(reference_clearance, 0.005),
        0.0,
        1.0,
    )
    opposing_contact = (
        config.opposing_contact_max
        * (1.0 - lift_phase)
        * opposition_quality
    )
    lift_progress = lift_phase * (
        config.lift_contact_success_max
        * actual_progress
        * opposition_quality
        - config.lift_deficit_penalty_max * lift_deficit
    )

    reference_upward = reference_linear[..., 2]
    directional_phase = jp.clip(
        reference_upward / config.reference_upward_speed_mps, 0.0, 1.0,
    )
    actual_upward = jp.clip(
        physical.object_v_com[..., 2] / config.actual_upward_speed_mps,
        0.0,
        1.0,
    )
    actual_downward = jp.clip(
        -physical.object_v_com[..., 2] / config.actual_downward_speed_mps,
        0.0,
        1.0,
    )
    lift_velocity = directional_phase * opposition_quality * (
        config.upward_velocity_reward_max * actual_upward
        - config.downward_velocity_penalty_max * actual_downward
    )
    lateral_speed = jp.linalg.norm(physical.object_v_com[..., :2], axis=-1)
    lateral_slip = (
        -config.lateral_slip_penalty_max
        * directional_phase
        * contact_activity
        * jp.clip(lateral_speed / config.lateral_speed_mps, 0.0, 1.0)
    )

    hold_phase = jp.clip(
        reference_clearance / config.hold_reference_clearance_m, 0.0, 1.0,
    )
    actual_airborne = jp.clip(
        actual_clearance / config.airborne_clearance_m, 0.0, 1.0,
    )
    hold_contact = (
        config.hold_contact_reward_max
        * hold_phase
        * actual_airborne
        * opposition_quality
    )
    contact_loss = (
        -config.contact_loss_penalty_max
        * hold_phase
        * actual_airborne
        * (1.0 - opposition_quality)
    )
    falling = (
        -config.falling_penalty_max
        * hold_phase
        * actual_airborne
        * jp.clip(
            -physical.object_v_com[..., 2] / config.falling_speed_mps,
            0.0,
            1.0,
        )
    )

    terms_without_severe = (
        base.object_position
        + base.object_rotation
        + base.object_velocity
        + base.hand_relative
        + base.fingers
        + base.geometry
        + contact_reward
        + thumb_contact
        + opposing_contact
        + lift_progress
        + lift_velocity
        + hold_contact
        + contact_loss
        + falling
        + lateral_slip
        + base.action
        + base.survival
    )
    valid = base.valid & jp.isfinite(terms_without_severe)
    severe = jp.where(base.severe < 0, config.severe_penalty, 0.0)
    total = jp.where(
        valid, terms_without_severe + severe, config.severe_penalty,
    )
    return V10Reward(
        total,
        base.object_position,
        base.object_rotation,
        base.object_velocity,
        base.hand_relative,
        base.fingers,
        base.geometry,
        contact_reward,
        thumb_contact,
        opposing_contact,
        lift_progress,
        lift_velocity,
        hold_contact,
        contact_loss,
        falling,
        lateral_slip,
        base.action,
        base.survival,
        severe,
        base.done,
        base.reason,
        valid,
        contact_scale,
        reference_motion_speed,
        hold_phase,
    )
