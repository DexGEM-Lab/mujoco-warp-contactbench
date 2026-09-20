"""Temporal grasp-and-lift Reward v10.1.

Reward v10.1 preserves the v10 physical terms, but makes grasp progression an
explicit per-environment state machine. Short contact spikes no longer receive
the same opposition/lift/hold credit as a persistent thumb-versus-finger grip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, NamedTuple

from sim.manorl.autonomy_contracts import (
    V101_REWARD_CONTRACT_ID,
    V101_REWARD_TERM_NAMES,
)
from sim.manorl.autonomy_reward_v10 import (
    RewardV10Config,
    _contact_qualities,
    compute_reward as compute_reward_v10,
)
from sim.manorl.autonomy_v4 import _gather


@dataclass(frozen=True)
class RewardV101Config:
    """Immutable v10.1 reward contract and temporal thresholds."""

    contract_id: str = V101_REWARD_CONTRACT_ID
    base: RewardV10Config = field(
        default_factory=lambda: RewardV10Config(severe_penalty=-100.0)
    )
    opposition_quality_threshold: float = 0.5
    stable_opposition_frames: int = 8
    lift_reward_multiplier: float = 1.75
    hold_growth_frames: int = 60
    hold_growth_max_bonus: float = 1.0
    stalled_grace_frames: int = 12
    stalled_decay_frames: int = 60
    stalled_penalty_max: float = 6.0
    minimum_lift_speed_mps: float = 0.005
    post_grasp_drop_speed_mps: float = 0.20
    post_grasp_drop_penalty_max: float = 12.0
    landed_reset_clearance_m: float = 0.001

    def __post_init__(self) -> None:
        if self.contract_id != V101_REWARD_CONTRACT_ID:
            raise ValueError("reward v10.1 contract identifier drifted")
        integer_fields = (
            self.stable_opposition_frames,
            self.hold_growth_frames,
            self.stalled_grace_frames,
            self.stalled_decay_frames,
        )
        if any(type(value) is not int or value < 1 for value in integer_fields):
            raise ValueError("reward v10.1 frame thresholds must be positive integers")
        scalars = (
            self.opposition_quality_threshold,
            self.lift_reward_multiplier,
            self.hold_growth_max_bonus,
            self.stalled_penalty_max,
            self.minimum_lift_speed_mps,
            self.post_grasp_drop_speed_mps,
            self.post_grasp_drop_penalty_max,
            self.landed_reset_clearance_m,
        )
        if not all(math.isfinite(value) and value > 0 for value in scalars):
            raise ValueError("reward v10.1 coefficients must be positive finite")
        if self.opposition_quality_threshold > 1:
            raise ValueError("reward v10.1 opposition threshold must not exceed one")


DEFAULT_REWARD_V101_CONFIG = RewardV101Config()
REWARD_NAMES = V101_REWARD_TERM_NAMES


class RewardV101State(NamedTuple):
    opposition_streak: Any
    airborne_hold_streak: Any
    stalled_contact_streak: Any
    had_airborne: Any


class V101Reward(NamedTuple):
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
    stalled_contact: Any
    post_grasp_drop: Any
    action: Any
    survival: Any
    severe: Any
    done: Any
    reason: Any
    valid: Any
    contact_scale: Any
    reference_motion_speed: Any
    hold_phase: Any
    opposition_loaded: Any
    positive_lift: Any
    stable_airborne: Any


def initial_reward_state(num_envs: int, *, jp) -> RewardV101State:
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("reward v10.1 state requires a positive environment count")
    zeros = jp.zeros((num_envs,), dtype=jp.int32)
    return RewardV101State(zeros, zeros, zeros, jp.zeros((num_envs,), dtype=bool))


def reset_reward_state(state: RewardV101State, mask, *, jp) -> RewardV101State:
    mask = jp.asarray(mask, dtype=bool)
    zeros = jp.zeros_like(state.opposition_streak)
    return RewardV101State(
        jp.where(mask, zeros, state.opposition_streak),
        jp.where(mask, zeros, state.airborne_hold_streak),
        jp.where(mask, zeros, state.stalled_contact_streak),
        jp.where(mask, False, state.had_airborne),
    )


def reward_parameters_v101(
    config: RewardV101Config = DEFAULT_REWARD_V101_CONFIG,
) -> dict[str, object]:
    return {
        "id": config.contract_id,
        "base": {
            "id": config.base.contract_id,
            "coefficients": {
                name: value
                for name, value in config.base.__dict__.items()
                if name != "contract_id"
            },
        },
        "state_machine": {
            name: value
            for name, value in config.__dict__.items()
            if name not in {"contract_id", "base"}
        },
        "stages": [
            "approach",
            "thumb_contact",
            "persistent_opposition",
            "positive_lift",
            "stable_airborne_hold",
        ],
    }


def compute_reward(
    physical,
    contact,
    cache,
    index,
    executed_action,
    state: RewardV101State,
    env_ref=None,
    *,
    config: RewardV101Config = DEFAULT_REWARD_V101_CONFIG,
) -> tuple[V101Reward, RewardV101State]:
    """Return the post-action reward and next temporal state."""
    import jax.numpy as jp

    base = compute_reward_v10(
        physical,
        contact,
        cache,
        index,
        executed_action,
        env_ref,
        config=config.base,
    )
    gathered = _gather(cache, index, env_ref=env_ref)
    _, _, opposition_quality, _ = _contact_qualities(
        contact, cache, gathered, config.base, jp=jp
    )
    opposition_loaded = opposition_quality >= config.opposition_quality_threshold
    opposition_streak = jp.where(
        opposition_loaded, state.opposition_streak + 1, 0
    )
    stable_grip = opposition_streak >= config.stable_opposition_frames

    actual_clearance = physical.object_bottom - cache.table_height
    airborne = actual_clearance >= config.base.airborne_clearance_m
    had_airborne = state.had_airborne | airborne
    airborne_hold_streak = jp.where(
        stable_grip & airborne, state.airborne_hold_streak + 1, 0
    )
    positive_lift = stable_grip & (
        physical.object_v_com[..., 2] >= config.minimum_lift_speed_mps
    )
    stalled = stable_grip & ~airborne & ~positive_lift
    stalled_contact_streak = jp.where(
        stalled, state.stalled_contact_streak + 1, 0
    )

    stalled_excess = jp.maximum(
        stalled_contact_streak - config.stalled_grace_frames, 0
    )
    contact_decay = jp.exp(
        -stalled_excess.astype(physical.object_v_com.dtype)
        / float(config.stalled_decay_frames)
    )
    contact_reward = base.contact * contact_decay
    thumb_contact = base.thumb_contact * contact_decay
    stable_float = stable_grip.astype(base.total.dtype)
    opposing_contact = base.opposing_contact * stable_float
    lift_progress = (
        jp.minimum(base.lift_progress, 0.0)
        + jp.maximum(base.lift_progress, 0.0)
        * stable_float
        * config.lift_reward_multiplier
    )
    lift_velocity = (
        jp.minimum(base.lift_velocity, 0.0)
        + jp.maximum(base.lift_velocity, 0.0)
        * stable_float
        * config.lift_reward_multiplier
    )
    hold_growth = 1.0 + config.hold_growth_max_bonus * jp.clip(
        airborne_hold_streak.astype(base.total.dtype)
        / float(config.hold_growth_frames),
        0.0,
        1.0,
    )
    hold_contact = base.hold_contact * stable_float * hold_growth
    stalled_contact = -config.stalled_penalty_max * jp.clip(
        stalled_excess.astype(base.total.dtype)
        / float(config.stalled_decay_frames),
        0.0,
        1.0,
    )
    post_grasp_drop = (
        -config.post_grasp_drop_penalty_max
        * had_airborne.astype(base.total.dtype)
        * jp.clip(
            -physical.object_v_com[..., 2] / config.post_grasp_drop_speed_mps,
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
        + base.contact_loss
        + base.falling
        + base.lateral_slip
        + stalled_contact
        + post_grasp_drop
        + base.action
        + base.survival
    )
    valid = base.valid & jp.isfinite(terms_without_severe)
    total = jp.where(
        valid,
        terms_without_severe + base.severe,
        config.base.severe_penalty,
    )

    landed_after_airborne = had_airborne & (
        actual_clearance <= config.landed_reset_clearance_m
    )
    next_state = RewardV101State(
        jp.where(landed_after_airborne, 0, opposition_streak),
        jp.where(landed_after_airborne, 0, airborne_hold_streak),
        jp.where(landed_after_airborne, 0, stalled_contact_streak),
        jp.where(landed_after_airborne, False, had_airborne),
    )
    reward = V101Reward(
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
        base.contact_loss,
        base.falling,
        base.lateral_slip,
        stalled_contact,
        post_grasp_drop,
        base.action,
        base.survival,
        base.severe,
        base.done,
        base.reason,
        valid,
        base.contact_scale,
        base.reference_motion_speed,
        base.hold_phase,
        opposition_loaded,
        positive_lift,
        stable_grip & airborne,
    )
    return reward, next_state
