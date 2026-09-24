"""Reward v11: faithful port of case/contact-conditioned-autonomy/v5-log-reward.

Nine terms only (the v4 structural layout): object_position, object_rotation,
object_velocity, hand_relative, fingers, geometry, action, survival, severe.
Tracking triple uses the log-curve family with the static rotation override;
coefficients position 0.5, velocity 4.0, severe 75. There is deliberately NO
contact/lift/hold/falling term (docs/manorl_autonomy_v4_reward.md §8).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, NamedTuple

from sim.manorl.autonomy_contracts import (
    V11_REWARD_CONTRACT_ID,
    V11_REWARD_TERM_NAMES,
)
from sim.manorl.autonomy_v4 import (
    _gather,
    anchor_delta_and_velocity,
    motion_gate_weight,
    quat_conj,
    quat_mul,
    quat_unrotate,
    reference_lengths,
    shortest_angle,
)

REWARD_POSITION_COEF = 0.5
REWARD_VELOCITY_COEF = 4.0
REWARD_HAND_RELATIVE_COEF = 0.125
REWARD_GEOMETRY_COEF = 1.2
REWARD_SEVERE_PENALTY = 75.0
POSITION_HALF_M = 0.015
POSITION_HORIZON_M = 0.10
VELOCITY_HALF = 0.5
VELOCITY_HORIZON = 2.5
ROTATION_HALF_DEG = 35.0
ROTATION_HORIZON_DEG = 90.0
ROTATION_MAX = 0.3
ROTATION_MIN = -0.5
STATIC_ROT_ERROR_DEG = 45.0
STATIC_ROT_WEIGHT = 0.75
MOTION_GATE_LOW = 0.01


@dataclass(frozen=True)
class RewardV11Config:
    contract_id: str = V11_REWARD_CONTRACT_ID
    severe_penalty: float = -REWARD_SEVERE_PENALTY

    def __post_init__(self) -> None:
        if self.contract_id != V11_REWARD_CONTRACT_ID:
            raise ValueError("reward v11 contract identifier drifted")
        if not math.isfinite(self.severe_penalty) or self.severe_penalty >= 0:
            raise ValueError("reward v11 severe penalty must be finite and negative")


DEFAULT_REWARD_V11_CONFIG = RewardV11Config()
REWARD_NAMES = V11_REWARD_TERM_NAMES


class V11Reward(NamedTuple):
    total: Any
    object_position: Any
    object_rotation: Any
    object_velocity: Any
    hand_relative: Any
    fingers: Any
    geometry: Any
    action: Any
    survival: Any
    severe: Any
    done: Any
    reason: Any
    valid: Any


def log_fraction(u, half, horizon, *, jp):
    """Normalized log ramp: 0 at u=0, exactly 0.5 at u=half, 1 at u>=horizon."""
    if not (0.0 < half < horizon / 2.0):
        raise ValueError("log curve requires 0 < half < horizon/2")
    s = half * half / (horizon - 2.0 * half)
    return jp.clip(
        jp.log1p(jp.maximum(jp.asarray(u), 0.0) / s) / jp.log1p(horizon / s),
        0.0,
        1.0,
    )


def reward_parameters_v11(
    config: RewardV11Config = DEFAULT_REWARD_V11_CONFIG,
    object_radius: float = 0.04330042412758056,
) -> dict[str, object]:
    return {
        "id": V11_REWARD_CONTRACT_ID,
        "source": "case/contact-conditioned-autonomy/v5-log-reward",
        "terms": list(V11_REWARD_TERM_NAMES),
        "coefficients": {
            "object_position": REWARD_POSITION_COEF,
            "object_velocity": REWARD_VELOCITY_COEF,
            "hand_relative": REWARD_HAND_RELATIVE_COEF,
            "fingers": 0.2,
            "geometry": REWARD_GEOMETRY_COEF,
            "action": -0.002,
            "survival": 0.001,
            "severe_failure": config.severe_penalty,
        },
        "motion_gate": {
            "low_mps": MOTION_GATE_LOW,
            "static_floor": 0.01,
            "object_radius_m": float(object_radius),
        },
        "static_rotation_override": {
            "condition": f"v_eff <= {MOTION_GATE_LOW} and rotation error > {STATIC_ROT_ERROR_DEG} deg",
            "gate_weight": STATIC_ROT_WEIGHT,
        },
        "tracking_curves": {
            "object_position": {"half_m": POSITION_HALF_M, "horizon_m": POSITION_HORIZON_M, "axis_weights": [0.2, 0.2, 0.8]},
            "object_velocity": {"half": VELOCITY_HALF, "horizon": VELOCITY_HORIZON, "linear_norm_mps": 0.25, "angular_norm_radps": 2.0},
            "object_rotation": {"half_deg": ROTATION_HALF_DEG, "horizon_deg": ROTATION_HORIZON_DEG, "value_at_zero": ROTATION_MAX, "floor": ROTATION_MIN},
        },
        "deliberately_absent": [
            "force magnitude", "contact/thumb/opposing", "lift/hold",
            "falling/post_grasp_drop", "slip", "contact_loss", "stalled",
        ],
    }


def compute_reward(
    physical,
    contact,
    cache,
    index,
    executed_action,
    env_ref=None,
    *,
    config: RewardV11Config = DEFAULT_REWARD_V11_CONFIG,
) -> V11Reward:
    """Faithful v5-log-reward: nine-term log-curve reward, no contact layer."""
    import jax.numpy as j

    i = _gather(cache, index, env_ref=env_ref)
    C = j.asarray
    target_p = C(cache.object_origin)[i]
    target_q = C(cache.object_quat_xyzw)[i]
    dp = physical.object_origin - target_p
    pos = REWARD_POSITION_COEF * (
        0.2 * (1.0 - log_fraction(j.abs(dp[:, 0]), POSITION_HALF_M, POSITION_HORIZON_M, jp=j))
        + 0.2 * (1.0 - log_fraction(j.abs(dp[:, 1]), POSITION_HALF_M, POSITION_HORIZON_M, jp=j))
        + 0.8 * (1.0 - log_fraction(j.abs(dp[:, 2]), POSITION_HALF_M, POSITION_HORIZON_M, jp=j))
    )
    deg = shortest_angle(physical.object_quat_xyzw, target_q) * 180 / j.pi
    rot = ROTATION_MIN + (ROTATION_MAX - ROTATION_MIN) * (
        1.0 - log_fraction(deg, ROTATION_HALF_DEG, ROTATION_HORIZON_DEG, jp=j)
    )
    vel = REWARD_VELOCITY_COEF * 0.1 * (
        1.0
        - log_fraction(
            j.sqrt(
                j.sum(((physical.object_v_com - C(cache.object_v_com)[i]) / 0.25) ** 2, axis=-1)
                + j.sum(((physical.object_w - C(cache.object_w)[i]) / 2.0) ** 2, axis=-1)
            ),
            VELOCITY_HALF,
            VELOCITY_HORIZON,
            jp=j,
        )
    )
    rel = quat_unrotate(physical.object_quat_xyzw, physical.palm_origin - physical.object_origin)
    rrel = quat_unrotate(target_q, C(cache.palm_origin)[i] - target_p)
    relq = quat_mul(quat_conj(physical.object_quat_xyzw), physical.palm_quat_xyzw)
    rrelq = quat_mul(quat_conj(target_q), C(cache.palm_quat_xyzw)[i])
    hand = REWARD_HAND_RELATIVE_COEF * j.exp(
        -j.sum(((rel - rrel) / 0.04) ** 2, axis=-1)
        - (shortest_angle(relq, rrelq) / 0.35) ** 2
    )
    fingers = 0.2 * j.exp(
        -j.mean((physical.q_raw[:, 6:] - C(cache.q_feasible)[i][:, 6:]) ** 2, axis=-1) / 0.35**2
    )
    _, _, delta, _ = anchor_delta_and_velocity(
        physical, C(cache.region_anchor_hand)[i], C(cache.region_anchor_object)[i]
    )
    e = delta - C(cache.delta_ref)[i]
    weight = C(cache.proximity)[i] * C(cache.confidence)[i] * C(cache.valid)[i]
    geometry = REWARD_GEOMETRY_COEF * j.sum(
        weight * j.exp(-j.sum(e * e, axis=-1) / 0.01**2), axis=-1
    ) / j.maximum(j.sum(weight, axis=-1), 1.0)
    v_ref = C(cache.object_v_com)[i]
    w_ref = C(cache.object_w)[i]
    v_eff = j.sqrt(
        j.sum(v_ref * v_ref, axis=-1)
        + (j.asarray(cache.object_radius) * j.linalg.norm(w_ref, axis=-1)) ** 2
    )
    w_obj = motion_gate_weight(v_eff)
    w_rot = j.where(
        (v_eff <= MOTION_GATE_LOW) & (deg > STATIC_ROT_ERROR_DEG),
        STATIC_ROT_WEIGHT,
        w_obj,
    )
    pos = w_obj * pos
    rot = w_rot * rot
    vel = w_obj * vel
    action = -0.002 * j.mean(j.clip(executed_action, -1, 1) ** 2, axis=-1)
    survival = j.full_like(pos, 0.001)
    fallen = physical.object_bottom < cache.table_height - 0.05
    deviation = j.linalg.norm(dp, axis=-1) > 0.10
    finite = (
        j.isfinite(pos + rot + vel + hand + fingers + geometry + action)
        & physical.valid
        & contact.valid
    )
    severe = j.where(
        finite,
        j.where(fallen | deviation, config.severe_penalty, 0.0),
        config.severe_penalty,
    )
    reason = (
        (j.asarray(index) >= reference_lengths(cache, env_ref) - 1).astype(j.int32)
        | j.where(deviation, 2, 0)
        | j.where(fallen, 4, 0)
        | j.where(~finite, 8, 0)
    )
    total = j.where(
        finite,
        pos + rot + vel + hand + fingers + geometry + action + survival + severe,
        config.severe_penalty,
    )
    return V11Reward(
        total, pos, rot, vel, hand, fingers, geometry, action, survival,
        severe, reason != 0, reason, finite,
    )
