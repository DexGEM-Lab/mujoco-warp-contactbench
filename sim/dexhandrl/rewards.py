"""Versioned ManoRL-aligned task reward for DexHand021Pro.

The object/reference task is shared with ManoRL ``dev``.  DexHand contributes
only the physical observations needed to evaluate that task (contact bodies and
residual state) plus one explicitly separate fingertip-penetration regularizer.
The module contains no simulator calls, YAML loading, or training-side logging.

``compute_device_reward`` is the JAX device kernel; ``compute_reward_host`` is
its NumPy mirror used for fixed-tensor host/device parity and by diagnostic
paths that must not depend on JAX.  Both return the same :class:`DeviceReward`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, NamedTuple

import numpy as np

# Stable additive column order persisted by PPO/evaluation.  Their sum equals
# the raw environment reward ``total`` for every step.
REWARD_COMPONENTS: Final = (
    "distance_x",
    "distance_y",
    "distance_z",
    "rotation",
    "action_penalty",
    "contact",
    "object_stability",
    "survival",
    "fingertip_penetration",
    "deviation_penalty",
)

PPO_REWARD_SCALE: Final[float] = 0.5

# Stable reason codes shared with current ManoRL telemetry:
# 0 = ongoing, 1 = trajectory complete (success), 2 = deviation failure.
TERMINATION_REASON_NONE: Final[int] = 0
TERMINATION_REASON_SUCCESS: Final[int] = 1
TERMINATION_REASON_DEVIATION: Final[int] = 2

# The contact threshold is a versioned contract value, not a constant. Under
# all_assets@7228b5c / mano_assets@cde03ef, the full residual-off N8 trace had
# 406 positive strict hand↔object force slots: all exceeded 0.2 N, while 5 N
# retained only 88.2%. Thus 0.2 N removes the near-zero numerical tail without
# discarding observed physical contacts. The contract ID encodes the value so
# later asset/force evidence requires an explicit versioned change.
REWARD_CONTRACT_ID_PREFIX: Final = "dexhand021pro_shared_task_early{early}_contact{threshold:g}n_late{grace}_penalty3x_penetration{penetration:g}_v1"
PPO_REWARD_CONTRACT_ID_PREFIX: Final = "dexhand021pro_shared_task_early{early}_contact{threshold:g}n_late{grace}_penalty3x_penetration{penetration:g}_ppo_scale{scale:g}_v1"
THRESHOLD_STATUS: Final = "calibrated_n8_decomposed_assets_7228b5c"


@dataclass(frozen=True)
class DexHandRewardConfig:
    """Shared ManoRL task equations plus the explicit DexHand regularizer.

    ``early_phase_steps`` is the DexHand pure-mocap phase from its task YAML
    (130).  ManoRL ``dev`` uses 30; the phase *semantics* are identical (only
    the action penalty is live) while the phase length is embodiment-specific.
    """

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
    direct_contact_reward_scale: float = 1.0
    contact_force_threshold_N: float = 0.2
    late_contact_grace_frames: int = 10
    late_contact_penalty_multiplier: float = 3.0
    max_object_stability_reward: float = 0.4
    object_stability_reference_speed_m_s: float = 0.1
    survival_reward: float = 0.001
    early_phase_steps: int = 130
    max_deviation_distance_m: float = 0.10
    deviation_penalty: float = 0.0
    fingertip_penetration_weight: float = 1.5

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, tuple):
                if not value or not all(np.isfinite(item) for item in value):
                    raise ValueError(f"{name} must contain finite values")
            elif isinstance(value, int):
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
            elif not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.contact_force_threshold_N < 0:
            raise ValueError("contact_force_threshold_N must be non-negative")
        if self.max_deviation_distance_m < 0 or self.deviation_penalty < 0:
            raise ValueError("deviation distance and penalty must be non-negative")
        if self.object_stability_reference_speed_m_s < 0:
            raise ValueError("object stability reference speed must be non-negative")
        if self.fingertip_penetration_weight < 0:
            raise ValueError("fingertip penetration weight must be non-negative")

    @property
    def contract_id(self) -> str:
        return REWARD_CONTRACT_ID_PREFIX.format(
            early=self.early_phase_steps,
            threshold=self.contact_force_threshold_N,
            grace=self.late_contact_grace_frames,
            penetration=self.fingertip_penetration_weight,
        )

    @property
    def ppo_contract_id(self) -> str:
        return PPO_REWARD_CONTRACT_ID_PREFIX.format(
            early=self.early_phase_steps,
            threshold=self.contact_force_threshold_N,
            grace=self.late_contact_grace_frames,
            penetration=self.fingertip_penetration_weight,
            scale=PPO_REWARD_SCALE,
        )

    def record(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "ppo_contract_id": self.ppo_contract_id,
            "ppo_reward_scale": PPO_REWARD_SCALE,
            "component_order": list(REWARD_COMPONENTS),
            "threshold_status": THRESHOLD_STATUS,
            "config": asdict(self),
            "shared_task_source": "ManoRL dev source-aligned reward equations (v1 semantics)",
            "early_phase_note": "DexHand 130-frame pure-mocap phase; dev uses 30. Phase semantics identical.",
            "phase_window_semantics": "trajectory object-motion window proxy; no measured contact interval is present in the retarget Lance rows",
            "threshold_evidence": {
                "asset_revision": "all_assets@7228b5cf / mano_assets@cde03ef",
                "evaluation": "residual-off cube1 N8 primary worlds, full active horizons",
                "positive_force_slots": 406,
                "minimum_positive_force_N": 0.5646176338195801,
                "fraction_above_0p2N": 1.0,
                "fraction_above_5N": 0.8817733990147784,
            },
            "embodiment_adapter": {
                "contact_slots": "16 DexHand021Pro bodies in source order",
                "residual_state": "3D base offset + 16 active finger offsets",
                "fingertip_penetration": "separate DexHand regularizer, zeroed in early phase",
            },
        }


SOURCE_ALIGNED_REWARD_CONFIG: Final = DexHandRewardConfig()


class DeviceReward(NamedTuple):
    """Complete device reward and diagnostics in stable field order."""

    total: Any
    task_total: Any
    embodiment_total: Any
    components: Any
    distance_x: Any
    distance_y: Any
    distance_z: Any
    ungated_distance_x: Any
    ungated_distance_y: Any
    ungated_distance_z: Any
    rotation: Any
    rotation_error_deg: Any
    position_penalty: Any
    joint_penalty: Any
    action_penalty: Any
    raw_contact: Any
    contact: Any
    distance_gate: Any
    object_stability: Any
    object_speed: Any
    survival: Any
    fingertip_penetration: Any
    early_phase: Any
    deviation_penalty: Any
    terminated: Any
    truncated: Any
    reason: Any
    valid: Any


def _float32_threshold(value: float, array_dtype: Any) -> Any:
    """Strict float32 comparison gate used by ManoRL's device path.

    Warp contact forces are float32.  A Python 0.2 rounds *up* to the nearest
    float32 (0.20000000298...), which would silently reject a force that the
    float64 host reward counts.  Matching ManoRL, the gate is the next float32
    below the declared threshold so both paths agree outside a one-ULP band.
    """
    return np.nextafter(np.asarray(value, dtype=np.float32), -np.inf)


def _require_batch(jp: Any, value: Any, batch: int, trailing: tuple[int, ...], name: str) -> Any:
    value = jp.asarray(value)
    if value.shape != (batch, *trailing):
        raise ValueError(f"{name} must have shape {(batch, *trailing)}")
    return value


def _shared_terms(
    *, frame: Any, starts: Any, ends: Any, lengths: Any, obj: Any, target: Any,
    quat: Any, target_quat: Any, velocity: Any, force: Any, expected: Any,
    base: Any, joint: Any, active: Any, tip_penalty: Any, rotation_disabled: Any,
    config: DexHandRewardConfig, jp: Any, threshold_gate: Any,
) -> dict[str, Any]:
    """Single shared body for the JAX and NumPy reward computations."""
    distance = jp.abs(obj - target)
    ungated = jp.asarray(config.distance_scales, dtype=obj.dtype) * jp.exp(
        -config.distance_decay * distance
    )

    ax, ay, az, aw = (quat[:, index] for index in range(4))
    bx, by, bz, bw = (
        -target_quat[:, 0],
        -target_quat[:, 1],
        -target_quat[:, 2],
        target_quat[:, 3],
    )
    product_xyz = jp.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=1,
    )
    degrees = 2.0 * jp.arcsin(jp.minimum(jp.linalg.norm(product_xyz, axis=1), 1.0))
    degrees = degrees * 180.0 / 3.14159265359
    segment_1 = config.rotation_segment_1_coeff * degrees**2 + 1.0
    offset = degrees - config.rotation_segment_1_threshold_deg
    segment_2 = (
        config.rotation_segment_2_coeff * offset**2
        + config.rotation_segment_2_linear_coeff * offset
        + config.rotation_segment_2_constant
    )
    rotation_value = jp.where(
        degrees <= config.rotation_segment_1_threshold_deg,
        segment_1,
        jp.where(
            degrees <= config.rotation_segment_2_threshold_deg,
            segment_2,
            config.rotation_segment_3_value,
        ),
    )
    rotation = jp.where(
        rotation_disabled.astype(bool),
        0.0,
        config.rotation_scale * rotation_value + config.rotation_base_penalty,
    )

    position_base = jp.sum(jp.abs(base * config.position_penalty_scale), axis=1)
    joint_base = jp.sum(
        jp.where(active.astype(bool), jp.abs(joint * config.joint_penalty_scale), 0.0),
        axis=1,
    )
    joint_base = joint_base / jp.maximum(active.sum(axis=1), 1.0) * config.reference_joint_count
    position_penalty = (
        -config.action_penalty_scale * config.position_penalty_weight * position_base
    )
    joint_penalty = -config.action_penalty_scale * config.joint_penalty_weight * joint_base
    action_penalty = position_penalty + joint_penalty

    magnitudes = jp.linalg.norm(force, axis=-1)
    expected_count = jp.sum(expected, axis=1)
    correct = jp.sum((magnitudes > threshold_gate) * (expected > 0.5), axis=1)
    raw_contact = jp.where(
        expected_count > 0,
        correct / jp.maximum(expected_count, 1.0) * config.max_contact_reward,
        0.0,
    )
    valid_window = ends >= starts
    within = valid_window & (frame >= starts) & (frame <= ends)
    any_contact = jp.any(magnitudes > threshold_gate, axis=1)
    late_contact = valid_window & (frame > ends + config.late_contact_grace_frames) & any_contact
    contact = jp.where(
        late_contact,
        -config.late_contact_penalty_multiplier * config.max_contact_reward,
        jp.where(within, raw_contact, 0.0),
    ) * config.direct_contact_reward_scale
    distance_gate = jp.where(within, raw_contact, 0.0)
    distance_gate = jp.where(valid_window & (frame > ends), config.max_contact_reward, distance_gate)
    distance_terms = ungated * distance_gate[:, None]

    speed = jp.linalg.norm(velocity, axis=1)
    reference_speed = jp.asarray(config.object_stability_reference_speed_m_s, dtype=obj.dtype)
    safe_reference_speed = jp.where(reference_speed > 0, reference_speed, 1.0)
    stability_base = config.max_object_stability_reward * jp.exp(-(speed / safe_reference_speed) ** 2)
    stability_base = jp.where(reference_speed > 0, stability_base, 0.0)
    stability = jp.where(valid_window & (frame > ends), stability_base, 0.0)
    survival = jp.full((obj.shape[0],), config.survival_reward, dtype=obj.dtype)

    early = frame < config.early_phase_steps
    non_early_task = (
        distance_terms.sum(axis=1)
        + rotation
        + action_penalty
        + contact
        + stability
        + survival
    )
    task_total = jp.where(early, action_penalty, non_early_task)
    fingertip_penetration = jp.where(early, 0.0, config.fingertip_penetration_weight * tip_penalty)

    error = jp.linalg.norm(obj - target, axis=1)
    terminated = (~early) & (error > config.max_deviation_distance_m)
    truncated = frame >= lengths - 1
    deviation_penalty = jp.where(terminated, -config.deviation_penalty, 0.0)
    reason = jp.where(
        terminated,
        TERMINATION_REASON_DEVIATION,
        jp.where(truncated, TERMINATION_REASON_SUCCESS, TERMINATION_REASON_NONE),
    ).astype(jp.int32)
    embodiment_total = fingertip_penetration
    total = task_total + embodiment_total + deviation_penalty

    early_float = early.astype(obj.dtype)
    components = jp.stack(
        (
            jp.where(early, 0.0, distance_terms[:, 0]),
            jp.where(early, 0.0, distance_terms[:, 1]),
            jp.where(early, 0.0, distance_terms[:, 2]),
            jp.where(early, 0.0, rotation),
            action_penalty,
            jp.where(early, 0.0, contact),
            jp.where(early, 0.0, stability),
            jp.where(early, 0.0, survival),
            fingertip_penetration,
            deviation_penalty,
        ),
        axis=1,
    )
    return {
        "total": total, "task_total": task_total, "embodiment_total": embodiment_total,
        "components": components,
        "distance_x": jp.where(early, 0.0, distance_terms[:, 0]),
        "distance_y": jp.where(early, 0.0, distance_terms[:, 1]),
        "distance_z": jp.where(early, 0.0, distance_terms[:, 2]),
        "ungated_x": ungated[:, 0], "ungated_y": ungated[:, 1], "ungated_z": ungated[:, 2],
        "rotation": jp.where(early, 0.0, rotation),
        "rotation_error_deg": degrees,
        "position_penalty": position_penalty, "joint_penalty": joint_penalty,
        "action_penalty": action_penalty,
        "raw_contact": raw_contact,
        "contact": jp.where(early, 0.0, contact),
        "distance_gate": distance_gate,
        "object_stability": jp.where(early, 0.0, stability),
        "object_speed": speed,
        "survival": jp.where(early, 0.0, survival),
        "fingertip_penetration": fingertip_penetration,
        "early_phase": early,
        "deviation_penalty": deviation_penalty,
        "terminated": terminated, "truncated": truncated, "reason": reason,
        "early_float": early_float,
    }


def compute_device_reward(
    *,
    frame: Any,
    motion_window_start_frame: Any,
    motion_window_end_frame: Any,
    lengths: Any,
    object_position: Any,
    target_position: Any,
    object_quat_xyzw: Any,
    target_quat_xyzw: Any,
    object_linear_velocity: Any,
    hand_contact_force: Any,
    expected_contact_mask: Any,
    base_offset: Any,
    joint_offset: Any,
    active_joint_mask: Any,
    fingertip_inside_penalty: Any,
    rotation_disabled: Any,
    config: DexHandRewardConfig = SOURCE_ALIGNED_REWARD_CONFIG,
) -> DeviceReward:
    """Evaluate the ManoRL task equations on batched JAX arrays.

    ``motion_window_start/end_frame`` are object-motion boundaries retained by
    the retarget dataset. They proxy the shared reward phase because these rows
    contain no measured contact interval. ``fingertip_inside_penalty`` is a
    non-positive raw DexHand penalty, exposed separately from the task total.
    """

    import jax.numpy as jp

    frame = jp.asarray(frame)
    starts = jp.asarray(motion_window_start_frame)
    ends = jp.asarray(motion_window_end_frame)
    lengths = jp.asarray(lengths)
    batch = frame.shape[0]
    if any(value.shape != (batch,) for value in (starts, ends, lengths)):
        raise ValueError("reward phase vectors must be (batch,)")

    obj = _require_batch(jp, object_position, batch, (3,), "object_position")
    target = _require_batch(jp, target_position, batch, (3,), "target_position")
    quat = _require_batch(jp, object_quat_xyzw, batch, (4,), "object_quat_xyzw")
    target_quat = _require_batch(jp, target_quat_xyzw, batch, (4,), "target_quat_xyzw")
    velocity = _require_batch(jp, object_linear_velocity, batch, (3,), "object_linear_velocity")
    force = _require_batch(jp, hand_contact_force, batch, (16, 3), "hand_contact_force")
    expected = _require_batch(jp, expected_contact_mask, batch, (16,), "expected_contact_mask")
    base = _require_batch(jp, base_offset, batch, (3,), "base_offset")
    joint = _require_batch(jp, joint_offset, batch, (16,), "joint_offset")
    active = _require_batch(jp, active_joint_mask, batch, (16,), "active_joint_mask")
    tip_penalty = jp.asarray(fingertip_inside_penalty)
    rotation_disabled = jp.asarray(rotation_disabled)
    if tip_penalty.shape != (batch,) or rotation_disabled.shape != (batch,):
        raise ValueError("penetration/rotation masks must be (batch,)")
    gate = jp.asarray(_float32_threshold(config.contact_force_threshold_N, obj.dtype), dtype=obj.dtype)

    terms = _shared_terms(
        frame=frame, starts=starts, ends=ends, lengths=lengths, obj=obj, target=target,
        quat=quat, target_quat=target_quat, velocity=velocity, force=force, expected=expected,
        base=base, joint=joint, active=active, tip_penalty=tip_penalty,
        rotation_disabled=rotation_disabled, config=config, jp=jp, threshold_gate=gate,
    )
    config_values = []
    for value in asdict(config).values():
        config_values.extend(value if isinstance(value, tuple) else (value,))
    valid = (
        jp.all(jp.isfinite(obj))
        & jp.all(jp.isfinite(target))
        & jp.all(jp.isfinite(quat))
        & jp.all(jp.isfinite(target_quat))
        & jp.all(jp.isfinite(velocity))
        & jp.all(jp.isfinite(force))
        & jp.all(jp.isfinite(base))
        & jp.all(jp.isfinite(joint))
        & jp.all((expected == 0) | (expected == 1))
        & jp.all(jp.isfinite(terms["total"]))
        & jp.all(jp.isfinite(terms["components"]))
        & jp.asarray(all(np.isfinite(config_values)))
    )
    return DeviceReward(
        total=terms["total"],
        task_total=terms["task_total"],
        embodiment_total=terms["embodiment_total"],
        components=terms["components"],
        distance_x=terms["distance_x"],
        distance_y=terms["distance_y"],
        distance_z=terms["distance_z"],
        ungated_distance_x=terms["ungated_x"],
        ungated_distance_y=terms["ungated_y"],
        ungated_distance_z=terms["ungated_z"],
        rotation=terms["rotation"],
        rotation_error_deg=terms["rotation_error_deg"],
        position_penalty=terms["position_penalty"],
        joint_penalty=terms["joint_penalty"],
        action_penalty=terms["action_penalty"],
        raw_contact=terms["raw_contact"],
        contact=terms["contact"],
        distance_gate=terms["distance_gate"],
        object_stability=terms["object_stability"],
        object_speed=terms["object_speed"],
        survival=terms["survival"],
        fingertip_penetration=terms["fingertip_penetration"],
        early_phase=terms["early_phase"],
        deviation_penalty=terms["deviation_penalty"],
        terminated=terms["terminated"],
        truncated=terms["truncated"],
        reason=terms["reason"],
        valid=valid,
    )


def compute_reward_host(
    *,
    frame: Any,
    motion_window_start_frame: Any,
    motion_window_end_frame: Any,
    lengths: Any,
    object_position: Any,
    target_position: Any,
    object_quat_xyzw: Any,
    target_quat_xyzw: Any,
    object_linear_velocity: Any,
    hand_contact_force: Any,
    expected_contact_mask: Any,
    base_offset: Any,
    joint_offset: Any,
    active_joint_mask: Any,
    fingertip_inside_penalty: Any,
    rotation_disabled: Any,
    config: DexHandRewardConfig = SOURCE_ALIGNED_REWARD_CONFIG,
) -> DeviceReward:
    """NumPy mirror of :func:`compute_device_reward` for host/device parity."""

    frame = np.asarray(frame)
    starts = np.asarray(motion_window_start_frame)
    ends = np.asarray(motion_window_end_frame)
    lengths = np.asarray(lengths)
    batch = frame.shape[0]
    if any(value.shape != (batch,) for value in (starts, ends, lengths)):
        raise ValueError("reward phase vectors must be (batch,)")
    arrays = {
        "object_position": np.asarray(object_position, dtype=np.float32),
        "target_position": np.asarray(target_position, dtype=np.float32),
        "object_quat_xyzw": np.asarray(object_quat_xyzw, dtype=np.float32),
        "target_quat_xyzw": np.asarray(target_quat_xyzw, dtype=np.float32),
        "object_linear_velocity": np.asarray(object_linear_velocity, dtype=np.float32),
        "hand_contact_force": np.asarray(hand_contact_force, dtype=np.float32),
        "expected_contact_mask": np.asarray(expected_contact_mask, dtype=np.float32),
        "base_offset": np.asarray(base_offset, dtype=np.float32),
        "joint_offset": np.asarray(joint_offset, dtype=np.float32),
        "active_joint_mask": np.asarray(active_joint_mask, dtype=np.float32),
        "fingertip_inside_penalty": np.asarray(fingertip_inside_penalty, dtype=np.float32),
        "rotation_disabled": np.asarray(rotation_disabled, dtype=bool),
    }
    expected_shapes = {
        "object_position": (batch, 3), "target_position": (batch, 3),
        "object_quat_xyzw": (batch, 4), "target_quat_xyzw": (batch, 4),
        "object_linear_velocity": (batch, 3), "hand_contact_force": (batch, 16, 3),
        "expected_contact_mask": (batch, 16), "base_offset": (batch, 3),
        "joint_offset": (batch, 16), "active_joint_mask": (batch, 16),
        "fingertip_inside_penalty": (batch,), "rotation_disabled": (batch,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    gate = _float32_threshold(config.contact_force_threshold_N, np.float32)
    terms = _shared_terms(
        frame=frame, starts=starts, ends=ends, lengths=lengths,
        obj=arrays["object_position"], target=arrays["target_position"],
        quat=arrays["object_quat_xyzw"], target_quat=arrays["target_quat_xyzw"],
        velocity=arrays["object_linear_velocity"], force=arrays["hand_contact_force"],
        expected=arrays["expected_contact_mask"], base=arrays["base_offset"],
        joint=arrays["joint_offset"], active=arrays["active_joint_mask"],
        tip_penalty=arrays["fingertip_inside_penalty"],
        rotation_disabled=arrays["rotation_disabled"], config=config, jp=np, threshold_gate=gate,
    )
    config_values = []
    for value in asdict(config).values():
        config_values.extend(value if isinstance(value, tuple) else (value,))
    valid = bool(
        np.all(np.isfinite(arrays["object_position"]))
        and np.all(np.isfinite(arrays["target_position"]))
        and np.all(np.isfinite(arrays["object_quat_xyzw"]))
        and np.all(np.isfinite(arrays["target_quat_xyzw"]))
        and np.all(np.isfinite(arrays["object_linear_velocity"]))
        and np.all(np.isfinite(arrays["hand_contact_force"]))
        and np.all(np.isfinite(arrays["base_offset"]))
        and np.all(np.isfinite(arrays["joint_offset"]))
        and np.all((arrays["expected_contact_mask"] == 0) | (arrays["expected_contact_mask"] == 1))
        and np.all(np.isfinite(terms["total"]))
        and np.all(np.isfinite(terms["components"]))
        and all(np.isfinite(value) for value in config_values)
    )
    return DeviceReward(
        total=terms["total"],
        task_total=terms["task_total"],
        embodiment_total=terms["embodiment_total"],
        components=terms["components"],
        distance_x=terms["distance_x"],
        distance_y=terms["distance_y"],
        distance_z=terms["distance_z"],
        ungated_distance_x=terms["ungated_x"],
        ungated_distance_y=terms["ungated_y"],
        ungated_distance_z=terms["ungated_z"],
        rotation=terms["rotation"],
        rotation_error_deg=terms["rotation_error_deg"],
        position_penalty=terms["position_penalty"],
        joint_penalty=terms["joint_penalty"],
        action_penalty=terms["action_penalty"],
        raw_contact=terms["raw_contact"],
        contact=terms["contact"],
        distance_gate=terms["distance_gate"],
        object_stability=terms["object_stability"],
        object_speed=terms["object_speed"],
        survival=terms["survival"],
        fingertip_penetration=terms["fingertip_penetration"],
        early_phase=terms["early_phase"],
        deviation_penalty=terms["deviation_penalty"],
        terminated=terms["terminated"],
        truncated=terms["truncated"],
        reason=terms["reason"],
        valid=valid,
    )
