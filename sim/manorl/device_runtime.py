"""JAX primitives for the opt-in ManoRL device transition.

These functions consume the compact portions of an MJX state that the ManoRL
ABI actually reads. They deliberately do not materialize a ``Data`` object,
capacity-sized geometry diagnostics, or a full ``qpos/qvel/xpos/xquat`` host
snapshot.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import numpy as np

from sim.manorl.contracts import KEYPOINT_NAMES


class DeviceContactReduction(NamedTuple):
    """Reduced contact features plus a device-resident fail-closed status."""

    keypoint_forces: Any
    hand_object_forces: Any
    per_world_count: Any
    valid: Any


class DeviceTermination(NamedTuple):
    """Device equivalent of the termination ABI fields plus fail-closed validity."""

    reset: Any
    deviation_reset: Any
    deviation_penalty: Any
    reason_code: Any
    valid: Any


class DeviceReward(NamedTuple):
    """Compact source-order reward diagnostics plus fail-closed validity.

    These fields are the complete host ``RewardDiagnostics`` contract.  They
    are produced while the reward kernel is live, so training telemetry need
    not invent values or reconstruct a physical snapshot on the host.
    """

    total: Any
    distance_x: Any
    distance_y: Any
    distance_z: Any
    ungated_distance_x: Any
    ungated_distance_y: Any
    ungated_distance_z: Any
    rotation: Any
    position_penalty: Any
    joint_penalty: Any
    action_penalty: Any
    raw_contact: Any
    contact: Any
    distance_gate: Any
    object_stability: Any
    object_speed: Any
    survival: Any
    early_phase: Any
    deviation_penalty: Any
    valid: Any


class DeviceTaskCounters(NamedTuple):
    """Per-world transition state after the host-compatible commit order."""

    progress: Any
    trajectory_steps: Any
    episode_returns: Any
    reset_mask: Any
    control_call: Any


class DeviceTransitionBatch(NamedTuple):
    """The training-only JAX CUDA egress from one device transition.

    Policy tensors remain JAX arrays until the skrl boundary borrows them with
    CUDA DLPack. Host reward/termination telemetry is intentionally owned by
    the environment and is therefore not represented here.
    """

    observation: Any
    reward: Any
    reset: Any
    reason_code: Any
    deviation_reset: Any
    valid: Any


class DevicePhysicalFeatures(NamedTuple):
    """The source-order state fields consumed by observation/reward code."""

    mano_dof_pos: Any
    hand_position: Any
    hand_orientation_xyzw: Any
    hand_keypoint_orientations_xyzw: Any
    object_position: Any
    object_orientation_xyzw: Any
    object_linear_velocity: Any
    hand_keypoint_positions: Any
    fingertip_positions: Any
    valid: Any


def _device_ordinal(device: Any) -> int | None:
    """Return a runtime device ordinal when its backend exposes one."""

    for attribute in ("local_hardware_id", "id", "index"):
        value = getattr(device, attribute, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _is_nvidia_jax_cuda_device(device: Any) -> bool:
    """Recognize CUDA-backed JAX GPU devices without accepting ROCm."""

    if getattr(device, "platform", None) not in {"gpu", "cuda"}:
        return False
    device_kind = getattr(device, "device_kind", None)
    return device_kind is None or "nvidia" in str(device_kind).casefold()


def _require_matching_cuda_devices(
    *, jax_device: Any, torch_device: Any, direction: str, error_type: type[Exception]
) -> None:
    if not _is_nvidia_jax_cuda_device(jax_device):
        raise error_type(f"{direction} requires an NVIDIA CUDA JAX device")
    jax_ordinal = _device_ordinal(jax_device)
    torch_ordinal = _device_ordinal(torch_device)
    if jax_ordinal is not None and torch_ordinal is not None and jax_ordinal != torch_ordinal:
        raise error_type(
            f"{direction} changed CUDA device ordinal from {torch_ordinal} to {jax_ordinal}"
        )


def torch_to_jax_cuda(tensor: Any) -> Any:
    """Borrow a CUDA Torch tensor in JAX through same-device DLPack only."""

    import jax

    if not getattr(tensor, "is_cuda", False):
        raise ValueError("device runtime accepts CUDA Torch tensors only")
    array = jax.dlpack.from_dlpack(tensor)
    _require_matching_cuda_devices(
        jax_device=array.device,
        torch_device=tensor.device,
        direction="Torch-to-JAX DLPack conversion",
        error_type=RuntimeError,
    )
    return array


def jax_to_torch_cuda(array: Any) -> Any:
    """Borrow a CUDA JAX array in Torch through same-device DLPack only."""

    import torch

    if not _is_nvidia_jax_cuda_device(array.device):
        raise ValueError("device runtime returns NVIDIA CUDA JAX arrays only")
    tensor = torch.from_dlpack(array)
    if not tensor.is_cuda:
        raise RuntimeError("JAX-to-Torch DLPack conversion did not retain CUDA placement")
    _require_matching_cuda_devices(
        jax_device=array.device,
        torch_device=tensor.device,
        direction="JAX-to-Torch DLPack conversion",
        error_type=RuntimeError,
    )
    return tensor


def _normalize_wxyz_to_xyzw(quaternion: Any) -> tuple[Any, Any]:
    import jax.numpy as jp

    xyzw = quaternion[..., (1, 2, 3, 0)]
    norm = jp.linalg.norm(xyzw, axis=-1, keepdims=True)
    valid = jp.all(jp.isfinite(xyzw)) & jp.all(norm > 1e-12)
    return xyzw / jp.maximum(norm, 1e-12), valid


def _quat_rotate_xyzw(quaternion: Any, vectors: Any) -> Any:
    import jax.numpy as jp

    q_vec, q_w = quaternion[..., :3], quaternion[..., 3:4]
    return (
        vectors * (2.0 * q_w * q_w - 1.0)
        + jp.cross(q_vec, vectors) * q_w * 2.0
        + q_vec * jp.sum(q_vec * vectors, axis=-1, keepdims=True) * 2.0
    )


def extract_mjx_physical_features(
    *,
    qpos: Any,
    qvel: Any,
    xpos: Any,
    xquat: Any,
    hand_qpos_start: int,
    hand_dof: int,
    object_body_id: int,
    object_qvel_address: int,
    keypoint_body_ids: Sequence[int],
    fingertip_keypoint_ids: Sequence[int],
    fingertip_local_offsets: Any,
) -> DevicePhysicalFeatures:
    """Gather precisely the single-hand physical fields needed by ManoRL.

    The IDs are compile-time model metadata. The function is JIT-safe and
    performs no state-sized copies: its outputs are the 28D hand, one object,
    sixteen keypoints, five fingertips, and one validity scalar.
    """

    import jax.numpy as jp

    ids = np.asarray(keypoint_body_ids, dtype=np.int32)
    tips = np.asarray(fingertip_keypoint_ids, dtype=np.int32)
    offsets = jp.asarray(fingertip_local_offsets)
    if ids.shape != (len(KEYPOINT_NAMES),) or len(set(ids.tolist())) != len(KEYPOINT_NAMES):
        raise ValueError("one distinct source-order body is required per keypoint")
    if tips.shape != (5,) or np.any(tips < 0) or np.any(tips >= len(KEYPOINT_NAMES)):
        raise ValueError("fingertip_keypoint_ids must contain five valid source indices")
    if offsets.shape != (5, 3):
        raise ValueError("fingertip_local_offsets must have shape (5, 3)")
    if hand_dof != 28:
        raise ValueError("device physical extraction supports the exact 28-DoF layout only")
    qpos, qvel, xpos, xquat = map(jp.asarray, (qpos, qvel, xpos, xquat))
    if qpos.ndim != 2 or qvel.ndim != 2 or xpos.ndim != 3 or xquat.ndim != 3:
        raise ValueError("MJX state must be batched qpos/qvel/xpos/xquat arrays")
    batch = qpos.shape[0]
    if qvel.shape[0] != batch or xpos.shape[:2] != xquat.shape[:2] or xpos.shape[0] != batch:
        raise ValueError("MJX state batch dimensions must agree")
    if not 0 <= object_body_id < xpos.shape[1] or object_qvel_address < 0:
        raise ValueError("object body/qvel metadata is outside MJX state")
    if hand_qpos_start < 0 or hand_qpos_start + hand_dof > qpos.shape[1] or object_qvel_address + 3 > qvel.shape[1]:
        raise ValueError("hand/object state metadata is outside MJX state")
    keypoints = xpos[:, ids]
    keypoint_quat, keypoint_valid = _normalize_wxyz_to_xyzw(xquat[:, ids])
    object_quat, object_valid = _normalize_wxyz_to_xyzw(xquat[:, object_body_id])
    hand_quat = keypoint_quat[:, 0]
    tips_world = keypoints[:, tips] + _quat_rotate_xyzw(keypoint_quat[:, tips], offsets[None])
    features = DevicePhysicalFeatures(
        qpos[:, hand_qpos_start : hand_qpos_start + hand_dof],
        keypoints[:, 0],
        hand_quat,
        keypoint_quat,
        xpos[:, object_body_id],
        object_quat,
        qvel[:, object_qvel_address : object_qvel_address + 3],
        keypoints,
        tips_world,
        jp.all(keypoint_valid) & object_valid & jp.all(jp.isfinite(qpos)) & jp.all(jp.isfinite(qvel)),
    )
    return features


def build_device_observation_28(
    *,
    physical: DevicePhysicalFeatures,
    hand_keypoint_contact_forces: Any,
    target_object_position: Any,
    target_object_orientation_xyzw: Any,
    target_object_pos_next_5: Any,
    cumulative_offset: Any,
    cumulative_joint_offset: Any,
    point_cloud_local: Any,
    point_cloud_scale: Any,
    object_geometry: Any,
    expected_contact_mask: Any,
    action_ids: Any,
    object_support_points: Any,
    table_surface_height: float,
    mano_dof_lower: Any,
    mano_dof_upper: Any,
) -> tuple[Any, Any]:
    """Build the exact 480D 28-DoF source layout and its policy-clipped view.

    All variable inputs are batched. ``point_cloud_local`` supports the static
    normalized template by broadcasting ``(64, 3)`` and reset templates as
    ``(batch, 64, 3)``; support points are already deterministically reduced
    and padded by the owning environment.
    """

    import jax.numpy as jp

    batch = physical.mano_dof_pos.shape[0]
    def array(value: Any, trailing: tuple[int, ...], name: str) -> Any:
        value = jp.asarray(value)
        if value.shape != (batch, *trailing):
            raise ValueError(f"{name} must have shape {(batch, *trailing)}")
        return value
    force = array(hand_keypoint_contact_forces, (16, 3), "hand_keypoint_contact_forces")
    target_pos = array(target_object_position, (3,), "target_object_position")
    target_quat = array(target_object_orientation_xyzw, (4,), "target_object_orientation_xyzw")
    target_next = array(target_object_pos_next_5, (3,), "target_object_pos_next_5")
    cumulative_offset = array(cumulative_offset, (3,), "cumulative_offset")
    cumulative_joint_offset = array(cumulative_joint_offset, (22,), "cumulative_joint_offset")
    geometry = array(object_geometry, (12,), "object_geometry")
    expected = array(expected_contact_mask, (16,), "expected_contact_mask")
    support = jp.asarray(object_support_points)
    if support.ndim == 2:
        support = jp.broadcast_to(support, (batch, *support.shape))
    if support.ndim != 3 or support.shape[0] != batch or support.shape[2] != 3 or support.shape[1] < 1:
        raise ValueError("object_support_points must be (points, 3) or (batch, points, 3)")
    points = jp.asarray(point_cloud_local)
    if points.shape == (64, 3):
        points = jp.broadcast_to(points, (batch, 64, 3))
    if points.shape != (batch, 64, 3):
        raise ValueError("point_cloud_local must be (64, 3) or (batch, 64, 3)")
    scale = jp.asarray(point_cloud_scale)
    if scale.shape == (3,):
        scale = jp.broadcast_to(scale, (batch, 3))
    scale = array(scale, (3,), "point_cloud_scale")
    lower, upper = jp.asarray(mano_dof_lower), jp.asarray(mano_dof_upper)
    if lower.shape != (28,) or upper.shape != (28,):
        raise ValueError("mano_dof limits must be 28-wide")
    actions = jp.asarray(action_ids)
    if actions.shape != (batch,):
        raise ValueError("action_ids must be batched")
    object_quat = physical.object_orientation_xyzw
    target_support_quat = target_quat / jp.maximum(jp.linalg.norm(target_quat, axis=-1, keepdims=True), 1e-12)
    finger = 2.0 * (physical.mano_dof_pos[:, 6:] - lower[None, 6:]) / (upper[None, 6:] - lower[None, 6:]) - 1.0
    magnitude = jp.linalg.norm(force, axis=-1)
    direction = force / jp.maximum(magnitude[..., None], 1e-6)
    # NumPy promotes Warp float32 forces before comparing against the Python
    # 0.2 threshold. Match that strict comparison in float32 JAX: its nearest
    # representable 0.2 is slightly above the real-valued source threshold.
    contact_gate = jp.asarray(np.nextafter(np.float32(0.2), -np.inf), dtype=force.dtype)
    direction = direction * (magnitude > contact_gate)[..., None]
    cloud_world = _quat_rotate_xyzw(object_quat[:, None], points * scale[:, None]) + physical.object_position[:, None]
    cloud = (cloud_world - physical.hand_position[:, None]) / scale[:, None]
    object_min_z = jp.min(_quat_rotate_xyzw(object_quat[:, None], support)[..., 2] + physical.object_position[:, None, 2], axis=1)
    target_min_z = jp.min(_quat_rotate_xyzw(target_support_quat[:, None], support)[..., 2] + target_pos[:, None, 2], axis=1)
    clearance = jp.stack((
        object_min_z - table_surface_height,
        physical.object_position[:, 2] - table_surface_height,
        jp.min(physical.hand_keypoint_positions[..., 2], axis=1) - table_surface_height,
        jp.min(physical.fingertip_positions[..., 2], axis=1) - table_surface_height,
        target_min_z - table_surface_height,
        target_pos[:, 2] - table_surface_height,
    ), axis=1)
    one_hot = jp.eye(50, dtype=physical.mano_dof_pos.dtype)[jp.clip(actions.astype(jp.int32) - 1, 0, 49)]
    raw = jp.concatenate((
        physical.mano_dof_pos[:, :6], finger, physical.hand_orientation_xyzw,
        physical.object_position, object_quat, physical.hand_position,
        (physical.fingertip_positions - physical.object_position[:, None]).reshape(batch, -1),
        target_pos, target_quat, cumulative_offset, target_next, clearance,
        cloud.reshape(batch, -1), one_hot, geometry,
        (physical.hand_keypoint_positions - physical.object_position[:, None]).reshape(batch, -1),
        jp.tanh(magnitude * 0.025), cumulative_joint_offset,
        direction.reshape(batch, -1), expected,
    ), axis=1)
    if raw.shape != (batch, 480):
        raise RuntimeError(f"device observation layout drifted to {raw.shape}")
    valid = (
        physical.valid
        & jp.all(jp.isfinite(raw))
        & jp.all(scale > 0)
        & jp.all(upper > lower)
        & jp.isfinite(jp.asarray(table_surface_height))
        & jp.all((expected == 0) | (expected == 1))
        & jp.all((actions >= 1) & (actions <= 50))
    )
    return raw, valid


def check_device_termination(
    *, object_position: Any, target_position: Any, progress: Any,
    trajectory_lengths: Any, early_mask: Any, max_deviation_distance: float,
    deviation_penalty: float,
) -> DeviceTermination:
    """JAX form of :func:`check_termination` for the narrow device path."""

    import jax.numpy as jp

    object_position, target_position = jp.asarray(object_position), jp.asarray(target_position)
    progress, lengths, early = jp.asarray(progress), jp.asarray(trajectory_lengths), jp.asarray(early_mask)
    if object_position.ndim != 2 or object_position.shape[1] != 3 or target_position.shape != object_position.shape:
        raise ValueError("object and target positions must both be (batch, 3)")
    batch = object_position.shape[0]
    if progress.shape != (batch,) or lengths.shape != (batch,) or early.shape != (batch,):
        raise ValueError("termination vectors must be (batch,)")
    if (not np.isfinite(max_deviation_distance) or not np.isfinite(deviation_penalty)
            or max_deviation_distance < 0 or deviation_penalty < 0):
        raise ValueError("termination distances and penalty must be finite and non-negative")
    deviation = jp.linalg.norm(object_position - target_position, axis=1) > max_deviation_distance
    deviation = deviation & ~early.astype(bool)
    reset = (progress >= lengths - 1) | deviation
    penalty = jp.where(deviation, -deviation_penalty, 0.0)
    reason = jp.where(deviation, 2, jp.where(reset, 1, 0)).astype(jp.int32)
    valid = (
        jp.all(jp.isfinite(object_position))
        & jp.all(jp.isfinite(target_position))
        & jp.all(jp.isfinite(penalty))
        & jp.all((reason >= 0) & (reason <= 2))
    )
    return DeviceTermination(reset, deviation, penalty, reason, valid)


def compute_device_reward_28(
    *, object_position: Any, target_object_position: Any,
    object_orientation_xyzw: Any, target_object_orientation_xyzw: Any,
    cumulative_offset: Any, cumulative_joint_offset: Any, active_joint_mask: Any,
    hand_object_force_on_object_world_N: Any, expected_contact_mask: Any,
    expected_contact_weights: Any, object_linear_velocity: Any, trajectory_steps: Any,
    contact_start_frames: Any, contact_end_frames: Any, rotation_disabled_mask: Any,
    early_phase_starts: Any, early_phase_steps: int, termination: DeviceTermination, config: Any,
) -> DeviceReward:
    """Exact source reward equations evaluated on JAX arrays.

    CUDA MJX inputs are float32, whereas the legacy decoder promotes them to
    float64.  Consequently a force within one float32 ULP of 0.2 N can choose
    the adjacent strict-gate outcome; outside that documented rounding band the
    reward contract is numerically equivalent.
    """

    import jax.numpy as jp

    obj, target = jp.asarray(object_position), jp.asarray(target_object_position)
    quat, target_quat = jp.asarray(object_orientation_xyzw), jp.asarray(target_object_orientation_xyzw)
    offsets, joints, active = jp.asarray(cumulative_offset), jp.asarray(cumulative_joint_offset), jp.asarray(active_joint_mask)
    force, expected, weights, velocity = map(jp.asarray, (hand_object_force_on_object_world_N, expected_contact_mask, expected_contact_weights, object_linear_velocity))
    steps, starts, ends = map(jp.asarray, (trajectory_steps, contact_start_frames, contact_end_frames))
    disabled, early_starts = map(jp.asarray, (rotation_disabled_mask, early_phase_starts))
    batch = obj.shape[0]
    required = ((batch, 3), (batch, 3), (batch, 4), (batch, 4), (batch, 16, 3), (batch, 16), (batch, 16), (batch, 3))
    if tuple(value.shape for value in (obj, target, quat, target_quat, force, expected, weights, velocity)) != required:
        raise ValueError("device reward tensors differ from the 28-DoF reward ABI")
    if offsets.shape != (batch, 3) or joints.shape != (batch, 22) or active.shape != (batch, 22):
        raise ValueError("device reward supports one 28-DoF hand and 22 joint residuals")
    if any(value.shape != (batch,) for value in (steps, starts, ends, disabled, early_starts)):
        raise ValueError("device reward window vectors must be (batch,)")
    if early_phase_steps < 0:
        raise ValueError("early_phase_steps must be non-negative")
    distance = jp.abs(obj - target)
    ungated = jp.asarray(config.distance_scales) * jp.exp(-config.distance_decay * distance)
    ax, ay, az, aw = (quat[:, i] for i in range(4))
    bx, by, bz, bw = (-target_quat[:, 0], -target_quat[:, 1], -target_quat[:, 2], target_quat[:, 3])
    product = jp.stack((aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), axis=1)
    degrees = 2.0 * jp.arcsin(jp.minimum(jp.linalg.norm(product, axis=1), 1.0)) * 180.0 / 3.14159265359
    segment_1 = config.rotation_segment_1_coeff * degrees**2 + 1.0
    offset = degrees - config.rotation_segment_1_threshold_deg
    segment_2 = config.rotation_segment_2_coeff * offset**2 + config.rotation_segment_2_linear_coeff * offset + config.rotation_segment_2_constant
    rotation_value = jp.where(degrees <= config.rotation_segment_1_threshold_deg, segment_1, jp.where(degrees <= config.rotation_segment_2_threshold_deg, segment_2, config.rotation_segment_3_value))
    rotation = jp.where(disabled.astype(bool), 0.0, config.rotation_scale * rotation_value + config.rotation_base_penalty)
    position_base = jp.sum(jp.abs(offsets * config.position_penalty_scale), axis=1)
    joint_base = jp.sum(jp.where(active.astype(bool), jp.abs(joints * config.joint_penalty_scale), 0.0), axis=1)
    joint_base = joint_base / jp.maximum(active.sum(axis=1), 1.0) * config.reference_joint_count
    position_penalty = -config.action_penalty_scale * config.position_penalty_weight * position_base
    joint_penalty = -config.action_penalty_scale * config.joint_penalty_weight * joint_base
    action_penalty = position_penalty + joint_penalty
    magnitudes = jp.linalg.norm(force, axis=-1)
    weighted_expected = jp.sum(expected * weights, axis=1)
    weighted_correct = jp.sum((magnitudes > config.contact_force_threshold) * expected * weights, axis=1)
    raw_contact = jp.where(weighted_expected > 0, weighted_correct / weighted_expected * config.max_contact_reward, 0.0)
    within = (steps >= starts) & (steps <= ends)
    valid_window = ends >= starts
    contact = jp.where(within, raw_contact, 0.0) * config.direct_contact_reward_scale
    distance_gate = jp.where(within & valid_window, jp.where(within, raw_contact, 0.0), 0.0)
    distance_gate = jp.where((steps > ends) & valid_window, config.max_contact_reward, distance_gate)
    distance_terms = ungated * distance_gate[:, None]
    speed = jp.linalg.norm(velocity, axis=1)
    # Host reward semantics deliberately disable stability for a non-positive
    # reference speed. Use a safe denominator because both jp.where branches
    # are evaluated and 0/0 would otherwise poison the final reduction.
    reference_speed = jp.asarray(config.object_stability_reference_speed, dtype=obj.dtype)
    safe_reference_speed = jp.where(reference_speed > 0, reference_speed, 1.0)
    stability_base = config.max_object_stability_reward * jp.exp(-(speed / safe_reference_speed) ** 2)
    stability_base = jp.where(reference_speed > 0, stability_base, 0.0)
    stability = jp.where(valid_window & (steps > ends), stability_base, 0.0)
    survival = jp.full((batch,), config.survival_reward, dtype=obj.dtype)
    early = (steps >= early_starts) & (steps < early_starts + early_phase_steps)
    total = jp.where(early, action_penalty, distance_terms.sum(axis=1) + rotation + action_penalty + contact + stability + survival) + termination.deviation_penalty
    config_values = tuple(np.asarray(value, dtype=np.float64) for value in vars(config).values())
    valid_inputs = (
        jp.all(jp.isfinite(obj)) & jp.all(jp.isfinite(target))
        & jp.all(jp.isfinite(quat)) & jp.all(jp.isfinite(target_quat))
        & jp.all(jp.isfinite(offsets)) & jp.all(jp.isfinite(joints))
        & jp.all(jp.isfinite(force)) & jp.all(jp.isfinite(expected))
        & jp.all(jp.isfinite(weights)) & jp.all(jp.isfinite(velocity))
        & jp.all(jp.isfinite(steps)) & jp.all(jp.isfinite(starts)) & jp.all(jp.isfinite(ends))
        & jp.all(jp.isfinite(early_starts))
        & jp.all((expected == 0) | (expected == 1)) & jp.all(weights >= 0)
        & jp.all(jp.isfinite(termination.deviation_penalty))
        & jp.asarray(all(np.all(np.isfinite(value)) for value in config_values))
        & termination.valid
    )
    valid = valid_inputs & jp.all(jp.isfinite(total)) & jp.all(jp.isfinite(stability))
    return DeviceReward(
        total, distance_terms[:, 0], distance_terms[:, 1], distance_terms[:, 2],
        ungated[:, 0], ungated[:, 1], ungated[:, 2], rotation,
        position_penalty, joint_penalty, action_penalty, raw_contact, contact,
        distance_gate, stability, speed, survival, early,
        termination.deviation_penalty, valid,
    )


def advance_device_task_counters(
    *, progress: Any, trajectory_steps: Any, episode_returns: Any, pending_reset: Any,
    reward_total: Any, next_reset: Any, control_call: Any,
) -> DeviceTaskCounters:
    """Commit counters in ``step`` order, including delayed partial resets."""

    import jax.numpy as jp

    progress, steps, returns, pending = map(jp.asarray, (progress, trajectory_steps, episode_returns, pending_reset))
    reward, reset, call = map(jp.asarray, (reward_total, next_reset, control_call))
    batch = progress.shape[0]
    if any(value.shape != (batch,) for value in (steps, returns, pending, reward, reset)):
        raise ValueError("device task counters must be batched equally")
    stepped = steps + 1
    stepped = jp.where(progress == 0, 0, stepped)
    progressed = progress + 1
    # Physics/control use the pre-reset values. The reset is deliberately
    # applied before extraction, termination, reward and this return commit.
    progressed = jp.where(pending.astype(bool), 0, progressed)
    stepped = jp.where(pending.astype(bool), 0, stepped)
    returns = jp.where(pending.astype(bool), 0.0, returns) + reward
    return DeviceTaskCounters(progressed, stepped, returns, reset.astype(bool), call + 1)


def reduce_warp_contacts(
    *, nacon: Any, nefc: Any, geom: Any, world: Any, dimension: Any, addresses: Any,
    friction: Any, frame: Any, constraint_force: Any, ngeom: int,
    keypoint_geom_ids: Sequence[int], object_geom_ids: Sequence[int], compute_dtype: str = "float32",
) -> DeviceContactReduction:
    """Reduce pinned MJX-Warp contacts without host materialization."""

    import jax
    import jax.numpy as jp

    if compute_dtype not in {"float32", "float64"}:
        raise ValueError("compute_dtype must be float32 or float64")
    if compute_dtype == "float64" and not jax.config.x64_enabled:
        raise ValueError("float64 contact reduction requires jax_enable_x64")
    floating_dtype = jp.float64 if compute_dtype == "float64" else jp.float32
    if ngeom < 1:
        raise ValueError("ngeom must be positive")
    keypoint_ids, object_ids = np.asarray(keypoint_geom_ids, dtype=np.int32), np.asarray(object_geom_ids, dtype=np.int32)
    if keypoint_ids.shape != (len(KEYPOINT_NAMES),) or len(set(keypoint_ids.tolist())) != len(KEYPOINT_NAMES):
        raise ValueError("one distinct source-order geom is required per keypoint")
    if object_ids.ndim != 1 or not len(object_ids) or np.any(keypoint_ids < 0) or np.any(keypoint_ids >= ngeom) or np.any(object_ids < 0) or np.any(object_ids >= ngeom) or np.intersect1d(keypoint_ids, object_ids).size:
        raise ValueError("keypoint and object geom ids must be disjoint valid geoms")
    geom, world, dimension, addresses = map(jp.asarray, (geom, world, dimension, addresses))
    friction, frame, force = jp.asarray(friction, dtype=floating_dtype), jp.asarray(frame, dtype=floating_dtype), jp.asarray(constraint_force, dtype=floating_dtype)
    nefc_values = jp.asarray(nefc).reshape(-1)
    if geom.ndim != 2 or geom.shape[1] != 2:
        raise ValueError("geom must have shape (capacity, 2)")
    capacity, batch = geom.shape[0], force.shape[0]
    if world.shape != (capacity,) or dimension.shape != (capacity,) or addresses.shape != (capacity, 4) or friction.ndim != 2 or friction.shape[0] != capacity or friction.shape[1] < 2 or frame.shape != (capacity, 3, 3) or force.ndim != 2 or nefc_values.shape not in {(1,), (batch,)}:
        raise ValueError("MJX-Warp contact ABI shapes differ from the pinned contract")
    nefc = jp.broadcast_to(nefc_values, (batch,)) if nefc_values.shape == (1,) else nefc_values
    count, slots = jp.asarray(nacon).reshape(()), jp.arange(capacity, dtype=jp.int32)
    live = slots < count
    safe_world, safe_geom, safe_address = jp.clip(world, 0, batch - 1), jp.clip(geom, 0, ngeom - 1), jp.clip(addresses, 0, force.shape[1] - 1)
    addresses_valid = jp.all((addresses >= 0) & (addresses < nefc[safe_world, None]), axis=1)
    row_valid = (world >= 0) & (world < batch) & (dimension == 3) & jp.all((geom >= 0) & (geom < ngeom), axis=1) & addresses_valid
    valid = (count >= 0) & (count < capacity) & jp.all((nefc >= 0) & (nefc < force.shape[1])) & jp.all(jp.where(live, row_valid, True))
    contribution = live & row_valid
    pyramid = force[safe_world[:, None], safe_address]
    local_force = jp.stack((jp.sum(pyramid, axis=1), (pyramid[:, 0] - pyramid[:, 1]) * friction[:, 0], (pyramid[:, 2] - pyramid[:, 3]) * friction[:, 1]), axis=1)
    world_force = jp.einsum("ni,nij->nj", local_force, frame)
    valid = valid & jp.all(jp.where(contribution[:, None], jp.isfinite(world_force), True))
    world_force = jp.where(contribution[:, None], world_force, 0.0)
    lookup = jp.full((ngeom,), -1, dtype=jp.int32).at[jp.asarray(keypoint_ids)].set(jp.arange(len(KEYPOINT_NAMES), dtype=jp.int32))
    first, second = lookup[safe_geom[:, 0]], lookup[safe_geom[:, 1]]
    keypoint_forces = jp.zeros((batch, len(KEYPOINT_NAMES), 3), dtype=world_force.dtype)
    first_is, second_is = contribution & (first >= 0), contribution & (second >= 0)
    keypoint_forces = keypoint_forces.at[safe_world, jp.maximum(first, 0)].add(-world_force * first_is[:, None])
    keypoint_forces = keypoint_forces.at[safe_world, jp.maximum(second, 0)].add(world_force * second_is[:, None])
    object_device = jp.asarray(object_ids)
    first_object, second_object = jp.any(safe_geom[:, 1, None] == object_device[None], axis=1), jp.any(safe_geom[:, 0, None] == object_device[None], axis=1)
    first_hand, second_hand = contribution & (first >= 0) & (second < 0) & first_object, contribution & (second >= 0) & (first < 0) & second_object
    hand_object = jp.zeros_like(keypoint_forces)
    hand_object = hand_object.at[safe_world, jp.maximum(first, 0)].add(world_force * first_hand[:, None])
    hand_object = hand_object.at[safe_world, jp.maximum(second, 0)].add(-world_force * second_hand[:, None])
    counts = jp.zeros((batch,), dtype=jp.int32).at[safe_world].add(contribution.astype(jp.int32))
    valid = valid & jp.all(jp.isfinite(keypoint_forces)) & jp.all(jp.isfinite(hand_object))
    return DeviceContactReduction(keypoint_forces, hand_object, counts, valid)


def reduce_warp_contacts_for_right_policy(
    *, right_keypoint_geom_ids: Sequence[int], left_keypoint_geom_ids: Sequence[int], **kwargs: Any,
) -> DeviceContactReduction:
    """Reduce a dual-hand scene while exposing a right-hand policy interface.

    Both reducers inspect the same global buffer.  The right forces are the
    policy observation; forces delivered to the object are summed for reward.
    Contact counts describe global buffer occupancy, so the equal independent
    scans are checked and one count is retained rather than added.
    """

    import jax.numpy as jp

    right = reduce_warp_contacts(keypoint_geom_ids=right_keypoint_geom_ids, **kwargs)
    left = reduce_warp_contacts(keypoint_geom_ids=left_keypoint_geom_ids, **kwargs)
    counts_match = jp.all(right.per_world_count == left.per_world_count)
    return DeviceContactReduction(
        right.keypoint_forces,
        right.hand_object_forces + left.hand_object_forces,
        right.per_world_count,
        right.valid & left.valid & counts_match,
    )
