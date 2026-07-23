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


def torch_to_jax_cuda(tensor: Any) -> Any:
    """Borrow a CUDA Torch tensor in JAX through same-device DLPack only."""

    import jax

    if not getattr(tensor, "is_cuda", False):
        raise ValueError("device runtime accepts CUDA Torch tensors only")
    array = jax.dlpack.from_dlpack(tensor)
    if array.device.platform != "cuda":
        raise RuntimeError("Torch-to-JAX DLPack conversion did not retain CUDA placement")
    return array


def jax_to_torch_cuda(array: Any) -> Any:
    """Borrow a CUDA JAX array in Torch through same-device DLPack only."""

    import torch

    if getattr(array.device, "platform", None) != "cuda":
        raise ValueError("device runtime returns CUDA JAX arrays only")
    tensor = torch.from_dlpack(array)
    if not tensor.is_cuda:
        raise RuntimeError("JAX-to-Torch DLPack conversion did not retain CUDA placement")
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
