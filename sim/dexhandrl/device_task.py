"""Pure JAX task kernels for batched DexHand021Pro rollouts.

This is intentionally a task layer, not an MJX producer: callers provide
batched physics tensors and pinned contact reductions, retain ownership of
reset/stepping, and can JIT these functions without importing MuJoCo or Warp.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from sim.dexhandrl.obs_layout import OBS_DIM_021PRO

_DIRECT_ACTIVE = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14], dtype=np.int32)
_DIRECT_FULL = np.asarray([6, 7, 8, 9, 10, 11, 14, 15, 18, 19, 22, 23], dtype=np.int32)
_ROUTED_ACTIVE = np.asarray([6, 9, 12, 15], dtype=np.int32)
_ROUTED_FULL_3 = np.asarray([12, 16, 20, 24], dtype=np.int32)
_ROUTED_FULL_4 = np.asarray([13, 17, 21, 25], dtype=np.int32)
_JOINT_SCALES = np.asarray([.01, .035, .02, .02, .01, .03, .02, .01, .03, .02, .01, .03, .02, .01, .03, .02], dtype=np.float32) * 3.0
_JOINT_MAX = np.asarray([.1, .35, .2, .2, .1, .4, .2, .1, .4, .2, .1, .4, .2, .1, .4, .2], dtype=np.float32) * 2.0


class DeviceControl(NamedTuple):
    """Stateful active-space correction and MuJoCo actuator-ordered control."""

    reference_active: Any
    corrected_active: Any
    applied_active: Any
    full_target: Any
    ctrl: Any
    base_offset: Any
    joint_offset: Any
    valid: Any


class DeviceReward(NamedTuple):
    """Raw and weighted reward diagnostics in source reward-term order.

    The columns are ``alive, x, y, z, contact, stability, rotation,
    fingertip, action_position, action_joint``. ``total`` includes terminal
    failure/timeout additions supplied by the caller.
    """

    total: Any
    raw: Any
    weighted: Any
    terminated: Any
    truncated: Any
    valid: Any


def _require_batch(value: Any, batch: int, trailing: tuple[int, ...], name: str) -> Any:
    import jax.numpy as jp
    value = jp.asarray(value)
    if value.shape != (batch, *trailing):
        raise ValueError(f"{name} must have shape {(batch, *trailing)}")
    return value


def route_full_to_active(*, full: Any) -> Any:
    """Exact batched 26D-to-22D active extraction.

    This is the device equivalent of :func:`full_to_active_dof`: direct joints
    are copied while each coupled distal target is the non-negative sum of its
    joint3 and joint4 targets.
    """
    import jax.numpy as jp
    full = jp.asarray(full)
    if full.ndim != 2 or full.shape[1] != 26:
        raise ValueError("full must be (batch,26)")
    active = jp.zeros((full.shape[0], 22), dtype=full.dtype).at[:, :6].set(full[:, :6])
    active = active.at[:, 6 + jp.asarray(_DIRECT_ACTIVE)].set(full[:, jp.asarray(_DIRECT_FULL)])
    return active.at[:, 6 + jp.asarray(_ROUTED_ACTIVE)].set(
        jp.maximum(full[:, jp.asarray(_ROUTED_FULL_3)], 0.0)
        + jp.maximum(full[:, jp.asarray(_ROUTED_FULL_4)], 0.0)
    )


def route_active_to_full(*, active: Any, current_full: Any, route_upper: Any) -> Any:
    """Exact batched 22D-to-26D DexHand distal routing.

    The routed joints preserve their live joint3/joint4 split, fill joint3
    first on closing, and empty joint3 first on opening, matching
    :func:`active_to_full_dof`.
    """
    import jax.numpy as jp
    active, current = jp.asarray(active), jp.asarray(current_full)
    if active.ndim != 2 or active.shape[1] != 22 or current.shape != (active.shape[0], 26):
        raise ValueError("active/current_full must be (batch,22)/(batch,26)")
    upper = jp.asarray(route_upper)
    if upper.shape != (4, 2):
        raise ValueError("route_upper must be (4, 2) for index/middle/ring/pinky")
    full = current.at[:, :6].set(active[:, :6])
    full = full.at[:, jp.asarray(_DIRECT_FULL)].set(active[:, 6 + jp.asarray(_DIRECT_ACTIVE)])
    a = active[:, 6 + jp.asarray(_ROUTED_ACTIVE)]
    joint3, joint4 = full[:, jp.asarray(_ROUTED_FULL_3)], full[:, jp.asarray(_ROUTED_FULL_4)]
    upper3 = jp.minimum(upper[:, 0], 1.2)[None]
    upper4 = upper[:, 1][None]
    desired = jp.minimum(jp.maximum(a, 0.0), upper3 + upper4)
    current3, current4 = jp.maximum(joint3, 0.0), jp.maximum(joint4, 0.0)
    delta = desired - current3 - current4
    increase = jp.maximum(delta, 0.0)
    next3 = current3 + jp.minimum(increase, jp.maximum(upper3 - current3, 0.0))
    next4 = current4 + jp.minimum(increase - (next3 - current3), jp.maximum(upper4 - current4, 0.0))
    decrease = jp.maximum(-delta, 0.0)
    decrease3 = jp.minimum(decrease, jp.maximum(next3, 0.0))
    next3 = next3 - decrease3
    next4 = next4 - jp.minimum(decrease - decrease3, jp.maximum(next4, 0.0))
    full = full.at[:, jp.asarray(_ROUTED_FULL_3)].set(next3)
    return full.at[:, jp.asarray(_ROUTED_FULL_4)].set(next4)


def apply_device_control(
    *, action: Any, reference_active: Any, current_full: Any, previous_active: Any,
    base_offset: Any, joint_offset: Any, active_joint_mask: Any, frame: Any,
    start_frame: Any, last_contact_frame: Any, use_residual: Any,
    early_phase_frames: int, base_decay: Any, post_contact_base_decay: Any,
    joint_decay: float, post_contact_joint_decay: float, base_scales: Any,
    base_max_offset: float, active_max_deltas: Any, active_lower: Any,
    active_upper: Any, route_upper: Any, actuator_ids: Any, nu: int,
) -> DeviceControl:
    """Apply residual recurrence, ordered filters, routing, and ctrl scatter."""
    import jax.numpy as jp
    action, reference, current, previous = map(jp.asarray, (action, reference_active, current_full, previous_active))
    batch = action.shape[0]
    if action.shape != (batch, 22) or reference.shape != (batch, 22) or current.shape != (batch, 26) or previous.shape != (batch, 22):
        raise ValueError("control tensors must be batched 22/26-wide")
    base_offset, joint_offset = _require_batch(base_offset, batch, (3,), "base_offset"), _require_batch(joint_offset, batch, (16,), "joint_offset")
    active_mask = _require_batch(active_joint_mask, batch, (16,), "active_joint_mask")
    frame, start, last = map(jp.asarray, (frame, start_frame, last_contact_frame))
    if any(x.shape != (batch,) for x in (frame, start, last)) or early_phase_frames < 0 or nu < 26:
        raise ValueError("control frame/capacity ABI is invalid")
    max_delta, lower, upper = map(jp.asarray, (active_max_deltas, active_lower, active_upper))
    if max_delta.shape != (22,) or lower.shape != (22,) or upper.shape != (22,) or jp.asarray(actuator_ids).shape != (26,):
        raise ValueError("active limits and actuator ids must be 22/26-wide")
    after = frame > last
    early = (frame - start) < early_phase_frames
    enabled = jp.asarray(use_residual).astype(bool) & ~early
    bdecay = jp.where(after[:, None], jp.asarray(post_contact_base_decay)[None], jp.asarray(base_decay)[None])
    jdecay = jp.where(after, post_contact_joint_decay, joint_decay)
    next_base = jp.clip(bdecay * base_offset + jp.clip(action[:, :3], -1., 1.) * jp.asarray(base_scales)[None], -base_max_offset, base_max_offset)
    next_joint = jdecay[:, None] * joint_offset + jp.clip(action[:, 6:], -1., 1.) * active_mask * jp.asarray(_JOINT_SCALES)[None]
    next_joint = jp.clip(next_joint, -jp.asarray(_JOINT_MAX)[None], jp.asarray(_JOINT_MAX)[None]) * active_mask
    next_base, next_joint = jp.where(enabled[:, None], next_base, 0.), jp.where(enabled[:, None], next_joint, 0.)
    # Scalar parity first routes the raw Lance active reference through the
    # live distal split, then extracts its corrected active representation.
    # The latter can differ from the raw reference when a coupled target is
    # clamped by the live joint distribution or route limits.
    reference_full = route_active_to_full(active=reference, current_full=current, route_upper=route_upper)
    corrected = route_full_to_active(full=reference_full)
    corrected = corrected.at[:, :3].add(next_base).at[:, 6:].add(next_joint)
    applied = jp.clip(previous + jp.clip(corrected - previous, -max_delta, max_delta), lower, upper)
    full = route_active_to_full(active=applied, current_full=current, route_upper=route_upper)
    ids = jp.asarray(actuator_ids)
    safe_ids = jp.clip(ids, 0, nu - 1)
    ctrl = jp.zeros((batch, nu), dtype=full.dtype).at[:, safe_ids].set(full)
    valid = (jp.all(jp.isfinite(action)) & jp.all(jp.isfinite(full)) & jp.all(jp.isfinite(ctrl)) & jp.all((ids >= 0) & (ids < nu)) & jp.all(max_delta > 0) & jp.all(lower <= upper))
    return DeviceControl(reference, corrected, applied, full, ctrl, next_base, next_joint, valid)


def gather_trajectory(*, values: Any, frame: Any, lengths: Any) -> Any:
    """Gather padded per-world trajectory values, clamped to each valid end."""
    import jax.numpy as jp
    values, frame, lengths = jp.asarray(values), jp.asarray(frame), jp.asarray(lengths)
    if values.ndim < 2 or frame.shape != (values.shape[0],) or lengths.shape != (values.shape[0],):
        raise ValueError("trajectory values/frame/lengths must be batched")
    index = jp.clip(jp.minimum(frame, lengths - 1), 0, values.shape[1] - 1).astype(jp.int32)
    return values[jp.arange(values.shape[0]), index]


def _quat_rotate(q: Any, v: Any) -> Any:
    import jax.numpy as jp
    xyz, w = q[..., :3], q[..., 3:4]
    return v * (2 * w * w - 1) + 2 * w * jp.cross(xyz, v) + 2 * xyz * jp.sum(xyz * v, axis=-1, keepdims=True)


def build_device_observation(
    *, action_onehot: Any, geometry: Any, current_full: Any, active_finger_lower: Any,
    active_finger_upper: Any, hand_position: Any, hand_quat_xyzw: Any,
    fingertip_positions: Any, contact_body_positions: Any, episode_time: Any,
    object_position: Any, object_quat_xyzw: Any, object_linear_velocity: Any,
    target_position: Any, target_quat_xyzw: Any, target_next_5: Any,
    expected_contact_mask: Any, hand_contact_force: Any, object_contact_force: Any,
    base_offset: Any, joint_offset: Any, point_cloud_local: Any,
) -> tuple[Any, Any]:
    """Build the exact batched 461D policy layout from device-resident tensors."""
    import jax.numpy as jp
    current = jp.asarray(current_full); batch = current.shape[0]
    if current.shape != (batch, 26):
        raise ValueError("current_full must be (batch,26)")
    onehot = _require_batch(action_onehot, batch, (50,), "action_onehot")
    geometry = _require_batch(geometry, batch, (12,), "geometry")
    hp, hq = _require_batch(hand_position, batch, (3,), "hand_position"), _require_batch(hand_quat_xyzw, batch, (4,), "hand_quat_xyzw")
    tips, bodies = _require_batch(fingertip_positions, batch, (5,3), "fingertip_positions"), _require_batch(contact_body_positions, batch, (16,3), "contact_body_positions")
    obj, oq, vel = _require_batch(object_position,batch,(3,),"object_position"), _require_batch(object_quat_xyzw,batch,(4,),"object_quat_xyzw"), _require_batch(object_linear_velocity,batch,(3,),"object_linear_velocity")
    target, tq, next5 = _require_batch(target_position,batch,(3,),"target_position"), _require_batch(target_quat_xyzw,batch,(4,),"target_quat"), _require_batch(target_next_5,batch,(3,),"target_next_5")
    expected, force = _require_batch(expected_contact_mask,batch,(16,),"expected_contact_mask"), _require_batch(hand_contact_force,batch,(16,3),"hand_contact_force")
    object_force = _require_batch(object_contact_force,batch,(3,),"object_contact_force")
    base, joint = _require_batch(base_offset,batch,(3,),"base_offset"), _require_batch(joint_offset,batch,(16,),"joint_offset")
    lower, upper = jp.asarray(active_finger_lower), jp.asarray(active_finger_upper)
    if lower.shape != (16,) or upper.shape != (16,): raise ValueError("active finger limits must be 16-wide")
    cloud = jp.asarray(point_cloud_local)
    if cloud.shape == (64,3): cloud = jp.broadcast_to(cloud, (batch,64,3))
    if cloud.shape != (batch,64,3): raise ValueError("point_cloud_local must be (64,3) or (batch,64,3)")
    active = jp.zeros((batch,22), dtype=current.dtype).at[:,:6].set(current[:,:6])
    active = active.at[:,6 + jp.asarray(_DIRECT_ACTIVE)].set(current[:,jp.asarray(_DIRECT_FULL)])
    active = active.at[:,6 + jp.asarray(_ROUTED_ACTIVE)].set(jp.maximum(current[:,jp.asarray(_ROUTED_FULL_3)],0) + jp.maximum(current[:,jp.asarray(_ROUTED_FULL_4)],0))
    fingers = 2 * (active[:,6:] - lower) / jp.maximum(upper - lower, 1e-6) - 1
    masked = force * expected[...,None]
    magnitude = jp.clip(jp.linalg.norm(masked,axis=-1) / 100., 0., 1.)
    direction = masked / jp.maximum(jp.linalg.norm(masked,axis=-1,keepdims=True), 1e-8)
    points_world = _quat_rotate(oq[:,None], cloud) + obj[:,None]
    inv_hand = jp.concatenate((-hq[:,:3], hq[:,3:]), axis=1)
    points_hand = _quat_rotate(inv_hand[:,None], points_world - hp[:,None])
    time = _require_batch(episode_time,batch,(),"episode_time")
    observation = jp.concatenate((onehot, geometry, fingers, jp.concatenate((hp,hq),axis=1), tips.reshape(batch,-1), bodies.reshape(batch,-1), time[:,None], obj, oq, vel, target, tq, next5, expected, magnitude, direction.reshape(batch,-1), jp.clip(jp.linalg.norm(object_force,axis=-1)/100.,0.,1.)[:,None], base, joint, points_hand.reshape(batch,-1)), axis=1)
    if observation.shape != (batch, OBS_DIM_021PRO): raise RuntimeError("DexHand observation layout drifted")
    valid = jp.all(jp.isfinite(observation)) & jp.all(upper > lower) & jp.all((expected == 0) | (expected == 1))
    return observation, valid


def fingertip_inside_object_penalty(
    *, fingertips_world: Any, object_position: Any, object_quat_xyzw: Any,
    primitive_type: Any, primitive_params: Any, active_fingertip_mask: Any,
    min_depth: float, max_depth: float, max_penalty: float,
) -> Any:
    """Device equivalent of DexHand's cube/cylinder/sphere penetration term."""
    import jax.numpy as jp
    tips, position, quat = jp.asarray(fingertips_world), jp.asarray(object_position), jp.asarray(object_quat_xyzw)
    batch = tips.shape[0]
    if tips.shape != (batch, 5, 3) or position.shape != (batch, 3) or quat.shape != (batch, 4):
        raise ValueError("fingertip/object tensors must be (batch,5,3)/(batch,3)/(batch,4)")
    params, mask, kind = jp.asarray(primitive_params), jp.asarray(active_fingertip_mask), jp.asarray(primitive_type)
    if params.shape != (batch, 3) or mask.shape != (batch, 5) or kind.shape != (batch,) or max_depth <= min_depth:
        raise ValueError("primitive/mask/penetration contract is invalid")
    inverse = jp.concatenate((-quat[:, :3], quat[:, 3:]), axis=1)
    local = _quat_rotate(inverse[:, None], tips - position[:, None])
    box_depth = jp.minimum(jp.minimum(params[:, None, 0] * .5 - jp.abs(local[..., 0]), params[:, None, 1] * .5 - jp.abs(local[..., 1])), params[:, None, 2] * .5 - jp.abs(local[..., 2]))
    radial = jp.sqrt(jp.sum(local[..., :2] ** 2, axis=-1))
    cylinder_depth = jp.minimum(params[:, None, 0] - radial, params[:, None, 1] * .5 - jp.abs(local[..., 2]))
    sphere_depth = params[:, None, 0] - jp.linalg.norm(local, axis=-1)
    depth = jp.where(kind[:, None] == 0, box_depth, jp.where(kind[:, None] == 1, cylinder_depth, jp.where(kind[:, None] == 2, sphere_depth, 0.)))
    ramp = jp.clip((jp.maximum(depth, 0.) - min_depth) / (max_depth - min_depth), 0., 1.)
    return -max_penalty * jp.sum(ramp * mask, axis=1) / jp.maximum(jp.sum(mask, axis=1), 1.)


def compute_device_reward_done(
    *, frame: Any, start_frame: Any, last_contact_frame: Any, lengths: Any,
    object_position: Any, target_position: Any, object_quat_xyzw: Any,
    target_quat_xyzw: Any, object_linear_velocity: Any, hand_contact_force: Any,
    expected_contact_mask: Any, base_offset: Any, joint_offset: Any,
    active_joint_mask: Any, fingertip_inside_penalty: Any, rotation_disabled: Any,
    early_phase_frames: int, contact_force_threshold: float,
    max_contact_reward: float, max_stability_reward: float, stability_scale: float,
    object_terminal_threshold: float, reward_weights: Any,
    termination_failure_reward: float, termination_timeout_reward: float,
) -> DeviceReward:
    """Compute scalar-environment parity reward components and done masks."""
    import jax.numpy as jp
    frame, start, last, lengths = map(jp.asarray,(frame,start_frame,last_contact_frame,lengths)); batch=frame.shape[0]
    obj,target,quat,tquat,vel = (_require_batch(x,batch,s,n) for x,s,n in ((object_position,(3,),"object_position"),(target_position,(3,),"target_position"),(object_quat_xyzw,(4,),"object_quat"),(target_quat_xyzw,(4,),"target_quat"),(object_linear_velocity,(3,),"object_linear_velocity")))
    force,expected = _require_batch(hand_contact_force,batch,(16,3),"hand_contact_force"),_require_batch(expected_contact_mask,batch,(16,),"expected_contact_mask")
    base,joint,mask = _require_batch(base_offset,batch,(3,),"base_offset"),_require_batch(joint_offset,batch,(16,),"joint_offset"),_require_batch(active_joint_mask,batch,(16,),"active_joint_mask")
    tip_penalty,disabled = jp.asarray(fingertip_inside_penalty),jp.asarray(rotation_disabled)
    weights=jp.asarray(reward_weights)
    if any(x.shape != (batch,) for x in (start,last,lengths,tip_penalty,disabled)) or weights.shape != (10,): raise ValueError("reward ABI vectors differ from pinned contract")
    early=(frame-start)<early_phase_frames; after=frame>last
    has=jp.linalg.norm(force,axis=-1)>contact_force_threshold
    quality=jp.sum(has* (expected>.5),axis=1)/jp.maximum(jp.sum(expected,axis=1),1.)*max_contact_reward
    quality=jp.where(after,0.,quality)
    tracking=quality[:,None]*jp.exp(-40.*jp.abs(obj-target))
    stability=jp.where(after,max_stability_reward*jp.exp(-(jp.linalg.norm(vel,axis=1)/max(stability_scale,1e-6))**2),0.)
    dot=jp.clip(jp.abs(jp.sum(quat*tquat,axis=1))/jp.maximum(jp.linalg.norm(quat,axis=1)*jp.linalg.norm(tquat,axis=1),1e-12),0.,1.)
    deg=2*jp.arccos(dot)*180/jp.pi; rot=jp.where(deg<=20,.4*(-.00125*deg**2+1)-.1,jp.where(deg<=90,.4*(-.00003175*(deg-20)**2-.019206*(deg-20)+.5)-.1,-.5)); rot=jp.where(disabled.astype(bool),0.,rot)
    position=-jp.sum(jp.abs(base*100),axis=1); action_joint=-jp.sum(jp.abs(joint*10),axis=1)/jp.maximum(jp.sum(mask,axis=1),1)*8; position=jp.where(after,position*.5,position); action_joint=jp.where(after,action_joint*.5,action_joint)
    raw=jp.stack((jp.ones((batch,),obj.dtype),tracking[:,0],tracking[:,1],tracking[:,2],quality,stability,rot,tip_penalty,position,action_joint),axis=1)
    raw=jp.where(early[:,None],raw.at[:,1:7].set(0.),raw)
    weighted=raw*weights[None]; error=jp.linalg.norm(obj-target,axis=1); terminated=(~early)&(error>object_terminal_threshold); truncated=frame>=lengths-1
    termination_reward = jp.where(
        terminated,
        jp.asarray(termination_failure_reward, dtype=obj.dtype),
        jp.where(truncated, jp.asarray(termination_timeout_reward, dtype=obj.dtype), 0.0),
    )
    total=jp.sum(weighted,axis=1)+termination_reward
    valid=(jp.all(jp.isfinite(raw))&jp.all(jp.isfinite(total))
           &jp.isfinite(jp.asarray(termination_failure_reward))
           &jp.isfinite(jp.asarray(termination_timeout_reward))
           &jp.all((expected==0)|(expected==1)))
    return DeviceReward(total,raw,weighted,terminated,truncated,valid)
