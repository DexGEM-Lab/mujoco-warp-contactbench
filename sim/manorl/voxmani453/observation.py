"""453 current-state ABI built from ManoRL physics; no v5-vector conversion."""

from sim.manorl.autonomy_v4 import (
    ANTIWINDUP_ERROR,
    DOF_RATE,
    _gather,
    quat_rotate,
    relative_rot6,
    reference_lengths,
    shortest_angle,
)
from .config import PAST_FRAMES, SCENE_SIGMA, T_MAX


def pose(position, quaternion_xyzw):
    import jax.numpy as j

    return j.concatenate(
        (position / SCENE_SIGMA, quaternion_xyzw[..., (3, 0, 1, 2)]), -1
    )


def reference_keypoints(cache, index, env_ref=None):
    import jax.numpy as j

    if cache.hand_region_pose_reference is None:
        raise ValueError("453 observations require reference hand region poses")
    i = _gather(cache, index, env_ref=env_ref)
    local = j.asarray(cache.hand_region_pose_reference)[i][..., :3]
    return j.asarray(cache.object_origin)[i][..., None, :] + quat_rotate(
        j.asarray(cache.object_quat_xyzw)[i][..., None, :], local
    )


def build_raw_observation_voxmani453(
    physical,
    contact,
    cache,
    index,
    previous_command,
    env_ref=None,
    *,
    past_pose=None,
    past_pad=None,
    previous_action=None,
):
    import jax.numpy as j

    if contact.object_all_torque is None:
        raise ValueError("453 force decomposition requires total object torque")
    b = physical.q_raw.shape[0]
    i = _gather(cache, index, env_ref=env_ref)
    last_index = j.broadcast_to(j.asarray(reference_lengths(cache, env_ref)) - 1, (b,))
    final = _gather(cache, last_index, env_ref=env_ref)
    current_pose = pose(physical.object_origin, physical.object_quat_xyzw)
    if past_pose is None:
        # Only initial/reset construction; live runtime supplies measured history.
        past_pose = j.broadcast_to(current_pose[:, None], (b, len(PAST_FRAMES), 7))
        past_pad = j.ones((b, len(PAST_FRAMES)), dtype=bool)
    if previous_action is None:
        previous_action = j.zeros((b, 28), physical.q_raw.dtype)
    gravity = j.asarray(
        cache.gravity_world if cache.gravity_world is not None else [0.0, 0.0, -9.81]
    )
    weight = j.maximum(cache.object_mass * j.linalg.norm(gravity), 1e-6)
    torque_scale = j.maximum(weight * cache.object_radius, 1e-6)
    forces = contact.paired_force_on_object
    hand_force = forces.sum(1)
    hand_torque = contact.paired_torque_com.sum(1)
    force_obs = j.concatenate(
        (
            j.broadcast_to(cache.object_mass * gravity / weight, (b, 3)),
            (forces / weight).reshape(b, 48),
            hand_torque / torque_scale,
            (contact.object_all_force - hand_force) / weight,
            (contact.object_all_torque - hand_torque) / torque_scale,
        ),
        -1,
    )
    # V4 stores qdot/rate; reconstruct SI joint speeds before VoxMani /10.
    finger_velocity = physical.qdot[:, 6:] * j.asarray(DOF_RATE)[6:] / 10
    palm_v = physical.region_v_com[:, 0] + j.cross(
        physical.palm_w, physical.palm_origin - physical.region_com[:, 0]
    )
    ref_object = j.asarray(cache.object_origin)[i]
    ref_q = j.asarray(cache.object_quat_xyzw)[i]
    ref_hand = reference_keypoints(cache, index, env_ref)
    keypoint_delta = physical.region_origin - ref_hand
    hand_rms = j.sqrt(j.mean(j.sum(keypoint_delta**2, -1), -1))
    finger_rms = j.sqrt(
        j.mean((physical.q_raw[:, 6:] - j.asarray(cache.q_feasible)[i][:, 6:]) ** 2, -1)
    )
    last = j.maximum(last_index, 1)
    now = j.asarray(index) * cache.control_timestep
    end = last * cache.control_timestep
    # Privilege retains dense v11 reference semantics and has critic-only routing.
    priv = j.concatenate(
        (
            (physical.object_origin - ref_object) / SCENE_SIGMA,
            shortest_angle(physical.object_quat_xyzw, ref_q)[:, None] / j.pi,
            physical.object_v_com - j.asarray(cache.object_v_com)[i],
            hand_rms[:, None] / 0.1,
            finger_rms[:, None],
            (now / end)[:, None],
            ((end - now) / T_MAX)[:, None],
            j.ones((b, 1)),
            j.minimum(end - now, cache.control_timestep)[:, None] / T_MAX,
        ),
        -1,
    )
    # ManoRL v11 ends at the reference end: its final palm is the terminal goal.
    terminal = j.concatenate(
        (
            (j.asarray(cache.palm_origin)[final] - physical.palm_origin) / SCENE_SIGMA,
            relative_rot6(
                physical.palm_quat_xyzw, j.asarray(cache.palm_quat_xyzw)[final]
            ),
            (j.asarray(index) >= last)[:, None].astype(physical.q_raw.dtype),
        ),
        -1,
    )
    raw = j.concatenate(
        (
            current_pose,
            past_pose.reshape(b, 35),
            past_pad.astype(physical.q_raw.dtype),
            pose(physical.region_origin, physical.region_quat_xyzw).reshape(b, 112),
            physical.q_raw[:, 6:],
            finger_velocity,
            # Reducer gives force ON object; negate for reaction ON hand bones.
            j.tanh(-forces / 10).reshape(b, 48),
            force_obs,
            j.concatenate((palm_v, physical.palm_w / 10), -1),
            j.concatenate((physical.object_v_com, physical.object_w / 10), -1),
            j.zeros((b, 1)),
            previous_action,
            priv,
            j.tanh(hand_force[:, 2] / weight)[:, None],
            j.mean((contact.paired_count > 0).astype(physical.q_raw.dtype), -1)[
                :, None
            ],
            (previous_command - physical.q_raw) / j.asarray(ANTIWINDUP_ERROR),
            (keypoint_delta / SCENE_SIGMA).reshape(b, 48),
            terminal,
        ),
        -1,
    ).astype(j.float32)
    if raw.shape[-1] != 453:
        raise AssertionError(f"453 ABI drift: {raw.shape}")
    return raw


def update_history(history, counts, current_pose, execute):
    """Newest-first measured poses, with per-world reset counts."""
    import jax.numpy as j

    candidate = j.concatenate((current_pose[:, None], history[:, :-1]), 1)
    history = j.where(execute, candidate, history.at[:, 0].set(current_pose))
    counts = j.minimum(counts + j.asarray(execute, j.int32), 16)
    lag = j.asarray(PAST_FRAMES)
    pad = counts[:, None] < lag[None]
    past = j.where(pad[..., None], current_pose[:, None], history[:, lag])
    return history, counts, past, pad
