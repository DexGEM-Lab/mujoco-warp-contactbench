"""Structured reference records carried separately from the 453 ABI in PPO states.

States are an immutable snapshot of both streams. PPO memory owns each snapshot;
minibatches and bootstrap never consult the runtime's current reference index.
frame_id is serialized for provenance and is never encoded by the network.
"""

import torch

from .goal_stream import GoalStream, default_specs

GOAL_OFFSETS = (0, 6, 12, 24)
GOAL_SLOTS = len(GOAL_OFFSETS) + 1  # final reference endpoint
STREAM_SPECS = default_specs()
STATE_DIM = sum(GOAL_SLOTS * (spec.raw_dim + 4) + 1 for spec in STREAM_SPECS.values())


def pack_streams(streams):
    fields = []
    for name in STREAM_SPECS:
        stream = streams[name].validate_contract(raw_dim=STREAM_SPECS[name].raw_dim)
        if stream.length != GOAL_SLOTS:
            raise ValueError("PPO stream snapshot requires the configured bucket width")
        fields.extend(
            (
                stream.raw.flatten(1),
                stream.mask.float(),
                stream.dt,
                stream.frame_id.float(),
                stream.type_id.float(),
                stream.presence[:, None],
            )
        )
    return torch.cat(fields, -1)


def unpack_streams(states):
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(f"structured states must be [B,{STATE_DIM}]")
    streams = {}
    offset = 0
    for name, spec in STREAM_SPECS.items():
        width = GOAL_SLOTS * spec.raw_dim
        raw = states[:, offset : offset + width].reshape(-1, GOAL_SLOTS, spec.raw_dim)
        offset += width
        mask, dt, frame, kind = [
            states[:, offset + k * GOAL_SLOTS : offset + (k + 1) * GOAL_SLOTS]
            for k in range(4)
        ]
        offset += 4 * GOAL_SLOTS
        presence = states[:, offset]
        offset += 1
        streams[name] = GoalStream(
            raw, mask.bool(), dt, frame.long(), kind.long(), presence
        )
    return streams


def reference_states(runtime, indices=None):
    """Sample both dense ManoRL reference channels; masks shorten near the end."""
    import jax.numpy as j
    from sim.manorl.autonomy_v4 import _gather, reference_lengths, quat_rotate, rot6

    cache, env_ref = runtime.cache, runtime.env_ref
    index = runtime.indices if indices is None else indices
    last = j.broadcast_to(j.asarray(reference_lengths(cache, env_ref)) - 1, index.shape)
    targets = j.concatenate(
        (index[:, None] + j.asarray(GOAL_OFFSETS)[None], last[:, None]), -1
    )
    # Deduplicate clamped records; compact as a leading-True mask without sorting data.
    valid = j.concatenate(
        (targets[:, :-1] < last[:, None], j.ones((runtime.num_envs, 1), bool)), -1
    )
    order = j.argsort(~valid, axis=1, stable=True)
    targets = j.take_along_axis(j.minimum(targets, last[:, None]), order, 1)
    mask = j.take_along_axis(valid, order, 1)
    gathered = (
        _gather(cache, targets, env_ref=None)
        if env_ref is None
        else (env_ref[:, None], targets)
    )
    object_p = j.asarray(cache.object_origin)[gathered]
    object_q = j.asarray(cache.object_quat_xyzw)[gathered]
    canonical = j.asarray(cache.points_object_local)[j.asarray([0, 16, 32, 48])]
    placed = object_p[..., None, :] + quat_rotate(object_q[..., None, :], canonical)
    obj = j.concatenate(
        (
            object_p,
            rot6(object_q),
            (targets == last[:, None])[..., None],
            placed.reshape(runtime.num_envs, GOAL_SLOTS, 12),
        ),
        -1,
    )
    palm_p = j.asarray(cache.palm_origin)[gathered]
    palm_q = j.asarray(cache.palm_quat_xyzw)[gathered]
    local = j.asarray(cache.hand_region_pose_reference)[gathered][..., :3]
    keypoints = object_p[..., None, :] + quat_rotate(object_q[..., None, :], local)
    hand = j.concatenate(
        (
            palm_p,
            rot6(palm_q),
            j.asarray(cache.q_feasible)[gathered][..., 6:],
            keypoints.reshape(runtime.num_envs, GOAL_SLOTS, 48),
        ),
        -1,
    )
    dt = (targets - index[:, None]) * cache.control_timestep
    fields = []
    for raw, kind in ((obj, 0), (hand, 2)):
        raw = j.where(mask[..., None], raw, 0.0)
        fields.extend(
            (
                raw.reshape(runtime.num_envs, -1),
                mask.astype(j.float32),
                j.where(mask, dt, 0.0),
                j.where(mask, targets, 0.0).astype(j.float32),
                j.where(mask, kind, 0.0).astype(j.float32),
                j.ones((runtime.num_envs, 1)),
            )
        )
    return j.concatenate(fields, -1).astype(j.float32)
