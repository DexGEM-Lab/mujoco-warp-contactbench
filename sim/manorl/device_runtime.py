"""JAX primitives for the opt-in ManoRL device transition.

These functions intentionally accept the MJX-Warp private contact arrays directly.
They never build the capacity-sized ``(batch, ngeom, 3)`` diagnostic tensor used
by the NumPy snapshot path: the production transition only needs primary-hand
keypoint forces and hand-on-object forces.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import numpy as np

from sim.manorl.contracts import KEYPOINT_NAMES


class DeviceContactReduction(NamedTuple):
    """Reduced contact features plus a device-resident fail-closed status.

    ``valid`` is scalar and must be checked by the caller before consuming the
    forces. It covers the capacity, world, condim, geom, and solved-constraint
    invariants that the reference producer raises for on the host.
    """

    keypoint_forces: Any
    hand_object_forces: Any
    per_world_count: Any
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


def reduce_warp_contacts(
    *,
    nacon: Any,
    nefc: Any,
    geom: Any,
    world: Any,
    dimension: Any,
    addresses: Any,
    friction: Any,
    frame: Any,
    constraint_force: Any,
    ngeom: int,
    keypoint_geom_ids: Sequence[int],
    object_geom_ids: Sequence[int],
) -> DeviceContactReduction:
    """Reduce the pinned MJX-Warp contact ABI without host materialization.

    The allocated contact capacity is static under JIT while ``nacon`` is a
    dynamic global live-count. Every padded row is masked before it can affect
    a gather or scatter. Invalid live rows are similarly sanitized for safe
    gathers, then reported through ``valid`` so the environment can fail closed
    on the same transition without copying the contact buffers to NumPy.
    """

    import jax.numpy as jp

    if ngeom < 1:
        raise ValueError("ngeom must be positive")
    keypoint_ids = np.asarray(keypoint_geom_ids, dtype=np.int32)
    object_ids = np.asarray(object_geom_ids, dtype=np.int32)
    if keypoint_ids.shape != (len(KEYPOINT_NAMES),) or len(set(keypoint_ids.tolist())) != len(KEYPOINT_NAMES):
        raise ValueError("one distinct source-order geom is required per keypoint")
    if object_ids.ndim != 1 or not len(object_ids):
        raise ValueError("at least one object collision geom is required")
    if (
        np.any(keypoint_ids < 0)
        or np.any(keypoint_ids >= ngeom)
        or np.any(object_ids < 0)
        or np.any(object_ids >= ngeom)
        or np.intersect1d(keypoint_ids, object_ids).size
    ):
        raise ValueError("keypoint and object geom ids must be disjoint valid geoms")

    geom = jp.asarray(geom)
    world = jp.asarray(world)
    dimension = jp.asarray(dimension)
    addresses = jp.asarray(addresses)
    friction = jp.asarray(friction)
    frame = jp.asarray(frame)
    force = jp.asarray(constraint_force)
    nefc = jp.asarray(nefc).reshape(-1)
    if geom.ndim != 2 or geom.shape[1] != 2:
        raise ValueError("geom must have shape (capacity, 2)")
    capacity = geom.shape[0]
    batch = force.shape[0]
    if (
        world.shape != (capacity,)
        or dimension.shape != (capacity,)
        or addresses.shape != (capacity, 4)
        or friction.ndim != 2
        or friction.shape[0] != capacity
        or friction.shape[1] < 2
        or frame.shape != (capacity, 3, 3)
        or force.ndim != 2
        or nefc.shape != (batch,)
    ):
        raise ValueError("MJX-Warp contact ABI shapes differ from the pinned contract")

    count = jp.asarray(nacon).reshape(())
    slots = jp.arange(capacity, dtype=jp.int32)
    live = slots < count
    safe_world = jp.clip(world, 0, batch - 1)
    safe_geom = jp.clip(geom, 0, ngeom - 1)
    safe_address = jp.clip(addresses, 0, force.shape[1] - 1)
    addresses_valid = jp.all((addresses >= 0) & (addresses < nefc[safe_world, None]), axis=1)
    row_valid = (
        (world >= 0)
        & (world < batch)
        & (dimension == 3)
        & jp.all((geom >= 0) & (geom < ngeom), axis=1)
        & addresses_valid
    )
    capacity_valid = (count >= 0) & (count < capacity) & jp.all((nefc >= 0) & (nefc < force.shape[1]))
    valid = capacity_valid & jp.all(jp.where(live, row_valid, True))
    contribution = live & row_valid

    pyramid = force[safe_world[:, None], safe_address]
    local_force = jp.stack(
        (
            jp.sum(pyramid, axis=1),
            (pyramid[:, 0] - pyramid[:, 1]) * friction[:, 0],
            (pyramid[:, 2] - pyramid[:, 3]) * friction[:, 1],
        ),
        axis=1,
    )
    world_force = jp.einsum("ni,nij->nj", local_force, frame)
    world_force = jp.where(contribution[:, None], world_force, 0.0)

    keypoint_lookup = jp.full((ngeom,), -1, dtype=jp.int32).at[jp.asarray(keypoint_ids)].set(
        jp.arange(len(KEYPOINT_NAMES), dtype=jp.int32)
    )
    first_keypoint = keypoint_lookup[safe_geom[:, 0]]
    second_keypoint = keypoint_lookup[safe_geom[:, 1]]
    keypoint_forces = jp.zeros((batch, len(KEYPOINT_NAMES), 3), dtype=world_force.dtype)
    first_is_keypoint = contribution & (first_keypoint >= 0)
    second_is_keypoint = contribution & (second_keypoint >= 0)
    # Scatter all static slots; inactive/non-keypoint rows carry a zero value.
    # Boolean indexing would create dynamic shapes and cannot be JIT compiled.
    safe_first_keypoint = jp.maximum(first_keypoint, 0)
    safe_second_keypoint = jp.maximum(second_keypoint, 0)
    keypoint_forces = keypoint_forces.at[safe_world, safe_first_keypoint].add(
        -world_force * first_is_keypoint[:, None]
    )
    keypoint_forces = keypoint_forces.at[safe_world, safe_second_keypoint].add(
        world_force * second_is_keypoint[:, None]
    )

    object_ids_device = jp.asarray(object_ids)
    first_object = jp.any(safe_geom[:, 1, None] == object_ids_device[None, :], axis=1)
    second_object = jp.any(safe_geom[:, 0, None] == object_ids_device[None, :], axis=1)
    first_hand = contribution & (first_keypoint >= 0) & (second_keypoint < 0) & first_object
    second_hand = contribution & (second_keypoint >= 0) & (first_keypoint < 0) & second_object
    hand_object_forces = jp.zeros_like(keypoint_forces)
    hand_object_forces = hand_object_forces.at[safe_world, safe_first_keypoint].add(
        world_force * first_hand[:, None]
    )
    hand_object_forces = hand_object_forces.at[safe_world, safe_second_keypoint].add(
        -world_force * second_hand[:, None]
    )
    per_world_count = jp.zeros((batch,), dtype=jp.int32).at[safe_world].add(contribution.astype(jp.int32))
    valid = valid & jp.all(jp.isfinite(keypoint_forces)) & jp.all(jp.isfinite(hand_object_forces))
    return DeviceContactReduction(keypoint_forces, hand_object_forces, per_world_count, valid)
