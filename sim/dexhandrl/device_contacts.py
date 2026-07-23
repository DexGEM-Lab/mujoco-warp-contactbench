"""Pure JAX decoding of DexHand's pinned global MJX-Warp contact ABI.

The Warp implementation stores all worlds in one capacity-sized contact table.
This module deliberately consumes that representation directly: no ``Data``
materialization, Python contact loop, or host array conversion occurs inside a
compiled reduction.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Sequence


class DeviceContactSummary(NamedTuple):
    """Per-world force features plus a fail-closed ABI validity scalar."""

    hand: Any
    object: Any
    count: Any
    valid: Any


def reduce_dexhand_contacts(
    *,
    nacon: Any,
    nefc: Any,
    contact_geom: Any,
    contact_worldid: Any,
    contact_dim: Any,
    contact_efc_address: Any,
    contact_friction: Any,
    contact_frame: Any,
    efc_force: Any,
    ngeom: int,
    geom_to_contact_body: Sequence[int],
    object_geom_flags: Sequence[bool] | Sequence[int],
) -> DeviceContactSummary:
    """Decode pyramidal ``condim=3`` contacts from global Warp buffers.

    ``geom_to_contact_body`` maps each model geom to a DexHand contact body in
    ``[0, 15]`` or ``-1``.  ``object_geom_flags`` identifies every geom owned
    by the task object, allowing multi-geom hands and objects.  Invalid active
    rows never index out of bounds and make ``valid`` false rather than being
    silently interpreted as zero contact.
    """
    import jax.numpy as jp

    if ngeom < 1:
        raise ValueError("ngeom must be positive")
    body_lookup = jp.asarray(geom_to_contact_body, dtype=jp.int32)
    object_flags = jp.asarray(object_geom_flags, dtype=jp.bool_)
    if body_lookup.shape != (ngeom,) or object_flags.shape != (ngeom,):
        raise ValueError("geom mappings must be one entry per model geom")

    geom = jp.asarray(contact_geom)
    world = jp.asarray(contact_worldid)
    dim = jp.asarray(contact_dim)
    addresses = jp.asarray(contact_efc_address)
    friction = jp.asarray(contact_friction)
    frame = jp.asarray(contact_frame)
    force = jp.asarray(efc_force)
    nefc_values = jp.asarray(nefc).reshape(-1)
    if geom.ndim != 2 or geom.shape[1] != 2:
        raise ValueError("contact__geom must be (capacity, 2)")
    capacity = geom.shape[0]
    if force.ndim != 2:
        raise ValueError("efc__force must be (batch, constraint_capacity)")
    batch, force_capacity = force.shape
    if (
        world.shape != (capacity,)
        or dim.shape != (capacity,)
        or addresses.shape != (capacity, 4)
        or friction.ndim != 2
        or friction.shape[0] != capacity
        or friction.shape[1] < 2
        or frame.shape not in {(capacity, 3, 3), (capacity, 9)}
        or nefc_values.shape not in {(1,), (batch,)}
    ):
        raise ValueError("MJX-Warp contact ABI shapes differ from the pinned contract")
    frame = frame.reshape(capacity, 3, 3)
    nefc_world = (
        jp.broadcast_to(nefc_values, (batch,))
        if nefc_values.shape == (1,)
        else nefc_values
    )
    count = jp.asarray(nacon).reshape(())
    slots = jp.arange(capacity, dtype=jp.int32)
    live = slots < count
    safe_world = jp.clip(world, 0, batch - 1)
    safe_geom = jp.clip(geom, 0, ngeom - 1)
    safe_address = jp.clip(addresses, 0, max(force_capacity - 1, 0))
    addresses_valid = jp.all(
        (addresses >= 0) & (addresses < nefc_world[safe_world, None]), axis=1
    )
    row_valid = (
        (world >= 0)
        & (world < batch)
        & (dim == 3)
        & jp.all((geom >= 0) & (geom < ngeom), axis=1)
        & addresses_valid
    )
    valid = (
        (count >= 0)
        & (count < capacity)
        & jp.all((nefc_world >= 0) & (nefc_world < force_capacity))
        & jp.all(jp.where(live, row_valid, True))
    )
    contribution = live & row_valid
    pyramid = force[safe_world[:, None], safe_address]
    local = jp.stack(
        (
            jp.sum(pyramid, axis=1),
            (pyramid[:, 0] - pyramid[:, 1]) * friction[:, 0],
            (pyramid[:, 2] - pyramid[:, 3]) * friction[:, 1],
        ),
        axis=1,
    )
    world_force = jp.einsum("ni,nij->nj", local, frame)
    valid = valid & jp.all(jp.where(contribution[:, None], jp.isfinite(world_force), True))
    world_force = jp.where(contribution[:, None], world_force, 0.0)

    lookup = body_lookup
    mapping_valid = jp.all((lookup >= -1) & (lookup < 16)) & jp.all(~((lookup >= 0) & object_flags))
    valid = valid & mapping_valid
    body1, body2 = lookup[safe_geom[:, 0]], lookup[safe_geom[:, 1]]
    obj = object_flags
    object1, object2 = obj[safe_geom[:, 0]], obj[safe_geom[:, 1]]
    hand = jp.zeros((batch, 16, 3), dtype=world_force.dtype)
    first_hand = contribution & (body1 >= 0)
    second_hand = contribution & (body2 >= 0)
    hand = hand.at[safe_world, jp.maximum(body1, 0)].add(-world_force * first_hand[:, None])
    hand = hand.at[safe_world, jp.maximum(body2, 0)].add(world_force * second_hand[:, None])
    object_force = jp.zeros((batch, 3), dtype=world_force.dtype)
    object_force = object_force.at[safe_world].add(-world_force * (contribution & object1)[:, None])
    object_force = object_force.at[safe_world].add(world_force * (contribution & object2)[:, None])
    counts = jp.zeros((batch,), dtype=jp.int32).at[safe_world].add(contribution.astype(jp.int32))
    valid = valid & jp.all(jp.isfinite(hand)) & jp.all(jp.isfinite(object_force))
    return DeviceContactSummary(hand, object_force, counts, valid)
