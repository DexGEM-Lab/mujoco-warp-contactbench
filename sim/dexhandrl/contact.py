"""Shared MuJoCo contact-force decoding for replay and RL environments."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def decode_contact_force_world(
    *,
    frame: np.ndarray,
    friction: np.ndarray,
    condim: int,
    efc_addresses: np.ndarray,
    efc_force: np.ndarray,
    pyramidal: bool,
) -> tuple[np.ndarray, float]:
    """Decode one MuJoCo contact into world-frame linear force and normal force."""

    local = np.zeros(6, dtype=np.float32)
    addresses = np.asarray(efc_addresses, dtype=np.int32).reshape(-1)
    forces = np.asarray(efc_force, dtype=np.float32).reshape(-1)
    dim = max(1, min(int(condim), 6))
    if pyramidal:
        base_address = int(addresses[0]) if addresses.size else -1
        if base_address < 0:
            return np.zeros(3, dtype=np.float32), 0.0
        if dim == 1:
            if base_address < len(forces):
                local[0] = forces[base_address]
        else:
            mu = np.asarray(friction, dtype=np.float32).reshape(-1)
            for index in range(dim - 1):
                address = base_address + 2 * index
                direction_1 = forces[address] if address < len(forces) else 0.0
                direction_2 = forces[address + 1] if address + 1 < len(forces) else 0.0
                local[0] += direction_1 + direction_2
                local[index + 1] = (direction_1 - direction_2) * (
                    mu[index] if index < len(mu) else 0.0
                )
    else:
        for index in range(dim):
            address = int(addresses[index]) if index < len(addresses) else -1
            if 0 <= address < len(forces):
                local[index] = forces[address]

    force_world = local[:3] @ np.asarray(frame, dtype=np.float32).reshape(3, 3)
    return np.asarray(force_world, dtype=np.float32), float(local[0])


def collect_contact_summary(
    *,
    active_contacts: int,
    contact_geom: np.ndarray,
    contact_frame: np.ndarray,
    contact_friction: np.ndarray,
    contact_dim: np.ndarray,
    contact_efc_address: np.ndarray,
    efc_force: np.ndarray,
    pyramidal: bool,
    geom_body_names: Mapping[int, str],
    geom_names: Mapping[int, str],
    contact_body_names: Sequence[str],
    object_name: str,
) -> dict[str, Any]:
    """Aggregate contact forces using the same body/object convention everywhere."""

    active = max(0, int(active_contacts))
    body_vectors = {name: np.zeros(3, dtype=np.float32) for name in contact_body_names}
    object_vector = np.zeros(3, dtype=np.float32)
    hand_object_pairs: list[dict[str, Any]] = []
    if active <= 0:
        return {
            "contact_force_vectors": np.zeros((len(contact_body_names), 3), dtype=np.float32),
            "object_force_vector": object_vector,
            "hand_object_pairs": hand_object_pairs,
            "active_contact_count": 0,
        }

    geoms = np.asarray(contact_geom)[:active]
    frames = np.asarray(contact_frame)[:active]
    frictions = np.asarray(contact_friction)[:active]
    dims = np.asarray(contact_dim)[:active]
    addresses = np.asarray(contact_efc_address)[:active]
    for contact_index, (geom1, geom2) in enumerate(geoms):
        geom1_id = int(geom1)
        geom2_id = int(geom2)
        body1 = geom_body_names.get(geom1_id, "")
        body2 = geom_body_names.get(geom2_id, "")
        force_world, normal_force = decode_contact_force_world(
            frame=frames[contact_index],
            friction=frictions[contact_index],
            condim=int(dims[contact_index]),
            efc_addresses=addresses[contact_index],
            efc_force=efc_force,
            pyramidal=pyramidal,
        )
        if not np.any(force_world):
            continue

        force_on_body1 = -force_world
        force_on_body2 = force_world
        if body1 in body_vectors:
            body_vectors[body1] += force_on_body1
        if body2 in body_vectors:
            body_vectors[body2] += force_on_body2
        if body1 == object_name:
            object_vector += force_on_body1
        elif body2 == object_name:
            object_vector += force_on_body2

        if body1 == object_name and body2 in body_vectors:
            hand_body = body2
            force_on_hand = force_on_body2
        elif body2 == object_name and body1 in body_vectors:
            hand_body = body1
            force_on_hand = force_on_body1
        else:
            continue
        hand_object_pairs.append(
            {
                "contact_index": int(contact_index),
                "body_name": hand_body,
                "force_world_N": [float(value) for value in force_on_hand],
                "force_magnitude_N": float(np.linalg.norm(force_on_hand)),
                "force_normal_N": float(abs(normal_force)),
                "geom1": geom1_id,
                "geom2": geom2_id,
                "geom1_name": geom_names.get(geom1_id, ""),
                "geom2_name": geom_names.get(geom2_id, ""),
                "body1": body1,
                "body2": body2,
            }
        )

    return {
        "contact_force_vectors": np.stack(
            [body_vectors[name] for name in contact_body_names], axis=0
        ),
        "object_force_vector": object_vector,
        "hand_object_pairs": hand_object_pairs,
        "active_contact_count": active,
    }
