#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "sim"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.ball_pit.common import BallPitSpec, physics_dt, spec_from_args  # noqa: E402
from common.contact_schema import (  # noqa: E402
    LANCE_GENERATED_SCHEMA,
    make_contact_entry,
    make_contact_pair,
    transform_force_to_local,
    validate_contact_sequence,
)
from sim.scene import (  # noqa: E402
    HAND_LINK_NAMES,
    build_ball_pit_scene_xml,
    hand_joint_addresses,
    set_hand_pose_mjx,
)

DEFAULT_SCENE_COPY = REPO_ROOT / "outputs/mjx_warp_contact_error_scene.xml"


def _device_get(jax: Any, value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _stats(values_m: list[float] | np.ndarray) -> dict[str, Any]:
    values = np.asarray(values_m, dtype=np.float64)
    mm = values * 1000.0
    if len(mm) == 0:
        return {
            "count": 0,
            "mean_mm": None,
            "std_mm": None,
            "min_mm": None,
            "median_mm": None,
            "p75_mm": None,
            "p90_mm": None,
            "p95_mm": None,
            "p99_mm": None,
            "max_mm": None,
        }
    pct = np.percentile(mm, [0, 50, 75, 90, 95, 99, 100])
    return {
        "count": int(len(mm)),
        "mean_mm": float(np.mean(mm)),
        "std_mm": float(np.std(mm)),
        "min_mm": float(pct[0]),
        "median_mm": float(pct[1]),
        "p75_mm": float(pct[2]),
        "p90_mm": float(pct[3]),
        "p95_mm": float(pct[4]),
        "p99_mm": float(pct[5]),
        "max_mm": float(pct[6]),
    }


def _point_triangle_dist2(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    p = points
    a, b, c = tri[0], tri[1], tri[2]
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ap @ ab
    d2 = ap @ ac
    out = np.empty((p.shape[0],), dtype=np.float64)
    done = np.zeros((p.shape[0],), dtype=bool)

    mask = (d1 <= 0.0) & (d2 <= 0.0)
    out[mask] = np.sum((p[mask] - a) ** 2, axis=1)
    done |= mask

    bp = p - b
    d3 = bp @ ab
    d4 = bp @ ac
    mask = (~done) & (d3 >= 0.0) & (d4 <= d3)
    out[mask] = np.sum((p[mask] - b) ** 2, axis=1)
    done |= mask

    vc = d1 * d4 - d3 * d2
    mask = (~done) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    if np.any(mask):
        v = d1[mask] / (d1[mask] - d3[mask])
        proj = a + v[:, None] * ab
        out[mask] = np.sum((p[mask] - proj) ** 2, axis=1)
    done |= mask

    cp = p - c
    d5 = cp @ ab
    d6 = cp @ ac
    mask = (~done) & (d6 >= 0.0) & (d5 <= d6)
    out[mask] = np.sum((p[mask] - c) ** 2, axis=1)
    done |= mask

    vb = d5 * d2 - d1 * d6
    mask = (~done) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    if np.any(mask):
        w = d2[mask] / (d2[mask] - d6[mask])
        proj = a + w[:, None] * ac
        out[mask] = np.sum((p[mask] - proj) ** 2, axis=1)
    done |= mask

    va = d3 * d6 - d5 * d4
    mask = (~done) & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    if np.any(mask):
        w = (d4[mask] - d3[mask]) / ((d4[mask] - d3[mask]) + (d5[mask] - d6[mask]))
        proj = b + w[:, None] * (c - b)
        out[mask] = np.sum((p[mask] - proj) ** 2, axis=1)
    done |= mask

    mask = ~done
    if np.any(mask):
        denom = va[mask] + vb[mask] + vc[mask]
        v = vb[mask] / denom
        w = vc[mask] / denom
        proj = a + ab * v[:, None] + ac * w[:, None]
        out[mask] = np.sum((p[mask] - proj) ** 2, axis=1)
    return out


def _point_to_mesh_distance(point: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> float:
    point_batch = np.asarray(point, dtype=np.float64).reshape(1, 3)
    min_dist2 = np.inf
    for face in faces:
        dist2 = float(_point_triangle_dist2(point_batch, vertices[face])[0])
        if dist2 < min_dist2:
            min_dist2 = dist2
    return float(np.sqrt(min_dist2))


def _array_for_world(value: np.ndarray, worldid: int) -> np.ndarray:
    if value.ndim >= 3 and value.shape[0] == 1:
        return value[int(worldid)]
    return value


def _geom_body_maps(mujoco: Any, model: Any) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    body_names: dict[int, str] = {}
    geom_names: dict[int, str] = {}
    geom_body_names: dict[int, str] = {}
    for body_id in range(model.nbody):
        body_names[body_id] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    for geom_id in range(model.ngeom):
        geom_names[geom_id] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        geom_body_names[geom_id] = body_names[int(model.geom_bodyid[geom_id])]
    return body_names, geom_names, geom_body_names


def _hand_collision_meshes(mujoco: Any, model: Any, geom_names: dict[int, str], geom_body_names: dict[int, str]) -> dict[int, dict[str, Any]]:
    meshes: dict[int, dict[str, Any]] = {}
    for geom_id in range(model.ngeom):
        body_name = geom_body_names[geom_id]
        geom_name = geom_names[geom_id]
        if body_name not in HAND_LINK_NAMES or not geom_name.endswith("_collision"):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        face_adr = int(model.mesh_faceadr[mesh_id])
        face_num = int(model.mesh_facenum[mesh_id])
        meshes[geom_id] = {
            "body_name": body_name,
            "geom_name": geom_name,
            "vertices": np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float64),
            "faces": np.asarray(model.mesh_face[face_adr : face_adr + face_num], dtype=np.int64),
        }
    return meshes


def _native_contact_force_rows(efc_force: np.ndarray, addresses: np.ndarray) -> list[float]:
    out: list[float] = []
    for raw_addr in np.ravel(addresses):
        addr = int(raw_addr)
        if 0 <= addr < len(efc_force):
            out.append(float(efc_force[addr]))
    return out


def _floating_base_offset(mujoco: Any, model: Any) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "floating_base")
    if body_id < 0:
        return [0.0, 0.0, 0.0]
    return [float(v) for v in model.body_pos[int(body_id)]]


def _hand_qpos_vector(qpos: np.ndarray, addresses: dict[str, tuple[int, int]], base_offset: list[float]) -> list[float]:
    ordered_names = ("ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz")
    out = [0.0] * 26
    for idx, name in enumerate(ordered_names):
        if name in addresses:
            out[idx] = float(qpos[addresses[name][0]])
    for idx, value in enumerate(base_offset[:3]):
        out[idx] += float(value)
    finger_names = sorted(name for name in addresses if name not in set(ordered_names))
    for out_idx, name in enumerate(finger_names[:20], start=6):
        out[out_idx] = float(qpos[addresses[name][0]])
    return out


def _group_contactbench_entries(frame_contacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame_contacts:
        key = (str(row["link_name"]), str(row["object_name"]))
        if key not in grouped:
            grouped[key] = {
                "pairs": [],
                "total_world": np.zeros(3, dtype=np.float64),
                "wrist_rotation": np.asarray(row["wrist_rotation_world_from_local"], dtype=np.float64),
                "joint_rotation": np.asarray(row["joint_rotation_world_from_local"], dtype=np.float64),
                "object_rotation": np.asarray(row["object_rotation_world_from_local"], dtype=np.float64),
            }
        force_normal = np.asarray(row["force_normal_proxy_world"], dtype=np.float64)
        pair = make_contact_pair(
            force_normal=force_normal.tolist(),
            force_tangential=[0.0, 0.0, 0.0],
            pos_world=row["pos_world"],
            pos_wrist=row["pos_wrist"],
            pos_joint=row["pos_joint"],
            pos_object=row["pos_object"],
        )
        grouped[key]["pairs"].append(pair)
        grouped[key]["total_world"] += force_normal

    entries: list[dict[str, Any]] = []
    for (joint_name, object_name), values in sorted(grouped.items()):
        total_world = values["total_world"].tolist()
        entries.append(
            make_contact_entry(
                joint_name=joint_name,
                object_name=object_name,
                total_force_world=total_world,
                total_force_wrist=transform_force_to_local(total_world, values["wrist_rotation"]),
                total_force_joint=transform_force_to_local(total_world, values["joint_rotation"]),
                total_force_object=transform_force_to_local(total_world, values["object_rotation"]),
                contact_pairs=values["pairs"],
            )
        )
    return entries


def _export_warp_contacts(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    requested_device = str(getattr(args, "device", "gpu"))
    if requested_device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif requested_device == "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    else:
        raise ValueError(f"unsupported MJX-Warp device: {requested_device}")

    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    actual_backend = jax.default_backend()
    if requested_device == "gpu" and actual_backend != "gpu":
        raise RuntimeError(f"expected JAX GPU backend, got {actual_backend} devices={jax.devices()}")
    if requested_device == "cpu" and actual_backend != "cpu":
        raise RuntimeError(f"expected JAX CPU backend, got {actual_backend} devices={jax.devices()}")

    spec = spec_from_args(args)
    scene_path = build_ball_pit_scene_xml(Path(tempfile.mkdtemp(prefix="contactbench_mjx_warp_contact_")) / "ball_pit.xml", spec)
    args.scene_copy.parent.mkdir(parents=True, exist_ok=True)
    args.scene_copy.write_text(scene_path.read_text(encoding="utf-8"), encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    hand_addresses = hand_joint_addresses(mujoco, model)
    _, geom_names, geom_body_names = _geom_body_maps(mujoco, model)
    hand_meshes = _hand_collision_meshes(mujoco, model, geom_names, geom_body_names)
    object_names = tuple(f"ball_{idx:03d}" for idx in range(spec.ball_count))
    object_name_set = set(object_names)
    object_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in object_names]
    if any(body_id < 0 for body_id in object_body_ids):
        missing = [name for name, body_id in zip(object_names, object_body_ids) if body_id < 0]
        raise RuntimeError(f"missing object bodies: {missing[:8]}")
    palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    if palm_body_id < 0:
        raise RuntimeError("missing palm body")
    hand_base_offset = _floating_base_offset(mujoco, model)

    mx = mjx.put_model(model, impl="warp")
    dx = mjx.make_data(model, impl="warp", naconmax=args.naconmax, njmax=args.njmax)
    step = jax.jit(mjx.step)

    contacts_by_frame: list[list[dict[str, Any]]] = []
    contactbench_by_frame: list[list[dict[str, Any]]] = []
    object_positions_by_frame: list[list[list[float]]] = []
    hand_trajectory_frames: list[dict[str, Any]] = []
    all_hand_dist: list[float] = []
    all_ball_dist: list[float] = []
    all_contact_dist: list[float] = []
    all_normal_force_proxy: list[float] = []
    by_link_hand_dist: dict[str, list[float]] = defaultdict(list)
    by_link_ball_dist: dict[str, list[float]] = defaultdict(list)
    by_link_contact_dist: dict[str, list[float]] = defaultdict(list)
    largest_contacts: list[dict[str, Any]] = []

    start = time.perf_counter()
    dx = set_hand_pose_mjx(jnp, dx, 0, spec, hand_addresses)
    for _ in range(spec.settle_frames * spec.substeps):
        dx = set_hand_pose_mjx(jnp, dx, 0, spec, hand_addresses)
        dx = step(mx, dx)
        dx = set_hand_pose_mjx(jnp, dx, 0, spec, hand_addresses)
    jax.block_until_ready(dx.qpos)

    for frame in range(spec.rollout_frames):
        dx = set_hand_pose_mjx(jnp, dx, frame, spec, hand_addresses)
        for _ in range(spec.substeps):
            dx = set_hand_pose_mjx(jnp, dx, frame, spec, hand_addresses)
            dx = step(mx, dx)
            dx = set_hand_pose_mjx(jnp, dx, frame, spec, hand_addresses)
        jax.block_until_ready(dx.qpos)

        impl = dx._impl
        nacon = _device_get(jax, impl.nacon)
        active_contacts = int(nacon[0]) if nacon.shape else int(nacon)
        contact_geom = _device_get(jax, impl.contact__geom)[:active_contacts]
        contact_pos = _device_get(jax, impl.contact__pos)[:active_contacts]
        contact_dist = _device_get(jax, impl.contact__dist)[:active_contacts]
        contact_frame = _device_get(jax, impl.contact__frame)[:active_contacts]
        contact_worldid = _device_get(jax, impl.contact__worldid)[:active_contacts]
        contact_efc_address = _device_get(jax, impl.contact__efc_address)[:active_contacts]
        contact_dim = _device_get(jax, impl.contact__dim)[:active_contacts]
        efc_force = _device_get(jax, impl.efc__force)
        geom_xpos_all = _device_get(jax, dx.geom_xpos)
        geom_xmat_all = _device_get(jax, dx.geom_xmat)
        xpos_all = _device_get(jax, dx.xpos)
        xmat_all = _device_get(jax, dx.xmat)
        qpos_all = _device_get(jax, dx.qpos)

        frame_contacts: list[dict[str, Any]] = []
        for contact_index in range(active_contacts):
            worldid = int(contact_worldid[contact_index]) if len(contact_worldid) else 0
            geom1 = int(contact_geom[contact_index, 0])
            geom2 = int(contact_geom[contact_index, 1])
            body1 = geom_body_names.get(geom1, "")
            body2 = geom_body_names.get(geom2, "")
            if body1 in HAND_LINK_NAMES and body2 in object_name_set:
                hand_geom_id, object_geom_id = geom1, geom2
                link_name, object_name = body1, body2
            elif body2 in HAND_LINK_NAMES and body1 in object_name_set:
                hand_geom_id, object_geom_id = geom2, geom1
                link_name, object_name = body2, body1
            else:
                continue
            if hand_geom_id not in hand_meshes:
                continue

            geom_xpos = _array_for_world(geom_xpos_all, worldid)
            geom_xmat = _array_for_world(geom_xmat_all, worldid)
            xpos = _array_for_world(xpos_all, worldid)
            xmat = _array_for_world(xmat_all, worldid)
            pos_world = np.asarray(contact_pos[contact_index], dtype=np.float64)

            hand_geom_origin = np.asarray(geom_xpos[hand_geom_id], dtype=np.float64)
            hand_geom_rotation = np.asarray(geom_xmat[hand_geom_id], dtype=np.float64).reshape(3, 3)
            pos_hand_geom = hand_geom_rotation.T @ (pos_world - hand_geom_origin)
            hand_mesh = hand_meshes[hand_geom_id]
            distance_to_hand_mesh = _point_to_mesh_distance(pos_hand_geom, hand_mesh["vertices"], hand_mesh["faces"])

            hand_body_id = int(model.geom_bodyid[hand_geom_id])
            object_body_id = int(model.geom_bodyid[object_geom_id])
            wrist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
            hand_body_origin = np.asarray(xpos[hand_body_id], dtype=np.float64)
            hand_body_rotation = np.asarray(xmat[hand_body_id], dtype=np.float64).reshape(3, 3)
            object_body_origin = np.asarray(xpos[object_body_id], dtype=np.float64)
            object_body_rotation = np.asarray(xmat[object_body_id], dtype=np.float64).reshape(3, 3)
            wrist_origin = np.asarray(xpos[wrist_body_id], dtype=np.float64)
            wrist_rotation = np.asarray(xmat[wrist_body_id], dtype=np.float64).reshape(3, 3)
            pos_joint = hand_body_rotation.T @ (pos_world - hand_body_origin)
            pos_object = object_body_rotation.T @ (pos_world - object_body_origin)
            pos_wrist = wrist_rotation.T @ (pos_world - wrist_origin)

            ball_center = np.asarray(geom_xpos[object_geom_id], dtype=np.float64)
            ball_radius = float(model.geom_size[object_geom_id, 0])
            signed_distance_to_ball_surface = float(np.linalg.norm(pos_world - ball_center) - ball_radius)
            efc_rows = _native_contact_force_rows(efc_force, contact_efc_address[contact_index])
            normal_force_proxy = abs(efc_rows[0]) if efc_rows else 0.0
            contact_frame_world = np.asarray(contact_frame[contact_index], dtype=np.float64).reshape(3, 3)
            normal_axis_world = contact_frame_world[0]
            force_normal_proxy = (normal_axis_world * normal_force_proxy).tolist()

            row = {
                "frame": int(frame),
                "contact_index": int(contact_index),
                "worldid": int(worldid),
                "link_name": link_name,
                "object_name": object_name,
                "geom1": int(geom1),
                "geom2": int(geom2),
                "hand_geom_id": int(hand_geom_id),
                "object_geom_id": int(object_geom_id),
                "hand_geom_name": geom_names[hand_geom_id],
                "object_geom_name": geom_names[object_geom_id],
                "pos_world": [float(v) for v in pos_world],
                "pos_wrist": [float(v) for v in pos_wrist],
                "pos_joint": [float(v) for v in pos_joint],
                "pos_object": [float(v) for v in pos_object],
                "pos_hand_geom": [float(v) for v in pos_hand_geom],
                "wrist_rotation_world_from_local": wrist_rotation.tolist(),
                "joint_rotation_world_from_local": hand_body_rotation.tolist(),
                "object_rotation_world_from_local": object_body_rotation.tolist(),
                "contact_frame": contact_frame_world.tolist(),
                "contact_dist_m": float(contact_dist[contact_index]),
                "contact_dim": int(contact_dim[contact_index]),
                "efc_address": [int(v) for v in np.ravel(contact_efc_address[contact_index])],
                "efc_force_rows": efc_rows,
                "normal_force_proxy_N": float(normal_force_proxy),
                "force_normal_proxy_world": [float(v) for v in force_normal_proxy],
                "distance_to_hand_mesh_m": float(distance_to_hand_mesh),
                "signed_distance_to_ball_surface_m": float(signed_distance_to_ball_surface),
                "abs_distance_to_ball_surface_m": abs(float(signed_distance_to_ball_surface)),
            }
            frame_contacts.append(row)
            all_hand_dist.append(distance_to_hand_mesh)
            all_ball_dist.append(abs(float(signed_distance_to_ball_surface)))
            all_contact_dist.append(float(contact_dist[contact_index]))
            all_normal_force_proxy.append(float(normal_force_proxy))
            by_link_hand_dist[link_name].append(distance_to_hand_mesh)
            by_link_ball_dist[link_name].append(abs(float(signed_distance_to_ball_surface)))
            by_link_contact_dist[link_name].append(float(contact_dist[contact_index]))
            if len(largest_contacts) < 25 or distance_to_hand_mesh > min(c["distance_to_hand_mesh_m"] for c in largest_contacts):
                largest_contacts.append(row)
                largest_contacts.sort(key=lambda item: item["distance_to_hand_mesh_m"], reverse=True)
                largest_contacts = largest_contacts[:25]

        contacts_by_frame.append(frame_contacts)
        contactbench_by_frame.append(_group_contactbench_entries(frame_contacts))

        xpos0 = _array_for_world(xpos_all, 0)
        qpos0 = qpos_all[0] if qpos_all.ndim == 2 and qpos_all.shape[0] == 1 else qpos_all
        palm_pos = [float(v) for v in np.asarray(xpos0[palm_body_id], dtype=np.float64)]
        urdf_dof = _hand_qpos_vector(np.asarray(qpos0, dtype=np.float64), hand_addresses, hand_base_offset)
        hand_trajectory_frames.append(
            {
                "mano_global_pos": palm_pos,
                "mano_global_rot_aa": urdf_dof[3:6],
                "mano_hand_pose": urdf_dof[3:6] + [0.0] * 45,
                "mano_joint_pos": [palm_pos for _ in range(21)],
                "urdf_dof": urdf_dof,
            }
        )
        object_positions_by_frame.append(
            [[float(v) for v in np.asarray(xpos0[body_id], dtype=np.float64)] for body_id in object_body_ids]
        )

        if args.progress_interval and frame % args.progress_interval == 0:
            print(
                f"frame={frame} hand_ball_contacts={len(frame_contacts)} elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )

    contact_counts = [len(frame) for frame in contacts_by_frame]
    error_stats = {
        "distance_definition": "unsigned nearest distance from raw MJX-Warp _impl.contact__pos to the corresponding hand collision geom surface; contact point transformed into geom-local coordinates with dx.geom_xpos/dx.geom_xmat",
        "distance_to_hand_mesh": _stats(all_hand_dist),
        "abs_distance_to_ball_surface": _stats(all_ball_dist),
        "contact_dist": _stats(all_contact_dist),
        "normal_force_proxy": _stats(all_normal_force_proxy),
        "contact_counts_by_frame": {
            "frames": int(len(contact_counts)),
            "nonempty_frames": int(np.count_nonzero(contact_counts)),
            "max_contacts_in_frame": int(np.max(contact_counts)) if contact_counts else 0,
            "total_contacts": int(np.sum(contact_counts)),
        },
        "by_link": {
            link_name: {
                "distance_to_hand_mesh": _stats(by_link_hand_dist[link_name]),
                "abs_distance_to_ball_surface": _stats(by_link_ball_dist[link_name]),
                "contact_dist": _stats(by_link_contact_dist[link_name]),
            }
            for link_name in sorted(by_link_hand_dist)
        },
        "largest_distance_contacts": [
            {
                key: value
                for key, value in row.items()
                if key
                in {
                    "frame",
                    "contact_index",
                    "link_name",
                    "object_name",
                    "pos_world",
                    "contact_dist_m",
                    "normal_force_proxy_N",
                    "distance_to_hand_mesh_m",
                    "signed_distance_to_ball_surface_m",
                    "abs_distance_to_ball_surface_m",
                }
            }
            for row in largest_contacts
        ],
    }

    contact_entry_count = int(sum(len(frame) for frame in contactbench_by_frame))
    contact_pair_count = int(sum(len(entry["contact_pairs"]) for frame in contactbench_by_frame for entry in frame))
    validate_contact_sequence(contactbench_by_frame)

    metadata = {
        "backend": "mjx_warp",
        "device": requested_device,
        "benchmark_case": spec.case_name,
        "source": f"MJX-Warp {requested_device} native contact export from Data._impl contact buffers",
        "api_stability_note": "Uses MJX-Warp internal _impl fields; not the stable MuJoCo CPU data.contact/mj_contactForce API.",
        "force_convention": "normal_force_proxy_on_hand_link_by_ball_from_mjx_warp_efc_first_row",
        "force_note": "ContactBench export stores a normal-force proxy from the first contact constraint row and zero tangential force. Native efc rows are preserved in the native sidecar. It is not a MuJoCo mj_contactForce 6D wrench.",
        "contact_position_note": "pos_world is raw MJX-Warp _impl.contact__pos. It is not projected onto the ball or hand mesh.",
        "object_trajectory_note": "All balls are exported from MJX-Warp dx.xpos. rot_aa is zero because sphere orientation is contact-invariant here.",
        "hand_trajectory_note": "hands fields are exported from MJX-Warp state. urdf_dof translation includes the MJCF floating_base offset so URDF mesh replay aligns with MuJoCo world contacts.",
        "hand_base_offset": hand_base_offset,
        "scene_xml": str(scene_path),
        "scene_copy": str(args.scene_copy),
        "mujoco_version": mujoco.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "ball_count": spec.ball_count,
        "ball_radius": spec.ball_radius,
        "fps": spec.fps,
        "substeps": spec.substeps,
        "physics_dt": physics_dt(spec),
        "settle_frames": spec.settle_frames,
        "rollout_frames": spec.rollout_frames,
        "duration_seconds": spec.duration_seconds,
        "frames": spec.rollout_frames,
        "nonempty_frames": int(np.count_nonzero(contact_counts)),
        "contact_entries": contact_entry_count,
        "contact_pairs": contact_pair_count,
        "naconmax": args.naconmax,
        "njmax": args.njmax,
    }
    native_payload = {
        "schema": "contactbench.mjx_warp_native_contact.v1",
        "metadata": metadata,
        "error_stats": error_stats,
        "contact": contacts_by_frame,
    }
    hand_trajectory = {
        key: [frame[key] for frame in hand_trajectory_frames]
        for key in ("mano_global_pos", "mano_global_rot_aa", "mano_hand_pose", "mano_joint_pos", "urdf_dof")
    }
    object_trajectories = [
        {
            "object_name": name,
            "pos": [object_positions_by_frame[frame][idx] for frame in range(len(object_positions_by_frame))],
            "rot_aa": [[0.0, 0.0, 0.0] for _ in range(len(object_positions_by_frame))],
        }
        for idx, name in enumerate(object_names)
    ]
    contactbench_payload = {
        "schema": LANCE_GENERATED_SCHEMA,
        "note": "MJX-Warp ball-pit hand-ball native contact export converted to ContactBench contact shape. Forces are normal-force proxies, not MuJoCo mj_contactForce 6D wrenches.",
        "metadata": metadata,
        "hand_trajectory": hand_trajectory,
        "object_trajectories": object_trajectories,
        "contact": contactbench_by_frame,
    }
    return native_payload, {"metadata": metadata, **error_stats}, contactbench_payload
