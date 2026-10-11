"""Canonical geometry and kinematic tree derived from pinned ManoRL assets."""

import numpy as np
import torch
from .network import CanonicalBuffers, SkeletonTree
from .config import SCENE_SIGMA


def _balanced_kmeans(
    points: np.ndarray, clusters: int, rng: np.random.Generator
) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    count = len(points)
    if count % clusters:
        raise ValueError("balanced k-means requires equal integral capacities")
    capacity = count // clusters
    chosen = [int(rng.integers(count))]
    nearest = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    while len(chosen) < clusters:
        nearest[chosen] = -1.0
        chosen.append(int(np.argmax(nearest)))
        nearest = np.minimum(
            nearest, np.sum((points - points[chosen[-1]]) ** 2, axis=1)
        )
    centers = points[chosen].copy()
    previous = None
    for _ in range(128):
        distance = np.sum((points[:, None] - centers[None]) ** 2, axis=2)
        rows, columns = linear_sum_assignment(np.repeat(distance, capacity, axis=1))
        labels = np.empty(count, dtype=np.int64)
        labels[rows] = columns // capacity
        groups = np.stack(
            [np.flatnonzero(labels == cluster) for cluster in range(clusters)]
        )
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels
        centers = points[groups].mean(axis=1)
    return groups[np.argsort(groups[:, 0], kind="stable")]


def skeleton(model, body_ids):
    """Derive all 22 joint owners from compiled metadata, including stacked axes."""
    bone_of_body = {int(body): bone for bone, body in enumerate(body_ids)}
    joint_index = {
        joint: int(model.jnt_qposadr[joint]) - 6
        for joint in range(model.njnt)
        if 6 <= model.jnt_qposadr[joint] < 28
    }
    parents = np.full(22, -1, dtype=np.int64)
    children = parents.copy()
    edges = []
    depths = {0: 0}
    for child, body in enumerate(body_ids[1:], 1):
        chain = [int(body)]
        ancestor = int(model.body_parentid[body])
        while ancestor not in bone_of_body:
            if ancestor == 0:
                raise ValueError("hand skeleton is not rooted at palm")
            chain.append(ancestor)
            ancestor = int(model.body_parentid[ancestor])
        parent = bone_of_body[ancestor]
        joints = []
        for chain_body in reversed(chain):
            start = int(model.body_jntadr[chain_body])
            for joint in range(start, start + int(model.body_jntnum[chain_body])):
                if joint not in joint_index:
                    raise ValueError("non-finger joint on a bone edge")
                index = joint_index[joint]
                if parents[index] != -1:
                    raise ValueError("duplicate skeleton joint owner")
                parents[index], children[index] = parent, child
                joints.append(index)
        if not joints or parent not in depths:
            raise ValueError("invalid skeleton edge or bone order")
        edges.append((parent, child, tuple(joints)))
        depths[child] = depths[parent] + 1
    if len(edges) != 15 or np.any(parents < 0) or max(depths.values()) != 3:
        raise ValueError("expected 16 bones and a 22-joint depth-three skeleton")
    return SkeletonTree(tuple(edges), tuple(children.tolist()), tuple(parents.tolist()))


def build_canonical(object_type="cube2"):
    """Static collision geometry: same meshes/sample rule as the v5 runtime."""
    from sim.manorl.assets import compile_model_metadata_only, object_collision_vertices
    from sim.manorl.environment import (
        MjxWarpPhysicalProducer,
        _FINGERTIP_LOCAL_OFFSETS,
        _FINGERTIP_NAMES,
    )
    from sim.manorl.contracts import KEYPOINT_NAMES
    from sim.manorl.autonomy_v4 import (
        _deterministic_surface_samples,
        _mesh_body_geometry,
    )

    mujoco, model = compile_model_metadata_only(
        object_type=object_type, hand_side="right", physics_timestep=1 / 480
    )
    producer = MjxWarpPhysicalProducer(
        mujoco, model, object_type=object_type, hand_sides=("right",)
    )
    tree = skeleton(model, producer.keypoint_body_ids)
    hand = []
    for gid in producer.keypoint_geom_ids:
        vertices, faces = _mesh_body_geometry(model, gid)
        hand.append(_deterministic_surface_samples(vertices, faces, 128)[::8])
    vertices = np.asarray(object_collision_vertices(object_type)).reshape(-1, 3)
    points = _deterministic_surface_samples(
        vertices, np.arange(len(vertices)).reshape(-1, 3), 64
    )
    rng = np.random.default_rng(42)
    fine = _balanced_kmeans(points, 16, rng)
    coarse = _balanced_kmeans(points[fine].mean(1), 4, rng)
    coarse_of_fine = np.empty(16, dtype=np.int64)
    coarse_of_fine[coarse] = np.arange(4)[:, None]
    # Table bounds from compiled scene, retaining ManoRL's world coordinate frame.
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if table_id < 0 or model.geom_type[table_id] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("canonical desk requires the ManoRL floor plane")
    center = np.asarray(model.geom_pos[table_id])
    extent = np.asarray(model.geom_size[table_id])[:2]
    xs, ys = (
        np.linspace(center[k] - extent[k], center[k] + extent[k], 8) for k in (0, 1)
    )
    desk = np.asarray(
        [
            (xs[x + dx], ys[y + dy], center[2])
            for y in range(0, 8, 2)
            for x in range(0, 8, 2)
            for dy in range(2)
            for dx in range(2)
        ]
    )
    f = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float32)
    l = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.long)
    buffers = CanonicalBuffers(
        obj_points=f(points[None] / SCENE_SIGMA),
        fine_groups=l(fine),
        coarse_of_fine=l(coarse_of_fine),
        coarse_point_idx=l(fine[:, :2].reshape(-1)),
        coarse_pt_groups=l((coarse[:, :, None] * 2 + np.arange(2)).reshape(4, 8)),
        desk_points=f(desk / SCENE_SIGMA),
        desk_groups=l(np.arange(64).reshape(16, 4)),
        hand_points=f(np.asarray(hand) / SCENE_SIGMA),
        tip_bone=l([KEYPOINT_NAMES.index(name) for name in _FINGERTIP_NAMES]),
        tip_offsets=f(_FINGERTIP_LOCAL_OFFSETS / SCENE_SIGMA),
    )
    return buffers, tree
