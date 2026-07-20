"""Gym-like DexHandRL MJX environment for skrl-style training."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import yaml

try:  # pragma: no cover - optional dependency
    from gymnasium import spaces as gym_spaces
except ImportError:  # pragma: no cover - optional dependency
    gym_spaces = None

from sim.dexhandrl.constants import (
    ACTION_DIM_021PRO,
    DEFAULT_ISAAC_SOURCE_ROOT,
    DEFAULT_LANCE_PATH,
    DEXHAND021PRO_CONTACT_BODY_NAMES,
    DEXHAND021PRO_FINGERTIP_BODY_NAMES,
    POINT_CLOUD_DIM_021PRO,
)
from sim.dexhandrl.contact import collect_contact_summary, decode_contact_force_world
from sim.dexhandrl.hand_mapping import (
    DEXHAND021PRO_FULL_DOF_NAMES,
    active_to_full_dof,
    dexhand021pro_route_joint_upper_limits,
    full_dof_from_qpos,
    full_to_active_dof,
    make_ctrl_vector,
    resolve_mujoco_ids,
    raw_finger_to_full_dof,
    set_hand_qpos,
    set_object_freejoint,
    wxyz_to_xyzw,
)
from sim.dexhandrl.lance_loader import DexHandTrajectory, load_trajectory
from sim.dexhandrl.parity import (
    add_termination_reward,
    apply_reward_weights,
    build_minimal_observation,
    encode_geometry_12d,
    geometry_from_isaac_urdf,
    load_isaac_reward_contract,
    object_tracking_rewards,
    rotation_angle_error_deg,
    rotation_tracking_reward,
)
from sim.dexhandrl.scene import build_dexhandrl_scene_xml


_SIMPLE_CONTACT_TO_BODY = {
    "palm": "RFH1",
    "thumb1": "RH0_0",
    "thumb2": "RH0_2",
    "thumb3": "RH0_3",
    "index1": "RH1_0",
    "index2": "RH1_2",
    "index3": "RH1_3",
    "middle1": "RH2_0",
    "middle2": "RH2_2",
    "middle3": "RH2_3",
    "ring1": "RH3_0",
    "ring2": "RH3_2",
    "ring3": "RH3_3",
    "pinky1": "RH4_0",
    "pinky2": "RH4_2",
    "pinky3": "RH4_3",
}

_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_FINGER_TO_JOINT_INDICES_021PRO = {
    "thumb": [0, 1, 2, 3],
    "index": [4, 5, 6],
    "middle": [7, 8, 9],
    "ring": [10, 11, 12],
    "pinky": [13, 14, 15],
}
_ACTIVATED_CONTROL_PRIMARY_DOF_INDICES = (6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24)

# Keep the historical private import path available to parity tests and callers.
_decode_contact_force_world = decode_contact_force_world


@dataclass(frozen=True)
class SimpleBoxSpace:
    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: np.dtype = np.float32


@dataclass
class DexHandRLMJXEnvConfig:
    lance_path: Path = DEFAULT_LANCE_PATH
    object_name: str = "cube1"
    action: str = "01"
    sequence_index: int = 0
    uuid: str | None = None
    device: str = "cpu"
    use_residual: bool = True
    dt: float = 0.01
    substeps: int = 10
    settle_frames: int = 0
    pre_contact_frames: int = 200
    post_contact_frames: int = 250
    object_reset_z_offset: float = 0.01
    reference_joint_source: str = "active"
    early_phase_frames: int = 130
    object_position_terminal_threshold: float = 0.1
    termination_failure_penalty: float = -200.0
    termination_timeout_penalty: float = 0.0
    alive_reward: float = 0.001
    contact_force_threshold: float = 5.0
    object_stability_velocity_scale: float = 0.1
    max_contact_reward: float = 0.4
    max_object_stability_reward: float = 0.4
    fingertip_inside_object_min_depth: float = 0.001
    fingertip_inside_object_max_depth: float = 0.003
    max_fingertip_inside_object_penalty: float = 0.4
    object_friction: str = "0.5 0.01 0.001"
    ground_friction: str = "0.5 0.5 0"
    base_pos_kp: float = 5000.0
    base_pos_kv: float = 500.0
    base_pos_force: float = 500.0
    base_rot_kp: float = 10000.0
    base_rot_kv: float = 125.0
    base_rot_force: float = 100.0
    finger_kp: float = 15.0
    finger_kv: float = 0.0
    finger_force: float = 50.0
    hand_self_collision_mode: str = "disable_within_finger"
    base_position_decay: tuple[float, float, float] = (0.9, 0.9, 0.9)
    post_contact_base_position_decay: tuple[float, float, float] = (0.9, 0.9, 0.9)
    joint_decay: float = 0.9
    post_contact_joint_decay: float = 1.0
    base_position_scales: tuple[float, float, float] = (0.002, 0.002, 0.003)
    base_position_max_offset: float = 0.04
    naconmax: int = 8192
    njmax: int = 8192
    seed: int | None = 0
    isaac_source_root: Path | None = None
    contact_bodies_file: Path | None = None
    task_config_path: Path | None = None

    def __post_init__(self) -> None:
        root = Path(self.isaac_source_root or DEFAULT_ISAAC_SOURCE_ROOT)
        self.isaac_source_root = root
        if self.contact_bodies_file is None:
            self.contact_bodies_file = root / "assets/all_assets/Assets/object_grasps_simple.yaml"
        else:
            self.contact_bodies_file = Path(self.contact_bodies_file)
        if self.task_config_path is None:
            self.task_config_path = root / "dexhand_env/cfg/task/Dexhand021proReconstruction.yaml"
        else:
            self.task_config_path = Path(self.task_config_path)


def _make_box_space(low: float, high: float, shape: tuple[int, ...], dtype=np.float32):
    low_arr = np.full(shape, low, dtype=dtype)
    high_arr = np.full(shape, high, dtype=dtype)
    if gym_spaces is not None:  # pragma: no cover - optional dependency
        return gym_spaces.Box(low=low_arr, high=high_arr, dtype=dtype)
    return SimpleBoxSpace(low=low_arr, high=high_arr, shape=shape, dtype=np.dtype(dtype))


def _hash_seed(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFFFFFF


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _rotate_points_wxyz(points: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    rot = _quat_wxyz_to_matrix(quat_wxyz)
    return np.asarray(points, dtype=np.float32) @ rot.T


def _rotate_points_inverse_wxyz(points: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    rot = _quat_wxyz_to_matrix(quat_wxyz)
    return np.asarray(points, dtype=np.float32) @ rot


@lru_cache(maxsize=32)
def _load_obj_mesh(mesh_path: str, scale: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    scale_arr = np.asarray(scale, dtype=np.float32)
    with Path(mesh_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append(
                        [
                            float(parts[1]) * float(scale_arr[0]),
                            float(parts[2]) * float(scale_arr[1]),
                            float(parts[3]) * float(scale_arr[2]),
                        ]
                    )
            elif line.startswith("f "):
                parts = line.split()[1:]
                indices: list[int] = []
                for item in parts:
                    token = item.split("/")[0]
                    if token:
                        indices.append(int(token) - 1)
                if len(indices) >= 3:
                    for idx in range(1, len(indices) - 1):
                        faces.append((indices[0], indices[idx], indices[idx + 1]))
    vertices_arr = np.asarray(vertices, dtype=np.float32)
    faces_arr = np.asarray(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    return vertices_arr, faces_arr


def _sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if vertices.size == 0:
        raise RuntimeError("mesh has no vertices")
    if faces.size == 0:
        indices = rng.choice(len(vertices), size=num_points, replace=True)
        return vertices[indices]
    tris = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1
    )
    if float(areas.sum()) <= 0.0:
        indices = rng.choice(len(vertices), size=num_points, replace=True)
        return vertices[indices]

    # This mirrors trimesh.sample.sample_surface, which IsaacGym uses for
    # irregular-object point clouds.
    random = rng.random
    weight_cum = np.cumsum(areas)
    face_pick = random(num_points) * weight_cum[-1]
    face_index = np.searchsorted(weight_cum, face_pick)
    tri_origins = vertices[faces[:, 0]]
    tri_vectors = vertices[faces[:, 1:]].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    random_lengths = random((len(tri_vectors), 2, 1))
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    return (sample_vector + tri_origins).astype(np.float32)


def _farthest_point_sample(
    points: np.ndarray,
    num_points: int,
    seed: int | None = None,
) -> np.ndarray:
    """Downsample an oversampled surface cloud with deterministic FPS."""

    source = np.asarray(points, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {source.shape}")
    if source.shape[0] == 0:
        raise ValueError("points must contain at least one point")
    if num_points < 1:
        raise ValueError(f"num_points must be positive, got {num_points}")
    if num_points > source.shape[0]:
        raise ValueError(
            f"cannot sample {num_points} points from {source.shape[0]} source points"
        )

    selected = np.empty(num_points, dtype=np.int64)
    selected[0] = 0 if seed is None else int(seed) % source.shape[0]
    min_dists = np.full(source.shape[0], np.inf, dtype=np.float32)
    for index in range(1, num_points):
        last = source[selected[index - 1]]
        distances = np.sum((source - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, distances)
        selected[index] = int(np.argmax(min_dists))
    return source[selected]


def _sample_box_surface(size: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    half = 0.5 * np.asarray(size, dtype=np.float32).reshape(3)
    areas = np.asarray(
        [
            half[1] * half[2],
            half[1] * half[2],
            half[0] * half[2],
            half[0] * half[2],
            half[0] * half[1],
            half[0] * half[1],
        ],
        dtype=np.float64,
    )
    probs = areas / max(float(areas.sum()), 1e-8)
    face_ids = rng.choice(6, size=num_points, replace=True, p=probs)
    uv = rng.uniform(-1.0, 1.0, size=(num_points, 2)).astype(np.float32)
    pts = np.zeros((num_points, 3), dtype=np.float32)
    for face_id in range(6):
        mask = face_ids == face_id
        if not np.any(mask):
            continue
        uv_face = uv[mask]
        if face_id == 0:
            pts[mask] = np.column_stack([np.full(mask.sum(), half[0]), uv_face[:, 0] * half[1], uv_face[:, 1] * half[2]])
        elif face_id == 1:
            pts[mask] = np.column_stack([np.full(mask.sum(), -half[0]), uv_face[:, 0] * half[1], uv_face[:, 1] * half[2]])
        elif face_id == 2:
            pts[mask] = np.column_stack([uv_face[:, 0] * half[0], np.full(mask.sum(), half[1]), uv_face[:, 1] * half[2]])
        elif face_id == 3:
            pts[mask] = np.column_stack([uv_face[:, 0] * half[0], np.full(mask.sum(), -half[1]), uv_face[:, 1] * half[2]])
        elif face_id == 4:
            pts[mask] = np.column_stack([uv_face[:, 0] * half[0], uv_face[:, 1] * half[1], np.full(mask.sum(), half[2])])
        else:
            pts[mask] = np.column_stack([uv_face[:, 0] * half[0], uv_face[:, 1] * half[1], np.full(mask.sum(), -half[2])])
    return pts


def _sample_cylinder_surface(size: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    radius = float(size[0])
    length = float(size[1])
    side_area = 2.0 * np.pi * radius * length
    cap_area = 2.0 * np.pi * radius * radius
    probs = np.asarray([side_area, cap_area, cap_area], dtype=np.float64)
    probs = probs / max(float(probs.sum()), 1e-8)
    region = rng.choice(3, size=num_points, replace=True, p=probs)
    pts = np.zeros((num_points, 3), dtype=np.float32)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=num_points).astype(np.float32)
    z = rng.uniform(-0.5 * length, 0.5 * length, size=num_points).astype(np.float32)
    r = np.sqrt(rng.random(num_points)).astype(np.float32) * radius
    xy = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).astype(np.float32)
    side_mask = region == 0
    if np.any(side_mask):
        pts[side_mask, 0:2] = np.column_stack([radius * np.cos(theta[side_mask]), radius * np.sin(theta[side_mask])])
        pts[side_mask, 2] = z[side_mask]
    top_mask = region == 1
    if np.any(top_mask):
        pts[top_mask, 0:2] = xy[top_mask]
        pts[top_mask, 2] = 0.5 * length
    bottom_mask = region == 2
    if np.any(bottom_mask):
        pts[bottom_mask, 0:2] = xy[bottom_mask]
        pts[bottom_mask, 2] = -0.5 * length
    return pts


def _sample_sphere_surface(radius: float, num_points: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=(num_points, 3)).astype(np.float32)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return radius * vec / norm


def _sample_object_point_cloud(
    geometry: dict[str, object],
    object_name: str,
    isaac_source_root: Path,
    num_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    geom_type = str(geometry["type"])
    size = np.asarray(geometry["size"], dtype=np.float32)
    if geom_type == "box":
        return _sample_box_surface(size, num_points, rng)
    if geom_type == "cylinder":
        return _sample_cylinder_surface(size, num_points, rng)
    if geom_type == "sphere":
        return _sample_sphere_surface(float(size[0]), num_points, rng)
    mesh_path = geometry.get("mesh_path")
    if not mesh_path:
        raise RuntimeError(f"irregular object {object_name!r} has no mesh_path")
    scale = tuple(float(v) for v in geometry.get("scale", (1.0, 1.0, 1.0)))
    vertices, faces = _load_obj_mesh(str(mesh_path), scale)
    oversampled = _sample_mesh_surface(vertices, faces, num_points * 10, rng)
    return _farthest_point_sample(oversampled, num_points, seed=42)


def _contact_body_mapping_from_file(contact_file: Path) -> dict[str, dict[str, list[str]]]:
    body_yaml = yaml.safe_load(contact_file.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, list[str]]] = {}
    for object_name, actions in body_yaml.items():
        if not isinstance(actions, dict):
            continue
        mapping[object_name] = {}
        for action_id, simple_names in actions.items():
            if isinstance(simple_names, str):
                simple_list = [simple_names]
            else:
                simple_list = list(simple_names or [])
            body_names = []
            for simple_name in simple_list:
                body_name = _SIMPLE_CONTACT_TO_BODY.get(str(simple_name))
                if body_name is None:
                    raise RuntimeError(f"unsupported simplified contact body name: {simple_name!r}")
                body_names.append(body_name)
            mapping[object_name][str(action_id)] = body_names
    return mapping


def _rotation_disabled_from_config(config_path: Path, object_name: str, action: str) -> bool:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    task = data.get("task", {})
    disable_map = task.get("disableObjectRotation", {}) or {}
    value = disable_map.get(object_name)
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() == "all"
    return str(action) in {str(item) for item in value}


def _normalize_contact_force(values: np.ndarray, max_contact_force: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32) / float(max_contact_force), 0.0, 1.0)


def _normalize_active_positions(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    denom = np.clip(upper - lower, 1e-6, None)
    normalized = 2.0 * (np.asarray(values, dtype=np.float32) - lower) / denom - 1.0
    return np.clip(normalized, -1.0, 1.0)


def _mask_contact_vectors(vectors: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply IsaacGym's per-action contact-body mask to force vectors."""

    force_vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, 3)
    contact_mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    if force_vectors.shape[0] != contact_mask.shape[0]:
        raise ValueError(
            "contact vector/body mask mismatch: "
            f"vectors={force_vectors.shape}, mask={contact_mask.shape}"
        )
    return force_vectors * contact_mask[:, None]


def _reference_full_targets(
    traj: DexHandTrajectory,
    frame: int,
    source: str,
    current_full: np.ndarray | None,
    route_joint_upper_limits: dict[int, tuple[float, float]],
) -> np.ndarray:
    if source == "active":
        return active_to_full_dof(
            traj.hand_dof[frame],
            current_full=current_full,
            route_joint_upper_limits=route_joint_upper_limits,
        )
    if source == "full_raw":
        if traj.hand_finger_dof_full_raw is None:
            raise RuntimeError("Lance row does not contain dexhand_joint_angles_full_raw")
        return raw_finger_to_full_dof(traj.hand_dof[frame], traj.hand_finger_dof_full_raw[frame])
    raise ValueError(f"unsupported reference joint source {source!r}")


class DexHandRLMJXEnv:
    """Single-trajectory DexHandRL MJX environment.

    The environment exposes the same 22D action and 461D observation layout as
    the IsaacGym reconstruction task. It is intentionally lightweight: one
    loaded Lance trajectory, one MJX state, gym-like reset/step, and skrl-ready
    spaces.
    """

    metadata = {"render_modes": [], "render_fps": 100}
    render_mode = None
    num_actions = ACTION_DIM_021PRO
    num_observations = 461

    def __init__(self, cfg: DexHandRLMJXEnvConfig):
        self.cfg = cfg
        self.reward_contract = load_isaac_reward_contract(cfg.task_config_path)
        self.reward_weights = self.reward_contract.reward_weights
        supported_reward_terms = {
            "alive",
            "object_pos_tracking_x",
            "object_pos_tracking_y",
            "object_pos_tracking_z",
            "contact_quality",
            "object_stability_velocity",
            "object_rot_tracking",
            "fingertip_inside_object_penalty",
            "action_penalty_position",
            "action_penalty_joint",
            "termination_success",
            "termination_failure_penalty",
            "termination_timeout_penalty",
        }
        unsupported = sorted(
            name
            for name, weight in self.reward_weights.items()
            if weight != 0.0 and name not in supported_reward_terms
        )
        if unsupported:
            raise RuntimeError(
                "MuJoCo reward parity does not implement enabled IsaacGym terms: "
                f"{unsupported}"
            )
        if cfg.device == "cpu":
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        elif cfg.device == "gpu":
            os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        else:
            raise ValueError(f"unsupported MJX device: {cfg.device!r}")

        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx

        self.jax = jax
        self.jnp = jnp
        self.mujoco = mujoco
        self.mjx = mjx

        actual_backend = jax.default_backend()
        if cfg.device == "cpu" and actual_backend != "cpu":
            raise RuntimeError(f"expected JAX CPU backend, got {actual_backend}; devices={jax.devices()}")
        if cfg.device == "gpu" and actual_backend != "gpu":
            raise RuntimeError(f"expected JAX GPU backend, got {actual_backend}; devices={jax.devices()}")

        self.trajectory = load_trajectory(
            lance_path=cfg.lance_path,
            object_name=cfg.object_name,
            action=cfg.action,
            sequence_index=cfg.sequence_index,
            uuid=cfg.uuid,
        )
        self.start_frame = max(0, self.trajectory.movement_start_frame - cfg.pre_contact_frames)
        self.end_frame = min(
            self.trajectory.frames - 1,
            self.trajectory.movement_end_frame + cfg.post_contact_frames,
        )
        self.episode_length_frames = max(1, self.end_frame - self.start_frame + 1)
        self.last_contact_frame = int(self.trajectory.movement_end_frame)
        self.max_episode_time_s = float(self.episode_length_frames * cfg.dt)

        self.geometry = geometry_from_isaac_urdf(cfg.object_name, isaac_source_root=cfg.isaac_source_root)
        self.geometry_encoding = encode_geometry_12d(self.geometry)
        self.object_primitive_type, self.object_primitive_params = self._infer_object_primitive()
        self.object_point_cloud_local = _sample_object_point_cloud(
            self.geometry,
            object_name=cfg.object_name,
            isaac_source_root=cfg.isaac_source_root,
            num_points=POINT_CLOUD_DIM_021PRO // 3,
            seed=42,
        )

        scene_path = Path(tempfile.mkdtemp(prefix="dexhandrl_mjx_scene_")) / "dexhandrl.xml"
        self.scene_path = build_dexhandrl_scene_xml(
            output_path=scene_path,
            object_name=cfg.object_name,
            isaac_source_root=cfg.isaac_source_root,
            dt=cfg.dt,
            substeps=cfg.substeps,
            base_pos_kp=cfg.base_pos_kp,
            base_pos_kv=cfg.base_pos_kv,
            base_pos_force=cfg.base_pos_force,
            base_rot_kp=cfg.base_rot_kp,
            base_rot_kv=cfg.base_rot_kv,
            base_rot_force=cfg.base_rot_force,
            finger_kp=cfg.finger_kp,
            finger_kv=cfg.finger_kv,
            finger_force=cfg.finger_force,
            hand_self_collision_mode=cfg.hand_self_collision_mode,
            object_friction=cfg.object_friction,
            ground_friction=cfg.ground_friction,
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.ids = resolve_mujoco_ids(mujoco, self.model, object_name=cfg.object_name)
        self.route_joint_upper_limits = dexhand021pro_route_joint_upper_limits(mujoco, self.model)
        self.mx = mjx.put_model(self.model, impl="warp")
        self.forward = jax.jit(mjx.forward)
        self.step_fn = jax.jit(mjx.step)

        self.geom_body_names = {
            geom_id: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[geom_id]))
            or ""
            for geom_id in range(self.model.ngeom)
        }
        self.geom_names = {
            geom_id: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            for geom_id in range(self.model.ngeom)
        }
        self.body_name_to_id = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (
                list(DEXHAND021PRO_CONTACT_BODY_NAMES)
                + list(DEXHAND021PRO_FINGERTIP_BODY_NAMES)
                + ["right_hand_base", "RFH1", cfg.object_name]
            )
        }
        self.body_name_to_id = {k: int(v) for k, v in self.body_name_to_id.items() if int(v) >= 0}

        self.hand_pose_body_name = "right_hand_base" if "right_hand_base" in self.body_name_to_id else "RFH1"
        if self.hand_pose_body_name not in self.body_name_to_id:
            raise RuntimeError("could not resolve a hand base body for hand_pose observation")
        self.hand_pose_body_id = self.body_name_to_id[self.hand_pose_body_name]
        self.object_body_id = self.ids.object_body_id

        self.contact_body_names = list(DEXHAND021PRO_CONTACT_BODY_NAMES)
        self.contact_body_ids = [self.body_name_to_id[name] for name in self.contact_body_names]
        self.fingertip_body_ids = [self.body_name_to_id[name] for name in DEXHAND021PRO_FINGERTIP_BODY_NAMES]

        self.contact_bodies_per_object_action = _contact_body_mapping_from_file(cfg.contact_bodies_file)
        self.expected_contact_mask = self._build_expected_contact_mask(cfg.object_name, cfg.action)
        self.active_finger_names = self._build_active_finger_names(cfg.object_name, cfg.action)
        self.active_joint_mask = self._build_active_joint_mask(self.active_finger_names)
        self.active_fingertip_mask = self._build_active_fingertip_mask(self.active_finger_names)
        self.active_joint_count = max(int(self.active_joint_mask.sum()), 1)
        self.active_fingertip_count = max(int(self.active_fingertip_mask.sum()), 1)
        self.rotation_disabled = _rotation_disabled_from_config(cfg.task_config_path, cfg.object_name, cfg.action)

        self.active_control_primary_joint_names = [
            DEXHAND021PRO_FULL_DOF_NAMES[idx] for idx in _ACTIVATED_CONTROL_PRIMARY_DOF_INDICES
        ]
        self.active_finger_lower, self.active_finger_upper = self._build_active_finger_limits()

        self.observation_space = _make_box_space(-np.inf, np.inf, (self.num_observations,), dtype=np.float32)
        self.action_space = _make_box_space(-1.0, 1.0, (self.num_actions,), dtype=np.float32)
        self.state_space = self.observation_space
        self.num_envs = 1
        self.num_agents = 1

        self._dx = None
        self._current_frame = self.start_frame
        self._episode_step_count = 0
        self._done = False
        self.base_position_offset = np.zeros(3, dtype=np.float32)
        self.joint_offset = np.zeros(16, dtype=np.float32)
        self._last_full_targets: np.ndarray | None = None
        self._last_obs = np.zeros((self.num_observations,), dtype=np.float32)
        self._last_info: dict[str, Any] = {}

    def _build_expected_contact_mask(self, object_name: str, action: str) -> np.ndarray:
        bodies = self.contact_bodies_per_object_action.get(object_name, {}).get(str(action))
        if bodies is None:
            raise RuntimeError(f"no active contact body config for ({object_name!r}, {action!r})")
        mask = np.zeros((len(DEXHAND021PRO_CONTACT_BODY_NAMES),), dtype=np.float32)
        body_to_index = {name: idx for idx, name in enumerate(DEXHAND021PRO_CONTACT_BODY_NAMES)}
        for body_name in bodies:
            if body_name not in body_to_index:
                raise RuntimeError(f"configured contact body {body_name!r} is not in the 16-body contact set")
            mask[body_to_index[body_name]] = 1.0
        return mask

    def _build_active_finger_names(self, object_name: str, action: str) -> list[str]:
        bodies = self.contact_bodies_per_object_action.get(object_name, {}).get(str(action))
        if bodies is None:
            raise RuntimeError(f"no active contact body config for ({object_name!r}, {action!r})")
        active = set()
        for body_name in bodies:
            if body_name.startswith("RH0_"):
                active.add("thumb")
            elif body_name.startswith("RH1_"):
                active.add("index")
            elif body_name.startswith("RH2_"):
                active.add("middle")
            elif body_name.startswith("RH3_"):
                active.add("ring")
            elif body_name.startswith("RH4_"):
                active.add("pinky")
        return sorted(active)

    def _build_active_joint_mask(self, active_finger_names: list[str]) -> np.ndarray:
        mask = np.zeros((16,), dtype=np.float32)
        for finger in active_finger_names:
            for idx in _FINGER_TO_JOINT_INDICES_021PRO[finger]:
                mask[idx] = 1.0
        return mask

    def _build_active_fingertip_mask(self, active_finger_names: list[str]) -> np.ndarray:
        mask = np.zeros((5,), dtype=np.float32)
        name_to_index = {name: idx for idx, name in enumerate(_FINGER_NAMES)}
        for finger in active_finger_names:
            mask[name_to_index[finger]] = 1.0
        return mask

    def _build_active_finger_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []
        for joint_name in self.active_control_primary_joint_names:
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"MuJoCo model is missing joint {joint_name!r}")
            joint_range = np.asarray(self.model.jnt_range[joint_id], dtype=np.float32)
            lower.append(float(joint_range[0]))
            upper.append(float(joint_range[1]))
        return np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)

    def _infer_object_primitive(self) -> tuple[int, np.ndarray]:
        geom_type = str(self.geometry["type"])
        size = np.asarray(self.geometry["size"], dtype=np.float32)
        if geom_type == "box":
            return 0, size
        if geom_type == "cylinder":
            return 1, np.asarray([size[0], size[1], 0.0], dtype=np.float32)
        if geom_type == "sphere":
            return 2, np.asarray([size[0], 0.0, 0.0], dtype=np.float32)
        return 3, np.zeros((3,), dtype=np.float32)

    def _set_initial_state(self) -> None:
        initial_full = _reference_full_targets(
            self.trajectory,
            frame=self.start_frame,
            source=self.cfg.reference_joint_source,
            current_full=None,
            route_joint_upper_limits=self.route_joint_upper_limits,
        )
        initial_object_pos = np.asarray(self.trajectory.object_pos[self.start_frame], dtype=np.float32).copy()
        initial_object_pos[2] += float(self.cfg.object_reset_z_offset)
        dx = self.mjx.make_data(self.model, impl="warp", naconmax=self.cfg.naconmax, njmax=self.cfg.njmax)
        dx = set_hand_qpos(self.jnp, dx, self.ids, initial_full)
        dx = set_object_freejoint(self.jnp, dx, self.ids, initial_object_pos, self.trajectory.object_quat_xyzw[self.start_frame])
        ctrl = make_ctrl_vector(initial_full, self.ids.actuator_ids, self.model.nu)
        dx = dx.replace(ctrl=self.jnp.asarray(ctrl, dtype=dx.ctrl.dtype))
        dx = self.forward(self.mx, dx)
        for _ in range(max(self.cfg.settle_frames, 0) * max(self.cfg.substeps, 1)):
            dx = self.step_fn(self.mx, dx)
        self.jax.block_until_ready(dx.qpos)
        self._dx = dx
        self._last_full_targets = initial_full.copy()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.cfg.seed = seed
        self._current_frame = self.start_frame
        self._episode_step_count = 0
        self._done = False
        self.base_position_offset[:] = 0.0
        self.joint_offset[:] = 0.0
        self._set_initial_state()
        obs, info = self._build_observation(target_frame=self._current_frame)
        self._last_obs = obs
        self._last_info = info
        return obs, info

    def _apply_residual_action(self, action: np.ndarray, target_frame: int, current_full: np.ndarray | None) -> np.ndarray:
        if not self.cfg.use_residual:
            self.base_position_offset[:] = 0.0
            self.joint_offset[:] = 0.0
            return _reference_full_targets(
                self.trajectory,
                frame=target_frame,
                source=self.cfg.reference_joint_source,
                current_full=current_full,
                route_joint_upper_limits=self.route_joint_upper_limits,
            )

        in_early_phase = (target_frame - self.start_frame) < self.cfg.early_phase_frames
        if in_early_phase:
            self.base_position_offset[:] = 0.0
            self.joint_offset[:] = 0.0
        action = np.asarray(action, dtype=np.float32).reshape(self.num_actions)
        action = np.clip(action, -1.0, 1.0)
        if in_early_phase:
            return _reference_full_targets(
                self.trajectory,
                frame=target_frame,
                source=self.cfg.reference_joint_source,
                current_full=current_full,
                route_joint_upper_limits=self.route_joint_upper_limits,
            )

        after_last_contact = target_frame > self.last_contact_frame
        base_decay = np.asarray(
            self.cfg.post_contact_base_position_decay if after_last_contact else self.cfg.base_position_decay,
            dtype=np.float32,
        )
        joint_decay = float(self.cfg.post_contact_joint_decay if after_last_contact else self.cfg.joint_decay)
        base_scale = np.asarray(self.cfg.base_position_scales, dtype=np.float32)
        joint_actions = action[6:22] * self.active_joint_mask
        joint_scales = np.asarray(
            [
                0.01,
                0.035,
                0.02,
                0.02,
                0.01,
                0.03,
                0.02,
                0.01,
                0.03,
                0.02,
                0.01,
                0.03,
                0.02,
                0.01,
                0.03,
                0.02,
            ],
            dtype=np.float32,
        ) * 3.0
        joint_max = np.asarray(
            [
                0.1,
                0.35,
                0.2,
                0.2,
                0.1,
                0.4,
                0.2,
                0.1,
                0.4,
                0.2,
                0.1,
                0.4,
                0.2,
                0.1,
                0.4,
                0.2,
            ],
            dtype=np.float32,
        ) * 2.0

        self.base_position_offset = base_decay * self.base_position_offset + action[0:3] * base_scale
        self.joint_offset = joint_decay * self.joint_offset + joint_actions * joint_scales
        self.base_position_offset = np.clip(
            self.base_position_offset,
            -self.cfg.base_position_max_offset,
            self.cfg.base_position_max_offset,
        )
        self.joint_offset = np.clip(self.joint_offset, -joint_max, joint_max)
        self.joint_offset *= self.active_joint_mask

        reference_full = _reference_full_targets(
            self.trajectory,
            frame=target_frame,
            source=self.cfg.reference_joint_source,
            current_full=current_full,
            route_joint_upper_limits=self.route_joint_upper_limits,
        ).copy()
        corrected_active = full_to_active_dof(reference_full)
        corrected_active[0:3] += self.base_position_offset
        corrected_active[6:] += self.joint_offset
        return active_to_full_dof(
            corrected_active,
            current_full=current_full if current_full is not None else reference_full,
            route_joint_upper_limits=self.route_joint_upper_limits,
        )

    def _collect_contacts(self) -> dict[str, Any]:
        dx = self._dx
        if dx is None:
            raise RuntimeError("environment is not initialized")
        impl = dx._impl
        nacon = np.asarray(impl.nacon)
        active_contacts = int(nacon[0]) if nacon.shape else int(nacon)
        if active_contacts >= self.cfg.naconmax:
            raise RuntimeError(
                f"MJX contact buffer reached naconmax={self.cfg.naconmax}; "
                "increase naconmax to avoid contact truncation"
            )

        contact_geom = np.asarray(impl.contact__geom)[:active_contacts]
        contact_frame = np.asarray(impl.contact__frame)[:active_contacts]
        contact_friction = np.asarray(impl.contact__friction)[:active_contacts]
        contact_dim = np.asarray(impl.contact__dim)[:active_contacts]
        contact_efc_address = np.asarray(impl.contact__efc_address)[:active_contacts]
        efc_force = np.asarray(impl.efc__force)
        pyramidal = int(self.model.opt.cone) == int(self.mujoco.mjtCone.mjCONE_PYRAMIDAL)
        return collect_contact_summary(
            active_contacts=active_contacts,
            contact_geom=contact_geom,
            contact_frame=contact_frame,
            contact_friction=contact_friction,
            contact_dim=contact_dim,
            contact_efc_address=contact_efc_address,
            efc_force=efc_force,
            pyramidal=pyramidal,
            geom_body_names=self.geom_body_names,
            geom_names=self.geom_names,
            contact_body_names=self.contact_body_names,
            object_name=self.cfg.object_name,
        )

    def _compute_fingertip_penalty(self, fingertips_world: np.ndarray, object_pos: np.ndarray, object_quat_xyzw: np.ndarray) -> float:
        if self.object_primitive_type == 3:
            return 0.0
        tips = np.asarray(fingertips_world, dtype=np.float32).reshape(5, 3)
        object_pos = np.asarray(object_pos, dtype=np.float32).reshape(3)
        object_quat_wxyz = np.asarray([object_quat_xyzw[3], object_quat_xyzw[0], object_quat_xyzw[1], object_quat_xyzw[2]], dtype=np.float32)
        tips_local = _rotate_points_inverse_wxyz(tips - object_pos[None, :], object_quat_wxyz)
        depth = np.zeros((5,), dtype=np.float32)
        eps = 1e-6
        if self.object_primitive_type == 0:
            half_extents = 0.5 * self.object_primitive_params.reshape(3)
            margin = half_extents[None, :] - np.abs(tips_local)
            inside = np.all(margin >= -eps, axis=-1)
            depth = np.where(inside, np.clip(np.min(margin, axis=-1), 0.0, None), 0.0)
        elif self.object_primitive_type == 1:
            radius = float(self.object_primitive_params[0])
            half_length = 0.5 * float(self.object_primitive_params[1])
            radial_distance = np.sqrt(tips_local[:, 0] ** 2 + tips_local[:, 1] ** 2)
            radial_margin = radius - radial_distance
            axial_margin = half_length - np.abs(tips_local[:, 2])
            inside = (radial_margin >= -eps) & (axial_margin >= -eps)
            depth = np.where(inside, np.clip(np.minimum(radial_margin, axial_margin), 0.0, None), 0.0)
        elif self.object_primitive_type == 2:
            radius = float(self.object_primitive_params[0])
            depth = np.clip(radius - np.linalg.norm(tips_local, axis=-1), 0.0, None)
        min_depth = float(self.cfg.fingertip_inside_object_min_depth)
        max_depth = float(self.cfg.fingertip_inside_object_max_depth)
        if max_depth <= min_depth:
            raise RuntimeError(
                "invalid fingertip penetration thresholds: "
                f"min_depth={min_depth}, max_depth={max_depth}"
            )
        penetration_ramp = np.clip(
            (depth - min_depth) / (max_depth - min_depth),
            0.0,
            1.0,
        )
        active_penetration_ramp = penetration_ramp * self.active_fingertip_mask
        per_fingertip_penalty = self.cfg.max_fingertip_inside_object_penalty / float(self.active_fingertip_count)
        return -float(np.sum(active_penetration_ramp * per_fingertip_penalty))

    def _build_observation(self, target_frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        dx = self._dx
        if dx is None:
            raise RuntimeError("environment is not initialized")
        qpos = np.asarray(dx.qpos)
        qvel = np.asarray(dx.qvel)
        object_pos = np.asarray(dx.xpos[self.object_body_id], dtype=np.float32)
        object_quat_xyzw = wxyz_to_xyzw(np.asarray(dx.xquat[self.object_body_id], dtype=np.float32))
        object_lin_vel = np.asarray(qvel[self.ids.object_free_dof_addr : self.ids.object_free_dof_addr + 3], dtype=np.float32)
        hand_pos = np.asarray(dx.xpos[self.hand_pose_body_id], dtype=np.float32)
        hand_quat_xyzw = wxyz_to_xyzw(np.asarray(dx.xquat[self.hand_pose_body_id], dtype=np.float32))
        hand_pose = np.concatenate([hand_pos, hand_quat_xyzw]).astype(np.float32)
        current_full = full_dof_from_qpos(qpos, self.ids)
        current_active = full_to_active_dof(current_full)
        active_finger_pos = current_active[6:]
        active_finger_pos = _normalize_active_positions(active_finger_pos, self.active_finger_lower, self.active_finger_upper)
        fingertip_positions_world = np.asarray([dx.xpos[body_id] for body_id in self.fingertip_body_ids], dtype=np.float32).reshape(15)
        body_pose_world = np.asarray([dx.xpos[self.body_name_to_id[name]] for name in self.contact_body_names], dtype=np.float32).reshape(48)
        contact_summary = self._collect_contacts()
        contact_force_vectors = _mask_contact_vectors(
            contact_summary["contact_force_vectors"],
            self.expected_contact_mask,
        )
        contact_force_magnitude = _normalize_contact_force(np.linalg.norm(contact_force_vectors, axis=-1), 100.0)
        contact_force_directions = np.zeros_like(contact_force_vectors, dtype=np.float32)
        norms = np.linalg.norm(contact_force_vectors, axis=-1, keepdims=True)
        contact_force_directions = np.divide(
            contact_force_vectors,
            norms,
            out=contact_force_directions,
            where=norms > 1e-8,
        )
        object_contact_force_magnitude = float(
            _normalize_contact_force(np.linalg.norm(contact_summary["object_force_vector"]), 100.0)
        )
        episode_time = float(
            np.clip(
                ((target_frame - self.start_frame) * self.cfg.dt) / max(self.max_episode_time_s, 1e-6),
                0.0,
                1.0,
            )
        )
        target_pos = np.asarray(self.trajectory.object_pos[target_frame], dtype=np.float32)
        target_quat = np.asarray(self.trajectory.object_quat_xyzw[target_frame], dtype=np.float32)
        lookahead_frame = min(target_frame + 5, self.end_frame)
        target_pos_next_5 = np.asarray(self.trajectory.object_pos[lookahead_frame], dtype=np.float32)
        obs = build_minimal_observation(
            action=self.cfg.action,
            object_name=self.cfg.object_name,
            geometry=self.geometry,
            active_finger_dof_pos=active_finger_pos,
            hand_pose=hand_pose,
            fingertip_positions_world=fingertip_positions_world,
            body16_positions_world=body_pose_world,
            episode_time=episode_time,
            object_position=object_pos,
            object_orientation=object_quat_xyzw,
            object_linear_velocity=object_lin_vel,
            target_object_position=target_pos,
            target_object_orientation=target_quat,
            target_object_pos_next_5=target_pos_next_5,
            expected_contact_mask=self.expected_contact_mask,
            contact_force_magnitude=contact_force_magnitude.astype(np.float32),
            contact_force_directions=contact_force_directions.reshape(-1).astype(np.float32),
            object_contact_force_magnitude=object_contact_force_magnitude,
            cumulative_offset=self.base_position_offset,
            cumulative_joint_offset=self.joint_offset,
            object_point_cloud_hand_frame=self._object_point_cloud_hand_frame(hand_pos, hand_quat_xyzw, object_pos, object_quat_xyzw),
            isaac_source_root=self.cfg.isaac_source_root,
        )
        target_error = object_pos - target_pos
        target_active = (
            full_to_active_dof(self._last_full_targets)
            if self._last_full_targets is not None
            else current_active
        )
        hand_target_error = current_active - target_active
        finger_target_abs_error = np.abs(hand_target_error[6:])
        rotation_error_deg = rotation_angle_error_deg(object_quat_xyzw, target_quat)
        reward_terms = self._compute_reward_terms(
            target_frame=target_frame,
            object_pos=object_pos,
            object_quat_xyzw=object_quat_xyzw,
            object_lin_vel=object_lin_vel,
            target_pos=target_pos,
            target_quat=target_quat,
            contact_summary=contact_summary,
            fingertip_positions_world=fingertip_positions_world,
            hand_pose=hand_pose,
            position_error=target_error,
            rotation_error_deg=rotation_error_deg,
        )
        info = {
            "trajectory_uuid": self.trajectory.uuid,
            "row_id": int(self.trajectory.row_id),
            "object_name": self.cfg.object_name,
            "action": self.cfg.action,
            "sequence_index": int(self.cfg.sequence_index),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "current_frame": int(target_frame),
            "movement_start_frame": int(self.trajectory.movement_start_frame),
            "movement_end_frame": int(self.trajectory.movement_end_frame),
            "episode_time_s": float((self._current_frame - self.start_frame) * self.cfg.dt),
            "episode_progress": float((target_frame - self.start_frame) / max(self.episode_length_frames - 1, 1)),
            "object_position_error_m": float(np.linalg.norm(target_error)),
            "object_position": object_pos,
            "target_object_position": target_pos,
            "object_rotation_error_deg": float(rotation_error_deg),
            "hand_active_l2_error": float(np.linalg.norm(hand_target_error)),
            "finger_active_mean_abs_error_rad": float(finger_target_abs_error.mean()),
            "finger_active_max_abs_error_rad": float(finger_target_abs_error.max()),
            "contact_force_vectors": contact_summary["contact_force_vectors"],
            "object_force_vector": contact_summary["object_force_vector"],
            "hand_object_pairs": contact_summary["hand_object_pairs"],
            "active_contact_count": int(contact_summary["active_contact_count"]),
            "reward_terms": reward_terms,
            "rotation_disabled": bool(self.rotation_disabled),
            "use_residual": bool(self.cfg.use_residual),
        }
        return obs, info

    def _object_point_cloud_hand_frame(
        self,
        hand_pos: np.ndarray,
        hand_quat_xyzw: np.ndarray,
        object_pos: np.ndarray,
        object_quat_xyzw: np.ndarray,
    ) -> np.ndarray:
        object_quat_wxyz = np.asarray([object_quat_xyzw[3], object_quat_xyzw[0], object_quat_xyzw[1], object_quat_xyzw[2]], dtype=np.float32)
        hand_quat_wxyz = np.asarray([hand_quat_xyzw[3], hand_quat_xyzw[0], hand_quat_xyzw[1], hand_quat_xyzw[2]], dtype=np.float32)
        points_world = _rotate_points_wxyz(self.object_point_cloud_local, object_quat_wxyz) + np.asarray(object_pos, dtype=np.float32)
        points_hand = _rotate_points_inverse_wxyz(points_world - np.asarray(hand_pos, dtype=np.float32), hand_quat_wxyz)
        return points_hand.reshape(-1).astype(np.float32)

    def _compute_reward_terms(
        self,
        *,
        target_frame: int,
        object_pos: np.ndarray,
        object_quat_xyzw: np.ndarray,
        object_lin_vel: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        contact_summary: dict[str, Any],
        fingertip_positions_world: np.ndarray,
        hand_pose: np.ndarray,
        position_error: np.ndarray,
        rotation_error_deg: float,
    ) -> dict[str, float]:
        active_phase = (target_frame - self.start_frame) < self.cfg.early_phase_frames
        after_last_contact = target_frame > self.last_contact_frame
        body_force_norm = np.linalg.norm(np.asarray(contact_summary["contact_force_vectors"], dtype=np.float32), axis=-1)
        has_contact = body_force_norm > self.cfg.contact_force_threshold
        correct_contacts = has_contact * (self.expected_contact_mask > 0.5)
        expected_count = max(float(self.expected_contact_mask.sum()), 1.0)
        contact_quality = float(correct_contacts.sum()) / expected_count * self.cfg.max_contact_reward
        if after_last_contact:
            contact_quality = 0.0

        obj_tracking = object_tracking_rewards(object_pos, target_pos)
        reward_object_pos_x = contact_quality * obj_tracking["object_pos_tracking_x"]
        reward_object_pos_y = contact_quality * obj_tracking["object_pos_tracking_y"]
        reward_object_pos_z = contact_quality * obj_tracking["object_pos_tracking_z"]

        speed = float(np.linalg.norm(object_lin_vel))
        object_stability_velocity = self.cfg.max_object_stability_reward * float(
            np.exp(-((speed / max(self.cfg.object_stability_velocity_scale, 1e-6)) ** 2))
        )
        if not after_last_contact:
            object_stability_velocity = 0.0

        rot_tracking = rotation_tracking_reward(rotation_error_deg)
        if self.rotation_disabled:
            rot_tracking = 0.0

        fingertip_penalty = self._compute_fingertip_penalty(
            fingertips_world=fingertip_positions_world,
            object_pos=object_pos,
            object_quat_xyzw=object_quat_xyzw,
        )

        base_penalty = -float(np.sum(np.abs(self.base_position_offset * 100.0)))
        joint_penalty = -float(np.sum(np.abs(self.joint_offset * 10.0)) / float(self.active_joint_count) * 8.0)
        if after_last_contact:
            base_penalty *= 0.5
            joint_penalty *= 0.5

        if active_phase:
            reward_object_pos_x = 0.0
            reward_object_pos_y = 0.0
            reward_object_pos_z = 0.0
            contact_quality = 0.0
            object_stability_velocity = 0.0
            rot_tracking = 0.0

        raw_terms = {
            "alive": 1.0,
            "object_pos_tracking_x": reward_object_pos_x,
            "object_pos_tracking_y": reward_object_pos_y,
            "object_pos_tracking_z": reward_object_pos_z,
            "contact_quality": contact_quality,
            "object_stability_velocity": object_stability_velocity,
            "object_rot_tracking": rot_tracking,
            "fingertip_inside_object_penalty": fingertip_penalty,
            "action_penalty_position": base_penalty,
            "action_penalty_joint": joint_penalty,
        }
        return apply_reward_weights(raw_terms, self.reward_weights)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        action = np.asarray(action, dtype=np.float32).reshape(self.num_actions)
        target_frame_before = self._current_frame
        current_qpos = np.asarray(self._dx.qpos)
        current_full = (
            full_dof_from_qpos(current_qpos, self.ids)
            if self.cfg.reference_joint_source == "active"
            else None
        )
        full_targets = self._apply_residual_action(action, target_frame_before, current_full)
        self._last_full_targets = full_targets.copy()
        ctrl = make_ctrl_vector(full_targets, self.ids.actuator_ids, self.model.nu)
        self._dx = self._dx.replace(ctrl=self.jnp.asarray(ctrl, dtype=self._dx.ctrl.dtype))
        self._dx = self.forward(self.mx, self._dx)
        for _ in range(max(self.cfg.substeps, 1)):
            self._dx = self.step_fn(self.mx, self._dx)
        self.jax.block_until_ready(self._dx.qpos)

        self._episode_step_count += 1
        self._current_frame = min(self._current_frame + 1, self.end_frame)
        obs, info = self._build_observation(target_frame=self._current_frame)
        reward_terms = info["reward_terms"]
        terminated = False
        truncated = self._current_frame >= self.end_frame
        object_error = float(info["object_position_error_m"])
        if not ((self._current_frame - self.start_frame) < self.cfg.early_phase_frames):
            terminated = object_error > self.cfg.object_position_terminal_threshold
        reward = float(reward_terms["total"])
        if terminated:
            termination_reward = add_termination_reward(
                reward_terms,
                "termination_failure_penalty",
                self.reward_weights,
            )
            reward += termination_reward
            info["termination_reason"] = "object_position_tracking_error"
        elif truncated:
            termination_reward = add_termination_reward(
                reward_terms,
                "termination_timeout_penalty",
                self.reward_weights,
            )
            reward += termination_reward
            info["termination_reason"] = "timeout"
        self._done = terminated or truncated
        self._last_obs = obs
        self._last_info = info
        info["applied_action"] = action
        info["full_targets"] = full_targets
        info["base_position_offset"] = self.base_position_offset.copy()
        info["joint_offset"] = self.joint_offset.copy()
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._dx = None

    def state(self) -> np.ndarray:
        return np.asarray(self._last_obs, dtype=np.float32)
