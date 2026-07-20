"""Observation, reward, and action utilities kept close to IsaacGym semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from sim.dexhandrl.constants import (
    DEFAULT_ISAAC_SOURCE_ROOT,
    NUM_ACTION_TYPES_021PRO,
    OBJECT_GEOMETRY_ENCODING_DIM,
    POINT_CLOUD_DIM_021PRO,
)


def action_type_onehot(action: str, num_action_types: int = NUM_ACTION_TYPES_021PRO) -> np.ndarray:
    out = np.zeros((num_action_types,), dtype=np.float32)
    idx = max(0, min(num_action_types - 1, int(action) - 1))
    out[idx] = 1.0
    return out


def _mesh_extents_from_obj(path: Path, scale: tuple[float, float, float]) -> list[float]:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append(
                [
                    float(parts[1]) * scale[0],
                    float(parts[2]) * scale[1],
                    float(parts[3]) * scale[2],
                ]
            )
    if not vertices:
        raise RuntimeError(f"OBJ mesh has no vertices: {path}")
    arr = np.asarray(vertices, dtype=np.float64)
    return (arr.max(axis=0) - arr.min(axis=0)).astype(float).tolist()


def geometry_from_isaac_urdf(object_name: str, isaac_source_root: Path = DEFAULT_ISAAC_SOURCE_ROOT) -> dict[str, object]:
    urdf_path = isaac_source_root / "assets/all_assets/Assets/sim/mano_objects_urdf" / f"{object_name}.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"object URDF not found: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    geom = root.find(".//collision/geometry") or root.find(".//visual/geometry")
    if geom is None:
        raise RuntimeError(f"URDF has no geometry: {urdf_path}")
    if geom.find("box") is not None:
        size = [float(v) for v in geom.find("box").attrib["size"].split()]
        return {"type": "box", "size": size, "urdf_path": str(urdf_path)}
    if geom.find("cylinder") is not None:
        elem = geom.find("cylinder")
        return {"type": "cylinder", "size": [float(elem.attrib["radius"]), float(elem.attrib["length"])], "urdf_path": str(urdf_path)}
    if geom.find("sphere") is not None:
        return {"type": "sphere", "size": [float(geom.find("sphere").attrib["radius"])], "urdf_path": str(urdf_path)}
    mesh = geom.find("mesh")
    if mesh is None:
        raise RuntimeError(f"URDF geometry is not supported: {urdf_path}")
    scale = tuple(float(v) for v in mesh.attrib.get("scale", "1 1 1").split())
    mesh_path = (urdf_path.parent / mesh.attrib["filename"]).resolve()
    return {
        "type": "irregular",
        "size": _mesh_extents_from_obj(mesh_path, scale),
        "mesh_path": str(mesh_path),
        "scale": scale,
        "urdf_path": str(urdf_path),
    }


def encode_geometry_12d(geometry: dict[str, object], max_size: float = 0.2) -> np.ndarray:
    enc = np.zeros((OBJECT_GEOMETRY_ENCODING_DIM,), dtype=np.float32)
    geom_type = str(geometry["type"])
    size = [float(v) for v in geometry["size"]]
    if geom_type == "box":
        enc[0:3] = np.clip(np.asarray(size[:3], dtype=np.float32) / max_size, 0.0, 1.0)
    elif geom_type == "cylinder":
        diameter = 2.0 * size[0]
        enc[3:6] = np.clip(np.asarray([diameter, diameter, size[1]], dtype=np.float32) / max_size, 0.0, 1.0)
    elif geom_type == "sphere":
        diameter = 2.0 * size[0]
        enc[6:9] = np.clip(np.asarray([diameter, diameter, diameter], dtype=np.float32) / max_size, 0.0, 1.0)
    elif geom_type == "irregular":
        enc[9:12] = np.clip(np.asarray(size[:3], dtype=np.float32) / max_size, 0.0, 1.0)
    return enc


def rotation_angle_error_deg(q_xyzw: np.ndarray, target_xyzw: np.ndarray) -> float:
    q = np.asarray(q_xyzw, dtype=np.float64)
    target = np.asarray(target_xyzw, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    target = target / max(np.linalg.norm(target), 1e-12)
    dot = float(abs(np.dot(q, target)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def rotation_tracking_reward(angle_deg: float) -> float:
    theta = float(angle_deg)
    if theta <= 20.0:
        return float(0.4 * (-0.00125 * theta * theta + 1.0) - 0.1)
    if theta <= 90.0:
        theta_offset = theta - 20.0
        return float(
            0.4
            * (
                -0.00003175 * theta_offset * theta_offset
                - 0.019206 * theta_offset
                + 0.5
            )
            - 0.1
        )
    return -0.5


@dataclass(frozen=True)
class IsaacRewardContract:
    reward_weights: dict[str, float]
    early_phase_frames: int
    object_position_terminal_threshold: float
    contact_force_threshold: float
    object_stability_velocity_scale: float
    fingertip_inside_object_min_depth: float
    fingertip_inside_object_max_depth: float


def load_isaac_reward_contract(config_path: Path) -> IsaacRewardContract:
    """Load reward-defining values from the canonical IsaacGym task YAML."""

    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    task = data.get("task")
    if not isinstance(task, dict):
        raise RuntimeError(f"task config has no 'task' mapping: {path}")
    reward_weights = task.get("reward_weights")
    termination = task.get("termination")
    if not isinstance(reward_weights, dict) or not isinstance(termination, dict):
        raise RuntimeError(f"task config has no reward/termination contract: {path}")

    required_weights = {
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
    missing = sorted(required_weights - reward_weights.keys())
    if missing:
        raise RuntimeError(f"task reward contract is missing weights {missing}: {path}")

    return IsaacRewardContract(
        reward_weights={str(name): float(weight) for name, weight in reward_weights.items()},
        early_phase_frames=int(task["early_phase_frames"]),
        object_position_terminal_threshold=float(
            termination["object_position_terminal_threshold"]
        ),
        contact_force_threshold=float(task["contact_force_threshold"]),
        object_stability_velocity_scale=float(task["object_stability_velocity_scale"]),
        fingertip_inside_object_min_depth=float(
            task["fingertip_inside_object_min_depth"]
        ),
        fingertip_inside_object_max_depth=float(
            task["fingertip_inside_object_max_depth"]
        ),
    )


def apply_reward_weights(
    raw_terms: dict[str, float], reward_weights: dict[str, float]
) -> dict[str, float]:
    """Match IsaacGym RewardCalculator's raw/weighted logging contract."""

    components: dict[str, float] = {}
    total = 0.0
    for name, raw_value in raw_terms.items():
        weight = float(reward_weights.get(name, 0.0))
        if weight == 0.0:
            continue
        raw = float(raw_value)
        weighted = raw * weight
        components[name] = raw
        components[f"{name}_weighted"] = weighted
        total += weighted
    components["total"] = float(total)
    return components


def add_termination_reward(
    components: dict[str, float],
    name: str,
    reward_weights: dict[str, float],
) -> float:
    """Add one IsaacGym-style one-shot termination component in place."""

    weighted = float(reward_weights[name])
    components[name] = 1.0
    components[f"{name}_weighted"] = weighted
    components["total"] = float(components.get("total", 0.0) + weighted)
    return weighted


def object_tracking_rewards(object_pos: np.ndarray, target_pos: np.ndarray) -> dict[str, float]:
    diff = np.abs(np.asarray(object_pos, dtype=np.float32) - np.asarray(target_pos, dtype=np.float32))
    base = np.exp(-40.0 * diff)
    return {
        "object_pos_tracking_x": float(base[0]),
        "object_pos_tracking_y": float(base[1]),
        "object_pos_tracking_z": float(base[2]),
    }


def build_minimal_observation(
    *,
    action: str,
    object_name: str,
    active_finger_dof_pos: np.ndarray,
    hand_pose: np.ndarray,
    fingertip_positions_world: np.ndarray,
    body16_positions_world: np.ndarray,
    episode_time: float,
    object_position: np.ndarray,
    object_orientation: np.ndarray,
    object_linear_velocity: np.ndarray,
    target_object_position: np.ndarray,
    target_object_orientation: np.ndarray,
    target_object_pos_next_5: np.ndarray,
    expected_contact_mask: np.ndarray,
    contact_force_magnitude: np.ndarray,
    contact_force_directions: np.ndarray,
    object_contact_force_magnitude: float,
    cumulative_offset: np.ndarray | None = None,
    cumulative_joint_offset: np.ndarray | None = None,
    object_point_cloud_hand_frame: np.ndarray | None = None,
    geometry: dict[str, object] | None = None,
    isaac_source_root: Path = DEFAULT_ISAAC_SOURCE_ROOT,
) -> np.ndarray:
    """Build a 461D observation vector in the current IsaacGym key order."""

    if geometry is None:
        geometry = geometry_from_isaac_urdf(object_name, isaac_source_root=isaac_source_root)
    parts = [
        action_type_onehot(action),
        encode_geometry_12d(geometry),
        np.asarray(active_finger_dof_pos, dtype=np.float32).reshape(16),
        np.asarray(hand_pose, dtype=np.float32).reshape(7),
        np.asarray(fingertip_positions_world, dtype=np.float32).reshape(15),
        np.asarray(body16_positions_world, dtype=np.float32).reshape(48),
        np.asarray([episode_time], dtype=np.float32),
        np.asarray(object_position, dtype=np.float32).reshape(3),
        np.asarray(object_orientation, dtype=np.float32).reshape(4),
        np.asarray(object_linear_velocity, dtype=np.float32).reshape(3),
        np.asarray(target_object_position, dtype=np.float32).reshape(3),
        np.asarray(target_object_orientation, dtype=np.float32).reshape(4),
        np.asarray(target_object_pos_next_5, dtype=np.float32).reshape(3),
        np.asarray(expected_contact_mask, dtype=np.float32).reshape(16),
        np.asarray(contact_force_magnitude, dtype=np.float32).reshape(16),
        np.asarray(contact_force_directions, dtype=np.float32).reshape(48),
        np.asarray([object_contact_force_magnitude], dtype=np.float32),
        np.zeros((3,), dtype=np.float32) if cumulative_offset is None else np.asarray(cumulative_offset, dtype=np.float32).reshape(3),
        np.zeros((16,), dtype=np.float32) if cumulative_joint_offset is None else np.asarray(cumulative_joint_offset, dtype=np.float32).reshape(16),
        np.zeros((POINT_CLOUD_DIM_021PRO,), dtype=np.float32)
        if object_point_cloud_hand_frame is None
        else np.asarray(object_point_cloud_hand_frame, dtype=np.float32).reshape(POINT_CLOUD_DIM_021PRO),
    ]
    obs = np.concatenate(parts).astype(np.float32)
    if obs.shape != (461,):
        raise RuntimeError(f"DexHand021Pro observation parity expected 461D, got {obs.shape}")
    return obs
