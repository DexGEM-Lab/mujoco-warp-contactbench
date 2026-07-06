from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

LANCE_GENERATED_SCHEMA = "contactbench.generated_data_schema.v1"


def vec3(value: Iterable[float], name: str) -> list[float]:
    out = [float(v) for v in value]
    if len(out) != 3:
        raise ValueError(f"{name} must have length 3, got {len(out)}")
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must contain only finite values: {out}")
    return out


def zero3() -> list[float]:
    return [0.0, 0.0, 0.0]


def transform_point_to_local(
    point_world: Iterable[float],
    origin_world: Iterable[float],
    rotation_world_from_local: Iterable[Iterable[float]],
) -> list[float]:
    p = vec3(point_world, "point_world")
    o = vec3(origin_world, "origin_world")
    r = [[float(x) for x in row] for row in rotation_world_from_local]
    if len(r) != 3 or any(len(row) != 3 for row in r):
        raise ValueError("rotation_world_from_local must be 3x3")
    delta = [p[i] - o[i] for i in range(3)]
    return [sum(r[row][col] * delta[row] for row in range(3)) for col in range(3)]


def transform_force_to_local(
    force_world: Iterable[float],
    rotation_world_from_local: Iterable[Iterable[float]],
) -> list[float]:
    f = vec3(force_world, "force_world")
    r = [[float(x) for x in row] for row in rotation_world_from_local]
    if len(r) != 3 or any(len(row) != 3 for row in r):
        raise ValueError("rotation_world_from_local must be 3x3")
    return [sum(r[row][col] * f[row] for row in range(3)) for col in range(3)]


def make_contact_pair(
    *,
    force_normal: Iterable[float],
    force_tangential: Iterable[float] | None = None,
    pos_world: Iterable[float],
    pos_wrist: Iterable[float],
    pos_joint: Iterable[float],
    pos_object: Iterable[float],
) -> dict[str, list[float]]:
    pair = {
        "force_normal": vec3(force_normal, "force_normal"),
        "force_tangential": vec3(force_tangential or zero3(), "force_tangential"),
        "pos_world": vec3(pos_world, "pos_world"),
        "pos_wrist": vec3(pos_wrist, "pos_wrist"),
        "pos_joint": vec3(pos_joint, "pos_joint"),
        "pos_object": vec3(pos_object, "pos_object"),
    }
    validate_contact_point(pair)
    return pair


def make_contact_entry(
    *,
    joint_name: str,
    object_name: str,
    total_force_world: Iterable[float],
    total_force_wrist: Iterable[float],
    total_force_joint: Iterable[float],
    total_force_object: Iterable[float],
    contact_pairs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    entry = {
        "joint_name": str(joint_name),
        "object_name": str(object_name),
        "total_force_world": vec3(total_force_world, "total_force_world"),
        "total_force_wrist": vec3(total_force_wrist, "total_force_wrist"),
        "total_force_joint": vec3(total_force_joint, "total_force_joint"),
        "total_force_object": vec3(total_force_object, "total_force_object"),
        "contact_pairs": list(contact_pairs),
    }
    validate_contact_entry(entry)
    return entry


def validate_contact_point(point: dict[str, Any]) -> None:
    for key in (
        "force_normal",
        "force_tangential",
        "pos_world",
        "pos_wrist",
        "pos_joint",
        "pos_object",
    ):
        if key not in point:
            raise ValueError(f"contact point missing {key}")
        vec3(point[key], key)


def validate_contact_entry(entry: dict[str, Any]) -> None:
    for key in ("joint_name", "object_name"):
        if not isinstance(entry.get(key), str) or not entry[key]:
            raise ValueError(f"contact entry {key} must be a non-empty string")
    for key in (
        "total_force_world",
        "total_force_wrist",
        "total_force_joint",
        "total_force_object",
    ):
        if key not in entry:
            raise ValueError(f"contact entry missing {key}")
        vec3(entry[key], key)
    pairs = entry.get("contact_pairs")
    if not isinstance(pairs, list):
        raise ValueError("contact_pairs must be a list")
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("each contact_pairs item must be an object")
        validate_contact_point(pair)


def validate_contact_sequence(contact: Any) -> int:
    if not isinstance(contact, list):
        raise ValueError("contact must be a list over timesteps")
    total = 0
    for frame_idx, frame_contacts in enumerate(contact):
        if not isinstance(frame_contacts, list):
            raise ValueError(f"contact[{frame_idx}] must be a list")
        for entry in frame_contacts:
            if not isinstance(entry, dict):
                raise ValueError(f"contact[{frame_idx}] item must be an object")
            validate_contact_entry(entry)
            total += 1
    return total


def write_contact_fixture(path: str | Path, contact: list[list[dict[str, Any]]], metadata: dict[str, Any] | None = None) -> None:
    validate_contact_sequence(contact)
    payload = {
        "schema": LANCE_GENERATED_SCHEMA,
        "note": "Fixture mirrors the generated_data_schema.jsonc contact field.",
        "metadata": metadata or {},
        "contact": contact,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_contact_fixture(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_contact_sequence(payload["contact"])
    return payload


def make_minimal_generated_trajectory(contact: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Build a small generated-trajectory-shaped object for smoke tests."""
    import numpy as np

    validate_contact_sequence(contact)
    total_frames = len(contact)
    return {
        "index": {
            "capMachine": "sim",
            "operator": "contactbench",
            "scene": "mano_object_contact",
            "is_generated": True,
        },
        "trajectory_metadata": {
            "data_fps": 60,
            "total_frames": total_frames,
            "hand_names": ["mano_hand_s02"],
            "object_names": ["contact_object"],
            "mano_hand_shapes": [np.zeros(10, dtype=np.float32)],
            "raw_data_info": {
                "capMachine": "sim",
                "operator": "contactbench",
                "scene": "mano_object_contact",
                "id": 0,
            },
            "trajectory_info": {
                "object_move": [
                    {"object_name": "contact_object", "start_frame": 0, "end_frame": max(0, total_frames - 1)}
                ]
            },
            "capture_info": None,
            "train_info": {"commit_hash": "unknown", "reward_value": 0.0},
        },
        "timestamp": np.arange(total_frames, dtype=np.float64) / 60.0,
        "hands": [
            {
                "mano_global_pos": np.zeros((total_frames, 3), dtype=np.float32),
                "mano_global_rot_aa": np.zeros((total_frames, 3), dtype=np.float32),
                "mano_hand_pose": np.zeros((total_frames, 48), dtype=np.float32),
                "mano_joint_pos": np.zeros((total_frames, 21, 3), dtype=np.float32),
                "urdf_dof": np.zeros((total_frames, 26), dtype=np.float32),
            }
        ],
        "objects": [
            {
                "rot_aa": np.zeros((total_frames, 3), dtype=np.float32),
                "pos": np.zeros((total_frames, 3), dtype=np.float32),
            }
        ],
        "contact": contact,
    }


def sample_contact_sequence(backend: str) -> list[list[dict[str, Any]]]:
    force_world = [0.0, 0.0, 4.2]
    point_world = [-0.12, 0.002, 0.03]
    pair = make_contact_pair(
        force_normal=force_world,
        force_tangential=[0.15, 0.0, 0.0],
        pos_world=point_world,
        pos_wrist=point_world,
        pos_joint=[0.0, 0.0, 0.006],
        pos_object=[0.0, 0.0, -0.025],
    )
    entry = make_contact_entry(
        joint_name="index_dip",
        object_name="contact_object",
        total_force_world=[0.15, 0.0, 4.2],
        total_force_wrist=[0.15, 0.0, 4.2],
        total_force_joint=[0.15, 0.0, 4.2],
        total_force_object=[0.15, 0.0, 4.2],
        contact_pairs=[pair],
    )
    entry["backend_note"] = backend
    return [[], [entry]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a ContactBench Lance contact fixture.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    payload = read_contact_fixture(args.path)
    count = validate_contact_sequence(payload["contact"])
    print(f"validated {count} contact entries from {args.path}")


if __name__ == "__main__":
    main()
