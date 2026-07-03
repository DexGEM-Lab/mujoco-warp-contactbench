#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "3rd_party"
LANCE_MANAGER_ROOT = THIRD_PARTY / "lance_manager"
for path in (REPO_ROOT, THIRD_PARTY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.ball_pit.common import BallPitSpec, hand_pose_at  # noqa: E402
from common.contact_schema import read_contact_fixture, validate_contact_sequence  # noqa: E402
from lance_manager.lance_dataset_manager import LanceDatasetManager  # noqa: E402
from lance_manager.schema.manager import SchemaManager  # noqa: E402
from lance_manager.schema.utils.converter import DictToArrowConverter  # noqa: E402
from lance_manager.schema.utils.parser import SchemaParser  # noqa: E402
from lance_manager.schema.utils.validator import create_validator_from_schema  # noqa: E402

GENERATED_SCHEMA = LANCE_MANAGER_ROOT / "schema/schemas/generated_data_schema.jsonc"
LOGS = REPO_ROOT / "logs"
DEFAULT_OUTPUT = LOGS / "mujoco_cpu_mjx_warp_gpu_ball_pit_generated.lance"
DEFAULT_INPUT_NAMES = (
    "mujoco_cpu_ball_pit_contact_10s.json",
    "mjx_warp_gpu_ball_pit_contact_10s.json",
)


class SingleProcessSchemaConverter:
    def __init__(self, schema_path: Path):
        self._parser = SchemaParser(schema_path)
        self._converter = DictToArrowConverter(self._parser)
        self._validator = create_validator_from_schema(str(schema_path))
        self._reserved_fields = set(self._parser.schema_dict.get("reserved_fields", []))

    def convert(self, data_dict: dict[str, Any]):
        processed = SchemaManager._process_reserved_fields(data_dict, self._reserved_fields)
        self._validator(**processed)
        return self._converter.convert(processed)

    def close(self) -> None:
        return None


def _resolve_default_input(name: str) -> Path:
    generated = LOGS / name
    if generated.exists():
        return generated
    raise FileNotFoundError(f"missing {name}; run the JSON debug export scripts first")


def _default_inputs() -> list[Path]:
    return [_resolve_default_input(name) for name in DEFAULT_INPUT_NAMES]


def _backend_to_scene(backend: str) -> str:
    # Must satisfy 3rd_party/lance_manager/schema/configs/3_scene.yaml enum.
    # Both CPU MuJoCo and MJX-Warp GPU represent the same MuJoCo ball-pit scene.
    return "cube1_02"


def _hand_trajectory(spec: BallPitSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_frames = int(spec.rollout_frames)
    mano_global_pos = np.zeros((total_frames, 3), dtype=np.float32)
    mano_global_rot_aa = np.zeros((total_frames, 3), dtype=np.float32)
    mano_hand_pose = np.zeros((total_frames, 48), dtype=np.float32)
    mano_joint_pos = np.zeros((total_frames, 21, 3), dtype=np.float32)
    urdf_dof = np.zeros((total_frames, 26), dtype=np.float32)
    for frame in range(total_frames):
        pos, euler = hand_pose_at(frame, spec)
        mano_global_pos[frame] = np.asarray(pos, dtype=np.float32)
        mano_global_rot_aa[frame] = np.asarray(euler, dtype=np.float32)
        urdf_dof[frame, 0:3] = mano_global_pos[frame]
        urdf_dof[frame, 3:6] = mano_global_rot_aa[frame]
        mano_hand_pose[frame, 0:3] = mano_global_rot_aa[frame]
        mano_joint_pos[frame, :, :] = mano_global_pos[frame]
    return mano_global_pos, mano_global_rot_aa, mano_hand_pose, mano_joint_pos, urdf_dof


def _array(value: Any, *, shape: tuple[int, ...], field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} contains non-finite values")
    return arr


def _tx3_array(value: Any, *, total_frames: int, field_name: str) -> np.ndarray:
    return _array(value, shape=(total_frames, 3), field_name=field_name)


def _hand_trajectory_from_payload(payload: dict[str, Any], spec: BallPitSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trajectory = payload.get("hand_trajectory")
    if trajectory is None:
        return _hand_trajectory(spec)
    if not isinstance(trajectory, dict):
        raise ValueError("hand_trajectory must be an object when present")
    total_frames = int(spec.rollout_frames)
    mano_global_pos = _array(trajectory.get("mano_global_pos"), shape=(total_frames, 3), field_name="hand_trajectory.mano_global_pos")
    mano_global_rot_aa = _array(trajectory.get("mano_global_rot_aa"), shape=(total_frames, 3), field_name="hand_trajectory.mano_global_rot_aa")
    mano_hand_pose = _array(trajectory.get("mano_hand_pose"), shape=(total_frames, 48), field_name="hand_trajectory.mano_hand_pose")
    mano_joint_pos = _array(trajectory.get("mano_joint_pos"), shape=(total_frames, 21, 3), field_name="hand_trajectory.mano_joint_pos")
    urdf_dof = _array(trajectory.get("urdf_dof"), shape=(total_frames, 26), field_name="hand_trajectory.urdf_dof")
    return mano_global_pos, mano_global_rot_aa, mano_hand_pose, mano_joint_pos, urdf_dof


def _fallback_contact_object_trajectories(contact: list[list[dict[str, Any]]], object_names: list[str], total_frames: int) -> np.ndarray:
    positions: dict[str, np.ndarray] = {name: np.zeros((total_frames, 3), dtype=np.float32) for name in object_names}
    counts: dict[str, np.ndarray] = {name: np.zeros((total_frames,), dtype=np.int32) for name in object_names}
    for frame_idx, frame_contacts in enumerate(contact):
        if frame_idx >= total_frames:
            break
        for entry in frame_contacts:
            obj = entry.get("object_name")
            if obj not in positions:
                continue
            for pair in entry.get("contact_pairs", []):
                positions[obj][frame_idx] += np.asarray(pair["pos_object"], dtype=np.float32)
                counts[obj][frame_idx] += 1
    objects = []
    for name in object_names:
        pos = positions[name]
        nonzero = counts[name] > 0
        if np.any(nonzero):
            pos[nonzero] = pos[nonzero] / counts[name][nonzero, None]
        objects.append({"rot_aa": np.zeros((total_frames, 3), dtype=np.float32), "pos": pos.astype(np.float32)})
    return np.asarray(objects, dtype=object)


def _object_trajectories_from_payload(
    payload: dict[str, Any],
    contact: list[list[dict[str, Any]]],
    total_frames: int,
) -> tuple[list[str], np.ndarray]:
    trajectories = payload.get("object_trajectories")
    contact_object_names = {entry["object_name"] for frame in contact for entry in frame}
    if trajectories is None:
        object_names = sorted(contact_object_names) or ["ball_000"]
        return object_names, _fallback_contact_object_trajectories(contact, object_names, total_frames)
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("object_trajectories must be a non-empty list when present")

    object_names: list[str] = []
    objects: list[dict[str, np.ndarray]] = []
    seen: set[str] = set()
    for idx, item in enumerate(trajectories):
        if not isinstance(item, dict):
            raise ValueError(f"object_trajectories[{idx}] must be an object")
        name = str(item.get("object_name") or item.get("name") or "")
        if not name:
            raise ValueError(f"object_trajectories[{idx}] missing object_name")
        if name in seen:
            raise ValueError(f"duplicate object_trajectories object_name: {name}")
        seen.add(name)
        pos = _tx3_array(item.get("pos"), total_frames=total_frames, field_name=f"object_trajectories[{idx}].pos")
        rot_aa = _tx3_array(
            item.get("rot_aa", np.zeros((total_frames, 3), dtype=np.float32)),
            total_frames=total_frames,
            field_name=f"object_trajectories[{idx}].rot_aa",
        )
        object_names.append(name)
        objects.append({"rot_aa": rot_aa, "pos": pos})

    missing = sorted(contact_object_names.difference(seen))
    if missing:
        raise ValueError(f"contact object_name values missing from object_trajectories: {missing[:8]}")
    return object_names, np.asarray(objects, dtype=object)


def _contact_to_numpy(contact: list[list[dict[str, Any]]]) -> np.ndarray:
    frames = []
    for frame_contacts in contact:
        entries = []
        for entry in frame_contacts:
            pairs = []
            for pair in entry.get("contact_pairs", []):
                pairs.append(
                    {
                        "force_normal": np.asarray(pair["force_normal"], dtype=np.float32),
                        "force_tangential": np.asarray(pair["force_tangential"], dtype=np.float32),
                        "pos_world": np.asarray(pair["pos_world"], dtype=np.float32),
                        "pos_wrist": np.asarray(pair["pos_wrist"], dtype=np.float32),
                        "pos_joint": np.asarray(pair["pos_joint"], dtype=np.float32),
                        "pos_object": np.asarray(pair["pos_object"], dtype=np.float32),
                    }
                )
            entries.append(
                {
                    "joint_name": str(entry["joint_name"]),
                    "object_name": str(entry["object_name"]),
                    "total_force_world": np.asarray(entry["total_force_world"], dtype=np.float32),
                    "total_force_wrist": np.asarray(entry["total_force_wrist"], dtype=np.float32),
                    "total_force_joint": np.asarray(entry["total_force_joint"], dtype=np.float32),
                    "total_force_object": np.asarray(entry["total_force_object"], dtype=np.float32),
                    "contact_pairs": np.asarray(pairs, dtype=object),
                }
            )
        frames.append(np.asarray(entries, dtype=object))
    return np.asarray(frames, dtype=object)


def trajectory_from_payload(payload: dict[str, Any], *, raw_id: int, source_name: str = "sim") -> dict[str, Any]:
    contact = payload["contact"]
    validate_contact_sequence(contact)
    metadata = payload.get("metadata", {})
    backend = str(metadata.get("backend", source_name))
    fps = float(metadata.get("fps", 100.0))
    total_frames = int(metadata.get("frames", len(contact)))
    if total_frames != len(contact):
        total_frames = len(contact)
    spec = BallPitSpec(
        scenario=str(metadata.get("scenario", "filled_tank")),
        ball_count=int(metadata.get("ball_count", 120)),
        ball_radius=float(metadata.get("ball_radius", 0.03)),
        fps=fps,
        substeps=int(metadata.get("substeps", 2)),
        settle_frames=int(metadata.get("settle_frames", 0)),
        rollout_frames=total_frames,
        duration_seconds=float(metadata.get("duration_seconds", total_frames / fps)),
    )
    hand_pos, hand_rot, hand_pose, hand_joints, urdf_dof = _hand_trajectory_from_payload(payload, spec)
    object_names, objects = _object_trajectories_from_payload(payload, contact, total_frames)
    scene = _backend_to_scene(backend)
    return {
        "index": {
            "capMachine": "capMachine01",
            "operator": "s01",
            "scene": scene,
            "is_generated": True,
        },
        "trajectory_metadata": {
            "data_fps": int(round(fps)),
            "total_frames": total_frames,
            "hand_names": np.asarray(["mano_hand_s02"], dtype=object),
            "object_names": np.asarray(object_names, dtype=object),
            "mano_hand_shapes": np.zeros((1, 10), dtype=np.float32),
            "raw_data_info": {
                "capMachine": "capMachine01",
                "operator": "s01",
                "scene": scene,
                "id": int(raw_id),
            },
            "trajectory_info": {
                "object_move": np.asarray(
                    [
                        {"object_name": name, "start_frame": 0, "end_frame": max(0, total_frames - 1)}
                        for name in object_names
                    ],
                    dtype=object,
                ),
            },
            "train_info": {
                "commit_hash": "mujoco-warp-contactbench-local",
                "reward_value": float(metadata.get("contact_pairs", 0)),
            },
        },
        "timestamp": (np.arange(total_frames, dtype=np.float64) / fps),
        "hands": np.asarray(
            [
                {
                    "mano_global_pos": hand_pos,
                    "mano_global_rot_aa": hand_rot,
                    "mano_hand_pose": hand_pose,
                    "mano_joint_pos": hand_joints,
                    "urdf_dof": urdf_dof,
                }
            ],
            dtype=object,
        ),
        "objects": objects,
        "contact": _contact_to_numpy(contact),
    }


def trajectory_from_contact(path: Path, raw_id: int) -> dict[str, Any]:
    return trajectory_from_payload(read_contact_fixture(path), raw_id=raw_id, source_name=path.stem)


def write_payloads_to_lance(
    payloads: list[dict[str, Any]],
    *,
    output: Path,
    replace: bool = False,
    processes: int = 1,
    source_names: list[str] | None = None,
) -> dict[str, Any]:
    if replace and output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    manager = SingleProcessSchemaConverter(GENERATED_SCHEMA) if processes <= 1 else SchemaManager(GENERATED_SCHEMA, processes=processes)
    dataset = LanceDatasetManager(output)
    rows: list[dict[str, Any]] = []
    try:
        for idx, payload in enumerate(payloads):
            source_name = source_names[idx] if source_names and idx < len(source_names) else f"payload_{idx}"
            trajectory = trajectory_from_payload(payload, raw_id=idx, source_name=source_name)
            table = manager.convert(trajectory)
            dataset.writer.write(table, dedup=False)
            rows.append({"source": source_name, "rows": table.num_rows})
            print(f"wrote {source_name} rows={table.num_rows}", flush=True)
        dataset.writer.flush()
    finally:
        manager.close()
    return {"output": str(output), "rows": rows}


def write_contacts_to_lance(
    inputs: list[Path],
    *,
    output: Path,
    replace: bool = False,
    processes: int = 1,
) -> dict[str, Any]:
    payloads = [read_contact_fixture(path) for path in inputs]
    result = write_payloads_to_lance(
        payloads,
        output=output,
        replace=replace,
        processes=processes,
        source_names=[str(path) for path in inputs],
    )
    result["inputs"] = [str(path) for path in inputs]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MuJoCo CPU and MJX-Warp GPU contact fixtures to full generated_data Lance.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inputs", type=Path, nargs="*", default=None)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()

    inputs = list(args.inputs) if args.inputs else _default_inputs()
    result = write_contacts_to_lance(
        inputs,
        output=args.output,
        replace=args.replace,
        processes=args.processes,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
