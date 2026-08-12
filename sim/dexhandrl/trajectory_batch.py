"""Immutable, host-resident trajectory windows for future batched DexHandRL physics.

This module deliberately separates task contact configuration from the Lance
contact annotation.  The former conditions rewards/controls; the latter remains
trajectory evidence and is never used to derive task masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from sim.dexhandrl.constants import DEXHAND021PRO_CONTACT_BODY_NAMES
from sim.dexhandrl.lance_loader import DexHandTrajectory, load_trajectories_by_row_ids
from sim.dexhandrl.parity import action_type_onehot

DEXHAND_ALL_ACTIONS = ("01", "02", "03", "04", "09", "10", "18")


def resolve_balanced_action_assignments(
    rows: Sequence[dict[str, Any]],
    *,
    object_name: str,
    num_envs: int,
    sequence_index: int = 0,
    duplicate_first_for_extra: bool = False,
) -> list[dict[str, Any]]:
    """Resolve all seven actions deterministically across a vector batch.

    Production training cycles trajectory identities independently within each
    action.  A bounded eight-world evaluation may instead duplicate world 0 in
    world 7, giving one exact same-input determinism control while retaining
    one world for every action.
    """

    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    routes: dict[str, list[dict[str, Any]]] = {}
    for action in DEXHAND_ALL_ACTIONS:
        selected = [
            row
            for row in rows
            if row.get("object_name") == object_name and str(row.get("action")) == action
        ]
        selected.sort(key=lambda row: (int(row["row_id"]), str(row["uuid"])))
        if not selected:
            raise RuntimeError(
                f"no trajectories found for required action={action!r} "
                f"object={object_name!r}"
            )
        routes[action] = selected

    assignments: list[dict[str, Any]] = []
    for env_id in range(num_envs):
        if duplicate_first_for_extra and env_id >= len(DEXHAND_ALL_ACTIONS):
            source = assignments[env_id % len(DEXHAND_ALL_ACTIONS)]
            assignments.append({**source, "determinism_duplicate_of": env_id % len(DEXHAND_ALL_ACTIONS)})
            continue
        action = DEXHAND_ALL_ACTIONS[env_id % len(DEXHAND_ALL_ACTIONS)]
        route = routes[action]
        route_index = (
            sequence_index + env_id // len(DEXHAND_ALL_ACTIONS)
        ) % len(route)
        assignments.append({**route[route_index], "sequence_index": route_index})
    return assignments


_SIMPLE_CONTACT_TO_BODY = {
    "palm": "RFH1", "thumb1": "RH0_0", "thumb2": "RH0_2", "thumb3": "RH0_3",
    "index1": "RH1_0", "index2": "RH1_2", "index3": "RH1_3",
    "middle1": "RH2_0", "middle2": "RH2_2", "middle3": "RH2_3",
    "ring1": "RH3_0", "ring2": "RH3_2", "ring3": "RH3_3",
    "pinky1": "RH4_0", "pinky2": "RH4_2", "pinky3": "RH4_3",
}
_FINGER_TO_JOINTS = {
    "thumb": (0, 1, 2, 3), "index": (4, 5, 6), "middle": (7, 8, 9),
    "ring": (10, 11, 12), "pinky": (13, 14, 15),
}
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class DexHandTrajectoryBatch:
    """Padded trajectory windows indexed by immutable environment assignment."""

    hand_active: np.ndarray
    hand_finger_dof_full_raw: np.ndarray | None
    full_raw_validity: np.ndarray
    object_pos: np.ndarray
    object_quat_xyzw: np.ndarray
    lance_contact_annotation: np.ndarray
    lengths: np.ndarray
    start_frames: np.ndarray
    movement_start_local: np.ndarray
    movement_end_local: np.ndarray
    row_ids: np.ndarray
    action_indices: np.ndarray
    actions: tuple[str, ...]
    uuids: tuple[str, ...]
    action_onehot50: np.ndarray
    task_expected_contact_masks: np.ndarray
    task_contact_body_names: tuple[tuple[str, ...], ...]
    active_joint_masks: np.ndarray
    active_fingertip_masks: np.ndarray
    rotation_disabled: np.ndarray
    # Lineage evidence retained from the read-only Lance index (default empty
    # for direct dataclass construction in tests).
    seed_uuids: tuple[str, ...] = ()

    @property
    def num_envs(self) -> int:
        return int(self.hand_active.shape[0])

    @property
    def frames(self) -> int:
        return int(self.hand_active.shape[1])


def _task_contracts(
    assignments: Sequence[dict[str, Any]], contact_bodies_file: Path, task_config_path: Path,
) -> dict[str, tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, bool]]:
    if not contact_bodies_file.is_file():
        raise RuntimeError(f"required contact-body config is missing: {contact_bodies_file}")
    if not task_config_path.is_file():
        raise RuntimeError(f"required task config is missing: {task_config_path}")
    contact_data = yaml.safe_load(contact_bodies_file.read_text(encoding="utf-8"))
    task_data = yaml.safe_load(task_config_path.read_text(encoding="utf-8"))
    if not isinstance(contact_data, dict) or not isinstance(task_data, dict):
        raise RuntimeError("DexHand task/contact configuration must be YAML mappings")
    disable_map = (task_data.get("task") or {}).get("disableObjectRotation", {}) or {}
    body_index = {name: index for index, name in enumerate(DEXHAND021PRO_CONTACT_BODY_NAMES)}
    result = {}
    for spec in assignments:
        object_name, action = str(spec["object_name"]), str(spec["action"])
        key = f"{object_name}/{action}"
        if key in result:
            continue
        configured = (contact_data.get(object_name) or {}).get(action)
        if not isinstance(configured, list) or not configured:
            raise RuntimeError(f"no active contact body config for ({object_name!r}, {action!r})")
        names = []
        for simple_name in configured:
            body_name = _SIMPLE_CONTACT_TO_BODY.get(str(simple_name))
            if body_name is None:
                raise RuntimeError(f"unsupported simplified contact body name: {simple_name!r}")
            names.append(body_name)
        mask = np.zeros(16, dtype=np.float32)
        for name in names:
            if name not in body_index:
                raise RuntimeError(f"configured contact body {name!r} is not in the 16-body contact set")
            mask[body_index[name]] = 1.0
        fingers = {name.split("_")[0] for name in names if name.startswith("RH")}
        finger_names = {"RH0": "thumb", "RH1": "index", "RH2": "middle", "RH3": "ring", "RH4": "pinky"}
        active_fingers = {finger_names[name] for name in fingers}
        joints = np.zeros(16, dtype=np.float32)
        tips = np.zeros(5, dtype=np.float32)
        for finger in active_fingers:
            joints[list(_FINGER_TO_JOINTS[finger])] = 1.0
            tips[_FINGER_NAMES.index(finger)] = 1.0
        disabled = disable_map.get(object_name)
        rotation_disabled = str(disabled).lower() == "all" if isinstance(disabled, str) else action in {str(x) for x in (disabled or [])}
        result[key] = (mask, tuple(names), joints, tips, rotation_disabled)
    return result


def pack_trajectory_batch(
    trajectories: Sequence[DexHandTrajectory], assignments: Sequence[dict[str, Any]], *,
    pre_contact_frames: int, post_contact_frames: int, contact_bodies_file: Path, task_config_path: Path,
) -> DexHandTrajectoryBatch:
    """Window and pad loaded rows. Frame zero is reset; frame one is first decision."""
    if len(trajectories) != len(assignments) or not trajectories:
        raise ValueError("trajectories and assignments must be non-empty and aligned")
    contracts = _task_contracts(assignments, Path(contact_bodies_file), Path(task_config_path))
    windows = []
    for traj, spec in zip(trajectories, assignments):
        if traj.row_id != int(spec["row_id"]) or traj.uuid != str(spec["uuid"]):
            raise RuntimeError("loaded Lance rows do not match the requested trajectory assignment")
        start = max(0, traj.movement_start_frame - int(pre_contact_frames))
        end = min(traj.frames - 1, traj.movement_end_frame + int(post_contact_frames))
        windows.append((start, end + 1))
    lengths = np.asarray([stop - start for start, stop in windows], dtype=np.int32)
    n, t = len(trajectories), int(lengths.max())
    hand = np.zeros((n, t, 22), dtype=np.float32)
    obj_pos = np.zeros((n, t, 3), dtype=np.float32)
    obj_quat = np.zeros((n, t, 4), dtype=np.float32)
    lance_contact = np.zeros((n, t, 16), dtype=np.float32)
    any_raw = any(traj.hand_finger_dof_full_raw is not None for traj in trajectories)
    raw = np.zeros((n, t, 20), dtype=np.float32) if any_raw else None
    raw_valid = np.zeros((n, t), dtype=bool)
    for env_id, (traj, (start, stop)) in enumerate(zip(trajectories, windows)):
        length = stop - start
        hand[env_id, :length] = traj.hand_dof[start:stop]
        obj_pos[env_id, :length] = traj.object_pos[start:stop]
        obj_quat[env_id, :length] = traj.object_quat_xyzw[start:stop]
        lance_contact[env_id, :length] = traj.expected_contact_mask[start:stop]
        if raw is not None and traj.hand_finger_dof_full_raw is not None:
            raw[env_id, :length] = traj.hand_finger_dof_full_raw[start:stop]
            raw_valid[env_id, :length] = True
        if length < t:
            hand[env_id, length:] = hand[env_id, length - 1]
            obj_pos[env_id, length:] = obj_pos[env_id, length - 1]
            obj_quat[env_id, length:] = obj_quat[env_id, length - 1]
            if raw is not None and raw_valid[env_id, length - 1]:
                raw[env_id, length:] = raw[env_id, length - 1]
                raw_valid[env_id, length:] = True
    actions = tuple(str(spec["action"]) for spec in assignments)
    task = [contracts[f"{spec['object_name']}/{spec['action']}"] for spec in assignments]
    return DexHandTrajectoryBatch(
        hand_active=hand, hand_finger_dof_full_raw=raw, full_raw_validity=raw_valid,
        object_pos=obj_pos, object_quat_xyzw=obj_quat, lance_contact_annotation=lance_contact,
        lengths=lengths, start_frames=np.asarray([start for start, _ in windows], dtype=np.int32),
        movement_start_local=np.asarray([traj.movement_start_frame - start for traj, (start, _) in zip(trajectories, windows)], dtype=np.int32),
        movement_end_local=np.asarray([traj.movement_end_frame - start for traj, (start, _) in zip(trajectories, windows)], dtype=np.int32),
        row_ids=np.asarray([traj.row_id for traj in trajectories], dtype=np.int64),
        action_indices=np.asarray([DEXHAND_ALL_ACTIONS.index(action) for action in actions], dtype=np.int32),
        actions=actions, uuids=tuple(traj.uuid for traj in trajectories),
        seed_uuids=tuple(
            str((traj.metadata.get("index") or {}).get("seed_uuid") or "")
            for traj in trajectories
        ),
        action_onehot50=np.stack([action_type_onehot(action) for action in actions]),
        task_expected_contact_masks=np.stack([entry[0] for entry in task]),
        task_contact_body_names=tuple(entry[1] for entry in task),
        active_joint_masks=np.stack([entry[2] for entry in task]), active_fingertip_masks=np.stack([entry[3] for entry in task]),
        rotation_disabled=np.asarray([entry[4] for entry in task], dtype=bool),
    )


def load_trajectory_batch(lance_path: Path | str, assignments: Sequence[dict[str, Any]], **kwargs: Any) -> DexHandTrajectoryBatch:
    """Bulk-load unique rows once, then pack assignment-aligned windows."""
    row_ids = [int(spec["row_id"]) for spec in assignments]
    expected_uuids: dict[int, str] = {}
    for spec, row_id in zip(assignments, row_ids):
        uuid = str(spec["uuid"])
        prior_uuid = expected_uuids.setdefault(row_id, uuid)
        if prior_uuid != uuid:
            raise RuntimeError(f"conflicting UUID assignments for Lance row_id={row_id}")
    loaded = load_trajectories_by_row_ids(lance_path, row_ids, expected_uuids=expected_uuids)
    by_row_id = {traj.row_id: traj for traj in loaded}
    try:
        trajectories = [by_row_id[row_id] for row_id in row_ids]
    except KeyError as exc:
        raise RuntimeError(f"Lance bulk load omitted requested row_id={exc.args[0]}") from exc
    return pack_trajectory_batch(trajectories, assignments, **kwargs)
