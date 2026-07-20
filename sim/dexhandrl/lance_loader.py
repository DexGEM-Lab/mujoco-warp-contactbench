"""Load DexHandRL synthetic/retarget Lance trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from sim.dexhandrl.constants import (
    DEFAULT_LANCE_PATH,
    DEXHAND021PRO_ACTIVE_DOF_NAMES,
    DEXHAND021PRO_CONTACT_BODY_NAMES,
)


SUPPORTED_RETARGET_SCHEMAS = {"dexhand_retarget_minimal_v1", "dexhand_synthetic_minimal_v1"}
SUPPORTED_HAND_FORMAT = "dexhand021pro_controls_world_v1"


@dataclass(frozen=True)
class DexHandTrajectory:
    """One row-oriented DexHandRL trajectory loaded from Lance."""

    lance_path: Path
    uuid: str
    row_id: int
    object_name: str
    action: str
    hand_dof: np.ndarray
    hand_finger_dof_full_raw: np.ndarray | None
    object_pos: np.ndarray
    object_quat_xyzw: np.ndarray
    expected_contact_mask: np.ndarray
    movement_start_frame: int
    movement_end_frame: int
    metadata: dict[str, Any]

    @property
    def frames(self) -> int:
        return int(self.hand_dof.shape[0])


def _import_lance() -> Any:
    try:
        return importlib.import_module("lance")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "DexHandRL Lance loading requires pylance (import name 'lance'). "
            "Install the target repo environment before running replay."
        ) from exc


def _unwrap_columns(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    for col in range(start, stop):
        out[:, col] = np.unwrap(out[:, col])
    return out


def _movement_range(uuid: str, object_name: str, total_frames: int, trajectory_info: dict[str, Any]) -> tuple[int, int]:
    object_move = trajectory_info.get("object_move") if isinstance(trajectory_info, dict) else None
    if not isinstance(object_move, list) or len(object_move) != 1:
        raise RuntimeError(f"trajectory {uuid} must contain exactly one object_move entry")
    move = object_move[0] or {}
    if move.get("object_name") != object_name:
        raise RuntimeError(
            f"trajectory {uuid} object_move.object_name={move.get('object_name')!r}, "
            f"expected {object_name!r}"
        )
    start = int(move["start_frame"])
    end = int(move["end_frame"])
    if start < 0 or end < start or end >= total_frames:
        raise RuntimeError(f"trajectory {uuid} invalid movement range {start}..{end}")
    return start, end


def _validate_row(row: dict[str, Any]) -> None:
    index = row.get("index") or {}
    metadata = row.get("trajectory_metadata") or {}
    if index.get("retarget_schema") not in SUPPORTED_RETARGET_SCHEMAS:
        raise RuntimeError(f"unsupported retarget schema {index.get('retarget_schema')!r}")
    if index.get("hand_model_type") != "dexhand021pro":
        raise RuntimeError(f"expected dexhand021pro, got {index.get('hand_model_type')!r}")
    if metadata.get("dexhand_dof_dim") != len(DEXHAND021PRO_ACTIVE_DOF_NAMES):
        raise RuntimeError(f"expected 22D dexhand dof, got {metadata.get('dexhand_dof_dim')!r}")
    if metadata.get("dexhand_dof_format") != SUPPORTED_HAND_FORMAT:
        raise RuntimeError(f"expected {SUPPORTED_HAND_FORMAT}, got {metadata.get('dexhand_dof_format')!r}")
    dof_names = tuple(metadata.get("dexhand_dof_names") or ())
    if dof_names and dof_names != DEXHAND021PRO_ACTIVE_DOF_NAMES:
        raise RuntimeError("Lance dexhand_dof_names do not match the IsaacGym 021pro active target order")


def _contact_mask_from_row(row: dict[str, Any], frames: int) -> np.ndarray:
    mask = np.zeros((frames, len(DEXHAND021PRO_CONTACT_BODY_NAMES)), dtype=np.float32)
    body_to_idx = {name: idx for idx, name in enumerate(DEXHAND021PRO_CONTACT_BODY_NAMES)}
    contact_by_frame = row.get("contact") or []
    if not isinstance(contact_by_frame, list):
        return mask
    for frame_idx, entries in enumerate(contact_by_frame[:frames]):
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            body_name = entry.get("body_name") or entry.get("joint_name")
            if body_name in body_to_idx:
                mask[frame_idx, body_to_idx[body_name]] = 1.0
    return mask


def list_trajectories(lance_path: Path | str = DEFAULT_LANCE_PATH, limit: int | None = 20) -> list[dict[str, Any]]:
    """Return lightweight trajectory index rows from the DexHandRL Lance dataset."""

    lance = _import_lance()
    ds = lance.dataset(str(lance_path))
    table = ds.to_table(columns=["index", "trajectory_metadata"], limit=limit, with_row_id=True)
    rows = []
    for fallback_row_id, row in enumerate(table.to_pylist()):
        index = row.get("index") or {}
        metadata = row.get("trajectory_metadata") or {}
        rows.append(
            {
                "row_id": int(row.get("_rowid", fallback_row_id)),
                "uuid": index.get("uuid"),
                "object_name": index.get("scene"),
                "action": index.get("gesture"),
                "frames": metadata.get("total_frames"),
                "schema": index.get("retarget_schema"),
            }
        )
    return rows


def load_trajectory(
    lance_path: Path | str = DEFAULT_LANCE_PATH,
    object_name: str = "cube1",
    action: str = "01",
    sequence_index: int = 0,
    uuid: str | None = None,
) -> DexHandTrajectory:
    """Load one DexHand021Pro trajectory by object/action or exact UUID."""

    lance = _import_lance()
    path = Path(lance_path)
    ds = lance.dataset(str(path))
    optional_columns = ["source", "contact"]
    schema_names = set(ds.schema.names)
    columns = ["index", "trajectory_metadata", "hands", "objects"]
    columns.extend(name for name in optional_columns if name in schema_names)
    if uuid:
        table = ds.to_table(
            columns=columns,
            filter=f"index.uuid = '{uuid}'",
            limit=2,
            with_row_id=True,
        )
    else:
        table = ds.to_table(
            columns=columns,
            filter=f"index.scene = '{object_name}' and index.gesture = '{action}'",
            limit=sequence_index + 1,
            with_row_id=True,
        )
    rows = table.to_pylist()
    if uuid:
        if len(rows) != 1:
            raise RuntimeError(f"expected exactly one Lance row for uuid={uuid}, got {len(rows)}")
        row = rows[0]
        row_id = int(row.get("_rowid", -1))
    else:
        if len(rows) <= sequence_index:
            raise RuntimeError(
                f"no Lance row for object={object_name!r} action={action!r} "
                f"sequence_index={sequence_index} in {path}"
            )
        row = rows[sequence_index]
        row_id = int(row.get("_rowid", sequence_index))

    _validate_row(row)
    index = row["index"] or {}
    metadata = row["trajectory_metadata"] or {}
    actual_object = str(index["scene"])
    actual_action = str(index["gesture"])
    total_frames = int(metadata["total_frames"])
    movement_start, movement_end = _movement_range(
        uuid=str(index["uuid"]),
        object_name=actual_object,
        total_frames=total_frames,
        trajectory_info=metadata["trajectory_info"],
    )

    hands = row.get("hands") or []
    objects = row.get("objects") or []
    if len(hands) != 1 or len(objects) != 1:
        raise RuntimeError("DexHandRL minimal Lance row must contain one hand and one object")
    hand_dof = np.asarray(hands[0]["dexhand_dof"], dtype=np.float32)
    raw_finger_dof = hands[0].get("dexhand_joint_angles_full_raw")
    hand_finger_dof_full_raw = (
        None if raw_finger_dof is None else np.asarray(raw_finger_dof, dtype=np.float32)
    )
    object_pos = np.asarray(objects[0]["pos"], dtype=np.float32)
    object_quat_xyzw = np.asarray(objects[0]["rot_quat_xyzw"], dtype=np.float32)
    if hand_dof.shape != (total_frames, len(DEXHAND021PRO_ACTIVE_DOF_NAMES)):
        raise RuntimeError(f"invalid hand_dof shape {hand_dof.shape}")
    if (
        hand_finger_dof_full_raw is not None
        and hand_finger_dof_full_raw.shape != (total_frames, 20)
    ):
        raise RuntimeError(f"invalid dexhand_joint_angles_full_raw shape {hand_finger_dof_full_raw.shape}")
    if object_pos.shape != (total_frames, 3):
        raise RuntimeError(f"invalid object_pos shape {object_pos.shape}")
    if object_quat_xyzw.shape != (total_frames, 4):
        raise RuntimeError(f"invalid object_quat shape {object_quat_xyzw.shape}")

    hand_dof = _unwrap_columns(hand_dof, 3, 6)
    expected_contact_mask = _contact_mask_from_row(row, frames=total_frames)
    return DexHandTrajectory(
        lance_path=path,
        uuid=str(index["uuid"]),
        row_id=row_id,
        object_name=actual_object,
        action=actual_action,
        hand_dof=hand_dof,
        hand_finger_dof_full_raw=hand_finger_dof_full_raw,
        object_pos=object_pos,
        object_quat_xyzw=object_quat_xyzw,
        expected_contact_mask=expected_contact_mask,
        movement_start_frame=movement_start,
        movement_end_frame=movement_end,
        metadata={"index": index, "trajectory_metadata": metadata, "source": row.get("source") or {}},
    )
