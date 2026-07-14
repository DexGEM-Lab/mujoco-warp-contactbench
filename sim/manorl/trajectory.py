"""Narrow Lance reader for the one accepted ManoRL trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from sim.manorl.contracts import (
    DATASET_ROW_INDEX,
    EXPECTED_DATASET_VERSION,
    JOINT_NAMES,
    OBJECT_INDEX,
    OBJECT_TYPE,
    REFERENCE_FRAME_COUNT,
    SOURCE_DATA_FPS,
    SOURCE_FRAME_COUNT,
    SOURCE_SLICE,
    TRAJECTORY_IDENTITY,
    TrajectoryIdentity,
)

LANCE_COLUMNS = ("index", "trajectory_metadata", "timestamp", "hands", "objects")


def _immutable(array: NDArray[np.floating[Any]], *, dtype: Any = np.float64) -> NDArray[Any]:
    result = np.ascontiguousarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


def rotvec_to_xyzw(rotvec: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Convert source axis-angle rotation vectors to normalized task XYZW quaternions."""

    vectors = np.asarray(rotvec, dtype=np.float64)
    if vectors.shape[-1:] != (3,) or not np.all(np.isfinite(vectors)):
        raise ValueError("rotation vectors must be finite with final dimension 3")
    quaternions = Rotation.from_rotvec(vectors).as_quat()
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    return quaternions


def xyzw_to_wxyz(quaternion: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape[-1:] != (4,):
        raise ValueError("quaternions must have final dimension 4")
    return values[..., (3, 0, 1, 2)]


def wxyz_to_xyzw(quaternion: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape[-1:] != (4,):
        raise ValueError("quaternions must have final dimension 4")
    return values[..., (1, 2, 3, 0)]


@dataclass(frozen=True)
class ReferenceTrajectory:
    identity: TrajectoryIdentity
    dataset_version: int
    source_indices: NDArray[np.int64]
    timestamps: NDArray[np.float64]
    q_ref: NDArray[np.float64]
    object_pos_raw: NDArray[np.float64]
    object_pos: NDArray[np.float64]
    object_quat_xyzw: NDArray[np.float64]
    object_z_shift: float

    def __post_init__(self) -> None:
        expected = REFERENCE_FRAME_COUNT
        shapes = {
            "source_indices": self.source_indices.shape == (expected,),
            "timestamps": self.timestamps.shape == (expected,),
            "q_ref": self.q_ref.shape == (expected, len(JOINT_NAMES)),
            "object_pos_raw": self.object_pos_raw.shape == (expected, 3),
            "object_pos": self.object_pos.shape == (expected, 3),
            "object_quat_xyzw": self.object_quat_xyzw.shape == (expected, 4),
        }
        invalid = [name for name, valid in shapes.items() if not valid]
        if invalid:
            raise ValueError(f"invalid reference trajectory shapes: {', '.join(invalid)}")
        for name in (
            "timestamps",
            "q_ref",
            "object_pos_raw",
            "object_pos",
            "object_quat_xyzw",
        ):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} contains non-finite values")
        if not np.isfinite(self.object_z_shift):
            raise ValueError("object_z_shift must be finite")
        if not np.allclose(np.linalg.norm(self.object_quat_xyzw, axis=1), 1.0, atol=1e-10):
            raise ValueError("object quaternions must be normalized")
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")


def _derive_identity(row: dict[str, Any]) -> str:
    source_path = row["index"].get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("row index must contain a non-empty source_path")
    identity = Path(source_path).parent.name
    if identity.count("_") != 2:
        raise ValueError(f"source_path does not encode an object_action_sequence: {source_path!r}")
    return identity


def _validate_row(row: dict[str, Any]) -> None:
    index = row["index"]
    metadata = row["trajectory_metadata"]
    expected = TRAJECTORY_IDENTITY
    if index["uuid"] != expected.uuid:
        raise ValueError(f"row UUID mismatch: {index['uuid']!r}")
    if index["file_uuid"] != expected.file_uuid:
        raise ValueError(f"file UUID mismatch: {index['file_uuid']!r}")
    if _derive_identity(row) != expected.identity:
        raise ValueError(f"trajectory identity mismatch: {_derive_identity(row)!r}")
    if metadata["object_names"][OBJECT_INDEX] != OBJECT_TYPE:
        raise ValueError(f"object index 0 must be {OBJECT_TYPE}")
    if metadata["hand_names"] != ["right"]:
        raise ValueError(f"expected one right hand, got {metadata['hand_names']!r}")
    if int(metadata["total_frames"]) != SOURCE_FRAME_COUNT:
        raise ValueError(
            f"expected {SOURCE_FRAME_COUNT} source frames, got {metadata['total_frames']}"
        )
    if int(metadata["data_fps"]) != SOURCE_DATA_FPS:
        raise ValueError(f"expected {SOURCE_DATA_FPS} Hz source data, got {metadata['data_fps']}")
    movement = metadata["trajectory_info"]["object_move"]
    expected_movement = [
        {
            "object_name": OBJECT_TYPE,
            "start_frame": expected.movement_start_raw,
            "end_frame": expected.movement_end_raw,
        }
    ]
    if movement != expected_movement:
        raise ValueError(f"movement range mismatch: {movement!r}")
    if len(row["hands"]) != 1 or len(row["objects"]) <= OBJECT_INDEX:
        raise ValueError("expected one hand and object index 0")


def _initial_support_shift(
    initial_position: NDArray[np.float64],
    initial_quaternion_xyzw: NDArray[np.float64],
) -> float:
    from sim.manorl.assets import object_collision_vertices

    rotated = Rotation.from_quat(initial_quaternion_xyzw).apply(
        object_collision_vertices().copy()
    )
    return -float(np.min(rotated[:, 2] + initial_position[2]))


def trajectory_from_row(row: dict[str, Any], dataset_version: int) -> ReferenceTrajectory:
    """Validate one decoded Lance row and construct the immutable accepted slice."""

    _validate_row(row)
    source_count = int(row["trajectory_metadata"]["total_frames"])
    timestamps_all = np.asarray(row["timestamp"], dtype=np.float64)
    q_all = np.asarray(row["hands"][0]["urdf_dof"], dtype=np.float64)
    object_pos_all = np.asarray(row["objects"][OBJECT_INDEX]["pos"], dtype=np.float64)
    object_rotvec_all = np.asarray(row["objects"][OBJECT_INDEX]["rot_aa"], dtype=np.float64)
    expected_shapes = {
        "timestamp": (source_count,),
        "urdf_dof": (source_count, len(JOINT_NAMES)),
        "object position": (source_count, 3),
        "object axis-angle": (source_count, 3),
    }
    arrays = {
        "timestamp": timestamps_all,
        "urdf_dof": q_all,
        "object position": object_pos_all,
        "object axis-angle": object_rotvec_all,
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(f"{name} shape {arrays[name].shape} != {expected_shape}")
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"{name} contains non-finite values")
    timestamp_delta = np.diff(timestamps_all)
    if np.any(timestamp_delta <= 0):
        raise ValueError("source timestamps are not strictly increasing")
    # The accepted capture intervals range from 1 to 65 ms despite the source
    # metadata reporting 111 Hz. The source control loop uses its fixed 5 ms
    # schedule, so timestamps establish ordering only and never simulation time.

    start, stop = SOURCE_SLICE
    q_ref = q_all[start:stop].copy()
    q_ref[:, 3:6] = np.unwrap(q_ref[:, 3:6], axis=0, period=2.0 * np.pi)
    object_pos_raw = object_pos_all[start:stop].copy()
    object_pos = object_pos_raw.copy()
    object_quat_xyzw = rotvec_to_xyzw(object_rotvec_all[start:stop])
    z_shift = _initial_support_shift(object_pos[0], object_quat_xyzw[0])
    object_pos[:, 2] += z_shift

    return ReferenceTrajectory(
        identity=TRAJECTORY_IDENTITY,
        dataset_version=dataset_version,
        source_indices=_immutable(np.arange(start, stop), dtype=np.int64),
        timestamps=_immutable(timestamps_all[start:stop]),
        q_ref=_immutable(q_ref),
        object_pos_raw=_immutable(object_pos_raw),
        object_pos=_immutable(object_pos),
        object_quat_xyzw=_immutable(object_quat_xyzw),
        object_z_shift=z_shift,
    )


def load_reference_trajectory(
    dataset_path: str | Path = TRAJECTORY_IDENTITY.dataset_path,
    *,
    expected_dataset_version: int = EXPECTED_DATASET_VERSION,
) -> ReferenceTrajectory:
    """Load exactly row 1 and only the five required top-level Lance columns."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"required Lance dataset is absent: {path}")
    try:
        import lance
    except ImportError as exc:
        raise RuntimeError("pylance is required to read the accepted trajectory") from exc

    dataset = lance.dataset(str(path))
    version_value = getattr(dataset, "version", None)
    if version_value is None:
        raise ValueError("Lance dataset did not expose a version; identity cannot be fixed")
    dataset_version = int(version_value)
    if dataset_version != expected_dataset_version:
        raise ValueError(
            f"dataset version {dataset_version} != accepted version {expected_dataset_version}"
        )
    table = dataset.take([DATASET_ROW_INDEX], columns=list(LANCE_COLUMNS))
    rows = table.to_pylist()
    if len(rows) != 1:
        raise ValueError(f"dataset.take returned {len(rows)} rows, expected exactly one")
    return trajectory_from_row(rows[0], dataset_version)
