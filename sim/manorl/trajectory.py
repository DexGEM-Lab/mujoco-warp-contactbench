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
GENERATED_CUBE1_DATASET_PATH = Path(
    "/mnt/nas-222-project/mocap/dataAugmentation/for_retargeting/new_all_with_keypoints/"
    "lance_new_all_generated_mano/new_all_generated_mano.lance"
)
GENERATED_CUBE1_DATASET_VERSION = 236
GENERATED_CUBE1_ROW_INDEX = 507
GENERATED_CUBE1_UUID = "00f45dd5-6699-5be1-8948-d6f7b623da48"
GENERATED_CUBE1_MOVEMENT = (267, 541)
GENERATED_PADDING = 250
CUBE1_ACTION_01_BATCH_ROWS = (
    (0, "97f4b8a1-19f4-5c1c-a051-162f21fcfc84", "cube1_01_003", 1481, 681, 971),
    (1, "d5bc2bc6-9458-52d0-bccc-66c9ec21bae3", "cube1_01_009", 1373, 690, 982),
    (2, "7ff8a804-55f7-5bce-be50-e09dbbfb4018", "cube1_01_011", 1348, 486, 760),
    (3, "9fdd20ef-8b9e-5ce3-8bbc-d0c60c3c55f2", "cube1_01_012", 1562, 721, 1036),
    (4, "a700b2bd-bb93-5e62-a120-afb2807eb4a6", "cube1_01_014", 1111, 460, 757),
    (5, "2024ca74-970e-5ad8-bb8c-d49f8615ac9a", "cube1_01_025", 1258, 587, 858),
    (7, "4adcb856-a43c-5fab-9a88-f0726797ce41", "cube1_01_042", 1302, 676, 956),
    (14, "0063573a-2137-550f-9bb1-cecd5b92f71e", "cube1_01_092", 1118, 458, 697),
    (15, "bde3b93e-0679-5ce3-96b7-6862b9b44ef7", "cube1_01_093", 1530, 524, 750),
    (16, "ec08a573-5050-552d-8175-8e18343f2cc0", "cube1_01_094", 1043, 502, 745),
)


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
        expected = int(self.source_indices.shape[0])
        shapes = {
            "source_indices": expected >= 2,
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


@dataclass(frozen=True)
class TrajectoryBatch:
    """Immutable per-world reference assignments, mirroring Isaac batch loading."""

    trajectories: tuple[ReferenceTrajectory, ...]

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError("trajectory batch must be non-empty")
        if any(not isinstance(trajectory, ReferenceTrajectory) for trajectory in self.trajectories):
            raise TypeError("trajectory batch must contain ReferenceTrajectory values")
        if any(trajectory.identity.identity.split("_")[0] != OBJECT_TYPE for trajectory in self.trajectories):
            raise ValueError("trajectory batch must contain only cube1 trajectories")

    @property
    def num_envs(self) -> int:
        return len(self.trajectories)

    @property
    def lengths(self) -> NDArray[np.int64]:
        return _immutable(np.asarray([len(trajectory.q_ref) for trajectory in self.trajectories]), dtype=np.int64)


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


def generated_cube1_row_507_from_row(row: dict[str, Any], dataset_version: int) -> ReferenceTrajectory:
    """Build the explicitly selected generated cube1 trajectory.

    The generated dataset does not carry a source ``gesture``/``source_path``
    label, so this selector deliberately binds a versioned row and UUID rather
    than claiming it is the accepted ``cube1_01_009`` source action.
    """

    if dataset_version != GENERATED_CUBE1_DATASET_VERSION:
        raise ValueError(
            f"generated dataset version {dataset_version} != {GENERATED_CUBE1_DATASET_VERSION}"
        )
    index = row["index"]
    metadata = row["trajectory_metadata"]
    if index.get("uuid") != GENERATED_CUBE1_UUID or index.get("scene") != OBJECT_TYPE:
        raise ValueError("generated row does not match the selected cube1 UUID/scene")
    if index.get("is_generated") is not True:
        raise ValueError("selected generated row must declare is_generated=true")
    if metadata.get("object_names") != [OBJECT_TYPE] or metadata.get("hand_names") != ["right"]:
        raise ValueError("selected generated row must contain one right hand and cube1")
    if int(metadata.get("total_frames", -1)) != 735 or int(metadata.get("data_fps", -1)) != 100:
        raise ValueError("selected generated row frame/fps contract changed")
    movement = metadata.get("trajectory_info", {}).get("object_move")
    expected_movement = [
        {"object_name": OBJECT_TYPE, "start_frame": GENERATED_CUBE1_MOVEMENT[0], "end_frame": GENERATED_CUBE1_MOVEMENT[1]}
    ]
    if movement != expected_movement:
        raise ValueError(f"selected generated row movement changed: {movement!r}")
    if len(row["hands"]) != 1 or len(row["objects"]) != 1:
        raise ValueError("selected generated row must contain one hand and one object")

    source_count = int(metadata["total_frames"])
    timestamps_all = np.asarray(row["timestamp"], dtype=np.float64)
    q_all = np.asarray(row["hands"][0]["urdf_dof"], dtype=np.float64)
    object_pos_all = np.asarray(row["objects"][0]["pos"], dtype=np.float64)
    object_rotvec_all = np.asarray(row["objects"][0]["rot_aa"], dtype=np.float64)
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
        if arrays[name].shape != expected_shape or not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"generated {name} is not finite with shape {expected_shape}")
    if np.any(np.diff(timestamps_all) <= 0):
        raise ValueError("generated source timestamps are not strictly increasing")

    movement_start, movement_end = GENERATED_CUBE1_MOVEMENT
    start = max(0, movement_start - GENERATED_PADDING)
    stop = min(source_count, movement_end + GENERATED_PADDING)
    if stop - start < 2:
        raise ValueError("generated movement window does not provide two replay references")
    q_ref = q_all[start:stop].copy()
    q_ref[:, 3:6] = np.unwrap(q_ref[:, 3:6], axis=0, period=2.0 * np.pi)
    object_pos_raw = object_pos_all[start:stop].copy()
    object_quat_xyzw = rotvec_to_xyzw(object_rotvec_all[start:stop])
    z_shift = _initial_support_shift(object_pos_raw[0], object_quat_xyzw[0])
    object_pos = object_pos_raw.copy()
    object_pos[:, 2] += z_shift
    identity = TrajectoryIdentity(
        dataset_path=str(GENERATED_CUBE1_DATASET_PATH),
        dataset_version=dataset_version,
        row_index=GENERATED_CUBE1_ROW_INDEX,
        object_index=0,
        uuid=GENERATED_CUBE1_UUID,
        file_uuid="",
        identity="cube1_generated_row_507",
        source_start=start,
        source_stop=stop,
        movement_start_raw=movement_start,
        movement_end_raw=movement_end,
    )
    return ReferenceTrajectory(
        identity=identity,
        dataset_version=dataset_version,
        source_indices=_immutable(np.arange(start, stop), dtype=np.int64),
        timestamps=_immutable(timestamps_all[start:stop]),
        q_ref=_immutable(q_ref),
        object_pos_raw=_immutable(object_pos_raw),
        object_pos=_immutable(object_pos),
        object_quat_xyzw=_immutable(object_quat_xyzw),
        object_z_shift=z_shift,
    )


def load_generated_cube1_row_507(
    dataset_path: str | Path = GENERATED_CUBE1_DATASET_PATH,
) -> ReferenceTrajectory:
    """Load exactly the selected generated cube1 row 507 from Lance version 236."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"generated Lance dataset is absent: {path}")
    try:
        import lance
    except ImportError as exc:
        raise RuntimeError("pylance is required to read the generated trajectory") from exc
    dataset = lance.dataset(str(path))
    version = int(getattr(dataset, "version", -1))
    if version != GENERATED_CUBE1_DATASET_VERSION:
        raise ValueError(f"generated dataset version {version} != {GENERATED_CUBE1_DATASET_VERSION}")
    rows = dataset.take([GENERATED_CUBE1_ROW_INDEX], columns=list(LANCE_COLUMNS)).to_pylist()
    if len(rows) != 1:
        raise ValueError("generated dataset did not return exactly row 507")
    return generated_cube1_row_507_from_row(rows[0], version)


def _cube1_action_01_trajectory_from_row(
    row: dict[str, Any],
    dataset_version: int,
    *,
    row_index: int,
    uuid: str,
    identity: str,
    source_count: int,
    movement_start: int,
    movement_end: int,
) -> ReferenceTrajectory:
    """Decode one explicit cube1/action-01 Lance assignment with full padding."""

    if dataset_version != EXPECTED_DATASET_VERSION:
        raise ValueError(f"dataset version {dataset_version} != accepted version {EXPECTED_DATASET_VERSION}")
    index = row["index"]
    metadata = row["trajectory_metadata"]
    expected_path = f"cube1/{identity}/{identity}_mano.npy"
    if (
        index.get("uuid") != uuid
        or index.get("scene") != OBJECT_TYPE
        or index.get("gesture") != "01"
        or index.get("source_path") != expected_path
    ):
        raise ValueError(f"batch row {row_index} identity contract changed")
    if metadata.get("object_names") != [OBJECT_TYPE] or metadata.get("hand_names") != ["right"]:
        raise ValueError(f"batch row {row_index} must contain one right hand and cube1")
    if int(metadata.get("total_frames", -1)) != source_count or int(metadata.get("data_fps", -1)) != SOURCE_DATA_FPS:
        raise ValueError(f"batch row {row_index} source frame/fps contract changed")
    movement = metadata.get("trajectory_info", {}).get("object_move")
    if movement != [{"object_name": OBJECT_TYPE, "start_frame": movement_start, "end_frame": movement_end}]:
        raise ValueError(f"batch row {row_index} movement contract changed")
    if len(row["hands"]) != 1 or len(row["objects"]) != 1:
        raise ValueError(f"batch row {row_index} must contain one hand and one object")

    timestamps_all = np.asarray(row["timestamp"], dtype=np.float64)
    q_all = np.asarray(row["hands"][0]["urdf_dof"], dtype=np.float64)
    object_pos_all = np.asarray(row["objects"][0]["pos"], dtype=np.float64)
    object_rotvec_all = np.asarray(row["objects"][0]["rot_aa"], dtype=np.float64)
    arrays = {
        "timestamp": timestamps_all,
        "urdf_dof": q_all,
        "object position": object_pos_all,
        "object axis-angle": object_rotvec_all,
    }
    expected_shapes = {
        "timestamp": (source_count,),
        "urdf_dof": (source_count, len(JOINT_NAMES)),
        "object position": (source_count, 3),
        "object axis-angle": (source_count, 3),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape or not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"batch row {row_index} invalid {name}")
    if np.any(np.diff(timestamps_all) <= 0):
        raise ValueError(f"batch row {row_index} timestamps are not strictly increasing")

    start = movement_start - GENERATED_PADDING
    stop = movement_end + GENERATED_PADDING
    if start < 0 or stop > source_count:
        raise ValueError(f"batch row {row_index} lacks the required 250-frame movement padding")
    q_ref = q_all[start:stop].copy()
    q_ref[:, 3:6] = np.unwrap(q_ref[:, 3:6], axis=0, period=2.0 * np.pi)
    object_pos_raw = object_pos_all[start:stop].copy()
    object_quat_xyzw = rotvec_to_xyzw(object_rotvec_all[start:stop])
    z_shift = _initial_support_shift(object_pos_raw[0], object_quat_xyzw[0])
    object_pos = object_pos_raw.copy()
    object_pos[:, 2] += z_shift
    identity_contract = TrajectoryIdentity(
        dataset_path=str(TRAJECTORY_IDENTITY.dataset_path),
        dataset_version=dataset_version,
        row_index=row_index,
        object_index=0,
        uuid=uuid,
        file_uuid=str(index.get("file_uuid", "")),
        identity=identity,
        source_start=start,
        source_stop=stop,
        movement_start_raw=movement_start,
        movement_end_raw=movement_end,
    )
    return ReferenceTrajectory(
        identity=identity_contract,
        dataset_version=dataset_version,
        source_indices=_immutable(np.arange(start, stop), dtype=np.int64),
        timestamps=_immutable(timestamps_all[start:stop]),
        q_ref=_immutable(q_ref),
        object_pos_raw=_immutable(object_pos_raw),
        object_pos=_immutable(object_pos),
        object_quat_xyzw=_immutable(object_quat_xyzw),
        object_z_shift=z_shift,
    )


def load_cube1_action_01_batch10(
    dataset_path: str | Path = TRAJECTORY_IDENTITY.dataset_path,
) -> TrajectoryBatch:
    """Load ten explicit, fully padded cube1/action-01 trajectory assignments."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"required Lance dataset is absent: {path}")
    try:
        import lance
    except ImportError as exc:
        raise RuntimeError("pylance is required to read the cube1 action-01 batch") from exc
    dataset = lance.dataset(str(path))
    version = int(getattr(dataset, "version", -1))
    if version != EXPECTED_DATASET_VERSION:
        raise ValueError(f"dataset version {version} != accepted version {EXPECTED_DATASET_VERSION}")
    row_indices = [entry[0] for entry in CUBE1_ACTION_01_BATCH_ROWS]
    rows = dataset.take(row_indices, columns=list(LANCE_COLUMNS)).to_pylist()
    if len(rows) != len(CUBE1_ACTION_01_BATCH_ROWS):
        raise ValueError("cube1 action-01 batch did not return all selected rows")
    trajectories = tuple(
        _cube1_action_01_trajectory_from_row(
            row,
            version,
            row_index=row_index,
            uuid=uuid,
            identity=identity,
            source_count=source_count,
            movement_start=movement_start,
            movement_end=movement_end,
        )
        for row, (row_index, uuid, identity, source_count, movement_start, movement_end)
        in zip(rows, CUBE1_ACTION_01_BATCH_ROWS, strict=True)
    )
    return TrajectoryBatch(trajectories)
