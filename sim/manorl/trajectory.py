"""Versioned Lance readers and deterministic ManoRL training selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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
LANCE_DISCOVERY_COLUMNS = ("index", "trajectory_metadata")
LANCE_DECODE_CHUNK_SIZE = 128
TRAJECTORY_IDENTITY_SCHEMA = "object_action_sequence"
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
    resolved_pairs: tuple[ObjectActionPair, ...] = ()
    selection_mode: str | None = None

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError("trajectory batch must be non-empty")
        if any(not isinstance(trajectory, ReferenceTrajectory) for trajectory in self.trajectories):
            raise TypeError("trajectory batch must contain ReferenceTrajectory values")
        if any(not isinstance(pair, ObjectActionPair) for pair in self.resolved_pairs):
            raise TypeError("resolved_pairs must contain ObjectActionPair values")
        if self.selection_mode not in (None, "pairs", "all"):
            raise ValueError("selection_mode must be 'pairs', 'all', or None")

    @property
    def num_envs(self) -> int:
        return len(self.trajectories)

    @property
    def lengths(self) -> NDArray[np.int64]:
        return _immutable(np.asarray([len(trajectory.q_ref) for trajectory in self.trajectories]), dtype=np.int64)


@dataclass(frozen=True, order=True)
class ObjectActionPair:
    """One exact Lance object/action pair with a canonical two-digit action ID."""

    object_type: str
    action_id: str

    def __post_init__(self) -> None:
        object_type = str(self.object_type)
        action_id = str(self.action_id)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", object_type) is None:
            raise ValueError(
                "trajectory selector object must use letters, digits, '.', or '-' and cannot contain '_'"
            )
        if not action_id.isdigit() or not 1 <= int(action_id) <= 50:
            raise ValueError("trajectory selector action must be a source action id in [1, 50]")
        object.__setattr__(self, "object_type", object_type)
        object.__setattr__(self, "action_id", f"{int(action_id):02d}")

    @property
    def canonical(self) -> str:
        return f"{self.object_type}:{self.action_id}"


def parse_trajectory_selector(selector: str) -> tuple[ObjectActionPair, ...] | None:
    """Parse ``all`` or a comma-separated list of exact ``object:action`` pairs.

    ``None`` means all eligible pairs. Explicit pairs are de-duplicated after
    action normalization and returned in deterministic sorted order.
    """

    if not isinstance(selector, str) or not selector.strip():
        raise ValueError("trajectory selector must be 'all' or a non-empty object:action list")
    value = selector.strip()
    if value == "all":
        return None
    if value.lower() == "all":
        raise ValueError("trajectory selector all mode must be spelled exactly 'all'")
    tokens = value.split(",")
    if any(not token.strip() for token in tokens):
        raise ValueError("trajectory selector contains an empty pair")
    pairs: list[ObjectActionPair] = []
    for token in tokens:
        fields = token.strip().split(":")
        if len(fields) != 2 or not all(field.strip() for field in fields):
            raise ValueError(
                f"malformed trajectory selector pair {token!r}; expected object:action"
            )
        pairs.append(ObjectActionPair(fields[0].strip(), fields[1].strip()))
    duplicates = sorted(pair.canonical for pair in set(pairs) if pairs.count(pair) > 1)
    if duplicates:
        raise ValueError(f"duplicate trajectory selector pair(s): {', '.join(duplicates)}")
    return tuple(sorted(pairs))


@dataclass(frozen=True)
class TrajectorySelection:
    """Versioned Lance query over one pair, explicit pairs, or every pair."""

    object_type: str = "cube1"
    gesture: str = "01"
    selector: str | None = None
    dataset_path: Path = Path(TRAJECTORY_IDENTITY.dataset_path)
    expected_dataset_version: int = EXPECTED_DATASET_VERSION
    pre_padding: int = GENERATED_PADDING
    post_padding: int = GENERATED_PADDING

    def __post_init__(self) -> None:
        ObjectActionPair(self.object_type, self.gesture)
        if self.selector is not None:
            parse_trajectory_selector(self.selector)
        if self.pre_padding < 0 or self.post_padding < 0:
            raise ValueError("trajectory padding must be non-negative")

    @property
    def action_id(self) -> str:
        return ObjectActionPair(self.object_type, self.gesture).action_id

    @property
    def requested_pairs(self) -> tuple[ObjectActionPair, ...] | None:
        if self.selector is None:
            return (ObjectActionPair(self.object_type, self.gesture),)
        return parse_trajectory_selector(self.selector)

    @property
    def mode(self) -> str:
        return "all" if self.requested_pairs is None else "pairs"

    @property
    def canonical_selector(self) -> str:
        pairs = self.requested_pairs
        return "all" if pairs is None else ",".join(pair.canonical for pair in pairs)

    @property
    def require_full_padding(self) -> bool:
        """Keep the historical single-pair slice exact; clip opt-in selectors."""

        return self.selector is None


@dataclass(frozen=True)
class EnvTrajectoryAssignment:
    """Stable provenance for the fixed trajectory owned by one vector environment."""

    env_id: int
    trajectory: ReferenceTrajectory

    @property
    def identity(self) -> TrajectoryIdentity:
        return self.trajectory.identity


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
    object_type: str = OBJECT_TYPE,
) -> float:
    from sim.manorl.assets import object_collision_vertices

    rotated = Rotation.from_quat(initial_quaternion_xyzw).apply(
        object_collision_vertices(object_type).copy()
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


def _selection_row_sort_key(row_index_and_row: tuple[int, dict[str, Any]]) -> tuple[str, str, int, int]:
    row_index, row = row_index_and_row
    index = row["index"]
    source_path = str(index.get("source_path", ""))
    identity = _derive_identity(row)
    object_type, action_id, sequence = identity.split("_")
    if index.get("scene") != object_type or str(index.get("gesture", "")).zfill(2) != action_id:
        raise ValueError(f"Lance index/source_path identity mismatch at row {row_index}: {source_path!r}")
    if not sequence.isdigit():
        raise ValueError(f"Lance source_path sequence is not numeric at row {row_index}: {source_path!r}")
    return object_type, action_id, int(sequence), row_index


@dataclass(frozen=True)
class _TrajectoryCandidate:
    row_index: int
    pair: ObjectActionPair
    identity: str
    sequence: int


def _candidate_from_metadata_row(
    row: dict[str, Any], *, row_index: int, selection: TrajectorySelection
) -> _TrajectoryCandidate | None:
    """Return lightweight eligibility metadata without decoding frame arrays."""

    index = row.get("index")
    metadata = row.get("trajectory_metadata")
    if not isinstance(index, dict) or not isinstance(metadata, dict):
        return None
    # The accepted s02 source index predates the optional ``is_generated``
    # field.  Missing means a source row here; an explicit true value is the
    # only generated-row marker accepted by the contract.
    if index.get("is_generated", False) is not False:
        return None
    try:
        identity = _derive_identity(row)
    except (KeyError, ValueError):
        return None
    fields = identity.split("_")
    if len(fields) != 3 or not fields[2].isdigit():
        return None
    object_type, action_raw, sequence_raw = fields
    try:
        pair = ObjectActionPair(object_type, action_raw)
        index_pair = ObjectActionPair(str(index.get("scene", "")), str(index.get("gesture", "")))
    except ValueError:
        return None
    if pair != index_pair or action_raw != pair.action_id:
        return None
    source_path = str(index.get("source_path", ""))
    if Path(source_path).parent.name != identity:
        return None
    object_names = metadata.get("object_names")
    if (
        not isinstance(object_names, list)
        or object_type not in object_names
        or metadata.get("hand_names") != ["right"]
    ):
        return None
    try:
        source_count = int(metadata["total_frames"])
        movement = metadata["trajectory_info"]["object_move"]
        movement_entry = next(
            item for item in movement if item.get("object_name") == object_type
        )
        start_raw = int(movement_entry["start_frame"])
        end_raw = int(movement_entry["end_frame"])
    except (KeyError, TypeError, ValueError, StopIteration):
        return None
    requested_start = start_raw - selection.pre_padding
    requested_stop = end_raw + selection.post_padding
    if selection.require_full_padding and (
        requested_start < 0 or requested_stop > source_count
    ):
        return None
    start = max(0, requested_start)
    stop = min(source_count, requested_stop)
    if stop - start < 2:
        return None
    return _TrajectoryCandidate(
        row_index=row_index,
        pair=pair,
        identity=identity,
        sequence=int(sequence_raw),
    )


def _discover_trajectory_candidates(
    dataset: Any, selection: TrajectorySelection
) -> tuple[tuple[ObjectActionPair, ...], dict[ObjectActionPair, tuple[_TrajectoryCandidate, ...]]]:
    """Resolve pairs from lightweight Lance index/metadata columns only."""

    discovery_rows = dataset.to_table(columns=list(LANCE_DISCOVERY_COLUMNS)).to_pylist()
    candidates: dict[ObjectActionPair, list[_TrajectoryCandidate]] = {}
    for row_index, row in enumerate(discovery_rows):
        candidate = _candidate_from_metadata_row(row, row_index=row_index, selection=selection)
        if candidate is not None:
            candidates.setdefault(candidate.pair, []).append(candidate)

    requested_pairs = selection.requested_pairs
    if requested_pairs is None:
        resolved_pairs = tuple(sorted(candidates))
        if not resolved_pairs:
            raise LookupError(
                "no eligible non-generated Lance object/action pairs with exact "
                f"{TRAJECTORY_IDENTITY_SCHEMA} identities"
            )
    else:
        unmatched = tuple(pair for pair in requested_pairs if pair not in candidates)
        if unmatched:
            available = ", ".join(pair.canonical for pair in sorted(candidates)) or "none"
            missing = ", ".join(pair.canonical for pair in unmatched)
            raise LookupError(
                f"trajectory selector pair(s) matched no eligible rows: {missing}; "
                f"available pairs: {available}"
            )
        resolved_pairs = requested_pairs

    resolved_candidates = {
        pair: tuple(
            sorted(candidates[pair], key=lambda item: (item.sequence, item.identity, item.row_index))
        )
        for pair in resolved_pairs
    }
    return resolved_pairs, resolved_candidates


def _selected_trajectory_from_row(
    row: dict[str, Any],
    dataset_version: int,
    *,
    row_index: int,
    selection: TrajectorySelection,
    expected_pair: ObjectActionPair,
) -> ReferenceTrajectory:
    """Decode one fully padded source row selected by object and gesture."""

    index = row["index"]
    metadata = row["trajectory_metadata"]
    identity = _derive_identity(row)
    object_type, action_id, _ = identity.split("_")
    if ObjectActionPair(object_type, action_id) != expected_pair:
        raise ValueError(f"row {row_index} is not {expected_pair.canonical}")
    if index.get("scene") != object_type or str(index.get("gesture", "")).zfill(2) != action_id:
        raise ValueError(f"row {row_index} Lance index identity does not match source_path")
    object_names = metadata.get("object_names")
    if not isinstance(object_names, list) or object_type not in object_names:
        raise ValueError(f"row {row_index} does not contain selected object {object_type!r}")
    object_index = object_names.index(object_type)
    if metadata.get("hand_names") != ["right"] or len(row.get("hands", [])) != 1:
        raise ValueError(f"row {row_index} must contain exactly one right hand")
    if object_index >= len(row.get("objects", [])):
        raise ValueError(f"row {row_index} selected object index is absent from object state")
    source_count = int(metadata.get("total_frames", -1))
    movement = metadata.get("trajectory_info", {}).get("object_move", [])
    entry = next((item for item in movement if item.get("object_name") == object_type), None)
    if entry is None:
        raise ValueError(f"row {row_index} has no object_move entry for {object_type!r}")
    start_raw, end_raw = int(entry["start_frame"]), int(entry["end_frame"])
    requested_start = start_raw - selection.pre_padding
    requested_stop = end_raw + selection.post_padding
    if selection.require_full_padding and (
        requested_start < 0 or requested_stop > source_count
    ):
        raise ValueError(
            f"row {row_index} cannot provide [{start_raw}-{selection.pre_padding}, {end_raw}+{selection.post_padding})"
        )
    start = max(0, requested_start)
    stop = min(source_count, requested_stop)
    if stop - start < 2:
        raise ValueError(f"row {row_index} selected source window has fewer than two frames")
    timestamps_all = np.asarray(row["timestamp"], dtype=np.float64)
    q_all = np.asarray(row["hands"][0]["urdf_dof"], dtype=np.float64)
    object_pos_all = np.asarray(row["objects"][object_index]["pos"], dtype=np.float64)
    object_rotvec_all = np.asarray(row["objects"][object_index]["rot_aa"], dtype=np.float64)
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
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape or not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"row {row_index} has invalid {name}")
    if np.any(np.diff(timestamps_all) <= 0):
        raise ValueError(f"row {row_index} timestamps are not strictly increasing")
    q_ref = q_all[start:stop].copy()
    q_ref[:, 3:6] = np.unwrap(q_ref[:, 3:6], axis=0, period=2.0 * np.pi)
    object_pos_raw = object_pos_all[start:stop].copy()
    object_quat_xyzw = rotvec_to_xyzw(object_rotvec_all[start:stop])
    z_shift = _initial_support_shift(
        object_pos_raw[0], object_quat_xyzw[0], object_type
    )
    object_pos = object_pos_raw.copy()
    object_pos[:, 2] += z_shift
    return ReferenceTrajectory(
        identity=TrajectoryIdentity(
            dataset_path=str(selection.dataset_path),
            dataset_version=dataset_version,
            row_index=row_index,
            object_index=object_index,
            uuid=str(index["uuid"]),
            file_uuid=str(index.get("file_uuid", "")),
            identity=identity,
            source_start=start,
            source_stop=stop,
            movement_start_raw=start_raw,
            movement_end_raw=end_raw,
        ),
        dataset_version=dataset_version,
        source_indices=_immutable(np.arange(start, stop), dtype=np.int64),
        timestamps=_immutable(timestamps_all[start:stop]),
        q_ref=_immutable(q_ref),
        object_pos_raw=_immutable(object_pos_raw),
        object_pos=_immutable(object_pos),
        object_quat_xyzw=_immutable(object_quat_xyzw),
        object_z_shift=z_shift,
    )


def _decode_valid_candidates(
    dataset: Any,
    dataset_version: int,
    *,
    selection: TrajectorySelection,
    resolved_pairs: tuple[ObjectActionPair, ...],
    candidates_by_pair: dict[ObjectActionPair, tuple[_TrajectoryCandidate, ...]],
    limits: dict[ObjectActionPair, int],
) -> dict[ObjectActionPair, tuple[ReferenceTrajectory, ...]]:
    """Decode valid pair candidates in cross-pair, bounded-memory chunks."""

    if any(limits.get(pair, 0) < 1 for pair in resolved_pairs):
        raise ValueError("candidate decode limits must be positive")
    decoded: dict[ObjectActionPair, list[ReferenceTrajectory]] = {
        pair: [] for pair in resolved_pairs
    }
    rejected: dict[ObjectActionPair, list[str]] = {pair: [] for pair in resolved_pairs}
    cursors = {pair: 0 for pair in resolved_pairs}
    while True:
        requests: list[tuple[ObjectActionPair, _TrajectoryCandidate]] = []
        requested_counts = {pair: 0 for pair in resolved_pairs}
        while len(requests) < LANCE_DECODE_CHUNK_SIZE:
            progressed = False
            for pair in resolved_pairs:
                candidates = candidates_by_pair[pair]
                remaining = limits[pair] - len(decoded[pair]) - requested_counts[pair]
                if remaining <= 0 or cursors[pair] >= len(candidates):
                    continue
                candidate = candidates[cursors[pair]]
                cursors[pair] += 1
                requested_counts[pair] += 1
                requests.append((pair, candidate))
                progressed = True
                if len(requests) == LANCE_DECODE_CHUNK_SIZE:
                    break
            if not progressed:
                break
        if not requests:
            break
        rows = dataset.take(
            [candidate.row_index for _, candidate in requests],
            columns=list(LANCE_COLUMNS),
        ).to_pylist()
        if len(rows) != len(requests):
            raise ValueError("Lance did not return every candidate trajectory row")
        for (pair, candidate), row in zip(requests, rows, strict=True):
            try:
                trajectory = _selected_trajectory_from_row(
                    row,
                    dataset_version,
                    row_index=candidate.row_index,
                    selection=selection,
                    expected_pair=pair,
                )
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                rejected[pair].append(f"row {candidate.row_index}: {exc}")
                continue
            decoded[pair].append(trajectory)
    for pair in resolved_pairs:
        if decoded[pair]:
            continue
        detail = "; ".join(rejected[pair][:4]) or "no full rows were decoded"
        raise LookupError(f"no valid Lance trajectories for {pair.canonical}; {detail}")
    return {pair: tuple(decoded[pair]) for pair in resolved_pairs}


def load_assigned_trajectory_batch(selection: TrajectorySelection, *, num_envs: int) -> TrajectoryBatch:
    """Discover eligible pairs and deterministically assign them to vector worlds.

    Pair slots round-robin over sorted resolved pairs; each pair independently
    round-robins its sorted exact-identity trajectories. Full rows are decoded
    in bounded chunks and malformed candidates are skipped before assignment.
    """

    if not isinstance(selection, TrajectorySelection):
        raise TypeError("selection must be a TrajectorySelection")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if not selection.dataset_path.exists():
        raise FileNotFoundError(f"Lance dataset is absent: {selection.dataset_path}")
    try:
        import lance
    except ImportError as exc:
        raise RuntimeError("pylance is required to assign ManoRL trajectories") from exc
    dataset = lance.dataset(str(selection.dataset_path))
    version = int(getattr(dataset, "version", -1))
    if version != selection.expected_dataset_version:
        raise ValueError(f"dataset version {version} != requested {selection.expected_dataset_version}")
    resolved_pairs, candidates_by_pair = _discover_trajectory_candidates(dataset, selection)
    pair_count = len(resolved_pairs)
    pair_slot_counts = {pair: 0 for pair in resolved_pairs}
    for env_id in range(num_envs):
        pair = resolved_pairs[env_id % pair_count]
        pair_slot_counts[pair] += 1
    decoded_by_pair = _decode_valid_candidates(
        dataset,
        version,
        selection=selection,
        resolved_pairs=resolved_pairs,
        candidates_by_pair=candidates_by_pair,
        limits={
            pair: min(pair_slot_counts[pair], len(candidates_by_pair[pair]))
            for pair in resolved_pairs
        },
    )
    pair_slots = {pair: 0 for pair in resolved_pairs}
    assignments: list[ReferenceTrajectory] = []
    for env_id in range(num_envs):
        pair = resolved_pairs[env_id % pair_count]
        decoded = decoded_by_pair[pair]
        pair_slot = pair_slots[pair]
        assignments.append(decoded[pair_slot % len(decoded)])
        pair_slots[pair] += 1
    return TrajectoryBatch(
        tuple(assignments),
        resolved_pairs=resolved_pairs,
        selection_mode=selection.mode,
    )
