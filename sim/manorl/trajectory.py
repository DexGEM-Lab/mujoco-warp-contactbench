"""Versioned Lance readers and deterministic ManoRL training selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from sim.manorl.contracts import (
    DEFAULT_HAND_DATASET_PATH,
    DATASET_ROW_INDEX,
    EXPECTED_DATASET_VERSION,
    JOINT_DOF,
    LEGACY_JOINT_NAMES,
    canonical_hand_sides,
    normalize_hand_side,
    OBJECT_INDEX,
    OBJECT_TYPE,
    SOURCE_DATA_FPS,
    SOURCE_FRAME_COUNT,
    SOURCE_SLICE,
    TRAJECTORY_IDENTITY,
    TrajectoryIdentity,
)
from sim.manorl.hand_layout import HandActionLayout

LANCE_COLUMNS = ("index", "trajectory_metadata", "timestamp", "hands", "objects")
LANCE_DISCOVERY_COLUMNS = ("index", "trajectory_metadata")
LANCE_DECODE_CHUNK_SIZE = 128
HAND_DATASET_PATH = Path(DEFAULT_HAND_DATASET_PATH)
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
DEFAULT_PRE_PADDING = 100
DEFAULT_POST_PADDING = 250
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
    # ``q_ref`` is the primary/reference hand retained for the historical
    # single-hand API.  New rows keep every detected side here so action
    # routing can control one or both without relying on Lance list order.
    hand_sides: tuple[str, ...] = ("right",)
    q_ref_by_side: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    selected_hand_sides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        expected = int(self.source_indices.shape[0])
        shapes = {
            "source_indices": expected >= 2,
            "timestamps": self.timestamps.shape == (expected,),
            "q_ref": self.q_ref.ndim == 2 and self.q_ref.shape[0] == expected
            and self.q_ref.shape[1] in (len(LEGACY_JOINT_NAMES), JOINT_DOF),
            "object_pos_raw": self.object_pos_raw.shape == (expected, 3),
            "object_pos": self.object_pos.shape == (expected, 3),
            "object_quat_xyzw": self.object_quat_xyzw.shape == (expected, 4),
        }
        invalid = [name for name, valid in shapes.items() if not valid]
        if invalid:
            raise ValueError(f"invalid reference trajectory shapes: {', '.join(invalid)}")
        sides = canonical_hand_sides(self.hand_sides)
        if self.q_ref_by_side:
            raw_map = dict(self.q_ref_by_side)
        else:
            raw_map = {"right": self.q_ref}
        normalized_map: dict[str, NDArray[np.float64]] = {}
        for side, values in raw_map.items():
            normalized = normalize_hand_side(str(side), allow_auto=False, allow_both=False)
            array = np.asarray(values, dtype=np.float64)
            if array.shape != self.q_ref.shape or not np.all(np.isfinite(array)):
                raise ValueError(f"q_ref_by_side[{normalized!r}] must match q_ref and be finite")
            normalized_map[normalized] = _immutable(array)
        if set(normalized_map) != set(sides):
            raise ValueError("hand_sides and q_ref_by_side keys do not match")
        selected = self.selected_hand_sides or sides
        selected = canonical_hand_sides(selected)
        if not set(selected).issubset(set(sides)):
            raise ValueError("selected hand side is absent from the trajectory")
        object.__setattr__(self, "hand_sides", sides)
        object.__setattr__(self, "q_ref_by_side", normalized_map)
        object.__setattr__(self, "selected_hand_sides", selected)
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

    @property
    def dof_dim(self) -> int:
        return int(self.q_ref.shape[1])

    @property
    def action_layout(self) -> HandActionLayout:
        return HandActionLayout(
            self.hand_sides,
            self.selected_hand_sides,
            dof_per_hand=self.dof_dim,
        )

    def q_ref_for(self, hand_side: str) -> NDArray[np.float64]:
        side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
        try:
            return self.q_ref_by_side[side]
        except KeyError as exc:
            raise KeyError(f"trajectory has no {side} hand reference") from exc


@dataclass(frozen=True)
class TrajectoryBatch:
    """Immutable per-world reference assignments, mirroring Isaac batch loading."""

    trajectories: tuple[ReferenceTrajectory, ...]
    resolved_pairs: tuple[ObjectActionPair, ...] = ()
    selection_mode: str | None = None
    pair_assignment_cycle: int = 0

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError("trajectory batch must be non-empty")
        if any(not isinstance(trajectory, ReferenceTrajectory) for trajectory in self.trajectories):
            raise TypeError("trajectory batch must contain ReferenceTrajectory values")
        if any(not isinstance(pair, ObjectActionPair) for pair in self.resolved_pairs):
            raise TypeError("resolved_pairs must contain ObjectActionPair values")
        if self.selection_mode not in (None, "pairs", "all"):
            raise ValueError("selection_mode must be 'pairs', 'all', or None")
        if (
            not isinstance(self.pair_assignment_cycle, int)
            or isinstance(self.pair_assignment_cycle, bool)
            or self.pair_assignment_cycle < 0
        ):
            raise ValueError("pair_assignment_cycle must be a non-negative integer")

    @property
    def num_envs(self) -> int:
        return len(self.trajectories)

    @property
    def lengths(self) -> NDArray[np.int64]:
        return _immutable(np.asarray([len(trajectory.q_ref) for trajectory in self.trajectories]), dtype=np.int64)

    @property
    def hand_sides(self) -> tuple[str, ...]:
        """Union of sides present in the batch, in canonical order."""

        return canonical_hand_sides(
            tuple(side for trajectory in self.trajectories for side in trajectory.hand_sides)
        )

    @property
    def action_dim(self) -> int:
        dims = {trajectory.action_layout.action_dim for trajectory in self.trajectories}
        if len(dims) != 1:
            raise ValueError("trajectory batch contains incompatible hand action layouts")
        return next(iter(dims))


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
    expected_dataset_version: int | None = None
    pre_padding: int = DEFAULT_PRE_PADDING
    post_padding: int = DEFAULT_POST_PADDING
    hand_side: str = "auto"
    pair_assignment_cycle: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        # Modern captures use descriptive labels such as
        # ``001-Palmar-Pinch``.  The leading numeric token is the stable
        # source action id; retain the human-readable ``gesture`` field for
        # callers, but normalize validation/lookup through ``action_id``.
        action_id = _gesture_action_id(self.gesture)
        if action_id is None:
            raise ValueError(
                f"gesture must start with a source action id in [1, 50], got {self.gesture!r}"
            )
        ObjectActionPair(self.object_type, action_id)
        if self.selector is not None:
            parse_trajectory_selector(self.selector)
        if self.expected_dataset_version is not None and (
            not isinstance(self.expected_dataset_version, int)
            or isinstance(self.expected_dataset_version, bool)
            or self.expected_dataset_version < 1
        ):
            raise ValueError("expected_dataset_version must be a positive integer")
        if self.pre_padding < 0 or self.post_padding < 0:
            raise ValueError("trajectory padding must be non-negative")
        if (
            not isinstance(self.pair_assignment_cycle, int)
            or isinstance(self.pair_assignment_cycle, bool)
            or self.pair_assignment_cycle < 0
        ):
            raise ValueError("pair_assignment_cycle must be a non-negative integer")
        normalize_hand_side(self.hand_side)

    @property
    def action_id(self) -> str:
        action_id = _gesture_action_id(self.gesture)
        if action_id is None:  # pragma: no cover - guarded by __post_init__
            raise ValueError(f"gesture does not contain a valid source action id: {self.gesture!r}")
        return action_id

    @property
    def requested_pairs(self) -> tuple[ObjectActionPair, ...] | None:
        if self.selector is None:
            return (ObjectActionPair(self.object_type, self.action_id),)
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


def detect_hand_sides(row_or_metadata: Mapping[str, Any] | dict[str, Any]) -> tuple[str, ...]:
    """Detect available hand sides from Lance metadata, independent of list order."""

    if not isinstance(row_or_metadata, Mapping):
        raise TypeError("row_or_metadata must be a mapping")
    metadata = row_or_metadata.get("trajectory_metadata", row_or_metadata)
    if not isinstance(metadata, Mapping):
        raise ValueError("trajectory metadata is missing")
    names = metadata.get("hand_names")
    if names is None:
        # Legacy generated rows omitted side metadata and were right-hand
        # captures by contract.
        return ("right",)
    if isinstance(names, (str, bytes)) or names is None:
        raise ValueError("trajectory_metadata.hand_names must be a non-empty sequence")
    try:
        names = list(names)
    except TypeError as exc:
        raise ValueError("trajectory_metadata.hand_names must be a non-empty sequence") from exc
    if not names:
        raise ValueError("trajectory_metadata.hand_names must be a non-empty list")
    try:
        sides = canonical_hand_sides(tuple(str(value) for value in names))
    except (TypeError, ValueError) as exc:
        # Preserve the normalization reason (especially an unsupported label)
        # in the public decoder error while retaining row context.
        raise ValueError(f"invalid trajectory hand_names: {names!r}: {exc}") from exc
    if len(sides) != len(names):
        raise ValueError("trajectory_metadata.hand_names contains duplicate sides")
    return sides


def resolve_hand_selection(
    available_sides: object,
    hand_side: str = "auto",
) -> tuple[str, ...]:
    """Resolve ``auto``/``both``/explicit control selection for a dataset."""

    available = canonical_hand_sides(available_sides)
    selection = normalize_hand_side(hand_side)
    if selection == "auto":
        return available
    if selection == "both":
        if set(available) != {"left", "right"}:
            raise ValueError(f"requested both hands; dataset has {available}")
        return available
    if selection not in available:
        raise ValueError(f"requested {selection} hand is absent; dataset has {available}")
    return (selection,)


def _hand_rows_by_side(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sides = detect_hand_sides(row)
    hands = row.get("hands")
    if not isinstance(hands, list) or len(hands) != len(sides):
        raise ValueError(f"dataset contains {len(hands) if isinstance(hands, list) else 'invalid'} hands for {sides}")
    # Metadata names are the only authoritative association.  We construct a
    # map once, then use canonical side names everywhere else.
    raw_names = row["trajectory_metadata"].get("hand_names")
    if raw_names is None:
        ordered_sides = ["right"]
    else:
        ordered_sides = [
            normalize_hand_side(str(value), allow_auto=False, allow_both=False)
            for value in raw_names
        ]
    mapping = {side: hand for side, hand in zip(ordered_sides, hands, strict=True)}
    if set(mapping) != set(sides):
        raise ValueError("hand_names contains duplicate or unsupported side labels")
    return mapping


def _safe_row_identity(row: dict[str, Any], row_index: int = 0) -> str:
    """Derive a stable identity for both old source and new capture rows."""

    index = row.get("index", {})
    source_path = index.get("source_path") if isinstance(index, dict) else None
    if isinstance(source_path, str) and source_path:
        candidate = Path(source_path).parent.name
        if candidate.count("_") == 2:
            return candidate
    scene = str(index.get("scene", "object")) if isinstance(index, dict) else "object"
    gesture = str(index.get("gesture", "01")) if isinstance(index, dict) else "01"
    # Keep environment's object_action_sequence convention even when the new
    # capture uses a descriptive gesture string.
    leading_action = re.match(r"0*(\d+)", gesture)
    if leading_action is not None and 1 <= int(leading_action.group(1)) <= 50:
        gesture_slug = f"{int(leading_action.group(1)):02d}"
    else:
        gesture_slug = re.sub(r"[^A-Za-z0-9]+", "", gesture) or "01"
    # New capture rows do not carry source_path and often reuse a capture
    # session id for every row.  The Lance row index is therefore the stable
    # sequence discriminator; row zero intentionally becomes ``..._001``.
    sequence = int(row_index) + 1
    metadata = row.get("trajectory_metadata", {})
    if isinstance(metadata, dict):
        raw_id = metadata.get("raw_data_info", {}).get("id") if isinstance(metadata.get("raw_data_info"), dict) else None
        if isinstance(source_path, str) and source_path and isinstance(raw_id, (int, np.integer)):
            sequence = int(raw_id)
    return f"{scene}_{gesture_slug}_{sequence:03d}"


def _derive_identity(row: dict[str, Any]) -> str:
    source_path = row["index"].get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("row index must contain a non-empty source_path")
    identity = Path(source_path).parent.name
    if identity.count("_") != 2:
        raise ValueError(f"source_path does not encode an object_action_sequence: {source_path!r}")
    return identity


def _gesture_action_id(value: object) -> str | None:
    """Extract a canonical source action id from numeric/descriptive labels."""

    match = re.match(r"0*(\d+)", str(value).strip())
    if match is None:
        return None
    number = int(match.group(1))
    return f"{number:02d}" if 1 <= number <= 50 else None


def _modern_row_pair_identity(
    row: dict[str, Any], *, row_index: int
) -> tuple[ObjectActionPair, str] | None:
    """Resolve scene/gesture identity for rows without source_path metadata."""

    index = row.get("index")
    metadata = row.get("trajectory_metadata")
    if not isinstance(index, dict) or not isinstance(metadata, dict):
        return None
    object_type = str(index.get("scene", "")).strip()
    action_id = _gesture_action_id(index.get("gesture", ""))
    if not object_type or action_id is None:
        return None
    try:
        pair = ObjectActionPair(object_type, action_id)
    except ValueError:
        return None
    identity = _safe_row_identity(row, row_index)
    return pair, identity


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
    try:
        from sim.manorl.assets import object_collision_vertices

        rotated = Rotation.from_quat(initial_quaternion_xyzw).apply(
            object_collision_vertices(object_type).copy()
        )
    except (FileNotFoundError, ValueError):
        # Dataset discovery is useful before an object runtime is installed.
        # Keep the raw capture pose in that case and let environment creation
        # report the explicit unsupported-object error later.
        return 0.0
    return -float(np.min(rotated[:, 2] + initial_position[2]))


def trajectory_from_lance_row(
    row: dict[str, Any],
    dataset_version: int,
    *,
    row_index: int = 0,
    hand_side: str = "auto",
    pre_padding: int = 0,
    post_padding: int = 0,
) -> ReferenceTrajectory:
    """Decode a modern Lance row with one, left/right, or both hands.

    The row's ``hand_names`` metadata defines side association; the physical
    order of ``hands`` is never interpreted as right-first.  ``hand_side``
    controls which slots receive actions, while all side references remain in
    ``q_ref_by_side`` for reference following.
    """

    if not isinstance(row, dict):
        raise TypeError("row must be a decoded Lance mapping")
    metadata = row.get("trajectory_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("row lacks trajectory_metadata")
    sides = detect_hand_sides(row)
    selected = resolve_hand_selection(sides, hand_side)
    hand_rows = _hand_rows_by_side(row)
    source_count = int(metadata.get("total_frames", 0))
    timestamps = np.asarray(row.get("timestamp", ()), dtype=np.float64)
    if source_count <= 0:
        source_count = len(timestamps)
    if timestamps.shape != (source_count,) or not np.all(np.isfinite(timestamps)):
        raise ValueError(f"timestamp shape {timestamps.shape} does not match total_frames={source_count}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("source timestamps are not strictly increasing")

    object_names = metadata.get("object_names")
    if not isinstance(object_names, list):
        object_names = [str(row.get("index", {}).get("scene", OBJECT_TYPE))]
    objects = row.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("row must contain at least one object state")
    object_type = str(row.get("index", {}).get("scene", object_names[0]))
    object_index = object_names.index(object_type) if object_type in object_names else 0
    if object_index >= len(objects):
        raise ValueError("selected object state is absent")
    object_state = objects[object_index]
    object_pos_all = np.asarray(object_state.get("pos", ()), dtype=np.float64)
    object_rotvec_all = np.asarray(object_state.get("rot_aa", ()), dtype=np.float64)
    if object_pos_all.shape != (source_count, 3) or object_rotvec_all.shape != (source_count, 3):
        raise ValueError("object pose arrays do not match total_frames")
    if not np.all(np.isfinite(object_pos_all)) or not np.all(np.isfinite(object_rotvec_all)):
        raise ValueError("object pose arrays contain non-finite values")

    movement = metadata.get("trajectory_info", {}).get("object_move", [])
    movement_entry = next(
        (entry for entry in movement if entry.get("object_name") == object_type), None
    ) if isinstance(movement, list) else None
    if movement_entry is None:
        movement_start, movement_end = 0, source_count - 1
    else:
        movement_start = int(movement_entry.get("start_frame", 0))
        movement_end = int(movement_entry.get("end_frame", source_count - 1))
    if not 0 <= movement_start <= movement_end < source_count:
        raise ValueError(
            "object movement range must be an inclusive interval within total_frames"
        )
    start = max(0, movement_start - int(pre_padding))
    # Lance movement metadata names an inclusive ``end_frame``.  Convert it
    # once at the Python slice boundary so zero-padding selections retain the
    # final movement frame.
    stop = min(source_count, movement_end + 1 + int(post_padding))
    if stop - start < 2:
        raise ValueError("selected row window must contain at least two frames")

    q_by_side: dict[str, NDArray[np.float64]] = {}
    dof_dim: int | None = None
    for side in sides:
        q_all = np.asarray(hand_rows[side].get("urdf_dof", ()), dtype=np.float64)
        if q_all.ndim != 2 or q_all.shape[0] != source_count or q_all.shape[1] not in (26, JOINT_DOF):
            raise ValueError(f"{side} urdf_dof must have shape ({source_count}, 26 or {JOINT_DOF})")
        if dof_dim is None:
            dof_dim = int(q_all.shape[1])
        if q_all.shape[1] != dof_dim or not np.all(np.isfinite(q_all)):
            raise ValueError("all hand references must share one finite DOF width")
        values = q_all[start:stop].copy()
        values[:, 3:6] = np.unwrap(values[:, 3:6], axis=0, period=2.0 * np.pi)
        q_by_side[side] = _immutable(values)
    assert dof_dim is not None
    object_pos_raw = object_pos_all[start:stop].copy()
    object_quat_xyzw = rotvec_to_xyzw(object_rotvec_all[start:stop])
    z_shift = _initial_support_shift(object_pos_raw[0], object_quat_xyzw[0], object_type)
    object_pos = object_pos_raw.copy()
    object_pos[:, 2] += z_shift
    primary = "right" if "right" in selected else selected[0]
    index = row.get("index", {})
    identity = _safe_row_identity(row, row_index)
    trajectory_identity = TrajectoryIdentity(
        dataset_path=str(row.get("dataset_path", "")),
        dataset_version=int(dataset_version),
        row_index=int(row_index),
        object_index=int(object_index),
        uuid=str(index.get("uuid", "")),
        file_uuid=str(index.get("file_uuid", "")),
        identity=identity,
        source_start=int(start),
        source_stop=int(stop),
        movement_start_raw=int(movement_start),
        movement_end_raw=int(movement_end),
    )
    return ReferenceTrajectory(
        identity=trajectory_identity,
        dataset_version=int(dataset_version),
        source_indices=_immutable(np.arange(start, stop), dtype=np.int64),
        timestamps=_immutable(timestamps[start:stop]),
        q_ref=q_by_side[primary],
        object_pos_raw=_immutable(object_pos_raw),
        object_pos=_immutable(object_pos),
        object_quat_xyzw=_immutable(object_quat_xyzw),
        object_z_shift=z_shift,
        hand_sides=sides,
        q_ref_by_side=q_by_side,
        selected_hand_sides=selected,
    )


def trajectory_from_row(
    row: dict[str, Any],
    dataset_version: int,
    *,
    hand_side: str = "auto",
) -> ReferenceTrajectory:
    """Validate one decoded Lance row and construct the immutable accepted slice."""

    # Modern capture rows carry descriptive indices, 28-wide DOFs, or more
    # than one side.  Route them through the metadata-driven decoder before
    # applying the historical source identity contract.
    metadata = row.get("trajectory_metadata", {}) if isinstance(row, dict) else {}
    index = row.get("index", {}) if isinstance(row, dict) else {}
    hands = row.get("hands", []) if isinstance(row, dict) else []
    modern = (
        not isinstance(index, dict)
        or not isinstance(index.get("source_path"), str)
        or not index.get("source_path")
        or (
            isinstance(metadata, dict)
            and metadata.get("hand_names") not in (None, ["right"])
        )
    )
    if not modern and isinstance(hands, list) and hands:
        first_q = np.asarray(hands[0].get("urdf_dof", ())) if isinstance(hands[0], dict) else np.asarray(())
        modern = first_q.ndim == 2 and first_q.shape[-1] == JOINT_DOF
    if modern:
        return trajectory_from_lance_row(row, dataset_version, hand_side=hand_side)

    _validate_row(row)
    source_count = int(row["trajectory_metadata"]["total_frames"])
    timestamps_all = np.asarray(row["timestamp"], dtype=np.float64)
    q_all = np.asarray(row["hands"][0]["urdf_dof"], dtype=np.float64)
    object_pos_all = np.asarray(row["objects"][OBJECT_INDEX]["pos"], dtype=np.float64)
    object_rotvec_all = np.asarray(row["objects"][OBJECT_INDEX]["rot_aa"], dtype=np.float64)
    expected_shapes = {
        "timestamp": (source_count,),
        "urdf_dof": (source_count, len(LEGACY_JOINT_NAMES)),
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
        hand_sides=("right",),
        q_ref_by_side={"right": _immutable(q_ref)},
        selected_hand_sides=("right",),
        object_pos_raw=_immutable(object_pos_raw),
        object_pos=_immutable(object_pos),
        object_quat_xyzw=_immutable(object_quat_xyzw),
        object_z_shift=z_shift,
    )


def load_reference_trajectory(
    dataset_path: str | Path = TRAJECTORY_IDENTITY.dataset_path,
    *,
    expected_dataset_version: int | None = EXPECTED_DATASET_VERSION,
    hand_side: str = "auto",
    row_index: int = DATASET_ROW_INDEX,
) -> ReferenceTrajectory:
    """Load one reference row, auto-detecting modern hand-side metadata."""

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
    if expected_dataset_version is not None and dataset_version != expected_dataset_version:
        raise ValueError(
            f"dataset version {dataset_version} != accepted version {expected_dataset_version}"
        )
    table = dataset.take([int(row_index)], columns=list(LANCE_COLUMNS))
    rows = table.to_pylist()
    if len(rows) != 1:
        raise ValueError(f"dataset.take returned {len(rows)} rows, expected exactly one")
    row = dict(rows[0])
    row["dataset_path"] = str(path)
    if path.resolve() == HAND_DATASET_PATH.resolve() or detect_hand_sides(row) != ("right",):
        return trajectory_from_lance_row(
            row,
            dataset_version,
            row_index=int(row_index),
            hand_side=hand_side,
        )
    return trajectory_from_row(row, dataset_version, hand_side=hand_side)


def load_hand_reference_trajectory(
    dataset_path: str | Path = HAND_DATASET_PATH,
    *,
    row_index: int = 0,
    hand_side: str = "auto",
    expected_dataset_version: int | None = None,
    pre_padding: int = 0,
    post_padding: int = 0,
) -> ReferenceTrajectory:
    """Load a modern left/right/bimanual Lance row with data-driven sides."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Lance dataset is absent: {path}")
    try:
        import lance
    except ImportError as exc:
        raise RuntimeError("pylance is required to read the hand trajectory") from exc
    dataset = lance.dataset(str(path))
    version_value = getattr(dataset, "version", None)
    if version_value is None:
        raise ValueError("Lance dataset did not expose a version")
    version = int(version_value)
    if expected_dataset_version is not None and version != int(expected_dataset_version):
        raise ValueError(f"dataset version {version} != requested {expected_dataset_version}")
    rows = dataset.take([int(row_index)], columns=list(LANCE_COLUMNS)).to_pylist()
    if len(rows) != 1:
        raise ValueError(f"dataset.take returned {len(rows)} rows, expected exactly one")
    row = dict(rows[0])
    row["dataset_path"] = str(path)
    return trajectory_from_lance_row(
        row,
        version,
        row_index=int(row_index),
        hand_side=hand_side,
        pre_padding=pre_padding,
        post_padding=post_padding,
    )


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
        "urdf_dof": (source_count, len(LEGACY_JOINT_NAMES)),
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
        "urdf_dof": (source_count, len(LEGACY_JOINT_NAMES)),
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
    source_path = index.get("source_path")
    modern_identity = _modern_row_pair_identity(row, row_index=row_index)
    if isinstance(source_path, str) and source_path:
        try:
            identity = _derive_identity(row)
        except (KeyError, ValueError):
            return None
    elif modern_identity is not None:
        _, identity = modern_identity
    else:
        return None
    fields = identity.split("_")
    if len(fields) != 3 or not fields[2].isdigit():
        return None
    object_type, action_raw, sequence_raw = fields
    try:
        pair = ObjectActionPair(object_type, action_raw)
    except ValueError:
        return None
    index_action = _gesture_action_id(index.get("gesture", ""))
    if str(index.get("scene", "")) != object_type or index_action != pair.action_id:
        return None
    if isinstance(source_path, str) and source_path and Path(source_path).parent.name != identity:
        return None
    object_names = metadata.get("object_names")
    if isinstance(object_names, list) and object_type not in object_names:
        return None
    try:
        sides = detect_hand_sides(row)
        if not sides:
            return None
    except (TypeError, ValueError):
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
    if requested_start < 0:
        return None
    requested_stop = end_raw + selection.post_padding
    if not source_path:
        # Modern capture metadata uses an inclusive movement end.  Historical
        # source-path datasets retain their established exclusive-stop ABI.
        requested_stop += 1
    # Modern captures are allowed to clip at dataset boundaries even when a
    # historical selection requests 250-frame source padding.
    require_full_padding = selection.require_full_padding and bool(source_path)
    if require_full_padding and (
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
    # Modern capture rows identify themselves through ``scene``/``gesture``
    # and hand metadata, without the legacy source_path/object_names contract.
    # Reuse the authoritative side-aware decoder so metadata order never maps
    # a left hand into the right slot.
    modern = (
        not isinstance(index.get("source_path"), str)
        or metadata.get("hand_names") not in (None, ["right"])
    )
    if not modern and isinstance(row.get("hands"), list) and row["hands"]:
        first_q = np.asarray(row["hands"][0].get("urdf_dof", ()))
        modern = first_q.ndim == 2 and first_q.shape[-1] == JOINT_DOF
    if modern:
        resolved = _modern_row_pair_identity(row, row_index=row_index)
        if resolved is None or resolved[0] != expected_pair:
            raise ValueError(f"row {row_index} is not {expected_pair.canonical}")
        modern_row = dict(row)
        modern_row["dataset_path"] = str(selection.dataset_path)
        return trajectory_from_lance_row(
            modern_row,
            dataset_version,
            row_index=row_index,
            hand_side=selection.hand_side,
            pre_padding=selection.pre_padding,
            post_padding=selection.post_padding,
        )
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
    if requested_start < 0:
        raise ValueError(
            f"row {row_index} movement start {start_raw} cannot provide {selection.pre_padding}-frame pre-padding"
        )
    if selection.require_full_padding and requested_stop > source_count:
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
        "urdf_dof": (source_count, len(LEGACY_JOINT_NAMES)),
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

    if any(limits.get(pair, 0) < 0 for pair in resolved_pairs):
        raise ValueError("candidate decode limits must be non-negative")
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
        if limits[pair] == 0 or decoded[pair]:
            continue
        detail = "; ".join(rejected[pair][:4]) or "no full rows were decoded"
        raise LookupError(f"no valid Lance trajectories for {pair.canonical}; {detail}")
    return {pair: tuple(decoded[pair]) for pair in resolved_pairs}


def load_assigned_trajectory_batch(selection: TrajectorySelection, *, num_envs: int) -> TrajectoryBatch:
    """Discover eligible pairs and deterministically assign them to vector worlds.

    Pair slots round-robin over sorted resolved pairs; each pair independently
    round-robins its sorted exact-identity trajectories. ``pair_assignment_cycle``
    advances every pair by one local slot-window, so successive fixed-size runs
    cover long pairs without increasing simultaneous world residency. Full rows
    are decoded in bounded chunks and malformed candidates are skipped.
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
    dataset = lance.dataset(
        str(selection.dataset_path),
        **(
            {"version": selection.expected_dataset_version}
            if selection.expected_dataset_version is not None
            else {}
        ),
    )
    version = int(getattr(dataset, "version", -1))
    if selection.expected_dataset_version is not None and version != selection.expected_dataset_version:
        raise ValueError(f"dataset version {version} != requested {selection.expected_dataset_version}")
    resolved_pairs, candidates_by_pair = _discover_trajectory_candidates(dataset, selection)
    pair_count = len(resolved_pairs)
    pair_slot_counts = {pair: 0 for pair in resolved_pairs}
    for env_id in range(num_envs):
        pair = resolved_pairs[env_id % pair_count]
        pair_slot_counts[pair] += 1
    rotated_candidates_by_pair: dict[ObjectActionPair, tuple[_TrajectoryCandidate, ...]] = {}
    for pair in resolved_pairs:
        candidates = candidates_by_pair[pair]
        slots = pair_slot_counts[pair]
        offset = (
            selection.pair_assignment_cycle * slots
        ) % len(candidates)
        rotated_candidates_by_pair[pair] = candidates[offset:] + candidates[:offset]
    decoded_by_pair = _decode_valid_candidates(
        dataset,
        version,
        selection=selection,
        resolved_pairs=resolved_pairs,
        candidates_by_pair=rotated_candidates_by_pair,
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
        pair_assignment_cycle=selection.pair_assignment_cycle,
    )
