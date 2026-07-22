from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.manorl.assets import object_collision_vertices
from sim.manorl.contracts import DATASET_PATH, EXPECTED_DATASET_VERSION
from sim.manorl.trajectory import (
    CUBE1_ACTION_01_BATCH_ROWS,
    LANCE_COLUMNS,
    ObjectActionPair,
    TrajectorySelection,
    load_assigned_trajectory_batch,
    load_cube1_action_01_batch10,
    load_reference_trajectory,
    parse_trajectory_selector,
    trajectory_from_row,
)


def _dataset_available() -> tuple[bool, str]:
    if not Path(DATASET_PATH).exists():
        return False, f"accepted Lance dataset absent: {DATASET_PATH}"
    try:
        import lance  # noqa: F401
    except ImportError:
        return False, "pylance distribution (imported as lance) is not installed"
    return True, ""


@pytest.mark.integration
def test_exact_row_one_identity_slice_and_support_shift() -> None:
    available, reason = _dataset_available()
    if not available:
        pytest.skip(reason)
    trajectory = load_reference_trajectory()
    assert trajectory.dataset_version == EXPECTED_DATASET_VERSION
    assert trajectory.source_indices[0] == 440
    assert trajectory.source_indices[-1] == 1231
    assert trajectory.q_ref.shape == (792, 26)
    assert trajectory.object_pos_raw.shape == (792, 3)
    assert trajectory.object_pos.shape == (792, 3)
    assert trajectory.object_quat_xyzw.shape == (792, 4)
    assert not trajectory.q_ref.flags.writeable
    assert trajectory.identity.identity == "cube1_01_009"
    rotated_vertices = Rotation.from_quat(trajectory.object_quat_xyzw[0]).apply(
        object_collision_vertices().copy()
    )
    support_bottom = np.min(rotated_vertices[:, 2] + trajectory.object_pos[0, 2])
    assert support_bottom == pytest.approx(0.0, abs=1e-12)
    assert trajectory.object_z_shift == pytest.approx(0.006806849331337565, abs=1e-12)
    np.testing.assert_allclose(
        trajectory.object_pos[:, 2] - trajectory.object_pos_raw[:, 2],
        trajectory.object_z_shift,
        atol=1e-15,
    )


@pytest.mark.integration
def test_cube1_action_01_batch_has_explicit_distinct_padded_assignments() -> None:
    available, reason = _dataset_available()
    if not available:
        pytest.skip(reason)
    batch = load_cube1_action_01_batch10()
    assert batch.num_envs == 10
    assert [trajectory.identity.row_index for trajectory in batch.trajectories] == [
        row[0] for row in CUBE1_ACTION_01_BATCH_ROWS
    ]
    assert [trajectory.identity.identity for trajectory in batch.trajectories] == [
        row[2] for row in CUBE1_ACTION_01_BATCH_ROWS
    ]
    assert batch.lengths.tolist() == [790, 792, 774, 815, 797, 771, 780, 739, 726, 743]
    assert len({tuple(trajectory.object_pos[0]) for trajectory in batch.trajectories}) == 10


@pytest.mark.integration
def test_selector_assigns_all_eligible_rows_before_repeating() -> None:
    available, reason = _dataset_available()
    if not available:
        pytest.skip(reason)
    batch = load_assigned_trajectory_batch(TrajectorySelection("cube1", "01"), num_envs=12)
    assert [trajectory.identity.row_index for trajectory in batch.trajectories] == [
        0, 1, 2, 3, 4, 5, 7, 14, 15, 16, 0, 1
    ]
    assert {trajectory.identity.identity.split("_")[1] for trajectory in batch.trajectories} == {"01"}


@pytest.mark.integration
def test_selector_accepts_non_default_cube1_gesture() -> None:
    available, reason = _dataset_available()
    if not available:
        pytest.skip(reason)
    batch = load_assigned_trajectory_batch(TrajectorySelection("cube1", "03"), num_envs=3)
    assert [trajectory.identity.row_index for trajectory in batch.trajectories] == [71, 72, 74]
    assert [trajectory.identity.identity for trajectory in batch.trajectories] == [
        "cube1_03_004", "cube1_03_005", "cube1_03_007"
    ]


def test_pair_selector_normalizes_and_preserves_exact_pairs() -> None:
    assert parse_trajectory_selector("cube2:1,cube1:02") == (
        ObjectActionPair("cube1", "02"),
        ObjectActionPair("cube2", "01"),
    )
    assert TrajectorySelection(selector="all").canonical_selector == "all"
    assert TrajectorySelection("cube1", "1").canonical_selector == "cube1:01"


def test_trajectory_selection_defaults_to_100_pre_and_250_post_padding() -> None:
    selection = TrajectorySelection()
    assert (selection.pre_padding, selection.post_padding) == (100, 250)


@pytest.mark.parametrize(
    "selector, message",
    [
        ("", "non-empty"),
        ("cube1:01,", "empty"),
        ("cube1", "expected object:action"),
        ("cube1:01,cube1:1", "duplicate"),
        ("ALL", "spelled exactly"),
    ],
)
def test_pair_selector_rejects_malformed_values(selector: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_trajectory_selector(selector)


@pytest.mark.integration
def test_all_pair_selector_discovers_s02_pairs_without_suffix_rows() -> None:
    available, reason = _dataset_available()
    if not available:
        pytest.skip(reason)
    batch = load_assigned_trajectory_batch(TrajectorySelection(selector="all"), num_envs=77)
    assert len(batch.resolved_pairs) == 77
    assert batch.resolved_pairs == tuple(sorted(batch.resolved_pairs))
    assert batch.resolved_pairs[0].canonical == "cube1:01"
    assert batch.resolved_pairs[-1].canonical == "sphere3:02"
    assert all(identity.count("_") == 2 for identity in (t.identity.identity for t in batch.trajectories))


@pytest.mark.integration
def test_all_s02_pairs_have_valid_runtime_grasp_mappings() -> None:
    available, reason = _dataset_available()
    if not available:
        pytest.skip(reason)
    from sim.manorl.environment import _expected_keypoint_ids

    batch = load_assigned_trajectory_batch(TrajectorySelection(selector="all"), num_envs=77)
    for pair in batch.resolved_pairs:
        keypoint_ids = _expected_keypoint_ids(pair.object_type, pair.action_id)
        assert keypoint_ids.ndim == 1
        assert len(keypoint_ids) >= 2


@pytest.mark.integration
def test_explicit_pair_selector_does_not_form_cartesian_product() -> None:
    available, reason = _dataset_available()
    if not available:
        pytest.skip(reason)
    selection = TrajectorySelection(selector="cube1:01,cube2:01")
    batch = load_assigned_trajectory_batch(selection, num_envs=2)
    assert batch.resolved_pairs == (
        ObjectActionPair("cube1", "01"),
        ObjectActionPair("cube2", "01"),
    )
    assert [t.identity.identity.split("_")[:2] for t in batch.trajectories] == [
        ["cube1", "01"],
        ["cube2", "01"],
    ]


def _accepted_fake_row(timestamps: np.ndarray) -> dict[str, object]:
    return {
        "index": {
            "uuid": "d5bc2bc6-9458-52d0-bccc-66c9ec21bae3",
            "file_uuid": "e6fe4732-72cd-5ab7-93e6-2e62dc0263a5",
            "gesture": "01",
            "source_path": "cube1/cube1_01_009/cube1_01_009_mano.npy",
        },
        "trajectory_metadata": {
            "object_names": ["cube1"],
            "hand_names": ["right"],
            "raw_data_info": {"id": 9},
            "total_frames": 1373,
            "data_fps": 111,
            "trajectory_info": {
                "object_move": [
                    {"object_name": "cube1", "start_frame": 690, "end_frame": 982}
                ]
            },
        },
        "timestamp": timestamps,
        "hands": [{"urdf_dof": np.zeros((1373, 26))}],
        "objects": [{"pos": np.repeat([[0.0, 0.0, 1.0]], 1373, axis=0), "rot_aa": np.zeros((1373, 3))}],
    }


def test_timestamps_require_strict_monotonicity_only() -> None:
    deltas = np.resize(np.array([0.001, 0.065], dtype=np.float64), 1372)
    timestamps = np.concatenate([[0.0], np.cumsum(deltas)])
    trajectory = trajectory_from_row(_accepted_fake_row(timestamps), EXPECTED_DATASET_VERSION)
    assert trajectory.timestamps.shape == (792,)

    non_monotonic = timestamps.copy()
    non_monotonic[301] = non_monotonic[300]
    with pytest.raises(ValueError, match="strictly increasing"):
        trajectory_from_row(_accepted_fake_row(non_monotonic), EXPECTED_DATASET_VERSION)


def test_assigned_loader_skips_invalid_full_candidate_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "source.lance"
    path.mkdir()
    timestamps = np.arange(1373, dtype=np.float64) / 111.0
    invalid = _accepted_fake_row(timestamps.copy())
    invalid["timestamp"][100] = invalid["timestamp"][99]
    valid = deepcopy(_accepted_fake_row(timestamps))
    rows = [invalid, valid]
    for sequence, row in enumerate(rows, start=1):
        identity = f"cube1_01_{sequence:03d}"
        row["index"].update(
            scene="cube1",
            source_path=f"cube1/{identity}/{identity}_mano.npy",
            uuid=f"row-{sequence}",
        )

    class FakeTable:
        def __init__(self, values: list[dict[str, object]]) -> None:
            self.values = values

        def to_pylist(self) -> list[dict[str, object]]:
            return self.values

    class FakeDataset:
        version = EXPECTED_DATASET_VERSION

        def to_table(self, *, columns: list[str]) -> FakeTable:
            assert columns == ["index", "trajectory_metadata"]
            return FakeTable(rows)

        def take(self, indices: list[int], *, columns: list[str]) -> FakeTable:
            assert columns == list(LANCE_COLUMNS)
            return FakeTable([rows[index] for index in indices])

    import lance

    monkeypatch.setattr(lance, "dataset", lambda _: FakeDataset())
    batch = load_assigned_trajectory_batch(
        TrajectorySelection("cube1", "01", dataset_path=path), num_envs=1
    )
    assert batch.trajectories[0].identity.row_index == 1
    assert batch.trajectories[0].identity.identity == "cube1_01_002"


def test_assigned_loader_does_not_decode_pairs_without_environment_slots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "source.lance"
    path.mkdir()
    timestamps = np.arange(1373, dtype=np.float64) / 111.0
    rows = [_accepted_fake_row(timestamps), deepcopy(_accepted_fake_row(timestamps))]
    for row in rows:
        row["index"]["scene"] = "cube1"
    rows[1]["index"].update(
        gesture="02",
        source_path="cube1/cube1_02_001/cube1_02_001_mano.npy",
        uuid="row-2",
    )
    taken_indices: list[int] = []

    class FakeTable:
        def __init__(self, values: list[dict[str, object]]) -> None:
            self.values = values

        def to_pylist(self) -> list[dict[str, object]]:
            return self.values

    class FakeDataset:
        version = EXPECTED_DATASET_VERSION

        def to_table(self, *, columns: list[str]) -> FakeTable:
            assert columns == ["index", "trajectory_metadata"]
            return FakeTable(rows)

        def take(self, indices: list[int], *, columns: list[str]) -> FakeTable:
            assert columns == list(LANCE_COLUMNS)
            taken_indices.extend(indices)
            return FakeTable([rows[index] for index in indices])

    import lance

    monkeypatch.setattr(lance, "dataset", lambda _: FakeDataset())
    monkeypatch.setattr("sim.manorl.trajectory._initial_support_shift", lambda *_: 0.0)
    batch = load_assigned_trajectory_batch(
        TrajectorySelection(selector="all", dataset_path=path), num_envs=1
    )
    assert [pair.canonical for pair in batch.resolved_pairs] == ["cube1:01", "cube1:02"]
    assert [trajectory.identity.row_index for trajectory in batch.trajectories] == [0]
    assert taken_indices == [0]


def test_loader_uses_exact_take_and_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "source.lance"
    path.mkdir()
    calls: list[tuple[list[int], list[str]]] = []

    class FakeTable:
        def to_pylist(self) -> list[dict[str, object]]:
            raise MarkerError

    class FakeDataset:
        version = EXPECTED_DATASET_VERSION

        def take(self, indices: list[int], *, columns: list[str]) -> FakeTable:
            calls.append((indices, columns))
            return FakeTable()

    class MarkerError(Exception):
        pass

    import lance

    monkeypatch.setattr(lance, "dataset", lambda _: FakeDataset())
    with pytest.raises(MarkerError):
        load_reference_trajectory(path)
    assert calls == [([1], list(LANCE_COLUMNS))]


def test_loader_rejects_missing_dataset_version_before_take(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "source.lance"
    path.mkdir()

    class FakeDataset:
        def take(self, *_: object, **__: object) -> object:
            raise AssertionError("take must not run without a fixed dataset version")

    import lance

    monkeypatch.setattr(lance, "dataset", lambda _: FakeDataset())
    with pytest.raises(ValueError, match="did not expose a version"):
        load_reference_trajectory(path)


def test_loader_rejects_dataset_version_before_take(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "source.lance"
    path.mkdir()

    class FakeDataset:
        version = EXPECTED_DATASET_VERSION + 1

        def take(self, *_: object, **__: object) -> object:
            raise AssertionError("take must not run for the wrong version")

    import lance

    monkeypatch.setattr(lance, "dataset", lambda _: FakeDataset())
    with pytest.raises(ValueError, match="dataset version"):
        load_reference_trajectory(path)
