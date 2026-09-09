from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.manorl.assets import object_collision_vertices
from sim.manorl.contracts import (
    CONTROL_TIMESTEP,
    DATASET_PATH,
    EXPECTED_DATASET_VERSION,
    TrajectoryIdentity,
)
from sim.manorl.trajectory import (
    CUBE1_ACTION_01_BATCH_ROWS,
    LANCE_COLUMNS,
    ObjectActionPair,
    ReferenceTrajectory,
    TrajectorySelection,
    _candidate_from_metadata_row,
    _selected_trajectory_from_row,
    load_assigned_trajectory_batch,
    load_cube1_action_01_batch10,
    load_reference_trajectory,
    parse_trajectory_selector,
    resample_reference_trajectory,
    trajectory_from_lance_row,
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


def test_trajectory_selection_defaults_to_180_pre_and_250_post_padding() -> None:
    selection = TrajectorySelection()
    assert (selection.pre_padding, selection.post_padding) == (180, 250)
    assert selection.reference_fps is None
    assert TrajectorySelection(reference_fps=100).reference_fps == 100
    assert TrajectorySelection(reference_fps=120).reference_fps == 120
    assert selection.pair_assignment_cycle == 0
    with pytest.raises(ValueError, match="reference_fps"):
        TrajectorySelection(reference_fps=200)
    with pytest.raises(ValueError, match="pair_assignment_cycle"):
        TrajectorySelection(pair_assignment_cycle=-1)


def _four_frame_reference() -> ReferenceTrajectory:
    q_ref = np.zeros((4, 28), dtype=np.float64)
    q_ref[:, 0] = np.arange(4, dtype=np.float64)
    q_ref[:, 3] = np.deg2rad([170.0, 179.0, -179.0, -170.0])
    object_pos = np.zeros((4, 3), dtype=np.float64)
    object_pos[:, 0] = np.arange(4, dtype=np.float64)
    object_quat = Rotation.from_euler(
        "z", np.asarray([0.0, 30.0, 60.0, 90.0])[:, None], degrees=True
    ).as_quat()
    return ReferenceTrajectory(
        identity=TrajectoryIdentity(
            dataset_path="fixture.lance",
            dataset_version=1,
            row_index=0,
            object_index=0,
            uuid="fixture",
            file_uuid="fixture-file",
            identity="cube1_01_001",
            source_start=10,
            source_stop=14,
            movement_start_raw=11,
            movement_end_raw=12,
        ),
        dataset_version=1,
        source_indices=np.arange(10, 14, dtype=np.int64),
        timestamps=np.asarray([0.10, 0.11, 0.12, 0.13]),
        q_ref=q_ref,
        object_pos_raw=object_pos,
        object_pos=object_pos,
        object_quat_xyzw=object_quat,
        object_z_shift=0.0,
    )


@pytest.mark.parametrize("fps", [100, 120])
def test_reference_resampling_couples_source_and_policy_fps(fps: int) -> None:
    trajectory = resample_reference_trajectory(
        _four_frame_reference(), reference_fps=fps
    )

    assert trajectory.reference_fps == fps
    assert trajectory.control_fps == fps
    assert len(trajectory.q_ref) == 4
    assert np.diff(trajectory.timestamps).tolist() == pytest.approx([1.0 / fps] * 3)
    np.testing.assert_allclose(trajectory.q_ref[:, 0], np.arange(4))
    assert np.rad2deg(trajectory.q_ref[2, 3]) == pytest.approx(181.0)
    np.testing.assert_array_equal(trajectory.source_indices, [10, 11, 12, 13])
    assert (trajectory.movement_start_step, trajectory.movement_end_step) == (1, 2)


def test_reference_resampling_rejects_mixed_public_clock_modes() -> None:
    with pytest.raises(ValueError, match="reference_fps == control_fps"):
        resample_reference_trajectory(
            _four_frame_reference(), reference_fps=100, control_fps=120
        )
    with pytest.raises(ValueError, match="reference_fps == control_fps"):
        TrajectorySelection(reference_fps=100, control_fps=120)


def test_reference_resampling_retains_internal_legacy_200hz_clock() -> None:
    trajectory = resample_reference_trajectory(
        _four_frame_reference(), reference_fps=100, control_fps=200
    )

    assert trajectory.control_fps == 200
    assert len(trajectory.q_ref) == 7
    assert np.diff(trajectory.timestamps).tolist() == pytest.approx(
        [CONTROL_TIMESTEP] * 6
    )
    np.testing.assert_allclose(trajectory.q_ref[:, 0], np.arange(7) * 0.5)
    assert np.rad2deg(trajectory.q_ref[3, 3]) == pytest.approx(180.0)
    np.testing.assert_array_equal(
        trajectory.source_indices, [10, 10, 11, 11, 12, 12, 13]
    )
    assert (trajectory.movement_start_step, trajectory.movement_end_step) == (2, 4)

    implicit_legacy_control = replace(trajectory, control_fps=None)
    canonical = resample_reference_trajectory(
        implicit_legacy_control, reference_fps=100, control_fps=200
    )
    assert canonical.control_fps == 200


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


def test_modern_capture_edge_holds_missing_padding_to_exact_duration() -> None:
    q = np.zeros((6, 28), dtype=np.float64)
    q[:, 0] = np.arange(6)
    row = {
        "index": {
            "uuid": "00000000-0000-0000-0000-000000000001",
            "scene": "cube1",
            "gesture": "001-Palmar-Pinch",
            "is_generated": False,
        },
        "trajectory_metadata": {
            "total_frames": 6,
            "data_fps": 100,
            "hand_names": ["right"],
            "object_names": ["cube1"],
            "trajectory_info": {
                "object_move": [
                    {"object_name": "cube1", "start_frame": 1, "end_frame": 3}
                ]
            },
        },
        "timestamp": (np.arange(6, dtype=np.float64) / 100.0).tolist(),
        "hands": [{"hand_name": "right", "urdf_dof": q.tolist()}],
        "objects": [
            {
                "pos": np.zeros((6, 3), dtype=np.float64).tolist(),
                "rot_aa": np.zeros((6, 3), dtype=np.float64).tolist(),
            }
        ],
    }
    selection = TrajectorySelection(
        object_type="cube1",
        gesture="01",
        selector="cube1:01",
        pre_padding=3,
        post_padding=4,
        reference_fps=100,
    )
    candidate = _candidate_from_metadata_row(row, row_index=0, selection=selection)
    assert candidate is not None

    trajectory = trajectory_from_lance_row(
        row,
        295,
        pre_padding=3,
        post_padding=4,
        hand_side="right",
    )

    assert len(trajectory.q_ref) == 10
    assert (trajectory.movement_start_step, trajectory.movement_end_step) == (3, 5)
    assert len(trajectory.q_ref) - 1 - trajectory.movement_end_step == 4
    np.testing.assert_array_equal(
        trajectory.source_indices, [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]
    )
    np.testing.assert_array_equal(
        trajectory.q_ref[:, 0], [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]
    )
    np.testing.assert_allclose(np.diff(trajectory.timestamps), 0.01, atol=1e-15)


def _composite_scene_row(
    *, active_object: str = "mayonnaisebottle"
) -> dict[str, object]:
    frames = 6
    scene_objects = ["bowl", "mayonnaisebottle"]
    active_index = scene_objects.index(active_object)
    return {
        "index": {
            "uuid": "composite-row",
            "file_uuid": "composite-file",
            "scene": ",".join(scene_objects),
            "gesture": "005-mayonnaisebottle-bowl-pick-up",
            "is_generated": False,
        },
        "trajectory_metadata": {
            "total_frames": frames,
            "data_fps": 100,
            "hand_names": ["right"],
            "trajectory_info": {
                "object_move": [
                    {
                        "object_name": active_object,
                        "start_frame": 1,
                        "end_frame": 4,
                    }
                ]
            },
        },
        "timestamp": (np.arange(frames, dtype=np.float64) / 100.0).tolist(),
        "hands": [
            {
                "hand_name": "right",
                "urdf_dof": np.zeros((frames, 28), dtype=np.float64).tolist(),
            }
        ],
        "objects": [
            {
                "pos": np.repeat([[float(index), 0.0, 1.0]], frames, axis=0).tolist(),
                "rot_aa": np.zeros((frames, 3), dtype=np.float64).tolist(),
            }
            for index in range(len(scene_objects))
        ],
        "_expected_active_index": active_index,
    }


def test_composite_scene_uses_unique_manipulated_object_and_ordered_state() -> None:
    row = _composite_scene_row()
    expected_index = row.pop("_expected_active_index")
    selection = TrajectorySelection(
        object_type="mayonnaisebottle",
        gesture="05",
        selector="mayonnaisebottle:05",
        pre_padding=0,
        post_padding=0,
        hand_side="right",
    )

    candidate = _candidate_from_metadata_row(row, row_index=7, selection=selection)
    assert candidate is not None
    assert candidate.pair == ObjectActionPair("mayonnaisebottle", "05")
    assert candidate.identity == "mayonnaisebottle_05_008"

    trajectory = _selected_trajectory_from_row(
        row,
        5,
        row_index=7,
        selection=selection,
        expected_pair=ObjectActionPair("mayonnaisebottle", "05"),
    )
    assert trajectory.identity.identity == "mayonnaisebottle_05_008"
    assert trajectory.identity.object_index == expected_index == 1
    np.testing.assert_allclose(trajectory.object_pos_raw[:, 0], 1.0)
    assert (trajectory.movement_start_step, trajectory.movement_end_step) == (0, 3)


def test_composite_scene_rejects_ambiguous_manipulated_objects() -> None:
    row = _composite_scene_row()
    row.pop("_expected_active_index")
    row["trajectory_metadata"]["trajectory_info"]["object_move"].append(
        {"object_name": "bowl", "start_frame": 1, "end_frame": 4}
    )
    selection = TrajectorySelection(
        object_type="mayonnaisebottle",
        gesture="05",
        selector="mayonnaisebottle:05",
    )

    assert _candidate_from_metadata_row(row, row_index=0, selection=selection) is None
    with pytest.raises(ValueError, match="exactly one manipulated object"):
        trajectory_from_lance_row(row, dataset_version=5)


def test_candidate_rejects_negative_movement_pre_padding_margin(tmp_path: Path) -> None:
    row = deepcopy(_accepted_fake_row(np.arange(1373, dtype=np.float64) / 111.0))
    row["index"].update(
        scene="cube1",
        source_path="cube1/cube1_01_009/cube1_01_009_mano.npy",
    )
    row["trajectory_metadata"]["trajectory_info"]["object_move"][0]["start_frame"] = 50
    selection = TrajectorySelection(
        "cube1", "01", dataset_path=tmp_path / "source.lance", pre_padding=100
    )
    assert _candidate_from_metadata_row(row, row_index=0, selection=selection) is None
    with pytest.raises(ValueError, match="pre-padding"):
        _selected_trajectory_from_row(
            row,
            EXPECTED_DATASET_VERSION,
            row_index=0,
            selection=selection,
            expected_pair=ObjectActionPair("cube1", "01"),
        )


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


def test_assigned_loader_rotates_each_pair_by_local_slot_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "source.lance"
    path.mkdir()
    timestamps = np.arange(1373, dtype=np.float64) / 111.0
    rows = [deepcopy(_accepted_fake_row(timestamps)) for _ in range(3)]
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

    monkeypatch.setattr(lance, "dataset", lambda _: FakeDataset(), raising=False)
    monkeypatch.setattr("sim.manorl.trajectory._initial_support_shift", lambda *_: 0.0)
    observed = []
    for cycle in range(4):
        batch = load_assigned_trajectory_batch(
            TrajectorySelection(
                "cube1", "01", dataset_path=path, pair_assignment_cycle=cycle
            ),
            num_envs=1,
        )
        observed.append(batch.trajectories[0].identity.row_index)
        assert batch.pair_assignment_cycle == cycle

    assert observed == [0, 1, 2, 0]


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


def test_composite_scene_ground_shift_preserves_all_relative_positions(monkeypatch):
    import sim.manorl.trajectory as module
    row = _composite_scene_row()
    # Active object is deliberately second; first body is the lowest support.
    row["objects"][0]["pos"] = np.tile([0., 0., .1], (6, 1)).tolist()
    row["objects"][1]["pos"] = np.tile([0., 0., .4], (6, 1)).tolist()
    monkeypatch.setattr(module, "_initial_support_shift", lambda pos, quat, name: -pos[2])
    trajectory = trajectory_from_lance_row(row, 5)
    assert trajectory.scene_object_types == ("bowl", "mayonnaisebottle")
    assert trajectory.object_z_shift == pytest.approx(-.1)
    np.testing.assert_allclose(trajectory.scene_object_initial_pos[:, 2], [0., .3])
    np.testing.assert_allclose(trajectory.object_pos[:, 2], .3)
    np.testing.assert_allclose(trajectory.q_ref[:, 2], -.1)
    assert not trajectory.scene_object_initial_pos.flags.writeable
    resampled = resample_reference_trajectory(trajectory, reference_fps=100)
    np.testing.assert_array_equal(resampled.scene_object_initial_pos, trajectory.scene_object_initial_pos)
    np.testing.assert_allclose(resampled.object_pos[0], trajectory.scene_object_initial_pos[1])


def test_composite_scene_rejects_malformed_passive_pose():
    row = _composite_scene_row()
    row["objects"][0]["pos"][0][0] = float("nan")
    with pytest.raises(ValueError, match="bowl pose arrays contain non-finite"):
        trajectory_from_lance_row(row, 5)
