from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.manorl.assets import object_collision_vertices
from sim.manorl.contracts import DATASET_PATH, EXPECTED_DATASET_VERSION
from sim.manorl.trajectory import (
    LANCE_COLUMNS,
    load_reference_trajectory,
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
    assert trajectory.source_indices[0] == 10
    assert trajectory.source_indices[-1] == 603
    assert trajectory.q_ref.shape == (594, 26)
    assert trajectory.object_pos_raw.shape == (594, 3)
    assert trajectory.object_pos.shape == (594, 3)
    assert trajectory.object_quat_xyzw.shape == (594, 4)
    assert not trajectory.q_ref.flags.writeable
    assert trajectory.identity.identity == "powerdrill_02_002"
    rotated_vertices = Rotation.from_quat(trajectory.object_quat_xyzw[0]).apply(
        object_collision_vertices().copy()
    )
    support_bottom = np.min(rotated_vertices[:, 2] + trajectory.object_pos[0, 2])
    assert support_bottom == pytest.approx(0.0, abs=1e-12)
    assert trajectory.object_z_shift == pytest.approx(0.0005318821542129315, abs=1e-12)
    np.testing.assert_allclose(
        trajectory.object_pos[:, 2] - trajectory.object_pos_raw[:, 2],
        trajectory.object_z_shift,
        atol=1e-15,
    )


def _accepted_fake_row(timestamps: np.ndarray) -> dict[str, object]:
    return {
        "index": {
            "uuid": "e49b87fb-51c1-44eb-aade-666b5e617959",
            "file_uuid": "20260528022141_a5fb81e3",
            "gesture": "002-test",
        },
        "trajectory_metadata": {
            "object_names": ["powerdrill"],
            "hand_names": ["right"],
            "raw_data_info": {"id": 1},
            "total_frames": 605,
            "data_fps": 100,
            "trajectory_info": {
                "object_move": [
                    {"object_name": "powerdrill", "start_frame": 260, "end_frame": 444}
                ]
            },
        },
        "timestamp": timestamps,
        "hands": [{"urdf_dof": np.zeros((605, 26))}],
        "objects": [{"pos": np.repeat([[0.0, 0.0, 1.0]], 605, axis=0), "rot_aa": np.zeros((605, 3))}],
    }


def test_timestamp_jitter_is_validated_by_rate_and_envelope() -> None:
    deltas = np.resize(np.array([0.006, 0.014], dtype=np.float64), 604)
    timestamps = np.concatenate([[0.0], np.cumsum(deltas)])
    trajectory = trajectory_from_row(_accepted_fake_row(timestamps), EXPECTED_DATASET_VERSION)
    assert trajectory.timestamps.shape == (594,)

    bad_rate = np.arange(605, dtype=np.float64) * 0.012
    with pytest.raises(ValueError, match="mean timestamp interval"):
        trajectory_from_row(_accepted_fake_row(bad_rate), EXPECTED_DATASET_VERSION)

    bad_jitter = np.arange(605, dtype=np.float64) * 0.01
    bad_jitter[301:] += 0.006
    with pytest.raises(ValueError, match="jitter"):
        trajectory_from_row(_accepted_fake_row(bad_jitter), EXPECTED_DATASET_VERSION)


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
