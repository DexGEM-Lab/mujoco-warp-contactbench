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
