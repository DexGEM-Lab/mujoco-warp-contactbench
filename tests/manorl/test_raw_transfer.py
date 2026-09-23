from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.manorl import assets
from sim.manorl.trajectory import (
    RAW_CAPTURE_TRANSFER_CONTRACT,
    ObjectActionPair,
    TrajectorySelection,
    _selected_trajectory_from_row,
)
from sim.manorl.trajectory_package import (
    load_trajectory_package,
    write_trajectory_package,
)


def _raw_row(*, fps: int = 100, scene: str = "cube1", gesture: str = "001-Palmar-Pinch") -> dict:
    frames = 5
    timestamps = np.arange(frames, dtype=np.float64) / float(fps)
    right = np.zeros((frames, 28), dtype=np.float64)
    right[:, 0] = timestamps
    right[:, 6] = np.linspace(0.0, 0.4, frames)
    left = right.copy()
    left[:, 0] *= -1.0
    object_names = scene.split(",")
    objects = []
    for slot, _name in enumerate(object_names):
        position = np.zeros((frames, 3), dtype=np.float64)
        position[:, 0] = slot * 0.05 + timestamps
        rotvec = np.zeros((frames, 3), dtype=np.float64)
        rotvec[:, 2] = timestamps
        objects.append({"pos": position.tolist(), "rot_aa": rotvec.tolist()})
    active = object_names[0]
    return {
        "index": {
            "uuid": "00000000-0000-0000-0000-000000000001",
            "file_uuid": "capture-file",
            "operator": "recorded-operator",
            "scene": scene,
            "gesture": gesture,
            "is_generated": False,
        },
        "trajectory_metadata": {
            "total_frames": frames,
            "data_fps": fps,
            "hand_names": ["left", "right"],
            "mano_hand_shapes": [[-0.1] * 10, [0.2] * 10],
            "object_names": object_names,
            "trajectory_info": {
                "object_move": [
                    {"object_name": active, "start_frame": 1, "end_frame": 3}
                ]
            },
        },
        "timestamp": timestamps.tolist(),
        "hands": [
            {"hand_name": "left", "urdf_dof": left.tolist()},
            {"hand_name": "right", "urdf_dof": right.tolist()},
        ],
        "objects": objects,
    }


def _selection(path: Path, *, override: str = "") -> TrajectorySelection:
    return TrajectorySelection(
        selector="all",
        dataset_path=path,
        expected_dataset_version=7,
        pre_padding=0,
        post_padding=0,
        hand_side="right",
        drop_uncontrolled_hands=True,
        target_object_overrides=override,
        reference_fps=120,
        raw_transfer=True,
    )


def test_raw_transfer_preserves_duration_final_pose_and_full_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    row = _raw_row(fps=100)
    monkeypatch.setattr(
        "sim.manorl.trajectory._initial_scene_support_shift",
        lambda *_args: 0.03,
    )
    selection = _selection(tmp_path / "source.lance")
    trajectory = _selected_trajectory_from_row(
        row,
        7,
        row_index=4,
        selection=selection,
        expected_pair=ObjectActionPair("cube1", "01"),
    )

    assert trajectory.reference_contract == RAW_CAPTURE_TRANSFER_CONTRACT
    assert trajectory.hand_sides == ("right",)
    assert trajectory.selected_hand_sides == ("right",)
    assert trajectory.source_operator == "recorded-operator"
    assert trajectory.source_hand_betas == (0.2,) * 10
    assert trajectory.identity.source_start == 0
    assert trajectory.identity.source_stop == 5
    assert trajectory.movement_start_step == 2
    assert trajectory.movement_end_step == 4
    assert len(trajectory.q_ref) == 6
    assert trajectory.timestamps[-1] == pytest.approx(5.0 / 120.0)
    assert trajectory.timestamps[-1] - 0.04 < 1.0 / 120.0
    np.testing.assert_allclose(trajectory.q_ref[-1, 0], 0.04)
    np.testing.assert_allclose(trajectory.q_ref[:, 2], 0.03)
    np.testing.assert_allclose(trajectory.object_pos_raw[-1], [0.04, 0.0, 0.0])
    np.testing.assert_allclose(trajectory.object_pos[-1], [0.04, 0.0, 0.03])
    expected_quat = Rotation.from_rotvec([0.0, 0.0, 0.04]).as_quat()
    assert abs(np.dot(trajectory.object_quat_xyzw[-1], expected_quat)) == pytest.approx(1.0)


def test_raw_transfer_preserves_compound_scene_with_explicit_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    row = _raw_row(scene="bottle,cap", gesture="018-Lateral")
    row["trajectory_metadata"]["trajectory_info"]["object_move"][0][
        "object_name"
    ] = "cap,bottle"
    monkeypatch.setattr(
        "sim.manorl.trajectory._initial_scene_support_shift",
        lambda *_args: 0.0,
    )
    selection = _selection(tmp_path / "source.lance", override="bottle:18")
    trajectory = _selected_trajectory_from_row(
        row,
        7,
        row_index=8,
        selection=selection,
        expected_pair=ObjectActionPair("bottle", "18"),
    )

    assert trajectory.identity.identity == "bottle_18_009"
    assert trajectory.identity.object_index == 0
    assert trajectory.scene_object_types == ("bottle", "cap")
    np.testing.assert_allclose(trajectory.scene_object_initial_pos[1], [0.05, 0.0, 0.0])


def test_raw_transfer_package_binds_source_and_fixed_physical_hand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    row = _raw_row()
    monkeypatch.setattr(
        "sim.manorl.trajectory._initial_scene_support_shift",
        lambda *_args: 0.0,
    )
    selection = _selection(tmp_path / "source.lance")
    trajectory = _selected_trajectory_from_row(
        row,
        7,
        row_index=0,
        selection=selection,
        expected_pair=ObjectActionPair("cube1", "01"),
    )
    monkeypatch.setattr(assets, "EXPLICIT_ASSET_MANIFEST", "/fixed/cheyingtong.json")
    monkeypatch.setattr(assets, "MANO_OPERATOR", "cheyingtong")
    monkeypatch.setattr(
        assets,
        "asset_provenance",
        lambda: {
            "asset_source_repository": "repo",
            "asset_source_commit": "commit",
            "asset_manifest_sha256": "a" * 64,
        },
    )
    package = write_trajectory_package(
        tmp_path / "raw.mtp",
        [trajectory],
        selection=selection,
        dataset_schema_digest="schema",
        discovery_digest="discovery",
    )
    catalog = load_trajectory_package(package)

    assert catalog.manifest["physical_hand_asset_profile"]["hand_operator"] == "cheyingtong"
    assert catalog.manifest["selection"]["raw_transfer"] is True
    loaded = catalog.trajectories[0]
    assert loaded.reference_contract == RAW_CAPTURE_TRANSFER_CONTRACT
    assert loaded.source_operator == "recorded-operator"
    assert loaded.source_hand_betas == (0.2,) * 10
