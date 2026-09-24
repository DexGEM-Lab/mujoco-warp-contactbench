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


def _raw_row(
    *,
    fps: int = 100,
    scene: str = "cube1",
    gesture: str = "001-Palmar-Pinch",
    frames: int = 5,
    move_start: int = 1,
    move_end: int = 3,
) -> dict:
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
                    {
                        "object_name": active,
                        "start_frame": move_start,
                        "end_frame": move_end,
                    }
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


def test_multi_source_package_assignment_keeps_both_source_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dataclasses import replace

    from sim.manorl.trajectory_package import load_assigned_trajectory_packages

    monkeypatch.setattr(
        "sim.manorl.trajectory._initial_scene_support_shift",
        lambda *_args: 0.0,
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
    first_selection = _selection(tmp_path / "remake.lance")
    second_selection = replace(
        first_selection,
        dataset_path=tmp_path / "guangguan.lance",
        expected_dataset_version=8,
    )
    first = _selected_trajectory_from_row(
        _raw_row(),
        7,
        row_index=0,
        selection=first_selection,
        expected_pair=ObjectActionPair("cube1", "01"),
    )
    second = _selected_trajectory_from_row(
        _raw_row(),
        8,
        row_index=0,
        selection=second_selection,
        expected_pair=ObjectActionPair("cube1", "01"),
    )
    first_package = write_trajectory_package(
        tmp_path / "remake.mtp",
        [first],
        selection=first_selection,
        dataset_schema_digest="schema-a",
        discovery_digest="discovery-a",
    )
    second_package = write_trajectory_package(
        tmp_path / "guangguan.mtp",
        [second],
        selection=second_selection,
        dataset_schema_digest="schema-b",
        discovery_digest="discovery-b",
    )
    runtime_selection = replace(
        first_selection,
        dataset_path=tmp_path / "unused.lance",
        expected_dataset_version=None,
    )
    batch = load_assigned_trajectory_packages(
        [first_package, second_package],
        runtime_selection,
        num_envs=2,
    )

    assert {item.identity.dataset_path for item in batch.trajectories} == {
        str((tmp_path / "remake.lance")),
        str((tmp_path / "guangguan.lance")),
    }
    assert batch.trajectory_package is not None
    assert batch.trajectory_package["schema"] == "manorl.trajectory_package_bundle.v1"


def test_raw_training_profile_uses_small_regularized_all_joint_residual() -> None:
    from sim.manorl.environment import EnvironmentConfig, _expected_keypoint_ids
    from sim.manorl.contracts import KEYPOINT_NAMES
    from tools.train_manorl_cube1 import TrainingBudget

    budget = TrainingBudget(
        trajectory_packages=("remake.mtp", "guangguan.mtp"),
        dataset_version=None,
        reference_fps=120,
        hand_side="right",
        drop_uncontrolled_hands=True,
        raw_transfer=True,
        expected_contact_mode="raw_gesture",
        residual_joint_mode="all",
        action_penalty_scale=1.0,
        pre_padding=0,
        post_padding=0,
        position_scale=0.001,
        max_position_offset=0.01,
        joint_scale_multiplier=1.0,
        joint_max_offset_multiplier=1.0,
    )
    residual = budget.residual_action_config
    assert residual.position_scale == (0.001, 0.001, 0.001)
    assert residual.max_position_offset == (0.01, 0.01, 0.01)
    assert residual.joint_scale_multiplier == 1.0
    assert residual.joint_max_offset_multiplier == 1.0
    config = EnvironmentConfig(
        expected_contact_mode=budget.expected_contact_mode,
        residual_joint_mode=budget.residual_joint_mode,
    )
    assert config.residual_joint_mode == "all"
    names = tuple(KEYPOINT_NAMES[index] for index in _expected_keypoint_ids("bottle", "18", "raw_gesture"))
    assert names == ("thumb_ip", "index_dip")


def test_raw_training_profile_rejects_operator_or_clock_fallbacks() -> None:
    from tools.train_manorl_cube1 import TrainingBudget

    with pytest.raises(ValueError, match="raw_transfer training requires"):
        TrainingBudget(raw_transfer=True, trajectory_package="raw.mtp")


def test_training_cli_builds_one_fixed_hand_multi_source_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tools import train_manorl_cube1 as tool

    remake = tmp_path / "remake.mtp"
    guangguan = tmp_path / "guangguan.mtp"
    remake.mkdir()
    guangguan.mkdir()
    captured: dict[str, object] = {}

    def fake_run(output: Path, budget):
        captured["output"] = output
        captured["budget"] = budget
        return {"status": "mocked"}

    monkeypatch.setattr(tool, "run", fake_run)
    assert tool.main([
        "--output", str(tmp_path / "run"),
        "--updates", "1",
        "--num-envs", "2",
        "--evaluation-enabled", "false",
        "--trajectory-packages", str(remake),
        "--trajectory-packages", str(guangguan),
        "--all-pairs",
        "--raw-transfer",
        "--reference-fps", "120",
        "--hand-side", "right",
        "--drop-uncontrolled-hands",
        "--pre-padding", "0",
        "--post-padding", "0",
        "--target-object-overrides", "bottle:18",
        "--expected-contact-mode", "raw_gesture",
        "--residual-joint-mode", "all",
        "--action-penalty-scale", "1.0",
        "--position-scale", "0.001",
        "--max-position-offset", "0.01",
        "--joint-scale-multiplier", "1.0",
        "--joint-max-offset-multiplier", "1.0",
        "--unified-object-batch",
        "--wandb", "false",
    ]) == 0
    budget = captured["budget"]
    assert budget.raw_transfer is True
    assert budget.hand_side == "right"
    assert budget.trajectory_packages == (str(remake), str(guangguan))
    assert budget.expected_contact_mode == "raw_gesture"
    assert budget.residual_joint_mode == "all"


def test_runtime_pre_padding_crops_episode_start_to_movement_window() -> None:
    from sim.manorl.trajectory_package import _apply_runtime_pre_padding

    row = _raw_row(fps=120, frames=200, move_start=100, move_end=150)
    selection = _selection(Path("/tmp/source.lance"))
    trajectory = _selected_trajectory_from_row(
        row,
        7,
        row_index=0,
        selection=selection,
        expected_pair=ObjectActionPair("cube1", "01"),
    )
    assert trajectory.movement_start_step is not None
    # Zero keeps the complete capture untouched (identical object).
    assert _apply_runtime_pre_padding(trajectory, 0) is trajectory
    # A pre-padding beyond the movement start clamps to frame zero.
    assert _apply_runtime_pre_padding(trajectory, 10_000) is trajectory

    start = int(trajectory.movement_start_step) - 80
    assert start > 0
    cropped = _apply_runtime_pre_padding(trajectory, 80)
    assert len(cropped.q_ref) == len(trajectory.q_ref) - start
    assert cropped.movement_start_step == trajectory.movement_start_step - start
    assert cropped.movement_end_step == trajectory.movement_end_step - start
    assert cropped.identity == trajectory.identity
    np.testing.assert_allclose(cropped.q_ref[0], trajectory.q_ref[start])
    np.testing.assert_allclose(cropped.object_pos[0], trajectory.object_pos[start])
    np.testing.assert_allclose(
        cropped.object_quat_xyzw[0], trajectory.object_quat_xyzw[start]
    )
    for side, values in cropped.q_ref_by_side.items():
        np.testing.assert_allclose(values[0], trajectory.q_ref_by_side[side][start])


def test_package_assignment_applies_runtime_pre_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dataclasses import replace as dc_replace

    from sim.manorl.trajectory_package import load_assigned_trajectory_packages

    monkeypatch.setattr(
        "sim.manorl.trajectory._initial_scene_support_shift",
        lambda *_args: 0.0,
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
    selection = _selection(tmp_path / "source.lance")
    trajectory = _selected_trajectory_from_row(
        _raw_row(fps=120, frames=200, move_start=100, move_end=150),
        7,
        row_index=0,
        selection=selection,
        expected_pair=ObjectActionPair("cube1", "01"),
    )
    package = write_trajectory_package(
        tmp_path / "raw.mtp",
        [trajectory],
        selection=selection,
        dataset_schema_digest="schema",
        discovery_digest="discovery",
    )
    assignment_selection = dc_replace(
        selection, pre_padding=80, expected_dataset_version=None
    )
    batch = load_assigned_trajectory_packages(
        [package], assignment_selection, num_envs=2
    )
    assigned = batch.trajectories[0]
    start = int(trajectory.movement_start_step) - 80
    assert start > 0
    assert len(assigned.q_ref) == len(trajectory.q_ref) - start
    assert assigned.movement_start_step == trajectory.movement_start_step - start
    np.testing.assert_allclose(assigned.q_ref[0], trajectory.q_ref[start])
