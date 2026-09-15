"""New-capture defaults, physical hand omission, and checkpoint boundaries."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sim.manorl.abi import ResidualActionConfig, process_residual_actions
from sim.manorl.checkpoint import CheckpointFormatError, _validate_environment_signature
from sim.manorl.contracts import KEYPOINT_NAMES
from sim.manorl.environment import EnvironmentConfig, _active_joint_mask, _expected_keypoint_ids
from sim.manorl.trajectory import (
    ObjectActionPair, TrajectorySelection, _candidate_from_metadata_row,
    _selected_trajectory_from_row, trajectory_from_lance_row,
)
from sim.manorl.trajectory_package import write_trajectory_package, load_trajectory_package, assign_trajectory_catalog
from tools.train_manorl_cube1 import TrainingBudget


def capture_row():
    n = 6
    return {
        "index": {"scene": "bowl,egg_cup", "gesture": "004-pour", "uuid": "u", "file_uuid": "f", "is_generated": False},
        "trajectory_metadata": {
            "total_frames": n, "hand_names": ["left", "right"],
            "trajectory_info": {"object_move": [{"object_name": "egg_cup", "start_frame": 1, "end_frame": 4}]},
        },
        "timestamp": (np.arange(n) / 120).tolist(),
        "hands": [{"urdf_dof": np.full((n, 28), np.nan)}, {"urdf_dof": np.zeros((n, 28))}],
        "objects": [{"pos": np.tile([i, 0., .1], (n, 1)), "rot_aa": np.zeros((n, 3))} for i in range(2)],
    }


def test_training_defaults_and_five_distal_sites_enable_all_fingers():
    budget = TrainingBudget()
    assert budget.residual_action_config.position_scale == (.002,) * 3
    assert budget.residual_action_config.max_position_offset == (.01,) * 3
    assert budget.expected_contact_mode == "five_fingertips"
    ids = _expected_keypoint_ids("egg_cup", "04", budget.expected_contact_mode)
    assert tuple(KEYPOINT_NAMES[i] for i in ids) == (
        "thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip"
    )
    mask = np.zeros((1, 16)); mask[0, ids] = 1
    assert mask.sum() == 5
    assert _active_joint_mask(mask, finger_dof=22).all()
    # Source-map / generic environment defaults still bind historical checkpoints.
    assert EnvironmentConfig().expected_contact_mode == "source_mapping"
    assert ResidualActionConfig().position_scale == (.003,) * 3


def test_residual_xyz_contribution_and_saturating_cap():
    cfg = TrainingBudget().residual_action_config
    offset = np.zeros((1, 3)); joints = np.zeros((1, 22))
    for step in range(100, 140):
        out = process_residual_actions(
            np.ones((1, 28)), trajectory_steps=np.array([step]),
            cumulative_offset=offset, cumulative_joint_offset=joints,
            mocap_targets=np.zeros((1, 28)), joint_lower=np.full(28, -10.),
            joint_upper=np.full(28, 10.), active_joint_mask=np.ones((1, 22), bool),
            config=cfg,
        )
        if step == 100:
            np.testing.assert_allclose(out.cumulative_offset, .002)
        offset = out.cumulative_offset; joints = out.cumulative_joint_offset
    np.testing.assert_allclose(offset, .01)


def test_right_only_decoder_does_not_require_valid_left_pose(monkeypatch):
    import sim.manorl.trajectory as module
    monkeypatch.setattr(module, "_initial_scene_support_shift", lambda *args: 0.)
    row = capture_row()
    selection = TrajectorySelection(selector="egg_cup:04", hand_side="right",
        reference_fps=120, pre_padding=0, post_padding=0, drop_uncontrolled_hands=True)
    candidate = _candidate_from_metadata_row(row, row_index=7, selection=selection)
    assert candidate.pair == ObjectActionPair("egg_cup", "04")
    assert candidate.identity == "egg_cup_04_008"
    t = _selected_trajectory_from_row(row, 3, row_index=7, selection=selection,
        expected_pair=candidate.pair)
    assert t.hand_sides == t.selected_hand_sides == ("right",)
    assert set(t.q_ref_by_side) == {"right"}
    assert t.action_layout.action_dim == 28
    assert t.scene_object_types == ("bowl", "egg_cup")
    assert t.identity.object_index == 1
    assert t.q_ref.shape == (4, 28)
    with pytest.raises(ValueError, match="finite"):
        trajectory_from_lance_row(row, 3, hand_side="right")


def test_combined_target_not_silently_selected():
    row = capture_row()
    row["trajectory_metadata"]["trajectory_info"]["object_move"][0]["object_name"] = "egg_cup,bowl"
    assert _candidate_from_metadata_row(row, row_index=0, selection=TrajectorySelection(selector="all")) is None


def test_contact_mode_checkpoint_mismatch_fails_and_legacy_is_source_mapping():
    metadata = {"runtime_config": {"environment": {}}}
    agent = SimpleNamespace(manorl_environment_signature={"expected_contact_mode": "source_mapping"})
    _validate_environment_signature(metadata, agent)
    agent.manorl_environment_signature["expected_contact_mode"] = "five_fingertips"
    with pytest.raises(CheckpointFormatError, match="expected_contact_mode"):
        _validate_environment_signature(metadata, agent)
    metadata["runtime_config"]["environment"]["expected_contact_mode"] = "five_fingertips"
    _validate_environment_signature(metadata, agent)


def test_right_only_package_round_trip_and_selection_contract(tmp_path, monkeypatch):
    import sim.manorl.trajectory as module
    monkeypatch.setattr(module, "_initial_scene_support_shift", lambda *args: 0.)
    s = TrajectorySelection(selector="egg_cup:04", dataset_path=tmp_path / "source.lance",
        expected_dataset_version=3, hand_side="right", reference_fps=120,
        pre_padding=0, post_padding=0, drop_uncontrolled_hands=True)
    row = capture_row()
    t = _selected_trajectory_from_row(row, 3, row_index=0, selection=s,
        expected_pair=ObjectActionPair("egg_cup", "04"))
    p = write_trajectory_package(tmp_path / "right.mtp", (t,), selection=s,
        dataset_schema_digest="schema", discovery_digest="discovery")
    c = load_trajectory_package(p)
    b = assign_trajectory_catalog(c, s, num_envs=2)
    with pytest.raises(ValueError, match="target_object_overrides"):
        assign_trajectory_catalog(c, replace(s, target_object_overrides="egg_cup:04"), num_envs=1)
    assert b.hand_sides == ("right",)
    assert b.action_dim == 28
    assert b.trajectories[0].scene_object_types == ("bowl", "egg_cup")
    with pytest.raises(ValueError, match="drop_uncontrolled_hands"):
        assign_trajectory_catalog(c, replace(s, drop_uncontrolled_hands=False), num_envs=1)


def test_package_checks_explicit_hand_profile(tmp_path, monkeypatch):
    from sim.manorl import assets
    import sim.manorl.trajectory as module
    monkeypatch.setattr(module, "_initial_scene_support_shift", lambda *args: 0.)
    s = TrajectorySelection(selector="egg_cup:04", hand_side="right", reference_fps=120,
        pre_padding=0, post_padding=0, drop_uncontrolled_hands=True)
    t = _selected_trajectory_from_row(capture_row(), 3, row_index=0, selection=s,
        expected_pair=ObjectActionPair("egg_cup", "04"))
    monkeypatch.setattr(assets, "EXPLICIT_ASSET_MANIFEST", "profile.json")
    monkeypatch.setattr(assets, "asset_provenance", lambda: {"asset_manifest_sha256": "profile-a"})
    p = write_trajectory_package(tmp_path / "profile.mtp", (t,), selection=s,
        dataset_schema_digest="schema", discovery_digest="discovery")
    assert load_trajectory_package(p).manifest["source_hand_asset_profile"]["asset_manifest_sha256"] == "profile-a"
    monkeypatch.setattr(assets, "asset_provenance", lambda: {"asset_manifest_sha256": "profile-b"})
    with pytest.raises(ValueError, match="source hand asset profile mismatch"):
        load_trajectory_package(p)


def test_sept15_preset_keeps_right_only_and_requested_padding():
    from pathlib import Path
    import subprocess
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", "-c", 'source configs/manorl/sept15_right.env; printf "%s\\n" "$MANORL_HAND_SIDE" "$MANORL_DROP_UNCONTROLLED_HANDS" "$MANORL_REFERENCE_FPS" "$MANORL_PRE_PADDING" "$MANORL_POST_PADDING"'],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines() == ["right", "true", "120", "60", "250"]


@pytest.mark.parametrize("annotation", ["egg_cup,bowl", "bowl,egg_cup", "egg_cup"])
def test_explicit_cup_override_preserves_source_and_object_order(annotation, monkeypatch):
    from copy import deepcopy
    import sim.manorl.trajectory as module
    monkeypatch.setattr(module, "_initial_scene_support_shift", lambda *args: 0.)
    row = capture_row()
    row["trajectory_metadata"]["trajectory_info"]["object_move"][0]["object_name"] = annotation
    original = deepcopy(row["trajectory_metadata"])
    selection = TrajectorySelection(selector="all", hand_side="right", reference_fps=120,
        pre_padding=60, post_padding=250, drop_uncontrolled_hands=True,
        target_object_overrides="egg_cup:4")
    assert selection.target_object_overrides == "egg_cup:04"
    c = _candidate_from_metadata_row(row, row_index=2, selection=selection)
    assert c.pair == ObjectActionPair("egg_cup", "04")
    t = _selected_trajectory_from_row(row, 4, row_index=2, selection=selection, expected_pair=c.pair)
    assert t.identity.identity == "egg_cup_04_003"
    assert t.identity.object_index == 1
    assert t.scene_object_types == ("bowl", "egg_cup")
    assert t.hand_sides == ("right",) and t.movement_start_step == 60
    assert row["trajectory_metadata"] == original


def test_override_must_not_invent_unannotated_target():
    s = TrajectorySelection(target_object_overrides="bowl:04")
    with pytest.raises(ValueError, match="absent"):
        _candidate_from_metadata_row(capture_row(), row_index=0, selection=s)
    with pytest.raises(ValueError, match="one object per action"):
        TrajectorySelection(target_object_overrides="bowl:04,egg_cup:04")
    with pytest.raises(ValueError, match="one object per action"):
        TrajectorySelection(target_object_overrides="all")


def test_dense_capture_constraint_capacity_is_explicit():
    assert TrainingBudget().constraint_capacity == 512
    assert TrainingBudget(constraint_capacity=2048).constraint_capacity == 2048
