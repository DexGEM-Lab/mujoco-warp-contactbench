from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim.manorl.trajectory import (
    RL_EPISODE_REFERENCE_CONTRACT,
    RL_EPISODE_TARGET120_CONTRACT,
    RL_EPISODE_RESAMPLING_ID,
    TrajectorySelection,
    _discover_rl_episode_candidates,
    _rl_episode_candidate_from_discovery_row,
    resample_timestamped_reference_trajectory,
    trajectory_from_rl_episode_row,
    window_reference_trajectory,
)
from sim.manorl.trajectory_package import (
    load_trajectory_package,
    write_trajectory_package,
)
from sim.manorl.reference_motion import REFERENCE_MOTION_ANNOTATION_CONTRACT


def rl_episode_row(*, frames: int = 5, timestep: float = 0.005) -> dict[str, object]:
    timestamps = np.arange(frames, dtype=np.float64) * timestep
    q = np.zeros((frames, 28), dtype=np.float64)
    q[:, 0] = np.arange(frames, dtype=np.float64)
    q[:, 2] = 0.3
    pos = np.zeros((frames, 3), dtype=np.float64)
    pos[:, 0] = np.arange(frames, dtype=np.float64) * 0.01
    pos[:, 2] = 0.1
    uuid = "rl-row-uuid"
    return {
        "index": {
            "uuid": uuid,
            "seed_uuid": "seed-uuid",
            "scene": "mayonnaisebottle",
            "action_code": "05",
            "gesture": "05",
            "is_generated": True,
        },
        "trajectory_metadata": {
            "hand_names": ["right"],
            "hand_slots": ["right", "left"],
            "gesture": "005-mayonnaisebottle-bowl-pick-up",
            # Deliberately disagrees with the exact 200 Hz timestamps. The RL
            # adapter must preserve timestamp duration rather than metadata FPS.
            "data_fps": 100,
            "total_frames": frames,
        },
        "timestamp": timestamps.tolist(),
        "hands": [{"hand_name": "right", "urdf_dof": q.tolist()}],
        "objects": [
            {
                "pos": pos.tolist(),
                "rot_aa": np.zeros((frames, 3), dtype=np.float64).tolist(),
            }
        ],
        "provenance": {
            "source_rl_lance_path": "/source/generated_mano.lance",
            "source_rl_version": 13,
            "source_rl_row": 225,
            "source_rl_uuid": uuid,
        },
    }


def target120_rl_episode_row(*, frames: int = 5, timestep: float = 0.005) -> dict[str, object]:
    row = rl_episode_row(frames=frames, timestep=timestep)
    state = np.asarray(row["hands"][0]["urdf_dof"], dtype=np.float64)
    target = state.copy()
    target[:, 0] += np.linspace(0.0, 0.25, frames)
    target[:, 6:] += 0.1
    row["hands"][0]["urdf_dof_target"] = target.tolist()
    row["trajectory_metadata"]["state_target_contract"] = (
        RL_EPISODE_TARGET120_CONTRACT
    )
    return row


def fixed_asset_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from sim.manorl import assets

    monkeypatch.setattr(assets, "EXPLICIT_ASSET_MANIFEST", "/tmp/cheyingtong.json")
    monkeypatch.setattr(assets, "MANO_OPERATOR", "cheyingtong")
    monkeypatch.setattr(assets, "validate_asset_manifest", lambda *args, **kwargs: {})


def test_rl_episode_decoder_uses_fixed_hand_and_preserves_world_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixed_asset_profile(monkeypatch)
    row = rl_episode_row()
    trajectory = trajectory_from_rl_episode_row(
        row,
        1,
        row_index=12100,
        dataset_path=tmp_path / "refined.lance",
    )

    np.testing.assert_array_equal(trajectory.q_ref, row["hands"][0]["urdf_dof"])
    np.testing.assert_array_equal(trajectory.object_pos, row["objects"][0]["pos"])
    assert trajectory.object_z_shift == 0.0
    assert trajectory.identity.identity == "mayonnaisebottle_05_12101"
    assert trajectory.identity.uuid == row["index"]["uuid"]
    assert trajectory.hand_sides == trajectory.selected_hand_sides == ("right",)
    assert (trajectory.movement_start_step, trajectory.movement_end_step) == (0, 4)


def test_target120_decoder_separates_measured_state_and_actuator_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixed_asset_profile(monkeypatch)
    row = target120_rl_episode_row()
    trajectory = trajectory_from_rl_episode_row(
        row, 1, row_index=0, dataset_path=tmp_path / "target120.lance"
    )

    np.testing.assert_array_equal(
        trajectory.q_state_ref, row["hands"][0]["urdf_dof"]
    )
    np.testing.assert_array_equal(
        trajectory.q_ref, row["hands"][0]["urdf_dof_target"]
    )
    assert trajectory.state_target_contract == RL_EPISODE_TARGET120_CONTRACT

    resampled = resample_timestamped_reference_trajectory(
        trajectory, reference_fps=120
    )
    assert not np.array_equal(resampled.q_ref, resampled.q_state_ref)
    np.testing.assert_allclose(resampled.q_state_ref[0], trajectory.q_state_ref[0])
    np.testing.assert_allclose(resampled.q_ref[-1], trajectory.q_ref[-1])


def test_target_track_requires_explicit_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixed_asset_profile(monkeypatch)
    row = target120_rl_episode_row()
    row["trajectory_metadata"].pop("state_target_contract")
    with pytest.raises(ValueError, match="explicit target120 contract"):
        trajectory_from_rl_episode_row(
            row, 1, row_index=0, dataset_path=tmp_path / "invalid.lance"
        )


def test_rl_episode_decoder_consumes_reference_motion_annotation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixed_asset_profile(monkeypatch)
    row = rl_episode_row()
    row["trajectory_metadata"]["trajectory_info"] = {
        "object_move": [
            {
                "object_name": "mayonnaisebottle",
                "start_frame": 2,
                "end_frame": 4,
            }
        ]
    }
    row["trajectory_metadata"]["reference_motion_annotation"] = {
        "contract": REFERENCE_MOTION_ANNOTATION_CONTRACT,
        "start_frame": 2,
        "end_frame": 4,
    }
    trajectory = trajectory_from_rl_episode_row(
        row,
        1,
        row_index=0,
        dataset_path=tmp_path / "annotated.lance",
    )

    assert trajectory.identity.movement_start_raw == 2
    assert trajectory.identity.movement_end_raw == 4
    assert (trajectory.movement_start_step, trajectory.movement_end_step) == (2, 4)


def test_rl_episode_decoder_requires_explicit_fixed_asset_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sim.manorl import assets

    monkeypatch.setattr(assets, "EXPLICIT_ASSET_MANIFEST", "")
    with pytest.raises(ValueError, match="explicit fixed-hand"):
        trajectory_from_rl_episode_row(
            rl_episode_row(),
            1,
            row_index=0,
            dataset_path=tmp_path / "refined.lance",
        )


def test_timestamp_resampling_preserves_elapsed_duration_not_declared_fps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixed_asset_profile(monkeypatch)
    source = trajectory_from_rl_episode_row(
        rl_episode_row(),
        1,
        row_index=0,
        dataset_path=tmp_path / "refined.lance",
    )
    resampled = resample_timestamped_reference_trajectory(source, reference_fps=120)

    # Four 5 ms intervals span 20 ms. A 120 Hz grid needs three intervals,
    # holding the exact final source pose for less than one output step.
    assert len(resampled.q_ref) == 4
    np.testing.assert_allclose(resampled.timestamps, np.arange(4) / 120.0)
    np.testing.assert_allclose(resampled.q_ref[:, 0], [0.0, 5.0 / 3.0, 10.0 / 3.0, 4.0])
    np.testing.assert_allclose(resampled.object_pos[-1], source.object_pos[-1])
    assert resampled.reference_fps == resampled.control_fps == 120
    assert (resampled.movement_start_step, resampled.movement_end_step) == (0, 3)


def test_timestamp_resampled_window_enforces_pre60_with_real_margin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixed_asset_profile(monkeypatch)
    row = rl_episode_row(frames=200, timestep=0.005)
    row["trajectory_metadata"]["trajectory_info"] = {
        "object_move": [
            {
                "object_name": "mayonnaisebottle",
                "start_frame": 100,
                "end_frame": 199,
            }
        ]
    }
    row["trajectory_metadata"]["reference_motion_annotation"] = {
        "contract": REFERENCE_MOTION_ANNOTATION_CONTRACT,
        "start_frame": 100,
        "end_frame": 199,
        "confidence": "high",
    }
    source = trajectory_from_rl_episode_row(
        row, 1, row_index=0, dataset_path=tmp_path / "annotated.lance"
    )
    resampled = resample_timestamped_reference_trajectory(
        source, reference_fps=120
    )
    window = window_reference_trajectory(
        resampled, pre_padding=60, post_padding=0
    )

    assert resampled.movement_start_step == 60
    assert window.movement_start_step == 60
    assert window.identity.source_start == 0
    assert window.pre_edge_hold_steps == 0
    assert len(window.q_ref) == len(resampled.q_ref)
    assert np.all(np.diff(window.timestamps) > 0)


def test_timestamp_resampled_window_edge_holds_missing_pre60(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixed_asset_profile(monkeypatch)
    row = rl_episode_row(frames=160, timestep=0.005)
    row["trajectory_metadata"]["trajectory_info"] = {
        "object_move": [
            {
                "object_name": "mayonnaisebottle",
                "start_frame": 60,
                "end_frame": 159,
            }
        ]
    }
    row["trajectory_metadata"]["reference_motion_annotation"] = {
        "contract": REFERENCE_MOTION_ANNOTATION_CONTRACT,
        "start_frame": 60,
        "end_frame": 159,
        "confidence": "high",
    }
    source = trajectory_from_rl_episode_row(
        row, 1, row_index=0, dataset_path=tmp_path / "annotated.lance"
    )
    resampled = resample_timestamped_reference_trajectory(
        source, reference_fps=120
    )
    assert resampled.movement_start_step == 36
    window = window_reference_trajectory(
        resampled, pre_padding=60, post_padding=0
    )

    assert window.movement_start_step == 60
    assert window.pre_edge_hold_steps == 24
    np.testing.assert_array_equal(window.source_indices[:25], 0)
    assert window.identity.source_start == 0


def test_rl_episode_discovery_filters_high_confidence_rows(tmp_path: Path) -> None:
    high = rl_episode_row(frames=160)
    high["trajectory_metadata"]["reference_motion_annotation"] = {
        "contract": REFERENCE_MOTION_ANNOTATION_CONTRACT,
        "start_frame": 60,
        "end_frame": 159,
        "confidence": "high",
    }
    high["trajectory_metadata"]["trajectory_info"] = {
        "object_move": [
            {
                "object_name": "mayonnaisebottle",
                "start_frame": 60,
                "end_frame": 159,
            }
        ]
    }
    low = json.loads(json.dumps(high))
    low["index"]["uuid"] = "low-uuid"
    low["provenance"]["source_rl_uuid"] = "low-uuid"
    low["trajectory_metadata"]["reference_motion_annotation"]["confidence"] = "low"

    class Table:
        def to_pylist(self):
            return [high, low]

    class Dataset:
        def to_table(self, *, columns):
            return Table()

    selection = TrajectorySelection(
        selector="mayonnaisebottle:05",
        dataset_path=tmp_path / "annotated.lance",
        expected_dataset_version=1,
        pre_padding=60,
        post_padding=0,
        hand_side="right",
        reference_fps=120,
    )
    pairs, rows = _discover_rl_episode_candidates(
        Dataset(), selection, confidence="high"
    )
    assert [pair.canonical for pair in pairs] == ["mayonnaisebottle:05"]
    assert len(rows[pairs[0]]) == 1
    assert rows[pairs[0]][0].row_index == 0


def test_rl_episode_discovery_records_observed_clock_and_source_provenance() -> None:
    row = rl_episode_row()
    candidate = _rl_episode_candidate_from_discovery_row(row, row_index=12100)

    assert candidate is not None
    assert candidate.pair.canonical == "mayonnaisebottle:05"
    assert candidate.identity == "mayonnaisebottle_05_12101"
    assert candidate.source_provenance == {
        "contract": RL_EPISODE_REFERENCE_CONTRACT,
        "source_rl_lance_path": "/source/generated_mano.lance",
        "source_rl_version": 13,
        "source_rl_row": 225,
        "source_rl_uuid": "rl-row-uuid",
        "seed_uuid": "seed-uuid",
        "declared_data_fps": 100,
        "observed_median_timestep_seconds": pytest.approx(0.005),
        "observed_median_fps": pytest.approx(200.0),
        "source_frame_count": 5,
        "source_duration_seconds": pytest.approx(0.02),
    }


def test_rl_episode_package_preserves_import_and_row_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixed_asset_profile(monkeypatch)
    source_path = tmp_path / "refined.lance"
    source_path.mkdir()
    row = rl_episode_row()
    trajectory = resample_timestamped_reference_trajectory(
        trajectory_from_rl_episode_row(
            row,
            1,
            row_index=0,
            dataset_path=source_path,
        ),
        reference_fps=120,
    )
    candidate = _rl_episode_candidate_from_discovery_row(row, row_index=0)
    assert candidate is not None
    selection = TrajectorySelection(
        selector="mayonnaisebottle:05",
        dataset_path=source_path,
        expected_dataset_version=1,
        pre_padding=0,
        post_padding=0,
        hand_side="right",
        reference_fps=120,
    )
    output = tmp_path / "mayo.mtp"
    write_trajectory_package(
        output,
        [trajectory],
        selection=selection,
        dataset_schema_digest="schema",
        discovery_digest="discovery",
        compiler={"rl_episode_reference": True},
        source_candidates=[
            {
                "row_index": candidate.row_index,
                "pair": candidate.pair.canonical,
                "identity": candidate.identity,
                "sequence": candidate.sequence,
                "provenance": dict(candidate.source_provenance or {}),
            }
        ],
    )

    catalog = load_trajectory_package(output)
    manifest = catalog.manifest
    assert manifest["reference_resampling"] == RL_EPISODE_RESAMPLING_ID
    assert manifest["selection"]["source_reference_contract"] == RL_EPISODE_REFERENCE_CONTRACT
    source_record = manifest["source_catalog"]["candidates"][0]
    assert source_record["provenance"]["source_rl_row"] == 225
    assert source_record["provenance"]["observed_median_fps"] == pytest.approx(200.0)
