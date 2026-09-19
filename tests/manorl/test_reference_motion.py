from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pyarrow as pa
import pytest

from sim.manorl.reference_motion import (
    REFERENCE_MOTION_ANNOTATION_CONTRACT,
    ReferenceMotionConfig,
    annotate_rl_episode_row,
    detect_reference_motion,
)
from tools.annotate_rl_episode_motion import (
    ANNOTATED_RL_LANCE_CONTRACT,
    annotated_schema,
)


def _trajectory(
    *,
    frames: int = 200,
    fps: int = 100,
    motion_start: int | None = 80,
    rotation_only: bool = False,
    no_quiet: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.arange(frames, dtype=np.float64) / fps
    positions = np.zeros((frames, 3), dtype=np.float64)
    rotations = np.zeros((frames, 3), dtype=np.float64)
    # A short gravity-settling transient must not become task-motion onset.
    positions[:11, 2] = np.linspace(0.006, 0.0, 11)
    if no_quiet:
        positions[:, 1] += 0.0015 * np.sin(np.arange(frames) * 1.7)
        rotations[:, 1] += 0.02 * np.sin(np.arange(frames) * 1.3)
    if motion_start is not None:
        ramp = np.linspace(0.0, 0.06, frames - motion_start)
        if rotation_only:
            rotations[motion_start:, 2] += ramp * 8.0
        else:
            positions[motion_start:, 0] += ramp
    return positions, rotations, timestamps


def _row() -> dict[str, object]:
    positions, rotations, timestamps = _trajectory()
    frames = len(timestamps)
    return {
        "index": {
            "uuid": "row-uuid",
            "seed_uuid": "seed-uuid",
            "scene": "mayonnaisebottle",
            "action_code": "04",
            "gesture": "04",
            "is_generated": True,
        },
        "trajectory_metadata": {
            "hand_names": ["right"],
            "hand_slots": ["right", "left"],
            "gesture": "04",
            "data_fps": 100,
            "total_frames": frames,
        },
        "timestamp": timestamps.tolist(),
        "hands": [
            {
                "hand_name": "right",
                "urdf_dof": np.zeros((frames, 28), dtype=np.float32).tolist(),
            }
        ],
        "objects": [
            {"pos": positions.astype(np.float32).tolist(), "rot_aa": rotations.astype(np.float32).tolist()}
        ],
        "provenance": {
            "source_rl_lance_path": "/source.lance",
            "source_rl_version": 1,
            "source_rl_row": 2,
            "source_rl_uuid": "row-uuid",
        },
    }


def test_detector_ignores_settling_and_finds_translation_onset() -> None:
    positions, rotations, timestamps = _trajectory(motion_start=80)
    result = detect_reference_motion(positions, rotations, timestamps)

    assert result.status == "high"
    assert result.confidence == "high"
    assert result.motion_mode == "translation"
    assert 78 <= result.start_frame <= 82
    assert result.stable_start_frame is not None
    assert result.end_frame == 199


def test_detector_finds_rotation_only_onset() -> None:
    positions, rotations, timestamps = _trajectory(
        motion_start=60, rotation_only=True
    )
    result = detect_reference_motion(positions, rotations, timestamps)

    assert result.status == "high"
    assert result.motion_mode == "rotation"
    assert 58 <= result.start_frame <= 62


def test_detector_labels_settling_only_row_explicitly() -> None:
    positions, rotations, timestamps = _trajectory(motion_start=None)
    result = detect_reference_motion(positions, rotations, timestamps)

    assert result.status == "initial_transient_only"
    assert result.confidence == "low"
    assert result.start_frame == 0
    assert result.motion_mode in {"translation", "none"}


def test_detector_uses_low_confidence_fallback_without_quiet_window() -> None:
    positions, rotations, timestamps = _trajectory(motion_start=90, no_quiet=True)
    config = ReferenceMotionConfig(
        quiet_translation_speed_m_s=0.001,
        quiet_rotation_speed_rad_s=0.01,
    )
    result = detect_reference_motion(
        positions, rotations, timestamps, config=config
    )

    assert result.status in {"low_no_quiet", "low_unconfirmed"}
    assert result.confidence == "low"
    assert result.start_frame > 0


def test_row_annotation_preserves_source_and_adds_standard_movement() -> None:
    source = _row()
    original = deepcopy(source)
    annotated, result = annotate_rl_episode_row(source)

    assert source == original
    assert annotated["hands"] == source["hands"]
    assert annotated["objects"] == source["objects"]
    metadata = annotated["trajectory_metadata"]
    movement = metadata["trajectory_info"]["object_move"]
    assert movement == [
        {
            "object_name": "mayonnaisebottle",
            "start_frame": result.start_frame,
            "end_frame": len(source["timestamp"]) - 1,
        }
    ]
    assert metadata["reference_motion_annotation"]["contract"] == (
        REFERENCE_MOTION_ANNOTATION_CONTRACT
    )


def test_annotated_schema_preserves_fields_and_declares_contract() -> None:
    source = pa.schema(
        [
            pa.field(
                "index",
                pa.struct(
                    [
                        pa.field("uuid", pa.string()),
                        pa.field("scene", pa.string()),
                    ]
                ),
            ),
            pa.field(
                "trajectory_metadata",
                pa.struct(
                    [
                        pa.field("gesture", pa.string()),
                        pa.field("total_frames", pa.int64()),
                    ]
                ),
            ),
            pa.field("timestamp", pa.list_(pa.float64())),
        ]
    )
    result = annotated_schema(source, ReferenceMotionConfig())

    assert result.names == source.names
    metadata_type = result.field("trajectory_metadata").type
    assert [field.name for field in metadata_type] == [
        "gesture",
        "total_frames",
        "trajectory_info",
        "reference_motion_annotation",
    ]
    assert result.metadata[b"manorl:annotation_contract"] == (
        ANNOTATED_RL_LANCE_CONTRACT.encode("ascii")
    )


def test_row_annotation_rejects_existing_movement_contract() -> None:
    source = _row()
    source["trajectory_metadata"]["trajectory_info"] = {"object_move": []}
    with pytest.raises(ValueError, match="already carries trajectory_info"):
        annotate_rl_episode_row(source)
