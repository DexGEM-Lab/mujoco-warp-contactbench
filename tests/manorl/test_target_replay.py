from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from sim.manorl.target_replay import (
    LANCE_TARGET_REPLAY_COLUMNS,
    TARGET_REPLAY_ROW_CONTRACT,
    TargetDofReplay,
    TargetReplaySourceError,
    load_target_replay_source,
    target_replay_source_from_row,
)


def _row() -> dict[str, object]:
    frames = 4
    recorded_qpos = np.zeros((frames, 28), dtype=np.float64)
    recorded_qpos[:, 0] = np.linspace(0.0, 0.01, frames)
    target_qpos = recorded_qpos.copy()
    target_qpos[:, 7] = np.linspace(0.0, 0.2, frames)
    object_position = np.zeros((frames, 3), dtype=np.float64)
    object_position[:, 2] = 0.043606
    return {
        "index": {
            "uuid": "generated-uuid",
            "seed_uuid": "source-uuid",
            "scene": "cube1",
            "is_generated": True,
        },
        "trajectory_metadata": {
            "total_frames": frames,
            "data_fps": 200.0,
            "hand_names": ["right"],
            "hand_slots": ["right", "left"],
            "object_names": ["cube1"],
            "trajectory_info": {
                "object_move": [
                    {"object_name": "cube1", "start_frame": 1, "end_frame": 2}
                ]
            },
        },
        "timestamp": (np.arange(frames, dtype=np.float64) * 0.005).tolist(),
        "hands": [
            {
                "hand_name": "right",
                "urdf_dof": recorded_qpos.tolist(),
                "urdf_dof_target": target_qpos.tolist(),
            },
            {
                "hand_name": None,
                "urdf_dof": [],
                "urdf_dof_target": [],
            },
        ],
        "objects": [
            {
                "pos": object_position.tolist(),
                "rot_aa": np.zeros((frames, 3), dtype=np.float64).tolist(),
            }
        ],
        "provenance": {
            "contract": TARGET_REPLAY_ROW_CONTRACT,
            "dataset_path": "/source/guangguan.lance",
            "dataset_version": 295,
            "row_index": 1613,
            "source_identity": "cube1_01_1614",
            "checkpoint_update": 2200,
            "checkpoint_sha256": "a" * 64,
            "checkpoint_metadata_json": json.dumps(
                {
                    "runtime_config": {
                        "environment": {
                            "warp_ccd": {
                                "ccd_iterations": 16,
                                "contacts_per_world": 16,
                            }
                        }
                    }
                }
            ),
        },
    }


def _source(row: dict[str, object] | None = None):
    return target_replay_source_from_row(
        _row() if row is None else row,
        dataset_path="/generated/merged.lance",
        dataset_version=1,
        row_index=2239,
    )


def test_direct_row_preserves_generated_and_source_lineage() -> None:
    source = _source()

    assert source.dataset_path == Path("/generated/merged.lance")
    assert source.dataset_version == 1
    assert source.row_index == 2239
    assert source.generated_uuid == "generated-uuid"
    assert source.source_dataset_path == "/source/guangguan.lance"
    assert source.source_dataset_version == 295
    assert source.source_row_index == 1613
    assert source.source_uuid == "source-uuid"
    assert source.source_identity == "cube1_01_1614"
    assert source.frames == 4 and source.transitions == 3
    assert source.movement_start == 1 and source.movement_end == 2
    assert source.warp_ccd_iterations == 16
    assert source.warp_ccd_contacts_per_world == 16
    assert not source.target_qpos.flags.writeable


def test_direct_loader_requests_only_replay_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeTable:
        def to_pylist(self):
            return [_row()]

    class FakeDataset:
        def take(self, indices, *, columns):
            captured["indices"] = indices
            captured["columns"] = columns
            return FakeTable()

    def dataset(path, *, version):
        captured["path"] = path
        captured["version"] = version
        return FakeDataset()

    monkeypatch.setitem(sys.modules, "lance", SimpleNamespace(dataset=dataset))
    source = load_target_replay_source(
        "/generated/merged.lance", dataset_version=1, row_index=2239
    )

    assert source.source_identity == "cube1_01_1614"
    assert captured == {
        "path": "/generated/merged.lance",
        "version": 1,
        "indices": [2239],
        "columns": list(LANCE_TARGET_REPLAY_COLUMNS),
    }


def test_row_rejects_invalid_timestamps() -> None:
    row = _row()
    row["timestamp"] = [0.0, 0.005, 0.012, 0.015]
    with pytest.raises(TargetReplaySourceError, match="timestamps"):
        _source(row)


def test_row_rejects_invalid_target_shape() -> None:
    row = _row()
    row["hands"][0]["urdf_dof_target"] = [[0.0] * 27 for _ in range(4)]
    with pytest.raises(TargetReplaySourceError, match="urdf_dof_target shape"):
        _source(row)


def test_row_rejects_wrong_contract() -> None:
    row = _row()
    row["provenance"]["contract"] = "different_contract"
    with pytest.raises(TargetReplaySourceError, match="unsupported target replay"):
        _source(row)


def test_row_rejects_missing_right_hand() -> None:
    row = _row()
    row["trajectory_metadata"]["hand_slots"] = ["left", "unused"]
    row["trajectory_metadata"]["hand_names"] = ["left"]
    with pytest.raises(TargetReplaySourceError, match="right-hand slot"):
        _source(row)


def test_row_rejects_partial_warp_ccd_contract() -> None:
    row = _row()
    checkpoint = json.loads(row["provenance"]["checkpoint_metadata_json"])
    del checkpoint["runtime_config"]["environment"]["warp_ccd"]["contacts_per_world"]
    row["provenance"]["checkpoint_metadata_json"] = json.dumps(checkpoint)
    with pytest.raises(TargetReplaySourceError, match="must provide both"):
        _source(row)


def test_cpu_replay_requires_override_for_row_ccd() -> None:
    with pytest.raises(TargetReplaySourceError, match=r"require device=.*gpu"):
        TargetDofReplay(_source(), device="cpu")


def test_cli_numeric_and_frame_validation() -> None:
    from tools.replay_manorl_target_dof import _positive_float, _validate_frame_limit

    assert _positive_float("0.25") == 0.25
    for value in ("nan", "inf", "-1", "0"):
        with pytest.raises(argparse.ArgumentTypeError):
            _positive_float(value)
    assert _validate_frame_limit(None, 3) == 3
    assert _validate_frame_limit(2, 3) == 2
    with pytest.raises(ValueError, match="exceeds row transitions"):
        _validate_frame_limit(4, 3)
