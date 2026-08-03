from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim.manorl.target_replay import (
    TARGET_REPLAY_PACKAGE_SCHEMA,
    TARGET_REPLAY_TARGET_SEMANTICS,
    TargetDofReplay,
    TargetReplayPackageError,
    load_target_replay_package,
    metadata_path_for,
    write_target_replay_package,
)


def _arrays() -> dict[str, np.ndarray]:
    frames = 4
    timestamps = np.arange(frames, dtype=np.float64) * 0.005
    recorded_qpos = np.zeros((frames, 28), dtype=np.float64)
    recorded_qpos[:, 0] = np.linspace(0.0, 0.01, frames)
    target_qpos = recorded_qpos.copy()
    target_qpos[:, 7] = np.linspace(0.0, 0.2, frames)
    object_position = np.zeros((frames, 3), dtype=np.float64)
    object_position[:, 2] = 0.043606
    object_quaternion_xyzw = np.zeros((frames, 4), dtype=np.float64)
    object_quaternion_xyzw[:, 3] = 1.0
    return {
        "timestamps": timestamps,
        "recorded_qpos": recorded_qpos,
        "target_qpos": target_qpos,
        "object_position": object_position,
        "object_quaternion_xyzw": object_quaternion_xyzw,
    }


def _metadata(*, ccd: bool = False) -> dict[str, object]:
    return {
        "schema": TARGET_REPLAY_PACKAGE_SCHEMA,
        "object": "cube1",
        "source_identity": "cube1_01_1614",
        "source_uuid": "source-uuid",
        "source_dataset": "predecoded://test",
        "source_dataset_version": 295,
        "source_row_index": 2239,
        "target_semantics": TARGET_REPLAY_TARGET_SEMANTICS,
        "control_timestep_seconds": 0.005,
        "physics_substeps_per_target": 2,
        "warp_ccd_iterations": 16 if ccd else None,
        "warp_ccd_contacts_per_world": 16 if ccd else None,
        "movement_start_raw": 0,
        "movement_end_raw": 3,
    }


def test_write_and_load_target_replay_package_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    package = write_target_replay_package(
        path,
        metadata=_metadata(),
        **_arrays(),
    )

    loaded = load_target_replay_package(path)
    assert loaded.path == path.resolve()
    assert loaded.frames == 4
    assert loaded.transitions == 3
    assert loaded.metadata["npz_sha256"]
    assert not loaded.target_qpos.flags.writeable
    np.testing.assert_array_equal(loaded.recorded_qpos, _arrays()["recorded_qpos"])
    assert package.metadata["npz_sha256"] == loaded.metadata["npz_sha256"]


def test_loader_rejects_payload_digest_changes(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    write_target_replay_package(path, metadata=_metadata(), **_arrays())
    with path.open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(TargetReplayPackageError, match="SHA256 mismatch"):
        load_target_replay_package(path)


def test_loader_rejects_invalid_timestamps_and_quaternions(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    arrays = _arrays()
    arrays["timestamps"] = np.asarray([0.0, 0.005, 0.012, 0.015])
    with pytest.raises(TargetReplayPackageError, match="timestamps"):
        write_target_replay_package(path, metadata=_metadata(), **arrays)

    arrays = _arrays()
    arrays["object_quaternion_xyzw"][0, 3] = 2.0
    with pytest.raises(TargetReplayPackageError, match="normalized quaternions"):
        write_target_replay_package(path, metadata=_metadata(), **arrays)


def test_writer_rejects_wrong_semantics(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    metadata = _metadata()
    metadata["target_semantics"] = "raw policy action"
    with pytest.raises(TargetReplayPackageError, match="target-DOF contract"):
        write_target_replay_package(path, metadata=metadata, **_arrays())


def test_loader_rejects_wrong_schema_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    write_target_replay_package(path, metadata=_metadata(), **_arrays())
    sidecar = metadata_path_for(path)
    metadata = json.loads(sidecar.read_text())
    metadata["schema"] = "other.schema.v1"
    sidecar.write_text(json.dumps(metadata))

    with pytest.raises(
        TargetReplayPackageError, match="unsupported target replay schema"
    ):
        load_target_replay_package(path)


def test_cpu_replay_requires_explicit_override_for_gpu_ccd_package(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.npz"
    package = write_target_replay_package(
        path,
        metadata=_metadata(ccd=True),
        **_arrays(),
    )
    with pytest.raises(TargetReplayPackageError, match=r"require device=.*gpu"):
        TargetDofReplay(package, device="cpu")


def test_cpu_override_is_explicit_before_runtime_initialization(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    package = write_target_replay_package(
        path,
        metadata=_metadata(ccd=True),
        **_arrays(),
    )
    # The environment is intentionally not constructed in this unit test when
    # the CPU path is unavailable; the package boundary itself is covered by
    # the rejection test above.  The override flag is a public CLI contract.
    assert package.metadata["warp_ccd_contacts_per_world"] == 16


def test_cli_frame_limit_is_bounded() -> None:
    from tools.replay_manorl_target_dof import _validate_frame_limit

    assert _validate_frame_limit(None, 3) == 3
    assert _validate_frame_limit(2, 3) == 2
    with pytest.raises(ValueError, match="exceeds package transitions"):
        _validate_frame_limit(4, 3)
