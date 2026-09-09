from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sim.manorl.checkpoint import CheckpointFormatError, load_skrl_checkpoint, save_skrl_checkpoint
from sim.manorl.contracts import TrajectoryIdentity
from sim.manorl.trajectory import ReferenceTrajectory, TrajectorySelection
from sim.manorl.trajectory_package import (
    TRAJECTORY_PACKAGE_SCHEMA,
    TrajectoryPackageError,
    assign_trajectory_catalog,
    load_trajectory_package,
    write_trajectory_package,
)
from tools.compile_manorl_trajectory_package import _selection_payload
from tools.publish_manorl_trajectory_package import publish


def _trajectory(pair: str, sequence: int, frames: int) -> ReferenceTrajectory:
    object_type, action = pair.split(":")
    identity = f"{object_type}_{action}_{sequence:03d}"
    source_indices = np.arange(10, 10 + frames, dtype=np.int64)
    timestamps = source_indices.astype(np.float64) / 120.0
    base = np.arange(frames * 28, dtype=np.float64).reshape(frames, 28) / 1000.0
    left = base + 1.0
    right = base + 2.0
    object_pos_raw = np.stack(
        [np.linspace(0.0, 0.1, frames), np.zeros(frames), np.ones(frames) * 0.2],
        axis=1,
    )
    object_pos = object_pos_raw.copy()
    object_pos[:, 2] += 0.01
    quaternions = np.zeros((frames, 4), dtype=np.float64)
    quaternions[:, 3] = 1.0
    return ReferenceTrajectory(
        identity=TrajectoryIdentity(
            dataset_path="/dataset/source.lance",
            dataset_version=295,
            row_index=sequence,
            object_index=0,
            uuid=f"uuid-{identity}",
            file_uuid="file-uuid",
            identity=identity,
            source_start=10,
            source_stop=10 + frames,
            movement_start_raw=11,
            movement_end_raw=10 + frames - 1,
        ),
        dataset_version=295,
        source_indices=source_indices,
        timestamps=timestamps,
        q_ref=right,
        object_pos_raw=object_pos_raw,
        object_pos=object_pos,
        object_quat_xyzw=quaternions,
        object_z_shift=0.01,
        hand_sides=("left", "right"),
        q_ref_by_side={"left": left, "right": right},
        selected_hand_sides=("right",),
        reference_fps=120,
        control_fps=120,
        movement_start_step=1,
        movement_end_step=frames - 1,
    )


def _selection(**overrides: object) -> TrajectorySelection:
    values = {
        "selector": "all",
        "dataset_path": Path("/dataset/source.lance"),
        "expected_dataset_version": 295,
        "pre_padding": 180,
        "post_padding": 250,
        "hand_side": "right",
        "reference_fps": 120,
    }
    values.update(overrides)
    return TrajectorySelection(**values)


def test_compiler_selection_payload_preserves_explicit_pairs() -> None:
    selection = _selection(selector="cylinder6:03,cylinder6:09")
    payload = _selection_payload(selection)
    assert payload["selector"] == "cylinder6:03,cylinder6:09"
    assert payload["pre_padding"] == 180
    assert payload["post_padding"] == 250


def _catalog_values() -> tuple[ReferenceTrajectory, ...]:
    return (
        _trajectory("cube1:01", 3, 4),
        _trajectory("cube1:01", 1, 3),
        _trajectory("cube2:04", 8, 5),
        _trajectory("cube2:04", 2, 4),
    )


def test_package_round_trip_is_lance_free_and_mmap_assignable(tmp_path: Path) -> None:
    package = write_trajectory_package(
        tmp_path / "catalog.mtp",
        _catalog_values(),
        selection=_selection(),
        dataset_schema_digest="schema",
        discovery_digest="discovery",
        compiler={"test": True},
    )
    sys.modules.pop("lance", None)
    sys.modules.pop("pyarrow", None)
    catalog = load_trajectory_package(package)
    assert "lance" not in sys.modules
    assert "pyarrow" not in sys.modules
    assert catalog.manifest["schema"] == TRAJECTORY_PACKAGE_SCHEMA
    assert [item.identity.identity for item in catalog.trajectories] == [
        "cube1_01_001",
        "cube1_01_003",
        "cube2_04_002",
        "cube2_04_008",
    ]
    expected = {item.identity.identity: item for item in _catalog_values()}
    for actual in catalog.trajectories:
        source = expected[actual.identity.identity]
        np.testing.assert_array_equal(actual.source_indices, source.source_indices)
        np.testing.assert_array_equal(actual.timestamps, source.timestamps)
        np.testing.assert_array_equal(actual.q_ref_for("left"), source.q_ref_for("left"))
        np.testing.assert_array_equal(actual.q_ref_for("right"), source.q_ref_for("right"))
        np.testing.assert_array_equal(actual.object_pos_raw, source.object_pos_raw)
        np.testing.assert_array_equal(actual.object_pos, source.object_pos)
        np.testing.assert_array_equal(actual.object_quat_xyzw, source.object_quat_xyzw)

    batch = assign_trajectory_catalog(catalog, _selection(), num_envs=10)
    assert batch.num_envs == 10
    assert batch.trajectory_package == catalog.checkpoint_metadata
    assert [item.identity.identity for item in batch.trajectories[:6]] == [
        "cube1_01_001",
        "cube2_04_002",
        "cube1_01_003",
        "cube2_04_008",
        "cube1_01_001",
        "cube2_04_002",
    ]


def test_package_assignment_cycle_rotates_each_pair_slot_window(tmp_path: Path) -> None:
    package = write_trajectory_package(
        tmp_path / "catalog.mtp",
        _catalog_values(),
        selection=_selection(),
        dataset_schema_digest="schema",
        discovery_digest="discovery",
    )
    catalog = load_trajectory_package(package)
    batch = assign_trajectory_catalog(
        catalog,
        _selection(pair_assignment_cycle=1),
        num_envs=2,
    )
    assert [item.identity.identity for item in batch.trajectories] == [
        "cube1_01_003",
        "cube2_04_008",
    ]


def test_package_records_rejections_and_rotates_raw_candidate_order(tmp_path: Path) -> None:
    candidates = [
        {"row_index": 1, "identity": "cube1_01_001", "pair": "cube1:01", "sequence": 1},
        {"row_index": 2, "identity": "cube1_01_002", "pair": "cube1:01", "sequence": 2},
        {"row_index": 3, "identity": "cube1_01_003", "pair": "cube1:01", "sequence": 3},
        {"row_index": 2, "identity": "cube2_04_002", "pair": "cube2:04", "sequence": 2},
        {"row_index": 8, "identity": "cube2_04_008", "pair": "cube2:04", "sequence": 8},
    ]
    rejections = [
        {
            "row_index": 2,
            "identity": "cube1_01_002",
            "pair": "cube1:01",
            "error_type": "ValueError",
            "message": "source timestamps are not strictly increasing",
        }
    ]
    package = write_trajectory_package(
        tmp_path / "catalog.mtp",
        _catalog_values(),
        selection=_selection(),
        dataset_schema_digest="schema",
        discovery_digest="discovery",
        source_candidates=candidates,
        source_rejections=rejections,
    )
    catalog = load_trajectory_package(package)
    assert catalog.manifest["source_catalog"]["candidate_count"] == 5
    assert catalog.manifest["source_catalog"]["rejected_count"] == 1
    batch = assign_trajectory_catalog(
        catalog,
        _selection(pair_assignment_cycle=2),
        num_envs=2,
    )
    assert [item.identity.identity for item in batch.trajectories] == [
        "cube1_01_003",
        "cube2_04_002",
    ]


def test_package_rejects_hash_corruption_and_selection_mismatch(tmp_path: Path) -> None:
    package = write_trajectory_package(
        tmp_path / "catalog.mtp",
        _catalog_values(),
        selection=_selection(),
        dataset_schema_digest="schema",
        discovery_digest="discovery",
    )
    catalog = load_trajectory_package(package)
    with pytest.raises(TrajectoryPackageError, match="selection mismatch"):
        assign_trajectory_catalog(catalog, _selection(reference_fps=100), num_envs=1)

    source_indices = package / "source_indices.npy"
    payload = bytearray(source_indices.read_bytes())
    payload[-1] ^= 1
    source_indices.write_bytes(payload)
    with pytest.raises(TrajectoryPackageError, match="hash mismatch"):
        load_trajectory_package(package)


def test_package_refuses_manifest_or_ready_tampering(tmp_path: Path) -> None:
    package = write_trajectory_package(
        tmp_path / "catalog.mtp",
        _catalog_values(),
        selection=_selection(),
        dataset_schema_digest="schema",
        discovery_digest="discovery",
    )
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["trajectory_count"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(TrajectoryPackageError, match="manifest digest mismatch"):
        load_trajectory_package(package)


def test_publish_exposes_content_addressed_ready_package(tmp_path: Path) -> None:
    source = write_trajectory_package(
        tmp_path / "source.mtp",
        _catalog_values(),
        selection=_selection(),
        dataset_schema_digest="schema",
        discovery_digest="discovery",
    )
    source_catalog = load_trajectory_package(source)
    published = publish(source, tmp_path / "published", name_prefix="catalog")
    assert published.name == f"catalog-{source_catalog.package_digest}"
    assert (published / "READY").read_text().strip() == source_catalog.package_digest
    assert load_trajectory_package(published).catalog_digest == source_catalog.catalog_digest
    with pytest.raises(FileExistsError):
        publish(source, tmp_path / "published", name_prefix="catalog")


def test_training_package_path_never_calls_direct_lance_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tools.train_manorl_cube1 as tool

    sentinel = object()
    package = tmp_path / "catalog.mtp"
    package.mkdir()
    selection = _selection()
    budget = tool.TrainingBudget(
        trajectory_package=str(package),
        dataset_path=str(selection.dataset_path),
        dataset_version=295,
        reference_fps=120,
        hand_side="right",
    )
    monkeypatch.setattr(
        tool,
        "load_assigned_trajectory_package",
        lambda path, selected, *, num_envs: (
            sentinel
            if path == str(package) and selected == selection and num_envs == 8192
            else None
        ),
    )
    monkeypatch.setattr(
        tool,
        "load_assigned_trajectory_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct Lance fallback")),
    )
    assert tool._load_selected_trajectories(selection, budget, num_envs=8192) is sentinel


def test_strict_checkpoint_binds_trajectory_package_digest(tmp_path: Path) -> None:
    class Agent:
        device = "cpu"

        def __init__(self, signature: dict[str, object]) -> None:
            self.manorl_environment_signature = signature
            self.policy = SimpleNamespace(action_dim=28, observation_dim=480, use_film=True)
            self.loaded: str | None = None

        def save(self, path: str) -> None:
            torch.save(
                {
                    "policy": {},
                    "value": {},
                    "optimizer": {},
                    "observation_preprocessor": {},
                    "value_preprocessor": {},
                },
                path,
            )

        def load(self, path: str) -> None:
            self.loaded = path

    package_signature = {
        "action_dim": 28,
        "observation_dim": 480,
        "trajectory_package_schema": TRAJECTORY_PACKAGE_SCHEMA,
        "trajectory_package_digest": "a" * 64,
        "trajectory_package_manifest_sha256": "b" * 64,
        "trajectory_catalog_digest": "c" * 64,
    }
    checkpoint = save_skrl_checkpoint(
        Agent(package_signature),
        tmp_path / "package.pt",
        runtime_config={"model": {"use_film": True}, "environment": package_signature},
    )
    target = Agent(dict(package_signature))
    load_skrl_checkpoint(target, checkpoint)
    assert target.loaded == str(checkpoint)

    mismatched = Agent({**package_signature, "trajectory_catalog_digest": "d" * 64})
    with pytest.raises(CheckpointFormatError, match="trajectory_catalog_digest"):
        load_skrl_checkpoint(mismatched, checkpoint)

    direct_target = Agent({"action_dim": 28, "observation_dim": 480})
    with pytest.raises(CheckpointFormatError, match="trajectory-package target runtime"):
        load_skrl_checkpoint(direct_target, checkpoint)

    direct_checkpoint = save_skrl_checkpoint(
        Agent({"action_dim": 28, "observation_dim": 480}),
        tmp_path / "direct.pt",
        runtime_config={
            "model": {"use_film": True},
            "environment": {"action_dim": 28, "observation_dim": 480},
        },
    )
    with pytest.raises(CheckpointFormatError, match="missing current MuJoCo hand signature"):
        load_skrl_checkpoint(target, direct_checkpoint)


def test_package_preserves_composite_scene_initial_states(tmp_path):
    from dataclasses import replace
    from sim.manorl.trajectory_package import SCENE_TRAJECTORY_PACKAGE_SCHEMA
    source = _trajectory("cube1:01", 1, 3)
    source = replace(
        source,
        scene_object_types=("cube1", "cube2"),
        scene_object_initial_pos=np.array([source.object_pos[0], [.4, .2, .1]]),
        scene_object_initial_quat_xyzw=np.tile([0., 0., 0., 1.], (2, 1)),
    )
    package = write_trajectory_package(
        tmp_path / "scene.mtp", (source,), selection=_selection(),
        dataset_schema_digest="schema", discovery_digest="discovery",
    )
    catalog = load_trajectory_package(package)
    assert catalog.manifest["schema"] == SCENE_TRAJECTORY_PACKAGE_SCHEMA
    actual = catalog.trajectories[0]
    assert actual.scene_object_types == source.scene_object_types
    np.testing.assert_array_equal(actual.scene_object_initial_pos, source.scene_object_initial_pos)
    np.testing.assert_array_equal(actual.scene_object_initial_quat_xyzw, source.scene_object_initial_quat_xyzw)
