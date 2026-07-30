from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.environment import MaterializedContactBuffers, MaterializedState
from sim.manorl.contracts import TrajectoryIdentity
from sim.manorl.lance_v2 import (
    FORCE_DIRECTION_CONTRACT,
    MANO_GLOBAL_FRAME_CONTRACT,
    SYNTHETIC_LANCE_V22_CONTRACT,
    build_v2_row,
    build_v2_schema,
    corrected_contact_frames,
    write_v2_lance,
)
from sim.manorl.mano_pose import right_urdf_trajectory_to_mano_48d
from sim.manorl.trajectory import ReferenceTrajectory, TrajectoryBatch, TrajectorySelection
import tools.export_manorl_synthetic_lance as exporter_module
from tools.export_manorl_synthetic_lance import (
    _load_predecoded_batch,
    export_checkpoint_rollouts,
)
from tools.validate_manorl_synthetic_lance import (
    _is_retryable_nested_decode_failure,
)


def _state() -> MaterializedState:
    body_count = 4
    xpos = np.zeros((1, body_count, 3), dtype=np.float64)
    xpos[0, 1] = (0.1, 0.2, 0.3)
    xpos[0, 2] = (0.5, 0.5, 0.5)
    xpos[0, 3] = (0.0, 0.0, 0.0)
    xquat = np.zeros((1, body_count, 4), dtype=np.float64)
    xquat[..., 0] = 1.0
    return MaterializedState(
        qpos=np.zeros((1, 28), dtype=np.float64),
        qvel=np.zeros((1, 28), dtype=np.float64),
        xpos=xpos,
        xquat=xquat,
        keypoints=np.zeros((1, 16, 3), dtype=np.float64),
        keypoint_quats=np.broadcast_to(np.asarray([0.0, 0.0, 0.0, 1.0]), (1, 16, 4)).copy(),
        fingertips=np.zeros((1, 5, 3), dtype=np.float64),
    )


def _buffers() -> MaterializedContactBuffers:
    return MaterializedContactBuffers(
        count=1,
        capacity=1,
        geom=np.asarray([[0, 16]], dtype=np.int64),
        position=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        world=np.asarray([0], dtype=np.int64),
        dimension=np.asarray([3], dtype=np.int64),
        addresses=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        nefc=np.asarray([4], dtype=np.int64),
        friction=np.zeros((1, 5), dtype=np.float64),
        frame=np.broadcast_to(np.eye(3), (1, 3, 3)).copy(),
        constraint_force=np.asarray([[1.0, 2.0, 0.0, 0.0]], dtype=np.float64),
        raw_metadata={},
        host_metadata={},
    )


def test_nested_arrow_decode_exceptions_are_retryable() -> None:
    pyarrow_error = (
        "rows = dataset.take([row_index]).to_pylist()\n"
        "TypeError: pyarrow.lib.FloatScalar.__new__(X): X is not a type object"
    )
    assert _is_retryable_nested_decode_failure(1, pyarrow_error)
    assert _is_retryable_nested_decode_failure(139, "segmentation fault")
    assert not _is_retryable_nested_decode_failure(1, "ValueError: schema changed")


def test_v2_normal_force_has_no_legacy_half_scale_and_uses_actual_frames() -> None:
    state = _state()
    model = SimpleNamespace(geom_bodyid=np.asarray([1] + [0] * 15 + [2], dtype=np.int64))
    entries = corrected_contact_frames(
        buffers=_buffers(),
        state=state,
        model=model,
        keypoint_geom_ids=tuple(range(16)),
        object_geom_ids=(16,),
        object_body_id=2,
        wrist_body_id=3,
        object_name="cube2",
    )[0]
    assert len(entries) == 1
    entry = entries[0]
    # sum([1, 2, 0, 0]) = 3; v1's 0.5 scale would incorrectly produce 1.5.
    np.testing.assert_allclose(entry["total_force_world"], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(entry["total_force_joint"], entry["total_force_world"])
    np.testing.assert_allclose(entry["total_force_object"], entry["total_force_world"])
    np.testing.assert_allclose(entry["contact_pairs"][0]["pos_joint"], [0.9, 1.8, 2.7])
    np.testing.assert_allclose(entry["contact_pairs"][0]["pos_object"], [0.5, 1.5, 2.5])
    assert FORCE_DIRECTION_CONTRACT.endswith("scale_1p0")


def test_mano_48d_conversion_matches_source_layout() -> None:
    values = np.zeros((2, 28), dtype=np.float64)
    values[1, 6] = 0.2
    converted = right_urdf_trajectory_to_mano_48d(values)
    assert converted.shape == (2, 48)
    np.testing.assert_array_equal(converted[:, :3], 0.0)
    assert np.linalg.norm(converted[1, 3:]) > 0.0


def test_v2_schema_has_explicit_contract_and_28d_rollout_fields() -> None:
    schema = build_v2_schema(observation_dim=480, action_dim=28)
    assert schema.metadata[b"schema_version"] == SYNTHETIC_LANCE_V22_CONTRACT.encode()
    assert schema.metadata[b"mano_global_frame_contract"] == MANO_GLOBAL_FRAME_CONTRACT.encode()
    assert schema.field("trajectory_metadata").type[0].name == "data_fps"
    assert schema.field("hands").type.value_type[6].name == "urdf_dof_target"
    assert schema.field("rollout").type[0].name == "transition_count"
    assert schema.field("provenance").type[5].name == "checkpoint_update"
    assert schema.field("provenance").type[-1].name == "generation_attempt"


def test_v2_row_requires_complete_t_and_t_minus_one_alignment() -> None:
    identity = TrajectoryIdentity(
        dataset_path="/source.lance", dataset_version=295, row_index=3, object_index=0,
        uuid="499dab41-1e12-4595-a040-c5ec979fc3bb", file_uuid="f",
        identity="cube2_02_0004", source_start=10, source_stop=12,
        movement_start_raw=10, movement_end_raw=11,
    )
    trajectory = ReferenceTrajectory(
        identity=identity, dataset_version=295, source_indices=np.asarray([10, 11]),
        timestamps=np.asarray([0.0, 0.005]), q_ref=np.zeros((2, 28)),
        object_pos_raw=np.zeros((2, 3)), object_pos=np.zeros((2, 3)),
        object_quat_xyzw=np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2), object_z_shift=0.0,
    )
    urdf_dof = np.zeros((2, 28), dtype=np.float64)
    urdf_dof[:, :3] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    urdf_dof[:, 3:6] = [[0.2, -0.3, 0.4], [-0.5, 0.1, 0.7]]
    states = {
        # Deliberately disagree with the URDF root; MANO globals must ignore
        # physical wrist-body pose and use the refined floating-root contract.
        "hand_position": np.full((2, 3), 99.0),
        "hand_orientation_xyzw": np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2),
        "mano_joint_pos": np.zeros((2, 21, 3)), "urdf_dof": urdf_dof,
        "urdf_dof_target": np.zeros((2, 28)), "object_position": np.zeros((2, 3)),
        "object_orientation_xyzw": np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2),
    }
    rollout = {
        "observation_t": np.zeros((1, 480)), "next_observation": np.zeros((1, 480)),
        "policy_mean_action": np.zeros((1, 28)), "processed_action": np.zeros((1, 28)),
        "cumulative_position_residual": np.zeros((1, 3)), "cumulative_joint_residual": np.zeros((1, 22)),
        "command_reference_index": np.asarray([0]), "command_source_frame_index": np.asarray([10]),
        "reference_target": np.zeros((1, 28)), "processed_target": np.zeros((1, 28)),
        "controller_target": np.zeros((1, 28)), "reward": np.asarray([1.0]),
        "raw_contact_reward": np.asarray([0.4]), "contact_reward": np.asarray([0.4]),
        "terminated": np.asarray([True]), "termination_reason_code": np.asarray([1]),
    }
    row = build_v2_row(
        trajectory=trajectory, source_index={},
        source_metadata={
            "hand_names": ["left", "right"],
            "mano_hand_shapes": [[1.0] * 10, [2.0] * 10],
        },
        states=states,
        contacts=[[], []], rollout=rollout,
        provenance={"checkpoint_path": "/c.pt", "checkpoint_sha256": "abc", "checkpoint_update": 1, "checkpoint_metadata": {}, "software_commit": "deadbeef", "seed": 42, "episode_index": 0, "generation_attempt": 1},
    )
    assert row["trajectory_metadata"]["data_fps"] == 200
    assert row["trajectory_metadata"]["total_frames"] == 2
    assert row["rollout"]["transition_count"] == 1
    assert row["provenance"]["force_contract"] == FORCE_DIRECTION_CONTRACT
    assert row["provenance"]["seed"] == 42
    assert row["provenance"]["episode_index"] == 0
    assert row["provenance"]["generation_attempt"] == 1
    np.testing.assert_array_equal(
        row["hands"][0]["mano_global_pos"],
        np.asarray(row["hands"][0]["urdf_dof"])[:, :3],
    )
    expected_rot = Rotation.from_euler(
        "XYZ", np.asarray(row["hands"][0]["urdf_dof"])[:, 3:6]
    )
    stored_rot = Rotation.from_rotvec(row["hands"][0]["mano_global_rot_aa"])
    np.testing.assert_allclose(
        (stored_rot.inv() * expected_rot).magnitude(), 0.0, atol=1e-6
    )
    assert row["trajectory_metadata"]["hand_names"] == ["right"]
    np.testing.assert_array_equal(
        row["trajectory_metadata"]["mano_hand_shapes"], [[2.0] * 10]
    )


def test_predecoded_manifest_selects_unique_hashed_identity_window(tmp_path) -> None:
    dataset = tmp_path / "source.lance"
    dataset.mkdir()
    records = []
    for sequence in (1, 2):
        identity = TrajectoryIdentity(
            dataset_path=str(dataset), dataset_version=295, row_index=sequence,
            object_index=0, uuid=f"00000000-0000-0000-0000-00000000000{sequence}",
            file_uuid="f", identity=f"cube2_02_{sequence:04d}", source_start=0,
            source_stop=2, movement_start_raw=0, movement_end_raw=1,
        )
        trajectory = ReferenceTrajectory(
            identity=identity, dataset_version=295, source_indices=np.asarray([0, 1]),
            timestamps=np.asarray([0.0, 0.005]), q_ref=np.zeros((2, 28)),
            object_pos_raw=np.zeros((2, 3)), object_pos=np.zeros((2, 3)),
            object_quat_xyzw=np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2), object_z_shift=0.0,
        )
        path = tmp_path / f"{identity.identity}.pkl"
        path.write_bytes(pickle.dumps(trajectory))
        from sim.manorl.lance_v2 import file_sha256
        records.append({"identity": identity.identity, "pair": "cube2:02", "pickle_sha256": file_sha256(path)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset_path": str(dataset), "dataset_version": 295, "hand_side": "right", "valid_records": records}))
    batch = _load_predecoded_batch(
        TrajectorySelection("cube2", "02", dataset_path=dataset, expected_dataset_version=295, hand_side="right", pair_assignment_cycle=1),
        num_envs=1, manifest_path=manifest,
    )
    assert [item.identity.identity for item in batch.trajectories] == ["cube2_02_0002"]


def test_repeated_synthesis_isolates_five_attempt_rounds(
    tmp_path, monkeypatch
) -> None:
    identity = TrajectoryIdentity(
        dataset_path=str(tmp_path / "source.lance"), dataset_version=295,
        row_index=3, object_index=0,
        uuid="499dab41-1e12-4595-a040-c5ec979fc3bb", file_uuid="f",
        identity="cube2_02_0004", source_start=0, source_stop=2,
        movement_start_raw=0, movement_end_raw=1,
    )
    trajectory = ReferenceTrajectory(
        identity=identity, dataset_version=295, source_indices=np.asarray([0, 1]),
        timestamps=np.asarray([0.0, 0.005]), q_ref=np.zeros((2, 28)),
        object_pos_raw=np.zeros((2, 3)), object_pos=np.zeros((2, 3)),
        object_quat_xyzw=np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2), object_z_shift=0.0,
    )
    batch = TrajectoryBatch((trajectory,))
    checkpoint = tmp_path / "checkpoint-000500.pt"
    observed_controls: list[dict[str, object]] = []

    monkeypatch.setattr(exporter_module, "_validate_checkpoint_path", lambda path: path)
    monkeypatch.setattr(
        exporter_module, "load_assigned_trajectory_batch", lambda selection, num_envs: batch
    )
    monkeypatch.setattr(exporter_module, "file_sha256", lambda path: "checkpoint-sha")
    monkeypatch.setattr(exporter_module, "checkpoint_runtime_metadata", lambda path: {})
    monkeypatch.setattr(exporter_module, "_software_commit", lambda: "commit")

    def fake_child(command, check):
        output = Path(command[command.index("--output") + 1])
        control_path = Path(command[command.index("--internal-attempt-control") + 1])
        control = json.loads(control_path.read_text())
        observed_controls.append(control)
        output.mkdir(parents=True, exist_ok=True)
        name = identity.identity
        episode = control["episode_indices"][name]
        child_manifest = output.parent / f"{output.name}.manifest.json"
        child_manifest.write_text(
            json.dumps(
                {
                    "synthesis": {
                        "counters": {
                            name: {"attempts": 1, "saved": 1, "failures": []}
                        }
                    },
                    "generated_uuids": [f"generated-{episode}"],
                    "row_source_identities": [name],
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(exporter_module.subprocess, "run", fake_child)
    monkeypatch.setitem(
        sys.modules,
        "lance",
        SimpleNamespace(dataset=lambda path: SimpleNamespace(count_rows=lambda: 5)),
    )
    output = tmp_path / "repeated.lance"
    result = export_checkpoint_rollouts(
        checkpoint=checkpoint,
        output=output,
        selection=TrajectorySelection(
            "cube2", "02", dataset_path=tmp_path / "source.lance",
            expected_dataset_version=295, hand_side="right",
        ),
        num_envs=1,
        device="cpu",
        episodes_per_identity=5,
        max_attempts_per_identity=10,
    )

    assert result["rows"] == 5
    assert output.is_dir()
    assert [control["attempt_seed"] for control in observed_controls] == [42, 43, 44, 45, 46]
    assert [control["episode_indices"][identity.identity] for control in observed_controls] == list(range(5))
    assert [control["attempt_numbers"][identity.identity] for control in observed_controls] == [1, 2, 3, 4, 5]
    manifest = json.loads((tmp_path / "repeated.lance.manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["rows"] == 5
    assert manifest["synthesis"]["attempt_isolation"] == "one_fresh_process_per_attempt_round"
    assert manifest["synthesis"]["counters"][identity.identity]["attempts"] == 5
    assert manifest["synthesis"]["counters"][identity.identity]["saved"] == 5

def test_v2_writer_round_trip_preserves_nested_contract(tmp_path) -> None:
    # Keep this writer test deliberately small; GPU rollout tests validate the
    # full row builder and contact invariants separately.
    output = tmp_path / "synthetic_v22.lance"
    row = {
        "index": {"uuid": "u", "seed_uuid": "s", "capMachine": "m", "operator": "o", "scene": "cube2", "is_generated": True},
        "trajectory_metadata": {
            "data_fps": 200, "total_frames": 1, "gesture": "02", "hand_names": ["right"], "hand_slots": ["right", "left"], "object_names": ["cube2"], "mano_hand_shapes": [[0.0] * 10],
            "raw_data_info": {"capMachine": "m", "operator": "o", "scene": "cube2", "id": 1},
            "trajectory_info": {"object_move": [{"object_name": "cube2", "start_frame": 0, "end_frame": 0}]}, "capture_info": None, "train_info": {"commit_hash": "c", "reward_value": 0.0},
        },
        "timestamp": [0.0],
        "hands": [{"hand_name": "right", "mano_global_pos": [[0.0] * 3], "mano_global_rot_aa": [[0.0] * 3], "mano_hand_pose": [[0.0] * 48], "mano_joint_pos": [[[0.0] * 3] * 21], "urdf_dof": [[0.0] * 28], "urdf_dof_target": [[0.0] * 28]}, {"hand_name": None, "mano_global_pos": [], "mano_global_rot_aa": [], "mano_hand_pose": [], "mano_joint_pos": [], "urdf_dof": [], "urdf_dof_target": []}],
        "objects": [{"rot_aa": [[0.0] * 3], "pos": [[0.0] * 3]}], "contact": [[]],
        "reference": {"source_frame_index": [0], "hand_urdf_dof": [[0.0] * 28], "object_pos": [[0.0] * 3], "object_rot_aa": [[0.0] * 3]},
        "rollout": {"transition_count": 0, "observation_t": [], "next_observation": [], "policy_mean_action": [], "processed_action": [], "cumulative_position_residual": [], "cumulative_joint_residual": [], "command_reference_index": [], "command_source_frame_index": [], "reference_target": [], "processed_target": [], "controller_target": [], "reward": [], "raw_contact_reward": [], "contact_reward": [], "terminated": [], "termination_reason_code": []},
        "provenance": {"contract": SYNTHETIC_LANCE_V22_CONTRACT, "force_contract": FORCE_DIRECTION_CONTRACT, "policy_mode": "deterministic_mean", "checkpoint_path": "p", "checkpoint_sha256": "h", "checkpoint_update": 1, "checkpoint_metadata_json": "{}", "dataset_path": "d", "dataset_version": 1, "row_index": 0, "source_identity": "id", "software_commit": "c", "seed": 42, "episode_index": 0, "generation_attempt": 1},
    }
    write_v2_lance([row], output=output, observation_dim=480, action_dim=28)
    import lance
    dataset = lance.dataset(str(output))
    assert dataset.count_rows() == 1
    assert dataset.schema.metadata[b"schema_version"] == SYNTHETIC_LANCE_V22_CONTRACT.encode()
    assert dataset.take([0]).to_pylist()[0]["trajectory_metadata"]["data_fps"] == 200
