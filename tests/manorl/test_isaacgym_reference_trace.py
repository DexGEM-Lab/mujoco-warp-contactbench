from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[2] / "tools" / "isaacgym_reference_trace.py"
_SPEC = importlib.util.spec_from_file_location("isaacgym_reference_trace", MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
trace = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = trace
_SPEC.loader.exec_module(trace)


def exact_entry() -> dict[str, object]:
    return {
        "dataset_path": str(trace.DATASET_PATH),
        "row_index": 1,
        "object_index": 0,
        "uuid": trace.UUID,
        "file_uuid": trace.FILE_UUID,
        "trajectory_name": trace.TRAJECTORY_NAME,
        "object_type": trace.OBJECT_TYPE,
        "action_id": "02",
        "sequence_id": "002",
    }


def exact_row() -> dict[str, object]:
    return {
        "index": {
            "uuid": trace.UUID,
            "file_uuid": trace.FILE_UUID,
            "gesture": "002-take",
        },
        "trajectory_metadata": {
            "object_names": ["powerdrill"],
            "raw_data_info": {"id": 1},
            "hand_names": ["right"],
            "total_frames": 605,
            "data_fps": 100,
            "trajectory_info": {
                "object_move": [
                    {
                        "object_name": "powerdrill",
                        "start_frame": 260,
                        "end_frame": 444,
                    }
                ]
            },
        },
    }


def valid_metadata() -> dict[str, object]:
    return {
        "schema": trace.SCHEMA,
        "trajectory_identity": {
            "dataset_path": str(trace.DATASET_PATH),
            "row_index": 1,
            "object_index": 0,
            "uuid": trace.UUID,
            "file_uuid": trace.FILE_UUID,
            "identity": trace.TRAJECTORY_NAME,
            "dataset_version": 3325,
            "source_slice": {"start": 10, "stop": 604, "stop_exclusive": True},
            "reference_frame_count": 594,
            "physical_call_count": 593,
        },
        "lance_index_entry": {},
        "source_repository": {},
        "config": {},
        "checks": {},
        "action_invariance": {},
        "metrics": {},
        "non_comparable_fields": [],
        "claims": {"isaac_parity": "trace_generated_not_yet_compared"},
    }


def valid_arrays() -> dict[str, np.ndarray]:
    return {
        name: np.zeros((trace.PHYSICS_CALLS, *tail), dtype=dtype)
        for name, (tail, dtype) in trace.REQUIRED_ARRAY_SPECS.items()
    }


def test_module_import_does_not_import_runtime_dependencies() -> None:
    assert "isaacgym" not in sys.modules
    assert "torch" not in sys.modules
    assert "hydra" not in sys.modules


def test_protected_source_hash_paths_are_correct_and_state_capture_is_read_only(
    tmp_path: Path,
) -> None:
    expected_paths = (
        "IsaacGymEnvs/isaacgymenvs/tasks/mano_hand.py",
        "IsaacGymEnvs/isaacgymenvs/tasks/base/vec_task.py",
        "IsaacGymEnvs/isaacgymenvs/utils/lance_mocap_backend.py",
        "IsaacGymEnvs/isaacgymenvs/cfg/task/MANOHand.yaml",
        "IsaacGymEnvs/isaacgymenvs/cfg/config.yaml",
    )
    assert trace.SOURCE_HASH_PATHS == expected_paths
    source = tmp_path / "source"
    for relative in expected_paths:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "add", *expected_paths], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-q", "-m", "fixture"], check=True
    )
    before = trace.capture_source_state(source)
    assert set(before["hashes"]) == set(expected_paths)
    assert all(len(digest) == 64 for digest in before["hashes"].values())
    assert trace.capture_source_state(source) == before


def test_strict_selector_accepts_the_sole_exact_match() -> None:
    decoy = exact_entry()
    decoy["uuid"] = "object-action-only-decoy"
    selected = trace.select_strict_index_entry(
        [decoy, exact_entry()], trace.LANCE_BASE_PATH, ["powerdrill"], [2]
    )
    assert selected == exact_entry()
    assert trace.normalize_action_ids("2,02") == ("02", "02")


@pytest.mark.parametrize(
    ("field", "defect"),
    [
        ("dataset_path", "/tmp/wrong.lance"),
        ("row_index", 2),
        ("object_index", 1),
        ("uuid", "wrong-uuid"),
        ("file_uuid", "wrong-file-uuid"),
        ("trajectory_name", "powerdrill_02_003"),
        ("object_type", "hammer"),
        ("action_id", "03"),
        ("sequence_id", "003"),
    ],
)
def test_strict_selector_rejects_each_identity_defect(field: str, defect: object) -> None:
    entry = exact_entry()
    entry[field] = defect
    with pytest.raises(RuntimeError, match="found 0"):
        trace.select_strict_index_entry(
            [entry], trace.LANCE_BASE_PATH, ["powerdrill"], ["02"]
        )


def test_strict_selector_rejects_duplicates_and_wrong_filters() -> None:
    with pytest.raises(RuntimeError, match="found 2"):
        trace.select_strict_index_entry(
            [exact_entry(), exact_entry()],
            trace.LANCE_BASE_PATH,
            ["powerdrill"],
            ["02"],
        )
    with pytest.raises(ValueError, match="object_types"):
        trace.select_strict_index_entry(
            [exact_entry()], trace.LANCE_BASE_PATH, ["hammer"], ["02"]
        )
    with pytest.raises(ValueError, match="action filter"):
        trace.select_strict_index_entry(
            [exact_entry()], trace.LANCE_BASE_PATH, ["powerdrill"], ["03"]
        )


def test_row_identity_derivation_and_validation() -> None:
    row = exact_row()
    assert trace.derive_row_identity(row) == "powerdrill_02_002"
    assert trace.validate_row_identity(row) == "powerdrill_02_002"

    altered_id = copy.deepcopy(row)
    altered_id["trajectory_metadata"]["raw_data_info"]["id"] = 2
    with pytest.raises(ValueError, match="row identity mismatch"):
        trace.validate_row_identity(altered_id)

    altered_gesture = copy.deepcopy(row)
    altered_gesture["index"]["gesture"] = "003-take"
    with pytest.raises(ValueError, match="row identity mismatch"):
        trace.validate_row_identity(altered_gesture)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("index", "uuid"), "wrong", "UUID"),
        (("index", "file_uuid"), "wrong", "file UUID"),
        (("trajectory_metadata", "hand_names"), ["left"], "right hand"),
        (("trajectory_metadata", "total_frames"), 604, "605 source frames"),
        (("trajectory_metadata", "data_fps"), 60, "100 Hz"),
        (
            ("trajectory_metadata", "trajectory_info", "object_move"),
            [],
            "movement-frame identity",
        ),
    ],
)
def test_row_validation_rejects_authoritative_defects(
    path: tuple[str, ...], value: object, message: str
) -> None:
    row = copy.deepcopy(exact_row())
    owner = row
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        trace.validate_row_identity(row)


def test_fixed_593_call_schedule_has_exact_endpoints() -> None:
    schedule = [trace.schedule_point(call) for call in range(593)]
    assert len(schedule) == 593
    assert [(point.target_index, point.reference_index) for point in schedule[:3]] == [
        (0, 0),
        (0, 1),
        (1, 2),
    ]
    assert (schedule[-1].target_index, schedule[-1].reference_index) == (591, 592)
    assert [point.source_reference_index for point in schedule] == list(range(10, 603))
    assert schedule[-1].source_target_index == 601
    with pytest.raises(ValueError):
        trace.schedule_point(593)


def test_terminal_policy_rejects_early_and_accepts_only_final_completion() -> None:
    trace.validate_terminal_policy(591, False, False, 0)
    with pytest.raises(RuntimeError, match="early reset"):
        trace.validate_terminal_policy(100, True, False, 2)
    with pytest.raises(RuntimeError, match="final call"):
        trace.validate_terminal_policy(592, False, False, 0)
    with pytest.raises(RuntimeError, match="final call"):
        trace.validate_terminal_policy(592, True, False, 1)
    with pytest.raises(RuntimeError, match="final call"):
        trace.validate_terminal_policy(592, True, True, 2)
    trace.validate_terminal_policy(592, True, True, 1)


def test_schema_validator_accepts_exact_shapes_and_rejects_shape_dtype_and_identity() -> None:
    metadata = valid_metadata()
    arrays = valid_arrays()
    trace.validate_trace_schema(metadata, arrays)

    wrong_shape = dict(arrays)
    wrong_shape["q_target"] = np.zeros((592, 26), dtype=np.float32)
    with pytest.raises(ValueError, match="q_target shape"):
        trace.validate_trace_schema(metadata, wrong_shape)

    wrong_dtype = dict(arrays)
    wrong_dtype["target_index"] = wrong_dtype["target_index"].astype(np.int32)
    with pytest.raises(ValueError, match="target_index dtype"):
        trace.validate_trace_schema(metadata, wrong_dtype)

    wrong_identity = copy.deepcopy(metadata)
    wrong_identity["trajectory_identity"]["dataset_version"] = 3324
    with pytest.raises(ValueError, match="dataset_version mismatch"):
        trace.validate_trace_schema(wrong_identity, arrays)


def test_metrics_keep_translation_rotation_and_finger_units_separate() -> None:
    arrays = valid_arrays()
    arrays["object_quat_xyzw"][:, 3] = 1.0
    arrays["object_reference_quat_xyzw"][:, 3] = 1.0
    arrays["hand_palm_quat_xyzw"][:, 3] = 1.0
    arrays["hand_qpos"][:, 0] = 0.01
    arrays["hand_qpos"][:, 3] = np.float32(2.0 * np.pi - 0.1)
    arrays["hand_qpos"][:, 6] = 0.2
    metrics = trace._compute_metrics(arrays)
    assert metrics["hand_translation_rmse_m"] == pytest.approx(0.01)
    assert metrics["hand_angular_rmse_rad"] == pytest.approx(0.1, abs=1e-6)
    assert metrics["finger_joint_rmse_rad"] == pytest.approx(
        0.2 / np.sqrt(20), abs=1e-6
    )
    assert metrics["object_position_rmse_m"] == 0.0


def test_relative_output_is_resolved_before_runtime_chdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    caller = tmp_path / "caller"
    runtime = tmp_path / "runtime"
    caller.mkdir()
    runtime.mkdir()
    monkeypatch.chdir(caller)
    paths = trace.artifact_paths("outputs/reference")
    assert paths.prefix == (caller / "outputs/reference").resolve()
    monkeypatch.chdir(runtime)
    assert paths.final_json == caller / "outputs/reference.json"


def test_run_rejects_source_output_and_restores_cwd_on_early_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    with pytest.raises(ValueError, match="outside the protected sibling"):
        trace.run(trace.SOURCE_REPO / "outputs/reference")

    caller = tmp_path / "caller"
    runtime = tmp_path / "runtime"
    caller.mkdir()
    runtime.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setattr(trace, "RUNTIME_CWD", runtime)

    def fail_source_capture() -> dict[str, object]:
        raise RuntimeError("synthetic pre-import failure")

    monkeypatch.setattr(trace, "capture_source_state", fail_source_capture)
    with pytest.raises(RuntimeError, match="synthetic pre-import failure"):
        trace.run("outputs/reference")
    assert Path.cwd() == caller
    assert (caller / "outputs/reference.failure.json").is_file()
    assert not (runtime / "outputs").exists()


def test_failure_outputs_are_disjoint_and_cannot_occupy_final_names(tmp_path: Path) -> None:
    paths = trace.artifact_paths(tmp_path / "reference.json")
    trace.write_failure_artifacts(
        paths,
        {"schema": trace.SCHEMA, "status": "failed"},
        {"target_index": np.array([0], dtype=np.int64)},
    )
    assert paths.failure_json.is_file()
    assert paths.partial_npz.is_file()
    assert not paths.final_json.exists()
    assert not paths.final_npz.exists()
    assert len(
        {
            paths.failure_json,
            paths.partial_npz,
            paths.final_json,
            paths.final_npz,
        }
    ) == 4
