from __future__ import annotations

import inspect
import numpy as np
import pytest

from tools.replay_atomic_benchmark_pilot import (
    decode_row,
    render_row_directory_name,
    run_physics,
    static_layout,
    validate_gpu_binding,
    wxyz,
)


def row_fixture() -> dict:
    n = 4
    return {
        "index": {"uuid": "u", "seed_uuid": "s"},
        "trajectory_metadata": {
            "gesture": "004-cup-pour",
            "data_fps": 120,
            "total_frames": n,
            "object_names": ["egg_cup", "bowl"],
            "trajectory_info": {
                "object_move": [
                    {"object_name": "egg_cup", "start_frame": 1, "end_frame": 3}
                ]
            },
        },
        "provenance": {
            "contract": "synthetic_mano_target_replay_visual_v2_contact",
            "control_fps": 120,
            "physics_fps": 480,
            "physics_substeps_per_control": 4,
            "source_identity": "egg_cup_04_001",
        },
        "hands": [
            {
                "urdf_dof": np.zeros((n, 28)).tolist(),
                "urdf_dof_target": np.ones((n, 28)).tolist(),
            }
        ],
        "objects": [
            {"pos": np.zeros((n, 3)).tolist(), "rot_aa": np.zeros((n, 3)).tolist()},
            {"pos": np.ones((n, 3)).tolist(), "rot_aa": np.zeros((n, 3)).tolist()},
        ],
    }


def test_decode_row_preserves_multiobject_target_contract() -> None:
    decoded = decode_row(9, row_fixture())
    assert decoded["row_index"] == 9
    assert decoded["names"] == ("egg_cup", "bowl")
    assert decoded["target"] == "egg_cup"
    np.testing.assert_array_equal(decoded["commands"], 1.0)
    np.testing.assert_array_equal(decoded["object_recorded_pos"]["bowl"], 1.0)
    np.testing.assert_array_equal(
        decoded["object_recorded_quat_wxyz"]["egg_cup"],
        np.repeat([[1.0, 0.0, 0.0, 0.0]], 4, axis=0),
    )


def test_decode_rejects_wrong_clock() -> None:
    row = row_fixture()
    row["provenance"]["physics_fps"] = 400
    with pytest.raises(ValueError, match="incompatible replay contract"):
        decode_row(0, row)


def test_static_layout_excludes_physical_objects() -> None:
    from tools.replay_atomic_benchmark_pilot import ALL_OBJECTS

    layout = {
        "objects": {
            name: {"pos": [index, index + 1, index + 2]}
            for index, name in enumerate(ALL_OBJECTS)
        }
    }
    result = static_layout(layout, ("egg_cup", "bowl"))
    assert "egg_cup" not in result and "bowl" not in result
    assert set(result) == set(ALL_OBJECTS) - {"egg_cup", "bowl"}
    np.testing.assert_array_equal(result["trash_bin"][1], [1, 0, 0, 0])


def test_wxyz_identity() -> None:
    np.testing.assert_allclose(wxyz([0, 0, 0]), [1, 0, 0, 0])


def test_duplicate_padding_rows_use_collision_free_trace_directories() -> None:
    assert render_row_directory_name(154, sequence=1, occurrence_count=5) == "row0154_slot0"
    assert render_row_directory_name(154, sequence=5, occurrence_count=5) == "row0154_slot4"
    assert render_row_directory_name(154, sequence=1, occurrence_count=1) == "row0154"
    with pytest.raises(ValueError, match="positive"):
        render_row_directory_name(154, sequence=0, occurrence_count=5)


def test_run_physics_exposes_allocation_only_capacity_overrides() -> None:
    parameters = inspect.signature(run_physics).parameters
    assert parameters["contact_capacity_per_world"].default == 1024
    assert parameters["constraint_capacity"].default == 4096
    assert parameters["ccd_contacts_per_world"].default == 256


def test_compute_and_egl_gpu_binding_must_match(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "3")
    assert validate_gpu_binding(3)["physical_gpu"] == 3
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    with pytest.raises(RuntimeError, match="identical physical index"):
        validate_gpu_binding(3)
