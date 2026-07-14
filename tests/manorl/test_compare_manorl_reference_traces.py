from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[2] / "tools" / "compare_manorl_reference_traces.py"
_SPEC = importlib.util.spec_from_file_location(
    "compare_manorl_reference_traces", MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
compare = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = compare
_SPEC.loader.exec_module(compare)


def valid_arrays(backend: str) -> dict[str, np.ndarray]:
    specs = (
        compare.MUJOCO_ARRAY_SPECS
        if backend == "mujoco"
        else compare.ISAAC_ARRAY_SPECS
    )
    arrays = {
        name: np.zeros((compare.TRACE_STEPS, *tail_shape), dtype=dtype)
        for name, (tail_shape, dtype) in specs.items()
    }
    calls = np.arange(compare.TRACE_STEPS, dtype=np.int64)
    arrays["target_index"][:] = np.maximum(calls - 1, 0)
    arrays["reference_index"][:] = calls
    arrays["source_reference_index"][:] = calls + compare.EXPECTED_IDENTITY["source_start"]
    arrays["sim_time"][:] = (calls.astype(np.float64) + 1.0) * compare.CONTROL_DT
    arrays["object_quat_xyzw"][:, 3] = 1.0
    arrays["object_reference_quat_xyzw"][:, 3] = 1.0
    if backend == "isaac":
        arrays["source_target_index"][:] = (
            arrays["target_index"] + compare.EXPECTED_IDENTITY["source_start"]
        )
        arrays["hand_palm_quat_xyzw"][:, 3] = 1.0
    return arrays


def valid_metadata(backend: str) -> dict[str, Any]:
    identity = {
        "dataset_path": compare.EXPECTED_IDENTITY["dataset_path"],
        "dataset_version": compare.EXPECTED_IDENTITY["dataset_version"],
        "row_index": 1,
        "object_index": 0,
        "uuid": compare.EXPECTED_IDENTITY["uuid"],
        "file_uuid": compare.EXPECTED_IDENTITY["file_uuid"],
        "identity": compare.EXPECTED_IDENTITY["identity"],
    }
    if backend == "mujoco":
        identity.update(
            source_start=compare.EXPECTED_IDENTITY["source_start"],
            source_stop=compare.EXPECTED_IDENTITY["source_stop"],
            loaded_dataset_version=compare.EXPECTED_IDENTITY["dataset_version"],
        )
        return {
            "schema": compare.MUJOCO_SCHEMA,
            "trajectory_identity": identity,
            "backend": "mjx-warp",
            "device": "gpu",
            "trace_path": "mujoco.npz",
            "trace_steps": compare.TRACE_STEPS,
            "metrics": {},
            "stability_gates": {
                "finite": True,
                "warning_free": True,
                "below_mjx_contact_capacity_boundary": True,
                "below_mjx_constraint_capacity_boundary": True,
                "max_abs_dof_velocity_below_100": True,
                "zero_wrist_effort_saturation": True,
            },
            "diagnostic_passed": True,
            "config": {},
            "claims": {"isaac_parity": "not_evaluated_no_isaac_trace"},
        }
    identity["source_slice"] = {
        "start": compare.EXPECTED_IDENTITY["source_start"],
        "stop": compare.EXPECTED_IDENTITY["source_stop"],
        "stop_exclusive": True,
    }
    identity["reference_frame_count"] = compare.TRACE_STEPS + 1
    identity["physical_call_count"] = compare.TRACE_STEPS
    return {
        "schema": compare.ISAAC_SCHEMA,
        "trajectory_identity": identity,
        "lance_index_entry": {},
        "source_repository": {},
        "config": {},
        "checks": {key: True for key in compare.ISAAC_CHECK_KEYS},
        "action_invariance": {},
        "metrics": {},
        "non_comparable_fields": [],
        "claims": {"isaac_parity": "trace_generated_not_yet_compared"},
    }


def write_artifacts(
    prefix: Path,
    backend: str,
    arrays: dict[str, np.ndarray] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    source_arrays = valid_arrays(backend) if arrays is None else arrays
    np.savez_compressed(f"{prefix}.npz", **source_arrays)
    Path(f"{prefix}.json").write_text(
        json.dumps(valid_metadata(backend) if metadata is None else metadata),
        encoding="utf-8",
    )


def artifact_pair(
    tmp_path: Path,
    *,
    mujoco_arrays: dict[str, np.ndarray] | None = None,
    isaac_arrays: dict[str, np.ndarray] | None = None,
    mujoco_metadata: dict[str, Any] | None = None,
    isaac_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    mujoco_prefix = tmp_path / "inputs" / "mujoco" / "trace"
    isaac_prefix = tmp_path / "inputs" / "isaac" / "trace"
    write_artifacts(mujoco_prefix, "mujoco", mujoco_arrays, mujoco_metadata)
    write_artifacts(isaac_prefix, "isaac", isaac_arrays, isaac_metadata)
    return mujoco_prefix, isaac_prefix


def run_comparison(
    tmp_path: Path,
    **artifacts: Any,
) -> dict[str, Any]:
    mujoco_prefix, isaac_prefix = artifact_pair(tmp_path, **artifacts)
    return compare.compare_traces(
        mujoco_prefix, isaac_prefix, tmp_path / "output" / "comparison"
    )


def test_fixtures_match_complete_backend_producer_abis() -> None:
    mujoco = valid_arrays("mujoco")
    isaac = valid_arrays("isaac")

    assert set(mujoco) == set(compare.MUJOCO_ARRAY_SPECS)
    assert set(isaac) == set(compare.ISAAC_ARRAY_SPECS)
    assert len(mujoco) == 22
    assert len(isaac) == 42
    for arrays, specs in (
        (mujoco, compare.MUJOCO_ARRAY_SPECS),
        (isaac, compare.ISAAC_ARRAY_SPECS),
    ):
        for name, (tail_shape, dtype) in specs.items():
            assert arrays[name].shape == (compare.TRACE_STEPS, *tail_shape)
            assert arrays[name].dtype == dtype


def test_exact_trace_supports_parity_and_near_zero_ratios_are_not_applicable(
    tmp_path: Path,
) -> None:
    mujoco = valid_arrays("mujoco")
    isaac = valid_arrays("isaac")
    mujoco["contact_count"][0] = 4
    mujoco["has_contact"][0] = True
    isaac["contact_count"][1:3] = 2
    isaac["has_contact"][1:3] = True

    report = run_comparison(tmp_path, mujoco_arrays=mujoco, isaac_arrays=isaac)

    assert report["schema"] == "manorl.reference_comparison.v1"
    assert report["comparison_status"] == "passed"
    assert report["claims"] == {
        "parity": "supported_same_order_comparable_scale",
        "failed_gates": [],
        "scope": "evidence classification only; not a training-success claim",
    }
    assert all(report["gates"].values())
    finger_ratio = report["scale_comparison"]["metrics"]["finger_joint_rmse_rad"]
    assert finger_ratio["applicable"] is False
    assert finger_ratio["ratio_mujoco_over_isaac"] is None
    assert "both backend metrics" in finger_ratio["reason"]
    contacts = report["non_comparable_contact_summaries"]
    assert contacts["used_as_gate"] is False
    assert contacts["mujoco"]["total_contact_count"] == 4
    assert contacts["isaac"]["total_contact_count"] == 4
    assert contacts["mujoco"]["active_call_fraction"] != contacts["isaac"][
        "active_call_fraction"
    ]


@pytest.mark.parametrize(
    ("backend", "removed_field"),
    [
        ("mujoco", "actuator_force_substeps"),
        ("isaac", "mocap_target_unclamped"),
    ],
)
def test_truncated_npz_field_set_is_rejected(
    tmp_path: Path, backend: str, removed_field: str
) -> None:
    arrays = valid_arrays(backend)
    arrays.pop(removed_field)
    kwargs = {f"{backend}_arrays": arrays}
    mujoco_prefix, isaac_prefix = artifact_pair(tmp_path, **kwargs)

    with pytest.raises(compare.TraceContractError, match=f"missing=.*{removed_field}"):
        compare.compare_traces(mujoco_prefix, isaac_prefix, tmp_path / "comparison")


@pytest.mark.parametrize("backend", ["mujoco", "isaac"])
def test_extra_npz_field_is_rejected(tmp_path: Path, backend: str) -> None:
    arrays = valid_arrays(backend)
    arrays["unexpected_array"] = np.zeros(compare.TRACE_STEPS, dtype=np.float64)
    kwargs = {f"{backend}_arrays": arrays}
    mujoco_prefix, isaac_prefix = artifact_pair(tmp_path, **kwargs)

    with pytest.raises(compare.TraceContractError, match="extra=.*unexpected_array"):
        compare.compare_traces(mujoco_prefix, isaac_prefix, tmp_path / "comparison")


@pytest.mark.parametrize(
    ("backend", "field", "replacement", "error"),
    [
        (
            "mujoco",
            "actuator_force_substeps",
            np.zeros((compare.TRACE_STEPS, 26), dtype=np.float64),
            "shape",
        ),
        (
            "isaac",
            "termination_reason_code",
            np.zeros(compare.TRACE_STEPS, dtype=np.int64),
            "dtype",
        ),
    ],
)
def test_backend_specific_npz_shape_and_dtype_are_rejected(
    tmp_path: Path,
    backend: str,
    field: str,
    replacement: np.ndarray,
    error: str,
) -> None:
    arrays = valid_arrays(backend)
    arrays[field] = replacement
    kwargs = {f"{backend}_arrays": arrays}
    mujoco_prefix, isaac_prefix = artifact_pair(tmp_path, **kwargs)

    with pytest.raises(compare.TraceContractError, match=f"{field} {error}"):
        compare.compare_traces(mujoco_prefix, isaac_prefix, tmp_path / "comparison")


def test_identity_mismatch_fails_before_output(tmp_path: Path) -> None:
    metadata = valid_metadata("isaac")
    metadata["trajectory_identity"]["uuid"] = "wrong-uuid"
    mujoco_prefix, isaac_prefix = artifact_pair(tmp_path, isaac_metadata=metadata)
    output = tmp_path / "comparison.json"

    with pytest.raises(compare.TraceContractError, match="identity mismatch"):
        compare.compare_traces(mujoco_prefix, isaac_prefix, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("backend", "section", "canonical_key", "substitute_key"),
    [
        ("mujoco", "stability_gates", "finite", "no_nan"),
        ("isaac", "checks", "exact_index_entry", "index_entry_present"),
    ],
)
def test_missing_canonical_producer_gate_or_check_is_rejected(
    tmp_path: Path,
    backend: str,
    section: str,
    canonical_key: str,
    substitute_key: str,
) -> None:
    metadata = valid_metadata(backend)
    metadata[section].pop(canonical_key)
    metadata[section][substitute_key] = True
    kwargs = {f"{backend}_metadata": metadata}
    mujoco_prefix, isaac_prefix = artifact_pair(tmp_path, **kwargs)

    with pytest.raises(
        compare.TraceContractError,
        match=f"{section} fields differ: missing=.*{canonical_key}.*extra=.*{substitute_key}",
    ):
        compare.compare_traces(mujoco_prefix, isaac_prefix, tmp_path / "comparison")


def test_inconsistent_mujoco_diagnostic_flag_is_rejected(tmp_path: Path) -> None:
    metadata = valid_metadata("mujoco")
    metadata["diagnostic_passed"] = False
    mujoco_prefix, isaac_prefix = artifact_pair(tmp_path, mujoco_metadata=metadata)

    with pytest.raises(compare.TraceContractError, match="must equal all canonical"):
        compare.compare_traces(mujoco_prefix, isaac_prefix, tmp_path / "comparison")


def test_required_metadata_sections_and_canonical_claims_are_rejected(
    tmp_path: Path,
) -> None:
    missing_config = valid_metadata("mujoco")
    missing_config.pop("config")
    mujoco_prefix, isaac_prefix = artifact_pair(
        tmp_path / "missing", mujoco_metadata=missing_config
    )
    with pytest.raises(compare.TraceContractError, match="config.*must be an object"):
        compare.compare_traces(mujoco_prefix, isaac_prefix, tmp_path / "missing-output")

    wrong_claim = valid_metadata("isaac")
    wrong_claim["claims"]["isaac_parity"] = "not_evaluated_no_isaac_trace"
    mujoco_prefix, isaac_prefix = artifact_pair(
        tmp_path / "claim", isaac_metadata=wrong_claim
    )
    with pytest.raises(compare.TraceContractError, match="canonical parity claim"):
        compare.compare_traces(mujoco_prefix, isaac_prefix, tmp_path / "claim-output")


def test_schedule_mismatch_fails_before_metrics(tmp_path: Path) -> None:
    isaac = valid_arrays("isaac")
    isaac["reference_index"][250] = 249
    mujoco_prefix, isaac_prefix = artifact_pair(tmp_path, isaac_arrays=isaac)

    with pytest.raises(compare.TraceContractError, match="schedule mismatch.*call 250"):
        compare.compare_traces(mujoco_prefix, isaac_prefix, tmp_path / "comparison")


def test_raw_reference_mismatch_fails_before_metrics(tmp_path: Path) -> None:
    isaac = valid_arrays("isaac")
    isaac["object_reference_pos_raw"][100, 2] = 1.0e-9
    mujoco_prefix, isaac_prefix = artifact_pair(tmp_path, isaac_arrays=isaac)

    with pytest.raises(compare.TraceContractError, match="shared reference mismatch"):
        compare.compare_traces(mujoco_prefix, isaac_prefix, tmp_path / "comparison")


def test_transformed_reference_xy_shift_with_actual_is_rejected_before_metrics(
    tmp_path: Path,
) -> None:
    isaac = valid_arrays("isaac")
    isaac["object_pos"][:, 0] = 0.01
    isaac["object_reference_pos"][:, 0] = 0.01
    mujoco_prefix, isaac_prefix = artifact_pair(tmp_path, isaac_arrays=isaac)

    with pytest.raises(
        compare.TraceContractError,
        match="object_reference_pos_transformed_xy",
    ):
        compare.compare_traces(mujoco_prefix, isaac_prefix, tmp_path / "comparison")


def test_reference_orientation_rotation_with_actual_is_rejected_before_metrics(
    tmp_path: Path,
) -> None:
    isaac = valid_arrays("isaac")
    angle = 0.01
    rotated = np.array(
        [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)],
        dtype=np.float32,
    )
    isaac["object_quat_xyzw"][:] = rotated
    isaac["object_reference_quat_xyzw"][:] = rotated
    mujoco_prefix, isaac_prefix = artifact_pair(tmp_path, isaac_arrays=isaac)

    with pytest.raises(
        compare.TraceContractError,
        match="object_reference_orientation.*max geodesic",
    ):
        compare.compare_traces(mujoco_prefix, isaac_prefix, tmp_path / "comparison")


def test_reference_quaternion_sign_and_transformed_z_are_not_shared_input_gates(
    tmp_path: Path,
) -> None:
    isaac = valid_arrays("isaac")
    isaac["object_reference_quat_xyzw"] *= -1.0
    isaac["object_pos"][:, 2] = 0.02
    isaac["object_reference_pos"][:, 2] = 0.02

    report = run_comparison(tmp_path, isaac_arrays=isaac)

    assert report["comparison_status"] == "passed"
    assert report["validation"]["shared_input_contracts"][
        "object_reference_orientation"
    ]["max_geodesic_difference_rad"] == pytest.approx(0.0)
    assert report["transformed_object_reference_z_difference"]["whole_trace"][
        "max_abs_m"
    ] == pytest.approx(0.02)


def test_wrist_two_pi_difference_is_wrapped_for_commands_and_tracking(
    tmp_path: Path,
) -> None:
    mujoco = valid_arrays("mujoco")
    isaac = valid_arrays("isaac")
    mujoco["q_target"][:, 3] = 2.0 * np.pi + 1.0e-8
    mujoco["hand_qpos"][:, 4] = 2.0 * np.pi - 2.0e-8
    isaac["hand_qpos"][:, 4] = 0.0

    report = run_comparison(tmp_path, mujoco_arrays=mujoco, isaac_arrays=isaac)

    command = report["q_command_comparison"]
    assert command["per_dof_rmse"][3] == pytest.approx(1.0e-8, abs=1.0e-12)
    assert command["largest_discrepancy"]["dof_index"] == 3
    wrist = report["tracking_metrics"]["mujoco"]["whole_trace"][
        "wrist_wrapped_angular_norm_rmse_rad"
    ]
    assert wrist == pytest.approx(2.0e-8, abs=1.0e-12)


def test_phase_boundaries_use_post_step_reference_index(tmp_path: Path) -> None:
    mujoco = valid_arrays("mujoco")
    isaac = valid_arrays("isaac")
    for arrays in (mujoco, isaac):
        arrays["hand_qpos"][[249, 250, 542, 543], 0] = 1.0

    report = run_comparison(tmp_path, mujoco_arrays=mujoco, isaac_arrays=isaac)
    phases = report["tracking_metrics"]["mujoco"]["phases"]

    assert phases["pre_motion"]["call_count"] == 250
    assert phases["movement"]["call_count"] == 293
    assert phases["post_motion"]["call_count"] == 248
    assert phases["pre_motion"]["hand_translation_norm_rmse_m"] == pytest.approx(
        np.sqrt(1.0 / 250.0)
    )
    assert phases["movement"]["hand_translation_norm_rmse_m"] == pytest.approx(
        np.sqrt(2.0 / 293.0)
    )
    assert phases["post_motion"]["hand_translation_norm_rmse_m"] == pytest.approx(
        np.sqrt(1.0 / 248.0)
    )
    assert phases["movement"]["raw_source_reference_index"] == {
        "start": 690,
        "stop_inclusive": 982,
    }


@pytest.mark.parametrize("failure", ["scale", "ordering"])
def test_scale_or_order_failure_is_not_supported(tmp_path: Path, failure: str) -> None:
    mujoco = valid_arrays("mujoco")
    isaac = valid_arrays("isaac")
    if failure == "scale":
        rank_pattern = np.arange(1, 21, dtype=np.float64) * 1.0e-3
        isaac["hand_qpos"][:, 6:] = rank_pattern
        mujoco["hand_qpos"][:, 6:] = 4.0 * rank_pattern
    else:
        mujoco["hand_qpos"][:, 3] = 0.02
        isaac["hand_qpos"][:, 4] = 0.02
        mujoco["hand_qpos"][:, 6:10] = 0.01
        isaac["hand_qpos"][:, 10:14] = 0.01

    report = run_comparison(tmp_path, mujoco_arrays=mujoco, isaac_arrays=isaac)

    assert report["comparison_status"] == "not_supported"
    assert report["claims"]["parity"] == "not_supported"
    if failure == "scale":
        evidence = report["scale_comparison"]["metrics"]["finger_joint_rmse_rad"]
        assert evidence["ratio_mujoco_over_isaac"] == pytest.approx(4.0)
        assert evidence["within_range"] is False
        assert "comparable_scale" in report["claims"]["failed_gates"]
    else:
        assert report["ordering"]["worst_wrist_axis"]["same_identity"] is False
        assert report["ordering"]["worst_finger_group"]["same_identity"] is False
        assert "same_worst_wrist_axis" in report["claims"]["failed_gates"]
        assert "same_worst_finger_group" in report["claims"]["failed_gates"]


def test_scaled_constant_finger_errors_have_identical_all_tied_ranking(
    tmp_path: Path,
) -> None:
    mujoco = valid_arrays("mujoco")
    isaac = valid_arrays("isaac")
    mujoco["hand_qpos"][:, 6:] = 0.002
    isaac["hand_qpos"][:, 6:] = 0.001

    report = run_comparison(tmp_path, mujoco_arrays=mujoco, isaac_arrays=isaac)

    ranking = report["ordering"]["finger_per_dof_rmse_spearman"]
    assert ranking["correlation"] == 1.0
    assert ranking["method"] == "individually_constant_vectors"
    assert ranking["constant_vector_atol"] == compare.FINGER_RANK_CONSTANT_ATOL
    assert report["scale_comparison"]["metrics"]["finger_joint_rmse_rad"][
        "ratio_mujoco_over_isaac"
    ] == pytest.approx(2.0)
    assert report["comparison_status"] == "passed"


def test_one_sided_constant_finger_errors_have_undefined_ranking(
    tmp_path: Path,
) -> None:
    mujoco = valid_arrays("mujoco")
    isaac = valid_arrays("isaac")
    mujoco["hand_qpos"][:, 6:] = 0.002
    isaac["hand_qpos"][:, 6:] = np.linspace(0.001, 0.002, 20)

    report = run_comparison(tmp_path, mujoco_arrays=mujoco, isaac_arrays=isaac)

    ranking = report["ordering"]["finger_per_dof_rmse_spearman"]
    assert ranking["correlation"] is None
    assert ranking["passed"] is False
    assert ranking["method"] == "undefined_one_sided_constant_vector"
    assert report["comparison_status"] == "not_supported"
    assert "finger_rank_correlation_at_least_0_8" in report["claims"][
        "failed_gates"
    ]


def test_non_negligible_metric_with_near_zero_denominator_is_explicit_na(
    tmp_path: Path,
) -> None:
    mujoco = valid_arrays("mujoco")
    isaac = valid_arrays("isaac")
    mujoco["object_pos"][:, 0] = 0.01

    report = run_comparison(tmp_path, mujoco_arrays=mujoco, isaac_arrays=isaac)

    evidence = report["scale_comparison"]["metrics"][
        "object_position_norm_rmse_m"
    ]
    assert evidence["applicable"] is False
    assert evidence["ratio_mujoco_over_isaac"] is None
    assert evidence["gate_passed"] is False
    assert "denominator" in evidence["reason"]
    assert report["claims"]["parity"] == "not_supported"


def test_failed_producer_contract_prevents_pass_but_contact_does_not(
    tmp_path: Path,
) -> None:
    metadata = valid_metadata("mujoco")
    metadata["stability_gates"]["warning_free"] = False
    metadata["diagnostic_passed"] = False
    mujoco = valid_arrays("mujoco")
    mujoco["contact_count"][:] = 100
    mujoco["has_contact"][:] = True

    report = run_comparison(
        tmp_path, mujoco_arrays=mujoco, mujoco_metadata=metadata
    )

    assert report["producer_contracts"]["mujoco"]["passed"] is False
    assert report["gates"]["producer_contracts"] is False
    assert report["non_comparable_contact_summaries"]["used_as_gate"] is False
    assert report["claims"]["failed_gates"] == ["producer_contracts"]


def test_atomic_output_and_relative_paths_resolve_from_caller(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    caller = tmp_path / "caller"
    inputs = tmp_path / "elsewhere" / "inputs"
    caller.mkdir()
    mujoco_prefix, isaac_prefix = artifact_pair(inputs)
    monkeypatch.chdir(caller)
    replace_calls: list[tuple[Path, Path, bool]] = []
    original_replace = os.replace

    def recording_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        replace_calls.append((Path(source), Path(destination), Path(source).is_file()))
        original_replace(source, destination)

    monkeypatch.setattr(compare.os, "replace", recording_replace)
    report = compare.compare_traces(
        Path(f"{mujoco_prefix}.npz"),
        Path(f"{isaac_prefix}.json"),
        "relative/output/comparison",
    )

    output = caller / "relative" / "output" / "comparison.json"
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert replace_calls == [(replace_calls[0][0], output, True)]
    assert replace_calls[0][0].parent == output.parent
    assert replace_calls[0][0] != output
    assert list(output.parent.glob(".comparison.json.*.tmp")) == []
    assert not (mujoco_prefix.parent / "relative").exists()
