#!/usr/bin/env python3
"""Compare fixed MuJoCo and Isaac Gym ManoRL reference replay traces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


OUTPUT_SCHEMA = "manorl.reference_comparison.v1"
MUJOCO_SCHEMA = "manorl.mujoco.reference_replay.v1"
ISAAC_SCHEMA = "manorl.isaacgym.reference_replay.v1"
TRACE_STEPS = 791
DOFS = 26
CONTROL_DT = 0.005
SIM_TIME_ATOL = 2.0e-7
QUATERNION_NORM_ATOL = 1.0e-3
HAND_REFERENCE_ATOL = 1.0e-6
RAW_OBJECT_REFERENCE_ATOL = 1.0e-12
TRANSFORMED_OBJECT_REFERENCE_XY_ATOL = 1.0e-6
OBJECT_REFERENCE_ORIENTATION_GEODESIC_ATOL = 1.0e-6
# At this scale, differences are numerical noise rather than rank information.
FINGER_RANK_CONSTANT_ATOL = 1.0e-12
RATIO_MIN = 1.0 / 3.0
RATIO_MAX = 3.0

EXPECTED_IDENTITY = {
    "dataset_path": (
        "/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_remake/"
        "npy_s02_v3.lance"
    ),
    "dataset_version": 132,
    "row_index": 1,
    "object_index": 0,
    "uuid": "d5bc2bc6-9458-52d0-bccc-66c9ec21bae3",
    "file_uuid": "e6fe4732-72cd-5ab7-93e6-2e62dc0263a5",
    "identity": "cube1_01_009",
    "source_start": 440,
    "source_stop": 1232,
}

JOINT_NAMES = (
    "ARTx",
    "ARTy",
    "ARTz",
    "ARRx",
    "ARRy",
    "ARRz",
    "j1_thumb_cmc_abd",
    "j1_thumb_cmc_flex",
    "j1_thumb_mcp",
    "j1_thumb_ip",
    "j2_index_mcp_abd",
    "j2_index_mcp_flex",
    "j2_index_pip",
    "j2_index_dip",
    "j3_middle_mcp_abd",
    "j3_middle_mcp_flex",
    "j3_middle_pip",
    "j3_middle_dip",
    "j4_ring_mcp_abd",
    "j4_ring_mcp_flex",
    "j4_ring_pip",
    "j4_ring_dip",
    "j5_pinky_mcp_abd",
    "j5_pinky_mcp_flex",
    "j5_pinky_pip",
    "j5_pinky_dip",
)

# Exact stacked output ABI from sim.manorl.mjx_sim.StepTrace.
MUJOCO_ARRAY_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "target_index": ((), np.dtype("int64")),
    "reference_index": ((), np.dtype("int64")),
    "source_reference_index": ((), np.dtype("int64")),
    "sim_time": ((), np.dtype("float64")),
    "q_target": ((DOFS,), np.dtype("float64")),
    "hand_qpos": ((DOFS,), np.dtype("float64")),
    "hand_qvel": ((DOFS,), np.dtype("float64")),
    "actuator_force_substeps": ((2, DOFS), np.dtype("float64")),
    "hand_reference": ((DOFS,), np.dtype("float64")),
    "object_pos": ((3,), np.dtype("float64")),
    "object_quat_xyzw": ((4,), np.dtype("float64")),
    "object_reference_pos_raw": ((3,), np.dtype("float64")),
    "object_reference_pos": ((3,), np.dtype("float64")),
    "object_reference_quat_xyzw": ((4,), np.dtype("float64")),
    "contact_count": ((), np.dtype("int64")),
    "has_contact": ((), np.dtype("bool")),
    "hand_object_contact_count": ((), np.dtype("int64")),
    "hand_object_min_distance": ((), np.dtype("float64")),
    "hand_object_min_distance_valid": ((), np.dtype("bool")),
    "warning_count": ((), np.dtype("int64")),
    "contact_capacity_saturated": ((), np.dtype("bool")),
    "constraint_capacity_saturated": ((), np.dtype("bool")),
}

# Exact committed tools.isaacgym_reference_trace.REQUIRED_ARRAY_SPECS.
ISAAC_ARRAY_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "target_index": ((), np.dtype("int64")),
    "reference_index": ((), np.dtype("int64")),
    "source_target_index": ((), np.dtype("int64")),
    "source_reference_index": ((), np.dtype("int64")),
    "sim_time": ((), np.dtype("float64")),
    "q_target": ((DOFS,), np.dtype("float32")),
    "mocap_target_unclamped": ((DOFS,), np.dtype("float32")),
    "hand_qpos": ((DOFS,), np.dtype("float32")),
    "hand_qvel": ((DOFS,), np.dtype("float32")),
    "hand_reference": ((DOFS,), np.dtype("float64")),
    "hand_reference_task_raw": ((DOFS,), np.dtype("float32")),
    "object_pos": ((3,), np.dtype("float32")),
    "object_quat_xyzw": ((4,), np.dtype("float32")),
    "object_linvel": ((3,), np.dtype("float32")),
    "object_angvel": ((3,), np.dtype("float32")),
    "object_reference_pos_raw": ((3,), np.dtype("float64")),
    "object_reference_pos": ((3,), np.dtype("float32")),
    "object_reference_quat_xyzw": ((4,), np.dtype("float32")),
    "hand_palm_pos": ((3,), np.dtype("float32")),
    "hand_palm_quat_xyzw": ((4,), np.dtype("float32")),
    "raw_action": ((DOFS,), np.dtype("float32")),
    "processed_action": ((DOFS,), np.dtype("float32")),
    "cumulative_offset": ((3,), np.dtype("float32")),
    "cumulative_joint_offset": ((20,), np.dtype("float32")),
    "progress_before_pre": ((), np.dtype("int64")),
    "progress_after_post": ((), np.dtype("int64")),
    "trajectory_step_before_pre": ((), np.dtype("int64")),
    "trajectory_step_after_pre": ((), np.dtype("int64")),
    "trajectory_step_after_post": ((), np.dtype("int64")),
    "reset": ((), np.dtype("bool")),
    "termination_reason_code": ((), np.dtype("int32")),
    "termination_object_target_dist": ((), np.dtype("float32")),
    "trajectory_complete_reset_mask": ((), np.dtype("bool")),
    "deviation_reset_mask": ((), np.dtype("bool")),
    "reward": ((), np.dtype("float32")),
    "contact_keypoint_force_xyz": ((16, 3), np.dtype("float32")),
    "contact_keypoint_force_magnitude": ((16,), np.dtype("float32")),
    "object_contact_force_xyz": ((3,), np.dtype("float32")),
    "object_contact_force_magnitude": ((), np.dtype("float32")),
    "expected_contact_mask": ((16,), np.dtype("float32")),
    "contact_count": ((), np.dtype("int64")),
    "has_contact": ((), np.dtype("bool")),
}

MUJOCO_STABILITY_GATE_KEYS = frozenset(
    {
        "finite",
        "warning_free",
        "below_mjx_contact_capacity_boundary",
        "below_mjx_constraint_capacity_boundary",
        "max_abs_dof_velocity_below_100",
        "zero_wrist_effort_saturation",
    }
)
ISAAC_CHECK_KEYS = frozenset(
    {
        "exact_index_entry",
        "live_row_identity",
        "dataset_version_132",
        "resolved_configuration",
        "exact_slice_440_1232",
        "one_environment",
        "26_dofs",
        "counter_schedule_all_calls",
        "no_reset_before_call_790",
        "final_trajectory_completion_reset",
        "exactly_791_simulate_calls",
        "source_repository_unchanged",
    }
)

PHASES = {
    "pre_motion": (0, 249),
    "movement": (250, 542),
    "post_motion": (543, 790),
}

FINGER_GROUPS = {
    "thumb": slice(6, 10),
    "index": slice(10, 14),
    "middle": slice(14, 18),
    "ring": slice(18, 22),
    "pinky": slice(22, 26),
}

PRIMARY_METRICS = {
    "hand_translation_norm_rmse_m": 1.0e-6,
    "wrist_wrapped_angular_norm_rmse_rad": 1.0e-5,
    "finger_joint_rmse_rad": 1.0e-5,
    "object_position_norm_rmse_m": 1.0e-6,
    "object_quaternion_geodesic_rmse_rad": 1.0e-5,
}


class TraceContractError(ValueError):
    """Raised when producer artifacts do not meet the comparison input contract."""


def _prefix_paths(prefix: str | os.PathLike[str]) -> tuple[Path, Path]:
    path = Path(prefix).expanduser().resolve()
    if path.suffix in {".npz", ".json"}:
        path = path.with_suffix("")
    return Path(f"{path}.npz"), Path(f"{path}.json")


def _output_path(output: str | os.PathLike[str]) -> Path:
    path = Path(output).expanduser().resolve()
    if path.suffix == "":
        return path.with_suffix(".json")
    if path.suffix != ".json":
        raise ValueError("--output must be a path prefix or end in .json")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TraceContractError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise TraceContractError(f"JSON artifact {path} must contain an object")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        raise TraceContractError(f"cannot read NPZ artifact {path}: {error}") from error


def _require_mapping(owner: Mapping[str, Any], key: str, backend: str) -> Mapping[str, Any]:
    value = owner.get(key)
    if not isinstance(value, Mapping):
        raise TraceContractError(f"{backend} JSON {key!r} must be an object")
    return value


def _normalize_identity(metadata: Mapping[str, Any], backend: str) -> dict[str, Any]:
    identity = _require_mapping(metadata, "trajectory_identity", backend)
    if backend == "mujoco":
        required = (
            "dataset_path",
            "row_index",
            "object_index",
            "uuid",
            "file_uuid",
            "identity",
            "source_start",
            "source_stop",
            "loaded_dataset_version",
        )
        missing = [key for key in required if key not in identity]
        if missing:
            raise TraceContractError(f"MuJoCo trajectory_identity missing fields: {missing}")
        dataset_version = identity["loaded_dataset_version"]
        if "dataset_version" in identity and identity["dataset_version"] != dataset_version:
            raise TraceContractError(
                "MuJoCo trajectory_identity dataset_version disagrees with loaded_dataset_version"
            )
        normalized = {
            "dataset_path": identity["dataset_path"],
            "dataset_version": dataset_version,
            "row_index": identity["row_index"],
            "object_index": identity["object_index"],
            "uuid": identity["uuid"],
            "file_uuid": identity["file_uuid"],
            "identity": identity["identity"],
            "source_start": identity["source_start"],
            "source_stop": identity["source_stop"],
        }
    else:
        required = (
            "dataset_path",
            "dataset_version",
            "row_index",
            "object_index",
            "uuid",
            "file_uuid",
            "identity",
            "source_slice",
        )
        missing = [key for key in required if key not in identity]
        if missing:
            raise TraceContractError(f"Isaac trajectory_identity missing fields: {missing}")
        source_slice = identity["source_slice"]
        if not isinstance(source_slice, Mapping):
            raise TraceContractError("Isaac trajectory_identity.source_slice must be an object")
        if source_slice.get("stop_exclusive") is not True:
            raise TraceContractError("Isaac source_slice.stop_exclusive must be true")
        normalized = {
            "dataset_path": identity["dataset_path"],
            "dataset_version": identity["dataset_version"],
            "row_index": identity["row_index"],
            "object_index": identity["object_index"],
            "uuid": identity["uuid"],
            "file_uuid": identity["file_uuid"],
            "identity": identity["identity"],
            "source_start": source_slice.get("start"),
            "source_stop": source_slice.get("stop"),
        }
    return normalized


def _validate_metadata(
    metadata: Mapping[str, Any], expected_schema: str, backend: str
) -> dict[str, Any]:
    if metadata.get("schema") != expected_schema:
        raise TraceContractError(
            f"{backend} schema {metadata.get('schema')!r} != {expected_schema!r}"
        )
    producer_identity = _require_mapping(metadata, "trajectory_identity", backend)
    claims = _require_mapping(metadata, "claims", backend)
    if backend == "mujoco":
        for section in ("metrics", "config"):
            _require_mapping(metadata, section, backend)
        if metadata.get("backend") not in {"mjx-warp", "mujoco-cpu"}:
            raise TraceContractError("MuJoCo backend must be 'mjx-warp' or 'mujoco-cpu'")
        if metadata.get("device") not in {"cpu", "gpu"}:
            raise TraceContractError("MuJoCo device must be 'cpu' or 'gpu'")
        if not isinstance(metadata.get("trace_path"), str) or not metadata["trace_path"]:
            raise TraceContractError("MuJoCo trace_path must be a non-empty string")
        if dict(claims) != {"isaac_parity": "not_evaluated_no_isaac_trace"}:
            raise TraceContractError("MuJoCo claims must contain the canonical parity claim")
        if metadata.get("trace_steps") != TRACE_STEPS:
            raise TraceContractError(f"MuJoCo trace_steps must be {TRACE_STEPS}")
    else:
        for section in (
            "lance_index_entry",
            "source_repository",
            "config",
            "action_invariance",
            "metrics",
        ):
            _require_mapping(metadata, section, backend)
        non_comparable = metadata.get("non_comparable_fields")
        if not isinstance(non_comparable, list):
            raise TraceContractError("Isaac JSON 'non_comparable_fields' must be an array")
        if dict(claims) != {"isaac_parity": "trace_generated_not_yet_compared"}:
            raise TraceContractError("Isaac claims must contain the canonical parity claim")
        if producer_identity.get("physical_call_count") != TRACE_STEPS:
            raise TraceContractError(
                f"Isaac trajectory_identity.physical_call_count must be {TRACE_STEPS}"
            )
        if producer_identity.get("reference_frame_count") != TRACE_STEPS + 1:
            raise TraceContractError(
                f"Isaac trajectory_identity.reference_frame_count must be {TRACE_STEPS + 1}"
            )
    identity = _normalize_identity(metadata, backend)
    if identity != EXPECTED_IDENTITY:
        differing = {
            key: {"actual": identity.get(key), "expected": expected}
            for key, expected in EXPECTED_IDENTITY.items()
            if identity.get(key) != expected
        }
        raise TraceContractError(f"{backend} trajectory identity mismatch: {differing}")
    return identity


def _validate_array_schema(arrays: Mapping[str, np.ndarray], backend: str) -> None:
    specs = MUJOCO_ARRAY_SPECS if backend == "mujoco" else ISAAC_ARRAY_SPECS
    missing = sorted(set(specs) - set(arrays))
    extra = sorted(set(arrays) - set(specs))
    if missing or extra:
        raise TraceContractError(
            f"{backend} NPZ fields differ: missing={missing}, extra={extra}"
        )
    for name, (tail_shape, expected_dtype) in specs.items():
        array = np.asarray(arrays[name])
        expected_shape = (TRACE_STEPS, *tail_shape)
        if array.shape != expected_shape:
            raise TraceContractError(
                f"{backend} {name} shape {array.shape} != {expected_shape}"
            )
        if array.dtype != expected_dtype:
            raise TraceContractError(
                f"{backend} {name} dtype {array.dtype} != {expected_dtype}"
            )
        if array.dtype.kind in "fiu" and not np.isfinite(array).all():
            raise TraceContractError(f"{backend} {name} contains non-finite values")
    contact_count = np.asarray(arrays["contact_count"])
    has_contact = np.asarray(arrays["has_contact"])
    if np.any(contact_count < 0):
        raise TraceContractError(f"{backend} contact_count contains negative values")
    if not np.array_equal(has_contact, contact_count > 0):
        raise TraceContractError(f"{backend} has_contact disagrees with contact_count")
    for name in ("object_quat_xyzw", "object_reference_quat_xyzw"):
        norms = np.linalg.norm(np.asarray(arrays[name], dtype=np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=QUATERNION_NORM_ATOL):
            maximum = float(np.max(np.abs(norms - 1.0)))
            raise TraceContractError(
                f"{backend} {name} is not normalized; max norm error={maximum}"
            )


def _validate_schedule(arrays: Mapping[str, np.ndarray], backend: str) -> None:
    calls = np.arange(TRACE_STEPS, dtype=np.int64)
    expected_target = np.maximum(calls - 1, 0)
    expected_reference = calls
    expected_source_reference = calls + EXPECTED_IDENTITY["source_start"]
    checks = {
        "target_index": expected_target,
        "reference_index": expected_reference,
        "source_reference_index": expected_source_reference,
    }
    for name, expected in checks.items():
        actual = np.asarray(arrays[name], dtype=np.int64)
        mismatch = np.flatnonzero(actual != expected)
        if mismatch.size:
            call = int(mismatch[0])
            raise TraceContractError(
                f"{backend} schedule mismatch for {name} at call {call}: "
                f"{int(actual[call])} != {int(expected[call])}"
            )
    expected_time = (calls.astype(np.float64) + 1.0) * CONTROL_DT
    actual_time = np.asarray(arrays["sim_time"], dtype=np.float64)
    mismatch = np.flatnonzero(
        ~np.isclose(actual_time, expected_time, rtol=0.0, atol=SIM_TIME_ATOL)
    )
    if mismatch.size:
        call = int(mismatch[0])
        raise TraceContractError(
            f"{backend} sim_time mismatch at call {call}: "
            f"{actual_time[call]} != {expected_time[call]} within {SIM_TIME_ATOL}"
        )


def _max_abs_difference(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    return float(np.max(np.abs(difference)))


def _validate_shared_inputs(
    mujoco: Mapping[str, np.ndarray], isaac: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    contracts = {
        "hand_reference": {
            "atol": HAND_REFERENCE_ATOL,
            "max_abs_difference": _max_abs_difference(
                mujoco["hand_reference"], isaac["hand_reference"]
            ),
        },
        "object_reference_pos_raw": {
            "atol": RAW_OBJECT_REFERENCE_ATOL,
            "max_abs_difference": _max_abs_difference(
                mujoco["object_reference_pos_raw"], isaac["object_reference_pos_raw"]
            ),
        },
        "object_reference_pos_transformed_xy": {
            "atol": TRANSFORMED_OBJECT_REFERENCE_XY_ATOL,
            "max_abs_difference": _max_abs_difference(
                np.asarray(mujoco["object_reference_pos"])[:, :2],
                np.asarray(isaac["object_reference_pos"])[:, :2],
            ),
        },
    }
    for name, evidence in contracts.items():
        if evidence["max_abs_difference"] > evidence["atol"]:
            raise TraceContractError(
                f"shared reference mismatch for {name}: max abs difference "
                f"{evidence['max_abs_difference']} > {evidence['atol']}"
            )
        evidence["passed"] = True
        evidence["rtol"] = 0.0

    orientation = _quaternion_geodesic(
        mujoco["object_reference_quat_xyzw"],
        isaac["object_reference_quat_xyzw"],
    )
    maximum_orientation = float(np.max(orientation))
    contracts["object_reference_orientation"] = {
        "comparison": "quaternion-sign-invariant geodesic angle",
        "max_geodesic_difference_rad": maximum_orientation,
        "atol_rad": OBJECT_REFERENCE_ORIENTATION_GEODESIC_ATOL,
        "passed": maximum_orientation <= OBJECT_REFERENCE_ORIENTATION_GEODESIC_ATOL,
    }
    if maximum_orientation > OBJECT_REFERENCE_ORIENTATION_GEODESIC_ATOL:
        raise TraceContractError(
            "shared reference mismatch for object_reference_orientation: max geodesic "
            f"difference {maximum_orientation} > "
            f"{OBJECT_REFERENCE_ORIENTATION_GEODESIC_ATOL} rad"
        )
    return contracts


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _quaternion_geodesic(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    actual_values = np.asarray(actual, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    actual_unit = actual_values / np.linalg.norm(actual_values, axis=1, keepdims=True)
    reference_unit = reference_values / np.linalg.norm(
        reference_values, axis=1, keepdims=True
    )
    dot = np.clip(np.abs(np.sum(actual_unit * reference_unit, axis=1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _phase_metrics(
    hand_error: np.ndarray,
    object_position_error: np.ndarray,
    object_orientation_error: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected_hand = hand_error[mask]
    selected_object_position = object_position_error[mask]
    selected_object_orientation = object_orientation_error[mask]
    per_dof_rmse = np.sqrt(np.mean(np.square(selected_hand), axis=0))
    return {
        "call_count": int(np.count_nonzero(mask)),
        "hand_translation_norm_rmse_m": _rmse(
            np.linalg.norm(selected_hand[:, :3], axis=1)
        ),
        "wrist_wrapped_angular_norm_rmse_rad": _rmse(
            np.linalg.norm(selected_hand[:, 3:6], axis=1)
        ),
        "finger_joint_rmse_rad": _rmse(selected_hand[:, 6:]),
        "hand_per_dof_rmse": per_dof_rmse.tolist(),
        "object_position_norm_rmse_m": _rmse(
            np.linalg.norm(selected_object_position, axis=1)
        ),
        "object_quaternion_geodesic_rmse_rad": _rmse(
            selected_object_orientation
        ),
    }


def _tracking_metrics(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    hand_error = np.asarray(arrays["hand_qpos"], dtype=np.float64) - np.asarray(
        arrays["hand_reference"], dtype=np.float64
    )
    hand_error[:, 3:6] = _wrap_angle(hand_error[:, 3:6])
    object_position_error = np.asarray(
        arrays["object_pos"], dtype=np.float64
    ) - np.asarray(arrays["object_reference_pos"], dtype=np.float64)
    object_orientation_error = _quaternion_geodesic(
        arrays["object_quat_xyzw"], arrays["object_reference_quat_xyzw"]
    )
    reference_indices = np.asarray(arrays["reference_index"])
    result = {
        "whole_trace": _phase_metrics(
            hand_error,
            object_position_error,
            object_orientation_error,
            np.ones(TRACE_STEPS, dtype=bool),
        ),
        "phases": {},
    }
    for name, (start, stop) in PHASES.items():
        mask = (reference_indices >= start) & (reference_indices <= stop)
        metrics = _phase_metrics(
            hand_error, object_position_error, object_orientation_error, mask
        )
        metrics["post_step_reference_index"] = {"start": start, "stop_inclusive": stop}
        metrics["raw_source_reference_index"] = {
            "start": start + EXPECTED_IDENTITY["source_start"],
            "stop_inclusive": stop + EXPECTED_IDENTITY["source_start"],
        }
        result["phases"][name] = metrics
    return result


def _command_comparison(
    mujoco: Mapping[str, np.ndarray], isaac: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    difference = np.asarray(mujoco["q_target"], dtype=np.float64) - np.asarray(
        isaac["q_target"], dtype=np.float64
    )
    difference[:, 3:6] = _wrap_angle(difference[:, 3:6])
    absolute = np.abs(difference)
    flat_index = int(np.argmax(absolute))
    call, dof = np.unravel_index(flat_index, absolute.shape)
    return {
        "difference_convention": {
            "translation_and_fingers": "ordinary_mujoco_minus_isaac",
            "wrist_euler_dofs_3_to_5": "wrapped_mujoco_minus_isaac_in_minus_pi_to_pi",
        },
        "per_dof_rmse": np.sqrt(np.mean(np.square(difference), axis=0)).tolist(),
        "per_dof_max_abs": np.max(absolute, axis=0).tolist(),
        "largest_discrepancy": {
            "call_index": int(call),
            "post_step_reference_index": int(mujoco["reference_index"][call]),
            "dof_index": int(dof),
            "dof_name": JOINT_NAMES[dof],
            "absolute_difference": float(absolute[call, dof]),
            "signed_mujoco_minus_isaac": float(difference[call, dof]),
        },
    }


def _z_difference_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = values[mask]
    return {
        "call_count": int(np.count_nonzero(mask)),
        "signed_mean_m": float(np.mean(selected)),
        "rmse_m": _rmse(selected),
        "max_abs_m": float(np.max(np.abs(selected))),
    }


def _object_reference_z_comparison(
    mujoco: Mapping[str, np.ndarray], isaac: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    difference = np.asarray(
        mujoco["object_reference_pos"], dtype=np.float64
    )[:, 2] - np.asarray(isaac["object_reference_pos"], dtype=np.float64)[:, 2]
    reference_indices = np.asarray(mujoco["reference_index"])
    result = {
        "difference_convention": "mujoco_transformed_z_minus_isaac_transformed_z",
        "reason_reported_separately": "backend support transforms can differ",
        "whole_trace": _z_difference_summary(
            difference, np.ones(TRACE_STEPS, dtype=bool)
        ),
        "phases": {},
    }
    for name, (start, stop) in PHASES.items():
        mask = (reference_indices >= start) & (reference_indices <= stop)
        result["phases"][name] = _z_difference_summary(difference, mask)
    return result


def _producer_contracts(
    mujoco_metadata: Mapping[str, Any], isaac_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    stability = _require_mapping(mujoco_metadata, "stability_gates", "mujoco")
    missing_gates = sorted(MUJOCO_STABILITY_GATE_KEYS - set(stability))
    extra_gates = sorted(set(stability) - MUJOCO_STABILITY_GATE_KEYS)
    if missing_gates or extra_gates:
        raise TraceContractError(
            "MuJoCo stability_gates fields differ: "
            f"missing={missing_gates}, extra={extra_gates}"
        )
    if any(type(value) is not bool for value in stability.values()):
        raise TraceContractError("MuJoCo stability_gates values must all be boolean")
    diagnostic = mujoco_metadata.get("diagnostic_passed")
    if type(diagnostic) is not bool:
        raise TraceContractError("MuJoCo diagnostic_passed must be boolean")
    expected_diagnostic = all(stability.values())
    if diagnostic != expected_diagnostic:
        raise TraceContractError(
            "MuJoCo diagnostic_passed must equal all canonical stability_gates"
        )

    checks = _require_mapping(isaac_metadata, "checks", "isaac")
    missing_checks = sorted(ISAAC_CHECK_KEYS - set(checks))
    extra_checks = sorted(set(checks) - ISAAC_CHECK_KEYS)
    if missing_checks or extra_checks:
        raise TraceContractError(
            f"Isaac checks fields differ: missing={missing_checks}, extra={extra_checks}"
        )
    if any(type(value) is not bool for value in checks.values()):
        raise TraceContractError("Isaac checks values must all be boolean")
    mujoco_passed = diagnostic
    isaac_passed = bool(all(checks.values()))
    return {
        "mujoco": {
            "diagnostic_passed": diagnostic,
            "stability_gates": dict(stability),
            "passed": mujoco_passed,
        },
        "isaac": {"checks": dict(checks), "passed": isaac_passed},
        "passed": mujoco_passed and isaac_passed,
    }


def _worst_ordering(
    mujoco_metrics: Mapping[str, Any], isaac_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    mujoco_dof = np.asarray(mujoco_metrics["whole_trace"]["hand_per_dof_rmse"])
    isaac_dof = np.asarray(isaac_metrics["whole_trace"]["hand_per_dof_rmse"])
    mujoco_wrist = int(np.argmax(mujoco_dof[3:6])) + 3
    isaac_wrist = int(np.argmax(isaac_dof[3:6])) + 3

    def group_values(per_dof: np.ndarray) -> dict[str, float]:
        return {
            name: _rmse(per_dof[group])
            for name, group in FINGER_GROUPS.items()
        }

    mujoco_groups = group_values(mujoco_dof)
    isaac_groups = group_values(isaac_dof)
    mujoco_group = max(mujoco_groups, key=mujoco_groups.__getitem__)
    isaac_group = max(isaac_groups, key=isaac_groups.__getitem__)
    mujoco_fingers = mujoco_dof[6:]
    isaac_fingers = isaac_dof[6:]
    mujoco_constant = bool(
        np.allclose(
            mujoco_fingers,
            mujoco_fingers[0],
            rtol=0.0,
            atol=FINGER_RANK_CONSTANT_ATOL,
        )
    )
    isaac_constant = bool(
        np.allclose(
            isaac_fingers,
            isaac_fingers[0],
            rtol=0.0,
            atol=FINGER_RANK_CONSTANT_ATOL,
        )
    )
    if mujoco_constant and isaac_constant:
        correlation = 1.0
        correlation_method = "individually_constant_vectors"
        correlation_reason = (
            "both vectors are constant within the rank tolerance, so both encode "
            "the same all-tied ordering regardless of magnitude"
        )
    elif mujoco_constant or isaac_constant:
        correlation = float("nan")
        correlation_method = "undefined_one_sided_constant_vector"
        correlation_reason = (
            "only one vector is constant, so rank correlation is undefined"
        )
    else:
        result = spearmanr(mujoco_fingers, isaac_fingers)
        correlation = float(result.statistic)
        correlation_method = "scipy.stats.spearmanr"
        correlation_reason = None
    correlation_passed = bool(np.isfinite(correlation) and correlation >= 0.8)
    return {
        "worst_wrist_axis": {
            "mujoco": {
                "dof_index": mujoco_wrist,
                "dof_name": JOINT_NAMES[mujoco_wrist],
                "rmse_rad": float(mujoco_dof[mujoco_wrist]),
            },
            "isaac": {
                "dof_index": isaac_wrist,
                "dof_name": JOINT_NAMES[isaac_wrist],
                "rmse_rad": float(isaac_dof[isaac_wrist]),
            },
            "same_identity": mujoco_wrist == isaac_wrist,
        },
        "worst_finger_group": {
            "mujoco": {
                "group": mujoco_group,
                "group_per_dof_rmse_rms_rad": mujoco_groups[mujoco_group],
                "all_groups": mujoco_groups,
            },
            "isaac": {
                "group": isaac_group,
                "group_per_dof_rmse_rms_rad": isaac_groups[isaac_group],
                "all_groups": isaac_groups,
            },
            "same_identity": mujoco_group == isaac_group,
        },
        "finger_per_dof_rmse_spearman": {
            "correlation": correlation if np.isfinite(correlation) else None,
            "minimum_required": 0.8,
            "constant_vector_atol": FINGER_RANK_CONSTANT_ATOL,
            "passed": correlation_passed,
            "method": correlation_method,
            "reason": correlation_reason
            if correlation_reason is not None
            else (None if np.isfinite(correlation) else "correlation undefined"),
        },
    }


def _scale_comparison(
    mujoco_metrics: Mapping[str, Any], isaac_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    gate_passed = True
    for name, threshold in PRIMARY_METRICS.items():
        mujoco_value = float(mujoco_metrics["whole_trace"][name])
        isaac_value = float(isaac_metrics["whole_trace"][name])
        non_negligible = max(mujoco_value, isaac_value) > threshold
        evidence: dict[str, Any] = {
            "mujoco": mujoco_value,
            "isaac": isaac_value,
            "non_negligible_threshold": threshold,
            "non_negligible_in_either_backend": non_negligible,
        }
        if not non_negligible:
            evidence.update(
                applicable=False,
                ratio_mujoco_over_isaac=None,
                within_range=None,
                reason="both backend metrics are at or below the non-negligible threshold",
                gate_passed=True,
            )
        elif isaac_value <= threshold:
            evidence.update(
                applicable=False,
                ratio_mujoco_over_isaac=None,
                within_range=None,
                reason="Isaac denominator is at or below the near-zero threshold",
                gate_passed=False,
            )
            gate_passed = False
        else:
            ratio = mujoco_value / isaac_value
            within = RATIO_MIN <= ratio <= RATIO_MAX
            evidence.update(
                applicable=True,
                ratio_mujoco_over_isaac=ratio,
                within_range=within,
                reason=None,
                gate_passed=within,
            )
            gate_passed = gate_passed and within
        results[name] = evidence
    return {
        "accepted_ratio_range_inclusive": [RATIO_MIN, RATIO_MAX],
        "metrics": results,
        "passed": gate_passed,
    }


def _contact_summary(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    counts = np.asarray(arrays["contact_count"], dtype=np.int64)
    active = np.asarray(arrays["has_contact"], dtype=bool)
    return {
        "total_contact_count": int(np.sum(counts)),
        "maximum_contact_count_per_call": int(np.max(counts)),
        "active_call_fraction": float(np.mean(active)),
        "active_call_count": int(np.count_nonzero(active)),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def compare_traces(
    mujoco_prefix: str | os.PathLike[str],
    isaac_prefix: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate, compare, atomically write, and return one comparison report."""

    mujoco_npz_path, mujoco_json_path = _prefix_paths(mujoco_prefix)
    isaac_npz_path, isaac_json_path = _prefix_paths(isaac_prefix)
    output_path = _output_path(output)
    mujoco_metadata = _load_json(mujoco_json_path)
    isaac_metadata = _load_json(isaac_json_path)
    mujoco_arrays = _load_npz(mujoco_npz_path)
    isaac_arrays = _load_npz(isaac_npz_path)

    mujoco_identity = _validate_metadata(mujoco_metadata, MUJOCO_SCHEMA, "mujoco")
    isaac_identity = _validate_metadata(isaac_metadata, ISAAC_SCHEMA, "isaac")
    if mujoco_identity != isaac_identity:
        raise TraceContractError("normalized MuJoCo and Isaac trajectory identities disagree")
    _validate_array_schema(mujoco_arrays, "mujoco")
    _validate_array_schema(isaac_arrays, "isaac")
    _validate_schedule(mujoco_arrays, "mujoco")
    _validate_schedule(isaac_arrays, "isaac")
    shared_inputs = _validate_shared_inputs(mujoco_arrays, isaac_arrays)

    producer_contracts = _producer_contracts(mujoco_metadata, isaac_metadata)
    command_comparison = _command_comparison(mujoco_arrays, isaac_arrays)
    mujoco_metrics = _tracking_metrics(mujoco_arrays)
    isaac_metrics = _tracking_metrics(isaac_arrays)
    ordering = _worst_ordering(mujoco_metrics, isaac_metrics)
    scale = _scale_comparison(mujoco_metrics, isaac_metrics)
    gates = {
        "producer_contracts": producer_contracts["passed"],
        "input_contracts": True,
        "comparable_scale": scale["passed"],
        "same_worst_wrist_axis": ordering["worst_wrist_axis"]["same_identity"],
        "same_worst_finger_group": ordering["worst_finger_group"]["same_identity"],
        "finger_rank_correlation_at_least_0_8": ordering[
            "finger_per_dof_rmse_spearman"
        ]["passed"],
    }
    failed_gates = [name for name, passed in gates.items() if not passed]
    parity_supported = not failed_gates
    report: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "comparison_status": "passed" if parity_supported else "not_supported",
        "inputs": {
            "mujoco": {
                "npz": str(mujoco_npz_path),
                "json": str(mujoco_json_path),
                "schema": MUJOCO_SCHEMA,
            },
            "isaac": {
                "npz": str(isaac_npz_path),
                "json": str(isaac_json_path),
                "schema": ISAAC_SCHEMA,
            },
            "normalized_shared_identity": mujoco_identity,
        },
        "validation": {
            "trace_steps": TRACE_STEPS,
            "schedule": "command 0,0,1,...,789; post reference 0..790; source reference 440..1230",
            "sim_time_atol": SIM_TIME_ATOL,
            "quaternion_norm_atol": QUATERNION_NORM_ATOL,
            "shared_input_contracts": shared_inputs,
        },
        "producer_contracts": producer_contracts,
        "q_command_comparison": command_comparison,
        "tracking_metrics": {"mujoco": mujoco_metrics, "isaac": isaac_metrics},
        "transformed_object_reference_z_difference": _object_reference_z_comparison(
            mujoco_arrays, isaac_arrays
        ),
        "ordering": ordering,
        "scale_comparison": scale,
        "non_comparable_contact_summaries": {
            "explanation": (
                "MuJoCo contact_count counts contact manifolds; Isaac contact_count counts "
                "thresholded force summaries at 16 hand keypoints. These summaries are not "
                "compared and are not parity gates."
            ),
            "used_as_gate": False,
            "mujoco": _contact_summary(mujoco_arrays),
            "isaac": _contact_summary(isaac_arrays),
        },
        "gates": gates,
        "claims": {
            "parity": "supported_same_order_comparable_scale"
            if parity_supported
            else "not_supported",
            "failed_gates": failed_gates,
            "scope": "evidence classification only; not a training-success claim",
        },
    }
    _atomic_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco-prefix", required=True)
    parser.add_argument("--isaac-prefix", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = compare_traces(
        arguments.mujoco_prefix, arguments.isaac_prefix, arguments.output
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["comparison_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
