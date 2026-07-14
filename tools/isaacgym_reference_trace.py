#!/usr/bin/env python3
"""Generate the fixed MANOHand Isaac Gym reference trace.

The module-level API is deliberately offline-safe: Isaac Gym, Torch, Hydra,
Lance, and the source repository are imported only by :func:`run`, after the
runtime working directory has been selected.  The trajectory identity is part
of this program's contract and is not exposed through the CLI.

Runtime invocation must set ``PYTHONDONTWRITEBYTECODE=1``.  This prevents Python
from writing bytecode into the source checkout while it is being measured.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import traceback
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np


SCHEMA = "manorl.isaacgym.reference_replay.v1"
SEMANTIC_FIXTURE_SCHEMA = "manorl.isaacgym.semantic_fixture.v1"
SEED = 42
TARGET_REPO = Path(__file__).resolve().parents[1]
SOURCE_REPO = TARGET_REPO.parent / "manohand_reconstruction"
RUNTIME_CWD = SOURCE_REPO / "IsaacGymEnvs/isaacgymenvs"
DATASET_PATH = Path(
    "/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_remake/"
    "npy_s02_v3.lance"
)
LANCE_BASE_PATH = DATASET_PATH.parent
LANCE_INDEX_PATH = "cfg/lance_human_p1_remake_npy_s02_v3_index.json"
EXPECTED_DATASET_VERSION = 132
ROW_INDEX = 1
OBJECT_INDEX = 0
UUID = "d5bc2bc6-9458-52d0-bccc-66c9ec21bae3"
FILE_UUID = "e6fe4732-72cd-5ab7-93e6-2e62dc0263a5"
TRAJECTORY_NAME = "cube1_01_009"
OBJECT_TYPE = "cube1"
ACTION_ID = "01"
SEQUENCE_ID = "009"
SOURCE_START = 440
SOURCE_STOP = 1232
REFERENCE_FRAMES = 792
PHYSICS_CALLS = 791
DOFS = 26
MAX_GPU_CONTACT_PAIRS = 1_554_432
NUM_SUBSCENES = 0
REFERENCE_MAX_DEVIATION_DISTANCE = 1_000_000.0

EXPECTED_IDENTITY = {
    "dataset_path": str(DATASET_PATH),
    "row_index": ROW_INDEX,
    "object_index": OBJECT_INDEX,
    "uuid": UUID,
    "file_uuid": FILE_UUID,
    "trajectory_name": TRAJECTORY_NAME,
    "object_type": OBJECT_TYPE,
    "action_id": ACTION_ID,
    "sequence_id": SEQUENCE_ID,
}

HYDRA_OVERRIDES = [
    "task=MANOHand",
    "num_envs=1",
    "physics_engine=physx",
    "pipeline=gpu",
    "sim_device=cuda:0",
    "rl_device=cuda:0",
    "graphics_device_id=0",
    f"num_subscenes={NUM_SUBSCENES}",
    "headless=true",
    "force_render=false",
]

TASK_UPDATES: dict[str, Any] = {
    "task.env.numEnvs": 1,
    "task.env.useBatchMocap": True,
    "task.env.mocapDataSource": "lance",
    "task.env.mocapLanceBasePath": str(LANCE_BASE_PATH),
    "task.env.mocapLanceIndexPath": LANCE_INDEX_PATH,
    "task.env.mocapObjectType": OBJECT_TYPE,
    "task.env.mocapActions": ACTION_ID,
    "task.env.loadMultiObject": False,
    "task.env.sequentialMocapAssignment": "sequential",
    "task.env.balanceEnvsByAction": False,
    "task.env.movementFramePrePadding": 250,
    "task.env.movementFramePostPadding": 250,
    "task.env.useResidualActions": False,
    "task.env.earlyPhaseMocapSteps": 0,
    "task.env.useRelativeControl": False,
    "task.env.actionsMovingAverage": 1.0,
    "task.env.controlFrequencyInv": 1,
    "task.env.forceScale": 0.0,
    "task.env.generate.objectPerturbation.enable": False,
    "task.env.generate.enableDataGeneration": False,
    "task.env.generate.skipExistingTrajectories": False,
    "task.env.generate.batchSynthesis.enable": False,
    "task.env.generate.batchSynthesis.resumeIncomplete": False,
    "task.env.generate.batchSynthesis.autoTerminateOnCompletion": False,
    "task.env.successTracking.enableSuccessTracking": False,
    "task.env.rerunLogging.enable": False,
    "task.env.unseenObject.enable": False,
    "task.env.enableDebugVis": False,
    "task.env.enableContactVisualization": False,
    "task.env.enableCameraSensors": False,
    "task.env.enableCameraOrbit": False,
    "task.env.printNumSuccesses": False,
    "task.env.pointCloudEncoding.isDynamic": False,
    "task.sim.physx.max_gpu_contact_pairs": MAX_GPU_CONTACT_PAIRS,
    # MuJoCo records one fixed 791-call horizon regardless of tracking error.
    # Disable only this source training reset so both traces preserve every call.
    "task.env.maxDeviationDistance": REFERENCE_MAX_DEVIATION_DISTANCE,
}


def task_updates_for_mode(*, semantic_only: bool) -> dict[str, Any]:
    """Keep PD trace and current-source semantic contracts intentionally separate."""

    updates = dict(TASK_UPDATES)
    if semantic_only:
        updates.update({
            "task.env.useResidualActions": True,
            "task.env.earlyPhaseMocapSteps": 100,
        })
    return updates

SOURCE_HASH_PATHS = (
    "IsaacGymEnvs/isaacgymenvs/tasks/mano_hand.py",
    "IsaacGymEnvs/isaacgymenvs/tasks/base/vec_task.py",
    "IsaacGymEnvs/isaacgymenvs/utils/lance_mocap_backend.py",
    "IsaacGymEnvs/isaacgymenvs/cfg/task/MANOHand.yaml",
    "IsaacGymEnvs/isaacgymenvs/cfg/config.yaml",
)

# (shape after the fixed 791-step axis, exact output dtype)
REQUIRED_ARRAY_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "target_index": ((), np.dtype("int64")),
    "reference_index": ((), np.dtype("int64")),
    "source_target_index": ((), np.dtype("int64")),
    "source_reference_index": ((), np.dtype("int64")),
    "sim_time": ((), np.dtype("float64")),
    "q_target": ((26,), np.dtype("float32")),
    "mocap_target_unclamped": ((26,), np.dtype("float32")),
    "hand_qpos": ((26,), np.dtype("float32")),
    "hand_qvel": ((26,), np.dtype("float32")),
    "hand_reference": ((26,), np.dtype("float64")),
    "hand_reference_task_raw": ((26,), np.dtype("float32")),
    "object_pos": ((3,), np.dtype("float32")),
    "object_quat_xyzw": ((4,), np.dtype("float32")),
    "object_linvel": ((3,), np.dtype("float32")),
    "object_angvel": ((3,), np.dtype("float32")),
    "object_reference_pos_raw": ((3,), np.dtype("float64")),
    "object_reference_pos": ((3,), np.dtype("float32")),
    "object_reference_quat_xyzw": ((4,), np.dtype("float32")),
    "hand_palm_pos": ((3,), np.dtype("float32")),
    "hand_palm_quat_xyzw": ((4,), np.dtype("float32")),
    "raw_action": ((26,), np.dtype("float32")),
    "processed_action": ((26,), np.dtype("float32")),
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

# These calls isolate every piecewise transition in the current source task:
# early phase, contact-window start/end, post-window behavior, and terminal.
SEMANTIC_REGULAR_CALLS = (0, 99, 100, 101, 102, 249, 250, 542, 543, 789, 790)
SEMANTIC_DELAYED_RESET_CALL = 791
SEMANTIC_SAMPLE_COUNT = len(SEMANTIC_REGULAR_CALLS) + 1
SEMANTIC_SAMPLE_KIND_POST_STEP = 0
SEMANTIC_SAMPLE_KIND_DELAYED_RESET_POST = 1
SEMANTIC_ARRAY_SPECS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "physical_call": ((), np.dtype("int64")),
    "sample_kind": ((), np.dtype("int8")),
    "pulse_case": ((), np.dtype("int8")),
    "raw_action": ((26,), np.dtype("float32")),
    "processed_action": ((26,), np.dtype("float32")),
    "mocap_target": ((26,), np.dtype("float32")),
    "controller_target": ((26,), np.dtype("float32")),
    "action_trajectory_step": ((), np.dtype("int64")),
    "action_cumulative_offset": ((3,), np.dtype("float32")),
    "action_cumulative_joint_offset": ((20,), np.dtype("float32")),
    "source_raw_observation": ((476,), np.dtype("float32")),
    "mano_dof_pos": ((26,), np.dtype("float32")),
    "mano_dof_lower": ((26,), np.dtype("float32")),
    "mano_dof_upper": ((26,), np.dtype("float32")),
    "hand_position": ((3,), np.dtype("float32")),
    "hand_orientation_xyzw": ((4,), np.dtype("float32")),
    "object_position": ((3,), np.dtype("float32")),
    "object_orientation_xyzw": ((4,), np.dtype("float32")),
    "object_linear_velocity": ((3,), np.dtype("float32")),
    "target_object_position": ((3,), np.dtype("float32")),
    "target_object_orientation_xyzw": ((4,), np.dtype("float32")),
    "target_object_pos_next_5": ((3,), np.dtype("float32")),
    "cumulative_offset": ((3,), np.dtype("float32")),
    "cumulative_joint_offset": ((20,), np.dtype("float32")),
    "point_cloud_relative": ((64, 3), np.dtype("float32")),
    "point_cloud_local_template": ((64, 3), np.dtype("float32")),
    "point_cloud_scale": ((3,), np.dtype("float32")),
    "point_cloud_normalized": ((), np.dtype("bool")),
    "object_geometry": ((12,), np.dtype("float32")),
    "object_support_points": ((256, 3), np.dtype("float32")),
    "object_support_point_count": ((), np.dtype("int64")),
    "table_surface_height": ((), np.dtype("float32")),
    "table_clearance_support_point_cap": ((), np.dtype("int64")),
    "hand_keypoint_positions": ((16, 3), np.dtype("float32")),
    "fingertip_positions": ((5, 3), np.dtype("float32")),
    "hand_keypoint_contact_forces": ((16, 3), np.dtype("float32")),
    "object_contact_force": ((3,), np.dtype("float32")),
    "object_gravity_force": ((), np.dtype("float32")),
    "expected_contact_mask": ((16,), np.dtype("float32")),
    "expected_contact_weights": ((16,), np.dtype("float32")),
    "action_type": ((50,), np.dtype("float32")),
    "action_id": ((), np.dtype("int64")),
    "active_joint_mask": ((20,), np.dtype("bool")),
    "rotation_disabled_mask": ((), np.dtype("bool")),
    "early_phase_start": ((), np.dtype("int64")),
    "progress": ((), np.dtype("int64")),
    "trajectory_step": ((), np.dtype("int64")),
    "contact_start_frame": ((), np.dtype("int64")),
    "contact_end_frame": ((), np.dtype("int64")),
    "reset": ((), np.dtype("bool")),
    "deviation_reset": ((), np.dtype("bool")),
    "deviation_penalty": ((), np.dtype("float32")),
    "reward_total": ((), np.dtype("float32")),
    "reward_distance_x": ((), np.dtype("float32")),
    "reward_distance_y": ((), np.dtype("float32")),
    "reward_distance_z": ((), np.dtype("float32")),
    "reward_rotation": ((), np.dtype("float32")),
    "reward_action_penalty": ((), np.dtype("float32")),
    "reward_position_penalty": ((), np.dtype("float32")),
    "reward_joint_penalty": ((), np.dtype("float32")),
    "reward_contact": ((), np.dtype("float32")),
    "reward_object_stability": ((), np.dtype("float32")),
    "reward_object_speed": ((), np.dtype("float32")),
}

NON_COMPARABLE_FIELDS = [
    "actuator_force_substeps: Isaac Gym exposes no per-internal-PhysX-substep generalized actuator force tensor",
    "contact_count: Isaac counts thresholded force summaries at 16 hand keypoints, whereas MuJoCo counts contact manifolds",
    "contact_capacity_saturated: no per-step saturation signal is exposed by the task",
    "constraint_capacity_saturated: no task tensor corresponds to MuJoCo nefc",
    "exact contact locations and forces: simulator and collision geometry dependent",
    "sim_time: derived from physical-call count and configured dt rather than an authoritative simulator clock tensor",
    "object reference Z: derived from Isaac URDF collision geometry and table configuration",
]


@dataclasses.dataclass(frozen=True)
class SchedulePoint:
    physical_call: int
    target_index: int
    reference_index: int
    source_target_index: int
    source_reference_index: int


@dataclasses.dataclass(frozen=True)
class ArtifactPaths:
    prefix: Path
    final_npz: Path
    final_json: Path
    failure_json: Path
    partial_npz: Path


def normalize_action_ids(actions: Any) -> tuple[str, ...] | None:
    """Normalize action IDs to zero-padded two-character strings."""
    if actions is None:
        return None
    if isinstance(actions, str):
        values: Iterable[Any] = actions.split(",")
    elif isinstance(actions, (int, np.integer)):
        values = (actions,)
    else:
        values = actions
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if not text.isdigit():
            raise ValueError(f"invalid action ID {value!r}")
        normalized.append(text.zfill(2))
    return tuple(normalized)


def derive_row_identity(row: Mapping[str, Any], object_index: int = OBJECT_INDEX) -> str:
    """Derive ``object_action_sequence`` from an authoritative source path."""
    del object_index
    source_path = row["index"].get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("row index must contain a non-empty source_path")
    identity = Path(source_path).parent.name
    if identity.count("_") != 2:
        raise ValueError(f"source path does not encode an object_action_sequence: {source_path!r}")
    return identity


def validate_row_identity(row: Mapping[str, Any]) -> str:
    """Validate every settled identity-bearing field in the physical Lance row."""
    index = row["index"]
    metadata = row["trajectory_metadata"]
    if index["uuid"] != UUID:
        raise ValueError("row UUID mismatch")
    if index["file_uuid"] != FILE_UUID:
        raise ValueError("row file UUID mismatch")
    identity = derive_row_identity(row)
    if identity != TRAJECTORY_NAME:
        raise ValueError(f"row identity mismatch: {identity!r}")
    if metadata["object_names"][OBJECT_INDEX] != OBJECT_TYPE:
        raise ValueError(f"object index 0 is not {OBJECT_TYPE}")
    if metadata["hand_names"] != ["right"]:
        raise ValueError("accepted row must contain exactly the right hand")
    if int(metadata["total_frames"]) != 1373:
        raise ValueError("accepted row must contain 1373 source frames")
    if int(metadata["data_fps"]) != 111:
        raise ValueError("accepted row must be 111 Hz")
    expected_move = [
        {"object_name": OBJECT_TYPE, "start_frame": 690, "end_frame": 982}
    ]
    if metadata["trajectory_info"]["object_move"] != expected_move:
        raise ValueError("movement-frame identity mismatch")
    return identity


def _entry_dataset_path(entry: Mapping[str, Any], lance_root: os.PathLike[str] | str) -> str:
    path = entry.get("dataset_path")
    if path is None:
        path = os.path.join(os.fspath(lance_root), str(entry["dataset_name"]))
    return os.path.realpath(os.fspath(path))


def select_strict_index_entry(
    entries: Iterable[Mapping[str, Any]],
    lance_root: os.PathLike[str] | str,
    object_types: Iterable[Any],
    actions: Any,
) -> dict[str, Any]:
    """Return the sole exact index match; reject broad matches and ambiguity."""
    if {str(value) for value in object_types} != {OBJECT_TYPE}:
        raise ValueError(f"wrapper requires object_types={{{OBJECT_TYPE!r}}}")
    if set(normalize_action_ids(actions) or ()) != {ACTION_ID}:
        raise ValueError(f"wrapper requires action filter {{{ACTION_ID!r}}}")

    expected_path = os.path.realpath(DATASET_PATH)
    matches: list[dict[str, Any]] = []
    for entry in entries:
        try:
            is_match = (
                _entry_dataset_path(entry, lance_root) == expected_path
                and int(entry["row_index"]) == ROW_INDEX
                and int(entry["object_index"]) == OBJECT_INDEX
                and entry.get("uuid") == UUID
                and entry.get("file_uuid") == FILE_UUID
                and entry.get("trajectory_name") == TRAJECTORY_NAME
                and entry.get("object_type") == OBJECT_TYPE
                and str(entry.get("action_id")).zfill(2) == ACTION_ID
                and str(entry.get("sequence_id")).zfill(3) == SEQUENCE_ID
            )
        except (KeyError, TypeError, ValueError):
            is_match = False
        if is_match:
            matches.append(dict(entry))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one accepted index entry, found {len(matches)}"
        )
    return matches[0]


def schedule_point(physical_call: int) -> SchedulePoint:
    """Map one of the exactly 791 physical calls to target/reference frames."""
    if not 0 <= physical_call < PHYSICS_CALLS:
        raise ValueError(f"physical_call must be in [0,{PHYSICS_CALLS})")
    target = max(physical_call - 1, 0)
    reference = physical_call
    return SchedulePoint(
        physical_call=physical_call,
        target_index=target,
        reference_index=reference,
        source_target_index=SOURCE_START + target,
        source_reference_index=SOURCE_START + reference,
    )


def validate_terminal_policy(
    physical_call: int,
    reset: bool,
    trajectory_complete: bool,
    termination_reason_code: int,
) -> None:
    """Require no early reset and the sole completion reset on call 790."""
    if physical_call < PHYSICS_CALLS - 1:
        if reset or trajectory_complete:
            raise RuntimeError(f"early reset on physical call {physical_call}")
        return
    if physical_call != PHYSICS_CALLS - 1:
        raise ValueError("terminal policy only covers the fixed replay schedule")
    if not reset or not trajectory_complete or termination_reason_code != 1:
        raise RuntimeError(
            "final call must terminate through trajectory completion with reason 1"
        )


def artifact_paths(output: os.PathLike[str] | str) -> ArtifactPaths:
    """Resolve output before runtime chdir and derive disjoint artifact paths."""
    prefix = Path(output).expanduser()
    if not prefix.is_absolute():
        prefix = Path.cwd() / prefix
    prefix = prefix.resolve()
    if prefix.suffix in {".npz", ".json"}:
        prefix = prefix.with_suffix("")
    return ArtifactPaths(
        prefix=prefix,
        final_npz=Path(f"{prefix}.npz"),
        final_json=Path(f"{prefix}.json"),
        failure_json=Path(f"{prefix}.failure.json"),
        partial_npz=Path(f"{prefix}.partial.npz"),
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _temporary_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        return Path(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_failure_artifacts(
    paths: ArtifactPaths,
    failure: Mapping[str, Any],
    partial_arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Atomically write only failure-specific names, never final artifact names."""
    if paths.failure_json in {paths.final_json, paths.final_npz} or paths.partial_npz in {
        paths.final_json,
        paths.final_npz,
    }:
        raise AssertionError("failure artifact paths overlap final artifact paths")
    _atomic_json(paths.failure_json, failure)
    if partial_arrays:
        temporary = _temporary_npz(paths.partial_npz, partial_arrays)
        os.replace(temporary, paths.partial_npz)


def validate_trace_schema(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    """Validate the fixed JSON identity and every NPZ field, shape, and dtype."""
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")
    identity = metadata.get("trajectory_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("trajectory_identity must be an object")
    expected_identity = {
        "dataset_path": str(DATASET_PATH),
        "row_index": ROW_INDEX,
        "object_index": OBJECT_INDEX,
        "uuid": UUID,
        "file_uuid": FILE_UUID,
        "identity": TRAJECTORY_NAME,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "source_slice": {"start": SOURCE_START, "stop": SOURCE_STOP, "stop_exclusive": True},
        "reference_frame_count": REFERENCE_FRAMES,
        "physical_call_count": PHYSICS_CALLS,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"trajectory_identity.{key} mismatch")
    for section in (
        "lance_index_entry",
        "source_repository",
        "config",
        "checks",
        "action_invariance",
        "metrics",
        "non_comparable_fields",
        "claims",
    ):
        if section not in metadata:
            raise ValueError(f"missing JSON section {section!r}")
    if metadata["claims"].get("isaac_parity") != "trace_generated_not_yet_compared":
        raise ValueError("unsupported Isaac parity claim")
    missing = sorted(set(REQUIRED_ARRAY_SPECS) - set(arrays))
    extra = sorted(set(arrays) - set(REQUIRED_ARRAY_SPECS))
    if missing or extra:
        raise ValueError(f"NPZ fields differ: missing={missing}, extra={extra}")
    for name, (tail_shape, expected_dtype) in REQUIRED_ARRAY_SPECS.items():
        array = np.asarray(arrays[name])
        expected_shape = (PHYSICS_CALLS, *tail_shape)
        if array.shape != expected_shape:
            raise ValueError(f"{name} shape {array.shape} != {expected_shape}")
        if array.dtype != expected_dtype:
            raise ValueError(f"{name} dtype {array.dtype} != {expected_dtype}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")


def validate_semantic_fixture_schema(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    """Validate the sparse source-state contract consumed by target semantics."""

    if metadata.get("schema") != SEMANTIC_FIXTURE_SCHEMA:
        raise ValueError(f"semantic fixture schema must be {SEMANTIC_FIXTURE_SCHEMA!r}")
    if metadata.get("regular_sample_calls") != list(SEMANTIC_REGULAR_CALLS):
        raise ValueError("semantic fixture sample calls differ from the settled phase boundaries")
    identity = metadata.get("trajectory_identity")
    if not isinstance(identity, Mapping) or identity.get("identity") != TRAJECTORY_NAME:
        raise ValueError("semantic fixture identity mismatch")
    discriminator = metadata.get("termination_discriminator")
    if not isinstance(discriminator, Mapping) or set(discriminator.get("cases", ())) != {
        "below_threshold",
        "above_threshold",
        "early_above_threshold",
        "trajectory_complete",
    }:
        raise ValueError("semantic fixture lacks the current-source termination discriminator")
    configuration = metadata.get("configuration")
    if not isinstance(configuration, Mapping) or configuration.get("useResidualActions") is not True or configuration.get(
        "earlyPhaseMocapSteps"
    ) != 100:
        raise ValueError("semantic fixture must use the current-source residual/early configuration")
    pulse_cases = metadata.get("pulse_cases")
    if not isinstance(pulse_cases, Mapping) or set(pulse_cases) != {
        str(PULSE_NONE),
        str(PULSE_EARLY_MASK),
        str(PULSE_TRANSITION),
        str(PULSE_ACCUMULATE_POSITIVE),
        str(PULSE_ACCUMULATE_NEGATIVE),
    }:
        raise ValueError("semantic fixture lacks the bounded residual action pulse contract")
    missing = sorted(set(SEMANTIC_ARRAY_SPECS) - set(arrays))
    extra = sorted(set(arrays) - set(SEMANTIC_ARRAY_SPECS))
    if missing or extra:
        raise ValueError(f"semantic fixture fields differ: missing={missing}, extra={extra}")
    for name, (tail_shape, expected_dtype) in SEMANTIC_ARRAY_SPECS.items():
        array = np.asarray(arrays[name])
        expected_shape = (SEMANTIC_SAMPLE_COUNT, *tail_shape)
        if array.shape != expected_shape:
            raise ValueError(f"semantic fixture {name} shape {array.shape} != {expected_shape}")
        if array.dtype != expected_dtype:
            raise ValueError(f"semantic fixture {name} dtype {array.dtype} != {expected_dtype}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"semantic fixture {name} contains non-finite values")
    expected_calls = [*SEMANTIC_REGULAR_CALLS, SEMANTIC_DELAYED_RESET_CALL]
    if arrays["physical_call"].tolist() != expected_calls:
        raise ValueError("semantic fixture physical-call array differs from metadata")
    expected_kinds = [SEMANTIC_SAMPLE_KIND_POST_STEP] * len(SEMANTIC_REGULAR_CALLS) + [
        SEMANTIC_SAMPLE_KIND_DELAYED_RESET_POST
    ]
    if arrays["sample_kind"].tolist() != expected_kinds:
        raise ValueError("semantic fixture sample kinds do not distinguish delayed reset")
    expected_pulses = {
        99: PULSE_EARLY_MASK,
        100: PULSE_TRANSITION,
        101: PULSE_ACCUMULATE_POSITIVE,
        102: PULSE_ACCUMULATE_NEGATIVE,
    }
    for call, pulse in expected_pulses.items():
        index = arrays["physical_call"].tolist().index(call)
        if int(arrays["pulse_case"][index]) != pulse:
            raise ValueError(f"semantic fixture call {call} lacks required residual pulse {pulse}")


def _write_semantic_fixture(
    output: os.PathLike[str] | str,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> tuple[Path, Path]:
    paths = artifact_paths(output)
    if paths.prefix == SOURCE_REPO or SOURCE_REPO in paths.prefix.parents:
        raise ValueError("semantic fixture output must be outside the protected sibling source repository")
    validate_semantic_fixture_schema(metadata, arrays)
    if paths.final_npz.exists() or paths.final_json.exists():
        raise FileExistsError("refusing to replace an existing semantic fixture")
    npz_temporary = _temporary_npz(paths.final_npz, arrays)
    try:
        _atomic_json(paths.final_json, metadata)
        os.replace(npz_temporary, paths.final_npz)
    except BaseException:
        try:
            npz_temporary.unlink()
        except FileNotFoundError:
            pass
        try:
            paths.final_json.unlink()
        except FileNotFoundError:
            pass
        raise
    return paths.final_npz, paths.final_json


def _required_attr(owner: Any, name: str) -> Any:
    if not hasattr(owner, name):
        raise AttributeError(f"MANOHand missing required authoritative attribute {name!r}")
    return getattr(owner, name)


def _scalar(tensor: Any, dtype: type[int] | type[float] | type[bool]) -> Any:
    value = tensor.item() if hasattr(tensor, "item") else tensor
    return dtype(value)


def _numpy(tensor: Any, dtype: str) -> np.ndarray:
    value = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    return np.asarray(value, dtype=dtype).copy()


def _git(source_repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.rstrip("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_source_state(source_repo: Path = SOURCE_REPO) -> dict[str, Any]:
    """Capture the source checkout state that must remain invariant."""
    return {
        "path": str(source_repo),
        "commit": _git(source_repo, "rev-parse", "HEAD"),
        "status": _git(source_repo, "status", "--porcelain=v1", "--untracked-files=all"),
        "staged_status": _git(source_repo, "diff", "--cached", "--name-status"),
        "hashes": {
            relative: _sha256(source_repo / relative) for relative in SOURCE_HASH_PATHS
        },
    }


def _source_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return before == after


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _stack_records(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not records:
        return {}
    keys = tuple(records[0])
    if any(tuple(record) != keys for record in records):
        raise ValueError("record fields changed during replay")
    return {name: np.stack([np.asarray(record[name]) for record in records]) for name in keys}


def _write_success(paths: ArtifactPaths, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    validate_trace_schema(metadata, arrays)
    paths.final_json.parent.mkdir(parents=True, exist_ok=True)
    if paths.final_npz.exists() or paths.final_json.exists():
        raise FileExistsError("refusing to replace an existing final trace artifact")
    npz_temporary = _temporary_npz(paths.final_npz, arrays)
    fd, json_temporary_name = tempfile.mkstemp(
        prefix=f".{paths.final_json.name}.", suffix=".tmp", dir=paths.final_json.parent
    )
    json_temporary = Path(json_temporary_name)
    installed: list[Path] = []
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(npz_temporary, paths.final_npz)
        installed.append(paths.final_npz)
        os.replace(json_temporary, paths.final_json)
        installed.append(paths.final_json)
    except BaseException:
        for path in installed:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for temporary in (npz_temporary, json_temporary):
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
        raise


def _assert_resolved_config(
    task_cfg: Mapping[str, Any], *, residual_actions: bool = False, early_phase_steps: int = 0
) -> None:
    checks = {
        "task name": task_cfg["name"] == "MANOHand",
        "physics engine": task_cfg["physics_engine"] == "physx",
        "one environment": task_cfg["env"]["numEnvs"] == 1,
        "residual action mode": task_cfg["env"]["useResidualActions"] is residual_actions,
        "early mocap phase": task_cfg["env"]["earlyPhaseMocapSteps"] == early_phase_steps,
        "deviation termination disabled": task_cfg["env"]["maxDeviationDistance"]
        == REFERENCE_MAX_DEVIATION_DISTANCE,
        "control frequency": task_cfg["env"]["controlFrequencyInv"] == 1,
        "dt": task_cfg["sim"]["dt"] == 0.005,
        "substeps": task_cfg["sim"]["substeps"] == 2,
        "GPU pipeline": task_cfg["sim"]["use_gpu_pipeline"] is True,
        "PhysX GPU": task_cfg["sim"]["physx"]["use_gpu"] is True,
        "bounded GPU contact capacity": task_cfg["sim"]["physx"]["max_gpu_contact_pairs"]
        == MAX_GPU_CONTACT_PAIRS,
        "single GPU scene": task_cfg["sim"]["physx"]["num_subscenes"] == NUM_SUBSCENES,
        "contact collection": task_cfg["sim"]["physx"]["contact_collection"] == 1,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"resolved configuration mismatch: {failures}")


def _install_strict_find(backend: Any, lance_module: Any, patch_state: dict[str, Any]) -> Any:
    original = backend.LanceTrajectoryIndex.find

    def strict_find(self: Any, object_types: Iterable[Any], actions: Any) -> list[dict[str, Any]]:
        item = select_strict_index_entry(self.entries, self.lance_root, object_types, actions)
        dataset = lance_module.dataset(str(DATASET_PATH))
        version = int(dataset.version)
        if version != EXPECTED_DATASET_VERSION:
            raise RuntimeError(
                f"Lance dataset version {version} != required {EXPECTED_DATASET_VERSION}"
            )
        table = dataset.take(
            [ROW_INDEX],
            columns=["index", "trajectory_metadata", "hands", "objects", "timestamp"],
        )
        if table.num_rows != 1:
            raise RuntimeError("row 1 lookup did not return exactly one row")
        row = table.to_pylist()[0]
        validate_row_identity(row)

        reference = backend.make_lance_reference(item, self.lance_root)
        parsed = backend.parse_lance_reference(reference)
        expected_parsed = {
            "dataset_path": os.path.realpath(DATASET_PATH),
            "row_index": ROW_INDEX,
            "object_index": OBJECT_INDEX,
            "uuid": UUID,
            "file_uuid": FILE_UUID,
            "trajectory_name": TRAJECTORY_NAME,
        }
        actual_parsed = dict(parsed)
        actual_parsed["dataset_path"] = os.path.realpath(actual_parsed["dataset_path"])
        for key, expected in expected_parsed.items():
            if actual_parsed.get(key) != expected:
                raise RuntimeError(f"Lance reference round-trip mismatch for {key}")
        accepted = dict(item)
        accepted["loader_key"] = reference
        accepted["path"] = reference
        patch_state.update(
            dataset_version=version,
            row_identity=TRAJECTORY_NAME,
            accepted_entry=dict(accepted),
            lance_reference=reference,
            matching_count=1,
        )
        return [accepted]

    backend.LanceTrajectoryIndex.find = strict_find
    return original


def _prove_action_invariance(env: Any, torch: Any) -> dict[str, Any]:
    names = (
        "trajectory_steps",
        "progress_buf",
        "cumulative_offset",
        "cumulative_joint_offset",
        "actions",
        "mocap_targets",
        "cur_targets",
        "prev_targets",
    )
    snapshot = {name: _required_attr(env, name).clone() for name in names}

    def restore() -> None:
        for name, saved in snapshot.items():
            _required_attr(env, name).copy_(saved)

    zero = torch.zeros((1, DOFS), device=env.device, dtype=torch.float32)
    nonzero = torch.linspace(-1.0, 1.0, DOFS, device=env.device, dtype=torch.float32).reshape(1, DOFS)
    env.pre_physics_step(zero)
    zero_target = env.cur_targets.clone()
    restore()
    env.pre_physics_step(nonzero)
    nonzero_target = env.cur_targets.clone()
    processed_nonzero = bool(torch.count_nonzero(env.actions).item() > 0)
    target_equal = bool(torch.equal(zero_target, nonzero_target))
    max_difference = float(torch.max(torch.abs(zero_target - nonzero_target)).item())
    offset_zero = bool(torch.count_nonzero(env.cumulative_offset).item() == 0)
    joint_offset_zero = bool(torch.count_nonzero(env.cumulative_joint_offset).item() == 0)
    residual_disabled = env.use_residual_actions is False
    restore()
    result = {
        "same_state_zero_nonzero_target_bitwise_equal": target_equal,
        "max_absolute_target_difference": max_difference,
        "processed_nonzero_action_has_nonzero": processed_nonzero,
        "cumulative_offset_zero": offset_zero,
        "cumulative_joint_offset_zero": joint_offset_zero,
        "useResidualActions": env.use_residual_actions,
    }
    if not all((target_equal, processed_nonzero, offset_zero, joint_offset_zero, residual_disabled)):
        raise RuntimeError(f"same-state residual-off invariance failed: {result}")
    return result


def _compute_metrics(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    numeric_finite = all(
        np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "fc"
    )
    if not numeric_finite:
        raise RuntimeError("trace contains non-finite numeric values")
    hand_error = arrays["hand_qpos"].astype(np.float64) - arrays["hand_reference"]
    hand_translation_error = hand_error[:, :3]
    hand_angular_error = (hand_error[:, 3:6] + np.pi) % (2.0 * np.pi) - np.pi
    finger_error = hand_error[:, 6:]
    per_dof_error = np.concatenate(
        (hand_translation_error, hand_angular_error, finger_error), axis=1
    )
    object_position_error = (
        arrays["object_pos"].astype(np.float64)
        - arrays["object_reference_pos"].astype(np.float64)
    )
    actual_quat = arrays["object_quat_xyzw"].astype(np.float64)
    reference_quat = arrays["object_reference_quat_xyzw"].astype(np.float64)
    actual_norm = np.linalg.norm(actual_quat, axis=1)
    reference_norm = np.linalg.norm(reference_quat, axis=1)
    palm_norm = np.linalg.norm(arrays["hand_palm_quat_xyzw"].astype(np.float64), axis=1)
    quaternion_errors = {
        "object_quaternion_norm_max_error": float(np.max(np.abs(actual_norm - 1.0))),
        "reference_quaternion_norm_max_error": float(
            np.max(np.abs(reference_norm - 1.0))
        ),
        "hand_palm_quaternion_norm_max_error": float(np.max(np.abs(palm_norm - 1.0))),
    }
    if max(quaternion_errors.values()) > 1.0e-3:
        raise RuntimeError(f"quaternion norm tolerance exceeded: {quaternion_errors}")
    actual_unit = actual_quat / actual_norm[:, None]
    reference_unit = reference_quat / reference_norm[:, None]
    quat_dot = np.clip(np.abs(np.sum(actual_unit * reference_unit, axis=1)), 0.0, 1.0)
    angular_error = 2.0 * np.arccos(quat_dot)
    return {
        "all_numeric_arrays_finite": True,
        "hand_translation_rmse_m": float(
            np.sqrt(np.mean(np.sum(hand_translation_error**2, axis=1)))
        ),
        "hand_angular_rmse_rad": float(
            np.sqrt(np.mean(np.sum(hand_angular_error**2, axis=1)))
        ),
        "finger_joint_rmse_rad": float(np.sqrt(np.mean(finger_error**2))),
        "hand_per_dof_rmse": np.sqrt(np.mean(per_dof_error**2, axis=0)).tolist(),
        "hand_per_dof_max_abs": np.max(np.abs(per_dof_error), axis=0).tolist(),
        "object_position_rmse_m": float(
            np.sqrt(np.mean(np.sum(object_position_error**2, axis=1)))
        ),
        "object_position_max_m": float(
            np.max(np.linalg.norm(object_position_error, axis=1))
        ),
        "object_orientation_error_rad_mean": float(np.mean(angular_error)),
        "object_orientation_error_rad_max": float(np.max(angular_error)),
        **quaternion_errors,
    }


def _capture_record(
    env: Any,
    pre_capture: Mapping[str, np.ndarray],
    schedule: SchedulePoint,
    progress_before: int,
    trajectory_before: int,
    trajectory_after_pre: int,
    hand_reference_raw: np.ndarray,
    hand_reference_unwrapped: np.ndarray,
    object_reference_pos_raw: np.ndarray,
) -> dict[str, Any]:
    reset = _scalar(_required_attr(env, "reset_buf")[0], bool)
    reason = _scalar(_required_attr(env, "termination_reason_code")[0], int)
    complete = _scalar(_required_attr(env, "trajectory_complete_reset_mask")[0], bool)
    deviation_owner = getattr(env, "deviation_reset_mask", None)
    deviation = False if deviation_owner is None else _scalar(deviation_owner[0], bool)
    magnitudes = _required_attr(env, "contact_force_magnitudes")[0]
    threshold = float(_required_attr(env, "contact_force_threshold"))
    contact_count = int((magnitudes >= threshold).sum().item())
    progress_after = _scalar(_required_attr(env, "progress_buf")[0], int)
    trajectory_after_post = _scalar(_required_attr(env, "trajectory_steps")[0], int)
    if progress_after != schedule.physical_call + 1:
        raise RuntimeError("progress schedule diverged")
    if trajectory_after_post != schedule.reference_index:
        raise RuntimeError("reference schedule diverged")
    validate_terminal_policy(schedule.physical_call, reset, complete, reason)

    return {
        "target_index": np.int64(schedule.target_index),
        "reference_index": np.int64(schedule.reference_index),
        "source_target_index": np.int64(schedule.source_target_index),
        "source_reference_index": np.int64(schedule.source_reference_index),
        "sim_time": np.float64((schedule.physical_call + 1) * 0.005),
        "q_target": pre_capture["q_target"],
        "mocap_target_unclamped": pre_capture["mocap_target_unclamped"],
        "hand_qpos": _numpy(_required_attr(env, "mano_dof_pos")[0], "float32"),
        "hand_qvel": _numpy(_required_attr(env, "mano_dof_vel")[0], "float32"),
        "hand_reference": np.asarray(hand_reference_unwrapped, dtype="float64"),
        "hand_reference_task_raw": np.asarray(hand_reference_raw, dtype="float32"),
        "object_pos": _numpy(_required_attr(env, "object_positions")[0], "float32"),
        "object_quat_xyzw": _numpy(_required_attr(env, "object_orientations")[0], "float32"),
        "object_linvel": _numpy(_required_attr(env, "object_linvels")[0], "float32"),
        "object_angvel": _numpy(_required_attr(env, "object_angvels")[0], "float32"),
        "object_reference_pos_raw": np.asarray(object_reference_pos_raw, dtype="float64"),
        "object_reference_pos": _numpy(_required_attr(env, "target_object_positions")[0], "float32"),
        "object_reference_quat_xyzw": _numpy(_required_attr(env, "target_object_orientations")[0], "float32"),
        "hand_palm_pos": _numpy(_required_attr(env, "hand_positions")[0], "float32"),
        "hand_palm_quat_xyzw": _numpy(_required_attr(env, "hand_orientations")[0], "float32"),
        "raw_action": pre_capture["raw_action"],
        "processed_action": pre_capture["processed_action"],
        "cumulative_offset": pre_capture["cumulative_offset"],
        "cumulative_joint_offset": pre_capture["cumulative_joint_offset"],
        "progress_before_pre": np.int64(progress_before),
        "progress_after_post": np.int64(progress_after),
        "trajectory_step_before_pre": np.int64(trajectory_before),
        "trajectory_step_after_pre": np.int64(trajectory_after_pre),
        "trajectory_step_after_post": np.int64(trajectory_after_post),
        "reset": np.bool_(reset),
        "termination_reason_code": np.int32(reason),
        "termination_object_target_dist": np.float32(
            _scalar(_required_attr(env, "termination_object_target_dist")[0], float)
        ),
        "trajectory_complete_reset_mask": np.bool_(complete),
        "deviation_reset_mask": np.bool_(deviation),
        "reward": np.float32(_scalar(_required_attr(env, "rew_buf")[0], float)),
        "contact_keypoint_force_xyz": _numpy(
            _required_attr(env, "hand_keypoint_contact_forces")[0], "float32"
        ),
        "contact_keypoint_force_magnitude": _numpy(magnitudes, "float32"),
        "object_contact_force_xyz": _numpy(
            _required_attr(env, "object_contact_forces")[0], "float32"
        ),
        "object_contact_force_magnitude": np.float32(
            _scalar(_required_attr(env, "object_contact_magnitudes")[0], float)
        ),
        "expected_contact_mask": _numpy(
            _required_attr(env, "expected_contact_mask_tensor")[0], "float32"
        ),
        "contact_count": np.int64(contact_count),
        "has_contact": np.bool_(contact_count > 0),
    }


def _capture_semantic_record(
    env: Any,
    record: Mapping[str, Any],
    pre_capture: Mapping[str, np.ndarray],
    physical_call: int,
    *,
    sample_kind: int,
    pulse_case: int,
) -> dict[str, Any]:
    """Capture source-resolved inputs and named reward outputs at one phase boundary."""

    if bool(record["deviation_reset_mask"]):
        raise RuntimeError("fixed-horizon semantic fixture cannot include a deviation reset")
    point_cloud = env.get_point_cloud_observation(mode="raw")[0]
    fingertip_positions = env.observation_encoder._compute_finger_tip_positions_world()[0]
    active_joint_mask = env._get_active_joint_mask()[0]
    gravity_forces = _required_attr(env, "object_gravity_forces")
    if gravity_forces is None:
        raise RuntimeError("MANOHand did not expose object gravity force for reward semantics")
    point_template = _required_attr(_required_attr(env, "pc_encoder"), "pointcloud_template")
    if point_template is None:
        raise RuntimeError("MANOHand did not expose a static point-cloud template")
    point_normalized = bool(_required_attr(env, "pc_normalize"))
    point_scale = getattr(env.pc_encoder, "pc_scale", None)
    if point_normalized and point_scale is None:
        raise RuntimeError("normalized source point-cloud template has no source scale")
    if point_scale is None:
        point_scale = env.pc_encoder.pointcloud_template.new_ones(3)
    support_points = env.observation_encoder._get_object_support_points(
        OBJECT_TYPE, dtype=point_template.dtype
    )
    support_count = int(support_points.shape[0])
    if not 1 <= support_count <= 256:
        raise RuntimeError(f"source support-point count {support_count} is outside fixture capacity")
    padded_support = point_template.new_zeros((256, 3))
    padded_support[:support_count] = support_points
    trajectory_info = _required_attr(env, "trajectory_manager").get_trajectory_info(0)
    action_id = trajectory_info.get("action_id") if trajectory_info else None
    if action_id is None or not str(action_id).isdigit():
        raise RuntimeError("source trajectory manager did not expose a numeric action_id")
    return {
        "physical_call": np.int64(physical_call),
        "sample_kind": np.int8(sample_kind),
        "pulse_case": np.int8(pulse_case),
        "raw_action": pre_capture["raw_action"],
        "processed_action": pre_capture["processed_action"],
        "mocap_target": pre_capture["mocap_target_unclamped"],
        "controller_target": pre_capture["q_target"],
        "action_trajectory_step": pre_capture["trajectory_step"],
        "action_cumulative_offset": pre_capture["action_cumulative_offset"],
        "action_cumulative_joint_offset": pre_capture["action_cumulative_joint_offset"],
        "source_raw_observation": _numpy(_required_attr(env, "obs_buf")[0], "float32"),
        "mano_dof_pos": _numpy(_required_attr(env, "mano_dof_pos")[0], "float32"),
        "mano_dof_lower": _numpy(_required_attr(env, "mano_dof_lower_limits"), "float32"),
        "mano_dof_upper": _numpy(_required_attr(env, "mano_dof_upper_limits"), "float32"),
        "hand_position": _numpy(_required_attr(env, "hand_positions")[0], "float32"),
        "hand_orientation_xyzw": _numpy(_required_attr(env, "hand_orientations")[0], "float32"),
        "object_position": _numpy(_required_attr(env, "object_positions")[0], "float32"),
        "object_orientation_xyzw": _numpy(_required_attr(env, "object_orientations")[0], "float32"),
        "object_linear_velocity": _numpy(_required_attr(env, "object_linvels")[0], "float32"),
        "target_object_position": _numpy(_required_attr(env, "target_object_positions")[0], "float32"),
        "target_object_orientation_xyzw": _numpy(
            _required_attr(env, "target_object_orientations")[0], "float32"
        ),
        "target_object_pos_next_5": _numpy(
            _required_attr(env, "target_object_pos_next_5")[0], "float32"
        ),
        "cumulative_offset": _numpy(_required_attr(env, "cumulative_offset")[0], "float32"),
        "cumulative_joint_offset": _numpy(
            _required_attr(env, "cumulative_joint_offset")[0], "float32"
        ),
        "point_cloud_relative": _numpy(point_cloud, "float32").reshape(64, 3),
        "point_cloud_local_template": _numpy(point_template, "float32"),
        "point_cloud_scale": _numpy(point_scale, "float32"),
        "point_cloud_normalized": np.bool_(point_normalized),
        "object_geometry": _numpy(_required_attr(env, "object_geometry")[0], "float32"),
        "object_support_points": _numpy(padded_support, "float32"),
        "object_support_point_count": np.int64(support_count),
        "table_surface_height": np.float32(_required_attr(env, "table_surface_height")),
        "table_clearance_support_point_cap": np.int64(
            env.observation_encoder.table_clearance_support_point_cap
        ),
        "hand_keypoint_positions": _numpy(
            _required_attr(env, "hand_keypoint_positions")[0], "float32"
        ),
        "fingertip_positions": _numpy(fingertip_positions, "float32"),
        "hand_keypoint_contact_forces": _numpy(
            _required_attr(env, "hand_keypoint_contact_forces")[0], "float32"
        ),
        "object_contact_force": _numpy(
            _required_attr(env, "object_contact_forces")[0], "float32"
        ),
        "object_gravity_force": np.float32(_scalar(gravity_forces[0], float)),
        "expected_contact_mask": _numpy(
            _required_attr(env, "expected_contact_mask_tensor")[0], "float32"
        ),
        "expected_contact_weights": _numpy(
            _required_attr(env, "contact_reward_calculator").expected_contact_weights[0], "float32"
        ),
        "action_type": _numpy(_required_attr(env, "action_type")[0], "float32"),
        "action_id": np.int64(int(action_id)),
        "active_joint_mask": _numpy(active_joint_mask, "bool"),
        "rotation_disabled_mask": np.bool_(
            _scalar(_required_attr(env, "rotation_disabled_mask")[0], bool)
        ),
        "early_phase_start": np.int64(
            _scalar(env._get_early_phase_starts_tensor()[0], int)
        ),
        "progress": np.int64(_scalar(_required_attr(env, "progress_buf")[0], int)),
        "trajectory_step": np.int64(_scalar(_required_attr(env, "trajectory_steps")[0], int)),
        "contact_start_frame": np.int64(_scalar(_required_attr(env, "env_contact_start_frames")[0], int)),
        "contact_end_frame": np.int64(_scalar(_required_attr(env, "env_contact_end_frames")[0], int)),
        "reset": np.bool_(record["reset"]),
        "deviation_reset": np.bool_(record["deviation_reset_mask"]),
        # The guarded source configuration disables deviation termination so all
        # selected samples have exactly zero termination penalty.
        "deviation_penalty": np.float32(0.0),
        "reward_total": np.float32(record["reward"]),
        "reward_distance_x": np.float32(_scalar(_required_attr(env, "distance_reward_x")[0], float)),
        "reward_distance_y": np.float32(_scalar(_required_attr(env, "distance_reward_y")[0], float)),
        "reward_distance_z": np.float32(_scalar(_required_attr(env, "distance_reward_z")[0], float)),
        "reward_rotation": np.float32(_scalar(_required_attr(env, "rotation_reward")[0], float)),
        "reward_action_penalty": np.float32(_scalar(_required_attr(env, "action_penalty")[0], float)),
        "reward_position_penalty": np.float32(_scalar(_required_attr(env, "position_penalty")[0], float)),
        "reward_joint_penalty": np.float32(_scalar(_required_attr(env, "joint_penalty")[0], float)),
        "reward_contact": np.float32(_scalar(_required_attr(env, "contact_reward")[0], float)),
        "reward_object_stability": np.float32(
            _scalar(_required_attr(env, "object_stability_reward")[0], float)
        ),
        "reward_object_speed": np.float32(_scalar(_required_attr(env, "object_speed")[0], float)),
    }


PULSE_NONE = 0
PULSE_EARLY_MASK = 1
PULSE_TRANSITION = 2
PULSE_ACCUMULATE_POSITIVE = 3
PULSE_ACCUMULATE_NEGATIVE = 4


def _semantic_action(torch: Any, env: Any, *, physical_call: int) -> tuple[Any, int]:
    """Exercise wrist, active and inactive fingers across the 99/100 boundary."""

    cases = {
        99: (PULSE_EARLY_MASK, 1.0),
        100: (PULSE_TRANSITION, 1.0),
        101: (PULSE_ACCUMULATE_POSITIVE, 1.0),
        102: (PULSE_ACCUMULATE_NEGATIVE, -1.0),
    }
    case = cases.get(physical_call)
    action = torch.zeros((1, DOFS), device=env.device, dtype=torch.float32)
    if case is None:
        return action, PULSE_NONE
    pulse_case, sign = case
    active = env._get_active_joint_mask()[0]
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    inactive_indices = torch.nonzero(~active, as_tuple=False).flatten()
    if len(active_indices) == 0 or len(inactive_indices) == 0:
        raise RuntimeError("semantic action fixture requires both an active and inactive finger joint")
    # Values exceed policy bounds deliberately: the source pre-step clamp, then
    # active/inactive masking, must be observable in the recorded result.
    action[0, 3] = 2.0 * sign
    action[0, 6 + active_indices[0]] = 2.0 * sign
    action[0, 6 + inactive_indices[0]] = -2.0 * sign
    return action, pulse_case


def _capture_termination_discriminator(
    env: Any,
    torch: Any,
    *,
    max_deviation_distance: float,
    deviation_penalty: float,
) -> dict[str, Any]:
    """Measure source termination truth values under current-training settings.

    This is deliberately independent of the fixed-horizon replay override.
    It mutates only live runtime buffers and restores them before returning.
    """

    manager = _required_attr(env, "termination_manager")
    saved = {
        "object_positions": _required_attr(env, "object_positions").clone(),
        "target_object_positions": _required_attr(env, "target_object_positions").clone(),
        "progress_buf": _required_attr(env, "progress_buf").clone(),
        "trajectory_steps": _required_attr(env, "trajectory_steps").clone(),
        "manager_threshold": manager.max_deviation_distance,
        "manager_penalty": manager.deviation_penalty,
    }
    cases = (
        ("below_threshold", 100, 0, max_deviation_distance * 0.999),
        ("above_threshold", 100, 0, max_deviation_distance * 1.001),
        ("early_above_threshold", 99, 0, max_deviation_distance * 1.001),
        ("trajectory_complete", 100, REFERENCE_FRAMES - 1, 0.0),
    )
    observed: dict[str, Any] = {}
    try:
        manager.max_deviation_distance = float(max_deviation_distance)
        manager.deviation_penalty = float(deviation_penalty)
        for name, trajectory_step, progress, displacement in cases:
            env.object_positions.copy_(saved["target_object_positions"])
            env.object_positions[0, 0] += displacement
            env.progress_buf.fill_(progress)
            env.trajectory_steps.fill_(trajectory_step)
            reset, deviation, penalty = manager.check_termination()
            observed[name] = {
                "trajectory_step": trajectory_step,
                "progress": progress,
                "object_target_distance": displacement,
                "early_mask": bool(env._get_early_phase_mask()[0].item()),
                "reset": bool(reset[0].item()),
                "deviation_reset": bool(deviation[0].item()),
                "deviation_penalty": float(penalty[0].item()),
            }
    finally:
        env.object_positions.copy_(saved["object_positions"])
        env.target_object_positions.copy_(saved["target_object_positions"])
        env.progress_buf.copy_(saved["progress_buf"])
        env.trajectory_steps.copy_(saved["trajectory_steps"])
        manager.max_deviation_distance = saved["manager_threshold"]
        manager.deviation_penalty = saved["manager_penalty"]

    expected = {
        "below_threshold": (False, False, 0.0),
        "above_threshold": (True, True, -float(deviation_penalty)),
        "early_above_threshold": (False, False, 0.0),
        "trajectory_complete": (True, False, 0.0),
    }
    for name, (reset, deviation, penalty) in expected.items():
        actual = observed[name]
        if (actual["reset"], actual["deviation_reset"]) != (reset, deviation) or not math.isclose(
            actual["deviation_penalty"], penalty, abs_tol=1e-6
        ):
            raise RuntimeError(f"source termination discriminator failed for {name}: {actual}")
    return {
        "max_deviation_distance": float(max_deviation_distance),
        "deviation_penalty": float(deviation_penalty),
        "cases": observed,
    }


def run(
    output: os.PathLike[str] | str,
    device: str = "cuda:0",
    headless: bool = True,
    semantic_only: bool = False,
) -> tuple[Path, Path]:
    """Run either the fixed trace or the current-source sparse semantic fixture."""
    if device != "cuda:0":
        raise ValueError("the settled reference trace requires device cuda:0")
    if headless is not True:
        raise ValueError("the settled reference trace requires headless execution")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("PYTHONDONTWRITEBYTECODE=1 is required to protect the source checkout")

    caller_cwd = Path.cwd()
    paths = artifact_paths(output)
    if paths.prefix == SOURCE_REPO or SOURCE_REPO in paths.prefix.parents:
        raise ValueError("output must be outside the protected sibling source repository")
    records: list[dict[str, Any]] = []
    semantic_records: list[dict[str, Any]] = []
    source_before: dict[str, Any] | None = None
    source_after: dict[str, Any] | None = None
    patch_state: dict[str, Any] = {"matching_count": 0}
    resolved_config: dict[str, Any] | None = None
    current_source_termination: dict[str, float] | None = None
    last_state: dict[str, Any] = {}
    captured_warnings: list[str] = []
    backend: Any = None
    original_find: Any = None

    try:
        os.chdir(RUNTIME_CWD)
        source_before = capture_source_state()

        import sys

        # The sibling package mixes `isaacgymenvs.*` imports with historical
        # top-level `tasks.*`/`utils.*` imports. Both roots are therefore
        # required; inserting them explicitly makes ownership independent of an
        # editable-install side effect.
        for import_root in (str(RUNTIME_CWD), str(RUNTIME_CWD.parent)):
            if import_root not in sys.path:
                sys.path.insert(0, import_root)

        import isaacgym  # noqa: F401  # must precede torch
        import torch

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        import isaacgymenvs  # noqa: F401  # registers OmegaConf resolvers
        import lance
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from isaacgymenvs.utils import lance_mocap_backend as backend_module

        backend = backend_module
        with initialize_config_dir(version_base=None, config_dir=str(RUNTIME_CWD / "cfg")):
            cfg = compose(config_name="config.yaml", overrides=HYDRA_OVERRIDES)
        current_source_cfg = OmegaConf.to_container(cfg.task, resolve=True)
        if not isinstance(current_source_cfg, dict):
            raise TypeError("current source task configuration is not a dict")
        current_source_termination = {
            "max_deviation_distance": float(current_source_cfg["env"]["maxDeviationDistance"]),
            "deviation_penalty": float(current_source_cfg["env"]["deviationPenalty"]),
        }
        task_updates = task_updates_for_mode(semantic_only=semantic_only)
        for key, value in task_updates.items():
            OmegaConf.update(cfg, key, value, merge=False)
        task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
        if not isinstance(task_cfg, dict):
            raise TypeError("resolved task configuration is not a dict")
        resolved_config = task_cfg
        _assert_resolved_config(
            task_cfg,
            residual_actions=semantic_only,
            early_phase_steps=100 if semantic_only else 0,
        )

        original_find = _install_strict_find(backend, lance, patch_state)
        from isaacgymenvs.tasks.mano_hand import MANOHand

        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            env = MANOHand(
                cfg=task_cfg,
                rl_device=device,
                sim_device=device,
                graphics_device_id=0,
                headless=True,
                virtual_screen_capture=False,
                force_render=False,
            )
            captured_warnings.extend(
                f"{item.category.__name__}: {item.message}" for item in warning_records
            )

        if bool(env.use_residual_actions) is not semantic_only:
            raise RuntimeError("constructed environment residual-action mode differs from requested contract")
        if int(env.num_envs) != 1 or int(env.num_mano_dofs) != DOFS:
            raise RuntimeError("constructed environment dimension mismatch")
        if int(env.control_freq_inv) != 1:
            raise RuntimeError("constructed environment control frequency mismatch")
        if _numpy(env.env_trajectory_lengths, "int64").tolist() != [REFERENCE_FRAMES]:
            raise RuntimeError("trajectory length is not 792")
        mocap = env.env_mocap_data[0]
        expected_slice = {
            "original_start_frame": SOURCE_START,
            "original_end_frame": SOURCE_STOP,
            "length": REFERENCE_FRAMES,
        }
        for key, expected in expected_slice.items():
            if int(mocap[key]) != expected:
                raise RuntimeError(f"mocap slice field {key} mismatch")

        env.reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
        if int(env.trajectory_steps[0].item()) != 0 or int(env.progress_buf[0].item()) != 0:
            raise RuntimeError("explicit reset did not initialize both counters to zero")
        invariance = None if semantic_only else _prove_action_invariance(env, torch)

        full_reference_raw = np.concatenate(
            (
                _numpy(env._hand_pos_lookup[0], "float32"),
                _numpy(env._hand_rot_lookup[0], "float32"),
                _numpy(env._finger_joints_lookup[0], "float32"),
            ),
            axis=1,
        )
        if full_reference_raw.shape != (REFERENCE_FRAMES, DOFS):
            raise RuntimeError(f"raw hand reference shape mismatch: {full_reference_raw.shape}")
        full_reference = full_reference_raw.astype(np.float64)
        full_reference[:, 3:6] = np.unwrap(
            full_reference[:, 3:6], axis=0, period=2.0 * math.pi
        )
        raw_object_table = lance.dataset(str(DATASET_PATH)).take(
            [ROW_INDEX], columns=["objects"]
        )
        raw_object_rows = raw_object_table.to_pylist()
        if len(raw_object_rows) != 1:
            raise RuntimeError("raw object reference lookup did not return exactly one row")
        raw_object_reference = np.asarray(
            raw_object_rows[0]["objects"][OBJECT_INDEX]["pos"], dtype=np.float64
        )[SOURCE_START:SOURCE_STOP]
        if raw_object_reference.shape != (REFERENCE_FRAMES, 3) or not np.isfinite(
            raw_object_reference
        ).all():
            raise RuntimeError("raw Lance object reference is invalid")

        for physical_call in range(PHYSICS_CALLS):
            schedule = schedule_point(physical_call)
            if semantic_only:
                raw_action, pulse_case = _semantic_action(torch, env, physical_call=physical_call)
            else:
                raw_action = torch.zeros((1, DOFS), device=env.device, dtype=torch.float32)
                pulse_case = PULSE_NONE
            if tuple(raw_action.shape) != (1, DOFS) or not bool(torch.isfinite(raw_action).all()):
                raise RuntimeError("invalid replay action")
            action = torch.clamp(raw_action, -env.clip_actions, env.clip_actions)
            progress_before = int(env.progress_buf[0].item())
            trajectory_before = int(env.trajectory_steps[0].item())
            action_cumulative_offset_before = _numpy(
                _required_attr(env, "cumulative_offset")[0], "float32"
            )
            action_cumulative_joint_offset_before = _numpy(
                _required_attr(env, "cumulative_joint_offset")[0], "float32"
            )
            if trajectory_before != schedule.target_index:
                raise RuntimeError(
                    f"target schedule diverged on call {physical_call}: "
                    f"{trajectory_before} != {schedule.target_index}"
                )
            env.pre_physics_step(action)
            trajectory_after_pre = int(env.trajectory_steps[0].item())
            pre_capture = {
                "raw_action": _numpy(raw_action[0], "float32"),
                "processed_action": _numpy(_required_attr(env, "actions")[0], "float32"),
                "mocap_target_unclamped": _numpy(
                    _required_attr(env, "mocap_targets")[0], "float32"
                ),
                "q_target": _numpy(_required_attr(env, "cur_targets")[0], "float32"),
                "cumulative_offset": _numpy(
                    _required_attr(env, "cumulative_offset")[0], "float32"
                ),
                "cumulative_joint_offset": _numpy(
                    _required_attr(env, "cumulative_joint_offset")[0], "float32"
                ),
                "trajectory_step": np.int64(trajectory_before),
                "action_cumulative_offset": action_cumulative_offset_before,
                "action_cumulative_joint_offset": action_cumulative_joint_offset_before,
            }
            env.gym.simulate(env.sim)
            env.post_physics_step()
            record = _capture_record(
                env,
                pre_capture,
                schedule,
                progress_before,
                trajectory_before,
                trajectory_after_pre,
                full_reference_raw[schedule.reference_index],
                full_reference[schedule.reference_index],
                raw_object_reference[schedule.reference_index],
            )
            records.append(record)
            if semantic_only and physical_call in SEMANTIC_REGULAR_CALLS:
                semantic_records.append(
                    _capture_semantic_record(
                        env,
                        record,
                        pre_capture,
                        physical_call,
                        sample_kind=SEMANTIC_SAMPLE_KIND_POST_STEP,
                        pulse_case=pulse_case,
                    )
                )
            env.control_steps += 1
            last_state = {
                "last_completed_physical_call": physical_call,
                "progress": int(record["progress_after_post"]),
                "trajectory_step": int(record["trajectory_step_after_post"]),
                "reset": bool(record["reset"]),
                "termination_reason_code": int(record["termination_reason_code"]),
                "trajectory_complete_reset_mask": bool(
                    record["trajectory_complete_reset_mask"]
                ),
            }

        arrays = _stack_records(records)
        if len(records) != PHYSICS_CALLS:
            raise RuntimeError("replay did not complete exactly 791 physical calls")

        if semantic_only:
            if not bool(records[-1]["reset"]):
                raise RuntimeError("semantic fixture requires a terminal reset signal at physical call 790")
            raw_action, pulse_case = _semantic_action(torch, env, physical_call=SEMANTIC_DELAYED_RESET_CALL)
            action = torch.clamp(raw_action, -env.clip_actions, env.clip_actions)
            delayed_trajectory_before = np.int64(
                _scalar(_required_attr(env, "trajectory_steps")[0], int)
            )
            delayed_cumulative_offset_before = _numpy(
                _required_attr(env, "cumulative_offset")[0], "float32"
            )
            delayed_cumulative_joint_offset_before = _numpy(
                _required_attr(env, "cumulative_joint_offset")[0], "float32"
            )
            env.pre_physics_step(action)
            delayed_pre_capture = {
                "raw_action": _numpy(raw_action[0], "float32"),
                "processed_action": _numpy(_required_attr(env, "actions")[0], "float32"),
                "mocap_target_unclamped": _numpy(_required_attr(env, "mocap_targets")[0], "float32"),
                "q_target": _numpy(_required_attr(env, "cur_targets")[0], "float32"),
                "cumulative_offset": _numpy(_required_attr(env, "cumulative_offset")[0], "float32"),
                "cumulative_joint_offset": _numpy(
                    _required_attr(env, "cumulative_joint_offset")[0], "float32"
                ),
                "trajectory_step": delayed_trajectory_before,
                "action_cumulative_offset": delayed_cumulative_offset_before,
                "action_cumulative_joint_offset": delayed_cumulative_joint_offset_before,
            }
            # The next source post-step consumes the pending terminal reset and
            # writes the reset state. It is fixture-only and never enters the
            # immutable 791-call reference trace.
            env.gym.simulate(env.sim)
            env.post_physics_step()
            delayed_record = {
                "reset": np.bool_(_scalar(_required_attr(env, "reset_buf")[0], bool)),
                "deviation_reset_mask": np.bool_(False),
                "reward": np.float32(_scalar(_required_attr(env, "rew_buf")[0], float)),
            }
            semantic_records.append(
                _capture_semantic_record(
                    env,
                    delayed_record,
                    delayed_pre_capture,
                    SEMANTIC_DELAYED_RESET_CALL,
                    sample_kind=SEMANTIC_SAMPLE_KIND_DELAYED_RESET_POST,
                    pulse_case=pulse_case,
                )
            )
            assert current_source_termination is not None
            termination_discriminator = _capture_termination_discriminator(
                env,
                torch,
                **current_source_termination,
            )

        source_after = capture_source_state()
        if not _source_unchanged(source_before, source_after):
            raise RuntimeError("source repository state or protected file hashes changed")

        if semantic_only:
            semantic_arrays = _stack_records(semantic_records)
            semantic_metadata = _jsonable({
                "schema": SEMANTIC_FIXTURE_SCHEMA,
                "regular_sample_calls": list(SEMANTIC_REGULAR_CALLS),
                "delayed_reset_post_after_call": PHYSICS_CALLS - 1,
                "trajectory_identity": {
                    "identity": TRAJECTORY_NAME,
                    "dataset_path": str(DATASET_PATH),
                    "dataset_version": patch_state["dataset_version"],
                    "source_slice": [SOURCE_START, SOURCE_STOP],
                },
                "source_repository": {"before": source_before, "after": source_after},
                "configuration": {
                    "useResidualActions": task_cfg["env"]["useResidualActions"],
                    "earlyPhaseMocapSteps": task_cfg["env"]["earlyPhaseMocapSteps"],
                    "maxDeviationDistance": task_cfg["env"]["maxDeviationDistance"],
                    "deviationPenalty": task_cfg["env"]["deviationPenalty"],
                    "headless": True,
                },
                "pulse_cases": {
                    str(PULSE_NONE): "zero_action",
                    str(PULSE_EARLY_MASK): "early_mask_wrist_active_inactive",
                    str(PULSE_TRANSITION): "transition_wrist_active_inactive",
                    str(PULSE_ACCUMULATE_POSITIVE): "accumulate_positive_wrist_active_inactive",
                    str(PULSE_ACCUMULATE_NEGATIVE): "accumulate_negative_wrist_active_inactive",
                },
                "termination_discriminator": termination_discriminator,
                "purpose": (
                    "resolved source inputs and named reward terms for target-side observation/action/reward verification; "
                    "not a cross-simulator physical-state equality claim"
                ),
            })
            # ``run`` changes cwd to the read-only source runtime. Reuse the
            # target-absolute prefix resolved before that change rather than
            # reinterpreting a caller-relative path under the sibling source.
            return _write_semantic_fixture(paths.prefix, semantic_metadata, semantic_arrays)

        metrics = _compute_metrics(arrays)
        metadata: dict[str, Any] = {
            "schema": SCHEMA,
            "trajectory_identity": {
                "dataset_path": str(DATASET_PATH),
                "row_index": ROW_INDEX,
                "object_index": OBJECT_INDEX,
                "uuid": UUID,
                "file_uuid": FILE_UUID,
                "identity": TRAJECTORY_NAME,
                "dataset_version": patch_state["dataset_version"],
                "source_slice": {
                    "start": SOURCE_START,
                    "stop": SOURCE_STOP,
                    "stop_exclusive": True,
                },
                "reference_frame_count": REFERENCE_FRAMES,
                "physical_call_count": PHYSICS_CALLS,
            },
            "lance_index_entry": {
                "entry": _jsonable(patch_state["accepted_entry"]),
                "reference": patch_state["lance_reference"],
            },
            "source_repository": {"before": source_before, "after": source_after},
            "config": {
                "constructor": {
                    "rl_device": device,
                    "sim_device": device,
                    "graphics_device_id": 0,
                    "headless": True,
                    "virtual_screen_capture": False,
                    "force_render": False,
                },
                "hydra_overrides": HYDRA_OVERRIDES,
                "task_updates": TASK_UPDATES,
                "resolved": {
                    "dt": task_cfg["sim"]["dt"],
                    "substeps": task_cfg["sim"]["substeps"],
                    "gravity": task_cfg["sim"]["gravity"],
                    "use_gpu_pipeline": task_cfg["sim"]["use_gpu_pipeline"],
                    "physx_use_gpu": task_cfg["sim"]["physx"]["use_gpu"],
                    "physx_max_gpu_contact_pairs": task_cfg["sim"]["physx"]["max_gpu_contact_pairs"],
                    "physx_num_subscenes": task_cfg["sim"]["physx"]["num_subscenes"],
                    "contact_collection": task_cfg["sim"]["physx"]["contact_collection"],
                    "maxDeviationDistance": task_cfg["env"]["maxDeviationDistance"],
                    "useResidualActions": task_cfg["env"]["useResidualActions"],
                },
                "deviation_termination": "disabled_for_fixed_horizon_reference",
                "target_schedule": "command 0,0,1,...,789; reference 0..790",
                "quaternion_convention": "all source and actual quaternions are XYZW",
                "seed": SEED,
                "python_dont_write_bytecode": True,
            },
            "checks": {
                "exact_index_entry": True,
                "live_row_identity": True,
                "dataset_version_132": True,
                "resolved_configuration": True,
                "exact_slice_440_1232": True,
                "one_environment": True,
                "26_dofs": True,
                "counter_schedule_all_calls": True,
                "no_reset_before_call_790": True,
                "final_trajectory_completion_reset": True,
                "exactly_791_simulate_calls": True,
                "source_repository_unchanged": True,
            },
            "action_invariance": invariance,
            "metrics": metrics,
            "warnings": {
                "messages": captured_warnings,
                "stdout_stderr_log": f"{paths.prefix}.log (external invocation log)",
            },
            "non_comparable_fields": NON_COMPARABLE_FIELDS,
            "claims": {"isaac_parity": "trace_generated_not_yet_compared"},
        }
        metadata = _jsonable(metadata)
        _write_success(paths, metadata, arrays)
        return paths.final_npz, paths.final_json
    except BaseException as error:
        try:
            source_after = capture_source_state() if source_before is not None else None
        except BaseException as source_error:
            source_after = {"capture_error": repr(source_error)}
        partial = _stack_records(records) if records else None
        failure = {
            "schema": SCHEMA,
            "status": "failed",
            "exception": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "completed_physical_calls": len(records),
            "last_state": last_state,
            "patch_state": _jsonable(patch_state),
            "resolved_config": _jsonable(resolved_config),
            "source_repository": {
                "before": _jsonable(source_before),
                "after": _jsonable(source_after),
                "unchanged": bool(
                    source_before is not None
                    and source_after is not None
                    and _source_unchanged(source_before, source_after)
                ),
            },
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "warnings": captured_warnings,
        }
        write_failure_artifacts(paths, failure, partial)
        raise
    finally:
        if backend is not None and original_find is not None:
            backend.LanceTrajectoryIndex.find = original_find
        os.chdir(caller_cwd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="output prefix for .npz and .json")
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="emit source current-configuration semantic samples instead of the residual-off PD trace",
    )
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="run headless (required and enabled by default)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run(
        arguments.output,
        device=arguments.device,
        headless=arguments.headless,
        semantic_only=arguments.semantic_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
