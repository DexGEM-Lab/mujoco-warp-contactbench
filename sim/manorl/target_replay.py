"""Versioned target-DOF replay packages and MJX-Warp playback.

A target replay package is deliberately smaller than a Lance row.  It contains
only the recorded physical state needed to initialize and score a replay, the
post-controller target vectors to apply as ``data.ctrl``, and inspectable JSON
metadata.  The loader never opens Lance or a policy checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.contracts import (
    CONTROL_TIMESTEP,
    JOINT_DOF,
    PHYSICS_SUBSTEPS_PER_TARGET,
    TrajectoryIdentity,
)
from sim.manorl.trajectory import (
    ReferenceTrajectory,
    TrajectoryBatch,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

# Keep the first validated package spelling stable; it is already used by the
# Server1→Server2 replay artifact and is part of the external package contract.
TARGET_REPLAY_PACKAGE_SCHEMA = "manorl_target_dof_replay_package_v1"
TARGET_REPLAY_TARGET_SEMANTICS = (
    "hands[0].urdf_dof_target is the post-command_target controller ctrl vector"
)
_REQUIRED_ARRAYS = (
    "timestamps",
    "recorded_qpos",
    "target_qpos",
    "object_position",
    "object_quaternion_xyzw",
)
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*_[0-9]{1,2}_[0-9]+$")


class TargetReplayPackageError(ValueError):
    """Raised when a replay package violates its versioned contract."""


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one package payload or sidecar."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_path_for(package_path: Path) -> Path:
    """Resolve the required human-readable sidecar for an NPZ package."""

    if package_path.suffix.lower() != ".npz":
        raise TargetReplayPackageError(
            f"target replay package must have a .npz suffix: {package_path}"
        )
    return package_path.with_suffix(".json")


def _readonly(
    values: np.ndarray, *, dtype: np.dtype[Any] = np.dtype(np.float64)
) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _required_string(metadata: Mapping[str, Any], name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value.strip():
        raise TargetReplayPackageError(f"metadata.{name} must be a non-empty string")
    return value


def _optional_positive_int(metadata: Mapping[str, Any], name: str) -> int | None:
    value = metadata.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TargetReplayPackageError(
            f"metadata.{name} must be a positive integer or null"
        )
    return int(value)


def _validate_metadata(
    metadata: Mapping[str, Any], package_path: Path
) -> dict[str, Any]:
    if metadata.get("schema") != TARGET_REPLAY_PACKAGE_SCHEMA:
        raise TargetReplayPackageError(
            f"unsupported target replay schema: {metadata.get('schema')!r}"
        )
    result = dict(metadata)
    for name in (
        "object",
        "source_identity",
        "source_uuid",
        "target_semantics",
        "npz_sha256",
    ):
        _required_string(result, name)
    if result["target_semantics"] != TARGET_REPLAY_TARGET_SEMANTICS:
        raise TargetReplayPackageError(
            "package target_semantics does not identify post-command_target controller vectors"
        )
    if not _IDENTITY_RE.fullmatch(result["source_identity"]):
        raise TargetReplayPackageError(
            "metadata.source_identity must be object_action_sequence, for example cube1_01_1614"
        )
    if result["object"] != result["source_identity"].split("_", 1)[0]:
        raise TargetReplayPackageError("metadata.object disagrees with source_identity")
    for name in ("source_dataset_version", "source_row_index"):
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TargetReplayPackageError(
                f"metadata.{name} must be a non-negative integer"
            )
    control_timestep = result.get("control_timestep_seconds")
    if isinstance(control_timestep, bool) or not isinstance(
        control_timestep, (int, float)
    ):
        raise TargetReplayPackageError(
            "metadata.control_timestep_seconds must be numeric"
        )
    if not math.isfinite(float(control_timestep)) or float(control_timestep) <= 0:
        raise TargetReplayPackageError(
            "metadata.control_timestep_seconds must be finite and positive"
        )
    if not math.isclose(
        float(control_timestep), CONTROL_TIMESTEP, rel_tol=0.0, abs_tol=1e-12
    ):
        raise TargetReplayPackageError(
            f"package control timestep {control_timestep} does not match project timestep {CONTROL_TIMESTEP}"
        )
    substeps = result.get("physics_substeps_per_target")
    if substeps != PHYSICS_SUBSTEPS_PER_TARGET:
        raise TargetReplayPackageError(
            f"package physics_substeps_per_target={substeps!r} does not match {PHYSICS_SUBSTEPS_PER_TARGET}"
        )
    frames = result.get("frames")
    transitions = result.get("transitions")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 2:
        raise TargetReplayPackageError("metadata.frames must be an integer >= 2")
    if transitions != frames - 1:
        raise TargetReplayPackageError("metadata.transitions must equal frames - 1")
    _optional_positive_int(result, "warp_ccd_iterations")
    _optional_positive_int(result, "warp_ccd_contacts_per_world")
    if len(result["npz_sha256"]) != hashlib.sha256().digest_size * 2:
        raise TargetReplayPackageError(
            "metadata.npz_sha256 must be a hexadecimal SHA256 digest"
        )
    try:
        int(result["npz_sha256"], 16)
    except ValueError as exc:
        raise TargetReplayPackageError(
            "metadata.npz_sha256 must be hexadecimal"
        ) from exc
    if not package_path.is_file():
        raise TargetReplayPackageError(
            f"target replay payload does not exist: {package_path}"
        )
    actual_sha256 = sha256_file(package_path)
    if actual_sha256 != result["npz_sha256"]:
        raise TargetReplayPackageError(
            f"target replay payload SHA256 mismatch: expected {result['npz_sha256']}, got {actual_sha256}"
        )
    return result


def _validate_arrays(
    arrays: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    missing = [name for name in _REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise TargetReplayPackageError(f"target replay payload omits arrays: {missing}")
    values = {
        name: np.asarray(arrays[name], dtype=np.float64) for name in _REQUIRED_ARRAYS
    }
    timestamps = values["timestamps"]
    recorded_qpos = values["recorded_qpos"]
    target_qpos = values["target_qpos"]
    object_position = values["object_position"]
    object_quaternion = values["object_quaternion_xyzw"]
    frames = int(metadata["frames"])
    expected = {
        "timestamps": (frames,),
        "recorded_qpos": (frames, JOINT_DOF),
        "target_qpos": (frames, JOINT_DOF),
        "object_position": (frames, 3),
        "object_quaternion_xyzw": (frames, 4),
    }
    for name, shape in expected.items():
        if values[name].shape != shape:
            raise TargetReplayPackageError(
                f"payload.{name} shape {values[name].shape} does not match {shape}"
            )
        if not np.all(np.isfinite(values[name])):
            raise TargetReplayPackageError(f"payload.{name} contains non-finite values")
    deltas = np.diff(timestamps)
    if np.any(deltas <= 0):
        raise TargetReplayPackageError("payload.timestamps must be strictly increasing")
    if not np.allclose(
        deltas,
        float(metadata["control_timestep_seconds"]),
        rtol=0.0,
        atol=1e-10,
    ):
        raise TargetReplayPackageError(
            "payload.timestamps must advance by metadata.control_timestep_seconds"
        )
    quaternion_norms = np.linalg.norm(object_quaternion, axis=1)
    if not np.allclose(quaternion_norms, 1.0, rtol=0.0, atol=1e-6):
        raise TargetReplayPackageError(
            "payload.object_quaternion_xyzw must contain normalized quaternions"
        )
    return {name: _readonly(value) for name, value in values.items()}


@dataclass(frozen=True)
class TargetReplayPackage:
    """Validated immutable arrays and metadata for one target replay."""

    path: Path
    metadata: Mapping[str, Any]
    timestamps: np.ndarray
    recorded_qpos: np.ndarray
    target_qpos: np.ndarray
    object_position: np.ndarray
    object_quaternion_xyzw: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def transitions(self) -> int:
        return self.frames - 1

    @property
    def source_identity(self) -> str:
        return str(self.metadata["source_identity"])

    @property
    def object_type(self) -> str:
        return str(self.metadata["object"])


def load_target_replay_package(path: str | Path) -> TargetReplayPackage:
    """Load and validate one NPZ package plus its JSON sidecar.

    This function intentionally has no Lance, checkpoint, JAX, or MuJoCo
    imports.  It is safe to use on a replay-only host before allocating a GPU.
    """

    package_path = Path(path).expanduser().resolve()
    sidecar = metadata_path_for(package_path)
    if not sidecar.is_file():
        raise TargetReplayPackageError(
            f"target replay metadata sidecar is missing: {sidecar}"
        )
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetReplayPackageError(
            f"could not read target replay metadata: {sidecar}"
        ) from exc
    if not isinstance(metadata, dict):
        raise TargetReplayPackageError("target replay metadata must be a JSON object")
    metadata = _validate_metadata(metadata, package_path)
    try:
        with np.load(package_path, allow_pickle=False) as payload:
            arrays = {name: payload[name] for name in payload.files}
    except (OSError, ValueError) as exc:
        raise TargetReplayPackageError(
            f"could not read target replay payload: {package_path}"
        ) from exc
    validated = _validate_arrays(arrays, metadata)
    return TargetReplayPackage(
        path=package_path,
        metadata=metadata,
        timestamps=validated["timestamps"],
        recorded_qpos=validated["recorded_qpos"],
        target_qpos=validated["target_qpos"],
        object_position=validated["object_position"],
        object_quaternion_xyzw=validated["object_quaternion_xyzw"],
    )


def _json_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    if "schema" in result and result["schema"] != TARGET_REPLAY_PACKAGE_SCHEMA:
        raise TargetReplayPackageError(
            f"metadata.schema must be {TARGET_REPLAY_PACKAGE_SCHEMA!r}"
        )
    if (
        "target_semantics" in result
        and result["target_semantics"] != TARGET_REPLAY_TARGET_SEMANTICS
    ):
        raise TargetReplayPackageError(
            "metadata.target_semantics is not the target-DOF contract"
        )
    result["schema"] = TARGET_REPLAY_PACKAGE_SCHEMA
    result["target_semantics"] = TARGET_REPLAY_TARGET_SEMANTICS
    return result


def write_target_replay_package(
    path: str | Path,
    *,
    metadata: Mapping[str, Any],
    timestamps: Any,
    recorded_qpos: Any,
    target_qpos: Any,
    object_position: Any,
    object_quaternion_xyzw: Any,
    replace: bool = False,
) -> TargetReplayPackage:
    """Atomically write a validated NPZ package and JSON sidecar.

    The payload and sidecar are written through process-local partial names;
    a failed write never publishes a package that the loader can mistake for
    complete.
    """

    package_path = Path(path).expanduser().resolve()
    if package_path.suffix.lower() != ".npz":
        raise TargetReplayPackageError("target replay output must have a .npz suffix")
    sidecar = metadata_path_for(package_path)
    if (package_path.exists() or sidecar.exists()) and not replace:
        raise FileExistsError(
            f"target replay output exists; pass replace=True explicitly: {package_path}"
        )
    arrays = {
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "recorded_qpos": np.asarray(recorded_qpos, dtype=np.float64),
        "target_qpos": np.asarray(target_qpos, dtype=np.float64),
        "object_position": np.asarray(object_position, dtype=np.float64),
        "object_quaternion_xyzw": np.asarray(object_quaternion_xyzw, dtype=np.float64),
    }
    normalized_metadata = _json_safe_metadata(metadata)
    # The writer needs the frame count before the loader can validate the arrays.
    normalized_metadata["frames"] = int(arrays["timestamps"].shape[0])
    normalized_metadata["transitions"] = normalized_metadata["frames"] - 1
    normalized_metadata["npz_sha256"] = "0" * 64
    # Validate all metadata except the payload digest before writing.
    _validate_metadata_without_payload_digest(normalized_metadata)
    normalized_arrays = _validate_arrays(arrays, normalized_metadata)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_payload = package_path.with_name(
        f".{package_path.name}.{os.getpid()}.partial"
    )
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.partial")
    try:
        with temporary_payload.open("wb") as stream:
            np.savez_compressed(stream, **normalized_arrays)
        normalized_metadata["npz_sha256"] = sha256_file(temporary_payload)
        temporary_sidecar.write_text(
            json.dumps(normalized_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not replace and (package_path.exists() or sidecar.exists()):
            raise FileExistsError(
                f"target replay output appeared during write: {package_path}"
            )
        os.replace(temporary_payload, package_path)
        os.replace(temporary_sidecar, sidecar)
    except BaseException:
        for temporary in (temporary_payload, temporary_sidecar):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise
    return load_target_replay_package(package_path)


def _validate_metadata_without_payload_digest(metadata: Mapping[str, Any]) -> None:
    """Validate writer metadata without requiring a digest that is not known yet."""

    if metadata.get("npz_sha256") != "0" * 64:
        raise TargetReplayPackageError("internal writer digest sentinel is invalid")
    for name in (
        "object",
        "source_identity",
        "source_uuid",
        "target_semantics",
    ):
        _required_string(metadata, name)
    if metadata["target_semantics"] != TARGET_REPLAY_TARGET_SEMANTICS:
        raise TargetReplayPackageError(
            "metadata target_semantics is not the target-DOF contract"
        )
    if not _IDENTITY_RE.fullmatch(str(metadata["source_identity"])):
        raise TargetReplayPackageError(
            "metadata.source_identity is not object_action_sequence"
        )
    if str(metadata["object"]) != str(metadata["source_identity"]).split("_", 1)[0]:
        raise TargetReplayPackageError("metadata.object disagrees with source_identity")
    for name in ("source_dataset_version", "source_row_index"):
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TargetReplayPackageError(
                f"metadata.{name} must be a non-negative integer"
            )
    timestep = metadata.get("control_timestep_seconds")
    if not isinstance(timestep, (int, float)) or isinstance(timestep, bool):
        raise TargetReplayPackageError(
            "metadata.control_timestep_seconds must be numeric"
        )
    if not math.isclose(float(timestep), CONTROL_TIMESTEP, rel_tol=0.0, abs_tol=1e-12):
        raise TargetReplayPackageError(
            "metadata control timestep does not match project timestep"
        )
    if metadata.get("physics_substeps_per_target") != PHYSICS_SUBSTEPS_PER_TARGET:
        raise TargetReplayPackageError(
            "metadata physics substep count does not match project contract"
        )
    if (
        int(metadata["frames"]) < 2
        or int(metadata["transitions"]) != int(metadata["frames"]) - 1
    ):
        raise TargetReplayPackageError("metadata frames/transitions are inconsistent")
    _optional_positive_int(metadata, "warp_ccd_iterations")
    _optional_positive_int(metadata, "warp_ccd_contacts_per_world")


def _trajectory_for_package(package: TargetReplayPackage) -> ReferenceTrajectory:
    metadata = package.metadata
    identity = TrajectoryIdentity(
        dataset_path=str(metadata.get("source_dataset", "predecoded://target-replay")),
        dataset_version=int(metadata["source_dataset_version"]),
        row_index=int(metadata["source_row_index"]),
        object_index=0,
        uuid=str(metadata["source_uuid"]),
        file_uuid=str(metadata.get("source_file_uuid", "")),
        identity=package.source_identity,
        source_start=0,
        source_stop=package.frames,
        movement_start_raw=int(metadata.get("movement_start_raw", 0)),
        movement_end_raw=int(metadata.get("movement_end_raw", package.frames - 1)),
    )
    if identity.movement_start_raw < 0 or identity.movement_end_raw >= package.frames:
        raise TargetReplayPackageError("movement range is outside the replay package")
    return ReferenceTrajectory(
        identity=identity,
        dataset_version=identity.dataset_version,
        source_indices=_readonly(np.arange(package.frames), dtype=np.dtype(np.int64)),
        timestamps=package.timestamps,
        q_ref=package.recorded_qpos,
        object_pos_raw=package.object_position,
        object_pos=package.object_position,
        object_quat_xyzw=package.object_quaternion_xyzw,
        object_z_shift=0.0,
        hand_sides=("right",),
        q_ref_by_side={"right": package.recorded_qpos},
        selected_hand_sides=("right",),
    )


@dataclass(frozen=True)
class ReplayState:
    """One post-physics replay state in task array conventions."""

    qpos: np.ndarray
    object_position: np.ndarray
    object_quaternion_xyzw: np.ndarray


class TargetDofReplay:
    """Apply a validated target package to the existing MJX-Warp environment."""

    def __init__(
        self,
        package: TargetReplayPackage,
        *,
        device: str = "gpu",
        allow_physics_override: bool = False,
    ) -> None:
        if device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'")
        self.package = package
        self.device = device
        self.physics_overrides: list[str] = []
        ccd_iterations = package.metadata.get("warp_ccd_iterations")
        ccd_contacts = package.metadata.get("warp_ccd_contacts_per_world")
        if device == "cpu" and (ccd_iterations is not None or ccd_contacts is not None):
            if not allow_physics_override:
                raise TargetReplayPackageError(
                    "this package records explicit Warp CCD settings that require device='gpu'; "
                    "rerun with --allow-physics-override for an explicitly non-identical CPU diagnostic"
                )
            self.physics_overrides.append(
                "CPU replay omitted package Warp CCD allocation"
            )
            ccd_iterations = None
            ccd_contacts = None
        from sim.manorl.environment import (
            EnvironmentConfig,
            MujocoManoEnvironment,
            recommended_warp_contact_capacity,
        )

        trajectory = _trajectory_for_package(package)
        self.environment = MujocoManoEnvironment(
            TrajectoryBatch((trajectory,)),
            EnvironmentConfig(
                device=device,
                num_envs=1,
                hand_side="right",
                residual_enabled=False,
                point_sampling_backend="numpy_per_env",
                capture_transition_diagnostics=False,
                contact_capacity=recommended_warp_contact_capacity(1, ("right",)),
                warp_ccd_iterations=ccd_iterations,
                warp_ccd_contacts_per_world=ccd_contacts,
            ),
        )
        self._frame = 0
        self.reset()

    @property
    def frame(self) -> int:
        """Current state index, where frame zero is the initialized package state."""

        return self._frame

    def reset(self) -> ReplayState:
        """Restore the recorded frame-0 physical state without a native physics step."""

        environment = self.environment
        qpos = np.asarray(environment.data.qpos, dtype=np.float64).copy()
        qvel = np.zeros_like(np.asarray(environment.data.qvel, dtype=np.float64))
        ctrl = np.asarray(environment.data.ctrl, dtype=np.float64).copy()
        hand_slice = environment.producer.hand_qpos_slices["right"]
        object_address = environment.producer.object_qpos_address
        qpos[0, hand_slice] = self.package.recorded_qpos[0]
        qpos[0, object_address : object_address + 3] = self.package.object_position[0]
        qpos[0, object_address + 3 : object_address + 7] = xyzw_to_wxyz(
            self.package.object_quaternion_xyzw[0]
        )
        ctrl[0, :JOINT_DOF] = self.package.target_qpos[0]
        environment.data = environment.data.replace(
            qpos=environment.jax.device_put(
                environment.jp.asarray(qpos), environment.device
            ),
            qvel=environment.jax.device_put(
                environment.jp.asarray(qvel), environment.device
            ),
            ctrl=environment.jax.device_put(
                environment.jp.asarray(ctrl), environment.device
            ),
        )
        environment.data = environment._forward_fn(environment.data)
        self._frame = 0
        return self.state()

    def state(self) -> ReplayState:
        """Materialize the current hand qpos and object pose."""

        environment = self.environment
        qpos = np.asarray(
            environment.data.qpos[0, environment.producer.hand_qpos_slices["right"]],
            dtype=np.float64,
        ).copy()
        object_position = np.asarray(
            environment.data.xpos[0, environment.producer.object_body_id],
            dtype=np.float64,
        ).copy()
        object_quaternion = wxyz_to_xyzw(
            np.asarray(
                environment.data.xquat[0, environment.producer.object_body_id],
                dtype=np.float64,
            )
        ).copy()
        return ReplayState(qpos, object_position, object_quaternion)

    def step(self) -> ReplayState:
        """Apply the next post-controller target and exactly two Warp substeps."""

        if self._frame >= self.package.transitions:
            raise IndexError(
                "target replay is at its final frame; call reset() before stepping again"
            )
        environment = self.environment
        ctrl = np.asarray(environment.data.ctrl, dtype=np.float64).copy()
        ctrl[0, :JOINT_DOF] = self.package.target_qpos[self._frame]
        environment.data = environment.data.replace(
            ctrl=environment.jax.device_put(
                environment.jp.asarray(ctrl), environment.device
            )
        )
        # These are intentionally the existing environment's private compiled
        # kernels: no second native simulation is allowed to advance the state.
        for _ in range(PHYSICS_SUBSTEPS_PER_TARGET):
            environment.data = environment._step_fn(environment.data)
            environment._check_warp_ccd_overflow()
        self._frame += 1
        return self.state()

    def host_data(self) -> Any:
        """Return one native snapshot for visualization-only mirroring."""

        return self.environment.host_data(0)

    def headless_report(
        self,
        *,
        transitions: int | None = None,
        max_object_position_error_m: float = 0.05,
        max_object_rotation_error_rad: float = 0.25,
        max_qpos_abs_error: float = 1.0,
    ) -> dict[str, Any]:
        """Replay a bounded prefix and compare it with recorded physical states."""

        if transitions is None:
            transitions = self.package.transitions
        if (
            isinstance(transitions, bool)
            or not isinstance(transitions, int)
            or not 1 <= transitions <= self.package.transitions
        ):
            raise ValueError(
                f"transitions must be an integer in [1, {self.package.transitions}]"
            )
        initial_state = self.reset()
        initial_qpos_error = float(
            np.max(np.abs(initial_state.qpos - self.package.recorded_qpos[0]))
        )
        initial_object_position_error = float(
            np.linalg.norm(
                initial_state.object_position - self.package.object_position[0]
            )
        )
        initial_object_rotation_error = _rotation_error_rad(
            initial_state.object_quaternion_xyzw,
            self.package.object_quaternion_xyzw[0],
        )
        q_errors: list[float] = []
        object_errors: list[float] = []
        rotation_errors: list[float] = []
        replay_z = [float(self.state().object_position[2])]
        for index in range(transitions):
            state = self.step()
            replay_z.append(float(state.object_position[2]))
            q_errors.append(
                float(
                    np.max(np.abs(state.qpos - self.package.recorded_qpos[index + 1]))
                )
            )
            object_errors.append(
                float(
                    np.linalg.norm(
                        state.object_position - self.package.object_position[index + 1]
                    )
                )
            )
            rotation_errors.append(
                _rotation_error_rad(
                    state.object_quaternion_xyzw,
                    self.package.object_quaternion_xyzw[index + 1],
                )
            )
        q_values = np.asarray(q_errors, dtype=np.float64)
        object_values = np.asarray(object_errors, dtype=np.float64)
        rotation_values = np.asarray(rotation_errors, dtype=np.float64)
        report: dict[str, Any] = {
            "schema": "manorl.target_dof_replay_result.v1",
            "status": (
                "pass"
                if (
                    float(np.max(object_values)) <= max_object_position_error_m
                    and float(np.max(rotation_values)) <= max_object_rotation_error_rad
                    and float(np.max(q_values)) <= max_qpos_abs_error
                )
                else "fail"
            ),
            "package": str(self.package.path),
            "package_schema": self.package.metadata["schema"],
            "source_identity": self.package.source_identity,
            "source_uuid": self.package.metadata["source_uuid"],
            "object": self.package.object_type,
            "device": self.device,
            "states": transitions + 1,
            "transitions": transitions,
            "physics_substeps_per_target": PHYSICS_SUBSTEPS_PER_TARGET,
            "control_timestep_seconds": CONTROL_TIMESTEP,
            "physics_overrides": list(self.physics_overrides),
            "initial_max_qpos_abs_error": initial_qpos_error,
            "initial_object_position_error_m": initial_object_position_error,
            "initial_object_rotation_error_rad": initial_object_rotation_error,
            "max_qpos_abs_error": float(np.max(q_values)),
            "mean_qpos_abs_error": float(np.mean(q_values)),
            "rms_qpos_abs_error": float(np.sqrt(np.mean(q_values * q_values))),
            "max_object_position_error_m": float(np.max(object_values)),
            "mean_object_position_error_m": float(np.mean(object_values)),
            "rms_object_position_error_m": float(
                np.sqrt(np.mean(object_values * object_values))
            ),
            "max_object_rotation_error_rad": float(np.max(rotation_values)),
            "mean_object_rotation_error_rad": float(np.mean(rotation_values)),
            "recorded_max_lift_m": float(
                np.max(self.package.object_position[: transitions + 1, 2])
                - self.package.object_position[0, 2]
            ),
            "replay_max_lift_m": float(
                max(replay_z) - self.package.object_position[0, 2]
            ),
            "thresholds": {
                "max_object_position_error_m": max_object_position_error_m,
                "max_object_rotation_error_rad": max_object_rotation_error_rad,
                "max_qpos_abs_error": max_qpos_abs_error,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return report


def _rotation_error_rad(first_xyzw: np.ndarray, second_xyzw: np.ndarray) -> float:
    first = Rotation.from_quat(np.asarray(first_xyzw, dtype=np.float64))
    second = Rotation.from_quat(np.asarray(second_xyzw, dtype=np.float64))
    return float((first.inv() * second).magnitude())


def render_target_replay(
    replay: TargetDofReplay,
    *,
    speed: float = 0.25,
    loop: bool = True,
    transitions: int | None = None,
    print_every: int = 25,
    azimuth: float = 135.0,
    elevation: float = -20.0,
    distance: float = 0.65,
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.08),
) -> None:
    """Open a passive MuJoCo viewer while MJX-Warp remains the sole simulator."""

    if speed <= 0 or not math.isfinite(speed):
        raise ValueError("speed must be finite and positive")
    if print_every < 1:
        raise ValueError("print_every must be positive")
    if transitions is None:
        transitions = replay.package.transitions
    if (
        not isinstance(transitions, int)
        or isinstance(transitions, bool)
        or not 1 <= transitions <= replay.package.transitions
    ):
        raise ValueError(f"transitions must be in [1, {replay.package.transitions}]")
    from mujoco import viewer as mujoco_viewer
    from sim.manorl.assets import COLLISION_GEOM_GROUP
    from sim.manorl.view_environment import (
        _compile_native_viewer_model,
        _mirror_native_viewer_data,
        _require_graphical_session,
        _viewer_lock,
    )

    _require_graphical_session()
    mujoco, viewer_model = _compile_native_viewer_model(replay.environment)
    render_data = mujoco.MjData(viewer_model)
    _mirror_native_viewer_data(mujoco, viewer_model, render_data, replay.host_data())
    with mujoco_viewer.launch_passive(
        viewer_model, render_data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        with _viewer_lock(viewer):
            viewer.opt.geomgroup[COLLISION_GEOM_GROUP] = 0
            viewer.cam.azimuth = azimuth
            viewer.cam.elevation = elevation
            viewer.cam.distance = distance
            viewer.cam.lookat[:] = lookat
            viewer.sync()
        while viewer.is_running():
            replay.reset()
            with _viewer_lock(viewer):
                _mirror_native_viewer_data(
                    mujoco, viewer_model, render_data, replay.host_data()
                )
                viewer.sync()
            for index in range(transitions):
                if not viewer.is_running():
                    break
                started = time.perf_counter()
                state = replay.step()
                with _viewer_lock(viewer):
                    _mirror_native_viewer_data(
                        mujoco, viewer_model, render_data, replay.host_data()
                    )
                    viewer.sync()
                if index % print_every == 0:
                    print(
                        f"frame={index + 1}/{transitions} object_z={state.object_position[2]:.5f}",
                        flush=True,
                    )
                time.sleep(
                    max(
                        0.0,
                        CONTROL_TIMESTEP / speed - (time.perf_counter() - started),
                    )
                )
            if not loop:
                return


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Atomically write one JSON replay result."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        temporary.write_text(
            json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
