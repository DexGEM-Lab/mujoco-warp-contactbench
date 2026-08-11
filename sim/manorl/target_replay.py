"""Direct Lance-row target-DOF replay in MJX-Warp.

The source of truth is one explicit synthetic Lance row.  The decoder extracts
only the arrays and lineage needed for replay, then applies the row's
post-controller targets directly to ``data.ctrl``.  No NPZ/JSON package or
policy checkpoint is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    JOINT_DOF,
    SimulationClock,
    TrajectoryIdentity,
    simulation_clock,
)
from sim.manorl.trajectory import (
    ReferenceTrajectory,
    TrajectoryBatch,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

TARGET_REPLAY_V22_ROW_CONTRACT = "synthetic_mano_28d_checkpoint_rollout_v2_2"
TARGET_REPLAY_V23_ROW_CONTRACT = "synthetic_mano_28d_checkpoint_rollout_v2_3"
TARGET_REPLAY_ROW_CONTRACT = TARGET_REPLAY_V22_ROW_CONTRACT
TARGET_REPLAY_COMPACT_ROW_CONTRACTS = frozenset(
    (
        "synthetic_mano_target_replay_visual_v1",
        "synthetic_mano_target_replay_visual_v2_contact",
    )
)
TARGET_REPLAY_FULL_ROW_CONTRACTS = frozenset(
    (TARGET_REPLAY_V22_ROW_CONTRACT, TARGET_REPLAY_V23_ROW_CONTRACT)
)
TARGET_REPLAY_ROW_CONTRACTS = frozenset(
    (*TARGET_REPLAY_FULL_ROW_CONTRACTS, *TARGET_REPLAY_COMPACT_ROW_CONTRACTS)
)
LANCE_TARGET_REPLAY_COLUMNS = (
    "index",
    "trajectory_metadata",
    "timestamp",
    "hands",
    "objects",
    "provenance",
)
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*_[0-9]{1,2}_[0-9]+$")


class TargetReplaySourceError(ValueError):
    """Raised when a Lance row cannot satisfy the target replay contract."""


def _readonly(
    values: Any, *, dtype: np.dtype[Any] = np.dtype(np.float64)
) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _mapping(row: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = row.get(name)
    if not isinstance(value, dict):
        raise TargetReplaySourceError(f"Lance row field {name!r} must be a mapping")
    return value


def _list(row: Mapping[str, Any], name: str) -> list[Any]:
    value = row.get(name)
    if not isinstance(value, list):
        raise TargetReplaySourceError(f"Lance row field {name!r} must be a list")
    return value


def _required_string(values: Mapping[str, Any], name: str, *, context: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise TargetReplaySourceError(f"{context}.{name} must be a non-empty string")
    return value


def _nonnegative_int(values: Mapping[str, Any], name: str, *, context: str) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TargetReplaySourceError(
            f"{context}.{name} must be a non-negative integer"
        )
    return int(value)


def _optional_positive_int(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TargetReplaySourceError(f"{context} must be a positive integer or null")
    return int(value)


def _row_clock(
    contract: str,
    provenance: Mapping[str, Any],
    data_fps: Any,
) -> tuple[str, int | None, SimulationClock]:
    if isinstance(data_fps, bool) or not isinstance(data_fps, int):
        raise TargetReplaySourceError("trajectory_metadata.data_fps must be an integer")
    try:
        clock = simulation_clock(data_fps)
    except (TypeError, ValueError) as exc:
        raise TargetReplaySourceError(
            "trajectory_metadata.data_fps is not a supported replay clock"
        ) from exc
    source_contract = (
        _required_string(provenance, "source_contract", context="provenance")
        if contract in TARGET_REPLAY_COMPACT_ROW_CONTRACTS
        else contract
    )
    if source_contract not in TARGET_REPLAY_FULL_ROW_CONTRACTS:
        raise TargetReplaySourceError(
            f"unsupported target replay source contract: {source_contract!r}"
        )
    explicit = contract in (
        TARGET_REPLAY_V23_ROW_CONTRACT,
        *TARGET_REPLAY_COMPACT_ROW_CONTRACTS,
    )
    reference_fps = provenance.get("reference_fps")
    if source_contract == TARGET_REPLAY_V22_ROW_CONTRACT:
        if clock.policy_fps != 200 or reference_fps is not None:
            raise TargetReplaySourceError("legacy v2.2 replay rows must use 200 Hz")
        reference_fps = None
    else:
        if reference_fps not in (None, 100, 120):
            raise TargetReplaySourceError("provenance.reference_fps is invalid")
        if clock.policy_fps in (100, 120) and reference_fps != clock.policy_fps:
            raise TargetReplaySourceError(
                "public reference/control replay clocks must be coupled"
            )
    if explicit:
        for name, expected in (
            ("control_fps", clock.policy_fps),
            ("physics_fps", clock.physics_fps),
            ("physics_substeps_per_control", clock.physics_substeps_per_control),
        ):
            if provenance.get(name) != expected:
                raise TargetReplaySourceError(f"provenance.{name} is inconsistent")
        for name, expected in (
            ("control_timestep_seconds", clock.control_timestep),
            ("physics_timestep_seconds", clock.physics_timestep),
        ):
            value = provenance.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-15)
            ):
                raise TargetReplaySourceError(f"provenance.{name} is inconsistent")
    return source_contract, reference_fps, clock


def _checkpoint_warp_ccd(
    provenance: Mapping[str, Any]
) -> tuple[int | None, int | None]:
    """Extract per-world CCD settings from serialized checkpoint metadata."""

    candidates: list[Mapping[str, Any]] = []
    direct = provenance.get("warp_ccd")
    if isinstance(direct, dict):
        candidates.append(direct)
    raw_metadata = provenance.get("checkpoint_metadata_json")
    if isinstance(raw_metadata, str) and raw_metadata:
        try:
            decoded = json.loads(raw_metadata)
        except json.JSONDecodeError as exc:
            raise TargetReplaySourceError(
                "provenance.checkpoint_metadata_json is invalid JSON"
            ) from exc
        if isinstance(decoded, dict):
            runtime = decoded.get("runtime_config")
            environment = (
                runtime.get("environment") if isinstance(runtime, dict) else None
            )
            warp = (
                environment.get("warp_ccd") if isinstance(environment, dict) else None
            )
            if isinstance(warp, dict):
                candidates.append(warp)
    candidates.append(provenance)
    for candidate in candidates:
        raw_iterations = candidate.get(
            "ccd_iterations", candidate.get("warp_ccd_iterations")
        )
        raw_contacts = candidate.get(
            "contacts_per_world", candidate.get("warp_ccd_contacts_per_world")
        )
        if raw_iterations is None and raw_contacts is None:
            continue
        iterations = _optional_positive_int(
            raw_iterations, context="Warp CCD iterations"
        )
        contacts = _optional_positive_int(
            raw_contacts, context="Warp CCD contacts_per_world"
        )
        if iterations is None or contacts is None:
            raise TargetReplaySourceError(
                "Warp CCD provenance must provide both iterations and contacts_per_world"
            )
        return iterations, contacts
    return None, None


def _right_hand(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    hands = _list(row, "hands")
    slots = metadata.get("hand_slots")
    names = metadata.get("hand_names")
    hand_index: int | None = None
    if isinstance(slots, list) and len(slots) == len(hands) and "right" in slots:
        hand_index = slots.index("right")
    elif isinstance(names, list) and len(names) == len(hands) and "right" in names:
        hand_index = names.index("right")
    elif isinstance(names, list) and names == ["right"] and hands:
        # Corrected v2.2 stores canonical right/left slots while hand_names lists
        # only active hands.  The right slot is canonical index zero.
        hand_index = 0
    if hand_index is None or not 0 <= hand_index < len(hands):
        raise TargetReplaySourceError("Lance row has no resolvable right-hand slot")
    hand = hands[hand_index]
    if not isinstance(hand, dict):
        raise TargetReplaySourceError("resolved right-hand slot must be a mapping")
    hand_name = hand.get("hand_name")
    if hand_name not in (None, "right"):
        raise TargetReplaySourceError(
            f"resolved right-hand slot reports incompatible hand_name={hand_name!r}"
        )
    return hand


def _object_state(
    row: Mapping[str, Any], metadata: Mapping[str, Any], object_type: str
) -> dict[str, Any]:
    objects = _list(row, "objects")
    names = metadata.get("object_names")
    if not isinstance(names, list) or object_type not in names:
        raise TargetReplaySourceError(
            "trajectory_metadata.object_names does not identify index.scene"
        )
    object_index = names.index(object_type)
    if not 0 <= object_index < len(objects) or not isinstance(
        objects[object_index], dict
    ):
        raise TargetReplaySourceError("Lance row lacks the selected object state")
    return objects[object_index]


def _movement_range(
    metadata: Mapping[str, Any], object_type: str, frames: int
) -> tuple[int, int]:
    trajectory_info = metadata.get("trajectory_info")
    movement = (
        trajectory_info.get("object_move")
        if isinstance(trajectory_info, dict)
        else None
    )
    entry = (
        next(
            (
                candidate
                for candidate in movement
                if isinstance(candidate, dict)
                and candidate.get("object_name") == object_type
            ),
            None,
        )
        if isinstance(movement, list)
        else None
    )
    start_value = entry.get("start_frame", 0) if entry is not None else 0
    end_value = entry.get("end_frame", frames - 1) if entry is not None else frames - 1
    if (
        isinstance(start_value, bool)
        or not isinstance(start_value, int)
        or isinstance(end_value, bool)
        or not isinstance(end_value, int)
    ):
        raise TargetReplaySourceError(
            "object movement range must contain integer frames"
        )
    start = int(start_value)
    end = int(end_value)
    if not 0 <= start <= end < frames:
        raise TargetReplaySourceError(
            f"object movement range [{start}, {end}] is outside {frames} replay frames"
        )
    return start, end


@dataclass(frozen=True)
class TargetReplaySource:
    """Validated direct Lance row and its generated/source lineage."""

    dataset_path: Path
    dataset_version: int
    row_index: int
    object_type: str
    generated_uuid: str
    source_dataset_path: str
    source_dataset_version: int
    source_row_index: int
    source_uuid: str
    source_identity: str
    checkpoint_update: int
    checkpoint_sha256: str
    row_contract: str
    source_contract: str
    reference_fps: int | None
    clock: SimulationClock
    warp_ccd_iterations: int | None
    warp_ccd_contacts_per_world: int | None
    movement_start: int
    movement_end: int
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
    def control_fps(self) -> int:
        return self.clock.policy_fps

    @property
    def control_timestep(self) -> float:
        return self.clock.control_timestep

    @property
    def physics_substeps_per_control(self) -> int:
        return self.clock.physics_substeps_per_control


def target_replay_source_from_row(
    row: Mapping[str, Any],
    *,
    dataset_path: str | Path,
    dataset_version: int,
    row_index: int,
) -> TargetReplaySource:
    """Validate one decoded synthetic row without opening Lance."""

    if not isinstance(row, Mapping):
        raise TargetReplaySourceError("decoded Lance row must be a mapping")
    path = Path(dataset_path).expanduser().resolve()
    if (
        isinstance(dataset_version, bool)
        or not isinstance(dataset_version, int)
        or dataset_version < 1
    ):
        raise TargetReplaySourceError("dataset_version must be a positive integer")
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise TargetReplaySourceError("row_index must be a non-negative integer")
    index = _mapping(row, "index")
    metadata = _mapping(row, "trajectory_metadata")
    provenance = _mapping(row, "provenance")
    contract = _required_string(provenance, "contract", context="provenance")
    if contract not in TARGET_REPLAY_ROW_CONTRACTS:
        raise TargetReplaySourceError(
            f"unsupported target replay row contract: {contract!r}"
        )
    object_type = _required_string(index, "scene", context="index")
    if index.get("is_generated") is not True:
        raise TargetReplaySourceError("index.is_generated must be true")
    generated_uuid = _required_string(index, "uuid", context="index")
    source_uuid = _required_string(index, "seed_uuid", context="index")
    source_identity = _required_string(
        provenance, "source_identity", context="provenance"
    )
    if not _IDENTITY_RE.fullmatch(source_identity):
        raise TargetReplaySourceError(
            "provenance.source_identity must be object_action_sequence"
        )
    if source_identity.split("_", 1)[0] != object_type:
        raise TargetReplaySourceError(
            "index.scene disagrees with provenance.source_identity"
        )
    source_dataset_path = _required_string(
        provenance, "dataset_path", context="provenance"
    )
    source_dataset_version = _nonnegative_int(
        provenance, "dataset_version", context="provenance"
    )
    source_row_index = _nonnegative_int(provenance, "row_index", context="provenance")
    checkpoint_update = _nonnegative_int(
        provenance, "checkpoint_update", context="provenance"
    )
    checkpoint_sha256 = _required_string(
        provenance, "checkpoint_sha256", context="provenance"
    )
    if len(checkpoint_sha256) != 64:
        raise TargetReplaySourceError(
            "provenance.checkpoint_sha256 must be a 64-character digest"
        )
    try:
        int(checkpoint_sha256, 16)
    except ValueError as exc:
        raise TargetReplaySourceError(
            "provenance.checkpoint_sha256 must be hexadecimal"
        ) from exc

    timestamps = np.asarray(row.get("timestamp", ()), dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise TargetReplaySourceError(
            "timestamp must be a one-dimensional array with at least two frames"
        )
    frames = len(timestamps)
    total_frames = metadata.get("total_frames")
    if (
        isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames != frames
    ):
        raise TargetReplaySourceError(
            f"trajectory_metadata.total_frames={total_frames!r} does not match {frames} timestamps"
        )
    data_fps = metadata.get("data_fps")
    source_contract, reference_fps, clock = _row_clock(contract, provenance, data_fps)
    hand = _right_hand(row, metadata)
    object_state = _object_state(row, metadata, object_type)
    recorded_qpos = np.asarray(hand.get("urdf_dof", ()), dtype=np.float64)
    target_qpos = np.asarray(hand.get("urdf_dof_target", ()), dtype=np.float64)
    object_position = np.asarray(object_state.get("pos", ()), dtype=np.float64)
    object_rotvec = np.asarray(object_state.get("rot_aa", ()), dtype=np.float64)
    expected = {
        "timestamp": (frames,),
        "urdf_dof": (frames, JOINT_DOF),
        "urdf_dof_target": (frames, JOINT_DOF),
        "object_position": (frames, 3),
        "object_rot_aa": (frames, 3),
    }
    values = {
        "timestamp": timestamps,
        "urdf_dof": recorded_qpos,
        "urdf_dof_target": target_qpos,
        "object_position": object_position,
        "object_rot_aa": object_rotvec,
    }
    for name, shape in expected.items():
        if values[name].shape != shape:
            raise TargetReplaySourceError(
                f"{name} shape {values[name].shape} does not match {shape}"
            )
        if not np.all(np.isfinite(values[name])):
            raise TargetReplaySourceError(f"{name} contains non-finite values")
    if not math.isclose(float(timestamps[0]), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise TargetReplaySourceError("timestamps must start at zero")
    deltas = np.diff(timestamps)
    if np.any(deltas <= 0) or not np.allclose(
        deltas, clock.control_timestep, rtol=0.0, atol=1e-10
    ):
        raise TargetReplaySourceError(
            f"timestamps must advance strictly by {clock.control_timestep} seconds"
        )
    object_quaternion = Rotation.from_rotvec(object_rotvec).as_quat()
    quaternion_norms = np.linalg.norm(object_quaternion, axis=1)
    if not np.allclose(quaternion_norms, 1.0, rtol=0.0, atol=1e-10):
        raise TargetReplaySourceError(
            "object rotations did not produce unit quaternions"
        )
    movement_start, movement_end = _movement_range(metadata, object_type, frames)
    ccd_iterations, ccd_contacts = _checkpoint_warp_ccd(provenance)
    return TargetReplaySource(
        dataset_path=path,
        dataset_version=dataset_version,
        row_index=row_index,
        object_type=object_type,
        generated_uuid=generated_uuid,
        source_dataset_path=source_dataset_path,
        source_dataset_version=source_dataset_version,
        source_row_index=source_row_index,
        source_uuid=source_uuid,
        source_identity=source_identity,
        checkpoint_update=checkpoint_update,
        checkpoint_sha256=checkpoint_sha256,
        row_contract=contract,
        source_contract=source_contract,
        reference_fps=reference_fps,
        clock=clock,
        warp_ccd_iterations=ccd_iterations,
        warp_ccd_contacts_per_world=ccd_contacts,
        movement_start=movement_start,
        movement_end=movement_end,
        timestamps=_readonly(timestamps),
        recorded_qpos=_readonly(recorded_qpos),
        target_qpos=_readonly(target_qpos),
        object_position=_readonly(object_position),
        object_quaternion_xyzw=_readonly(object_quaternion),
    )


def load_target_replay_source(
    dataset_path: str | Path,
    *,
    dataset_version: int,
    row_index: int,
) -> TargetReplaySource:
    """Read exactly one target trajectory directly from Lance."""

    path = Path(dataset_path).expanduser().resolve()
    try:
        import lance

        dataset = lance.dataset(str(path), version=dataset_version)
        rows = dataset.take(
            [row_index], columns=list(LANCE_TARGET_REPLAY_COLUMNS)
        ).to_pylist()
    except Exception as exc:
        raise TargetReplaySourceError(
            f"could not decode Lance row {row_index} from {path} at version {dataset_version}"
        ) from exc
    if len(rows) != 1:
        raise TargetReplaySourceError(
            f"Lance take returned {len(rows)} rows for requested row {row_index}"
        )
    return target_replay_source_from_row(
        rows[0],
        dataset_path=path,
        dataset_version=dataset_version,
        row_index=row_index,
    )


def _trajectory_for_source(source: TargetReplaySource) -> ReferenceTrajectory:
    identity = TrajectoryIdentity(
        dataset_path=source.source_dataset_path,
        dataset_version=source.source_dataset_version,
        row_index=source.source_row_index,
        object_index=0,
        uuid=source.source_uuid,
        file_uuid="",
        identity=source.source_identity,
        source_start=0,
        source_stop=source.frames,
        movement_start_raw=source.movement_start,
        movement_end_raw=source.movement_end,
    )
    return ReferenceTrajectory(
        identity=identity,
        dataset_version=identity.dataset_version,
        source_indices=_readonly(np.arange(source.frames), dtype=np.dtype(np.int64)),
        timestamps=source.timestamps,
        q_ref=source.recorded_qpos,
        object_pos_raw=source.object_position,
        object_pos=source.object_position,
        object_quat_xyzw=source.object_quaternion_xyzw,
        object_z_shift=0.0,
        hand_sides=("right",),
        q_ref_by_side={"right": source.recorded_qpos},
        selected_hand_sides=("right",),
        reference_fps=source.reference_fps,
        control_fps=source.control_fps,
        movement_start_step=source.movement_start,
        movement_end_step=source.movement_end,
    )


@dataclass(frozen=True)
class ReplayState:
    """One post-physics replay state in task array conventions."""

    qpos: np.ndarray
    object_position: np.ndarray
    object_quaternion_xyzw: np.ndarray


class TargetDofReplay:
    """Apply one direct Lance target sequence to the MJX-Warp environment."""

    def __init__(
        self,
        source: TargetReplaySource,
        *,
        device: str = "gpu",
        allow_physics_override: bool = False,
    ) -> None:
        if device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'")
        self.source = source
        self.device = device
        self.physics_overrides: list[str] = []
        ccd_iterations = source.warp_ccd_iterations
        ccd_contacts = source.warp_ccd_contacts_per_world
        if device == "cpu" and (ccd_iterations is not None or ccd_contacts is not None):
            if not allow_physics_override:
                raise TargetReplaySourceError(
                    "this Lance row records explicit Warp CCD settings that require device='gpu'; "
                    "rerun with --allow-physics-override for a non-identical CPU diagnostic"
                )
            self.physics_overrides.append(
                "CPU replay omitted Lance row Warp CCD allocation"
            )
            ccd_iterations = None
            ccd_contacts = None
        from sim.manorl.environment import (
            EnvironmentConfig,
            MujocoManoEnvironment,
            recommended_warp_contact_capacity,
        )

        self.environment = MujocoManoEnvironment(
            TrajectoryBatch((_trajectory_for_source(source),)),
            EnvironmentConfig(
                device=device,
                num_envs=1,
                hand_side="right",
                residual_enabled=False,
                point_sampling_backend="numpy_per_env",
                capture_transition_diagnostics=False,
                reference_fps=source.reference_fps,
                control_fps=source.control_fps,
                post_padding=0,
                contact_capacity=recommended_warp_contact_capacity(1, ("right",)),
                warp_ccd_iterations=ccd_iterations,
                warp_ccd_contacts_per_world=ccd_contacts,
            ),
        )
        self._frame = 0
        self.reset()

    @property
    def frame(self) -> int:
        return self._frame

    def reset(self) -> ReplayState:
        """Restore the row's recorded frame-0 reset state."""

        environment = self.environment
        qpos = np.asarray(environment.data.qpos, dtype=np.float64).copy()
        qvel = np.zeros_like(np.asarray(environment.data.qvel, dtype=np.float64))
        ctrl = np.asarray(environment.data.ctrl, dtype=np.float64).copy()
        hand_slice = environment.producer.hand_qpos_slices["right"]
        object_address = environment.producer.object_qpos_address
        qpos[0, hand_slice] = self.source.recorded_qpos[0]
        qpos[0, object_address : object_address + 3] = self.source.object_position[0]
        qpos[0, object_address + 3 : object_address + 7] = xyzw_to_wxyz(
            self.source.object_quaternion_xyzw[0]
        )
        ctrl[0, :JOINT_DOF] = self.source.target_qpos[0]
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
        """Apply the next row target and its recorded MJX-Warp substeps."""

        if self._frame >= self.source.transitions:
            raise IndexError("target replay is at its final frame; call reset()")
        environment = self.environment
        ctrl = np.asarray(environment.data.ctrl, dtype=np.float64).copy()
        ctrl[0, :JOINT_DOF] = self.source.target_qpos[self._frame]
        environment.data = environment.data.replace(
            ctrl=environment.jax.device_put(
                environment.jp.asarray(ctrl), environment.device
            )
        )
        for _ in range(self.source.physics_substeps_per_control):
            environment.data = environment._step_fn(environment.data)
            environment._check_warp_ccd_overflow()
        self._frame += 1
        return self.state()

    def host_data(self) -> Any:
        return self.environment.host_data(0)

    def headless_report(
        self,
        *,
        transitions: int | None = None,
        max_object_position_error_m: float = 0.05,
        max_object_rotation_error_rad: float = 0.25,
        max_qpos_abs_error: float = 1.0,
    ) -> dict[str, Any]:
        if transitions is None:
            transitions = self.source.transitions
        if (
            isinstance(transitions, bool)
            or not isinstance(transitions, int)
            or not 1 <= transitions <= self.source.transitions
        ):
            raise ValueError(f"transitions must be in [1, {self.source.transitions}]")
        initial_state = self.reset()
        initial_qpos_error = float(
            np.max(np.abs(initial_state.qpos - self.source.recorded_qpos[0]))
        )
        initial_object_position_error = float(
            np.linalg.norm(
                initial_state.object_position - self.source.object_position[0]
            )
        )
        initial_object_rotation_error = _rotation_error_rad(
            initial_state.object_quaternion_xyzw,
            self.source.object_quaternion_xyzw[0],
        )
        q_errors: list[float] = []
        object_errors: list[float] = []
        rotation_errors: list[float] = []
        replay_z = [float(initial_state.object_position[2])]
        for index in range(transitions):
            state = self.step()
            replay_z.append(float(state.object_position[2]))
            q_errors.append(
                float(np.max(np.abs(state.qpos - self.source.recorded_qpos[index + 1])))
            )
            object_errors.append(
                float(
                    np.linalg.norm(
                        state.object_position - self.source.object_position[index + 1]
                    )
                )
            )
            rotation_errors.append(
                _rotation_error_rad(
                    state.object_quaternion_xyzw,
                    self.source.object_quaternion_xyzw[index + 1],
                )
            )
        q_values = np.asarray(q_errors, dtype=np.float64)
        object_values = np.asarray(object_errors, dtype=np.float64)
        rotation_values = np.asarray(rotation_errors, dtype=np.float64)
        return {
            "schema": "manorl.target_dof_lance_replay_result.v1",
            "status": (
                "pass"
                if (
                    float(np.max(object_values)) <= max_object_position_error_m
                    and float(np.max(rotation_values)) <= max_object_rotation_error_rad
                    and float(np.max(q_values)) <= max_qpos_abs_error
                )
                else "fail"
            ),
            "lance_dataset": str(self.source.dataset_path),
            "lance_dataset_version": self.source.dataset_version,
            "lance_row_index": self.source.row_index,
            "generated_uuid": self.source.generated_uuid,
            "source_dataset": self.source.source_dataset_path,
            "source_dataset_version": self.source.source_dataset_version,
            "source_row_index": self.source.source_row_index,
            "source_uuid": self.source.source_uuid,
            "source_identity": self.source.source_identity,
            "checkpoint_update": self.source.checkpoint_update,
            "checkpoint_sha256": self.source.checkpoint_sha256,
            "row_contract": self.source.row_contract,
            "source_contract": self.source.source_contract,
            "object": self.source.object_type,
            "device": self.device,
            "states": transitions + 1,
            "transitions": transitions,
            "reference_fps": self.source.reference_fps,
            "control_fps": self.source.control_fps,
            "control_timestep_seconds": self.source.control_timestep,
            "physics_fps": self.source.clock.physics_fps,
            "physics_timestep_seconds": self.source.clock.physics_timestep,
            "physics_substeps_per_control": self.source.physics_substeps_per_control,
            "physics_substeps_per_target": self.source.physics_substeps_per_control,
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
                np.max(self.source.object_position[: transitions + 1, 2])
                - self.source.object_position[0, 2]
            ),
            "replay_max_lift_m": float(
                max(replay_z) - self.source.object_position[0, 2]
            ),
            "thresholds": {
                "max_object_position_error_m": max_object_position_error_m,
                "max_object_rotation_error_rad": max_object_rotation_error_rad,
                "max_qpos_abs_error": max_qpos_abs_error,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


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
    """Open a native viewer while MJX-Warp remains the sole simulator."""

    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    if print_every < 1:
        raise ValueError("print_every must be positive")
    if transitions is None:
        transitions = replay.source.transitions
    if (
        not isinstance(transitions, int)
        or isinstance(transitions, bool)
        or not 1 <= transitions <= replay.source.transitions
    ):
        raise ValueError(f"transitions must be in [1, {replay.source.transitions}]")
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
                        replay.source.control_timestep / speed
                        - (time.perf_counter() - started),
                    )
                )
            if not loop:
                return


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
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
