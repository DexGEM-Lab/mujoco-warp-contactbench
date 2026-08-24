"""Post-terminal quality gate for production ManoRL synthesis rows."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any, Final, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from sim.manorl.observations import CONTACT_FORCE_THRESHOLD


SYNTHESIS_ACCEPTANCE_CONTRACT: Final = (
    "manorl_synthesis_complete_final_rotation_xyz_mean35deg_"
    "hand_object_contact_gt0p2n_gt100frames_v1"
)
SYNTHESIS_FINAL_ROTATION_XYZ_MEAN_ERROR_MAX_DEG: Final[float] = 35.0
SYNTHESIS_HAND_OBJECT_CONTACT_FRAME_COUNT_THRESHOLD: Final[int] = 100
SYNTHESIS_HAND_OBJECT_CONTACT_MINIMUM_FRAMES: Final[int] = (
    SYNTHESIS_HAND_OBJECT_CONTACT_FRAME_COUNT_THRESHOLD + 1
)

FAILURE_TRAJECTORY_INCOMPLETE: Final = "deviation_before_source_completion"
FAILURE_FINAL_ROTATION: Final = (
    "final_object_rotation_xyz_mean_error_above_35deg"
)
FAILURE_CONTACT_FRAMES: Final = "hand_object_contact_frames_not_above_100"


@dataclass(frozen=True)
class SynthesisAcceptanceResult:
    """Observed values and decision for one candidate episode."""

    accepted: bool
    trajectory_complete: bool
    termination_reason_code: int
    final_rotation_xyz_abs_error_deg: tuple[float, float, float]
    final_rotation_xyz_mean_error_deg: float
    hand_object_contact_frames: int
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": SYNTHESIS_ACCEPTANCE_CONTRACT,
            "accepted": self.accepted,
            "trajectory_complete": self.trajectory_complete,
            "termination_reason_code": self.termination_reason_code,
            "final_rotation_xyz_abs_error_deg": list(
                self.final_rotation_xyz_abs_error_deg
            ),
            "final_rotation_xyz_mean_error_deg": (
                self.final_rotation_xyz_mean_error_deg
            ),
            "final_rotation_xyz_mean_error_max_deg": (
                SYNTHESIS_FINAL_ROTATION_XYZ_MEAN_ERROR_MAX_DEG
            ),
            "hand_object_contact_frames": self.hand_object_contact_frames,
            "hand_object_contact_force_threshold_N": CONTACT_FORCE_THRESHOLD,
            "hand_object_contact_frame_count_comparison": ">",
            "hand_object_contact_frame_count_threshold": (
                SYNTHESIS_HAND_OBJECT_CONTACT_FRAME_COUNT_THRESHOLD
            ),
            "hand_object_contact_minimum_frames": (
                SYNTHESIS_HAND_OBJECT_CONTACT_MINIMUM_FRAMES
            ),
            "failure_reasons": list(self.failure_reasons),
        }


def synthesis_acceptance_manifest() -> dict[str, object]:
    """Describe the fixed, conjunctive production acceptance contract."""

    return {
        "contract": SYNTHESIS_ACCEPTANCE_CONTRACT,
        "operator": "all",
        "rules": {
            "trajectory_complete": {
                "required": True,
                "termination_reason_code": 1,
            },
            "final_object_rotation": {
                "representation": "intrinsic_XYZ_euler_degrees",
                "measurement_precision": "persisted_float32_rotvec",
                "difference": "per_axis_shortest_wrapped_absolute",
                "aggregation": "arithmetic_mean_xyz",
                "comparison": "<=",
                "maximum_mean_error_deg": (
                    SYNTHESIS_FINAL_ROTATION_XYZ_MEAN_ERROR_MAX_DEG
                ),
            },
            "hand_object_contact_frames": {
                "scope": "solved_right_hand_target_object_normal_force",
                "measurement_precision": "persisted_float32_force_vectors",
                "force_magnitude_comparison": ">",
                "force_threshold_N": CONTACT_FORCE_THRESHOLD,
                "frame_count_comparison": ">",
                "frame_count_threshold": (
                    SYNTHESIS_HAND_OBJECT_CONTACT_FRAME_COUNT_THRESHOLD
                ),
                "minimum_passing_frames": (
                    SYNTHESIS_HAND_OBJECT_CONTACT_MINIMUM_FRAMES
                ),
            },
        },
    }


def _normalized_quaternion_xyzw(
    name: str, values: NDArray[np.floating] | tuple[float, ...]
) -> NDArray[np.float64]:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{name} must be one finite XYZW quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} quaternion norm must be positive")
    return quaternion / norm


def final_rotation_xyz_abs_error_deg(
    simulated_xyzw: NDArray[np.floating] | tuple[float, ...],
    reference_xyzw: NDArray[np.floating] | tuple[float, ...],
) -> NDArray[np.float64]:
    """Return independent intrinsic-XYZ Euler errors with per-axis wrapping."""

    simulated = _normalized_quaternion_xyzw("simulated", simulated_xyzw)
    reference = _normalized_quaternion_xyzw("reference", reference_xyzw)
    # scipy chooses a deterministic branch. Suppress the expected warning at
    # a gimbal singularity; the returned third angle is still deterministic.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        simulated_xyz = Rotation.from_quat(simulated).as_euler(
            "XYZ", degrees=True
        )
        reference_xyz = Rotation.from_quat(reference).as_euler(
            "XYZ", degrees=True
        )
    wrapped = (simulated_xyz - reference_xyz + 180.0) % 360.0 - 180.0
    return np.abs(np.asarray(wrapped, dtype=np.float64))


def persisted_rotation_quaternion_xyzw(
    quaternion_xyzw: NDArray[np.floating] | tuple[float, ...],
) -> NDArray[np.float64]:
    """Round-trip one rotation through the Lance float32 rotvec representation."""

    normalized = _normalized_quaternion_xyzw("object", quaternion_xyzw)
    rotvec = Rotation.from_quat(normalized).as_rotvec().astype(np.float32)
    return Rotation.from_rotvec(rotvec.astype(np.float64)).as_quat()


def count_hand_object_contact_frames(
    contact_frames: Sequence[Sequence[dict[str, Any]]],
    *,
    target_object_name: str,
    hand_name: str = "right",
) -> int:
    """Count persisted right-hand/target-object frames strictly above 0.2 N."""

    if not isinstance(target_object_name, str) or not target_object_name:
        raise ValueError("target_object_name must be a non-empty string")
    if not isinstance(hand_name, str) or not hand_name:
        raise ValueError("hand_name must be a non-empty string")
    count = 0
    for entries in contact_frames:
        frame_has_contact = False
        for entry in entries:
            if (
                entry.get("hand_name") != hand_name
                or entry.get("object_name") != target_object_name
            ):
                continue
            for pair in entry.get("contact_pairs") or ():
                # Lance stores force_normal as float32. Compare in that exact
                # representation so exporter and validator agree at 0.2 N.
                force = np.asarray(pair.get("force_normal") or (), dtype=np.float32)
                if force.shape != (3,) or not np.all(np.isfinite(force)):
                    raise ValueError("force_normal must be one finite XYZ vector")
                if np.linalg.norm(force) > np.float32(CONTACT_FORCE_THRESHOLD):
                    frame_has_contact = True
                    break
            if frame_has_contact:
                break
        count += int(frame_has_contact)
    return count


def evaluate_synthesis_acceptance(
    *,
    trajectory_complete: bool,
    termination_reason_code: int,
    simulated_final_object_quaternion_xyzw: NDArray[np.floating]
    | tuple[float, ...],
    reference_final_object_quaternion_xyzw: NDArray[np.floating]
    | tuple[float, ...],
    hand_object_contact_frames: int,
) -> SynthesisAcceptanceResult:
    """Evaluate the three required production predicates without fallbacks."""

    if (
        not isinstance(hand_object_contact_frames, (int, np.integer))
        or isinstance(hand_object_contact_frames, (bool, np.bool_))
        or int(hand_object_contact_frames) < 0
    ):
        raise ValueError("hand_object_contact_frames must be a non-negative integer")
    if (
        not isinstance(termination_reason_code, (int, np.integer))
        or isinstance(termination_reason_code, (bool, np.bool_))
    ):
        raise ValueError("termination_reason_code must be an integer")

    xyz_error = final_rotation_xyz_abs_error_deg(
        simulated_final_object_quaternion_xyzw,
        reference_final_object_quaternion_xyzw,
    )
    mean_error = float(np.mean(xyz_error))
    complete = bool(trajectory_complete) and int(termination_reason_code) == 1
    contact_frames = int(hand_object_contact_frames)
    failures: list[str] = []
    if not complete:
        failures.append(FAILURE_TRAJECTORY_INCOMPLETE)
    if mean_error > SYNTHESIS_FINAL_ROTATION_XYZ_MEAN_ERROR_MAX_DEG:
        failures.append(FAILURE_FINAL_ROTATION)
    if contact_frames <= SYNTHESIS_HAND_OBJECT_CONTACT_FRAME_COUNT_THRESHOLD:
        failures.append(FAILURE_CONTACT_FRAMES)

    return SynthesisAcceptanceResult(
        accepted=not failures,
        trajectory_complete=complete,
        termination_reason_code=int(termination_reason_code),
        final_rotation_xyz_abs_error_deg=tuple(float(value) for value in xyz_error),
        final_rotation_xyz_mean_error_deg=mean_error,
        hand_object_contact_frames=contact_frames,
        failure_reasons=tuple(failures),
    )
