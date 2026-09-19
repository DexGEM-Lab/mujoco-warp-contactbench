#!/usr/bin/env python3
"""Summarize one natural ManoRL frozen-evaluation trace from physical evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


LOADED_FORCE_THRESHOLD_N = 0.02
AIRBORNE_CLEARANCE_M = 0.005
STRICT_HOLD_SECONDS = 0.25
FALLEN_CLEARANCE_M = -0.05
THUMB_REGIONS = slice(13, 16)
OTHER_FINGER_REGIONS = slice(1, 13)
CHECKPOINT_UPDATE_PATTERN = re.compile(r"\.update(\d+)\.pt$")


@dataclass(frozen=True)
class EvaluationSummary:
    checkpoint: str
    checkpoint_sha256: str
    checkpoint_update: int | None
    identity: str
    frames: int
    control_timestep_s: float
    return_total: float
    terminal_reason_code: int
    natural_horizon: bool
    deviation: bool
    fallen: bool
    nonfinite_reason: bool
    strict_hold_frames_required: int
    strict_grasp_success: bool
    loaded_contact_frames: int
    thumb_loaded_frames: int
    other_finger_loaded_frames: int
    opposing_loaded_frames: int
    airborne_frames: int
    loaded_airborne_frames: int
    thumb_loaded_airborne_frames: int
    opposing_loaded_airborne_frames: int
    longest_loaded_airborne_frames: int
    longest_opposing_loaded_airborne_frames: int
    contact_loss_fraction_after_first_loaded: float | None
    contact_loss_fraction_after_first_loaded_airborne: float | None
    peak_bottom_clearance_m: float
    minimum_bottom_clearance_m: float
    mean_object_path_error_m: float
    maximum_object_path_error_m: float
    final_object_path_error_m: float
    minimum_vertical_velocity_m_per_s: float
    selection_rank: tuple[float, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest_true(values: np.ndarray) -> int:
    if values.ndim != 1:
        raise ValueError("longest_true requires one-dimensional input")
    best = current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def loss_fraction_after(values: np.ndarray, trigger: np.ndarray) -> float | None:
    indices = np.flatnonzero(trigger)
    if not len(indices):
        return None
    suffix = values[int(indices[0]) :]
    return float(np.mean(~suffix))


def checkpoint_update(path: Path) -> int | None:
    match = CHECKPOINT_UPDATE_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def load_trace(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("full_horizon_diagnostic") is not False:
        raise ValueError("checkpoint selection requires a natural, non-diagnostic evaluation")
    if payload.get("diagnostic_boundary") != "stopped at natural first termination":
        raise ValueError("evaluation did not stop at the natural first termination")
    return payload


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    required = (
        "actual_object_position",
        "reference_object_position",
        "paired_force_on_object",
        "bottom_clearance",
        "reason_code",
        "natural_prefix_steps",
        "full_horizon_diagnostic",
    )
    with np.load(path) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError("evaluation artifact is missing: " + ", ".join(missing))
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if bool(arrays["full_horizon_diagnostic"]):
        raise ValueError("checkpoint selection rejects diagnostic continuation artifacts")
    return arrays


def validate_artifact(arrays: dict[str, np.ndarray], expected_frames: int) -> None:
    shapes = {
        "actual_object_position": (expected_frames, 3),
        "reference_object_position": (expected_frames, 3),
        "paired_force_on_object": (expected_frames, 16, 3),
        "bottom_clearance": (expected_frames,),
        "reason_code": (expected_frames,),
    }
    for name, expected in shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {arrays[name].shape}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    natural_prefix = int(arrays["natural_prefix_steps"])
    if natural_prefix != expected_frames:
        raise ValueError(
            f"natural prefix {natural_prefix} does not match artifact frame count {expected_frames}"
        )


def terminal_reason(trace: dict[str, object], arrays: dict[str, np.ndarray]) -> int:
    natural = trace.get("natural_first_termination")
    if not isinstance(natural, dict) or "reason_code" not in natural:
        raise ValueError("natural evaluation did not reach a terminal reason")
    reason = int(natural["reason_code"])
    if int(arrays["reason_code"][-1]) != reason:
        raise ValueError("trace and artifact terminal reasons disagree")
    if reason == 0:
        raise ValueError("natural terminal reason cannot be zero")
    return reason


def summarize_evaluation(
    *,
    trace_path: Path,
    artifact_path: Path,
    checkpoint_path: Path,
) -> EvaluationSummary:
    trace = load_trace(trace_path)
    arrays = load_artifact(artifact_path)
    frames = int(trace.get("steps", 0))
    if frames < 1:
        raise ValueError("evaluation trace must contain at least one frame")
    validate_artifact(arrays, frames)
    if Path(str(trace.get("checkpoint"))).resolve() != checkpoint_path.resolve():
        raise ValueError("trace checkpoint does not match requested checkpoint")

    provenance = trace.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("evaluation trace is missing provenance")
    checkpoint_provenance = provenance.get("checkpoint")
    if not isinstance(checkpoint_provenance, dict):
        raise ValueError("evaluation trace is missing checkpoint provenance")
    clock = checkpoint_provenance.get("clock")
    if not isinstance(clock, dict):
        raise ValueError("evaluation trace is missing physical clock provenance")
    control_timestep = float(clock["control_timestep"])
    if not math.isfinite(control_timestep) or control_timestep <= 0:
        raise ValueError("control timestep must be finite and positive")

    forces = np.linalg.norm(arrays["paired_force_on_object"], axis=-1)
    loaded_regions = forces > LOADED_FORCE_THRESHOLD_N
    loaded = loaded_regions.any(axis=1)
    thumb_loaded = loaded_regions[:, THUMB_REGIONS].any(axis=1)
    other_finger_loaded = loaded_regions[:, OTHER_FINGER_REGIONS].any(axis=1)
    opposing_loaded = thumb_loaded & other_finger_loaded
    airborne = arrays["bottom_clearance"] > AIRBORNE_CLEARANCE_M
    loaded_airborne = loaded & airborne
    thumb_loaded_airborne = thumb_loaded & airborne
    opposing_loaded_airborne = opposing_loaded & airborne

    actual_position = arrays["actual_object_position"]
    reference_position = arrays["reference_object_position"]
    path_error = np.linalg.norm(actual_position - reference_position, axis=1)
    if frames > 1:
        vertical_velocity = np.diff(actual_position[:, 2]) / control_timestep
        minimum_vertical_velocity = float(vertical_velocity.min())
    else:
        minimum_vertical_velocity = 0.0

    reason = terminal_reason(trace, arrays)
    hold_frames = max(1, math.ceil(STRICT_HOLD_SECONDS / control_timestep))
    longest_loaded_airborne = longest_true(loaded_airborne)
    longest_opposing_airborne = longest_true(opposing_loaded_airborne)
    natural_horizon = reason == 1
    fallen = bool(reason & 4) or bool(arrays["bottom_clearance"].min() < FALLEN_CLEARANCE_M)
    strict_success = natural_horizon and longest_loaded_airborne >= hold_frames
    contact_loss_after_loaded = loss_fraction_after(loaded, loaded)
    contact_loss_after_airborne = loss_fraction_after(loaded, loaded_airborne)

    rank = (
        float(strict_success),
        float(natural_horizon),
        float(np.count_nonzero(loaded_airborne)),
        float(longest_loaded_airborne),
        float(np.count_nonzero(opposing_loaded_airborne)),
        float(np.count_nonzero(opposing_loaded)),
        float(np.count_nonzero(thumb_loaded)),
        -float(fallen),
        -float(contact_loss_after_airborne if contact_loss_after_airborne is not None else 1.0),
        -float(path_error.mean()),
    )
    return EvaluationSummary(
        checkpoint=str(checkpoint_path.resolve()),
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_update=checkpoint_update(checkpoint_path),
        identity=str(trace["identity"]),
        frames=frames,
        control_timestep_s=control_timestep,
        return_total=float(trace["return"]),
        terminal_reason_code=reason,
        natural_horizon=natural_horizon,
        deviation=bool(reason & 2),
        fallen=fallen,
        nonfinite_reason=bool(reason & 8),
        strict_hold_frames_required=hold_frames,
        strict_grasp_success=strict_success,
        loaded_contact_frames=int(np.count_nonzero(loaded)),
        thumb_loaded_frames=int(np.count_nonzero(thumb_loaded)),
        other_finger_loaded_frames=int(np.count_nonzero(other_finger_loaded)),
        opposing_loaded_frames=int(np.count_nonzero(opposing_loaded)),
        airborne_frames=int(np.count_nonzero(airborne)),
        loaded_airborne_frames=int(np.count_nonzero(loaded_airborne)),
        thumb_loaded_airborne_frames=int(np.count_nonzero(thumb_loaded_airborne)),
        opposing_loaded_airborne_frames=int(np.count_nonzero(opposing_loaded_airborne)),
        longest_loaded_airborne_frames=longest_loaded_airborne,
        longest_opposing_loaded_airborne_frames=longest_opposing_airborne,
        contact_loss_fraction_after_first_loaded=contact_loss_after_loaded,
        contact_loss_fraction_after_first_loaded_airborne=contact_loss_after_airborne,
        peak_bottom_clearance_m=float(arrays["bottom_clearance"].max()),
        minimum_bottom_clearance_m=float(arrays["bottom_clearance"].min()),
        mean_object_path_error_m=float(path_error.mean()),
        maximum_object_path_error_m=float(path_error.max()),
        final_object_path_error_m=float(path_error[-1]),
        minimum_vertical_velocity_m_per_s=minimum_vertical_velocity,
        selection_rank=rank,
    )


def write_summary(path: Path, summary: EvaluationSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize_evaluation(
        trace_path=args.trace.resolve(),
        artifact_path=args.artifact.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
    )
    if args.output:
        write_summary(args.output.resolve(), summary)
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
