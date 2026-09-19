"""Reference-object movement onset detection for complete RL episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

REFERENCE_MOTION_ANNOTATION_CONTRACT = (
    "manorl.reference_object_quiescent_departure.v1"
)


@dataclass(frozen=True)
class ReferenceMotionConfig:
    """Physical-time thresholds for separating settling from task motion."""

    stable_window_seconds: float = 0.10
    short_quiet_seconds: float = 0.04
    velocity_smoothing_seconds: float = 0.025
    quiet_translation_speed_m_s: float = 0.02
    quiet_rotation_speed_rad_s: float = 0.25
    start_translation_m: float = 0.002
    confirm_translation_m: float = 0.005
    start_rotation_rad: float = math.radians(2.0)
    confirm_rotation_rad: float = math.radians(5.0)
    persistence_seconds: float = 0.10
    confirmation_seconds: float = 0.50
    persistence_fraction: float = 0.70
    fallback_search_fraction: float = 0.35

    def __post_init__(self) -> None:
        positive = (
            "stable_window_seconds",
            "short_quiet_seconds",
            "velocity_smoothing_seconds",
            "quiet_translation_speed_m_s",
            "quiet_rotation_speed_rad_s",
            "start_translation_m",
            "confirm_translation_m",
            "start_rotation_rad",
            "confirm_rotation_rad",
            "persistence_seconds",
            "confirmation_seconds",
            "fallback_search_fraction",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.start_translation_m >= self.confirm_translation_m:
            raise ValueError("translation confirmation must exceed start threshold")
        if self.start_rotation_rad >= self.confirm_rotation_rad:
            raise ValueError("rotation confirmation must exceed start threshold")
        if not 0.0 < self.persistence_fraction <= 1.0:
            raise ValueError("persistence_fraction must be within (0, 1]")
        if not 0.0 < self.fallback_search_fraction <= 1.0:
            raise ValueError("fallback_search_fraction must be within (0, 1]")

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class ReferenceMotionAnnotation:
    """One complete row's movement-onset result in source-frame coordinates."""

    start_frame: int
    end_frame: int
    confidence: str
    status: str
    motion_mode: str
    observed_fps: float
    stable_start_frame: int | None
    stable_end_frame: int | None
    candidate_frame: int | None
    onset_translation_from_baseline_m: float
    onset_rotation_from_baseline_rad: float
    max_translation_from_baseline_m: float
    max_rotation_from_baseline_rad: float

    def __post_init__(self) -> None:
        if not 0 <= self.start_frame <= self.end_frame:
            raise ValueError("movement annotation frames are invalid")
        if self.confidence not in {"high", "low"}:
            raise ValueError("movement annotation confidence must be high or low")
        if self.status not in {
            "high",
            "low_no_quiet",
            "low_unconfirmed",
            "initial_transient_only",
        }:
            raise ValueError(f"unsupported movement annotation status: {self.status}")
        if self.motion_mode not in {
            "translation",
            "rotation",
            "both",
            "none",
        }:
            raise ValueError(f"unsupported movement mode: {self.motion_mode}")
        if not math.isfinite(self.observed_fps) or self.observed_fps <= 0.0:
            raise ValueError("observed_fps must be finite and positive")
        stable = (self.stable_start_frame, self.stable_end_frame)
        if (stable[0] is None) != (stable[1] is None):
            raise ValueError("stable window frames must be provided together")
        if stable[0] is not None and not 0 <= stable[0] <= stable[1] <= self.end_frame:
            raise ValueError("stable window lies outside the trajectory")
        if self.candidate_frame is not None and not 0 <= self.candidate_frame <= self.end_frame:
            raise ValueError("candidate frame lies outside the trajectory")
        metrics = (
            self.onset_translation_from_baseline_m,
            self.onset_rotation_from_baseline_rad,
            self.max_translation_from_baseline_m,
            self.max_rotation_from_baseline_rad,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in metrics):
            raise ValueError("movement annotation metrics must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": REFERENCE_MOTION_ANNOTATION_CONTRACT,
            **asdict(self),
        }


def _true_runs(mask: NDArray[np.bool_], minimum_length: int) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("run mask must be one-dimensional")
    if minimum_length < 1:
        raise ValueError("minimum run length must be positive")
    boundaries = np.diff(np.concatenate(([False], values, [False])).astype(np.int8))
    starts = np.flatnonzero(boundaries == 1)
    stops = np.flatnonzero(boundaries == -1)
    return [
        (int(start), int(stop - 1))
        for start, stop in zip(starts, stops, strict=True)
        if stop - start >= minimum_length
    ]


def _odd_window(seconds: float, fps: float, *, minimum: int = 3) -> int:
    value = max(minimum, int(round(seconds * fps)))
    return value if value % 2 else value + 1


def _motion_mode(
    translation: float,
    rotation: float,
    *,
    translation_threshold: float,
    rotation_threshold: float,
) -> str:
    translated = translation >= translation_threshold
    rotated = rotation >= rotation_threshold
    if translated and rotated:
        return "both"
    if translated:
        return "translation"
    if rotated:
        return "rotation"
    return "none"


def detect_reference_motion(
    positions: NDArray[np.floating[Any]],
    rotation_vectors: NDArray[np.floating[Any]],
    timestamps: NDArray[np.floating[Any]],
    *,
    config: ReferenceMotionConfig = ReferenceMotionConfig(),
) -> ReferenceMotionAnnotation:
    """Find the first sustained reference-pose departure after initial settling.

    A raw first-difference detector labels gravity settling as task motion. This
    detector first finds a physical-time quiescent window, anchors the settled
    pose there, then finds a persistent 2 mm / 2 degree departure that reaches
    5 mm / 5 degrees within the confirmation horizon. Rows without an absolute
    quiet window use the least-active early window with low confidence. Rows
    with no post-baseline departure are explicitly labeled as initial-transient
    only and start at frame zero rather than being silently dropped.
    """

    position = np.asarray(positions, dtype=np.float64)
    rotvec = np.asarray(rotation_vectors, dtype=np.float64)
    time = np.asarray(timestamps, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("positions must have shape (frames, 3)")
    frames = len(position)
    if frames < 3 or rotvec.shape != (frames, 3) or time.shape != (frames,):
        raise ValueError("reference pose/timestamp arrays have inconsistent shapes")
    if not all(np.all(np.isfinite(value)) for value in (position, rotvec, time)):
        raise ValueError("reference pose/timestamp arrays must be finite")
    deltas = np.diff(time)
    if np.any(deltas <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    median_timestep = float(np.median(deltas))
    fps = 1.0 / median_timestep

    rotations = Rotation.from_rotvec(rotvec)
    translation_speed = np.linalg.norm(np.diff(position, axis=0), axis=1) / deltas
    rotation_speed = (rotations[:-1].inv() * rotations[1:]).magnitude() / deltas
    smoothing = _odd_window(config.velocity_smoothing_seconds, fps)
    smooth_translation = median_filter(
        translation_speed, size=smoothing, mode="nearest"
    )
    smooth_rotation = median_filter(rotation_speed, size=smoothing, mode="nearest")
    quiet = (
        (smooth_translation <= config.quiet_translation_speed_m_s)
        & (smooth_rotation <= config.quiet_rotation_speed_rad_s)
    )
    stable_length = max(3, int(round(config.stable_window_seconds * fps)))
    stable_runs = _true_runs(quiet, stable_length)

    if stable_runs:
        stable_start = stable_runs[0][0]
        stable_end = min(stable_runs[0][1], stable_start + stable_length - 1)
        status = "high"
        confidence = "high"
    else:
        search_stop = min(
            len(quiet),
            max(stable_length, int(round(config.fallback_search_fraction * frames))),
        )
        normalized_activity = (
            smooth_translation / config.quiet_translation_speed_m_s
            + smooth_rotation / config.quiet_rotation_speed_rad_s
        )
        if search_stop < stable_length:
            stable_start = 0
        else:
            window_scores = np.convolve(
                normalized_activity[:search_stop],
                np.ones(stable_length, dtype=np.float64),
                mode="valid",
            )
            stable_start = int(np.argmin(window_scores))
        stable_end = min(frames - 2, stable_start + stable_length - 1)
        status = "low_no_quiet"
        confidence = "low"

    # Velocity index i describes the transition from pose i to i+1. Include
    # pose stable_end+1 when estimating the settled reference pose.
    baseline_stop = min(frames, stable_end + 2)
    baseline_position = np.median(position[stable_start:baseline_stop], axis=0)
    baseline_rotation = rotations[stable_start:baseline_stop].mean()
    translation_from_baseline = np.linalg.norm(
        position - baseline_position, axis=1
    )
    rotation_from_baseline = (baseline_rotation.inv() * rotations).magnitude()
    started = (
        (translation_from_baseline >= config.start_translation_m)
        | (rotation_from_baseline >= config.start_rotation_rad)
    )
    confirmed = (
        (translation_from_baseline >= config.confirm_translation_m)
        | (rotation_from_baseline >= config.confirm_rotation_rad)
    )
    persistence = max(2, int(round(config.persistence_seconds * fps)))
    confirmation = max(
        persistence, int(round(config.confirmation_seconds * fps))
    )

    candidate: int | None = None
    for frame in range(stable_end + 1, frames):
        if not started[frame]:
            continue
        persist_stop = min(frames, frame + persistence)
        if float(np.mean(started[frame:persist_stop])) < config.persistence_fraction:
            continue
        confirm_stop = min(frames, frame + confirmation)
        if np.any(confirmed[frame:confirm_stop]):
            candidate = frame
            break

    if candidate is None:
        unconfirmed = np.flatnonzero(started[stable_end + 1 :])
        if len(unconfirmed):
            candidate = stable_end + 1 + int(unconfirmed[0])
            status = "low_unconfirmed"
            confidence = "low"
        else:
            start_frame = 0
            max_translation = float(np.max(translation_from_baseline))
            max_rotation = float(np.max(rotation_from_baseline))
            mode = _motion_mode(
                max_translation,
                max_rotation,
                translation_threshold=config.confirm_translation_m,
                rotation_threshold=config.confirm_rotation_rad,
            )
            return ReferenceMotionAnnotation(
                start_frame=start_frame,
                end_frame=frames - 1,
                confidence="low",
                status="initial_transient_only",
                motion_mode=mode,
                observed_fps=fps,
                stable_start_frame=stable_start,
                stable_end_frame=stable_end,
                candidate_frame=None,
                onset_translation_from_baseline_m=float(
                    translation_from_baseline[start_frame]
                ),
                onset_rotation_from_baseline_rad=float(
                    rotation_from_baseline[start_frame]
                ),
                max_translation_from_baseline_m=max_translation,
                max_rotation_from_baseline_rad=max_rotation,
            )

    short_quiet = max(2, int(round(config.short_quiet_seconds * fps)))
    prior_quiet_runs = [
        run
        for run in _true_runs(quiet[:candidate], short_quiet)
        if run[0] >= stable_start
    ]
    start_frame = (
        prior_quiet_runs[-1][1] + 1 if prior_quiet_runs else candidate
    )
    start_frame = min(start_frame, candidate)
    mode = _motion_mode(
        float(translation_from_baseline[candidate]),
        float(rotation_from_baseline[candidate]),
        translation_threshold=config.start_translation_m,
        rotation_threshold=config.start_rotation_rad,
    )
    return ReferenceMotionAnnotation(
        start_frame=start_frame,
        end_frame=frames - 1,
        confidence=confidence,
        status=status,
        motion_mode=mode,
        observed_fps=fps,
        stable_start_frame=stable_start,
        stable_end_frame=stable_end,
        candidate_frame=candidate,
        onset_translation_from_baseline_m=float(
            translation_from_baseline[start_frame]
        ),
        onset_rotation_from_baseline_rad=float(rotation_from_baseline[start_frame]),
        max_translation_from_baseline_m=float(np.max(translation_from_baseline)),
        max_rotation_from_baseline_rad=float(np.max(rotation_from_baseline)),
    )


def annotate_rl_episode_row(
    row: Mapping[str, Any],
    *,
    config: ReferenceMotionConfig = ReferenceMotionConfig(),
) -> tuple[dict[str, Any], ReferenceMotionAnnotation]:
    """Return a source-preserving row with standard and diagnostic annotations."""

    if not isinstance(row, Mapping):
        raise TypeError("row must be a mapping")
    index = row.get("index")
    metadata = row.get("trajectory_metadata")
    objects = row.get("objects")
    timestamps = row.get("timestamp")
    if not isinstance(index, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("row lacks index or trajectory_metadata")
    object_name = str(index.get("scene", "")).strip()
    if not object_name or "," in object_name:
        raise ValueError("motion annotation requires exactly one scene object")
    if not isinstance(objects, list) or len(objects) != 1 or not isinstance(objects[0], Mapping):
        raise ValueError("motion annotation requires exactly one object pose track")
    state = objects[0]
    annotation = detect_reference_motion(
        np.asarray(state.get("pos", ()), dtype=np.float64),
        np.asarray(state.get("rot_aa", ()), dtype=np.float64),
        np.asarray(timestamps, dtype=np.float64),
        config=config,
    )
    total_frames = int(metadata.get("total_frames", -1))
    if total_frames != annotation.end_frame + 1:
        raise ValueError("trajectory_metadata.total_frames differs from object track")
    if metadata.get("trajectory_info") not in (None, {}):
        raise ValueError("row already carries trajectory_info; refusing silent overwrite")
    annotated_metadata = dict(metadata)
    annotated_metadata["trajectory_info"] = {
        "object_move": [
            {
                "object_name": object_name,
                "start_frame": annotation.start_frame,
                # End is the complete episode boundary. This task detects onset
                # only and does not claim a separately detected motion stop.
                "end_frame": annotation.end_frame,
            }
        ]
    }
    annotated_metadata["reference_motion_annotation"] = annotation.to_dict()
    result = dict(row)
    result["trajectory_metadata"] = annotated_metadata
    return result, annotation
