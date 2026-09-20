#!/usr/bin/env python3
"""Reduce ManoRL PPO JSONL into auditable fixed-window convergence evidence."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CONFIG_KEYS = (
    "config/learning_rate",
    "config/learning_epochs",
    "config/mini_batches",
    "config/rollouts",
    "config/teacher_anchor_beta",
    "config/teacher_anchor_passes",
)


@dataclass(frozen=True)
class WindowSummary:
    update_start: int
    update_end: int
    transitions_end: int
    updates: int
    valid_fraction: float
    completed_episodes: int
    reference_complete_count: int
    reference_complete_rate: float | None
    deviation_count: int
    deviation_rate: float | None
    fallen_count: int
    nonfinite_count: int
    loaded_airborne_fraction: float
    airborne_5mm_fraction: float
    paired_loaded_contact_fraction: float
    mean_object_path_error_m: float
    mean_episode_max_clearance_m: float | None
    mean_episode_loaded_contact_frames: float | None
    reward_mean: float
    exact_kl_mean: float
    teacher_anchor_loss_mean: float
    sampling_seconds_mean: float
    optimizer_seconds_mean: float


@dataclass(frozen=True)
class ConvergenceSummary:
    metrics_path: str
    target_updates: int
    latest_update: int
    latest_transitions: int
    row_count: int
    continuous_updates: bool
    all_rows_valid: bool
    fixed_config: dict[str, float | int | bool]
    evidence_status: str
    first_loaded_airborne_update: int | None
    first_reference_complete_update: int | None
    peak_loaded_airborne_update: int
    peak_loaded_airborne_fraction: float
    peak_paired_contact_update: int
    peak_paired_contact_fraction: float
    peak_bottom_clearance_update: int
    peak_bottom_clearance_m: float
    active_sampling_optimizer_seconds: float
    windows: tuple[WindowSummary, ...]
    last_200_updates: WindowSummary
    interpretation: str


def load_rows(path: Path) -> list[dict[str, float | int | bool]]:
    rows: list[dict[str, float | int | bool]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"metrics line {line_number} is not a JSON object")
        rows.append(payload)
    if not rows:
        raise ValueError("metrics JSONL is empty")
    return rows


def numeric(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metrics row is missing numeric key {key!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metrics key {key!r} is non-finite")
    return result


def optional_numeric(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metrics key {key!r} must be numeric when present")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metrics key {key!r} is non-finite")
    return result


def mean(rows: Iterable[dict[str, object]], key: str) -> float:
    values = [numeric(row, key) for row in rows]
    return sum(values) / len(values)


def weighted_optional_mean(
    rows: Iterable[dict[str, object]],
    *,
    value_key: str,
    weight_key: str,
) -> float | None:
    numerator = denominator = 0.0
    for row in rows:
        value = optional_numeric(row, value_key)
        weight = numeric(row, weight_key)
        if value is None or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator else None


def reduce_window(rows: list[dict[str, object]]) -> WindowSummary:
    if not rows:
        raise ValueError("cannot reduce an empty convergence window")
    completed = int(round(sum(numeric(row, "episodes/completed_count") for row in rows)))
    reference_complete = int(
        round(sum(numeric(row, "termination/reference_complete_count") for row in rows))
    )
    deviation = int(round(sum(numeric(row, "termination/deviation_count") for row in rows)))
    fallen = int(round(sum(numeric(row, "termination/fallen_count") for row in rows)))
    nonfinite = int(round(sum(numeric(row, "termination/nonfinite_count") for row in rows)))
    return WindowSummary(
        update_start=int(numeric(rows[0], "update")),
        update_end=int(numeric(rows[-1], "update")),
        transitions_end=int(numeric(rows[-1], "transitions")),
        updates=len(rows),
        valid_fraction=mean(rows, "valid"),
        completed_episodes=completed,
        reference_complete_count=reference_complete,
        reference_complete_rate=reference_complete / completed if completed else None,
        deviation_count=deviation,
        deviation_rate=deviation / completed if completed else None,
        fallen_count=fallen,
        nonfinite_count=nonfinite,
        loaded_airborne_fraction=mean(rows, "physics/loaded_airborne_fraction"),
        airborne_5mm_fraction=mean(rows, "physics/airborne_5mm_fraction"),
        paired_loaded_contact_fraction=mean(
            rows,
            "physics/paired_loaded_contact_fraction",
        ),
        mean_object_path_error_m=mean(rows, "physics/object_position_error_l2_mean"),
        mean_episode_max_clearance_m=weighted_optional_mean(
            rows,
            value_key="episodes/max_clearance_mean",
            weight_key="episodes/completed_count",
        ),
        mean_episode_loaded_contact_frames=weighted_optional_mean(
            rows,
            value_key="episodes/loaded_contact_frames_mean",
            weight_key="episodes/completed_count",
        ),
        reward_mean=mean(rows, "reward_mean"),
        exact_kl_mean=mean(rows, "info/exact_kl_mean"),
        teacher_anchor_loss_mean=mean(rows, "teacher_anchor/loss"),
        sampling_seconds_mean=mean(rows, "performance/sampling_time"),
        optimizer_seconds_mean=mean(rows, "performance/optimizer_time"),
    )


def fixed_config(rows: list[dict[str, object]]) -> dict[str, float | int | bool]:
    result: dict[str, float | int | bool] = {}
    first = rows[0]
    for key in CONFIG_KEYS:
        value = first.get(key)
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"first metrics row is missing config key {key!r}")
        if any(row.get(key) != value for row in rows[1:]):
            raise ValueError(f"training config changed within one metrics stream: {key}")
        result[key] = value
    return result


def first_positive_update(rows: list[dict[str, object]], key: str) -> int | None:
    for row in rows:
        if numeric(row, key) > 0:
            return int(numeric(row, "update"))
    return None


def peak_row(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    return max(rows, key=lambda row: numeric(row, key))


def summarize_convergence(
    path: Path,
    *,
    target_updates: int = 1000,
    window_updates: int = 50,
) -> ConvergenceSummary:
    if target_updates <= 0 or window_updates <= 0:
        raise ValueError("target and window updates must be positive")
    rows = load_rows(path)
    updates = [int(numeric(row, "update")) for row in rows]
    continuous = updates == list(range(updates[0], updates[-1] + 1))
    if updates[0] != 1:
        continuous = False
    windows = tuple(
        reduce_window(rows[start : start + window_updates])
        for start in range(0, len(rows), window_updates)
    )
    last_200 = reduce_window(rows[-min(200, len(rows)) :])
    loaded_peak = peak_row(rows, "physics/loaded_airborne_fraction")
    contact_peak = peak_row(rows, "physics/paired_loaded_contact_fraction")
    clearance_peak = peak_row(rows, "physics/bottom_clearance_max")
    all_valid = all(numeric(row, "valid") == 1.0 for row in rows)
    latest = updates[-1]
    first_complete = first_positive_update(rows, "termination/reference_complete_count")
    if latest < target_updates:
        status = "training_in_progress"
        interpretation = (
            "The fixed update budget is incomplete; convergence cannot be decided. "
            "Training-window contact or lift events are diagnostics, not frozen-policy success."
        )
    elif first_complete is None:
        status = "budget_complete_without_reference_completion"
        interpretation = (
            "The update budget completed without a natural reference-complete episode. "
            "The run is not converged to the full task even if transient lift/contact occurred."
        )
    else:
        status = "budget_complete_requires_frozen_evaluation"
        interpretation = (
            "At least one training episode reached the reference horizon. Frozen natural "
            "evaluation is still required before any grasp-success or convergence claim."
        )
    return ConvergenceSummary(
        metrics_path=str(path.resolve()),
        target_updates=target_updates,
        latest_update=latest,
        latest_transitions=int(numeric(rows[-1], "transitions")),
        row_count=len(rows),
        continuous_updates=continuous,
        all_rows_valid=all_valid,
        fixed_config=fixed_config(rows),
        evidence_status=status,
        first_loaded_airborne_update=first_positive_update(
            rows,
            "physics/loaded_airborne_fraction",
        ),
        first_reference_complete_update=first_complete,
        peak_loaded_airborne_update=int(numeric(loaded_peak, "update")),
        peak_loaded_airborne_fraction=numeric(
            loaded_peak,
            "physics/loaded_airborne_fraction",
        ),
        peak_paired_contact_update=int(numeric(contact_peak, "update")),
        peak_paired_contact_fraction=numeric(
            contact_peak,
            "physics/paired_loaded_contact_fraction",
        ),
        peak_bottom_clearance_update=int(numeric(clearance_peak, "update")),
        peak_bottom_clearance_m=numeric(clearance_peak, "physics/bottom_clearance_max"),
        active_sampling_optimizer_seconds=sum(
            numeric(row, "performance/sampling_time")
            + numeric(row, "performance/optimizer_time")
            for row in rows
        ),
        windows=windows,
        last_200_updates=last_200,
        interpretation=interpretation,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-updates", type=int, default=1000)
    parser.add_argument("--window-updates", type=int, default=50)
    args = parser.parse_args()
    summary = summarize_convergence(
        args.metrics.resolve(),
        target_updates=args.target_updates,
        window_updates=args.window_updates,
    )
    payload = asdict(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
