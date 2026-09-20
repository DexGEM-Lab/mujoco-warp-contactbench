"""Three-stage curriculum for contact-conditioned autonomous grasping."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np


CURRICULUM_STAGE_NAMES = {
    1: "approach_thumb_opposition",
    2: "stable_grip_low_lift",
    3: "full_lift_hold_recovery",
}


@dataclass(frozen=True)
class CurriculumConfig:
    opposition_threshold: float = 0.10
    positive_lift_threshold: float = 0.05
    stable_airborne_threshold: float = 0.03
    averaging_updates: int = 8
    minimum_updates_per_stage: int = 16

    def __post_init__(self) -> None:
        rates = (
            self.opposition_threshold,
            self.positive_lift_threshold,
            self.stable_airborne_threshold,
        )
        if any(not np.isfinite(value) or not 0 < value <= 1 for value in rates):
            raise ValueError("curriculum thresholds must be finite fractions")
        if (
            type(self.averaging_updates) is not int
            or self.averaging_updates < 1
            or type(self.minimum_updates_per_stage) is not int
            or self.minimum_updates_per_stage < self.averaging_updates
        ):
            raise ValueError("curriculum update windows are inconsistent")


def reference_stage_candidates(cache) -> dict[int, tuple[int, ...]]:
    """Derive deterministic approach/contact/lift/hold reset candidates."""
    proximity = (
        np.asarray(cache.proximity)
        * np.asarray(cache.confidence)
        * np.asarray(cache.valid)
    )
    intent = proximity.max(axis=1)
    contact_hits = np.flatnonzero(intent >= 0.5)
    contact = int(contact_hits[0]) if contact_hits.size else int(np.argmax(intent))

    clearance = np.asarray(cache.reference_bottom) - float(cache.table_height)
    upward = np.asarray(cache.object_v_com)[:, 2]
    lift_hits = np.flatnonzero((clearance >= 0.005) | (upward >= 0.02))
    lift = int(lift_hits[0]) if lift_hits.size else contact
    airborne_hits = np.flatnonzero(clearance >= 0.03)
    airborne = int(airborne_hits[0]) if airborne_hits.size else lift
    last = len(clearance) - 1

    def clipped(*values: int) -> tuple[int, ...]:
        return tuple(dict.fromkeys(max(0, min(last, value)) for value in values))

    return {
        1: clipped(contact - 24, contact - 8, contact),
        2: clipped(contact, lift - 8, lift),
        3: clipped(lift, airborne - 8, airborne),
    }


class CurriculumController:
    """Host-side stage controller driven only by update-level telemetry."""

    def __init__(self, config: CurriculumConfig = CurriculumConfig()) -> None:
        self.config = config
        self.stage = 1
        self.stage_updates = 0
        self._opposition = deque(maxlen=config.averaging_updates)
        self._positive_lift = deque(maxlen=config.averaging_updates)
        self._stable_airborne = deque(maxlen=config.averaging_updates)

    def _mean(self, values: deque[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def observe(self, metrics: Mapping[str, float]) -> dict[str, float]:
        keys = (
            "curriculum/opposing_loaded_fraction",
            "curriculum/positive_lift_fraction",
            "curriculum/stable_airborne_fraction",
        )
        if any(key not in metrics for key in keys):
            raise ValueError("curriculum telemetry is incomplete")
        values = tuple(float(metrics[key]) for key in keys)
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("curriculum telemetry must contain finite fractions")
        self._opposition.append(values[0])
        self._positive_lift.append(values[1])
        self._stable_airborne.append(values[2])
        self.stage_updates += 1

        opposing_mean = self._mean(self._opposition)
        positive_lift_mean = self._mean(self._positive_lift)
        stable_airborne_mean = self._mean(self._stable_airborne)
        promoted = False
        window_ready = (
            len(self._opposition) == self.config.averaging_updates
            and self.stage_updates >= self.config.minimum_updates_per_stage
        )
        if self.stage == 1 and window_ready and (
            opposing_mean >= self.config.opposition_threshold
        ):
            self.stage = 2
            promoted = True
        elif self.stage == 2 and window_ready and (
            positive_lift_mean >= self.config.positive_lift_threshold
        ):
            self.stage = 3
            promoted = True
        if promoted:
            self.stage_updates = 0
            self._opposition.clear()
            self._positive_lift.clear()
            self._stable_airborne.clear()
        return {
            "curriculum/stage": float(self.stage),
            "curriculum/promoted": float(promoted),
            "curriculum/opposing_loaded_window_mean": opposing_mean,
            "curriculum/positive_lift_window_mean": positive_lift_mean,
            "curriculum/stable_airborne_window_mean": stable_airborne_mean,
            "curriculum/final_threshold_met": float(
                stable_airborne_mean >= self.config.stable_airborne_threshold
            ),
        }


def curriculum_parameters(config: CurriculumConfig) -> dict[str, object]:
    return {
        "enabled": True,
        "stages": dict(CURRICULUM_STAGE_NAMES),
        "thresholds": {
            "opposing_loaded_fraction": config.opposition_threshold,
            "positive_lift_fraction": config.positive_lift_threshold,
            "stable_airborne_fraction": config.stable_airborne_threshold,
        },
        "averaging_updates": config.averaging_updates,
        "minimum_updates_per_stage": config.minimum_updates_per_stage,
        "reset_sampling": "three deterministic key-frame candidates per reference and stage",
    }
