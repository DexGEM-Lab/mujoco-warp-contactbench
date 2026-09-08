"""Low-overhead telemetry for autonomous PPO training and evaluation.

The physical environment owns all measurements.  This module only reduces the
already-produced transition ``info`` dictionaries at update boundaries and
reads native skrl/RlGamesPPO tracking data.  Missing native statistics remain
absent rather than being replaced with guessed values.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np


# Names intentionally mirror the mature ManoRL logger where possible.
_PPO_TRACKING_NAMES: dict[str, tuple[str, ...]] = {
    "losses/a_loss": ("Loss / Policy loss",),
    "losses/c_loss": ("Loss / Value loss",),
    "losses/entropy": ("Loss / Entropy loss",),
    "info/exact_kl_mean": ("Learning / Exact KL mean",),
    "info/exact_kl_min": ("Learning / Exact KL min",),
    "info/exact_kl_max": ("Learning / Exact KL max",),
    "info/approximate_kl_mean": ("Learning / Approximate KL mean",),
    "info/last_lr": ("Learning / Learning rate",),
    "info/lr_start": ("Learning / Learning rate start",),
    "info/lr_min": ("Learning / Learning rate min",),
    "info/lr_max": ("Learning / Learning rate max",),
    "info/lr_scheduler_increases": ("Learning / Scheduler increases",),
    "info/lr_scheduler_decreases": ("Learning / Scheduler decreases",),
    "info/completed_minibatches": ("Learning / Completed minibatches",),
    "info/policy_std": ("Policy / Standard deviation",),
}


def latest_ppo_metrics(agent: Any) -> dict[str, float]:
    """Return only native PPO scalars available after the latest update."""

    tracking = getattr(agent, "tracking_data", None)
    if not tracking:
        return {}
    result: dict[str, float] = {}
    for public_name, source_names in _PPO_TRACKING_NAMES.items():
        values = next((tracking.get(name) for name in source_names if tracking.get(name)), None)
        if values:
            value = values[-1]
            if np.isfinite(float(value)):
                result[public_name] = float(value)
    # Clip fraction is deliberately omitted: RlGamesPPO does not expose it.
    return result


def _scalar(value: Any) -> float | None:
    try:
        array = np.asarray(value)
        if array.size != 1:
            return None
        result = float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _longest_true(values: Sequence[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


@dataclass
class TelemetryAccumulator:
    """Reduce transition measurements once per PPO update.

    ``add`` accepts the host-side info already emitted by the environment; it
    does not inspect or copy device tensors.  A future batched adapter can call
    ``add_batch`` with one info mapping per environment and retain the same
    update-boundary schema.
    """

    started_at: float = field(default_factory=perf_counter)
    rewards: list[float] = field(default_factory=list)
    terms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    infos: list[Mapping[str, Any]] = field(default_factory=list)

    def add(self, info: Mapping[str, Any], reward: float, terms: Mapping[str, Any] | None = None) -> None:
        self.rewards.append(float(reward))
        self.infos.append(info)
        for name, value in (terms or info.get("terms", {})).items():
            scalar = _scalar(value)
            if scalar is not None:
                self.terms[str(name)].append(scalar)

    def add_batch(self, infos: Sequence[Mapping[str, Any]], rewards: Sequence[float], terms: Sequence[Mapping[str, Any]] | None = None) -> None:
        if len(infos) != len(rewards):
            raise ValueError("telemetry infos and rewards must have equal lengths")
        for index, (info, reward) in enumerate(zip(infos, rewards)):
            self.add(info, reward, None if terms is None else terms[index])

    def reduce(self, *, update: int, transitions: int, update_elapsed_seconds: float | None = None, agent: Any = None) -> dict[str, float]:
        metrics: dict[str, float] = {
            "global_step": float(transitions),
            "update": float(update),
            "transitions": float(transitions),
            "reward_mean": _mean(self.rewards) or 0.0,
            "performance/update_time": float(update_elapsed_seconds if update_elapsed_seconds is not None else perf_counter() - self.started_at),
        }
        if metrics["performance/update_time"] > 0:
            metrics["performance/step_fps"] = float(transitions / metrics["performance/update_time"])
            metrics["performance/update_fps"] = float(1.0 / metrics["performance/update_time"])
        metrics["episodes/length_mean"] = float(len(self.infos))
        if self.infos and any(bool(info.get("terminated", False) or info.get("truncated", False)) for info in self.infos):
            metrics["episodes/return_mean"] = metrics["reward_mean"] * metrics["episodes/length_mean"]
        for name, values in self.terms.items():
            value = _mean(values)
            if value is not None:
                metrics[f"manorl/{name}_mean"] = value
                metrics[name] = value
        # These fields are emitted by Cube2AutonomousMJX and are all computed
        # from actual states, never reference substitutions.
        path_errors: list[float] = []
        orientation_errors: list[float] = []
        lifts: list[float] = []
        target_lifts: list[float] = []
        slips: list[float] = []
        contact: list[bool] = []
        airborne_contact: list[bool] = []
        phases: dict[str, int] = defaultdict(int)
        for info in self.infos:
            actual = np.asarray(info.get("object_position", []), dtype=float)
            target = np.asarray(info.get("target_object_position", []), dtype=float)
            if actual.shape == (3,) and target.shape == (3,):
                path_errors.append(float(np.linalg.norm(actual - target)))
            actual_q = np.asarray(info.get("object_quaternion_xyzw", []), dtype=float)
            target_q = np.asarray(info.get("target_object_quaternion_xyzw", []), dtype=float)
            if actual_q.shape == (4,) and target_q.shape == (4,):
                actual_q = actual_q / max(np.linalg.norm(actual_q), 1e-12)
                target_q = target_q / max(np.linalg.norm(target_q), 1e-12)
                orientation_errors.append(float(1.0 - min(1.0, abs(float(np.dot(actual_q, target_q))))))
            for key, values in (("actual_peak_lift", lifts), ("target_peak_lift", target_lifts), ("lift_error", [])):
                value = _scalar(info.get(key))
                if value is not None and key != "lift_error": values.append(value)
            motion = np.asarray(info.get("relative_contact_motion", []), dtype=float)
            if motion.size and motion.size % 3 == 0:
                slips.append(float(np.linalg.norm(motion.reshape(-1, 3), axis=1).mean()))
            force = np.asarray(info.get("hand_object_force", []), dtype=float)
            force_contact = bool(force.size and np.linalg.norm(force.reshape(-1, 3), axis=1).max() > 0.02)
            contact.append(force_contact)
            actual_z = _scalar(actual[2]) if actual.shape == (3,) else None
            airborne_contact.append(bool(force_contact and actual_z is not None and actual_z > 0.03))
            phase = info.get("failure_phase")
            if phase is not None: phases[str(phase)] += 1
        if path_errors: metrics["physics/path_rmse"] = float(np.sqrt(np.mean(np.square(path_errors))) )
        if orientation_errors: metrics["physics/orientation_error"] = float(np.mean(orientation_errors))
        if lifts: metrics["physics/peak_lift"] = float(max(lifts))
        if target_lifts: metrics["physics/target_peak_lift"] = float(max(target_lifts))
        if slips: metrics["physics/slip_proxy"] = float(np.mean(slips))
        if contact:
            metrics["physics/contact_frames"] = float(sum(contact))
            metrics["physics/contact_frame_fraction"] = float(np.mean(contact))
            metrics["physics/sustained_contact_frames"] = float(_longest_true(contact))
        if airborne_contact:
            metrics["physics/airborne_contact_frames"] = float(sum(airborne_contact))
            metrics["physics/sustained_airborne_contact_frames"] = float(_longest_true(airborne_contact))
        for phase, count in phases.items(): metrics[f"episodes/phase/{phase}"] = float(count)
        metrics.update(latest_ppo_metrics(agent) if agent is not None else {})
        return metrics

    def clear(self) -> None:
        self.rewards.clear(); self.terms.clear(); self.infos.clear(); self.started_at = perf_counter()


def configure_wandb_axis(run: Any) -> None:
    """Use physical environment transitions as the single W&B step axis."""

    define_metric = getattr(run, "define_metric", None)
    if callable(define_metric):
        define_metric("transitions")
        define_metric("*", step_metric="transitions")


def log_update(run: Any, metrics: Mapping[str, Any]) -> None:
    """Log one reduced update and use transitions as the W&B history step."""

    if run is None:
        return
    payload = dict(metrics)
    step = int(payload["transitions"])
    run.log(payload, step=step)
