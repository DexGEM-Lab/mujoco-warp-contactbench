"""Low-overhead telemetry for autonomous PPO training and evaluation.

The physical environment owns all measurements.  This module only reduces the
already-produced transition ``info`` dictionaries at update boundaries and
reads native skrl/RlGamesPPO tracking data.  Missing native statistics remain
absent rather than being replaced with guessed values.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter
from typing import Any, Mapping, Sequence

from sim.manorl.assets import object_collision_vertices
from sim.manorl.contracts import FLOOR_TOP_Z

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


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv, qw = q[:3], q[3]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


@lru_cache(maxsize=4)
def _cached_collision_vertices(object_type: str) -> np.ndarray:
    return np.asarray(object_collision_vertices(object_type), dtype=float).copy()


def genuine_airborne_contact(info: Mapping[str, Any], *, clearance_m: float = 0.01, force_threshold_N: float = 0.02, object_type: str = "cube2") -> bool:
    """Require genuine force and bottom-vertex clearance above the table."""
    force = np.asarray(info.get("hand_object_force", []), dtype=float)
    if not force.size or np.linalg.norm(force.reshape(-1, 3), axis=1).max() <= force_threshold_N:
        return False
    position = np.asarray(info.get("object_position", []), dtype=float)
    quat = np.asarray(info.get("object_quaternion_xyzw", []), dtype=float)
    if position.shape != (3,) or quat.shape != (4,):
        return False
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    vertices = _cached_collision_vertices(object_type)
    bottom = float(np.min(np.asarray([_quat_rotate(quat, vertex) for vertex in vertices])[:, 2] + position[2]))
    return bottom > FLOOR_TOP_Z + clearance_m


def _orientation_angles(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    actual = actual / max(np.linalg.norm(actual), 1e-12)
    target = target / max(np.linalg.norm(target), 1e-12)
    angle_rad = float(2.0 * np.arccos(np.clip(abs(float(np.dot(actual, target))), 0.0, 1.0)))
    return angle_rad, float(np.degrees(angle_rad))


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
    ongoing_episode_return: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    ongoing_episode_length: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    completed_episode_returns: list[float] = field(default_factory=list)
    completed_episode_lengths: list[int] = field(default_factory=list)

    def add(self, info: Mapping[str, Any], reward: float, terms: Mapping[str, Any] | None = None, *, terminated: bool = False, truncated: bool = False, env_id: int = 0) -> None:
        self.rewards.append(float(reward))
        self.infos.append(info)
        self.ongoing_episode_return[env_id] += float(reward)
        self.ongoing_episode_length[env_id] += 1
        if terminated or truncated:
            self.completed_episode_returns.append(self.ongoing_episode_return.pop(env_id, 0.0))
            self.completed_episode_lengths.append(self.ongoing_episode_length.pop(env_id, 0))
        for name, value in (terms or info.get("terms", {})).items():
            scalar = _scalar(value)
            if scalar is not None:
                self.terms[str(name)].append(scalar)

    def add_batch(self, infos: Sequence[Mapping[str, Any]], rewards: Sequence[float], terms: Sequence[Mapping[str, Any]] | None = None, terminated: Sequence[bool] | None = None, truncated: Sequence[bool] | None = None) -> None:
        if len(infos) != len(rewards):
            raise ValueError("telemetry infos and rewards must have equal lengths")
        for index, (info, reward) in enumerate(zip(infos, rewards)):
            self.add(info, reward, None if terms is None else terms[index], terminated=bool(terminated[index]) if terminated is not None else False, truncated=bool(truncated[index]) if truncated is not None else False, env_id=index)

    def reduce(self, *, update: int, transitions: int, window_transitions: int | None = None, update_elapsed_seconds: float | None = None, agent: Any = None) -> dict[str, float]:
        metrics: dict[str, float] = {
            "global_step": float(transitions),
            "update": float(update),
            "transitions": float(transitions),
            "reward_mean": _mean(self.rewards) or 0.0,
            "performance/update_time": float(update_elapsed_seconds if update_elapsed_seconds is not None else perf_counter() - self.started_at),
        }
        if metrics["performance/update_time"] > 0:
            measured_transitions = int(window_transitions if window_transitions is not None else len(self.infos))
            metrics["performance/step_fps"] = float(measured_transitions / metrics["performance/update_time"])
            metrics["performance/update_fps"] = float(1.0 / metrics["performance/update_time"])
        if self.completed_episode_lengths:
            metrics["episodes/completed_count"] = float(len(self.completed_episode_lengths))
            metrics["episodes/length_mean"] = float(np.mean(self.completed_episode_lengths))
            metrics["episodes/return_mean"] = float(np.mean(self.completed_episode_returns))
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
                angle_rad, angle_degrees = _orientation_angles(actual_q, target_q)
                orientation_errors.append(angle_rad)
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
            airborne_contact.append(genuine_airborne_contact(info))
            phase = info.get("failure_phase")
            if phase is not None: phases[str(phase)] += 1
        if path_errors: metrics["physics/path_rmse"] = float(np.sqrt(np.mean(np.square(path_errors))) )
        if orientation_errors:
            metrics["physics/orientation_error_rad"] = float(np.mean(orientation_errors))
            metrics["physics/orientation_error_degrees"] = float(np.degrees(np.mean(orientation_errors)))
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
        """Clear only the current update window; episode state crosses PPO cuts."""
        self.rewards.clear(); self.terms.clear(); self.infos.clear(); self.started_at = perf_counter()
        self.completed_episode_returns.clear(); self.completed_episode_lengths.clear()


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
