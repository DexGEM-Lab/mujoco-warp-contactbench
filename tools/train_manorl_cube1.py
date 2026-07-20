#!/usr/bin/env python3
"""Bounded CUDA PPO training for a selected object/gesture Lance trajectory batch."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Protocol
from uuid import uuid4

import numpy as np
import torch

from sim.manorl.abi import (
    ENVIRONMENT_CONTRACT_ID,
    ResidualActionConfig,
    TARGET_MAX_DEVIATION_DISTANCE,
    TERMINATION_REASON_FAILURE,
    TERMINATION_REASON_NONE,
    TERMINATION_REASON_SUCCESS,
)
from sim.manorl.checkpoint import load_skrl_checkpoint, save_skrl_checkpoint
from sim.manorl.cli import parse_cli_bool
from sim.manorl.contracts import DATASET_PATH
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv
from sim.manorl.observations import CONTACT_FORCE_THRESHOLD
from sim.manorl.rerun_recorder import ManoRerunRecorder
from sim.manorl.rewards import (
    PPO_REWARD_CONTRACT_ID,
    PPO_REWARD_SCALE,
    REWARD_CONTRACT_ID,
    REWARD_HAND_OBJECT_THRESHOLD_N,
)
from sim.manorl.skrl_runtime import ManoPPOConfig, ManoSkrlRuntime
from sim.manorl.trajectory import (
    TRAJECTORY_IDENTITY_SCHEMA,
    TrajectorySelection,
    load_assigned_trajectory_batch,
)

WARP_BROADPHASE_CONTACTS_PER_WORLD = 31
UNIFIED_WARP_CONTACTS_PER_WORLD = 64
WARP_CONTACT_CAPACITY_MARGIN = 64
DEFAULT_WANDB_TAGS = ("manorl", "mujoco", "skrl")
REWARD_UPDATE_COMPONENTS = (
    "total",
    "distance_x",
    "distance_y",
    "distance_z",
    "rotation",
    "action_penalty",
    "contact",
    "object_stability",
    "survival",
    "deviation_penalty",
)
# Isaac Gym's completed-episode keys are the names used by its episode
# statistics tracker. These aliases are backed by the same target diagnostics.
EPISODE_REWARD_ALIASES = (
    ("episode_reward", "total"),
    ("distance_reward", None),
    ("distance_reward_x", "distance_x"),
    ("distance_reward_y", "distance_y"),
    ("distance_reward_z", "distance_z"),
    ("rotation_reward", "rotation"),
    ("action_penalty", "action_penalty"),
    ("contact_reward", "contact"),
    ("object_stability_reward", "object_stability"),
    ("survival_reward", "survival"),
)
GYM_INSTANT_REWARD_ALIASES = (
    ("distance_reward_x_instant/step", "distance_x"),
    ("distance_reward_y_instant/step", "distance_y"),
    ("distance_reward_z_instant/step", "distance_z"),
    ("rotation_reward_instant/step", "rotation"),
    ("action_penalty_instant/step", "action_penalty"),
    ("contact_reward_instant/step", "contact"),
    ("object_stability_reward_instant/step", "object_stability"),
    ("survival_reward_instant/step", "survival"),
)
GYM_STEP_ALIASES = (
    "rewards/frame",
    "rewards/iter",
    "rewards/step",
    "rewards/time",
    "episode_lengths/frame",
    "episode_lengths/iter",
    "episode_lengths/step",
)
# The Gym reference also publishes mu/* policy summaries. skrl exposes only
# sampled actions on this path, so emitting mu/* here would be fabricated.
GYM_INSTANT_GROUP_COMPONENTS = {
    "distance_x": "distance_reward_x_instant",
    "distance_y": "distance_reward_y_instant",
    "distance_z": "distance_reward_z_instant",
    "rotation": "rotation_reward_instant",
    "action_penalty": "action_penalty_instant",
    "position_penalty": "position_penalty_instant",
    "joint_penalty": "joint_penalty_instant",
    "contact": "contact_reward_instant",
    "object_stability": "object_stability_reward_instant",
    "object_speed": "object_speed_instant",
}
GYM_EPISODE_GROUP_COMPONENTS = {
    "distance_x": "distance_reward_x",
    "distance_y": "distance_reward_y",
    "distance_z": "distance_reward_z",
    "rotation": "rotation_reward",
    "survival": "survival_reward",
    "action_penalty": "action_penalty",
    "contact": "contact_reward",
    "object_stability": "object_stability_reward",
}
GROUPED_REWARD_COMPONENTS = tuple(
    dict.fromkeys(("total", *GYM_INSTANT_GROUP_COMPONENTS, *GYM_EPISODE_GROUP_COMPONENTS))
)
SKRL_TRACKING_METRICS = (
    (("Loss / Policy loss",), "losses/a_loss"),
    (("Loss / Value loss",), "losses/c_loss"),
    (("Loss / Entropy loss",), "losses/entropy"),
    (("Loss / Bounds loss",), "losses/bounds_loss"),
    (("Learning / Learning rate",), "info/last_lr"),
    (("Learning / Learning rate multiplier", "Learning / LR multiplier"), "info/lr_mul"),
    (("Learning / E-clip", "Policy / E-clip", "Info / E-clip"), "info/e_clip"),
    (("Info / Clip fraction", "Stats / Clip fraction"), "info/clip_frac"),
    (("Info / KL", "Stats / KL divergence"), "info/kl"),
    (("Learning / Learning rate start",), "info/lr_start"),
    (("Learning / Learning rate min",), "info/lr_min"),
    (("Learning / Learning rate max",), "info/lr_max"),
    (("Learning / Exact KL mean",), "info/exact_kl_mean"),
    (("Learning / Exact KL min",), "info/exact_kl_min"),
    (("Learning / Exact KL max",), "info/exact_kl_max"),
    (("Learning / Approximate KL mean",), "info/approximate_kl_mean"),
    (("Learning / Scheduler increases",), "info/lr_scheduler_increases"),
    (("Learning / Scheduler decreases",), "info/lr_scheduler_decreases"),
    (("Learning / Completed minibatches",), "info/completed_minibatches"),
    (("Policy / Standard deviation",), "info/policy_std"),
    (("Stats / Algorithm update time (ms)",), "performance/algorithm_update_time_ms"),
)


@dataclass(frozen=True)
class WandbOptions:
    enabled: bool = True
    project: str = "one_policy"
    group: str = "s02"
    entity: str = ""
    name: str | None = None
    tags: tuple[str, ...] = DEFAULT_WANDB_TAGS


class TrainingObserver(Protocol):
    """Render post-step state without owning PPO stepping or policy actions."""

    close_requested: bool

    def observe(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class TrainingBudget:
    # These are the validated single-task convergence defaults.  Smaller
    # budgets remain available as explicit diagnostic overrides.
    num_envs: int = 2048
    updates: int = 8000
    wall_clock_seconds: float | None = None
    seed: int = 42
    rerun_output: str | None = None
    rerun_grpc_url: str | None = None
    rerun_high_return_dir: str | None = None
    rerun_high_return_threshold: float | None = None
    rerun_high_return_following: int = 5
    rerun_env_id: int = 0
    rerun_stride: int = 1
    object_type: str = "cube1"
    gesture: str = "01"
    trajectory_selector: str | None = None
    dataset_path: str = DATASET_PATH
    residual_enabled: bool = True
    use_film: bool = True
    terminal: bool = True
    wandb: WandbOptions = WandbOptions()
    checkpoint_interval_updates: int | None = 200
    minibatch_size: int | None = None
    # Evaluation is intentionally one fixed trajectory by default.  A larger
    # count is an explicit diagnostic mode and must not be mistaken for the
    # Gym single-environment reference result.
    evaluation_num_envs: int | None = 1
    headless: bool = True
    viewer_envs: int = 1
    viewer_stride: int = 1
    console_format: Literal["human", "json"] = "human"
    device_resident_controls: bool = False
    profile_phases: bool = False
    capture_transition_diagnostics: bool | None = None
    unified_object_batch: bool = False

    @property
    def transitions(self) -> int:
        return self.num_envs * ManoPPOConfig().rollouts * self.updates

    @property
    def resolved_minibatch_size(self) -> int:
        if self.minibatch_size is not None:
            return self.minibatch_size
        rollout_batch = self.num_envs * ManoPPOConfig().rollouts
        return math.gcd(ManoPPOConfig().minibatch_size, rollout_batch)

    @property
    def resolved_capture_transition_diagnostics(self) -> bool:
        if self.capture_transition_diagnostics is None:
            return not self.device_resident_controls
        return self.capture_transition_diagnostics

    @property
    def resolved_evaluation_num_envs(self) -> int:
        maximum = min(self.num_envs, 128)
        num_envs = self.evaluation_num_envs if self.evaluation_num_envs is not None else maximum
        if not 1 <= num_envs <= maximum:
            raise ValueError(f"evaluation_num_envs must be within 1..{maximum}")
        return num_envs


@dataclass(frozen=True)
class EvaluationGroupResult:
    label: str
    kind: Literal["object", "object_action"]
    num_envs: int
    return_mean: float
    reward_mean: float
    action_abs_mean: float
    final_object_target_distance: float
    max_object_target_distance: float
    contact_reward_mean: float
    success_count: int
    failure_count: int


@dataclass(frozen=True)
class EvaluationResult:
    mode: str
    calls: int
    return_mean: float
    reward_mean: float
    action_abs_mean: float
    final_object_target_distance: float
    max_object_target_distance: float
    contact_reward_mean: float
    reset_seen: bool
    timeout_seen: bool
    completed_horizon: bool
    rewards_by_call: list[float]
    object_target_distance_by_call: list[float]
    success_seen: bool = False
    failure_seen: bool = False
    termination_reason_code: int = 0
    groups: tuple[EvaluationGroupResult, ...] = ()


@dataclass(frozen=True)
class _MetricGroupAxis:
    kind: Literal["object", "object_action"]
    labels: tuple[str, ...]
    env_group_ids: np.ndarray


@dataclass(frozen=True)
class _EnvironmentTelemetryLayout:
    object_types: tuple[str, ...]
    action_ids: tuple[str, ...]
    identities: tuple[str, ...]
    axes: tuple[_MetricGroupAxis, ...]


def _environment_telemetry_layout(
    environment: Any, *, num_envs: int
) -> _EnvironmentTelemetryLayout | None:
    """Resolve stable Gym-style object and object/action groups for vector rows."""

    object_values = getattr(environment, "object_types", None)
    action_values = getattr(environment, "action_ids", None)
    if object_values is None or action_values is None:
        return None
    object_types = tuple(str(value) for value in object_values)
    action_array = np.asarray(action_values)
    if len(object_types) != num_envs or action_array.shape != (num_envs,):
        raise RuntimeError("environment object/action telemetry has an invalid vector shape")
    try:
        action_ids = tuple(f"{int(value):02d}" for value in action_array)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("environment action telemetry must contain integer action IDs") from exc
    pair_values = tuple(zip(object_types, action_ids, strict=True))
    object_names = tuple(sorted(set(object_types)))
    pair_names = tuple(sorted(set(pair_values)))
    object_indices = {name: index for index, name in enumerate(object_names)}
    pair_indices = {name: index for index, name in enumerate(pair_names)}
    trajectories = getattr(environment, "trajectories", ())
    if len(trajectories) == num_envs:
        identities = tuple(str(item.identity.identity) for item in trajectories)
    else:
        identities = tuple(
            f"{object_type}_{action_id}_env{env_id:06d}"
            for env_id, (object_type, action_id) in enumerate(pair_values)
        )
    return _EnvironmentTelemetryLayout(
        object_types=object_types,
        action_ids=action_ids,
        identities=identities,
        axes=(
            _MetricGroupAxis(
                kind="object",
                labels=tuple(f"object_{name}" for name in object_names),
                env_group_ids=np.asarray(
                    [object_indices[name] for name in object_types], dtype=np.int64
                ),
            ),
            _MetricGroupAxis(
                kind="object_action",
                labels=tuple(f"{object_type}_{action_id}" for object_type, action_id in pair_names),
                env_group_ids=np.asarray(
                    [pair_indices[name] for name in pair_values], dtype=np.int64
                ),
            ),
        ),
    )


class _GroupedUpdateTelemetry:
    """Accumulate one PPO update at Gym's object and object/action levels."""

    _instant_sources = ("total", *GYM_INSTANT_GROUP_COMPONENTS)

    def __init__(self, layout: _EnvironmentTelemetryLayout) -> None:
        self.layout = layout
        self.component_sums = {
            axis.kind: np.zeros((len(axis.labels), len(self._instant_sources)), dtype=np.float64)
            for axis in layout.axes
        }
        self.sample_counts = {
            axis.kind: np.zeros(len(axis.labels), dtype=np.int64) for axis in layout.axes
        }
        self.completed_counts = {
            axis.kind: np.zeros(len(axis.labels), dtype=np.int64) for axis in layout.axes
        }
        self.success_counts = {
            axis.kind: np.zeros(len(axis.labels), dtype=np.int64) for axis in layout.axes
        }
        self.failure_counts = {
            axis.kind: np.zeros(len(axis.labels), dtype=np.int64) for axis in layout.axes
        }
        self.episode_returns = {
            axis.kind: [[] for _ in axis.labels] for axis in layout.axes
        }
        self.episode_components = {
            axis.kind: {
                source: [[] for _ in axis.labels] for source in GYM_EPISODE_GROUP_COMPONENTS
            }
            for axis in layout.axes
        }

    @staticmethod
    def _bincount(
        axis: _MetricGroupAxis, values: np.ndarray | None = None
    ) -> np.ndarray:
        return np.bincount(
            axis.env_group_ids,
            weights=values,
            minlength=len(axis.labels),
        )

    def add_step(
        self,
        *,
        components: dict[str, np.ndarray],
        completed: np.ndarray,
        success: np.ndarray,
        failure: np.ndarray,
        episode_returns: np.ndarray,
        episode_component_totals: dict[str, np.ndarray],
    ) -> None:
        num_envs = len(self.layout.object_types)
        required = set(self._instant_sources) | set(GYM_EPISODE_GROUP_COMPONENTS)
        if set(components) != required:
            missing = sorted(required - set(components))
            extra = sorted(set(components) - required)
            raise RuntimeError(f"grouped reward telemetry mismatch; missing={missing}, extra={extra}")
        if any(np.asarray(values).shape != (num_envs,) for values in components.values()):
            raise RuntimeError("grouped reward telemetry has an invalid vector shape")
        for values in (completed, success, failure, episode_returns):
            if np.asarray(values).shape != (num_envs,):
                raise RuntimeError("grouped episode telemetry has an invalid vector shape")
        component_matrix = np.stack(
            [np.asarray(components[source], dtype=np.float64) for source in self._instant_sources]
        )
        for axis in self.layout.axes:
            self.sample_counts[axis.kind] += self._bincount(axis).astype(np.int64)
            for component_index, values in enumerate(component_matrix):
                self.component_sums[axis.kind][:, component_index] += self._bincount(axis, values)
            self.completed_counts[axis.kind] += self._bincount(
                axis, completed.astype(np.float64)
            ).astype(np.int64)
            self.success_counts[axis.kind] += self._bincount(
                axis, success.astype(np.float64)
            ).astype(np.int64)
            self.failure_counts[axis.kind] += self._bincount(
                axis, failure.astype(np.float64)
            ).astype(np.int64)
            if not completed.any():
                continue
            completed_group_ids = axis.env_group_ids[completed]
            completed_returns = episode_returns[completed]
            for group_id in np.unique(completed_group_ids):
                selected = completed_group_ids == group_id
                self.episode_returns[axis.kind][int(group_id)].extend(
                    completed_returns[selected].tolist()
                )
                for source in GYM_EPISODE_GROUP_COMPONENTS:
                    self.episode_components[axis.kind][source][int(group_id)].extend(
                        np.asarray(episode_component_totals[source], dtype=np.float64)[completed][
                            selected
                        ].tolist()
                    )

    def metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        source_indices = {source: index for index, source in enumerate(self._instant_sources)}
        for axis in self.layout.axes:
            for group_id, label in enumerate(axis.labels):
                samples = int(self.sample_counts[axis.kind][group_id])
                if samples < 1:
                    continue
                sums = self.component_sums[axis.kind][group_id]
                metrics[f"reward_mean/{label}"] = float(sums[source_indices["total"]] / samples)
                for source, metric_name in GYM_INSTANT_GROUP_COMPONENTS.items():
                    metrics[f"{metric_name}/{label}"] = float(
                        sums[source_indices[source]] / samples
                    )
                completed = int(self.completed_counts[axis.kind][group_id])
                successes = int(self.success_counts[axis.kind][group_id])
                failures = int(self.failure_counts[axis.kind][group_id])
                metrics[f"attempts/{label}"] = float(completed)
                metrics[f"successes/{label}"] = float(successes)
                metrics[f"failures/{label}"] = float(failures)
                metrics[f"success_rate/{label}"] = (
                    100.0 * successes / completed if completed else 0.0
                )
                returns = np.asarray(
                    self.episode_returns[axis.kind][group_id], dtype=np.float64
                )
                if returns.size == 0:
                    continue
                episode_mean = float(returns.mean())
                metrics[f"episode_reward/{label}"] = episode_mean
                metrics[f"episode_cumulative/episode_reward/{label}"] = episode_mean
                metrics[f"episode_cumulative_min/episode_reward/{label}_min"] = float(
                    returns.min()
                )
                metrics[f"episode_cumulative_max/episode_reward/{label}_max"] = float(
                    returns.max()
                )
                distance_total = np.zeros(returns.size, dtype=np.float64)
                for source, metric_name in GYM_EPISODE_GROUP_COMPONENTS.items():
                    values = np.asarray(
                        self.episode_components[axis.kind][source][group_id],
                        dtype=np.float64,
                    )
                    if values.shape != returns.shape:
                        raise RuntimeError("grouped episode reward telemetry lost alignment")
                    value_mean = float(values.mean())
                    metrics[f"{metric_name}/{label}"] = value_mean
                    metrics[f"episode_cumulative/{metric_name}/{label}"] = value_mean
                    if source in {"distance_x", "distance_y", "distance_z"}:
                        distance_total += values
                metrics[f"distance_reward/{label}"] = float(distance_total.mean())
                metrics[f"episode_cumulative/distance_reward/{label}"] = float(
                    distance_total.mean()
                )
        return metrics


def _wandb_run_name(output: Path, budget: TrainingBudget) -> str:
    if budget.wandb.name:
        return budget.wandb.name
    selector = budget.trajectory_selector
    if selector is None:
        selector = f"{budget.object_type}-{int(budget.gesture):02d}"
    selector_label = selector.replace(":", "-").replace(",", "_")
    return f"{output.name}-{selector_label}"


def _parse_wandb_tags(values: list[str]) -> tuple[str, ...]:
    tags = tuple(tag.strip() for value in values for tag in value.split(",") if tag.strip())
    return tags or DEFAULT_WANDB_TAGS


def _wandb_config(
    *,
    budget: TrainingBudget,
    ppo_config: ManoPPOConfig,
    trajectory_assignments: list[dict[str, object]],
    evaluation_ppo_config: ManoPPOConfig,
    evaluation_trajectory_assignments: list[dict[str, object]],
    trajectory_selection: dict[str, object] | None = None,
    device: dict[str, object],
) -> dict[str, object]:
    residual_action = ResidualActionConfig()
    config = {
        "training_budget": {
            **asdict(budget),
            "planned_transitions": budget.transitions,
            "resolved_capture_transition_diagnostics": budget.resolved_capture_transition_diagnostics,
        },
        "ppo_config": asdict(ppo_config),
        "wandb": {
            "primary_axis": "completed_ppo_updates",
            "secondary_metrics": ["transitions"],
            "shared_policy_run": True,
            "aggregation_levels": ["global", "object", "object_action"],
            "object_label": "object_{object}",
            "object_action_label": "{object}_{action:02d}",
            "grouped_instant_metrics": list(GYM_INSTANT_GROUP_COMPONENTS.values()),
            "grouped_episode_metrics": [
                "episode_reward",
                *GYM_EPISODE_GROUP_COMPONENTS.values(),
            ],
        },
        "evaluation": {
            "num_envs": len(evaluation_trajectory_assignments),
            "coverage": "at_least_one_environment_per_resolved_pair",
            "ppo_config": asdict(evaluation_ppo_config),
            "trajectory_assignments": evaluation_trajectory_assignments,
        },
        "reward": {
            "environment_contract": REWARD_CONTRACT_ID,
            "contact_force_threshold_N": REWARD_HAND_OBJECT_THRESHOLD_N,
            "ppo_contract": PPO_REWARD_CONTRACT_ID,
            "ppo_scale": PPO_REWARD_SCALE,
            "isaacgym_ppo_scale": 0.5,
        },
        "environment": {
            "contract": ENVIRONMENT_CONTRACT_ID,
            "observation_contact_threshold_N": CONTACT_FORCE_THRESHOLD,
            "residual_action": {
                "position_scale": list(residual_action.position_scale),
                "gamma_xy": residual_action.gamma_xy,
                "gamma_z": residual_action.gamma_z,
                "gamma_joints": residual_action.gamma_joints,
                "max_position_offset": list(residual_action.max_position_offset),
                "joint_scale": list(residual_action.joint_scale),
                "max_joint_offset": list(residual_action.max_joint_offset),
                "early_phase_steps": residual_action.early_phase_steps,
                "rotation_effective_scale": 0.00025,
            },
            "max_deviation_distance": TARGET_MAX_DEVIATION_DISTANCE if budget.terminal else 1_000_000.0,
        },
        "trajectory_assignments": trajectory_assignments,
        "trajectory_selection": trajectory_selection,
        "device": device,
    }
    try:
        return json.loads(json.dumps(config, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("W&B config must be JSON-serializable") from exc


@contextmanager
def _wandb_run(
    *, output: Path, budget: TrainingBudget, config: dict[str, object]
) -> Iterator[tuple[Any | None, Any | None]]:
    if not budget.wandb.enabled:
        yield None, None
        return
    wandb_dir = output.parent
    wandb_dir.mkdir(parents=True, exist_ok=True)
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B is enabled but the wandb package is unavailable") from exc
    try:
        run = wandb.init(
            project=budget.wandb.project,
            entity=budget.wandb.entity or None,
            group=budget.wandb.group,
            name=_wandb_run_name(output, budget),
            tags=list(budget.wandb.tags),
            config=config,
            dir=str(wandb_dir),
        )
    except Exception as exc:
        raise RuntimeError("W&B initialization failed") from exc
    if run is None:
        raise RuntimeError("W&B initialization returned no run")
    try:
        yield run, wandb
    except BaseException as body_error:
        try:
            run.finish(exit_code=1)
        except BaseException as cleanup_error:
            body_error.add_note(f"W&B cleanup after training failure also failed: {cleanup_error!r}")
        raise
    else:
        try:
            run.finish()
        except BaseException as cleanup_error:
            raise RuntimeError("W&B cleanup after successful training failed") from cleanup_error


def _log_wandb_update(run: Any, wandb: Any, update: dict[str, Any]) -> None:
    completed_updates = int(update["update"])
    transitions = int(update["environment_transitions"])
    metrics = {
        "global_step": completed_updates,
        "transitions": transitions,
        "update": completed_updates,
        "reward_mean": update["reward_mean"],
        "action_abs_mean": update["action_abs_mean"],
        "reset_count": update["reset_count"],
        "success_count": update.get("success_count", 0.0),
        "failure_count": update.get("failure_count", 0.0),
        "completed_episode_count": update["completed_episode_count"],
        "elapsed_seconds": update["elapsed_seconds"],
        "update_environment_transitions_per_second": update["update_environment_transitions_per_second"],
        "cumulative_environment_transitions_per_second": update[
            "cumulative_environment_transitions_per_second"
        ],
    }
    metrics["manorl/reward_mean"] = update["manorl/reward_mean"]
    metrics.update({
        f"manorl/{name}_mean": update[f"manorl/{name}_mean"]
        for name in REWARD_UPDATE_COMPONENTS
    })
    # Preserve the established raw W&B component panels from rollout means.
    metrics.update({
        name: update[f"manorl/{name}_mean"]
        for name in REWARD_UPDATE_COMPONENTS
    })
    metrics.update({
        name: update[name]
        for name in (
            "performance/total_fps",
            "performance/step_fps",
            "performance/update_time",
            "performance/play_time",
            "performance/algorithm_update_time_ms",
            "info/epochs",
            "losses/a_loss",
            "losses/c_loss",
            "losses/entropy",
            "info/last_lr",
            "info/lr_start",
            "info/lr_min",
            "info/lr_max",
            "info/exact_kl_mean",
            "info/exact_kl_min",
            "info/exact_kl_max",
            "info/approximate_kl_mean",
            "info/lr_scheduler_increases",
            "info/lr_scheduler_decreases",
            "info/completed_minibatches",
            "info/policy_std",
            *GYM_STEP_ALIASES,
            *(alias for alias, _ in GYM_INSTANT_REWARD_ALIASES),
            "distance_reward_instant/step",
            *(f"episode_cumulative/{alias}" for alias, _ in EPISODE_REWARD_ALIASES),
            *(f"episode_cumulative_min/{alias}_min" for alias, _ in EPISODE_REWARD_ALIASES),
            *(f"episode_cumulative_max/{alias}_max" for alias, _ in EPISODE_REWARD_ALIASES),
        )
        if name in update
    })
    direct_episode_keys = tuple(alias for alias, _ in EPISODE_REWARD_ALIASES)
    # ``action_penalty`` already has a public rollout panel. Keep that raw
    # W&B key stable; its selected-group Gym alias remains unambiguous.
    direct_raw_collisions = {"action_penalty"}
    metrics.update({
        name: update[name]
        for name in update
        if (
            name not in direct_raw_collisions
            and (
                name in direct_episode_keys
                or any(name.startswith(f"{alias}/") for alias in direct_episode_keys)
            )
        )
        or name.startswith((
            "episode_cumulative/",
            "episode_cumulative_min/",
            "episode_cumulative_max/",
        ))
    })
    if "episode_return_mean" in update:
        # Completed-episode aliases are emitted only after a reset. Rollout
        # means use the explicit ``manorl/*_mean`` namespace above.
        episode_total_mean = update.get("episode_total_mean")
        if episode_total_mean is None:
            episode_total_mean = update["episode_return_mean"]
        metrics["episode_return_mean"] = update["episode_return_mean"]
        metrics["episode_reward"] = episode_total_mean
        metrics["episode_cumulative/total"] = episode_total_mean
        metrics["episode_cumulative/episode_reward"] = episode_total_mean
        metrics["episode_cumulative/total_mean"] = episode_total_mean
        if "episode_total_min" in update:
            metrics["episode_cumulative/total_min"] = update["episode_total_min"]
        if "episode_total_max" in update:
            metrics["episode_cumulative/total_max"] = update["episode_total_max"]
        metrics["episode_return_distribution"] = wandb.Histogram(update["episode_return_values"])
    grouped_metrics = update.get("grouped_metrics", {})
    if not isinstance(grouped_metrics, dict) or any(
        not isinstance(name, str) or not isinstance(value, (int, float))
        for name, value in grouped_metrics.items()
    ):
        raise RuntimeError("grouped W&B telemetry must be a flat numeric mapping")
    metrics.update(grouped_metrics)
    run.log(metrics, step=completed_updates)


def _evaluation_metrics(result: EvaluationResult) -> dict[str, object]:
    metrics = {
        f"evaluation/{result.mode}/{key}": value
        for key, value in asdict(result).items()
        if key not in {"rewards_by_call", "object_target_distance_by_call", "mode", "groups"}
    }
    for group in result.groups:
        for key, value in asdict(group).items():
            if key in {"label", "kind"}:
                continue
            metrics[f"evaluation/{result.mode}/{key}/{group.label}"] = value
    return metrics


def _log_wandb_evaluations(
    run: Any, results: list[EvaluationResult], *, update: int, transitions: int
) -> None:
    metrics: dict[str, object] = {
        "global_step": update,
        "update": update,
        "transitions": transitions,
    }
    for result in results:
        result_metrics = _evaluation_metrics(result)
        metrics.update(result_metrics)
        if result.mode in {"untrained", "trained"}:
            metrics.update({
                key.replace(f"evaluation/{result.mode}/", "evaluation/policy/"): value
                for key, value in result_metrics.items()
            })
    run.log(metrics, step=update)
    run.summary.update(metrics)


def _log_wandb_artifacts(run: Any, wandb: Any, *, output: Path, paths: list[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot log missing W&B artifacts: {', '.join(str(path) for path in missing)}")
    artifact = wandb.Artifact(name=f"{output.name}-artifacts", type="manorl-training")
    for path in paths:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)


def _episode_records_path(output: Path) -> Path:
    return output.with_suffix(".episodes.jsonl")


def _partial_episode_records_path(output: Path) -> Path:
    return output.with_suffix(".episodes.jsonl.partial")


@contextmanager
def _episode_records_file(episodes_path: Path) -> Iterator[Any]:
    partial_path = episodes_path.with_suffix(f"{episodes_path.suffix}.partial")
    with partial_path.open("x", encoding="utf-8") as episode_file:
        yield episode_file
    os.replace(partial_path, episodes_path)


def _transitions_per_second(transitions: int, elapsed_seconds: float) -> float:
    return float(transitions / elapsed_seconds) if elapsed_seconds > 0.0 else 0.0


def _latest_skrl_tracking_metrics(agent: Any) -> dict[str, float]:
    """Read the latest native skrl scalars without inventing missing values."""

    tracking_data = getattr(agent, "tracking_data", None)
    if tracking_data is None:
        return {}
    metrics: dict[str, float] = {}
    for source_names, metric_name in SKRL_TRACKING_METRICS:
        values = next((tracking_data.get(name) for name in source_names if tracking_data.get(name)), None)
        if values:
            metrics[metric_name] = float(values[-1])
    return metrics


def _public_update_metrics(update: dict[str, Any]) -> dict[str, Any]:
    """Exclude in-memory W&B histogram samples from persisted update metrics."""

    private = {"episode_return_values", "grouped_metrics"}
    return {name: value for name, value in update.items() if name not in private}


def _completed_episode_record(
    *,
    update: int,
    update_step: int,
    vector_step: int,
    num_envs: int,
    completed: np.ndarray,
    episode_returns: np.ndarray,
    termination_reason_codes: np.ndarray | None = None,
    telemetry_layout: _EnvironmentTelemetryLayout | None = None,
    episode_component_totals: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": "manorl.completed_episode_returns.v1",
        "update": update,
        "update_step": update_step,
        "vector_step": vector_step,
        "environment_transitions": vector_step * num_envs,
        "env_ids": np.flatnonzero(completed).tolist(),
        "returns": np.asarray(episode_returns, dtype=np.float64)[completed].tolist(),
    }
    if termination_reason_codes is not None:
        reason_codes = np.asarray(termination_reason_codes, dtype=np.int32)
        if reason_codes.shape != completed.shape:
            raise ValueError("termination reason codes must match completed mask")
        record["termination_reason_codes"] = reason_codes[completed].tolist()
        record["success_env_ids"] = np.flatnonzero(
            completed & (reason_codes == TERMINATION_REASON_SUCCESS)
        ).tolist()
        record["failure_env_ids"] = np.flatnonzero(
            completed & (reason_codes == TERMINATION_REASON_FAILURE)
        ).tolist()
    if telemetry_layout is not None:
        if len(telemetry_layout.object_types) != completed.size:
            raise ValueError("episode telemetry layout must match completed mask")
        record["schema"] = "manorl.completed_episode_returns.v2"
        record["object_types"] = np.asarray(telemetry_layout.object_types)[completed].tolist()
        record["action_ids"] = np.asarray(telemetry_layout.action_ids)[completed].tolist()
        record["identities"] = np.asarray(telemetry_layout.identities)[completed].tolist()
        if episode_component_totals is not None:
            record["reward_components"] = {
                metric_name: np.asarray(episode_component_totals[source], dtype=np.float64)[
                    completed
                ].tolist()
                for source, metric_name in GYM_EPISODE_GROUP_COMPONENTS.items()
            }
    return record


def _write_episode_record(
    episode_file: Any, record: dict[str, object], *, console_format: Literal["human", "json"] = "json"
) -> None:
    """Flush exact return arrays durably before rendering their console summary."""

    serialized = json.dumps(record, sort_keys=True, separators=(",", ":"))
    episode_file.write(serialized + "\n")
    episode_file.flush()
    _emit_console(console_format, "completed_episodes", record)


def _format_completed_episode_record(record: dict[str, object]) -> str:
    returns = np.asarray(record["returns"], dtype=np.float64)
    return (
        f"episodes update={record['update']} rollout_step={record['update_step']} "
        f"count={returns.size} mean={returns.mean():.4f} min={returns.min():.4f} max={returns.max():.4f}"
    )


def _format_training_update(update: dict[str, Any]) -> str:
    episode = ""
    if "episode_return_mean" in update or "episode_total_mean" in update:
        episode_total_mean = update.get("episode_total_mean")
        if episode_total_mean is None:
            episode_total_mean = update["episode_return_mean"]
        episode = (
            f" episodes={int(update['completed_episode_count'])}"
            f" episode_total_mean={episode_total_mean:.4f}"
        )
        if "episode_total_min" in update and "episode_total_max" in update:
            episode += (
                f" episode_total_min={update['episode_total_min']:.4f}"
                f" episode_total_max={update['episode_total_max']:.4f}"
            )
    return (
        f"update={int(update['update'])} transitions={int(update['environment_transitions'])} "
        f"reward={update['reward_mean']:.4f} contact={update.get('manorl/contact_mean', update.get('contact', 0.0)):.4f} "
        f"resets={int(update['reset_count'])} success={int(update.get('success_count', 0))} "
        f"failure={int(update.get('failure_count', 0))} speed={update['update_environment_transitions_per_second']:.1f}/s "
        f"elapsed={update['elapsed_seconds']:.1f}s{episode}"
    )


def _emit_console(console_format: Literal["human", "json"], event: str, payload: dict[str, Any]) -> None:
    if console_format == "json":
        if event == "completed_episodes":
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)
        else:
            print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)
        return
    if event == "completed_episodes":
        print(_format_completed_episode_record(payload), flush=True)
    elif event == "training_update":
        print(_format_training_update(payload["metrics"]), flush=True)
    elif event == "checkpoint":
        print(f"checkpoint update={int(payload['update'])} path={payload['path']}", flush=True)
    elif event == "training_complete":
        throughput = payload["throughput"]
        print(
            f"training complete transitions={int(throughput['environment_transitions'])} "
            f"elapsed={throughput['elapsed_seconds']:.1f}s "
            f"speed={throughput['final_environment_transitions_per_second']:.1f}/s",
            flush=True,
        )


def _assert_cuda_runtime() -> None:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA-capable target Torch is required for fast ManoRL training")


def _evaluation_group_results(
    layout: _EnvironmentTelemetryLayout | None,
    *,
    returns: np.ndarray,
    reward_means: np.ndarray,
    action_abs_means: np.ndarray,
    final_distances: np.ndarray,
    max_distances: np.ndarray,
    contact_means: np.ndarray,
    success: np.ndarray,
    failure: np.ndarray,
) -> tuple[EvaluationGroupResult, ...]:
    if layout is None:
        return ()
    groups: list[EvaluationGroupResult] = []
    for axis in layout.axes:
        for group_id, label in enumerate(axis.labels):
            selected = axis.env_group_ids == group_id
            groups.append(EvaluationGroupResult(
                label=label,
                kind=axis.kind,
                num_envs=int(selected.sum()),
                return_mean=float(returns[selected].mean()),
                reward_mean=float(reward_means[selected].mean()),
                action_abs_mean=float(action_abs_means[selected].mean()),
                final_object_target_distance=float(final_distances[selected].mean()),
                max_object_target_distance=float(max_distances[selected].max()),
                contact_reward_mean=float(contact_means[selected].mean()),
                success_count=int(success[selected].sum()),
                failure_count=int(failure[selected].sum()),
            ))
    return tuple(groups)


def _evaluate(runtime: ManoSkrlRuntime, mode: Literal["zero", "untrained", "trained"]) -> EvaluationResult:
    environment = runtime.gymnasium_env.environment
    # ``deterministic_actions`` only selects the Gaussian mean. PointNet/FiLM
    # still contains BatchNorm, so evaluation must also switch skrl modules to
    # eval mode or the reported policy changes with vector batch statistics.
    runtime.agent.enable_models_training_mode(False)
    observations, _ = runtime.env.reset()
    num_envs = environment.config.num_envs
    layout = _environment_telemetry_layout(environment, num_envs=num_envs)
    returns = np.zeros(num_envs, dtype=np.float64)
    action_abs_sums = np.zeros(num_envs, dtype=np.float64)
    reward_sums = np.zeros(num_envs, dtype=np.float64)
    contact_sums = np.zeros(num_envs, dtype=np.float64)
    sample_counts = np.zeros(num_envs, dtype=np.int64)
    max_distances = np.zeros(num_envs, dtype=np.float64)
    final_distances = np.zeros(num_envs, dtype=np.float64)
    reason_codes = np.full(num_envs, TERMINATION_REASON_NONE, dtype=np.int32)
    success = np.zeros(num_envs, dtype=bool)
    failure = np.zeros(num_envs, dtype=bool)
    timeout = np.zeros(num_envs, dtype=bool)
    active = np.ones(num_envs, dtype=bool)
    rewards_by_call: list[float] = []
    distance_means: list[float] = []
    trajectory_lengths = np.asarray(
        getattr(environment, "trajectory_lengths", np.full(num_envs, 792)), dtype=np.int64
    )
    if trajectory_lengths.shape != (num_envs,) or np.any(trajectory_lengths < 2):
        raise RuntimeError("evaluation trajectory lengths have an invalid vector shape")
    max_calls = int(trajectory_lengths.max()) + 1
    for call in range(max_calls):
        if mode == "zero":
            actions = torch.zeros((num_envs, 26), device=runtime.device)
        else:
            actions = runtime.deterministic_actions(observations)
        observations, rewards, terminated, truncated, info = runtime.env.step(actions)
        if not torch.isfinite(rewards).all() or not torch.isfinite(observations).all():
            raise RuntimeError(f"non-finite {mode} evaluation value at call {call}")
        reward_values = rewards.detach().cpu().numpy().reshape(-1).astype(np.float64)
        action_values = actions.detach().abs().mean(dim=1).cpu().numpy().astype(np.float64)
        if reward_values.shape != (num_envs,) or action_values.shape != (num_envs,):
            raise RuntimeError("evaluation policy tensors have an invalid vector shape")
        active_before = active.copy()
        returns[active_before] += reward_values[active_before]
        reward_sums[active_before] += reward_values[active_before]
        action_abs_sums[active_before] += action_values[active_before]
        sample_counts[active_before] += 1
        rewards_by_call.append(float(reward_values[active_before].mean()))
        physical = environment.last_physical
        diagnostics = environment.last_reward
        if physical is None or diagnostics is None:
            raise RuntimeError("environment omitted physical/reward diagnostics during evaluation")
        indices = environment.trajectory_steps
        target = environment.reference_object_pos[
            np.arange(num_envs), indices
        ]
        distances = np.linalg.norm(physical.object_position - target, axis=1)
        contacts = np.asarray(diagnostics.contact, dtype=np.float64)
        if distances.shape != (num_envs,) or contacts.shape != (num_envs,):
            raise RuntimeError("evaluation physical diagnostics have an invalid vector shape")
        distance_means.append(float(distances[active_before].mean()))
        max_distances[active_before] = np.maximum(
            max_distances[active_before], distances[active_before]
        )
        contact_sums[active_before] += contacts[active_before]
        done = (terminated | truncated).detach().cpu().numpy().reshape(-1).astype(bool)
        truncated_values = truncated.detach().cpu().numpy().reshape(-1).astype(bool)
        if done.shape != (num_envs,) or truncated_values.shape != (num_envs,):
            raise RuntimeError("evaluation termination tensors have an invalid vector shape")
        newly_done = active_before & done
        termination = environment.last_termination
        if termination is None:
            raise RuntimeError("environment omitted termination diagnostics during evaluation")
        step_reason_codes = np.asarray(termination.reason_code, dtype=np.int32).reshape(-1)
        step_success = np.asarray(termination.success, dtype=bool).reshape(-1)
        step_failure = np.asarray(termination.failure, dtype=bool).reshape(-1)
        if any(
            values.shape != (num_envs,)
            for values in (step_reason_codes, step_success, step_failure)
        ):
            raise RuntimeError("evaluation termination diagnostics have an invalid vector shape")
        if newly_done.any():
            final_distances[newly_done] = distances[newly_done]
            reason_codes[newly_done] = step_reason_codes[newly_done]
            success[newly_done] = step_success[newly_done]
            failure[newly_done] = step_failure[newly_done]
            timeout[newly_done] = truncated_values[newly_done]
            active[newly_done] = False
        if not active.any():
            reward_means = reward_sums / sample_counts
            action_abs_means = action_abs_sums / sample_counts
            contact_means = contact_sums / sample_counts
            unique_reason_codes = np.unique(reason_codes)
            termination_reason_code = (
                int(unique_reason_codes[0]) if unique_reason_codes.size == 1 else TERMINATION_REASON_NONE
            )
            return EvaluationResult(
                mode=mode,
                calls=call + 1,
                return_mean=float(returns.mean()),
                reward_mean=float(reward_means.mean()),
                action_abs_mean=float(action_abs_means.mean()),
                final_object_target_distance=float(final_distances.mean()),
                max_object_target_distance=float(max_distances.max()),
                contact_reward_mean=float(contact_means.mean()),
                reset_seen=True,
                timeout_seen=bool(timeout.any()),
                completed_horizon=bool(success.all()),
                rewards_by_call=rewards_by_call,
                object_target_distance_by_call=distance_means,
                success_seen=bool(success.any()),
                failure_seen=bool(failure.any()),
                termination_reason_code=termination_reason_code,
                groups=_evaluation_group_results(
                    layout,
                    returns=returns,
                    reward_means=reward_means,
                    action_abs_means=action_abs_means,
                    final_distances=final_distances,
                    max_distances=max_distances,
                    contact_means=contact_means,
                    success=success,
                    failure=failure,
                ),
            )
    remaining = np.flatnonzero(active).tolist()
    raise RuntimeError(
        f"{mode} evaluation did not terminate every environment; remaining env_ids={remaining[:16]}"
    )


def _post_interaction_runs_optimizer(
    agent: Any, *, timestep: int, fallback_rollout_index: int, fallback_rollouts: int
) -> bool:
    cfg = getattr(agent, "cfg", None)
    rollouts = int(getattr(cfg, "rollouts", fallback_rollouts))
    learning_starts = int(getattr(cfg, "learning_starts", 0))
    rollout = int(getattr(agent, "_rollout", fallback_rollout_index))
    training = bool(getattr(agent, "training", True))
    return training and not (rollout + 1) % rollouts and timestep >= learning_starts


def _reset_done_observations(
    runtime: ManoSkrlRuntime, observations: torch.Tensor, done: torch.Tensor
) -> torch.Tensor:
    """Resolve terminal rows to reset ``s0`` before the next policy action."""

    mask = torch.as_tensor(done, device=observations.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != observations.shape[0]:
        raise ValueError("done mask and observation batch size must match")
    if not bool(mask.any()):
        return observations
    runtime_reset_done = getattr(runtime, "reset_done", None)
    if callable(runtime_reset_done):
        return runtime_reset_done(observations, mask)
    env_ids = torch.nonzero(mask, as_tuple=False).reshape(-1).cpu().numpy().astype(np.int64)
    try:
        reset_observations, _ = runtime.env.reset(options={"env_ids": env_ids})
    except TypeError:
        # Minimal fake runtimes used by pure training-loop tests may only expose
        # a full reset. Production always takes the indexed branch above.
        reset_observations, _ = runtime.env.reset()
    reset_observations = torch.as_tensor(
        reset_observations, device=observations.device, dtype=observations.dtype
    )
    if reset_observations.shape != observations.shape:
        raise RuntimeError("reset observation batch does not match policy observations")
    updated = observations.clone()
    row_ids = torch.nonzero(mask, as_tuple=False).reshape(-1)
    updated[row_ids] = reset_observations[row_ids]
    return updated


def _train(
    runtime: ManoSkrlRuntime,
    budget: TrainingBudget,
    recorder: ManoRerunRecorder | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    on_completed_episodes: Callable[[dict[str, object]], None] | None = None,
    observer: TrainingObserver | None = None,
) -> tuple[list[dict[str, Any]], int, float]:
    environment = runtime.gymnasium_env.environment
    config = runtime.config
    runtime.agent.enable_training_mode(True)
    observations, _ = runtime.env.reset()
    if budget.profile_phases:
        reset_environment_profile = getattr(environment, "reset_phase_profile", None)
        if callable(reset_environment_profile):
            reset_environment_profile()
        reset_conversion_profile = getattr(runtime, "reset_conversion_phase_profile", None)
        if callable(reset_conversion_profile):
            reset_conversion_profile()
    started = time.monotonic()
    updates: list[dict[str, Any]] = []
    global_timestep = 0
    pending_done = torch.zeros(
        (environment.config.num_envs, 1), device=observations.device, dtype=torch.bool
    )
    telemetry_layout = _environment_telemetry_layout(
        environment, num_envs=environment.config.num_envs
    )
    episode_component_totals: dict[str, np.ndarray] = {
        source: np.zeros(environment.config.num_envs, dtype=np.float64)
        for source in GYM_EPISODE_GROUP_COMPONENTS
    }
    episode_component_totals.update({
        alias: np.zeros(environment.config.num_envs, dtype=np.float64)
        for alias, _ in EPISODE_REWARD_ALIASES
        if alias not in episode_component_totals
    })
    profile_totals: dict[str, float] = {}
    profile_counts: dict[str, int] = {}
    profile_cuda = str(getattr(runtime, "device", getattr(runtime.agent, "device", "cpu"))).startswith("cuda")
    episode_lengths = np.zeros(environment.config.num_envs, dtype=np.int64)

    def synchronize() -> None:
        if profile_cuda:
            torch.cuda.synchronize()

    def record_phase(name: str, elapsed: float) -> None:
        profile_totals[name] = profile_totals.get(name, 0.0) + elapsed
        profile_counts[name] = profile_counts.get(name, 0) + 1

    def phase_start(name: str) -> float | None:
        if not budget.profile_phases:
            return None
        synchronize()
        return time.perf_counter()

    def phase_stop(name: str, phase_started: float | None) -> float | None:
        if phase_started is None:
            return None
        synchronize()
        elapsed = time.perf_counter() - phase_started
        record_phase(name, elapsed)
        return elapsed

    def phase_summary() -> dict[str, dict[str, float | int]]:
        summary = {
            name: {
                "total_seconds": total,
                "calls": profile_counts[name],
                "mean_seconds": total / profile_counts[name],
            }
            for name, total in profile_totals.items()
        }
        conversion_profile = getattr(runtime, "conversion_phase_profile", lambda: {})()
        overlap = set(summary) & set(conversion_profile)
        if overlap:
            raise RuntimeError(f"duplicate training phase names: {sorted(overlap)}")
        return {**summary, **conversion_profile}
    for update in range(budget.updates):
        if (
            budget.wall_clock_seconds is not None
            and time.monotonic() - started >= budget.wall_clock_seconds
        ):
            break
        update_started = time.monotonic()
        rewards: list[torch.Tensor] = []
        action_magnitudes: list[torch.Tensor] = []
        reward_components: dict[str, list[np.ndarray]] = {name: [] for name in REWARD_UPDATE_COMPONENTS}
        completed_episode_returns: list[float] = []
        completed_episode_lengths: list[int] = []
        completed_episode_components: dict[str, list[float]] = {
            alias: [] for alias, _ in EPISODE_REWARD_ALIASES
        }
        instant_component_means: dict[str, float] = {}
        reset_count = 0
        rollout_started = time.perf_counter()
        success_count = 0
        failure_count = 0
        grouped_telemetry = (
            _GroupedUpdateTelemetry(telemetry_layout)
            if telemetry_layout is not None
            else None
        )
        for update_step in range(config.rollouts):
            boundary_reset_phase = phase_start("episode_boundary_reset")
            if bool(pending_done.any()):
                observations = _reset_done_observations(runtime, observations, pending_done)
                pending_done.zero_()
            phase_stop("episode_boundary_reset", boundary_reset_phase)
            policy_phase = phase_start("policy_action")
            with torch.no_grad():
                actions, _ = runtime.agent.act(
                    observations, None, timestep=global_timestep, timesteps=budget.transitions
                )
            phase_stop("policy_action", policy_phase)
            environment_step_phase = phase_start("runtime_env_step")
            next_observations, reward, terminated, truncated, infos = runtime.env.step(actions)
            done = terminated | truncated
            phase_stop("runtime_env_step", environment_step_phase)
            observer_phase = phase_start("observer_and_rerun")
            if observer is not None:
                observer.observe()
            if recorder is not None and (
                global_timestep % budget.rerun_stride == 0 or bool(done.any().item())
            ):
                recorder.record_transition()
            phase_stop("observer_and_rerun", observer_phase)
            finite_phase = phase_start("rollout_finite_checks")
            finite_rollout = torch.isfinite(next_observations).all() and torch.isfinite(reward).all()
            phase_stop("rollout_finite_checks", finite_phase)
            if not finite_rollout:
                raise RuntimeError(f"non-finite rollout value at global timestep {global_timestep}")
            recording_phase = phase_start("transition_recording")
            runtime.agent.record_transition(
                observations=observations,
                states=None,
                actions=actions,
                rewards=reward,
                next_observations=next_observations,
                next_states=None,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=global_timestep,
                timesteps=budget.transitions,
            )
            phase_stop("transition_recording", recording_phase)
            interaction_timestep = global_timestep + 1
            runs_optimizer = _post_interaction_runs_optimizer(
                runtime.agent,
                timestep=interaction_timestep,
                fallback_rollout_index=update_step,
                fallback_rollouts=config.rollouts,
            )
            post_phase = phase_start("post_interaction_all_calls")
            runtime.agent.post_interaction(timestep=interaction_timestep, timesteps=budget.transitions)
            post_elapsed = phase_stop("post_interaction_all_calls", post_phase)
            if runs_optimizer and post_elapsed is not None:
                record_phase("ppo_optimizer_rollout_boundary", post_elapsed)
            observations = next_observations
            pending_done = done
            rewards.append(reward.detach())
            action_magnitudes.append(actions.detach().abs())
            telemetry_phase = phase_start("host_telemetry")
            diagnostics = environment.last_reward
            termination = getattr(environment, "last_termination", None)
            if diagnostics is None:
                raise RuntimeError("environment omitted reward diagnostics during training")
            if termination is None:
                snapshot = environment.last_transition
                if snapshot is None:
                    raise RuntimeError("environment omitted termination state during training")
                termination = snapshot.termination
                episode_returns = np.asarray(snapshot.episode_return, dtype=np.float64)
            else:
                episode_returns = np.asarray(environment.episode_returns, dtype=np.float64)
            diagnostic_arrays = {
                name: np.asarray(getattr(diagnostics, name), dtype=np.float64)
                for name in REWARD_UPDATE_COMPONENTS
            }
            for name, values in diagnostic_arrays.items():
                reward_components[name].append(values)
                instant_component_means[name] = float(values.mean())
            episode_lengths += 1
            for alias, source_name in EPISODE_REWARD_ALIASES:
                if telemetry_layout is not None and alias in GYM_EPISODE_GROUP_COMPONENTS:
                    continue
                if source_name is None:
                    episode_component_totals[alias] += (
                        diagnostic_arrays["distance_x"]
                        + diagnostic_arrays["distance_y"]
                        + diagnostic_arrays["distance_z"]
                    )
                else:
                    episode_component_totals[alias] += diagnostic_arrays[source_name]
            grouped_components: dict[str, np.ndarray] | None = None
            if grouped_telemetry is not None:
                grouped_components = {
                    source: np.asarray(getattr(diagnostics, source), dtype=np.float64)
                    for source in GROUPED_REWARD_COMPONENTS
                }
                for source in GYM_EPISODE_GROUP_COMPONENTS:
                    episode_component_totals[source] += grouped_components[source]
            completed = np.asarray(termination.reset, dtype=bool)
            failure_value = getattr(termination, "failure", None)
            if failure_value is None:
                failure_value = getattr(termination, "deviation_reset", np.zeros_like(completed))
            failure = np.asarray(failure_value, dtype=bool)
            success_value = getattr(termination, "success", None)
            success = np.asarray(
                completed & ~failure if success_value is None else success_value,
                dtype=bool,
            )
            reason_value = getattr(termination, "reason_code", None)
            if reason_value is None:
                reason_value = np.where(
                    failure,
                    TERMINATION_REASON_FAILURE,
                    np.where(success, TERMINATION_REASON_SUCCESS, TERMINATION_REASON_NONE),
                )
            reason_codes = np.asarray(reason_value, dtype=np.int32)
            if (
                success.shape != completed.shape
                or failure.shape != completed.shape
                or reason_codes.shape != completed.shape
            ):
                raise RuntimeError("environment termination telemetry has an invalid shape")
            if np.any((success | failure) != completed) or np.any(success & failure):
                raise RuntimeError("environment termination masks are inconsistent")
            success_count += int(success.sum())
            failure_count += int(failure.sum())
            completed_episode_returns.extend(episode_returns[completed].tolist())
            if grouped_telemetry is not None:
                if grouped_components is None:
                    raise RuntimeError("grouped reward telemetry was not initialized")
                grouped_telemetry.add_step(
                    components=grouped_components,
                    completed=completed,
                    success=success,
                    failure=failure,
                    episode_returns=episode_returns,
                    episode_component_totals=episode_component_totals,
                )
            if completed.any():
                completed_episode_lengths.extend(episode_lengths[completed].tolist())
                for alias, _ in EPISODE_REWARD_ALIASES:
                    completed_episode_components[alias].extend(
                        episode_component_totals[alias][completed].tolist()
                    )
                episode_lengths[completed] = 0
            if completed.any() and on_completed_episodes is not None:
                on_completed_episodes(_completed_episode_record(
                    update=update + 1,
                    update_step=update_step + 1,
                    vector_step=global_timestep + 1,
                    num_envs=environment.config.num_envs,
                    completed=completed,
                    episode_returns=episode_returns,
                    termination_reason_codes=reason_codes
                    if hasattr(termination, "reason_code")
                    else None,
                    telemetry_layout=telemetry_layout,
                    episode_component_totals=episode_component_totals,
                ))
            if completed.any():
                for values in episode_component_totals.values():
                    values[completed] = 0.0
            phase_stop("host_telemetry", telemetry_phase)
            reset_item_phase = phase_start("reset_count_host_item")
            reset_increment = int(done.sum().item())
            phase_stop("reset_count_host_item", reset_item_phase)
            reset_count += reset_increment
            global_timestep += 1
        rollout_elapsed = time.perf_counter() - rollout_started
        parameter_finite_phase = phase_start("trainer_parameter_finite_checks")
        finite_parameters = all(torch.isfinite(parameter).all() for parameter in runtime.model.parameters())
        phase_stop("trainer_parameter_finite_checks", parameter_finite_phase)
        if not finite_parameters:
            raise RuntimeError(f"PPO update {update} produced non-finite model parameters")
        update_telemetry_phase = phase_start("update_host_telemetry")
        component_means = {
            name: float(np.concatenate(values).mean())
            for name, values in reward_components.items()
        }
        update_finished = time.monotonic()
        cumulative_elapsed = update_finished - started
        update_elapsed = update_finished - update_started
        update_transitions = config.rollouts * environment.config.num_envs
        reward_mean = float(torch.cat(rewards).mean().item())
        update_metrics: dict[str, Any] = {
            "update": float(update + 1),
            "environment_transitions": float((update + 1) * update_transitions),
            "reward_mean": reward_mean,
            "action_abs_mean": float(torch.cat(action_magnitudes).mean().item()),
            "reset_count": float(reset_count),
            "success_count": float(success_count),
            "failure_count": float(failure_count),
            "completed_episode_count": float(len(completed_episode_returns)),
            "elapsed_seconds": cumulative_elapsed,
            "update_elapsed_seconds": update_elapsed,
            "update_environment_transitions_per_second": _transitions_per_second(
                update_transitions, update_elapsed
            ),
            "cumulative_environment_transitions_per_second": _transitions_per_second(
                (update + 1) * update_transitions, cumulative_elapsed
            ),
            "performance/total_fps": _transitions_per_second(
                (update + 1) * update_transitions, cumulative_elapsed
            ),
            "performance/step_fps": _transitions_per_second(update_transitions, update_elapsed),
            "performance/update_time": update_elapsed,
            "performance/play_time": rollout_elapsed,
            "info/epochs": float(update + 1),
            # Gym reward aliases are completed-episode metrics and are added
            # only below when an episode actually finishes.
            "manorl/reward_mean": reward_mean,
            **{
                alias: instant_component_means[source_name]
                for alias, source_name in GYM_INSTANT_REWARD_ALIASES
            },
            # RewardCalculator defines distance_reward as x + y + z.
            "distance_reward_instant/step": sum(
                instant_component_means[name] for name in ("distance_x", "distance_y", "distance_z")
            ),
            **{f"manorl/{name}_mean": value for name, value in component_means.items()},
        }
        if grouped_telemetry is not None:
            update_metrics["grouped_metrics"] = grouped_telemetry.metrics()
        update_metrics.update(_latest_skrl_tracking_metrics(runtime.agent))
        if completed_episode_returns:
            episode_return_array = np.asarray(completed_episode_returns, dtype=np.float64)
            episode_total_mean = float(np.mean(episode_return_array))
            episode_length_array = np.asarray(completed_episode_lengths, dtype=np.float64)
            update_metrics["episode_return_mean"] = episode_total_mean
            update_metrics["episode_total_mean"] = episode_total_mean
            update_metrics["episode_total_min"] = float(np.min(episode_return_array))
            update_metrics["episode_total_max"] = float(np.max(episode_return_array))
            update_metrics["episode_reward"] = episode_total_mean
            update_metrics["rewards/iter"] = episode_total_mean
            update_metrics["rewards/step"] = episode_total_mean
            update_metrics["rewards/time"] = episode_total_mean
            update_metrics["rewards/frame"] = episode_total_mean
            update_metrics["episode_lengths/frame"] = float(np.mean(episode_length_array))
            update_metrics["episode_lengths/iter"] = float(np.mean(episode_length_array))
            update_metrics["episode_lengths/step"] = float(np.mean(episode_length_array))
            # RLGPUAlgoObserver writes completed episode_data keys directly.
            for alias, values in completed_episode_components.items():
                component_values = np.asarray(values, dtype=np.float64)
                update_metrics[alias] = float(np.mean(component_values))
                update_metrics[f"episode_cumulative/{alias}"] = float(np.mean(component_values))
                update_metrics[f"episode_cumulative_min/{alias}_min"] = float(np.min(component_values))
                update_metrics[f"episode_cumulative_max/{alias}_max"] = float(np.max(component_values))
            # This bounded trainer resolves one global object/gesture group.
            # Mirror the source observer's selected-group reward/component keys.
            groups = (f"{budget.object_type}_{budget.gesture}", f"object_{budget.object_type}")
            for group in groups:
                update_metrics[f"episode_reward/{group}"] = episode_total_mean
                for alias, values in completed_episode_components.items():
                    if alias == "episode_reward":
                        continue
                    component_values = np.asarray(values, dtype=np.float64)
                    key = f"{alias}/{group}"
                    component_mean = float(np.mean(component_values))
                    update_metrics[key] = component_mean
                    update_metrics[f"episode_cumulative/{key}"] = component_mean
                    update_metrics[f"episode_cumulative_min/{key}_min"] = float(np.min(component_values))
                    update_metrics[f"episode_cumulative_max/{key}_max"] = float(np.max(component_values))
            # One update retains at most one rollout batch: 48 * 4096 = 196,608
            # values for the target Server2 scale, then this list is discarded.
            update_metrics["episode_return_values"] = completed_episode_returns
        phase_stop("update_host_telemetry", update_telemetry_phase)
        if budget.profile_phases:
            update_metrics["training_phase_profile"] = phase_summary()
        updates.append(_public_update_metrics(update_metrics))
        if on_update is not None:
            on_update(update_metrics)
        # A GLFW close is a request to stop after this complete PPO rollout.
        if observer is not None and observer.close_requested:
            break
    runtime.training_phase_profile = phase_summary() if budget.profile_phases else {}
    return updates, global_timestep * environment.config.num_envs, time.monotonic() - started


def _numbered_checkpoint_path(output: Path, completed_updates: int) -> Path:
    return output / f"checkpoint-{completed_updates:06d}.pt"


def _last_checkpoint_path(output: Path) -> Path:
    return output / "last.pt"


def _checkpoint_runtime_config(
    runtime: ManoSkrlRuntime,
    update: dict[str, float],
    *,
    trajectory_selection: dict[str, object] | None = None,
) -> dict[str, object]:
    config = {
        **runtime.checkpoint_metadata(),
        "training_progress": {
            "completed_updates": int(update["update"]),
            "environment_transitions": int(update["environment_transitions"]),
        },
    }
    if trajectory_selection is not None:
        config["trajectory_selection"] = trajectory_selection
    return config


def _checkpoint_sidecar_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".json")


def _temporary_checkpoint_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f".{checkpoint.name}.{uuid4().hex}.tmp")


@contextmanager
def _owned_initial_checkpoint(output: Path) -> Iterator[Path]:
    """Reserve an output-owned native checkpoint for the initial-state boundary."""

    output.mkdir(parents=True, exist_ok=True)
    for _ in range(16):
        checkpoint = output / f".initial-{uuid4().hex}.pt"
        try:
            checkpoint.touch(exist_ok=False)
        except FileExistsError:
            continue
        sidecar = _checkpoint_sidecar_path(checkpoint)
        if sidecar.exists():
            checkpoint.unlink()
            continue
        try:
            yield checkpoint
        finally:
            checkpoint.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
        return
    raise RuntimeError("could not reserve a unique initial checkpoint path")


def _evaluation_ppo_config(training_config: ManoPPOConfig, *, num_envs: int) -> ManoPPOConfig:
    """Use the largest training-compatible minibatch that divides the evaluation batch."""

    batch_size = training_config.rollouts * num_envs
    minibatch_size = math.gcd(training_config.minibatch_size, batch_size)
    if minibatch_size < 1:
        raise ValueError("evaluation PPO minibatch has no valid divisor")
    config = replace(training_config, minibatch_size=minibatch_size, profile_phases=False)
    config.skrl_config(num_envs=num_envs, device="cuda")
    return config


def _trajectory_assignments(trajectories: Any) -> list[dict[str, object]]:
    return [
        {
            "env_id": env_id,
            "identity": item.identity.identity,
            "object": item.identity.identity.split("_")[0],
            "action": item.identity.identity.split("_")[1],
            "row_index": item.identity.row_index,
            "uuid": item.identity.uuid,
            "source_slice": [item.identity.source_start, item.identity.source_stop],
        }
        for env_id, item in enumerate(trajectories.trajectories)
    ]


def _trajectory_selection_metadata(
    selection: TrajectorySelection,
    trajectories: Any,
    *,
    assignments: list[dict[str, object]],
    evaluation_assignments: list[dict[str, object]],
) -> dict[str, object]:
    resolved_pairs = [
        {"object": pair.object_type, "action": pair.action_id}
        for pair in getattr(trajectories, "resolved_pairs", ())
    ]
    if not resolved_pairs:
        pairs = {
            (fields[0], fields[1])
            for item in assignments
            if len(fields := str(item["identity"]).split("_")) == 3
        }
        resolved_pairs = [
            {"object": object_type, "action": action_id}
            for object_type, action_id in sorted(pairs)
        ]
    return {
        "selector": selection.canonical_selector,
        "mode": selection.mode,
        "include_suffix_files": False,
        "identity_schema": TRAJECTORY_IDENTITY_SCHEMA,
        "padding_policy": "full" if selection.require_full_padding else "clip_to_source",
        "dataset_path": str(selection.dataset_path),
        "dataset_version": selection.expected_dataset_version,
        "resolved_pairs": resolved_pairs,
        "assignments": assignments,
        "evaluation_assignments": evaluation_assignments,
    }


def _build_training_observer(
    environment: MujocoManoEnvironment, budget: TrainingBudget
) -> TrainingObserver | None:
    if budget.headless:
        return None
    from sim.manorl.view_environment import TrainingViewer

    return TrainingViewer(
        environment,
        tile_envs=budget.viewer_envs,
        stride=budget.viewer_stride,
        quiet=budget.console_format == "json",
    )


def _build_evaluation_runtime(
    *,
    selection: TrajectorySelection,
    budget: TrainingBudget,
    training_config: ManoPPOConfig,
    num_envs: int | None = None,
) -> tuple[ManoSkrlRuntime, ManoPPOConfig, list[dict[str, object]]]:
    num_envs = budget.resolved_evaluation_num_envs if num_envs is None else num_envs
    maximum = min(budget.num_envs, 128)
    if not 1 <= num_envs <= maximum:
        raise ValueError(f"evaluation num_envs must be within 1..{maximum}")
    trajectories = load_assigned_trajectory_batch(selection, num_envs=num_envs)
    contacts_per_world = (
        UNIFIED_WARP_CONTACTS_PER_WORLD
        if budget.unified_object_batch
        else WARP_BROADPHASE_CONTACTS_PER_WORLD
    )
    physical = MujocoManoEnvironment(
        trajectories,
        EnvironmentConfig(
            num_envs=num_envs,
            device="gpu",
            residual_enabled=budget.residual_enabled,
            max_deviation_distance=TARGET_MAX_DEVIATION_DISTANCE if budget.terminal else 1_000_000.0,
            contact_capacity=max(128, contacts_per_world * num_envs + WARP_CONTACT_CAPACITY_MARGIN),
            unified_object_batch=budget.unified_object_batch,
        ),
    )
    ppo_config = _evaluation_ppo_config(training_config, num_envs=num_envs)
    return ManoSkrlRuntime(ManoGymnasiumVectorEnv(physical), ppo_config), ppo_config, _trajectory_assignments(trajectories)


def _full_coverage_evaluation_num_envs(
    budget: TrainingBudget, trajectories: Any
) -> int:
    """Use at least one deterministic evaluation environment per resolved pair."""

    pair_count = max(1, len(getattr(trajectories, "resolved_pairs", ())))
    num_envs = max(budget.resolved_evaluation_num_envs, pair_count)
    maximum = min(budget.num_envs, 128)
    if num_envs > maximum:
        raise ValueError(
            f"full-pair evaluation needs {num_envs} environments but the bounded maximum is {maximum}"
        )
    return num_envs


def _save_checkpoint_atomically(
    agent: Any, checkpoint: Path, *, runtime_config: dict[str, object]
) -> Path:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = _temporary_checkpoint_path(checkpoint)
    temporary_sidecar = _checkpoint_sidecar_path(temporary_checkpoint)
    sidecar = _checkpoint_sidecar_path(checkpoint)
    try:
        save_skrl_checkpoint(agent, temporary_checkpoint, runtime_config=runtime_config)
        metadata = json.loads(temporary_sidecar.read_text(encoding="utf-8"))
        metadata["checkpoint_file"] = checkpoint.name
        temporary_sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_checkpoint, checkpoint)
        os.replace(temporary_sidecar, sidecar)
    finally:
        temporary_checkpoint.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
    return checkpoint


def _last_sidecar_content(checkpoint: Path, last_checkpoint: Path) -> bytes:
    metadata = json.loads(_checkpoint_sidecar_path(checkpoint).read_text(encoding="utf-8"))
    runtime_config = dict(metadata["runtime_config"])
    runtime_config.pop("training_progress", None)
    metadata["checkpoint_file"] = last_checkpoint.name
    metadata["runtime_config"] = runtime_config
    metadata["progress_metadata"] = "immutable numbered and final checkpoint sidecars"
    return (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _update_last_checkpoint(output: Path, checkpoint: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    last_checkpoint = _last_checkpoint_path(output)
    last_sidecar = _checkpoint_sidecar_path(last_checkpoint)
    sidecar_content = _last_sidecar_content(checkpoint, last_checkpoint)
    if last_sidecar.exists():
        if not last_sidecar.is_file() or last_sidecar.read_bytes() != sidecar_content:
            raise ValueError(f"last checkpoint compatibility sidecar does not match: {last_sidecar}")
    else:
        if last_checkpoint.exists():
            raise FileExistsError(f"last checkpoint exists without compatibility sidecar: {last_checkpoint}")
        temporary_sidecar = _temporary_checkpoint_path(last_sidecar)
        try:
            temporary_sidecar.write_bytes(sidecar_content)
            os.replace(temporary_sidecar, last_sidecar)
        finally:
            temporary_sidecar.unlink(missing_ok=True)
    temporary_checkpoint = _temporary_checkpoint_path(last_checkpoint)
    try:
        shutil.copyfile(checkpoint, temporary_checkpoint)
        os.replace(temporary_checkpoint, last_checkpoint)
    finally:
        temporary_checkpoint.unlink(missing_ok=True)
    return last_checkpoint


def _save_periodic_checkpoint(
    runtime: ManoSkrlRuntime,
    output: Path,
    update: dict[str, float],
    *,
    trajectory_selection: dict[str, object] | None = None,
) -> Path:
    checkpoint = _numbered_checkpoint_path(output, int(update["update"]))
    sidecar = _checkpoint_sidecar_path(checkpoint)
    if checkpoint.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to replace periodic checkpoint artifact: {checkpoint}")
    return _save_checkpoint_atomically(
        runtime.agent,
        checkpoint,
        runtime_config=_checkpoint_runtime_config(
            runtime, update, trajectory_selection=trajectory_selection
        ),
    )


def _maybe_save_periodic_checkpoint(
    runtime: ManoSkrlRuntime,
    output: Path,
    checkpoint_interval_updates: int | None,
    update: dict[str, float],
    *,
    trajectory_selection: dict[str, object] | None = None,
) -> Path | None:
    completed_updates = int(update["update"])
    if checkpoint_interval_updates is None or completed_updates % checkpoint_interval_updates != 0:
        return None
    checkpoint = _save_periodic_checkpoint(
        runtime, output, update, trajectory_selection=trajectory_selection
    )
    _update_last_checkpoint(output, checkpoint)
    return checkpoint


def run(output: Path, budget: TrainingBudget) -> dict[str, Any]:
    _assert_cuda_runtime()
    if output.suffix:
        raise ValueError("--output must be a prefix without a suffix")
    if budget.rerun_output is not None and budget.rerun_grpc_url is not None:
        raise ValueError("rerun_output and rerun_grpc_url are mutually exclusive")
    if budget.device_resident_controls and (
        budget.rerun_output is not None
        or budget.rerun_grpc_url is not None
        or budget.rerun_high_return_dir is not None
        or budget.rerun_high_return_threshold is not None
        or not budget.headless
    ):
        raise ValueError("device-resident-controls requires headless training without Rerun transition recording")
    if budget.rerun_high_return_threshold is not None and budget.rerun_high_return_dir is None:
        raise ValueError("rerun_high_return_threshold requires rerun_high_return_dir")
    if budget.rerun_high_return_dir is not None and budget.rerun_output is None:
        raise ValueError("rerun_high_return_dir requires rerun_output")
    if budget.rerun_high_return_following < 0:
        raise ValueError("rerun_high_return_following must be non-negative")
    if (
        budget.rerun_high_return_dir is not None
        and budget.rerun_stride != 1
    ):
        raise ValueError("high-return Rerun replay requires rerun_stride=1 to preserve every action")
    output = output.resolve()
    checkpoint = output.with_suffix(".pt")
    last_checkpoint = _last_checkpoint_path(output)
    metrics_path = output.with_suffix(".json")
    trace_path = output.with_suffix(".eval.npz")
    episodes_path = _episode_records_path(output)
    partial_episodes_path = _partial_episode_records_path(output)
    rerun_path = Path(budget.rerun_output).resolve() if budget.rerun_output else None
    rerun_high_return_dir = (
        Path(budget.rerun_high_return_dir).resolve() if budget.rerun_high_return_dir else None
    )
    artifacts = (
        checkpoint,
        _checkpoint_sidecar_path(checkpoint),
        metrics_path,
        trace_path,
        episodes_path,
        partial_episodes_path,
    )
    if output.exists() or any(path.exists() for path in artifacts):
        raise FileExistsError("refusing to replace an existing training artifact prefix")
    torch.manual_seed(budget.seed)
    np.random.seed(budget.seed)
    torch.cuda.manual_seed_all(budget.seed)

    contacts_per_world = (
        UNIFIED_WARP_CONTACTS_PER_WORLD
        if budget.unified_object_batch
        else WARP_BROADPHASE_CONTACTS_PER_WORLD
    )
    contact_capacity = max(
        128,
        contacts_per_world * budget.num_envs + WARP_CONTACT_CAPACITY_MARGIN,
    )
    selection = TrajectorySelection(
        object_type=budget.object_type,
        gesture=budget.gesture,
        selector=budget.trajectory_selector,
        dataset_path=Path(budget.dataset_path),
    )
    trajectories = load_assigned_trajectory_batch(selection, num_envs=budget.num_envs)
    evaluation_num_envs = _full_coverage_evaluation_num_envs(budget, trajectories)
    assigned_object_types = {
        item.identity.identity.split("_")[0]
        for item in getattr(trajectories, "trajectories", ())
    }
    if len(assigned_object_types) > 1 and (
        budget.rerun_output is not None
        or budget.rerun_grpc_url is not None
        or budget.rerun_high_return_dir is not None
        or not budget.headless
    ):
        raise ValueError(
            "mixed-object training supports headless execution only; "
            "the viewer and Rerun recorder require one MuJoCo object model"
        )
    physical = MujocoManoEnvironment(
        trajectories,
        EnvironmentConfig(
            num_envs=budget.num_envs,
            device="gpu",
            residual_enabled=budget.residual_enabled,
            max_deviation_distance=TARGET_MAX_DEVIATION_DISTANCE if budget.terminal else 1_000_000.0,
            contact_capacity=contact_capacity,
            device_resident_controls=budget.device_resident_controls,
            capture_transition_diagnostics=budget.resolved_capture_transition_diagnostics,
            profile_phases=budget.profile_phases,
            unified_object_batch=budget.unified_object_batch,
        ),
    )
    ppo_config = ManoPPOConfig(
        minibatch_size=budget.resolved_minibatch_size,
        use_film=budget.use_film,
        profile_phases=budget.profile_phases,
    )
    ppo_config.skrl_config(num_envs=budget.num_envs, device="cuda")
    runtime = ManoSkrlRuntime(ManoGymnasiumVectorEnv(physical), ppo_config)
    if runtime.device != "cuda":
        raise RuntimeError(f"skrl runtime must train on CUDA, got {runtime.device!r}")
    trajectory_assignments = _trajectory_assignments(trajectories)
    evaluation_runtime, evaluation_ppo_config, evaluation_trajectory_assignments = _build_evaluation_runtime(
        selection=selection,
        budget=budget,
        training_config=ppo_config,
        num_envs=evaluation_num_envs,
    )
    trajectory_selection = _trajectory_selection_metadata(
        selection,
        trajectories,
        assignments=trajectory_assignments,
        evaluation_assignments=evaluation_trajectory_assignments,
    )
    device = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "skrl": runtime.device,
        "jax": physical.jax.default_backend(),
    }
    wandb_config = _wandb_config(
        budget=budget,
        ppo_config=ppo_config,
        trajectory_assignments=trajectory_assignments,
        evaluation_ppo_config=evaluation_ppo_config,
        evaluation_trajectory_assignments=evaluation_trajectory_assignments,
        trajectory_selection=trajectory_selection,
        device=device,
    )
    with _wandb_run(output=output, budget=budget, config=wandb_config) as (wandb_run, wandb):
        with _owned_initial_checkpoint(output) as initial_checkpoint:
            _save_checkpoint_atomically(
                runtime.agent,
                initial_checkpoint,
                runtime_config=_checkpoint_runtime_config(
                    runtime,
                    {"update": 0.0, "environment_transitions": 0.0},
                    trajectory_selection=trajectory_selection,
                ),
            )
            load_skrl_checkpoint(evaluation_runtime.agent, initial_checkpoint)
            zero_baseline = _evaluate(evaluation_runtime, "zero")
            untrained = _evaluate(evaluation_runtime, "untrained")
            # Make the checkpoint boundary executable: PPO starts from exactly
            # the native policy, value, optimizer, and normalizer state reported
            # by the untrained comparison, independent of RNG construction.
            load_skrl_checkpoint(runtime.agent, initial_checkpoint)
        # Keep this bounded evaluator alive for the trained row. Reconstructing
        # all object routes after a large training runtime leaves JAX/Warp
        # allocator caches competing with a second set of evaluator buffers.
        if wandb_run is not None:
            _log_wandb_evaluations(wandb_run, [zero_baseline, untrained], update=0, transitions=0)
        recorder: ManoRerunRecorder | None = None
        observer: TrainingObserver | None = None
        published_rerun: Path | None = None
        periodic_checkpoints: list[Path] = []

        def on_update(update: dict[str, Any]) -> None:
            periodic_checkpoint = _maybe_save_periodic_checkpoint(
                runtime,
                output,
                budget.checkpoint_interval_updates,
                update,
                trajectory_selection=trajectory_selection,
            )
            if periodic_checkpoint is not None:
                periodic_checkpoints.append(periodic_checkpoint)
                _emit_console(
                    budget.console_format,
                    "checkpoint",
                    {"update": update["update"], "path": str(periodic_checkpoint)},
                )
            _emit_console(
                budget.console_format,
                "training_update",
                {"metrics": _public_update_metrics(update)},
            )
            if wandb_run is not None:
                _log_wandb_update(wandb_run, wandb, update)

        try:
            recorder = (
                ManoRerunRecorder(
                    physical,
                    rerun_path or "outputs/manorl/live_training.rrd",
                    env_id=budget.rerun_env_id,
                    grpc_url=budget.rerun_grpc_url,
                    archive_dir=rerun_high_return_dir,
                    archive_threshold=budget.rerun_high_return_threshold,
                    archive_following=budget.rerun_high_return_following,
                )
                if rerun_path is not None or budget.rerun_grpc_url is not None
                else None
            )
            observer = _build_training_observer(physical, budget)
            episodes_path.parent.mkdir(parents=True, exist_ok=True)
            with _episode_records_file(episodes_path) as episode_file:
                updates, transitions, elapsed = _train(
                    runtime,
                    budget,
                    recorder,
                    on_update=on_update,
                    on_completed_episodes=lambda record: _write_episode_record(
                        episode_file, record, console_format=budget.console_format
                    ),
                    observer=observer,
                )
        finally:
            try:
                if observer is not None:
                    observer.close()
            finally:
                if recorder is not None:
                    published_rerun = recorder.close()
        throughput = {
            "environment_transitions": transitions,
            "elapsed_seconds": elapsed,
            "final_environment_transitions_per_second": _transitions_per_second(transitions, elapsed),
        }
        _emit_console(budget.console_format, "training_complete", {"throughput": throughput})
        rerun_artifact = None if published_rerun is None else str(published_rerun)
        final_update = {
            "update": float(len(updates)),
            "environment_transitions": float(transitions),
        }
        _save_checkpoint_atomically(
            runtime.agent,
            checkpoint,
            runtime_config=_checkpoint_runtime_config(
                runtime, final_update, trajectory_selection=trajectory_selection
            ),
        )
        _update_last_checkpoint(output, checkpoint)
        environment_result = {
            "residual_enabled": physical.config.residual_enabled,
            "observation_contact_threshold_N": CONTACT_FORCE_THRESHOLD,
            "residual_action": asdict(physical.config.residual_action),
            "max_deviation_distance": physical.config.max_deviation_distance,
            "device_resident_controls": physical.config.device_resident_controls,
            "capture_transition_diagnostics": physical.config.capture_transition_diagnostics,
        }
        learning_starts = runtime.agent.cfg.learning_starts
        phase_profile = {
            "environment": physical.phase_profile() if hasattr(physical, "phase_profile") else {},
            "contact": physical.contact_profile() if hasattr(physical, "contact_profile") else {},
            "training": getattr(runtime, "training_phase_profile", {}),
        }
        # The full-pair evaluator must not execute while the 2,048-world
        # training runtime is still owned. Preserve plain result metadata first,
        # then remove every training-only owner before reusing the bounded
        # evaluator constructed for the initial comparison.
        observer = None
        recorder = None
        del runtime
        del physical
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Evaluate exactly what a user will later load. skrl preprocessor/module
        # state may differ after PPO training, so loading the final native
        # checkpoint remains the reproducibility boundary even though the
        # already-built physical evaluator is reused.
        load_skrl_checkpoint(evaluation_runtime.agent, checkpoint)
        trained = _evaluate(evaluation_runtime, "trained")
        np.savez_compressed(
            trace_path,
            zero_reward=np.asarray(zero_baseline.rewards_by_call, dtype=np.float32),
            untrained_reward=np.asarray(untrained.rewards_by_call, dtype=np.float32),
            trained_reward=np.asarray(trained.rewards_by_call, dtype=np.float32),
            zero_object_target_distance=np.asarray(zero_baseline.object_target_distance_by_call, dtype=np.float32),
            untrained_object_target_distance=np.asarray(untrained.object_target_distance_by_call, dtype=np.float32),
            trained_object_target_distance=np.asarray(trained.object_target_distance_by_call, dtype=np.float32),
        )
        result = {
            "schema": "manorl.cube1_fast_training.v1",
            "trajectory_selection": trajectory_selection,
            "checkpoint_conversion": "tools/convert_gym_checkpoint.py",
            "initialization": {
                "actor_mean": "source_default",
                "initial_log_std": -0.99,
                "ppo_learning_rate": ppo_config.learning_rate,
            },
            "reward": {
                "environment_contract": REWARD_CONTRACT_ID,
                "contact_force_threshold_N": REWARD_HAND_OBJECT_THRESHOLD_N,
                "ppo_contract": PPO_REWARD_CONTRACT_ID,
                "ppo_scale": PPO_REWARD_SCALE,
                "isaacgym_ppo_scale": 0.5,
            },
            "environment": environment_result,
            "environment_contract": ENVIRONMENT_CONTRACT_ID,
            "learning_starts": learning_starts,
            "budget": {
                **asdict(budget),
                "planned_transitions": budget.transitions,
                "resolved_capture_transition_diagnostics": budget.resolved_capture_transition_diagnostics,
                "warp_contact_capacity": contact_capacity,
                "evaluation_num_envs": evaluation_num_envs,
                "evaluation_ppo_config": asdict(evaluation_ppo_config),
            },
            "actual": {"transitions": transitions, "elapsed_seconds": elapsed, "updates": len(updates)},
            "throughput": throughput,
            "phase_profile": phase_profile,
            "device": device,
            "baseline": asdict(zero_baseline),
            "untrained": asdict(untrained),
            "trained": asdict(trained),
            "acceptance": {
                "trained_completed_horizon": trained.completed_horizon,
                "trained_calls_not_before_zero_reference": trained.calls >= zero_baseline.calls,
                "trained_return_exceeds_untrained": trained.return_mean > untrained.return_mean,
                "accepted": (
                    trained.calls >= zero_baseline.calls
                    and trained.return_mean > untrained.return_mean
                ),
            },
            "updates": updates,
            "artifacts": {
                "checkpoint": str(checkpoint),
                "last_checkpoint": str(last_checkpoint),
                "periodic_checkpoints": [str(path) for path in periodic_checkpoints],
                "evaluation_trace": str(trace_path),
                "episode_returns": str(episodes_path),
                "rerun": rerun_artifact,
            },
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if wandb_run is not None:
            _log_wandb_evaluations(
                wandb_run, [trained], update=len(updates), transitions=transitions
            )
            acceptance_metrics = {
                "global_step": len(updates),
                "update": len(updates),
                "transitions": transitions,
                **throughput,
                **{f"acceptance/{key}": value for key, value in result["acceptance"].items()},
            }
            wandb_run.log(acceptance_metrics, step=len(updates))
            wandb_run.summary.update(acceptance_metrics)
            artifact_paths = [
                checkpoint,
                _checkpoint_sidecar_path(checkpoint),
                last_checkpoint,
                _checkpoint_sidecar_path(last_checkpoint),
                metrics_path,
                trace_path,
                episodes_path,
            ]
            for periodic_checkpoint in periodic_checkpoints:
                artifact_paths.extend(
                    [periodic_checkpoint, _checkpoint_sidecar_path(periodic_checkpoint)]
                )
            if published_rerun is not None:
                artifact_paths.append(published_rerun)
            _log_wandb_artifacts(wandb_run, wandb, output=output, paths=artifact_paths)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--evaluation-num-envs", type=int, default=1)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(DATASET_PATH),
        help="pinned Lance dataset path (default: repository contract path)",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        help="override resolved Gym minibatch size (default: largest 4096-compatible divisor)",
    )
    parser.add_argument("--wall-clock-seconds", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rerun-output", type=Path, help="optional .rrd transition recording for one training env")
    parser.add_argument(
        "--rerun-grpc-url",
        help="optional Rerun gRPC URL (for example rerun+http://127.0.0.1:9876/proxy) for live web streaming",
    )
    parser.add_argument(
        "--rerun-high-return-dir",
        type=Path,
        help="optional directory for preserving completed local .rrd episodes above the threshold",
    )
    parser.add_argument(
        "--rerun-high-return-threshold",
        type=float,
        help="minimum completed episode return to preserve under --rerun-high-return-dir",
    )
    parser.add_argument(
        "--rerun-high-return-following",
        type=int,
        default=5,
        help="number of completed episodes after a threshold hit to preserve (default: 5)",
    )
    parser.add_argument("--rerun-env-id", type=int, default=0)
    parser.add_argument("--rerun-stride", type=int, default=1)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--all-pairs",
        action="store_true",
        help="select every eligible non-suffix object/action pair in the Lance dataset",
    )
    selection_group.add_argument(
        "--pairs",
        metavar="OBJECT:ACTION[,OBJECT:ACTION...]",
        help="select exact object/action pairs, for example cube1:01,cube2:01",
    )
    selection_group.add_argument(
        "--trajectory-selector",
        metavar="all|OBJECT:ACTION[,OBJECT:ACTION...]",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--object",
        dest="object_type",
        help="legacy single-pair object selector; requires --gesture",
    )
    parser.add_argument(
        "--gesture",
        help="legacy single-pair action selector; requires --object",
    )
    parser.add_argument("--use_residual", type=parse_cli_bool, default=True, metavar="{true,false}")
    parser.add_argument("--film", type=parse_cli_bool, default=True, metavar="{true,false}")
    parser.add_argument("--terminal", type=parse_cli_bool, default=True, metavar="{true,false}")
    parser.add_argument("--headless", type=parse_cli_bool, default=True, metavar="{true,false}")
    parser.add_argument("--viewer-envs", type=int, default=1, help="number of training worlds to tile when headless=false")
    parser.add_argument("--viewer-stride", type=int, default=1, help="render every N completed vector steps when headless=false")
    parser.add_argument("--console-format", choices=("human", "json"), default="human")
    parser.add_argument(
        "--device-resident-controls",
        type=parse_cli_bool,
        default=False,
        metavar="{true,false}",
        help="keep controller targets and delayed reset writes on the MJX device",
    )
    parser.add_argument(
        "--capture-transition-diagnostics",
        type=parse_cli_bool,
        default=None,
        metavar="{true,false}",
        help="capture transition snapshots independently; default is the inverse of device-resident-controls",
    )
    parser.add_argument(
        "--profile-phases",
        action="store_true",
        help="synchronize CUDA/JAX at explicit rollout and PPO phase boundaries and emit timings",
    )
    parser.add_argument(
        "--unified-object-batch",
        action="store_true",
        help="use one fixed-topology MJX model for mixed-object batches",
    )
    parser.add_argument("--wandb", type=parse_cli_bool, default=True, metavar="{true,false}")
    parser.add_argument("--wandb-project", default="one_policy")
    parser.add_argument("--wandb-group", default="s02")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-tags", action="append", default=[], metavar="TAG[,TAG...]")
    args = parser.parse_args(argv)
    selector = (
        "all"
        if args.all_pairs
        else args.pairs
        if args.pairs is not None
        else args.trajectory_selector
    )
    if selector is not None and (args.object_type is not None or args.gesture is not None):
        parser.error("--all-pairs/--pairs cannot be combined with --object or --gesture")
    if (args.object_type is None) != (args.gesture is None):
        parser.error("--object and --gesture must be supplied together")
    object_type = args.object_type or "cube1"
    gesture = args.gesture or "01"
    try:
        parsed_selection = TrajectorySelection(
            object_type=object_type,
            gesture=gesture,
            selector=selector,
            dataset_path=args.dataset_path,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.updates < 1 or args.num_envs < 1 or args.rerun_stride < 1:
        parser.error("updates, num-envs, and rerun-stride must be positive")
    evaluation_num_envs_maximum = min(args.num_envs, 128)
    if args.evaluation_num_envs is not None and not 1 <= args.evaluation_num_envs <= evaluation_num_envs_maximum:
        parser.error(f"evaluation-num-envs must be within 1..{evaluation_num_envs_maximum} when provided")
    resolved_minibatch_size = TrainingBudget(
        num_envs=args.num_envs, minibatch_size=args.minibatch_size
    ).resolved_minibatch_size
    try:
        ManoPPOConfig(minibatch_size=resolved_minibatch_size).skrl_config(
            num_envs=args.num_envs, device="cuda"
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.checkpoint_interval_updates is not None and args.checkpoint_interval_updates < 1:
        parser.error("checkpoint-interval-updates must be positive when provided")
    if args.wall_clock_seconds is not None and (
        not math.isfinite(args.wall_clock_seconds) or args.wall_clock_seconds <= 0
    ):
        parser.error("wall-clock-seconds must be a finite positive value when provided")
    if args.rerun_output is not None and args.rerun_grpc_url is not None:
        parser.error("--rerun-output and --rerun-grpc-url are mutually exclusive")
    if args.rerun_high_return_threshold is not None and not math.isfinite(args.rerun_high_return_threshold):
        parser.error("rerun-high-return-threshold must be finite when provided")
    if args.rerun_high_return_threshold is not None and args.rerun_high_return_dir is None:
        parser.error("rerun-high-return-threshold requires --rerun-high-return-dir")
    if args.rerun_high_return_dir is not None and args.rerun_output is None:
        parser.error("rerun-high-return-dir requires --rerun-output")
    if args.rerun_high_return_following < 0:
        parser.error("rerun-high-return-following must be non-negative")
    if not 0 <= args.rerun_env_id < args.num_envs:
        parser.error("rerun-env-id must be within num-envs")
    if not 1 <= args.viewer_envs <= args.num_envs:
        parser.error("viewer-envs must be within num-envs")
    if args.viewer_stride < 1:
        parser.error("viewer-stride must be positive")
    result = run(
        args.output,
        TrainingBudget(
            num_envs=args.num_envs,
            updates=args.updates,
            wall_clock_seconds=args.wall_clock_seconds,
            seed=args.seed,
            rerun_output=str(args.rerun_output.resolve()) if args.rerun_output is not None else None,
            rerun_grpc_url=args.rerun_grpc_url,
            rerun_high_return_dir=(
                str(args.rerun_high_return_dir.resolve()) if args.rerun_high_return_dir is not None else None
            ),
            rerun_high_return_threshold=args.rerun_high_return_threshold,
            rerun_high_return_following=args.rerun_high_return_following,
            rerun_env_id=args.rerun_env_id,
            rerun_stride=args.rerun_stride,
            object_type=object_type,
            gesture=gesture,
            trajectory_selector=(
                None if selector is None else parsed_selection.canonical_selector
            ),
            dataset_path=str(args.dataset_path.resolve()),
            residual_enabled=args.use_residual,
            use_film=args.film,
            terminal=args.terminal,
            wandb=WandbOptions(
                enabled=args.wandb,
                project=args.wandb_project,
                group=args.wandb_group,
                entity=args.wandb_entity,
                name=args.wandb_name,
                tags=_parse_wandb_tags(args.wandb_tags),
            ),
            checkpoint_interval_updates=args.checkpoint_interval_updates,
            minibatch_size=args.minibatch_size,
            evaluation_num_envs=args.evaluation_num_envs,
            headless=args.headless,
            viewer_envs=args.viewer_envs,
            viewer_stride=args.viewer_stride,
            console_format=args.console_format,
            device_resident_controls=args.device_resident_controls,
            profile_phases=args.profile_phases,
            capture_transition_diagnostics=args.capture_transition_diagnostics,
            unified_object_batch=args.unified_object_batch,
        ),
    )
    if args.console_format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"artifacts metrics={args.output.with_suffix('.json')} checkpoint={args.output.with_suffix('.pt')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
