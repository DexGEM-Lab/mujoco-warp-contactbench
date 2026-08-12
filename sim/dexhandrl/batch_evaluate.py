"""Deterministic N8 physical evaluation for DexHand021Pro MJX-Warp.

Worlds 0..6 cover cube1 actions 01/02/03/04/09/10/18.  World 7 is an exact
identity duplicate of world 0 and is excluded from aggregate task metrics; it
exists solely to detect nondeterministic batch physics, control, or policy
inference.  Evaluation runs every assigned trajectory to its full packed
horizon.  The production 0.10 m deviation threshold is reported as a
counterfactual failure event instead of truncating the physical trace.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from sim.dexhandrl.asset_paths import asset_provenance, contact_bodies_path, task_config_path
from sim.dexhandrl.constants import (
    ACTION_DIM_021PRO,
    DEFAULT_ISAAC_SOURCE_ROOT,
    DEFAULT_LANCE_PATH,
    DEFAULT_OUTPUT_ROOT,
)
from sim.dexhandrl.contracts import DexHandPhysicsConfig
from sim.dexhandrl.lance_loader import lance_dataset_metadata, list_trajectories
from sim.dexhandrl.provenance import code_provenance
from sim.dexhandrl.rewards import (
    PPO_REWARD_SCALE,
    REWARD_COMPONENTS,
    SOURCE_ALIGNED_REWARD_CONFIG,
)
from sim.dexhandrl.skrl_batch_env import DexHandRLBatchConfig, DexHandRLMJXBatchEnv
from sim.dexhandrl.trajectory_batch import (
    DEXHAND_ALL_ACTIONS,
    load_trajectory_batch,
    resolve_balanced_action_assignments,
)


EVALUATION_SCHEMA = "dexhandrl.n8_physical_evaluation.v1"
DEFAULT_FORCE_THRESHOLDS_N = (0.2, 5.0)
PRIMARY_WORLD_COUNT = len(DEXHAND_ALL_ACTIONS)


@dataclass(frozen=True)
class N8EvaluationConfig:
    mode: str = "reference"
    device: str = "gpu"
    num_envs: int = 8
    lance_path: Path = DEFAULT_LANCE_PATH
    isaac_source_root: Path = DEFAULT_ISAAC_SOURCE_ROOT
    sequence_index: int = 0
    pre_contact_frames: int = 200
    post_contact_frames: int = 250
    settle_frames: int = 1
    seed: int = 0
    checkpoint: Path | None = None
    output_dir: Path = DEFAULT_OUTPUT_ROOT / "dexhandrl_n8_evaluation"
    force_thresholds_N: tuple[float, ...] = DEFAULT_FORCE_THRESHOLDS_N
    reward_contact_threshold_N: float = 0.2
    deviation_threshold_m: float = 0.10
    max_steps: int | None = None
    headless: bool = True
    tile_envs: int = 8
    render_stride: int = 1
    speed: float = 1.0
    pointnet_enabled: bool = True
    film_enabled: bool = True
    separate: bool = True
    hand_friction: tuple[float, float, float] = (0.5, 0.01, 0.001)

    def __post_init__(self) -> None:
        if self.mode not in {"reference", "untrained", "trained"}:
            raise ValueError("mode must be reference, untrained, or trained")
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be cpu or gpu")
        if self.num_envs != 8:
            raise ValueError("the N8 evaluator requires exactly 8 environments")
        if self.mode == "trained" and self.checkpoint is None:
            raise ValueError("trained mode requires --checkpoint")
        if self.mode != "trained" and self.checkpoint is not None:
            raise ValueError("--checkpoint is accepted only in trained mode")
        if self.pre_contact_frames < 0 or self.post_contact_frames < 0 or self.settle_frames < 0:
            raise ValueError("trajectory padding and settle frames must be non-negative")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided")
        if not 1 <= self.tile_envs <= self.num_envs:
            raise ValueError("tile_envs must be within [1, num_envs]")
        if self.render_stride < 1 or self.speed <= 0:
            raise ValueError("render_stride and speed must be positive")
        if self.deviation_threshold_m < 0 or self.reward_contact_threshold_N < 0:
            raise ValueError("deviation/contact thresholds must be non-negative")
        friction = tuple(float(value) for value in self.hand_friction)
        if len(friction) != 3 or any(not np.isfinite(value) or value < 0 for value in friction):
            raise ValueError("hand_friction must contain three finite non-negative values")
        object.__setattr__(self, "hand_friction", friction)
        thresholds = tuple(float(value) for value in self.force_thresholds_N)
        if not thresholds or any(not np.isfinite(value) or value < 0 for value in thresholds):
            raise ValueError("force thresholds must be finite and non-negative")
        object.__setattr__(self, "force_thresholds_N", tuple(sorted(set(thresholds))))


@dataclass(frozen=True)
class N8EvaluationResult:
    summary_path: Path
    trace_path: Path
    summary: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {"count": int(array.size), "finite_count": 0, "mean": None, "min": None, "max": None, "final": None}
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": float(finite.mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "final": float(finite[-1]),
    }


def _force_distribution(values: np.ndarray, thresholds_N: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    positive = finite[finite > 0.0]

    def percentiles(data: np.ndarray) -> dict[str, float | None]:
        labels = (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99), ("p99_9", 99.9), ("max", 100))
        if not data.size:
            return {name: None for name, _ in labels}
        return {name: float(np.percentile(data, percentile)) for name, percentile in labels}

    threshold_records = {}
    for threshold in thresholds_N:
        count = int(np.count_nonzero(finite > threshold))
        threshold_records[f"{float(threshold):g}"] = {
            "threshold_N": float(threshold),
            "above_count": count,
            "above_fraction_all": float(count / finite.size) if finite.size else 0.0,
            "above_fraction_positive": float(count / positive.size) if positive.size else 0.0,
        }
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "positive_count": int(positive.size),
        "zero_count": int(np.count_nonzero(finite == 0.0)),
        "percentiles_all_N": percentiles(finite),
        "percentiles_positive_N": percentiles(positive),
        "thresholds": threshold_records,
    }


def _capacity_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0, "pass_below_80_percent": False}
    first = samples[0]
    contact_capacity = int(first["actual_contact_buffer_shape"][0])
    ccd_capacity = int(first["actual_ccd_capacity"])
    constraint_capacity = int(first["actual_constraint_buffer_shape"][1])
    contact_max = max(int(sample["contact_high_water"]) for sample in samples)
    ccd_max = max(int(sample["ncollision_high_water"]) for sample in samples)
    constraint_max = max(int(sample["constraint_high_water"]) for sample in samples)
    ratios = {
        "contact": float(contact_max / contact_capacity) if contact_capacity else math.inf,
        "ccd": float(ccd_max / ccd_capacity) if ccd_capacity else math.inf,
        "constraint": float(constraint_max / constraint_capacity) if constraint_capacity else math.inf,
    }
    return {
        "sample_count": len(samples),
        "requested_global_contact_capacity": int(first["requested_global_contact_capacity"]),
        "requested_ccd_capacity": int(first["requested_ccd_capacity"]),
        "requested_per_world_constraint_capacity": int(first["requested_per_world_constraint_capacity"]),
        "actual_contact_buffer_shape": list(first["actual_contact_buffer_shape"]),
        "actual_ccd_capacity": ccd_capacity,
        "actual_constraint_buffer_shape": list(first["actual_constraint_buffer_shape"]),
        "max_contact_rows": contact_max,
        "max_collision_candidates": ccd_max,
        "max_constraints_per_world": constraint_max,
        "utilization": ratios,
        "pass_below_80_percent": bool(all(value < 0.8 for value in ratios.values())),
    }


def _policy_checkpoint_sidecar(checkpoint: Path, training_reward: Any) -> dict[str, Any]:
    sidecar = checkpoint.with_suffix(".json")
    if not sidecar.is_file():
        raise RuntimeError(f"trained evaluation requires checkpoint sidecar: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    reward = payload.get("reward")
    if not isinstance(reward, dict):
        raise RuntimeError("checkpoint sidecar has no versioned reward contract")
    expected = training_reward.record()
    if reward.get("contract_id") != expected["contract_id"]:
        raise RuntimeError(
            f"checkpoint reward contract {reward.get('contract_id')!r} != evaluator {expected['contract_id']!r}"
        )
    if reward.get("ppo_contract_id") != expected["ppo_contract_id"]:
        raise RuntimeError("checkpoint PPO reward contract differs from evaluator")
    if reward.get("ppo_reward_scale") != expected["ppo_reward_scale"]:
        raise RuntimeError("checkpoint PPO reward scale differs from evaluator")
    if reward.get("config") != expected["config"]:
        raise RuntimeError("checkpoint reward configuration differs from evaluator")
    return payload


def _build_policy(cfg: N8EvaluationConfig, spaces: Any, training_reward: Any) -> tuple[Any, Any, Any, dict[str, Any] | None]:
    if cfg.mode == "reference":
        return None, None, None, None
    import torch
    from sim.dexhandrl.skrl_batch_runtime import DLPackBridge
    from sim.dexhandrl.skrl_train import SkrlLauncherConfig, _build_agent, _seed_launcher

    _seed_launcher(cfg.seed)
    launcher = SkrlLauncherConfig(
        device=cfg.device,
        num_envs=cfg.num_envs,
        lance_path=cfg.lance_path,
        isaac_source_root=cfg.isaac_source_root,
        object_name="cube1",
        action="all",
        sequence_index=cfg.sequence_index,
        use_residual=True,
        reward=training_reward,
        early_phase_frames=training_reward.early_phase_steps,
        pre_contact_frames=cfg.pre_contact_frames,
        post_contact_frames=cfg.post_contact_frames,
        object_position_terminal_threshold=training_reward.max_deviation_distance_m,
        settle_frames=cfg.settle_frames,
        seed=cfg.seed,
        updates=1,
        rollouts=1,
        learning_epochs=1,
        mini_batches=1,
        output_dir=cfg.output_dir / "policy_runtime",
        pointnet_enabled=cfg.pointnet_enabled,
        film_enabled=cfg.film_enabled,
        separate=cfg.separate,
    )
    agent, model_config = _build_agent(spaces, launcher)
    sidecar = None
    if cfg.mode == "trained":
        assert cfg.checkpoint is not None
        sidecar = _policy_checkpoint_sidecar(cfg.checkpoint, training_reward)
        agent.load(str(cfg.checkpoint))
    agent.enable_training_mode(False, apply_to_models=True)
    bridge = DLPackBridge(production_gpu=cfg.device == "gpu")
    return agent, model_config, bridge, sidecar


def _mean_policy_actions(agent: Any, observations: Any) -> Any:
    import torch

    with torch.no_grad():
        normalized = agent._observation_preprocessor(observations)
        means, _ = agent.policy.compute({"observations": normalized}, role="policy")
        return torch.clamp(means, -1.0, 1.0).contiguous()


def _per_world_summary(
    *, world: int, trace: dict[str, np.ndarray], batch: Any,
    thresholds_N: tuple[float, ...], deviation_threshold_m: float,
    early_phase_steps: int,
) -> dict[str, Any]:
    live = trace["active"][:, world].astype(bool)
    if not np.any(live):
        raise RuntimeError(f"world {world} produced no live transition")
    values = {
        name: np.asarray(array)[live, world]
        for name, array in trace.items()
        if name not in {"active", "global_valid", "duplicate_qpos", "duplicate_qvel"}
    }
    frames = values["frame"].astype(np.int64)
    forces = values["contact_magnitudes"]
    expected = np.asarray(batch.task_expected_contact_masks[world], dtype=bool)
    expected_count = max(int(expected.sum()), 1)
    within = (frames >= int(batch.movement_start_local[world])) & (frames <= int(batch.movement_end_local[world]))
    late = frames > int(batch.movement_end_local[world]) + SOURCE_ALIGNED_REWARD_CONFIG.late_contact_grace_frames
    component_sums = {
        name: float(values["components"][:, index].sum())
        for index, name in enumerate(REWARD_COMPONENTS)
    }
    contacts: dict[str, Any] = {}
    for threshold in thresholds_N:
        above = forces > float(threshold)
        expected_fraction = (above & expected[None, :]).sum(axis=1) / expected_count
        contacts[f"{threshold:g}N"] = {
            "threshold_N": float(threshold),
            "any_contact_frames": int(np.count_nonzero(np.any(above, axis=1))),
            "window_frames": int(np.count_nonzero(within)),
            "window_any_contact_frames": int(np.count_nonzero(within & np.any(above, axis=1))),
            "window_expected_contact_fraction_mean": (
                float(expected_fraction[within].mean()) if np.any(within) else None
            ),
            "late_contact_frames": int(np.count_nonzero(late & np.any(above, axis=1))),
        }
    deviation = (values["object_position_error"] > deviation_threshold_m) & (frames >= early_phase_steps)
    first_deviation = np.flatnonzero(deviation)
    annotation = np.asarray(batch.lance_contact_annotation[world, : int(batch.lengths[world])])
    per_step_finite = np.ones(len(frames), dtype=bool)
    for name in (
        "reward", "components", "object_position_error", "object_rotation_error_deg",
        "object_speed", "object_angular_speed", "object_position", "target_object_position",
        "object_height_delta", "target_height_delta", "hand_active_l2_error",
        "finger_active_mean_abs_error", "finger_active_max_abs_error", "contact_magnitudes",
        "fingertip_penetration_max_depth_m", "fingertip_penetration_mean_depth_m",
    ):
        array = np.asarray(values[name])
        per_step_finite &= np.all(np.isfinite(array.reshape(len(array), -1)), axis=1)
    return {
        "env_id": world,
        "is_primary": world < PRIMARY_WORLD_COUNT,
        "duplicate_of": 0 if world == 7 else None,
        "object_name": "cube1",
        "action": str(batch.actions[world]),
        "row_id": int(batch.row_ids[world]),
        "uuid": str(batch.uuids[world]),
        "seed_uuid": str(batch.seed_uuids[world]) if batch.seed_uuids else None,
        "start_frame": int(batch.start_frames[world]),
        "movement_start_local": int(batch.movement_start_local[world]),
        "movement_end_local": int(batch.movement_end_local[world]),
        "packed_length": int(batch.lengths[world]),
        "recorded_transitions": int(live.sum()),
        "first_recorded_frame": int(frames[0]),
        "final_recorded_frame": int(frames[-1]),
        "completed_full_horizon": bool(np.any(values["truncated"])),
        "raw_environment_return": float(values["reward"].sum()),
        "ppo_scaled_return": float(values["reward"].sum() * PPO_REWARD_SCALE),
        "raw_environment_reward": _finite_stats(values["reward"]),
        "task_reward": _finite_stats(values["task_reward"]),
        "embodiment_reward": _finite_stats(values["embodiment_reward"]),
        "component_sums": component_sums,
        "object_position_error_m": _finite_stats(values["object_position_error"]),
        "object_rotation_error_deg": _finite_stats(values["object_rotation_error_deg"]),
        "object_speed_m_s": _finite_stats(values["object_speed"]),
        "object_angular_speed_rad_s": _finite_stats(values["object_angular_speed"]),
        "object_lift_m": _finite_stats(values["object_height_delta"]),
        "target_lift_m": _finite_stats(values["target_height_delta"]),
        "lift_tracking_error_m": _finite_stats(values["object_height_delta"] - values["target_height_delta"]),
        "hand_active_l2_error": _finite_stats(values["hand_active_l2_error"]),
        "finger_active_mean_abs_error_rad": _finite_stats(values["finger_active_mean_abs_error"]),
        "finger_active_max_abs_error_rad": _finite_stats(values["finger_active_max_abs_error"]),
        "fingertip_penetration_penalty": _finite_stats(values["fingertip_penetration"]),
        "fingertip_penetration_max_depth_m": _finite_stats(values["fingertip_penetration_max_depth_m"]),
        "fingertip_penetration_mean_depth_m": _finite_stats(values["fingertip_penetration_mean_depth_m"]),
        "contact": contacts,
        "lance_contact_annotation_nonzero_frames": int(np.count_nonzero(np.any(annotation > 0.5, axis=1))),
        "deviation_10cm": {
            "threshold_m": float(deviation_threshold_m),
            "frames": int(np.count_nonzero(deviation)),
            "first_local_frame": int(frames[first_deviation[0]]) if first_deviation.size else None,
        },
        "termination_reason_final": int(values["reason"][-1]),
        "finite_transition_count": int(np.count_nonzero(per_step_finite)),
        "all_transitions_finite": bool(np.all(per_step_finite)),
    }


def _mean_or_none(values: Iterable[Any]) -> float | None:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    return float(finite.mean()) if finite.size else None


def _aggregate_worlds(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "world_count": len(worlds),
        "raw_environment_return_mean": _mean_or_none(world["raw_environment_return"] for world in worlds),
        "ppo_scaled_return_mean": _mean_or_none(world["ppo_scaled_return"] for world in worlds),
        "position_error_mean_m": _mean_or_none(world["object_position_error_m"]["mean"] for world in worlds),
        "position_error_max_m_mean": _mean_or_none(world["object_position_error_m"]["max"] for world in worlds),
        "rotation_error_mean_deg": _mean_or_none(world["object_rotation_error_deg"]["mean"] for world in worlds),
        "max_object_speed_m_s_mean": _mean_or_none(world["object_speed_m_s"]["max"] for world in worlds),
        "max_object_angular_speed_rad_s_mean": _mean_or_none(world["object_angular_speed_rad_s"]["max"] for world in worlds),
        "max_fingertip_penetration_depth_m": max(
            (float(world["fingertip_penetration_max_depth_m"]["max"] or 0.0) for world in worlds),
            default=0.0,
        ),
        "all_full_horizon": bool(all(world["completed_full_horizon"] for world in worlds)),
        "all_finite": bool(all(world["all_transitions_finite"] for world in worlds)),
        "deviation_failure_worlds": int(sum(world["deviation_10cm"]["first_local_frame"] is not None for world in worlds)),
    }


def _determinism_summary(trace: dict[str, np.ndarray], duplicate_of: int = 0, duplicate_world: int = 7) -> dict[str, Any]:
    live = trace["active"][:, duplicate_of].astype(bool) & trace["active"][:, duplicate_world].astype(bool)
    if not np.any(live):
        return {"compared_transitions": 0, "pass_1e_6": False}

    def maximum(name: str) -> float:
        lhs = np.asarray(trace[name])[live, duplicate_of]
        rhs = np.asarray(trace[name])[live, duplicate_world]
        return float(np.max(np.abs(lhs - rhs), initial=0.0))

    qpos = np.asarray(trace["duplicate_qpos"])[live]
    qvel = np.asarray(trace["duplicate_qvel"])[live]
    values = {
        "action_max_abs_diff": maximum("actions"),
        "reward_max_abs_diff": maximum("reward"),
        "object_position_max_abs_diff_m": maximum("object_position"),
        "object_rotation_error_max_abs_diff_deg": maximum("object_rotation_error_deg"),
        "contact_force_max_abs_diff_N": maximum("contact_magnitudes"),
        "qpos_max_abs_diff": float(np.max(np.abs(qpos[:, 0] - qpos[:, 1]), initial=0.0)),
        "qvel_max_abs_diff": float(np.max(np.abs(qvel[:, 0] - qvel[:, 1]), initial=0.0)),
    }
    return {
        "duplicate_of": duplicate_of,
        "duplicate_world": duplicate_world,
        "compared_transitions": int(np.count_nonzero(live)),
        **values,
        "bitwise_qpos_equal": bool(np.array_equal(qpos[:, 0], qpos[:, 1])),
        "bitwise_qvel_equal": bool(np.array_equal(qvel[:, 0], qvel[:, 1])),
        "pass_1e_6": bool(max(values.values()) <= 1e-6),
    }


def evaluate_n8(cfg: N8EvaluationConfig) -> N8EvaluationResult:
    rows = list_trajectories(cfg.lance_path, limit=None)
    assignments = resolve_balanced_action_assignments(
        rows,
        object_name="cube1",
        num_envs=cfg.num_envs,
        sequence_index=cfg.sequence_index,
        duplicate_first_for_extra=True,
    )
    root = Path(cfg.isaac_source_root)
    batch = load_trajectory_batch(
        cfg.lance_path,
        assignments,
        pre_contact_frames=cfg.pre_contact_frames,
        post_contact_frames=cfg.post_contact_frames,
        contact_bodies_file=contact_bodies_path(root),
        task_config_path=task_config_path(root),
    )
    training_reward = replace(
        SOURCE_ALIGNED_REWARD_CONFIG,
        contact_force_threshold_N=cfg.reward_contact_threshold_N,
        max_deviation_distance_m=cfg.deviation_threshold_m,
    )
    # Full-horizon physical traces are more informative than reset-at-first-
    # deviation traces. Deviation is recomputed below at the production 0.10 m
    # threshold. Reward values remain unchanged because the configured one-shot
    # deviation penalty is zero.
    evaluation_reward = replace(training_reward, max_deviation_distance_m=1_000_000.0)
    use_residual = cfg.mode != "reference"
    base_physics = DexHandPhysicsConfig()
    physics = replace(
        base_physics,
        material=replace(base_physics.material, hand_friction=cfg.hand_friction),
    )
    env = DexHandRLMJXBatchEnv(DexHandRLBatchConfig(
        trajectories=batch,
        action="all",
        device=cfg.device,
        object_name="cube1",
        physics=physics,
        settle_frames=cfg.settle_frames,
        use_residual=use_residual,
        reward=evaluation_reward,
        isaac_source_root=root,
    ))
    renderer = None
    try:
        from sim.dexhandrl.skrl_train import _BatchSpaces

        spaces = _BatchSpaces(env.num_envs)
        agent, model_config, bridge, checkpoint_sidecar = _build_policy(cfg, spaces, training_reward)
        reset = env.reset_device()
        torch_observations = bridge.jax_to_torch(reset.observation) if bridge is not None else None
        zero_actions = env.jp.zeros((env.num_envs, ACTION_DIM_021PRO), dtype=env.jp.float32)
        if not cfg.headless:
            from sim.dexhandrl.view_environment import TiledDexHandRenderer

            renderer = TiledDexHandRenderer(env, tile_envs=cfg.tile_envs)
            if not renderer.render(env.render_snapshot()):
                raise RuntimeError("viewer closed before the first evaluation transition")

        trace_lists: dict[str, list[np.ndarray]] = {name: [] for name in (
            "active", "global_valid", "frame", "actions", "reward", "task_reward",
            "embodiment_reward", "components", "terminated", "truncated", "reason",
            "object_position_error", "object_rotation_error_deg", "object_speed",
            "object_angular_speed", "object_position", "target_object_position",
            "object_height_delta", "target_height_delta", "hand_active_l2_error",
            "finger_active_mean_abs_error", "finger_active_max_abs_error",
            "contact_magnitudes", "raw_contact", "contact_gate", "fingertip_penetration",
            "fingertip_penetration_max_depth_m", "fingertip_penetration_mean_depth_m",
            "duplicate_qpos", "duplicate_qvel",
        )}
        capacity_samples: list[dict[str, Any]] = []
        active = np.ones(env.num_envs, dtype=bool)
        steps = 0
        viewer_closed = False
        started = time.perf_counter()
        while np.any(active):
            if cfg.max_steps is not None and steps >= cfg.max_steps:
                break
            step_started = time.perf_counter()
            if agent is None:
                actions_device = zero_actions
                actions_host = np.zeros((env.num_envs, ACTION_DIM_021PRO), dtype=np.float32)
            else:
                assert bridge is not None and torch_observations is not None
                actions_torch = _mean_policy_actions(agent, torch_observations)
                actions_host = actions_torch.detach().cpu().numpy().astype(np.float32, copy=True)
                actions_device = bridge.torch_to_jax(actions_torch)
            transition = env.step_device(actions_device)
            if bridge is not None:
                torch_observations = bridge.jax_to_torch(transition.observation)
            telemetry = env.telemetry()
            capacity_samples.append(telemetry)
            frame = np.asarray(env.state.frame).copy()
            transition_values = {
                name: np.asarray(getattr(transition, name)).copy()
                for name in (
                    "reward", "task_reward", "embodiment_reward", "components", "terminated",
                    "truncated", "reason", "object_position_error", "object_rotation_error_deg",
                    "object_speed", "object_angular_speed", "object_position",
                    "target_object_position", "object_height_delta", "target_height_delta",
                    "hand_active_l2_error", "finger_active_mean_abs_error",
                    "finger_active_max_abs_error", "contact_magnitudes", "raw_contact",
                    "contact_gate", "fingertip_penetration",
                    "fingertip_penetration_max_depth_m", "fingertip_penetration_mean_depth_m",
                )
            }
            trace_lists["active"].append(active.copy())
            trace_lists["global_valid"].append(np.asarray(transition.valid).reshape(()).copy())
            trace_lists["frame"].append(frame)
            trace_lists["actions"].append(actions_host)
            for name, value in transition_values.items():
                trace_lists[name].append(value)
            trace_lists["duplicate_qpos"].append(np.asarray(telemetry["qpos"])[[0, 7]].copy())
            trace_lists["duplicate_qvel"].append(np.asarray(telemetry["qvel"])[[0, 7]].copy())
            ended = np.asarray(transition.terminated) | np.asarray(transition.truncated)
            active &= ~ended
            steps += 1
            if renderer is not None and steps % cfg.render_stride == 0:
                if not renderer.render(env.render_snapshot()):
                    viewer_closed = True
                    break
                remaining = env.cfg.physics.control_dt / cfg.speed - (time.perf_counter() - step_started)
                if remaining > 0:
                    time.sleep(remaining)

        if not trace_lists["reward"]:
            raise RuntimeError("N8 evaluation produced no transition")
        trace = {name: np.stack(values) for name, values in trace_lists.items()}
        worlds = [
            _per_world_summary(
                world=world,
                trace=trace,
                batch=batch,
                thresholds_N=cfg.force_thresholds_N,
                deviation_threshold_m=cfg.deviation_threshold_m,
                early_phase_steps=training_reward.early_phase_steps,
            )
            for world in range(env.num_envs)
        ]
        primary = worlds[:PRIMARY_WORLD_COUNT]
        per_action = {world["action"]: _aggregate_worlds([world]) for world in primary}

        frames = trace["frame"]
        live = trace["active"].astype(bool)
        forces = trace["contact_magnitudes"]
        starts = np.asarray(batch.movement_start_local)[None, :]
        ends = np.asarray(batch.movement_end_local)[None, :]
        window = live & (frames >= starts) & (frames <= ends)
        late = live & (frames > ends + training_reward.late_contact_grace_frames)
        expected = np.asarray(batch.task_expected_contact_masks, dtype=bool)[None, :, :]
        all_live_slots = np.broadcast_to(live[:, :, None], forces.shape)
        window_slots = np.broadcast_to(window[:, :, None], forces.shape)
        late_slots = np.broadcast_to(late[:, :, None], forces.shape)
        force_distributions = {
            "all_live_hand_object_slots": _force_distribution(forces[all_live_slots], cfg.force_thresholds_N),
            "movement_window_expected_slots": _force_distribution(forces[window_slots & expected], cfg.force_thresholds_N),
            "movement_window_unexpected_slots": _force_distribution(forces[window_slots & ~expected], cfg.force_thresholds_N),
            "outside_movement_window_slots": _force_distribution(forces[all_live_slots & ~window_slots], cfg.force_thresholds_N),
            "late_slots_after_grace": _force_distribution(forces[late_slots], cfg.force_thresholds_N),
        }
        determinism = _determinism_summary(trace)
        capacity = _capacity_summary(capacity_samples)
        global_valid = bool(np.all(trace["global_valid"]))
        aggregate = _aggregate_worlds(primary)
        gates = {
            "all_transitions_device_valid": global_valid,
            "all_primary_worlds_finite": aggregate["all_finite"],
            "all_primary_worlds_full_horizon": aggregate["all_full_horizon"],
            "capacity_below_80_percent": capacity["pass_below_80_percent"],
            # Contact-rich worlds can bifurcate after tiny world-index/order
            # differences. Preserve this as variance evidence, not as a gate on
            # whether the simulator baseline is interpretable for learning.
            "duplicate_determinism_1e_6": determinism["pass_1e_6"],
        }
        simulator_gate_names = (
            "all_transitions_device_valid",
            "all_primary_worlds_finite",
            "all_primary_worlds_full_horizon",
            "capacity_below_80_percent",
        )
        gates["simulator_contract_ready"] = bool(
            all(gates[name] for name in simulator_gate_names) and not viewer_closed
        )
        gates["physical_interpretation_ready"] = gates["simulator_contract_ready"]
        assignment_manifest = [
            {
                "env_id": env_id,
                "object_name": spec["object_name"],
                "action": spec["action"],
                "row_id": int(spec["row_id"]),
                "uuid": str(spec["uuid"]),
                "seed_uuid": spec.get("seed_uuid"),
                "sequence_index": int(spec["sequence_index"]),
                "determinism_duplicate_of": spec.get("determinism_duplicate_of"),
            }
            for env_id, spec in enumerate(assignments)
        ]
        summary = {
            "schema": EVALUATION_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": cfg.mode,
            "policy_semantics": {
                "reference": "zero 22D actions with use_residual=false",
                "untrained": "fixed-seed untrained policy mean actions with use_residual=true",
                "trained": "checkpoint policy mean actions with use_residual=true",
            }[cfg.mode],
            "device": cfg.device,
            "runtime_backend": "mjx-warp",
            "num_envs": cfg.num_envs,
            "elapsed_seconds": float(time.perf_counter() - started),
            "recorded_batch_steps": int(steps),
            "max_steps": cfg.max_steps,
            "viewer_closed": viewer_closed,
            "headless": cfg.headless,
            "code": code_provenance(),
            "dataset": lance_dataset_metadata(cfg.lance_path),
            "assets": asset_provenance("cube1", root),
            "assignments": assignment_manifest,
            "trajectory_window_contract": {
                "pre_contact_frames": cfg.pre_contact_frames,
                "post_contact_frames": cfg.post_contact_frames,
                "reward_window_source": "trajectory_metadata.trajectory_info.object_move",
                "warning": "retarget Lance contact annotation is retained as evidence and is not used to synthesize the task mask/window",
            },
            "training_reward_contract": training_reward.record(),
            "evaluation_reward_contract": evaluation_reward.record(),
            "evaluation_termination_semantics": {
                "full_horizon_max_deviation_distance_m": evaluation_reward.max_deviation_distance_m,
                "counterfactual_production_deviation_threshold_m": cfg.deviation_threshold_m,
            },
            "physics_contract": env.cfg.physics.record(),
            "abi": {"observation_dim": env.num_observations, "policy_action_dim": env.num_actions, "target_routing": "22D active to state-dependent 26D full target"},
            "model_config": asdict(model_config) if model_config is not None else None,
            "checkpoint": str(cfg.checkpoint) if cfg.checkpoint is not None else None,
            "checkpoint_sidecar": checkpoint_sidecar,
            "force_threshold_status": training_reward.record()["threshold_status"],
            "force_distributions_N": force_distributions,
            "capacity": capacity,
            "determinism": determinism,
            "gates": gates,
            "aggregate_primary_worlds": aggregate,
            "per_action_primary_worlds": per_action,
            "worlds": worlds,
        }
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cfg.output_dir / f"{cfg.mode}_summary.json"
        trace_path = cfg.output_dir / f"{cfg.mode}_trace.npz"
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
        np.savez_compressed(trace_path, **trace)
        return N8EvaluationResult(summary_path=summary_path, trace_path=trace_path, summary=summary)
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(part) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("force thresholds must be comma-separated numbers") from error
    if not thresholds:
        raise argparse.ArgumentTypeError("at least one force threshold is required")
    return thresholds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "untrained", "trained"), default="reference")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--lance-path", type=Path, default=DEFAULT_LANCE_PATH)
    parser.add_argument("--isaac-source-root", type=Path, default=DEFAULT_ISAAC_SOURCE_ROOT)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--pre-contact-frames", type=int, default=200)
    parser.add_argument("--post-contact-frames", type=int, default=250)
    parser.add_argument("--settle-frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "dexhandrl_n8_evaluation")
    parser.add_argument("--force-thresholds", type=_parse_thresholds, default=DEFAULT_FORCE_THRESHOLDS_N)
    parser.add_argument("--reward-contact-threshold", type=float, default=0.2)
    parser.add_argument("--deviation-threshold", type=float, default=0.10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-envs", type=int, default=8)
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pointnet-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--film-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--separate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hand-friction", default="0.5,0.01,0.001", help="sliding,torsional,rolling friction")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        hand_friction = tuple(float(value) for value in args.hand_friction.replace(" ", ",").split(",") if value)
    except ValueError as error:
        raise SystemExit("--hand-friction must contain three numbers") from error
    cfg = N8EvaluationConfig(
        mode=args.mode,
        device=args.device,
        num_envs=args.num_envs,
        lance_path=args.lance_path,
        isaac_source_root=args.isaac_source_root,
        sequence_index=args.sequence_index,
        pre_contact_frames=args.pre_contact_frames,
        post_contact_frames=args.post_contact_frames,
        settle_frames=args.settle_frames,
        seed=args.seed,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        force_thresholds_N=args.force_thresholds,
        reward_contact_threshold_N=args.reward_contact_threshold,
        deviation_threshold_m=args.deviation_threshold,
        max_steps=args.max_steps,
        headless=args.headless,
        tile_envs=args.tile_envs,
        render_stride=args.render_stride,
        speed=args.speed,
        pointnet_enabled=args.pointnet_enabled,
        film_enabled=args.film_enabled,
        separate=args.separate,
        hand_friction=hand_friction,
    )
    if not cfg.headless:
        if not os.environ.get("DISPLAY"):
            raise SystemExit("visible N8 evaluation requires DISPLAY (use export DISPLAY=:1)")
    result = evaluate_n8(cfg)
    print(f"N8 {cfg.mode} summary: {result.summary_path}", flush=True)
    print(f"N8 {cfg.mode} trace: {result.trace_path}", flush=True)
    print(json.dumps(_jsonable({
        "gates": result.summary["gates"],
        "aggregate_primary_worlds": result.summary["aggregate_primary_worlds"],
        "determinism": result.summary["determinism"],
        "capacity": result.summary["capacity"],
    }), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
