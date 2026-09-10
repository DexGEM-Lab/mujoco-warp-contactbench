"""Device-side update telemetry for the v4 autonomous PPO route.

The runtime owns physical truth.  This reducer accepts compact per-transition
Torch tensors captured immediately after ``runtime.step`` and preserves episode
state through PPO rollout cuts.  It intentionally has no reward/controller or
physics side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

REWARD_NAMES = (
    "object_position", "object_rotation", "object_velocity", "hand_relative",
    "fingers", "geometry", "action", "survival", "severe",
)
REASON_BITS = {"reference_complete": 1, "deviation": 2, "fallen": 4, "nonfinite": 8}


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


@dataclass
class V4TelemetryAccumulator:
    """Tensor-only episode/update accounting for fixed-size batched PPO."""
    num_envs: int
    device: torch.device

    def __post_init__(self) -> None:
        self.episode_returns = torch.zeros((self.num_envs,), device=self.device)
        self.episode_lengths = torch.zeros((self.num_envs,), device=self.device)
        self.episode_terms = torch.zeros((self.num_envs, len(REWARD_NAMES)), device=self.device)
        self.episode_max_clearance = torch.full((self.num_envs,), -torch.inf, device=self.device)
        self.episode_loaded_frames = torch.zeros((self.num_envs,), device=self.device)
        self._reset_window()

    def _reset_window(self) -> None:
        self.frames = torch.zeros((), device=self.device)
        self.reward_sum = torch.zeros((len(REWARD_NAMES) + 1,), device=self.device)
        self.reward_min = torch.full((len(REWARD_NAMES) + 1,), torch.inf, device=self.device)
        self.reward_max = torch.full((len(REWARD_NAMES) + 1,), -torch.inf, device=self.device)
        self.reason_counts = torch.zeros((len(REASON_BITS),), device=self.device)
        self.horizon_only = torch.zeros((), device=self.device)
        self.completed_returns: list[torch.Tensor] = []
        self.completed_lengths: list[torch.Tensor] = []
        self.completed_terms: list[torch.Tensor] = []
        self.completed_clearance: list[torch.Tensor] = []
        self.completed_loaded_frames: list[torch.Tensor] = []
        self.final_progress: list[torch.Tensor] = []
        self.sums: dict[str, torch.Tensor] = {}
        self.maxes: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}

    def _sum(self, name: str, value: torch.Tensor) -> None:
        self.sums[name] = self.sums.get(name, torch.zeros((), device=self.device)) + value.sum()

    def _count(self, name: str, value: torch.Tensor) -> None:
        self.counts[name] = self.counts.get(name, torch.zeros((), device=self.device)) + value.sum()

    def _max(self, name: str, value: torch.Tensor) -> None:
        candidate = value.max()
        self.maxes[name] = candidate if name not in self.maxes else torch.maximum(self.maxes[name], candidate)

    def add(self, sample: Mapping[str, torch.Tensor]) -> None:
        """Add one post-transition B-row snapshot before any reset."""
        terms = sample["reward_terms"]
        total = sample["reward_total"]
        if terms.shape != (self.num_envs, len(REWARD_NAMES)) or total.shape != (self.num_envs,):
            raise ValueError("v4 telemetry reward shapes are incompatible with the batch")
        frames = torch.as_tensor(float(self.num_envs), device=self.device)
        self.frames += frames
        all_rewards = torch.cat((terms, total[:, None]), dim=1)
        self.reward_sum += all_rewards.sum(0)
        self.reward_min = torch.minimum(self.reward_min, all_rewards.min(0).values)
        self.reward_max = torch.maximum(self.reward_max, all_rewards.max(0).values)
        self.episode_returns += total
        self.episode_lengths += 1
        self.episode_terms += terms
        valid = sample["valid"].to(torch.float32)
        self._sum("reward_reconciliation_abs", (terms.sum(1) - total).abs() * valid)
        self._sum("reward_reconciliation_denominator", valid)

        # Error fields use their own physical denominators, never reward values.
        position_error = sample["position_error_abs"]
        for axis, values in zip("xyz", position_error.unbind(1)):
            self._sum(f"object_position_error_{axis}_abs", values)
        l2 = torch.linalg.vector_norm(position_error, dim=1)
        self._sum("object_position_error_l2", l2); self._sum("object_position_error_l2_sq", l2.square())
        for name in ("object_rotation_error_rad", "palm_position_error", "finger_raw_error", "finger_feasible_error",
                     "bottom_clearance", "reference_bottom_clearance", "origin_lift_delta"):
            self._sum(name, sample[name])
        self._max("bottom_clearance", sample["bottom_clearance"])
        loaded = sample["paired_loaded"].to(torch.float32)
        self._sum("paired_loaded_regions", loaded); self._count("paired_loaded_region_denominator", torch.ones_like(loaded))
        self._sum("paired_contact_count", sample["paired_contact_count"])
        self._sum("paired_force_norm", sample["paired_force_norm"])
        self._sum("object_all_force_norm", sample["object_all_force_norm"])
        self._sum("paired_torque_com_norm", sample["paired_torque_com_norm"])
        active = sample["paired_active"].to(torch.float32)
        self._sum("active_slip", sample["tangential_slip"] * active)
        self._sum("active_slip_denominator", active)
        airborne = sample["airborne"].to(torch.float32)
        any_loaded = loaded.amax(1)
        self._sum("airborne", airborne); self._sum("loaded_airborne", airborne * any_loaded)
        self._sum("action_raw_abs", sample["action_raw_abs"])
        self._max("action_raw_abs", sample["action_raw_abs"])
        self._sum("action_executed_norm", sample["action_executed_norm"])
        self._sum("action_clipped", sample["action_clipped"].to(torch.float32))
        self._count("action_denominator", torch.ones_like(sample["action_clipped"], dtype=torch.float32))
        self._sum("command_envelope_utilization", sample["command_envelope_utilization"])
        self._sum("antiwindup_active", sample["antiwindup_active"].to(torch.float32))

        reason = sample["reason"].to(torch.int32)
        done = reason != 0
        for i, bit in enumerate(REASON_BITS.values()):
            self.reason_counts[i] += ((reason & bit) != 0).sum()
        pure_horizon = (reason == REASON_BITS["reference_complete"])
        self.horizon_only += pure_horizon.sum()
        loaded_frame = any_loaded
        self.episode_max_clearance = torch.maximum(self.episode_max_clearance, sample["bottom_clearance"])
        self.episode_loaded_frames += loaded_frame
        # Empty device tensors are harmless at reduction time; avoiding a
        # Python bool here keeps CUDA rollout collection asynchronous.
        self.completed_returns.append(self.episode_returns[done])
        self.completed_lengths.append(self.episode_lengths[done])
        self.completed_terms.append(self.episode_terms[done])
        self.completed_clearance.append(self.episode_max_clearance[done])
        self.completed_loaded_frames.append(self.episode_loaded_frames[done])
        self.final_progress.append(sample["reference_progress"][done])
        self.episode_returns = torch.where(done, torch.zeros_like(self.episode_returns), self.episode_returns)
        self.episode_lengths = torch.where(done, torch.zeros_like(self.episode_lengths), self.episode_lengths)
        self.episode_terms = torch.where(done[:, None], torch.zeros_like(self.episode_terms), self.episode_terms)
        self.episode_max_clearance = torch.where(done, torch.full_like(self.episode_max_clearance, -torch.inf), self.episode_max_clearance)
        self.episode_loaded_frames = torch.where(done, torch.zeros_like(self.episode_loaded_frames), self.episode_loaded_frames)

    def reduce(self, *, update: int, transitions: int) -> dict[str, float]:
        if not bool(self.frames > 0):
            raise ValueError("cannot reduce an empty telemetry window")
        out: dict[str, float] = {"update": float(update), "transitions": float(transitions), "telemetry/frames": _scalar(self.frames)}
        for i, name in enumerate((*REWARD_NAMES, "total")):
            out[f"reward/{name}"] = _scalar(self.reward_sum[i] / self.frames)
            out[f"reward/{name}_min"] = _scalar(self.reward_min[i])
            out[f"reward/{name}_max"] = _scalar(self.reward_max[i])
        for axis in "xyz": out[f"physics/object_position_error_{axis}_abs_mean"] = _scalar(self.sums[f"object_position_error_{axis}_abs"] / self.frames)
        out["physics/object_position_error_l2_mean"] = _scalar(self.sums["object_position_error_l2"] / self.frames)
        out["physics/object_position_error_l2_rmse"] = _scalar(torch.sqrt(self.sums["object_position_error_l2_sq"] / self.frames))
        for name in ("object_rotation_error_rad", "palm_position_error", "finger_raw_error", "finger_feasible_error", "bottom_clearance", "reference_bottom_clearance", "origin_lift_delta", "paired_contact_count", "paired_force_norm", "object_all_force_norm", "paired_torque_com_norm"):
            out[f"physics/{name}_mean"] = _scalar(self.sums[name] / self.frames)
        out["action/raw_abs_mean"] = _scalar(self.sums["action_raw_abs"] / self.frames)
        out["action/raw_abs_max"] = _scalar(self.maxes["action_raw_abs"])
        out["action/executed_norm_mean"] = _scalar(self.sums["action_executed_norm"] / self.frames)
        out["action/command_envelope_utilization_mean"] = _scalar(self.sums["command_envelope_utilization"] / self.frames)
        out["reward/component_sum_minus_total_abs_mean"] = _scalar(self.sums["reward_reconciliation_abs"] / torch.clamp_min(self.sums["reward_reconciliation_denominator"], 1.))
        out["reward/component_sum_denominator"] = _scalar(self.sums["reward_reconciliation_denominator"])
        out["physics/bottom_clearance_max"] = _scalar(self.maxes["bottom_clearance"])
        out["physics/paired_loaded_contact_fraction"] = _scalar(self.sums["paired_loaded_regions"] / self.counts["paired_loaded_region_denominator"])
        active = self.sums["active_slip_denominator"]
        if bool(active > 0): out["physics/contact_active_tangential_slip_mean"] = _scalar(self.sums["active_slip"] / active)
        out["physics/airborne_5mm_fraction"] = _scalar(self.sums["airborne"] / self.frames)
        out["physics/loaded_airborne_fraction"] = _scalar(self.sums["loaded_airborne"] / self.frames)
        out["action/raw_clip_fraction"] = _scalar(self.sums["action_clipped"] / self.counts["action_denominator"])
        out["action/antiwindup_fraction"] = _scalar(self.sums["antiwindup_active"] / self.frames)
        completed = torch.cat(self.completed_returns) if self.completed_returns else torch.empty((0,), device=self.device)
        completed_count = int(completed.numel())
        out["episodes/completed_count"] = float(completed_count)
        if completed_count:
            lengths = torch.cat(self.completed_lengths); returns = completed; terms = torch.cat(self.completed_terms)
            out.update({"episodes/return_mean": _scalar(returns.mean()), "episodes/return_min": _scalar(returns.min()), "episodes/return_max": _scalar(returns.max()), "episodes/length_mean": _scalar(lengths.mean()), "episodes/length_min": _scalar(lengths.min()), "episodes/length_max": _scalar(lengths.max()), "episodes/return_denominator": float(completed_count)})
            for i, name in enumerate(REWARD_NAMES):
                values = terms[:, i]
                out[f"reward/episode_{name}_return_mean"] = _scalar(values.mean())
                out[f"reward/episode_{name}_return_min"] = _scalar(values.min())
                out[f"reward/episode_{name}_return_max"] = _scalar(values.max())
            out["episodes/max_clearance_mean"] = _scalar(torch.cat(self.completed_clearance).mean())
            out["episodes/loaded_contact_frames_mean"] = _scalar(torch.cat(self.completed_loaded_frames).mean())
            out["episodes/final_reference_progress_mean"] = _scalar(torch.cat(self.final_progress).mean())
        for i, name in enumerate(REASON_BITS):
            out[f"termination/{name}_count"] = _scalar(self.reason_counts[i])
            if completed_count: out[f"termination/{name}_rate"] = _scalar(self.reason_counts[i] / completed_count)
        out["termination/horizon_only_count"] = _scalar(self.horizon_only)
        if completed_count: out["termination/horizon_only_rate"] = _scalar(self.horizon_only / completed_count)
        self._reset_window()
        return out
