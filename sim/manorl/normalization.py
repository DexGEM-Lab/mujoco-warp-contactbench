"""Source-equivalent PointNet-safe running observation normalization."""

from __future__ import annotations

import torch
from torch import nn

from sim.manorl.observations import OBSERVATION_SLICES


class PointCloudAwareRunningStandardScaler(nn.Module):
    """Normalize raw observations while sharing XYZ statistics across points.

    rl-games' source normalizer carries ordinary per-observation statistics and
    separate shared XYZ point-cloud statistics. Repeating the three shared
    moments across the 64 point slots preserves PointNet permutation invariance.
    """

    def __init__(self, *, size: int = 476, epsilon: float = 1e-8, clip_threshold: float = 5.0, device: str = "cpu"):
        super().__init__()
        pc_slice = OBSERVATION_SLICES["object_point_cloud_raw"]
        if size != 476 or (pc_slice.stop - pc_slice.start) != 64 * 3:
            raise ValueError("the source PointNet scaler requires the 476D / 64x3 ABI")
        self.size = size
        self.pc_start = pc_slice.start
        self.pc_end = pc_slice.stop
        self.num_points = 64
        self.epsilon = float(epsilon)
        self.clip_threshold = float(clip_threshold)
        self.register_buffer("running_mean", torch.zeros(size, dtype=torch.float64, device=device))
        self.register_buffer("running_var", torch.ones(size, dtype=torch.float64, device=device))
        self.register_buffer("count", torch.ones((), dtype=torch.float64, device=device))
        self.register_buffer("pc_running_mean", torch.zeros(3, dtype=torch.float64, device=device))
        self.register_buffer("pc_running_var", torch.ones(3, dtype=torch.float64, device=device))
        self.register_buffer("pc_count", torch.ones((), dtype=torch.float64, device=device))

    @staticmethod
    def _moments(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("normalizer expects a non-empty rank-2 batch")
        count = torch.as_tensor(len(values), dtype=torch.float64, device=values.device)
        mean = values.mean(dim=0, dtype=torch.float64)
        variance = values.var(dim=0, unbiased=True) if len(values) > 1 else torch.zeros_like(mean)
        return mean, variance, count

    @staticmethod
    def _merge(
        mean: torch.Tensor, variance: torch.Tensor, count: torch.Tensor,
        batch_mean: torch.Tensor, batch_variance: torch.Tensor, batch_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = batch_mean - mean
        total = count + batch_count
        m2 = variance * count + batch_variance * batch_count + delta.square() * count * batch_count / total
        return mean + delta * batch_count / total, m2 / total, total

    @torch.no_grad()
    def update(self, observations: torch.Tensor) -> None:
        values = observations.detach().to(device=self.running_mean.device, dtype=torch.float64)
        if values.ndim == 3:
            values = values.flatten(0, 1)
        if values.ndim != 2 or values.shape[1:] != (self.size,) or not torch.isfinite(values).all():
            raise ValueError(f"normalizer requires finite (*, {self.size}) observations")
        mean, variance, count = self._moments(values)
        self.running_mean, self.running_var, self.count = self._merge(
            self.running_mean, self.running_var, self.count, mean, variance, count
        )
        points = values[:, self.pc_start:self.pc_end].reshape(-1, 3)
        pc_mean, pc_variance, pc_count = self._moments(points)
        self.pc_running_mean, self.pc_running_var, self.pc_count = self._merge(
            self.pc_running_mean, self.pc_running_var, self.pc_count, pc_mean, pc_variance, pc_count
        )
        self.running_mean[self.pc_start:self.pc_end] = self.pc_running_mean.repeat(self.num_points)
        self.running_var[self.pc_start:self.pc_end] = self.pc_running_var.repeat(self.num_points)

    def forward(
        self, observations: torch.Tensor | None, *, train: bool = False, inverse: bool = False, no_grad: bool = True
    ) -> torch.Tensor | None:
        if observations is None:
            return None
        if train:
            self.update(observations)
        def apply() -> torch.Tensor:
            mean, variance = self.running_mean.float(), self.running_var.float()
            if inverse:
                return torch.sqrt(variance + self.epsilon) * torch.clamp(
                    observations, -self.clip_threshold, self.clip_threshold
                ) + mean
            return torch.clamp(
                (observations - mean) / torch.sqrt(variance + self.epsilon),
                -self.clip_threshold, self.clip_threshold,
            )
        if no_grad:
            with torch.no_grad():
                return apply()
        return apply()
