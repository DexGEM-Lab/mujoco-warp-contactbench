"""PointNet encoder for flattened 3D point clouds."""

from __future__ import annotations

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """Encode a fixed-size flattened point cloud into a dense feature vector."""

    def __init__(self, feature_dim: int = 64, num_points: int = 64):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_points = int(num_points)

        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(256, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, point_cloud_flat: torch.Tensor) -> torch.Tensor:
        if point_cloud_flat.ndim != 2:
            raise RuntimeError(f"expected flattened point cloud with rank 2, got {tuple(point_cloud_flat.shape)}")
        expected_dim = self.num_points * 3
        if point_cloud_flat.shape[1] != expected_dim:
            raise RuntimeError(
                f"point cloud dimension mismatch: expected {expected_dim}, got {point_cloud_flat.shape[1]}"
            )

        batch_size = point_cloud_flat.shape[0]
        points = point_cloud_flat.reshape(batch_size, self.num_points, 3)
        point_features = self.point_mlp(points)
        global_features = point_features.max(dim=1)[0]
        return self.global_mlp(global_features)
