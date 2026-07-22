"""Source-authoritative ManoRL PointNet actor-critic for skrl.

The production model is the validated PointNet + FiLM policy.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from sim.manorl.gymnasium_env import ACTION_DIM, OBSERVATION_DIM
from sim.manorl.observations import ObservationLayout, observation_layout_for_dimension

POINT_COUNT = 64
POINT_FEATURE_DIM = 64
CONDITION_DIM = 62
CONDITION_EMBED_DIM = 32
BASE_FEATURE_DIM = 286
HIDDEN_UNITS = (512, 512, 256, 128)
LOG_STD_LIMITS = (-10.0, 2.0)


def _layout_for_observation_dim(observation_dim: int) -> ObservationLayout:
    """Resolve a live Gym space to the shared observation layout contract."""

    return observation_layout_for_dimension(int(observation_dim))


class PointNetEncoder(nn.Module):
    """Source PointNet: shared 3→64→128→256 MLP, max pool, 256→64 MLP."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(),
        ).to(device)
        self.global_mlp = nn.Sequential(
            nn.Linear(256, POINT_FEATURE_DIM), nn.LayerNorm(POINT_FEATURE_DIM), nn.ReLU()
        ).to(device)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[1:] != (POINT_COUNT, 3):
            raise ValueError("PointNet requires (batch, 64, 3) point clouds")
        features = self.point_mlp(points.reshape(-1, 3)).reshape(len(points), POINT_COUNT, 256)
        return self.global_mlp(features.max(dim=1).values)


class FiLMLayer(nn.Module):
    """Source identity-centered feature modulation namespace."""

    def __init__(self, feature_dim: int, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.film_generator = nn.Linear(CONDITION_EMBED_DIM, 2 * feature_dim).to(device)
        nn.init.normal_(self.film_generator.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.film_generator.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        delta_gamma, beta = self.film_generator(condition).chunk(2, dim=-1)
        return (1.0 + delta_gamma) * features + beta


class FiLMBlock(nn.Module):
    """Source identity-centered FiLM: Linear → (1 + gamma) * x + beta → ELU."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim).to(device)
        self.film = FiLMLayer(output_dim, device)
        self.activation = nn.ELU().to(device)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.activation(self.film(self.linear(features), condition))


class ManoActorCritic(GaussianMixin, DeterministicMixin, Model):
    """One shared skrl module exposing Gaussian policy and deterministic value.

    State-dict names intentionally describe the target architecture rather than
    masquerading as rl-games checkpoint keys. ``state_dict_manifest`` supplies
    the explicit namespace/shape contract used by the Gym checkpoint converter.
    """

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device: str | torch.device = "cpu",
        use_film: bool = True,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device)
        self.observation_dim = int(self.num_observations)
        self.action_dim = int(self.num_actions)
        self.observation_layout = _layout_for_observation_dim(self.observation_dim)
        expected_action_dim = (
            self.observation_layout.dof_dim * self.observation_layout.hand_count
        )
        if self.action_dim != expected_action_dim:
            raise ValueError(
                "ManoRL action and observation layouts do not match: "
                f"observation width {self.observation_dim} requires "
                f"{expected_action_dim} actions, got {self.action_dim}"
            )
        self.base_feature_dim = (
            self.observation_layout.slices["object_point_cloud_raw"].start
            + POINT_FEATURE_DIM
            + 48
            + 16
            + 48
            + 16
            + self.observation_layout.cumulative_joint_dim
        )
        GaussianMixin.__init__(
            self, clip_actions=False, clip_mean_actions=False, clip_log_std=True,
            min_log_std=LOG_STD_LIMITS[0], max_log_std=LOG_STD_LIMITS[1], reduction="sum", role="policy",
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")
        self.use_film = bool(use_film)
        self.pointnet = PointNetEncoder(device)
        self.condition_encoder = nn.Sequential(
            nn.Linear(CONDITION_DIM, CONDITION_EMBED_DIM), nn.ReLU()
        ).to(device)
        if self.use_film:
            self.actor_backbone = nn.ModuleList(
                [
                    FiLMBlock(self.base_feature_dim, HIDDEN_UNITS[0], device),
                    nn.Linear(HIDDEN_UNITS[0], HIDDEN_UNITS[1]).to(device),
                    nn.ELU(),
                    nn.Linear(HIDDEN_UNITS[1], HIDDEN_UNITS[2]).to(device),
                    nn.ELU(),
                    nn.Linear(HIDDEN_UNITS[2], HIDDEN_UNITS[3]).to(device),
                    nn.ELU(),
                ]
            )
            actor_input_dim = self.base_feature_dim
        else:
            # Single-task mode keeps the source condition features as ordinary
            # inputs while removing FiLM modulation. This preserves action and
            # geometry information for an eventual multi-task run.
            actor_input_dim = self.base_feature_dim + CONDITION_EMBED_DIM
            self.actor_backbone = nn.ModuleList(
                [
                    nn.Linear(actor_input_dim, HIDDEN_UNITS[0]).to(device),
                    nn.ELU(),
                    nn.Linear(HIDDEN_UNITS[0], HIDDEN_UNITS[1]).to(device),
                    nn.ELU(),
                    nn.Linear(HIDDEN_UNITS[1], HIDDEN_UNITS[2]).to(device),
                    nn.ELU(),
                    nn.Linear(HIDDEN_UNITS[2], HIDDEN_UNITS[3]).to(device),
                    nn.ELU(),
                ]
            )
        critic_input_dim = self.base_feature_dim + CONDITION_EMBED_DIM
        self.actor_head = nn.Linear(HIDDEN_UNITS[-1], self.action_dim).to(device)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), -0.99, dtype=torch.float32, device=device)
        )
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, HIDDEN_UNITS[0]), nn.ELU(),
            nn.Linear(HIDDEN_UNITS[0], HIDDEN_UNITS[1]), nn.ELU(),
            nn.Linear(HIDDEN_UNITS[1], HIDDEN_UNITS[2]), nn.ELU(),
            nn.Linear(HIDDEN_UNITS[2], HIDDEN_UNITS[3]), nn.ELU(),
            nn.Linear(HIDDEN_UNITS[3], 1),
        )
        self.critic = self.critic.to(device)
        self._initialize_source_weights()

    def _slice(self, observations: torch.Tensor, name: str) -> torch.Tensor:
        section = self.observation_layout.slices[name]
        return observations[:, section]

    def _initialize_source_weights(self) -> None:
        """Match source init: orthogonal MLPs, untouched PointNet/condition/FiLM."""
        skipped = set(self.pointnet.modules())
        skipped.update(self.condition_encoder.modules())
        for block in self.actor_backbone:
            if isinstance(block, FiLMBlock):
                skipped.update(block.film.modules())
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in skipped:
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _features(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 2 or observations.shape[1:] != (self.observation_dim,):
            raise ValueError(
                f"ManoActorCritic requires (batch, {self.observation_dim}) observations"
            )
        point_cloud = self._slice(observations, "object_point_cloud_raw").reshape(-1, POINT_COUNT, 3)
        point_features = self.pointnet(point_cloud)
        hand = torch.cat([
            self._slice(observations, "hand_keypoints"),
            self._slice(observations, "contact_forces"),
            self._slice(observations, "contact_force_directions"),
            self._slice(observations, "expected_contact_mask"),
        ], dim=-1)
        base = torch.cat([
            observations[:, : self.observation_layout.slices["object_point_cloud_raw"].start],
            point_features,
            hand,
            self._slice(observations, "cumulative_joint_offset"),
        ], dim=-1)
        condition_input = torch.cat([
            self._slice(observations, "action_types"), self._slice(observations, "object_geometry")
        ], dim=-1)
        condition = self.condition_encoder(condition_input)
        if base.shape[1] != self.base_feature_dim:
            raise RuntimeError(
                f"source feature split changed: expected {self.base_feature_dim}, got {base.shape[1]}"
            )
        return base, condition

    def compute(self, inputs: Mapping[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observations = inputs["observations"]
        base, condition = self._features(observations)
        if role == "policy":
            if self.use_film:
                features = self.actor_backbone[0](base, condition)
                for layer in self.actor_backbone[1:]:
                    features = layer(features)
            else:
                features = torch.cat([base, condition], dim=-1)
                for layer in self.actor_backbone:
                    features = layer(features)
            mean = self.actor_head(features)
            return mean, {"log_std": self.log_std.expand_as(mean)}
        if role == "value":
            return self.critic(torch.cat([base, condition], dim=-1)), {}
        raise ValueError("role must be 'policy' or 'value'")

    def act(self, inputs: dict[str, torch.Tensor], *, role: str = "") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        if role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
        raise ValueError("role must be 'policy' or 'value'")

    def state_dict_manifest(self) -> dict[str, tuple[int, ...]]:
        """Exact target state-dict ABI, intended for future explicit conversion."""
        return {name: tuple(parameter.shape) for name, parameter in self.state_dict().items()}

    @classmethod
    def expected_state_dict_manifest(cls) -> dict[str, tuple[int, ...]]:
        # Uses canonical spaces only to make the expected namespace executable.
        import gymnasium as gym
        model = cls(
            gym.spaces.Box(-5.0, 5.0, shape=(OBSERVATION_DIM,), dtype=float),
            None,
            gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=float),
        )
        return model.state_dict_manifest()
