"""VoxMani-inspired lightweight token model for ManoRL autonomy v6.

Only the observation encoder and actor-critic network change. The ManoRL v4
environment, reward, PPO implementation, action command and raw-Normal policy
distribution remain authoritative.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from skrl.models.torch import Model
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin

from sim.manorl.autonomy_contracts import (
    ACTION_DIM,
    ACTION_CONTRACT_ID,
    CHECKPOINT_FORMAT_V6,
    CURRENT_TOKENS_V6,
    GOAL_TOKENS_V6,
    MODEL_DIM_V6,
    OBSERVATION_CONTRACT_ID_V6,
    POLICY_SAMPLING_CONTRACT,
    RAW_OBSERVATION_DIM_V6,
    REWARD_CONTRACT_ID,
    raw_observation_slices_v6,
)

ACTOR_CRITIC_ARCHITECTURE_ID_V6 = (
    "manorl.autonomy.actor_critic.v6.region-token-cross-attention"
)


def actor_critic_architecture_v6() -> dict[str, Any]:
    return {
        "id": ACTOR_CRITIC_ARCHITECTURE_ID_V6,
        "raw_observation_dim": RAW_OBSERVATION_DIM_V6,
        "model_dim": MODEL_DIM_V6,
        "current_tokens": CURRENT_TOKENS_V6,
        "goal_tokens": GOAL_TOKENS_V6,
        "object_tokens": "16 patches x 4 points",
        "hand_tokens": "16 regions x 16 points; current/reference share point stem",
        "contact_binding": (
            "per-region confidence, valid, active, count, force xyz, slip xyz"
        ),
        "fusion": "2 x (pre-LN self-attention + cross-attention), 4 heads",
        "actor_critic": "shared observation token encoder; independent fusion towers",
        "action_head": "flat 28D mean; raw Normal then physical clip",
        "action_dim": ACTION_DIM,
    }


class PointStem(nn.Module):
    """Pointwise feature stem shared across actor and critic consumers."""

    def __init__(self, output_dim: int = MODEL_DIM_V6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)


class AttentionBlock(nn.Module):
    """Pre-LN attention and feed-forward residual block."""

    def __init__(
        self,
        *,
        model_dim: int = MODEL_DIM_V6,
        heads: int = 4,
        feedforward_dim: int = 512,
        cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.memory_norm = nn.LayerNorm(model_dim) if cross_attention else None
        self.attention = nn.MultiheadAttention(
            model_dim, heads, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.SiLU(),
            nn.Linear(feedforward_dim, model_dim),
        )

    def forward(
        self, tokens: torch.Tensor, memory: torch.Tensor | None = None
    ) -> torch.Tensor:
        query = self.query_norm(tokens)
        source = (
            query
            if memory is None
            else self.memory_norm(memory)
        )
        attended, _ = self.attention(
            query, source, source, need_weights=False
        )
        tokens = tokens + attended
        return tokens + self.feedforward(self.feedforward_norm(tokens))


class FusionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attention = AttentionBlock()
        self.cross_attention = AttentionBlock(cross_attention=True)

    def forward(
        self, current: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        return self.cross_attention(self.self_attention(current), goal)


class FusionTower(nn.Module):
    """Independent actor or critic token-fusion tower."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(FusionLayer() for _ in range(2))
        self.head = nn.Sequential(
            nn.LayerNorm(MODEL_DIM_V6),
            nn.Linear(MODEL_DIM_V6, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim),
        )

    def forward(
        self, current: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            current = layer(current, goal)
        return self.head(current[:, -1])


class V6ObservationEncoder(nn.Module):
    """Convert the flat ABI into current tokens and reference-hand memory."""

    def __init__(self) -> None:
        super().__init__()
        self.slices = raw_observation_slices_v6()
        self.object_stem = PointStem()
        self.hand_stem = PointStem()
        self.object_patch_embedding = nn.Embedding(16, MODEL_DIM_V6)
        self.hand_region_embedding = nn.Embedding(16, MODEL_DIM_V6)
        self.current_hand_type = nn.Parameter(
            torch.zeros(1, 1, MODEL_DIM_V6)
        )
        self.reference_hand_type = nn.Parameter(
            torch.zeros(1, 1, MODEL_DIM_V6)
        )
        self.region_contact = nn.Sequential(
            nn.Linear(10, MODEL_DIM_V6),
            nn.LayerNorm(MODEL_DIM_V6),
            nn.SiLU(),
        )
        numeric_fields = (
            "autonomous_actual",
            "autonomous_reference",
            "autonomous_future",
            "object_wrench",
            "action_types",
            "object_geometry",
        )
        numeric_width = sum(
            self.slices[name].stop - self.slices[name].start
            for name in numeric_fields
        )
        self.numeric_fields = numeric_fields
        self.state = nn.Sequential(
            nn.Linear(numeric_width, 256),
            nn.SiLU(),
            nn.Linear(256, MODEL_DIM_V6),
        )
        self.readout = nn.Parameter(
            torch.zeros(1, 1, MODEL_DIM_V6)
        )

    def _block(self, observation: torch.Tensor, name: str) -> torch.Tensor:
        return observation[:, self.slices[name]]

    def _hand_tokens(
        self, points: torch.Tensor, *, reference: bool
    ) -> torch.Tensor:
        batch = points.shape[0]
        point_features = self.hand_stem(
            points.reshape(batch, 16, 16, 3)
        )
        pooled = point_features.amax(dim=2)
        region_ids = torch.arange(16, device=points.device)
        token_type = (
            self.reference_hand_type
            if reference
            else self.current_hand_type
        )
        return (
            pooled
            + self.hand_region_embedding(region_ids)[None]
            + token_type
        )

    def forward(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            observation.ndim != 2
            or observation.shape[1] != RAW_OBSERVATION_DIM_V6
        ):
            raise ValueError(
                "v6 model requires (batch,2205) raw observations"
            )
        batch = observation.shape[0]
        object_points = self._block(
            observation, "object_point_cloud_raw"
        ).reshape(batch, 16, 4, 3)
        object_tokens = self.object_stem(object_points).amax(dim=2)
        object_ids = torch.arange(16, device=observation.device)
        object_tokens = (
            object_tokens + self.object_patch_embedding(object_ids)[None]
        )

        current_hand = self._hand_tokens(
            self._block(observation, "hand_point_cloud_actual_raw"),
            reference=False,
        )
        contact = self._block(
            observation, "hand_region_contact"
        ).reshape(batch, 16, 10)
        current_hand = current_hand + self.region_contact(contact)
        goal = self._hand_tokens(
            self._block(observation, "hand_point_cloud_reference_raw"),
            reference=True,
        )

        numeric = torch.cat(
            [self._block(observation, name) for name in self.numeric_fields],
            dim=-1,
        )
        state = self.state(numeric).unsqueeze(1)
        readout = self.readout.expand(batch, -1, -1)
        current = torch.cat(
            (object_tokens, current_hand, state, readout), dim=1
        )
        if (
            current.shape[1] != CURRENT_TOKENS_V6
            or goal.shape[1] != GOAL_TOKENS_V6
        ):
            raise AssertionError("v6 token layout drifted")
        return current, goal


class AutonomyActorCriticV6(
    GaussianMixin, DeterministicMixin, Model
):
    """v6 policy with a shared token encoder and independent fusion towers."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device = "cpu",
        *,
        clip_actions: bool = False,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=None,
            action_space=action_space,
            device=device,
        )
        if (int(self.num_observations), int(self.num_actions)) != (
            RAW_OBSERVATION_DIM_V6,
            ACTION_DIM,
        ):
            raise ValueError("v6 actor requires raw 2205 and 28 actions")
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_mean_actions=False,
            clip_log_std=True,
            min_log_std=-5.0,
            max_log_std=2.0,
            reduction="sum",
            role="policy",
        )
        DeterministicMixin.__init__(
            self, clip_actions=False, role="value"
        )
        self.encoder = V6ObservationEncoder().to(device)
        self.actor = FusionTower(ACTION_DIM).to(device)
        self.critic = FusionTower(1).to(device)
        # Preserve the current ManoRL distribution initialization.
        self.log_std = nn.Parameter(
            torch.full((ACTION_DIM,), -1.0, device=device)
        )

    def checkpoint_architecture(self) -> dict[str, Any]:
        return actor_critic_architecture_v6()

    def checkpoint_contracts(self) -> dict[str, str]:
        return {
            "checkpoint_format": CHECKPOINT_FORMAT_V6,
            "observation_contract": OBSERVATION_CONTRACT_ID_V6,
            "reward_contract": REWARD_CONTRACT_ID,
            "action_contract": ACTION_CONTRACT_ID,
        }

    def act(self, inputs, role: str = ""):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        if role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
        raise ValueError("role must be policy or value")

    def compute(self, inputs, role: str = ""):
        current, goal = self.encoder(inputs["observations"])
        if role == "policy":
            mean = self.actor(current, goal)
            return mean, {"log_std": self.log_std.expand_as(mean)}
        if role == "value":
            return self.critic(current, goal), {}
        raise ValueError("role must be policy or value")

    @property
    def policy_sampling_contract(self) -> dict[str, str]:
        return dict(POLICY_SAMPLING_CONTRACT)
