"""Intermediate ManoRL v5-to-v6 actor-critic ablations.

These models preserve the ManoRL environment, reward, PPO, action distribution,
and physical command path. Each class changes only the observation encoder
declared by its checkpoint contract.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from skrl.models.torch import Model
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin

from sim.manorl.autonomy_contracts import (
    ACTION_CONTRACT_ID,
    ACTION_DIM,
    CHECKPOINT_FORMAT_V525,
    CHECKPOINT_FORMAT_V55,
    CHECKPOINT_FORMAT_V575,
    ENCODED_OBSERVATION_DIM_V525,
    ENCODED_OBSERVATION_DIM_V55,
    ENCODED_OBSERVATION_DIM_V575,
    OBSERVATION_CONTRACT_ID_V525,
    OBSERVATION_CONTRACT_ID_V55,
    OBSERVATION_CONTRACT_ID_V575,
    RAW_OBSERVATION_DIM_V525,
    RAW_OBSERVATION_DIM_V55,
    RAW_OBSERVATION_DIM_V575,
    REWARD_CONTRACT_ID,
    raw_observation_slices_v55,
    raw_observation_slices_v575,
)
from sim.manorl.model import PointNetEncoder

ACTOR_CRITIC_ARCHITECTURE_ID_V525 = (
    "manorl.autonomy.actor_critic.v5.25.region-hand"
)
ACTOR_CRITIC_ARCHITECTURE_ID_V55 = (
    "manorl.autonomy.actor_critic.v5.5.region-hand-goal"
)
ACTOR_CRITIC_ARCHITECTURE_ID_V575 = (
    "manorl.autonomy.actor_critic.v5.75.contact-wrench"
)


def actor_critic_architecture_v525(
    *, separate_critic: bool = False
) -> dict[str, Any]:
    return {
        "id": ACTOR_CRITIC_ARCHITECTURE_ID_V525,
        "raw_observation_dim": RAW_OBSERVATION_DIM_V525,
        "encoded_feature_dim": ENCODED_OBSERVATION_DIM_V525,
        "object_encoder": "global PointNet 64 points -> 64",
        "hand_encoder": (
            "16 regions x 16 points -> shared point MLP -> "
            "64D region tokens + identity embedding -> max pool"
        ),
        "attention": "none",
        "value_trunk": "separate" if separate_critic else "shared",
        "action_dim": ACTION_DIM,
    }


class RegionHandEncoder(nn.Module):
    """Encode 16 hand regions without discarding their identity before pooling."""

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.point_stem = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )
        self.region_embedding = nn.Embedding(16, feature_dim)

    def tokens(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[1:] != (256, 3):
            raise ValueError("region hand encoder requires (batch,256,3)")
        batch = points.shape[0]
        features = self.point_stem(points.reshape(batch, 16, 16, 3))
        pooled = features.amax(dim=2)
        region_ids = torch.arange(16, device=points.device)
        return pooled + self.region_embedding(region_ids)[None]

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.tokens(points).amax(dim=1)


class HandGoalCrossAttention(nn.Module):
    """One pre-normalized current-hand to region-goal attention layer."""

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(feature_dim)
        self.goal_norm = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            feature_dim, 4, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(feature_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.SiLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(
        self, current: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        query = self.query_norm(current)
        memory = self.goal_norm(goal)
        attended, _ = self.attention(
            query, memory, memory, need_weights=False
        )
        fused = current + attended
        return fused + self.feedforward(self.feedforward_norm(fused))


class AutonomyActorCriticV525(
    GaussianMixin, DeterministicMixin, Model
):
    """v5.25: v5 MLP policy with an identity-preserving current-hand encoder."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device = "cpu",
        *,
        separate_critic: bool = False,
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
            RAW_OBSERVATION_DIM_V525,
            ACTION_DIM,
        ):
            raise ValueError("v5.25 actor requires raw 1342 and 28 actions")
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
        self.object_pointnet = PointNetEncoder(device)
        self.hand_encoder = RegionHandEncoder().to(device)
        self.policy_trunk = nn.Sequential(
            nn.Linear(ENCODED_OBSERVATION_DIM_V525, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        ).to(device)
        self.value_trunk = (
            nn.Sequential(
                nn.Linear(ENCODED_OBSERVATION_DIM_V525, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
            ).to(device)
            if separate_critic
            else None
        )
        self.separate_critic = bool(separate_critic)
        self.mean = nn.Linear(128, ACTION_DIM).to(device)
        self.value = nn.Linear(128, 1).to(device)
        self.log_std = nn.Parameter(
            torch.full((ACTION_DIM,), -1.0, device=device)
        )

    def _encoded(self, observation: torch.Tensor) -> torch.Tensor:
        if (
            observation.ndim != 2
            or observation.shape[1] != RAW_OBSERVATION_DIM_V525
        ):
            raise ValueError(
                "v5.25 model requires (batch,1342) raw observations"
            )
        object_cloud = observation[:, 320:512].reshape(-1, 64, 3)
        hand_cloud = observation[:, 512:1280].reshape(-1, 256, 3)
        return torch.cat(
            (
                observation[:, :320],
                self.object_pointnet(object_cloud),
                self.hand_encoder(hand_cloud),
                observation[:, 1280:],
            ),
            dim=-1,
        )

    def checkpoint_architecture(self) -> dict[str, Any]:
        return actor_critic_architecture_v525(
            separate_critic=self.separate_critic
        )

    def checkpoint_contracts(self) -> dict[str, str]:
        return {
            "checkpoint_format": CHECKPOINT_FORMAT_V525,
            "observation_contract": OBSERVATION_CONTRACT_ID_V525,
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
        encoded = self._encoded(inputs["observations"])
        if role == "policy":
            mean = self.mean(self.policy_trunk(encoded))
            return mean, {"log_std": self.log_std.expand_as(mean)}
        if role == "value":
            trunk = self.value_trunk or self.policy_trunk
            return self.value(trunk(encoded)), {}
        raise ValueError("role must be policy or value")


def actor_critic_architecture_v55(
    *, separate_critic: bool = False
) -> dict[str, Any]:
    return {
        "id": ACTOR_CRITIC_ARCHITECTURE_ID_V55,
        "raw_observation_dim": RAW_OBSERVATION_DIM_V55,
        "encoded_feature_dim": ENCODED_OBSERVATION_DIM_V55,
        "object_encoder": "global PointNet 64 points -> 64",
        "hand_encoder": "16 current regions x 16 points -> 16 x 64",
        "goal_encoder": "16 object-frame pose9 -> 16 x 64",
        "attention": "1 x current-hand queries to region-goal cross-attention",
        "value_trunk": "separate" if separate_critic else "shared",
        "action_dim": ACTION_DIM,
    }


def actor_critic_architecture_v575(
    *, separate_critic: bool = False
) -> dict[str, Any]:
    return {
        "id": ACTOR_CRITIC_ARCHITECTURE_ID_V575,
        "raw_observation_dim": RAW_OBSERVATION_DIM_V575,
        "encoded_feature_dim": ENCODED_OBSERVATION_DIM_V575,
        "object_encoder": "global PointNet 64 points -> 64",
        "hand_encoder": (
            "16 current regions x 16 points + bound contact10 -> 16 x 64"
        ),
        "goal_encoder": "16 object-frame pose9 -> 16 x 64",
        "attention": "1 x current-hand queries to region-goal cross-attention",
        "object_wrench": "15D gravity/hand/other force and torque",
        "value_trunk": "separate" if separate_critic else "shared",
        "action_dim": ACTION_DIM,
    }


class _AutonomyActorCriticRegionGoal(
    GaussianMixin, DeterministicMixin, Model
):
    """Shared implementation for the v5.5 and v5.75 controlled ablations."""

    raw_observation_dim: int
    encoded_feature_dim: int
    slices_factory: Any

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device = "cpu",
        *,
        separate_critic: bool = False,
        clip_actions: bool = False,
        bind_dynamic_contact: bool,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=None,
            action_space=action_space,
            device=device,
        )
        if (int(self.num_observations), int(self.num_actions)) != (
            self.raw_observation_dim,
            ACTION_DIM,
        ):
            raise ValueError(
                f"{type(self).__name__} requires raw "
                f"{self.raw_observation_dim} and 28 actions"
            )
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
        self.slices = self.slices_factory()
        self.object_pointnet = PointNetEncoder(device)
        self.hand_encoder = RegionHandEncoder().to(device)
        self.goal_pose = nn.Sequential(
            nn.Linear(9, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        ).to(device)
        self.goal_region_embedding = nn.Embedding(16, 64).to(device)
        self.cross_attention = HandGoalCrossAttention().to(device)
        self.bind_dynamic_contact = bool(bind_dynamic_contact)
        self.region_contact = (
            nn.Sequential(
                nn.Linear(10, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
            ).to(device)
            if bind_dynamic_contact
            else None
        )
        self.policy_trunk = nn.Sequential(
            nn.Linear(self.encoded_feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        ).to(device)
        self.value_trunk = (
            nn.Sequential(
                nn.Linear(self.encoded_feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
            ).to(device)
            if separate_critic
            else None
        )
        self.separate_critic = bool(separate_critic)
        self.mean = nn.Linear(128, ACTION_DIM).to(device)
        self.value = nn.Linear(128, 1).to(device)
        self.log_std = nn.Parameter(
            torch.full((ACTION_DIM,), -1.0, device=device)
        )

    def _block(
        self, observation: torch.Tensor, name: str
    ) -> torch.Tensor:
        return observation[:, self.slices[name]]

    def _contact_features(
        self, observation: torch.Tensor
    ) -> torch.Tensor:
        intent = self._block(
            observation, "contact_intent"
        )
        confidence = intent[:, :16, None]
        valid = intent[:, 16:32, None]
        force = intent[:, 32:].reshape(-1, 16, 3)
        dynamic = self._block(
            observation, "hand_region_dynamic_contact"
        ).reshape(-1, 16, 5)
        return torch.cat(
            (
                confidence,
                valid,
                dynamic[:, :, :2],
                force,
                dynamic[:, :, 2:],
            ),
            dim=-1,
        )

    def _encoded(self, observation: torch.Tensor) -> torch.Tensor:
        if (
            observation.ndim != 2
            or observation.shape[1] != self.raw_observation_dim
        ):
            raise ValueError(
                f"{type(self).__name__} requires "
                f"(batch,{self.raw_observation_dim}) raw observations"
            )
        batch = observation.shape[0]
        object_cloud = self._block(
            observation, "object_point_cloud_raw"
        ).reshape(batch, 64, 3)
        hand_cloud = self._block(
            observation, "hand_point_cloud_raw"
        ).reshape(batch, 256, 3)
        current = self.hand_encoder.tokens(hand_cloud)
        if self.region_contact is not None:
            current = current + self.region_contact(
                self._contact_features(observation)
            )
        goal_pose = self._block(
            observation, "hand_region_goal_pose"
        ).reshape(batch, 16, 9)
        region_ids = torch.arange(16, device=observation.device)
        goal = (
            self.goal_pose(goal_pose)
            + self.goal_region_embedding(region_ids)[None]
        )
        hand = self.cross_attention(current, goal).amax(dim=1)
        numeric_names = [
            "autonomous_actual",
            "autonomous_reference",
            "autonomous_future",
            "contact_intent",
        ]
        if self.bind_dynamic_contact:
            numeric_names.append("object_wrench")
        numeric_names.extend(("action_types", "object_geometry"))
        numeric = torch.cat(
            [self._block(observation, name) for name in numeric_names],
            dim=-1,
        )
        encoded = torch.cat(
            (numeric, self.object_pointnet(object_cloud), hand), dim=-1
        )
        if encoded.shape[1] != self.encoded_feature_dim:
            raise AssertionError("intermediate encoded feature ABI drifted")
        return encoded

    def act(self, inputs, role: str = ""):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        if role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
        raise ValueError("role must be policy or value")

    def compute(self, inputs, role: str = ""):
        encoded = self._encoded(inputs["observations"])
        if role == "policy":
            mean = self.mean(self.policy_trunk(encoded))
            return mean, {"log_std": self.log_std.expand_as(mean)}
        if role == "value":
            trunk = self.value_trunk or self.policy_trunk
            return self.value(trunk(encoded)), {}
        raise ValueError("role must be policy or value")


class AutonomyActorCriticV55(_AutonomyActorCriticRegionGoal):
    raw_observation_dim = RAW_OBSERVATION_DIM_V55
    encoded_feature_dim = ENCODED_OBSERVATION_DIM_V55
    slices_factory = staticmethod(raw_observation_slices_v55)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, bind_dynamic_contact=False, **kwargs)

    def checkpoint_architecture(self) -> dict[str, Any]:
        return actor_critic_architecture_v55(
            separate_critic=self.separate_critic
        )

    def checkpoint_contracts(self) -> dict[str, str]:
        return {
            "checkpoint_format": CHECKPOINT_FORMAT_V55,
            "observation_contract": OBSERVATION_CONTRACT_ID_V55,
            "reward_contract": REWARD_CONTRACT_ID,
            "action_contract": ACTION_CONTRACT_ID,
        }


class AutonomyActorCriticV575(_AutonomyActorCriticRegionGoal):
    raw_observation_dim = RAW_OBSERVATION_DIM_V575
    encoded_feature_dim = ENCODED_OBSERVATION_DIM_V575
    slices_factory = staticmethod(raw_observation_slices_v575)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, bind_dynamic_contact=True, **kwargs)

    def checkpoint_architecture(self) -> dict[str, Any]:
        return actor_critic_architecture_v575(
            separate_critic=self.separate_critic
        )

    def checkpoint_contracts(self) -> dict[str, str]:
        return {
            "checkpoint_format": CHECKPOINT_FORMAT_V575,
            "observation_contract": OBSERVATION_CONTRACT_ID_V575,
            "reward_contract": REWARD_CONTRACT_ID,
            "action_contract": ACTION_CONTRACT_ID,
        }
