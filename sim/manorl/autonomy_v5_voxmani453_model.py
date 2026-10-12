"""Full VoxMani 453 architecture with ManoRL raw-Normal/skrl interface."""

import torch
from torch import nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from sim.manorl.autonomy_contracts import (
    ACTION_CONTRACT_ID,
    CHECKPOINT_FORMAT_VOXMANI453,
    OBSERVATION_CONTRACT_ID_VOXMANI453,
    POLICY_SAMPLING_CONTRACT,
    REWARD_CONTRACT_ID,
)
from sim.manorl.voxmani453.canonical import build_canonical
from sim.manorl.voxmani453.goal_stream import empty_streams
from sim.manorl.voxmani453.network import ObsEncoder, Tower
from sim.manorl.voxmani453.streams import STREAM_SPECS, unpack_streams


def actor_critic_architecture_voxmani453():
    return {
        "id": "manorl.autonomy.actor_critic.v5.voxmani453",
        "upstream_commit": "16a80fc6615c3f82194acd34e45b98d459359325",
        "raw_observation_dim": 453,
        "action_dim": 28,
        "model_dim": 128,
        "attention_heads": 4,
        "ffn_dim": 512,
        "trunk_layers": 4,
        "skeleton_layers": 3,
        "skeleton_link_dim": 64,
        "value_trunk": "separate",
        "learned_encoder_sharing": False,
        "current_tokens": 70,
        "object_stream_raw_dim": 22,
        "hand_stream_raw_dim": 79,
        "per_joint_readout": True,
        "critic_only_fields": ["priv_13", "critic_keypoint_error"],
        "reference_adapter": "dense-manorl-0-6-12-24-end-v1",
        "coordinate_frame": "manorl-world",
        "canonical_surface": "manorl-collision",
        "critic_final_joint_update": False,
        "terminal_rotation": "inverse-current-times-final-columnwise-rot6",
    }


class AutonomyActorCriticVoxMani453(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        *,
        device="cpu",
        clip_actions=False,
        canonical=None,
        object_type="cube2",
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        if (self.num_observations, self.num_actions) != (453, 28):
            raise ValueError("VoxMani model requires 453 observations and 28 actions")
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
        DeterministicMixin.__init__(self, clip_actions=False, role="value")
        buffers, tree = (
            canonical if canonical is not None else build_canonical(object_type)
        )
        # Pure buffer expansion may be shared; every learned parameter is tower-local.
        self.encoder = ObsEncoder(buffers, skeleton_on=True)
        self.actor = Tower(
            tree,
            is_critic=False,
            skeleton_on=True,
            trunk_layers=4,
            stream_specs=STREAM_SPECS,
        )
        self.critic = Tower(
            tree,
            is_critic=True,
            skeleton_on=True,
            trunk_layers=4,
            stream_specs=STREAM_SPECS,
        )
        self.log_std = nn.Parameter(torch.full((28,), -1.0))
        self.observation_dim, self.action_dim = 453, 28
        self.separate_critic = True
        self.to(device)

    def checkpoint_architecture(self):
        return actor_critic_architecture_voxmani453()

    def checkpoint_contracts(self):
        return {
            "checkpoint_format": CHECKPOINT_FORMAT_VOXMANI453,
            "observation_contract": OBSERVATION_CONTRACT_ID_VOXMANI453,
            "action_contract": ACTION_CONTRACT_ID,
            "reward_contract": getattr(self, "reward_contract_id", REWARD_CONTRACT_ID),
        }

    def policy_sampling_contract(self):
        return dict(POLICY_SAMPLING_CONTRACT)

    def act(self, inputs, role=""):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        if role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
        raise ValueError("role must be policy or value")

    def compute(self, inputs, role=""):
        observation = inputs["observations"]
        if observation.ndim != 2 or observation.shape[1] != 453:
            raise ValueError("VoxMani observation must be [B,453]")
        streams = inputs.get("goal_streams")
        if streams is None:
            states = inputs.get("states")
            streams = (
                unpack_streams(states)
                if states is not None
                else empty_streams(
                    observation.shape[0], STREAM_SPECS, device=observation.device
                )
            )
        if set(streams) != set(STREAM_SPECS):
            raise ValueError("both object and hand streams must be supplied")
        for name, spec in STREAM_SPECS.items():
            stream = streams[name].validate_contract(raw_dim=spec.raw_dim)
            if (
                stream.batch != observation.shape[0]
                or stream.device != observation.device
            ):
                raise ValueError("stream batch/device mismatch")
        encoded = self.encoder.encode(observation)
        if role == "policy":
            features = self.actor.forward_varlen(encoded["actor"], streams)
            mean = self.actor.head(features)
            return mean, {"log_std": self.log_std.expand_as(mean)}
        if role == "value":
            features = self.critic.forward_varlen(encoded["critic"], streams)
            return self.critic.head(features)[:, None], {}
        raise ValueError("role must be policy or value")
