"""skrl-compatible actor and value models for DexHandRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
from loguru import logger
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from sim.dexhandrl.obs_layout import (
    ACTION_DIM_021PRO,
    DEFAULT_FILM_GENERATOR_HIDDEN_DIM,
    DEFAULT_MLP_UNITS,
    DEFAULT_POINTNET_FEATURE_DIM,
    DEFAULT_POINTNET_NUM_POINTS,
    FILM_CONDITION_KEYS,
    OBS_COMPONENT_SLICES,
    OBS_DIM_021PRO,
    POINT_CLOUD_SLICE,
)
from sim.dexhandrl.pointnet_encoder import PointNetEncoder


@dataclass(frozen=True)
class SkrlDexHandModelConfig:
    obs_dim: int = OBS_DIM_021PRO
    action_dim: int = ACTION_DIM_021PRO
    pointnet_enabled: bool = True
    pointnet_num_points: int = DEFAULT_POINTNET_NUM_POINTS
    pointnet_feature_dim: int = DEFAULT_POINTNET_FEATURE_DIM
    film_enabled: bool = True
    film_condition_keys: tuple[str, ...] = FILM_CONDITION_KEYS
    film_apply_layers: tuple[int, ...] = (0, 1)
    film_generator_hidden_dim: int = DEFAULT_FILM_GENERATOR_HIDDEN_DIM
    mlp_units: tuple[int, ...] = DEFAULT_MLP_UNITS
    activation: str = "elu"
    separate: bool = True
    fixed_sigma: bool = True
    log_std_init: float = -0.99
    clip_actions: bool = False
    clip_mean_actions: bool = False
    clip_log_std: bool = True
    min_log_std: float = -20.0
    max_log_std: float = 2.0
    reduction: str = "sum"


def _get_activation(name: str) -> nn.Module:
    lookup = {
        "none": nn.Identity,
        "identity": nn.Identity,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    key = str(name).lower()
    if key not in lookup:
        raise KeyError(f"unsupported activation {name!r}")
    return lookup[key]()


def _linear_indices(network: nn.Sequential) -> list[int]:
    return [idx for idx, layer in enumerate(network) if isinstance(layer, nn.Linear)]


def _extract_tensor(inputs: Any, preferred_key: str) -> torch.Tensor:
    if isinstance(inputs, dict):
        fallback_key = "states" if preferred_key == "observations" else "observations"
        if preferred_key in inputs and inputs[preferred_key] is not None:
            return inputs[preferred_key]
        if fallback_key in inputs and inputs[fallback_key] is not None:
            return inputs[fallback_key]
        raise RuntimeError(f"unexpected skrl input keys: {sorted(inputs.keys())}")
    return inputs


def _slice_tensor(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return tensor[:, start:end]


class _DexHandFeatureBackbone(nn.Module):
    def __init__(self, observation_space: Any, device: Any, cfg: SkrlDexHandModelConfig, *, role: str):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.role = role

        self.obs_dim = int(observation_space.shape[0] if hasattr(observation_space, "shape") else cfg.obs_dim)
        self.action_dim = int(cfg.action_dim)
        self.use_pointnet = bool(cfg.pointnet_enabled)
        self.use_film = bool(cfg.film_enabled)
        self.separate = bool(cfg.separate)
        self.fixed_sigma = bool(cfg.fixed_sigma)
        self.film_condition_keys = tuple(cfg.film_condition_keys)
        self.film_condition_slices = [OBS_COMPONENT_SLICES[key] for key in self.film_condition_keys]
        self.film_condition_dim = sum(end - start for start, end in self.film_condition_slices)
        self.pointnet_slice = POINT_CLOUD_SLICE
        self.pointnet_dim = self.pointnet_slice[1] - self.pointnet_slice[0]
        self.non_pc_slice_ranges = [self.pointnet_slice]
        if self.use_film and not self.separate:
            raise RuntimeError("skrl FiLM mode requires separate=true to match the IsaacGym network contract")
        if self.use_film:
            self.non_pc_slice_ranges.extend(self.film_condition_slices)

        if self.use_pointnet:
            self.pointnet_encoder = PointNetEncoder(
                feature_dim=cfg.pointnet_feature_dim,
                num_points=cfg.pointnet_num_points,
            )
            fused_dim = cfg.pointnet_feature_dim + (self.obs_dim - self.pointnet_dim - self.film_condition_dim)
        else:
            self.pointnet_encoder = None
            fused_dim = self.obs_dim - self.film_condition_dim

        if fused_dim < 0:
            raise RuntimeError(
                f"invalid observation layout: obs_dim={self.obs_dim}, pointnet_dim={self.pointnet_dim}, "
                f"film_condition_dim={self.film_condition_dim}"
        )
        self.fused_dim = fused_dim

        self._film_condition_tensor_slices = [slice(start, end) for start, end in self.film_condition_slices]
        self.shared_trunk = None
        self.actor_trunk = None
        self.critic_trunk = None
        self.actor_linear_indices: list[int] = []
        self.actor_linear_indices_to_apply: set[int] = set()
        self.actor_film_generators = nn.ModuleDict()
        self.critic_linear_indices: list[int] = []
        if self.separate or self.use_film:
            self.actor_trunk = self._build_mlp(self.fused_dim, cfg.mlp_units, cfg.activation)
            self.actor_linear_indices = _linear_indices(self.actor_trunk)
            self.actor_linear_indices_to_apply = {
                self.actor_linear_indices[layer_idx]
                for layer_idx in cfg.film_apply_layers
                if 0 <= layer_idx < len(self.actor_linear_indices)
            }
            if self.use_film and self.actor_linear_indices_to_apply:
                for linear_idx in sorted(self.actor_linear_indices_to_apply):
                    layer = self.actor_trunk[linear_idx]
                    assert isinstance(layer, nn.Linear)
                    generator = nn.Sequential(
                        nn.Linear(self.film_condition_dim, cfg.film_generator_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(cfg.film_generator_hidden_dim, 2 * layer.out_features),
                    )
                    nn.init.normal_(generator[-1].weight, std=0.001)
                    nn.init.zeros_(generator[-1].bias)
                    self.actor_film_generators[str(linear_idx)] = generator

            critic_input_dim = self.fused_dim + self.film_condition_dim if self.use_film and role == "value" else self.fused_dim
            self.critic_trunk = self._build_mlp(critic_input_dim, cfg.mlp_units, cfg.activation)
            self.critic_linear_indices = _linear_indices(self.critic_trunk)
        else:
            self.shared_trunk = self._build_mlp(self.fused_dim, cfg.mlp_units, cfg.activation)
            self.critic_linear_indices = _linear_indices(self.shared_trunk)

        self._log_layout()

    def _build_mlp(self, input_dim: int, units: Iterable[int], activation_name: str) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in units:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(_get_activation(activation_name))
            prev_dim = int(hidden_dim)
        return nn.Sequential(*layers)

    def _log_layout(self) -> None:
        trunk_mode = "shared" if self.shared_trunk is not None else "separate"
        logger.info(
            f"skrl {self.role} backbone: obs_dim={self.obs_dim}, fused_dim={self.fused_dim}, "
            f"film_condition_dim={self.film_condition_dim}, pointnet={self.use_pointnet}, trunk_mode={trunk_mode}"
        )

    def _exclude_slices(self, obs_tensor: torch.Tensor, slices: list[tuple[int, int]]) -> torch.Tensor:
        if not slices:
            return obs_tensor
        kept: list[torch.Tensor] = []
        cursor = 0
        for start, end in sorted(slices, key=lambda item: item[0]):
            if cursor < start:
                kept.append(obs_tensor[:, cursor:start])
            cursor = max(cursor, end)
        if cursor < obs_tensor.shape[1]:
            kept.append(obs_tensor[:, cursor:])
        if not kept:
            return obs_tensor.new_zeros((obs_tensor.shape[0], 0))
        return torch.cat(kept, dim=-1)

    def _extract_film_condition(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        pieces = [_slice_tensor(obs_tensor, start, end) for start, end in self.film_condition_slices]
        return torch.cat(pieces, dim=-1)

    def _apply_film(self, features: torch.Tensor, film_params: torch.Tensor) -> torch.Tensor:
        delta_gamma, beta = film_params.chunk(2, dim=-1)
        return (1.0 + delta_gamma) * features + beta

    def _forward_actor_trunk(self, fused_features: torch.Tensor, film_condition: torch.Tensor | None) -> torch.Tensor:
        if self.shared_trunk is not None:
            return self.shared_trunk(fused_features)
        if not self.use_film:
            return self.actor_trunk(fused_features)
        if film_condition is None:
            raise RuntimeError("film condition requested but not provided")
        x = fused_features
        for idx, layer in enumerate(self.actor_trunk):
            x = layer(x)
            if idx in self.actor_linear_indices_to_apply:
                generator = self.actor_film_generators[str(idx)] if str(idx) in self.actor_film_generators else None
                if generator is not None:
                    x = self._apply_film(x, generator(film_condition))
        return x

    def _forward_critic_trunk(self, fused_features: torch.Tensor, film_condition: torch.Tensor | None) -> torch.Tensor:
        if self.shared_trunk is not None:
            return self.shared_trunk(fused_features)
        if self.use_film and film_condition is not None and self.role == "value":
            critic_input = torch.cat([fused_features, film_condition], dim=-1)
        else:
            critic_input = fused_features
        return self.critic_trunk(critic_input)

    def _encode_features(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        film_condition = self._extract_film_condition(states) if self.use_film else None
        if self.use_pointnet:
            point_cloud = states[:, self.pointnet_slice[0] : self.pointnet_slice[1]]
            excluded = [self.pointnet_slice]
            if self.use_film:
                excluded.extend(self.film_condition_slices)
            non_pc_obs = self._exclude_slices(states, excluded)
            pc_features = self.pointnet_encoder(point_cloud)  # type: ignore[operator]
            fused_features = torch.cat([non_pc_obs, pc_features], dim=-1)
        else:
            excluded = list(self.film_condition_slices) if self.use_film else []
            fused_features = self._exclude_slices(states, excluded)
        return fused_features, film_condition


class DexHandSkrlPolicy(GaussianMixin, Model):
    """Stochastic actor that mirrors the IsaacGym DexHand actor path."""

    def __init__(
        self,
        observation_space: Any,
        state_space: Any,
        action_space: Any,
        device: Any,
        cfg: SkrlDexHandModelConfig | None = None,
    ) -> None:
        cfg = cfg or SkrlDexHandModelConfig()
        try:
            Model.__init__(
                self,
                observation_space=observation_space,
                state_space=state_space,
                action_space=action_space,
                device=device,
            )
        except TypeError:  # pragma: no cover - older skrl releases
            Model.__init__(self, observation_space, action_space, device)
        try:
            GaussianMixin.__init__(
                self,
                clip_actions=cfg.clip_actions,
                clip_mean_actions=cfg.clip_mean_actions,
                clip_log_std=cfg.clip_log_std,
                min_log_std=cfg.min_log_std,
                max_log_std=cfg.max_log_std,
                reduction=cfg.reduction,
                role="policy",
            )
        except TypeError:  # pragma: no cover - older skrl releases
            GaussianMixin.__init__(
                self,
                clip_actions=cfg.clip_actions,
                clip_mean_actions=cfg.clip_mean_actions,
                clip_log_std=cfg.clip_log_std,
                min_log_std=cfg.min_log_std,
                max_log_std=cfg.max_log_std,
                reduction=cfg.reduction,
            )
        self.backbone = _DexHandFeatureBackbone(observation_space, device, cfg, role="policy")
        actor_last_dim = cfg.mlp_units[-1] if cfg.mlp_units else self.backbone.fused_dim
        self.mu = nn.Linear(actor_last_dim, self.backbone.action_dim)
        if self.backbone.fixed_sigma:
            self.log_std = nn.Parameter(
                torch.full((self.backbone.action_dim,), float(cfg.log_std_init), device=self.device)
            )
        else:
            self.log_std = nn.Linear(actor_last_dim, self.backbone.action_dim)
        self.to(self.device)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        states = _extract_tensor(inputs, "observations").to(self.device)
        fused_features, film_condition = self.backbone._encode_features(states)
        actor_features = self.backbone._forward_actor_trunk(fused_features, film_condition)
        mean_actions = self.mu(actor_features)
        if self.backbone.fixed_sigma:
            log_std = self.log_std.unsqueeze(0).expand_as(mean_actions)
        else:
            log_std = self.log_std(actor_features)
        return mean_actions, {"log_std": log_std}


class DexHandSkrlValue(DeterministicMixin, Model):
    """Value function that mirrors the IsaacGym critic path."""

    def __init__(
        self,
        observation_space: Any,
        state_space: Any,
        action_space: Any,
        device: Any,
        cfg: SkrlDexHandModelConfig | None = None,
    ) -> None:
        cfg = cfg or SkrlDexHandModelConfig()
        try:
            Model.__init__(
                self,
                observation_space=observation_space,
                state_space=state_space,
                action_space=action_space,
                device=device,
            )
        except TypeError:  # pragma: no cover - older skrl releases
            Model.__init__(self, observation_space, action_space, device)
        try:
            DeterministicMixin.__init__(self, clip_actions=cfg.clip_actions, role="value")
        except TypeError:  # pragma: no cover - older skrl releases
            DeterministicMixin.__init__(self, clip_actions=cfg.clip_actions)
        self.backbone = _DexHandFeatureBackbone(observation_space, device, cfg, role="value")
        critic_last_dim = cfg.mlp_units[-1] if cfg.mlp_units else self.backbone.fused_dim
        self.value = nn.Linear(critic_last_dim, 1)
        self.to(self.device)

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        states = _extract_tensor(inputs, "states").to(self.device)
        fused_features, film_condition = self.backbone._encode_features(states)
        critic_features = self.backbone._forward_critic_trunk(fused_features, film_condition)
        value = self.value(critic_features)
        return value, {}


class DexHandSkrlSharedModel(GaussianMixin, DeterministicMixin, Model):
    """Shared policy/value model used when the IsaacGym network shares its trunk."""

    def __init__(
        self,
        observation_space: Any,
        state_space: Any,
        action_space: Any,
        device: Any,
        cfg: SkrlDexHandModelConfig | None = None,
    ) -> None:
        cfg = cfg or SkrlDexHandModelConfig()
        try:
            Model.__init__(
                self,
                observation_space=observation_space,
                state_space=state_space,
                action_space=action_space,
                device=device,
            )
        except TypeError:  # pragma: no cover - older skrl releases
            Model.__init__(self, observation_space, action_space, device)
        try:
            GaussianMixin.__init__(
                self,
                clip_actions=cfg.clip_actions,
                clip_mean_actions=cfg.clip_mean_actions,
                clip_log_std=cfg.clip_log_std,
                min_log_std=cfg.min_log_std,
                max_log_std=cfg.max_log_std,
                reduction=cfg.reduction,
                role="policy",
            )
        except TypeError:  # pragma: no cover - older skrl releases
            GaussianMixin.__init__(
                self,
                clip_actions=cfg.clip_actions,
                clip_mean_actions=cfg.clip_mean_actions,
                clip_log_std=cfg.clip_log_std,
                min_log_std=cfg.min_log_std,
                max_log_std=cfg.max_log_std,
                reduction=cfg.reduction,
            )
        try:
            DeterministicMixin.__init__(self, clip_actions=cfg.clip_actions, role="value")
        except TypeError:  # pragma: no cover - older skrl releases
            DeterministicMixin.__init__(self, clip_actions=cfg.clip_actions)
        self.backbone = _DexHandFeatureBackbone(observation_space, device, cfg, role="shared")
        shared_last_dim = cfg.mlp_units[-1] if cfg.mlp_units else self.backbone.fused_dim
        self.mu = nn.Linear(shared_last_dim, self.backbone.action_dim)
        self.value = nn.Linear(shared_last_dim, 1)
        if self.backbone.fixed_sigma:
            self.log_std = nn.Parameter(
                torch.full((self.backbone.action_dim,), float(cfg.log_std_init), device=self.device)
            )
        else:
            self.log_std = nn.Linear(shared_last_dim, self.backbone.action_dim)
        self.to(self.device)

    def act(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        if role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
        raise RuntimeError(f"unsupported skrl role {role!r} for shared DexHand model")

    def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[torch.Tensor, dict[str, Any]]:
        preferred_key = "observations" if role == "policy" else "states"
        states = _extract_tensor(inputs, preferred_key).to(self.device)
        fused_features, film_condition = self.backbone._encode_features(states)
        shared_features = self.backbone._forward_actor_trunk(fused_features, film_condition)
        if role == "policy":
            mean_actions = self.mu(shared_features)
            if self.backbone.fixed_sigma:
                log_std = self.log_std.unsqueeze(0).expand_as(mean_actions)
            else:
                log_std = self.log_std(shared_features)
            return mean_actions, {"log_std": log_std}
        if role == "value":
            value = self.value(shared_features)
            return value, {}
        raise RuntimeError(f"unsupported skrl role {role!r} for shared DexHand model")


def build_skrl_models(
    observation_space: Any,
    state_space: Any,
    action_space: Any,
    device: Any,
    cfg: SkrlDexHandModelConfig | None = None,
) -> dict[str, nn.Module]:
    cfg = cfg or SkrlDexHandModelConfig()
    if cfg.separate:
        policy = DexHandSkrlPolicy(observation_space, state_space, action_space, device, cfg)
        value = DexHandSkrlValue(observation_space, state_space, action_space, device, cfg)
    else:
        shared_model = DexHandSkrlSharedModel(observation_space, state_space, action_space, device, cfg)
        policy = shared_model
        value = shared_model
    return {"policy": policy, "value": value}
