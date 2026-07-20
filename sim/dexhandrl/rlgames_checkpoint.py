"""Compatibility loading for DexHandRL rl-games checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from sim.dexhandrl.obs_layout import OBS_DIM_021PRO


class RlGamesRunningMeanStd(nn.Module):
    """Running normalizer with rl-games-compatible math and state names."""

    def __init__(
        self,
        size: int | list[int] | Any,
        *,
        epsilon: float = 1e-5,
        clip_threshold: float = 5.0,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if hasattr(size, "shape"):
            normalized_size = int(np.prod(size.shape))
        elif isinstance(size, (list, tuple)):
            normalized_size = int(np.prod(size))
        else:
            normalized_size = int(size)
        self.epsilon = float(epsilon)
        self.clip_threshold = float(clip_threshold)
        resolved_device = torch.device(device or "cpu")
        self.register_buffer("running_mean", torch.zeros(normalized_size, dtype=torch.float64, device=resolved_device))
        self.register_buffer("running_var", torch.ones(normalized_size, dtype=torch.float64, device=resolved_device))
        self.register_buffer("count", torch.ones((), dtype=torch.float64, device=resolved_device))

    def _update(self, values: torch.Tensor) -> None:
        feature_count = int(self.running_mean.numel())
        if values.numel() % feature_count != 0:
            raise RuntimeError(
                f"normalizer input shape {tuple(values.shape)} is incompatible with "
                f"feature count {feature_count}"
            )
        samples = values.reshape(-1, feature_count)
        batch_mean = samples.mean(dim=0).to(dtype=torch.float64)
        batch_count = int(samples.shape[0])
        batch_var = samples.var(
            dim=0,
            correction=1 if batch_count > 1 else 0,
        ).to(dtype=torch.float64)
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        first_moment = self.running_var * self.count
        second_moment = batch_var * batch_count
        combined = first_moment + second_moment + delta.square() * self.count * batch_count / total_count
        self.running_mean.copy_(self.running_mean + delta * batch_count / total_count)
        self.running_var.copy_(combined / total_count)
        self.count.copy_(total_count)

    def _compute(self, values: torch.Tensor, *, train: bool, inverse: bool) -> torch.Tensor:
        if train:
            self._update(values)
        mean = self.running_mean.float()
        scale = torch.sqrt(self.running_var.float() + self.epsilon)
        if inverse:
            clipped = torch.clamp(values, -self.clip_threshold, self.clip_threshold)
            return clipped * scale + mean
        normalized = (values - mean) / scale
        return torch.clamp(normalized, -self.clip_threshold, self.clip_threshold)

    def forward(
        self,
        values: torch.Tensor | None,
        *,
        train: bool = False,
        inverse: bool = False,
        no_grad: bool = True,
    ) -> torch.Tensor | None:
        if values is None:
            return None
        if no_grad:
            with torch.no_grad():
                return self._compute(values, train=train, inverse=inverse)
        return self._compute(values, train=train, inverse=inverse)


@dataclass(frozen=True)
class RlGamesCheckpointLoadReport:
    path: Path
    epoch: int | None
    frame: int | None
    last_mean_rewards: float | None
    observation_dim: int
    policy_tensors: int
    value_tensors: int


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        safe_globals: list[Any] = []
        numpy_scalar = np._core.multiarray.scalar
        safe_globals.append((numpy_scalar, "numpy.core.multiarray.scalar"))
        safe_globals.append(np.dtype)
        numpy_dtypes = getattr(np, "dtypes", None)
        if numpy_dtypes is not None:
            for dtype_name in ("Float32DType", "Float64DType", "Int32DType", "Int64DType"):
                dtype_class = getattr(numpy_dtypes, dtype_name, None)
                if dtype_class is not None:
                    safe_globals.append(dtype_class)
        with torch.serialization.safe_globals(safe_globals):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError(f"not an rl-games checkpoint with a model state dict: {path}")
    return payload


def _add_prefix_mapping(
    source: dict[str, torch.Tensor],
    mapping: dict[str, str],
    source_prefix: str,
    target_prefix: str,
) -> None:
    matched = False
    for source_key in source:
        if source_key.startswith(source_prefix):
            suffix = source_key[len(source_prefix) :]
            mapping[source_key] = target_prefix + suffix
            matched = True
    if not matched:
        raise RuntimeError(f"checkpoint has no tensors under {source_prefix!r}")


def _copy_mapped_tensors(
    module: nn.Module,
    source: dict[str, torch.Tensor],
    mapping: dict[str, str],
    *,
    component: str,
) -> int:
    target_state = module.state_dict()
    updates: dict[str, torch.Tensor] = {}
    for source_key, target_key in mapping.items():
        if source_key not in source:
            raise RuntimeError(f"{component}: checkpoint is missing {source_key!r}")
        if target_key not in target_state:
            raise RuntimeError(f"{component}: skrl model is missing target tensor {target_key!r}")
        source_tensor = source[source_key]
        target_tensor = target_state[target_key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise RuntimeError(
                f"{component}: tensor shape mismatch for {source_key!r} -> {target_key!r}: "
                f"checkpoint={tuple(source_tensor.shape)}, skrl={tuple(target_tensor.shape)}"
            )
        updates[target_key] = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
    module.load_state_dict(updates, strict=False)
    return len(updates)


def _load_normalizer(
    normalizer: Any,
    source: dict[str, torch.Tensor],
    prefix: str,
    *,
    expected_dim: int,
) -> None:
    if not isinstance(normalizer, RlGamesRunningMeanStd):
        raise RuntimeError(
            "rl-games checkpoint loading requires RlGamesRunningMeanStd preprocessors; "
            f"got {type(normalizer).__name__}"
        )
    mean = source.get(f"{prefix}.running_mean")
    var = source.get(f"{prefix}.running_var")
    count = source.get(f"{prefix}.count")
    if mean is None or var is None or count is None:
        raise RuntimeError(f"checkpoint is missing {prefix} running statistics")
    if mean.numel() != expected_dim or var.numel() != expected_dim:
        raise RuntimeError(
            f"checkpoint {prefix} dimension is {mean.numel()}, expected {expected_dim}. "
            "Use a checkpoint trained with the current observation layout."
        )
    normalizer.running_mean.copy_(mean.reshape_as(normalizer.running_mean).to(normalizer.running_mean))
    normalizer.running_var.copy_(var.reshape_as(normalizer.running_var).to(normalizer.running_var))
    normalizer.count.copy_(count.reshape_as(normalizer.count).to(normalizer.count))


def load_rlgames_checkpoint(agent: Any, checkpoint_path: str | Path) -> RlGamesCheckpointLoadReport:
    """Load compatible rl-games policy/value weights and normalization into skrl."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"rl-games checkpoint not found: {path}")
    payload = _load_payload(path)
    source = payload["model"]

    obs_mean = source.get("running_mean_std.running_mean")
    if obs_mean is None:
        raise RuntimeError("checkpoint does not contain input normalization statistics")
    observation_dim = int(obs_mean.numel())
    if observation_dim != OBS_DIM_021PRO:
        raise RuntimeError(
            f"checkpoint observation dimension is {observation_dim}, but the current DexHand021Pro layout is "
            f"{OBS_DIM_021PRO}. This checkpoint cannot be loaded without changing observation semantics."
        )

    policy_mapping = {
        "a2c_network.log_std": "log_std",
        "a2c_network.mu.weight": "mu.weight",
        "a2c_network.mu.bias": "mu.bias",
    }
    _add_prefix_mapping(
        source,
        policy_mapping,
        "a2c_network.pointnet_encoder.",
        "backbone.pointnet_encoder.",
    )
    _add_prefix_mapping(source, policy_mapping, "a2c_network.actor_mlp.", "backbone.actor_trunk.")
    _add_prefix_mapping(
        source,
        policy_mapping,
        "a2c_network.film_generators.0.",
        "backbone.actor_film_generators.0.",
    )
    _add_prefix_mapping(
        source,
        policy_mapping,
        "a2c_network.film_generators.1.",
        "backbone.actor_film_generators.2.",
    )

    value_mapping = {
        "a2c_network.value.weight": "value.weight",
        "a2c_network.value.bias": "value.bias",
    }
    _add_prefix_mapping(
        source,
        value_mapping,
        "a2c_network.pointnet_encoder.",
        "backbone.pointnet_encoder.",
    )
    _add_prefix_mapping(source, value_mapping, "a2c_network.critic_mlp.", "backbone.critic_trunk.")

    policy_tensors = _copy_mapped_tensors(agent.policy, source, policy_mapping, component="policy")
    value_tensors = _copy_mapped_tensors(agent.value, source, value_mapping, component="value")
    _load_normalizer(
        agent._observation_preprocessor,
        source,
        "running_mean_std",
        expected_dim=OBS_DIM_021PRO,
    )
    _load_normalizer(
        agent._state_preprocessor,
        source,
        "running_mean_std",
        expected_dim=OBS_DIM_021PRO,
    )
    _load_normalizer(
        agent._value_preprocessor,
        source,
        "value_mean_std",
        expected_dim=1,
    )

    return RlGamesCheckpointLoadReport(
        path=path,
        epoch=int(payload["epoch"]) if payload.get("epoch") is not None else None,
        frame=int(payload["frame"]) if payload.get("frame") is not None else None,
        last_mean_rewards=(
            float(payload["last_mean_rewards"])
            if payload.get("last_mean_rewards") is not None
            else None
        ),
        observation_dim=observation_dim,
        policy_tensors=policy_tensors,
        value_tensors=value_tensors,
    )
