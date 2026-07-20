"""Strict conversion of the validated IsaacGym/rl-games ManoRL checkpoint.

The source checkpoint is intentionally treated as read-only.  Conversion emits
the native skrl module mapping and a provenance sidecar; no source-specific
``rl_games`` object is imported at runtime.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from sim.manorl.checkpoint import (
    CHECKPOINT_FORMAT,
    GYM_CONVERTED_ENVIRONMENT_CONTRACT,
    GYM_PPO_REWARD_CONTRACT,
    GYM_REWARD_CONTRACT,
    GYM_SOURCE_FORMAT,
    _metadata_path,
)
from sim.manorl.abi import CHECKPOINT_SIDECAR_RESIDUAL_ACTION
from sim.manorl.model import ManoActorCritic
from sim.manorl.rewards import (
    CHECKPOINT_SIDECAR_PPO_REWARD_SCALE,
    CHECKPOINT_SIDECAR_REWARD_CONFIG,
)

SOURCE_PREFIX = "a2c_network.mano_net."
SOURCE_MODEL_KEY = "model"
SOURCE_OBSERVATION_NORMALIZER = "running_mean_std"
SOURCE_VALUE_NORMALIZER = "value_mean_std"

SOURCE_NORMALIZER_KEYS = (
    "running_mean", "running_var", "count",
    "pc_running_mean", "pc_running_var", "pc_count",
)
SOURCE_VALUE_KEYS = ("running_mean", "running_var", "count")
SUPPORTED_FLOAT_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)


class GymCheckpointFormatError(ValueError):
    """Raised when a source checkpoint cannot be converted safely."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest without modifying or locking ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GymCheckpointFormatError(f"source Gym checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise GymCheckpointFormatError(f"cannot read source Gym checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise GymCheckpointFormatError("source Gym checkpoint must be a mapping")
    if not isinstance(payload.get(SOURCE_MODEL_KEY), dict):
        raise GymCheckpointFormatError("source checkpoint lacks the rl-games model mapping")
    return payload


def _target_model() -> ManoActorCritic:
    model = ManoActorCritic(
        gym.spaces.Box(-5.0, 5.0, shape=(476,), dtype=float),
        None,
        gym.spaces.Box(-1.0, 1.0, shape=(26,), dtype=float),
        device="cpu",
    )
    model.eval()
    return model


def _tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise GymCheckpointFormatError(f"source tensor {name!r} is not a torch.Tensor")
    if tuple(value.shape) != shape:
        raise GymCheckpointFormatError(
            f"source tensor {name!r} has shape {tuple(value.shape)}, expected {shape}"
        )
    if not value.is_floating_point() or value.dtype not in SUPPORTED_FLOAT_DTYPES:
        raise GymCheckpointFormatError(
            f"source tensor {name!r} has unsupported dtype {value.dtype}"
        )
    if not torch.isfinite(value).all():
        raise GymCheckpointFormatError(f"source tensor {name!r} must be finite floating point")
    converted = value.detach().to(dtype=dtype, device="cpu").clone()
    if not torch.isfinite(converted).all():
        raise GymCheckpointFormatError(
            f"source tensor {name!r} becomes non-finite when converted to {dtype}"
        )
    return converted


def _map_model(source_model: dict[str, Any], target: ManoActorCritic) -> dict[str, torch.Tensor]:
    target_state = target.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    expected_source: set[str] = set()
    for target_name, target_value in target_state.items():
        source_name = (
            SOURCE_PREFIX + "condition_encoder.encoder.0." + target_name.rsplit(".", 1)[1]
            if target_name.startswith("condition_encoder.0.")
            else SOURCE_PREFIX + target_name
        )
        expected_source.add(source_name)
        mapped[target_name] = _tensor(
            source_model.get(source_name),
            name=source_name,
            shape=tuple(target_value.shape),
            dtype=target_value.dtype,
        )
    expected_auxiliary = {
        *(f"{SOURCE_OBSERVATION_NORMALIZER}.{name}" for name in SOURCE_NORMALIZER_KEYS),
        *(f"{SOURCE_VALUE_NORMALIZER}.{name}" for name in SOURCE_VALUE_KEYS),
    }
    expected_all = expected_source | expected_auxiliary
    source_model_keys = set(source_model)
    if source_model_keys != expected_all:
        missing = sorted(expected_all - source_model_keys)
        extra = sorted(source_model_keys - expected_all)
        raise GymCheckpointFormatError(
            f"source model parameter set differs from target; missing={missing}, extra={extra}"
        )
    return mapped


def _map_normalizer(source_model: dict[str, Any], prefix: str, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name in names:
        value = source_model.get(f"{prefix}.{name}")
        expected_shape = (3,) if name.startswith("pc_") and name not in {"pc_count"} else (() if name in {"count", "pc_count"} else (476,))
        if prefix == SOURCE_VALUE_NORMALIZER:
            expected_shape = () if name == "count" else (1,)
        result[name] = _tensor(
            value,
            name=f"{prefix}.{name}",
            shape=expected_shape,
            dtype=torch.float64,
        )
    return result


def _source_dtypes(
    source_model: dict[str, Any], *, prefix: str
) -> list[str]:
    return sorted(
        {str(value.dtype).removeprefix("torch.") for name, value in source_model.items()
         if name.startswith(prefix) and isinstance(value, torch.Tensor)}
    )


def _conversion_metadata(source_model: dict[str, Any]) -> dict[str, object]:
    return {
        "migration_profile": "isaacgym_mano_v1",
        "model_mapping": "rl-games.a2c_network.mano_net_to_skrl.ManoActorCritic",
        "pointnet": {"points": 64, "feature_dim": 64},
        "film": {
            "enabled": True,
            "condition_dim": 62,
            "embedding_dim": 32,
            "actor_layers": [0],
        },
        "normalizer": {
            "kind": "source_pointcloud_shared_xyz",
            "epsilon": 1.0e-5,
            "frozen_for_inference": True,
        },
        "dtype_conversion": {
            "model": {
                "source": _source_dtypes(source_model, prefix=SOURCE_PREFIX),
                "target": "float32",
            },
            "observation_normalizer": {
                "source": _source_dtypes(
                    source_model, prefix=SOURCE_OBSERVATION_NORMALIZER + "."
                ),
                "target": "float64",
            },
            "value_normalizer": {
                "source": _source_dtypes(
                    source_model, prefix=SOURCE_VALUE_NORMALIZER + "."
                ),
                "target": "float64",
            },
        },
    }


def _runtime_metadata() -> dict[str, object]:
    residual = CHECKPOINT_SIDECAR_RESIDUAL_ACTION
    reward = CHECKPOINT_SIDECAR_REWARD_CONFIG
    return {
        "model": {"use_film": True},
        "point_cloud": {
            "enabled": True,
            "dynamic": True,
            "points": 64,
            "coordinate_frame": "hand_relative",
            "normalize": False,
            "sampling_backend": "torch_cuda_global",
            "global_seed": 42,
        },
        "compatibility": {
            "name": "gym_eval_aligned",
            "early_phase_steps": 50,
            "movement_pre_padding": 250,
            "point_template_mode": "dynamic_reset",
        },
        "residual_action": asdict(residual),
        "reward": {
            **asdict(reward),
            "force_basis": "mujoco_pair_filtered_hand_object_force",
            "ppo_reward_scale": CHECKPOINT_SIDECAR_PPO_REWARD_SCALE,
        },
        "deterministic_policy_mode": "mean_clipped_to_action_space",
    }


def convert_gym_checkpoint(source: str | Path, output: str | Path) -> Path:
    """Convert one source ``.pth`` into a collision-safe native ``.pt``."""

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if source_path == output_path:
        raise GymCheckpointFormatError("source and output checkpoint paths must differ")
    if output_path.exists() or _metadata_path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite conversion output: {output_path}")
    if not source_path.is_file():
        raise GymCheckpointFormatError(f"source Gym checkpoint does not exist: {source_path}")

    source_sha256 = sha256_file(source_path)
    payload = _load_source(source_path)
    source_model = payload[SOURCE_MODEL_KEY]
    model = _target_model()
    model_state = _map_model(source_model, model)
    observation_state = _map_normalizer(source_model, SOURCE_OBSERVATION_NORMALIZER, SOURCE_NORMALIZER_KEYS)
    value_state = _map_normalizer(source_model, SOURCE_VALUE_NORMALIZER, SOURCE_VALUE_KEYS)
    if sha256_file(source_path) != source_sha256:
        raise GymCheckpointFormatError("source Gym checkpoint changed during conversion")

    # Construct a valid empty Adam state for skrl's parameter groups.  Inference
    # does not use its moments, but native training/resume loaders require the
    # optimizer module to have the same structural boundary.
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)
    modules = {
        "policy": model_state,
        "value": model_state,
        "optimizer": optimizer.state_dict(),
        "observation_preprocessor": {
            "running_mean": observation_state["running_mean"],
            "running_var": observation_state["running_var"],
            "count": observation_state["count"],
            "pc_running_mean": observation_state["pc_running_mean"],
            "pc_running_var": observation_state["pc_running_var"],
            "pc_count": observation_state["pc_count"],
        },
        "value_preprocessor": {
            "running_mean": value_state["running_mean"],
            "running_variance": value_state["running_var"],
            "current_count": value_state["count"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modules, output_path)
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_file": output_path.name,
        "reward_contract": GYM_REWARD_CONTRACT,
        "ppo_reward_contract": GYM_PPO_REWARD_CONTRACT,
        "environment_contract": GYM_CONVERTED_ENVIRONMENT_CONTRACT,
        "source": {
            "format": GYM_SOURCE_FORMAT,
            "path": str(source_path),
            "sha256": source_sha256,
            **(
                {"checkpoint_epoch": payload["epoch"]}
                if type(payload.get("epoch")) is int
                else {}
            ),
            **(
                {"checkpoint_frame": payload["frame"]}
                if type(payload.get("frame")) is int
                else {}
            ),
        },
        "conversion": _conversion_metadata(source_model),
        "runtime_config": _runtime_metadata(),
    }
    _metadata_path(output_path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
