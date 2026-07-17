"""Native skrl checkpoint boundary for the ManoRL target runtime.

Raw Isaac rl-games checkpoints are rejected at this boundary. They must first
pass through the Gym checkpoint converter, which emits an auditable native
checkpoint and source-compatible sidecar.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from sim.manorl.abi import ENVIRONMENT_CONTRACT_ID
from sim.manorl.rewards import PPO_REWARD_CONTRACT_ID, REWARD_CONTRACT_ID

if TYPE_CHECKING:
    from skrl.agents.torch.ppo import PPO


CHECKPOINT_FORMAT = "manorl.skrl.ppo.v2"
GYM_SOURCE_FORMAT = "isaacgym.rl-games.mano_hand.v1"
GYM_REWARD_CONTRACT = "isaacgym_mano_reward_contact_1x_threshold_2n_v1"
GYM_PPO_REWARD_CONTRACT = "isaacgym_mano_reward_contact_1x_threshold_2n_shaper_0p5_v1"
GYM_CONVERTED_ENVIRONMENT_CONTRACT = "isaacgym_mano_film_dynamic_residual_eval_v1"
_REQUIRED_MODULES = frozenset({"policy", "value", "optimizer", "observation_preprocessor", "value_preprocessor"})


class CheckpointFormatError(ValueError):
    """Raised when a checkpoint does not satisfy the native target contract."""


def _validate_finite_tensors(value: Any, *, path: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise CheckpointFormatError(f"checkpoint tensor {path!r} contains NaN or Inf")
        return
    if isinstance(value, dict):
        for name, item in value.items():
            _validate_finite_tensors(item, path=f"{path}.{name}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_tensors(item, path=f"{path}[{index}]")


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _load_modules(path: Path, *, device: str | torch.device) -> dict[str, Any]:
    try:
        modules = torch.load(path, map_location=device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckpointFormatError(f"cannot read checkpoint: {path}") from exc
    if not isinstance(modules, dict):
        raise CheckpointFormatError("native skrl checkpoint must be a module mapping")
    if "model" in modules or "env_state" in modules:
        raise CheckpointFormatError(
            "raw rl-games checkpoint input is unsupported: convert it with tools/convert_gym_checkpoint.py first"
        )
    missing = _REQUIRED_MODULES - modules.keys()
    if missing:
        raise CheckpointFormatError(f"checkpoint is missing native skrl modules: {sorted(missing)}")
    _validate_finite_tensors(modules, path="modules")
    return modules


def save_skrl_checkpoint(agent: "PPO", path: str | Path, *, runtime_config: dict[str, object]) -> Path:
    """Save the native agent state and a format/configuration sidecar."""

    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(checkpoint))
    payload = {
        "format": CHECKPOINT_FORMAT,
        "reward_contract": REWARD_CONTRACT_ID,
        "ppo_reward_contract": PPO_REWARD_CONTRACT_ID,
        "environment_contract": ENVIRONMENT_CONTRACT_ID,
        "runtime_config": runtime_config,
        "checkpoint_file": checkpoint.name,
    }
    _metadata_path(checkpoint).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checkpoint


def _load_metadata(checkpoint: Path) -> dict[str, Any]:
    metadata_file = _metadata_path(checkpoint)
    if not metadata_file.is_file():
        raise CheckpointFormatError("native skrl checkpoint sidecar is required")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointFormatError("checkpoint sidecar is not valid JSON") from exc
    if metadata.get("format") != CHECKPOINT_FORMAT or metadata.get("checkpoint_file") != checkpoint.name:
        raise CheckpointFormatError(
            "checkpoint sidecar does not describe this native ManoRL skrl v2 checkpoint"
        )
    for field, display_name in (
        ("reward_contract", "reward contract"),
        ("ppo_reward_contract", "PPO reward contract"),
    ):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise CheckpointFormatError(f"checkpoint {display_name} is missing")
    return metadata


def _validate_environment_contract(
    metadata: dict[str, Any], *, allow_converted: bool = False
) -> None:
    environment_contract = metadata.get("environment_contract")
    if not isinstance(environment_contract, str) or not environment_contract:
        raise CheckpointFormatError("checkpoint environment contract is missing")
    if allow_converted and metadata.get("source") is not None:
        return
    if environment_contract != ENVIRONMENT_CONTRACT_ID:
        raise CheckpointFormatError(
            f"checkpoint environment contract {environment_contract!r} != required {ENVIRONMENT_CONTRACT_ID!r}"
        )


def checkpoint_runtime_metadata(path: str | Path) -> dict[str, Any]:
    """Read and validate the native sidecar without loading agent modules."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointFormatError(f"checkpoint does not exist: {checkpoint}")
    metadata = _load_metadata(checkpoint)
    _validate_conversion_metadata(metadata)
    _validate_environment_contract(metadata, allow_converted=True)
    return metadata


def _validate_model_compatibility(metadata: dict[str, Any], agent: "PPO") -> None:
    runtime_config = metadata.get("runtime_config")
    if not isinstance(runtime_config, dict):
        return
    model_config = runtime_config.get("model")
    expected_film = model_config.get("use_film") if isinstance(model_config, dict) else None
    if expected_film is None:
        return
    policy = getattr(agent, "policy", None)
    model_film = getattr(policy, "use_film", None)
    if model_film is not None and bool(model_film) != bool(expected_film):
        raise CheckpointFormatError(
            f"checkpoint requires use_film={bool(expected_film)}, but target runtime has use_film={bool(model_film)}"
        )


def _validate_conversion_metadata(metadata: dict[str, Any]) -> None:
    """Validate provenance shape without freezing one migration's runtime values."""

    source = metadata.get("source")
    if source is None:
        return
    if not isinstance(source, dict):
        raise CheckpointFormatError("converted checkpoint source provenance must be a mapping")
    if "format" in source and (
        not isinstance(source["format"], str) or not source["format"]
    ):
        raise CheckpointFormatError("converted checkpoint source format is invalid")
    if "path" in source and (
        not isinstance(source["path"], str) or not source["path"]
    ):
        raise CheckpointFormatError("converted checkpoint source path is invalid")
    if "sha256" in source and (
        not isinstance(source["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None
    ):
        raise CheckpointFormatError("converted checkpoint source SHA256 is invalid")
    for field in ("checkpoint_epoch", "checkpoint_frame"):
        if field in source and type(source[field]) is not int:
            raise CheckpointFormatError(f"converted checkpoint {field} provenance is invalid")
    if not source:
        raise CheckpointFormatError("converted checkpoint source provenance is empty")
    if not isinstance(metadata.get("conversion"), dict):
        raise CheckpointFormatError("converted checkpoint conversion metadata is missing")
    if not isinstance(metadata.get("runtime_config"), dict):
        raise CheckpointFormatError("converted checkpoint runtime configuration is missing")


def load_skrl_checkpoint_for_inference(agent: "PPO", path: str | Path) -> Path:
    """Load a native checkpoint for visualization under the current environment contract."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointFormatError(f"checkpoint does not exist: {checkpoint}")
    _load_modules(checkpoint, device=agent.device)
    metadata = _load_metadata(checkpoint)
    _validate_conversion_metadata(metadata)
    _validate_environment_contract(metadata, allow_converted=True)
    _validate_model_compatibility(metadata, agent)
    agent.load(str(checkpoint))
    return checkpoint


def load_skrl_checkpoint(agent: "PPO", path: str | Path) -> Path:
    """Load a native checkpoint only when it matches the current training objective."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointFormatError(f"checkpoint does not exist: {checkpoint}")
    _load_modules(checkpoint, device=agent.device)
    metadata = _load_metadata(checkpoint)
    _validate_conversion_metadata(metadata)
    reward_contract = metadata["reward_contract"]
    if reward_contract != REWARD_CONTRACT_ID:
        raise CheckpointFormatError(
            f"checkpoint reward contract {reward_contract!r} != required {REWARD_CONTRACT_ID!r}"
        )
    ppo_reward_contract = metadata["ppo_reward_contract"]
    if ppo_reward_contract != PPO_REWARD_CONTRACT_ID:
        raise CheckpointFormatError(
            f"checkpoint PPO reward contract {ppo_reward_contract!r} != required {PPO_REWARD_CONTRACT_ID!r}"
        )
    _validate_environment_contract(metadata)
    _validate_model_compatibility(metadata, agent)
    agent.load(str(checkpoint))
    return checkpoint
