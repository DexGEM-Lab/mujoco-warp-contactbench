"""Native skrl checkpoint boundary for the ManoRL target runtime.

Isaac rl-games checkpoints are deliberately rejected here. Loading them would
silently assert an unverified parameter, normalizer, and optimizer conversion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from sim.manorl.abi import ENVIRONMENT_CONTRACT_ID
from sim.manorl.rewards import PPO_REWARD_CONTRACT_ID, REWARD_CONTRACT_ID

if TYPE_CHECKING:
    from skrl.agents.torch.ppo import PPO


CHECKPOINT_FORMAT = "manorl.skrl.ppo.v2"
_REQUIRED_MODULES = frozenset({"policy", "value", "optimizer", "observation_preprocessor", "value_preprocessor"})


class CheckpointFormatError(ValueError):
    """Raised when a checkpoint does not satisfy the native target contract."""


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
            "rl-games checkpoint input is unsupported: no parameter, normalizer, or optimizer conversion is implemented"
        )
    missing = _REQUIRED_MODULES - modules.keys()
    if missing:
        raise CheckpointFormatError(f"checkpoint is missing native skrl modules: {sorted(missing)}")
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
            "checkpoint sidecar does not describe the native ManoRL skrl v2 raw-1.0x reward format; "
            "legacy 0.5x-reward checkpoints cannot load"
        )
    for field, display_name in (
        ("reward_contract", "reward contract"),
        ("ppo_reward_contract", "PPO reward contract"),
    ):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise CheckpointFormatError(f"checkpoint {display_name} is missing")
    return metadata


def _validate_environment_contract(metadata: dict[str, Any]) -> None:
    environment_contract = metadata.get("environment_contract")
    if not isinstance(environment_contract, str) or not environment_contract:
        raise CheckpointFormatError("checkpoint environment contract is missing")
    if environment_contract != ENVIRONMENT_CONTRACT_ID:
        raise CheckpointFormatError(
            f"checkpoint environment contract {environment_contract!r} != required {ENVIRONMENT_CONTRACT_ID!r}"
        )


def load_skrl_checkpoint_for_inference(agent: "PPO", path: str | Path) -> Path:
    """Load a native checkpoint for visualization under the current environment contract."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointFormatError(f"checkpoint does not exist: {checkpoint}")
    _load_modules(checkpoint, device=agent.device)
    metadata = _load_metadata(checkpoint)
    _validate_environment_contract(metadata)
    agent.load(str(checkpoint))
    return checkpoint


def load_skrl_checkpoint(agent: "PPO", path: str | Path) -> Path:
    """Load a native checkpoint only when it matches the current training objective."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointFormatError(f"checkpoint does not exist: {checkpoint}")
    _load_modules(checkpoint, device=agent.device)
    metadata = _load_metadata(checkpoint)
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
    agent.load(str(checkpoint))
    return checkpoint
