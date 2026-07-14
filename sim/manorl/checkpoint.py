"""Native skrl checkpoint boundary for the ManoRL target runtime.

Isaac rl-games checkpoints are deliberately rejected here. Loading them would
silently assert an unverified parameter, normalizer, and optimizer conversion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from skrl.agents.torch.ppo import PPO


CHECKPOINT_FORMAT = "manorl.skrl.ppo.v1"
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
        "runtime_config": runtime_config,
        "checkpoint_file": checkpoint.name,
    }
    _metadata_path(checkpoint).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checkpoint


def load_skrl_checkpoint(agent: "PPO", path: str | Path) -> Path:
    """Load only a checkpoint produced by :func:`save_skrl_checkpoint`."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointFormatError(f"checkpoint does not exist: {checkpoint}")
    _load_modules(checkpoint, device=agent.device)
    metadata_file = _metadata_path(checkpoint)
    if not metadata_file.is_file():
        raise CheckpointFormatError("native skrl checkpoint sidecar is required")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointFormatError("checkpoint sidecar is not valid JSON") from exc
    if metadata.get("format") != CHECKPOINT_FORMAT or metadata.get("checkpoint_file") != checkpoint.name:
        raise CheckpointFormatError("checkpoint sidecar does not describe this native ManoRL skrl format")
    agent.load(str(checkpoint))
    return checkpoint
