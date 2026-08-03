"""Native skrl checkpoint boundary for the MuJoCo ManoRL runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from sim.manorl.abi import ENVIRONMENT_CONTRACT_ID, LEGACY_ENVIRONMENT_CONTRACT_IDS
from sim.manorl.rewards import (
    LEGACY_PPO_REWARD_CONTRACT_IDS,
    LEGACY_REWARD_CONTRACT_IDS,
    PPO_REWARD_CONTRACT_ID,
    REWARD_CONTRACT_ID,
)

if TYPE_CHECKING:
    from skrl.agents.torch.ppo import PPO


CHECKPOINT_FORMAT = "manorl.skrl.ppo.v2"
_REQUIRED_MODULES = frozenset({"policy", "value", "optimizer", "observation_preprocessor", "value_preprocessor"})
_ENVIRONMENT_SIGNATURE_FIELDS = (
    "resolved_hand_side",
    "available_hand_sides",
    "controlled_hand_sides",
    "reference_following_hand_sides",
    "action_dim",
    "observation_dim",
    "model_action_dim",
    "reference_fps",
    "warp_ccd",
)
_ENVIRONMENT_SIDE_SEQUENCE_FIELDS = frozenset(
    {
        "available_hand_sides",
        "controlled_hand_sides",
        "reference_following_hand_sides",
    }
)


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
            "Isaac Gym/rl-games checkpoints are unsupported; use a native 28-DoF MuJoCo checkpoint"
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


def _validate_environment_contract(metadata: dict[str, Any]) -> None:
    environment_contract = metadata.get("environment_contract")
    if not isinstance(environment_contract, str) or not environment_contract:
        raise CheckpointFormatError("checkpoint environment contract is missing")
    supported = {ENVIRONMENT_CONTRACT_ID, *LEGACY_ENVIRONMENT_CONTRACT_IDS}
    if environment_contract not in supported:
        raise CheckpointFormatError(
            f"checkpoint environment contract {environment_contract!r} is not a required supported contract; supported={sorted(supported)!r}"
        )


def checkpoint_runtime_metadata(path: str | Path) -> dict[str, Any]:
    """Read and validate the native sidecar without loading agent modules."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointFormatError(f"checkpoint does not exist: {checkpoint}")
    metadata = _load_metadata(checkpoint)
    _validate_environment_contract(metadata)
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


def _canonical_residual_action(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize tuple/list JSON differences and legacy implicit 1x multipliers."""

    normalized = json.loads(json.dumps(value, sort_keys=True))
    normalized.setdefault("joint_scale_multiplier", 1.0)
    normalized.setdefault("joint_max_offset_multiplier", 1.0)
    return normalized


def _canonical_warp_ccd(value: object) -> object:
    """Remove aggregate capacity, which scales with the target vector batch."""

    normalized = json.loads(json.dumps(value, sort_keys=True))
    if isinstance(normalized, dict):
        normalized.pop("naccdmax", None)
    return normalized


def _validate_environment_signature(metadata: dict[str, Any], agent: "PPO") -> None:
    """Reject side/layout mismatches when a checkpoint records the new fields.

    New sidecars must match every field the target runtime can resolve. Policy
    tensor dimensions provide a fallback for callers which construct an agent
    outside ``ManoSkrlRuntime``; the attached runtime signature additionally
    distinguishes left from right when both policies are 28-wide.
    """

    runtime_config = metadata.get("runtime_config")
    if not isinstance(runtime_config, dict):
        return
    checkpoint_environment = runtime_config.get("environment")
    if not isinstance(checkpoint_environment, dict):
        return

    target = getattr(agent, "manorl_environment_signature", None)
    if isinstance(target, dict):
        target_environment = target
        missing = [
            field
            for field in _ENVIRONMENT_SIGNATURE_FIELDS
            if field not in checkpoint_environment
            and target_environment.get(field) is not None
        ]
        if missing:
            raise CheckpointFormatError(
                "checkpoint environment is missing current MuJoCo hand signature "
                f"fields: {missing}"
            )
    else:
        policy = getattr(agent, "policy", None)
        target_environment = {
            "action_dim": getattr(policy, "action_dim", None),
            "observation_dim": getattr(policy, "observation_dim", None),
        }

    for field in _ENVIRONMENT_SIGNATURE_FIELDS:
        if field not in checkpoint_environment:
            continue
        target_value = target_environment.get(field)
        if target_value is None:
            continue
        checkpoint_value = checkpoint_environment[field]
        if field in _ENVIRONMENT_SIDE_SEQUENCE_FIELDS:
            if not isinstance(checkpoint_value, (list, tuple)) or not isinstance(
                target_value, (list, tuple)
            ):
                raise CheckpointFormatError(
                    f"checkpoint environment {field} is not a hand-side sequence"
                )
            checkpoint_value = tuple(checkpoint_value)
            target_value = tuple(target_value)
        elif field == "warp_ccd":
            checkpoint_value = _canonical_warp_ccd(checkpoint_value)
            target_value = _canonical_warp_ccd(target_value)
        if checkpoint_value != target_value:
            raise CheckpointFormatError(
                f"checkpoint environment {field}={checkpoint_value!r} does not match "
                f"target runtime {target_value!r}"
            )

    checkpoint_residual = checkpoint_environment.get("residual_action")
    target_residual = target_environment.get("residual_action")
    if isinstance(checkpoint_residual, dict) and isinstance(target_residual, dict):
        checkpoint_residual = _canonical_residual_action(checkpoint_residual)
        target_residual = _canonical_residual_action(target_residual)
        if checkpoint_residual != target_residual:
            raise CheckpointFormatError(
                "checkpoint environment residual_action does not match target runtime"
            )


def _validate_inference_reward_contract(metadata: dict[str, Any]) -> None:
    pair = (metadata["reward_contract"], metadata["ppo_reward_contract"])
    supported = {
        (REWARD_CONTRACT_ID, PPO_REWARD_CONTRACT_ID),
        *zip(LEGACY_REWARD_CONTRACT_IDS, LEGACY_PPO_REWARD_CONTRACT_IDS, strict=True),
    }
    if pair not in supported:
        raise CheckpointFormatError(
            f"checkpoint inference reward contracts {pair!r} are not a supported matched pair"
        )


def load_skrl_checkpoint_for_inference(agent: "PPO", path: str | Path) -> Path:
    """Load a current or legacy-reward native checkpoint for inference."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointFormatError(f"checkpoint does not exist: {checkpoint}")
    _load_modules(checkpoint, device=agent.device)
    metadata = _load_metadata(checkpoint)
    _validate_inference_reward_contract(metadata)
    _validate_environment_contract(metadata)
    _validate_model_compatibility(metadata, agent)
    _validate_environment_signature(metadata, agent)
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
    _validate_model_compatibility(metadata, agent)
    _validate_environment_signature(metadata, agent)
    agent.load(str(checkpoint))
    return checkpoint
