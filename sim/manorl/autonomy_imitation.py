"""Bounded real-physics teacher collection and actor-only imitation initialization.

The teacher is a diagnostic command pursuit rule.  It is used only to produce
an offline initialization dataset: every observation is captured before its
teacher action and the action enters the normal fused rate-limited control map.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from sim.manorl.autonomy_contracts import (
    ACTION_V3_CONTRACT_ID,
    CHECKPOINT_V3_FORMAT,
    OBSERVATION_V3_CONTRACT_ID,
    POLICY_SAMPLING_CONTRACT,
    REWARD_V3_CONTRACT_ID,
)


TEACHER_CONTRACT_ID = "manorl.autonomy.diagnostic_command_pursuit.v1"
IMITATION_TRAINING_MODE = "imitation initialization"


def validate_teacher_train_identity(identity: str, train_indices: list[int], identity_index: int) -> None:
    """Keep the diagnostic dataset strictly on the named training reference."""
    if identity != "cube2_02_2833" or identity_index not in train_indices:
        raise ValueError("pretrain is restricted to known TRAIN identity cube2_02_2833")


def teacher_command_pursuit_action(
    aligned_reference_q_next: np.ndarray,
    previous_command: np.ndarray,
    rate_per_second: np.ndarray,
    control_timestep: float,
) -> np.ndarray:
    """Return the clipped normalized action that pursues the next reference.

    This is intentionally only an action target.  Applying it through the
    runtime still uses measured-state tracking envelopes and physical limits.
    """
    target = np.asarray(aligned_reference_q_next, dtype=np.float32)
    previous = np.asarray(previous_command, dtype=np.float32)
    rate = np.asarray(rate_per_second, dtype=np.float32)
    if target.shape != (28,) or previous.shape != (28,) or rate.shape != (28,):
        raise ValueError("teacher target, previous command and rate must be 28-wide")
    if not np.isfinite(control_timestep) or control_timestep <= 0 or np.any(rate <= 0):
        raise ValueError("teacher timestep and rates must be positive and finite")
    return np.clip((target - previous) / (rate * float(control_timestep)), -1.0, 1.0).astype(np.float32)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def collect_real_teacher_rollout(adapter: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collect one full-start physical rollout under the diagnostic teacher.

    The returned observation row is the live fused-runtime observation before
    the corresponding teacher action. No state/object overwrite occurs.
    """
    observations, _ = adapter.reset()
    runtime = adapter.runtime
    rows: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    commands: list[np.ndarray] = []
    contact_force: list[float] = []
    wrist_error: list[float] = []
    reward: list[float] = []
    done = False
    while not done:
        index = int(_as_numpy(runtime.indices[0]))
        next_index = min(index + 1, runtime.length - 1)
        action = teacher_command_pursuit_action(
            np.asarray(runtime.aligned_q_ref[next_index]),
            _as_numpy(runtime.previous_command[0]),
            np.asarray(runtime.rate),
            runtime.clock.control_timestep,
        )
        # Capture before action: this contains real current physical/contact
        # state plus the reference fields available to the neural policy.
        rows.append(_as_numpy(observations[0]).astype(np.float32, copy=True))
        actions.append(action)
        _, step_reward, terminal, _ = adapter.step(torch.as_tensor(action[None], device=adapter.device))
        commands.append(_as_numpy(runtime.last_command[0]).astype(np.float32, copy=True))
        physical, contact = runtime.last_physical, runtime.last_contact
        contact_force.append(float(np.linalg.norm(_as_numpy(contact.hand_object_forces[0]), axis=-1).max()))
        phase = min(int(_as_numpy(runtime.indices[0])), runtime.length - 1)
        wrist_error.append(float(np.linalg.norm(_as_numpy(physical.mano_dof_pos[0])[:3] - runtime.aligned_q_ref[phase, :3])))
        reward.append(float(_as_numpy(step_reward[0, 0])))
        done = bool(_as_numpy(terminal[0, 0]))
        if not done:
            observations = adapter.prepare_action()
    dataset = {
        "observations": np.stack(rows),
        "teacher_actions": np.stack(actions),
        "executed_commands": np.stack(commands),
    }
    metrics = {
        "frames": len(rows),
        "teacher_wrist_rmse_m": float(np.sqrt(np.mean(np.square(wrist_error)))),
        "teacher_contact_frames_force_gt_0p02N": int(np.count_nonzero(np.asarray(contact_force) > .02)),
        "teacher_max_hand_object_force_N": float(max(contact_force, default=0.0)),
        "teacher_reward_mean": float(np.mean(reward)),
        "teacher_action_saturation_fraction": float(np.mean(np.abs(dataset["teacher_actions"]) >= .999)),
    }
    return dataset, metrics


def fit_policy_mean(
    model: torch.nn.Module,
    dataset: dict[str, np.ndarray],
    *,
    gradient_steps: int = 2000,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    loss_name: str = "huber",
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Fit the existing actor mean head; the critic head receives no loss."""
    if gradient_steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("gradient_steps, batch_size and learning_rate must be positive")
    if loss_name not in {"mse", "huber"}:
        raise ValueError("loss_name must be mse or huber")
    observations = torch.as_tensor(dataset["observations"], dtype=torch.float32, device=device)
    targets = torch.as_tensor(dataset["teacher_actions"], dtype=torch.float32, device=device)
    if observations.ndim != 2 or targets.shape != (observations.shape[0], 28):
        raise ValueError("imitation dataset must have (frames, observation) and (frames, 28) targets")
    # The existing trunk is shared with value prediction; optimizing its policy
    # path is the smallest faithful use of the existing MLP, while the value
    # head stays untouched and PPO creates its own optimizer on warm start.
    parameters = [*model.net.parameters(), *model.mean.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model.train()
    with torch.no_grad():
        initial = model.compute({"observations": observations}, role="policy")[0]
        initial_mse = float(F.mse_loss(initial, targets).detach().cpu())
    losses: list[float] = []
    for _ in range(gradient_steps):
        selection = torch.randint(observations.shape[0], (batch_size,), generator=generator, device="cpu").to(device)
        mean, _ = model.compute({"observations": observations[selection]}, role="policy")
        loss = F.mse_loss(mean, targets[selection]) if loss_name == "mse" else F.huber_loss(mean, targets[selection])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        neural = model.compute({"observations": observations}, role="policy")[0]
        mse = float(F.mse_loss(neural, targets).detach().cpu())
        mae = float(F.l1_loss(neural, targets).detach().cpu())
    return {
        "gradient_steps": float(gradient_steps), "batch_size": float(batch_size), "learning_rate": float(learning_rate),
        "initial_policy_teacher_mse": initial_mse, "final_policy_teacher_mse": mse,
        "final_policy_teacher_mae": mae, "final_batch_loss": losses[-1], "mean_batch_loss": float(np.mean(losses)),
    }


def imitation_checkpoint_payload(*, model: torch.nn.Module, config: dict[str, Any], provenance: dict[str, Any], teacher_config: dict[str, Any], fit_metrics: dict[str, Any]) -> dict[str, Any]:
    """Build a frozen-loader-compatible actor initialization checkpoint."""
    if not hasattr(model, "checkpoint_architecture"):
        raise TypeError("imitation checkpoint model must declare its architecture")
    return {
        "checkpoint_format": CHECKPOINT_V3_FORMAT,
        "observation_contract": OBSERVATION_V3_CONTRACT_ID,
        "reward_contract": REWARD_V3_CONTRACT_ID,
        "action_contract": ACTION_V3_CONTRACT_ID,
        "policy_sampling_contract": dict(POLICY_SAMPLING_CONTRACT),
        "model": model.state_dict(), "model_architecture": model.checkpoint_architecture(),
        "optimizer": None,
        "global_policy_step": 0, "policy_steps": 0, "environment_transitions": 0,
        "config": config,
        "provenance": {**provenance, "training_mode": IMITATION_TRAINING_MODE, "normalizer": None},
        "teacher": {"contract": TEACHER_CONTRACT_ID, **teacher_config},
        "fit_metrics": fit_metrics,
    }


def save_dataset(path: str | Path, dataset: dict[str, np.ndarray]) -> None:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)
