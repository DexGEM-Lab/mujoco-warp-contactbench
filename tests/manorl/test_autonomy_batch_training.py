"""v4 PPO failure-boundary regressions."""
from __future__ import annotations

import gymnasium as gym
import pytest
import torch

import sim.manorl.autonomy_batch_training as training
from sim.manorl.autonomy_contracts import ACTION_DIM, RAW_OBSERVATION_DIM
from sim.manorl.autonomy_training import AutonomyActorCritic


class _FiniteAdapter:
    """PPO fixture with raw observations and a terminal/reset boundary."""

    def __init__(self) -> None:
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(RAW_OBSERVATION_DIM,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,))
        self.phase = 0
        self.terminal_next: list[torch.Tensor] = []

    def reset(self):
        self.phase = 0
        return torch.zeros((1, RAW_OBSERVATION_DIM)), {}

    def step(self, actions: torch.Tensor):
        assert actions.shape == (1, ACTION_DIM)
        self.phase += 1
        terminal_next = torch.full((1, RAW_OBSERVATION_DIM), float(self.phase))
        self.terminal_next.append(terminal_next)
        return terminal_next, torch.ones((1, 1)), torch.tensor([[self.phase == 1]]), {"valid": torch.tensor(True)}

    def prepare_action(self):
        if self.phase == 1:
            self.phase = 0
        return torch.full((1, RAW_OBSERVATION_DIM), float(self.phase))

    def _to_torch(self, value):
        return value

    def compact_summary(self):
        return {name: torch.tensor(0.0) for name in ("object_motion", "contact_force", "path")}


def test_v4_raw_normal_logprob_ratio_is_identity_before_update():
    model = AutonomyActorCritic(
        gym.spaces.Box(-1.0, 1.0, shape=(RAW_OBSERVATION_DIM,)),
        gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,)),
        device="cpu", clip_actions=False,
    )
    with torch.no_grad():
        model.mean.bias.fill_(4.0)
        model.log_std.fill_(-4.6051702)
    observations = torch.zeros((2, RAW_OBSERVATION_DIM))
    actions, old = model.act({"observations": observations}, role="policy")
    _, replay = model.act({"observations": observations, "taken_actions": actions}, role="policy")
    assert (actions.abs() > 1.0).any()
    torch.testing.assert_close(torch.exp(replay["log_prob"] - old["log_prob"]), torch.ones_like(old["log_prob"]), rtol=1e-6, atol=1e-6)


def test_v4_nonfinite_update_writes_diagnostic_and_withholds_final_checkpoint(tmp_path, monkeypatch):
    original_builder = training.build_batched_runtime
    updates = 0

    def poison_after_second_update(*args, **kwargs):
        model, agent = original_builder(*args, **kwargs)
        original_update = agent.update

        def update(*update_args, **update_kwargs):
            nonlocal updates
            original_update(*update_args, **update_kwargs)
            updates += 1
            if updates == 2:
                next(model.parameters()).data.fill_(float("nan"))

        agent.update = update
        return model, agent

    monkeypatch.setattr(training, "build_batched_runtime", poison_after_second_update)
    checkpoint = tmp_path / "finite.pt"
    with pytest.raises(RuntimeError, match="non-finite PPO update 2"):
        training.run_batched_ppo(
            _FiniteAdapter(), updates=2, rollouts=2, learning_epochs=1, mini_batches=1,
            checkpoint=checkpoint, checkpoint_interval=1,
        )
    assert (tmp_path / "finite.update000001.pt").exists()
    diagnostic = tmp_path / "finite.nonfinite-update000002.json"
    assert diagnostic.exists()
    assert "model." in diagnostic.read_text()
    assert checkpoint.exists()  # update 1 remains the last known-finite checkpoint
    assert not (tmp_path / "finite.final.pt").exists()
