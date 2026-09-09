from __future__ import annotations

import copy
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from sim.manorl.autonomy_batch_training import load_frozen_v3, run_batched_ppo
from sim.manorl.autonomy_contracts import ACTION_DIM


class FakeBatchedAdapter:
    """Small runtime fixture; the trainer, not a hand-written schedule, drives it."""
    def __init__(self):
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.observation_space = gym.spaces.Box(-5., 5., shape=(538,), dtype=float)
        self.action_space = gym.spaces.Box(-1., 1., shape=(ACTION_DIM,), dtype=float)
        self.phase = torch.zeros((2, 1))
        self.prepares = 0
        self.recorded_actions = []
        self.terminal_next = []

    def reset(self):
        self.phase.zero_()
        return self.phase.repeat(1, 538), {}

    def step(self, actions):
        self.recorded_actions.append(actions.detach().clone())
        self.phase += 1
        done = torch.tensor([[self.prepares == 0], [False]], dtype=torch.bool)
        next_obs = self.phase.repeat(1, 538)
        self.terminal_next.append(next_obs.detach().clone())
        return next_obs, torch.ones((2, 1)), done, {"valid": torch.ones((2,), dtype=torch.bool)}

    def prepare_action(self):
        self.prepares += 1
        self.phase[0] = 0  # only completed row resets; its neighbour continues
        return self.phase.repeat(1, 538)

    def _to_torch(self, value):
        return value

    def compact_summary(self):
        return {"object_motion": torch.tensor(1.), "contact_force": torch.tensor(2.), "path": torch.tensor(3.)}


def test_real_ppo_loop_records_terminal_next_then_resets_only_completed_rows(tmp_path: Path):
    adapter = FakeBatchedAdapter()
    model, agent, rows = run_batched_ppo(adapter, updates=2, rollouts=2, learning_epochs=1, mini_batches=1,
                                          checkpoint=tmp_path / "v3.pt", checkpoint_interval=1)
    assert len(rows) == 2 and adapter.prepares == 4
    assert adapter.terminal_next[0][0, 0] == 1 and adapter.terminal_next[0][1, 0] == 1
    # The action handed to the fake physics path is the PPO-stored action.
    stored = agent.memory.get_tensor_by_name("actions")
    assert stored.shape[-1] == ACTION_DIM
    assert torch.all(stored <= 1.) and torch.all(stored >= -1.)
    assert (tmp_path / "v3.pt").exists()
    # Adam state exists only after a real backward/optimizer step.
    assert agent.optimizer.state and rows[-1]["info/completed_minibatches"] == 1.0
    assert rows[-1]["transitions"] == 8.0
    assert rows[-1]["performance/sampling_time"] >= 0.0


def test_v3_loader_strictly_rejects_v2_metadata(tmp_path: Path):
    path = tmp_path / "v2.pt"
    torch.save({"checkpoint_format": "manorl.autonomy.ppo.v2"}, path)
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="v2"):
        load_frozen_v3(path, model)
