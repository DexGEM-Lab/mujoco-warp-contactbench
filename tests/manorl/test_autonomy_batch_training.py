from __future__ import annotations

from pathlib import Path

import sim.manorl.autonomy_batch_training as batch_training

import gymnasium as gym
import pytest
import torch

from sim.manorl.autonomy_batch_training import load_frozen_v3, run_batched_ppo
from sim.manorl.autonomy_contracts import ACTION_DIM, ACTION_V3_CONTRACT_ID


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
        # Current runtime validity is one global JAX scalar, not a B-vector.
        return next_obs, torch.ones((2, 1)), done, {"valid": torch.tensor(True)}

    def prepare_action(self):
        self.prepares += 1
        self.phase[0] = 0  # only completed row resets; its neighbour continues
        return self.phase.repeat(1, 538)

    def _to_torch(self, value):
        return value

    def compact_summary(self):
        return {"object_motion": torch.tensor(1.), "contact_force": torch.tensor(2.), "path": torch.tensor(3.)}


def test_real_ppo_loop_global_valid_window_metrics_and_streaming_callback(tmp_path: Path, monkeypatch):
    adapter = FakeBatchedAdapter()
    # Four calls/update: sample start/end then optimizer start/end. Each
    # window has 4 B=2 transitions, sample time 2 and optimizer time 1.
    ticks = iter([0., 2., 2., 3.] * 3)
    monkeypatch.setattr(batch_training.time, "perf_counter", lambda: next(ticks))
    observed = []
    model, agent, rows = run_batched_ppo(adapter, updates=3, rollouts=2, learning_epochs=1, mini_batches=1,
                                          checkpoint=tmp_path / "v3.pt", checkpoint_interval=1,
                                          on_update=lambda row: observed.append((row, (tmp_path / f"v3.update{int(row['update']):06d}.pt").exists())))
    assert len(rows) == 3 and adapter.prepares == 6
    assert len(observed) == 3 and all(numbered for _, numbered in observed)
    # Cumulative transitions advance 4/update while all window-normalized
    # metrics remain invariant rather than decaying/growing by update index.
    assert [row["transitions"] for row in rows] == [4., 8., 12.]
    assert all(row["window_transitions"] == 4. and row["reward_mean"] == 1. for row in rows)
    assert all(row["performance/sampling_transitions_per_second"] == 2. for row in rows)
    assert all(row["performance/total_transitions_per_second"] == 4. / 3. for row in rows)
    assert rows[0]["episodes/completed_count"] == 1. and rows[0]["episodes/return_mean"] == 1.
    assert "episodes/completed_count" not in rows[1] # live B=1 episode crosses rollout/update cuts
    assert rows[-1]["policy_steps"] == 6.
    assert rows[-1]["valid"] == 1.
    assert adapter.terminal_next[0][0, 0] == 1 and adapter.terminal_next[0][1, 0] == 1
    assert adapter.terminal_next[0][0, 0] == 1 and adapter.terminal_next[0][1, 0] == 1
    # The action handed to the fake physics path is the PPO-stored action.
    stored = agent.memory.get_tensor_by_name("actions")
    assert stored.shape[-1] == ACTION_DIM
    assert torch.all(stored <= 1.) and torch.all(stored >= -1.)
    assert (tmp_path / "v3.pt").exists()
    # Adam state exists only after a real backward/optimizer step.
    assert agent.optimizer.state and rows[-1]["info/completed_minibatches"] == 1.0
    assert (tmp_path / "v3.pt").exists() and (tmp_path / "v3.final.pt").exists()
    checkpoint = torch.load(tmp_path / "v3.pt", weights_only=False)
    assert checkpoint["policy_steps"] == 6 and checkpoint["environment_transitions"] == 12
    assert checkpoint["global_policy_step"] == 12


def test_v3_loader_rejects_wrong_provenance_before_model_load(tmp_path: Path):
    path = tmp_path / "wrong-package.pt"
    torch.save({"checkpoint_format": "manorl.autonomy.ppo.v3", "observation_contract": "manorl.autonomy.observation.v3",
                "reward_contract": "manorl.autonomy.reward.v3", "action_contract": ACTION_V3_CONTRACT_ID,
                "provenance": {"package_digest": "wrong"}, "model": {}}, path)
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_frozen_v3(path, torch.nn.Linear(1, 1), expected_provenance={"package_digest": "expected"})


def test_v3_loader_strictly_rejects_v2_metadata(tmp_path: Path):
    path = tmp_path / "v2.pt"
    torch.save({"checkpoint_format": "manorl.autonomy.ppo.v2"}, path)
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="v2"):
        load_frozen_v3(path, model)
