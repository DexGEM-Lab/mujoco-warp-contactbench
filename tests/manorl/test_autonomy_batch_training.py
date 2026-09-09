from __future__ import annotations

from pathlib import Path

import sim.manorl.autonomy_batch_training as batch_training

import gymnasium as gym
import pytest
import torch

from sim.manorl.autonomy_batch_training import load_frozen_v3, run_batched_ppo
from sim.manorl.autonomy_contracts import ACTION_DIM, ACTION_V3_CONTRACT_ID, CHECKPOINT_V3_FORMAT, OBSERVATION_V3_CONTRACT_ID, REWARD_V3_CONTRACT_ID
from sim.manorl.autonomy_imitation import (
    fit_policy_mean, imitation_checkpoint_payload, teacher_command_pursuit_action,
    validate_teacher_train_identity,
)
from sim.manorl.autonomy_training import AutonomyActorCritic


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


def test_v31_loader_rejects_wrong_provenance_before_model_load(tmp_path: Path):
    path = tmp_path / "wrong-package.pt"
    torch.save({"checkpoint_format": CHECKPOINT_V3_FORMAT, "observation_contract": OBSERVATION_V3_CONTRACT_ID,
                "reward_contract": REWARD_V3_CONTRACT_ID, "action_contract": ACTION_V3_CONTRACT_ID,
                "provenance": {"package_digest": "wrong"}, "model": {}}, path)
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_frozen_v3(path, torch.nn.Linear(1, 1), expected_provenance={"package_digest": "expected"})


def test_installed_gaussian_mixin_preserves_logprob_for_clipped_actions():
    # This is the production mixin path.  Saturation must not create a PPO
    # old/new log-probability mismatch when the policy is unchanged.
    from skrl.models.torch import GaussianMixin, Model

    class SaturatingGaussian(GaussianMixin, Model):
        def __init__(self):
            Model.__init__(self, observation_space=gym.spaces.Box(-1., 1., shape=(1,), dtype=float),
                           action_space=gym.spaces.Box(-1., 1., shape=(1,), dtype=float), device="cpu")
            GaussianMixin.__init__(self, clip_actions=True, clip_mean_actions=False,
                                   clip_log_std=True, min_log_std=-5., max_log_std=2., reduction="sum")

        def compute(self, inputs, role=""):
            return torch.zeros((inputs["observations"].shape[0], 1)), {"log_std": torch.full((1,), 1.25)}

    torch.manual_seed(0)
    policy = SaturatingGaussian()
    observations = torch.zeros((4096, 1))
    actions, old = policy.act({"observations": observations})
    _, new = policy.act({"observations": observations, "taken_actions": actions})
    assert (actions.abs() == 1.).any(dim=1).float().mean() > .7
    torch.testing.assert_close(new["log_prob"], old["log_prob"], rtol=0., atol=0.)


def test_v3_loader_strictly_rejects_v2_metadata(tmp_path: Path):
    path = tmp_path / "v2.pt"
    torch.save({"checkpoint_format": "manorl.autonomy.ppo.v2"}, path)
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="v2"):
        load_frozen_v3(path, model)


def test_pretrain_rejects_holdout_identity():
    with pytest.raises(ValueError, match="known TRAIN"):
        validate_teacher_train_identity("cube2_02_9999", [0], 1)


def test_teacher_targets_are_clipped_rate_increments():
    rate = torch.ones(ACTION_DIM).numpy()
    action = teacher_command_pursuit_action(
        aligned_reference_q_next=torch.full((ACTION_DIM,), 3.).numpy(),
        previous_command=torch.zeros(ACTION_DIM).numpy(), rate_per_second=rate,
        control_timestep=.5,
    )
    assert action.shape == (ACTION_DIM,)
    assert (action == 1.).all()
    # The reference only supplies the target action. The measured-state map
    # remains reference-independent once this action has been chosen.
    from sim.manorl.autonomy_contracts import rate_limited_command
    kwargs = dict(previous_command=torch.zeros(ACTION_DIM).numpy(), action=action,
                  lower=-torch.ones(ACTION_DIM).numpy() * 10, upper=torch.ones(ACTION_DIM).numpy() * 10,
                  rate_per_second=rate, measured_qpos=torch.zeros(ACTION_DIM).numpy(),
                  max_tracking_error=torch.ones(ACTION_DIM).numpy() * 10, control_timestep=.5)
    torch.testing.assert_close(torch.as_tensor(rate_limited_command(**kwargs)), torch.full((ACTION_DIM,), .5, dtype=torch.float64))


def test_imitation_checkpoint_loads_and_warmstarts_without_optimizer_resume(tmp_path: Path):
    space = gym.spaces.Box(-5., 5., shape=(538,), dtype=float)
    action_space = gym.spaces.Box(-1., 1., shape=(ACTION_DIM,), dtype=float)
    source = AutonomyActorCritic(space, action_space, device="cpu")
    checkpoint = tmp_path / "imitation.pt"
    provenance = {"package_digest": "package", "identity_split": {"digest": "split"}, "witness_digest": "witness"}
    torch.save(imitation_checkpoint_payload(model=source, config={}, provenance=provenance,
                                            teacher_config={}, fit_metrics={}), checkpoint)
    target = AutonomyActorCritic(space, action_space, device="cpu")
    load_frozen_v3(checkpoint, target, expected_provenance=provenance)
    for got, expected in zip(target.parameters(), source.parameters()):
        torch.testing.assert_close(got, expected)


def test_ppo_warmstart_validates_and_uses_fresh_optimizer(tmp_path: Path):
    adapter = FakeBatchedAdapter()
    source = AutonomyActorCritic(adapter.observation_space, adapter.action_space, device="cpu")
    provenance = {"package_digest": "package", "identity_split": {"digest": "split"}, "witness_digest": "witness"}
    checkpoint = tmp_path / "init.pt"
    torch.save(imitation_checkpoint_payload(model=source, config={}, provenance=provenance,
                                            teacher_config={}, fit_metrics={}), checkpoint)
    _, agent, _ = run_batched_ppo(adapter, updates=1, rollouts=1, learning_epochs=1, mini_batches=1,
                                  init_checkpoint=checkpoint, provenance={**provenance, "ppo": {"rollouts": 1}})
    assert agent.optimizer.state  # populated only by this new PPO update


def test_actor_mean_fit_reduces_teacher_error():
    torch.manual_seed(0)
    space = gym.spaces.Box(-5., 5., shape=(4,), dtype=float)
    action_space = gym.spaces.Box(-1., 1., shape=(ACTION_DIM,), dtype=float)
    model = AutonomyActorCritic(space, action_space, device="cpu")
    observations = torch.randn(16, 4).numpy().astype("float32")
    targets = torch.zeros((16, ACTION_DIM)).numpy()
    metrics = fit_policy_mean(model, {"observations": observations, "teacher_actions": targets},
                              gradient_steps=30, batch_size=16, learning_rate=1e-2, loss_name="mse")
    assert metrics["final_policy_teacher_mse"] < metrics["initial_policy_teacher_mse"]
