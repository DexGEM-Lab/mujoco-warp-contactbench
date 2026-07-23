from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from sim.dexhandrl.skrl_batch_runtime import run_batch_ppo
from sim.dexhandrl.skrl_train import SkrlLauncherConfig, build_arg_parser


class _CpuBridge:
    """Test-only boundary: production uses DLPackBridge."""
    def torch_to_jax(self, value):
        return value.detach().clone().numpy().astype(np.float32)

    def jax_to_torch(self, value):
        return torch.as_tensor(value, dtype=torch.float32).clone()


@dataclass
class _Transition:
    observation: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    valid: np.ndarray


class _Env:
    num_envs = 2
    def __init__(self):
        self.order: list[str] = []
        self.reset_masks: list[list[float]] = []
        self.step_count = 0

    def reset_device(self):
        self.order.append("reset")
        return _Transition(np.array([[0.], [10.]], np.float32), np.zeros(2, np.float32),
                           np.zeros(2, bool), np.zeros(2, bool), np.ones(2, bool))

    def reset_done_device(self, mask):
        self.reset_masks.append(mask.tolist())
        self.order.append("reset_done")
        return _Transition(np.array([[5.], [10.]], np.float32), np.zeros(2, np.float32),
                           np.zeros(2, bool), np.zeros(2, bool), np.ones(2, bool))

    def step_device(self, actions):
        self.order.append(f"step{self.step_count}")
        terminal = np.array([self.step_count == 0, False])
        result = _Transition(np.array([[100. + self.step_count], [200. + self.step_count]], np.float32),
                             np.ones(2, np.float32), terminal, np.zeros(2, bool), np.ones(2, bool))
        self.step_count += 1
        return result

    def telemetry(self):
        return {"actual_contact_buffer_shape": (100, 4), "contact_high_water": 2,
                "actual_constraint_buffer_shape": (2, 100), "constraint_high_water": 2}


class _Agent:
    def __init__(self, order):
        self.order = order
        self.records = []
        self.posts = []

    def pre_interaction(self, *, timestep, **_): self.order.append(f"pre{timestep}")
    def act(self, observations, *_args, timestep, **_):
        self.order.append(f"act{timestep}")
        return torch.zeros((2, 22), dtype=torch.float32), None
    def record_transition(self, **kwargs):
        self.order.append(f"record{kwargs['timestep']}")
        self.records.append(kwargs)
    def post_interaction(self, *, timestep, **_):
        self.order.append(f"post{timestep}"); self.posts.append(timestep)
    def save(self, path): torch.save({"ok": True}, path)


@dataclass
class _ModelConfig:
    name: str = "fake"


def test_cpu_fake_loop_records_terminal_before_next_reset_and_checkpoints(tmp_path):
    cfg = SkrlLauncherConfig(output_dir=tmp_path, updates=2, rollouts=2, checkpoint_interval_updates=1)
    env = _Env()
    agent = _Agent(env.order)
    result = run_batch_ppo(env, agent, cfg, model_config=_ModelConfig(), assignment=[] , bridge=_CpuBridge())

    assert result.updates == 2
    assert result.transitions == 8
    assert agent.posts == [1, 2, 3, 4]
    assert env.reset_masks[0] == [1.0, 0.0]
    assert env.order.index("record0") < env.order.index("reset_done") < env.order.index("pre1")
    torch.testing.assert_close(agent.records[0]["next_observations"], torch.tensor([[100.], [200.]]))
    assert agent.records[0]["infos"]["time_outs"].tolist() == [[False], [False]]
    assert (tmp_path / "checkpoint-000001.pt").is_file()
    assert (tmp_path / "checkpoint-000002.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    assert len((tmp_path / "progress.jsonl").read_text().splitlines()) == 2


def test_real_cpu_two_world_one_update_changes_policy_parameters(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    pytest.importorskip("skrl")
    from sim.dexhandrl.skrl_train import _build_agent, _build_batch_env, _resolve_trajectory_specs

    cfg = SkrlLauncherConfig(action="all", device="cpu", num_envs=2, updates=1, rollouts=2,
                             learning_epochs=1, mini_batches=1, entropy_loss_scale=0.01, kl_threshold=0.0, checkpoint_interval_updates=1,
                             output_dir=tmp_path)
    if not cfg.lance_path.exists():
        pytest.skip("shared Lance archive is unavailable")
    specs = _resolve_trajectory_specs(cfg)
    env, spaces = _build_batch_env(cfg, specs)
    agent, model = _build_agent(spaces, cfg)
    before = [parameter.detach().clone() for model_instance in (agent.policy, agent.value) for parameter in model_instance.parameters()]
    try:
        result = run_batch_ppo(env, agent, cfg, model_config=model, assignment=specs)
    finally:
        env.close()
    assert result.updates == 1
    after = [parameter for model_instance in (agent.policy, agent.value) for parameter in model_instance.parameters()]
    assert all(torch.isfinite(parameter).all() for parameter in after)
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


def test_parser_accepts_action_all_and_batch_checkpoint_options():
    args = build_arg_parser().parse_args(["--action", "all", "--updates", "1", "--checkpoint-interval-updates", "3"])
    assert (args.action, args.updates, args.checkpoint_interval_updates) == ("all", 1, 3)
