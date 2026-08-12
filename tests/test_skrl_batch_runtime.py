from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from sim.dexhandrl.skrl_batch_runtime import DLPackBridge, _EpisodeTelemetry, _capacity_guard, _consume_tracking, run_batch_ppo
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
    reason: np.ndarray | None = None
    components: np.ndarray | None = None


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
        return {"requested_global_contact_capacity": 320,
                "requested_ccd_capacity": 192,
                "requested_per_world_constraint_capacity": 512,
                "actual_contact_buffer_shape": (100, 4), "contact_high_water": 2,
                "actual_ccd_capacity": 192, "actual_ncollision_shape": (1,), "ncollision_high_water": 2,
                "actual_constraint_buffer_shape": (2, 100), "constraint_high_water": 2}


class _Agent:
    def __init__(self, order):
        self.order = order
        self.records = []
        self.posts = []
        self.policy = torch.nn.Linear(1, 1)
        self.value = torch.nn.Linear(1, 1)
        self.init_calls = 0

    def init(self): self.init_calls += 1

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


def test_capacity_guard_fails_at_eighty_percent_of_actual_ccd_capacity():
    with __import__("pytest").raises(RuntimeError, match="CCD buffer reached 80%"):
        _capacity_guard({"actual_ccd_capacity": 64, "ncollision_high_water": 52})
    _capacity_guard({"actual_ccd_capacity": 64, "ncollision_high_water": 51})


def test_production_bridge_accepts_cuda_labeled_jax_buffers():
    buffer = SimpleNamespace(dtype="float32", flags=None, device=SimpleNamespace(platform="cuda", id=0))
    DLPackBridge(production_gpu=True)._check_jax(buffer)


def test_cpu_fake_loop_records_terminal_before_next_reset_and_checkpoints(tmp_path):
    cfg = SkrlLauncherConfig(output_dir=tmp_path, updates=2, rollouts=2, checkpoint_interval_updates=1)
    env = _Env()
    agent = _Agent(env.order)
    result = run_batch_ppo(env, agent, cfg, model_config=_ModelConfig(), assignment=[] , bridge=_CpuBridge())

    assert result.updates == 2
    assert result.transitions == 8
    assert agent.posts == [1, 2, 3, 4]
    assert agent.init_calls == 1
    assert env.reset_masks[0] == [1.0, 0.0]
    assert env.order.index("record0") < env.order.index("reset_done") < env.order.index("pre1")
    torch.testing.assert_close(agent.records[0]["next_observations"], torch.tensor([[100.], [200.]]))
    assert agent.records[0]["infos"]["time_outs"].tolist() == [[False], [False]]
    assert (tmp_path / "checkpoint-000001.pt").is_file()
    assert (tmp_path / "checkpoint-000002.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    sidecar = __import__("json").loads((tmp_path / "checkpoint-000001.json").read_text())
    assert sidecar["reward"]["ppo_reward_scale"] == 0.5
    assert sidecar["dataset"]["version"] == 30
    assert sidecar["assets"]["collision_mesh_count"] == 1
    assert sidecar["assets"]["visual_mesh"]["sha256"]
    assert sidecar["code"]["revision"] and isinstance(sidecar["code"]["dirty"], bool)
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


def test_reset_mask_borrow_survives_until_reset_output_and_actions_are_contiguous(tmp_path):
    class BorrowCheckingBridge(_CpuBridge):
        def __init__(self, env):
            self.env = env
            self.action_was_contiguous = False

        def torch_to_jax(self, value):
            if tuple(value.shape) == (2, 22):
                self.action_was_contiguous = value.is_contiguous()
            return value

        def jax_to_torch(self, value):
            borrowed = getattr(self.env, "borrowed_mask", None)
            if borrowed is not None:
                torch.testing.assert_close(borrowed, torch.tensor([1., 0.]))
            return super().jax_to_torch(value)

    class BorrowEnv(_Env):
        def reset_done_device(self, mask):
            self.borrowed_mask = mask
            return super().reset_done_device(mask)

    cfg = SkrlLauncherConfig(output_dir=tmp_path, updates=1, rollouts=2)
    env = BorrowEnv()
    agent = _Agent(env.order)
    agent.act = lambda observations, *_args, **_kwargs: (torch.zeros((22, 2), dtype=torch.float32).t(), None)
    bridge = BorrowCheckingBridge(env)
    run_batch_ppo(env, agent, cfg, model_config=_ModelConfig(), assignment=[], bridge=bridge)

    assert bridge.action_was_contiguous
    torch.testing.assert_close(env.borrowed_mask, torch.tensor([1., 0.]))


def test_nonfinite_agent_parameters_fail_after_update(tmp_path):
    cfg = SkrlLauncherConfig(output_dir=tmp_path, updates=1, rollouts=1)
    env = _Env()
    agent = _Agent(env.order)
    agent.policy.weight.data.fill_(float("nan"))
    import pytest

    with pytest.raises(RuntimeError, match="non-finite agent model parameters after PPO update 1"):
        run_batch_ppo(env, agent, cfg, model_config=_ModelConfig(), assignment=[], bridge=_CpuBridge())


def test_progress_honors_log_cadence_but_emits_checkpoint_boundary(tmp_path):
    cfg = SkrlLauncherConfig(output_dir=tmp_path, updates=3, rollouts=1, log_interval_updates=2, checkpoint_interval_updates=3)
    env = _Env()
    agent = _Agent(env.order)
    run_batch_ppo(env, agent, cfg, model_config=_ModelConfig(), assignment=[], bridge=_CpuBridge())

    records = [__import__("json").loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [record["update"] for record in records] == [1, 2, 3]
    assert (tmp_path / "checkpoint-000001.pt").is_file()
    assert all(record["contact_high_water"] == 2 and record["constraint_high_water"] == 2 for record in records)
    assert all(record["capacity"]["requested_global_contact_capacity"] == 320 for record in records)
    assert all(record["capacity"]["requested_ccd_capacity"] == 192 for record in records)
    assert all(record["capacity"]["actual_ccd_capacity"] == 192 for record in records)
    assert all(record["capacity"]["requested_per_world_constraint_capacity"] == 512 for record in records)
    assert all(record["metrics"]["performance/step_fps"] * record["metrics"]["performance/step_time"] == __import__("pytest").approx(2.0) for record in records)


def test_parser_accepts_action_all_and_batch_checkpoint_options():
    args = build_arg_parser().parse_args(["--action", "all", "--updates", "4", "--start-update", "2", "--checkpoint-interval-updates", "3"])
    assert (args.action, args.updates, args.start_update, args.checkpoint_interval_updates) == ("all", 4, 2, 3)


def test_resume_config_rejects_missing_or_invalid_bounds(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="start_update requires"):
        SkrlLauncherConfig(output_dir=tmp_path, updates=2, start_update=1)
    with pytest.raises(ValueError, match="less than total"):
        SkrlLauncherConfig(output_dir=tmp_path, updates=2, start_update=2, checkpoint=tmp_path / "checkpoint-000002.pt")


def test_episode_metrics_keep_raw_and_bootstrapped_shaped_rewards_separate(tmp_path):
    class TelemetryEnv(_Env):
        def __init__(self):
            super().__init__()
            self.batch = SimpleNamespace(action_indices=[0, 1])

        def step_device(self, actions):
            transition = super().step_device(actions)
            return _Transition(transition.observation, np.asarray([1., 2.], np.float32),
                np.asarray([True, False]), np.asarray([False, True]), np.ones(2, bool),
                np.asarray([1, 2], np.int32), np.arange(20, dtype=np.float32).reshape(2, 10))
    class ShapingAgent(_Agent):
        def record_transition(self, **kwargs):
            kwargs["rewards"].add_(10.)
            super().record_transition(**kwargs)
    cfg = SkrlLauncherConfig(output_dir=tmp_path, updates=1, rollouts=1)
    run_batch_ppo(TelemetryEnv(), ShapingAgent([]), cfg, model_config=_ModelConfig(), assignment=[], bridge=_CpuBridge())
    metrics = __import__("json").loads((tmp_path / "progress.jsonl").read_text())["metrics"]
    assert metrics["rewards"] == 1.5 and metrics["shaped_rewards"] == 11.5
    assert metrics["distance_x"] == 5.0
    assert metrics["distance_x_instant"] == 5.0
    assert metrics["distance_x/object_cube1"] == 5.0
    assert metrics["distance_y"] == 6.0
    assert metrics["distance_y_instant"] == 6.0
    assert metrics["reward_by_pair/cube1_01/total_reward"] == 1.0
    assert metrics["reward_by_pair_instant/cube1_01/total_reward"] == 1.0
    assert metrics["reward_by_pair/cube1_02/total_reward"] == 2.0
    assert "reward_by_pair/cube1_03/total_reward" not in metrics
    assert metrics["training/termination_rates/failure"] == .5
    assert metrics["training/termination_rates/success"] == .5
    assert metrics["components_consistency_gap"] == -93.5
    assert "x" not in metrics


def test_resume_runs_only_remaining_global_timesteps_and_validates_sidecar(tmp_path):
    checkpoint = tmp_path / "checkpoint-000001.pt"; checkpoint.write_bytes(b"placeholder")
    checkpoint.with_suffix(".json").write_text('{"update": 1}', encoding="utf-8")
    cfg = SkrlLauncherConfig(output_dir=tmp_path, checkpoint=checkpoint, updates=2, start_update=1, rollouts=2)
    env = _Env(); agent = _Agent(env.order)
    result = run_batch_ppo(env, agent, cfg, model_config=_ModelConfig(), assignment=[], bridge=_CpuBridge())
    assert result.updates == 1 and result.transitions == 4
    assert [record["timestep"] for record in agent.records] == [2, 3]
    assert (tmp_path / "checkpoint-000002.pt").is_file()


def test_tensorboard_uses_global_transition_axis_and_tracking_is_cleared(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    class TrackingAgent(_Agent):
        def __init__(self, order):
            super().__init__(order); self.tracking_data = {"Loss / Policy loss": [2.0], "Loss / Value loss": [12.0], "Learning / Learning rate": [0.0001]}
    logdir = tmp_path / "tb"
    cfg = SkrlLauncherConfig(output_dir=tmp_path, tensorboard_logdir=logdir, updates=1, rollouts=1)
    agent = TrackingAgent([])
    run_batch_ppo(_Env(), agent, cfg, model_config=_ModelConfig(), assignment=[], bridge=_CpuBridge())
    events = EventAccumulator(str(logdir)); events.Reload()
    assert events.Scalars("losses/a_loss")[0].value == 2.0
    assert events.Scalars("losses/c_loss")[0].value == 3.0
    assert events.Scalars("info/lr_mul")[0].value == 1.0
    assert events.Scalars("rewards/step")[0].step == 2
    assert events.Scalars("global_step")[0].value == 2
    assert agent.tracking_data == {}


def test_standard_skrl_tracking_map_unscales_critic_and_uses_scheduler_inputs():
    class Scheduler:
        def consume_kl_inputs(self):
            return [0.1, 0.3]

    policy = SimpleNamespace(
        log_std=torch.full((22,), -0.99),
        backbone=SimpleNamespace(fixed_sigma=True),
    )
    agent = SimpleNamespace(
        policy=policy,
        scheduler=Scheduler(),
        tracking_data={
            "Loss / Policy loss": [2.0],
            "Loss / Value loss": [12.0],
            "Learning / Learning rate": [1e-4],
        },
    )
    metrics = _consume_tracking(agent, SimpleNamespace(value_loss_scale=4.0, ratio_clip=.2))
    assert metrics["losses/a_loss"] == 2.0
    assert metrics["losses/c_loss"] == 3.0
    assert metrics["losses/bounds_loss"] == 0.0
    assert metrics["info/kl"] == .2
    assert metrics["info/lr_mul"] == 1.0 and metrics["info/e_clip"] == .2
    assert metrics["losses/entropy"] == __import__("pytest").approx(22 * (-.99 + .5 * (1 + np.log(2 * np.pi))))
    assert agent.tracking_data == {}


def test_fixed_entropy_requires_actual_fixed_backbone_and_22d_log_std():
    from sim.dexhandrl.skrl_batch_runtime import _fixed_gaussian_entropy

    assert _fixed_gaussian_entropy(SimpleNamespace(log_std=torch.zeros(22), backbone=SimpleNamespace(fixed_sigma=False))) is None
    assert _fixed_gaussian_entropy(SimpleNamespace(log_std=torch.zeros(21), backbone=SimpleNamespace(fixed_sigma=True))) is None
    assert _fixed_gaussian_entropy(SimpleNamespace(log_std=torch.zeros(22), backbone=SimpleNamespace(fixed_sigma=True))) is not None


def test_ppo_reward_shaper_applies_zero_point_five_in_place():
    from sim.dexhandrl.skrl_train import _reward_shaper_0_5

    rewards = torch.tensor([1.0, -2.0])
    shaped = _reward_shaper_0_5(rewards, 0, 1)
    assert shaped is rewards
    torch.testing.assert_close(rewards, torch.tensor([0.5, -1.0]))


def test_update_time_ends_before_validation_and_telemetry(tmp_path):
    import time

    class TimedEnv(_Env):
        def telemetry(self):
            time.sleep(.08)
            return super().telemetry()

    class TimedAgent(_Agent):
        def post_interaction(self, **kwargs):
            time.sleep(.01)
            super().post_interaction(**kwargs)

    cfg = SkrlLauncherConfig(output_dir=tmp_path, updates=1, rollouts=1)
    run_batch_ppo(TimedEnv(), TimedAgent([]), cfg, model_config=_ModelConfig(), assignment=[], bridge=_CpuBridge())
    metrics = __import__("json").loads((tmp_path / "progress.jsonl").read_text())["metrics"]
    assert .005 < metrics["performance/rl_update_time"] < .05
    assert (tmp_path / "metrics_schema.json").read_text().find('"games_to_track": 100') >= 0


def test_core_reward_window_is_rolling_while_components_are_current_update_samples():
    telemetry = _EpisodeTelemetry(1, [0], torch.device("cpu"))
    for reward in range(1, 102):
        telemetry.step(
            torch.tensor([float(reward)]), torch.tensor([float(reward)]),
            torch.tensor([[float(reward)] + [0.] * 9]), torch.tensor([True]), torch.tensor([1]),
        )
        metrics = telemetry.flush()
    assert metrics["rewards"] == __import__("pytest").approx(51.5)
    assert metrics["distance_x"] == 101.0
    assert "rewards/cube1_02" not in metrics


def test_all_action_requires_explicit_updates_before_runtime_construction():
    import pytest
    from sim.dexhandrl.skrl_train import main

    with pytest.raises(SystemExit, match="action=all requires explicit --updates"):
        main(["--action", "all"])


def test_initialization_ownership_keeps_single_action_trainer_in_charge():
    import inspect
    from sim.dexhandrl import skrl_batch_runtime, skrl_train

    agent_source = inspect.getsource(skrl_train._build_agent)
    runtime_source = inspect.getsource(skrl_batch_runtime.run_batch_ppo)
    assert ".init(" not in agent_source
    assert "agent = PPO(" in agent_source
    assert "DexHandTelemetryKLAdaptiveLR" in agent_source
    assert "RlGamesPPO" not in agent_source
    assert "init()" in runtime_source
    assert "init(trainer_cfg" not in runtime_source
