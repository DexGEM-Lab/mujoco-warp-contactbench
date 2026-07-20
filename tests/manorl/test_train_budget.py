from __future__ import annotations

import importlib.util
import io
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _load_tool():
    project_root = Path(__file__).parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    path = project_root / "tools" / "train_manorl_cube1.py"
    spec = importlib.util.spec_from_file_location("train_manorl_cube1_budget_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClock:
    def __init__(self, timestamps: list[float]) -> None:
        self.timestamps = iter(timestamps)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return next(self.timestamps)


class FakeAgent:
    device = "cpu"

    def __init__(self) -> None:
        self.act_calls = 0
        self.recorded_timesteps: list[int] = []
        self.post_interaction_timesteps: list[int] = []
        self.training_mode = False
        self.loaded_checkpoint: str | None = None

    def enable_training_mode(self, enabled: bool) -> None:
        self.training_mode = enabled

    def act(self, observations: torch.Tensor, *_: object, **__: object) -> tuple[torch.Tensor, None]:
        self.act_calls += 1
        return torch.ones_like(observations), None

    def record_transition(self, *, timestep: int, **_: object) -> None:
        self.recorded_timesteps.append(timestep)

    def post_interaction(self, *, timestep: int, **_: object) -> None:
        self.post_interaction_timesteps.append(timestep)

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": {},
                "value": {},
                "optimizer": {},
                "observation_preprocessor": {},
                "value_preprocessor": {},
            },
            path,
        )

    def load(self, path: str) -> None:
        self.loaded_checkpoint = path


class FakeEnv:
    def __init__(self, environment: SimpleNamespace) -> None:
        self.environment = environment
        self.step_calls = 0

    def reset(self) -> tuple[torch.Tensor, dict[str, object]]:
        return torch.zeros((1, 1)), {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        self.step_calls += 1
        values = {
            "total": 1.0,
            "distance_x": 0.1,
            "distance_y": 0.2,
            "distance_z": 0.3,
            "rotation": 0.4,
            "action_penalty": 0.0,
            "contact": 0.5,
            "object_stability": 0.6,
            "survival": 0.001,
            "deviation_penalty": 0.0,
        }
        completed = self.step_calls % 48 == 0
        self.environment.last_reward = SimpleNamespace(**{
            name: np.array([value], dtype=np.float64) for name, value in values.items()
        })
        self.environment.last_termination = SimpleNamespace(reset=np.array([completed], dtype=bool))
        self.environment.episode_returns = np.array([float(self.step_calls)], dtype=np.float64)
        self.environment.last_transition = None
        return (
            torch.zeros_like(actions),
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.bool),
            torch.zeros((1, 1), dtype=torch.bool),
            {},
        )


def _runtime() -> SimpleNamespace:
    environment = SimpleNamespace(
        config=SimpleNamespace(num_envs=1),
        last_reward=None,
        last_termination=None,
        episode_returns=np.zeros(1, dtype=np.float64),
        last_transition=None,
    )
    return SimpleNamespace(
        agent=FakeAgent(),
        env=FakeEnv(environment),
        gymnasium_env=SimpleNamespace(environment=environment),
        config=SimpleNamespace(rollouts=48),
        model=SimpleNamespace(parameters=lambda: [torch.ones(1)]),
    )


def test_train_without_wall_clock_cap_completes_all_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    runtime = _runtime()
    clock = FakeClock([0.0, 10.0, 14.0, 18.0, 20.0, 27.0])
    monkeypatch.setattr(tool.time, "monotonic", clock)

    updates, transitions, elapsed = tool._train(
        runtime, tool.TrainingBudget(num_envs=1, updates=2, wall_clock_seconds=None)
    )

    assert len(updates) == 2
    assert transitions == 96
    assert runtime.agent.training_mode is True
    assert runtime.agent.act_calls == 96
    assert runtime.env.step_calls == 96
    assert runtime.agent.recorded_timesteps == list(range(96))
    assert runtime.agent.post_interaction_timesteps == list(range(1, 97))
    assert [update["environment_transitions"] for update in updates] == [48.0, 96.0]
    assert [update["elapsed_seconds"] for update in updates] == [14.0, 20.0]
    assert [update["update_elapsed_seconds"] for update in updates] == [4.0, 2.0]
    assert [update["update_environment_transitions_per_second"] for update in updates] == [12.0, 24.0]
    assert [update["cumulative_environment_transitions_per_second"] for update in updates] == [48.0 / 14.0, 96.0 / 20.0]
    assert [update["completed_episode_count"] for update in updates] == [1.0, 1.0]
    assert [update["episode_return_mean"] for update in updates] == [48.0, 96.0]
    assert [update["episode_total_mean"] for update in updates] == [48.0, 96.0]
    assert [update["episode_total_min"] for update in updates] == [48.0, 96.0]
    assert [update["episode_total_max"] for update in updates] == [48.0, 96.0]
    assert [update["distance_reward_x_instant/step"] for update in updates] == [0.1, 0.1]
    assert [update["distance_reward_instant/step"] for update in updates] == pytest.approx([0.6, 0.6])
    assert [update["episode_lengths/step"] for update in updates] == [48.0, 48.0]
    assert [update["rewards/iter"] for update in updates] == [48.0, 96.0]
    assert [update["rewards/step"] for update in updates] == [48.0, 96.0]
    assert [update["rewards/time"] for update in updates] == [48.0, 96.0]
    assert [update["episode_cumulative/distance_reward_x"] for update in updates] == pytest.approx([4.8, 4.8])
    assert [update["episode_cumulative_min/distance_reward_x_min"] for update in updates] == pytest.approx([4.8, 4.8])
    assert [update["episode_cumulative_max/distance_reward_x_max"] for update in updates] == pytest.approx([4.8, 4.8])
    assert [update["episode_cumulative/contact_reward"] for update in updates] == pytest.approx([24.0, 24.0])
    assert [update["episode_reward/cube1_01"] for update in updates] == [48.0, 96.0]
    assert [update["episode_reward/object_cube1"] for update in updates] == [48.0, 96.0]
    assert [update["distance_reward"] for update in updates] == pytest.approx([28.8, 28.8])
    assert [update["distance_reward/cube1_01"] for update in updates] == pytest.approx([28.8, 28.8])
    assert [update["distance_reward/object_cube1"] for update in updates] == pytest.approx([28.8, 28.8])
    assert [update["episode_cumulative/distance_reward_x/cube1_01"] for update in updates] == pytest.approx([4.8, 4.8])
    assert [update["episode_cumulative_min/distance_reward_x/cube1_01_min"] for update in updates] == pytest.approx([4.8, 4.8])
    assert [update["episode_cumulative_max/distance_reward_x/cube1_01_max"] for update in updates] == pytest.approx([4.8, 4.8])
    assert all(update["reward_mean"] == update["manorl/reward_mean"] == 1.0 for update in updates)
    assert all(
        np.isclose(update["manorl/distance_x_mean"], 0.1)
        and update["manorl/action_penalty_mean"] == 0.0
        for update in updates
    )
    assert elapsed == 27.0
    assert clock.calls == 6


def test_profiled_train_attributes_all_post_calls_and_only_optimizer_boundaries() -> None:
    tool = _load_tool()
    runtime = _runtime()
    runtime.config.rollouts = 3
    runtime.agent.training = True
    runtime.agent._rollout = 0
    runtime.agent.cfg = SimpleNamespace(rollouts=3, learning_starts=4)
    original_post_interaction = runtime.agent.post_interaction

    def post_interaction(**kwargs: object) -> None:
        original_post_interaction(**kwargs)
        runtime.agent._rollout += 1

    runtime.agent.post_interaction = post_interaction

    updates, transitions, _ = tool._train(
        runtime, tool.TrainingBudget(num_envs=1, updates=2, profile_phases=True)
    )

    assert transitions == 6
    assert len(updates) == 2
    profile = runtime.training_phase_profile
    assert profile["runtime_env_step"]["calls"] == 6
    assert profile["rollout_finite_checks"]["calls"] == 6
    assert profile["host_telemetry"]["calls"] == 6
    assert profile["reset_count_host_item"]["calls"] == 6
    assert profile["post_interaction_all_calls"]["calls"] == 6
    assert profile["ppo_optimizer_rollout_boundary"]["calls"] == 1
    assert profile["trainer_parameter_finite_checks"]["calls"] == 2
    assert profile["update_host_telemetry"]["calls"] == 2


def test_profile_false_train_does_not_synchronize(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    runtime = _runtime()
    runtime.config.rollouts = 1
    runtime.device = "cuda"
    monkeypatch.setattr(
        tool.torch.cuda,
        "synchronize",
        lambda *_: (_ for _ in ()).throw(AssertionError("unexpected profiling sync")),
    )

    updates, transitions, _ = tool._train(
        runtime, tool.TrainingBudget(num_envs=1, updates=1, profile_phases=False)
    )

    assert transitions == 1
    assert len(updates) == 1
    assert runtime.training_phase_profile == {}


@pytest.mark.parametrize(
    ("controls", "diagnostics"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_training_budget_supports_four_way_controls_diagnostics_cross(
    controls: bool, diagnostics: bool
) -> None:
    tool = _load_tool()
    budget = tool.TrainingBudget(
        device_resident_controls=controls,
        capture_transition_diagnostics=diagnostics,
    )
    assert budget.resolved_capture_transition_diagnostics is diagnostics
    assert tool.TrainingBudget(
        device_resident_controls=controls
    ).resolved_capture_transition_diagnostics is (not controls)


def test_training_cli_parses_independent_diagnostics_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool()
    captured: list[object] = []
    monkeypatch.setattr(tool, "run", lambda _output, budget: captured.append(budget) or {})

    assert tool.main([
        "--output", str(tmp_path / "training"),
        "--device-resident-controls", "true",
        "--capture-transition-diagnostics", "true",
    ]) == 0
    budget = captured[0]
    assert budget.device_resident_controls is True
    assert budget.capture_transition_diagnostics is True
    assert budget.resolved_capture_transition_diagnostics is True


def test_training_observer_never_owns_steps_and_close_finishes_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    runtime = _runtime()
    runtime.config.rollouts = 3
    clock = FakeClock([0.0, 1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(tool.time, "monotonic", clock)

    class Observer:
        def __init__(self) -> None:
            self.close_requested = False
            self.observations = 0

        def observe(self) -> None:
            self.observations += 1
            self.close_requested = True

        def close(self) -> None:
            pass

    observer = Observer()
    updates, transitions, _ = tool._train(
        runtime, tool.TrainingBudget(num_envs=1, updates=2), observer=observer
    )

    assert observer.observations == 3
    assert runtime.env.step_calls == 3
    assert runtime.agent.recorded_timesteps == [0, 1, 2]
    assert len(updates) == 1
    assert transitions == 3


def test_console_formatters_keep_human_returns_compact_and_json_compatible(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool()
    record = {
        "schema": "manorl.completed_episode_returns.v1",
        "update": 1,
        "update_step": 2,
        "returns": [1.0, -3.0, 8.0],
    }

    summary = tool._format_completed_episode_record(record)
    assert "[" not in summary
    assert "count=3" in summary and "mean=2.0000" in summary
    tool._emit_console("json", "completed_episodes", record)
    assert json.loads(capsys.readouterr().out) == record
    tool._emit_console("json", "training_update", {"metrics": {"update": 1.0}})
    assert json.loads(capsys.readouterr().out) == {"event": "training_update", "metrics": {"update": 1.0}}
    tool._emit_console("json", "checkpoint", {"update": 1.0, "path": "checkpoint-000001.pt"})
    assert json.loads(capsys.readouterr().out) == {
        "event": "checkpoint", "update": 1.0, "path": "checkpoint-000001.pt"
    }
    throughput = {"environment_transitions": 48, "elapsed_seconds": 2.0}
    tool._emit_console("json", "training_complete", {"throughput": throughput})
    assert json.loads(capsys.readouterr().out) == {"event": "training_complete", "throughput": throughput}

    training_summary = tool._format_training_update({
        "update": 1.0,
        "environment_transitions": 48.0,
        "reward_mean": 0.5,
        "contact": 0.25,
        "reset_count": 1.0,
        "update_environment_transitions_per_second": 48.0,
        "elapsed_seconds": 1.0,
        "completed_episode_count": 2.0,
        "episode_total_mean": 3.0,
        "episode_total_min": -1.0,
        "episode_total_max": 7.0,
    })
    assert "episode_total_mean=3.0000" in training_summary
    assert "episode_total_min=-1.0000" in training_summary
    assert "episode_total_max=7.0000" in training_summary


def test_train_omits_episode_return_mean_without_completed_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    runtime = _runtime()
    runtime.config.rollouts = 2
    clock = FakeClock([0.0, 1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(tool.time, "monotonic", clock)

    updates, _, _ = tool._train(runtime, tool.TrainingBudget(num_envs=1, updates=1))

    assert updates[0]["completed_episode_count"] == 0.0
    assert "episode_return_mean" not in updates[0]
    assert "episode_total_mean" not in updates[0]
    assert "episode_total_min" not in updates[0]
    assert "episode_total_max" not in updates[0]
    assert "rewards/frame" not in updates[0]
    assert "rewards/iter" not in updates[0]
    assert "rewards/step" not in updates[0]
    assert "rewards/time" not in updates[0]
    assert "episode_reward" not in updates[0]
    assert "distance_reward_x" not in updates[0]
    assert "episode_reward/cube1_01" not in updates[0]
    assert "episode_lengths/step" not in updates[0]
    assert not any(name.startswith("episode_cumulative/") for name in updates[0])


def test_train_batches_same_step_completed_episode_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    runtime = _runtime()
    runtime.config.rollouts = 1
    runtime.gymnasium_env.environment.config.num_envs = 4
    runtime.env.reset = lambda: (torch.zeros((4, 1)), {})

    def step(actions: torch.Tensor):
        runtime.gymnasium_env.environment.last_reward = SimpleNamespace(**{
            name: np.full(4, 1.0, dtype=np.float64) for name in tool.REWARD_UPDATE_COMPONENTS
        })
        runtime.gymnasium_env.environment.last_transition = SimpleNamespace(
            termination=SimpleNamespace(reset=np.array([False, True, False, True])),
            episode_return=np.array([1.0, 2.5, 3.0, -4.0]),
        )
        return (
            torch.zeros_like(actions), torch.ones((4, 1)), torch.zeros((4, 1), dtype=torch.bool),
            torch.zeros((4, 1), dtype=torch.bool), {},
        )

    runtime.env.step = step
    clock = FakeClock([0.0, 1.0, 3.0, 5.0, 7.0])
    monkeypatch.setattr(tool.time, "monotonic", clock)
    records: list[dict[str, object]] = []
    updates, transitions, _ = tool._train(
        runtime, tool.TrainingBudget(num_envs=4, updates=1), on_completed_episodes=records.append
    )

    assert transitions == 4
    assert updates[0]["completed_episode_count"] == 2.0
    assert updates[0]["episode_return_mean"] == -0.75
    assert updates[0]["episode_total_mean"] == -0.75
    assert updates[0]["episode_total_min"] == -4.0
    assert updates[0]["episode_total_max"] == 2.5
    assert "episode_return_values" not in updates[0]
    assert records == [{
        "schema": "manorl.completed_episode_returns.v1", "update": 1, "update_step": 1,
        "vector_step": 1, "environment_transitions": 4, "env_ids": [1, 3], "returns": [2.5, -4.0],
    }]
    episode_file = io.StringIO()
    tool._write_episode_record(episode_file, records[0])
    assert json.loads(episode_file.getvalue()) == records[0]
    assert tool._episode_records_path(Path("outputs/manorl/run")) == Path("outputs/manorl/run.episodes.jsonl")


def test_train_emits_gym_style_object_and_pair_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    runtime = _runtime()
    runtime.config.rollouts = 1
    environment = runtime.gymnasium_env.environment
    environment.config.num_envs = 4
    environment.object_types = ("cube1", "cube1", "cube2", "cube2")
    environment.action_ids = np.asarray([1, 2, 1, 2], dtype=np.int64)
    environment.trajectories = tuple(
        SimpleNamespace(identity=SimpleNamespace(identity=identity))
        for identity in ("cube1_01_001", "cube1_02_001", "cube2_01_001", "cube2_02_001")
    )
    runtime.env.reset = lambda: (torch.zeros((4, 1)), {})
    completed = np.asarray([False, True, False, True])
    success = np.asarray([False, True, False, False])
    failure = np.asarray([False, False, False, True])
    component_values = np.arange(1.0, 5.0, dtype=np.float64)

    def step(actions: torch.Tensor):
        environment.last_reward = SimpleNamespace(**{
            source: component_values.copy()
            for source in set(tool.REWARD_UPDATE_COMPONENTS) | set(tool.GROUPED_REWARD_COMPONENTS)
        })
        environment.last_termination = SimpleNamespace(
            reset=completed,
            success=success,
            failure=failure,
            reason_code=np.asarray([0, 1, 0, 2], dtype=np.int32),
        )
        environment.episode_returns = np.asarray([1.0, 2.5, 3.0, -4.0])
        return (
            torch.zeros_like(actions),
            torch.ones((4, 1)),
            torch.as_tensor(completed).reshape(-1, 1),
            torch.zeros((4, 1), dtype=torch.bool),
            {},
        )

    runtime.env.step = step
    clock = FakeClock([0.0, 1.0, 3.0, 5.0, 7.0])
    monkeypatch.setattr(tool.time, "monotonic", clock)
    raw_updates: list[dict[str, object]] = []
    episode_records: list[dict[str, object]] = []
    updates, _, _ = tool._train(
        runtime,
        tool.TrainingBudget(num_envs=4, updates=1),
        on_update=raw_updates.append,
        on_completed_episodes=episode_records.append,
    )

    grouped = raw_updates[0]["grouped_metrics"]
    assert grouped["reward_mean/object_cube1"] == 1.5
    assert grouped["reward_mean/object_cube2"] == 3.5
    assert grouped["reward_mean/cube1_01"] == 1.0
    assert grouped["contact_reward_instant/cube2_02"] == 4.0
    assert grouped["attempts/object_cube1"] == 1.0
    assert grouped["success_rate/object_cube1"] == 100.0
    assert grouped["success_rate/object_cube2"] == 0.0
    assert grouped["episode_reward/cube1_02"] == 2.5
    assert grouped["episode_reward/cube2_02"] == -4.0
    assert grouped["distance_reward_x/cube1_02"] == 2.0
    assert grouped["distance_reward/cube2_02"] == 12.0
    assert "grouped_metrics" not in updates[0]
    assert episode_records[0]["schema"] == "manorl.completed_episode_returns.v2"
    assert episode_records[0]["object_types"] == ["cube1", "cube2"]
    assert episode_records[0]["action_ids"] == ["02", "02"]
    assert episode_records[0]["identities"] == ["cube1_02_001", "cube2_02_001"]
    assert episode_records[0]["reward_components"]["contact_reward"] == [2.0, 4.0]


def test_evaluate_waits_for_each_pair_first_termination() -> None:
    tool = _load_tool()

    class Agent:
        def enable_models_training_mode(self, _: bool) -> None:
            pass

    environment = SimpleNamespace(
        config=SimpleNamespace(num_envs=2),
        object_types=("cube1", "cube2"),
        action_ids=np.asarray([1, 2], dtype=np.int64),
        trajectories=tuple(
            SimpleNamespace(identity=SimpleNamespace(identity=value))
            for value in ("cube1_01_001", "cube2_02_001")
        ),
        trajectory_lengths=np.asarray([3, 5], dtype=np.int64),
        trajectory_steps=np.zeros(2, dtype=np.int64),
        reference_object_pos=np.zeros((2, 5, 3), dtype=np.float64),
        last_physical=None,
        last_reward=None,
        last_termination=None,
    )

    class Env:
        def __init__(self) -> None:
            self.calls = 0

        def reset(self):
            self.calls = 0
            return torch.zeros((2, 1)), {}

        def step(self, actions: torch.Tensor):
            self.calls += 1
            environment.trajectory_steps = np.minimum(
                np.asarray([self.calls, self.calls]), environment.trajectory_lengths - 1
            )
            positions = np.asarray(
                [[0.1 * self.calls, 0.0, 0.0], [0.2 * self.calls, 0.0, 0.0]],
                dtype=np.float64,
            )
            environment.last_physical = SimpleNamespace(object_position=positions)
            environment.last_reward = SimpleNamespace(contact=np.asarray([0.1, 0.2]))
            done = np.asarray([self.calls == 2, self.calls == 4])
            environment.last_termination = SimpleNamespace(
                reason_code=np.asarray([1 if done[0] else 0, 2 if done[1] else 0]),
                success=np.asarray([done[0], False]),
                failure=np.asarray([False, done[1]]),
            )
            return (
                torch.zeros((2, 1)),
                torch.as_tensor([[1.0], [2.0]]),
                torch.as_tensor(done).reshape(-1, 1),
                torch.zeros((2, 1), dtype=torch.bool),
                {},
            )

    runtime = SimpleNamespace(
        agent=Agent(),
        env=Env(),
        gymnasium_env=SimpleNamespace(environment=environment),
        device="cpu",
    )
    result = tool._evaluate(runtime, "zero")

    assert result.calls == 4
    assert result.return_mean == 5.0
    assert result.reward_mean == 1.5
    assert result.success_seen is True and result.failure_seen is True
    assert result.completed_horizon is False
    by_label = {group.label: group for group in result.groups}
    assert by_label["cube1_01"].return_mean == 2.0
    assert by_label["cube1_01"].success_count == 1
    assert by_label["cube2_02"].return_mean == 8.0
    assert by_label["cube2_02"].failure_count == 1


def test_full_pair_evaluation_uses_at_least_one_environment_per_pair() -> None:
    tool = _load_tool()
    trajectories = SimpleNamespace(resolved_pairs=tuple(range(77)))
    assert tool._full_coverage_evaluation_num_envs(
        tool.TrainingBudget(num_envs=2048, evaluation_num_envs=1), trajectories
    ) == 77
    with pytest.raises(ValueError, match="needs 77 environments"):
        tool._full_coverage_evaluation_num_envs(
            tool.TrainingBudget(num_envs=64, evaluation_num_envs=1), trajectories
        )


def test_episode_record_write_flushes_before_stdout_on_broken_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    events: list[str] = []

    class TrackingFile(io.StringIO):
        def write(self, value: str) -> int:
            events.append("write")
            return super().write(value)

        def flush(self) -> None:
            events.append("flush")
            super().flush()

    def broken_print(*_: object, **kwargs: object) -> None:
        assert kwargs["flush"] is True
        events.append("print")
        raise BrokenPipeError("closed stdout")

    episode_file = TrackingFile()
    monkeypatch.setattr("builtins.print", broken_print)
    record = {"schema": "manorl.completed_episode_returns.v1", "env_ids": [2], "returns": [3.5]}
    with pytest.raises(BrokenPipeError, match="closed stdout"):
        tool._write_episode_record(episode_file, record)

    assert events == ["write", "flush", "print"]
    assert json.loads(episode_file.getvalue()) == record


def test_episode_record_partial_publish_and_collision_protection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool()
    episodes_path = tmp_path / "run.episodes.jsonl"
    partial_path = tmp_path / "run.episodes.jsonl.partial"

    with tool._episode_records_file(episodes_path) as episode_file:
        episode_file.write('{"complete":true}\n')
    assert episodes_path.read_text(encoding="utf-8") == '{"complete":true}\n'
    assert not partial_path.exists()

    with pytest.raises(RuntimeError, match="interrupted"):
        with tool._episode_records_file(episodes_path.with_name("failed.episodes.jsonl")) as episode_file:
            episode_file.write('{"partial":true}\n')
            episode_file.flush()
            raise RuntimeError("interrupted")
    failed_partial = tmp_path / "failed.episodes.jsonl.partial"
    assert failed_partial.read_text(encoding="utf-8") == '{"partial":true}\n'
    assert not (tmp_path / "failed.episodes.jsonl").exists()

    collision_output = tmp_path / "collision"
    collision_partial = tool._partial_episode_records_path(collision_output)
    collision_partial.write_text('{"preserved":true}\n', encoding="utf-8")
    monkeypatch.setattr(tool, "_assert_cuda_runtime", lambda: None)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        tool.run(collision_output, tool.TrainingBudget())
    assert collision_partial.read_text(encoding="utf-8") == '{"preserved":true}\n'

    final_collision_output = tmp_path / "collision-final"
    final_collision = tool._episode_records_path(final_collision_output)
    final_collision.write_text('{"published":true}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        tool.run(final_collision_output, tool.TrainingBudget())
    assert final_collision.read_text(encoding="utf-8") == '{"published":true}\n'


def test_train_saves_periodic_native_checkpoints_and_updates_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool()
    runtime = _runtime()
    runtime.checkpoint_metadata = lambda: {"runtime": "fake"}
    clock = FakeClock(list(range(12)))
    monkeypatch.setattr(tool.time, "monotonic", clock)
    output = tmp_path / "training"
    checkpoint_paths: list[Path] = []

    def on_update(update: dict[str, float]) -> None:
        checkpoint = tool._maybe_save_periodic_checkpoint(runtime, output, 2, update)
        if checkpoint is not None:
            checkpoint_paths.append(checkpoint)

    updates, _, _ = tool._train(
        runtime, tool.TrainingBudget(num_envs=1, updates=5), on_update=on_update
    )

    assert len(updates) == 5
    assert checkpoint_paths == [
        output / "checkpoint-000002.pt",
        output / "checkpoint-000004.pt",
    ]
    assert not output.with_suffix(".pt").exists()
    for checkpoint, completed_updates in zip(checkpoint_paths, [2, 4], strict=True):
        sidecar = checkpoint.with_suffix(".pt.json")
        assert checkpoint.is_file()
        assert sidecar.is_file()
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        assert metadata["checkpoint_file"] == checkpoint.name
        assert metadata["runtime_config"]["training_progress"] == {
            "completed_updates": completed_updates,
            "environment_transitions": completed_updates * 48,
        }
        tool.load_skrl_checkpoint(runtime.agent, checkpoint)
        assert runtime.agent.loaded_checkpoint == str(checkpoint)

    last_checkpoint = output / "last.pt"
    assert last_checkpoint.read_bytes() == checkpoint_paths[-1].read_bytes()
    last_sidecar = last_checkpoint.with_suffix(".pt.json")
    last_sidecar_content = last_sidecar.read_bytes()
    last_metadata = json.loads(last_sidecar_content)
    assert last_metadata["checkpoint_file"] == "last.pt"
    assert "training_progress" not in last_metadata["runtime_config"]
    assert last_metadata["progress_metadata"] == "immutable numbered and final checkpoint sidecars"
    tool.load_skrl_checkpoint(runtime.agent, last_checkpoint)
    assert runtime.agent.loaded_checkpoint == str(last_checkpoint)

    final_checkpoint = output.with_suffix(".pt")
    tool._save_checkpoint_atomically(
        runtime.agent,
        final_checkpoint,
        runtime_config=tool._checkpoint_runtime_config(runtime, updates[-1]),
    )
    tool._update_last_checkpoint(output, final_checkpoint)
    assert last_checkpoint.read_bytes() == final_checkpoint.read_bytes()
    assert last_sidecar.read_bytes() == last_sidecar_content
    final_metadata = json.loads(final_checkpoint.with_suffix(".pt.json").read_text(encoding="utf-8"))
    assert final_metadata["runtime_config"]["training_progress"]["completed_updates"] == 5

    with pytest.raises(FileExistsError, match="periodic checkpoint artifact"):
        tool._maybe_save_periodic_checkpoint(runtime, output, 2, updates[1])


def test_periodic_checkpoint_namespaces_isolate_sibling_output_prefixes(tmp_path: Path) -> None:
    tool = _load_tool()
    runtime = _runtime()
    runtime.checkpoint_metadata = lambda: {"runtime": "fake"}
    update = {"update": 100.0, "environment_transitions": 4_800.0}
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first_checkpoint = tool._maybe_save_periodic_checkpoint(runtime, first_output, 100, update)
    second_checkpoint = tool._maybe_save_periodic_checkpoint(runtime, second_output, 100, update)

    assert first_checkpoint == first_output / "checkpoint-000100.pt"
    assert second_checkpoint == second_output / "checkpoint-000100.pt"
    assert first_checkpoint != second_checkpoint
    assert (first_output / "last.pt").is_file()
    assert (second_output / "last.pt").is_file()


def test_checkpoint_publication_failures_preserve_complete_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool()
    runtime = _runtime()
    runtime.checkpoint_metadata = lambda: {"runtime": "fake"}
    first_update = {"update": 2.0, "environment_transitions": 96.0}
    second_update = {"update": 4.0, "environment_transitions": 192.0}
    original_replace = tool.os.replace

    initial_output = tmp_path / "initial"
    initial_checkpoint = tool._save_periodic_checkpoint(runtime, initial_output, first_update)
    initial_last = initial_output / "last.pt"
    initial_sidecar = initial_last.with_suffix(".pt.json")

    def fail_initial_sidecar(source: Path, destination: Path) -> None:
        if destination == initial_sidecar:
            raise OSError("injected initial sidecar publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(tool.os, "replace", fail_initial_sidecar)
    with pytest.raises(OSError, match="initial sidecar"):
        tool._update_last_checkpoint(initial_output, initial_checkpoint)
    assert not initial_last.exists()
    assert not initial_sidecar.exists()
    assert not list(initial_output.glob(".last.pt.json.*.tmp*"))

    output = tmp_path / "training"
    first_checkpoint = tool._save_periodic_checkpoint(runtime, output, first_update)
    last_checkpoint = tool._update_last_checkpoint(output, first_checkpoint)
    last_sidecar = last_checkpoint.with_suffix(".pt.json")
    previous_payload = last_checkpoint.read_bytes()
    static_sidecar = last_sidecar.read_bytes()
    next_checkpoint = tool._save_periodic_checkpoint(runtime, output, second_update)

    def fail_last_payload(source: Path, destination: Path) -> None:
        if destination == last_checkpoint:
            raise OSError("injected last payload publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(tool.os, "replace", fail_last_payload)
    with pytest.raises(OSError, match="last payload"):
        tool._update_last_checkpoint(output, next_checkpoint)
    assert last_checkpoint.read_bytes() == previous_payload
    assert last_sidecar.read_bytes() == static_sidecar
    tool.load_skrl_checkpoint(runtime.agent, last_checkpoint)
    assert runtime.agent.loaded_checkpoint == str(last_checkpoint)
    assert not list(output.glob(".last.pt.*.tmp*"))

    def fail_sidecar_if_replaced(source: Path, destination: Path) -> None:
        if destination == last_sidecar:
            raise AssertionError("static last sidecar was replaced")
        original_replace(source, destination)

    monkeypatch.setattr(tool.os, "replace", fail_sidecar_if_replaced)
    tool._update_last_checkpoint(output, next_checkpoint)
    assert last_checkpoint.read_bytes() == next_checkpoint.read_bytes()
    assert last_sidecar.read_bytes() == static_sidecar
    tool.load_skrl_checkpoint(runtime.agent, last_checkpoint)
    assert runtime.agent.loaded_checkpoint == str(last_checkpoint)

    failed_output = tmp_path / "failed"
    failed_checkpoint = failed_output / "checkpoint-000002.pt"

    def fail_numbered_payload(source: Path, destination: Path) -> None:
        if destination == failed_checkpoint:
            raise OSError("injected numbered payload publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(tool.os, "replace", fail_numbered_payload)
    with pytest.raises(OSError, match="numbered payload"):
        tool._save_periodic_checkpoint(runtime, failed_output, first_update)
    assert not failed_checkpoint.exists()
    assert not failed_checkpoint.with_suffix(".pt.json").exists()
    assert not list(failed_output.glob(".checkpoint-000002.pt.*.tmp*"))


def test_train_with_wall_clock_cap_stops_before_an_update(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    runtime = _runtime()
    clock = FakeClock([0.0, 1.0, 1.0])
    monkeypatch.setattr(tool.time, "monotonic", clock)

    updates, transitions, elapsed = tool._train(
        runtime, tool.TrainingBudget(num_envs=1, updates=2, wall_clock_seconds=0.5)
    )

    assert updates == []
    assert transitions == 0
    assert elapsed == 1.0
    assert runtime.agent.act_calls == 0
    assert runtime.env.step_calls == 0
    assert clock.calls == 3


def test_cli_omits_wall_clock_cap_and_preserves_explicit_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool()
    captured = []
    monkeypatch.setattr(tool, "run", lambda output, budget: captured.append((output, budget)) or {})

    assert tool.main(["--output", str(tmp_path / "default")]) == 0
    assert captured[-1][1].wall_clock_seconds is None
    assert captured[-1][1].num_envs == 2048
    assert captured[-1][1].updates == 8000
    assert captured[-1][1].checkpoint_interval_updates == 200
    assert captured[-1][1].evaluation_num_envs == 1
    assert captured[-1][1].minibatch_size is None
    assert captured[-1][1].resolved_minibatch_size == 4096
    assert captured[-1][1].use_film is True
    assert captured[-1][1].residual_enabled is True
    assert captured[-1][1].terminal is True
    assert captured[-1][1].wandb.enabled is True
    assert captured[-1][1].dataset_path == tool.DATASET_PATH

    assert tool.main([
        "--output", str(tmp_path / "capped"),
        "--wall-clock-seconds", "17.5",
        "--checkpoint-interval-updates", "100",
    ]) == 0
    assert captured[-1][1].wall_clock_seconds == 17.5
    assert captured[-1][1].checkpoint_interval_updates == 100

    assert tool.main([
        "--output", str(tmp_path / "server2"), "--num-envs", "4096", "--minibatch-size", "4096",
    ]) == 0
    assert captured[-1][1].minibatch_size == 4096

    assert tool.main([
        "--output", str(tmp_path / "server2-default"), "--num-envs", "2048",
    ]) == 0
    assert captured[-1][1].minibatch_size is None
    assert captured[-1][1].resolved_minibatch_size == 4096


def test_cli_parses_all_and_exact_pair_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool()
    captured = []
    monkeypatch.setattr(tool, "run", lambda output, budget: captured.append((output, budget)) or {})

    dataset_path = tmp_path / "restored-s02.lance"
    assert tool.main([
        "--output", str(tmp_path / "all"),
        "--all-pairs",
        "--dataset-path", str(dataset_path),
    ]) == 0
    assert captured[-1][1].trajectory_selector == "all"
    assert captured[-1][1].dataset_path == str(dataset_path.resolve())

    assert tool.main([
        "--output", str(tmp_path / "pairs"),
        "--pairs", "cube2:1,cube1:02",
    ]) == 0
    assert captured[-1][1].trajectory_selector == "cube1:02,cube2:01"


def test_cli_rejects_pair_selector_conflicts(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tool = _load_tool()
    with pytest.raises(SystemExit, match="2"):
        tool.main([
            "--output", str(tmp_path / "invalid"),
            "--all-pairs", "--pairs", "cube1:01",
        ])
    assert "not allowed with argument" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        tool.main([
            "--output", str(tmp_path / "invalid-legacy"),
            "--pairs", "cube1:01", "--object", "cube1", "--gesture", "01",
        ])
    assert "cannot be combined" in capsys.readouterr().err


def test_cli_rejects_local_rerun_and_grpc_stream_together(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tool = _load_tool()

    with pytest.raises(SystemExit, match="2"):
        tool.main([
            "--output", str(tmp_path / "invalid"),
            "--rerun-output", str(tmp_path / "episode.rrd"),
            "--rerun-grpc-url", "rerun+http://127.0.0.1:9876/proxy",
            "--wandb", "false",
        ])

    assert "--rerun-output and --rerun-grpc-url are mutually exclusive" in capsys.readouterr().err


def test_training_observer_quiets_viewer_for_json_console(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    import sim.manorl.view_environment as viewer

    created: list[dict[str, object]] = []

    class FakeViewer:
        close_requested = False
        def __init__(self, _environment: object, **kwargs: object) -> None:
            created.append(kwargs)
        def observe(self) -> None:
            pass
        def close(self) -> None:
            pass

    monkeypatch.setattr(viewer, "TrainingViewer", FakeViewer)
    environment = SimpleNamespace()
    tool._build_training_observer(
        environment, tool.TrainingBudget(headless=False, console_format="json", viewer_envs=8, viewer_stride=2)
    )
    tool._build_training_observer(
        environment, tool.TrainingBudget(headless=False, console_format="human", viewer_envs=8, viewer_stride=2)
    )

    assert created == [
        {"tile_envs": 8, "stride": 2, "quiet": True},
        {"tile_envs": 8, "stride": 2, "quiet": False},
    ]


def test_json_console_mode_preserves_final_main_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool()
    result = {"schema": "manorl.cube1_fast_training.v1", "updates": [{"update": 1.0}]}
    monkeypatch.setattr(tool, "run", lambda *_: result)

    assert tool.main(["--output", str(tmp_path / "json"), "--console-format", "json"]) == 0
    assert json.loads(capsys.readouterr().out) == result


def test_cli_configures_human_console_and_validates_training_viewer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool()
    captured = []
    monkeypatch.setattr(tool, "run", lambda output, budget: captured.append((output, budget)) or {})

    assert tool.main([
        "--output", str(tmp_path / "visual"), "--num-envs", "8", "--evaluation-num-envs", "8",
        "--minibatch-size", "384", "--headless", "false", "--viewer-envs", "8", "--viewer-stride", "2",
    ]) == 0
    budget = captured[-1][1]
    assert budget.headless is False and budget.viewer_envs == 8 and budget.viewer_stride == 2
    assert budget.console_format == "human"
    assert "artifacts metrics=" in capsys.readouterr().out

    assert tool.main(["--output", str(tmp_path / "default")]) == 0
    default_budget = captured[-1][1]
    assert default_budget.headless is True and default_budget.viewer_envs == 1
    assert default_budget.viewer_stride == 1 and default_budget.console_format == "human"
    capsys.readouterr()

    with pytest.raises(SystemExit, match="2"):
        tool.main([
            "--output", str(tmp_path / "bad"), "--num-envs", "8", "--minibatch-size", "384", "--viewer-envs", "9"
        ])
    assert "viewer-envs must be within num-envs" in capsys.readouterr().err


def test_cli_rejects_minibatch_that_does_not_divide_rollout_batch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool()

    with pytest.raises(SystemExit, match="2"):
        tool.main(["--output", "output", "--num-envs", "1", "--minibatch-size", "4096"])

    assert "must divide rollout batch" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_invalid_checkpoint_interval(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool()

    with pytest.raises(SystemExit, match="2"):
        tool.main(["--output", "output", "--checkpoint-interval-updates", value])

    assert "checkpoint-interval-updates must be positive when provided" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_explicit_wall_clock_cap(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool()

    option = f"--wall-clock-seconds={value}" if value == "-inf" else "--wall-clock-seconds"
    arguments = ["--output", "output", option]
    if option == "--wall-clock-seconds":
        arguments.append(value)
    with pytest.raises(SystemExit, match="2"):
        tool.main(arguments)

    assert "wall-clock-seconds must be a finite positive value when provided" in capsys.readouterr().err


def test_evaluation_budget_defaults_to_single_world_and_supports_bounded_override() -> None:
    tool = _load_tool()

    training = tool.TrainingBudget(num_envs=4096, minibatch_size=4096)
    small = tool.TrainingBudget(num_envs=64)
    override = tool.TrainingBudget(num_envs=4096, evaluation_num_envs=96, minibatch_size=4096)
    bounded = tool.TrainingBudget(num_envs=4096, evaluation_num_envs=None, minibatch_size=4096)

    assert training.resolved_evaluation_num_envs == 1
    assert small.resolved_evaluation_num_envs == 1
    assert override.resolved_evaluation_num_envs == 96
    assert bounded.resolved_evaluation_num_envs == 128
    assert tool._evaluation_ppo_config(tool.ManoPPOConfig(minibatch_size=4096), num_envs=128).minibatch_size == 2048
    assert tool._evaluation_ppo_config(tool.ManoPPOConfig(minibatch_size=4096), num_envs=96).minibatch_size == 512
    for invalid in (0, 129, 4096):
        with pytest.raises(ValueError, match="evaluation_num_envs must be within 1..128"):
            _ = tool.TrainingBudget(num_envs=4096, evaluation_num_envs=invalid).resolved_evaluation_num_envs
    with pytest.raises(ValueError, match="evaluation_num_envs must be within 1..64"):
        _ = tool.TrainingBudget(num_envs=64, evaluation_num_envs=65).resolved_evaluation_num_envs


def test_cli_serializes_default_and_override_evaluation_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool()
    captured = []
    monkeypatch.setattr(tool, "run", lambda output, budget: captured.append(budget) or {})

    tool.main(["--output", str(tmp_path / "default"), "--num-envs", "4096", "--minibatch-size", "4096"])
    assert captured[-1].resolved_evaluation_num_envs == 1
    tool.main([
        "--output", str(tmp_path / "override"), "--num-envs", "4096", "--minibatch-size", "4096",
        "--evaluation-num-envs", "96",
    ])
    assert captured[-1].resolved_evaluation_num_envs == 96
    for invalid in ("0", "129", "4096"):
        with pytest.raises(SystemExit, match="2"):
            tool.main(["--output", str(tmp_path / f"invalid-{invalid}"), "--num-envs", "4096", "--evaluation-num-envs", invalid])
    with pytest.raises(SystemExit, match="2"):
        tool.main(["--output", str(tmp_path / "invalid-small"), "--num-envs", "64", "--evaluation-num-envs", "65"])
    assert "evaluation-num-envs must be within 1..64 when provided" in capsys.readouterr().err


def test_run_closes_recorder_when_training_viewer_construction_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool()
    recorder_closed: list[bool] = []
    rerun_output = tmp_path / "run.rrd"
    rerun_output.write_bytes(b"previous recording")
    rerun_output.with_name(".run.active.rrd").write_bytes(b"stale active stream")

    class Runtime:
        device = "cuda"
        def __init__(self, config: object) -> None:
            self.config = config
            self.agent = SimpleNamespace(cfg=SimpleNamespace(learning_starts=None))
            self.model = SimpleNamespace(parameters=lambda: [])
            self.gymnasium_env = SimpleNamespace(environment=None)
        def checkpoint_metadata(self) -> dict[str, object]:
            return {"runtime": "test"}

    class Physical:
        def __init__(self, config: object) -> None:
            self.config = config
            self.contact_start_frame = 1
            self.jax = SimpleNamespace(default_backend=lambda: "gpu")

    class Recorder:
        def __init__(self, *_: object, **__: object) -> None:
            pass
        def close(self) -> None:
            recorder_closed.append(True)
            return None

    def evaluation(*_: object, **__: object) -> tuple[Runtime, object, list[dict[str, object]]]:
        config = tool.ManoPPOConfig(minibatch_size=1)
        return Runtime(config), config, []

    monkeypatch.setattr(tool, "_assert_cuda_runtime", lambda: None)
    monkeypatch.setattr(tool, "load_assigned_trajectory_batch", lambda *args, **kwargs: object())
    monkeypatch.setattr(tool, "MujocoManoEnvironment", lambda _, config: Physical(config))
    monkeypatch.setattr(tool, "ManoGymnasiumVectorEnv", lambda physical: physical)
    monkeypatch.setattr(tool, "ManoSkrlRuntime", lambda _, config: Runtime(config))
    monkeypatch.setattr(tool, "_build_evaluation_runtime", evaluation)
    monkeypatch.setattr(tool, "_trajectory_assignments", lambda _: [])
    monkeypatch.setattr(tool, "_save_checkpoint_atomically", lambda _, path, **__: path)
    monkeypatch.setattr(tool, "load_skrl_checkpoint", lambda *_: None)
    monkeypatch.setattr(tool, "_evaluate", lambda _, mode: tool.EvaluationResult(mode, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, True, [], []))
    monkeypatch.setattr(tool, "ManoRerunRecorder", Recorder)
    monkeypatch.setattr(tool, "_build_training_observer", lambda *_: (_ for _ in ()).throw(RuntimeError("viewer failed")))

    with pytest.raises(RuntimeError, match="viewer failed"):
        tool.run(
            tmp_path / "run",
            tool.TrainingBudget(
                num_envs=1,
                updates=1,
                minibatch_size=1,
                rerun_output=str(rerun_output),
                wandb=tool.WandbOptions(enabled=False),
            ),
        )
    assert recorder_closed == [True]


def test_run_reuses_bounded_evaluator_and_native_checkpoint_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool()
    constructions: list[tuple[str, int, int]] = []
    loads: list[tuple[str, str]] = []
    modes: list[tuple[str, str]] = []
    initial_runtime_ref: list[weakref.ReferenceType[object]] = []
    training_runtime_ref: list[weakref.ReferenceType[object]] = []
    training_physical_ref: list[weakref.ReferenceType[object]] = []

    class Runtime:
        def __init__(self, name: str, config: object) -> None:
            self.name = name
            self.device = "cuda"
            self.config = config
            self.agent = SimpleNamespace(
                name=name,
                cfg=SimpleNamespace(learning_starts=config.learning_starts),
            )
            self.gymnasium_env = SimpleNamespace(environment=None)
            self.model = SimpleNamespace(parameters=lambda: [])

        def checkpoint_metadata(self) -> dict[str, object]:
            return {"runtime": self.name}

    class Physical:
        def __init__(self, config: object) -> None:
            self.config = config
            self.contact_start_frame = 250
            self.jax = SimpleNamespace(default_backend=lambda: "gpu")

    def trajectories(_: object, *, num_envs: int) -> SimpleNamespace:
        return SimpleNamespace(num_envs=num_envs)

    def build_physical(_: object, config: object) -> Physical:
        physical = Physical(config)
        if not constructions:
            training_physical_ref.append(weakref.ref(physical))
        return physical

    def build_runtime(physical: Physical, config: object) -> Runtime:
        name = "training" if not constructions else f"evaluation-{len(constructions)}"
        constructions.append((name, physical.config.num_envs, config.minibatch_size))
        runtime = Runtime(name, config)
        if name == "training":
            training_runtime_ref.append(weakref.ref(runtime))
        if name == "evaluation-1":
            initial_runtime_ref.append(weakref.ref(runtime))
        return runtime

    def assignments(trajectory_batch: SimpleNamespace) -> list[dict[str, object]]:
        return [{"env_id": index, "identity": f"prefix-{index}"} for index in range(trajectory_batch.num_envs)]

    def save(_: object, path: Path, **__: object) -> Path:
        return path

    def evaluate(runtime: Runtime, mode: str) -> object:
        if mode == "trained":
            assert runtime is initial_runtime_ref[0]()
            assert training_runtime_ref[0]() is None
            assert training_physical_ref[0]() is None
        modes.append((runtime.name, mode))
        return tool.EvaluationResult(mode, 1, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, False, False, True, [1.0], [0.0])

    def train(*_: object, **__: object) -> tuple[list[object], int, float]:
        assert initial_runtime_ref[0]() is not None
        return [], 0, 0.0

    monkeypatch.setattr(tool, "_assert_cuda_runtime", lambda: None)
    monkeypatch.setattr(tool, "load_assigned_trajectory_batch", trajectories)
    monkeypatch.setattr(tool, "MujocoManoEnvironment", build_physical)
    monkeypatch.setattr(tool, "ManoGymnasiumVectorEnv", lambda physical: physical)
    monkeypatch.setattr(tool, "ManoSkrlRuntime", build_runtime)
    monkeypatch.setattr(tool, "_trajectory_assignments", assignments)
    monkeypatch.setattr(tool, "_save_checkpoint_atomically", save)
    monkeypatch.setattr(tool, "_update_last_checkpoint", lambda output, checkpoint: output / "last.pt")
    monkeypatch.setattr(tool, "load_skrl_checkpoint", lambda agent, path: loads.append((agent.name, path.name)) or path)
    monkeypatch.setattr(tool, "_evaluate", evaluate)
    monkeypatch.setattr(tool, "_train", train)

    result = tool.run(
        tmp_path / "run",
        tool.TrainingBudget(
            num_envs=4096,
            updates=1,
            minibatch_size=4096,
            evaluation_num_envs=128,
            wandb=tool.WandbOptions(enabled=False),
        ),
    )

    assert constructions == [("training", 4096, 4096), ("evaluation-1", 128, 2048)]
    assert modes == [("evaluation-1", "zero"), ("evaluation-1", "untrained"), ("evaluation-1", "trained")]
    assert loads[0][0] == "evaluation-1" and loads[1][0] == "training"
    assert loads[0][1] == loads[1][1] and loads[0][1].startswith(".initial-")
    assert loads[2] == ("evaluation-1", "run.pt")
    assert result["trajectory_selection"]["evaluation_assignments"] == [{"env_id": i, "identity": f"prefix-{i}"} for i in range(128)]
    assert result["budget"]["evaluation_num_envs"] == 128
    assert result["learning_starts"] == 0
    assert not list((tmp_path / "run").glob(".initial-*"))


def test_owned_initial_checkpoint_cleans_up_after_failure_and_refuses_collisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool()
    with pytest.raises(RuntimeError, match="interrupted"):
        with tool._owned_initial_checkpoint(tmp_path) as checkpoint:
            tool._checkpoint_sidecar_path(checkpoint).write_text("sidecar", encoding="utf-8")
            raise RuntimeError("interrupted")
    assert not list(tmp_path.glob(".initial-*"))

    monkeypatch.setattr(tool, "uuid4", lambda: SimpleNamespace(hex="collision"))
    (tmp_path / ".initial-collision.pt").touch()
    with pytest.raises(RuntimeError, match="could not reserve"):
        with tool._owned_initial_checkpoint(tmp_path):
            pass
