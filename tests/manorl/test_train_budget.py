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
    assert all(update["reward_mean"] == update["total"] == 1.0 for update in updates)
    assert all(np.isclose(update["distance_x"], 0.1) and update["action_penalty"] == 0.0 for update in updates)
    assert elapsed == 27.0
    assert clock.calls == 6


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


def test_train_omits_episode_return_mean_without_completed_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    runtime = _runtime()
    runtime.config.rollouts = 2
    clock = FakeClock([0.0, 1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(tool.time, "monotonic", clock)

    updates, _, _ = tool._train(runtime, tool.TrainingBudget(num_envs=1, updates=1))

    assert updates[0]["completed_episode_count"] == 0.0
    assert "episode_return_mean" not in updates[0]


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
    assert "episode_return_values" not in updates[0]
    assert records == [{
        "schema": "manorl.completed_episode_returns.v1", "update": 1, "update_step": 1,
        "vector_step": 1, "environment_transitions": 4, "env_ids": [1, 3], "returns": [2.5, -4.0],
    }]
    episode_file = io.StringIO()
    tool._write_episode_record(episode_file, records[0])
    assert json.loads(episode_file.getvalue()) == records[0]
    assert tool._episode_records_path(Path("outputs/manorl/run")) == Path("outputs/manorl/run.episodes.jsonl")


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
    assert captured[-1][1].checkpoint_interval_updates is None
    assert captured[-1][1].minibatch_size == 1024

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


def test_evaluation_budget_defaults_to_bounded_prefix_and_uses_valid_minibatch() -> None:
    tool = _load_tool()

    training = tool.TrainingBudget(num_envs=4096, minibatch_size=4096)
    small = tool.TrainingBudget(num_envs=64)
    override = tool.TrainingBudget(num_envs=4096, evaluation_num_envs=96, minibatch_size=4096)

    assert training.resolved_evaluation_num_envs == 128
    assert small.resolved_evaluation_num_envs == 64
    assert override.resolved_evaluation_num_envs == 96
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
    assert captured[-1].resolved_evaluation_num_envs == 128
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
            tool.TrainingBudget(num_envs=1, updates=1, minibatch_size=1, rerun_output=str(tmp_path / "run.rrd")),
        )
    assert recorder_closed == [True]


def test_run_uses_bounded_fresh_evaluators_and_native_initial_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = _load_tool()
    constructions: list[tuple[str, int, int]] = []
    loads: list[tuple[str, str]] = []
    modes: list[tuple[str, str]] = []
    released: list[str] = []
    initial_runtime_ref: list[weakref.ReferenceType[object]] = []

    class Runtime:
        def __init__(self, name: str, config: object) -> None:
            self.name = name
            self.device = "cuda"
            self.config = config
            self.agent = SimpleNamespace(name=name, cfg=SimpleNamespace(learning_starts=None))
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
        return Physical(config)

    def build_runtime(physical: Physical, config: object) -> Runtime:
        name = "training" if not constructions else f"evaluation-{len(constructions)}"
        if name == "evaluation-2":
            assert released == ["evaluation-1"]
            assert initial_runtime_ref[0]() is None
        constructions.append((name, physical.config.num_envs, config.minibatch_size))
        runtime = Runtime(name, config)
        if name == "evaluation-1":
            initial_runtime_ref.append(weakref.ref(runtime))
            weakref.finalize(runtime, released.append, name)
        return runtime

    def assignments(trajectory_batch: SimpleNamespace) -> list[dict[str, object]]:
        return [{"env_id": index, "identity": f"prefix-{index}"} for index in range(trajectory_batch.num_envs)]

    def save(_: object, path: Path, **__: object) -> Path:
        return path

    def evaluate(runtime: Runtime, mode: str) -> object:
        modes.append((runtime.name, mode))
        return tool.EvaluationResult(mode, 1, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, False, False, True, [1.0], [0.0])

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
    monkeypatch.setattr(tool, "_train", lambda *args, **kwargs: ([], 0, 0.0))

    result = tool.run(tmp_path / "run", tool.TrainingBudget(num_envs=4096, updates=1, minibatch_size=4096))

    assert constructions == [("training", 4096, 4096), ("evaluation-1", 128, 2048), ("evaluation-2", 128, 2048)]
    assert modes == [("evaluation-1", "zero"), ("evaluation-1", "untrained"), ("evaluation-2", "trained")]
    assert loads[0][0] == "evaluation-1" and loads[1][0] == "training"
    assert loads[0][1] == loads[1][1] and loads[0][1].startswith(".initial-")
    assert loads[2] == ("evaluation-2", "run.pt")
    assert result["trajectory_selection"]["evaluation_assignments"] == [{"env_id": i, "identity": f"prefix-{i}"} for i in range(128)]
    assert result["budget"]["evaluation_num_envs"] == 128
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
