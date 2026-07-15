from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_tool():
    path = Path(__file__).parents[2] / "tools" / "train_manorl_cube1.py"
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
    def __init__(self) -> None:
        self.act_calls = 0
        self.recorded_timesteps: list[int] = []
        self.post_interaction_timesteps: list[int] = []
        self.training_mode = False

    def enable_training_mode(self, enabled: bool) -> None:
        self.training_mode = enabled

    def act(self, observations: torch.Tensor, *_: object, **__: object) -> tuple[torch.Tensor, None]:
        self.act_calls += 1
        return torch.ones_like(observations), None

    def record_transition(self, *, timestep: int, **_: object) -> None:
        self.recorded_timesteps.append(timestep)

    def post_interaction(self, *, timestep: int, **_: object) -> None:
        self.post_interaction_timesteps.append(timestep)


class FakeEnv:
    def __init__(self) -> None:
        self.step_calls = 0

    def reset(self) -> tuple[torch.Tensor, dict[str, object]]:
        return torch.zeros((1, 1)), {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        self.step_calls += 1
        return (
            torch.zeros_like(actions),
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.bool),
            torch.zeros((1, 1), dtype=torch.bool),
            {},
        )


def _runtime() -> SimpleNamespace:
    environment = SimpleNamespace(config=SimpleNamespace(num_envs=1))
    return SimpleNamespace(
        agent=FakeAgent(),
        env=FakeEnv(),
        gymnasium_env=SimpleNamespace(environment=environment),
        config=SimpleNamespace(rollouts=48),
        model=SimpleNamespace(parameters=lambda: [torch.ones(1)]),
    )


def test_train_without_wall_clock_cap_completes_all_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    runtime = _runtime()
    clock = FakeClock([0.0, 10_000.0, 10_001.0, 10_002.0])
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
    assert [update["elapsed_seconds"] for update in updates] == [10_000.0, 10_001.0]
    assert elapsed == 10_002.0
    assert clock.calls == 4


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

    assert tool.main([
        "--output", str(tmp_path / "capped"),
        "--wall-clock-seconds", "17.5",
    ]) == 0
    assert captured[-1][1].wall_clock_seconds == 17.5


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
