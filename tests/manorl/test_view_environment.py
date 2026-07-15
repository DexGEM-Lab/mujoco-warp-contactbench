from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from sim.manorl.view_environment import _tile_layout, parse_args


def _load_tool(name: str):
    path = Path(__file__).resolve().parents[2] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_viewer_cli_defaults_to_residual_and_terminal_modes() -> None:
    args = parse_args([])
    assert args.device == "cpu"
    assert args.speed == 0.25
    assert args.loop is True
    assert args.print_every == 10
    assert args.terminal is True
    assert args.use_residual is True
    assert args.trajectory == "accepted"
    assert args.num_envs == 1
    assert args.render_env == 0
    assert args.tile_envs == 1
    assert args.object_type is None
    assert args.gesture is None
    assert args.rerun_output is None

    one_shot = parse_args(
        [
            "--device",
            "gpu",
            "--no-loop",
            "--speed",
            "1.0",
            "--terminal",
            "false",
            "--use_residual",
            "false",
            "--trajectory",
            "generated-cube1-row-507",
            "--num-envs",
            "10",
            "--render-env",
            "9",
            "--tile-envs",
            "10",
            "--object",
            "cube1",
            "--gesture",
            "03",
            "--rerun-output",
            "outputs/manorl/viewer.rrd",
        ]
    )
    assert one_shot.device == "gpu"
    assert one_shot.loop is False
    assert one_shot.speed == 1.0
    assert one_shot.terminal is False
    assert one_shot.use_residual is False
    assert one_shot.trajectory == "generated-cube1-row-507"
    assert one_shot.num_envs == 10
    assert one_shot.render_env == 9
    assert one_shot.tile_envs == 10
    assert one_shot.object_type == "cube1"
    assert one_shot.gesture == "03"
    assert str(one_shot.rerun_output) == "outputs/manorl/viewer.rrd"


@pytest.mark.parametrize(
    ("mode_args", "expected_residual_enabled", "expected_terminal"),
    [([], True, True), (["--use_residual", "false", "--terminal", "false"], False, False)],
)
def test_record_rerun_cli_parses_runtime_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode_args: list[str],
    expected_residual_enabled: bool,
    expected_terminal: bool,
) -> None:
    record_manorl_rerun = _load_tool("record_manorl_rerun")
    captured_configs = []

    class FakeEnvironment:
        def __init__(self, trajectories, config) -> None:
            captured_configs.append(config)

        def step(self, actions) -> None:
            pass

    class FakeRecorder:
        def __init__(self, environment, output, env_id: int) -> None:
            pass

        def record_transition(self) -> None:
            pass

        def close(self) -> Path:
            return tmp_path / "recording.rrd"

    monkeypatch.setattr(
        record_manorl_rerun,
        "load_assigned_trajectory_batch",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(record_manorl_rerun, "_assignment_payload", lambda trajectories: [])
    monkeypatch.setattr(record_manorl_rerun, "MujocoManoEnvironment", FakeEnvironment)
    monkeypatch.setattr(record_manorl_rerun, "ManoRerunRecorder", FakeRecorder)

    assert record_manorl_rerun.main(
        ["--output", str(tmp_path / "recording.rrd"), "--steps", "1", *mode_args]
    ) == 0
    assert captured_configs[0].residual_enabled is expected_residual_enabled
    assert (captured_configs[0].max_deviation_distance == 0.1) is expected_terminal


@pytest.mark.parametrize(
    ("mode_args", "expected_residual_enabled", "expected_terminal"),
    [([], True, True), (["--use_residual", "false", "--terminal", "false"], False, False)],
)
def test_train_cube1_cli_parses_runtime_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode_args: list[str],
    expected_residual_enabled: bool,
    expected_terminal: bool,
) -> None:
    train_manorl_cube1 = _load_tool("train_manorl_cube1")
    captured_budgets = []

    def fake_run(output: Path, budget) -> dict[str, object]:
        captured_budgets.append(budget)
        return {}

    monkeypatch.setattr(train_manorl_cube1, "run", fake_run)

    assert train_manorl_cube1.main(
        ["--output", str(tmp_path / "training"), *mode_args]
    ) == 0
    assert captured_budgets[0].residual_enabled is expected_residual_enabled
    assert captured_budgets[0].terminal is expected_terminal


def test_tile_layout_covers_non_overlapping_grid() -> None:
    assert _tile_layout(1, width=100, height=80) == [(0, 0, 100, 80)]
    assert _tile_layout(4, width=100, height=80) == [
        (0, 40, 50, 40),
        (50, 40, 50, 40),
        (0, 0, 50, 40),
        (50, 0, 50, 40),
    ]


def test_viewer_requires_a_graphical_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from sim.manorl.view_environment import _require_graphical_session

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="graphical session"):
        _require_graphical_session()
