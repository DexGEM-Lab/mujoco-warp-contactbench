from __future__ import annotations

import pytest

from sim.manorl.view_environment import _tile_layout, parse_args


def test_viewer_cli_is_residual_off_and_loops_by_default() -> None:
    args = parse_args([])
    assert args.device == "cpu"
    assert args.speed == 0.25
    assert args.loop is True
    assert args.print_every == 10
    assert args.training_termination is False
    assert args.trajectory == "accepted"
    assert args.num_envs == 1
    assert args.render_env == 0
    assert args.tile_envs == 1

    one_shot = parse_args(
        [
            "--device",
            "gpu",
            "--no-loop",
            "--speed",
            "1.0",
            "--training-termination",
            "--trajectory",
            "generated-cube1-row-507",
            "--num-envs",
            "10",
            "--render-env",
            "9",
            "--tile-envs",
            "10",
        ]
    )
    assert one_shot.device == "gpu"
    assert one_shot.loop is False
    assert one_shot.speed == 1.0
    assert one_shot.training_termination is True
    assert one_shot.trajectory == "generated-cube1-row-507"
    assert one_shot.num_envs == 10
    assert one_shot.render_env == 9
    assert one_shot.tile_envs == 10


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
