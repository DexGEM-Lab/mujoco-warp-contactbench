from __future__ import annotations

import pytest


def test_viewer_cli_is_residual_off_and_loops_by_default() -> None:
    from sim.manorl.view_environment import parse_args

    args = parse_args([])
    assert args.device == "cpu"
    assert args.speed == 0.25
    assert args.loop is True
    assert args.print_every == 10
    assert args.training_termination is False
    assert args.trajectory == "accepted"

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
        ]
    )
    assert one_shot.device == "gpu"
    assert one_shot.loop is False
    assert one_shot.speed == 1.0
    assert one_shot.training_termination is True
    assert one_shot.trajectory == "generated-cube1-row-507"


def test_viewer_requires_a_graphical_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from sim.manorl.view_environment import _require_graphical_session

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="graphical session"):
        _require_graphical_session()
