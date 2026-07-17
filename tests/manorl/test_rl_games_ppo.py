from __future__ import annotations

import pytest
import torch

from sim.manorl.rl_games_ppo import RlGamesAdaptiveLR, rl_games_policy_kl


def test_policy_kl_matches_source_rl_games_formula() -> None:
    current_mean = torch.tensor([[0.1, -0.2], [0.4, 0.3]], dtype=torch.float32)
    current_std = torch.tensor([[0.7, 0.8], [0.9, 1.1]], dtype=torch.float32)
    reference_mean = torch.tensor([[0.0, -0.1], [0.2, 0.5]], dtype=torch.float32)
    reference_std = torch.tensor([[0.6, 0.9], [1.0, 0.8]], dtype=torch.float32)

    expected = (
        torch.log(reference_std / current_std + 1.0e-5)
        + (
            current_std.square() + (reference_mean - current_mean).square()
        )
        / (2.0 * (reference_std.square() + 1.0e-5))
        - 0.5
    ).sum(dim=-1).mean()

    torch.testing.assert_close(
        rl_games_policy_kl(
            current_mean, current_std, reference_mean, reference_std
        ),
        expected,
    )
    with pytest.raises(ValueError, match="identical shapes"):
        rl_games_policy_kl(
            current_mean[:, :1], current_std, reference_mean, reference_std
        )


def test_legacy_adaptive_scheduler_matches_source_thresholds() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=3.0e-4)
    scheduler = RlGamesAdaptiveLR(optimizer, kl_threshold=0.016)

    scheduler.step(0.04)
    assert scheduler.get_last_lr() == pytest.approx([2.0e-4])
    scheduler.step(0.02)
    assert scheduler.get_last_lr() == pytest.approx([2.0e-4])
    scheduler.step(0.004)
    assert scheduler.get_last_lr() == pytest.approx([3.0e-4])

    for _ in range(32):
        scheduler.step(1.0)
    assert scheduler.get_last_lr() == pytest.approx([1.0e-6])
    for _ in range(32):
        scheduler.step(0.0)
    assert scheduler.get_last_lr()[0] <= 1.0e-2


def test_scheduler_rejects_nonpositive_threshold() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=3.0e-4)
    with pytest.raises(ValueError, match="positive"):
        RlGamesAdaptiveLR(optimizer, kl_threshold=0.0)
