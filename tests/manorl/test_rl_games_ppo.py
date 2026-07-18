from __future__ import annotations

import pytest
import torch

from sim.manorl.rl_games_ppo import (
    RL_GAMES_BOUNDS_SOFT_BOUND,
    RlGamesAdaptiveLR,
    rl_games_bounds_loss,
    rl_games_critic_loss,
    rl_games_policy_kl,
)


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


def test_bounds_loss_matches_source_soft_bound_formula() -> None:
    policy_mean = torch.tensor(
        [[-1.3, -1.1, 0.0, 1.1, 1.4], [0.5, -0.5, 1.0, -1.0, 1.2]],
        dtype=torch.float32,
    )
    expected = (
        torch.clamp_min(policy_mean - 1.1, 0.0).square()
        + torch.clamp_max(policy_mean + 1.1, 0.0).square()
    ).sum(dim=-1).mean()

    assert RL_GAMES_BOUNDS_SOFT_BOUND == 1.1
    torch.testing.assert_close(rl_games_bounds_loss(policy_mean), expected)
    assert rl_games_bounds_loss(torch.tensor([[-1.1, 1.1]])).item() == 0.0


def test_critic_loss_matches_source_clipped_max_formula() -> None:
    current_values = torch.tensor([[0.5], [1.0]], dtype=torch.float32)
    reference_values = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
    returns = torch.tensor([[1.0], [0.5]], dtype=torch.float32)
    clipped_values = reference_values + torch.clamp(
        current_values - reference_values, -0.2, 0.2
    )
    expected = torch.maximum(
        (current_values - returns).square(),
        (clipped_values - returns).square(),
    ).mean()

    torch.testing.assert_close(
        rl_games_critic_loss(
            current_values,
            reference_values,
            returns,
            value_clip=0.2,
        ),
        expected,
    )
    torch.testing.assert_close(
        rl_games_critic_loss(
            current_values,
            reference_values,
            returns,
            value_clip=0.0,
        ),
        (current_values - returns).square().mean(),
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
