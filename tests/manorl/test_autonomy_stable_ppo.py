"""Stable-v6 PPO guard and anchor schedule tests."""
from __future__ import annotations

import pytest

from sim.manorl.autonomy_batch_training import teacher_anchor_beta_for_update
from sim.manorl.autonomy_training import teacher_anchor_metadata, v4_ppo_config
from sim.manorl.rl_games_ppo import target_kl_exceeded


def test_target_kl_is_opt_in_and_strictly_stops_above_threshold():
    assert not target_kl_exceeded(10.0, None)
    assert not target_kl_exceeded(0.05, 0.05)
    assert target_kl_exceeded(0.05001, 0.05)
    with pytest.raises(ValueError):
        target_kl_exceeded(0.0, 0.0)


def test_stable_v6_recipe_and_linear_anchor_schedule():
    config = v4_ppo_config(
        rollouts=32,
        learning_epochs=2,
        mini_batches=16,
        learning_rate=1e-5,
        target_kl=0.05,
    )
    assert config["learning_rate"] == 1e-5
    assert config["learning_epochs"] == 2
    assert teacher_anchor_beta_for_update(1.0, 0.2, update=1, updates=5) == 1.0
    assert teacher_anchor_beta_for_update(1.0, 0.2, update=5, updates=5) == 0.2
    assert teacher_anchor_beta_for_update(1.0, 0.2, update=3, updates=5) == pytest.approx(0.6)
    metadata = teacher_anchor_metadata(
        1.0, 2, final_beta=0.2, schedule_updates=1000
    )
    assert metadata["schedule"] == {
        "type": "linear_by_update",
        "start_beta": 1.0,
        "final_beta": 0.2,
        "updates": 1000,
    }
