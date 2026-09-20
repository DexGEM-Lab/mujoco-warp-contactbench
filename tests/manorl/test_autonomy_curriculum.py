"""Curriculum sampling and promotion tests."""
from __future__ import annotations

from sim.manorl.autonomy_curriculum import (
    CurriculumConfig,
    CurriculumController,
    reference_stage_candidates,
)
from tests.manorl.test_autonomy_v4 import _cache


def _metrics(opposition=0.0, lift=0.0, airborne=0.0):
    return {
        "curriculum/opposing_loaded_fraction": opposition,
        "curriculum/positive_lift_fraction": lift,
        "curriculum/stable_airborne_fraction": airborne,
    }


def test_reference_stage_candidates_are_ordered_and_bounded():
    cache = _cache(64)
    candidates = reference_stage_candidates(cache)
    assert set(candidates) == {1, 2, 3}
    assert all(values for values in candidates.values())
    assert all(
        0 <= frame < len(cache.q_feasible)
        for values in candidates.values()
        for frame in values
    )
    assert max(candidates[1]) <= max(candidates[2]) <= max(candidates[3])


def test_curriculum_promotes_only_after_window_and_stage_threshold():
    config = CurriculumConfig(averaging_updates=2, minimum_updates_per_stage=2)
    controller = CurriculumController(config)
    assert controller.observe(_metrics(opposition=0.2))["curriculum/stage"] == 1
    promoted = controller.observe(_metrics(opposition=0.2))
    assert promoted["curriculum/stage"] == 2
    assert promoted["curriculum/promoted"] == 1
    assert controller.observe(_metrics(lift=0.1))["curriculum/stage"] == 2
    promoted = controller.observe(_metrics(lift=0.1))
    assert promoted["curriculum/stage"] == 3
    assert promoted["curriculum/promoted"] == 1
