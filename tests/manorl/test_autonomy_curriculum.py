"""Curriculum sampling and promotion tests."""
from __future__ import annotations

from sim.manorl.autonomy_curriculum import (
    CurriculumConfig,
    CurriculumController,
    mixed_stage_frame,
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


def test_mixed_stage_frame_defaults_to_stage_candidates():
    candidates = reference_stage_candidates(_cache(64))
    for stage in (1, 2, 3):
        for env in range(100):
            assert mixed_stage_frame(candidates, stage, env) in candidates[stage]


def test_mixed_stage_frame_mixes_full_horizon_and_previous_stages():
    candidates = reference_stage_candidates(_cache(64))
    frames = {
        env: mixed_stage_frame(
            candidates, 3, env, mix_previous=0.25, full_horizon=0.15
        )
        for env in range(100)
    }
    assert all(frames[env] == 0 for env in range(15))
    stage1_or_2 = set(candidates[1]) | set(candidates[2])
    assert all(frames[env] in stage1_or_2 for env in range(15, 40))
    assert all(frames[env] in candidates[3] for env in range(40, 100))


def test_mixed_stage_frame_stage_one_skips_previous_mix():
    candidates = reference_stage_candidates(_cache(64))
    frames = {
        env: mixed_stage_frame(
            candidates, 1, env, mix_previous=0.25, full_horizon=0.15
        )
        for env in range(100)
    }
    assert all(frames[env] == 0 for env in range(15))
    assert all(frames[env] in candidates[1] for env in range(15, 100))
