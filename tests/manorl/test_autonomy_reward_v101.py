"""Temporal-state tests for Reward v10.1."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sim.manorl.autonomy_reward_v101 import (
    DEFAULT_REWARD_V101_CONFIG,
    compute_reward,
    initial_reward_state,
    reset_reward_state,
)
from tests.manorl.test_autonomy_reward_v10 import (
    _contact_with_regions,
    _motion_cache,
)
from tests.manorl.test_autonomy_v4 import _contact, _state


def test_persistent_opposition_unlocks_full_grip_reward():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    state = initial_reward_state(1, jp=jp)
    rewards = []
    for _ in range(DEFAULT_REWARD_V101_CONFIG.stable_opposition_frames):
        reward, state = compute_reward(
            _state(),
            _contact_with_regions(0, 13),
            _motion_cache(),
            jp.array([0]),
            jp.zeros((1, 28)),
            state,
        )
        rewards.append(float(reward.opposing_contact[0]))
    assert rewards[:-1] == [0.0] * (len(rewards) - 1)
    assert rewards[-1] > 0
    assert int(state.opposition_streak[0]) == 8


def test_stalled_grip_decays_contact_and_adds_penalty():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    state = initial_reward_state(1, jp=jp)
    first_contact = None
    final = None
    steps = (
        DEFAULT_REWARD_V101_CONFIG.stable_opposition_frames
        + DEFAULT_REWARD_V101_CONFIG.stalled_grace_frames
        + 20
    )
    grounded = _state()._replace(object_bottom=jp.array([0.0]))
    for index in range(steps):
        reward, state = compute_reward(
            grounded,
            _contact_with_regions(0, 13),
            _motion_cache(),
            jp.array([0]),
            jp.zeros((1, 28)),
            state,
        )
        if index == 0:
            first_contact = float(reward.contact[0])
        final = reward
    assert float(final.contact[0]) < first_contact
    assert float(final.stalled_contact[0]) < 0


def test_airborne_landing_clears_temporal_state_and_severe_is_minus_100():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    state = initial_reward_state(1, jp=jp)
    airborne = _state()._replace(
        object_bottom=jp.array([0.02]),
        object_v_com=jp.array([[0.0, 0.0, 0.02]]),
    )
    for _ in range(DEFAULT_REWARD_V101_CONFIG.stable_opposition_frames):
        reward, state = compute_reward(
            airborne,
            _contact_with_regions(0, 13),
            _motion_cache(0.2, clearance=0.1),
            jp.array([0]),
            jp.zeros((1, 28)),
            state,
        )
    assert bool(state.had_airborne[0])
    landed = _state()._replace(object_bottom=jp.array([0.0]))
    _, state = compute_reward(
        landed,
        _contact(),
        _motion_cache(),
        jp.array([0]),
        jp.zeros((1, 28)),
        state,
    )
    assert not bool(state.had_airborne[0])
    assert int(state.opposition_streak[0]) == 0

    fallen = _state()._replace(object_bottom=jp.array([-0.06]))
    severe, _ = compute_reward(
        fallen,
        _contact(),
        _motion_cache(),
        jp.array([0]),
        jp.zeros((1, 28)),
        initial_reward_state(1, jp=jp),
    )
    np.testing.assert_allclose(severe.severe, [-100.0], atol=0)


def test_explicit_reset_clears_only_selected_rows():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    state = initial_reward_state(2, jp=jp)._replace(
        opposition_streak=jp.array([5, 7]),
        airborne_hold_streak=jp.array([2, 3]),
        stalled_contact_streak=jp.array([1, 4]),
        had_airborne=jp.array([True, True]),
    )
    reset = reset_reward_state(state, jp.array([True, False]), jp=jp)
    np.testing.assert_array_equal(reset.opposition_streak, [0, 7])
    np.testing.assert_array_equal(reset.had_airborne, [False, True])
