"""Numerical tests for the v6-compatible Reward v10 implementation."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sim.manorl.autonomy_contracts import (
    V10_REWARD_CONTRACT_ID,
    V10_REWARD_TERM_NAMES,
    resolve_reward_contract,
    validate_v6_checkpoint_metadata,
)
from sim.manorl.autonomy_reward_v10 import RewardV10Config, compute_reward
from tests.manorl.test_autonomy_v4 import _cache, _contact, _state


def _motion_cache(speed: float = 0.0, *, clearance: float = 0.0):
    cache = _cache()
    velocity = np.zeros_like(cache.object_v_com)
    velocity[:, 2] = speed
    return replace(
        cache,
        object_v_com=velocity,
        reference_bottom=np.full_like(cache.reference_bottom, clearance),
        table_height=0.0,
    )


def _contact_with_regions(*regions: int, force: float = 0.02):
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    contact = _contact()
    forces = contact.paired_force_on_object
    counts = contact.paired_count
    for region in regions:
        forces = forces.at[0, region, 0].set(force)
        counts = counts.at[0, region].set(1)
    return contact._replace(
        paired_force_on_object=forces,
        paired_count=counts,
    )


def _sum(reward):
    return sum(getattr(reward, name) for name in V10_REWARD_TERM_NAMES)


def test_contact_thumb_and_opposition_require_loaded_regions():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    state, cache, action = _state(), _motion_cache(), jp.zeros((1, 28))
    nonthumb = compute_reward(
        state, _contact_with_regions(0), cache, jp.array([0]), action,
    )
    thumb = compute_reward(
        state, _contact_with_regions(13), cache, jp.array([0]), action,
    )
    opposing = compute_reward(
        state, _contact_with_regions(0, 13), cache, jp.array([0]), action,
    )
    assert float(nonthumb.contact[0]) > 0
    np.testing.assert_allclose(nonthumb.thumb_contact, [0.0], atol=0)
    assert float(thumb.thumb_contact[0]) > 0
    np.testing.assert_allclose(thumb.opposing_contact, [0.0], atol=0)
    np.testing.assert_allclose(opposing.opposing_contact, [2.0], atol=1e-6)


def test_motion_scaled_contact_preserves_v8_bounds():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    state, contact, action = (
        _state(),
        _contact_with_regions(0, 13),
        jp.zeros((1, 28)),
    )
    static = compute_reward(
        state, contact, _motion_cache(0.0), jp.array([0]), action,
    )
    moving = compute_reward(
        state, contact, _motion_cache(0.15), jp.array([0]), action,
    )
    np.testing.assert_allclose(static.contact_scale, [1.0], atol=1e-7)
    np.testing.assert_allclose(moving.contact_scale, [2.0], atol=1e-7)
    np.testing.assert_allclose(moving.contact, 2 * static.contact, atol=1e-6)
    np.testing.assert_allclose(
        moving.thumb_contact, 2 * static.thumb_contact, atol=1e-6,
    )


def test_lift_direction_slip_hold_loss_and_falling_terms():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    action = jp.zeros((1, 28))
    cache = _motion_cache(0.20, clearance=0.10)
    missing = compute_reward(
        _state()._replace(object_bottom=jp.array([0.0])),
        _contact(),
        cache,
        jp.array([0]),
        action,
    )
    np.testing.assert_allclose(missing.lift_progress, [-16.0], atol=1e-6)

    held_state = _state()._replace(
        object_bottom=jp.array([0.02]),
        object_v_com=jp.array([[0.10, 0.0, 0.10]]),
    )
    held = compute_reward(
        held_state,
        _contact_with_regions(0, 13),
        cache,
        jp.array([0]),
        action,
    )
    assert float(held.lift_velocity[0]) > 0
    np.testing.assert_allclose(held.lateral_slip, [-4.0], atol=1e-6)
    np.testing.assert_allclose(held.hold_contact, [12.0], atol=1e-6)
    np.testing.assert_allclose(held.contact_loss, [0.0], atol=2e-6)

    lost = compute_reward(
        held_state,
        _contact_with_regions(13),
        cache,
        jp.array([0]),
        action,
    )
    np.testing.assert_allclose(lost.contact_loss, [-20.0], atol=1e-6)

    falling = compute_reward(
        held_state._replace(
            object_v_com=jp.array([[0.0, 0.0, -0.20]])
        ),
        _contact_with_regions(0, 13),
        cache,
        jp.array([0]),
        action,
    )
    np.testing.assert_allclose(falling.falling, [-12.0], atol=1e-6)
    np.testing.assert_allclose(falling.total, _sum(falling), atol=3e-6)


def test_severe_failure_is_minus_400_and_v10_contract_is_supported():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    fallen = _state()._replace(object_bottom=jp.array([-0.06]))
    reward = compute_reward(
        fallen,
        _contact(),
        _motion_cache(),
        jp.array([0]),
        jp.zeros((1, 28)),
    )
    np.testing.assert_allclose(reward.severe, [-400.0], atol=0)
    assert int(reward.reason[0]) & 4
    assert resolve_reward_contract("v10") == (
        "v10",
        V10_REWARD_CONTRACT_ID,
    )
    validate_v6_checkpoint_metadata(
        {
            "checkpoint_format": "manorl.autonomy.ppo.v6",
            "observation_contract": (
                "manorl.autonomy.observation.v6."
                "region-token-cross-attention"
            ),
            "reward_contract": V10_REWARD_CONTRACT_ID,
            "action_contract": "manorl.autonomy.action.v4",
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("loaded_contact_force_n", 0.0),
        ("contact_motion_speed_high_mps", 0.01),
        ("hold_contact_reward_max", -1.0),
        ("airborne_clearance_m", np.inf),
        ("severe_penalty", 1.0),
    ),
)
def test_v10_config_rejects_invalid_coefficients(field, value):
    with pytest.raises(ValueError):
        RewardV10Config(**{field: value})


def test_v10_jit_matches_eager():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    physical = _state()._replace(
        object_bottom=jp.array([0.02]),
        object_v_com=jp.array([[0.01, 0.0, -0.10]]),
    )
    cache = _motion_cache(0.20, clearance=0.10)
    contact, action = _contact_with_regions(0, 13), jp.zeros((1, 28))
    eager = compute_reward(physical, contact, cache, jp.array([0]), action)
    compiled = jax.jit(
        lambda i: compute_reward(physical, contact, cache, i, action)
    )(jp.array([0]))
    for actual, expected in zip(compiled, eager):
        np.testing.assert_allclose(actual, expected, atol=3e-6)
