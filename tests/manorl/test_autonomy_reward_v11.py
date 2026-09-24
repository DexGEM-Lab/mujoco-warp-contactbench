"""Numerical tests for the faithful v11 (v5-log-reward port)."""
from __future__ import annotations

import numpy as np
import pytest

from sim.manorl.autonomy_contracts import (
    V11_REWARD_CONTRACT_ID,
    V11_REWARD_TERM_NAMES,
    resolve_reward_contract,
)
from sim.manorl.autonomy_reward_v11 import (
    MOTION_GATE_LOW,
    POSITION_HALF_M,
    POSITION_HORIZON_M,
    REWARD_GEOMETRY_COEF,
    REWARD_POSITION_COEF,
    REWARD_SEVERE_PENALTY,
    REWARD_VELOCITY_COEF,
    ROTATION_MAX,
    STATIC_ROT_ERROR_DEG,
    STATIC_ROT_WEIGHT,
    RewardV11Config,
    compute_reward,
    log_fraction,
)
from tests.manorl.test_autonomy_v4 import _cache, _contact, _state


def test_contract_resolution_and_term_layout():
    version, contract_id = resolve_reward_contract("v11")
    assert version == "v11"
    assert contract_id == V11_REWARD_CONTRACT_ID
    assert len(V11_REWARD_TERM_NAMES) == 9
    assert V11_REWARD_TERM_NAMES[0] == "object_position"
    assert "hold_contact" not in V11_REWARD_TERM_NAMES


def test_log_fraction_anchors():
    jp = pytest.importorskip("jax").numpy
    curve = log_fraction(
        jp.asarray([0.0, POSITION_HALF_M, POSITION_HORIZON_M, 2 * POSITION_HORIZON_M]),
        POSITION_HALF_M,
        POSITION_HORIZON_M,
        jp=jp,
    )
    np.testing.assert_allclose(np.asarray(curve), [0.0, 0.5, 1.0, 1.0], atol=1e-6)
    with pytest.raises(ValueError):
        log_fraction(jp.asarray([0.0]), 0.0, POSITION_HORIZON_M, jp=jp)


def test_aligned_step_scores_and_gate():
    jp = pytest.importorskip("jax").numpy
    state, cache, action = _state(), _cache(), jp.zeros((1, 28))
    reward = compute_reward(state, _contact(), cache, jp.array([0]), action)
    # static reference: position/velocity keep the 1% floor; rotation aligned +0.3 x floor
    assert float(reward.object_position[0]) == pytest.approx(
        REWARD_POSITION_COEF * 1.2 * 0.01, abs=1e-4
    )
    assert float(reward.object_velocity[0]) == pytest.approx(
        REWARD_VELOCITY_COEF * 0.1 * 0.01, abs=1e-4
    )
    assert float(reward.object_rotation[0]) == pytest.approx(ROTATION_MAX * 0.01, abs=1e-4)
    # fingers/hand/geometry at alignment near their maxima (no gate)
    assert float(reward.fingers[0]) > 0.19
    assert float(reward.geometry[0]) <= REWARD_GEOMETRY_COEF + 1e-4
    assert float(reward.severe[0]) == 0.0
    assert bool(reward.done[0]) is False


def test_total_is_sum_of_nine_terms():
    jp = pytest.importorskip("jax").numpy
    state, cache, action = _state(), _cache(), jp.zeros((1, 28))
    reward = compute_reward(state, _contact(), cache, jp.array([0]), action)
    parts = sum(float(getattr(reward, n)[0]) for n in V11_REWARD_TERM_NAMES)
    assert float(reward.total[0]) == pytest.approx(parts, abs=1e-4)


def test_static_rotation_override_escalates():
    jp = pytest.importorskip("jax").numpy
    state, cache, action = _state(), _cache(), jp.zeros((1, 28))
    reward = compute_reward(state, _contact(), cache, jp.array([0]), action)
    assert float(reward.object_rotation[0]) <= ROTATION_MAX * STATIC_ROT_WEIGHT + 1e-6
    # rotation term respects the static override constants
    assert STATIC_ROT_ERROR_DEG == 45.0
    assert STATIC_ROT_WEIGHT == 0.75
    assert MOTION_GATE_LOW == 0.01


def test_severe_penalty_configurable():
    config = RewardV11Config(severe_penalty=-75.0)
    assert config.severe_penalty == -REWARD_SEVERE_PENALTY
    with pytest.raises(ValueError):
        RewardV11Config(severe_penalty=10.0)
