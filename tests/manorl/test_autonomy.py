from __future__ import annotations

import numpy as np

from sim.manorl.autonomy import dense_autonomous_reward, shared_support_alignment, surface_intent
from sim.manorl.autonomy_contracts import (
    ACTION_DIM, OBSERVATION_DIM, OBSERVATION_CONTRACT, rate_limited_command,
)


def test_autonomous_contract_and_reference_independent_command_map() -> None:
    assert ACTION_DIM == 28
    assert OBSERVATION_DIM == OBSERVATION_CONTRACT.dimension
    previous = np.zeros(ACTION_DIM)
    action = np.full(ACTION_DIM, 0.5)
    limits = np.full(ACTION_DIM, 1.0)
    command = rate_limited_command(previous, action, -limits, limits, np.full(ACTION_DIM, 0.1))
    np.testing.assert_allclose(command, 0.05)
    # A fixed physical state/history/action has no reference argument and is
    # therefore invariant to any reference trajectory change.
    np.testing.assert_array_equal(command, rate_limited_command(previous, action, -limits, limits, np.full(ACTION_DIM, 0.1)))


def test_surface_intent_is_real_mesh_geometry() -> None:
    surface = np.asarray([[-.05, 0, 0], [.05, 0, 0], [0, .05, 0], [0, 0, .05]])
    points = np.asarray(surface[[0, 1, 2, 3]])
    points = np.concatenate((points, np.repeat(points[:1], 12, axis=0)))
    proximity, anchors, confidence = surface_intent(points, np.zeros(3), np.array([0, 0, 0, 1.]), surface)
    np.testing.assert_allclose(anchors, points)
    np.testing.assert_allclose(proximity, 1.0)
    np.testing.assert_allclose(confidence, 1.0)


def test_alignment_and_dense_reward_work_before_object_move() -> None:
    hand, obj, shift = shared_support_alignment(np.array([0., 0., .2]), np.array([0., 0., .1]), np.array([[0., 0., -.05], [0., 0., .05]]))
    np.testing.assert_allclose(shift, [0., 0., -.051])
    np.testing.assert_allclose(hand - obj, [.0, .0, .1])
    terms = dense_autonomous_reward(obj, obj, np.zeros(3), hand-obj, hand-obj, np.ones(16), np.zeros(16), np.zeros(28), phase=0.)
    assert terms["object_motion"] > 0.99
    assert terms["surface_proximity_contact"] > 0.99
    assert set(("object_motion", "reference_hand_object_relationship", "surface_proximity_contact", "stability", "release")) <= terms.keys()
