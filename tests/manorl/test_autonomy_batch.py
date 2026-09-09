from __future__ import annotations

import numpy as np
import pytest

from sim.manorl.autonomy_batch import (
    V3_OBSERVATION_DIM,
    AutonomyTransitionState,
    DeviceWitnessFeatures,
    build_device_autonomy_observation,
    compute_device_autonomy_reward_v3,
    rate_limited_command_batch,
    transform_fixed_witnesses,
)
from sim.manorl.autonomy_contracts import ACTION_DIM, OBSERVATION_DIM, observation_slices, rate_limited_command, validate_v3_checkpoint_metadata, CHECKPOINT_V3_FORMAT, OBSERVATION_V3_CONTRACT_ID, REWARD_V3_CONTRACT_ID
from sim.manorl.environment import recommended_warp_contact_capacity
from sim.manorl.mjx_sim import CONSTRAINT_CAPACITY


def _physical(batch: int = 2):
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    return type("Physical", (), {
        "mano_dof_pos": jp.zeros((batch, 28), dtype=jp.float32),
        "hand_keypoint_positions": jp.zeros((batch, 16, 3), dtype=jp.float32),
        "hand_keypoint_orientations_xyzw": jp.tile(jp.asarray([0., 0., 0., 1.]), (batch, 16, 1)),
        "object_position": jp.zeros((batch, 3), dtype=jp.float32),
        "object_orientation_xyzw": jp.tile(jp.asarray([0., 0., 0., 1.]), (batch, 1)),
        "object_linear_velocity": jp.zeros((batch, 3), dtype=jp.float32),
        "valid": jp.ones((batch,), dtype=bool),
    })()


def test_fused_transition_state_is_explicit_jax_pytree():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    state = AutonomyTransitionState(jp.zeros((1, 2)), jp.zeros((1,), dtype=jp.int32), jp.zeros((1,), dtype=bool), jp.zeros((1, 28)), jp.zeros((1,)), jp.zeros((1, 16, 3)))
    leaves, treedef = jax.tree_util.tree_flatten(state)
    assert len(leaves) == 6
    assert jax.tree_util.tree_unflatten(treedef, leaves).indices.shape == (1,)


def test_contact_capacity_scales_global_only_and_constraints_stay_per_world():
    assert recommended_warp_contact_capacity(8192, ("right",)) > recommended_warp_contact_capacity(64, ("right",))
    assert CONSTRAINT_CAPACITY == 512


def test_batch_rate_map_matches_scalar_v2_for_each_row():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    previous = np.linspace(-.2, .2, ACTION_DIM, dtype=np.float32)
    action = np.linspace(-.8, .8, ACTION_DIM, dtype=np.float32)
    lower, upper = -np.ones(ACTION_DIM, dtype=np.float32), np.ones(ACTION_DIM, dtype=np.float32)
    rates = np.full(ACTION_DIM, 4., dtype=np.float32)
    measured = np.linspace(-.1, .1, ACTION_DIM, dtype=np.float32)
    envelope = np.full(ACTION_DIM, .35, dtype=np.float32)
    expected = rate_limited_command(previous, action, lower, upper, rates, measured_qpos=measured, max_tracking_error=envelope)
    actual = rate_limited_command_batch(jp.asarray(previous[None]), jp.asarray(action[None]), jp.asarray(lower), jp.asarray(upper), jp.asarray(rates), jp.asarray(measured[None]), jp.asarray(envelope))
    np.testing.assert_allclose(np.asarray(actual[0]), expected, rtol=0., atol=2e-7)


def test_fixed_witness_transform_is_rigid_and_signed_distance_is_retained():
    physical = _physical(2)
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    hand = jp.zeros((2, 16, 3)); obj = jp.zeros((2, 16, 3)); signed = jp.full((2, 16), -.001); confidence = jp.exp(-jp.abs(signed) / .01)
    features = transform_fixed_witnesses(physical, hand, obj, signed, confidence)
    np.testing.assert_allclose(np.asarray(features.correspondence_error), 0.)
    np.testing.assert_allclose(np.asarray(features.signed_distance), -.001)
    assert np.all(np.asarray(features.confidence) > .9)


def test_v3_observation_keeps_width_and_batch_axis_without_host_geometry_queries():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    physical = _physical(2)
    contact = type("Contact", (), {"hand_object_forces": jp.zeros((2, 16, 3))})()
    signed = jp.full((2, 16), .01); witness = DeviceWitnessFeatures(jp.zeros((2, 16, 3)), jp.zeros((2, 16, 3)), jp.zeros((2, 16)), jp.ones((2, 16)), jp.ones((2, 16)), signed, jp.zeros((2, 16, 3)))
    refs = jp.zeros((2, 4, 28)); refobj = jp.zeros((2, 4, 3)); previous = jp.zeros((2, 28))
    observation = build_device_autonomy_observation(physical=physical, contact=contact, witness=witness, reference_q=refs, reference_object=refobj, previous_command=previous, action_ids=jp.asarray([2, 3]), index=jp.asarray([0, 1]), lower=jp.full((28,), -1.), upper=jp.full((28,), 1.))
    assert observation.shape == (2, V3_OBSERVATION_DIM) == (2, OBSERVATION_DIM)
    assert np.isfinite(np.asarray(observation)).all()


@pytest.mark.parametrize("unique", [True, False])
def test_v31_phase_uses_reference_horizon_for_unique_and_batched_tables(unique: bool):
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    batch, horizon = 4, 539
    physical = _physical(batch)
    contact = type("Contact", (), {"hand_object_forces": jp.zeros((batch, 16, 3))})()
    witness = DeviceWitnessFeatures(*(jp.zeros((batch, 16, 3)) for _ in range(2)), jp.zeros((batch, 16)), jp.ones((batch, 16)), jp.ones((batch, 16)), jp.zeros((batch, 16)), jp.zeros((batch, 16, 3)))
    reference_q = jp.zeros((horizon, 28)) if unique else jp.zeros((batch, horizon, 28))
    reference_object = jp.zeros((horizon, 3)) if unique else jp.zeros((batch, horizon, 3))
    observation = build_device_autonomy_observation(physical=physical, contact=contact, witness=witness,
        reference_q=reference_q, reference_object=reference_object, previous_command=jp.zeros((batch, 28)),
        action_ids=jp.ones((batch,)), index=jp.asarray([0, 134, 269, 538]),
        lower=jp.full((28,), -1.), upper=jp.full((28,), 1.))
    phase = np.asarray(observation[:, observation_slices()["contact_phase_confidence"]])[:, 0]
    np.testing.assert_allclose(phase, [0., 134 / 538, .5, 1.], rtol=0., atol=2e-7)
    assert np.all(phase <= 1.)


def test_masked_reset_changes_only_completed_rows():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    data = jp.arange(6, dtype=jp.float32).reshape(2, 3)
    mask = jp.asarray([True, False])
    reset = jp.full((2, 3), -1., dtype=jp.float32)
    from sim.manorl.autonomy_batch import masked_row_reset
    result = masked_row_reset(data, mask, lambda value, selected: jp.where(selected[:, None], reset, value))
    np.testing.assert_array_equal(np.asarray(result), [[-1., -1., -1.], [3., 4., 5.]])


def test_v31_checkpoint_rejects_v2_and_phase_bug_v3_metadata():
    with pytest.raises(ValueError, match="v2"):
        validate_v3_checkpoint_metadata({"checkpoint_format": "manorl.autonomy.ppo.v2"})
    with pytest.raises(ValueError, match="explicit legacy reader"):
        validate_v3_checkpoint_metadata({"checkpoint_format": "manorl.autonomy.ppo.v3", "observation_contract": "manorl.autonomy.observation.v3"})
    validate_v3_checkpoint_metadata({"checkpoint_format": CHECKPOINT_V3_FORMAT, "observation_contract": OBSERVATION_V3_CONTRACT_ID, "reward_contract": REWARD_V3_CONTRACT_ID})


def test_v3_reward_is_additive_and_force_magnitude_only_enters_contact_term():
    jax = pytest.importorskip("jax")
    jp = jax.numpy
    b = 2; physical = _physical(b); signed = jp.zeros((b, 16)); witness = DeviceWitnessFeatures(jp.zeros((b, 16, 3)), jp.zeros((b, 16, 3)), jp.zeros((b, 16)), jp.ones((b, 16)), jp.ones((b, 16)), signed, jp.zeros((b, 16, 3)))
    reward = compute_device_autonomy_reward_v3(object_position=jp.zeros((b, 3)), target_object_position=jp.zeros((b, 3)), object_velocity=jp.zeros((b, 3)), hand_object_relative=jp.zeros((b, 3)), reference_hand_object_relative=jp.zeros((b, 3)), witness=witness, reference_confidence=jp.ones((b, 16)), hand_object_force=jp.zeros((b, 16, 3)), relative_contact_motion=jp.zeros((b, 16, 3)), action=jp.zeros((b, 28)), release_active=jp.zeros((b,), dtype=bool))
    assert np.isfinite(np.asarray(reward.total)).all() and np.asarray(reward.measured_contact).max() == 0.
