"""Directional anti-windup tests against the actual jitted transition branch."""
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

from sim.manorl.autonomy_contracts import rate_limited_command
from sim.manorl.autonomy_batch import BatchedAutonomyRuntime
from sim.manorl.autonomy_v4 import DOF_RATE, ANTIWINDUP_ERROR


class Data(NamedTuple):
    qpos: object
    ctrl: object

    def replace(self, **kwargs):
        return self._replace(**kwargs)


class Physical(NamedTuple):
    q_raw: object
    valid: object


class Valid(NamedTuple):
    valid: object


@pytest.fixture
def runtime_command(monkeypatch):
    jax = pytest.importorskip("jax")
    j = jax.numpy
    runtime = BatchedAutonomyRuntime.__new__(BatchedAutonomyRuntime)
    runtime.jp = j
    runtime.rate, runtime.envelope = DOF_RATE.copy(), ANTIWINDUP_ERROR.copy()
    runtime.lower, runtime.upper = -np.ones(28, np.float32), np.ones(28, np.float32)
    runtime.cache = SimpleNamespace(control_timestep=1 / 120)
    runtime._physical = lambda data, previous: Physical(data.qpos, j.asarray(True))
    # Isolate the real command branch from physics/reward; no copied mapper.
    runtime._step_batch = runtime._forward_batch = lambda data: data
    runtime._observe = lambda data, index, command: (
        runtime._physical(data, command), Valid(j.asarray(True)), command, command
    )
    monkeypatch.setattr("sim.manorl.autonomy_batch.compute_reward", lambda *args: Valid(j.asarray(True)))

    @jax.jit
    def command(previous, measured, action, execute):
        result = runtime._transition(
            Data(measured, previous), j.zeros(previous.shape[0], j.int32), previous, action, execute
        )
        return result[2]

    return command


@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("error_factor,action_factor,delta_factor", [
    (1., 1., 0.),       # Exact margin: outward blocks, target retained.
    (2., 1., 0.),       # Push-back beyond margin never drags target to q.
    (1., -1., -1.),     # Reverse at margin always available.
    (2., -1., -1.),     # Reverse beyond margin always available.
    (2., 0., 0.),       # Zero action holds the loaded command.
    (.5, .5, .5),       # Inside margin: ordinary rate integration.
    (.99, 3., 1.),      # Can cross margin; raw action still clips to one.
])
def test_directional_numpy_and_jitted_transition_parity(runtime_command, sign, error_factor, action_factor, delta_factor):
    previous = (sign * error_factor * ANTIWINDUP_ERROR).astype(np.float32)
    measured = np.zeros(28, np.float32)
    action = np.full(28, sign * action_factor, np.float32)
    expected = previous.astype(np.float64) + sign * delta_factor * DOF_RATE.astype(np.float64) / 120
    expected = np.clip(expected, -1, 1)
    numpy_command = rate_limited_command(previous, action, -np.ones(28), np.ones(28), DOF_RATE,
        measured_qpos=measured, max_tracking_error=ANTIWINDUP_ERROR)
    np.testing.assert_allclose(numpy_command, expected, atol=1e-8)
    # Two opposite rows prove world-wise broadcast and direction, not only N1.
    actual = np.asarray(runtime_command(
        np.stack([previous, -previous]), np.stack([measured, measured]),
        np.stack([action, -action]), True))
    np.testing.assert_allclose(actual, np.stack([numpy_command, -numpy_command]), atol=1e-7)


@pytest.mark.parametrize("sign", [-1., 1.])
def test_joint_limits_still_clip_and_nonexecution_preserves_command(runtime_command, sign):
    previous = np.full(28, sign * .999, np.float32)
    action = np.full(28, sign * 5., np.float32)
    numpy_command = rate_limited_command(previous, action, -np.ones(28), np.ones(28), DOF_RATE,
        measured_qpos=previous, max_tracking_error=ANTIWINDUP_ERROR)
    np.testing.assert_array_equal(numpy_command, np.full(28, sign))
    actual = runtime_command(previous[None], previous[None], action[None], True)
    np.testing.assert_array_equal(actual, numpy_command[None])
    held = runtime_command(previous[None], previous[None], action[None], False)
    np.testing.assert_array_equal(held, previous[None])


def test_numpy_unmeasured_fallback_and_invalid_rate():
    zero, one = np.zeros(28), np.ones(28)
    np.testing.assert_allclose(rate_limited_command(zero, 3 * one, -one, one, DOF_RATE), DOF_RATE / 120)
    with pytest.raises(ValueError, match="rates"):
        rate_limited_command(zero, one, -one, one, zero)
