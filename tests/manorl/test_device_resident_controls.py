from __future__ import annotations

import numpy as np

from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.trajectory import load_reference_trajectory


def _environment(*, device_resident_controls: bool) -> MujocoManoEnvironment:
    return MujocoManoEnvironment(
        load_reference_trajectory(),
        EnvironmentConfig(
            num_envs=1,
            device="cpu",
            residual_enabled=True,
            max_deviation_distance=1_000_000.0,
            device_resident_controls=device_resident_controls,
            capture_transition_diagnostics=not device_resident_controls,
        ),
    )


def test_device_resident_controls_match_legacy_outputs_and_delayed_reset() -> None:
    """The opt-in path changes storage placement, not the source transition."""

    legacy = _environment(device_resident_controls=False)
    accelerated = _environment(device_resident_controls=True)
    action = np.linspace(-0.9, 0.9, 26, dtype=np.float64).reshape(1, 26)

    for environment in (legacy, accelerated):
        environment.progress[:] = 790
        environment.trajectory_steps[:] = 789

    legacy_output, legacy_reward, legacy_reset, legacy_extras = legacy.step(action)
    accelerated_output, accelerated_reward, accelerated_reset, accelerated_extras = accelerated.step(action)
    np.testing.assert_allclose(accelerated_output["obs"], legacy_output["obs"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(accelerated_reward, legacy_reward, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(accelerated_reset, legacy_reset)
    np.testing.assert_array_equal(accelerated_extras["time_outs"], legacy_extras["time_outs"])
    np.testing.assert_array_equal(accelerated_reset, [True])
    assert accelerated.last_controller_targets is None
    assert accelerated.last_transition is None
    assert accelerated.last_termination is not None
    assert accelerated.last_termination.reset.shape == (1,)

    legacy_output, legacy_reward, legacy_reset, legacy_extras = legacy.step(action)
    accelerated_output, accelerated_reward, accelerated_reset, accelerated_extras = accelerated.step(action)
    np.testing.assert_allclose(accelerated_output["obs"], legacy_output["obs"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(accelerated_reward, legacy_reward, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(accelerated_reset, legacy_reset)
    np.testing.assert_array_equal(accelerated_extras["time_outs"], legacy_extras["time_outs"])
    np.testing.assert_array_equal(accelerated.trajectory_steps, [0])
    assert accelerated.data.qpos.shape[0] == 1
