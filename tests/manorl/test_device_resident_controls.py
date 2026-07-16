from __future__ import annotations

import numpy as np
import torch

from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv
from sim.manorl.skrl_runtime import ProfiledGymnasiumWrapper
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


def test_phase_profile_is_opt_in_and_records_rollout_boundaries() -> None:
    action = np.linspace(-0.4, 0.4, 26, dtype=np.float64).reshape(1, 26)
    default = _environment(device_resident_controls=False)
    assert default.phase_profile() == {}

    profiled = MujocoManoEnvironment(
        load_reference_trajectory(),
        EnvironmentConfig(
            num_envs=1,
            device="cpu",
            residual_enabled=True,
            max_deviation_distance=1_000_000.0,
            profile_phases=True,
        ),
    )
    profiled.reset_mask[:] = True
    profiled.step(action)
    required = {
        "action_numpy_materialization",
        "action_conversion_processing",
        "controller_target_work",
        "mjx_physics",
        "delayed_reset_application",
        "reset_indexed_writes",
        "reset_full_batch_forward",
        "state_contact_extraction",
        "state_materialization",
        "contact_buffer_materialization",
        "python_contact_decode",
        "termination",
        "reward",
        "observation",
        "transition_recording",
    }
    assert required <= set(profiled.phase_profile())
    assert all(profiled.phase_profile()[name]["calls"] == 1 for name in required)

    contact = profiled.contact_profile()
    assert contact["nacon"]["calls"] == 1
    assert len(contact["nacon_per_step"]) == 1
    assert contact["nacon"]["max"] <= contact["capacity"]
    assert contact["raw_buffers"]["contact__geom"]["bytes"] > 0
    assert contact["host_materialization"]["contact__geom"]["bytes"] > 0
    assert "pcie" not in repr(contact).lower()


def test_profile_false_never_calls_explicit_sync_and_preserves_step_semantics() -> None:
    action = np.linspace(-0.4, 0.4, 26, dtype=np.float64).reshape(1, 26)
    default = _environment(device_resident_controls=False)
    comparison = _environment(device_resident_controls=False)

    def unexpected_sync() -> None:
        raise AssertionError("profile=false must not invoke the profiling synchronizer")

    default._profile_sync = unexpected_sync  # type: ignore[method-assign]
    actual = default.step(action)
    expected = comparison.step(action)
    np.testing.assert_allclose(actual[0]["obs"], expected[0]["obs"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(actual[2], expected[2])
    assert default.phase_profile() == {}
    assert default.contact_profile() == {}


def test_profiled_skrl_wrapper_splits_action_and_response_conversions() -> None:
    environment = _environment(device_resident_controls=False)
    wrapper = ProfiledGymnasiumWrapper(ManoGymnasiumVectorEnv(environment))
    observations, _ = wrapper.reset()
    wrapper.step(torch.zeros((1, 26), dtype=torch.float32, device=observations.device))
    assert wrapper.phase_profile()["skrl_cuda_action_to_numpy"]["calls"] == 1
    assert wrapper.phase_profile()["skrl_numpy_response_to_cuda"]["calls"] == 1


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
