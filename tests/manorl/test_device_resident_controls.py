from __future__ import annotations

import numpy as np
import pytest
import torch

import sim.manorl.environment as environment_module
from sim.manorl.environment import (
    EnvironmentConfig,
    MujocoManoEnvironment,
    _decode_contact_forces,
    _decode_contact_forces_reference,
)
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


def _contact_decoder_fixture() -> dict[str, object]:
    rng = np.random.default_rng(1234)
    batch = 4
    ngeom = 24
    keypoint_geom_ids = list(range(16))
    object_geom_ids = {20}
    count = 12
    geom = np.asarray(
        [
            (0, 20), (20, 1), (2, 3), (4, 20),
            (20, 5), (6, 7), (8, 20), (20, 9),
            (10, 11), (12, 20), (20, 13), (14, 15),
        ],
        dtype=np.int64,
    )
    world = np.asarray([0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 1], dtype=np.int64)
    dimension = np.full(count, 3, dtype=np.int64)
    addresses = np.asarray(
        [[0, 1, 2, 3], [3, 4, 5, 6], [1, 2, 3, 4], [0, 2, 4, 6]] * 3,
        dtype=np.int64,
    )[:count]
    nefc = np.asarray([8, 7, 8, 7], dtype=np.int64)
    friction = rng.normal(size=(count, 5)).astype(np.float64)
    frame = rng.normal(size=(count, 3, 3)).astype(np.float64)
    constraint_force = rng.normal(size=(batch, 8)).astype(np.float64)
    return {
        "count": count,
        "geom": geom,
        "world": world,
        "dimension": dimension,
        "addresses": addresses,
        "nefc": nefc,
        "friction": friction,
        "frame": frame,
        "constraint_force": constraint_force,
        "ngeom": ngeom,
        "keypoint_geom_ids": keypoint_geom_ids,
        "object_geom_ids": object_geom_ids,
    }


def test_vectorized_contact_decoder_matches_loop_reference() -> None:
    fixture = _contact_decoder_fixture()
    vectorized = _decode_contact_forces(**fixture)
    reference = _decode_contact_forces_reference(**fixture)
    for actual, expected in zip(vectorized, reference):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_vectorized_contact_decoder_rejects_malformed_geom_buffer() -> None:
    fixture = _contact_decoder_fixture()
    fixture["geom"] = np.asarray([0, 20], dtype=np.int64)
    with pytest.raises(RuntimeError, match="geom-pair buffer shape"):
        _decode_contact_forces(**fixture)


def test_runtime_vectorized_decoder_matches_reference_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectorized = _environment(device_resident_controls=False)
    reference = _environment(device_resident_controls=False)
    action = np.linspace(
        -0.65, 0.65, vectorized.action_dim, dtype=np.float64
    ).reshape(1, vectorized.action_dim)

    def assert_step_equal(
        vectorized_result: tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, np.ndarray]],
        reference_result: tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, np.ndarray]],
    ) -> None:
        vectorized_observation, vectorized_reward, vectorized_reset, vectorized_info = vectorized_result
        reference_observation, reference_reward, reference_reset, reference_info = reference_result
        np.testing.assert_allclose(vectorized_observation["obs"], reference_observation["obs"], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(vectorized_reward, reference_reward, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(vectorized_reset, reference_reset)
        np.testing.assert_allclose(vectorized_info["time_outs"], reference_info["time_outs"], rtol=0.0, atol=0.0)
        assert vectorized.last_physical is not None
        assert reference.last_physical is not None
        for field in (
            "hand_position",
            "object_position",
            "hand_keypoint_positions",
            "fingertip_positions",
            "hand_keypoint_contact_forces",
            "object_contact_force",
            "geom_contact_force_world_N",
            "hand_object_force_on_object_world_N",
        ):
            np.testing.assert_allclose(
                getattr(vectorized.last_physical, field),
                getattr(reference.last_physical, field),
                rtol=0.0,
                atol=1e-12,
            )
        np.testing.assert_array_equal(vectorized.last_physical.contact_count, reference.last_physical.contact_count)

    for _ in range(2):
        vectorized_result = vectorized.step(action)
        with monkeypatch.context() as context:
            context.setattr(
                environment_module,
                "_decode_contact_forces",
                environment_module._decode_contact_forces_reference,
            )
            reference_result = reference.step(action)
        assert_step_equal(vectorized_result, reference_result)


def test_phase_profile_is_opt_in_and_records_rollout_boundaries() -> None:
    default = _environment(device_resident_controls=False)
    action = np.linspace(
        -0.4, 0.4, default.action_dim, dtype=np.float64
    ).reshape(1, default.action_dim)
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
    default = _environment(device_resident_controls=False)
    comparison = _environment(device_resident_controls=False)
    action = np.linspace(
        -0.4, 0.4, default.action_dim, dtype=np.float64
    ).reshape(1, default.action_dim)

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
    wrapper.step(
        torch.zeros(
            (1, environment.action_dim),
            dtype=torch.float32,
            device=observations.device,
        )
    )
    assert wrapper.phase_profile()["skrl_cuda_action_to_numpy"]["calls"] == 1
    assert wrapper.phase_profile()["skrl_numpy_response_to_cuda"]["calls"] == 1


def test_device_resident_controls_match_legacy_outputs_and_delayed_reset() -> None:
    """The opt-in path changes storage placement, not the source transition."""

    legacy = _environment(device_resident_controls=False)
    accelerated = _environment(device_resident_controls=True)
    action = np.linspace(
        -0.9, 0.9, legacy.action_dim, dtype=np.float64
    ).reshape(1, legacy.action_dim)

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
