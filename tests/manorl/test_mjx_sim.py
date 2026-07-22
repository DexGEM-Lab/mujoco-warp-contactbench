from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sim.manorl.contracts import (
    CONTROL_STEP_COUNT,
    DATASET_PATH,
    JOINT_DOF,
    REFERENCE_FRAME_COUNT,
    SOURCE_SLICE,
    TRAJECTORY_IDENTITY,
    ServoConfig,
)
from sim.manorl.mjx_sim import (
    MjxWarpReplay,
    MujocoCpuReplay,
    command_target,
    run_reference_replay,
    source_counter_indices,
)
from sim.manorl.replay_reference import summarize_trace
from sim.manorl.trajectory import ReferenceTrajectory, load_reference_trajectory


def _runtime_ready(*, require_mjx: bool = False) -> tuple[bool, str]:
    required = ["mujoco", "scipy"]
    if require_mjx:
        required.extend(["jax", "mujoco.mjx", "warp"])
    absent = [name for name in required if importlib.util.find_spec(name) is None]
    if absent:
        return False, f"runtime modules absent: {', '.join(absent)}"
    return True, ""


def _synthetic_servo_trajectory() -> ReferenceTrajectory:
    """A controller-only fixture, never a replacement for Lance acceptance data."""

    q_ref = np.zeros((REFERENCE_FRAME_COUNT, 26), dtype=np.float64)
    q_ref[1:, :3] = [0.01, -0.005, 0.008]
    q_ref[1:, 3:6] = [0.05, -0.04, 0.03]
    q_ref[1:, 6:] = 0.02
    object_pos = np.repeat([[0.0, 0.0, 1.0]], REFERENCE_FRAME_COUNT, axis=0)
    object_quat = np.repeat([[0.0, 0.0, 0.0, 1.0]], REFERENCE_FRAME_COUNT, axis=0)
    source_indices = np.arange(*SOURCE_SLICE, dtype=np.int64)
    timestamps = np.arange(REFERENCE_FRAME_COUNT, dtype=np.float64) * 0.01
    return ReferenceTrajectory(
        identity=TRAJECTORY_IDENTITY,
        dataset_version=TRAJECTORY_IDENTITY.dataset_version,
        source_indices=source_indices,
        timestamps=timestamps,
        q_ref=q_ref,
        object_pos_raw=object_pos.copy(),
        object_pos=object_pos,
        object_quat_xyzw=object_quat,
        object_z_shift=0.0,
    )


def test_command_clamp_and_shortest_wrist_wrap() -> None:
    lower = np.full(26, -1.0)
    upper = np.full(26, 1.0)
    current = np.zeros(26)
    current[3:6] = [3.10, -3.10, 0.0]
    reference = np.full(26, 2.0)
    reference[3:6] = [-3.10, 3.10, 4.0 * np.pi]
    target = command_target(reference, current, lower, upper)
    np.testing.assert_array_equal(target[:3], 1.0)
    assert np.all(np.abs(target[3:6] - current[3:6]) <= np.pi)
    assert target[3] == pytest.approx(1.0)
    assert target[4] == pytest.approx(-1.0)
    assert target[5] == pytest.approx(0.0)


def test_residual_off_action_invariance_and_target_timing_cpu() -> None:
    available, reason = _runtime_ready()
    if not available:
        pytest.skip(reason)
    trajectory = _synthetic_servo_trajectory()
    zero_replay = MujocoCpuReplay(trajectory)
    random_replay = MujocoCpuReplay(trajectory)
    zero_trace = zero_replay.step(np.zeros(26))
    random_trace = random_replay.step(np.linspace(-10.0, 10.0, 26))
    assert zero_trace.target_index == 0
    assert zero_trace.reference_index == 0
    assert zero_trace.source_reference_index == SOURCE_SLICE[0]
    assert zero_trace.sim_time == pytest.approx(0.005)
    # Legacy q_ref/action input remains 26-wide, but the pinned runtime model
    # and replay trace are expanded to the revised 28-DoF actuator ABI.
    assert zero_trace.actuator_force_substeps.shape == (2, JOINT_DOF)
    assert zero_trace.hand_qpos.shape == (JOINT_DOF,)
    for field in (
        "q_target",
        "hand_qpos",
        "hand_qvel",
        "actuator_force_substeps",
        "object_pos",
        "object_quat_xyzw",
    ):
        np.testing.assert_array_equal(getattr(zero_trace, field), getattr(random_trace, field))

    # On the first changed command, ctrl remains a joint-position target while
    # actuator_force reports MuJoCo's dynamically computed generalized force.
    zero_replay.step(np.zeros(26))
    changed = zero_replay.step(np.zeros(26))
    np.testing.assert_allclose(zero_replay.data.ctrl, changed.q_target)
    assert not np.allclose(changed.actuator_force_substeps[-1], changed.q_target)


def test_source_counter_terminal_schedule_never_consumes_last_slice_frame() -> None:
    schedule = [source_counter_indices(step) for step in range(CONTROL_STEP_COUNT)]
    assert schedule[:4] == [(0, 0), (0, 1), (1, 2), (2, 3)]
    assert schedule[-1] == (CONTROL_STEP_COUNT - 2, CONTROL_STEP_COUNT - 1)
    assert all(
        target != REFERENCE_FRAME_COUNT - 1 and reference != REFERENCE_FRAME_COUNT - 1
        for target, reference in schedule
    )
    with pytest.raises(ValueError):
        source_counter_indices(CONTROL_STEP_COUNT)


def test_short_cpu_replay_is_finite() -> None:
    available, reason = _runtime_ready()
    if not available:
        pytest.skip(reason)
    trajectory = _synthetic_servo_trajectory()
    config, trace = run_reference_replay(
        trajectory, backend="mujoco-cpu", device="cpu", max_steps=5
    )
    assert config.backend == "mujoco-cpu"
    assert trace["target_index"].tolist() == [0, 0, 1, 2, 3]
    assert trace["reference_index"].tolist() == [0, 1, 2, 3, 4]
    assert trace["sim_time"][-1] == pytest.approx(0.025)
    assert trace["actuator_force_substeps"].shape == (5, 2, JOINT_DOF)
    assert all(
        np.all(np.isfinite(value))
        for value in trace.values()
        if np.issubdtype(value.dtype, np.number)
    )


@pytest.mark.mjx_warp
def test_short_mjx_warp_cpu_replay_is_finite() -> None:
    available, reason = _runtime_ready(require_mjx=True)
    if not available:
        pytest.skip(reason)
    trajectory = _synthetic_servo_trajectory()
    replay = MjxWarpReplay(trajectory, device="cpu")
    first = replay.step(np.zeros(26))
    second = replay.step(np.ones(26))
    assert first.target_index == 0 and first.reference_index == 0
    assert second.target_index == 0 and second.reference_index == 1
    assert second.sim_time == pytest.approx(0.01, abs=2e-7)
    for record in (first, second):
        assert np.all(np.isfinite(record.hand_qpos))
        assert np.all(np.isfinite(record.object_pos))
        assert np.all(np.isfinite(record.actuator_force_substeps))
        assert record.actuator_force_substeps.shape == (2, JOINT_DOF)


def test_nominal_native_servo_free_space_prefix_is_stable() -> None:
    available, reason = _runtime_ready()
    if not available:
        pytest.skip(reason)
    trajectory = _synthetic_servo_trajectory()
    config, trace = run_reference_replay(
        trajectory,
        backend="mujoco-cpu",
        device="cpu",
        max_steps=16,
        servo=ServoConfig(hand_contacts_enabled=False),
    )
    assert np.max(np.abs(trace["hand_qvel"])) < 100.0
    assert np.max(np.abs(trace["actuator_force_substeps"][:, :, :6])) < 5000.0
    assert np.max(trace["warning_count"]) == 0
    summary = summarize_trace(trajectory, config, trace, trace_path=Path("unused.npz"))
    assert summary["diagnostic_passed"] is True
    warned_trace = dict(trace)
    warned_trace["warning_count"] = trace["warning_count"].copy()
    warned_trace["warning_count"][0] = 1
    warned = summarize_trace(
        trajectory, config, warned_trace, trace_path=Path("unused.npz")
    )
    assert warned["diagnostic_passed"] is False
    assert warned["stability_gates"]["warning_free"] is False
