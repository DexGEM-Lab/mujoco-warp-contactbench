from __future__ import annotations

import os

import numpy as np
import pytest

from sim.dexhandrl.skrl_env import DexHandRLMJXEnv
from sim.dexhandrl.view_environment import (
    PassiveDexHandViewer,
    ZeroActionStepper,
    _parser,
    _require_graphical_display,
    _wait_for_passive_viewer_shutdown,
)
from tools.trace_dexhandrl_mujoco import PassiveTraceViewer


def test_viewer_parser_supports_native_zero_action_command(tmp_path):
    args = _parser().parse_args([
        "--object-name", "cube1", "--action", "02", "--sequence-index", "0",
        "--device", "cpu", "--no-use-residual", "--no-headless", "--max-steps", "3",
        "--no-loop", "--no-realtime", "--object-position-terminal-threshold", "0.2",
    ])
    assert (args.object_name, args.action, args.sequence_index, args.use_residual) == ("cube1", "02", 0, False)
    assert (args.headless, args.max_steps, args.loop, args.realtime) == (False, 3, False, False)


def test_graphical_viewer_rejects_headless_or_missing_display(monkeypatch):
    with pytest.raises(ValueError, match="--no-headless"):
        _require_graphical_display(headless=True)
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="requires DISPLAY"):
        _require_graphical_display(headless=False)


def test_zero_action_stepper_steps_production_shaped_environment_and_resets_when_looping():
    class Environment:
        num_actions = 22
        def __init__(self): self.resets, self.actions = 0, []
        def reset(self): self.resets += 1; return np.zeros(1), {}
        def step(self, action):
            self.actions.append(action.copy())
            return np.zeros(1), 0.0, False, True, {}

    env = Environment()
    stepper = ZeroActionStepper(env, loop=True)
    assert stepper.step() is not None
    assert env.resets == 2 and len(env.actions) == 1
    assert env.actions[0].shape == (22,) and not env.actions[0].any()


def test_environment_public_mirror_copies_all_dynamic_viewer_fields():
    state = type("State", (), {
        "qpos": np.array([1.0, 2.0]), "qvel": np.array([3.0]),
        "ctrl": np.array([4.0, 5.0]), "time": np.array(0.25),
    })()
    env = object.__new__(DexHandRLMJXEnv)
    env._dx = state
    data = type("Data", (), {
        "qpos": np.zeros(2), "qvel": np.zeros(1), "ctrl": np.zeros(2), "time": 0.0,
    })()
    env.copy_state_to_mjdata(data)
    np.testing.assert_array_equal(data.qpos, state.qpos)
    np.testing.assert_array_equal(data.qvel, state.qvel)
    np.testing.assert_array_equal(data.ctrl, state.ctrl)
    assert data.time == 0.25


def test_passive_viewer_close_returns_immediately_without_mujoco_weakref():
    class Handle:
        def __init__(self): self.close_count = 0
        def close(self): self.close_count += 1

    viewer = PassiveDexHandViewer(object(), object(), object(), Handle(), 0)
    viewer.close()
    viewer.close()
    assert viewer.handle.close_count == 1


def test_passive_viewer_waits_for_mujoco_thread_weakref_to_clear():
    class Handle:
        def __init__(self):
            self.close_count = 0
            self.running = object()
            self._sim = lambda: self.running
        def close(self): self.close_count += 1

    handle, elapsed = Handle(), [0.0]
    def clock(): return elapsed[0]
    def sleeper(delay):
        elapsed[0] += delay
        handle.running = None

    _wait_for_passive_viewer_shutdown(handle, timeout_s=1.0, clock=clock, sleeper=sleeper)
    assert elapsed[0] == pytest.approx(0.01)


def test_passive_viewer_shutdown_timeout_is_explicit():
    handle, elapsed = type("Handle", (), {})(), [0.0]
    handle._sim = lambda: object()
    def clock(): return elapsed[0]
    def sleeper(delay): elapsed[0] += delay

    with pytest.raises(RuntimeError, match="did not stop within 0.03s"):
        _wait_for_passive_viewer_shutdown(handle, timeout_s=0.03, clock=clock, sleeper=sleeper)


def test_trace_viewer_alias_inherits_passive_viewer_shutdown():
    assert PassiveTraceViewer is PassiveDexHandViewer


def test_passive_viewer_mirrors_without_step_calls():
    class Lock:
        def __enter__(self): return self
        def __exit__(self, *_): return False
    class Handle:
        cam = type("Camera", (), {"lookat": np.zeros(3), "distance": 0.0, "azimuth": 0.0, "elevation": 0.0})()
        def is_running(self): return True
        def lock(self): return Lock()
        def sync(self): self.synced = getattr(self, "synced", 0) + 1
        def close(self): pass
    class MuJoCo:
        def mj_forward(self, _model, data): data.xpos[:] = [[7, 8, 9]]
    data = type("Data", (), {"qpos": np.zeros(1), "qvel": np.zeros(1), "ctrl": np.zeros(1), "time": 0.0, "xpos": np.zeros((1, 3))})()
    source = type("Source", (), {"qpos": np.array([1.0]), "qvel": np.array([2.0]), "ctrl": np.array([3.0]), "time": 0.1})()
    handle = Handle()
    assert PassiveDexHandViewer(MuJoCo(), object(), data, handle, 0).sync(source)
    assert handle.synced == 1
    np.testing.assert_array_equal(data.ctrl, [3.0])
