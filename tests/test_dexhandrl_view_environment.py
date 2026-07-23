from __future__ import annotations

import os

import numpy as np
import pytest

import sim.dexhandrl.skrl_env as skrl_env_module
from sim.dexhandrl.skrl_env import DexHandRLMJXEnv, DexHandRLMJXEnvConfig
from sim.dexhandrl.view_environment import (
    PassiveDexHandViewer,
    ZeroActionStepper,
    _parser,
    _parse_cli_bool,
    _require_graphical_display,
    _resolve_viewer_options,
    _sleep_duration,
    _wait_for_passive_viewer_shutdown,
)
from tools.trace_dexhandrl_mujoco import PassiveTraceViewer


def test_viewer_parser_accepts_canonical_manorl_style_selectors_and_defaults():
    args = _parser().parse_args([
        "--object", "cube1", "--gesture", "02", "--sequence-index", "0",
        "--use_residual", "false", "--terminal", "true", "--speed", "0.25",
    ])
    options = _resolve_viewer_options(args)
    assert (args.object_name, args.action, args.sequence_index) == ("cube1", "02", 0)
    assert options.use_residual is False
    assert options.terminal_threshold == pytest.approx(0.1)
    assert (options.speed, options.sleep_enabled) == (0.25, True)
    assert args.headless is False
    assert args.runtime_backend == "mujoco-native"


def test_viewer_parser_defaults_are_manorl_aligned():
    options = _resolve_viewer_options(_parser().parse_args([]))
    assert options.use_residual is True
    assert options.terminal_threshold == pytest.approx(0.1)
    assert (options.speed, options.sleep_enabled) == (0.25, True)


def test_viewer_parser_allows_explicit_mjx_warp_selection():
    args = _parser().parse_args(["--runtime-backend", "mjx-warp", "--device", "gpu"])
    assert (args.runtime_backend, args.device) == ("mjx-warp", "gpu")


def test_viewer_parser_preserves_old_command_compatibility():
    args = _parser().parse_args([
        "--object-name", "cube1", "--action", "02", "--sequence-index", "0",
        "--no-use-residual", "--no-headless", "--max-steps", "3", "--no-loop",
        "--no-realtime", "--object-position-terminal-threshold", "0.2",
    ])
    options = _resolve_viewer_options(args)
    assert (args.object_name, args.action, args.sequence_index) == ("cube1", "02", 0)
    assert (args.headless, args.max_steps, args.loop) == (False, 3, False)
    assert (options.use_residual, options.terminal_threshold, options.sleep_enabled) == (False, 0.2, False)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--use_residual", "true", "--no-use-residual"], "use_residual"),
        (["--terminal", "false", "--object-position-terminal-threshold", "0.2"], "--terminal"),
        (["--speed", "0.5", "--realtime"], "--speed"),
    ],
)
def test_viewer_parser_rejects_conflicting_canonical_and_compatibility_forms(argv, message):
    with pytest.raises(ValueError, match=message):
        _resolve_viewer_options(_parser().parse_args(argv))


def test_viewer_terminal_mapping_preserves_dex_early_phase_configuration():
    enabled = _resolve_viewer_options(_parser().parse_args(["--terminal", "true"]))
    disabled = _resolve_viewer_options(_parser().parse_args(["--terminal", "false"]))
    assert enabled.terminal_threshold == pytest.approx(0.1)
    assert disabled.terminal_threshold == pytest.approx(1_000_000.0)
    assert DexHandRLMJXEnvConfig().early_phase_frames == 130


def test_viewer_speed_resolution_and_sleep_target():
    realtime = _resolve_viewer_options(_parser().parse_args(["--realtime"]))
    no_realtime = _resolve_viewer_options(_parser().parse_args(["--no-realtime"]))
    assert (realtime.speed, realtime.sleep_enabled) == (1.0, True)
    assert (no_realtime.speed, no_realtime.sleep_enabled) == (0.25, False)
    assert _sleep_duration(control_dt=0.01, speed=0.25, compute_time=0.01) == pytest.approx(0.03)
    assert _sleep_duration(control_dt=0.01, speed=0.25, compute_time=0.05) == 0.0
    with pytest.raises(ValueError, match="positive"):
        _resolve_viewer_options(_parser().parse_args(["--speed", "0"]))


def test_viewer_explicit_boolean_parser_matches_manorl_contract():
    assert _parse_cli_bool(" TRUE ") is True
    assert _parse_cli_bool("false") is False
    with pytest.raises(Exception, match="expected true or false"):
        _parse_cli_bool("yes")


def test_graphical_viewer_rejects_headless_or_missing_display(monkeypatch):
    with pytest.raises(ValueError, match="--headless"):
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


def test_native_contact_collection_normalizes_empty_buffers_and_unsolved_addresses(monkeypatch):
    captured: dict[str, object] = {}

    def collect(**kwargs):
        captured.update(kwargs)
        return {"active_contact_count": kwargs["active_contacts"]}

    monkeypatch.setattr(skrl_env_module, "collect_contact_summary", collect)
    env = object.__new__(DexHandRLMJXEnv)
    env.cfg = type("Cfg", (), {"runtime_backend": "mujoco-native", "object_name": "cube1"})()
    env._dx = type("Data", (), {"ncon": 0, "efc_force": np.zeros(0)})()
    env.model = type("Model", (), {"nconmax": 8, "opt": type("Opt", (), {"cone": 0})()})()
    # MuJoCo's native enum is PYRAMIDAL=0, ELLIPTIC=1.
    env.mujoco = type("MuJoCo", (), {"mjtCone": type("Cone", (), {"mjCONE_PYRAMIDAL": 0})})()
    env.geom_body_names, env.geom_names = {}, {}
    env.contact_body_names = ["RFH1"]
    assert env._collect_contacts() == {"active_contact_count": 0}
    assert np.asarray(captured["contact_geom"]).shape == (0, 2)
    assert np.asarray(captured["contact_frame"]).shape == (0, 9)
    assert np.asarray(captured["contact_friction"]).shape == (0, 5)
    assert np.asarray(captured["contact_dim"]).shape == (0,)
    assert np.asarray(captured["contact_efc_address"]).shape == (0, 6)
    assert captured["pyramidal"] is True

    unsolved = type(
        "Contact",
        (),
        {"geom1": 0, "geom2": 1, "frame": np.eye(3).ravel(), "friction": np.ones(5), "dim": 3, "efc_address": -1},
    )()
    env._dx = type("Data", (), {"ncon": 1, "contact": [unsolved], "efc_force": np.zeros(0)})()
    assert env._collect_contacts() == {"active_contact_count": 1}
    np.testing.assert_array_equal(captured["contact_efc_address"], [[-1] * 6])


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
