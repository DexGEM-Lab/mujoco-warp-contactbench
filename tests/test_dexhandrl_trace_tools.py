from __future__ import annotations
import argparse
import importlib, json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sim.dexhandrl.obs_layout import OBS_COMPONENT_SLICES
from sim.dexhandrl.trace_contract import (
    EXACT_IDENTITY,
    FINAL_STATE_FRAME,
    MAX_ACTIONS,
    RESET_DECISION_FRAME,
    SCHEMA,
    VECTOR_SPECS,
    validate_trace,
)
from tools.compare_dexhandrl_traces import compare
import tools.trace_dexhandrl_mujoco as mujoco_trace
from tools.trace_dexhandrl_mujoco import (
    ActionTraceIntervention,
    ControlTraceIntervention,
    ObservationTraceIntervention,
    PassiveTraceViewer,
    _parser,
    _passive_trace_viewer,
    _policy_observation,
    _validate_trace_interventions,
)
from tools.trace_dexhandrl_gym import (_authoritative_frame, _compose_train_config, _deterministic_action,
                                      _close_env_after_trace, _filter_exact_trajectory, _patch,
                                      _is_relative_to, _pre_action_observations,
                                      _preserve_bounded_run_play, _resolve_trace_paths,
                                      _restore_mapped_checkpoint,
                                      _validated_component_slices)

def _trace(frames=(62, 63), *, backend="mujoco"):
    arrays = {name: np.zeros((len(frames), width), dtype=np.float32) for name, width in VECTOR_SPECS.items()}
    arrays.update({"decision_frame": np.asarray(frames, dtype=np.int64), "state_frame": np.asarray(frames, dtype=np.int64) + 1, "reset": np.asarray([frame == RESET_DECISION_FRAME for frame in frames], dtype=np.bool_), "terminated": np.zeros(len(frames), np.bool_), "truncated": np.zeros(len(frames), np.bool_)})
    return {"schema": SCHEMA, "backend": backend, "identity": dict(EXACT_IDENTITY), "post_filters": []}, arrays

def _fresh_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )


def test_passive_trace_viewer_mirrors_state_without_physics_step() -> None:
    class Lock:
        def __init__(self, handle): self.handle = handle
        def __enter__(self):
            assert not self.handle.locked
            self.handle.locked = True
            self.handle.lock_count += 1
        def __exit__(self, *_):
            self.handle.locked = False
            return False
    class Handle:
        def __init__(self):
            self.lock_count = self.sync_count = self.close_count = 0
            self.locked = False
            self.running = True
            self.cam = type("Cam", (), {"lookat": np.zeros(3), "distance": 0, "azimuth": 0, "elevation": 0})()
        def is_running(self): return self.running
        def lock(self): return Lock(self)
        def sync(self):
            assert not self.locked
            self.sync_count += 1
        def close(self): self.close_count += 1
    class MuJoCo:
        def __init__(self): self.forward_count = 0
        def mj_forward(self, _model, data): self.forward_count += 1; data.xpos[:] = [[1, 2, 3]]
    data = type("Data", (), {"qpos": np.zeros(3), "qvel": np.zeros(2), "xpos": np.zeros((1, 3))})()
    handle, mujoco = Handle(), MuJoCo()
    viewer = PassiveTraceViewer(mujoco, object(), data, handle, 0)
    dx = type("DX", (), {"qpos": np.array([1, 2, 3]), "qvel": np.array([4, 5])})()

    assert viewer.sync(dx)
    np.testing.assert_array_equal(data.qpos, dx.qpos)
    np.testing.assert_array_equal(data.qvel, dx.qvel)
    assert mujoco.forward_count == handle.sync_count == handle.lock_count == 1
    assert handle.cam.lookat.tolist() == [1, 2, 3]
    handle.running = False
    assert not viewer.sync(dx)
    viewer.close()
    assert handle.close_count == 1


def test_passive_trace_viewer_headless_and_missing_display(monkeypatch):
    with _passive_trace_viewer(object(), headless=True) as viewer:
        assert viewer is None
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="requires DISPLAY"):
        PassiveTraceViewer.create(object())


def test_gym_trace_import_is_fresh_interpreter_offline_safe():
    result = _fresh_python(
        "import sys; import tools.trace_dexhandrl_gym; "
        "assert 'torch' not in sys.modules; assert 'isaacgym' not in sys.modules"
    )
    assert result.returncode == 0, result.stderr


def test_public_environment_imports_remain_available_lazily():
    result = _fresh_python(
        "import sys; from sim.dexhandrl import DexHandRLMJXEnv, DexHandRLMJXEnvConfig; "
        "from sim.dexhandrl.skrl_env import DexHandRLMJXEnv as DirectEnv; "
        "assert DexHandRLMJXEnv is DirectEnv; assert DexHandRLMJXEnvConfig.__name__ == 'DexHandRLMJXEnvConfig'; "
        "assert 'torch' not in sys.modules"
    )
    assert result.returncode == 0, result.stderr


def test_offline_safe_imports_do_not_load_runtime_modules():
    for module in ("tools.trace_dexhandrl_gym", "tools.trace_dexhandrl_mujoco", "tools.compare_dexhandrl_traces"):
        importlib.import_module(module)
    assert "isaacgym" not in sys.modules and "lance" not in sys.modules


def test_mujoco_trace_parser_exposes_finger_gains_with_production_defaults(tmp_path):
    args = _parser().parse_args(["--output", str(tmp_path / "trace.npz")])
    assert args.finger_kp == 15.0
    assert args.finger_kv == 0.0
    assert (args.base_rot_kp, args.base_rot_kv, args.base_rot_force) == (10000.0, 125.0, 100.0)

    args = _parser().parse_args([
        "--output", str(tmp_path / "trace.npz"), "--finger-kp", "7.5", "--finger-kv", "2.0",
        "--base-rot-kp", "5000", "--base-rot-kv", "500", "--base-rot-force", "200",
    ])
    assert args.finger_kp == 7.5
    assert args.finger_kv == 2.0
    assert (args.base_rot_kp, args.base_rot_kv, args.base_rot_force) == (5000.0, 500.0, 200.0)

@pytest.mark.parametrize("frames", [(61, 62), (62, 64)])
def test_trace_contract_rejects_invalid_decision_schedule(frames):
    metadata, arrays = _trace(frames)
    with pytest.raises(ValueError, match="decision_frame") : validate_trace(metadata, arrays)


def test_trace_contract_rejects_632_actions_and_accepts_full_v3_schedule():
    metadata, arrays = _trace(tuple(RESET_DECISION_FRAME + np.arange(MAX_ACTIONS + 1)))
    with pytest.raises(ValueError, match=f"1..{MAX_ACTIONS}"): validate_trace(metadata, arrays)

    metadata, arrays = _trace(tuple(RESET_DECISION_FRAME + np.arange(MAX_ACTIONS)))
    assert validate_trace(metadata, arrays) == MAX_ACTIONS
    assert arrays["decision_frame"][-1] == FINAL_STATE_FRAME - 1
    assert arrays["state_frame"][-1] == FINAL_STATE_FRAME


def test_gym_authoritative_frame_helper_reads_reset_and_progression():
    class Tensor:
        def __init__(self, value): self.value = value
        def item(self): return self.value

    env = type("Env", (), {"task": type("Task", (), {"task_states": {"current_frame_idx": [Tensor(62)]}})()})()
    assert _authoritative_frame(env) == RESET_DECISION_FRAME
    env.task.task_states["current_frame_idx"][0] = Tensor(63)
    assert _authoritative_frame(env) == RESET_DECISION_FRAME + 1

def test_trace_contract_rejects_wrong_state_dtype_and_filter_metadata():
    metadata, arrays = _trace(); arrays["state_frame"][1] = 62
    with pytest.raises(ValueError, match="state_frame"): validate_trace(metadata, arrays)
    _, arrays = _trace(); arrays["policy_mean"] = arrays["policy_mean"].astype(np.float64)
    with pytest.raises(ValueError, match="float32"): validate_trace(metadata, arrays)
    metadata, arrays = _trace(); metadata["post_filters"] = "absent"
    with pytest.raises(ValueError, match="post_filters"): validate_trace(metadata, arrays)

def test_trace_contract_rejects_nonfinite_values():
    metadata, arrays = _trace(); arrays["policy_mean"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"): validate_trace(metadata, arrays)


def test_trace_contract_validates_optional_live_processed_observation():
    metadata, arrays = _trace()
    arrays["live_processed_observation"] = np.zeros((2, 461), dtype=np.float32)
    assert validate_trace(metadata, arrays) == 2
    arrays["live_processed_observation"] = arrays["live_processed_observation"].astype(np.float64)
    with pytest.raises(ValueError, match="live_processed_observation.*float32"):
        validate_trace(metadata, arrays)
    arrays["live_processed_observation"] = np.zeros((1, 461), dtype=np.float32)
    with pytest.raises(ValueError, match="live_processed_observation.*shape"):
        validate_trace(metadata, arrays)

def test_gym_trace_resolves_caller_output_and_rejects_checkout_writes(tmp_path, monkeypatch):
    caller, gym_root = tmp_path / "caller", tmp_path / "gym"
    caller.mkdir(); gym_root.mkdir()
    assert _is_relative_to(gym_root / "nested" / "trace.npz", gym_root)
    assert not _is_relative_to(tmp_path / "other" / "trace.npz", gym_root)
    monkeypatch.chdir(caller)
    args = argparse.Namespace(output=Path("traces/gym.npz"), gym_root=gym_root)
    resolved_root, output, runtime = _resolve_trace_paths(args)
    assert resolved_root == gym_root.resolve()
    assert output == (caller / "traces/gym.npz").resolve()
    assert runtime == (caller / "traces/gym_runtime").resolve()
    with pytest.raises(ValueError, match="must be outside Gym root"):
        _resolve_trace_paths(argparse.Namespace(output=gym_root / "trace.npz", gym_root=gym_root))


def test_gym_trace_closes_actual_env_once_and_preserves_trace_failure():
    class Env:
        def __init__(self, close_error=None): self.close_error, self.close_calls = close_error, 0
        def close(self):
            self.close_calls += 1
            if self.close_error: raise self.close_error

    complete = Env()
    with _close_env_after_trace(complete):
        pass
    assert complete.close_calls == 1

    failed = Env(RuntimeError("close failure"))
    with pytest.raises(ValueError, match="trace failure"):
        with _close_env_after_trace(failed):
            raise ValueError("trace failure")
    assert failed.close_calls == 1

    close_only = Env(RuntimeError("close failure"))
    with pytest.raises(RuntimeError, match="close failure"):
        with _close_env_after_trace(close_only):
            pass
    assert close_only.close_calls == 1


def test_gym_trace_composes_absolute_checkout_config_without_package_lookup(tmp_path):
    calls = []

    class Initialize:
        def __enter__(self):
            calls.append("enter")
        def __exit__(self, *_):
            calls.append("exit")

    def initialize_config_dir(**kwargs):
        calls.append(("initialize", kwargs))
        return Initialize()

    def compose(**kwargs):
        calls.append(("compose", kwargs))
        return "configured"

    gym_root = tmp_path / "gym"
    checkpoint = tmp_path / "checkpoints" / "cube1_01.pth"
    cfg = _compose_train_config(argparse.Namespace(gym_root=gym_root, checkpoint=checkpoint),
                                initialize_config_dir, compose)
    assert cfg == "configured"
    assert calls[0] == ("initialize", {"config_dir": str((gym_root / "dexhand_env" / "cfg").resolve()), "version_base": None})
    assert calls[1] == "enter" and calls[-1] == "exit"
    composed = calls[2][1]
    assert composed["config_name"] == "config"
    assert "env.numEnvs=1" in composed["overrides"]
    assert "task.mocap.objectType=[cube1]" in composed["overrides"]
    assert "task.mocap.actions=[01]" in composed["overrides"]
    assert "task.pointCloudEncoding.isDynamic=false" in composed["overrides"]
    assert f"train.checkpoint={checkpoint}" in composed["overrides"]
    assert not any(item.startswith("hydra.") for item in composed["overrides"])


def test_gym_uuid_filter_fails_closed_and_selects_exact_row():
    exact = {"row_id": 22, "uuid": EXACT_IDENTITY["uuid"], "object_type": "cube1", "action_id": "01"}
    assert _filter_exact_trajectory({("cube1", "01"): [{**exact, "row_id": 16}, exact]}) == {("cube1", "01"): [exact]}
    with pytest.raises(RuntimeError, match="expected one row 22"):
        _filter_exact_trajectory({("cube1", "01"): [{**exact, "uuid": "wrong"}]})


def test_gym_pre_action_observations_are_copied_before_post_step_mutation():
    class Tensor:
        def __init__(self, value): self.value = np.asarray(value, dtype=np.float32)
        def __getitem__(self, index): return Tensor(self.value[index])
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value

    preprocessed = Tensor([[1.0] * 461])
    rms = Tensor([[2.0] * 461])
    processed_observation, rms_observation = _pre_action_observations(preprocessed, rms)
    preprocessed.value.fill(9.0)  # Simulates env_step overwriting the next observation buffer.
    assert processed_observation.shape == (461,)
    assert np.all(processed_observation == 1.0)
    assert np.all(rms_observation == 2.0)


def test_gym_runtime_component_slices_must_match_policy_contract_exactly():
    runtime = {name: list(bounds) for name, bounds in OBS_COMPONENT_SLICES.items()}
    assert _validated_component_slices(runtime, list(OBS_COMPONENT_SLICES)) == OBS_COMPONENT_SLICES
    runtime["active_finger_dof_pos"] = [62, 77]
    with pytest.raises(RuntimeError, match="differ"):
        _validated_component_slices(runtime, list(OBS_COMPONENT_SLICES))
    with pytest.raises(RuntimeError, match="differ"):
        _validated_component_slices({name: list(bounds) for name, bounds in OBS_COMPONENT_SLICES.items()},
                                    list(reversed(OBS_COMPONENT_SLICES)))


def test_gym_deterministic_action_clamps_and_rescales():
    torch = pytest.importorskip("torch")
    mu = torch.tensor([[-2.0, 0.25, 3.0]])
    action = _deterministic_action(mu, torch.tensor([-4.0, 2.0, 10.0]), torch.tensor([2.0, 6.0, 14.0]), True)
    assert torch.equal(action, torch.tensor([[-4.0, 4.5, 14.0]]))
    assert _deterministic_action(mu, None, None, False) is mu


def test_mapped_checkpoint_restore_uses_player_device_and_player_restore_state_rules():
    calls = []

    class Loader:
        def load(self, path, **kwargs):
            calls.append((path, kwargs))
            return {"model": {"weight": 1, "running_mean_std.running_mean": 2},
                    "running_mean_std": {"mean": 3}, "env_state": {"episode": 4}}

    class Model:
        def load_state_dict(self, state, strict=True): calls.append(("model", state, strict))
        class running_mean_std:
            @staticmethod
            def load_state_dict(state): calls.append(("rms", state))

    class Env:
        @staticmethod
        def set_env_state(state): calls.append(("env", state))

    player = type("Player", (), {"device": "cuda:0", "model": Model(), "normalize_input": True, "env": Env()})()
    _restore_mapped_checkpoint(player, "cuda1_checkpoint.pth", Loader())
    assert calls == [("cuda1_checkpoint.pth", {"map_location": "cuda:0", "weights_only": False}),
                     ("model", {"weight": 1, "running_mean_std.running_mean": 2}, True),
                     ("rms", {"mean": 3}), ("env", {"episode": 4})]


def test_mapped_checkpoint_restore_rejects_malformed_checkpoint_and_handles_old_torch():
    class OldTorch:
        def __init__(self, payload): self.payload, self.calls = payload, []
        def load(self, _path, **kwargs):
            self.calls.append(kwargs)
            if "weights_only" in kwargs: raise TypeError("load() got an unexpected keyword argument 'weights_only'")
            return self.payload

    player = type("Player", (), {"device": "cuda:0", "model": object(), "normalize_input": False, "env": None})()
    old = OldTorch({})
    with pytest.raises(RuntimeError, match="missing required 'model'"):
        _restore_mapped_checkpoint(player, "checkpoint.pth", old)
    assert old.calls == [{"map_location": "cuda:0", "weights_only": False}, {"map_location": "cuda:0"}]
    with pytest.raises(RuntimeError, match="requires a checkpoint path"):
        _restore_mapped_checkpoint(player, None, old)


def test_runtime_patch_restores_original_attribute():
    class Holder: value = "original"
    with _patch(Holder, "value", "replacement"):
        assert Holder.value == "replacement"
    assert Holder.value == "original"


def test_gym_apply_patch_reinstalls_bounded_run_play_and_restores_after_error():
    calls = []

    class FakeRunner:
        def run_play(self, _args):
            calls.append("original")

        def run(self):
            return self.run_play({})  # Gym's patched Runner.run dispatches dynamically.

    original_run_play = FakeRunner.run_play

    def upstream_run_play(self, _args):
        calls.append("upstream-player-run")

    def upstream_apply():
        calls.append("upstream-apply")
        FakeRunner.run_play = upstream_run_play

    class FakeTrain:
        apply_rl_games_patches = staticmethod(upstream_apply)

    original_apply = FakeTrain.apply_rl_games_patches

    def bounded_run_play(self, _args):
        calls.append("bounded-three-step-loop")

    with _patch(FakeRunner, "run_play", bounded_run_play), _preserve_bounded_run_play(FakeTrain, FakeRunner, bounded_run_play):
        FakeTrain.apply_rl_games_patches()
        assert FakeRunner.run_play is bounded_run_play
        FakeRunner().run()
        assert calls == ["upstream-apply", "bounded-three-step-loop"]

    assert FakeTrain.apply_rl_games_patches is original_apply
    assert FakeRunner.run_play is original_run_play

    def failing_upstream_apply():
        FakeRunner.run_play = upstream_run_play
        raise RuntimeError("upstream failure")

    FakeTrain.apply_rl_games_patches = staticmethod(failing_upstream_apply)
    failing_apply = FakeTrain.apply_rl_games_patches
    with pytest.raises(RuntimeError, match="upstream failure"):
        with _patch(FakeRunner, "run_play", bounded_run_play), _preserve_bounded_run_play(FakeTrain, FakeRunner, bounded_run_play):
            FakeTrain.apply_rl_games_patches()
    assert FakeTrain.apply_rl_games_patches is failing_apply
    assert FakeRunner.run_play is original_run_play


def _control_trace(tmp_path, *, metadata=None, arrays=None):
    source_metadata, source_arrays = _trace(backend="gym")
    if metadata is not None:
        source_metadata = metadata
    if arrays is not None:
        source_arrays = arrays
    path = tmp_path / "gym_control.npz"
    np.savez_compressed(path, metadata=np.asarray(json.dumps(source_metadata)), **source_arrays)
    return path


def test_action_trace_parser_and_main_wiring(tmp_path, monkeypatch):
    args = _parser().parse_args(["--output", str(tmp_path / "trace.npz"), "--action-trace", "gym.npz"])
    assert args.action_trace == Path("gym.npz")

    captured = {}
    monkeypatch.setattr(mujoco_trace, "run", lambda *args: captured.setdefault("args", args) or Path("unused.npz"))
    monkeypatch.setattr(sys, "argv", ["trace_dexhandrl_mujoco.py", "--output", str(tmp_path / "trace.npz"),
                                       "--action-trace", "gym.npz"])
    assert mujoco_trace.main() == 0
    assert captured["args"][-1] == Path("gym.npz")


def test_observation_trace_parser_accepts_repeated_components_and_phase_end(tmp_path):
    args = _parser().parse_args([
        "--output", str(tmp_path / "trace.npz"), "--observation-trace", "gym.npz",
        "--replace-observation-component", "contact_force_magnitude",
        "--replace-observation-component", "contact_force_directions",
        "--observation-intervention-end-frame", "440",
    ])
    assert args.observation_trace == Path("gym.npz")
    assert args.replace_observation_component == ["contact_force_magnitude", "contact_force_directions"]
    assert args.observation_intervention_end_frame == 440


@pytest.mark.parametrize(
    ("control_trace", "action_trace", "observation_trace", "components", "message"),
    [
        (None, None, Path("gym.npz"), (), "required together"),
        (None, None, None, ("contact_force_magnitude",), "required together"),
        (Path("control.npz"), None, Path("gym.npz"), ("contact_force_magnitude",), "mutually exclusive"),
        (Path("control.npz"), Path("action.npz"), None, (), "mutually exclusive"),
        (None, Path("action.npz"), Path("gym.npz"), ("contact_force_magnitude",), "mutually exclusive"),
        (None, None, Path("gym.npz"), ("contact_force_mag",), "exact OBS_COMPONENT_SLICES"),
        (None, None, None, (), "requires --observation-trace"),
        (None, None, Path("gym.npz"), ("contact_force_magnitude",), f"{RESET_DECISION_FRAME}..{FINAL_STATE_FRAME}"),
    ],
)
def test_trace_intervention_cli_pairing_fails_closed(control_trace, action_trace, observation_trace, components, message):
    end = RESET_DECISION_FRAME - 1 if message == "requires --observation-trace" else (
        FINAL_STATE_FRAME + 1 if message == f"{RESET_DECISION_FRAME}..{FINAL_STATE_FRAME}" else None
    )
    with pytest.raises(ValueError, match=message):
        _validate_trace_interventions(control_trace, action_trace, observation_trace, components, end)


def test_observation_trace_cli_wires_phase_end_to_run(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(mujoco_trace, "run", lambda *args: captured.setdefault("args", args) or Path("unused.npz"))
    monkeypatch.setattr(sys, "argv", ["trace_dexhandrl_mujoco.py", "--output", str(tmp_path / "trace.npz"),
                                       "--observation-trace", "gym.npz",
                                       "--replace-observation-component", "contact_force_magnitude",
                                       "--observation-intervention-end-frame", "440"])
    assert mujoco_trace.main() == 0
    assert captured["args"][-2] == 440
    assert captured["args"][-1] is None


def test_observation_trace_rejects_bad_source_component_and_frame(tmp_path):
    metadata, arrays = _trace(backend="mujoco")
    with pytest.raises(ValueError, match="Gym trace"):
        ObservationTraceIntervention.load(_control_trace(tmp_path, metadata=metadata), 1, ("contact_force_magnitude",))
    metadata, arrays = _trace(backend="gym")
    arrays["processed_observation"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ObservationTraceIntervention.load(_control_trace(tmp_path, metadata=metadata, arrays=arrays), 1, ("contact_force_magnitude",))
    with pytest.raises(ValueError, match="exact OBS_COMPONENT_SLICES"):
        ObservationTraceIntervention.load(_control_trace(tmp_path), 1, ("contact_force_mag",))
    path = _control_trace(tmp_path)
    with pytest.raises(ValueError, match="fewer than requested"):
        ObservationTraceIntervention.load(path, 3, ("contact_force_magnitude",))
    intervention = ObservationTraceIntervention.load(path, 1, ("contact_force_magnitude",))
    with pytest.raises(RuntimeError, match="decision frame"):
        intervention.apply(np.zeros(461, dtype=np.float32), RESET_DECISION_FRAME + 1)


def test_observation_trace_replaces_exact_slices_and_preserves_normal_input(tmp_path):
    path = _control_trace(tmp_path)
    intervention = ObservationTraceIntervention.load(
        path, 1, ("contact_force_magnitude", "contact_force_directions", "object_contact_force_magnitude")
    )
    source = intervention.source_observations[0]
    source[:] = np.arange(461, dtype=np.float32) + 1000
    live = np.arange(461, dtype=np.float32)
    hybrid = _policy_observation(live, RESET_DECISION_FRAME, intervention)
    assert intervention._row == 1
    expected = live.copy()
    for name in intervention.components:
        start, end = OBS_COMPONENT_SLICES[name]
        expected[start:end] = source[start:end]
    np.testing.assert_array_equal(hybrid, expected)
    untouched = np.ones(461, dtype=bool)
    for name in intervention.components:
        start, end = OBS_COMPONENT_SLICES[name]
        untouched[start:end] = False
    np.testing.assert_array_equal(hybrid[untouched], live[untouched])
    assert _policy_observation(live, RESET_DECISION_FRAME, None) is live


def test_observation_trace_phase_end_is_inclusive_and_consumes_pass_through_rows(tmp_path):
    metadata, arrays = _trace((RESET_DECISION_FRAME, RESET_DECISION_FRAME + 1), backend="gym")
    arrays["processed_observation"][0] = 10
    arrays["processed_observation"][1] = 20
    intervention = ObservationTraceIntervention.load(
        _control_trace(tmp_path, metadata=metadata, arrays=arrays), 2,
        ("contact_force_magnitude",), RESET_DECISION_FRAME,
    )
    live_first = np.zeros(461, dtype=np.float32)
    replaced = intervention.apply(live_first, RESET_DECISION_FRAME)
    start, end = OBS_COMPONENT_SLICES["contact_force_magnitude"]
    np.testing.assert_array_equal(replaced[start:end], np.full(end - start, 10, dtype=np.float32))
    live_second = np.full(461, 2, dtype=np.float32)
    passed_through = intervention.apply(live_second, RESET_DECISION_FRAME + 1)
    assert intervention._row == 2
    assert passed_through is not live_second
    np.testing.assert_array_equal(passed_through, live_second)
    assert intervention.metadata_record()["intervention_end_frame"] == RESET_DECISION_FRAME
    assert "consumed at every decision" in intervention.metadata_record()["switch_semantics"]


def test_observation_trace_metadata_states_policy_only_import(tmp_path):
    record = ObservationTraceIntervention.load(
        _control_trace(tmp_path), 1, ("contact_force_magnitude",)
    ).metadata_record()
    assert len(record["source_sha256"]) == 64
    assert record["source_backend"] == "gym"
    assert record["components"] == ["contact_force_magnitude"]
    assert "Only named Gym processed_observation slices" in record["semantics"]
    assert "no target, action" in record["semantics"]


def test_action_trace_rejects_non_gym_identity_nonfinite_and_short_sources(tmp_path):
    metadata, arrays = _trace(backend="mujoco")
    with pytest.raises(ValueError, match="Gym trace"):
        ActionTraceIntervention.load(_control_trace(tmp_path, metadata=metadata), 1)
    metadata, arrays = _trace(backend="gym"); metadata["identity"]["row_id"] = 9
    with pytest.raises(ValueError, match="identity"):
        ActionTraceIntervention.load(_control_trace(tmp_path, metadata=metadata), 1)
    metadata, arrays = _trace(backend="gym"); arrays["final_action"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ActionTraceIntervention.load(_control_trace(tmp_path, metadata=metadata, arrays=arrays), 1)
    with pytest.raises(ValueError, match="fewer than requested"):
        ActionTraceIntervention.load(_control_trace(tmp_path), 3)


def test_action_trace_returns_exact_rows_and_fails_closed_on_frame_or_exhaustion(tmp_path):
    path = _control_trace(tmp_path)
    intervention = ActionTraceIntervention.load(path, 1)
    expected = np.arange(22, dtype=np.float32) - 0.5
    intervention.final_actions[0] = expected
    np.testing.assert_array_equal(intervention.action(RESET_DECISION_FRAME), expected)
    assert intervention._row == 1
    np.testing.assert_array_equal(intervention.action(RESET_DECISION_FRAME + 1), np.zeros(22, dtype=np.float32))
    assert intervention._row == 2
    with pytest.raises(RuntimeError, match="exhausted"):
        intervention.action(RESET_DECISION_FRAME + 2)

    misaligned = ActionTraceIntervention.load(path, 1)
    with pytest.raises(RuntimeError, match="decision frame"):
        misaligned.action(RESET_DECISION_FRAME + 1)


def test_action_trace_metadata_and_source_keep_local_target_path(tmp_path):
    producer = Path(__file__).resolve().parents[1] / "tools" / "trace_dexhandrl_mujoco.py"
    source = producer.read_text(encoding="utf-8")
    assert 'patch = intervention.patch(env) if intervention is not None else nullcontext()' in source
    assert 'imported_action = action_intervention.action(decision_frame)' in source
    assert 'action = imported_action if imported_action is not None else local_action' in source
    assert 'reference_active_target=(source["reference_active_target"] if source is not None else' in source
    record = ActionTraceIntervention.load(_control_trace(tmp_path), 1).metadata_record()
    assert len(record["source_sha256"]) == 64
    assert record["source_backend"] == "gym"
    assert "Only aligned Gym final_action" in record["semantics"]
    assert "residual/control targets" in record["semantics"]


def test_control_trace_rejects_non_gym_identity_frame_and_nonfinite_sources(tmp_path):
    metadata, arrays = _trace(backend="mujoco")
    with pytest.raises(ValueError, match="Gym trace"):
        ControlTraceIntervention.load(_control_trace(tmp_path, metadata=metadata), 1)
    metadata, arrays = _trace(backend="gym"); metadata["identity"]["row_id"] = 9
    with pytest.raises(ValueError, match="identity"):
        ControlTraceIntervention.load(_control_trace(tmp_path, metadata=metadata), 1)
    metadata, arrays = _trace(backend="gym"); arrays["policy_mean"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ControlTraceIntervention.load(_control_trace(tmp_path, metadata=metadata, arrays=arrays), 1)
    intervention = ControlTraceIntervention.load(_control_trace(tmp_path), 1)
    with pytest.raises(RuntimeError, match="decision frame"):
        intervention.current(RESET_DECISION_FRAME + 1)


def test_control_trace_updates_exact_state_and_returns_exact_source_target(tmp_path):
    path = _control_trace(tmp_path)
    intervention = ControlTraceIntervention.load(path, 1)
    for name, value in (("base_offset_before", [0.1, 0.2, 0.3]), ("base_offset_after", [0.4, 0.5, 0.6]),
                        ("joint_offset_before", np.arange(16)), ("joint_offset_after", np.arange(16) + 1),
                        ("applied_active_target", np.arange(22)), ("reference_active_target", np.arange(22) + 2),
                        ("corrected_active_target", np.arange(22) + 3), ("full_target", np.arange(26) + 4)):
        intervention.arrays[name][0] = np.asarray(value, dtype=np.float32)
    env = type("Env", (), {})()
    env.base_position_offset = intervention.arrays["base_offset_before"][0].copy()
    env.joint_offset = intervention.arrays["joint_offset_before"][0].copy()
    target = intervention.apply(env, RESET_DECISION_FRAME)
    np.testing.assert_array_equal(target, intervention.arrays["full_target"][0])
    np.testing.assert_array_equal(env.base_position_offset, intervention.arrays["base_offset_after"][0])
    np.testing.assert_array_equal(env.joint_offset, intervention.arrays["joint_offset_after"][0])
    np.testing.assert_array_equal(env.active_prev_targets, intervention.arrays["applied_active_target"][0])
    np.testing.assert_array_equal(env._last_action_trace["full_target"], target)


def test_control_trace_rejects_offset_mismatch_and_normal_patch_is_unchanged(tmp_path):
    intervention = ControlTraceIntervention.load(_control_trace(tmp_path), 1)
    env = type("Env", (), {})()
    env.base_position_offset = np.ones(3, dtype=np.float32)
    env.joint_offset = np.zeros(16, dtype=np.float32)
    with pytest.raises(RuntimeError, match="base_position_offset mismatch"):
        intervention.apply(env, RESET_DECISION_FRAME)

    def normal_action(*_): return np.full(26, 7, dtype=np.float32)
    env._apply_residual_action = normal_action
    assert env._apply_residual_action(None, None, None)[0] == 7
    with intervention.patch(env):
        env.base_position_offset = np.zeros(3, dtype=np.float32)
        env.joint_offset = np.zeros(16, dtype=np.float32)
        assert env._apply_residual_action(None, RESET_DECISION_FRAME, None).shape == (26,)
    assert env._apply_residual_action is normal_action


def test_control_trace_intervention_metadata_states_imported_policy_semantics(tmp_path):
    record = ControlTraceIntervention.load(_control_trace(tmp_path), 1).metadata_record()
    assert len(record["source_sha256"]) == 64
    assert record["source_backend"] == "gym"
    assert "imported" in record["semantics"] and "MuJoCo" in record["semantics"]


def test_mujoco_trace_producer_declares_and_records_post_filters() -> None:
    producer = Path(__file__).resolve().parents[1] / "tools" / "trace_dexhandrl_mujoco.py"
    source = producer.read_text(encoding="utf-8")
    assert '"post_filters": ["velocity_clamp", "position_clamp"]' in source
    assert "class ControlTraceIntervention" in source
    assert '"applied_active_target": source["applied_active_target"].copy()' in source
    assert '"finger_pd": {"kp": cfg.finger_kp, "kv": cfg.finger_kv, "force": cfg.finger_force}' in source
    assert '"base_rotation_pd": {"kp": cfg.base_rot_kp, "kv": cfg.base_rot_kv, "force": cfg.base_rot_force}' in source


def test_comparator_reports_both_observation_slices_and_control_contract(tmp_path):
    meta, left = _trace(backend="gym"); _, right = _trace()
    right["processed_observation"][1, 62] = 1e-3; right["rms_observation"][1, 62] = 1e-3; right["applied_active_target"][0, 0] = 2e-3
    paths = []
    for name, arrays in (("gym", left), ("mujoco", right)):
        path = tmp_path / f"{name}.npz"; np.savez_compressed(path, metadata=np.asarray(json.dumps(meta if name == "gym" else {**meta, "backend": "mujoco"})), **arrays); paths.append(str(path))
    report = compare(*paths)
    assert report["shared_input"]["processed_observation_slices"]["active_finger_dof_pos"]["first_frame"] == 63
    assert report["shared_input"]["rms_observation_slices"]["active_finger_dof_pos"]["first_frame"] == 63
    assert report["first_divergence"]["shared_input_stage"] == "processed_observation"
    assert report["stage_semantics"]["applied_active_target"] == "a mismatch is a control-contract divergence"
