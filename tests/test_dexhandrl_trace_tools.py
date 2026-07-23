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


def test_mujoco_trace_producer_declares_and_records_post_filters() -> None:
    producer = Path(__file__).resolve().parents[1] / "tools" / "trace_dexhandrl_mujoco.py"
    source = producer.read_text(encoding="utf-8")
    assert 'applied = np.asarray(trace["applied_active_target"], dtype=np.float32)' in source
    assert 'applied_active_target=applied' in source
    assert '"post_filters": ["velocity_clamp", "position_clamp"]' in source


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
