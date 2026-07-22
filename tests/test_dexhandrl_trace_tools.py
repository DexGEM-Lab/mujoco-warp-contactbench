from __future__ import annotations
import importlib, json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sim.dexhandrl.obs_layout import OBS_COMPONENT_SLICES
from sim.dexhandrl.trace_contract import EXACT_IDENTITY, SCHEMA, VECTOR_SPECS, validate_trace
from tools.compare_dexhandrl_traces import compare
from tools.trace_dexhandrl_gym import (_deterministic_action, _filter_exact_trajectory, _patch,
                                      _pre_action_observations, _validated_component_slices)

def _trace(frames=(61, 62), *, backend="mujoco"):
    arrays = {name: np.zeros((len(frames), width), dtype=np.float32) for name, width in VECTOR_SPECS.items()}
    arrays.update({"decision_frame": np.asarray(frames, dtype=np.int64), "state_frame": np.asarray(frames, dtype=np.int64) + 1, "reset": np.asarray([frame == 61 for frame in frames], dtype=np.bool_), "terminated": np.zeros(len(frames), np.bool_), "truncated": np.zeros(len(frames), np.bool_)})
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

@pytest.mark.parametrize("frames", [(62, 63), (61, 61), (61, 63)])
def test_trace_contract_rejects_invalid_decision_schedule(frames):
    metadata, arrays = _trace(frames)
    with pytest.raises(ValueError, match="decision_frame") : validate_trace(metadata, arrays)

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


def test_runtime_patch_restores_original_attribute():
    class Holder: value = "original"
    with _patch(Holder, "value", "replacement"):
        assert Holder.value == "replacement"
    assert Holder.value == "original"


def test_comparator_reports_both_observation_slices_and_control_contract(tmp_path):
    meta, left = _trace(backend="gym"); _, right = _trace()
    right["processed_observation"][1, 62] = 1e-3; right["rms_observation"][1, 62] = 1e-3; right["applied_active_target"][0, 0] = 2e-3
    paths = []
    for name, arrays in (("gym", left), ("mujoco", right)):
        path = tmp_path / f"{name}.npz"; np.savez_compressed(path, metadata=np.asarray(json.dumps(meta if name == "gym" else {**meta, "backend": "mujoco"})), **arrays); paths.append(str(path))
    report = compare(*paths)
    assert report["shared_input"]["processed_observation_slices"]["active_finger_dof_pos"]["first_frame"] == 62
    assert report["shared_input"]["rms_observation_slices"]["active_finger_dof_pos"]["first_frame"] == 62
    assert report["first_divergence"]["shared_input_stage"] == "processed_observation"
    assert report["stage_semantics"]["applied_active_target"] == "a mismatch is a control-contract divergence"
