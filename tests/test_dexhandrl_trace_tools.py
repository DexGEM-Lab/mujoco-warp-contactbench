from __future__ import annotations
import importlib, json
import numpy as np
import pytest
from sim.dexhandrl.trace_contract import EXACT_IDENTITY, SCHEMA, VECTOR_SPECS, validate_trace
from tools.compare_dexhandrl_traces import compare

def _trace(frames=(61, 62), *, backend="mujoco"):
    arrays = {name: np.zeros((len(frames), width), dtype=np.float32) for name, width in VECTOR_SPECS.items()}
    arrays.update({"decision_frame": np.asarray(frames, dtype=np.int64), "state_frame": np.asarray(frames, dtype=np.int64) + 1, "reset": np.asarray([frame == 61 for frame in frames], dtype=np.bool_), "terminated": np.zeros(len(frames), np.bool_), "truncated": np.zeros(len(frames), np.bool_)})
    return {"schema": SCHEMA, "backend": backend, "identity": dict(EXACT_IDENTITY), "post_filters": []}, arrays

def test_offline_safe_imports_do_not_load_runtime_modules():
    for module in ("tools.trace_dexhandrl_gym", "tools.trace_dexhandrl_mujoco", "tools.compare_dexhandrl_traces"): importlib.import_module(module)
    import sys; assert "isaacgym" not in sys.modules and "lance" not in sys.modules

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
