from __future__ import annotations
import importlib
import json
import numpy as np
import pytest
from sim.dexhandrl.trace_contract import EXACT_IDENTITY, SCHEMA, VECTOR_SPECS, validate_trace
from tools.compare_dexhandrl_traces import compare

def _trace(frames=(61, 62)):
    arrays = {name: np.zeros((len(frames), width), dtype=np.float32) for name, width in VECTOR_SPECS.items()}
    arrays.update({"lance_frame": np.asarray(frames, dtype=np.int64), "reset": np.asarray([True, False]), "terminated": np.zeros(2, bool), "truncated": np.zeros(2, bool)})
    return {"schema": SCHEMA, "identity": dict(EXACT_IDENTITY)}, arrays

def test_offline_safe_imports_do_not_load_runtime_modules():
    for module in ("tools.trace_dexhandrl_gym", "tools.trace_dexhandrl_mujoco", "tools.compare_dexhandrl_traces"):
        importlib.import_module(module)
    import sys
    assert "isaacgym" not in sys.modules and "lance" not in sys.modules

def test_trace_contract_rejects_nonexact_identity():
    metadata, arrays = _trace(); metadata["identity"]["row_id"] = 21
    with pytest.raises(ValueError, match="exact identity mismatch"):
        validate_trace(metadata, arrays)

def test_trace_contract_rejects_nonfinite_values():
    metadata, arrays = _trace(); arrays["policy_mean"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_trace(metadata, arrays)

def test_comparator_reports_slice_and_first_divergence(tmp_path):
    meta, left = _trace(); _, right = _trace(); right["processed_observation"][1, 62] = 1e-3; right["filtered_active_target"][0, 0] = 2e-3
    paths = []
    for name, arrays in (("gym", left), ("mujoco", right)):
        path = tmp_path / f"{name}.npz"; np.savez_compressed(path, metadata=np.asarray(json.dumps(meta)), **arrays); paths.append(str(path))
    report = compare(*paths)
    assert report["shared_input"]["observation_slices"]["active_finger_dof_pos"]["first_frame"] == 62
    assert report["first_divergence"]["shared_input_stage"] == "processed_observation"
    assert report["stage_semantics"]["filtered_active_target"]["mujoco"] == "unfiltered; post_filters_absent"
