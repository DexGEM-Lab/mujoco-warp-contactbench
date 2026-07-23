from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.compare_dexhandrl_backends import _contact_count, _parser, compare_rollouts, metric, write_evidence


class FakeEnv:
    def __init__(
        self, backend: str, *, identity: str = "cube1", applied_delta: float = 0.0,
        reference_delta: float = 0.0, nan: bool = False, start_frame: int = 61,
        value_dtype: np.dtype = np.float32,
    ):
        self.cfg = SimpleNamespace(
            object_name=identity, action="02", sequence_index=0, use_residual=False,
            runtime_backend=backend, device="cpu", settle_frames=1, object_position_terminal_threshold=1e9,
            physics=SimpleNamespace(record=lambda: {"control_dt": 0.01, "substeps": 10}),
        )
        self.trajectory = SimpleNamespace(uuid="trajectory-0")
        self.ids = SimpleNamespace(object_free_qpos_addr=2)
        self.num_actions, self.num_observations = 22, 461
        self.start_frame = start_frame
        self._current_frame = start_frame
        self.value_dtype = value_dtype
        contact_state = {"ncon": np.asarray(1)} if backend == "mujoco-native" else {
            "_impl": SimpleNamespace(nacon=np.asarray([[1]]))
        }
        self._dx = SimpleNamespace(
            qpos=np.zeros(12, dtype=np.float32), qvel=np.zeros(11, dtype=np.float32),
            ctrl=np.zeros(26, dtype=np.float32), **contact_state,
        )
        self._dx.qpos[5] = 1.0
        self.applied_delta, self.reference_delta, self.nan, self.closed = applied_delta, reference_delta, nan, False
        self._last_action_trace = None

    def reset(self):
        self._current_frame = self.start_frame + 1
        return np.zeros(461, dtype=self.value_dtype), {}

    def step(self, action):
        assert action.shape == (22,)
        frame = self._current_frame
        target = np.full(22, frame, dtype=self.value_dtype)
        reference = target.copy(); reference[0] += self.reference_delta
        applied = target.copy(); applied[0] += self.applied_delta
        self._last_action_trace = {
            "decision_frame": frame, "reference_active_target": reference,
            "corrected_active_target": target, "applied_active_target": applied,
            "full_target": np.full(26, frame, dtype=self.value_dtype),
        }
        self._dx.ctrl[:] = self._last_action_trace["full_target"]
        self._current_frame += 1
        self._dx.qpos[0] = np.nan if self.nan else self._current_frame
        self._dx.qpos[5] = 1.0
        obs = np.full(461, self._current_frame, dtype=self.value_dtype)
        return obs, float(frame), False, False, {"full_targets": self._last_action_trace["full_target"]}

    def close(self):
        self.closed = True


def test_tool_import_and_defaults_are_offline_safe():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import tools.compare_dexhandrl_backends; assert 'lance' not in sys.modules; assert 'jax' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    args = _parser().parse_args([])
    assert (args.object_name, args.action, args.sequence_index, args.steps, args.mjx_device) == ("cube1", "02", 0, 50, "cpu")


def test_contact_count_reads_native_and_mjx_runtime_layouts_fail_closed():
    native = SimpleNamespace(
        cfg=SimpleNamespace(runtime_backend="mujoco-native"),
        _dx=SimpleNamespace(ncon=np.asarray(3)),
    )
    mjx = SimpleNamespace(
        cfg=SimpleNamespace(runtime_backend="mjx-warp"),
        _dx=SimpleNamespace(_impl=SimpleNamespace(nacon=np.asarray([[7]]))),
    )
    assert _contact_count(native) == 3
    assert _contact_count(mjx) == 7

    with pytest.raises(RuntimeError, match="does not expose"):
        _contact_count(SimpleNamespace(cfg=SimpleNamespace(runtime_backend="mjx-warp"), _dx=SimpleNamespace()))


def test_metric_reports_exactness_dtype_asymmetry_and_first_threshold_frame():
    left = np.zeros((3, 2), dtype=np.float32)
    right = left.copy(); right[1, 0] = 2e-4
    result = metric(left, right, np.array([62, 63, 64]))
    assert not result["exact"]
    assert result["native_dtype"] == result["mjx_dtype"] == "float32"
    assert result["dtype_equal"]
    assert result["first_nonzero"] == 63
    assert result["first_exceed"]["0.0001"] == 63
    assert result["first_exceed"]["0.001"] is None

    dtype_only = metric(np.array([[1]], dtype=np.float32), np.array([[1]], dtype=np.float64), np.array([62]))
    assert dtype_only["exact"]  # np.array_equal is numeric, not a dtype identity check.
    assert not dtype_only["dtype_equal"]
    assert (dtype_only["native_dtype"], dtype_only["mjx_dtype"]) == ("float32", "float64")


def test_compare_rejects_identity_reference_and_applied_target_invariants_and_closes():
    native, mjx = FakeEnv("mujoco-native"), FakeEnv("mjx-warp", identity="other")
    with pytest.raises(ValueError, match="identity mismatch"):
        compare_rollouts(native, mjx, steps=2)
    assert native.closed and mjx.closed

    native, mjx = FakeEnv("mujoco-native"), FakeEnv("mjx-warp", reference_delta=2e-4)
    with pytest.raises(ValueError, match="reference-active target must be exact"):
        compare_rollouts(native, mjx, steps=2)
    assert native.closed and mjx.closed

    native, mjx = FakeEnv("mujoco-native"), FakeEnv("mjx-warp", applied_delta=2e-4)
    with pytest.raises(ValueError, match="applied-active"):
        compare_rollouts(native, mjx, steps=2)
    assert native.closed and mjx.closed


def test_compare_rejects_physics_and_frame_contract_mismatches():
    native, mjx = FakeEnv("mujoco-native"), FakeEnv("mjx-warp")
    mjx.cfg.physics = SimpleNamespace(record=lambda: {"control_dt": 0.02, "substeps": 10})
    with pytest.raises(ValueError, match="physics contract mismatch"):
        compare_rollouts(native, mjx, steps=2)
    assert native.closed and mjx.closed

    native, mjx = FakeEnv("mujoco-native"), FakeEnv("mjx-warp", start_frame=62)
    with pytest.raises(ValueError, match="frame contract mismatch"):
        compare_rollouts(native, mjx, steps=2)
    assert native.closed and mjx.closed


def test_compare_rejects_nonfinite_arrays():
    native, mjx = FakeEnv("mujoco-native"), FakeEnv("mjx-warp", nan=True)
    with pytest.raises(ValueError, match="nonfinite"):
        compare_rollouts(native, mjx, steps=2)
    assert native.closed and mjx.closed


def test_each_backend_rejects_a_common_mode_frame_schedule_failure():
    class BadResetEnv(FakeEnv):
        def reset(self):
            self._current_frame = self.start_frame  # Both backends share the same bug.
            return np.zeros(461, dtype=self.value_dtype), {}

    native, mjx = BadResetEnv("mujoco-native"), BadResetEnv("mjx-warp")
    with pytest.raises(ValueError, match="reset frame"):
        compare_rollouts(native, mjx, steps=2)
    assert native.closed and mjx.closed


def test_evidence_reports_state_partitions_inputs_dtypes_and_provenance(tmp_path):
    report, arrays = compare_rollouts(
        FakeEnv("mujoco-native", value_dtype=np.float32),
        FakeEnv("mjx-warp", value_dtype=np.float64),
        steps=2,
    )
    output = tmp_path / "evidence.json"
    npz_path, digest = write_evidence(output, report, arrays)
    payload = json.loads(output.read_text())
    assert payload["schema"] == "dexhandrl.native_mjx_backend_evidence.v2"
    assert payload["artifact"]["sha256"] == digest
    assert "native_qpos" in np.load(npz_path).files
    assert "mjx_applied_active_target" in np.load(npz_path).files
    state = payload["comparison"]["qpos"]
    assert set(state) == {"all_rows", "reset_row", "post_step_rows"}
    assert state["reset_row"]["native_dtype"] == "float32"
    assert payload["comparison"]["reference_active_target"]["mjx_dtype"] == "float64"
    assert not payload["comparison"]["reference_active_target"]["dtype_equal"]
    assert payload["preregistered_target_exactness"]["reference_active_target"]["observed_exact"]
    assert "falsified" in payload["preregistered_target_exactness"]["corrected_active_target"]["preregistered_exactness"]
    assert payload["inputs"] == {
        "steps": 2, "actions": "zero 22D", "residual": False,
        "object_position_terminal_threshold": 1e9, "settle_frames": 1, "substeps": 10,
    }
    assert payload["provenance"]["git_head"]
    assert {"python", "numpy", "mujoco", "jax", "mjx"} <= payload["provenance"].keys()
    assert "full env.step wall time" in payload["timing"]["native"]["measurement"]
    assert "settle_frames * substeps" in payload["reset_note"]
    assert "current_full" in payload["target_routing_note"]
    assert "near-trivial" in payload["backend_authority"]
