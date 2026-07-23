from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.compare_dexhandrl_backends import _parser, compare_rollouts, metric, write_evidence


class FakeEnv:
    def __init__(self, backend: str, *, identity: str = "cube1", applied_delta: float = 0.0, nan: bool = False, start_frame: int = 62):
        self.cfg = SimpleNamespace(
            object_name=identity, action="02", sequence_index=0, use_residual=False,
            runtime_backend=backend, physics=SimpleNamespace(record=lambda: {"control_dt": 0.01, "substeps": 10}),
        )
        self.trajectory = SimpleNamespace(uuid="trajectory-0")
        self.ids = SimpleNamespace(object_free_qpos_addr=2)
        self.num_actions, self.num_observations = 22, 461
        self.start_frame = start_frame
        self._current_frame = start_frame
        self._dx = SimpleNamespace(
            qpos=np.zeros(12, dtype=np.float32), qvel=np.zeros(11, dtype=np.float32),
            ctrl=np.zeros(26, dtype=np.float32), ncon=np.asarray(1),
        )
        self._dx.qpos[5] = 1.0
        self.applied_delta, self.nan, self.closed = applied_delta, nan, False
        self._last_action_trace = None

    def reset(self):
        self._current_frame = self.start_frame
        return np.zeros(461, dtype=np.float32), {}

    def step(self, action):
        assert action.shape == (22,)
        frame = self._current_frame
        target = np.full(22, frame, dtype=np.float32)
        applied = target.copy(); applied[0] += self.applied_delta
        self._last_action_trace = {
            "decision_frame": frame, "reference_active_target": target,
            "corrected_active_target": target, "applied_active_target": applied,
            "full_target": np.full(26, frame, dtype=np.float32),
        }
        self._dx.ctrl[:] = self._last_action_trace["full_target"]
        self._current_frame += 1
        self._dx.qpos[0] = np.nan if self.nan else self._current_frame
        self._dx.qpos[5] = 1.0
        obs = np.full(461, self._current_frame, dtype=np.float32)
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


def test_metric_reports_exactness_and_first_threshold_frame():
    left = np.zeros((3, 2), dtype=np.float32)
    right = left.copy(); right[1, 0] = 2e-4
    result = metric(left, right, np.array([62, 63, 64]))
    assert not result["exact"]
    assert result["first_nonzero"] == 63
    assert result["first_exceed"]["0.0001"] == 63
    assert result["first_exceed"]["0.001"] is None


def test_compare_rejects_identity_and_applied_target_invariants_and_closes():
    native, mjx = FakeEnv("mujoco-native"), FakeEnv("mjx-warp", identity="other")
    with pytest.raises(ValueError, match="identity mismatch"):
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

    native, mjx = FakeEnv("mujoco-native"), FakeEnv("mjx-warp", start_frame=63)
    with pytest.raises(ValueError, match="frame contract mismatch"):
        compare_rollouts(native, mjx, steps=2)
    assert native.closed and mjx.closed


def test_compare_rejects_nonfinite_arrays():
    native, mjx = FakeEnv("mujoco-native"), FakeEnv("mjx-warp", nan=True)
    with pytest.raises(ValueError, match="nonfinite"):
        compare_rollouts(native, mjx, steps=2)
    assert native.closed and mjx.closed


def test_evidence_serialization_has_prefixed_arrays_and_contract_schema(tmp_path):
    report, arrays = compare_rollouts(FakeEnv("mujoco-native"), FakeEnv("mjx-warp"), steps=2)
    output = tmp_path / "evidence.json"
    npz_path, digest = write_evidence(output, report, arrays)
    payload = json.loads(output.read_text())
    assert payload["schema"] == "dexhandrl.native_mjx_backend_evidence.v1"
    assert payload["artifact"]["sha256"] == digest
    assert "native_qpos" in np.load(npz_path).files
    assert "mjx_applied_active_target" in np.load(npz_path).files
    assert payload["comparison"]["processed_obs_components"]["hand_pose"]["exact"]
    assert "near-trivial" in payload["backend_authority"]
