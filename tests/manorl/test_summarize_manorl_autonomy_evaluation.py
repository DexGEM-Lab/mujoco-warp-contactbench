from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.summarize_manorl_autonomy_evaluation import summarize_evaluation


def _fixture(
    root: Path,
    *,
    frames: int,
    reason: int,
    loaded_airborne_slice: slice | None,
    diagnostic: bool = False,
) -> tuple[Path, Path, Path]:
    checkpoint = root / "policy.update000050.pt"
    checkpoint.write_bytes(b"checkpoint")
    actual = np.zeros((frames, 3), dtype=np.float32)
    reference = np.zeros((frames, 3), dtype=np.float32)
    clearance = np.zeros((frames,), dtype=np.float32)
    force = np.zeros((frames, 16, 3), dtype=np.float32)
    if loaded_airborne_slice is not None:
        clearance[loaded_airborne_slice] = 0.01
        force[loaded_airborne_slice, 15, 0] = 0.03
        force[loaded_airborne_slice, 3, 0] = -0.03
    reason_code = np.zeros((frames,), dtype=np.int64)
    reason_code[-1] = reason
    artifact = root / "eval.npz"
    np.savez_compressed(
        artifact,
        actual_object_position=actual,
        reference_object_position=reference,
        paired_force_on_object=force,
        bottom_clearance=clearance,
        reason_code=reason_code,
        natural_prefix_steps=np.asarray(frames),
        full_horizon_diagnostic=np.asarray(diagnostic),
    )
    trace = root / "eval.json"
    trace.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "identity": "cube2_02_2833",
                "steps": frames,
                "return": 10.0,
                "natural_first_termination": {
                    "policy_step": frames - 1,
                    "reason_code": reason,
                },
                "full_horizon_diagnostic": diagnostic,
                "diagnostic_boundary": (
                    "natural termination recorded; continued after it"
                    if diagnostic
                    else "stopped at natural first termination"
                ),
                "provenance": {
                    "checkpoint": {
                        "clock": {
                            "control_timestep": 1 / 120,
                        }
                    }
                },
            }
        )
    )
    return trace, artifact, checkpoint


def test_summary_requires_natural_horizon_and_sustained_loaded_airborne(tmp_path: Path) -> None:
    trace, artifact, checkpoint = _fixture(
        tmp_path,
        frames=50,
        reason=1,
        loaded_airborne_slice=slice(10, 45),
    )
    summary = summarize_evaluation(
        trace_path=trace,
        artifact_path=artifact,
        checkpoint_path=checkpoint,
    )
    assert summary.natural_horizon is True
    assert summary.strict_hold_frames_required == 30
    assert summary.strict_grasp_success is True
    assert summary.loaded_airborne_frames == 35
    assert summary.thumb_loaded_frames == 35
    assert summary.opposing_loaded_frames == 35
    assert summary.checkpoint_update == 50


def test_summary_does_not_call_deviation_or_short_lift_success(tmp_path: Path) -> None:
    trace, artifact, checkpoint = _fixture(
        tmp_path,
        frames=20,
        reason=2,
        loaded_airborne_slice=slice(10, 15),
    )
    summary = summarize_evaluation(
        trace_path=trace,
        artifact_path=artifact,
        checkpoint_path=checkpoint,
    )
    assert summary.deviation is True
    assert summary.natural_horizon is False
    assert summary.strict_grasp_success is False
    assert summary.longest_loaded_airborne_frames == 5


def test_summary_rejects_diagnostic_continuation(tmp_path: Path) -> None:
    trace, artifact, checkpoint = _fixture(
        tmp_path,
        frames=20,
        reason=2,
        loaded_airborne_slice=None,
        diagnostic=True,
    )
    with pytest.raises(ValueError, match="natural, non-diagnostic"):
        summarize_evaluation(
            trace_path=trace,
            artifact_path=artifact,
            checkpoint_path=checkpoint,
        )
