from __future__ import annotations

import json
from pathlib import Path

from tools.summarize_manorl_training_convergence import summarize_convergence


def _row(update: int, *, complete: int = 0, loaded_airborne: float = 0.0) -> dict[str, float]:
    completed = 2
    return {
        "update": float(update),
        "transitions": float(update * 100),
        "valid": 1.0,
        "episodes/completed_count": float(completed),
        "termination/reference_complete_count": float(complete),
        "termination/deviation_count": float(completed - complete),
        "termination/fallen_count": 0.0,
        "termination/nonfinite_count": 0.0,
        "physics/loaded_airborne_fraction": loaded_airborne,
        "physics/airborne_5mm_fraction": loaded_airborne,
        "physics/paired_loaded_contact_fraction": 0.1,
        "physics/object_position_error_l2_mean": 0.01,
        "physics/bottom_clearance_max": loaded_airborne,
        "episodes/max_clearance_mean": loaded_airborne,
        "episodes/loaded_contact_frames_mean": 3.0,
        "reward_mean": 1.0,
        "info/exact_kl_mean": 0.02,
        "teacher_anchor/loss": 0.3,
        "performance/sampling_time": 2.0,
        "performance/optimizer_time": 3.0,
        "config/learning_rate": 3e-5,
        "config/learning_epochs": 4.0,
        "config/mini_batches": 16.0,
        "config/rollouts": 32.0,
        "config/teacher_anchor_beta": 1.0,
        "config/teacher_anchor_passes": 2.0,
    }


def _write(path: Path, rows: list[dict[str, float]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_incomplete_budget_remains_in_progress(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write(metrics, [_row(1), _row(2, loaded_airborne=0.1)])
    result = summarize_convergence(metrics, target_updates=3, window_updates=1)
    assert result.evidence_status == "training_in_progress"
    assert result.first_loaded_airborne_update == 2
    assert result.first_reference_complete_update is None
    assert result.continuous_updates is True
    assert result.all_rows_valid is True


def test_complete_budget_without_horizon_is_not_converged(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write(metrics, [_row(1), _row(2), _row(3, loaded_airborne=0.2)])
    result = summarize_convergence(metrics, target_updates=3, window_updates=2)
    assert result.evidence_status == "budget_complete_without_reference_completion"
    assert result.peak_loaded_airborne_update == 3
    assert result.last_200_updates.reference_complete_rate == 0.0


def test_horizon_signal_still_requires_frozen_evaluation(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write(metrics, [_row(1), _row(2, complete=1), _row(3)])
    result = summarize_convergence(metrics, target_updates=3, window_updates=3)
    assert result.evidence_status == "budget_complete_requires_frozen_evaluation"
    assert result.first_reference_complete_update == 2
    assert result.windows[0].reference_complete_rate == 1 / 6
