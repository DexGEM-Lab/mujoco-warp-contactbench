"""Current frozen-evaluation NPZ renderer input contracts."""
from __future__ import annotations

import numpy as np
import pytest

from tools.render_manorl_autonomy_evaluation import (
    REQUIRED_ARRAYS,
    load_evaluation_artifact,
)


def _arrays(frames: int = 2):
    return {
        "actual_qpos": np.zeros((frames, 28)),
        "reference_qpos_feasible": np.zeros((frames, 28)),
        "actual_object_position": np.zeros((frames, 3)),
        "reference_object_position": np.zeros((frames, 3)),
        "actual_object_quaternion_xyzw": np.tile(
            [0.0, 0.0, 0.0, 1.0], (frames, 1)
        ),
        "reference_object_quaternion_xyzw": np.tile(
            [0.0, 0.0, 0.0, 1.0], (frames, 1)
        ),
    }


def test_load_evaluation_artifact_accepts_current_pose_contract(tmp_path):
    path = tmp_path / "trace.npz"
    np.savez_compressed(path, **_arrays())
    loaded = load_evaluation_artifact(path)
    assert tuple(name in loaded for name in REQUIRED_ARRAYS) == (
        True,
    ) * len(REQUIRED_ARRAYS)


def test_load_evaluation_artifact_rejects_missing_or_nonfinite(tmp_path):
    missing = tmp_path / "missing.npz"
    arrays = _arrays()
    arrays.pop("actual_qpos")
    np.savez_compressed(missing, **arrays)
    with pytest.raises(ValueError, match="missing"):
        load_evaluation_artifact(missing)
    nonfinite = tmp_path / "nonfinite.npz"
    arrays = _arrays()
    arrays["actual_object_position"][0, 0] = np.nan
    np.savez_compressed(nonfinite, **arrays)
    with pytest.raises(ValueError, match="finite"):
        load_evaluation_artifact(nonfinite)
