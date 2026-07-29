from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_generic_train_and_inference_shell_syntax() -> None:
    subprocess.run(
        ["bash", "-n", str(ROOT / "train.sh"), str(ROOT / "inference.sh")],
        check=True,
    )


def test_generic_train_rejects_nonpositive_environment_count() -> None:
    result = subprocess.run(
        [str(ROOT / "train.sh"), "cube1", "0", "0"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "num_envs must be a positive integer" in result.stderr


def test_generic_inference_requires_checkpoint() -> None:
    result = subprocess.run(
        [str(ROOT / "inference.sh"), "cube1", "01", "20", "0"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Set CHECKPOINT or MANORL_CHECKPOINT" in result.stderr
