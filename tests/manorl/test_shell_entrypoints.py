from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_generic_train_and_inference_shell_syntax() -> None:
    subprocess.run(
        [
            "bash",
            "-n",
            str(ROOT / "train.sh"),
            str(ROOT / "inference.sh"),
            str(ROOT / "test.sh"),
            str(ROOT / "synthesize.sh"),
        ],
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


def test_generic_train_rejects_unsupported_reference_fps() -> None:
    result = subprocess.run(
        [str(ROOT / "train.sh"), "cube1", "1", "0"],
        text=True,
        capture_output=True,
        env={**os.environ, "MANORL_REFERENCE_FPS": "200"},
    )
    assert result.returncode == 2
    assert "MANORL_REFERENCE_FPS must be 100 or 120" in result.stderr


def test_generic_inference_requires_checkpoint() -> None:
    result = subprocess.run(
        [str(ROOT / "inference.sh"), "cube1", "01", "20", "0"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Set CHECKPOINT or MANORL_CHECKPOINT" in result.stderr


def test_synthesis_requires_checkpoint() -> None:
    result = subprocess.run(
        [str(ROOT / "synthesize.sh"), "cube2", "02", "5", "0"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "Set CHECKPOINT or MANORL_CHECKPOINT" in result.stderr


def test_synthesis_rejects_unknown_output_format(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-000001.pt"
    checkpoint.write_bytes(b"checkpoint")
    Path(f"{checkpoint}.json").write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [str(ROOT / "synthesize.sh"), "cube2", "02", "1", "0"],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "CHECKPOINT": str(checkpoint),
            "MANORL_PYTHON": "/bin/true",
            "MANORL_SYNTH_OUTPUT_FORMAT": "unknown",
        },
    )
    assert result.returncode == 2
    assert (
        "MANORL_SYNTH_OUTPUT_FORMAT must be full or compact-replay-visual"
        in result.stderr
    )


def test_reference_test_rejects_invalid_gpu() -> None:
    result = subprocess.run(
        [str(ROOT / "test.sh"), "invalid"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "physical_gpu must be a non-negative integer" in result.stderr


def test_reference_test_rejects_unsupported_reference_fps() -> None:
    result = subprocess.run(
        [str(ROOT / "test.sh"), "0"],
        text=True,
        capture_output=True,
        env={**os.environ, "MANORL_REFERENCE_FPS": "200"},
    )
    assert result.returncode == 2
    assert "MANORL_REFERENCE_FPS must be 100 or 120" in result.stderr


def test_reference_test_passes_explicit_padding(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.lance"
    dataset.mkdir()
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    fake_python.chmod(0o755)
    result = subprocess.run(
        [str(ROOT / "test.sh"), "2", "73", "91"],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "DISPLAY": ":99",
            "MANORL_PYTHON": str(fake_python),
            "MANORL_DATASET_PATH": str(dataset),
        },
        check=True,
    )
    arguments = result.stdout.splitlines()
    assert arguments[arguments.index("--pre-padding") + 1] == "73"
    assert arguments[arguments.index("--post-padding") + 1] == "91"


def test_reference_test_padding_defaults_match_training_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.lance"
    dataset.mkdir()
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    fake_python.chmod(0o755)
    result = subprocess.run(
        [str(ROOT / "test.sh"), "0"],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "DISPLAY": ":99",
            "MANORL_PYTHON": str(fake_python),
            "MANORL_DATASET_PATH": str(dataset),
        },
        check=True,
    )
    arguments = result.stdout.splitlines()
    assert arguments[arguments.index("--pre-padding") + 1] == "180"
    assert arguments[arguments.index("--post-padding") + 1] == "250"
