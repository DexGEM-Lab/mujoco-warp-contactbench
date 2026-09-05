from __future__ import annotations

import json
import os
import shlex
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
            str(ROOT / "scripts" / "view_manorl_lance.sh"),
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


def test_generic_train_rejects_invalid_padding(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.lance"
    dataset.mkdir()
    for name in ("MANORL_PRE_PADDING", "MANORL_POST_PADDING"):
        result = subprocess.run(
            [str(ROOT / "train.sh"), "banana", "1", "0"],
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "MANORL_DATASET_PATH": str(dataset),
                "MANORL_PYTHON": "/bin/true",
                name: "-1",
            },
        )
        assert result.returncode == 2
        assert f"{name} must be a non-negative integer" in result.stderr


def test_generic_train_passes_padding_overrides(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "READY").write_text("digest\n", encoding="utf-8")
    (package / "manifest.json").write_text(
        json.dumps({"resolved_pairs": ["banana:18"]}), encoding="utf-8"
    )
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import sys
if sys.argv[1] == '-c':
    print(json.dumps(sys.argv[-1]))
elif sys.argv[1] == '-':
    print('banana:18')
else:
    print('\\n'.join(sys.argv[1:]))
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output = tmp_path / "run"
    result = subprocess.run(
        [str(ROOT / "train.sh"), "banana", "1", "0"],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "MANORL_PYTHON": str(fake_python),
            "MANORL_TRAJECTORY_PACKAGE": str(package),
            "MANORL_PRE_PADDING": "180",
            "MANORL_POST_PADDING": "180",
            "MANORL_UPDATES": "1",
            "MANORL_CHECKPOINT_INTERVAL": "1",
            "MANORL_WANDB": "false",
            "MANORL_OUTPUT": str(output),
            "MANORL_TIMEOUT": "1m",
        },
        check=True,
    )
    arguments = result.stdout.splitlines()
    assert arguments[arguments.index("--pre-padding") + 1] == "180"
    assert arguments[arguments.index("--post-padding") + 1] == "180"
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pre_padding"] == 180
    assert manifest["post_padding"] == 180


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


def test_lance_viewer_meta_script_builds_one_env_command(tmp_path: Path) -> None:
    dataset = tmp_path / "daily.lance"
    dataset.mkdir()
    script = ROOT / "scripts" / "view_manorl_lance.sh"
    result = subprocess.run(
        [
            str(script),
            "--dry-run",
            "--dataset",
            str(dataset),
            "--dataset-version",
            "12",
            "--object",
            "banana",
            "--gesture",
            "18",
            "--reference-fps",
            "100",
            "--pre-padding",
            "180",
            "--post-padding",
            "180",
            "--hand-side",
            "right",
            "--display",
            ":1",
            "--python",
            "/bin/true",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    command_line = next(
        line.removeprefix("command: ")
        for line in result.stdout.splitlines()
        if line.startswith("command: ")
    )
    arguments = shlex.split(command_line)
    assert arguments[:4] == [
        "/bin/true",
        "-m",
        "sim.manorl.view_environment",
        "--device",
    ]
    assert arguments[arguments.index("--dataset-path") + 1] == str(dataset)
    assert arguments[arguments.index("--dataset-version") + 1] == "12"
    assert arguments[arguments.index("--object") + 1] == "banana"
    assert arguments[arguments.index("--gesture") + 1] == "18"
    assert arguments[arguments.index("--reference-fps") + 1] == "100"
    assert arguments[arguments.index("--pre-padding") + 1] == "180"
    assert arguments[arguments.index("--post-padding") + 1] == "180"
    assert arguments[arguments.index("--num-envs") + 1] == "1"
    assert arguments[arguments.index("--render-env") + 1] == "0"
    assert "--loop" in arguments
    assert "JAX_PLATFORMS=cpu" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=''" in result.stdout


def test_lance_viewer_meta_script_rejects_environment_count_override(tmp_path: Path) -> None:
    dataset = tmp_path / "daily.lance"
    dataset.mkdir()
    result = subprocess.run(
        [
            str(ROOT / "scripts" / "view_manorl_lance.sh"),
            "--dataset",
            str(dataset),
            "--object",
            "banana",
            "--gesture",
            "18",
            "--num-envs",
            "2",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "unknown argument: --num-envs" in result.stderr


def test_synthesis_rejects_negative_xy_offset(tmp_path: Path) -> None:
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
            "MANORL_SYNTH_OBJECT_XY_OFFSET_M": "-0.02",
        },
    )
    assert result.returncode == 2
    assert "MANORL_SYNTH_OBJECT_XY_OFFSET_M" in result.stderr


def test_synthesis_exporter_accepts_xy_offset_arg() -> None:
    result = subprocess.run(
        [
            "python",
            str(ROOT / "tools" / "export_manorl_synthetic_lance.py"),
            "--help",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "--object-xy-offset-m" in result.stdout


def test_default_synthesis_requires_pre60_bundle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-000001.pt"
    checkpoint.write_bytes(b"checkpoint")
    Path(f"{checkpoint}.json").write_text("{}", encoding="utf-8")
    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [str(ROOT / "synthesize.sh"), "cube2", "02", "1", "0"],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "CHECKPOINT": str(checkpoint),
            "MANORL_PYTHON": "/bin/true",
            "MANORL_SYNTH_ACCEPTED_PARENT": str(parent),
        },
    )
    assert result.returncode == 2
    assert "canonical pre60 bundle" in result.stderr


def test_synthesis_defaults_to_prefix_only_and_rejects_retreat(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-000001.pt"
    checkpoint.write_bytes(b"checkpoint")
    Path(f"{checkpoint}.json").write_text("{}", encoding="utf-8")
    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="utf-8")
    predecoded = tmp_path / "pre60-manifest.json"
    predecoded.write_text("{}", encoding="utf-8")
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    fake_python.chmod(0o755)
    result = subprocess.run(
        [str(ROOT / "synthesize.sh"), "cube2", "02", "1", "0"],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "CHECKPOINT": str(checkpoint),
            "MANORL_PYTHON": str(fake_python),
            "MANORL_SYNTH_ACCEPTED_PARENT": str(parent),
            "MANORL_PREDECODED_MANIFEST": str(predecoded),
        },
        check=True,
    )
    arguments = result.stdout.splitlines()
    assert "--approach-prefix" in arguments
    assert arguments[arguments.index("--approach-mode") + 1] == "far"
    assert "--accepted-parent" in arguments
    assert "--retreat-suffix" not in arguments
    assert "--allow-partial-yield" in arguments
    assert arguments[arguments.index("--max-attempts-per-identity") + 1] == "12"
    assert arguments[arguments.index("--predecoded-manifest") + 1] == str(
        predecoded.resolve()
    )

    rejected = subprocess.run(
        [str(ROOT / "synthesize.sh"), "cube2", "02", "1", "0"],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "CHECKPOINT": str(checkpoint),
            "MANORL_PYTHON": str(fake_python),
            "MANORL_SYNTH_ACCEPTED_PARENT": str(parent),
            "MANORL_PREDECODED_MANIFEST": str(predecoded),
            "MANORL_SYNTH_RETREAT_SUFFIX": "true",
        },
    )
    assert rejected.returncode == 2
    assert "no longer part of the default production contract" in rejected.stderr
