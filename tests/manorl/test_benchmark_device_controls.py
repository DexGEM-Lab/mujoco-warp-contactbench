from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import benchmark_manorl_device_controls as benchmark


SCRIPT = Path(__file__).parents[2] / "tools" / "benchmark_manorl_device_controls.py"


def _write_provenance(path: Path, *, schema: str = benchmark.PROVENANCE_SCHEMA) -> Path:
    files = {"tools/benchmark_manorl_device_controls.py": "test-sha256"}
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "source_worktree": str(Path(__file__).parents[2]),
                "source_commit": "d7c2898-test",
                "source_diff_sha256": "dirty-diff-test",
                "mirror_path": "local-test-mirror",
                "source_files": files,
                "mirror_file_hashes": files,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _command(output: Path, provenance: Path, *, seed: int = 42) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--device", "cpu",
        "--num-envs", "1",
        "--warmup-steps", "0",
        "--steps", "1",
        "--repeats", "1",
        "--order-seed", str(seed),
        "--output", str(output),
        "--provenance", str(provenance),
    ]


def _cpu_environment() -> dict[str, str]:
    return {**os.environ, "JAX_PLATFORMS": "cpu"}


def test_v3_subprocess_harness_publishes_independent_validated_samples(tmp_path: Path) -> None:
    provenance = _write_provenance(tmp_path / "source-provenance.json")
    output = tmp_path / "ablation.json"

    completed = subprocess.run(
        _command(output, provenance),
        env=_cpu_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == benchmark.SUMMARY_SCHEMA
    assert len(payload["results"]) == len(benchmark.MODE_CONFIGS)
    assert len({result["process_id"] for result in payload["results"]}) == len(benchmark.MODE_CONFIGS)
    assert all(result["process_id"] != os.getpid() for result in payload["results"])
    assert {result["mode"] for result in payload["results"]} == set(benchmark.MODE_CONFIGS)
    for artifact in payload["artifacts"]:
        assert all(Path(path).is_file() for path in artifact.values())
        assert json.loads(Path(artifact["json"]).read_text(encoding="utf-8"))["schema"] == benchmark.MODE_SCHEMA


def test_v3_subprocess_harness_fails_closed_on_child_failure(tmp_path: Path) -> None:
    seed = 42
    provenance = _write_provenance(tmp_path / "source-provenance.json")
    output = tmp_path / "ablation.json"
    order = list(benchmark.MODE_CONFIGS)
    random.Random(seed).shuffle(order)
    stale_child = output.with_name(f"{output.stem}.repeat00.order00.{order[0]}.json")
    stale_child.write_text("preserve stale child\n", encoding="utf-8")

    completed = subprocess.run(
        _command(output, provenance, seed=seed),
        env=_cpu_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode != 0
    assert not output.exists()
    assert stale_child.read_text(encoding="utf-8") == "preserve stale child\n"
    assert "benchmark child failed" in completed.stderr


def test_v3_harness_rejects_stale_child_schema(tmp_path: Path) -> None:
    provenance = _write_provenance(tmp_path / "source-provenance.json")
    expected = json.loads(provenance.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="child schema"):
        benchmark._validate_mode_payload(
            {"schema": "manorl.device_controls_ablation.v2", "provenance": expected, "result": {}},
            expected,
        )


def test_v3_subprocess_harness_rejects_stale_provenance_schema(tmp_path: Path) -> None:
    provenance = _write_provenance(tmp_path / "stale-provenance.json", schema="stale.v0")
    output = tmp_path / "ablation.json"

    completed = subprocess.run(
        _command(output, provenance),
        env=_cpu_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode != 0
    assert not output.exists()
    assert benchmark.PROVENANCE_SCHEMA in completed.stderr


def test_v3_cli_rejects_unvalidated_legacy_invocation() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--device", "cpu", "--num-envs", "1", "--steps", "1"],
        env=_cpu_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode != 0
    assert "requires --output and --provenance" in completed.stderr


def test_v3_child_rejects_zero_exit_without_json_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provenance = _write_provenance(tmp_path / "source-provenance.json")
    args = SimpleNamespace(
        device="cpu",
        num_envs=1,
        warmup_steps=0,
        steps=1,
        order_seed=42,
        profile_phases=False,
    )
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    with pytest.raises(RuntimeError, match="did not publish JSON"):
        benchmark._run_child(
            args,
            mode="legacy_controls_diagnostics",
            repeat=0,
            order_index=0,
            output=tmp_path / "missing.json",
            provenance=provenance,
        )
