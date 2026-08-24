from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim.manorl.approach_prefix import APPROACH_PREFIX_ONLY_PRODUCTION_CONTRACT
from tools import run_manorl_all_objects_near_far as runner


def test_default_runner_never_requests_retreat(monkeypatch, tmp_path: Path) -> None:
    observed: list[list[str]] = []

    def fake_run(command, *, env, check):
        observed.append(list(command))
        assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert env["MANORL_RUNNER_INHERITED"] == "yes"
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("MANORL_RUNNER_INHERITED", "yes")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc = runner._run_object_mode(
        object_name="banana",
        mode="near",
        gpu=2,
        parents_manifest=tmp_path / "parents.json",
        pairs="banana:01",
        n_env=1,
        output=tmp_path / "near.lance",
        seed=420001,
        replace=True,
    )
    assert rc == 0
    command = observed[0]
    assert "--approach-prefix" in command
    assert command[command.index("--approach-mode") + 1] == "near"
    assert "--retreat-suffix" not in command
    assert "--replace" in command
    assert command[command.index("--predecoded-manifest") + 1] == str(
        runner.PREDECODED_MANIFEST
    )
    assert command[command.index("--episodes-per-identity") + 1] == "5"
    assert command[command.index("--max-attempts-per-identity") + 1] == "12"


def test_default_runner_summary_declares_prefix_only_contract(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "OBJECTS", ("banana",))
    monkeypatch.setattr(runner, "_verified_sources", lambda object_name: ["banana_01_052"])
    monkeypatch.setattr(
        runner,
        "_write_parents_manifest",
        lambda object_name, output_dir: output_dir / "parents_banana.json",
    )
    parent_root = tmp_path / "parents"
    (parent_root / "banana").mkdir(parents=True)
    (parent_root / "banana" / "parents_by_action.json").write_text(
        json.dumps({"actions": {"01": {"banana_01_052": "/parent.json"}}})
    )
    monkeypatch.setattr(runner, "PARENT_ROOT", parent_root)
    monkeypatch.setattr(runner, "_run_object_mode", lambda **kwargs: 0)
    assert runner.main(["--output-dir", str(tmp_path), "--gpus", "0"]) == 0
    summary = json.loads((tmp_path / "run-summary.json").read_text())
    assert summary["contract"] == runner.RUN_CONTRACT
    assert summary["production_contract"] == APPROACH_PREFIX_ONLY_PRODUCTION_CONTRACT
    assert summary["retreat_suffix"] is None
    assert summary["episodes_per_identity_per_mode"] == 5
    assert summary["max_attempts_per_identity_per_mode"] == 12


def test_default_runner_surfaces_worker_setup_failure(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "OBJECTS", ("banana",))
    monkeypatch.setattr(
        runner, "_verified_sources", lambda object_name: ["banana_01_052"]
    )
    monkeypatch.setattr(
        runner,
        "_write_parents_manifest",
        lambda object_name, output_dir: (_ for _ in ()).throw(
            RuntimeError("broken parent manifest")
        ),
    )
    assert runner.main(["--output-dir", str(tmp_path), "--gpus", "0"]) == 1
    summary = json.loads((tmp_path / "run-summary.json").read_text())
    assert summary["failed"] is True
    assert summary["results"] == [
        {
            "gpu": 0,
            "object": "banana",
            "mode": "setup",
            "exit_code": 1,
            "error": "RuntimeError: broken parent manifest",
        }
    ]
