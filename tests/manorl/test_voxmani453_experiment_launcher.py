"""The remote experiment may select only a GPU with no compute occupants."""

from types import SimpleNamespace

import pytest

from tools import run_voxmani453_experiment as launcher


def test_idle_selection_excludes_compute_processes_and_reserved_memory(monkeypatch):
    rows = {
        "--query-compute-apps=gpu_uuid,pid": [["GPU-busy", "123"]],
        "--query-gpu=index,uuid,memory.used": [
            ["0", "GPU-busy", "23"],  # Low memory/utilization is insufficient.
            ["1", "GPU-display", "512"],
            ["2", "GPU-free", "23"],
        ],
    }
    monkeypatch.setattr(launcher, "nvidia_rows", lambda query: rows[query])
    assert launcher.idle_gpus() == [("2", "GPU-free")]


def test_missing_idle_gpu_fails_without_starting_training(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "idle_gpus", lambda: [])
    started = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: started.append(a))
    with pytest.raises(RuntimeError, match="No idle GPU"):
        launcher.acquire_idle_gpu(tmp_path, wait_seconds=0)
    assert not started


def test_nvidia_csv_keeps_physical_uuid_and_checks_command_success(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=" 2, GPU-physical-uuid, 23 \n")

    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.nvidia_rows("--query-gpu=index,uuid,memory.used") == [
        ["2", "GPU-physical-uuid", "23"]
    ]
    assert calls[0][1]["check"] is True
