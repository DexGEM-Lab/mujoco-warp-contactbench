from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _watcher():
    path = Path(__file__).resolve().parents[2] / "tools" / "watch_manorl_training.py"
    spec = importlib.util.spec_from_file_location("watch_manorl_training_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_watcher_summarizes_giant_episode_record_without_echoing_arrays(tmp_path: Path, capsys) -> None:
    watcher = _watcher()
    returns = list(range(20_000))
    line = json.dumps({
        "schema": "manorl.completed_episode_returns.v1", "update": 3, "returns": returns,
    })
    assert "[" not in watcher.summarize_line(line)
    assert "count=20000" in watcher.summarize_line(line)

    log = tmp_path / "run.train.log"
    log.write_text(line + "\n" + '{"event":"training_complete","throughput":{"environment_transitions":4}}\n', encoding="utf-8")
    assert watcher.main([str(log)]) == 0
    output = capsys.readouterr().out
    assert "count=20000" in output
    assert "training complete transitions=4" in output
    assert str(returns[:10]) not in output
