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
        "schema": "manorl.completed_episode_returns.v1", "update": 3, "update_step": 48,
        "vector_step": 144, "environment_transitions": 1_152, "env_ids": list(range(20_000)), "returns": returns,
    })
    update = json.dumps({
        "event": "training_update",
        "metrics": {"update": 3.0, "environment_transitions": 1_152.0, "reward_mean": -0.125},
    })
    assert watcher.summarize_line(line) == "episodes update=3 count=20000 mean=9999.5000 min=0.0000 max=19999.0000"
    assert watcher.summarize_line(update) == "update=3 transitions=1152 reward=-0.1250"

    log = tmp_path / "run.train.log"
    log.write_text(line + "\n" + update + "\n" + '{"event":"training_complete","throughput":{"environment_transitions":1152}}\n', encoding="utf-8")
    assert watcher.main([str(log)]) == 0
    output = capsys.readouterr().out
    assert "episodes update=3 count=20000 mean=9999.5000 min=0.0000 max=19999.0000" in output
    assert "update=3 transitions=1152 reward=-0.1250" in output
    assert "training complete transitions=1152" in output
    assert str(returns[:10]) not in output
