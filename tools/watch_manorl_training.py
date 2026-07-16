#!/usr/bin/env python3
"""Summarize ManoRL JSON or human training logs, optionally following a live file."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator


def summarize_record(record: dict[str, Any]) -> str | None:
    if record.get("schema") == "manorl.completed_episode_returns.v1":
        returns = record.get("returns", [])
        if not returns:
            return "episodes count=0"
        values = (float(value) for value in returns)
        first = next(values)
        count, total, minimum, maximum = 1, first, first, first
        for value in values:
            count += 1
            total += value
            minimum = min(minimum, value)
            maximum = max(maximum, value)
        return f"episodes update={record.get('update')} count={count} mean={total / count:.4f} min={minimum:.4f} max={maximum:.4f}"
    if record.get("event") == "training_update":
        metrics = record.get("metrics", {})
        return f"update={int(metrics.get('update', 0))} transitions={int(metrics.get('environment_transitions', 0))} reward={float(metrics.get('reward_mean', 0.0)):.4f}"
    if record.get("event") == "training_complete":
        throughput = record.get("throughput", {})
        return f"training complete transitions={int(throughput.get('environment_transitions', 0))}"
    return None


def summarize_line(line: str) -> str | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return line.rstrip() or None
    return summarize_record(record) if isinstance(record, dict) else None


def iter_log(path: Path, *, follow: bool, poll_seconds: float) -> Iterator[str]:
    with path.open(encoding="utf-8") as stream:
        while True:
            line = stream.readline()
            if line:
                yield line
            elif not follow:
                return
            else:
                time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    for line in iter_log(args.log, follow=args.follow, poll_seconds=args.poll_seconds):
        summary = summarize_line(line)
        if summary is not None:
            print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
