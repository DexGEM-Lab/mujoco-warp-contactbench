#!/usr/bin/env python3
"""Run one replay command, rejecting its first real Warp capacity overflow.

The pinned DataWarp lacks a live CCD count. Its native overflow diagnostic can
therefore appear even when Python exits successfully. This parent owns the log
and child process group, so it can reject and stop such a run before the next
action starts. It never retries or changes capacity automatically.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

OVERFLOW = re.compile(
    r"\b(?:CCD|nefc|nacon|narrowphase|constraint|contact) overflow - please increase\b",
    re.IGNORECASE,
)
OVERFLOW_EXIT_CODE = 86


def terminate(child: subprocess.Popen) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def run_guarded(command: list[str], log: Path) -> int:
    if not command:
        raise ValueError("replay command is required")
    receipt = Path(str(log) + ".guard.json")
    if log.exists() or receipt.exists():
        raise FileExistsError(f"guard requires fresh log and receipt: {log}")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    rejected = None
    with log.open("x") as stream:
        child = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1, start_new_session=True,
        )
        try:
            for line in child.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
                if OVERFLOW.search(line):
                    rejected = line.strip()
                    terminate(child)
                    break
            code = child.wait()
        finally:
            if child.poll() is None:
                terminate(child)
            child.stdout.close()
    state = "rejected_overflow" if rejected else ("complete" if code == 0 else "failed")
    with receipt.open("x") as stream:
        json.dump(
            {"state": state, "command": command, "pid": child.pid,
             "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
             "process_exit_code": code, "overflow": rejected},
            stream, indent=2,
        )
        stream.write("\n")
    if rejected:
        print("REJECTED_OVERFLOW: " + rejected, flush=True)
        return OVERFLOW_EXIT_CODE
    return code if code >= 0 else 128 - code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    raise SystemExit(run_guarded(command, args.log))


if __name__ == "__main__":
    main()
