"""Benchmark the device-resident homogeneous cube2 autonomy path.

Usage is intentionally explicit: operators choose the GPU and batch size; this
command reports transition throughput, warmup/compile wall time, validity and
reset counts without claiming a result before a run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

# Keep the documented direct-file invocation source-authoritative.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from sim.manorl.autonomy_batch import BatchedAutonomyRuntime
from sim.manorl.trajectory_package import load_trajectory_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--identity", default="cube2_02_2833")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        parser.error("--steps and --num-envs must be positive")
    catalog = load_trajectory_package(Path(args.package))
    trajectory = next((item for item in catalog.trajectories if item.identity.identity == args.identity), None)
    if trajectory is None:
        parser.error(f"identity absent from package: {args.identity}")
    compile_start = time.perf_counter()
    runtime = BatchedAutonomyRuntime(trajectory, num_envs=args.num_envs, device=args.device)
    compile_seconds = time.perf_counter() - compile_start
    actions = runtime.jp.zeros((args.num_envs, 28), dtype=runtime.jp.float32)
    warmup_start = time.perf_counter()
    runtime.step(actions)
    runtime.jp.asarray(runtime.observation).block_until_ready()
    warmup_seconds = time.perf_counter() - warmup_start
    start = time.perf_counter(); resets = 0; valid = 0
    for _ in range(args.steps):
        _, _, done, info = runtime.step(actions)
        runtime.jp.asarray(runtime.observation).block_until_ready()
        resets += int(np.asarray(done).sum())
        valid += int(np.asarray(info["valid"]).sum())
    elapsed = time.perf_counter() - start
    transitions = args.steps * args.num_envs
    print(json.dumps({
        "batch": args.num_envs,
        "steps": args.steps,
        "transitions": transitions,
        "compile_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds,
        "steady_seconds": elapsed,
        "environment_transitions_per_second": transitions / max(elapsed, 1e-12),
        "validity_count": valid,
        "reset_count": resets,
        "host_transfer_scope": "compact done/valid counters only; full observation/contact buffers remain device arrays",
        "peak_memory_bytes": None,
        "note": "peak memory requires operator-side GPU telemetry (e.g. nvidia-smi); no speed claim is implied",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
