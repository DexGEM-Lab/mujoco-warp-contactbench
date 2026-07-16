#!/usr/bin/env python3
"""Measure the four-way ManoRL environment control/diagnostic ablation.

Example (use an idle GPU only, with a fresh provenance sidecar):
  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
    /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
    tools/benchmark_manorl_device_controls.py --device gpu --num-envs 2048 \
    --warmup-steps 8 --steps 48 --repeats 3 --profile-phases \
    --output outputs/manorl/bench_phase_ablation_<stamp>.json \
    --provenance outputs/manorl/bench_phase_ablation_<stamp>.source_provenance.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.trajectory import TrajectorySelection, load_assigned_trajectory_batch


SUMMARY_SCHEMA = "manorl.device_controls_ablation.v3"
MODE_SCHEMA = "manorl.device_controls_ablation_mode.v3"
PROVENANCE_SCHEMA = "manorl.benchmark_source_provenance.v1"
MODE_CONFIGS = {
    "legacy_controls_diagnostics": (False, True),
    "legacy_controls_no_diagnostics": (False, False),
    "device_controls_diagnostics": (True, True),
    "device_controls_no_diagnostics": (True, False),
}
REQUIRED_RESULT_KEYS = {
    "mode",
    "device_resident_controls",
    "capture_transition_diagnostics",
    "repeat",
    "order_index",
    "device",
    "steps",
    "num_envs",
    "elapsed_seconds",
    "transitions_per_second",
    "reward_mean",
    "reset_count",
    "transition_diagnostics",
    "phase_profile",
    "contact_profile",
    "process_id",
}


def _mode_name(*, device_resident_controls: bool, capture_transition_diagnostics: bool) -> str:
    return (
        f"{'device' if device_resident_controls else 'legacy'}_controls_"
        f"{'diagnostics' if capture_transition_diagnostics else 'no_diagnostics'}"
    )


def _run_mode(
    args: argparse.Namespace,
    *,
    device_resident_controls: bool,
    capture_transition_diagnostics: bool,
    repeat: int,
    order_index: int,
) -> dict[str, Any]:
    contact_capacity = max(128, 31 * args.num_envs + 64)
    trajectories = load_assigned_trajectory_batch(
        TrajectorySelection(object_type="cube1", gesture="01"), num_envs=args.num_envs
    )
    environment = MujocoManoEnvironment(
        trajectories,
        EnvironmentConfig(
            num_envs=args.num_envs,
            device=args.device,
            contact_capacity=contact_capacity,
            device_resident_controls=device_resident_controls,
            capture_transition_diagnostics=capture_transition_diagnostics,
            profile_phases=args.profile_phases,
        ),
    )
    actions = np.linspace(-0.75, 0.75, args.num_envs * 26, dtype=np.float64).reshape(args.num_envs, 26)
    for _ in range(args.warmup_steps):
        environment.step(actions)
    environment.jax.block_until_ready(environment.data.qpos)
    environment.reset_phase_profile()

    started = time.perf_counter()
    rewards: list[float] = []
    resets = 0
    for _ in range(args.steps):
        _, reward, reset, _ = environment.step(actions)
        rewards.append(float(reward.mean()))
        resets += int(reset.sum())
    environment.jax.block_until_ready(environment.data.qpos)
    elapsed = time.perf_counter() - started
    result = {
        "mode": _mode_name(
            device_resident_controls=device_resident_controls,
            capture_transition_diagnostics=capture_transition_diagnostics,
        ),
        "device_resident_controls": device_resident_controls,
        "capture_transition_diagnostics": capture_transition_diagnostics,
        "repeat": repeat,
        "order_index": order_index,
        "device": str(environment.device),
        "steps": args.steps,
        "num_envs": args.num_envs,
        "elapsed_seconds": elapsed,
        "transitions_per_second": args.steps * args.num_envs / elapsed,
        "reward_mean": float(np.mean(rewards)),
        "reset_count": resets,
        "transition_diagnostics": environment.last_transition is not None,
        "process_id": os.getpid(),
        "phase_profile": environment.phase_profile() if args.profile_phases else {},
        "contact_profile": environment.contact_profile() if args.profile_phases else {},
    }
    # The parent launches one process per sample; process exit owns all JAX/Warp
    # allocations and caches before the next mode starts.
    return result


def _read_provenance(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"provenance artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError(
            f"provenance schema {payload.get('schema')!r} != {PROVENANCE_SCHEMA!r}: {path}"
        )
    for key in (
        "source_worktree",
        "source_commit",
        "source_diff_sha256",
        "mirror_path",
        "source_files",
        "mirror_file_hashes",
    ):
        if not payload.get(key):
            raise ValueError(f"provenance field {key!r} is missing or empty: {path}")
    if not isinstance(payload["source_files"], dict) or not isinstance(payload["mirror_file_hashes"], dict):
        raise ValueError("provenance source_files and mirror_file_hashes must be mappings")
    if payload["source_files"] != payload["mirror_file_hashes"]:
        raise ValueError("provenance source and mirror file hashes differ")
    return payload


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_mode_payload(payload: dict[str, Any], expected_provenance: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != MODE_SCHEMA:
        raise ValueError(f"child schema {payload.get('schema')!r} != {MODE_SCHEMA!r}")
    if payload.get("provenance") != expected_provenance:
        raise ValueError("child provenance does not match the requested source provenance")
    result = payload.get("result")
    if not isinstance(result, dict) or not REQUIRED_RESULT_KEYS <= set(result):
        raise ValueError("child result is missing required benchmark fields")
    if result["mode"] not in MODE_CONFIGS:
        raise ValueError(f"unknown child benchmark mode: {result.get('mode')!r}")
    if not isinstance(result["elapsed_seconds"], (int, float)) or result["elapsed_seconds"] <= 0:
        raise ValueError("child elapsed_seconds must be positive")
    return result


def _mode_arguments(args: argparse.Namespace) -> list[str]:
    arguments = [
        "--device", args.device,
        "--num-envs", str(args.num_envs),
        "--warmup-steps", str(args.warmup_steps),
        "--steps", str(args.steps),
        "--order-seed", str(args.order_seed),
    ]
    if args.profile_phases:
        arguments.append("--profile-phases")
    return arguments


def _run_child(
    args: argparse.Namespace,
    *,
    mode: str,
    repeat: int,
    order_index: int,
    output: Path,
    provenance: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    child_stdout = output.with_suffix(".stdout.log")
    child_stderr = output.with_suffix(".stderr.log")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *_mode_arguments(args),
        "--single-mode", mode,
        "--repeat", str(repeat),
        "--order-index", str(order_index),
        "--output", str(output),
        "--provenance", str(provenance),
    ]
    with child_stdout.open("w", encoding="utf-8") as stdout_file, child_stderr.open("w", encoding="utf-8") as stderr_file:
        completed = subprocess.run(command, stdout=stdout_file, stderr=stderr_file, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark child failed with exit code {completed.returncode}: {mode} repeat={repeat}; "
            f"stdout={child_stdout} stderr={child_stderr}"
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"benchmark child did not publish JSON artifact: {output}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    provenance_payload = _read_provenance(provenance)
    result = _validate_mode_payload(payload, provenance_payload)
    if result["mode"] != mode or result["repeat"] != repeat or result["order_index"] != order_index:
        raise ValueError(f"child metadata does not match invocation: {output}")
    artifacts = {"json": str(output), "stdout": str(child_stdout), "stderr": str(child_stderr)}
    if not all(Path(path).is_file() for path in artifacts.values()):
        raise RuntimeError(f"benchmark child artifacts are incomplete: {artifacts}")
    return result, artifacts


def _run_single(args: argparse.Namespace) -> int:
    if args.single_mode is None or args.output is None or args.provenance is None:
        raise ValueError("single-mode requires --output and --provenance")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace existing benchmark artifact: {args.output}")
    provenance = _read_provenance(args.provenance)
    device_resident_controls, capture_transition_diagnostics = MODE_CONFIGS[args.single_mode]
    result = _run_mode(
        args,
        device_resident_controls=device_resident_controls,
        capture_transition_diagnostics=capture_transition_diagnostics,
        repeat=args.repeat,
        order_index=args.order_index,
    )
    payload = {"schema": MODE_SCHEMA, "provenance": provenance, "result": result}
    _write_json_atomically(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _run_parent(args: argparse.Namespace) -> int:
    if args.output is None or args.provenance is None:
        raise ValueError("the parent harness requires --output and --provenance")
    if args.output.suffix != ".json":
        raise ValueError("--output must be a .json aggregate artifact")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace existing benchmark artifact: {args.output}")
    provenance = _read_provenance(args.provenance)
    modes = list(MODE_CONFIGS)
    order_rng = random.Random(args.order_seed)
    results: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []
    for repeat in range(args.repeats):
        order = list(modes)
        order_rng.shuffle(order)
        for order_index, mode in enumerate(order):
            child_output = args.output.with_name(
                f"{args.output.stem}.repeat{repeat:02d}.order{order_index:02d}.{mode}.json"
            )
            result, child_artifacts = _run_child(
                args,
                mode=mode,
                repeat=repeat,
                order_index=order_index,
                output=child_output,
                provenance=args.provenance,
            )
            results.append(result)
            artifacts.append(child_artifacts)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "provenance": provenance,
        "config": {
            "device": args.device,
            "num_envs": args.num_envs,
            "warmup_steps": args.warmup_steps,
            "steps": args.steps,
            "repeats": args.repeats,
            "order_seed": args.order_seed,
            "profile_phases": args.profile_phases,
        },
        "results": results,
        "artifacts": artifacts,
    }
    _write_json_atomically(args.output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--order-seed", type=int, default=42)
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument("--output", type=Path, help="aggregate or child JSON artifact; required for validated runs")
    parser.add_argument("--provenance", type=Path, help="source/mirror provenance JSON artifact")
    parser.add_argument("--single-mode", choices=tuple(MODE_CONFIGS), help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--order-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.num_envs < 1 or args.warmup_steps < 0 or args.steps < 1 or args.repeats < 1:
        parser.error("num-envs, steps, and repeats must be positive; warmup-steps must be non-negative")
    if args.repeat < 0 or args.order_index < 0:
        parser.error("repeat and order-index must be non-negative")
    if args.single_mode is not None:
        return _run_single(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
