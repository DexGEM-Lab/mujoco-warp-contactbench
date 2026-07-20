#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.sim_to_lance import main as sim_to_lance_main  # noqa: E402

from sim.dexhandrl.constants import DEFAULT_OUTPUT_ROOT  # noqa: E402

DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "mjx_warp_smoke.lance"


def check_mujoco() -> tuple[bool, str]:
    try:
        mujoco = importlib.import_module("mujoco")
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"mujoco unavailable: {exc}"
    version = getattr(mujoco, "__version__", "unknown")
    return True, f"mujoco available: {version}"


def check_jax_mjx_device(device: str) -> tuple[bool, str]:
    try:
        jax = importlib.import_module("jax")
        importlib.import_module("mujoco.mjx")
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"jax/mjx unavailable: {exc}"
    try:
        backend = jax.default_backend()
        devices = jax.devices()
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"jax imported but device query failed: {exc}"
    device_summary = ", ".join(f"{device.platform}:{device}" for device in devices)
    if backend != device:
        return False, f"expected JAX {device} backend, got {backend}; devices [{device_summary}]"
    return True, f"jax/mjx {device} available, devices [{device_summary}]"


def run_smoke_export(output: Path, device: str) -> int:
    argv = [
        "sim_to_lance.py",
        "--output",
        str(output),
        "--replace",
        "--device",
        device,
        "--scenario",
        "sparse_debug",
        "--ball-count",
        "8",
        "--duration-seconds",
        "0.2",
        "--settle-frames",
        "2",
        "--progress-interval",
        "10",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        return sim_to_lance_main()
    finally:
        sys.argv = old_argv


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the MJX-Warp sim-to-Lance pipeline.")
    parser.add_argument("--strict", action="store_true", help="Fail if MuJoCo, JAX, MJX, or requested MJX-Warp device export is unavailable.")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

    checks = [check_mujoco(), check_jax_mjx_device(args.device)]
    for ok, message in checks:
        print(f"{'ok' if ok else 'warn'}: {message}")
    if args.strict and not all(ok for ok, _ in checks):
        print("strict mode failed before export", file=sys.stderr)
        return 1

    result = run_smoke_export(args.output, args.device)
    if result != 0:
        return result
    if not args.output.exists():
        print(f"smoke export did not create {args.output}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "status": "ok"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
