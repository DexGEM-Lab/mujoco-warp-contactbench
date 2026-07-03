#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import subprocess
import sys
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.contact_schema import read_contact_fixture, validate_contact_sequence  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "logs" / "mujoco_mjx_live_contact.json"


def check_mujoco() -> tuple[bool, str]:
    try:
        mujoco = importlib.import_module("mujoco")
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"mujoco unavailable: {exc}"
    version = getattr(mujoco, "__version__", "unknown")
    return True, f"mujoco available: {version}"


def check_jax_and_mjx() -> tuple[bool, str]:
    try:
        jax = importlib.import_module("jax")
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"jax unavailable: {exc}"

    try:
        devices = jax.devices()
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"jax imported but device query failed: {exc}"

    device_summary = ", ".join(f"{device.platform}:{device}" for device in devices)
    try:
        importlib.import_module("mujoco.mjx")
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"jax devices [{device_summary}], mujoco.mjx unavailable: {exc}"

    return True, f"jax/mjx available, devices [{device_summary}]"


def run_export(output: Path, *, synthetic: bool = False, steps: int = 3) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(BACKEND_DIR / "export_contacts.py"),
        "--output",
        str(output),
        "--steps",
        str(int(steps)),
    ]
    if synthetic:
        command.append("--synthetic")
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def validate_fixture(path: Path) -> tuple[int, dict[str, Any]]:
    payload = read_contact_fixture(path)
    count = validate_contact_sequence(payload["contact"])
    if count < 1:
        raise ValueError("fixture must contain at least one contact entry")
    pair_count = sum(
        len(entry["contact_pairs"])
        for frame in payload["contact"]
        for entry in frame
    )
    if pair_count < 1:
        raise ValueError("fixture must contain at least one contact pair")
    return count, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the MuJoCo/MJX backend scaffold.")
    parser.add_argument("--strict", action="store_true", help="Fail if mujoco or jax/mjx is unavailable.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=3, help="Live MuJoCo rollout steps.")
    args = parser.parse_args()

    mujoco_ok, mujoco_message = check_mujoco()
    mjx_ok, mjx_message = check_jax_and_mjx()
    for ok, message in ((mujoco_ok, mujoco_message), (mjx_ok, mjx_message)):
        prefix = "ok" if ok else "warn"
        print(f"{prefix}: {message}")

    if args.strict and not mujoco_ok:
        print("strict mode failed because MuJoCo is unavailable", file=sys.stderr)
        return 1

    synthetic = not args.strict and not mujoco_ok
    if synthetic:
        print("warn: using synthetic schema fixture because MuJoCo is unavailable outside Docker")
    result = run_export(args.output, synthetic=synthetic, steps=args.steps)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        print(f"export_contacts.py failed with code {result.returncode}", file=sys.stderr)
        return result.returncode

    try:
        count, payload = validate_fixture(args.output)
    except Exception as exc:
        print(f"fixture validation failed: {exc}", file=sys.stderr)
        return 1

    metadata = payload.get("metadata", {})
    source = metadata.get("source", "synthetic_schema_fixture")
    if args.strict and source != "mujoco_live_rollout_mj_contactForce":
        print(f"strict mode expected live MuJoCo data, got {source}", file=sys.stderr)
        return 1
    print(f"ok: validated {count} contact entries in {args.output}")
    print(f"ok: source {source}")
    print(f"ok: hand asset {metadata.get('hand_asset', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
