#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "sim"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sim.dexhandrl.constants import DEFAULT_OUTPUT_ROOT  # noqa: E402
from sim.warp_contact_export import _export_warp_contacts  # noqa: E402
from tools.export_contactbench_lance import write_payloads_to_lance  # noqa: E402

DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "mjx_warp_contactbench_generated.lance"
DEFAULT_ERROR_OUTPUT = DEFAULT_OUTPUT_ROOT / "mjx_warp_contact_to_hand_mesh_error_raw.json"
DEFAULT_SCENE_COPY = DEFAULT_OUTPUT_ROOT / "mjx_warp_contact_error_scene.xml"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MJX-Warp simulation and write Lance directly.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu", help="MJX-Warp execution device.")
    parser.add_argument("--scenario", default="filled_tank", choices=("sparse_debug", "filled_tank"))
    parser.add_argument("--ball-count", type=int, default=120)
    parser.add_argument("--ball-radius", type=float, default=0.03)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--settle-frames", type=int, default=60)
    parser.add_argument("--rollout-frames", type=int, default=1000)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--naconmax", type=int, default=8192)
    parser.add_argument("--njmax", type=int, default=8192)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--error-output", type=Path, default=DEFAULT_ERROR_OUTPUT)
    parser.add_argument("--scene-copy", type=Path, default=DEFAULT_SCENE_COPY)
    parser.add_argument("--keep-debug-json", action="store_true")
    args = parser.parse_args()

    if args.replace and args.output.exists():
        shutil.rmtree(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.error_output.parent.mkdir(parents=True, exist_ok=True)
    args.scene_copy.parent.mkdir(parents=True, exist_ok=True)

    native_payload, error_payload, contactbench_payload = _export_warp_contacts(args)
    if args.keep_debug_json:
        _write_json(args.error_output, error_payload)

    result = write_payloads_to_lance(
        [contactbench_payload],
        output=args.output,
        replace=False,
        processes=args.processes,
        source_names=[f"mjx_warp_{args.device}_sim"],
    )
    result["simulation"] = {
        "backend": "mjx_warp",
        "device": args.device,
        "metadata": contactbench_payload["metadata"],
        "error_output": str(args.error_output) if args.keep_debug_json else None,
        "scene_copy": str(args.scene_copy),
        "native_contacts_total": sum(len(frame) for frame in native_payload["contact"]),
        "distance_to_hand_mesh": error_payload["distance_to_hand_mesh"],
    }
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
