#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "docker/mujoco_mjx"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.ball_pit.common import spec_from_args  # noqa: E402
from docker.mujoco_mjx.ball_pit_contact import run_ball_pit  # noqa: E402
from docker.mujoco_mjx.mjx_warp_contact_export import _export_gpu_contacts  # noqa: E402
from tools.export_contactbench_lance import write_payloads_to_lance  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "logs/mujoco_cpu_mjx_warp_gpu_ball_pit_generated.lance"
DEFAULT_CPU_IMAGE = REPO_ROOT / "logs/mujoco_cpu_ball_pit_contact_10s.svg"
DEFAULT_GPU_ERROR_OUTPUT = REPO_ROOT / "logs/mjx_warp_gpu_contact_to_hand_mesh_error_raw.json"
DEFAULT_GPU_SCENE_COPY = REPO_ROOT / "logs/mjx_warp_gpu_contact_error_scene.xml"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MuJoCo/MJX simulation and write Lance directly without contact JSON intermediates.")
    parser.add_argument("--backend", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--scenario", default="filled_tank", choices=("sparse_debug", "filled_tank"))
    parser.add_argument("--ball-count", type=int, default=120)
    parser.add_argument("--ball-radius", type=float, default=0.03)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--settle-frames", type=int, default=60)
    parser.add_argument("--rollout-frames", type=int, default=1000)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--cpu-image", type=Path, default=DEFAULT_CPU_IMAGE)
    parser.add_argument("--no-cpu-image", action="store_true")
    parser.add_argument("--naconmax", type=int, default=8192)
    parser.add_argument("--njmax", type=int, default=8192)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--gpu-error-output", type=Path, default=DEFAULT_GPU_ERROR_OUTPUT)
    parser.add_argument("--gpu-scene-copy", type=Path, default=DEFAULT_GPU_SCENE_COPY)
    parser.add_argument("--keep-gpu-debug-json", action="store_true")
    args = parser.parse_args()

    if args.replace and args.output.exists():
        shutil.rmtree(args.output)

    payloads: list[dict[str, Any]] = []
    source_names: list[str] = []
    summaries: list[dict[str, Any]] = []

    if args.backend in {"cpu", "both"}:
        cpu_result = run_ball_pit(
            spec_from_args(args),
            output=None,
            image=None if args.no_cpu_image else args.cpu_image,
            video=None,
        )
        payloads.append(cpu_result["payload"])
        source_names.append("mujoco_cpu_sim")
        summaries.append({
            "backend": "cpu",
            "metadata": cpu_result["metadata"],
            "image": cpu_result["image"],
        })

    if args.backend in {"gpu", "both"}:
        args.scene_copy = args.gpu_scene_copy
        native_payload, error_payload, contactbench_payload = _export_gpu_contacts(args)
        if args.keep_gpu_debug_json:
            _write_json(args.gpu_error_output, error_payload)
        payloads.append(contactbench_payload)
        source_names.append("mjx_warp_gpu_sim")
        summaries.append({
            "backend": "gpu",
            "metadata": contactbench_payload["metadata"],
            "error_output": str(args.gpu_error_output) if args.keep_gpu_debug_json else None,
            "scene_copy": str(args.gpu_scene_copy),
            "native_contacts_total": sum(len(frame) for frame in native_payload["contact"]),
            "distance_to_hand_mesh": error_payload["distance_to_hand_mesh"],
        })

    result = write_payloads_to_lance(
        payloads,
        output=args.output,
        replace=False,
        processes=args.processes,
        source_names=source_names,
    )
    result["simulations"] = summaries
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
