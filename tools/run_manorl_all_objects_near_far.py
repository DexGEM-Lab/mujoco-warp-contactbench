#!/usr/bin/env python3
"""Default all-object pre60 Far/Near approach-prefix-only production runner.

Every verified accepted parent targets five Far and five Near episodes with at
most twelve attempts per mode. Both modes prepend the established 4 cm approach
arc and preserve the complete original reference tail. Near uses movement-end+15
only to sample its start; this runner never requests a retreat suffix.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from sim.manorl.approach_prefix import APPROACH_PREFIX_ONLY_PRODUCTION_CONTRACT

PARENT_ROOT = Path("/home/jay/data/manorl_all75_parents_20260821")
PREDECODED_MANIFEST = Path("/mnt/user-home/jay/data/manorl_all75_pre60_vla_predecode/manifest.json")
CHECKPOINT = Path("/home/jay/data/manorl_banana_vla_checkpoint/checkpoint-001000.pt")
DATASET = Path(
    "/mnt/nas-222-projects/mocap_v2/lance_datasets/human_p1_guangguan/"
    "human_p1_guangguan_clean.lance"
)
OBJECTS = (
    "banana", "bowl", "cube1", "cube2", "cylinder2", "cylinder3", "cylinder4",
    "cylinder7", "iphone", "largeclamp", "mayonnaisebottle", "powerdrill",
    "scissor",
)
SEED_BASE = 420_000
RUN_CONTRACT = "manorl_all_objects_pre60_far_near_prefix_only_run_v2"


def _verified_sources(object_name: str) -> list[str]:
    manifest = json.loads(
        (PARENT_ROOT / object_name / "manifest.json").read_text(encoding="utf-8")
    )
    return sorted(Path(path).name.removesuffix(".json") for path in manifest["descriptors"])


def _write_parents_manifest(object_name: str, output_dir: Path) -> Path:
    by_action = json.loads(
        (PARENT_ROOT / object_name / "parents_by_action.json").read_text(encoding="utf-8")
    )
    parents = {}
    for action, mapping in by_action["actions"].items():
        parents.update(mapping)
    manifest = output_dir / f"parents_{object_name}.json"
    manifest.write_text(
        json.dumps(
            {
                "contract": "manorl_synthesis_accepted_parents_manifest_v1",
                "parents": parents,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _run_object_mode(
    *,
    object_name: str,
    mode: str,
    gpu: int,
    parents_manifest: Path,
    pairs: str,
    n_env: int,
    output: Path,
    seed: int,
    replace: bool = False,
) -> int:
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[1]
            / "tools"
            / "export_manorl_synthetic_lance.py"
        ),
        "--checkpoint",
        str(CHECKPOINT),
        "--output",
        str(output),
        "--object",
        object_name,
        "--gesture",
        "01",
        "--pairs",
        pairs,
        "--dataset-path",
        str(DATASET),
        "--dataset-version",
        "295",
        "--num-envs",
        str(n_env),
        "--device",
        "gpu",
        "--output-format",
        "compact-replay-visual",
        "--seed",
        str(seed),
        "--episodes-per-identity",
        "5",
        "--max-attempts-per-identity",
        "12",
        "--predecoded-manifest",
        str(PREDECODED_MANIFEST),
        "--approach-prefix",
        "--approach-mode",
        mode,
        "--accepted-parents-manifest",
        str(parents_manifest),
        "--allow-partial-yield",
    ]
    if replace:
        command.append("--replace")
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    return subprocess.run(command, env=env, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--objects", help="comma-separated subset")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = (
        tuple(args.objects.split(",")) if args.objects else OBJECTS
    )
    gpus = [int(gpu) for gpu in args.gpus.split(",")]

    # 每物体 verified 数
    per_object = {obj: _verified_sources(obj) for obj in objects}
    for obj, sources in per_object.items():
        print(f"{obj}: {len(sources)} verified sources", flush=True)

    # 按 source 数轮转分配 GPU（大物体先分）
    order = sorted(objects, key=lambda obj: -len(per_object[obj]))
    gpu_objects: dict[int, list[str]] = {gpu: [] for gpu in gpus}
    for index, obj in enumerate(order):
        gpu_objects[gpus[index % len(gpus)]].append(obj)

    import threading

    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    failed = False
    failed_lock = threading.Lock()

    def record_result(value: dict[str, Any]) -> None:
        nonlocal failed
        with results_lock:
            results.append(value)
            print(json.dumps(value, sort_keys=True), flush=True)
        if value.get("exit_code") != 0:
            with failed_lock:
                failed = True

    def gpu_worker(gpu: int, object_list: list[str]) -> None:
        for obj in object_list:
            try:
                sources = per_object[obj]
                n_env = len(sources)
                parents_manifest = _write_parents_manifest(obj, output_dir)
                by_action = json.loads(
                    (PARENT_ROOT / obj / "parents_by_action.json").read_text(
                        encoding="utf-8"
                    )
                )
                pairs = ",".join(
                    f"{obj}:{action}" for action in sorted(by_action["actions"])
                )
                for mode in ("far", "near"):
                    output = output_dir / f"{obj}_{mode}.lance"
                    seed = SEED_BASE + (1 if mode == "near" else 0)
                    rc = _run_object_mode(
                        object_name=obj,
                        mode=mode,
                        gpu=gpu,
                        parents_manifest=parents_manifest,
                        pairs=pairs,
                        n_env=n_env,
                        output=output,
                        seed=seed,
                        replace=args.replace,
                    )
                    record_result(
                        {"gpu": gpu, "object": obj, "mode": mode, "exit_code": rc}
                    )
            except Exception as exc:
                record_result(
                    {
                        "gpu": gpu,
                        "object": obj,
                        "mode": "setup",
                        "exit_code": 1,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    threads = [
        threading.Thread(target=gpu_worker, args=(gpu, object_list), daemon=True)
        for gpu, object_list in gpu_objects.items()
        if object_list
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    results.sort(key=lambda item: (item["gpu"], item["object"], item["mode"]))
    summary = {
        "contract": RUN_CONTRACT,
        "production_contract": APPROACH_PREFIX_ONLY_PRODUCTION_CONTRACT,
        "predecoded_manifest": str(PREDECODED_MANIFEST),
        "retreat_suffix": None,
        "episodes_per_identity_per_mode": 5,
        "max_attempts_per_identity_per_mode": 12,
        "output_dir": str(output_dir),
        "results": results,
        "failed": failed,
    }
    (output_dir / "run-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
