#!/usr/bin/env python3
"""All-banana strict accepted-parent prefix-retreat production runner.

Every eligible banana source identity (263) is bound to its own accepted-parent
v3 descriptor. Each source targets five augmented episodes (1:5) with at most
twelve attempts. Prefix (far, 4 cm arc) and v4 retreat are mandatory for every
accepted row. Source identities are grouped by action and distributed across
Server1 GPUs; each worker launches one bounded exporter process per action
batch and appends accepted rows to its per-GPU output.

Output rows use a fresh episode-index namespace (5..9) and action-separated
seed bases so they cannot collide with the historical 1:5 dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from sim.manorl.synthetic_parent import load_accepted_synthetic_parent

DESCRIPTOR_ROOT = "/home/jay/data/manorl_banana_all_strict_parents_20260821"
PREDECODED_MANIFEST = "/home/jay/data/manorl_banana_pre60_vla_predecode/manifest.json"
CHECKPOINT = "/home/jay/data/manorl_banana_vla_checkpoint/checkpoint-001000.pt"
PLAN_CONTRACT = "manorl_banana_all_strict_prefix_retreat_plan_v1"
RUN_CONTRACT = "manorl_banana_all_strict_prefix_retreat_run_v1"

ACTIONS = ("01", "02", "03", "04", "09", "14", "18")


def _parents_for_action(action: str) -> list[dict[str, Any]]:
    by_action = json.loads(
        (Path(DESCRIPTOR_ROOT) / "parents_by_action.json").read_text(encoding="utf-8")
    )
    mapping = by_action.get("actions", {}).get(action, {})
    records = []
    for identity, descriptor_path in sorted(mapping.items()):
        parent = load_accepted_synthetic_parent(descriptor_path)
        records.append(
            {
                "source_identity": identity,
                "descriptor_path": str(Path(descriptor_path).resolve()),
                "checkpoint_sha256": parent.checkpoint_sha256,
                "reference_fps": parent.reference_fps,
            }
        )
    return records


def _plan() -> dict[str, Any]:
    parents = []
    for action in ACTIONS:
        parents.extend(_parents_for_action(action))
    return {
        "contract": PLAN_CONTRACT,
        "parents": len(parents),
        "episodes_per_identity": 5,
        "max_attempts_per_identity": 12,
        "episode_index_offset": 5,
        "parents_plan": parents,
    }


def _write_plan(output_dir: Path) -> Path:
    plan = _plan()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return plan_path


def _manifest_for_action(action: str, descriptor_root: Path) -> Path:
    records = _parents_for_action(action)
    manifest_path = descriptor_root / f"parents_{action}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "contract": "manorl_synthesis_accepted_parents_manifest_v1",
                "parents": {
                    record["source_identity"]: record["descriptor_path"]
                    for record in records
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _run_action_worker(
    *,
    action: str,
    gpu: int,
    output_dir: Path,
    descriptor_root: Path,
    seed: int,
    replace: bool,
) -> int:
    records = _parents_for_action(action)
    if not records:
        return 0
    manifest = _manifest_for_action(action, descriptor_root)
    output = output_dir / f"banana_action_{action}_prefix_retreat.lance"
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[1]
            / "tools"
            / "export_manorl_synthetic_lance.py"
        ),
        "--checkpoint",
        CHECKPOINT,
        "--output",
        str(output),
        "--object",
        "banana",
        "--gesture",
        action,
        "--dataset-path",
        "/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance",
        "--dataset-version",
        "295",
        "--num-envs",
        str(len(records)),
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
        PREDECODED_MANIFEST,
        "--approach-prefix",
        "--approach-mode",
        "far",
        "--accepted-parents-manifest",
        str(manifest),
        "--retreat-suffix",
        "--allow-partial-yield",
    ]
    if replace:
        command.append("--replace")
    env = dict(
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        CUDA_VISIBLE_DEVICES=str(gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    return subprocess.run(command, env=env, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--action-seed", type=int, default=420_000)
    parser.add_argument("--action", choices=ACTIONS)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--write-plan-only", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    descriptor_root = Path(DESCRIPTOR_ROOT).resolve()
    if args.write_plan_only:
        plan_path = _write_plan(output_dir)
        print(json.dumps({"plan": str(plan_path)}, indent=2))
        return 0

    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    if not gpus:
        parser.error("--gpus must name at least one GPU")
    actions = (args.action,) if args.action else ACTIONS
    per_gpu: dict[int, list[str]] = {gpu: [] for gpu in gpus}
    for index, action in enumerate(actions):
        per_gpu[gpus[index % len(gpus)]].append(action)

    plan_path = _write_plan(output_dir)
    results: list[dict[str, Any]] = []
    failed = False
    for gpu, action_list in per_gpu.items():
        for action in action_list:
            seed = args.action_seed + int(action)
            rc = _run_action_worker(
                action=action,
                gpu=gpu,
                output_dir=output_dir,
                descriptor_root=descriptor_root,
                seed=seed,
                replace=args.replace,
            )
            results.append({"gpu": gpu, "action": action, "exit_code": rc})
            if rc != 0:
                failed = True
    summary = {
        "contract": RUN_CONTRACT,
        "plan": str(plan_path),
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
