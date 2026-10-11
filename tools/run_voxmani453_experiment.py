#!/usr/bin/env python3
"""Run real physics/PPO, strict resume and frozen evaluation on an idle GPU."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def nvidia_rows(query: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        [cell.strip() for cell in row]
        for row in csv.reader(io.StringIO(result.stdout))
        if row
    ]


def idle_gpus() -> list[tuple[str, str]]:
    """A zero utilization reading alone does not establish availability."""
    occupied = {row[0] for row in nvidia_rows("--query-compute-apps=gpu_uuid,pid")}
    return [
        (index, uuid)
        for index, uuid, used in nvidia_rows("--query-gpu=index,uuid,memory.used")
        if uuid not in occupied and int(used) < 512
    ]


def save_status(output: Path, **fields) -> None:
    target = output / "status.json"
    temporary = output / "status.json.tmp"
    temporary.write_text(json.dumps({"time": time.time(), **fields}, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(fields), flush=True)


def acquire_idle_gpu(output: Path, wait_seconds: int):
    deadline = time.monotonic() + wait_seconds
    while True:
        for index, uuid in idle_gpus():
            lock = open(f"/tmp/cty-voxmani453-{uuid}.lock", "a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            # Recheck after taking the per-GPU lock and before any CUDA import.
            if (index, uuid) in idle_gpus():
                return index, uuid, lock
            lock.close()
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "No idle GPU: all have compute processes or >=512 MiB in use"
            )
        save_status(output, stage="waiting_for_idle_gpu", pid=os.getpid())
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def run_stage(stage: str, argv: list[str], output: Path, env: dict, gpu: dict) -> None:
    log = output / f"{stage}.log"
    save_status(output, stage=stage, gpu=gpu, command=argv, pid=os.getpid())
    with log.open("w") as stream:
        subprocess.run(
            argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_results(output: Path, args) -> dict:
    import numpy as np
    import torch
    from sim.manorl.autonomy_contracts import V11_REWARD_CONTRACT_ID

    def load(folder: str):
        path = output / folder / "policy.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["model_architecture"]["raw_observation_dim"] == 453
        assert payload["model_architecture"]["action_dim"] == 28
        assert payload["reward_contract"] == V11_REWARD_CONTRACT_ID
        assert payload["optimizer"]["state"]
        assert all(
            torch.isfinite(t).all()
            for t in payload["model"].values()
            if t.is_floating_point()
        )
        return payload

    initial = load("train")
    resumed = load("resume")
    assert initial["update"] == args.updates
    assert resumed["update"] == args.updates + args.resume_updates
    assert (
        resumed["environment_transitions"]
        == (args.updates + args.resume_updates) * args.rollouts * args.num_envs
    )
    changed = [
        name
        for name, tensor in initial["model"].items()
        if tensor.is_floating_point()
        and not torch.equal(tensor, resumed["model"][name])
    ]
    assert any(name.startswith("actor.") for name in changed)
    assert any(name.startswith("critic.") for name in changed)
    metric_rows = []
    for folder in ("train", "resume"):
        rows = [
            json.loads(line)
            for line in (output / folder / "policy.pt.metrics.jsonl")
            .read_text()
            .splitlines()
        ]
        assert rows and all(
            row["valid"] == 1 and all(np.isfinite(v) for v in row.values())
            for row in rows
        )
        metric_rows.extend(rows)
    assert len(metric_rows) == args.updates + args.resume_updates
    reset = json.loads((output / "runtime_reset.json").read_text())
    assert (
        reset["subset_reset_preserves_other_world"] and reset["all_observations_finite"]
    )
    trace = json.loads((output / "evaluation.json").read_text())
    assert trace["steps"] > 0 and all(
        row["finite"] and row["valid"] for row in trace["trace"]
    )
    with np.load(output / "evaluation.npz") as artifact:
        assert artifact["actual_qpos"].shape == (trace["steps"], 28)
        assert artifact["paired_force_on_object"].shape == (trace["steps"], 16, 3)
        assert np.isfinite(artifact["actual_object_position"]).all()
    return {
        "training_updates": initial["update"],
        "resumed_updates": resumed["update"],
        "transitions": resumed["environment_transitions"],
        "changed_tensors_on_resume": len(changed),
        "evaluation_steps": trace["steps"],
        "evaluation_return": trace["return"],
        "natural_first_termination": trace["natural_first_termination"],
        "training_checkpoint_sha256": sha256(output / "train" / "policy.pt"),
        "resumed_checkpoint_sha256": sha256(output / "resume" / "policy.pt"),
        "boundary": "execution validation; task success and policy convergence are not established",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package", default=str(ROOT / "data/cube2_all_v295_f120_pre180_post180")
    )
    parser.add_argument("--identity", default="cube2_02_2833")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--resume-updates", type=int, default=2)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    if (
        min(args.num_envs, args.rollouts, args.updates, args.resume_updates) <= 0
        or args.wait_seconds < 0
    ):
        parser.error("counts must be positive and wait-seconds nonnegative")
    if args.num_envs * args.rollouts % 4:
        parser.error("num-envs * rollouts must be divisible by four minibatches")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "train/policy.pt.metrics.jsonl").exists():
        parser.error("output already contains an experiment; use a new directory")
    gpu = {}
    lock = None
    try:
        index, uuid, lock = acquire_idle_gpu(output, args.wait_seconds)
        gpu = {"physical_index": index, "uuid": uuid}
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": uuid,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": "4",
            "PYTHONUNBUFFERED": "1",
        }
        common = [
            sys.executable,
            "-u",
            str(ROOT / "tools/train_manorl_autonomy.py"),
            "train",
            "--policy-version",
            "v5.voxmani453",
            "--reward-version",
            "v11",
            "--package",
            args.package,
            "--identity",
            args.identity,
            "--device",
            "gpu",
            "--num-envs",
            str(args.num_envs),
            "--rollouts",
            str(args.rollouts),
            "--learning-epochs",
            "1",
            "--mini-batches",
            "4",
            "--learning-rate",
            "1e-5",
            "--teacher-anchor-beta",
            "0.2",
            "--teacher-anchor-passes",
            "1",
            "--checkpoint-interval",
            str(max(1, args.updates // 2)),
            "--no-wandb",
        ]
        train_path, resume_path = (
            output / "train/policy.pt",
            output / "resume/policy.pt",
        )
        run_stage(
            "runtime_reset",
            [
                sys.executable,
                "-u",
                str(ROOT / "tools/validate_voxmani453_runtime.py"),
                "--package",
                args.package,
                "--identity",
                args.identity,
                "--device",
                "gpu",
                "--output",
                str(output / "runtime_reset.json"),
            ],
            output,
            env,
            gpu,
        )
        run_stage(
            "train",
            common + ["--updates", str(args.updates), "--checkpoint", str(train_path)],
            output,
            env,
            gpu,
        )
        run_stage(
            "resume",
            common
            + [
                "--updates",
                str(args.updates + args.resume_updates),
                "--resume-checkpoint",
                str(train_path),
                "--checkpoint",
                str(resume_path),
            ],
            output,
            env,
            gpu,
        )
        frozen_hash = sha256(resume_path)
        run_stage(
            "evaluate",
            [
                sys.executable,
                "-u",
                str(ROOT / "tools/train_manorl_autonomy.py"),
                "evaluate",
                "--package",
                args.package,
                "--identity",
                args.identity,
                "--device",
                "gpu",
                "--checkpoint",
                str(resume_path),
                "--trace",
                str(output / "evaluation.json"),
                "--artifact",
                str(output / "evaluation.npz"),
            ],
            output,
            env,
            gpu,
        )
        if frozen_hash != sha256(resume_path):
            raise RuntimeError("frozen evaluation changed the checkpoint")
        result = verify_results(output, args)
        (output / "validation.json").write_text(
            json.dumps({"gpu": gpu, **result}, indent=2) + "\n"
        )
        save_status(output, stage="complete", gpu=gpu, **result)
        return 0
    except Exception as exc:
        save_status(output, stage="failed", gpu=gpu, error=str(exc))
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
