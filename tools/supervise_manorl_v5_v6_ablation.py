#!/usr/bin/env python3
"""Supervise the fixed ManoRL v5-to-v6 two-GPU training queue.

The supervisor never terminates GPU processes.  It launches a downstream job
only after every predecessor has written ``exit.status=0`` and one of the
allowed GPUs has no compute process and less than the configured idle-memory
threshold.  Each training process remains independently inspectable in tmux.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable


class JobState(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BROKEN = "broken"


@dataclass(frozen=True)
class QueueConfig:
    deployment: Path
    output_root: Path
    package: Path
    python: Path
    dexstream_root: Path
    poll_seconds: float
    idle_memory_limit_mib: int
    num_envs: int = 2048
    updates: int = 1000
    rollouts: int = 32
    learning_epochs: int = 4
    mini_batches: int = 16
    learning_rate: float = 3e-5
    checkpoint_interval: int = 50
    identity: str = "cube2_02_2833"

    @property
    def total_transitions(self) -> int:
        return self.num_envs * self.updates * self.rollouts


@dataclass(frozen=True)
class Job:
    key: str
    policy_version: str
    output_name: str
    session_name: str
    predecessors: tuple[str, ...]
    preferred_gpus: tuple[int, ...]
    externally_started: bool = False


@dataclass(frozen=True)
class GpuState:
    index: int
    uuid: str
    memory_used_mib: int
    compute_pids: tuple[int, ...]

    def is_idle(self, memory_limit_mib: int) -> bool:
        return not self.compute_pids and self.memory_used_mib < memory_limit_mib


JOBS: tuple[Job, ...] = (
    Job(
        key="v5",
        policy_version="v5",
        output_name="v5-formal",
        session_name="manorl-v5-formal",
        predecessors=(),
        preferred_gpus=(0,),
        externally_started=True,
    ),
    Job(
        key="v525",
        policy_version="v5.25",
        output_name="v525-formal",
        session_name="manorl-v525-formal",
        predecessors=(),
        preferred_gpus=(1,),
        externally_started=True,
    ),
    Job(
        key="v55",
        policy_version="v5.5",
        output_name="v55-formal",
        session_name="manorl-v55-formal",
        predecessors=("v5",),
        preferred_gpus=(0,),
    ),
    Job(
        key="v575",
        policy_version="v5.75",
        output_name="v575-formal",
        session_name="manorl-v575-formal",
        predecessors=("v525",),
        preferred_gpus=(1,),
    ),
    Job(
        key="v6",
        policy_version="v6",
        output_name="v6-formal",
        session_name="manorl-v6-formal",
        predecessors=("v55", "v575"),
        preferred_gpus=(0, 1),
    ),
)
JOB_BY_KEY = {job.key: job for job in JOBS}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_checked(argv: Iterable[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=True,
        text=True,
        capture_output=capture_output,
    )


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_event(config: QueueConfig, event: str, **fields: object) -> None:
    path = config.output_root / "queue.events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": utc_now(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def output_dir(config: QueueConfig, job: Job) -> Path:
    return config.output_root / job.output_name


def read_exit_code(config: QueueConfig, job: Job) -> int | None:
    path = output_dir(config, job) / "exit.status"
    if not path.exists():
        return None
    text = path.read_text().strip()
    try:
        return int(text)
    except ValueError as error:
        raise RuntimeError(f"invalid exit status for {job.key}: {text!r}") from error


def tmux_session_exists(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def inspect_job(config: QueueConfig, job: Job) -> JobState:
    exit_code = read_exit_code(config, job)
    if exit_code is not None:
        return JobState.SUCCEEDED if exit_code == 0 else JobState.FAILED
    if tmux_session_exists(job.session_name):
        return JobState.RUNNING
    run_dir = output_dir(config, job)
    if (run_dir / "launch.json").exists() or (run_dir / "policy.pt.metrics.jsonl").exists():
        return JobState.BROKEN
    return JobState.WAITING


def inspect_gpus() -> dict[int, GpuState]:
    gpu_rows = run_checked(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
    ).stdout.splitlines()
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    pids_by_uuid: dict[str, list[int]] = {}
    if processes.returncode == 0:
        for row in processes.stdout.splitlines():
            if not row.strip():
                continue
            uuid, pid = (part.strip() for part in row.split(",", maxsplit=1))
            pids_by_uuid.setdefault(uuid, []).append(int(pid))

    result: dict[int, GpuState] = {}
    for row in gpu_rows:
        index_text, uuid, memory_text = (part.strip() for part in row.split(",", maxsplit=2))
        index = int(index_text)
        result[index] = GpuState(
            index=index,
            uuid=uuid,
            memory_used_mib=int(memory_text),
            compute_pids=tuple(sorted(pids_by_uuid.get(uuid, []))),
        )
    return result


def validate_config(config: QueueConfig) -> None:
    required = (
        config.deployment / "tools" / "train_manorl_autonomy.py",
        config.package,
        config.python,
        config.dexstream_root,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required queue inputs are missing: " + ", ".join(missing))
    if config.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if config.idle_memory_limit_mib <= 0:
        raise ValueError("idle-memory-limit-mib must be positive")
    if config.total_transitions != 65_536_000:
        raise ValueError("the fixed experiment contract requires 65,536,000 transitions")


def deployment_commit(config: QueueConfig) -> str:
    marker = config.deployment / "DEPLOYED_COMMIT"
    if marker.exists():
        return marker.read_text().strip()
    result = run_checked(
        ["git", "-C", str(config.deployment), "rev-parse", "HEAD"],
        capture_output=True,
    )
    return result.stdout.strip()


def training_argv(config: QueueConfig, job: Job, checkpoint: Path) -> list[str]:
    return [
        str(config.python),
        "tools/train_manorl_autonomy.py",
        "train",
        "--package",
        str(config.package),
        "--identity",
        config.identity,
        "--device",
        "gpu",
        "--policy-version",
        job.policy_version,
        "--num-envs",
        str(config.num_envs),
        "--updates",
        str(config.updates),
        "--rollouts",
        str(config.rollouts),
        "--learning-epochs",
        str(config.learning_epochs),
        "--mini-batches",
        str(config.mini_batches),
        "--learning-rate",
        str(config.learning_rate),
        "--total-transitions",
        str(config.total_transitions),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-interval",
        str(config.checkpoint_interval),
        "--all-references",
        "--teacher-anchor-beta",
        "1",
        "--teacher-anchor-passes",
        "2",
        "--separate-critic",
        "--wandb",
        "--wandb-mode",
        "offline",
    ]


def choose_idle_gpu(config: QueueConfig, job: Job, gpus: dict[int, GpuState]) -> int | None:
    for index in job.preferred_gpus:
        state = gpus.get(index)
        if state is None:
            raise RuntimeError(f"GPU {index} is not visible")
        if state.is_idle(config.idle_memory_limit_mib):
            return index
    return None


def launch_job(config: QueueConfig, job: Job, gpu_index: int) -> None:
    if job.externally_started:
        raise ValueError(f"{job.key} is externally started and cannot be launched by this supervisor")
    run_dir = output_dir(config, job)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "policy.pt"
    exit_status = run_dir / "exit.status"
    log_path = run_dir / "train.log"
    launch_path = run_dir / "launch.json"
    if any(path.exists() for path in (exit_status, checkpoint, launch_path)):
        raise RuntimeError(f"refusing to overwrite existing run state in {run_dir}")
    if tmux_session_exists(job.session_name):
        raise RuntimeError(f"tmux session already exists: {job.session_name}")

    argv = training_argv(config, job, checkpoint)
    metadata: dict[str, object] = {
        "state": "requested",
        "requested_at_utc": utc_now(),
        "job": asdict(job),
        "gpu_index": gpu_index,
        "deployment": str(config.deployment),
        "deployed_commit": deployment_commit(config),
        "package": str(config.package),
        "checkpoint": str(checkpoint),
        "argv": argv,
        "protocol": {
            "num_envs": config.num_envs,
            "updates": config.updates,
            "rollouts": config.rollouts,
            "learning_epochs": config.learning_epochs,
            "mini_batches": config.mini_batches,
            "learning_rate": config.learning_rate,
            "total_transitions": config.total_transitions,
            "checkpoint_interval": config.checkpoint_interval,
            "all_references": True,
            "separate_critic": True,
            "teacher_anchor_beta": 1,
            "teacher_anchor_passes": 2,
            "wandb_mode": "offline",
        },
    }
    atomic_write_json(launch_path, metadata)

    temporary_status = run_dir / ".exit.status.tmp"
    environment = {
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MANORL_DEXSTREAM_ROOT": str(config.dexstream_root),
        "WANDB_DIR": str(run_dir / "wandb"),
        "WANDB_MODE": "offline",
        "PYTHONUNBUFFERED": "1",
    }
    exports = " ".join(f"export {key}={shlex.quote(value)};" for key, value in environment.items())
    command = " ".join(
        (
            "set +e;",
            f"cd {shlex.quote(str(config.deployment))};",
            exports,
            f"{shlex.join(argv)} >> {shlex.quote(str(log_path))} 2>&1;",
            "run_code=$?;",
            f"printf '%s\\n' \"$run_code\" > {shlex.quote(str(temporary_status))};",
            f"mv {shlex.quote(str(temporary_status))} {shlex.quote(str(exit_status))};",
            'exit "$run_code"',
        )
    )
    try:
        run_checked(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                job.session_name,
                "bash",
                "-lc",
                command,
            ]
        )
    except Exception:
        metadata["state"] = "launch_failed"
        metadata["launch_failed_at_utc"] = utc_now()
        atomic_write_json(launch_path, metadata)
        raise

    metadata["state"] = "running"
    metadata["launched_at_utc"] = utc_now()
    atomic_write_json(launch_path, metadata)
    append_event(config, "job_launched", job=job.key, gpu_index=gpu_index)


def status_payload(
    config: QueueConfig,
    states: dict[str, JobState],
    gpus: dict[int, GpuState],
) -> dict[str, object]:
    return {
        "updated_at_utc": utc_now(),
        "jobs": {key: state.value for key, state in states.items()},
        "gpus": {
            str(index): {
                "memory_used_mib": state.memory_used_mib,
                "compute_pids": list(state.compute_pids),
                "idle_for_queue": state.is_idle(config.idle_memory_limit_mib),
            }
            for index, state in sorted(gpus.items())
            if index in (0, 1)
        },
    }


def supervise(config: QueueConfig, *, once: bool) -> int:
    validate_config(config)
    append_event(
        config,
        "supervisor_started",
        pid=os.getpid(),
        poll_seconds=config.poll_seconds,
        idle_memory_limit_mib=config.idle_memory_limit_mib,
    )
    last_wait_signature: tuple[object, ...] | None = None
    while True:
        states = {job.key: inspect_job(config, job) for job in JOBS}
        gpus = inspect_gpus()
        atomic_write_json(config.output_root / "queue.status.json", status_payload(config, states, gpus))

        failed = [key for key, state in states.items() if state in (JobState.FAILED, JobState.BROKEN)]
        if failed:
            append_event(
                config,
                "queue_failed",
                failed_jobs=failed,
                states={key: value.value for key, value in states.items()},
            )
            return 1

        if states["v6"] is JobState.SUCCEEDED:
            atomic_write_json(
                config.output_root / "queue.completed.json",
                {"completed_at_utc": utc_now(), "states": {key: value.value for key, value in states.items()}},
            )
            append_event(config, "queue_completed")
            return 0

        launched = False
        for job in JOBS:
            if job.externally_started or states[job.key] is not JobState.WAITING:
                continue
            if not all(states[predecessor] is JobState.SUCCEEDED for predecessor in job.predecessors):
                continue
            gpu_index = choose_idle_gpu(config, job, gpus)
            if gpu_index is None:
                continue
            launch_job(config, job, gpu_index)
            launched = True
            break

        wait_signature = (
            tuple((key, value.value) for key, value in states.items()),
            tuple(
                (index, state.memory_used_mib, state.compute_pids)
                for index, state in sorted(gpus.items())
                if index in (0, 1)
            ),
        )
        if not launched and wait_signature != last_wait_signature:
            append_event(
                config,
                "queue_waiting",
                states={key: value.value for key, value in states.items()},
                gpus={
                    str(index): {
                        "memory_used_mib": state.memory_used_mib,
                        "compute_pids": list(state.compute_pids),
                    }
                    for index, state in sorted(gpus.items())
                    if index in (0, 1)
                },
            )
            last_wait_signature = wait_signature

        if once:
            return 0
        time.sleep(config.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment",
        type=Path,
        default=Path("/mnt/nas-222-projects/cty/manorl-autonomy/deployments/v5-v6-ablation-20260919"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/nas-222-projects/cty/manorl-autonomy/outputs/v5-v6-ablation-20260919"),
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("/mnt/nas-222-projects/cty/manorl-autonomy/data/cube2_02_v295_f120_pre180_post180"),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/cty/miniconda3/envs/contact-conditioned-autonomy/bin/python"),
    )
    parser.add_argument(
        "--dexstream-root",
        type=Path,
        default=Path(
            "/mnt/user-home/cty/workplace/"
            "mujoco-warp-contactbench-contact-conditioned-autonomy/assets/dexstream_digital_assets"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--idle-memory-limit-mib",
        type=int,
        default=1024,
        help="A GPU is launchable only below this memory use and with zero compute processes.",
    )
    parser.add_argument("--once", action="store_true", help="Inspect once, launch at most one ready job, and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = QueueConfig(
        deployment=args.deployment.resolve(),
        output_root=args.output_root.resolve(),
        package=args.package.resolve(),
        python=args.python.resolve(),
        dexstream_root=args.dexstream_root.resolve(),
        poll_seconds=args.poll_seconds,
        idle_memory_limit_mib=args.idle_memory_limit_mib,
    )
    return supervise(config, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
