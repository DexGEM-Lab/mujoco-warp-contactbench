#!/usr/bin/env python3
"""Evaluate and render completed ManoRL v5-to-v6 training runs on CPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from summarize_manorl_autonomy_evaluation import (  # noqa: E402
    EvaluationSummary,
    checkpoint_update,
    summarize_evaluation,
    write_summary,
)


class TrainingState(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BROKEN = "broken"


@dataclass(frozen=True)
class Experiment:
    key: str
    policy_version: str
    output_name: str
    session_name: str


@dataclass(frozen=True)
class EvaluationConfig:
    deployment: Path
    output_root: Path
    package: Path
    python: Path
    dexstream_root: Path
    poll_seconds: float
    identity: str = "cube2_02_2833"
    checkpoint_interval: int = 50
    final_update: int = 1000


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("v5", "v5", "v5-formal", "manorl-v5-formal"),
    Experiment("v525", "v5.25", "v525-formal", "manorl-v525-formal"),
    Experiment("v55", "v5.5", "v55-formal", "manorl-v55-formal"),
    Experiment("v575", "v5.75", "v575-formal", "manorl-v575-formal"),
    Experiment("v6", "v6", "v6-formal", "manorl-v6-formal"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_event(config: EvaluationConfig, event: str, **fields: object) -> None:
    path = config.output_root / "evaluation-queue.events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"timestamp_utc": utc_now(), "event": event, **fields},
                sort_keys=True,
            )
            + "\n"
        )


def run_dir(config: EvaluationConfig, experiment: Experiment) -> Path:
    return config.output_root / experiment.output_name


def tmux_session_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def training_state(config: EvaluationConfig, experiment: Experiment) -> TrainingState:
    root = run_dir(config, experiment)
    exit_path = root / "exit.status"
    if exit_path.exists():
        text = exit_path.read_text().strip()
        try:
            exit_code = int(text)
        except ValueError as error:
            raise RuntimeError(
                f"invalid training exit status for {experiment.key}: {text!r}"
            ) from error
        return TrainingState.SUCCEEDED if exit_code == 0 else TrainingState.FAILED
    if tmux_session_exists(experiment.session_name):
        return TrainingState.RUNNING
    if root.exists() and any(root.iterdir()):
        return TrainingState.BROKEN
    return TrainingState.WAITING


def validate_config(config: EvaluationConfig) -> None:
    required = (
        config.deployment / "tools" / "train_manorl_autonomy.py",
        config.deployment / "tools" / "render_manorl_autonomy_evaluation.py",
        config.package,
        config.python,
        config.dexstream_root,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("evaluation inputs are missing: " + ", ".join(missing))
    if config.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if config.checkpoint_interval <= 0 or config.final_update <= 0:
        raise ValueError("checkpoint interval and final update must be positive")
    if config.final_update % config.checkpoint_interval:
        raise ValueError("final update must be divisible by checkpoint interval")


def discover_periodic_checkpoints(
    config: EvaluationConfig,
    experiment: Experiment,
) -> list[Path]:
    checkpoints = list(run_dir(config, experiment).glob("policy.update*.pt"))
    by_update: dict[int, Path] = {}
    for path in checkpoints:
        update = checkpoint_update(path)
        if update is None:
            raise ValueError(f"unexpected periodic checkpoint name: {path.name}")
        if update in by_update:
            raise ValueError(f"duplicate checkpoint update {update}")
        by_update[update] = path
    expected = list(
        range(
            config.checkpoint_interval,
            config.final_update + 1,
            config.checkpoint_interval,
        )
    )
    if sorted(by_update) != expected:
        missing = sorted(set(expected) - set(by_update))
        unexpected = sorted(set(by_update) - set(expected))
        raise RuntimeError(
            f"{experiment.key} periodic checkpoint set mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return [by_update[update] for update in expected]


def evaluation_root(config: EvaluationConfig, experiment: Experiment) -> Path:
    return run_dir(config, experiment) / "evaluations"


def evaluation_paths(
    config: EvaluationConfig,
    experiment: Experiment,
    *,
    identity: str,
    checkpoint: Path,
    split: str,
) -> dict[str, Path]:
    update = checkpoint_update(checkpoint)
    if update is None:
        raise ValueError("periodic checkpoint update is required")
    root = evaluation_root(config, experiment) / split / identity / f"update{update:06d}"
    return {
        "root": root,
        "trace": root / "eval.json",
        "artifact": root / "eval.npz",
        "summary": root / "summary.json",
        "log": root / "evaluate.log",
        "command": root / "command.json",
        "exit": root / "exit.status",
    }


def evaluation_argv(
    config: EvaluationConfig,
    experiment: Experiment,
    *,
    identity: str,
    checkpoint: Path,
    trace: Path,
    artifact: Path,
) -> list[str]:
    return [
        str(config.python),
        str(config.deployment / "tools" / "train_manorl_autonomy.py"),
        "evaluate",
        "--package",
        str(config.package),
        "--identity",
        identity,
        "--device",
        "cpu",
        "--policy-version",
        experiment.policy_version,
        "--num-envs",
        "1",
        "--no-persistentworkspace",
        "--checkpoint",
        str(checkpoint),
        "--trace",
        str(trace),
        "--artifact",
        str(artifact),
    ]


def evaluation_environment(config: EvaluationConfig) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MANORL_DEXSTREAM_ROOT": str(config.dexstream_root),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def read_summary(path: Path) -> EvaluationSummary:
    payload = json.loads(path.read_text())
    payload["selection_rank"] = tuple(payload["selection_rank"])
    return EvaluationSummary(**payload)


def evaluate_one(
    config: EvaluationConfig,
    experiment: Experiment,
    *,
    identity: str,
    checkpoint: Path,
    split: str,
) -> tuple[EvaluationSummary, Path]:
    paths = evaluation_paths(
        config,
        experiment,
        identity=identity,
        checkpoint=checkpoint,
        split=split,
    )
    if paths["summary"].exists():
        return read_summary(paths["summary"]), paths["artifact"]
    if paths["root"].exists() and any(paths["root"].iterdir()):
        if paths["exit"].exists() and paths["exit"].read_text().strip() == "0":
            if not paths["trace"].exists() or not paths["artifact"].exists():
                raise RuntimeError(f"successful evaluation is missing artifacts: {paths['root']}")
            summary = summarize_evaluation(
                trace_path=paths["trace"],
                artifact_path=paths["artifact"],
                checkpoint_path=checkpoint,
            )
            write_summary(paths["summary"], summary)
            return summary, paths["artifact"]
        raise RuntimeError(f"refusing to overwrite partial evaluation: {paths['root']}")

    paths["root"].mkdir(parents=True, exist_ok=False)
    argv = evaluation_argv(
        config,
        experiment,
        identity=identity,
        checkpoint=checkpoint,
        trace=paths["trace"],
        artifact=paths["artifact"],
    )
    atomic_write_json(
        paths["command"],
        {
            "started_at_utc": utc_now(),
            "experiment": asdict(experiment),
            "identity": identity,
            "checkpoint": str(checkpoint),
            "argv": argv,
        },
    )
    started = time.monotonic()
    with paths["log"].open("w", encoding="utf-8") as log:
        result = subprocess.run(
            argv,
            cwd=config.deployment,
            env=evaluation_environment(config),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    paths["exit"].write_text(f"{result.returncode}\n")
    if result.returncode:
        raise RuntimeError(
            f"evaluation failed for {experiment.key} {identity} {checkpoint.name}; "
            f"see {paths['log']}"
        )
    summary = summarize_evaluation(
        trace_path=paths["trace"],
        artifact_path=paths["artifact"],
        checkpoint_path=checkpoint,
    )
    write_summary(paths["summary"], summary)
    append_event(
        config,
        "checkpoint_evaluated",
        experiment=experiment.key,
        identity=identity,
        split=split,
        checkpoint=checkpoint.name,
        elapsed_seconds=time.monotonic() - started,
        strict_grasp_success=summary.strict_grasp_success,
        natural_horizon=summary.natural_horizon,
        loaded_airborne_frames=summary.loaded_airborne_frames,
    )
    return summary, paths["artifact"]


def summary_key(summary: EvaluationSummary) -> tuple[tuple[float, ...], int]:
    return summary.selection_rank, summary.checkpoint_update or -1


def write_sweep_csv(path: Path, summaries: list[EvaluationSummary]) -> None:
    rows = [asdict(summary) for summary in summaries]
    for row in rows:
        row["selection_rank"] = json.dumps(row["selection_rank"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_identities(trace_path: Path) -> dict[str, tuple[str, ...]]:
    trace = json.loads(trace_path.read_text())
    checkpoint_provenance = trace["provenance"]["checkpoint"]
    assignment = checkpoint_provenance["reference_assignment"]
    identities = tuple(str(value) for value in assignment["identities"])
    split = checkpoint_provenance["identity_split"]
    result: dict[str, tuple[str, ...]] = {}
    for name in ("validation", "test"):
        indices = tuple(int(value) for value in split[f"{name}_indices"])
        if not indices:
            raise ValueError(f"{name} split is empty")
        if min(indices) < 0 or max(indices) >= len(identities):
            raise ValueError(f"{name} split index exceeds reference assignment")
        result[name] = tuple(identities[index] for index in indices)
    if set(result["validation"]) & set(result["test"]):
        raise ValueError("validation and test identities overlap")
    return result


def aggregate_heldout(
    validation: list[EvaluationSummary],
    test: list[EvaluationSummary],
) -> dict[str, object]:
    def aggregate(rows: list[EvaluationSummary]) -> dict[str, object]:
        count = len(rows)
        if not count:
            raise ValueError("held-out aggregate cannot be empty")
        return {
            "count": count,
            "strict_grasp_success_count": sum(row.strict_grasp_success for row in rows),
            "strict_grasp_success_rate": sum(row.strict_grasp_success for row in rows) / count,
            "natural_horizon_count": sum(row.natural_horizon for row in rows),
            "natural_horizon_rate": sum(row.natural_horizon for row in rows) / count,
            "mean_loaded_airborne_frames": sum(row.loaded_airborne_frames for row in rows) / count,
            "mean_opposing_loaded_frames": sum(row.opposing_loaded_frames for row in rows) / count,
            "mean_object_path_error_m": sum(row.mean_object_path_error_m for row in rows) / count,
            "rows": [asdict(row) for row in rows],
        }

    return {
        "validation": aggregate(validation),
        "test": aggregate(test),
    }


def probe_video(path: Path) -> dict[str, object]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames:format=duration,size",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(result.stdout)
        payload["verification_backend"] = "ffprobe"
        return payload

    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        frame_count = sum(1 for _ in reader)
    finally:
        reader.close()
    if frame_count < 1:
        raise RuntimeError("rendered video has no decodable frames")
    size = metadata.get("size")
    fps = float(metadata.get("fps", 0.0))
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or int(size[0]) < 1
        or int(size[1]) < 1
        or fps <= 0
    ):
        raise RuntimeError(f"rendered video metadata is invalid: {metadata!r}")
    return {
        "verification_backend": "imageio",
        "stream": {
            "width": int(size[0]),
            "height": int(size[1]),
            "r_frame_rate": fps,
            "nb_frames": frame_count,
        },
        "format": {
            "duration": frame_count / fps,
            "size": path.stat().st_size,
        },
    }


def render_best(
    config: EvaluationConfig,
    experiment: Experiment,
    *,
    artifact: Path,
    trajectory_summary: EvaluationSummary,
    checkpoint_summary: EvaluationSummary,
    trajectory_split: str,
) -> dict[str, object]:
    root = evaluation_root(config, experiment) / "best"
    output = root / "actual-vs-reference.mp4"
    selection = root / "selection.json"
    render_log = root / "render.log"
    probe_path = root / "video-probe.json"
    if selection.exists() and output.exists() and probe_path.exists():
        return json.loads(selection.read_text())
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to overwrite partial best rendering: {root}")
    root.mkdir(parents=True, exist_ok=False)
    argv = [
        str(config.python),
        str(config.deployment / "tools" / "render_manorl_autonomy_evaluation.py"),
        "--artifact",
        str(artifact),
        "--output",
        str(output),
    ]
    environment = evaluation_environment(config)
    environment["MUJOCO_GL"] = "egl"
    with render_log.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            argv,
            cwd=config.deployment,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if result.returncode or not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"render failed for {experiment.key}; see {render_log}")
    probe_payload = probe_video(output)
    atomic_write_json(probe_path, probe_payload)
    payload = {
        "created_at_utc": utc_now(),
        "experiment": asdict(experiment),
        "selected_checkpoint": asdict(checkpoint_summary),
        "selected_trajectory_split": trajectory_split,
        "selected_trajectory": asdict(trajectory_summary),
        "artifact": str(artifact),
        "video": str(output),
        "video_probe": probe_payload,
    }
    atomic_write_json(selection, payload)
    append_event(
        config,
        "best_video_rendered",
        experiment=experiment.key,
        checkpoint=checkpoint_summary.checkpoint,
        identity=trajectory_summary.identity,
        split=trajectory_split,
        video=str(output),
    )
    return payload


def evaluate_experiment(
    config: EvaluationConfig,
    experiment: Experiment,
) -> dict[str, object]:
    root = evaluation_root(config, experiment)
    completed_path = root / "completed.json"
    if completed_path.exists():
        return json.loads(completed_path.read_text())

    checkpoints = discover_periodic_checkpoints(config, experiment)
    canonical: list[EvaluationSummary] = []
    canonical_artifacts: dict[str, Path] = {}
    for checkpoint in checkpoints:
        summary, artifact = evaluate_one(
            config,
            experiment,
            identity=config.identity,
            checkpoint=checkpoint,
            split="checkpoint-sweep",
        )
        canonical.append(summary)
        canonical_artifacts[summary.checkpoint] = artifact
    write_sweep_csv(root / "checkpoint-sweep.csv", canonical)
    selected_checkpoint = max(canonical, key=summary_key)
    selected_trace = evaluation_paths(
        config,
        experiment,
        identity=config.identity,
        checkpoint=Path(selected_checkpoint.checkpoint),
        split="checkpoint-sweep",
    )["trace"]
    identities = split_identities(selected_trace)

    heldout_rows: dict[str, list[EvaluationSummary]] = {"validation": [], "test": []}
    trajectory_candidates: list[tuple[str, EvaluationSummary, Path]] = [
        (
            "checkpoint-sweep",
            selected_checkpoint,
            canonical_artifacts[selected_checkpoint.checkpoint],
        )
    ]
    for split_name in ("validation", "test"):
        for identity in identities[split_name]:
            summary, artifact = evaluate_one(
                config,
                experiment,
                identity=identity,
                checkpoint=Path(selected_checkpoint.checkpoint),
                split=split_name,
            )
            heldout_rows[split_name].append(summary)
            trajectory_candidates.append((split_name, summary, artifact))

    heldout = aggregate_heldout(
        heldout_rows["validation"],
        heldout_rows["test"],
    )
    atomic_write_json(root / "heldout-summary.json", heldout)
    best_split, best_trajectory, best_artifact = max(
        trajectory_candidates,
        key=lambda row: summary_key(row[1]),
    )
    video = render_best(
        config,
        experiment,
        artifact=best_artifact,
        trajectory_summary=best_trajectory,
        checkpoint_summary=selected_checkpoint,
        trajectory_split=best_split,
    )
    result = {
        "completed_at_utc": utc_now(),
        "experiment": asdict(experiment),
        "checkpoint_count": len(canonical),
        "selected_checkpoint": asdict(selected_checkpoint),
        "heldout": heldout,
        "best_video": video,
    }
    atomic_write_json(completed_path, result)
    append_event(
        config,
        "experiment_evaluation_completed",
        experiment=experiment.key,
        selected_checkpoint=selected_checkpoint.checkpoint,
        validation_strict_success_rate=heldout["validation"]["strict_grasp_success_rate"],
        test_strict_success_rate=heldout["test"]["strict_grasp_success_rate"],
    )
    return result


def queue_status(
    config: EvaluationConfig,
    states: dict[str, TrainingState],
) -> dict[str, object]:
    return {
        "updated_at_utc": utc_now(),
        "training": {key: state.value for key, state in states.items()},
        "evaluation": {
            experiment.key: (
                "completed"
                if (evaluation_root(config, experiment) / "completed.json").exists()
                else "waiting"
            )
            for experiment in EXPERIMENTS
        },
    }


def supervise(config: EvaluationConfig, *, once: bool) -> int:
    validate_config(config)
    append_event(config, "evaluation_supervisor_started", pid=os.getpid())
    last_signature: tuple[object, ...] | None = None
    while True:
        states = {
            experiment.key: training_state(config, experiment)
            for experiment in EXPERIMENTS
        }
        atomic_write_json(
            config.output_root / "evaluation-queue.status.json",
            queue_status(config, states),
        )
        failed = [
            key
            for key, state in states.items()
            if state in (TrainingState.FAILED, TrainingState.BROKEN)
        ]
        if failed:
            append_event(config, "evaluation_queue_failed", failed_training_jobs=failed)
            return 1

        ready = next(
            (
                experiment
                for experiment in EXPERIMENTS
                if states[experiment.key] is TrainingState.SUCCEEDED
                and not (evaluation_root(config, experiment) / "completed.json").exists()
            ),
            None,
        )
        if ready is not None:
            try:
                evaluate_experiment(config, ready)
            except Exception as error:
                atomic_write_json(
                    evaluation_root(config, ready) / "error.json",
                    {
                        "failed_at_utc": utc_now(),
                        "experiment": ready.key,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                append_event(
                    config,
                    "experiment_evaluation_failed",
                    experiment=ready.key,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                raise
            continue

        completed = [
            experiment.key
            for experiment in EXPERIMENTS
            if (evaluation_root(config, experiment) / "completed.json").exists()
        ]
        if len(completed) == len(EXPERIMENTS):
            atomic_write_json(
                config.output_root / "evaluation-queue.completed.json",
                {"completed_at_utc": utc_now(), "experiments": completed},
            )
            append_event(config, "evaluation_queue_completed")
            return 0

        signature = tuple(
            (
                experiment.key,
                states[experiment.key].value,
                experiment.key in completed,
            )
            for experiment in EXPERIMENTS
        )
        if signature != last_signature:
            append_event(
                config,
                "evaluation_queue_waiting",
                training={key: value.value for key, value in states.items()},
                completed=completed,
            )
            last_signature = signature
        if once:
            return 0
        time.sleep(config.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment",
        type=Path,
        default=Path(
            "/mnt/nas-222-projects/cty/manorl-autonomy/"
            "deployments/v5-v6-ablation-20260919"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/nas-222-projects/cty/manorl-autonomy/"
            "outputs/v5-v6-ablation-20260919"
        ),
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=Path(
            "/mnt/nas-222-projects/cty/manorl-autonomy/"
            "data/cube2_02_v295_f120_pre180_post180"
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(
            "/home/cty/miniconda3/envs/contact-conditioned-autonomy/bin/python"
        ),
    )
    parser.add_argument(
        "--dexstream-root",
        type=Path,
        default=Path(
            "/mnt/user-home/cty/workplace/"
            "mujoco-warp-contactbench-contact-conditioned-autonomy/"
            "assets/dexstream_digital_assets"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = EvaluationConfig(
        deployment=args.deployment.resolve(),
        output_root=args.output_root.resolve(),
        package=args.package.resolve(),
        python=args.python.resolve(),
        dexstream_root=args.dexstream_root.resolve(),
        poll_seconds=args.poll_seconds,
    )
    return supervise(config, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
