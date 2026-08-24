"""Compile pinned Lance rows into a Lance-free ManoRL trajectory package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

from sim.manorl.trajectory import (
    DEFAULT_POST_PADDING,
    DEFAULT_PRE_PADDING,
    DEFAULT_REFERENCE_FPS,
    SUPPORTED_REFERENCE_FPS,
    ReferenceTrajectory,
    TrajectorySelection,
)
from sim.manorl.trajectory_package import (
    TRAJECTORY_PACKAGE_SCHEMA,
    load_trajectory_package,
    write_trajectory_package,
)

_RETRYABLE_NATIVE_CODES = frozenset((-11, 139, -6, 134))


class NativeWorkerExhausted(RuntimeError):
    """Raised after a bounded set of native-fault worker attempts."""

    def __init__(self, message: str, attempts: list[dict[str, object]]) -> None:
        super().__init__(message)
        self.attempts = attempts


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _run_worker(
    command: list[str],
    *,
    log_dir: Path,
    label: str,
    max_attempts: int,
) -> list[dict[str, object]]:
    attempts: list[dict[str, object]] = []
    for attempt in range(1, max_attempts + 1):
        stdout_path = log_dir / f"{label}.attempt-{attempt}.stdout.log"
        stderr_path = log_dir / f"{label}.attempt-{attempt}.stderr.log"
        started = time.monotonic()
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        record: dict[str, object] = {
            "attempt": attempt,
            "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
        }
        attempts.append(record)
        if result.returncode == 0:
            return attempts
        if result.returncode not in _RETRYABLE_NATIVE_CODES:
            raise RuntimeError(
                f"isolated compiler worker {label} failed semantically with "
                f"exit {result.returncode}; see {stderr_path}"
            )
        if attempt == max_attempts:
            raise NativeWorkerExhausted(
                f"isolated compiler worker {label} exhausted {max_attempts} "
                f"native-fault attempts; see {log_dir}",
                attempts,
            )
    raise AssertionError("unreachable worker retry state")


def _selection_payload(selection: TrajectorySelection) -> dict[str, object]:
    if selection.expected_dataset_version is None or selection.reference_fps is None:
        raise ValueError("package compilation requires pinned dataset and reference clocks")
    return {
        "object_type": selection.object_type,
        "gesture": selection.gesture,
        "selector": selection.canonical_selector,
        "dataset_path": str(selection.dataset_path),
        "dataset_version": selection.expected_dataset_version,
        "pre_padding": selection.pre_padding,
        "post_padding": selection.post_padding,
        "hand_side": selection.hand_side,
        "reference_fps": selection.reference_fps,
        "control_fps": selection.resolved_control_fps,
    }


def compile_package(
    output: Path,
    *,
    selection: TrajectorySelection,
    shard_size: int,
    max_attempts: int,
) -> Path:
    """Run isolated Lance workers and atomically publish an MTP catalog."""

    if "lance" in sys.modules or "pyarrow" in sys.modules:
        raise RuntimeError("compiler coordinator must start without Lance/PyArrow imports")
    if output.exists():
        raise FileExistsError(f"refusing to replace trajectory package: {output}")
    build_root = output.parent / f".{output.name}.compile-{uuid4().hex}"
    log_dir = build_root / "logs"
    shard_dir = build_root / "shards"
    requests_dir = build_root / "requests"
    for path in (log_dir, shard_dir, requests_dir):
        path.mkdir(parents=True, exist_ok=False)
    selection_path = build_root / "selection.json"
    selection_path.write_text(
        json.dumps(_selection_payload(selection), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    worker_base = [
        sys.executable,
        "-m",
        "tools.compile_manorl_trajectory_worker",
        "--selection-json",
        str(selection_path),
    ]
    report: dict[str, Any] = {
        "schema": "manorl.trajectory_package.compile_report.v1",
        "package_schema": TRAJECTORY_PACKAGE_SCHEMA,
        "selection": _selection_payload(selection),
        "shard_size": shard_size,
        "max_attempts": max_attempts,
        "workers": [],
    }
    try:
        discovery_path = build_root / "discovery.json"
        attempts = _run_worker(
            [*worker_base[:3], "discover", *worker_base[3:], "--output", str(discovery_path)],
            log_dir=log_dir,
            label="discovery",
            max_attempts=max_attempts,
        )
        report["workers"].append({"label": "discovery", "attempts": attempts})
        discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
        candidates = discovery.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("isolated discovery returned no trajectory candidates")
        discovery_digest = _digest(discovery)
        schema_digest = str(discovery["dataset_schema_digest"])
        shard_count = (len(candidates) + shard_size - 1) // shard_size
        print(
            f"discovered trajectories={len(candidates)} pairs={len(discovery['resolved_pairs'])} "
            f"decode_shards={shard_count}",
            flush=True,
        )
        def decode_group(
            requests: list[dict[str, object]], label: str
        ) -> tuple[list[ReferenceTrajectory], list[dict[str, object]]]:
            requests_path = requests_dir / f"{label}.json"
            requests_path.write_text(
                json.dumps(requests, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            shard_output = shard_dir / f"{label}.mtp"
            rejections_output = requests_dir / f"{label}.rejections.json"
            command = [
                *worker_base[:3],
                "decode",
                *worker_base[3:],
                "--requests-json",
                str(requests_path),
                "--rejections-json",
                str(rejections_output),
                "--dataset-schema-digest",
                schema_digest,
                "--discovery-digest",
                discovery_digest,
                "--output",
                str(shard_output),
            ]
            try:
                attempts = _run_worker(
                    command,
                    log_dir=log_dir,
                    label=label,
                    max_attempts=max_attempts,
                )
            except NativeWorkerExhausted as exc:
                report["workers"].append(
                    {
                        "label": label,
                        "request_count": len(requests),
                        "attempts": exc.attempts,
                        "result": "native_fault_exhausted",
                    }
                )
                if len(requests) == 1:
                    raise RuntimeError(
                        f"candidate {requests[0]} persistently crashes the isolated decoder"
                    ) from exc
                midpoint = len(requests) // 2
                print(
                    f"bisecting native-fault shard={label} requests={len(requests)}",
                    flush=True,
                )
                left_trajectories, left_rejections = decode_group(
                    requests[:midpoint], label + "-L"
                )
                right_trajectories, right_rejections = decode_group(
                    requests[midpoint:], label + "-R"
                )
                return (
                    left_trajectories + right_trajectories,
                    left_rejections + right_rejections,
                )
            report["workers"].append(
                {
                    "label": label,
                    "request_count": len(requests),
                    "attempts": attempts,
                    "result": "decoded",
                }
            )
            if not rejections_output.is_file():
                raise RuntimeError(f"isolated shard {label} omitted its rejection ledger")
            rejections = json.loads(rejections_output.read_text(encoding="utf-8"))
            if not isinstance(rejections, list):
                raise RuntimeError(f"isolated shard {label} rejection ledger is invalid")
            decoded = (
                list(load_trajectory_package(shard_output).trajectories)
                if shard_output.is_dir()
                else []
            )
            if len(decoded) + len(rejections) != len(requests):
                raise RuntimeError(
                    f"isolated shard {label} accounted for "
                    f"{len(decoded) + len(rejections)} of {len(requests)} candidates"
                )
            return decoded, rejections

        trajectories: list[ReferenceTrajectory] = []
        rejections: list[dict[str, object]] = []
        for shard_index, start in enumerate(range(0, len(candidates), shard_size)):
            requests = candidates[start : start + shard_size]
            decoded, rejected = decode_group(requests, f"decode-{shard_index:05d}")
            trajectories.extend(decoded)
            rejections.extend(rejected)
            print(
                f"decoded shard={shard_index + 1}/{shard_count} "
                f"trajectories={len(decoded)} rejected={len(rejected)} "
                f"total_valid={len(trajectories)}",
                flush=True,
            )
        if len(trajectories) + len(rejections) != len(candidates):
            raise RuntimeError(
                f"accounted for {len(trajectories) + len(rejections)} of "
                f"{len(candidates)} candidates"
            )
        valid_pairs = sorted(
            {item.identity.identity.rsplit("_", 1)[0].replace("_", ":", 1) for item in trajectories}
        )
        if valid_pairs != discovery["resolved_pairs"]:
            raise RuntimeError("one or more discovered pairs has no valid trajectory")
        report["candidate_count"] = len(candidates)
        report["decoded_count"] = len(trajectories)
        report["rejected_count"] = len(rejections)
        report["rejections"] = rejections
        report["resolved_pairs"] = discovery["resolved_pairs"]
        report["discovery_digest"] = discovery_digest
        report_path = build_root / "compile_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        package = write_trajectory_package(
            output,
            trajectories,
            selection=selection,
            dataset_schema_digest=schema_digest,
            discovery_digest=discovery_digest,
            compiler={
                "coordinator": "tools.compile_manorl_trajectory_package",
                "worker": "tools.compile_manorl_trajectory_worker",
                "candidate_count": len(candidates),
                "decoded_count": len(trajectories),
                "rejected_count": len(rejections),
                "shard_size": shard_size,
                "max_attempts": max_attempts,
                "compile_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            },
            source_candidates=candidates,
            source_rejections=rejections,
        )
        # Shard packages are transient decode boundaries. The final package has
        # copied and hashed every byte, so retain compact logs/requests/report
        # evidence rather than a second full catalog.
        shutil.rmtree(shard_dir)
        evidence = output.with_name(output.name + ".compile-evidence")
        if evidence.exists():
            raise FileExistsError(f"compile evidence destination exists: {evidence}")
        shutil.move(str(build_root), evidence)
        if "lance" in sys.modules or "pyarrow" in sys.modules:
            raise RuntimeError("compiler coordinator imported Lance/PyArrow")
        return package
    except BaseException:
        failed = build_root.with_name(build_root.name + ".failed")
        if build_root.exists():
            os.replace(build_root, failed)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-version", type=int, required=True)
    parser.add_argument(
        "--pairs",
        help="optional comma-separated object:action subset; default compiles all pairs",
    )
    parser.add_argument("--reference-fps", type=int, choices=SUPPORTED_REFERENCE_FPS, default=DEFAULT_REFERENCE_FPS)
    parser.add_argument("--hand-side", choices=("right", "left", "both"), default="right")
    parser.add_argument("--pre-padding", type=int, default=DEFAULT_PRE_PADDING)
    parser.add_argument("--post-padding", type=int, default=DEFAULT_POST_PADDING)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    if args.dataset_version < 1:
        parser.error("dataset-version must be positive")
    if args.pre_padding < 0 or args.post_padding < 0:
        parser.error("padding must be non-negative")
    if args.shard_size < 1 or args.max_attempts < 1:
        parser.error("shard-size and max-attempts must be positive")
    selection = TrajectorySelection(
        selector=args.pairs or "all",
        dataset_path=args.dataset_path.expanduser().resolve(),
        expected_dataset_version=args.dataset_version,
        pre_padding=args.pre_padding,
        post_padding=args.post_padding,
        hand_side=args.hand_side,
        reference_fps=args.reference_fps,
    )
    package = compile_package(
        args.output.expanduser().resolve(),
        selection=selection,
        shard_size=args.shard_size,
        max_attempts=args.max_attempts,
    )
    catalog = load_trajectory_package(package)
    print(
        json.dumps(
            {
                "package": str(package),
                "package_digest": catalog.package_digest,
                "catalog_digest": catalog.catalog_digest,
                "trajectory_count": len(catalog.trajectories),
                "pair_count": len(catalog.resolved_pairs),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
