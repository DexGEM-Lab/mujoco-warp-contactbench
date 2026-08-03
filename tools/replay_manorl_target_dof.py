#!/usr/bin/env python3
"""Replay a versioned post-controller target-DOF package in MJX-Warp.

The input is a predecoded ``.npz`` package with a required adjacent ``.json``
sidecar.  This command never opens Lance and never loads a policy checkpoint.
Use ``--headless`` for machine-readable parity metrics or omit it to open the
native MuJoCo viewer while MJX-Warp remains the sole physics simulator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from sim.manorl.target_replay import (
    TargetReplayPackageError,
    TargetDofReplay,
    load_target_replay_package,
    render_target_replay,
    write_report,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        required=True,
        help="validated target replay .npz package; its .json sidecar is required",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="gpu",
        help="MJX-Warp device; GPU is required to preserve packages with explicit Warp CCD scratch",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run parity metrics and print JSON instead of opening a viewer",
    )
    parser.add_argument(
        "--frames",
        type=_positive_int,
        help="number of control transitions to replay (default: all package transitions)",
    )
    parser.add_argument(
        "--speed",
        type=_positive_float,
        default=0.25,
        help="GUI playback speed relative to recorded time (default: 0.25)",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="loop GUI playback (default: true)",
    )
    parser.add_argument(
        "--allow-physics-override",
        action="store_true",
        help="allow CPU replay to omit package-specific Warp CCD allocation; report the override",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path for headless metrics",
    )
    parser.add_argument(
        "--print-every",
        type=_positive_int,
        default=25,
        help="GUI progress print interval in frames (default: 25)",
    )
    parser.add_argument("--display", help="set DISPLAY before opening the GUI")
    parser.add_argument(
        "--xauthority", type=Path, help="set XAUTHORITY before opening the GUI"
    )
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument("--distance", type=_positive_float, default=0.65)
    parser.add_argument(
        "--lookat",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.08),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--max-object-position-error-m", type=_positive_float, default=0.05
    )
    parser.add_argument(
        "--max-object-rotation-error-rad", type=_positive_float, default=0.25
    )
    parser.add_argument("--max-qpos-abs-error", type=_positive_float, default=1.0)
    return parser


def _validate_frame_limit(frames: int | None, transitions: int) -> int:
    if frames is None:
        return transitions
    if frames > transitions:
        raise ValueError(f"--frames={frames} exceeds package transitions={transitions}")
    return frames


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.display:
        os.environ["DISPLAY"] = args.display
    if args.xauthority:
        os.environ["XAUTHORITY"] = str(args.xauthority.expanduser())
    try:
        package = load_target_replay_package(args.package)
        transitions = _validate_frame_limit(args.frames, package.transitions)
        replay = TargetDofReplay(
            package,
            device=args.device,
            allow_physics_override=args.allow_physics_override,
        )
        if args.headless:
            report = replay.headless_report(
                transitions=transitions,
                max_object_position_error_m=args.max_object_position_error_m,
                max_object_rotation_error_rad=args.max_object_rotation_error_rad,
                max_qpos_abs_error=args.max_qpos_abs_error,
            )
            if args.output:
                write_report(args.output, report)
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
            return 0 if report["status"] == "pass" else 1
        if args.output:
            raise ValueError("--output is only valid with --headless")
        render_target_replay(
            replay,
            speed=args.speed,
            loop=args.loop,
            transitions=transitions,
            print_every=args.print_every,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            lookat=tuple(args.lookat),
        )
        return 0
    except (TargetReplayPackageError, ValueError, FileNotFoundError) as exc:
        print(f"target-DOF replay error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
