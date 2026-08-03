#!/usr/bin/env python3
"""Replay one synthetic target-DOF trajectory directly from Lance.

The command reads exactly one dataset version and row, extracts the recorded
28D physical hand state and post-controller target vectors, then runs MJX-Warp.
Use ``--headless`` for JSON parity metrics or omit it for the MuJoCo viewer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

from sim.manorl.target_replay import (
    TargetDofReplay,
    TargetReplaySourceError,
    load_target_replay_source,
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


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="synthetic Lance dataset"
    )
    parser.add_argument("--dataset-version", type=_positive_int, required=True)
    parser.add_argument("--row-index", type=_nonnegative_int, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run parity metrics and print JSON instead of opening a viewer",
    )
    parser.add_argument(
        "--frames",
        type=_positive_int,
        help="number of control transitions to replay (default: all)",
    )
    parser.add_argument("--speed", type=_positive_float, default=0.25)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-physics-override",
        action="store_true",
        help="allow CPU replay to omit row-specific GPU Warp CCD allocation",
    )
    parser.add_argument("--output", type=Path, help="optional headless JSON report")
    parser.add_argument("--print-every", type=_positive_int, default=25)
    parser.add_argument("--display", help="set DISPLAY before opening the GUI")
    parser.add_argument(
        "--xauthority", type=Path, help="set XAUTHORITY before GUI launch"
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
        raise ValueError(f"--frames={frames} exceeds row transitions={transitions}")
    return frames


def _validate_gui_numbers(args: argparse.Namespace) -> None:
    values = (args.azimuth, args.elevation, *args.lookat)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("camera values must be finite")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output and not args.headless:
        print("target-DOF replay error: --output requires --headless", file=sys.stderr)
        return 2
    if args.display:
        os.environ["DISPLAY"] = args.display
    if args.xauthority:
        os.environ["XAUTHORITY"] = str(args.xauthority.expanduser())
    try:
        if not args.headless:
            _validate_gui_numbers(args)
        source = load_target_replay_source(
            args.dataset,
            dataset_version=args.dataset_version,
            row_index=args.row_index,
        )
        transitions = _validate_frame_limit(args.frames, source.transitions)
        replay = TargetDofReplay(
            source,
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
    except (TargetReplaySourceError, ValueError, FileNotFoundError) as exc:
        print(f"target-DOF replay error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
