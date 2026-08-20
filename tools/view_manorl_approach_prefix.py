"""View accepted-parent approach-prefix augmentation episodes in one window.

Single active MJX-Warp environment and one checkpoint policy runtime are
reused across every reset.  Each terminal advances ``--seed`` and reinstalls
the next seeded approach-prefixed reference derived from the same accepted
parent, so every episode starts from its own frame 0 with a freshly sampled
hand approach while the original pre60 reference and the deterministic
checkpoint policy continue unchanged.

Usage:

    python tools/view_manorl_approach_prefix.py \\
        --checkpoint /path/checkpoint-001000.pt \\
        --accepted-parent /path/accepted-parent.json \\
        --predecode-dir /path/predecode-bundle \\
        --seed 49

Output lines are machine-readable JSON on stdout:

    RESET    per-episode sampled start (xyz, xy radius, z offset, prefix)
    FRAME    periodic progress telemetry (every --print-every steps)
    TERMINAL episode end with termination reason code
    STOP     explicit bounded exit after --max-episodes
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sim.manorl.view_environment import view_approach_prefix_episodes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--accepted-parent", type=Path, required=True)
    parser.add_argument(
        "--predecode-dir",
        type=Path,
        required=True,
        help="predecoded pre60 bundle directory holding manifest.json and identity pickles",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=49,
        help="starting approach-prefix sampling seed (default: 49)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="simulation speed multiplier (default: 1.0 real-time)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="print a FRAME telemetry line every N control steps (default: 20)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="stop after this many completed episodes instead of looping until the window closes",
    )
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.speed <= 0.0:
        parser.error("--speed must be positive")
    if args.print_every < 1:
        parser.error("--print-every must be positive")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    if not args.predecode_dir.is_dir():
        parser.error(f"--predecode-dir is not a directory: {args.predecode_dir}")
    if not (args.predecode_dir / "manifest.json").is_file():
        parser.error(f"--predecode-dir lacks manifest.json: {args.predecode_dir}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    view_approach_prefix_episodes(
        checkpoint=args.checkpoint,
        accepted_parent=args.accepted_parent,
        predecode_dir=args.predecode_dir,
        start_seed=args.seed,
        speed=args.speed,
        print_every=args.print_every,
        max_episodes=args.max_episodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
