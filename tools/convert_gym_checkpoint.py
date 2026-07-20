"""Convert a validated IsaacGym/rl-games MANO checkpoint to native skrl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.manorl.gym_checkpoint import convert_gym_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="read-only rl-games .pth checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="new native .pt output; existing files are refused")
    args = parser.parse_args(argv)
    output = convert_gym_checkpoint(args.source, args.output)
    print(f"converted checkpoint={output}")
    print(f"sidecar={output.with_suffix(output.suffix + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
