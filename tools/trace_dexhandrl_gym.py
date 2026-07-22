#!/usr/bin/env python3
"""Runtime-only exact Gym trace wrapper; never writes the Gym checkout.

Importing this module or requesting --help is offline-safe. ``run`` imports Isaac
Gym first, snapshots the source checkout, and only monkeypatches live objects.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
from typing import Any
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sim.dexhandrl.trace_contract import EXACT_IDENTITY

GYM_ROOT = Path("/home/jay/dexrobot/FromSSH/dexhand_reconstruction/dexrobot_isaac")

def _git_snapshot(root: Path) -> tuple[str, str]:
    return (subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1"], text=True),
            subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip())

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--steps", type=int, default=3)
    p.add_argument("--gym-root", type=Path, default=GYM_ROOT); p.add_argument("--checkpoint", type=Path, required=True)
    return p

def _require(instance: Any, *names: str) -> None:
    absent = [name for name in names if not hasattr(instance, name)]
    if absent: raise RuntimeError(f"Gym trace hook unavailable: {type(instance).__name__} lacks {absent}")

def run(args: argparse.Namespace) -> None:
    """Prepare authoritative runtime wrappers and fail closed until a live player is supplied.

    The current Gym checkout exposes task-side boundaries (`obs_dict`,
    `action_processor`, `pre_physics_step`, `post_physics_step`) but no stable
    public one-env player factory. This tool deliberately will not reconstruct
    Hydra/player internals or write a helper into the sibling checkout.
    """
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("set PYTHONDONTWRITEBYTECODE=1 before a Gym trace")
    before = _git_snapshot(args.gym_root)
    old_cwd, old_path = Path.cwd(), list(sys.path)
    try:
        os.chdir(args.gym_root); sys.path.insert(0, str(args.gym_root))
        import isaacgym  # noqa: F401  # must precede torch
        import torch  # noqa: F401
        from dexhand_env.components.action.action_processor import ActionProcessor
        from dexhand_env.components.observation.observation_encoder import ObservationEncoder
        _require(ActionProcessor, "process_actions", "apply_pre_action_rule")
        _require(ObservationEncoder, "compute_observations", "concatenate_observations")
        raise RuntimeError(
            "Gym runtime boundary reached: provide a public exact one-env player factory. "
            "Required runtime capture hooks are DexHandBase.pre_physics_step/post_physics_step, "
            "task.obs_dict, and action_processor.{active_prev_targets,active_rule_targets,active_raw_targets,active_next_targets,full_dof_targets}. "
            f"Exact requested identity={EXACT_IDENTITY}; no Gym source was modified."
        )
    finally:
        os.chdir(old_cwd); sys.path[:] = old_path
        after = _git_snapshot(args.gym_root)
        if after != before: raise RuntimeError("Gym source checkout changed during trace wrapper execution")

def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv); run(args); return 0
if __name__ == "__main__": raise SystemExit(main())
