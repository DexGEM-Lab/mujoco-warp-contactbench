#!/usr/bin/env python
"""Public v4 PPO train/evaluate CLI for the pinned cube2 single reference."""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from sim.manorl.autonomy_batch_training import load_frozen_v4, run_batched_ppo
from sim.manorl.autonomy_contracts import ACTION_CONTRACT_ID, CHECKPOINT_FORMAT, OBSERVATION_CONTRACT_ID, REWARD_CONTRACT_ID
from sim.manorl.autonomy_training import AutonomyActorCritic, BatchedAutonomyAdapter, identity_split, seed_everything
from sim.manorl.trajectory_package import load_trajectory_package

DEFAULT_PACKAGE = "/home/jay/dexrobot/FromSSH/manoRL_mujoco/outputs/manorl/contact_conditioned_autonomy/cube2_02_v295_f120_pre180_post180"
DEFAULT_IDENTITY = "cube2_02_2833"


def _jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)): return value.item()
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value


def _git_revision(path: Path) -> str:
    marker = path / "DEPLOYED_COMMIT"
    if marker.exists():
        value = marker.read_text().strip()
        if re.fullmatch(r"[0-9a-f]{40}", value) is None: raise ValueError(f"{marker} must contain a full lowercase commit")
        return value
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _catalog_and_trajectory(args):
    catalog = load_trajectory_package(Path(args.package))
    trajectory = next((row for row in catalog.trajectories if row.identity.identity == args.identity), None)
    if trajectory is None: raise ValueError(f"identity {args.identity!r} is absent from package")
    return catalog, trajectory


def _adapter(args, trajectory, *, full_horizon_diagnostic=False):
    return BatchedAutonomyAdapter(trajectory, num_envs=args.num_envs, device=args.device, seed=args.seed,
        persistent_ccd_workspace=args.persistentworkspace, ccd_contacts_per_world=args.ccd_contacts_per_world,
        full_horizon_diagnostic=full_horizon_diagnostic)


def _provenance(catalog, split, adapter):
    """Cache bytes are recorded, never used as a compatibility gate.

    Their float build hash can vary across supported build paths; source/asset/package
    and ABI/clock are the physical compatibility contract.
    """
    return {"source_commit": _git_revision(ROOT), "asset_pin": _git_revision(ROOT / "assets/dexstream_digital_assets"),
            "package_digest": catalog.package_digest, "manifest_sha256": catalog.manifest_sha256,
            "catalog_digest": catalog.catalog_digest, "identity_split": split,
            "contracts": {"checkpoint": CHECKPOINT_FORMAT, "observation": OBSERVATION_CONTRACT_ID,
                          "action": ACTION_CONTRACT_ID, "reward": REWARD_CONTRACT_ID},
            "clock": dict(adapter.runtime.clock_metadata), "identity": adapter.runtime.trajectory.identity.identity,
            "cache_hash_recorded_not_compared": adapter.runtime.cache.content_hash,
            "contact_capacity": adapter.runtime.warp_contact_capacity,
            "constraint_capacity_per_world": adapter.runtime.warp_constraint_capacity}


def _wandb(args, metadata):
    if not args.wandb: return None
    import wandb
    return wandb.init(project=args.wandb_project or os.environ.get("WANDB_PROJECT", "mujoco-mano"),
                      entity=args.wandb_entity or os.environ.get("WANDB_ENTITY", "sunjay45711-dexerto"),
                      config=metadata, mode=args.wandb_mode or os.environ.get("WANDB_MODE", "online"), reinit=True)


def train(args):
    seed_everything(args.seed); catalog, trajectory = _catalog_and_trajectory(args); split = identity_split(catalog, seed=args.split_seed)
    identity_index = next(i for i, row in enumerate(catalog.trajectories) if row is trajectory)
    if identity_index not in split["train_indices"]: raise ValueError("train identity must be in the deterministic TRAIN split")
    if args.total_transitions is not None and args.total_transitions != args.updates * args.rollouts * args.num_envs:
        raise ValueError("total-transitions must equal updates * rollouts * num-envs")
    adapter = _adapter(args, trajectory); provenance = _provenance(catalog, split, adapter)
    config = {key: value for key, value in vars(args).items() if key not in {"fn", "wandb"}}
    metadata = {"training_contract": "manorl.autonomy.training.v4.single_reference", "config": config, "provenance": provenance}
    run = _wandb(args, metadata)
    try:
        def publish(row):
            print(json.dumps(row), flush=True)
            if run is not None: run.log(row, step=int(row["transitions"]))
        _, _, rows = run_batched_ppo(adapter, updates=args.updates, rollouts=args.rollouts,
            learning_epochs=args.learning_epochs, mini_batches=args.mini_batches, checkpoint=args.checkpoint,
            checkpoint_interval=args.checkpoint_interval, config=config, provenance=provenance, on_update=publish)
    except BaseException:
        if run is not None: run.finish(exit_code=1)
        raise
    else:
        if run is not None: run.finish(exit_code=0)
    print(json.dumps({"checkpoint": args.checkpoint, "updates": len(rows), "cache_hash": provenance["cache_hash_recorded_not_compared"]}), flush=True)


def evaluate(args):
    seed_everything(args.seed); catalog, trajectory = _catalog_and_trajectory(args)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    split = identity_split(catalog, seed=payload.get("provenance", {}).get("identity_split", {}).get("seed", args.split_seed))
    adapter = _adapter(args, trajectory, full_horizon_diagnostic=args.full_horizon_diagnostic)
    model = AutonomyActorCritic(adapter.observation_space, adapter.action_space, device=adapter.device)
    expected = {"asset_pin": _git_revision(ROOT / "assets/dexstream_digital_assets"),
                "package_digest": catalog.package_digest, "manifest_sha256": catalog.manifest_sha256,
                "catalog_digest": catalog.catalog_digest, "identity_split": split,
                "identity": trajectory.identity.identity}
    payload = load_frozen_v4(args.checkpoint, model, map_location=adapter.device, expected_provenance=expected)
    observations, _ = adapter.reset(); total = 0.0; trace = []; natural_first = None
    limit = args.steps if args.steps is not None else adapter.runtime.length - 1
    for policy_step in range(limit):
        with torch.no_grad():
            mean, _ = model.compute({"observations": observations}, role="policy")
            action = torch.clamp(mean, -1., 1.)
        next_obs, reward, done, info = adapter.step(action); total += float(reward[0, 0].cpu())
        natural_done = bool(np.asarray(adapter.runtime.last_reward.done)[0])
        if natural_done and natural_first is None:
            natural_first = {"policy_step": policy_step, "reason_code": int(np.asarray(adapter.runtime.last_reason)[0])}
        trace.append({"policy_step": policy_step, "reward": float(reward[0, 0].cpu()),
                      "natural_done": natural_done, "runtime_done": bool(done[0, 0].cpu()),
                      "reason_code": int(np.asarray(adapter.runtime.last_reason)[0]),
                      "cache_hash": adapter.runtime.cache.content_hash})
        if natural_done and not args.full_horizon_diagnostic: break
        observations = next_obs if args.full_horizon_diagnostic else adapter.prepare_action()
    result = {"format": "manorl.autonomy.frozen_evaluation.v4", "checkpoint": str(args.checkpoint),
              "identity": trajectory.identity.identity, "started_reference_frame": 0, "steps": len(trace),
              "return": total, "natural_first_termination": natural_first,
              "full_horizon_diagnostic": bool(args.full_horizon_diagnostic),
              "diagnostic_boundary": "natural termination recorded; continued after it" if args.full_horizon_diagnostic else "stopped at natural first termination",
              "provenance": {"checkpoint": payload["provenance"], "evaluation_cache_hash_recorded_not_compared": adapter.runtime.cache.content_hash}, "trace": trace}
    trace_path = Path(args.trace); trace_path.parent.mkdir(parents=True, exist_ok=True); trace_path.write_text(json.dumps(result, indent=2, default=_jsonable) + "\n")
    print(json.dumps({"trace": str(trace_path), "steps": len(trace), "return": total, "natural_first_termination": natural_first}), flush=True)


def inspect(args):
    catalog, trajectory = _catalog_and_trajectory(args); adapter = _adapter(args, trajectory)
    print(json.dumps({"identity": trajectory.identity.identity, "raw_observation": list(adapter.reset()[0].shape),
        "cache_hash": adapter.runtime.cache.content_hash, "clock": adapter.runtime.clock_metadata,
        "workspace": args.persistentworkspace, "ccd_contacts_per_world": args.ccd_contacts_per_world,
        "constraint_capacity_per_world": adapter.runtime.warp_constraint_capacity}))


def main():
    parser = argparse.ArgumentParser(description=__doc__); subs = parser.add_subparsers(required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--package", default=DEFAULT_PACKAGE); common.add_argument("--identity", default=DEFAULT_IDENTITY)
    common.add_argument("--device", choices=("cpu", "gpu"), default="gpu"); common.add_argument("--seed", type=int, default=0)
    common.add_argument("--split-seed", type=int, default=0); common.add_argument("--num-envs", type=int, default=4096)
    common.add_argument("--persistentworkspace", action=argparse.BooleanOptionalAction, default=True)
    common.add_argument("--ccd-contacts-per-world", type=int, default=121, help="cube2 B4096 CCD scratch; runtime keeps njmax=512/world")
    train_parser = subs.add_parser("train", parents=[common]); train_parser.add_argument("--updates", type=int, default=256); train_parser.add_argument("--rollouts", type=int, default=32)
    train_parser.add_argument("--learning-epochs", type=int, default=4); train_parser.add_argument("--mini-batches", type=int, default=16); train_parser.add_argument("--total-transitions", type=int)
    train_parser.add_argument("--checkpoint", default="outputs/manorl/contact_conditioned_autonomy/cube2_02_v4_ppo.pt"); train_parser.add_argument("--checkpoint-interval", type=int, default=16)
    train_parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True); train_parser.add_argument("--wandb-project"); train_parser.add_argument("--wandb-entity"); train_parser.add_argument("--wandb-mode"); train_parser.set_defaults(fn=train)
    eval_parser = subs.add_parser("evaluate", parents=[common]); eval_parser.add_argument("--checkpoint", required=True); eval_parser.add_argument("--steps", type=int); eval_parser.add_argument("--full-horizon-diagnostic", action="store_true")
    eval_parser.add_argument("--trace", default="outputs/manorl/contact_conditioned_autonomy/cube2_02_v4_eval.json"); eval_parser.set_defaults(fn=evaluate)
    inspect_parser = subs.add_parser("inspect", parents=[common]); inspect_parser.set_defaults(fn=inspect)
    args = parser.parse_args(); return args.fn(args)

if __name__ == "__main__": raise SystemExit(main())
