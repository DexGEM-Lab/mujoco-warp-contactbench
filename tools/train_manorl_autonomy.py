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

from sim.manorl.autonomy_batch_training import inspect_v4_resume, inspect_v4_warmstart, load_frozen_v4, resume_lineage, run_batched_ppo
from sim.manorl.autonomy_telemetry import configure_wandb_axis
from sim.manorl.autonomy_contracts import ACTION_CONTRACT_ID, CHECKPOINT_FORMAT, OBSERVATION_CONTRACT_ID, REWARD_CONTRACT_ID, validate_v4_checkpoint_metadata
from sim.manorl.autonomy_training import AutonomyActorCritic, BatchedAutonomyAdapter, actor_critic_architecture, identity_split, seed_everything, v4_ppo_config, validate_learning_rate, teacher_anchor_metadata
from sim.manorl.autonomy_v4_telemetry import REWARD_NAMES
from sim.manorl.autonomy_v4 import REWARD_MOTION_RADIUS, reward_parameters
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


def _training_references(args, catalog, trajectory, split):
    if getattr(args, "all_references", False):
        references = list(catalog.trajectories)
        if not references:
            raise ValueError("all-reference training requires a non-empty package")
        if any(t.identity.identity.split('_')[0] != "cube2" for t in references):
            raise ValueError("all-reference v4 currently requires shared cube2 object geometry")
        return references
    if not getattr(args, "all_train_references", False):
        return trajectory
    references = [catalog.trajectories[i] for i in split["train_indices"]]
    if len(references) != 40:
        raise ValueError("multi-reference training requires exactly 40 TRAIN identities")
    if any(t.identity.identity.split('_')[0] != "cube2" for t in references):
        raise ValueError("multi-reference v4 currently requires shared cube2 object geometry")
    return references


def _adapter(args, trajectory, *, full_horizon_diagnostic=False):
    return BatchedAutonomyAdapter(trajectory, num_envs=args.num_envs, device=args.device, seed=args.seed,
        persistent_ccd_workspace=args.persistentworkspace, ccd_contacts_per_world=args.ccd_contacts_per_world,
        full_horizon_diagnostic=full_horizon_diagnostic)


def _provenance(catalog, split, adapter):
    """Cache bytes are recorded, never used as a compatibility gate.

    Their float build hash can vary across supported build paths; source/asset/package
    and ABI/clock are the physical compatibility contract.
    """
    result = {"source_commit": _git_revision(ROOT), "asset_pin": _git_revision(ROOT / "assets/dexstream_digital_assets"),
            "package_digest": catalog.package_digest, "manifest_sha256": catalog.manifest_sha256,
            "catalog_digest": catalog.catalog_digest, "identity_split": split,
            "contracts": {"checkpoint": CHECKPOINT_FORMAT, "observation": OBSERVATION_CONTRACT_ID,
                          "action": ACTION_CONTRACT_ID, "reward": REWARD_CONTRACT_ID},
            "clock": dict(adapter.runtime.clock_metadata), "identity": adapter.runtime.trajectory.identity.identity,
            "cache_hash_recorded_not_compared": adapter.runtime.cache.content_hash,
            "contact_capacity": adapter.runtime.warp_contact_capacity,
            "constraint_capacity_per_world": adapter.runtime.warp_constraint_capacity,
            "reward_parameters": reward_parameters(getattr(adapter.runtime.cache, "object_radius", REWARD_MOTION_RADIUS))}
    if getattr(adapter.runtime, "trajectories", None) is not None:
        result["reference_assignment"] = {"mode": "fixed_round_robin_same_reference_reset",
            "identities": [t.identity.identity for t in adapter.runtime.trajectories]}
    return result


def _wandb(args, metadata):
    if not args.wandb: return None
    import wandb
    run_id = args.wandb_run_id or os.environ.get("WANDB_RUN_ID")
    mode = args.wandb_mode or os.environ.get("WANDB_MODE", "online")
    if args.resume_checkpoint and (not run_id or mode != "online"):
        raise ValueError("W&B resume requires explicit --wandb-run-id/WANDB_RUN_ID and online mode")
    options = {"id": run_id} if run_id else {}
    if args.resume_checkpoint:
        options["resume"] = "must"
    run = wandb.init(project=args.wandb_project or os.environ.get("WANDB_PROJECT", "mujoco-mano"),
                      entity=args.wandb_entity or os.environ.get("WANDB_ENTITY", "sunjay45711-dexerto"),
                      config=None if args.resume_checkpoint else metadata, mode=mode, reinit=True, **options)
    try:
        metadata["config"]["wandb_run_id"] = run.id
        # Resume without an init-time config conflict, preserving the checkpoint's
        # previous configuration in lineage before publishing the new total budget.
        run.config.update(metadata, allow_val_change=True)
        configure_wandb_axis(run)
    except BaseException:
        run.finish(exit_code=1)
        raise
    return run


def _resolved_telemetry_config(adapter, args):
    # Keep W&B metadata tied to the same default-plus-override factory passed
    # into the PPO builder. Per-update config/* fields record the live agent.
    ppo = v4_ppo_config(rollouts=args.rollouts, learning_epochs=args.learning_epochs, mini_batches=args.mini_batches, learning_rate=args.learning_rate)
    ppo.update({"normalize_observations": False, "normalize_values": False,
                # skrl 2.1.0 compute_gae standardizes advantages unconditionally.
                "normalize_advantages": True, "optimizer": "Adam",
                "adam_betas": [0.9, 0.999], "adam_eps": 1e-8,
                "rollout_batch_samples": args.rollouts * adapter.num_envs})
    return {"teacher_anchor": teacher_anchor_metadata(args.teacher_anchor_beta, args.teacher_anchor_passes), "ppo": ppo,
            "runtime": {"num_envs": adapter.num_envs, "raw_observation_dim": adapter.observation_dim,
                        "action_dim": adapter.action_dim, "control_timestep": adapter.runtime.cache.control_timestep,
                        "physics_substeps": 4,
                        "reference_count": len(adapter.runtime.trajectories) if getattr(adapter.runtime, "trajectories", None) is not None else 1,
                        "reference_assignment_mode": "all_package_references" if getattr(args, "all_references", False) else ("train_split_round_robin" if getattr(args, "all_train_references", False) else "single_reference")},
            "architecture": actor_critic_architecture(separate_critic=args.separate_critic),
            "thresholds": {"loaded_force_N": .02, "airborne_clearance_m": .005,
                           "severe_reason_bits": {"reference_complete": 1, "deviation": 2, "fallen": 4, "nonfinite": 8}},
            "reward_parameters": reward_parameters(getattr(adapter.runtime.cache, "object_radius", REWARD_MOTION_RADIUS))}


def train(args):
    validate_learning_rate(args.learning_rate)
    anchor = teacher_anchor_metadata(args.teacher_anchor_beta, args.teacher_anchor_passes)
    seed_everything(args.seed); catalog, trajectory = _catalog_and_trajectory(args); split = identity_split(catalog, seed=args.split_seed)
    identity_index = next(i for i, row in enumerate(catalog.trajectories) if row is trajectory)
    if args.all_references and args.all_train_references:
        raise ValueError("all-references and all-train-references are mutually exclusive")
    if not args.all_references and identity_index not in split["train_indices"]:
        raise ValueError("train identity must be in the deterministic TRAIN split")
    if args.total_transitions is not None and args.total_transitions != args.updates * args.rollouts * args.num_envs:
        raise ValueError("total-transitions must equal updates * rollouts * num-envs")
    references = _training_references(args, catalog, trajectory, split)
    adapter = _adapter(args, references); provenance = _provenance(catalog, split, adapter)
    if args.all_references and "reference_assignment" in provenance:
        provenance["reference_assignment"]["mode"] = "all_package_references"
        provenance["reference_assignment"]["reference_count"] = len(references)
    warmstart_checkpoint = str(Path(args.warmstart).expanduser().resolve()) if args.warmstart else None
    mode = "ppo_warmstart" if warmstart_checkpoint else "ppo_from_scratch"
    config = {key: value for key, value in vars(args).items() if key not in {"fn", "wandb"}}
    config["teacher_anchor"] = anchor; provenance["teacher_anchor"] = anchor
    warmstart_expected = {
        key: provenance[key] for key in (
            "asset_pin", "package_digest", "manifest_sha256", "catalog_digest",
            "identity_split", "contracts", "identity",
        )
    }
    warmstart_expected["clock"] = {
        key: provenance["clock"][key] for key in (
            "control_timestep", "physics_timestep", "physics_substeps",
        )
    }
    transfer_mode = None
    if warmstart_checkpoint is not None:
        _, transfer_mode = inspect_v4_warmstart(
            warmstart_checkpoint, actor_critic_architecture(separate_critic=args.separate_critic),
            expected_provenance=warmstart_expected,
        )
    lineage = {"separate_critic": bool(args.separate_critic), "warmstart_checkpoint": warmstart_checkpoint,
               "warmstart_transfer_mode": transfer_mode, "mode": mode}
    completed_updates = 0
    if args.resume_checkpoint:
        # Metadata and every model/Adam tensor are validated before creating a
        # W&B writer. A CPU model suffices; no B*rollout memory is allocated here.
        validation_model = AutonomyActorCritic(adapter.observation_space, adapter.action_space,
                                               device="cpu", separate_critic=args.separate_critic)
        validation_optimizer = torch.optim.Adam(validation_model.parameters(), lr=args.learning_rate)
        payload = inspect_v4_resume(args.resume_checkpoint, validation_model, validation_optimizer,
            expected_config=config, expected_provenance=provenance, updates=args.updates,
            cuda_device_count=torch.cuda.device_count() if args.device == "gpu" else None)
        completed_updates = payload["policy_steps"] // args.rollouts
        run_id = args.wandb_run_id or os.environ.get("WANDB_RUN_ID")
        if args.wandb:
            if not run_id:
                raise ValueError("W&B resume requires explicit --wandb-run-id or WANDB_RUN_ID")
            previous_id = payload["config"].get("wandb_run_id")
            if previous_id is not None and previous_id != run_id:
                raise ValueError("W&B resume run ID differs from checkpoint")
        lineage = {key: payload["provenance"][key] for key in ("warmstart_checkpoint", "warmstart_transfer_mode")
                   if key in payload["provenance"]}
        lineage.update(resume_lineage(args.resume_checkpoint, payload))
        del validation_model, validation_optimizer, payload
    config.update(lineage); provenance.update(lineage)
    config["wandb_run_id"] = args.wandb_run_id or os.environ.get("WANDB_RUN_ID")
    metadata = {"training_contract": "manorl.autonomy.training.v4.all_references" if args.all_references else ("manorl.autonomy.training.v4.multi_reference" if args.all_train_references else "manorl.autonomy.training.v4.single_reference"), "config": config,
                "resolved": _resolved_telemetry_config(adapter, args), "provenance": provenance}
    run = _wandb(args, metadata)
    try:
        metrics_path = Path(args.checkpoint).with_suffix(Path(args.checkpoint).suffix + ".metrics.jsonl")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        def publish(row):
            print(json.dumps(row), flush=True)
            # Local provenance stays available when W&B is offline/unavailable.
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
            if run is not None: run.log(row, step=int(row["transitions"]))
        _, _, rows = run_batched_ppo(adapter, updates=args.updates, rollouts=args.rollouts,
            learning_epochs=args.learning_epochs, mini_batches=args.mini_batches, learning_rate=args.learning_rate, checkpoint=args.checkpoint,
            checkpoint_interval=args.checkpoint_interval, config=config, provenance=provenance,
            separate_critic=args.separate_critic, warmstart=warmstart_checkpoint,
            teacher_anchor_beta=args.teacher_anchor_beta, teacher_anchor_passes=args.teacher_anchor_passes,
            resume_checkpoint=args.resume_checkpoint,
            expected_warmstart_provenance=warmstart_expected, on_update=publish)
    except BaseException:
        if run is not None: run.finish(exit_code=1)
        raise
    else:
        if run is not None: run.finish(exit_code=0)
    print(json.dumps({"checkpoint": args.checkpoint, "metrics": str(Path(args.checkpoint).with_suffix(Path(args.checkpoint).suffix + ".metrics.jsonl")), "updates": args.updates, "start_update": completed_updates, "completed_updates": completed_updates + len(rows), "additional_updates": len(rows), "cache_hash": provenance["cache_hash_recorded_not_compared"]}), flush=True)


def _checkpoint_separate_critic(payload):
    validate_v4_checkpoint_metadata(payload)
    architecture = payload.get("model_architecture")
    if architecture == actor_critic_architecture(separate_critic=False): return False
    if architecture == actor_critic_architecture(separate_critic=True): return True
    raise ValueError("checkpoint/model architecture mismatch")


def evaluate(args):
    if args.num_envs != 1: raise ValueError("frozen v4 evaluate requires --num-envs 1")
    if args.steps is not None and args.steps <= 0: raise ValueError("--steps must be positive")
    seed_everything(args.seed); catalog, trajectory = _catalog_and_trajectory(args)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    separate_critic = _checkpoint_separate_critic(payload)
    split = identity_split(catalog, seed=payload.get("provenance", {}).get("identity_split", {}).get("seed", args.split_seed))
    adapter = _adapter(args, trajectory, full_horizon_diagnostic=args.full_horizon_diagnostic)
    model = AutonomyActorCritic(adapter.observation_space, adapter.action_space, device=adapter.device,
                                separate_critic=separate_critic)
    # Identity is a conditioning input, not a compatibility contract: a frozen
    # checkpoint must accept references it was not trained on.  Compatibility
    # remains gated by asset/package/ABI/split; the identity itself is recorded
    # in the result provenance only.
    expected = {"asset_pin": _git_revision(ROOT / "assets/dexstream_digital_assets"),
                "package_digest": catalog.package_digest, "manifest_sha256": catalog.manifest_sha256,
                "catalog_digest": catalog.catalog_digest, "identity_split": split}
    payload = load_frozen_v4(args.checkpoint, model, map_location=adapter.device, expected_provenance=expected)
    observations, _ = adapter.reset(); total = 0.0; trace = []; natural_first = None; artifact = {"actual_object_position": [], "reference_object_position": [], "actual_palm_position": [], "reference_palm_position": [], "actual_qpos": [], "reference_qpos_raw": [], "reference_qpos_feasible": [], "reward_terms": [], "paired_force_on_object": [], "object_all_force": [], "bottom_clearance": [], "reason_code": []}
    limit = args.steps if args.steps is not None else adapter.runtime.length - 1
    for policy_step in range(limit):
        with torch.no_grad():
            mean, _ = model.compute({"observations": observations}, role="policy")
            action = torch.clamp(mean, -1., 1.)
        next_obs, reward, done, info = adapter.step(action); reward_value = float(reward[0, 0].cpu()); total += reward_value
        natural_done = bool(np.asarray(adapter.runtime.last_reward.done)[0])
        if natural_done and natural_first is None:
            natural_first = {"policy_step": policy_step, "reason_code": int(np.asarray(adapter.runtime.last_reason)[0])}
        i = min(int(np.asarray(adapter.runtime.indices)[0]), adapter.runtime.length - 1)
        physical, contact, diagnostics, cache = adapter.runtime.last_physical, adapter.runtime.last_contact, adapter.runtime.last_reward, adapter.runtime.cache
        reason_code = int(np.asarray(adapter.runtime.last_reason)[0])
        clearance = float(np.asarray(physical.object_bottom)[0] - cache.table_height)
        terms = [float(np.asarray(getattr(diagnostics, name))[0]) for name in ("object_position", "object_rotation", "object_velocity", "hand_relative", "fingers", "geometry", "action", "survival", "severe")]
        runtime_valid = bool(np.asarray(info["valid"])[0])
        finite = bool(np.isfinite(reward_value) and np.isfinite(clearance)
                      and np.isfinite(np.asarray(physical.object_origin)[0]).all()
                      and np.isfinite(np.asarray(terms)).all())
        trace.append({"policy_step": policy_step, "reward": reward_value,
                      "finite": finite, "valid": runtime_valid,
                      "natural_done": natural_done, "runtime_done": bool(done[0, 0].cpu()),
                      "reason_code": reason_code, "bottom_clearance_m": clearance,
                      "cache_hash": adapter.runtime.cache.content_hash})
        artifact["actual_object_position"].append(np.asarray(physical.object_origin)[0]); artifact["reference_object_position"].append(np.asarray(cache.object_origin)[i])
        artifact["actual_palm_position"].append(np.asarray(physical.palm_origin)[0]); artifact["reference_palm_position"].append(np.asarray(cache.palm_origin)[i])
        artifact["actual_qpos"].append(np.asarray(physical.q_raw)[0]); artifact["reference_qpos_raw"].append(np.asarray(cache.q_raw)[i]); artifact["reference_qpos_feasible"].append(np.asarray(cache.q_feasible)[i])
        artifact["reward_terms"].append(terms); artifact["paired_force_on_object"].append(np.asarray(contact.paired_force_on_object)[0]); artifact["object_all_force"].append(np.asarray(contact.object_all_force)[0]); artifact["bottom_clearance"].append(clearance); artifact["reason_code"].append(reason_code)
        if natural_done and not args.full_horizon_diagnostic: break
        observations = next_obs if args.full_horizon_diagnostic else adapter.prepare_action()
    result = {"format": "manorl.autonomy.frozen_evaluation.v4", "checkpoint": str(args.checkpoint),
              "identity": trajectory.identity.identity, "started_reference_frame": 0, "steps": len(trace),
              "return": total, "natural_first_termination": natural_first,
              "full_horizon_diagnostic": bool(args.full_horizon_diagnostic),
              "diagnostic_boundary": "natural termination recorded; continued after it" if args.full_horizon_diagnostic else "stopped at natural first termination",
              "natural_prefix_steps": len(trace) if natural_first is None else natural_first["policy_step"] + 1,
              "provenance": {"checkpoint": payload["provenance"], "evaluation_cache_hash_recorded_not_compared": adapter.runtime.cache.content_hash,
                             "evaluation_reward_parameters": reward_parameters(getattr(adapter.runtime.cache, "object_radius", REWARD_MOTION_RADIUS))}, "trace": trace}
    trace_path = Path(args.trace); trace_path.parent.mkdir(parents=True, exist_ok=True); trace_path.write_text(json.dumps(result, indent=2, default=_jsonable) + "\n")
    artifact_path = Path(args.artifact) if args.artifact else trace_path.with_suffix(".npz")
    artifact_path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(artifact_path, **{name: np.asarray(values) for name, values in artifact.items()}, reward_term_names=np.asarray(REWARD_NAMES), natural_prefix_steps=np.asarray(result["natural_prefix_steps"]), full_horizon_diagnostic=np.asarray(args.full_horizon_diagnostic))
    print(json.dumps({"trace": str(trace_path), "artifact": str(artifact_path), "steps": len(trace), "return": total, "natural_first_termination": natural_first}), flush=True)


def inspect(args):
    catalog, trajectory = _catalog_and_trajectory(args); adapter = _adapter(args, trajectory)
    print(json.dumps({"identity": trajectory.identity.identity, "raw_observation": list(adapter.reset()[0].shape),
        "cache_hash": adapter.runtime.cache.content_hash, "clock": adapter.runtime.clock_metadata,
        "workspace": args.persistentworkspace, "ccd_contacts_per_world": args.ccd_contacts_per_world,
        "constraint_capacity_per_world": adapter.runtime.warp_constraint_capacity}))


def build_parser():
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
    train_parser.add_argument("--learning-rate", type=float, default=3e-4, help="finite positive PPO Adam learning rate (default: 3e-4)")
    reference_group = train_parser.add_mutually_exclusive_group()
    reference_group.add_argument("--all-train-references", action="store_true", help="fixed round-robin assignment over all 40 deterministic TRAIN identities")
    reference_group.add_argument("--all-references", action="store_true", help="fixed round-robin assignment over every reference in the supplied package")
    train_parser.add_argument("--teacher-anchor-beta", type=float, default=0.0, help="optional post-PPO teacher-action MSE weight; training supervision only")
    train_parser.add_argument("--teacher-anchor-passes", type=int, default=2, help="full teacher-label minibatch passes after each PPO update")
    initialization = train_parser.add_mutually_exclusive_group()
    initialization.add_argument("--warmstart", help="strict v4 model-only checkpoint initialization; PPO uses a fresh full optimizer")
    initialization.add_argument("--resume-checkpoint", help="restore v4 model/Adam/RNG; --updates is the TOTAL cumulative target, with fresh full-start episodes")
    train_parser.add_argument("--wandb-run-id", help="existing W&B run ID; required explicitly or via WANDB_RUN_ID on resume")
    train_parser.add_argument("--separate-critic", action="store_true", help="use an independent value trunk; shared remains the default")
    train_parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True); train_parser.add_argument("--wandb-project"); train_parser.add_argument("--wandb-entity"); train_parser.add_argument("--wandb-mode"); train_parser.set_defaults(fn=train)
    eval_parser = subs.add_parser("evaluate", parents=[common]); eval_parser.set_defaults(num_envs=1); eval_parser.add_argument("--checkpoint", required=True); eval_parser.add_argument("--steps", type=int); eval_parser.add_argument("--full-horizon-diagnostic", action="store_true")
    eval_parser.add_argument("--trace", default="outputs/manorl/contact_conditioned_autonomy/cube2_02_v4_eval.json"); eval_parser.add_argument("--artifact", help="compressed physical frozen-evaluation trace (.npz); defaults beside --trace"); eval_parser.set_defaults(fn=evaluate)
    inspect_parser = subs.add_parser("inspect", parents=[common]); inspect_parser.set_defaults(fn=inspect)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv); return args.fn(args)

if __name__ == "__main__": raise SystemExit(main())
