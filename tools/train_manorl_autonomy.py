#!/usr/bin/env python
"""M3 formal skrl PPO train/evaluate entrypoint for cube2 autonomy."""
from __future__ import annotations
import argparse, json, os, hashlib, re, sys, subprocess
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import numpy as np
import torch
from sim.manorl.autonomy import Cube2AutonomousMJX, PACKAGE_DEFAULT
from sim.manorl.autonomy_contracts import ACTION_CONTRACT_ID, OBSERVATION_CONTRACT_ID, REWARD_CONTRACT_ID, ACTION_V3_CONTRACT_ID, OBSERVATION_V3_CONTRACT_ID, REWARD_V3_CONTRACT_ID
from sim.manorl.autonomy_training import AutonomyVectorEnv, BatchedAutonomyAdapter, build_runtime, identity_split, seed_everything, TRAINING_CONTRACT_ID
from sim.manorl.autonomy_batch_training import load_frozen_v3, run_batched_ppo
from sim.manorl.autonomy_telemetry import TelemetryAccumulator, configure_wandb_axis, log_update
from sim.manorl.trajectory_package import load_trajectory_package

def _catalog(path): return load_trajectory_package(Path(path))
def _jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)): return value.item()
    if isinstance(value, dict): return {str(k): _jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value
def _env(args,catalog,identity_index): return AutonomyVectorEnv(catalog.trajectories[identity_index],device=args.device,seed=args.seed,contact_conditioned=args.setting=="contact-conditioned")
def formal_rollout_schedule(*, horizon, rollouts, updates):
    """Reference schedule for explicit reset-after-terminal loop tests."""
    phase=0; records=[]; reset_count=1
    for _ in range(updates):
        for _ in range(rollouts):
            phase += 1; terminal = phase >= horizon; records.append((phase, terminal))
            if terminal: phase=0; reset_count += 1
    return records, reset_count
def _wandb_start(args, metadata):
    if not args.wandb: return None
    import wandb
    # The default is the canonical entity/project path; both are overrideable
    # for offline smoke tests or a separately configured project.
    run = wandb.init(
        project=getattr(args, "wandb_project", None) or os.environ.get("WANDB_PROJECT", "mujoco-mano"),
        entity=getattr(args, "wandb_entity", None) or os.environ.get("WANDB_ENTITY", "sunjay45711-dexerto"),
        config=metadata,
        mode=getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE", "online"),
        reinit=True,
    )
    configure_wandb_axis(run)
    return run

def formaltrain(args):
    seed_everything(args.seed)
    catalog = _catalog(args.package)
    split = identity_split(catalog, seed=args.split_seed)
    idx = args.identity_index
    if args.num_envs != 1:
        raise ValueError("M3 first learner is explicitly N=1; batched scaling is a later milestone")
    if args.total_transitions is not None and args.total_transitions != args.updates * args.rollouts * args.num_envs:
        raise ValueError("total-transitions must equal updates * rollouts * num-envs")
    if idx not in split["train_indices"]:
        raise ValueError("formaltrain identity must belong to deterministic TRAIN split")
    env = _env(args, catalog, idx)
    wrapper, model, agent = build_runtime(
        env, rollouts=args.rollouts, learning_epochs=args.learning_epochs,
        device="cpu" if args.device == "cpu" else "cuda",
    )
    identity = catalog.trajectories[idx].identity.identity
    metadata = {
        "training_contract": TRAINING_CONTRACT_ID, "setting": args.setting,
        "identity": identity, "identity_index": idx, "split": split,
        "contracts": {"action": ACTION_CONTRACT_ID, "observation": OBSERVATION_CONTRACT_ID, "reward": REWARD_CONTRACT_ID},
        "package_digest": catalog.package_digest, "manifest_sha256": catalog.manifest_sha256,
        "seed": args.seed, "split_seed": args.split_seed, "num_envs": 1,
        "rollouts": args.rollouts, "total_transitions": args.updates * args.rollouts,
        "checkpoint_path": str(args.checkpoint),
        "source": {"training_cli": "tools/train_manorl_autonomy.py", "training_contract": TRAINING_CONTRACT_ID},
    }
    run = _wandb_start(args, metadata)
    if run is not None and hasattr(run, "summary"):
        run.summary.update({"lineage/run_id": getattr(run, "id", None), "lineage/run_url": getattr(run, "url", None), "lineage/identity": identity})
    try:
        observations, _ = wrapper.reset()
        rows = []
        telemetry = TelemetryAccumulator()
        for update in range(args.updates):
            update_started = __import__("time").perf_counter()
            terminations = []
            truncations = []
            for timestep in range(args.rollouts):
                with torch.no_grad():
                    actions, _ = agent.act(observations, None, timestep=timestep, timesteps=args.rollouts)
                next_obs, reward, terminated, truncated, infos = wrapper.step(actions)
                rt = torch.as_tensor(reward); td = torch.as_tensor(terminated); tr = torch.as_tensor(truncated)
                agent.record_transition(observations=observations, states=None, actions=actions, rewards=rt,
                    next_observations=next_obs, next_states=None, terminated=td, truncated=tr, infos=infos,
                    timestep=timestep, timesteps=args.rollouts)
                agent.post_interaction(timestep=timestep + 1, timesteps=args.rollouts)
                info = infos if isinstance(infos, dict) else infos[0]
                telemetry.add(info, float(rt.mean()), info.get("terms", {}), terminated=bool(td.any()), truncated=bool(tr.any()))
                terminations.append(bool(td.any())); truncations.append(bool(tr.any())); observations = next_obs
                if bool((td | tr).any()): observations, _ = wrapper.reset()
            row = telemetry.reduce(update=update + 1, transitions=(update + 1) * args.rollouts,
                window_transitions=args.rollouts, update_elapsed_seconds=__import__("time").perf_counter() - update_started, agent=agent)
            row.update({"terminated": float(sum(terminations)), "truncated": float(sum(truncations))})
            rows.append(row); print(json.dumps(row, default=_jsonable), flush=True); log_update(run, row)
            telemetry.clear()
        out = Path(args.checkpoint); out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "manorl.autonomy.formalppo.v1", "training_contract": TRAINING_CONTRACT_ID,
            "contracts": {"action": ACTION_CONTRACT_ID, "observation": OBSERVATION_CONTRACT_ID, "reward": REWARD_CONTRACT_ID},
            "package": {"digest": catalog.package_digest, "manifest_sha256": catalog.manifest_sha256, "catalog_digest": catalog.catalog_digest},
            "split": split, "identity_index": idx, "identity": identity, "setting": args.setting, "seed": args.seed,
            "config": {k: v for k, v in vars(args).items() if k != "fn"}, "model": model.state_dict(),
            "optimizer": agent.optimizer.state_dict(), "normalizer": agent._observation_preprocessor.state_dict() if hasattr(agent._observation_preprocessor, "state_dict") else None,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": __import__("random").getstate(), "rows": rows}
        torch.save(payload, out)
        out.with_suffix(".json").write_text(json.dumps({k: v for k, v in payload.items() if k not in {"model", "optimizer", "normalizer", "torch_rng", "numpy_rng"}}, default=str, indent=2) + "\n")
        print(json.dumps({"checkpoint": str(out), "run_id": None if run is None else run.id, "run_url": None if run is None else run.url}))
    except BaseException:
        if run is not None:
            try: run.finish(exit_code=1)
            except BaseException: pass
        raise
    else:
        if run is not None: run.finish()
    return 0

def _git_revision(path: Path) -> str:
    deployed_commit = path / "DEPLOYED_COMMIT"
    if deployed_commit.exists():
        revision = deployed_commit.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(
                f"{deployed_commit} must contain one full 40-character lowercase hexadecimal commit"
            )
        return revision
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def batchtrain(args):
    """Launch direct DLPack batched v3 runtime training with canonical PPO."""
    seed_everything(args.seed)
    catalog = _catalog(args.package); split = identity_split(catalog, seed=args.split_seed)
    if args.identity_index not in split["train_indices"]:
        raise ValueError("batchtrain identity must belong to deterministic TRAIN split")
    if args.total_transitions is not None and args.total_transitions != args.num_envs * args.rollouts * args.updates:
        raise ValueError("total-transitions must equal num-envs * rollouts * updates")
    trajectory = catalog.trajectories[args.identity_index]
    adapter = BatchedAutonomyAdapter(trajectory, num_envs=args.num_envs, device=args.device, seed=args.seed,
                                     persistent_ccd_workspace=args.persistentworkspace,
                                     ccd_contacts_per_world=args.ccd_contacts_per_world)
    config = {k: v for k, v in vars(args).items() if k != "fn"}
    provenance = {"source": "tools/train_manorl_autonomy.py", "source_commit": _git_revision(_ROOT),
                  "asset_pin": _git_revision(_ROOT / "assets/dexstream_digital_assets"),
                  "package_digest": catalog.package_digest, "manifest_sha256": catalog.manifest_sha256,
                  "catalog_digest": catalog.catalog_digest, "identity_split": split,
                  "witness_digest": adapter.runtime.witness.digest,
                  "clock": {"policy_fps": adapter.runtime.clock.policy_fps, "physics_fps": adapter.runtime.clock.physics_fps, "substeps": adapter.runtime.clock.physics_substeps_per_control},
                  "v3_contract": {"action": ACTION_V3_CONTRACT_ID, "observation": OBSERVATION_V3_CONTRACT_ID, "reward": REWARD_V3_CONTRACT_ID},
                  "ppo": {"learning_epochs": args.learning_epochs, "mini_batches": args.mini_batches, "rollouts": args.rollouts}}
    metadata = {**provenance, "config": config, "identity": trajectory.identity.identity}
    run = _wandb_start(args, metadata)
    if run is not None:
        print(json.dumps({"wandb_run_id": run.id, "wandb_run_url": run.url, "identity": trajectory.identity.identity}), flush=True)
    def publish(row):
        print(json.dumps(row), flush=True)
        log_update(run, row)
    try:
        run_batched_ppo(adapter, updates=args.updates, rollouts=args.rollouts,
            learning_epochs=args.learning_epochs, mini_batches=args.mini_batches, checkpoint=args.checkpoint,
            checkpoint_interval=args.checkpoint_interval, config=config, provenance=provenance,
            on_update=publish)
    except BaseException:
        if run is not None: run.finish(exit_code=1)
        raise
    else:
        if run is not None: run.finish(exit_code=0)
    return 0


def batchevaluate(args):
    """Evaluate a frozen v3 model on a full-start runtime without optimisation."""
    seed_everything(args.seed); catalog = _catalog(args.package)
    trajectory = catalog.trajectories[args.identity_index]
    adapter = BatchedAutonomyAdapter(trajectory, num_envs=args.num_envs, device=args.device, seed=args.seed,
                                     persistent_ccd_workspace=args.persistentworkspace,
                                     ccd_contacts_per_world=args.ccd_contacts_per_world)
    from sim.manorl.autonomy_training import AutonomyActorCritic
    model = AutonomyActorCritic(adapter.observation_space, adapter.action_space, device=str(adapter.device))
    expected_provenance = {"package_digest": catalog.package_digest, "manifest_sha256": catalog.manifest_sha256,
                           "catalog_digest": catalog.catalog_digest, "identity_split": identity_split(catalog, seed=args.split_seed),
                           "witness_digest": adapter.runtime.witness.digest}
    payload = load_frozen_v3(args.checkpoint, model, map_location=adapter.device, expected_provenance=expected_provenance)
    observations, _ = adapter.reset(); steps = adapter.runtime.length - 1 if args.steps is None else args.steps
    trace, total = [], torch.zeros((args.num_envs, 1), device=adapter.device)
    for policy_step in range(steps):
        with torch.no_grad():
            mean, _ = model.compute({"observations": observations}, role="policy"); action = torch.clamp(mean, -1., 1.)
        terminal_next, reward, done, info = adapter.step(action); total += reward
        # Evaluation is N=1 by default; compact state copies here are deliberate
        # output artifacts rather than training fast-path telemetry.
        physical, contact = adapter.runtime.last_physical, adapter.runtime.last_contact
        trace.append({"policy_step": policy_step, "reward": float(reward[0, 0].detach().cpu()), "done": bool(done[0, 0].detach().cpu()),
                      "actual_q": np.asarray(physical.mano_dof_pos[0]).tolist(),
                      "object_position": np.asarray(physical.object_position[0]).tolist(),
                      "object_quaternion_xyzw": np.asarray(physical.object_orientation_xyzw[0]).tolist(),
                      "target_object_position": np.asarray(adapter.runtime.reference_obj[min(int(np.asarray(adapter.runtime.indices[0])), adapter.runtime.length - 1)]).tolist(),
                      "target_object_quaternion_xyzw": np.asarray(adapter.runtime.reference_quat[min(int(np.asarray(adapter.runtime.indices[0])), adapter.runtime.length - 1)]).tolist(),
                      "reference_q": np.asarray(adapter.runtime.reference_q[min(int(np.asarray(adapter.runtime.indices[0])), adapter.runtime.length - 1)]).tolist(),
                      "contact_force": np.asarray(contact.hand_object_forces[0]).tolist(),
                      "path": float(np.linalg.norm(np.asarray(physical.object_position[0]) - np.asarray(adapter.runtime.reference_obj[min(int(np.asarray(adapter.runtime.indices[0])), adapter.runtime.length - 1)]))),
                      "termination_reason": int(np.asarray(adapter.runtime.last_reason[0]))})
        observations = adapter.prepare_action()
        if bool(done.all().detach().cpu()): break
    out = {"format": "manorl.autonomy.batch-eval.v3", "checkpoint": str(args.checkpoint), "global_policy_step": payload["global_policy_step"], "identity": trajectory.identity.identity,
           "num_envs": args.num_envs, "steps": len(trace), "return": np.asarray(total.detach().cpu()).reshape(-1).tolist(), "trace": trace}
    Path(args.trace).parent.mkdir(parents=True, exist_ok=True); Path(args.trace).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"trace": args.trace, "steps": len(trace), "return": out["return"]}), flush=True)
    return 0


def evaluate(args):
    seed_everything(args.seed); catalog=_catalog(args.package); payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False); split=identity_split(catalog,seed=payload["split"]["seed"])
    if payload.get("format")!="manorl.autonomy.formalppo.v1" or payload.get("contracts") != {"action":ACTION_CONTRACT_ID,"observation":OBSERVATION_CONTRACT_ID,"reward":REWARD_CONTRACT_ID}: raise ValueError("incompatible autonomy PPO checkpoint contract")
    if payload["package"]["digest"] != catalog.package_digest or payload["split"]["digest"] != split["digest"]: raise ValueError("checkpoint package or split provenance mismatch")
    idx=args.identity_index
    if idx not in split["validation_indices"]+split["test_indices"] and not args.allow_train_eval: raise ValueError("evaluation identity must be VAL/TEST; pass --allow-train-eval for training identity")
    eval_args=argparse.Namespace(**vars(args)); eval_args.setting=payload.get("setting",args.setting); env=_env(eval_args,catalog,idx); runtime_device="cuda" if args.device=="gpu" else "cpu"; wrapper,model,agent=build_runtime(env,rollouts=2,learning_epochs=1,device=runtime_device); model.load_state_dict(payload["model"])
    if payload.get("normalizer") is not None and hasattr(agent._observation_preprocessor,"load_state_dict"): agent._observation_preprocessor.load_state_dict(payload["normalizer"])
    model.eval(); obs,_=wrapper.reset(); trace=[]; total=0.
    for _ in range(args.steps):
        with torch.no_grad(): inputs={"observations":agent._observation_preprocessor(obs),"states":None}; mean,_=model.compute(inputs,role="policy"); action=torch.clamp(mean,-1.,1.)
        obs,reward,terminated,truncated,info=wrapper.step(action); total+=float(reward.mean()); trace.append({"reward":float(reward.mean()),"terminated":bool(terminated.any()),"truncated":bool(truncated.any()),"info":_jsonable(info)})
        if bool((terminated|truncated).any()): break
    Path(args.trace).write_text(json.dumps({"format":"manorl.autonomy.formalppo-eval.v1","checkpoint_format":payload["format"],"identity":catalog.trajectories[idx].identity.identity,"identity_index":idx,"split_role":"train" if idx in split["train_indices"] else "validation" if idx in split["validation_indices"] else "test","contracts":payload["contracts"],"package":payload["package"],"steps":len(trace),"return":total,"trace":trace},indent=2)+"\n"); print(json.dumps({"identity":catalog.trajectories[idx].identity.identity,"steps":len(trace),"return":total,"trace":args.trace})); return 0

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="mode",required=True); common=argparse.ArgumentParser(add_help=False); common.add_argument("--package",default=str(PACKAGE_DEFAULT)); common.add_argument("--device",choices=("cpu","gpu"),default="cpu"); common.add_argument("--seed",type=int,default=0); common.add_argument("--split-seed",type=int,default=0); common.add_argument("--identity-index",type=int,default=0); common.add_argument("--setting",choices=("contact-conditioned","state-only"),default="contact-conditioned")
    t=sub.add_parser("formaltrain",parents=[common]); t.add_argument("--updates",type=int,default=1); t.add_argument("--num-envs",type=int,default=1); t.add_argument("--rollouts",type=int,default=2); t.add_argument("--total-transitions",type=int,default=None); t.add_argument("--learning-epochs",type=int,default=1); t.add_argument("--checkpoint",default="outputs/manorl/contact_conditioned_autonomy/cube2_02_formalppo.pt"); t.add_argument("--wandb",action=argparse.BooleanOptionalAction,default=True); t.add_argument("--wandb-project",default=None); t.add_argument("--wandb-entity",default=None); t.add_argument("--wandb-mode",default=None); t.set_defaults(fn=formaltrain)
    b=sub.add_parser("batchtrain",parents=[common]); b.add_argument("--num-envs",type=int,default=8192); b.add_argument("--rollouts",type=int,default=32); b.add_argument("--updates",type=int,default=256); b.add_argument("--total-transitions",type=int,default=None); b.add_argument("--learning-epochs",type=int,default=4); b.add_argument("--mini-batches",type=int,default=16); b.add_argument("--persistentworkspace",action=argparse.BooleanOptionalAction,default=True); b.add_argument("--ccd-contacts-per-world",type=int,default=None); b.add_argument("--checkpoint-interval",type=int,default=16); b.add_argument("--checkpoint",default="outputs/manorl/contact_conditioned_autonomy/batchppo-v3.pt"); b.add_argument("--wandb",action=argparse.BooleanOptionalAction,default=True); b.add_argument("--wandb-project",default=None); b.add_argument("--wandb-entity",default=None); b.add_argument("--wandb-mode",default=None); b.set_defaults(fn=batchtrain)
    be=sub.add_parser("batchevaluate",parents=[common]); be.add_argument("--num-envs",type=int,default=1); be.add_argument("--persistentworkspace",action=argparse.BooleanOptionalAction,default=True); be.add_argument("--ccd-contacts-per-world",type=int,default=None); be.add_argument("--checkpoint",required=True); be.add_argument("--steps",type=int,default=None); be.add_argument("--trace",default="outputs/manorl/contact_conditioned_autonomy/batchppo-v3-eval.json"); be.set_defaults(fn=batchevaluate)
    e=sub.add_parser("evaluate",parents=[common]); e.add_argument("--checkpoint",required=True); e.add_argument("--steps",type=int,default=8); e.add_argument("--trace",default="outputs/manorl/contact_conditioned_autonomy/formalppo_eval.json"); e.add_argument("--allow-train-eval",action="store_true"); e.set_defaults(fn=evaluate)
    a=p.parse_args(); return a.fn(a)
if __name__=="__main__": raise SystemExit(main())
