#!/usr/bin/env python
"""M3 formal skrl PPO train/evaluate entrypoint for cube2 autonomy."""
from __future__ import annotations
import argparse, json, os, hashlib
from pathlib import Path
import numpy as np
import torch
from sim.manorl.autonomy import Cube2AutonomousMJX, PACKAGE_DEFAULT
from sim.manorl.autonomy_contracts import ACTION_CONTRACT_ID, OBSERVATION_CONTRACT_ID, REWARD_CONTRACT_ID
from sim.manorl.autonomy_training import AutonomyVectorEnv, build_runtime, identity_split, seed_everything, TRAINING_CONTRACT_ID
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
    run=wandb.init(project=os.environ.get("WANDB_PROJECT","manorl-contact-conditioned-autonomy"),entity=os.environ.get("WANDB_ENTITY"),config=metadata,mode=os.environ.get("WANDB_MODE","online"),reinit=True)
    return run

def formaltrain(args):
    seed_everything(args.seed); catalog=_catalog(args.package); split=identity_split(catalog,seed=args.split_seed); idx=args.identity_index
    if args.num_envs != 1: raise ValueError("M3 first learner is explicitly N=1; batched scaling is a later milestone")
    if args.total_transitions is not None and args.total_transitions != args.updates * args.rollouts * args.num_envs: raise ValueError("total-transitions must equal updates * rollouts * num-envs")
    if idx not in split["train_indices"]: raise ValueError("formaltrain identity must belong to deterministic TRAIN split")
    env=_env(args,catalog,idx); wrapper,model,agent=build_runtime(env,rollouts=args.rollouts,learning_epochs=args.learning_epochs,device="cpu" if args.device=="cpu" else "cuda")
    metadata={"training_contract":TRAINING_CONTRACT_ID,"setting":args.setting,"identity":catalog.trajectories[idx].identity.identity,"identity_index":idx,"split":split,"contracts":{"action":ACTION_CONTRACT_ID,"observation":OBSERVATION_CONTRACT_ID,"reward":REWARD_CONTRACT_ID},"package_digest":catalog.package_digest,"manifest_sha256":catalog.manifest_sha256,"seed":args.seed,"split_seed":args.split_seed,"num_envs":1,"rollouts":args.rollouts,"total_transitions":args.updates*args.rollouts}; run=_wandb_start(args,metadata)
    observations,_=wrapper.reset(); rows=[]
    for update in range(args.updates):
        rewards=[]; terminations=[]; truncations=[]
        for timestep in range(args.rollouts):
            with torch.no_grad(): actions,_=agent.act(observations,None,timestep=timestep,timesteps=args.rollouts)
            next_obs,reward,terminated,truncated,infos=wrapper.step(actions); rt=torch.as_tensor(reward); td=torch.as_tensor(terminated); tr=torch.as_tensor(truncated)
            agent.record_transition(observations=observations,states=None,actions=actions,rewards=rt,next_observations=next_obs,next_states=None,terminated=td,truncated=tr,infos=infos,timestep=timestep,timesteps=args.rollouts); agent.post_interaction(timestep=timestep+1,timesteps=args.rollouts)
            rewards.append(float(rt.mean())); terminations.append(bool(td.any())); truncations.append(bool(tr.any())); observations=next_obs
            if bool((td | tr).any()): observations,_=wrapper.reset()
        row={"update":update,"reward_mean":float(np.mean(rewards)),"terminated":sum(terminations),"truncated":sum(truncations),"transitions":(update+1)*args.rollouts}; rows.append(row); print(json.dumps(row),flush=True); run and run.log(row)
    out=Path(args.checkpoint); out.parent.mkdir(parents=True,exist_ok=True); payload={"format":"manorl.autonomy.formalppo.v1","training_contract":TRAINING_CONTRACT_ID,"contracts":{"action":ACTION_CONTRACT_ID,"observation":OBSERVATION_CONTRACT_ID,"reward":REWARD_CONTRACT_ID},"package":{"digest":catalog.package_digest,"manifest_sha256":catalog.manifest_sha256,"catalog_digest":catalog.catalog_digest},"split":split,"identity_index":idx,"identity":catalog.trajectories[idx].identity.identity,"setting":args.setting,"seed":args.seed,"config":{k:v for k,v in vars(args).items() if k != "fn"},"model":model.state_dict(),"optimizer":agent.optimizer.state_dict(),"normalizer":agent._observation_preprocessor.state_dict() if hasattr(agent._observation_preprocessor,"state_dict") else None,"torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":__import__("random").getstate(),"rows":rows}; torch.save(payload,out); out.with_suffix(".json").write_text(json.dumps({k:v for k,v in payload.items() if k not in {"model","optimizer","normalizer","torch_rng","numpy_rng"}},default=str,indent=2)+"\n"); print(json.dumps({"checkpoint":str(out),"run_id":None if run is None else run.id,"run_url":None if run is None else run.url})); run and run.finish(); return 0

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
    t=sub.add_parser("formaltrain",parents=[common]); t.add_argument("--updates",type=int,default=1); t.add_argument("--num-envs",type=int,default=1); t.add_argument("--rollouts",type=int,default=2); t.add_argument("--total-transitions",type=int,default=None); t.add_argument("--learning-epochs",type=int,default=1); t.add_argument("--checkpoint",default="outputs/manorl/contact_conditioned_autonomy/cube2_02_formalppo.pt"); t.add_argument("--wandb",action=argparse.BooleanOptionalAction,default=True); t.set_defaults(fn=formaltrain)
    e=sub.add_parser("evaluate",parents=[common]); e.add_argument("--checkpoint",required=True); e.add_argument("--steps",type=int,default=8); e.add_argument("--trace",default="outputs/manorl/contact_conditioned_autonomy/formalppo_eval.json"); e.add_argument("--allow-train-eval",action="store_true"); e.set_defaults(fn=evaluate)
    a=p.parse_args(); return a.fn(a)
if __name__=="__main__": raise SystemExit(main())
