#!/usr/bin/env python
"""Full-start physical diagnostics for the M2 autonomy contract.

PPO is intentionally absent: the previous smoke optimizer had no GAE or
bootstrap semantics and is refused until the next milestone wires canonical
training. This tool records diagnostic actors only.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from sim.manorl.autonomy import Cube2AutonomousMJX, PACKAGE_DEFAULT
from sim.manorl.autonomy_contracts import ACTION_CONTRACT_ID, OBSERVATION_CONTRACT_ID, REWARD_CONTRACT_ID, ACTION_DIM
from sim.manorl.trajectory_package import load_trajectory_package

def load_env(package: Path, device: str, near_contact: bool, seed: int, identity_index: int):
    catalog = load_trajectory_package(package)
    if identity_index < 0 or identity_index >= len(catalog.trajectories): raise IndexError(f"identity index must be in [0,{len(catalog.trajectories)-1}]")
    trajectory = catalog.trajectories[identity_index]
    return Cube2AutonomousMJX(trajectory, device=device, near_contact=near_contact, seed=seed), trajectory, catalog

def diagnostic(args):
    package=Path(args.package); env,traj,catalog=load_env(package,args.device,False,args.seed,args.identity_index); env.reset(); trace=[]; total=0.; max_steps=args.steps if args.steps is not None else len(traj.q_ref)-1
    for _ in range(max_steps):
        i=min(env.step_index,len(env.aligned_q_ref)-1)
        if args.actor == "zero": action=np.zeros(ACTION_DIM,dtype=float)
        else:
            # The diagnostic actor consumes the next reference target and its
            # own previous command. Its output alone enters the command map.
            target=env.aligned_q_ref[min(i+1,len(env.aligned_q_ref)-1)]
            action=np.clip((target-env.previous_command)/(env.rate_per_second*env.clock.control_timestep),-1.,1.)
        step=env.step(action); total+=step.reward; info=step.info
        trace.append({"step":info["step"],"reward":step.reward,"terms":step.terms,"qpos":info["qpos"].tolist(),"qvel":info["qvel"].tolist(),"reference_hand_q":env.aligned_q_ref[i].tolist(),"command":info["command"].tolist(),"object_position":info["object_position"].tolist(),"object_quaternion_xyzw":env._state()[4].tolist(),"target_object_position":info["target_object_position"].tolist(),"target_object_quaternion_xyzw":traj.object_quat_xyzw[i].tolist(),"path_error":info["path_error"],"lift_error":info["lift_error"],"hand_object_force":info["hand_object_force"].reshape(-1).tolist(),"supporting_object_net_force":info["supporting_object_net_force"].tolist(),"contact_count":info["contact_count"],"surface_proximity":info["surface_proximity"].tolist(),"reference_surface_proximity":info["reference_surface_proximity"].tolist(),"surface_anchor_local":info["surface_anchor_local"].reshape(-1).tolist(),"reference_surface_anchor_local":info["reference_surface_anchor_local"].reshape(-1).tolist(),"relative_contact_motion":info["relative_contact_motion"].reshape(-1).tolist(),"failure_phase":info["failure_phase"]})
        if step.done: break
    config_hash=hashlib.sha256(json.dumps({"actor":args.actor,"identity_index":args.identity_index,"seed":args.seed,"device":args.device,"steps":max_steps},sort_keys=True).encode()).hexdigest()
    payload={"format":"manorl.autonomy.diagnostic.v2","actor":args.actor,"identity":traj.identity.identity,"identity_index":args.identity_index,"source":{"package_digest":catalog.package_digest,"manifest_sha256":catalog.manifest_sha256,"catalog_digest":catalog.catalog_digest},"contracts":{"action":ACTION_CONTRACT_ID,"observation":OBSERVATION_CONTRACT_ID,"reward":REWARD_CONTRACT_ID},"clock":{"policy_fps":env.clock.policy_fps,"control_timestep":env.clock.control_timestep,"physics_fps":env.clock.physics_fps,"physics_timestep":env.clock.physics_timestep,"substeps":env.clock.physics_substeps_per_control},"config_hash":config_hash,"reference_pursuit_actor_has_no_post_output_reference":args.actor=="reference-pursuit","steps":len(trace),"return":total,"trace":trace}
    out=Path(args.trace);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload)+"\n"); print(json.dumps({"actor":args.actor,"identity":traj.identity.identity,"return":total,"steps":len(trace),"failure_phase":trace[-1]["failure_phase"] if trace else "none","trace":str(out)})); return 0

def package_summary(args):
    catalog=load_trajectory_package(Path(args.package)); rows=[]
    for t in catalog.trajectories: rows.append({"identity":t.identity.identity,"frames":len(t.q_ref),"object_extent_m":np.ptp(np.asarray(t.object_pos_raw),axis=0).tolist(),"raw_initial_object_m":np.asarray(t.object_pos_raw[0]).tolist()})
    Path(args.output).write_text(json.dumps({"format":"manorl.autonomy.package-summary.v2","source":catalog.checkpoint_metadata,"count":len(rows),"rows":rows},indent=2)+"\n"); print(json.dumps({"count":len(rows),"output":args.output})); return 0

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="mode",required=True); common=argparse.ArgumentParser(add_help=False); common.add_argument("--package",default=str(PACKAGE_DEFAULT)); common.add_argument("--device",choices=("cpu","gpu"),default="cpu"); common.add_argument("--seed",type=int,default=0); common.add_argument("--identity-index",type=int,default=0)
    d=sub.add_parser("diagnostic",parents=[common]);d.add_argument("--actor",choices=("zero","reference-pursuit"),required=True);d.add_argument("--steps",type=int,default=None);d.add_argument("--trace",default="outputs/manorl/contact_conditioned_autonomy/diagnostic_v2.json");d.set_defaults(fn=diagnostic)
    s=sub.add_parser("package-summary",parents=[common]);s.add_argument("--output",default="outputs/manorl/contact_conditioned_autonomy/cube2_02_summary_v2.json");s.set_defaults(fn=package_summary)
    a=p.parse_args();return a.fn(a)
if __name__=="__main__":raise SystemExit(main())
