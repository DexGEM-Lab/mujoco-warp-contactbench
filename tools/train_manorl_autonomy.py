#!/usr/bin/env python
"""Inspection-only v4 autonomy CLI. Training is intentionally stopped."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sim.manorl.autonomy_contracts import CHECKPOINT_FORMAT,OBSERVATION_CONTRACT_ID,ACTION_CONTRACT_ID,REWARD_CONTRACT_ID,RAW_OBSERVATION_DIM
from sim.manorl.autonomy_training import AutonomyActorCritic
from sim.manorl.trajectory_package import load_trajectory_package

DEFAULT_PACKAGE="/home/jay/dexrobot/FromSSH/manoRL_mujoco/outputs/manorl/contact_conditioned_autonomy/cube2_02_v295_f120_pre180_post180"
def _trajectory(path,identity):
    return next(t for t in load_trajectory_package(path).trajectories if t.identity.identity==identity)
def inspect(args):
    t=_trajectory(args.package,args.identity)
    from sim.manorl.autonomy_batch import BatchedAutonomyRuntime
    r=BatchedAutonomyRuntime(t,num_envs=1,device=args.device)
    print(json.dumps({"contract":OBSERVATION_CONTRACT_ID,"identity":args.identity,"raw_shape":list(r.raw_observation.shape),"cache_hash":r.cache.content_hash,"control_dt":1/120,"training":"stopped"}))
def smoke(args):
    t=_trajectory(args.package,args.identity)
    from sim.manorl.autonomy_training import BatchedAutonomyAdapter
    adapter=BatchedAutonomyAdapter(t,num_envs=1,device=args.device)
    obs,_=adapter.reset(); model=AutonomyActorCritic(adapter.observation_space,adapter.action_space,device=adapter.device)
    if args.checkpoint:
        payload=torch.load(args.checkpoint,map_location=adapter.device,weights_only=False)
        required={"checkpoint_format":CHECKPOINT_FORMAT,"observation_contract":OBSERVATION_CONTRACT_ID,"action_contract":ACTION_CONTRACT_ID,"reward_contract":REWARD_CONTRACT_ID}
        if any(payload.get(k)!=v for k,v in required.items()): raise ValueError("checkpoint is not v4")
        model.load_state_dict(payload["model"],strict=True)
    with torch.no_grad(): action,_=model.compute({"observations":obs},role="policy")
    _,reward,done,info=adapter.step(action)
    print(json.dumps({"raw_shape":list(obs.shape),"action_shape":list(action.shape),"reward":float(reward[0]),"done":bool(done[0]),"valid":bool(np.asarray(info['valid'])[0]),"training":"stopped"}))
def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(required=True)
    for name,fn in (("inspect",inspect),("smoke",smoke)):
        q=sub.add_parser(name); q.add_argument("--package",default=DEFAULT_PACKAGE); q.add_argument("--identity",default="cube2_02_2833"); q.add_argument("--device",choices=("cpu","gpu"),default="cpu"); q.set_defaults(fn=fn)
        if name=="smoke": q.add_argument("--checkpoint")
    args=p.parse_args(); args.fn(args)
if __name__=="__main__": main()
