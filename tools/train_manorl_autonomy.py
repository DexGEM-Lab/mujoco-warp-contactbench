#!/usr/bin/env python
"""Train/evaluate the standalone cube2 autonomous PPO milestone."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sim.manorl.autonomy import Cube2AutonomousMJX, PACKAGE_DEFAULT
from sim.manorl.autonomy_contracts import ACTION_DIM, OBSERVATION_DIM, CHECKPOINT_FORMAT
from sim.manorl.trajectory import TrajectorySelection
from sim.manorl.trajectory_package import load_assigned_trajectory_package

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__(); self.body=nn.Sequential(nn.Linear(OBSERVATION_DIM,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh()); self.mean=nn.Linear(256,ACTION_DIM); self.value=nn.Linear(256,1); self.log_std=nn.Parameter(torch.full((ACTION_DIM,),-1.2))
    def forward(self,x):
        h=self.body(x); mean=self.mean(h); return mean, self.value(h).squeeze(-1), self.log_std.expand_as(mean)

def load_env(package: Path, device: str, near_contact: bool, seed: int):
    selection=TrajectorySelection(object_type='cube2',gesture='02',dataset_path=package,expected_dataset_version=295,pre_padding=180,post_padding=180,hand_side='right',reference_fps=120,control_fps=120)
    batch=load_assigned_trajectory_package(package,selection,num_envs=1)
    return Cube2AutonomousMJX(batch.trajectories[0],device=device,near_contact=near_contact,seed=seed), batch.trajectories[0]

def train(args):
    env,traj=load_env(Path(args.package),args.device,args.near_contact,args.seed); model=ActorCritic(); opt=torch.optim.Adam(model.parameters(),lr=args.lr); obs=env.reset(); rows=[]
    for update in range(args.updates):
        ob=[]; ac=[]; lp=[]; rw=[]; val=[]; done=False
        for _ in range(args.horizon):
            x=torch.from_numpy(obs).float().unsqueeze(0); mean,v,ls=model(x); dist=torch.distributions.Normal(mean,ls.exp()); a=dist.sample(); act=torch.tanh(a).detach().numpy()[0]; step=env.step(act); ob.append(x[0]); ac.append(a[0].detach()); lp.append(dist.log_prob(a).sum().detach()); rw.append(step.reward); val.append(v.detach().squeeze()); obs=env.reset() if step.done else step.observation; done=step.done
        returns=[]; g=0.
        for r in reversed(rw): g=r+args.gamma*g; returns.append(g)
        returns=torch.tensor(returns[::-1]); ob_t=torch.stack(ob); ac_t=torch.stack(ac); old_lp=torch.stack(lp); old_v=torch.stack(val)
        for _ in range(args.epochs):
            mean,v,ls=model(ob_t); dist=torch.distributions.Normal(mean,ls.exp()); new_lp=dist.log_prob(ac_t).sum(-1); ratio=(new_lp-old_lp).exp(); adv=returns-old_v; adv=(adv-adv.mean())/(adv.std()+1e-8); policy=-torch.minimum(ratio*adv,torch.clamp(ratio,1-.2,1+.2)*adv).mean(); value=.5*(returns-v).square().mean(); loss=policy+value-.001*dist.entropy().sum(-1).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        rows.append({'update':update,'reward_mean':float(np.mean(rw)),'episode_done':done}); print(json.dumps(rows[-1]),flush=True)
    out=Path(args.checkpoint); out.parent.mkdir(parents=True,exist_ok=True); torch.save({'format':CHECKPOINT_FORMAT,'observation_contract':'manorl.autonomy.observation.v1','action_contract':'manorl.autonomy.action.v1','reward_contract':'manorl.autonomy.reward.v1','package_digest':load_assigned_trajectory_package(Path(args.package),TrajectorySelection(object_type='cube2',gesture='02',dataset_path=Path(args.package),expected_dataset_version=295,pre_padding=180,post_padding=180,hand_side='right',reference_fps=120,control_fps=120),num_envs=1).trajectory_package['package_digest'],'model':model.state_dict(),'config':vars(args)},out); out.with_suffix('.json').write_text(json.dumps({'format':CHECKPOINT_FORMAT,'identity':traj.identity.identity,'package':str(args.package),'package_manifest_sha256':'e826de23d4586611001230d340eaf0c68752b09988eed7928bf8884f056d8706','rows':rows},indent=2)+'\n'); return 0

def evaluate(args):
    env,traj=load_env(Path(args.package),args.device,False,args.seed); payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False); model=ActorCritic(); model.load_state_dict(payload['model']); model.eval(); obs=env.reset(); total=0.; trace=[]
    for _ in range(args.steps):
        with torch.no_grad(): mean,_,_=model(torch.from_numpy(obs).float().unsqueeze(0)); act=torch.tanh(mean)[0].numpy()
        step=env.step(act); total+=step.reward; trace.append({'step':step.info['step'],'reward':step.reward,'object_position':step.info['object_position'].tolist(),'surface_proximity_mean':float(np.mean(step.info['surface_proximity']))}); obs=env.reset() if step.done else step.observation
        if step.done: break
    out=Path(args.trace); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'identity':traj.identity.identity,'return':total,'steps':len(trace),'trace':trace},indent=2)+'\n'); print(json.dumps({'return':total,'steps':len(trace),'trace':str(out)})); return 0

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='mode',required=True); common=argparse.ArgumentParser(add_help=False); common.add_argument('--package',default=str(PACKAGE_DEFAULT)); common.add_argument('--device',choices=('cpu','gpu'),default='cpu'); common.add_argument('--seed',type=int,default=0)
 t=sub.add_parser('train',parents=[common]); t.add_argument('--updates',type=int,default=1); t.add_argument('--horizon',type=int,default=32); t.add_argument('--epochs',type=int,default=2); t.add_argument('--lr',type=float,default=3e-4); t.add_argument('--gamma',type=float,default=.99); t.add_argument('--near-contact',action='store_true'); t.add_argument('--checkpoint',default='outputs/manorl/contact_conditioned_autonomy/cube2_02_autonomy.pt'); t.set_defaults(fn=train)
 e=sub.add_parser('evaluate',parents=[common]); e.add_argument('--checkpoint',required=True); e.add_argument('--steps',type=int,default=128); e.add_argument('--trace',default='outputs/manorl/contact_conditioned_autonomy/eval_trace.json'); e.set_defaults(fn=evaluate)
 a=p.parse_args(); return a.fn(a)
if __name__=='__main__': raise SystemExit(main())
