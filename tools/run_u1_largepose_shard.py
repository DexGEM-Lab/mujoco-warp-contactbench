#!/usr/bin/env python3
"""Disjoint slot shard for the large-pose campaign; physics/runtime unchanged."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from sim.manorl.u1_campaign import Ledger,child_uuid,digest,file_sha
from tools.run_u1_largepose_campaign import (VERSION,SOURCE_REGISTRY,PREFIX_FRAMES,
                                             build_plan,LargePoseRuntime)


def main():
 p=argparse.ArgumentParser();p.add_argument('--action',required=True);p.add_argument('--main-ledger',type=Path,required=True);p.add_argument('--shard-root',type=Path,required=True);p.add_argument('--slot-start',type=int,required=True);p.add_argument('--shard-index',type=int,required=True);p.add_argument('--shard-count',type=int,required=True);a=p.parse_args()
 if not(0<=a.shard_index<a.shard_count):raise ValueError('bad shard index')
 root=a.shard_root/f'shard{a.shard_index}';root.mkdir(parents=True,exist_ok=True)
 identity=digest(dict(version=VERSION,action=a.action,registry_sha=file_sha(SOURCE_REGISTRY),implementation={f:file_sha(Path(__file__).resolve().parents[1]/f)for f in('sim/manorl/u1_largepose.py','tools/run_u1_largepose_campaign.py')}))
 ledger=Ledger(root/(a.action+'.jsonl'),identity)
 mainrows=[json.loads(x)for x in a.main_ledger.read_text().splitlines()] if a.main_ledger.exists()else[]
 attempted={x['uuid']for x in mainrows if'uuid'in x}|{x['uuid']for x in ledger.rows if'uuid'in x}
 runtime=LargePoseRuntime(SOURCE_REGISTRY,a.action);slots=[s for s in build_plan()if s['slot']>=a.slot_start and(s['slot']-a.slot_start)%a.shard_count==a.shard_index]
 for slot in slots:
  sid=slot['slot']
  if sid in ledger.selected():continue
  for cand in slot['candidates']:
   uid=child_uuid(dict(version=VERSION,registry=file_sha(SOURCE_REGISTRY)),a.action,digest(slot),sid,cand)
   if uid in attempted:continue
   folder=root/a.action/uid;ledger.append(uuid=uid,slot=sid,candidate=cand,status='started');attempted.add(uid)
   try:
    delta=np.asarray(cand['delta'],float)
    from sim.manorl.start_augmentation import scene_contacts
    q0=runtime.I.initial['qpos'].copy();q0[:6]+=delta
    contacts=scene_contacts(runtime.m,q0,runtime.I.initial['qvel'],runtime.p['names'])
    penetrating=[c for c in contacts if c['distance_m']<0]
    inside=bool(np.all(q0[:6]>=runtime.m.jnt_range[:6,0])and np.all(q0[:6]<=runtime.m.jnt_range[:6,1]))
    if penetrating or not inside:raise ValueError(f'initial geometry preflight failed: penetration={penetrating}, wrist_in_range={inside}')
    first=runtime.replay(delta,PREFIX_FRAMES,folder/'first')
    if not first['accepted']:raise ValueError('first-pass gates failed')
    cmd=[sys.executable,'-m','tools.run_u1_largepose_campaign','second-pass','--action',a.action,'--staging',str(root),'--uuid',uid,'--frozen',str(folder/'first/target.npy')]
    with(folder/'second-process.log').open('w')as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    second=json.loads((folder/'second/result.json').read_text())
    if not second['accepted']or second['pid']==first['pid']:raise ValueError('independent gates failed')
    artifacts={str(x.resolve()):file_sha(x)for x in folder.rglob('*')if x.is_file()};ledger.append(uuid=uid,slot=sid,status='selected',artifacts=artifacts);break
   except Exception as e:ledger.append(uuid=uid,slot=sid,status='rejected',reason=repr(e))
  if sid not in ledger.selected():ledger.append(slot=sid,status='exhausted');raise RuntimeError(f'exhausted slot{sid}')
 expected={s['slot']for s in slots};selected=set(ledger.selected())
 if selected!=expected:raise RuntimeError(f'incomplete shard {selected^expected}')
 (root/'complete.json').write_text(json.dumps({'action':a.action,'shard_index':a.shard_index,'slots':sorted(selected),'selected':len(selected)},indent=2)+'\n')
 print(json.dumps({'selected':len(selected),'slots':sorted(selected)}))
if __name__=='__main__':main()
