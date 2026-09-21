#!/usr/bin/env python3
"""Run a disjoint slot shard with the unchanged native-U1 Runtime."""
from pathlib import Path
import argparse,fcntl,json,subprocess,sys
from sim.manorl.u1_campaign import Ledger,child_uuid,digest,file_sha,verify_signed
from tools.run_u1_campaign import Runtime

IMPLEMENTATION=('tools/run_u1_campaign.py','tools/u1_campaign_registry.py','sim/manorl/u1_campaign.py','tools/u1_interactive_session.py','sim/manorl/local_contact_repair.py','sim/manorl/mjx_sim.py')
CANONICAL_RUNNER='/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-uniform-direct-parent-repair/tools/run_u1_campaign.py'
def assigned_slots(slots,start,index,count):return [x for x in slots if x['slot']>=start and (x['slot']-start)%count==index]
def identity(registry,planned,action):
 impl={str(Path(CANONICAL_RUNNER)):file_sha(CANONICAL_RUNNER)}
 impl.update({f:file_sha(f)for f in IMPLEMENTATION[1:]})
 return digest(dict(registry=registry['digest'],plan=planned['digest'],action=action,implementation=impl))
def main():
 p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--main-attempts',type=Path,required=True);p.add_argument('--shard-root',type=Path,required=True);p.add_argument('--action',required=True);p.add_argument('--slot-start',type=int,required=True);p.add_argument('--shard-index',type=int,required=True);p.add_argument('--shard-count',type=int,required=True);a=p.parse_args();r=json.loads(a.registry.read_text());pl=json.loads(a.plan.read_text());verify_signed(r);verify_signed(pl);assert pl['registry_digest']==r['digest'];assert 0<=a.shard_index<a.shard_count
 root=a.shard_root/f'shard{a.shard_index}';root.mkdir(parents=True,exist_ok=True)
 with (root/'writer.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ledger=Ledger(root/'ledger.jsonl',identity(r,pl,a.action));mainrows=[]
  mp=a.main_attempts/(a.action+'.jsonl')
  if mp.exists():mainrows=[json.loads(x)for x in mp.read_text().splitlines()]
  attempted={x['uuid']for x in mainrows if 'uuid'in x}|{x['uuid']for x in ledger.rows if 'uuid'in x};runtime=Runtime(a.registry,a.action)
  for slot in assigned_slots(pl['slots'],a.slot_start,a.shard_index,a.shard_count):
   sid=slot['slot']
   if sid in ledger.selected():continue
   for candidate in slot['candidates']:
    uid=child_uuid(r['digest'],a.action,pl['digest'],sid,candidate)
    if uid in attempted:continue
    folder=root/a.action/uid;ledger.append(uuid=uid,slot=sid,candidate=candidate,status='started');attempted.add(uid)
    try:
     first=runtime.replay(candidate['delta'],folder/'first')
     if not first['accepted']:raise ValueError('first-pass gates failed')
     cmd=[sys.executable,'-m','tools.run_u1_campaign','second-pass','--registry',str(a.registry.resolve()),'--action',a.action,'--staging',str(folder/'second'),'--delta',json.dumps(candidate['delta']),'--frozen',str(folder/'first/target.npy')]
     with (folder/'second-process.log').open('w')as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
     second=json.loads((folder/'second/result.json').read_text())
     if not second['accepted']or second['pid']==first['pid']:raise ValueError('independent gates failed')
     artifacts={str(x.resolve()):file_sha(x)for x in folder.rglob('*')if x.is_file()};ledger.append(uuid=uid,slot=sid,status='selected',artifacts=artifacts);break
    except Exception as e:ledger.append(uuid=uid,slot=sid,status='rejected',reason=repr(e))
   if sid not in ledger.selected():ledger.append(slot=sid,status='exhausted');raise RuntimeError(f'exhausted slot{sid}')
  selected=ledger.selected();expected={x['slot']for x in assigned_slots(pl['slots'],a.slot_start,a.shard_index,a.shard_count)}
  if set(selected)!=expected:raise RuntimeError(f'shard incomplete {set(selected)^expected}')
  (root/'complete.json').write_text(json.dumps({'action':a.action,'shard_index':a.shard_index,'shard_count':a.shard_count,'slot_start':a.slot_start,'selected':len(selected),'slots':sorted(selected)},indent=2)+'\n');print(json.dumps({'selected':len(selected),'slots':sorted(selected)}))
if __name__=='__main__':main()
