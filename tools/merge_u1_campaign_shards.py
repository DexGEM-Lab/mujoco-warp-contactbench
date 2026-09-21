#!/usr/bin/env python3
"""Merge completed disjoint shard ledgers into one canonical action ledger."""
from pathlib import Path
import argparse,fcntl,json,os
from sim.manorl.u1_campaign import Ledger,file_sha,verify_signed
from tools.run_u1_campaign_shard import identity

def latest(rows):
 out={}
 for x in rows:
  if 'uuid'in x:out[x['uuid']]=x
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--attempts',type=Path,required=True);p.add_argument('--shard-root',type=Path,required=True);p.add_argument('--action',required=True);p.add_argument('--shard-count',type=int,required=True);a=p.parse_args();r=json.loads(a.registry.read_text());pl=json.loads(a.plan.read_text());verify_signed(r);verify_signed(pl);ident=identity(r,pl,a.action);a.attempts.mkdir(parents=True,exist_ok=True)
 with (a.attempts/(a.action+'.lock')).open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main=Ledger(a.attempts/(a.action+'.jsonl'),ident);seen={x['uuid']for x in main.rows if'uuid'in x};merged=[]
  for si in range(a.shard_count):
   root=a.shard_root/f'shard{si}';complete=json.loads((root/'complete.json').read_text());rows=[json.loads(x)for x in(root/'ledger.jsonl').read_text().splitlines()];assert all(x['identity']==ident for x in rows);assert not any(x['status']=='exhausted'for x in rows);fin=latest(rows)
   for x in rows:
    if x['status']=='started'and x['uuid']not in seen:main.append(uuid=x['uuid'],slot=x['slot'],candidate=x['candidate'],status='started');seen.add(x['uuid'])
    elif x['status']=='rejected':
     if x['uuid']not in seen:raise RuntimeError('rejection missing start')
     main.append(uuid=x['uuid'],slot=x['slot'],status='rejected',reason='shard'+str(si)+': '+x['reason'])
   for uid,x in sorted(fin.items(),key=lambda kv:kv[1]['slot']):
    if x['status']!='selected':continue
    src=root/a.action/uid;dst=a.attempts/a.action/uid
    if dst.exists():raise RuntimeError('destination exists '+uid)
    for path,h in x['artifacts'].items():assert file_sha(path)==h
    src.rename(dst);artifacts={str(f.resolve()):file_sha(f)for f in dst.rglob('*')if f.is_file()};main.append(uuid=uid,slot=x['slot'],status='selected',artifacts=artifacts);merged.append(x['slot'])
  selected=main.selected()
  if set(selected)!=set(range(160)):raise RuntimeError(f'canonical quota incomplete {len(selected)}')
  report={'action':a.action,'selected':160,'merged_slots':sorted(merged),'shards':a.shard_count,'identity':ident};(a.shard_root/'merge.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
