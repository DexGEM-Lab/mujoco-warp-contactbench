#!/usr/bin/env python3
"""Discover completed raw120 models and append their sources without stopping workers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sim.manorl.synthesis_admission import (
    append_prepared, claim_object, completed_job, eligible_gpus, ownership_record,
    refresh_source_manifest, register_existing, snapshot_checkpoint,
)
from sim.manorl.synthesis_campaign import CAMPAIGN_CONTRACT, PrefixConfig, QuotaLedger, atomic_json
from tools.watch_manorl_synthesis_campaign import lock_busy


def scan_once(args, ledger, config):
    root=args.campaign;events=[];errors=[]
    register_existing(ledger,args.registry,host=args.host,campaign_id=args.campaign_id,output=config['output'])
    existing={json.loads(row[0])['object'] for row in ledger.db.execute('SELECT spec FROM sources')}
    for report in sorted(args.runs.glob('*.json')):
        if report.name.endswith('.pt.json'):continue
        try:
            related=(report,report.with_suffix('.pt'),Path(str(report.with_suffix('.pt'))+'.json'))
            fingerprint=tuple((p.stat().st_size,p.stat().st_mtime_ns) if p.exists() else None for p in related)
            cache=getattr(args,'_report_cache',{})
            previous=cache.get(str(report))
            if previous is not None and previous[0]==fingerprint:
                job=previous[1]
            else:
                job=completed_job(report);cache[str(report)]=(fingerprint,job);args._report_cache=cache
            if job is None:continue
            obj=job['object']
            owner=ownership_record(args.registry,obj)
            if owner and (owner['host']!=args.host or owner['campaign_id']!=args.campaign_id):continue
            if obj in existing:continue
            pinned=snapshot_checkpoint(job,root)
            if not claim_object(args.registry,obj=obj,host=args.host,campaign_id=args.campaign_id,
                                checkpoint_sha=pinned['checkpoint_sha256'],output=config['output']):continue
            stage=root/'admissions'/(obj+'-'+pinned['checkpoint_sha256'][:16])
            stage.mkdir(parents=True,exist_ok=True)
            inventory=stage/'inventory.json';atomic_json(inventory,{'jobs':[pinned]})
            prepared=stage/'prepared'
            if not (prepared/'preparation.json').is_file():
                command=[sys.executable,'-m','tools.run_manorl_synthesis_campaign','prepare',
                    '--inventory',str(inventory),'--campaign',str(prepared),'--output',config['output'],
                    '--software-commit',args.software_commit,'--ccd-contacts',str(config['ccd_contacts_per_world']),
                    '--contacts-per-world',str(config['contacts_per_world']),
                    '--constraint-capacity',str(config['constraint_capacity']),
                    '--initial-penetration-tolerance-m',str(config.get('initial_penetration_tolerance_m',.001))]
                with (stage/'prepare.log').open('ab') as stream:
                    subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'})
            inserted=append_prepared(ledger,root,prepared,config)
            existing.add(obj)
            event={'object':obj,'added_sources':inserted,'added_target':5*inserted,'checkpoint_sha256':pinned['checkpoint_sha256']}
            atomic_json(stage/'admitted.json',{**event,'time':time.time(),'host':args.host})
            print(json.dumps({'event':'object_admitted',**event}),flush=True);events.append(event)
        except Exception as exc:
            error={'report':str(report),'error':type(exc).__name__+': '+str(exc)}
            errors.append(error);print(json.dumps({'event':'admission_error',**error}),flush=True)
    # Repair informational manifests from the authoritative DB after any prior
    # admission crash between SQL commit and JSON refresh.
    sources=refresh_source_manifest(ledger,root)
    policy=json.loads(args.gpu_policy.read_text())
    pool=eligible_gpus(policy,{s['object'] for s in sources})
    atomic_json(root/'gpu-pool.json',{'gpus':pool,'updated':time.time(),'policy':policy})
    summary=ledger.summary()
    status={'time':time.time(),'host':args.host,'campaign_id':args.campaign_id,'state':'active',
        'eligible_gpus':pool,'objects':sorted({s['object'] for s in sources}),
        'pairs':len({s['pair'] for s in sources}),'errors':errors,'admitted':events,
        'campaign':str(root),'output':config['output'],**summary}
    atomic_json(root/'admission-status.json',status)
    atomic_json(args.registry/'hosts'/f'{args.host}.json',status)
    if summary['source_success_policy'] is not None:
        atomic_json(args.registry/'source-filters'/f'{args.host}.json',
                    {'host':args.host,'campaign_id':args.campaign_id,**ledger.exclusion_report()})
    supervisor_status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    needs_watch=summary['remaining']>0 or supervisor_status.get('blocked') or supervisor_status.get('workers')
    if args.ensure_watch and pool and needs_watch and not lock_busy(root/'watch.lock'):
        allowed=sorted(set(policy.get('always',[]))|{int(g) for g in policy.get('after_objects',{})})
        path=root/f'admission-watch-{time.time_ns()}.log'
        with path.open('ab') as stream:
            process=subprocess.Popen([sys.executable,'-m','tools.watch_manorl_synthesis_campaign',
                '--campaign',str(root),'--gpus',','.join(map(str,allowed)),
                '--batch-size',str(args.batch_size),'--waves',str(args.waves)],
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'JAX_PLATFORMS':'cuda','MANORL_SOFTWARE_COMMIT':args.software_commit})
        print(json.dumps({'event':'watch_started','pid':process.pid,'log':str(path)}),flush=True)
    return status


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--registry',type=Path,required=True);parser.add_argument('--host',required=True)
    parser.add_argument('--campaign-id',required=True);parser.add_argument('--gpu-policy',type=Path,required=True)
    parser.add_argument('--software-commit',required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--interval',type=float,default=60.);parser.add_argument('--once',action='store_true')
    parser.add_argument('--ensure-watch',action='store_true');parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--waves',type=int,default=8)
    args=parser.parse_args(argv);args.campaign=args.campaign.resolve();args.registry=args.registry.resolve()
    if args.interval<1:raise ValueError('interval must be positive')
    root=args.campaign;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'admission.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (root/'config.json').exists():
        if args.output is None:raise ValueError('--output is required for a new host campaign')
        args.output.mkdir(parents=True,exist_ok=True)
        atomic_json(root/'config.json',{'contract':CAMPAIGN_CONTRACT,'prefix':asdict(PrefixConfig()),
            'output':str(args.output.resolve()),'ccd_contacts_per_world':16,'contacts_per_world':1024,
            'constraint_capacity':4096,'initial_penetration_tolerance_m':.001,'software_commit':args.software_commit})
    config=json.loads((root/'config.json').read_text())
    ledger=QuotaLedger(root/'quota.sqlite');ledger.initialize(config,[])
    try:
        while True:
            if not (root/'PAUSE').exists():
                try:scan_once(args,ledger,config)
                except Exception as exc:
                    error={'time':time.time(),'state':'error','error':type(exc).__name__+': '+str(exc)}
                    atomic_json(root/'admission-status.json',error)
                    print(json.dumps({'event':'admission_scan_failed',**error}),flush=True)
                    if args.once:raise
            if args.once:return 0
            time.sleep(args.interval)
    finally:
        ledger.close();lock.close()


if __name__=='__main__':raise SystemExit(main())
