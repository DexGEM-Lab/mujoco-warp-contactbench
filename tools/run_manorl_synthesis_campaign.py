#!/usr/bin/env python3
"""Prepare, run and supervise source-faithful raw120 1:5 compact synthesis."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

from sim.manorl.synthesis_campaign import (
    CAMPAIGN_CONTRACT, PrefixConfig, QuotaLedger, atomic_json,
    augment_source_prefix, canonical_hash, source_key, acquire_owner_lock, owner_lock_busy,
)


def emit(event, **values):
    print(json.dumps({'time':time.time(),'event':event,**values},sort_keys=True),flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def prepare(args):
    from sim.manorl.trajectory_package import load_trajectory_package, _apply_runtime_pre_padding
    from sim.manorl.lance_v2 import file_sha256
    root=args.campaign.resolve();root.mkdir(parents=True,exist_ok=True)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    inventory=read_json(args.inventory)
    prefix=PrefixConfig()
    if not np.isfinite(args.initial_penetration_tolerance_m) or not 0 <= args.initial_penetration_tolerance_m <= 0.01:
        raise ValueError('initial penetration tolerance must be between 0 and 0.01m')
    config={'contract':CAMPAIGN_CONTRACT,'prefix':asdict(prefix),'output':str(output),
        'ccd_contacts_per_world':args.ccd_contacts,'contacts_per_world':args.contacts_per_world,
        'constraint_capacity':args.constraint_capacity,'software_commit':args.software_commit,
        'initial_penetration_tolerance_m':args.initial_penetration_tolerance_m}
    if (root/'config.json').exists() and read_json(root/'config.json')!=config:
        raise ValueError('existing campaign configuration differs')
    catalogs={};sources=[];checkpoints={};seen=set();exceptions=Counter()
    for job in inventory['jobs']:
        checkpoint=Path(job['checkpoint']).resolve()
        metadata=read_json(str(checkpoint)+'.json')
        env=metadata['runtime_config']['environment']
        progress=metadata['runtime_config'].get('training_progress',{})
        if progress.get('completed_updates',0)<5000:
            raise ValueError('only completed 5000-update policies can enter this campaign')
        if env['compatibility']['early_phase_steps']!=80 or env['pre_padding']!=80:
            raise ValueError('checkpoint is not pre80/early80')
        cp_sha=file_sha256(checkpoint)
        checkpoints[cp_sha]={'path':str(checkpoint),'sha256':cp_sha,'metadata':metadata}
        pairs=set(job['pairs'])
        for package in job['packages']:
            package=str(Path(package).resolve())
            if package not in catalogs:
                emit('package_load',path=package)
                catalogs[package]=load_trajectory_package(package,verify_hashes=True)
            catalog=catalogs[package]
            for raw in catalog.trajectories:
                obj,action,_=raw.identity.identity.rsplit('_',2)
                if f'{obj}:{action}' not in pairs:continue
                source=_apply_runtime_pre_padding(raw,80)
                key=source_key(source)
                if key in seen:raise ValueError('duplicate raw source across inventory entries')
                seen.add(key)
                scene=tuple(source.scene_object_types or (obj,))
                group=obj+'-'+canonical_hash([scene,cp_sha])[:12]
                folder=root/'sources'/obj;folder.mkdir(parents=True,exist_ok=True)
                path=folder/(key+'.pkl')
                if not path.exists():
                    temporary=path.with_suffix('.tmp')
                    with temporary.open('wb') as f:pickle.dump(source,f,protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(temporary,path)
                sources.append({'key':key,'identity':source.identity.identity,'pair':f'{obj}:{action}',
                    'object':obj,'group_key':group,'scene_objects':list(scene),
                    'source_uuid':source.identity.uuid,'dataset_path':str(source.identity.dataset_path),
                    'dataset_version':source.identity.dataset_version,'row_index':source.identity.row_index,
                    'source_operator':source.source_operator,'source_hand_betas':source.source_hand_betas,
                    'pickle':str(path),'pickle_sha256':file_sha256(path),'checkpoint':str(checkpoint),
                    'checkpoint_sha256':cp_sha,'package_digest':catalog.package_digest,
                    'package_manifest_sha256':catalog.manifest_sha256,'frames':len(source.q_ref)})
                if source.movement_start_step!=80:exceptions['source_start_less_than_80']+=1
        emit('object_prepared',object=job['object'],sources=sum(s['object']==job['object'] for s in sources))
    sources.sort(key=lambda s:s['key'])
    atomic_json(root/'config.json',config)
    atomic_json(root/'sources.json',sources)
    atomic_json(root/'checkpoints.json',checkpoints)
    ledger=QuotaLedger(root/'quota.sqlite');ledger.initialize(config,sources)
    summary={**ledger.summary(),'objects':len({s['object'] for s in sources}),
        'pairs':len({s['pair'] for s in sources}),'groups':len({s['group_key'] for s in sources}),
        'source_exceptions':dict(exceptions)}
    atomic_json(root/'preparation.json',summary);emit('prepared',**summary)
    ledger.close()


def claim_wave(ledger, owner, group, batch_size, *, settle_pending=None):
    """Fill a short batch only after its predecessor's publication is settled.

    A small remaining quota may not hold two full batches of leases. Do not
    pad a short batch while rejected candidates are still owned by the prior
    publisher. Full batches retain the normal GPU/CPU publication overlap.
    """
    claims=ledger.claim(owner,group,batch_size)
    if len(claims)<batch_size and settle_pending is not None:
        emit('batch_wait_for_publication',owner=owner,group=group,
             available=len(claims),requested=batch_size)
        started=time.perf_counter()
        settle_pending()  # May fail: never release or reuse unverified claims.
        claims.extend(ledger.claim(owner,group,batch_size-len(claims)))
        emit('batch_claim_refilled',owner=owner,group=group,
             actual_candidates=len(claims),seconds=time.perf_counter()-started)
    return claims


def load_sources(claims):
    from sim.manorl.lance_v2 import file_sha256
    trajectories=[]
    for claim in claims:
        spec=claim['source'];path=Path(spec['pickle'])
        if file_sha256(path)!=spec['pickle_sha256']:raise RuntimeError('source pickle changed')
        with path.open('rb') as f:trajectory=pickle.load(f)
        if source_key(trajectory)!=spec['key']:raise RuntimeError('source namespace changed')
        trajectories.append(trajectory)
    return trajectories


def require_log_capture(path):
    expected=Path(path).resolve()
    for fd in (1,2):
        actual=Path(os.readlink(f'/proc/self/fd/{fd}'))
        if not actual.is_absolute() or actual.resolve()!=expected:
            raise RuntimeError('redirect both stdout and stderr directly to --log to capture native Warp warnings')


def log_has_overflow(path):
    if path is None or not path.exists():
        raise RuntimeError('Warp warning capture log is unavailable')
    # Only this worker's own log. A full-wave scan also catches startup overflows.
    import re
    pattern=re.compile(rb'CCD overflow|capacity saturated|contact buffer overflow|constraint buffer overflow',re.I)
    import ctypes
    sys.stdout.flush();sys.stderr.flush();ctypes.CDLL(None).fflush(None)
    with path.open('rb') as f:
        carry=b''
        while chunk:=f.read(1024*1024):
            if pattern.search(carry+chunk):return True
            carry=chunk[-128:]
    return False


def run_worker(args):
    from sim.manorl.synthesis_runtime import BatchSynthesisRuntime
    from sim.manorl.synthesis_publication import runtime_context
    from sim.manorl.synthesis_admission import execution_revision
    revision=execution_revision()
    root=args.campaign.resolve();config=read_json(root/'config.json');prefix=PrefixConfig(**config['prefix'])
    if min(args.batch_size,args.waves)<1:raise ValueError('batch size and waves must be positive')
    require_log_capture(args.log)
    ledger=QuotaLedger(root/'quota.sqlite');owner=args.owner or f'worker-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    owner_lock=acquire_owner_lock(root,owner)
    spool=root/'spool';spool.mkdir(exist_ok=True)
    runtime=None;pending=None;wave=0;total_accepted=0
    def heartbeat(**extra):
        ledger.heartbeat(owner,{'pid':os.getpid(),'pid_start':_process_start(os.getpid()),
            'group':args.group,'wave':wave,'accepted':total_accepted,
            'gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),'log':str(args.log),**extra})
    def reap_pending():
        nonlocal pending,total_accepted
        if pending is None:return
        process,path,output_handle=pending
        code=process.wait();output_handle.close()
        if code:raise RuntimeError(f'CPU publication failed exit={code}; preserved {path}')
        result=read_json(path.with_suffix('.result.json'))
        for item in result['records']:
            ledger.finish(item['claim'],item['diagnostic'],row_uuid=item['row_uuid'],shard=result['shard'] if item['row_uuid'] else None)
        total_accepted+=result['accepted']
        emit('published',owner=owner,wave=wave,accepted=result['accepted'],total_accepted=total_accepted,
             failures=result['failures'],payload_bytes=result['payload_bytes'],
             retreat_seconds=result['retreat_seconds'],publish_seconds=result['publish_seconds'],shard=result['shard'])
        # This is the campaign's own temporary IPC payload, not a Lance dataset.
        # Failed publications retain it. Durable accepted rows live on NAS.
        path.unlink();pending=None
    started=time.perf_counter()
    try:
        for wave in range(args.waves):
            if (root/'PAUSE').exists():break
            heartbeat(phase='claim')
            if pending is not None and pending[0].poll() is not None:reap_pending()
            claims=claim_wave(ledger,owner,args.group,args.batch_size,
                settle_pending=reap_pending if pending is not None else None)
            if not claims:break
            base=load_sources(claims)
            if runtime is None:
                padded=(base+[base[-1]]*args.batch_size)[:args.batch_size]
                t0=time.perf_counter()
                runtime=BatchSynthesisRuntime(padded,Path(claims[0]['source']['checkpoint']),
                    ccd_contacts=config['ccd_contacts_per_world'],contacts_per_world=config['contacts_per_world'],
                    constraint_capacity=config['constraint_capacity'],reuse_contact_snapshot=not args.no_contact_cache)
                emit('runtime_ready',seconds=time.perf_counter()-t0,batch_size=args.batch_size,group=args.group)
            valid=[];trajectories=[];samples=[];proposals=0
            while claims:
                proposals+=len(claims)
                for claim,source in zip(claims,base):
                    try:
                        trajectory,sample=augment_source_prefix(source,slot=claim['slot'],attempt=claim['attempt'],
                            joint_lower=runtime.environment.joint_lower,joint_upper=runtime.environment.joint_upper,config=prefix,
                            seed=claim.get('seed'))
                    except ValueError as exc:
                        ledger.finish(claim,{'accepted':False,'prefix_sampling_failure':str(exc)});continue
                    tolerance=config.get('initial_penetration_tolerance_m')
                    if tolerance is not None:
                        failure=runtime.initial_geometry_failure(trajectory,tolerance_m=tolerance)
                        if failure:
                            ledger.finish(claim,{'accepted':False,'initial_geometry_failure':failure,'prefix':sample});continue
                    valid.append(claim);trajectories.append(trajectory);samples.append(sample)
                if len(valid)==args.batch_size or proposals>=8*args.batch_size:break
                claims=ledger.claim(owner,args.group,args.batch_size-len(valid))
                base=load_sources(claims)
            if not valid:
                emit('sampling_rejected_wave',group=args.group,attempts=proposals);continue
            actual=len(valid)
            trajectories += [trajectories[-1]]*(args.batch_size-actual)
            samples += [samples[-1]]*(args.batch_size-actual)
            runtime.install(trajectories,[s['seed'] for s in samples])
            heartbeat(phase='rollout')
            results,timing=runtime.collect(trajectories,samples,valid,progress=lambda d:heartbeat(phase='rollout',**d))
            for result in results:
                result['diagnostic']['initial_geometry_prefilter_tolerance_m']=config.get('initial_penetration_tolerance_m')
            sys.stdout.flush();sys.stderr.flush()
            if log_has_overflow(args.log):
                raise RuntimeError('CAPACITY_BLOCKED: Warp overflow observed; no rows from this wave published')
            emit('wave_complete',group=args.group,wave=wave,proposals=proposals,actual_candidates=actual,**timing)
            # Bound postprocessing backlog to one wave. CPU writes overlap the
            # following GPU wave; only backpressure waits here.
            heartbeat(phase='publication_backpressure');reap_pending()
            bundle_id=uuid.uuid4().hex;path=spool/(bundle_id+'.pkl')
            bundle={'bundle_id':bundle_id,'output':config['output'],
                    'context':runtime_context(runtime,revision),'results':results}
            with path.open('wb') as f:pickle.dump(bundle,f,protocol=pickle.HIGHEST_PROTOCOL)
            log_handle=path.with_suffix('.log').open('wb')
            cpu_env={**os.environ,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
            process=subprocess.Popen([sys.executable,'-m','tools.run_manorl_synthesis_campaign','publish',str(path)],
                stdout=log_handle,stderr=subprocess.STDOUT,env=cpu_env,pass_fds=(owner_lock.fileno(),))
            pending=(process,path,log_handle)
        reap_pending()
        elapsed=time.perf_counter()-started
        heartbeat(phase='finished_round')
        emit('worker_round_done',owner=owner,group=args.group,accepted=total_accepted,seconds=elapsed,
             accepted_per_gpu_hour=total_accepted/max(elapsed,1e-9)*3600)
    finally:
        # Never release leases while a publisher still owns potentially durable
        # rows. Supervisor reconciliation handles abnormal exits.
        if pending is not None:
            process,path,handle=pending
            process.wait();handle.close()
        ledger.close();owner_lock.close()


def _process_start(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):return None


class AdoptedWorker:
    """Monitor an orphaned campaign process without spawning duplicate work."""
    def __init__(self, root, ledger, owner, details):
        self.root,self.ledger,self.owner,self.details=root,ledger,owner,details

    def poll(self):
        pid=self.details.get('pid');start=self.details.get('pid_start')
        if owner_lock_busy(self.root,self.owner) or (pid and start and _process_start(pid)==start):
            return None
        row=self.ledger.db.execute('SELECT details FROM workers WHERE owner=?',(self.owner,)).fetchone()
        phase=json.loads(row[0]).get('phase') if row else None
        return 0 if phase=='finished_round' else 1


def select_worker_group(groups, assigned, blocked, reserved, batch_size):
    """Prefer distinct groups, then share ready work across otherwise idle GPUs.

    Reserve one batch per launch decision until the next ledger snapshot so a
    small queue is not assigned to every free card before workers claim it.
    """
    candidates=[g for g in groups if g['group_key'] not in blocked
                and g['ready']-reserved.get(g['group_key'],0)>0]
    for group in candidates:
        if not assigned.get(group['group_key'],0):return group['group_key']
    shared=[g for g in candidates
            if g['ready']-reserved.get(g['group_key'],0)>=batch_size]
    if not shared:return None
    return min(shared,key=lambda g:assigned[g['group_key']])['group_key']


def supervise(args):
    from sim.manorl.synthesis_publication import reconcile_shards
    from sim.manorl.synthesis_admission import admitted_gpu_pool
    root=args.campaign.resolve();config=read_json(root/'config.json')
    # Only one supervisor may own this campaign; flock survives unrelated SSH loss.
    lock=(root/'supervisor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=QuotaLedger(root/'quota.sqlite');output=Path(config['output'])
    recovered=reconcile_shards(ledger,output)
    emit('reconciled',rows=recovered)
    gpus=[int(v) for v in args.gpus.split(',')]
    if not gpus or len(set(gpus))!=len(gpus):raise ValueError('GPU list must be unique')
    workers={}
    priority={name:i for i,name in enumerate(args.priority_objects.split(',')) if name}
    # A CPU publisher inherits its GPU owner's lock descriptor. Even if the GPU
    # process is killed, its claims remain unavailable until publication ends.
    for row in ledger.db.execute('SELECT owner,details FROM workers').fetchall():
        owner=row['owner'];details=json.loads(row['details']);pid=details.get('pid')
        current_start=_process_start(pid) if pid else None
        alive=owner_lock_busy(root,owner) or (current_start is not None and current_start==details.get('pid_start'))
        if alive:
            gpu=details.get('gpu')
            if gpu not in gpus or gpu in workers or not details.get('log'):
                raise RuntimeError(f'cannot safely adopt worker ownership: {owner}')
            workers[gpu]={'process':AdoptedWorker(root,ledger,owner,details),'handle':None,
                'owner':owner,'pid':pid,'pid_start':details.get('pid_start'),'group':details['group'],
                'log':details['log'],'started':time.time(),'adopted':True}
            emit('worker_adopted',owner=owner,gpu=gpu,pid=pid)
        else:ledger.release_owner(owner)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    state_path=root/'supervisor-state.json'
    prior=read_json(state_path) if state_path.exists() else {}
    last_scheduled=prior.get('last_scheduled',{});failures=Counter(prior.get('failures',{}));blocked=prior.get('blocked',{})
    stop=False
    def request_stop(signum,frame):
        nonlocal stop
        stop=True;emit('supervisor_stop_requested',signal=signum)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    while True:
        stop=stop or (root/'PAUSE').exists()
        summary=ledger.summary()
        for gpu,record in list(workers.items()):
            process=record['process']
            code=process.poll()
            if code is None or owner_lock_busy(root,record['owner']):continue
            if record['handle'] is not None:record['handle'].close()
            reconcile_shards(ledger,output)
            ledger.release_owner(record['owner'])
            group=record['group'];del workers[gpu]
            if code:
                failures[group]+=1
                tail=Path(record['log']).read_bytes()[-12000:].decode(errors='replace')
                blocked[group]={'exit_code':code,'log':record['log'],'tail':tail,'time':time.time()}
                emit('worker_failed',group=group,gpu=gpu,exit_code=code,log=record['log'])
            else:emit('worker_round_finished',group=group,gpu=gpu)
            atomic_json(state_path,{'last_scheduled':last_scheduled,'failures':dict(failures),'blocked':blocked})
        summary=ledger.summary()
        status={'time':time.time(),**summary,'workers':{str(g):{k:v for k,v in r.items() if k not in ('process','handle')}
            for g,r in workers.items()},'blocked':blocked,'restarts':dict(failures),'stop_requested':stop}
        atomic_json(root/'status.json',status)
        if summary['complete'] and not workers and not blocked:
            atomic_json(root/'completed.json',status);emit('campaign_complete',**summary);break
        if stop:
            if not workers:break
            time.sleep(10);continue
        assigned=Counter(r['group'] for r in workers.values());reserved=Counter()
        groups=sorted(ledger.groups(),key=lambda g:(last_scheduled.get(g['group_key'],0),
            priority.get(g['group_key'].rsplit('-',1)[0],len(priority)),g['min_attempts'],g['group_key']))
        for gpu in admitted_gpu_pool(root,gpus):
            if gpu in workers:continue
            group=select_worker_group(groups,assigned,blocked,reserved,args.batch_size)
            if group is None:continue
            # Sample actual foreign compute ownership immediately before launch.
            query=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            if query.stdout.strip():
                emit('gpu_busy',gpu=gpu,pids=query.stdout.strip());continue
            owner=f'g{gpu}-{uuid.uuid4().hex[:12]}'
            path=logs/(owner+'.log');handle=path.open('wb')
            command=[sys.executable,'-m','tools.run_manorl_synthesis_campaign','worker','--campaign',str(root),
                '--group',group,'--owner',owner,'--batch-size',str(args.batch_size),'--waves',str(args.waves),'--log',str(path)]
            environment={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'XLA_PYTHON_CLIENT_MEM_FRACTION':'0.65',
                         'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
            process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
            workers[gpu]={'process':process,'handle':handle,'pid':process.pid,'pid_start':_process_start(process.pid),
                'owner':owner,'group':group,'log':str(path),'started':time.time()}
            ledger.heartbeat(owner,{'pid':process.pid,'pid_start':_process_start(process.pid),'phase':'starting',
                'group':group,'gpu':gpu,'log':str(path)})
            last_scheduled[group]=time.time();assigned[group]+=1;reserved[group]+=args.batch_size
            emit('worker_started',gpu=gpu,group=group,pid=process.pid,owner=owner,log=str(path))
        # Persistence/waiting belongs to this server process, not the LLM monitor.
        time.sleep(10)
    ledger.close();lock.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--inventory',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--software-commit',required=True)
    p.add_argument('--ccd-contacts',type=int,default=16);p.add_argument('--contacts-per-world',type=int,default=1024)
    p.add_argument('--constraint-capacity',type=int,default=4096)
    p.add_argument('--initial-penetration-tolerance-m',type=float,default=0.001)
    for name in ('worker','supervise'):
        p=sub.add_parser(name);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--batch-size',type=int,default=64)
        p.add_argument('--waves',type=int,default=32)
        if name=='worker':
            p.add_argument('--group',required=True);p.add_argument('--owner');p.add_argument('--log',type=Path,required=True)
            p.add_argument('--no-contact-cache',action='store_true')
        else:
            p.add_argument('--gpus',default='2,3')
            p.add_argument('--priority-objects',default='cylinder6,cube2,sphere1,cylinder7,cylinder3,mayonnaisebottle')
    p=sub.add_parser('publish');p.add_argument('bundle',type=Path)
    p=sub.add_parser('migrate',help='Explicitly convert a legacy per-source-slot ledger to fixed pair budgets')
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,default=None,help='Override the publication root used for pre-migration reconciliation')
    p.add_argument('--dry-run',action='store_true',help='Report the planned migration without changing the ledger')
    p=sub.add_parser('set-source-filter',help='Stop retrying low-success sources; keep all saved rows and in-flight claims')
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--minimum-rate',type=float,default=.25)
    p.add_argument('--minimum-attempts',type=int,default=20)
    p=sub.add_parser('upgrade-source-policy',help='Quiescent pending-aware exclusion-guard upgrade with audited repair of premature exclusions')
    p.add_argument('--campaign',type=Path,required=True)
    p=sub.add_parser('status');p.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':prepare(args)
    elif args.command=='worker':run_worker(args)
    elif args.command=='supervise':supervise(args)
    elif args.command=='publish':
        from sim.manorl.synthesis_publication import publish_bundle
        result=publish_bundle(args.bundle);emit('publication_done',accepted=result['accepted'],shard=result['shard'])
    elif args.command=='migrate':
        from sim.manorl.synthesis_migration import migrate_ledger, write_migration_report
        report=migrate_ledger(args.campaign,output=args.output,dry_run=args.dry_run)
        if report['state']=='migrated':write_migration_report(args.campaign,report)
        emit('ledger_migration_'+report['state'].replace('-','_'),**report)
    elif args.command=='set-source-filter':
        root=args.campaign.resolve()
        if not (root/'quota.sqlite').is_file():raise ValueError('prepare the campaign before filtering sources')
        ledger=QuotaLedger(root/'quota.sqlite')
        try:
            added=ledger.configure_success_filter(minimum_rate=args.minimum_rate,minimum_attempts=args.minimum_attempts)
            report=ledger.exclusion_report();summary=ledger.summary()
            atomic_json(root/'source-filter-report.json',{**report,'summary':summary})
            emit('source_filter_enabled',added_exclusions=added,**summary)
        finally:ledger.close()
    elif args.command=='upgrade-source-policy':
        root=args.campaign.resolve()
        if not (root/'quota.sqlite').is_file():raise ValueError('prepare the campaign before upgrading its policy')
        ledger=QuotaLedger(root/'quota.sqlite')
        try:
            report=ledger.upgrade_eligibility_guard()
            atomic_json(root/'eligibility-upgrade-report.json',report)
            emit('eligibility_guard_upgraded',**report)
        finally:ledger.close()
    else:
        ledger=QuotaLedger(args.campaign/'quota.sqlite');print(json.dumps(ledger.summary(),indent=2));ledger.close()
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(1)
