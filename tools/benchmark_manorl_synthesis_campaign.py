#!/usr/bin/env python3
"""Fixed-candidate, warm-runtime end-to-end synthesis benchmark.

Run each batch-size variant in a fresh process on the same group/source limit.
Source/slot/attempt seeds are independent of batch size. Bench outputs never
satisfy production quota and reside in separate, non-overwriting directories.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import uuid

from sim.manorl.synthesis_campaign import PrefixConfig, atomic_json, augment_source_prefix
from tools.run_manorl_synthesis_campaign import emit, load_sources, log_has_overflow, read_json, require_log_capture


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--group',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--sources',type=int,default=16);p.add_argument('--candidate-limit',type=int,default=80)
    p.add_argument('--no-contact-cache',action='store_true');p.add_argument('--log',type=Path,required=True)
    p.add_argument('--start-prefilter',action='store_true')
    args=p.parse_args(argv)
    require_log_capture(args.log)
    if min(args.batch_size,args.sources,args.candidate_limit)<1:raise ValueError('benchmark sizes must be positive')
    if args.output.exists():raise FileExistsError('benchmark output already exists')
    args.output.mkdir(parents=True)
    config=read_json(args.campaign/'config.json');prefix=PrefixConfig(**config['prefix'])
    specs=[s for s in read_json(args.campaign/'sources.json') if s['group_key']==args.group][:args.sources]
    if not specs:raise ValueError('group contains no sources')
    claims=[]
    for source in specs:
        for slot in range(5):
            claims.append({'source':source,'slot':slot,'attempt':1,'token':'benchmark-'+uuid.uuid4().hex,'owner':'benchmark'})
    claims=claims[:args.candidate_limit]
    base=load_sources(claims)
    from sim.manorl.synthesis_runtime import BatchSynthesisRuntime
    from sim.manorl.synthesis_publication import runtime_context
    started=time.perf_counter()
    runtime=BatchSynthesisRuntime((base+[base[-1]]*args.batch_size)[:args.batch_size],Path(specs[0]['checkpoint']),
        ccd_contacts=config['ccd_contacts_per_world'],contacts_per_world=config['contacts_per_world'],
        constraint_capacity=config['constraint_capacity'],reuse_contact_snapshot=not args.no_contact_cache)
    initialize_seconds=time.perf_counter()-started
    candidates=[];sampling_failures=Counter()
    for claim,source in zip(claims,base):
        try:
            trajectory,sample=augment_source_prefix(source,slot=claim['slot'],attempt=1,
                joint_lower=runtime.environment.joint_lower,joint_upper=runtime.environment.joint_upper,config=prefix)
        except ValueError as exc:
            sampling_failures[str(exc)]+=1;continue
        if args.start_prefilter and runtime.initial_geometry_failure(trajectory) is not None:
            sampling_failures['initial_geometry_penetration']+=1;continue
        candidates.append((claim,trajectory,sample))
    if not candidates:
        raise RuntimeError('all fixed candidates failed geometric sampling: '+repr(sampling_failures))
    def collect(batch):
        trajectories=[v[1] for v in batch];samples=[v[2] for v in batch]
        trajectories += [trajectories[-1]]*(args.batch_size-len(batch))
        samples += [samples[-1]]*(args.batch_size-len(batch))
        runtime.install(trajectories,[v['seed'] for v in samples])
        results,timing=runtime.collect(trajectories,samples,[v[0] for v in batch],progress=lambda d:emit('benchmark_progress',**d))
        if log_has_overflow(args.log):raise RuntimeError('CAPACITY_BLOCKED: overflow in benchmark, no acceptance claim')
        return results,timing
    warm=time.perf_counter();_,warm_timing=collect(candidates[:args.batch_size]);warm_seconds=time.perf_counter()-warm
    measured=time.perf_counter();pending=None;summaries=[];waves=[]
    for offset in range(0,len(candidates),args.batch_size):
        results,timing=collect(candidates[offset:offset+args.batch_size]);waves.append(timing)
        if pending is not None:
            process,path,handle=pending
            if process.wait():raise RuntimeError('benchmark CPU publication failed; see '+str(path.with_suffix('.log')))
            handle.close();summaries.append(read_json(path.with_suffix('.result.json')))
        path=args.output/f'wave-{offset:06d}.pkl'
        with path.open('wb') as f:
            pickle.dump({'bundle_id':uuid.uuid4().hex,'output':str(args.output),'results':results,
                'context':runtime_context(runtime,config['software_commit'])},f,protocol=pickle.HIGHEST_PROTOCOL)
        handle=path.with_suffix('.log').open('wb')
        process=subprocess.Popen([sys.executable,'-m','tools.run_manorl_synthesis_campaign','publish',str(path)],
            stdout=handle,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'})
        pending=(process,path,handle)
    if pending is not None:
        process,path,handle=pending
        if process.wait():raise RuntimeError('benchmark CPU publication failed; see '+str(path.with_suffix('.log')))
        handle.close();summaries.append(read_json(path.with_suffix('.result.json')))
    seconds=time.perf_counter()-measured;accepted=sum(s['accepted'] for s in summaries)
    report={'group':args.group,'batch_size':args.batch_size,'fixed_candidates':len(claims),
        'sampled_candidates':len(candidates),'sampling_failures':dict(sampling_failures),
        'initialization_seconds':initialize_seconds,'warmup_seconds':warm_seconds,'warmup':warm_timing,
        'measured_seconds':seconds,'accepted_saved':accepted,'accepted_per_gpu_hour':3600*accepted/seconds,
        'contact_snapshot_cache':not args.no_contact_cache,'start_prefilter':args.start_prefilter,'waves':waves,
        'publication':[ {k:s[k] for k in ('accepted','failures','payload_bytes','retreat_seconds','publish_seconds','wall_seconds','shard')} for s in summaries]}
    atomic_json(args.output/'benchmark.json',report);emit('benchmark_complete',**report)


if __name__=='__main__':main()
