#!/usr/bin/env python3
"""Persistent, single-owner campaign monitor; independent of Pi's lifetime.

Restarts a departed supervisor while quotas remain. Existing GPU/CPU workers
are adopted by the supervisor, not duplicated. PAUSE is an explicit user stop.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from sim.manorl.synthesis_campaign import QuotaLedger, atomic_json


def should_continue(summary: dict, *, paused: bool) -> bool:
    return not paused and not summary['complete']


def lock_busy(path: Path) -> bool:
    with path.open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
    return False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--gpus',default='2,3')
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--waves',type=int,default=8)
    p.add_argument('--interval',type=float,default=15.)
    args=p.parse_args(argv);root=args.campaign.resolve()
    if not (root/'config.json').is_file() or not (root/'quota.sqlite').is_file():
        raise ValueError('prepare the campaign before starting its monitor')
    if args.interval<1 or min(args.batch_size,args.waves)<1:raise ValueError('invalid monitoring/run interval')
    own_lock=(root/'watch.lock').open('a+')
    fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads((root/'config.json').read_text());ledger=QuotaLedger(root/'quota.sqlite')
    stopping=False;child=None;handle=None;restarts=0
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        # Persistent operator intent prevents a replacement monitor undoing stop.
        atomic_json(root/'PAUSE',{'requested_at':time.time(),'signal':signum})
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while True:
            summary=ledger.summary();paused=stopping or (root/'PAUSE').exists()
            if not should_continue(summary,paused=paused):
                atomic_json(root/'watch-status.json',{'time':time.time(),'pid':os.getpid(),
                    'state':'paused' if paused else 'complete','restarts':restarts,**summary})
                if child is not None and child.poll() is None:child.terminate()
                return 0
            if child is not None and child.poll() is not None:
                print(json.dumps({'event':'supervisor_departed','exit_code':child.returncode,
                    'time':time.time(),'complete':False}),flush=True)
                handle.close();child=None;handle=None
            active=lock_busy(root/'supervisor.lock')
            if child is None and not active:
                log=root/f'supervisor-{time.time_ns()}.log';handle=log.open('ab')
                command=[sys.executable,'-m','tools.run_manorl_synthesis_campaign','supervise',
                    '--campaign',str(root),'--gpus',args.gpus,'--batch-size',str(args.batch_size),'--waves',str(args.waves)]
                child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                restarts+=1
                print(json.dumps({'event':'supervisor_started','pid':child.pid,'time':time.time(),'log':str(log)}),flush=True)
            status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
            worker_rows=ledger.db.execute('SELECT owner,heartbeat,details FROM workers').fetchall()
            latest_workers={row['owner']:{'heartbeat_age_s':time.time()-row['heartbeat'],**json.loads(row['details'])}
                            for row in worker_rows if time.time()-row['heartbeat']<7200}
            alerts=[]
            if status.get('blocked'):alerts.append('blocked_groups_require_intervention')
            if status and time.time()-status.get('time',0)>180:alerts.append('supervisor_status_stale')
            local_free=shutil.disk_usage(root).free;publish_free=shutil.disk_usage(config['output']).free
            if local_free<10*1024**3:alerts.append('local_space_below_10GiB')
            if publish_free<20*1024**3:alerts.append('publish_space_below_20GiB')
            if any(w['heartbeat_age_s']>1800 and w.get('phase')!='finished_round' for w in latest_workers.values()):
                alerts.append('worker_heartbeat_stale')
            atomic_json(root/'watch-status.json',{'time':time.time(),'pid':os.getpid(),'state':'running',
                'supervisor_pid':None if child is None else child.pid,'supervisor_lock_held':active,
                'restarts':restarts,'alerts':alerts,'local_free_bytes':local_free,'publish_free_bytes':publish_free,
                'recent_workers':latest_workers,**summary})
            time.sleep(args.interval)
    finally:
        if handle is not None:handle.close()
        ledger.close();own_lock.close()


if __name__=='__main__':raise SystemExit(main())
