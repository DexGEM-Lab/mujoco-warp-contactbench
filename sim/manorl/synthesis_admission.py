"""Append-only model admission with disjoint host ownership and local SQLite."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

from sim.manorl.synthesis_campaign import CAMPAIGN_CONTRACT, QuotaLedger, atomic_json, canonical_hash

REGISTRY_CONTRACT = 'manorl_synthesis_object_owner_v1'


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def completed_job(report: Path) -> dict | None:
    """Require both final training evidence and the matching native checkpoint."""
    data=json.loads(report.read_text())
    if data.get('schema')!='manorl.cube1_fast_training.v1':return None
    if data.get('actual',{}).get('updates',0)<5000:return None
    checkpoint=report.with_suffix('.pt');sidecar=Path(str(checkpoint)+'.json')
    if not checkpoint.is_file() or not sidecar.is_file():return None
    metadata=json.loads(sidecar.read_text());runtime=metadata.get('runtime_config',{})
    if metadata.get('checkpoint_file')!=checkpoint.name:return None
    if runtime.get('training_progress',{}).get('completed_updates',0)<5000:return None
    env=runtime.get('environment',{});budget=data.get('budget',{})
    if not (budget.get('raw_transfer') is True and env.get('pre_padding')==80
            and env.get('compatibility',{}).get('early_phase_steps')==80
            and env.get('reference_fps')==120 and env.get('physics_fps')==480
            and env.get('controlled_hand_sides')==['right']):
        return None
    pairs=runtime.get('trajectory_selection',{}).get('resolved_pairs',[])
    objects={p['object'] for p in pairs}
    if len(objects)!=1 or not budget.get('trajectory_packages'):
        raise ValueError(f'completed object report has ambiguous coverage: {report}')
    obj=next(iter(objects))
    if not re.fullmatch(r'[A-Za-z0-9_-]+',obj):raise ValueError('invalid object identifier')
    return {'object':obj,'checkpoint':str(checkpoint.resolve()),'report':str(report.resolve()),
            'pairs':sorted(p['object']+':'+p['action'] for p in pairs),
            'packages':budget['trajectory_packages']}


def ownership_record(registry: Path, obj: str) -> dict | None:
    path=registry/'objects'/f'{obj}.json'
    if not path.exists():return None
    value=json.loads(path.read_text())
    if value.get('contract')!=REGISTRY_CONTRACT:raise ValueError('unknown object ownership contract')
    return value


def claim_object(registry: Path, *, obj: str, host: str, campaign_id: str, checkpoint_sha: str, output: str) -> bool:
    """Exclusive-create an immutable NAS owner record; never steal ownership."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+',obj):raise ValueError('invalid object identifier')
    folder=registry/'objects';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{obj}.json'
    value={'contract':REGISTRY_CONTRACT,'object':obj,'host':host,'campaign_id':campaign_id,
           'checkpoint_sha256':checkpoint_sha,'output':output,'created':time.time()}
    try:
        with path.open('x') as stream:
            json.dump(value,stream,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        return True
    except FileExistsError:
        prior=ownership_record(registry,obj)
        if prior['host']!=host or prior['campaign_id']!=campaign_id:return False
        if prior['checkpoint_sha256']!=checkpoint_sha:
            raise ValueError(f'{obj} already bound to another checkpoint; explicit migration required')
        return True


def snapshot_checkpoint(job: dict, root: Path) -> dict:
    source=Path(job['checkpoint']);sidecar=Path(str(source)+'.json')
    digest=sha256(source);side_digest=sha256(sidecar)
    folder=root/'checkpoints'/job['object']/digest;folder.mkdir(parents=True,exist_ok=True)
    target=folder/source.name
    for old,new,expected in ((source,target,digest),(sidecar,Path(str(target)+'.json'),side_digest)):
        if not new.exists():
            temporary=new.with_name(new.name+'.copy-'+uuid.uuid4().hex)
            shutil.copyfile(old,temporary)
            if sha256(temporary)!=expected:raise RuntimeError('checkpoint changed while copying')
            os.replace(temporary,new)
        if sha256(new)!=expected:raise RuntimeError('immutable checkpoint snapshot differs')
    if sha256(source)!=digest or sha256(sidecar)!=side_digest:
        raise RuntimeError('final checkpoint changed during admission')
    return {**job,'checkpoint':str(target),'checkpoint_sha256':digest}


def register_existing(ledger: QuotaLedger, registry: Path, *, host: str, campaign_id: str, output: str) -> None:
    objects=defaultdict(set)
    for (raw,) in ledger.db.execute('SELECT spec FROM sources'):
        spec=json.loads(raw);objects[spec['object']].add(spec['checkpoint_sha256'])
    for obj,hashes in objects.items():
        if len(hashes)!=1:raise ValueError(f'local object {obj} has mixed checkpoints')
        if not claim_object(registry,obj=obj,host=host,campaign_id=campaign_id,
                            checkpoint_sha=next(iter(hashes)),output=output):
            raise RuntimeError(f'existing local object {obj} is owned by another host')


def refresh_source_manifest(ledger: QuotaLedger, campaign: Path) -> list[dict]:
    sources=[json.loads(row[0]) for row in ledger.db.execute('SELECT spec FROM sources ORDER BY key')]
    atomic_json(campaign/'sources.json',sources)
    return sources


def append_prepared(ledger: QuotaLedger, campaign: Path, prepared: Path, config: dict) -> int:
    """Idempotent admission: existing accepted/leased slots are never rewritten."""
    stage_config=json.loads((prepared/'config.json').read_text())
    for field in ('contract','prefix','ccd_contacts_per_world','contacts_per_world','constraint_capacity','initial_penetration_tolerance_m'):
        if stage_config.get(field)!=config.get(field):raise ValueError(f'admission changed generation contract: {field}')
    sources=json.loads((prepared/'sources.json').read_text())
    if not sources:raise ValueError('cannot admit an empty object')
    if len({s['object'] for s in sources})!=1:raise ValueError('admit one object at a time')
    for source in sources:
        if sha256(Path(source['pickle']))!=source['pickle_sha256']:
            raise RuntimeError('prepared source changed before admission')
    before=ledger.summary()['sources']
    ledger.initialize(config,sources)
    refresh_source_manifest(ledger,campaign)
    metadata=json.loads((campaign/'checkpoints.json').read_text()) if (campaign/'checkpoints.json').exists() else {}
    metadata.update(json.loads((prepared/'checkpoints.json').read_text()))
    atomic_json(campaign/'checkpoints.json',metadata)
    count=ledger.summary()['sources']-before
    if count and (campaign/'completed.json').exists():
        history=campaign/'completion-history';history.mkdir(exist_ok=True)
        os.rename(campaign/'completed.json',history/f'{time.time_ns()}.json')
    return count


def execution_revision() -> str:
    """Label the worker's immutable source, not the campaign's original revision."""
    marker=Path(__file__).resolve().parents[2]/'DEPLOYED_COMMIT'
    value=os.environ.get('MANORL_SOFTWARE_COMMIT')
    if not value and marker.is_file():value=marker.read_text().strip()
    if not value or not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})',value):
        raise RuntimeError('production worker requires an explicit immutable source revision')
    return value


def admitted_gpu_pool(campaign: Path, configured: list[int]) -> list[int]:
    path=campaign/'gpu-pool.json'
    if not path.exists():return list(configured)  # legacy fixed-pool campaigns
    values=json.loads(path.read_text()).get('gpus')
    if not isinstance(values,list) or any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in values):
        raise ValueError('GPU admission file must contain nonnegative integer gpus')
    if len(set(values))!=len(values) or not set(values).issubset(configured):
        raise ValueError('GPU admission exceeds the configured physical GPU pool')
    return [gpu for gpu in configured if gpu in values]


def eligible_gpus(policy: dict, admitted_objects: set[str]) -> list[int]:
    baseline=policy.get('always',[]);gates=policy.get('after_objects',{})
    allowed=set()
    for value in baseline:
        if not isinstance(value,int) or isinstance(value,bool) or value<0:raise ValueError('invalid GPU policy')
        allowed.add(value)
    for gpu,objects in gates.items():
        if not isinstance(objects,list) or not objects:raise ValueError('GPU gate must name completed objects')
        if all(obj in admitted_objects for obj in objects):allowed.add(int(gpu))
    return sorted(allowed)
