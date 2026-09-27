"""CPU-only retreat, verification and atomic compact shard publication.

Runs in a separate process from CUDA so Arrow/Lance cannot retain the GPU or
block the next rollout wave. Published shards contain their own recovery record.
"""
from __future__ import annotations

from collections import Counter
import errno
import gc
import json
import os
from pathlib import Path
import pickle
import shutil
import time
from types import SimpleNamespace
import uuid

import numpy as np

from sim.manorl.synthesis_campaign import CAMPAIGN_CONTRACT, atomic_json, ledger_schema_state


def runtime_context(runtime, software_commit: str) -> dict:
    return {'checkpoint':str(runtime.checkpoint), 'checkpoint_sha':runtime.checkpoint_sha,
        'metadata':runtime.metadata,'physical_manifest':runtime.physical_manifest,
        'object_name':runtime.object_name,'scene_names':runtime.scene_names,
        'ccd_iterations':runtime.environment.config.warp_ccd_iterations,
        'ccd_contacts':runtime.environment.config.warp_ccd_contacts_per_world,
        'software_commit':software_commit}


def _cpu_processor(context):
    from sim.manorl import assets
    from sim.manorl.synthesis_runtime import BatchSynthesisRuntime
    from sim.manorl.contracts import ServoConfig
    runtime=object.__new__(BatchSynthesisRuntime)
    runtime.checkpoint=Path(context['checkpoint']);runtime.checkpoint_sha=context['checkpoint_sha']
    runtime.metadata=context['metadata'];runtime.physical_manifest=context['physical_manifest']
    runtime.object_name=context['object_name'];runtime.scene_names=tuple(context['scene_names'])
    if len(runtime.scene_names)==1:
        mj,model=assets.compile_model(ServoConfig(),object_type=runtime.object_name,hand_side='right',physics_timestep=1/480)
    else:
        mj,model=assets.compile_unified_model(ServoConfig(),object_types=runtime.scene_names,hand_side='right',object_collisions=True,physics_timestep=1/480)
    model.opt.ccd_iterations=context['ccd_iterations']
    runtime.environment=SimpleNamespace(model=model,mujoco=mj,joint_lower=model.jnt_range[:28,0].copy(),
        joint_upper=model.jnt_range[:28,1].copy(),config=SimpleNamespace(warp_ccd_iterations=context['ccd_iterations'],
            warp_ccd_contacts_per_world=context['ccd_contacts']))
    from sim.manorl.environment import MjxWarpPhysicalProducer
    producer=MjxWarpPhysicalProducer(mj,model,object_type=runtime.object_name)
    runtime.hand_geoms=set(producer.keypoint_geom_ids_by_side['right'])
    runtime.object_qpos={}
    for name in runtime.scene_names:
        joint=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,assets.object_runtime(name).free_joint_name)
        runtime.object_qpos[name]=int(model.jnt_qposadr[joint])
    return runtime


def promote_verified_shard(temporary: Path, final: Path, *, attempts: int = 4) -> None:
    """Bounded, visible retry of CIFS sharing/permission transients only.

    Caller has already verified the row readback, payload hashes and commit
    manifest. Never overwrite a final shard or regenerate rows during retry.
    """
    if attempts < 1:raise ValueError('promotion attempts must be positive')
    if not (temporary/'campaign-commit.json').is_file():
        raise RuntimeError('cannot promote a shard without its recovery manifest')
    for attempt in range(attempts):
        if final.exists():raise FileExistsError('final shard exists; reconcile rather than overwrite')
        try:
            os.rename(temporary,final)
            return
        except OSError as exc:
            if (exc.errno not in (errno.EACCES,errno.EPERM,errno.EBUSY)
                    or attempt+1==attempts or final.exists() or not temporary.exists()):
                raise
            delay=min(2**attempt,4)
            print(json.dumps({'event':'publish_rename_retry','attempt':attempt+1,
                'errno':exc.errno,'error':str(exc),'delay_seconds':delay,
                'temporary':str(temporary),'final':str(final)}),flush=True)
            gc.collect()
            time.sleep(delay)


def publish_bundle(bundle_path: Path) -> dict:
    from sim.manorl.lance_v2 import write_compact_lance_stream, file_sha256
    import lance
    started=time.perf_counter()
    with bundle_path.open('rb') as f:bundle=pickle.load(f)
    context=bundle['context'];results=bundle['results']
    runtime=_cpu_processor(context)
    rows=[];records=[];failures=Counter()
    for result in results:
        row=runtime.compact_with_retreat(result,software_commit=context['software_commit'])
        record={'claim':result['claim'],'diagnostic':result['diagnostic'],'row_uuid':None}
        if row is not None:
            rows.append(row);record['row_uuid']=row['index']['uuid']
        else:
            reason=result['diagnostic'].get('retreat_failure') or result['diagnostic'].get('prefix_failure') or 'manipulation_gate'
            failures[str(reason)]+=1
        records.append(record)
    retreat_seconds=time.perf_counter()-started
    shard=None;payload_bytes=0
    if rows:
        root=Path(bundle['output'])/'shards'/runtime.object_name
        root.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(root).free < 10*1024**3:
            raise RuntimeError('publication filesystem has less than 10GiB free')
        name=bundle['bundle_id']+'.lance';shard=root/name
        temporary=root/('.'+name+'.partial')
        if shard.exists() or temporary.exists():
            raise FileExistsError('shard identity already exists; reconcile it instead of overwriting')
        write_compact_lance_stream(rows,output=temporary,batch_size=4)
        dataset=lance.dataset(str(temporary))
        readback=dataset.to_table(columns=['index']).to_pylist()
        expected=[row['index'] for row in rows]
        actual=[row['index'] for row in readback]
        if actual != expected or len({x['uuid'] for x in actual}) != len(expected):
            raise RuntimeError('published shard UUID/source readback mismatch')
        if 'rollout' in dataset.schema.names:
            raise RuntimeError('compact shard unexpectedly contains rollout data')
        # Release readback/scanner handles before renaming a directory on CIFS.
        # Keep the Python row/UUID evidence, not the live Lance dataset handle.
        del dataset
        gc.collect()
        # Re-read actual payloads, not only an in-memory Arrow table.
        hashes={}
        for path in sorted(temporary.rglob('*')):
            if path.is_file():
                payload_bytes+=path.stat().st_size
                hashes[str(path.relative_to(temporary))]=file_sha256(path)
        accepted_records=[r for r in records if r['row_uuid']]
        commit={'contract':CAMPAIGN_CONTRACT,'bundle_id':bundle['bundle_id'],
            'rows':accepted_records,'payload_sha256':hashes,'payload_bytes':payload_bytes,
            'checkpoint_metadata':context['metadata'],'physical_hand':context['physical_manifest']['hand_operator'],
            'software_commit':context['software_commit'],'phase_contract':{
                'prefix':'MJX-Warp solved','manipulation':'MJX-Warp checkpoint policy',
                'retreat':'offline kinematic XYZ; targets/keypoints translated; no added geometry penetration vs solved base',
                'contacts':'original MJX-Warp solved normal forces; no CPU reconstruction'}}
        atomic_json(temporary/'campaign-commit.json',commit)
        promote_verified_shard(temporary,shard)
        if lance.dataset(str(shard)).count_rows()!=len(rows):
            raise RuntimeError('final shard row count changed after promotion')
    summary={'contract':CAMPAIGN_CONTRACT,'bundle_id':bundle['bundle_id'],
        'shard':None if shard is None else str(shard),'records':records,
        'accepted':len(rows),'failures':dict(failures),'payload_bytes':payload_bytes,
        'retreat_seconds':retreat_seconds,'publish_seconds':time.perf_counter()-started-retreat_seconds,
        'wall_seconds':time.perf_counter()-started}
    atomic_json(bundle_path.with_suffix('.result.json'),summary)
    return summary


def reconcile_shards(ledger, output: Path, *, verify_hashes: bool = True) -> int:
    """Recover promotion-before-SQL crashes BEFORE dead-owner lease release.

    Reads both ledger generations: historical manifests use the same claim
    shape, and the explicit legacy-to-pair-budget migration reconciles promoted
    shards while the ledger is still in the legacy schema.
    """
    from sim.manorl.lance_v2 import file_sha256
    table, column = (('outputs', 'variant') if ledger_schema_state(ledger.db) == 'pair_budget'
                     else ('slots', 'slot'))
    recovered = 0
    for path in sorted((output/'shards').glob('*/*.lance/campaign-commit.json')):
        commit = json.loads(path.read_text())
        if commit.get('contract') != CAMPAIGN_CONTRACT:
            raise RuntimeError('foreign shard in campaign output')
        pending = []
        for record in commit['rows']:
            claim = record['claim']
            # 'slot' is the published lineage name of the variant index in both
            # the legacy and pair-budget schemas, so historical manifests keep
            # reconciling unchanged.
            row = ledger.db.execute(
                f'SELECT state,row_uuid,shard FROM {table} WHERE source_key=? AND {column}=?',
                (claim['source']['key'], claim['slot'])).fetchone()
            if row is None:
                # A completed authority transfer retired this source locally.
                # Skip ONLY an exact already-credited transferred row (source,
                # variant, UUID and its ORIGINAL shard); anything else — an
                # unknown source, a foreign UUID or a shard conflict — is a
                # real reconciliation failure and still fails loudly.
                matches = getattr(ledger, 'retired_credit_matches', None)
                if (matches is not None and matches(claim['source']['key'], claim['slot'],
                                                    record['row_uuid'], str(path.parent))):
                    continue
                raise RuntimeError('published source absent from ledger')
            if row['state']=='accepted':
                if row['row_uuid']!=record['row_uuid'] or row['shard']!=str(path.parent):
                    raise RuntimeError('two published rows compete for one source slot')
            else:pending.append(record)
        if pending:
            if verify_hashes:
                for relative,expected in commit['payload_sha256'].items():
                    if file_sha256(path.parent/relative)!=expected:raise RuntimeError('recovery shard hash mismatch')
            for record in pending:
                ledger.finish(record['claim'],record['diagnostic'],row_uuid=record['row_uuid'],shard=str(path.parent))
                recovered+=1
    return recovered
