"""Explicit single-object authority transfer between two campaign ledgers.

One object at a time, under externally established FULL campaign quiescence
(both campaigns drained: no leased outputs, no live worker owner locks). No
heuristics, no expiry, no automatic balancing — every step is an explicit
operator command, idempotent or refusing with the exact recovery state.

Protocol:

1. ``export``        source host: relationally complete object bundle + inputs
                     staged under a host-local directory and hashed; immutable
                     once written (mode 0444, never rewritten).
2. ``import``        destination host: verify staging manifest, campaign
                     config identity, registry ownership and collisions; copy
                     inputs byte-identically from the staging bundle rebasing
                     ONLY machine-local pickle/checkpoint paths; stage ledger
                     rows in ONE transaction; durable receipt; refresh
                     manifests.
3. ``detach``        source host: require the verified destination receipt and
                     unchanged source fingerprints; retire the object's ACTIVE
                     rows in ONE transaction while keeping exact retired-credit
                     receipts so shard reconciliation skips ONLY exact
                     already-credited rows; durable detach receipt.
4. ``commit-owner``  registry only, after BOTH receipts verify: append an
                     immutable transfer record and atomically flip the
                     current-owner pointer. The original owner record file is
                     never rewritten.
5. ``finalize``      each host: verify the registry now shows the destination
                     as owner and mark the local transfer row complete.

No source state is deleted before a verified destination receipt exists. No
automatic rollback exists after the destination can generate. An incomplete
handoff leaves a pending ``authority_transfers`` row that blocks claims and
campaign completion on both sides until finalized.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

from sim.manorl.synthesis_campaign import QuotaLedger, atomic_json, canonical_hash, owner_lock_busy
from sim.manorl.synthesis_admission import REGISTRY_CONTRACT, ownership_record

TRANSFER_CONTRACT = 'manorl_synthesis_object_authority_transfer_v1'
# Generation-contract fields that must be identical between the two campaigns
# (mirrors append_prepared's admission checks; local output/software fields
# deliberately excluded — the destination publishes under its own root).
GENERATION_FIELDS = ('contract', 'prefix', 'ccd_contacts_per_world', 'contacts_per_world',
                     'constraint_capacity', 'initial_penetration_tolerance_m')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _valid_object_name(obj: str) -> None:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', obj):
        raise ValueError('invalid object identifier')


def _object_pairs(db, obj: str) -> list[str]:
    pairs = sorted({row[0] for row in db.execute('SELECT DISTINCT pair FROM sources')
                    if row[0].rsplit(':', 1)[0] == obj})
    if not pairs:
        raise RuntimeError(f'no ledger sources for object {obj} in this campaign')
    return pairs


def _object_source_rows(db, obj: str) -> list:
    pairs = _object_pairs(db, obj)
    rows = db.execute(
        f'SELECT key,group_key,pair,spec FROM sources WHERE pair IN ({",".join("?" * len(pairs))}) '
        f'ORDER BY key', pairs).fetchall()
    bad = [row['key'] for row in rows if json.loads(row['spec']).get('object') != obj]
    if bad:
        raise RuntimeError(f'source specs disagree with pair prefix {obj}: {bad[:3]}')
    checkpoints = {json.loads(row['spec']).get('checkpoint_sha256') for row in rows}
    if None in checkpoints or len(checkpoints) != 1:
        raise RuntimeError(f'object {obj} must map to exactly one checkpoint')
    return rows


def bundle_fingerprint(db, obj: str) -> dict:
    """Deterministic relational snapshot of one object's ledger state."""
    rows = _object_source_rows(db, obj)
    keys = [row['key'] for row in rows]
    marks = ','.join('?' * len(keys))
    outputs = db.execute(
        f'SELECT pair,source_key,variant,state,attempts,owner,token,error,row_uuid,shard '
        f'FROM outputs WHERE source_key IN ({marks}) ORDER BY source_key,variant', keys).fetchall()
    excluded_keys = {row[0] for row in db.execute(
        f'SELECT source_key FROM source_exclusions WHERE source_key IN ({marks})', keys)}
    ready_excluded = [f'{row["source_key"]}:{row["variant"]}' for row in outputs
                      if row['state'] == 'ready' and row['source_key'] in excluded_keys]
    if ready_excluded:
        raise RuntimeError(f'excluded sources still hold ready rows (ledger invariant broken): '
                           f'{ready_excluded[:3]}')
    policy = db.execute('SELECT minimum_rate,minimum_attempts FROM source_success_policy WHERE id=1').fetchone()
    return {
        'sources': [[row['key'], row['group_key'], row['pair'], row['spec']] for row in rows],
        'pair_targets': [list(db.execute(
            'SELECT pair,original_sources,target_rows FROM pair_targets WHERE pair=?', (pair,)).fetchone())
            for pair in _object_pairs(db, obj)],
        'outputs': [[row['pair'], row['source_key'], row['variant'], row['state'], row['attempts'],
                     row['owner'], row['token'], row['error'], row['row_uuid'], row['shard']]
                    for row in outputs],
        'attempts': [list(row) for row in db.execute(
            f'SELECT token,source_key,slot,attempt,started,finished,diagnostic,allocated_seed '
            f'FROM attempts WHERE source_key IN ({marks}) ORDER BY token', keys)],
        'source_exclusions': [list(row) for row in db.execute(
            f'SELECT source_key,decided,settled_attempts,accepted_rows,minimum_rate,minimum_attempts '
            f'FROM source_exclusions WHERE source_key IN ({marks}) ORDER BY source_key', keys)],
        'source_exclusion_history': [list(row) for row in db.execute(
            f'SELECT id,source_key,action,decided,settled_attempts,accepted_rows,pending_claims,'
            f'minimum_rate,minimum_attempts,evidence FROM source_exclusion_history '
            f'WHERE source_key IN ({marks}) ORDER BY id', keys)],
        'policy': None if policy is None else dict(policy),
    }


def assert_campaign_quiescent(root: Path, ledger: QuotaLedger) -> None:
    """Documented FULL campaign quiescence: zero leased outputs anywhere and no
    worker owner lock currently held. Lock liveness is checked directly (flock
    of every recorded worker owner); timestamps are never extrapolated."""
    leased = int(ledger.db.execute("SELECT COUNT(*) FROM outputs WHERE state='leased'").fetchone()[0])
    if leased:
        raise RuntimeError(f'{leased} leased outputs remain: drain writers and settle publications '
                           'before transferring object authority')
    owners = [row[0] for row in ledger.db.execute('SELECT owner FROM workers')]
    live = sorted(owner for owner in owners if owner_lock_busy(root, owner))
    if live:
        raise RuntimeError(f'live worker owner locks held ({live}); stop this campaign\'s workers first')


def _place_bytes(path: Path, data: bytes) -> None:
    """Write-once semantics: existing identical content is a replay, any
    difference is a loud refusal. Existing paths are never overwritten."""
    if path.exists():
        if path.read_bytes() == data:
            return
        raise RuntimeError(f'refusing to overwrite existing transfer path with new content: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _place_file(source: Path, target: Path, expected_sha: str) -> None:
    actual = _sha256(source)
    if actual != expected_sha:
        raise RuntimeError(f'input changed under transfer: {source}')
    if target.exists():
        if _sha256(target) == expected_sha:
            return
        raise RuntimeError(f'refusing to overwrite existing destination input: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if _sha256(target) != expected_sha:
        raise RuntimeError(f'copied input does not verify: {target}')


def _read_staging(staging: Path) -> tuple[dict, dict]:
    manifest_path = staging / 'manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError(f'staging directory has no manifest.json: {staging}')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('contract') != TRANSFER_CONTRACT:
        raise RuntimeError('unknown transfer staging contract')
    return manifest, json.loads((staging / 'bundle.json').read_text())


def verify_staging(staging: Path) -> dict:
    """Verify every hashed file of the bundle exactly; refuse extra files."""
    staging = Path(staging).resolve()
    manifest, bundle = _read_staging(staging)
    allowed = set(manifest['files']) | {'manifest.json', 'journal.json'}
    actual = {str(path.relative_to(staging)) for path in staging.rglob('*') if path.is_file()}
    extras = sorted(actual - allowed)
    if extras:
        raise RuntimeError(f'unexpected extra files in staging directory: {extras[:5]}')
    for relative, expected in sorted(manifest['files'].items()):
        path = staging / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f'staging file missing or hash mismatch: {relative}')
    if bundle.get('transfer_id') != manifest['transfer_id']:
        raise RuntimeError('staging manifest and bundle disagree on transfer identity')
    return {'transfer_id': manifest['transfer_id'], 'object': manifest['object'],
            'bundle_sha256': manifest['bundle_sha256'], 'staging': str(staging)}


def _mark_immutable(staging: Path) -> None:
    for path in staging.rglob('*'):
        if path.is_file():
            os.chmod(path, 0o444)


def _authority_row(ledger: QuotaLedger, transfer_id: str):
    return ledger.db.execute('SELECT * FROM authority_transfers WHERE transfer_id=?',
                             (transfer_id,)).fetchone()


def _refuse_pending_for_object(ledger: QuotaLedger, obj: str, *, exclude: str | None = None) -> None:
    for row in ledger.db.execute(
            "SELECT transfer_id,direction,state FROM authority_transfers WHERE object=? AND state!='complete'",
            (obj,)).fetchall():
        if row['transfer_id'] != exclude:
            raise RuntimeError(f'object {obj} already has transfer {row["transfer_id"]} in state '
                               f'{row["direction"]}/{row["state"]}; finish or finalize it first')


def _write_receipt(path: Path, receipt: dict) -> None:
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise RuntimeError(f'refusing to overwrite an existing transfer receipt: {path}')
    atomic_json(path, receipt)


# --------------------------------------------------------------------------- #
# Step 1: export (source host)
# --------------------------------------------------------------------------- #

def export_object(campaign: Path, *, obj: str, registry: Path, staging: Path,
                  host: str, campaign_id: str, destination_host: str, destination_campaign: str,
                  transfer_id: str | None = None) -> dict:
    _valid_object_name(obj)
    root = Path(campaign).resolve()
    staging = Path(staging).resolve()
    config = json.loads((root / 'config.json').read_text())
    ledger = QuotaLedger(root / 'quota.sqlite')
    try:
        assert_campaign_quiescent(root, ledger)
        rows = _object_source_rows(ledger.db, obj)
        checkpoint_sha = next(iter({json.loads(row['spec'])['checkpoint_sha256'] for row in rows}))
        transfer_id = transfer_id or f'xfer-{time.strftime("%Y%m%dT%H%M%S")}-{uuid.uuid4().hex[:8]}'
        _refuse_pending_for_object(ledger, obj, exclude=transfer_id)
        owner = ownership_record(Path(registry), obj)
        if owner is None:
            raise RuntimeError(f'object {obj} has no registry owner record; nothing to transfer')
        if (owner['host'] != host or owner['campaign_id'] != campaign_id
                or owner['checkpoint_sha256'] != checkpoint_sha or owner.get('output') != config['output']):
            raise RuntimeError('registry owner record does not match this campaign; refusing '
                               f'ambiguous ownership: {owner}')
        fingerprint = bundle_fingerprint(ledger.db, obj)
        digest = canonical_hash(fingerprint)
        replay = _authority_row(ledger, transfer_id)
        if replay is not None:
            if replay['direction'] != 'outbound' or replay['state'] not in ('exported', 'detached'):
                raise RuntimeError(f'transfer {transfer_id} exists as {replay["direction"]}/{replay["state"]}')
            if replay['state'] != 'exported':
                raise RuntimeError(f'transfer {transfer_id} is already detached; use its receipts')
            if replay['bundle_sha256'] != digest:
                raise RuntimeError('source ledger changed since this transfer was exported; '
                                   'the staging bundle no longer matches the ledger')
        exported_at = float(replay['updated']) if replay is not None else time.time()
        inputs = {'pickles': {}, 'checkpoints': {}}
        for row in rows:
            spec = json.loads(row['spec'])
            _place_file(Path(spec['pickle']), staging / 'sources' / f'{row["key"]}.pkl',
                        spec['pickle_sha256'])
            inputs['pickles'][row['key']] = {'source_path': spec['pickle'],
                                             'sha256': spec['pickle_sha256']}
            if spec['checkpoint_sha256'] not in inputs['checkpoints']:
                sidecar = Path(str(spec['checkpoint']) + '.json')
                inputs['checkpoints'][spec['checkpoint_sha256']] = {
                    'name': Path(spec['checkpoint']).name,
                    'sha256': spec['checkpoint_sha256'],
                    'sidecar_name': sidecar.name,
                    'sidecar_sha256': _sha256(sidecar)}
        for sha, meta in sorted(inputs['checkpoints'].items()):
            source = next(Path(json.loads(row['spec'])['checkpoint']) for row in rows
                          if json.loads(row['spec'])['checkpoint_sha256'] == sha)
            folder = staging / 'checkpoints' / sha[:16]
            _place_file(source, folder / meta['name'], sha)
            _place_file(Path(str(source) + '.json'), folder / meta['sidecar_name'],
                        meta['sidecar_sha256'])
        bundle = {
            'contract': TRANSFER_CONTRACT, 'transfer_id': transfer_id, 'object': obj,
            'source': {'host': host, 'campaign_id': campaign_id, 'campaign_root': str(root),
                       'output': config['output']},
            'destination': {'host': destination_host, 'campaign_id': destination_campaign},
            'config': {field: config.get(field) for field in GENERATION_FIELDS},
            'owner_record': owner,
            'groups': sorted({row['group_key'] for row in rows}),
            'pairs': {row[0]: {'original_sources': row[1], 'target_rows': row[2]}
                      for row in fingerprint['pair_targets']},
            'inputs': inputs,
            'counts': {
                'sources': len(fingerprint['sources']),
                'outputs': len(fingerprint['outputs']),
                'accepted_rows': sum(1 for row in fingerprint['outputs'] if row[3] == 'accepted'),
                'attempts': len(fingerprint['attempts']),
                'exclusions': len(fingerprint['source_exclusions']),
                'exclusion_history': len(fingerprint['source_exclusion_history'])},
            'bundle_sha256': digest, 'exported_at': exported_at}
        _place_bytes(staging / 'bundle.json',
                     json.dumps(bundle, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b'\n')
        _place_bytes(staging / 'ledger-export.json',
                     json.dumps({'contract': TRANSFER_CONTRACT, 'transfer_id': transfer_id,
                                 'bundle_sha256': digest, 'fingerprint': fingerprint},
                                sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b'\n')
        manifest = {'contract': TRANSFER_CONTRACT, 'transfer_id': transfer_id, 'object': obj,
                    'bundle_sha256': digest,
                    'files': {str(path.relative_to(staging)): _sha256(path)
                              for path in sorted(staging.rglob('*'))
                              if path.is_file() and path.name not in ('manifest.json', 'journal.json')
                              and '.tmp-' not in path.name},
                    'unhashed': ['journal.json']}
        _place_bytes(staging / 'manifest.json',
                     json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b'\n')
        journal = {'contract': TRANSFER_CONTRACT, 'transfer_id': transfer_id, 'object': obj,
                   'phases': {'exported': {'host': host, 'at': exported_at}}, 'complete': False,
                   'note': 'intermediate handoff states never mark the overall campaign complete'}
        _place_bytes(staging / 'journal.json',
                     json.dumps(journal, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b'\n')
        if replay is None:
            with ledger.db:
                ledger.db.execute(
                    'INSERT INTO authority_transfers VALUES(?,?,?,?,?,?,?,?)',
                    (transfer_id, obj, 'outbound', destination_host, destination_campaign,
                     digest, 'exported', exported_at))
        _mark_immutable(staging)
        return {'contract': TRANSFER_CONTRACT, 'transfer_id': transfer_id, 'object': obj,
                'staging': str(staging), 'bundle_sha256': digest, 'state': 'exported',
                'counts': bundle['counts'], 'pairs': bundle['pairs']}
    finally:
        ledger.close()


# --------------------------------------------------------------------------- #
# Step 2: import (destination host)
# --------------------------------------------------------------------------- #

def _validate_destination(ledger: QuotaLedger, config: dict, bundle: dict, fingerprint: dict) -> None:
    for field in GENERATION_FIELDS:
        if bundle['config'].get(field) != config.get(field):
            raise ValueError(f'transfer rejected: campaign generation contract differs: {field}')
    keys = {row[0] for row in fingerprint['sources']}
    clash = sorted(keys & {row[0] for row in ledger.db.execute('SELECT key FROM sources')})
    if clash:
        raise RuntimeError(f'destination already holds these source identities: {clash[:5]}')
    pairs = {row[0] for row in fingerprint['pair_targets']}
    clash = sorted(pairs & {row[0] for row in ledger.db.execute('SELECT pair FROM pair_targets')})
    if clash:
        raise RuntimeError(f'destination already holds these pair budgets: {clash[:5]}')
    groups = {row[1] for row in fingerprint['sources']}
    clash = sorted(groups & {row[0] for row in ledger.db.execute('SELECT DISTINCT group_key FROM sources')})
    if clash:
        raise RuntimeError(f'destination group keys collide: {clash[:5]}')
    uuids = {row[9] for row in fingerprint['outputs'] if row[9]}
    clash = sorted(uuids & {row[0] for row in ledger.db.execute(
        'SELECT row_uuid FROM outputs WHERE row_uuid IS NOT NULL')})
    if clash:
        raise RuntimeError(f'destination already holds these published row UUIDs: {clash[:5]}')
    tokens = {row[0] for row in fingerprint['attempts']}
    clash = sorted(tokens & {row[0] for row in ledger.db.execute('SELECT token FROM attempts')})
    if clash:
        raise RuntimeError(f'destination attempt tokens collide: {clash[:5]}')
    seeds = [row[7] for row in fingerprint['attempts'] if row[7] is not None]
    if seeds:
        clash = sorted(set(seeds) & {row[0] for row in ledger.db.execute(
            'SELECT allocated_seed FROM attempts WHERE allocated_seed IS NOT NULL')})
        if clash:
            raise RuntimeError(f'destination allocated seeds collide: {clash[:5]}')


def import_object(campaign: Path, *, registry: Path, staging: Path, host: str, campaign_id: str) -> dict:
    root = Path(campaign).resolve()
    staging = Path(staging).resolve()
    verify_staging(staging)
    bundle = json.loads((staging / 'bundle.json').read_text())
    obj = bundle['object']
    fingerprint = json.loads((staging / 'ledger-export.json').read_text())['fingerprint']
    if canonical_hash(fingerprint) != bundle['bundle_sha256']:
        raise RuntimeError('staged ledger export does not match the bundle digest')
    if bundle['destination']['host'] != host or bundle['destination']['campaign_id'] != campaign_id:
        raise RuntimeError('this campaign is not the bundle\'s declared destination')
    owner = ownership_record(Path(registry), obj)
    if owner is None or owner['host'] != bundle['source']['host'] \
            or owner['campaign_id'] != bundle['source']['campaign_id']:
        raise RuntimeError('registry ownership is not the bundle\'s source campaign; refusing '
                           f'ambiguous authority: {owner}')
    if (Path(registry) / 'objects' / f'{obj}.current.json').exists():
        raise RuntimeError('object ownership was already transferred; refusing duplicate import')
    config = json.loads((root / 'config.json').read_text())
    ledger = QuotaLedger(root / 'quota.sqlite')
    try:
        assert_campaign_quiescent(root, ledger)
        replay = _authority_row(ledger, bundle['transfer_id'])
        if replay is not None:
            if replay['direction'] != 'inbound' or replay['state'] != 'imported':
                raise RuntimeError(f'transfer {bundle["transfer_id"]} exists as '
                                   f'{replay["direction"]}/{replay["state"]} on this campaign')
            if replay['bundle_sha256'] != bundle['bundle_sha256']:
                raise RuntimeError('existing inbound transfer has a different bundle identity')
        else:
            _refuse_pending_for_object(ledger, obj)
            _validate_destination(ledger, config, bundle, fingerprint)
        # Stage machine-local inputs first: byte-identical copies from the
        # staging bundle, rebasing ONLY the local pickle/checkpoint paths.
        pickles, checkpoints = {}, {}
        for key, group, pair, spec_text in fingerprint['sources']:
            spec = json.loads(spec_text)
            target = root / 'sources' / obj / f'{key}.pkl'
            _place_file(staging / 'sources' / f'{key}.pkl', target, spec['pickle_sha256'])
            pickles[key] = str(target)
            sha = spec['checkpoint_sha256']
            if sha not in checkpoints:
                meta = bundle['inputs']['checkpoints'][sha]
                staged = staging / 'checkpoints' / sha[:16]
                checkpoint = root / 'checkpoints' / obj / sha / meta['name']
                _place_file(staged / meta['name'], checkpoint, sha)
                _place_file(staged / meta['sidecar_name'], Path(str(checkpoint) + '.json'),
                            meta['sidecar_sha256'])
                checkpoints[sha] = (str(checkpoint), json.loads(
                    Path(str(checkpoint) + '.json').read_text()))
        imported_at = float(replay['updated']) if replay is not None else time.time()
        if replay is None:
            ledger.db.execute('BEGIN IMMEDIATE')
            try:
                for key, group, pair, spec_text in fingerprint['sources']:
                    spec = json.loads(spec_text)
                    spec['pickle'] = pickles[key]
                    spec['checkpoint'] = checkpoints[spec['checkpoint_sha256']][0]
                    ledger.db.execute('INSERT INTO sources(key,group_key,pair,spec) VALUES(?,?,?,?)',
                                      (key, group, pair, json.dumps(spec, sort_keys=True)))
                for pair, original_sources, target_rows in fingerprint['pair_targets']:
                    ledger.db.execute('INSERT INTO pair_targets VALUES(?,?,?)',
                                      (pair, original_sources, target_rows))
                for row in fingerprint['source_exclusions']:
                    ledger.db.execute('INSERT INTO source_exclusions VALUES(?,?,?,?,?,?)', row)
                for row in fingerprint['source_exclusion_history']:
                    ledger.db.execute(
                        'INSERT INTO source_exclusion_history(source_key,action,decided,settled_attempts,'
                        'accepted_rows,pending_claims,minimum_rate,minimum_attempts,evidence) '
                        'VALUES(?,?,?,?,?,?,?,?,?)', row[1:])
                for row in fingerprint['outputs']:
                    ledger.db.execute(
                        'INSERT INTO outputs(pair,source_key,variant,state,attempts,owner,token,error,'
                        'row_uuid,shard) VALUES(?,?,?,?,?,?,?,?,?,?)', row)
                for row in fingerprint['attempts']:
                    ledger.db.execute(
                        'INSERT INTO attempts(token,source_key,slot,attempt,started,finished,diagnostic,'
                        'allocated_seed) VALUES(?,?,?,?,?,?,?,?)', row)
                ledger.db.execute('INSERT INTO authority_transfers VALUES(?,?,?,?,?,?,?,?)',
                                  (bundle['transfer_id'], obj, 'inbound', bundle['source']['host'],
                                   bundle['source']['campaign_id'], bundle['bundle_sha256'],
                                   'imported', imported_at))
                ledger.db.commit()
            except BaseException:
                ledger.db.rollback()
                raise
        counts = {'sources': len(fingerprint['sources']),
                  'outputs': len(fingerprint['outputs']),
                  'accepted_rows': sum(1 for row in fingerprint['outputs'] if row[3] == 'accepted'),
                  'attempts': len(fingerprint['attempts']),
                  'exclusions': len(fingerprint['source_exclusions'])}
        receipt = {'contract': TRANSFER_CONTRACT, 'transfer_id': bundle['transfer_id'],
                   'direction': 'inbound', 'object': obj, 'state': 'imported',
                   'host': host, 'campaign_id': campaign_id, 'campaign_root': str(root),
                   'output': config['output'],
                   'peer': {'host': bundle['source']['host'], 'campaign_id': bundle['source']['campaign_id']},
                   'bundle_sha256': bundle['bundle_sha256'], 'counts': counts,
                   'pairs': bundle['pairs'], 'imported_at': imported_at,
                   'note': 'historical accepted rows keep their ORIGINAL NAS shard paths; '
                           'new rows publish under this campaign\'s output root'}
        _write_receipt(root / 'transfer-receipts' / f'{bundle["transfer_id"]}.json', receipt)
        from sim.manorl.synthesis_admission import refresh_source_manifest
        refresh_source_manifest(ledger, root)
        metadata = json.loads((root / 'checkpoints.json').read_text()) \
            if (root / 'checkpoints.json').exists() else {}
        for sha, (path_text, sidecar) in checkpoints.items():
            metadata[sha] = {'path': path_text, 'sha256': sha, 'metadata': sidecar}
        atomic_json(root / 'checkpoints.json', metadata)
        return receipt
    finally:
        ledger.close()


# --------------------------------------------------------------------------- #
# Step 3: detach (source host)
# --------------------------------------------------------------------------- #

def _load_receipt(path: Path, *, direction: str, state: str, bundle: dict) -> dict:
    receipt = json.loads(Path(path).read_text())
    if (receipt.get('contract') != TRANSFER_CONTRACT or receipt.get('direction') != direction
            or receipt.get('state') != state or receipt.get('transfer_id') != bundle['transfer_id']
            or receipt.get('object') != bundle['object']
            or receipt.get('bundle_sha256') != bundle['bundle_sha256']):
        raise RuntimeError(f'receipt {path} does not verify against the transfer bundle')
    return receipt


def detach_object(campaign: Path, *, staging: Path, destination_receipt: Path) -> dict:
    root = Path(campaign).resolve()
    staging = Path(staging).resolve()
    verify_staging(staging)
    bundle = json.loads((staging / 'bundle.json').read_text())
    obj = bundle['object']
    exported = json.loads((staging / 'ledger-export.json').read_text())
    receipt = _load_receipt(destination_receipt, direction='inbound', state='imported', bundle=bundle)
    if (receipt['peer']['host'] != bundle['source']['host']
            or receipt['peer']['campaign_id'] != bundle['source']['campaign_id']
            or receipt['host'] != bundle['destination']['host']
            or receipt['campaign_id'] != bundle['destination']['campaign_id']
            or receipt['counts']['accepted_rows'] != bundle['counts']['accepted_rows']
            or receipt['counts']['sources'] != bundle['counts']['sources']):
        raise RuntimeError('destination receipt does not match the declared transfer endpoints')
    receipt_path = root / 'transfer-receipts' / f'{bundle["transfer_id"]}.detach.json'
    ledger = QuotaLedger(root / 'quota.sqlite')
    try:
        assert_campaign_quiescent(root, ledger)
        row = _authority_row(ledger, bundle['transfer_id'])
        if row is None or row['direction'] != 'outbound' or row['bundle_sha256'] != bundle['bundle_sha256']:
            raise RuntimeError(f'transfer {bundle["transfer_id"]} is not an outbound bundle of this campaign')
        accepted = [[row[1], row[2], row[8], row[9]] for row in exported['fingerprint']['outputs']
                    if row[3] == 'accepted']
        if row['state'] == 'detached':
            # Idempotent replay: the ACTIVE rows are already retired; verify the
            # durable retired credit against the export instead of the (gone)
            # live rows, then return the existing receipt.
            retired = [[r['source_key'], r['variant'], r['row_uuid'], r['shard']]
                       for r in ledger.db.execute(
                           'SELECT source_key,variant,row_uuid,shard FROM transferred_credit '
                           'WHERE transfer_id=? ORDER BY source_key,variant', (bundle['transfer_id'],))]
            if retired != accepted:
                raise RuntimeError('existing retired-credit receipts do not match the exported bundle')
            if not receipt_path.is_file():
                raise RuntimeError('detach receipt missing; recovery: restore it from the '
                                   'campaign transfer-receipts archive before continuing')
            previous = json.loads(receipt_path.read_text())
            if previous['transfer_id'] != bundle['transfer_id'] or previous['state'] != 'detached':
                raise RuntimeError('existing detach receipt has a different identity')
            return previous
        if row['state'] != 'exported':
            raise RuntimeError(f'outbound transfer is {row["state"]}; cannot detach')
        fingerprint = bundle_fingerprint(ledger.db, obj)
        if fingerprint != exported['fingerprint']:
            raise RuntimeError('source ledger changed after export; fingerprints differ — re-export '
                               'is required before any detach')
        keys = [source[0] for source in fingerprint['sources']]
        pairs = [target[0] for target in fingerprint['pair_targets']]
        marks = ','.join('?' * len(keys))
        ledger.db.execute('BEGIN IMMEDIATE')
        try:
            for out in fingerprint['outputs']:
                if out[3] == 'accepted':
                    ledger.db.execute('INSERT INTO transferred_credit VALUES(?,?,?,?,?,?)',
                                      (bundle['transfer_id'], out[0], out[1], out[2], out[8], out[9]))
            removed = {
                'outputs': ledger.db.execute(
                    f'DELETE FROM outputs WHERE source_key IN ({marks})', keys).rowcount,
                'attempts': ledger.db.execute(
                    f'DELETE FROM attempts WHERE source_key IN ({marks})', keys).rowcount,
                'source_exclusions': ledger.db.execute(
                    f'DELETE FROM source_exclusions WHERE source_key IN ({marks})', keys).rowcount,
                'source_exclusion_history': ledger.db.execute(
                    f'DELETE FROM source_exclusion_history WHERE source_key IN ({marks})', keys).rowcount,
                'sources': ledger.db.execute(
                    f'DELETE FROM sources WHERE key IN ({marks})', keys).rowcount,
                'pair_targets': sum(ledger.db.execute(
                    'DELETE FROM pair_targets WHERE pair=?', (pair,)).rowcount for pair in pairs)}
            retained = ledger.db.execute(
                'SELECT COUNT(*) FROM transferred_credit WHERE transfer_id=?',
                (bundle['transfer_id'],)).fetchone()[0]
            if retained != len(accepted) or removed['outputs'] != len(fingerprint['outputs']):
                raise RuntimeError('detach accounting mismatch; transaction rolled back')
            detached_at = time.time()
            ledger.db.execute('UPDATE authority_transfers SET state=?,updated=? WHERE transfer_id=?',
                              ('detached', detached_at, bundle['transfer_id']))
            ledger.db.commit()
        except BaseException:
            ledger.db.rollback()
            raise
        detach_receipt = {'contract': TRANSFER_CONTRACT, 'transfer_id': bundle['transfer_id'],
                          'direction': 'outbound', 'object': obj, 'state': 'detached',
                          'host': bundle['source']['host'], 'campaign_id': bundle['source']['campaign_id'],
                          'campaign_root': str(root),
                          'peer': {'host': receipt['host'], 'campaign_id': receipt['campaign_id']},
                          'bundle_sha256': bundle['bundle_sha256'],
                          'destination_receipt': str(Path(destination_receipt).resolve()),
                          'retired_rows': accepted,
                          'removed': removed,
                          'detached_at': detached_at,
                          'note': 'retired credit is kept as exact receipts; shard reconciliation '
                                  'skips only exact already-credited transferred rows'}
        _write_receipt(receipt_path, detach_receipt)
        return detach_receipt
    finally:
        ledger.close()


# --------------------------------------------------------------------------- #
# Step 4: commit-owner (registry only)
# --------------------------------------------------------------------------- #

def ownership_record_without_override(registry: Path, obj: str) -> dict | None:
    path = Path(registry) / 'objects' / f'{obj}.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get('contract') != REGISTRY_CONTRACT:
        raise ValueError('unknown object ownership contract')
    return value


def commit_registry_owner(registry: Path, *, staging: Path, source_receipt: Path,
                          destination_receipt: Path) -> dict:
    registry = Path(registry).resolve()
    staging = Path(staging).resolve()
    verify_staging(staging)
    bundle = json.loads((staging / 'bundle.json').read_text())
    obj = bundle['object']
    outbound = _load_receipt(source_receipt, direction='outbound', state='detached', bundle=bundle)
    inbound = _load_receipt(destination_receipt, direction='inbound', state='imported', bundle=bundle)
    if (outbound['peer']['host'] != inbound['host']
            or outbound['peer']['campaign_id'] != inbound['campaign_id']
            or outbound['host'] != inbound['peer']['host']
            or outbound['campaign_id'] != inbound['peer']['campaign_id']):
        raise RuntimeError('source and destination receipts describe different transfer endpoints')
    if not (len(outbound['retired_rows']) == inbound['counts']['accepted_rows']
            == bundle['counts']['accepted_rows']):
        raise RuntimeError('receipts disagree on the retired accepted-row count')
    original = ownership_record_without_override(registry, obj)
    if original is None:
        raise RuntimeError(f'object {obj} has no original owner record to transfer from')
    if (original['host'] != bundle['source']['host']
            or original['campaign_id'] != bundle['source']['campaign_id']
            or original['checkpoint_sha256'] != bundle['owner_record']['checkpoint_sha256']):
        raise RuntimeError('original owner record does not match the exported bundle')
    pointer = registry / 'objects' / f'{obj}.current.json'
    if pointer.exists():
        current = json.loads(pointer.read_text())
        if current.get('transfer_id') != bundle['transfer_id'] or current.get('host') != inbound['host']:
            raise RuntimeError(f'object {obj} already has a different current owner pointer')
        return {'contract': TRANSFER_CONTRACT, 'transfer_id': bundle['transfer_id'], 'object': obj,
                'state': 'committed', 'registry': str(registry), 'replay': True}
    folder = registry / 'objects' / f'{obj}.transfers'
    folder.mkdir(parents=True, exist_ok=True)
    sequence = len(list(folder.glob('*.json'))) + 1
    record = {'contract': TRANSFER_CONTRACT, 'transfer_id': bundle['transfer_id'], 'object': obj,
              'sequence': sequence,
              'from': {'host': original['host'], 'campaign_id': original['campaign_id'],
                       'output': original.get('output')},
              'to': {'host': inbound['host'], 'campaign_id': inbound['campaign_id'],
                     'campaign_root': inbound['campaign_root'], 'output': inbound['output']},
              'checkpoint_sha256': original['checkpoint_sha256'],
              'bundle_sha256': bundle['bundle_sha256'],
              'receipts': {'source': str(Path(source_receipt).resolve()),
                           'destination': str(Path(destination_receipt).resolve()),
                           'retired_rows': len(outbound['retired_rows'])},
              'committed_at': time.time()}
    _place_bytes(folder / f'{sequence:05d}-{bundle["transfer_id"]}.json',
                 json.dumps(record, sort_keys=True, ensure_ascii=False).encode() + b'\n')
    current = {'contract': REGISTRY_CONTRACT, 'object': obj, 'host': inbound['host'],
               'campaign_id': inbound['campaign_id'],
               'checkpoint_sha256': original['checkpoint_sha256'], 'output': inbound['output'],
               'created': record['committed_at'], 'transfer_id': bundle['transfer_id'],
               'transferred_from': record['from']}
    atomic_json(pointer, current)
    return {'contract': TRANSFER_CONTRACT, 'transfer_id': bundle['transfer_id'], 'object': obj,
            'state': 'committed', 'registry': str(registry),
            'record': str(folder / f'{sequence:05d}-{bundle["transfer_id"]}.json'),
            'current_owner': current, 'replay': False}


# --------------------------------------------------------------------------- #
# Step 5: finalize (each host)
# --------------------------------------------------------------------------- #

def finalize_transfer(campaign: Path, *, registry: Path, staging: Path,
                      host: str, campaign_id: str) -> dict:
    root = Path(campaign).resolve()
    staging = Path(staging).resolve()
    verify_staging(staging)
    bundle = json.loads((staging / 'bundle.json').read_text())
    obj = bundle['object']
    owner = ownership_record(Path(registry), obj)
    if owner is None:
        raise RuntimeError(f'object {obj} has no registry owner record')
    ledger = QuotaLedger(root / 'quota.sqlite')
    try:
        row = _authority_row(ledger, bundle['transfer_id'])
        if row is None:
            raise RuntimeError(f'transfer {bundle["transfer_id"]} is not recorded in this campaign')
        if row['bundle_sha256'] != bundle['bundle_sha256']:
            raise RuntimeError('local transfer row has a different bundle identity')
        if row['direction'] == 'outbound':
            if row['state'] == 'exported':
                raise RuntimeError('outbound transfer is still exported; detach before finalizing')
            if row['state'] not in ('detached', 'complete'):  # complete = idempotent replay
                raise RuntimeError(f'outbound transfer is {row["state"]}; nothing to finalize')
            if (owner['host'] != bundle['destination']['host']
                    or owner['campaign_id'] != bundle['destination']['campaign_id']):
                raise RuntimeError('registry ownership has not been committed to the destination')
        else:
            if row['state'] == 'complete':
                pass  # idempotent replay falls through to the same owner check below
            elif row['state'] != 'imported':
                raise RuntimeError(f'inbound transfer is {row["state"]}; nothing to finalize')
        if row['direction'] == 'inbound' and (owner['host'] != host or owner['campaign_id'] != campaign_id):
            raise RuntimeError('registry ownership does not name this campaign as the owner')
        if row['state'] != 'complete':
            with ledger.db:
                ledger.db.execute('UPDATE authority_transfers SET state=?,updated=? WHERE transfer_id=?',
                                  ('complete', time.time(), bundle['transfer_id']))
        return {'contract': TRANSFER_CONTRACT, 'transfer_id': bundle['transfer_id'], 'object': obj,
                'direction': row['direction'], 'state': 'complete',
                'owner': {'host': owner['host'], 'campaign_id': owner['campaign_id']}}
    finally:
        ledger.close()
