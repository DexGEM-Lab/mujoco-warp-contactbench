"""Explicit single-object authority transfer between two real campaigns.

Every ledger is real SQLite; inputs are real files hashed and copied. The
tests assert the conservation contract of a completed handoff: global
source-key set, per-pair ORIGINAL-source x5 targets, accepted UUID/source/
variant/shard set, full attempt/seed history and the exclusion audit are all
preserved, authority becomes unique, and the source can no longer claim the
transferred object while the destination continues its lineage.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import pytest

from sim.manorl.synthesis_admission import claim_object, ownership_record, register_existing
from sim.manorl.synthesis_campaign import (
    CAMPAIGN_CONTRACT, PrefixConfig, QuotaLedger, atomic_json,
)
from sim.manorl.synthesis_publication import reconcile_shards
from sim.manorl.synthesis_transfer import (
    TRANSFER_CONTRACT,
    commit_registry_owner, detach_object, export_object, finalize_transfer,
    import_object, verify_staging,
)

CP_BYTES = b'fake-checkpoint-bytes-cube1'
CP_BYTES2 = b'fake-checkpoint-bytes-cylinder6'
SIDECAR = {'checkpoint_file': 'model.pt', 'runtime_config': {'environment': {'pre_padding': 80}}}


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_campaign(tmp_path, name, *, objects, host, campaign_id, registry, output=None,
                   ccd=16):
    """Real campaign: config, pickles, checkpoint snapshot, ledger, registry."""
    camp = tmp_path / name
    output = Path(output) if output else tmp_path / (name + '-output')
    config = {'contract': CAMPAIGN_CONTRACT, 'prefix': asdict(PrefixConfig()),
              'output': str(output), 'ccd_contacts_per_world': ccd,
              'contacts_per_world': 1024, 'constraint_capacity': 4096,
              'initial_penetration_tolerance_m': 0.001, 'software_commit': 'a' * 40}
    atomic_json(camp / 'config.json', config)
    sources = []
    checkpoints = {}
    for obj, entries in objects.items():
        cp = camp / 'checkpoints' / obj / ('cp-' + obj)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(CP_BYTES if obj == 'cube1' else CP_BYTES2)
        atomic_json(Path(str(cp) + '.json'), {**SIDECAR, 'object': obj})
        sha = _sha(cp)
        checkpoints[sha] = {'path': str(cp), 'sha256': sha, 'metadata': {**SIDECAR, 'object': obj}}
        for key, group, pair in entries:
            pickle = camp / 'sources' / obj / f'{key}.pkl'
            pickle.parent.mkdir(parents=True, exist_ok=True)
            pickle.write_bytes(f'pickle-{key}'.encode())
            sources.append({'key': key, 'group_key': group, 'pair': pair, 'object': obj,
                            'identity': f'{pair.replace(":", "_")}_{key}',
                            'pickle': str(pickle), 'pickle_sha256': _sha(pickle),
                            'checkpoint': str(cp), 'checkpoint_sha256': sha})
    ledger = QuotaLedger(camp / 'quota.sqlite')
    ledger.initialize(config, sources)
    atomic_json(camp / 'sources.json', sources)
    atomic_json(camp / 'checkpoints.json', checkpoints)
    for obj in objects:
        assert claim_object(registry, obj=obj, host=host, campaign_id=campaign_id,
                            checkpoint_sha=next(s['checkpoint_sha256'] for s in sources
                                                if s['object'] == obj), output=str(output))
    return camp, config, output, ledger


def drive_source_history(ledger, shard_of):
    """Accepted / failed / unfinished attempts plus an active exclusion."""
    # cube1:01 strong donor 'c1s0': three durable rows first.
    for _ in range(3):
        claim = ledger.claim('w', 'cube1-gA', 1)[0]
        ledger.finish(claim, {'accepted': True}, row_uuid=f'u-{claim["source"]["key"]}-'
                            f'{claim["slot"]}-{claim["attempt"]}', shard=shard_of(claim))
    # One failed retry on the next variant, then one abandoned (unfinished).
    claim = ledger.claim('w', 'cube1-gA', 1)[0]
    ledger.finish(claim, {'accepted': False, 'reason': 'prefix_sampling_failure'})
    claim = ledger.claim('w2', 'cube1-gA', 1)[0]
    ledger.release_owner('w2')  # attempt row stays started-but-unfinished
    # Weak same-pair donor: one settled failure, zero successes.
    claim = ledger.claim('w', 'cube1-gB', 1)[0]
    ledger.finish(claim, {'accepted': False, 'reason': 'manipulation_gate'})
    # cylinder6 keeps local credit that must NOT move.
    claim = ledger.claim('w', 'cyl-g', 1)[0]
    ledger.finish(claim, {'accepted': True}, row_uuid=f'u-{claim["source"]["key"]}-'
                        f'{claim["slot"]}-{claim["attempt"]}', shard=shard_of(claim))
    # Exclusion under the stored 25%/1 policy retires the weak donor's rows.
    ledger.configure_success_filter(minimum_rate=.25, minimum_attempts=1)
    assert ledger.summary()['sources_excluded'] == 1
    # cube1:01 stays two rows short of its FIXED budget (8/10) so the handoff
    # carries genuine outstanding work; cube1:02 is filled to 5/5.
    while True:
        current = ledger.db.execute("SELECT COUNT(*) FROM outputs WHERE pair='cube1:01' "
                                    "AND state='accepted'").fetchone()[0]
        if current >= 8:
            break
        wave = ledger.claim('w', 'cube1-gA', 4)
        assert wave
        for claim in wave[:8 - current]:
            ledger.finish(claim, {'accepted': True},
                          row_uuid=f'u-{claim["source"]["key"]}-{claim["slot"]}-{claim["attempt"]}',
                          shard=shard_of(claim))
        if len(wave) > 8 - current:
            ledger.release_owner('w')  # untouched claims return to ready
            break
    while True:
        wave = ledger.claim('w', 'cube1-gC', 4)
        if not wave:
            break
        for claim in wave:
            ledger.finish(claim, {'accepted': True},
                          row_uuid=f'u-{claim["source"]["key"]}-{claim["slot"]}-{claim["attempt"]}',
                          shard=shard_of(claim))


def global_state(*ledgers):
    def one(ledger):
        db = ledger.db
        return {
            'sources': {r[0] for r in db.execute('SELECT key FROM sources')},
            'pair_targets': {r[0]: (r[1], r[2]) for r in db.execute(
                'SELECT pair,original_sources,target_rows FROM pair_targets')},
            'accepted': sorted((r[0], r[1], r[2], r[3], r[4]) for r in db.execute(
                'SELECT source_key,variant,row_uuid,shard,pair FROM outputs WHERE state=\'accepted\'')),
            'attempts': sorted((r[0], r[1], r[2], r[3], r[4], r[5]) for r in db.execute(
                'SELECT token,source_key,slot,attempt,finished,allocated_seed FROM attempts')),
            'exclusions': sorted((r[0], r[1], r[2], r[3]) for r in db.execute(
                'SELECT source_key,decided,settled_attempts,accepted_rows FROM source_exclusions')),
            'history': sorted((r[0], r[1], r[2]) for r in db.execute(
                'SELECT source_key,action,evidence FROM source_exclusion_history'))}
    merged = {}
    for ledger in ledgers:
        for table, value in one(ledger).items():
            if isinstance(value, set):
                merged.setdefault(table, set()).update(value)
            elif isinstance(value, dict):
                merged.setdefault(table, {}).update(value)
            else:
                merged.setdefault(table, []).extend(value)
    for key, value in merged.items():
        if isinstance(value, list):
            merged[key] = sorted(value)  # one global order, independent of host split
    return merged


@pytest.fixture
def two_campaigns(tmp_path):
    registry = tmp_path / 'registry'
    s1c, s1_config, s1_output, s1 = build_campaign(
        tmp_path, 'server1', host='server1', campaign_id='s1-camp', registry=registry,
        objects={'cube1': [('c1s0', 'cube1-gA', 'cube1:01'), ('c1s1', 'cube1-gB', 'cube1:01'),
                           ('c1p0', 'cube1-gC', 'cube1:02')],
                 'cylinder6': [('cy0', 'cyl-g', 'cylinder6:01')]})
    shard_of = lambda claim: str(s1_output / 'shards' / claim['source']['object']
                                 / f'{claim["token"]}.lance')
    drive_source_history(s1, shard_of)
    s2c, s2_config, s2_output, s2 = build_campaign(
        tmp_path, 'server2', host='server2', campaign_id='s2-camp', registry=registry,
        objects={'cuboid3': [('cb0', 'cuboid3-g', 'cuboid3:01')]})
    claim = s2.claim('w', 'cuboid3-g', 1)[0]
    s2.finish(claim, {'accepted': True}, row_uuid=f'u-{claim["source"]["key"]}-'
              f'{claim["slot"]}-{claim["attempt"]}', shard=str(s2_output / 'shards/cuboid3/x.lance'))
    return {'registry': registry, 's1': (s1c, s1_config, s1_output, s1),
            's2': (s2c, s2_config, s2_output, s2)}


def run_handoff(factories, *, transfer_id='xfer-test-1'):
    registry, (s1c, s1_config, s1_out, s1), (s2c, s2_config, s2_out, s2) = (factories['registry'],
        factories['s1'], factories['s2'])
    staging = s1c.parent / 'staging' / transfer_id
    export = export_object(s1c, obj='cube1', registry=registry, staging=staging, host='server1',
                           campaign_id='s1-camp', destination_host='server2',
                           destination_campaign='s2-camp', transfer_id=transfer_id)
    inbound = import_object(s2c, registry=registry, staging=staging, host='server2',
                            campaign_id='s2-camp')
    outbound = detach_object(s1c, staging=staging,
                             destination_receipt=s2c / 'transfer-receipts' / f'{transfer_id}.json')
    committed = commit_registry_owner(registry, staging=staging,
                                      source_receipt=s1c / 'transfer-receipts' / f'{transfer_id}.detach.json',
                                      destination_receipt=s2c / 'transfer-receipts' / f'{transfer_id}.json')
    finalize_transfer(s1c, registry=registry, staging=staging, host='server1', campaign_id='s1-camp')
    finalize_transfer(s2c, registry=registry, staging=staging, host='server2', campaign_id='s2-camp')
    return {'export': export, 'inbound': inbound, 'outbound': outbound,
            'committed': committed, 'staging': staging}


def test_completed_handoff_conserves_global_state_and_moves_authority(two_campaigns):
    s1c, s1_config, s1_out, s1 = two_campaigns['s1']
    s2c, s2_config, s2_out, s2 = two_campaigns['s2']
    registry = two_campaigns['registry']
    before = global_state(s1, s2)
    cube1_keys = {key for key in before['sources']
                  if key in {'c1s0', 'c1s1', 'c1p0'}}
    reports = run_handoff(two_campaigns)
    after = global_state(s1, s2)
    assert after == before  # global conservation: sources, targets, credit, history
    # Source no longer holds the object; destination holds everything of it.
    assert not {r[0] for r in s1.db.execute('SELECT key FROM sources')} & cube1_keys
    assert {r[0] for r in s2.db.execute('SELECT key FROM sources')} >= cube1_keys
    assert s1.summary()['pairs'] == 1 and s1.summary()['accepted'] == 1  # cylinder6 only
    s2_summary = s2.summary()
    assert s2_summary['sources'] == 4 and s2_summary['accepted'] == 14
    assert s2_summary['target'] == 20 and not s2_summary['complete']  # cube1:01 still owes 2
    assert {p['pair']: p['target'] for p in s2.pair_status()} == {
        'cuboid3:01': 5, 'cube1:01': 10, 'cube1:02': 5}
    # Authority is unique and points at the destination.
    owner = ownership_record(registry, 'cube1')
    assert (owner['host'], owner['campaign_id'], owner['transfer_id']) == (
        'server2', 's2-camp', reports['committed']['transfer_id'])
    original = json.loads((registry / 'objects' / 'cube1.json').read_text())
    assert original['host'] == 'server1'  # immutable original record preserved
    records = list((registry / 'objects' / 'cube1.transfers').glob('*.json'))
    assert len(records) == 1 and json.loads(records[0].read_text())['transfer_id'] == 'xfer-test-1'
    # Retired credit receipts are exact and complete on the source.
    retired = s1.db.execute('SELECT source_key,variant,row_uuid,shard FROM transferred_credit '
                            'ORDER BY source_key,variant').fetchall()
    exported_accepted = [row for row in reports['outbound']['retired_rows']]
    assert len(retired) == len(exported_accepted) == reports['export']['counts']['accepted_rows'] == 13
    assert s1.summary()['transfers_pending'] == [] and s2.summary()['transfers_pending'] == []
    assert s1.pending_transfer_objects() == set()


def test_destination_rebases_only_machine_local_paths(two_campaigns):
    s1c, s1_config, s1_out, s1 = two_campaigns['s1']
    s2c, s2_config, s2_out, s2 = two_campaigns['s2']
    before_specs = {r[0]: json.loads(r[1]) for r in s1.db.execute(
        "SELECT key,spec FROM sources WHERE pair LIKE 'cube1:%'")}
    run_handoff(two_campaigns)
    for key, spec in before_specs.items():
        row = s2.db.execute('SELECT spec FROM sources WHERE key=?', (key,)).fetchone()
        imported = json.loads(row[0])
        assert imported['pickle'].startswith(str(s2c)) and imported['checkpoint'].startswith(str(s2c))
        assert imported['pickle'] != spec['pickle'] and imported['checkpoint'] != spec['checkpoint']
        # Byte-identical copies: dataset/source identity and hashes unchanged.
        assert imported['pickle_sha256'] == spec['pickle_sha256'] == _sha(imported['pickle'])
        assert imported['checkpoint_sha256'] == spec['checkpoint_sha256'] == _sha(imported['checkpoint'])
        for field in ('identity', 'object', 'pair', 'group_key'):
            assert imported[field] == spec[field]
    manifests = json.loads((s2c / 'sources.json').read_text())
    assert {m['key'] for m in manifests} >= set(before_specs)
    checkpoints = json.loads((s2c / 'checkpoints.json').read_text())
    imported_cp = next(sha for sha, meta in checkpoints.items() if meta['path'].startswith(str(s2c / 'checkpoints' / 'cube1')))
    assert imported_cp == before_specs['c1s0']['checkpoint_sha256']


def test_destination_continues_lineage_without_duplication(two_campaigns):
    s2c, s2_config, s2_out, s2 = two_campaigns['s2']
    s1c = two_campaigns['s1'][0]
    run_handoff(two_campaigns)
    max_variant = s2.db.execute(
        "SELECT MAX(variant) FROM outputs WHERE source_key='c1s0'").fetchone()[0]
    attempts_before = s2.db.execute(
        "SELECT attempts FROM outputs WHERE source_key='c1s0' AND variant=?", (max_variant,)).fetchone()[0]
    imported_seeds = {r[0] for r in s2.db.execute(
        'SELECT allocated_seed FROM attempts WHERE allocated_seed IS NOT NULL')}
    legacy_null = s2.db.execute(
        "SELECT COUNT(*) FROM attempts WHERE allocated_seed IS NULL AND source_key IN "
        "('c1s0','c1s1','c1p0')").fetchone()[0]
    assert legacy_null > 0  # legacy seed namespace preserved as NULL rows
    # New claims continue the pair: same pair, unique tokens, growing variants.
    wave = s2.claim('w', 'cube1-gA', 2)
    assert wave and {c['pair'] for c in wave} == {'cube1:01'}
    assert {c['source']['key'] for c in wave} <= {'c1s0'}  # weak donor stays excluded
    for claim in wave:
        assert claim['slot'] > max_variant or claim['attempt'] > attempts_before
        if claim['slot'] >= 5:
            assert claim['seed'] >= 2 ** 31 and claim['seed'] not in imported_seeds
        ledger_seed = s2.db.execute('SELECT allocated_seed FROM attempts WHERE token=?',
                                    (claim['token'],)).fetchone()[0]
        assert ledger_seed == claim.get('seed') or claim['slot'] < 5
    tokens = [c['token'] for c in wave]
    assert len(set(tokens)) == len(tokens)
    for claim in wave:
        s2.finish(claim, {'accepted': True}, row_uuid=f'new-{claim["token"]}',
                  shard=str(s2_out / 'shards' / 'cube1' / 'new.lance'))
    pairs = {r[0] for r in s2.db.execute(
        "SELECT DISTINCT pair FROM outputs WHERE row_uuid LIKE 'new-%'")}
    assert pairs == {'cube1:01'}  # claims after handoff stay in the same pair
    # Imported accepted rows still point at their ORIGINAL shard paths.
    old = s2.db.execute("SELECT shard FROM outputs WHERE source_key='c1p0' AND state='accepted'").fetchall()
    assert all(r[0].startswith(str(s1c)) for r in old)


def test_claims_and_completion_blocked_until_finalized(two_campaigns):
    s1c, s1_config, s1_out, s1 = two_campaigns['s1']
    s2c, s2_config, s2_out, s2 = two_campaigns['s2']
    registry = two_campaigns['registry']
    staging = s1c.parent / 'staging' / 'xfer-mid'
    export_object(s1c, obj='cube1', registry=registry, staging=staging, host='server1',
                  campaign_id='s1-camp', destination_host='server2', destination_campaign='s2-camp',
                  transfer_id='xfer-mid')
    with pytest.raises(RuntimeError, match='in flight'):
        s1.claim('w', 'cube1-gA', 1)
    journal = json.loads((staging / 'journal.json').read_text())
    assert journal['complete'] is False
    assert s1.summary()['complete'] is False and len(s1.summary()['transfers_pending']) == 1
    import_object(s2c, registry=registry, staging=staging, host='server2', campaign_id='s2-camp')
    with pytest.raises(RuntimeError, match='in flight'):
        s2.claim('w', 'cube1-gA', 1)
    # Even a nominally finished destination cannot report complete.
    s2.db.execute("UPDATE outputs SET state='accepted' WHERE pair='cuboid3:01'")
    s2.db.commit()
    summary = s2.summary()
    assert summary['complete'] is False and summary['transfers_pending'][0]['object'] == 'cube1'
    detach_object(s1c, staging=staging,
                  destination_receipt=s2c / 'transfer-receipts' / 'xfer-mid.json')
    commit_registry_owner(registry, staging=staging,
                          source_receipt=s1c / 'transfer-receipts' / 'xfer-mid.detach.json',
                          destination_receipt=s2c / 'transfer-receipts' / 'xfer-mid.json')
    assert s1.summary()['complete'] is False  # source must finalize too
    finalize_transfer(s1c, registry=registry, staging=staging, host='server1', campaign_id='s1-camp')
    assert s1.summary()['transfers_pending'] == []
    with pytest.raises(RuntimeError, match='in flight'):
        s2.claim('w', 'cube1-gC', 1)  # destination transfer row is still pending
    finalize_transfer(s2c, registry=registry, staging=staging, host='server2', campaign_id='s2-camp')
    assert s2.claim('w', 'cube1-gA', 1) != []  # outstanding transferred rows claimable after finalize


def test_source_reconcile_skips_only_exact_retired_receipts(two_campaigns, tmp_path):
    s1c, s1_config, s1_out, s1 = two_campaigns['s1']
    accepted = s1.db.execute(
        "SELECT source_key,variant,row_uuid,shard FROM outputs WHERE state='accepted' "
        "AND pair LIKE 'cube1:%' ORDER BY source_key,variant").fetchall()
    for row in accepted:  # historical transferred-object shards stay on the S1 root
        commit = Path(row['shard']) / 'campaign-commit.json'
        commit.parent.mkdir(parents=True, exist_ok=True)
        commit.write_text(json.dumps({'contract': CAMPAIGN_CONTRACT, 'rows': [
            {'claim': {'source': {'key': row['source_key']}, 'slot': row['variant']},
             'row_uuid': row['row_uuid'], 'diagnostic': {'accepted': True}}], 'payload_sha256': {}}))
    run_handoff(two_campaigns)
    assert reconcile_shards(s1, s1_out) == 0  # exact receipts: skipped, nothing recovered
    # An unknown published source must still fail loudly.
    foreign = s1_out / 'shards' / 'cube1' / 'foreign.lance' / 'campaign-commit.json'
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_text(json.dumps({'contract': CAMPAIGN_CONTRACT, 'rows': [
        {'claim': {'source': {'key': 'never-existed'}, 'slot': 0}, 'row_uuid': 'x'}],
        'payload_sha256': {}}))
    with pytest.raises(RuntimeError, match='absent from ledger'):
        reconcile_shards(s1, s1_out)
    foreign.unlink()
    # A changed UUID for a retired row must still fail loudly.
    row = accepted[0]
    tampered = s1_out / 'shards' / 'cube1' / 'tampered.lance' / 'campaign-commit.json'
    tampered.parent.mkdir(parents=True, exist_ok=True)
    tampered.write_text(json.dumps({'contract': CAMPAIGN_CONTRACT, 'rows': [
        {'claim': {'source': {'key': row['source_key']}, 'slot': row['variant']},
         'row_uuid': 'different-uuid'}], 'payload_sha256': {}}))
    with pytest.raises(RuntimeError, match='absent from ledger'):
        reconcile_shards(s1, s1_out)
    tampered.unlink()
    # The same UUID published from a different shard path must still fail.
    elsewhere = s1_out / 'shards' / 'cube1' / 'elsewhere.lance' / 'campaign-commit.json'
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    elsewhere.write_text(json.dumps({'contract': CAMPAIGN_CONTRACT, 'rows': [
        {'claim': {'source': {'key': row['source_key']}, 'slot': row['variant']},
         'row_uuid': row['row_uuid']}], 'payload_sha256': {}}))
    with pytest.raises(RuntimeError, match='absent from ledger'):
        reconcile_shards(s1, s1_out)


def test_precondition_and_identity_refusals(two_campaigns, tmp_path):
    s1c, s1_config, s1_out, s1 = two_campaigns['s1']
    s2c, s2_config, s2_out, s2 = two_campaigns['s2']
    registry = two_campaigns['registry']
    staging = s1c.parent / 'staging' / 'xfer-refuse'
    # Outstanding lease anywhere in the campaign refuses the export.
    lease = s1.claim('w', 'cyl-g', 1)[0]
    with pytest.raises(RuntimeError, match='leased outputs remain'):
        export_object(s1c, obj='cube1', registry=registry, staging=staging, host='server1',
                      campaign_id='s1-camp', destination_host='server2',
                      destination_campaign='s2-camp', transfer_id='xfer-refuse')
    s1.release_owner('w')
    export_object(s1c, obj='cube1', registry=registry, staging=staging, host='server1',
                  campaign_id='s1-camp', destination_host='server2', destination_campaign='s2-camp',
                  transfer_id='xfer-refuse')
    # Wrong destination identity refuses the import.
    with pytest.raises(RuntimeError, match='declared destination'):
        import_object(s2c, registry=registry, staging=staging, host='server9', campaign_id='s2-camp')
    # Registry not naming the exporter refuses the import (ambiguous authority).
    with pytest.raises(RuntimeError, match='ambiguous authority'):
        import_object(s2c, registry=tmp_path / 'empty-registry', staging=staging,
                      host='server2', campaign_id='s2-camp')
    # A generation-contract mismatch refuses the import.
    mismatch, _, m_out, _ledger = build_campaign(
        tmp_path, 'server3', host='server2', campaign_id='s2-camp', registry=registry,
        objects={'cuboid3': [('cb0', 'cuboid3-g', 'cuboid3:01')]}, ccd=32)
    with pytest.raises(ValueError, match='generation contract differs'):
        import_object(mismatch, registry=registry, staging=staging, host='server2', campaign_id='s2-camp')
    inbound = import_object(s2c, registry=registry, staging=staging, host='server2', campaign_id='s2-camp')
    # A tampered destination receipt refuses the detach.
    receipt_path = s2c / 'transfer-receipts' / 'xfer-refuse.json'
    forged = json.loads(receipt_path.read_text())
    forged['counts']['accepted_rows'] += 1
    (s2c / 'transfer-receipts' / 'forged.json').write_text(json.dumps(forged))
    with pytest.raises(RuntimeError, match='does not match the declared transfer endpoints'):
        detach_object(s1c, staging=staging, destination_receipt=s2c / 'transfer-receipts' / 'forged.json')
    forged['transfer_id'] = 'xfer-not-this-one'
    (s2c / 'transfer-receipts' / 'forged2.json').write_text(json.dumps(forged))
    with pytest.raises(RuntimeError, match='does not verify'):
        detach_object(s1c, staging=staging, destination_receipt=s2c / 'transfer-receipts' / 'forged2.json')
    # Source ledger changed after export: fingerprints differ, detach refuses.
    # (A rogue writer bypassing the claim guard simulated by direct SQL.)
    s1.db.execute("UPDATE outputs SET error='drift' WHERE source_key='c1p0'")
    s1.db.commit()
    with pytest.raises(RuntimeError, match='changed after export'):
        detach_object(s1c, staging=staging, destination_receipt=receipt_path)
    # Duplicate handoff of the same object refuses while one is pending.
    with pytest.raises(RuntimeError, match='already has transfer'):
        export_object(s1c, obj='cube1', registry=registry, staging=s1c.parent / 'staging' / 'other',
                      host='server1', campaign_id='s1-camp', destination_host='server2',
                      destination_campaign='s2-camp', transfer_id='xfer-other')
    # A staging file changed under the bundle refuses verification.
    target = staging / 'bundle.json'
    target.chmod(0o644)
    original = target.read_bytes()
    target.write_bytes(original + b' ')
    with pytest.raises(RuntimeError, match='hash mismatch'):
        verify_staging(staging)
    target.write_bytes(original)


def test_interrupted_steps_replay_idempotently(two_campaigns):
    s1c, s1_config, s1_out, s1 = two_campaigns['s1']
    s2c, s2_config, s2_out, s2 = two_campaigns['s2']
    registry = two_campaigns['registry']
    staging = s1c.parent / 'staging' / 'xfer-replay'
    export = export_object(s1c, obj='cube1', registry=registry, staging=staging, host='server1',
                           campaign_id='s1-camp', destination_host='server2',
                           destination_campaign='s2-camp', transfer_id='xfer-replay')
    replayed = export_object(s1c, obj='cube1', registry=registry, staging=staging, host='server1',
                             campaign_id='s1-camp', destination_host='server2',
                             destination_campaign='s2-camp', transfer_id='xfer-replay')
    assert replayed == export
    first = import_object(s2c, registry=registry, staging=staging, host='server2', campaign_id='s2-camp')
    second = import_object(s2c, registry=registry, staging=staging, host='server2', campaign_id='s2-camp')
    assert first == second
    assert s2.db.execute('SELECT COUNT(*) FROM sources').fetchone()[0] == 4  # no duplication
    receipt = s2c / 'transfer-receipts' / 'xfer-replay.json'
    detach_one = detach_object(s1c, staging=staging, destination_receipt=receipt)
    detach_two = detach_object(s1c, staging=staging, destination_receipt=receipt)
    assert detach_two == detach_one
    committed = commit_registry_owner(registry, staging=staging,
                                      source_receipt=s1c / 'transfer-receipts' / 'xfer-replay.detach.json',
                                      destination_receipt=receipt)
    assert commit_registry_owner(registry, staging=staging,
                                 source_receipt=s1c / 'transfer-receipts' / 'xfer-replay.detach.json',
                                 destination_receipt=receipt)['replay'] is True
    for camp, host, cid in ((s1c, 'server1', 's1-camp'), (s2c, 'server2', 's2-camp')):
        one = finalize_transfer(camp, registry=registry, staging=staging, host=host, campaign_id=cid)
        two = finalize_transfer(camp, registry=registry, staging=staging, host=host, campaign_id=cid)
        assert one == two and one['state'] == 'complete'


def test_finalize_refuses_before_registry_commit(two_campaigns):
    s1c, s1_config, s1_out, s1 = two_campaigns['s1']
    s2c, s2_config, s2_out, s2 = two_campaigns['s2']
    registry = two_campaigns['registry']
    staging = s1c.parent / 'staging' / 'xfer-order'
    export_object(s1c, obj='cube1', registry=registry, staging=staging, host='server1',
                  campaign_id='s1-camp', destination_host='server2', destination_campaign='s2-camp',
                  transfer_id='xfer-order')
    import_object(s2c, registry=registry, staging=staging, host='server2', campaign_id='s2-camp')
    with pytest.raises(RuntimeError, match='still exported'):
        finalize_transfer(s1c, registry=registry, staging=staging, host='server1', campaign_id='s1-camp')
    detach_object(s1c, staging=staging,
                  destination_receipt=s2c / 'transfer-receipts' / 'xfer-order.json')
    with pytest.raises(RuntimeError, match='not been committed'):
        finalize_transfer(s1c, registry=registry, staging=staging, host='server1', campaign_id='s1-camp')
    with pytest.raises(RuntimeError, match='does not name this campaign'):
        finalize_transfer(s2c, registry=registry, staging=staging, host='server2', campaign_id='s2-camp')


def test_admission_semantics_after_transfer(two_campaigns):
    s1c, s1_config, s1_out, s1 = two_campaigns['s1']
    s2c, s2_config, s2_out, s2 = two_campaigns['s2']
    registry = two_campaigns['registry']
    staging = s1c.parent / 'staging' / 'xfer-adm'
    export_object(s1c, obj='cube1', registry=registry, staging=staging, host='server1',
                  campaign_id='s1-camp', destination_host='server2', destination_campaign='s2-camp',
                  transfer_id='xfer-adm')
    # Mid-handoff the source's admission pass skips (never re-asserts) the object.
    register_existing(s1, registry, host='server1', campaign_id='s1-camp', output=str(s1_out))
    assert ownership_record(registry, 'cube1')['host'] == 'server1'
    import_object(s2c, registry=registry, staging=staging, host='server2', campaign_id='s2-camp')
    # Destination holds foreign-owned rows mid-handoff: admission must not reclaim.
    register_existing(s2, registry, host='server2', campaign_id='s2-camp', output=str(s2_out))
    assert ownership_record(registry, 'cube1')['host'] == 'server1'
    detach_object(s1c, staging=staging, destination_receipt=s2c / 'transfer-receipts' / 'xfer-adm.json')
    commit_registry_owner(registry, staging=staging,
                          source_receipt=s1c / 'transfer-receipts' / 'xfer-adm.detach.json',
                          destination_receipt=s2c / 'transfer-receipts' / 'xfer-adm.json')
    finalize_transfer(s1c, registry=registry, staging=staging, host='server1', campaign_id='s1-camp')
    finalize_transfer(s2c, registry=registry, staging=staging, host='server2', campaign_id='s2-camp')
    # After the handoff, the source cannot reclaim the object it no longer has,
    # and the destination registers as the rightful owner.
    register_existing(s1, registry, host='server1', campaign_id='s1-camp', output=str(s1_out))
    register_existing(s2, registry, host='server2', campaign_id='s2-camp', output=str(s2_out))
    assert ownership_record(registry, 'cube1')['host'] == 'server2'
    # A foreign host cannot claim the transferred object even with the same checkpoint.
    with pytest.raises(RuntimeError, match='owned by another host'):
        register_existing(_ledger_with(s2c), registry, host='server9', campaign_id='x', output=str(s2_out))


def _ledger_with(campaign):
    ledger = QuotaLedger(Path(campaign) / 'quota.sqlite')
    return ledger


def test_cli_wiring_runs_the_full_handoff(two_campaigns, capsys, tmp_path):
    import tools.run_manorl_synthesis_campaign as cli
    s1c, s1_config, s1_out, s1 = two_campaigns['s1']
    s2c, s2_config, s2_out, s2 = two_campaigns['s2']
    registry = two_campaigns['registry']
    staging = s1c.parent / 'cli-staging' / 'xfer-cli'
    cli.main(['transfer-export', '--campaign', str(s1c), '--object', 'cube1',
              '--registry', str(registry), '--staging', str(staging),
              '--host', 'server1', '--campaign-id', 's1-camp',
              '--destination-host', 'server2', '--destination-campaign', 's2-camp',
              '--transfer-id', 'xfer-cli'])
    cli.main(['transfer-import', '--campaign', str(s2c), '--registry', str(registry),
              '--staging', str(staging), '--host', 'server2', '--campaign-id', 's2-camp'])
    cli.main(['transfer-detach', '--campaign', str(s1c), '--registry', str(registry),
              '--staging', str(staging),
              '--destination-receipt', str(s2c / 'transfer-receipts' / 'xfer-cli.json')])
    cli.main(['transfer-commit-owner', '--registry', str(registry), '--staging', str(staging),
              '--source-receipt', str(s1c / 'transfer-receipts' / 'xfer-cli.detach.json'),
              '--destination-receipt', str(s2c / 'transfer-receipts' / 'xfer-cli.json')])
    cli.main(['transfer-finalize', '--campaign', str(s1c), '--registry', str(registry),
              '--staging', str(staging), '--host', 'server1', '--campaign-id', 's1-camp'])
    cli.main(['transfer-finalize', '--campaign', str(s2c), '--registry', str(registry),
              '--staging', str(staging), '--host', 'server2', '--campaign-id', 's2-camp'])
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e['event'] for e in events] == ['object_authority_exported',
                                            'object_authority_imported',
                                            'object_authority_detached',
                                            'object_authority_committed',
                                            'object_authority_finalized',
                                            'object_authority_finalized']
    assert ownership_record(registry, 'cube1')['host'] == 'server2'
