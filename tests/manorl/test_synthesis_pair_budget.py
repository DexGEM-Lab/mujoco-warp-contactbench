"""Fixed per-pair output budgets: donor reassignment, migration and lineage.

Every test uses real SQLite ledgers (and real Lance where publication is
involved), not mocks of the ledger itself.
"""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from sim.manorl.lance_v2 import generated_rollout_uuid
from sim.manorl.synthesis_campaign import (
    PAIR_BUDGET_CONTRACT, CAMPAIGN_CONTRACT, PrefixConfig, QuotaLedger,
    attempt_seed, augment_source_prefix, canonical_hash, source_key,
)
from sim.manorl.synthesis_migration import (
    MIGRATION_CONTRACT, LegacyQuotaLedger, migrate_ledger,
)


def source(key, group, pair):
    return {'key': key, 'group_key': group, 'pair': pair, 'identity': f'{pair.replace(":", "_")}_{key}'}


def drive(ledger, owner, group, plan, *, claim_count=64, rounds=4000):
    """Settle attempts per source until each source's plan is exhausted.

    Claims of sources outside the plan are released untouched, so they keep no
    settled attempt and their eligibility is unaffected.
    """
    remaining = {key: [accepts, fails] for key, (accepts, fails) in plan.items()}
    for _ in range(rounds):
        wave = ledger.claim(owner, group, claim_count)
        if not wave:
            break
        settled_any = False
        skipped = False
        for claim in wave:
            key = claim['source']['key']
            if key not in remaining:
                skipped = True  # foreign claim: released untouched below
                continue
            if remaining[key][0] > 0:
                remaining[key][0] -= 1
                ledger.finish(claim, {'accepted': True},
                              row_uuid=f'u-{key}-{claim["slot"]}-{claim["attempt"]}', shard='verified.lance')
                settled_any = True
            elif remaining[key][1] > 0:
                remaining[key][1] -= 1
                ledger.finish(claim, {'accepted': False})
                settled_any = True
            else:
                skipped = True  # budget exhausted: never settle a stray extra attempt
        done = all(a == 0 and f == 0 for a, f in remaining.values())
        if skipped:
            ledger.release_owner(owner)  # return untouched claims to ready
        if done or not settled_any:
            break
    else:
        raise AssertionError('drive did not converge')
    return {k: (plan[k][0] - v[0], plan[k][1] - v[1]) for k, v in remaining.items()}


def fill_pair(ledger, owner, group, *, claim_count=64, rounds=500):
    per_source = Counter()
    for _ in range(rounds):
        wave = ledger.claim(owner, group, claim_count)
        if not wave:
            return per_source
        for claim in wave:
            ledger.finish(claim, {'accepted': True},
                          row_uuid=f'u-{claim["source"]["key"]}-{claim["slot"]}-{claim["attempt"]}',
                          shard='verified.lance')
            per_source[claim['source']['key']] += 1
    raise AssertionError('pair never completed')


def test_excluded_sources_keep_pair_budget_and_survivors_supply_it(tmp_path):
    ledger = QuotaLedger(tmp_path / 'quota.sqlite')
    ledger.initialize({}, [source(f's{i:03d}', 'cube1-g', 'cube1:01') for i in range(100)])
    ledger.configure_success_filter(minimum_attempts=1)
    weak = ledger.claim('w', 'cube1-g', 50)
    assert len(weak) == 50
    for claim in weak:  # zero successes in one settled attempt each
        ledger.finish(claim, {'accepted': False})
    assert {c['source']['key'] for c in weak} == {f's{i:03d}' for i in range(50)}
    summary = ledger.summary()
    assert summary['target'] == 500 and summary['sources_excluded'] == 50
    assert summary['skipped_slots'] == 250 and summary['remaining'] == 500
    try:
        per_source = fill_pair(ledger, 'w', 'cube1-g')
        assert ledger.summary()['accepted'] == 500
        assert ledger.summary()['complete'] and ledger.summary()['pairs_met'] == 1
        assert set(per_source) == {f's{i:03d}' for i in range(50, 100)}
        assert max(per_source.values()) > 5  # donors exceed the old five-row ceiling
        assert min(per_source.values()) >= 5
        rows = ledger.db.execute(
            "SELECT pair,source_key,COUNT(*) FROM outputs WHERE state='accepted' GROUP BY pair,source_key").fetchall()
        assert {r['pair'] for r in rows} == {'cube1:01'}
    finally:
        ledger.close()


def test_mixed_action_group_never_borrows_across_pairs(tmp_path):
    ledger = QuotaLedger(tmp_path / 'quota.sqlite')
    sources = ([source(f'a{i}', 'cube1-g', 'cube1:01') for i in range(2)]
               + [source(f'b{i}', 'cube1-g', 'cube1:02') for i in range(2)])
    ledger.initialize({}, sources)
    ledger.configure_success_filter(minimum_attempts=1)
    drive(ledger, 'w', 'cube1-g', {f'a{i}': (0, 1) for i in range(2)})  # both weak pair donors excluded
    per_source = fill_pair(ledger, 'w', 'cube1-g')
    assert per_source == Counter({f'b{i}': 5 for i in range(2)})
    summary = ledger.summary()
    assert summary['target'] == 20 and summary['accepted'] == 10
    assert summary['blocked_pairs'] == ['cube1:01']
    assert summary['pairs_blocked'] == 1 and summary['pairs_met'] == 1 and not summary['complete']
    assert summary['remaining'] == 10 and summary['pair_debt'] == 10
    assert summary['blocked_pair_debt'] == 10 and summary['runnable_pair_debt'] == 0
    assert [p['pair'] for p in ledger.pair_status() if p['blocked']] == ['cube1:01']
    assert ledger.claim('w', 'cube1-g', 4) == []  # no runnable work, no cross-pair growth
    pairs = {r['pair'] for r in ledger.db.execute(
        "SELECT pair FROM outputs WHERE state='accepted'")}
    assert pairs == {'cube1:02'}
    borrowed = ledger.db.execute(
        "SELECT COUNT(*) FROM outputs WHERE pair='cube1:01' AND source_key IN ('b0','b1')").fetchone()[0]
    assert borrowed == 0  # donors are never drawn from another pair
    ledger.close()


def test_pair_without_eligible_donors_is_blocked_not_complete(tmp_path):
    ledger = QuotaLedger(tmp_path / 'quota.sqlite')
    ledger.initialize({}, [source('only', 'cube1-g', 'cube1:01')])
    ledger.configure_success_filter(minimum_attempts=1)
    drive(ledger, 'w', 'cube1-g', {'only': (0, 1)})
    summary = ledger.summary()
    assert summary['target'] == 5 and summary['accepted'] == 0
    assert summary['blocked_pairs'] == ['cube1:01'] and not summary['complete']
    assert ledger.claim('w', 'cube1-g', 5) == [] and ledger.groups() == []
    ledger.close()


def test_exactly_quarter_is_eligible_and_twenty_settled_floor(tmp_path):
    ledger = QuotaLedger(tmp_path / 'quota.sqlite')
    # 'weak' is excluded first so the pair deficit turns 'a' into a donor with
    # extra variants; separate groups keep the attempt histories independent.
    ledger.initialize({}, [source('a', 'g1', 'cube1:01'), source('weak', 'g2', 'cube1:01')])
    ledger.configure_success_filter()  # default .25 / 20 finished attempts
    drive(ledger, 'w', 'g2', {'weak': (0, 19)})  # 19 settled: below the attempt floor
    assert ledger.summary()['sources_excluded'] == 0
    drive(ledger, 'w', 'g2', {'weak': (0, 1)})  # 0/20: excluded at the floor
    assert ledger.summary()['sources_excluded'] == 1
    drive(ledger, 'w', 'g1', {'a': (5, 15)})  # 20th settlement: 5/20 == .25 exactly
    summary = ledger.summary()
    assert summary['sources_excluded'] == 1  # exactly .25 stays eligible
    assert summary['accepted'] == 5 and summary['target'] == 10
    drive(ledger, 'w', 'g1', {'a': (0, 1)})  # 5/21 strictly below
    summary = ledger.summary()
    assert summary['sources_excluded'] == 2
    assert summary['accepted'] == 5 and summary['excluded_accepted'] == 5
    assert summary['remaining'] == 5 and summary['blocked_pairs'] == ['cube1:01']  # target never shrank
    assert summary['blocked_pair_debt'] == 5 and summary['runnable_pair_debt'] == 0
    ledger.close()


def test_previously_full_donor_is_still_evaluated_and_credit_kept(tmp_path):
    ledger = QuotaLedger(tmp_path / 'quota.sqlite')
    # Separate groups keep the attempt histories independent; the PAIR is shared.
    ledger.initialize({}, [source('full', 'g1', 'cube1:01'),
                           source('weak', 'g2', 'cube1:01'),
                           source('healthy', 'g3', 'cube1:01')])
    ledger.configure_success_filter()
    drive(ledger, 'w', 'g2', {'weak': (0, 20)})  # excluded first: creates pair deficit
    # 'full' reaches five durable rows, then drops below .25 (5/21): the old
    # accepted>=5 exemption must no longer protect it.
    drive(ledger, 'w', 'g1', {'full': (5, 16)})
    assert ledger.summary()['sources_excluded'] == 2
    per_source = fill_pair(ledger, 'w', 'g3')
    summary = ledger.summary()
    assert summary['target'] == 15 and summary['accepted'] == 15 and summary['complete']
    assert summary['excluded_accepted'] == 5  # the excluded full source keeps its credit
    assert per_source == Counter({'healthy': 10})
    variants = [r['variant'] for r in ledger.db.execute(
        "SELECT variant FROM outputs WHERE source_key='healthy' AND state='accepted' ORDER BY variant")]
    assert variants == list(range(10))  # extra variants 5..9 supplied the debt
    ledger.close()


def test_extra_variant_seeds_uuids_and_strata(tmp_path):
    from test_synthesis_campaign import trajectory
    src = trajectory()
    low = np.full(28, -10.)
    high = -low
    config = PrefixConfig()
    key = source_key(src)
    # Slots 0-4 keep the exact historical derivation.
    for slot in range(5):
        for attempt in (1, 7):
            expected = int(canonical_hash([config.seed, key, slot, attempt, asdict(config)])[:8], 16) % (2**31 - 1)
            assert attempt_seed(key, slot, attempt, config) == expected
    legacy_seeds = {attempt_seed(key, s, a, config) for s in range(5) for a in range(1, 60)}
    seeds, uuids = {}, {}
    for variant in range(15):
        for attempt in (1, 2):
            augmented, sample = augment_source_prefix(src, slot=variant, attempt=attempt,
                                                       joint_lower=low, joint_upper=high)
            replay, replay_sample = augment_source_prefix(src, slot=variant, attempt=attempt,
                                                          joint_lower=low, joint_upper=high)
            assert sample == replay_sample and np.array_equal(augmented.q_ref, replay.q_ref)
            assert sample['seed'] not in seeds
            seeds[sample['seed']] = (variant, attempt)
            radius = float(np.linalg.norm(augmented.q_ref[0, :3] - src.q_ref[0, :3]))
            assert 0.05 <= radius <= 0.50  # widened counter never leaves the range
            stratum = variant % 5
            assert sample['radius_stratum_m'] == pytest.approx(
                [0.05 + stratum * 0.09, 0.05 + (stratum + 1) * 0.09], abs=1e-12)
            if variant >= 5:
                assert sample['seed'] not in legacy_seeds
            row_uuid = generated_rollout_uuid(
                src.identity.uuid, 'a' * 64, episode=variant,
                augmentation_identity=CAMPAIGN_CONTRACT + ':' + canonical_hash(sample))
            assert row_uuid not in uuids
            uuids[row_uuid] = (variant, attempt)
    base = augment_source_prefix(src, slot=2, attempt=3, joint_lower=low, joint_upper=high)[1]
    cycled = augment_source_prefix(src, slot=7, attempt=3, joint_lower=low, joint_upper=high)[1]
    assert base['radius_stratum_m'] == cycled['radius_stratum_m']
    assert base['azimuth_stratum'] == cycled['azimuth_stratum']
    assert base['seed'] != cycled['seed']
    with pytest.raises(ValueError):
        attempt_seed(key, -1, 1, config)
    with pytest.raises(ValueError):
        attempt_seed(key, 0, 0, config)


def test_concurrent_connections_claim_disjoint_and_never_overfill(tmp_path):
    path = tmp_path / 'quota.sqlite'
    QuotaLedger(path).initialize(
        {}, [source(f's{i}', 'g', 'cube1:01') for i in range(4)] + [source('x', 'g', 'cube1:02')])
    claimed, lock = [], threading.Lock()
    errors = []

    def worker(name):
        ledger = QuotaLedger(path)
        try:
            for _ in range(200):
                wave = ledger.claim(name, 'g', 3)
                if not wave:
                    break
                with lock:
                    claimed.extend(wave)
                for claim in wave:
                    ledger.finish(claim, {'accepted': True},
                                  row_uuid=f'u-{name}-{claim["source"]["key"]}-{claim["slot"]}',
                                  shard='verified.lance')
        except BaseException as exc:  # pragma: no cover - surfaced by assertion
            errors.append(exc)
        finally:
            ledger.close()

    threads = [threading.Thread(target=worker, args=(f'w{i}',)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    tokens = [c['token'] for c in claimed]
    identities = [(c['source']['key'], c['slot']) for c in claimed]
    assert len(set(tokens)) == len(tokens)
    assert len(set(identities)) == len(identities)  # atomic disjoint claims
    ledger = QuotaLedger(path)
    summary = ledger.summary()
    assert summary['target'] == 25 and summary['accepted'] == 25 and summary['complete']
    per_pair = dict(ledger.db.execute(
        "SELECT pair,SUM(state='accepted') FROM outputs GROUP BY pair").fetchall())
    assert per_pair == {'cube1:01': 20, 'cube1:02': 5}  # no pair overfilled
    ledger.close()


def build_legacy_campaign(tmp_path, *, with_lease=False):
    camp = tmp_path / 'legacy'
    camp.mkdir(parents=True, exist_ok=True)
    ledger = LegacyQuotaLedger(camp / 'quota.sqlite')
    ledger.initialize({}, [source('a', 'g', 'cube1:01'), source('b', 'g', 'cube1:01'),
                           source('c', 'g', 'cube1:02')])
    lease = None
    if with_lease:
        lease = ledger.claim('ghost', 'g', 1)[0]
    # partial pair credit on 'a' plus a finished retry history
    drive(ledger, 'w', 'g', {'a': (1, 8), 'b': (3, 0)})  # 'c' stays untouched
    ledger.configure_success_filter(minimum_attempts=4)  # 'a' at 1/9 -> legacy skip
    (camp / 'config.json').write_text(json.dumps({'output': str(tmp_path / 'out')}))
    return ledger, lease


def test_migration_preserves_credit_history_and_is_idempotent(tmp_path):
    ledger, _ = build_legacy_campaign(tmp_path)
    legacy_summary = ledger.summary()
    ledger.close()
    assert legacy_summary['target'] < legacy_summary['original_target']  # shrinking-target semantics
    report = migrate_ledger(tmp_path / 'legacy')
    assert report['state'] == 'migrated' and report['contract'] == MIGRATION_CONTRACT
    assert report['pair_targets'] == {'cube1:01': 10, 'cube1:02': 5}
    ledger = QuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')
    try:
        summary = ledger.summary()
        assert summary['contract'] == PAIR_BUDGET_CONTRACT
        assert summary['target'] == 15  # fixed again: exclusions cannot shrink it
        assert summary['accepted'] == 4 and summary['excluded_accepted'] == 1
        assert summary['blocked_pair_debt'] == 0 and summary['runnable_pair_debt'] == 11  # donors remain
        assert summary['sources_excluded'] == 1
        assert summary['attempts'] == legacy_summary['attempts']
        assert summary['source_success_policy']['minimum_rate'] == .25
        identities = [tuple(r) for r in ledger.db.execute(
            'SELECT source_key,variant,row_uuid,shard FROM outputs WHERE state=\'accepted\' ORDER BY source_key,variant')]
        assert report['before']['accepted_identities'] == identities
        # debt is reassignable to same-pair donors now
        per_source = fill_pair(ledger, 'w', 'g')
        assert ledger.summary()['accepted'] == 15 and ledger.summary()['complete']
        assert per_source['b'] == 6 and per_source['c'] == 5  # 'b' repays the retired debt
    finally:
        ledger.close()
    again = migrate_ledger(tmp_path / 'legacy')
    assert again['state'] == 'already_migrated'
    ledger = QuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')
    try:
        assert ledger.summary()['accepted'] == 15  # idempotent: nothing duplicated or lost
    finally:
        ledger.close()


def test_migration_refuses_outstanding_leases_without_touching_data(tmp_path):
    ledger, lease = build_legacy_campaign(tmp_path, with_lease=True)
    before = ledger.summary()
    ledger.close()
    with pytest.raises(RuntimeError, match='leased'):
        migrate_ledger(tmp_path / 'legacy')
    reopened = LegacyQuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')
    try:
        assert reopened.summary()['states'] == before['states']
        assert reopened.db.execute("SELECT state,token FROM slots WHERE source_key=? AND slot=?",
                                   (lease['source']['key'], lease['slot'])).fetchone()['token'] == lease['token']
    finally:
        reopened.close()
    with pytest.raises(Exception, match='legacy per-source-slot schema'):
        QuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')


def test_migration_reconciles_historical_manifests_before_converting(tmp_path):
    from sim.manorl.lance_v2 import file_sha256
    ledger, _ = build_legacy_campaign(tmp_path)
    claim = ledger.claim('dead-publisher', 'g', 1)[0]
    ledger.close()
    shard = tmp_path / 'out' / 'shards' / 'cube1' / 'wave.lance'
    shard.mkdir(parents=True)
    (shard / 'data').write_bytes(b'verified payload')
    (shard / 'campaign-commit.json').write_text(json.dumps({
        'contract': CAMPAIGN_CONTRACT,
        'rows': [{'claim': claim, 'diagnostic': {'accepted': True}, 'row_uuid': 'promoted-row'}],
        'payload_sha256': {'data': file_sha256(shard / 'data')}}))
    report = migrate_ledger(tmp_path / 'legacy')
    assert report['state'] == 'migrated' and report['recovered_promoted_rows'] == 1
    ledger = QuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')
    try:
        row = ledger.db.execute(
            "SELECT state,row_uuid,shard FROM outputs WHERE source_key=? AND variant=?",
            (claim['source']['key'], claim['slot'])).fetchone()
        assert tuple(row) == ('accepted', 'promoted-row', str(shard))
        assert ledger.summary()['accepted'] == 5  # recovered row keeps its pair credit
    finally:
        ledger.close()


def test_migration_rejects_tampered_historical_payload(tmp_path):
    from sim.manorl.lance_v2 import file_sha256
    ledger, _ = build_legacy_campaign(tmp_path)
    claim = ledger.claim('dead-publisher', 'g', 1)[0]
    ledger.close()
    shard = tmp_path / 'out' / 'shards' / 'cube1' / 'wave.lance'
    shard.mkdir(parents=True)
    (shard / 'data').write_bytes(b'payload')
    (shard / 'campaign-commit.json').write_text(json.dumps({
        'contract': CAMPAIGN_CONTRACT,
        'rows': [{'claim': claim, 'diagnostic': {'accepted': True}, 'row_uuid': 'promoted-row'}],
        'payload_sha256': {'data': file_sha256(shard / 'data')}}))
    (shard / 'data').write_bytes(b'tampered')
    with pytest.raises(RuntimeError, match='hash mismatch'):
        migrate_ledger(tmp_path / 'legacy')
    reopened = LegacyQuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')
    try:
        assert reopened.db.execute("SELECT COUNT(*) FROM slots WHERE state='accepted' AND row_uuid='promoted-row'").fetchone()[0] == 0
    finally:
        reopened.close()


def test_migrated_ledger_replays_publication_idempotently(tmp_path):
    from sim.manorl.synthesis_publication import reconcile_shards
    ledger, _ = build_legacy_campaign(tmp_path)
    claim = ledger.claim('dead-publisher', 'g', 1)[0]
    ledger.close()
    shard = tmp_path / 'out' / 'shards' / 'cube1' / 'wave.lance'
    shard.mkdir(parents=True)
    (shard / 'data').write_bytes(b'payload')
    from sim.manorl.lance_v2 import file_sha256
    (shard / 'campaign-commit.json').write_text(json.dumps({
        'contract': CAMPAIGN_CONTRACT,
        'rows': [{'claim': claim, 'diagnostic': {'accepted': True}, 'row_uuid': 'promoted-row'}],
        'payload_sha256': {'data': file_sha256(shard / 'data')}}))
    migrate_ledger(tmp_path / 'legacy')
    ledger = QuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')
    try:
        assert reconcile_shards(ledger, tmp_path / 'out') == 0  # restart recovery is a no-op replay
        assert ledger.summary()['accepted'] == 5
    finally:
        ledger.close()


def test_admission_and_watch_use_fixed_target_completion(tmp_path):
    from sim.manorl.synthesis_admission import append_prepared
    from tools.watch_manorl_synthesis_campaign import should_continue
    from tools.run_manorl_synthesis_campaign import select_worker_group
    root = tmp_path / 'campaign'
    root.mkdir()
    ledger = QuotaLedger(root / 'quota.sqlite')
    config = {'contract': 'test', 'prefix': {'quota': 5}}
    old = {'key': 'old', 'group_key': 'cube1-x', 'pair': 'cube1:01', 'object': 'cube1', 'checkpoint_sha256': 'a' * 64}
    ledger.initialize(config, [old])
    leased = ledger.claim('live', 'cube1-x', 2)
    ledger.finish(leased[0], {'accepted': True}, row_uuid='u-live-0', shard='verified.lance')
    fill_pair(ledger, 'live', 'cube1-x')
    assert ledger.summary()['accepted'] == 4
    assert not ledger.summary()['complete']  # the outstanding lease blocks completion
    ledger.finish(leased[1], {'accepted': True}, row_uuid='u-live-1', shard='verified.lance')
    assert ledger.summary()['complete'] and not should_continue({'complete': True}, paused=False)
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    blob = prepared / 'source.pkl'
    blob.write_bytes(b'example')
    from sim.manorl.synthesis_admission import sha256
    new = {'key': 'new', 'group_key': 'cuboid3-y', 'pair': 'cuboid3:01', 'object': 'cuboid3',
           'pickle': str(blob), 'pickle_sha256': sha256(blob), 'checkpoint_sha256': 'b' * 64}
    from sim.manorl.synthesis_campaign import atomic_json
    atomic_json(prepared / 'config.json', config)
    atomic_json(prepared / 'sources.json', [new])
    atomic_json(prepared / 'checkpoints.json', {})
    assert append_prepared(ledger, root, prepared, config) == 1
    summary = ledger.summary()
    assert summary['target'] == 10 and summary['pairs'] == 2 and not summary['complete']
    assert should_continue({'complete': summary['complete']}, paused=False)
    groups = ledger.groups()
    assert [g['group_key'] for g in groups] == ['cuboid3-y']
    assert select_worker_group(groups, {}, {}, {}, 1) == 'cuboid3-y'
    fill_pair(ledger, 'w', 'cuboid3-y')
    assert ledger.summary()['complete'] and ledger.summary()['pairs_met'] == 2
    ledger.close()


def test_append_admission_after_migration_preserves_column_semantics(tmp_path):
    """A migrated legacy table appends `pair` after `spec`; appended sources
    must land in the right columns regardless of column order."""
    ledger, _ = build_legacy_campaign(tmp_path)
    ledger.close()
    report = migrate_ledger(tmp_path / 'legacy')
    assert report['state'] == 'migrated'
    ledger = QuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')
    try:
        appended = source('appended', 'cube9-z', 'cube9:01')
        appended['identity'] = 'cube9_01_appended'
        ledger.initialize({}, [appended])
        row = ledger.db.execute(
            "SELECT group_key,pair,spec FROM sources WHERE key='appended'").fetchone()
        assert row['pair'] == 'cube9:01' and row['group_key'] == 'cube9-z'
        assert json.loads(row['spec'])['pair'] == 'cube9:01'  # spec/pair not swapped
        claim = ledger.claim('w', 'cube9-z', 1)[0]
        assert claim['source']['key'] == 'appended' and claim['pair'] == 'cube9:01'
        ledger.finish(claim, {'accepted': True}, row_uuid='appended-row', shard='v.lance')
        assert ledger.pair_status()[2] == {'pair': 'cube9:01', 'target': 5,
                                           'accepted': 1, 'remaining': 4,
                                           'eligible_donors': 1, 'met': False, 'blocked': False}
    finally:
        ledger.close()


def test_migration_reapplies_policy_to_formerly_exempt_full_donors(tmp_path):
    ledger = LegacyQuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')
    ledger.initialize({}, [source('full', 'g1', 'cube1:01'), source('donor', 'g2', 'cube1:01')])
    # Historical shape: attempts and five durable rows accumulated BEFORE the
    # filter was activated (the live policy was enabled late), then the legacy
    # activation exempted the already-full source via its accepted>=5 guard.
    drive(ledger, 'w', 'g1', {'full': (0, 20)})  # retryable failures first
    drive(ledger, 'w', 'g1', {'full': (5, 0)})   # then five durable rows
    assert ledger.configure_success_filter() == 0  # legacy exempted the full donor
    assert ledger.summary()['sources_excluded'] == 0
    settled = ledger.db.execute(
        "SELECT COUNT(*) FROM attempts WHERE source_key='full' AND finished IS NOT NULL").fetchone()[0]
    assert settled == 25
    ledger.close()
    (tmp_path / 'legacy' / 'config.json').write_text(json.dumps({'output': str(tmp_path / 'out')}))
    report = migrate_ledger(tmp_path / 'legacy')
    assert report['state'] == 'migrated'
    audit = report['policy_reapplication']
    assert audit['reapplied'] and audit['newly_excluded_formerly_exempt_donors'] == 1
    assert audit['exclusions_before'] == 0 and audit['exclusions_after'] == 1
    ledger = QuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')
    try:
        summary = ledger.summary()
        assert summary['accepted'] == 5  # credit unchanged by the exclusion
        assert summary['sources_excluded'] == 1 and summary['excluded_accepted'] == 5
        per_source = fill_pair(ledger, 'w', 'g2')
        assert per_source == Counter({'donor': 5})  # 'full' is no longer a donor
        assert ledger.summary()['complete'] and ledger.summary()['accepted'] == 10
    finally:
        ledger.close()


def _force_deficit_for_extra_variants(ledger, *, keep='b', exclude='a', group='g', pair='obj:01'):
    ledger.initialize({}, [source(exclude, group, pair), source(keep, group, pair)])
    ledger.configure_success_filter(minimum_attempts=1)
    drive(ledger, 'w', group, {exclude: (0, 1)})  # excluded -> deficit for donors
    for claim in ledger.claim('w', group, 5):  # settle the keeper's five originals
        ledger.finish(claim, {'accepted': True},
                      row_uuid=f'base-{claim["source"]["key"]}-{claim["slot"]}', shard='verified.lance')
    return ledger


def test_extra_variant_attempt_seeds_are_durably_allocated(tmp_path):
    from sim.manorl.synthesis_campaign import PrefixConfig, attempt_seed as derive
    ledger = QuotaLedger(tmp_path / 'quota.sqlite')
    ledger.initialize({}, [source('a', 'g', 'obj:01'), source('b', 'g', 'obj:01')])
    ledger.configure_success_filter(minimum_attempts=1)
    drive(ledger, 'w', 'g', {'a': (0, 1)})  # excluded -> deficit for donors
    try:
        legacy = ledger.claim('w', 'g', 1)[0]
        assert legacy['slot'] < 5
        assert legacy['seed'] == derive(legacy['source']['key'], legacy['slot'], legacy['attempt'], PrefixConfig())
        assert legacy['seed'] < 2**31  # legacy namespace unchanged
        row = ledger.db.execute("SELECT allocated_seed FROM attempts WHERE token=?",
                                (legacy['token'],)).fetchone()
        assert row['allocated_seed'] is None  # historical rows keep NULL
        ledger.finish(legacy, {'accepted': True}, row_uuid='legacy-row', shard='v.lance')
        for claim in ledger.claim('w', 'g', 4):  # settle the keeper's other originals
            ledger.finish(claim, {'accepted': True},
                          row_uuid=f'base-{claim["slot"]}', shard='v.lance')
        extra = ledger.claim('w', 'g', 8)
        assert extra and all(c['slot'] >= 5 for c in extra)
        for claim in extra:
            assert 2**31 <= claim['seed'] < 2**63  # positive int64 namespace
            row = ledger.db.execute("SELECT allocated_seed FROM attempts WHERE token=?",
                                    (claim['token'],)).fetchone()
            assert row['allocated_seed'] == claim['seed']
        seeds = {c['seed'] for c in extra}
        assert len(seeds) == len(extra) and all(s >= 2**31 for s in seeds)
        # restart: a second connection cannot register a duplicate allocation
        other = QuotaLedger(tmp_path / 'quota.sqlite')
        try:
            import sqlite3
            with pytest.raises(sqlite3.IntegrityError):
                with other.db:
                    other.db.execute("INSERT INTO attempts(token,allocated_seed) VALUES('steal',?)",
                                     (extra[0]['seed'],))
        finally:
            other.close()
        # retry after failure allocates a fresh registered seed for the new attempt
        ledger.finish(extra[0], {'accepted': False})
        retry = [c for c in ledger.claim('w', 'g', 8) if c['slot'] == extra[0]['slot']][0]
        assert retry['attempt'] == extra[0]['attempt'] + 1
        assert retry['seed'] != extra[0]['seed'] and retry['seed'] >= 2**31
        # allocated seeds propagate into the sampler output unchanged
        from test_synthesis_campaign import trajectory
        import numpy as np
        src = trajectory()
        low = np.full(28, -10.)
        _, sample = augment_source_prefix(src, slot=retry['slot'], attempt=retry['attempt'],
                                          joint_lower=low, joint_upper=-low, seed=retry['seed'])
        assert sample['seed'] == retry['seed']
        with pytest.raises(ValueError):
            augment_source_prefix(src, slot=0, attempt=1, joint_lower=low, joint_upper=-low, seed=-1)
        with pytest.raises(ValueError):
            augment_source_prefix(src, slot=0, attempt=1, joint_lower=low, joint_upper=-low, seed=2**63)
    finally:
        ledger.close()


def test_forced_seed_collision_resolves_deterministically(tmp_path):
    from sim.manorl.synthesis_campaign import PrefixConfig, attempt_seed as derive, _collision_candidate
    ledger = _force_deficit_for_extra_variants(QuotaLedger(tmp_path / 'q1.sqlite'))
    try:
        # The first extra claim is (b, variant 5, attempt 1): block its exact
        # candidate so the allocation must take the deterministic alternative.
        candidate = derive('b', 5, 1, PrefixConfig())
        assert candidate >= 2**31
        with ledger.db:
            ledger.db.execute("INSERT INTO attempts(token,source_key,slot,attempt,started,allocated_seed) "
                              "VALUES('blocker','b',5,0,0,?)", (candidate,))
        claim = ledger.claim('w', 'g', 1)[0]
        assert (claim['source']['key'], claim['slot'], claim['attempt']) == ('b', 5, 1)
        assert claim['seed'] != candidate and claim['seed'] >= 2**31
        assert claim['seed'] == _collision_candidate('b', 5, 1, PrefixConfig(), 1)
        # Deterministic resolution: an identical ledger yields the same seed.
        twin = _force_deficit_for_extra_variants(QuotaLedger(tmp_path / 'q2.sqlite'))
        try:
            with twin.db:
                twin.db.execute("INSERT INTO attempts(token,source_key,slot,attempt,started,allocated_seed) "
                                "VALUES('blocker','b',5,0,0,?)", (candidate,))
            twin_claim = twin.claim('w', 'g', 1)[0]
            assert twin_claim['seed'] == claim['seed']
            # Chained collision: block both the attempt-2 base candidate and
            # its first alternative; the allocator must walk to the second.
            ledger.finish(claim, {'accepted': False})  # retry slot 5 as attempt 2
            with ledger.db:
                ledger.db.execute("INSERT INTO attempts(token,source_key,slot,attempt,started,allocated_seed) "
                                  "VALUES('blocker2','b',5,0,0,?)", (derive('b', 5, 2, PrefixConfig()),))
                ledger.db.execute("INSERT INTO attempts(token,source_key,slot,attempt,started,allocated_seed) "
                                  "VALUES('blocker3','b',5,0,0,?)",
                                  (_collision_candidate('b', 5, 2, PrefixConfig(), 1),))
            chained = ledger.claim('w', 'g', 1)[0]
            assert (chained['source']['key'], chained['slot'], chained['attempt']) == ('b', 5, 2)
            assert chained['seed'] == _collision_candidate('b', 5, 2, PrefixConfig(), 2)
        finally:
            twin.close()
    finally:
        ledger.close()


def test_dry_run_projects_on_a_copy_and_never_mutates_the_original(tmp_path):
    from sim.manorl.lance_v2 import file_sha256
    ledger, _ = build_legacy_campaign(tmp_path)
    claim = ledger.claim('dead-publisher', 'g', 1)[0]
    ledger.close()
    shard = tmp_path / 'out' / 'shards' / 'cube1' / 'wave.lance'
    shard.mkdir(parents=True)
    (shard / 'data').write_bytes(b'verified payload')
    (shard / 'campaign-commit.json').write_text(json.dumps({
        'contract': CAMPAIGN_CONTRACT,
        'rows': [{'claim': claim, 'diagnostic': {'accepted': True}, 'row_uuid': 'promoted-row'}],
        'payload_sha256': {'data': file_sha256(shard / 'data')}}))
    original = tmp_path / 'legacy' / 'quota.sqlite'
    before_bytes = file_sha256(original)
    reopened = LegacyQuotaLedger(original)
    before_summary = reopened.summary()
    before_master = [tuple(r) for r in reopened.db.execute(
        "SELECT type,name FROM sqlite_master ORDER BY type,name")]
    before_attempts = reopened.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    reopened.close()

    report = migrate_ledger(tmp_path / 'legacy', dry_run=True)
    assert report['state'] == 'dry_run' and report['projected_state'] == 'migrated'
    assert report['recovered_promoted_rows'] == 1  # recoverable publication reported
    assert report['policy_reapplication']['reapplied'] is True

    assert file_sha256(original) == before_bytes  # original bytes untouched
    reopened = LegacyQuotaLedger(original)
    try:
        assert reopened.summary() == before_summary  # schema/credit/attempt state unchanged
        assert [tuple(r) for r in reopened.db.execute(
            "SELECT type,name FROM sqlite_master ORDER BY type,name")] == before_master
        assert reopened.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == before_attempts
        assert reopened.db.execute(
            "SELECT COUNT(*) FROM slots WHERE state='accepted' AND row_uuid='promoted-row'").fetchone()[0] == 0
    finally:
        reopened.close()
    with pytest.raises(Exception, match='legacy per-source-slot schema'):
        QuotaLedger(original)
    # the real run still performs the reported recovery on the original
    real = migrate_ledger(tmp_path / 'legacy')
    assert real['state'] == 'migrated' and real['recovered_promoted_rows'] == 1
    migrated = QuotaLedger(original)
    try:
        assert migrated.db.execute(
            "SELECT COUNT(*) FROM outputs WHERE row_uuid='promoted-row' AND state='accepted'").fetchone()[0] == 1
    finally:
        migrated.close()


def test_migration_transaction_rolls_back_completely_on_interruption(tmp_path, monkeypatch):
    import sim.manorl.synthesis_migration as migration
    ledger, _ = build_legacy_campaign(tmp_path)
    before = ledger.summary()
    ledger.close()
    original = tmp_path / 'legacy' / 'quota.sqlite'
    real_apply = migration.apply_pair_budget_schema

    def interrupted(db):
        real_apply(db)  # schema work happens, then the process "crashes"
        raise RuntimeError('simulated crash mid-migration')

    monkeypatch.setattr(migration, 'apply_pair_budget_schema', interrupted)
    with pytest.raises(RuntimeError, match='simulated crash'):
        migrate_ledger(tmp_path / 'legacy')
    monkeypatch.undo()
    reopened = LegacyQuotaLedger(original)
    try:
        assert reopened.summary() == before  # data, credit and policy intact
        assert reopened.db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='outputs'").fetchone()[0] == 0
        assert reopened.db.execute(
            "SELECT COUNT(*) FROM settings WHERE key='ledger_schema'").fetchone()[0] == 0
    finally:
        reopened.close()
    # the ledger remains fully migratable afterwards
    report = migrate_ledger(tmp_path / 'legacy')
    assert report['state'] == 'migrated'


def test_already_migrated_repairs_interrupted_candidate_eligibility(tmp_path, monkeypatch):
    import sim.manorl.synthesis_migration as migration
    # Reproduce an interrupted OLD candidate: schema marker committed, stored
    # policy never reapplied, formerly-full donor still eligible.
    ledger = LegacyQuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')
    ledger.initialize({}, [source('full', 'g1', 'cube1:01'), source('donor', 'g2', 'cube1:01')])
    drive(ledger, 'w', 'g1', {'full': (0, 20)})
    drive(ledger, 'w', 'g1', {'full': (5, 0)})
    assert ledger.configure_success_filter() == 0  # legacy exempted the full donor
    ledger.close()
    (tmp_path / 'legacy' / 'config.json').write_text(json.dumps({'output': str(tmp_path / 'out')}))
    monkeypatch.setattr(migration, '_reapply_policy', lambda db, rate, floor: 0)
    try:
        interrupted = migrate_ledger(tmp_path / 'legacy')
    finally:
        monkeypatch.undo()
    assert interrupted['state'] == 'migrated'
    marked = QuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')
    try:
        assert marked.summary()['sources_excluded'] == 0  # eligibility left uncorrected
    finally:
        marked.close()

    repaired = migrate_ledger(tmp_path / 'legacy')  # already-migrated invocation must repair
    assert repaired['state'] == 'already_migrated'
    assert repaired['repaired_eligibility_sources'] == 1
    audit = repaired['policy_reapplication']
    assert audit['newly_excluded_formerly_exempt_donors'] == 1 and audit['exclusions_after'] == 1
    ledger = QuotaLedger(tmp_path / 'legacy' / 'quota.sqlite')
    try:
        summary = ledger.summary()
        assert summary['accepted'] == 5  # original credit preserved
        assert summary['sources_excluded'] == 1 and summary['excluded_accepted'] == 5
        assert summary['target'] == 10 and summary['attempts'] == 25  # history preserved
        per_source = fill_pair(ledger, 'w', 'g2')
        assert per_source == Counter({'donor': 5})  # 'full' is no longer a donor
    finally:
        ledger.close()
    # A healthy migrated ledger needs no repair: re-running is a no-op.
    again = migrate_ledger(tmp_path / 'legacy')
    assert again['state'] == 'already_migrated' and again['repaired_eligibility_sources'] == 0


def settle_one(ledger, owner, group, *, accepted, key=None):
    claims = ledger.claim(owner, group, 1)
    assert claims, 'no claimable row for settlement'
    claim = claims[0]
    if key is not None:
        assert claim['source']['key'] == key
    if accepted:
        ledger.finish(claim, {'accepted': True},
                      row_uuid=f'u-{claim["source"]["key"]}-{claim["slot"]}-{claim["attempt"]}',
                      shard='verified.lance')
    else:
        ledger.finish(claim, {'accepted': False})
    return claim


def settled_state(ledger, key):
    row = ledger.db.execute(
        'SELECT (SELECT COUNT(*) FROM attempts WHERE source_key=? AND finished IS NOT NULL) f,'
        " (SELECT COUNT(*) FROM outputs WHERE source_key=? AND state='accepted') a,"
        " (SELECT COUNT(*) FROM outputs WHERE source_key=? AND state='leased') p",
        (key, key, key)).fetchone()
    return int(row['f']), int(row['a']), int(row['p'])


def excluded(ledger, key):
    return ledger.db.execute('SELECT 1 FROM source_exclusions WHERE source_key=?', (key,)).fetchone() is not None


def test_pending_claims_protect_healthy_donors_from_settlement_order(tmp_path):
    """Failure-first then delayed successes (5/21 with 3 pending -> 7/24):
    the instantaneous rule excluded exactly here; the guard must not."""
    ledger = QuotaLedger(tmp_path / 'quota.sqlite')
    ledger.initialize({}, [source('weak', 'gw', 'obj:01'), source('s', 'gs', 'obj:01')])
    ledger.configure_success_filter()  # .25 / 20
    for _ in range(20):
        settle_one(ledger, 'w', 'gw', accepted=False, key='weak')
    assert excluded(ledger, 'weak') and not excluded(ledger, 's')
    ledger.claim('grow', 'gs', 64)  # reassigns weak's retired debt to donor s
    ledger.release_owner('grow')   # abandoned without settling: no F added
    for _ in range(5):
        settle_one(ledger, 'w', 'gs', accepted=True, key='s')
    for _ in range(15):
        settle_one(ledger, 'w', 'gs', accepted=False, key='s')
    assert settled_state(ledger, 's') == (20, 5, 0)  # exactly .25 stays eligible
    pending = ledger.claim('gpu2', 'gs', 3)
    assert len(pending) == 3 and all(c['source']['key'] == 's' for c in pending)
    # failure settles first while two sibling claims are still in flight:
    # 5/20 -> 5/21 with 2 pending; the instantaneous rule excluded here.
    ledger.finish(pending[0], {'accepted': False})
    assert settled_state(ledger, 's') == (21, 5, 2)
    assert not excluded(ledger, 's')  # (5+2)/(21+2) >= .25: not excludable
    for claim in pending[1:]:  # delayed GPU successes publish later
        ledger.finish(claim, {'accepted': True},
                      row_uuid=f'late-{claim["slot"]}', shard='verified.lance')
    f, a, p = settled_state(ledger, 's')
    assert (f, a, p) == (23, 7, 0) and a / f >= .25
    assert not excluded(ledger, 's')  # delayed GPU successes kept the donor
    assert ledger.summary()['sources_excluded'] == 1
    ledger.close()


def test_exclusion_is_irrevocable_under_best_pending_outcome(tmp_path):
    """Once excluded under the guard, settlement of already-issued work alone
    must never lift the final rate above the threshold."""
    ledger = QuotaLedger(tmp_path / 'quota.sqlite')
    ledger.initialize({}, [source('weak', 'gw', 'obj:01'), source('s', 'gs', 'obj:01')])
    ledger.configure_success_filter()  # .25 / 20 floor
    for _ in range(20):
        settle_one(ledger, 'w', 'gw', accepted=False, key='weak')
    ledger.claim('grow', 'gs', 64); ledger.release_owner('grow')
    settle_one(ledger, 'w', 'gs', accepted=True, key='s')   # A=1
    for _ in range(16):
        settle_one(ledger, 'w', 'gs', accepted=False, key='s')  # F=17, P=0
    pending = ledger.claim('gpu2', 'gs', 4)  # four already-issued claims
    assert len(pending) == 4 and not excluded(ledger, 's')  # (1+4)/(17+4) >= .25
    for claim in pending[:3]:
        ledger.finish(claim, {'accepted': False})  # F=20, P=1: (1+1)/(21) < .25
    assert excluded(ledger, 's')
    record = ledger.db.execute('SELECT settled_attempts,accepted_rows FROM source_exclusions '
                               "WHERE source_key='s'").fetchone()
    assert (record['accepted_rows'], record['settled_attempts']) == (1, 20)
    assert settled_state(ledger, 's') == (20, 1, 1)  # one claim still in flight
    ledger.finish(pending[3], {'accepted': True}, row_uuid='best-outcome', shard='v.lance')
    f, a, p = settled_state(ledger, 's')
    assert (f, a, p) == (21, 2, 0) and a / f < .25
    assert excluded(ledger, 's')  # best pending outcome cannot lift the rate
    assert ledger.summary()['excluded_accepted'] == 2  # late success still credited
    ledger.close()

def test_source_far_below_threshold_is_excluded_despite_pending(tmp_path):
    ledger = QuotaLedger(tmp_path / 'quota.sqlite')
    ledger.initialize({}, [source('weak', 'gw', 'obj:01'), source('s', 'gs', 'obj:01')])
    ledger.configure_success_filter()
    for _ in range(20):
        settle_one(ledger, 'w', 'gw', accepted=False, key='weak')
    ledger.claim('grow', 'gs', 64); ledger.release_owner('grow')
    settle_one(ledger, 'w', 'gs', accepted=True, key='s')
    for _ in range(16):
        settle_one(ledger, 'w', 'gs', accepted=False, key='s')  # F=17, P=0
    pending = ledger.claim('gpu2', 'gs', 5)
    assert len(pending) == 5
    # (1+5)/(17+5) == .27: even five successes look rescuable; not excluded.
    assert not excluded(ledger, 's')
    for claim in pending[:3]:
        ledger.finish(claim, {'accepted': False})  # F=20, P=2
    # (1+2)/(20+2) < .25 at the floor: clearly below even if both remaining
    # pending attempts succeed -> excluded while two claims are STILL leased.
    assert excluded(ledger, 's')
    f, a, p = settled_state(ledger, 's')
    assert (a, f, p) == (1, 20, 2) and (a + p) / (f + p) < .25
    for claim in pending[3:]:
        ledger.finish(claim, {'accepted': True}, row_uuid=f'best-{claim["slot"]}', shard='v.lance')
    f, a, p = settled_state(ledger, 's')
    assert (a, f, p) == (3, 22, 0) and a / f < .25
    assert excluded(ledger, 's')  # best pending outcome could not lift it
    ledger.close()


def test_two_worker_pending_leases_and_release_paths(tmp_path):
    """Worker B's settlements must count worker A's same-source leases as P,
    and a dead-owner release (P drops, no F added) re-evaluates correctly."""
    ledger = QuotaLedger(tmp_path / 'quota.sqlite')
    ledger.initialize({}, [source('weak', 'gw', 'obj:01'), source('s', 'gs', 'obj:01')])
    ledger.configure_success_filter()
    for _ in range(20):
        settle_one(ledger, 'w', 'gw', accepted=False, key='weak')
    ledger.claim('grow', 'gs', 64); ledger.release_owner('grow')
    settle_one(ledger, 'w', 'gs', accepted=True, key='s')
    for _ in range(16):
        settle_one(ledger, 'w', 'gs', accepted=False, key='s')  # F=17, P=0
    held = ledger.claim('worker-a', 'gs', 2)  # a second worker holds two leases
    assert len(held) == 2 and not excluded(ledger, 's')  # (1+2)/(17+2) >= .25
    for _ in range(2):
        settle_one(ledger, 'worker-b', 'gs', accepted=False, key='s')  # B settles, A still holds
    assert settled_state(ledger, 's') == (19, 1, 2) and not excluded(ledger, 's')
    # exact optimistic boundary: (1+2)/(19+2) = 3/21 < .25 would exclude, so
    # push to the boundary case (A+P)/(F+P) == .25 exactly: 5/20 -> eligible.
    ledger2_state = settled_state(ledger, 's')
    assert ledger2_state == (19, 1, 2)
    # Release worker A (dead owner, recovery-before-release done by caller):
    # P drops to 0 without adding F -> 1/19 < .25 but F=19 < floor 20.
    assert ledger.release_owner('worker-a') == 2
    assert settled_state(ledger, 's') == (19, 1, 0) and not excluded(ledger, 's')
    # Abandoned unfinished attempts never count as F: the attempt rows for the
    # released claims have finished IS NULL and the rate uses F=19 only.
    unfinished = ledger.db.execute(
        "SELECT COUNT(*) FROM attempts WHERE source_key='s' AND finished IS NULL").fetchone()[0]
    assert unfinished >= 2 and settled_state(ledger, 's')[0] == 19
    settle_one(ledger, 'w', 'gs', accepted=False, key='s')  # 20th settlement, P=0
    assert excluded(ledger, 's')  # plain cutoff at the floor
    record = ledger.db.execute('SELECT settled_attempts FROM source_exclusions '
                               "WHERE source_key='s'").fetchone()
    assert record['settled_attempts'] == 20  # abandoned attempts not counted
    ledger.close()


def test_release_of_last_leased_owner_triggers_exclusion_decision(tmp_path):
    ledger = QuotaLedger(tmp_path / 'quota.sqlite')
    ledger.initialize({}, [source('weak', 'gw', 'obj:01'), source('s', 'gs', 'obj:01')])
    ledger.configure_success_filter()
    for _ in range(20):
        settle_one(ledger, 'w', 'gw', accepted=False, key='weak')
    ledger.claim('grow', 'gs', 64); ledger.release_owner('grow')
    for _ in range(5):
        settle_one(ledger, 'w', 'gs', accepted=True, key='s')
    for _ in range(15):
        settle_one(ledger, 'w', 'gs', accepted=False, key='s')  # 5/20 == .25 exact
    assert not excluded(ledger, 's')
    held = ledger.claim('ghost', 'gs', 2)
    assert len(held) == 2
    ledger.finish(held[0], {'accepted': False})  # 5/21 with one pending
    assert not excluded(ledger, 's')  # (5+1)/(21+1) >= .25
    ledger.release_owner('ghost')  # last leased owner released: P -> 0
    assert excluded(ledger, 's')  # 5/21 < .25 now decided at release time
    history = ledger.db.execute("SELECT action,pending_claims,evidence FROM source_exclusion_history "
                                "WHERE source_key='s' ORDER BY id").fetchall()
    assert [row['action'] for row in history] == ['excluded']
    assert 'release guard' in history[0]['evidence']
    ledger.close()


OLD_INSTANTANEOUS_TRIGGER_SQL = """CREATE TRIGGER source_success_after_settlement
  AFTER UPDATE OF finished ON attempts
  WHEN NEW.finished IS NOT NULL AND OLD.finished IS NULL
    AND EXISTS(SELECT 1 FROM source_success_policy WHERE id=1)
    AND NOT EXISTS(SELECT 1 FROM source_exclusions WHERE source_key=NEW.source_key)
  BEGIN
    INSERT OR IGNORE INTO source_exclusions
    SELECT NEW.source_key,NEW.finished,a.n,q.n,p.minimum_rate,p.minimum_attempts
    FROM source_success_policy p,
      (SELECT COUNT(*) n FROM attempts
       WHERE source_key=NEW.source_key AND finished IS NOT NULL) a,
      (SELECT COUNT(*) n FROM outputs
       WHERE source_key=NEW.source_key AND state='accepted') q
    WHERE p.id=1 AND a.n>=p.minimum_attempts AND q.n<a.n*p.minimum_rate;
    UPDATE outputs SET state='skipped'
      WHERE source_key=NEW.source_key AND state='ready'
        AND EXISTS(SELECT 1 FROM source_exclusions WHERE source_key=NEW.source_key);
  END"""


def install_old_trigger(ledger):
    """Reproduce a deployed 2a ledger running the pre-guard instantaneous rule."""
    with ledger.db:
        ledger.db.execute('DROP TRIGGER source_success_after_settlement')
        ledger.db.execute(OLD_INSTANTANEOUS_TRIGGER_SQL)


def trigger_sql(ledger):
    return ledger.db.execute("SELECT sql FROM sqlite_master WHERE type='trigger' "
                             "AND name='source_success_after_settlement'").fetchone()[0]


def build_deployed_2a_ledger(tmp_path):
    """Premature exclusion (43-like false positive), a genuinely low source,
    and untouched credit/history, produced under the OLD instantaneous rule."""
    ledger = QuotaLedger(tmp_path / 'quota.sqlite')
    ledger.initialize({}, [source('false', 'gf', 'obj:01'), source('helper', 'gx', 'obj:01'),
                           source('low', 'gl', 'obj:03'), source('healthy', 'gh', 'obj:02')])
    install_old_trigger(ledger)
    ledger.configure_success_filter()  # activates policy under old code semantics
    # Burn 'helper' first so all pair-debt growth lands on 'false' alone.
    for _ in range(20):
        settle_one(ledger, 'w', 'gx', accepted=False, key='helper')
    assert excluded(ledger, 'helper')
    # 'false': 16 failures, three claims leased by a GPU worker, then durable
    # accepts. The OLD rule ignores the pending successes and excludes at 4/20
    # mid-wave (the fifth accept can no longer be claimed for it).
    for _ in range(16):
        settle_one(ledger, 'w', 'gf', accepted=False, key='false')
    pending = ledger.claim('gpu', 'gf', 3)
    assert len(pending) == 3
    for _ in range(4):
        settle_one(ledger, 'w', 'gf', accepted=True, key='false')
    assert excluded(ledger, 'false')
    original = ledger.db.execute('SELECT * FROM source_exclusions WHERE source_key=?',
                                 ('false',)).fetchone()
    assert (original['accepted_rows'], original['settled_attempts']) == (4, 20)
    for i, claim in enumerate(pending):  # delayed successes settle afterwards
        if i < 2:
            ledger.finish(claim, {'accepted': True}, row_uuid=f'late-{i}', shard='v.lance')
        else:
            ledger.finish(claim, {'accepted': False})
    # 'low': genuinely below the rate even fully settled (2/20 when the old
    # rule fires; further failures settle after the decision).
    for _ in range(2):
        settle_one(ledger, 'w', 'gl', accepted=True, key='low')
    for _ in range(23):
        if excluded(ledger, 'low'):
            break
        settle_one(ledger, 'w', 'gl', accepted=False, key='low')
    assert excluded(ledger, 'low')
    # 'healthy': untouched donor with durable rows.
    for _ in range(5):
        settle_one(ledger, 'w', 'gh', accepted=True, key='healthy')
    return ledger, dict(original)


def test_upgrade_repairs_premature_exclusions_and_replaces_trigger(tmp_path):
    ledger, original = build_deployed_2a_ledger(tmp_path)
    assert 'leased' not in trigger_sql(ledger)  # old instantaneous rule active
    before_summary = ledger.summary()
    before_attempts = ledger.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    before_seeds = sorted(r[0] for r in ledger.db.execute(
        'SELECT allocated_seed FROM attempts WHERE allocated_seed IS NOT NULL'))
    false_skipped = ledger.db.execute(
        "SELECT COUNT(*) FROM outputs WHERE source_key='false' AND state='skipped'").fetchone()[0]
    assert false_skipped > 0

    report = ledger.upgrade_eligibility_guard()
    assert report['trigger_replaced'] and 'leased' in trigger_sql(ledger)
    assert report['repaired'] == 1
    repaired = report['reinstated_sources'][0]
    assert repaired['source_key'] == 'false' and repaired['now'] == '6/23'
    assert repaired['at_exclusion'] == '4/20' and repaired['late_settlements'] == 3
    assert not excluded(ledger, 'false') and excluded(ledger, 'low')

    # Original decision record preserved append-only BEFORE reinstating.
    history = ledger.db.execute('SELECT * FROM source_exclusion_history WHERE source_key=? '
                                'ORDER BY id', ('false',)).fetchall()
    assert [row['action'] for row in history] == ['reinstated']
    assert history[0]['decided'] == original['decided']
    assert history[0]['settled_attempts'] == original['settled_attempts'] == 20
    assert history[0]['accepted_rows'] == original['accepted_rows'] == 4
    assert 'premature exclusion repaired' in history[0]['evidence']

    # Credit, targets, attempt history and the seed registry are untouched;
    # skipped variants stay retired (no variant/seed reuse on reinstatement).
    after_summary = ledger.summary()
    assert after_summary['accepted'] == before_summary['accepted']
    assert after_summary['target'] == before_summary['target']
    assert ledger.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == before_attempts
    assert sorted(r[0] for r in ledger.db.execute(
        'SELECT allocated_seed FROM attempts WHERE allocated_seed IS NOT NULL')) == before_seeds
    assert ledger.db.execute(
        "SELECT COUNT(*) FROM outputs WHERE source_key='false' AND state='skipped'").fetchone()[0] == false_skipped

    # The reinstated donor supplies its pair debt with FRESH variants/seeds.
    fresh = ledger.claim('w', 'gf', 4)
    assert len(fresh) == 4 and all(c['source']['key'] == 'false' for c in fresh)
    assert all(c['slot'] >= 5 for c in fresh)  # no skipped variant/seed reuse
    variants = [r[0] for r in ledger.db.execute(
        "SELECT variant FROM outputs WHERE source_key='false' AND state='accepted' ORDER BY variant")]
    assert max(variants) >= 5

    # Safe re-exclusion: genuinely bad decisions after reinstatement remain
    # possible and audited under the (now guarded) normal paths.
    for claim in fresh:
        ledger.finish(claim, {'accepted': False})  # 6/27 < .25 at P=0
    assert excluded(ledger, 'false')
    actions = [row[0] for row in ledger.db.execute(
        'SELECT action FROM source_exclusion_history WHERE source_key=? ORDER BY id', ('false',))]
    assert actions == ['reinstated', 'excluded']

    # Idempotent: a second upgrade repairs nothing and changes nothing.
    again = ledger.upgrade_eligibility_guard()
    assert again['repaired'] == 0 and 'leased' in trigger_sql(ledger)
    ledger.close()


def test_upgrade_requires_quiescence_and_rolls_back_atomically(tmp_path):
    ledger, original = build_deployed_2a_ledger(tmp_path)
    # Re-create an outstanding lease to violate the quiescent precondition.
    with ledger.db:
        ledger.db.execute("UPDATE outputs SET state='leased',owner='live',token='probe' "
                          "WHERE source_key='false' AND variant=(SELECT MIN(variant) FROM outputs "
                          "WHERE source_key='false' AND state='skipped')")
    old_trigger = trigger_sql(ledger)
    with pytest.raises(RuntimeError, match='outstanding leased outputs'):
        ledger.upgrade_eligibility_guard()
    assert trigger_sql(ledger) == old_trigger  # untouched
    assert excluded(ledger, 'false') and excluded(ledger, 'low')
    with ledger.db:  # settle the probe back to its retired state
        ledger.db.execute("UPDATE outputs SET state='skipped',owner=NULL,token=NULL "
                          "WHERE owner='live' AND token='probe'")

    # Mid-transaction failure (history insert succeeds, delete blocked):
    # everything rolls back, including the trigger replacement.
    with ledger.db:
        ledger.db.execute("CREATE TRIGGER block_reinstate BEFORE DELETE ON source_exclusions "
                          "BEGIN SELECT RAISE(ABORT,'reinstate blocked'); END")
    with pytest.raises(Exception, match='reinstate blocked'):
        ledger.upgrade_eligibility_guard()
    assert trigger_sql(ledger) == old_trigger  # old rule restored by rollback
    assert excluded(ledger, 'false') and excluded(ledger, 'low')
    assert ledger.db.execute('SELECT COUNT(*) FROM source_exclusion_history').fetchone()[0] == 0
    row = ledger.db.execute('SELECT * FROM source_exclusions WHERE source_key=?', ('false',)).fetchone()
    assert dict(row) == original  # original decision intact
    with ledger.db:
        ledger.db.execute('DROP TRIGGER block_reinstate')
    # After removing the block the same upgrade succeeds.
    report = ledger.upgrade_eligibility_guard()
    assert report['repaired'] == 1 and 'leased' in trigger_sql(ledger)
    ledger.close()
