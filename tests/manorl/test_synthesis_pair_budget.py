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
