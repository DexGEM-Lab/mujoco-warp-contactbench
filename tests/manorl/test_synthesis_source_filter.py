"""Per-source donor filtering under fixed pair budgets.

Exclusions change which sources may generate rows; they never shrink a pair's
output target. Saved rows, leases and legacy credit are preserved.
"""
import sqlite3
import time

import pytest

from sim.manorl.synthesis_campaign import QuotaLedger


@pytest.fixture
def ledger(tmp_path):
    # One shared pair, two scene groups: eligibility is per source while the
    # output obligation belongs to the pair.
    value=QuotaLedger(tmp_path/'quota.sqlite')
    value.initialize({},[{'key':'a','group_key':'a','pair':'cube1:01','identity':'cube1_01_001'},
                         {'key':'b','group_key':'b','pair':'cube1:01','identity':'cube1_01_002'}])
    yield value
    value.close()


def attempt(ledger, group='a', *, accepted=False, diagnostic=None):
    claim=ledger.claim('worker',group,1)[0]
    if accepted:
        ledger.finish(claim,{'accepted':True},
            row_uuid=f"{group}-{claim['slot']}",shard='verified.lance')
    else:ledger.finish(claim,diagnostic or {'accepted':False})
    return claim


def test_filter_is_opt_in(ledger):
    for _ in range(25):attempt(ledger)
    summary=ledger.summary()
    assert summary['source_success_policy'] is None
    assert summary['sources_excluded']==0 and summary['remaining']==10


def test_twenty_settled_attempts_required_and_abandonment_not_counted(ledger):
    ledger.configure_success_filter()
    for _ in range(30):
        ledger.claim('abandoned','a',1)
        ledger.release_owner('abandoned')
    for _ in range(19):attempt(ledger)
    pending=ledger.claim('worker','a',1)[0]
    assert ledger.summary()['sources_excluded']==0
    assert not ledger.summary()['complete']
    ledger.finish(pending,{'accepted':False})
    summary=ledger.summary()
    assert summary['original_target']==10 and summary['target']==10
    assert summary['skipped_slots']==5 and not summary['complete']
    assert summary['blocked_pairs']==[]  # the surviving donor keeps the pair runnable
    assert summary['sources_retained']==1 and summary['sources_excluded']==1
    assert ledger.claim('worker','a',5)==[]  # the excluded group holds no claimable rows
    # the retired debt was reassigned to the donor in the other scene group
    assert [g['group_key'] for g in ledger.groups()]==['b']
    assert ledger.groups()[0]['ready']==10
    report=ledger.exclusion_report()['excluded_sources'][0]
    assert report['settled_attempts']==20 and report['accepted_rows']==0


def test_exact_quarter_is_retained_below_quarter_is_excluded(ledger):
    ledger.configure_success_filter(minimum_attempts=4)
    saved=attempt(ledger,accepted=True)
    for _ in range(3):attempt(ledger)
    assert ledger.summary()['sources_excluded']==0  # Exactly 1/4.
    attempt(ledger)  # Now 1/5.
    summary=ledger.summary()
    assert summary['accepted']==summary['excluded_accepted']==1
    assert summary['skipped_slots']==4 and not summary['complete']
    assert summary['remaining']==9 and summary['blocked_pairs']==[]
    row=ledger.db.execute('SELECT row_uuid,shard FROM outputs WHERE source_key=? AND variant=?',
                         ('a',saved['slot'])).fetchone()
    assert tuple(row)==(f"a-{saved['slot']}",'verified.lance')
    assert ledger.exclusion_report()['excluded_sources'][0]['success_rate_at_exclusion']==.2
    # the surviving donor repays the retired obligations with extra variants
    donor=ledger.claim('worker','b',9)
    assert len(donor)==9
    for claim in donor:
        ledger.finish(claim,{'accepted':True},row_uuid=f"b-{claim['slot']}",shard='verified.lance')
    assert ledger.summary()['complete'] and ledger.summary()['accepted']==10


def test_sampling_rejections_are_settled_generation_attempts(ledger):
    ledger.configure_success_filter()
    for i in range(20):
        diagnostic=({'accepted':False,'prefix_sampling_failure':'joint_limits'} if i%2
                    else {'accepted':False,'initial_geometry_failure':{'reason':'penetration'}})
        attempt(ledger,diagnostic=diagnostic)
    assert ledger.summary()['sources_excluded']==1


def test_historical_full_sources_and_rows_are_preserved(ledger):
    for _ in range(15):attempt(ledger)
    for _ in range(5):attempt(ledger,accepted=True)
    assert ledger.configure_success_filter()==0  # 5/20 == .25 exactly: still eligible
    summary=ledger.summary()
    assert summary['accepted']==5 and summary['sources_retained']==2
    assert summary['pairs_met']==0 and not summary['complete']  # the pair still owes five
    assert summary['remaining']==5


def test_historical_filter_preserves_inflight_tokens_and_accepts_late_rows(ledger):
    for _ in range(19):attempt(ledger)
    attempt(ledger,accepted=True)
    pending=ledger.claim('worker','a',2)
    assert ledger.configure_success_filter()==1
    assert ledger.summary()['states']=={'accepted':1,'leased':2,'ready':5,'skipped':2}
    for claim in pending:
        row=ledger.db.execute('SELECT state,token FROM outputs WHERE source_key=? AND variant=?',
                             ('a',claim['slot'])).fetchone()
        assert tuple(row)==('leased',claim['token'])
    assert not ledger.summary()['complete']
    ledger.finish(pending[0],{'accepted':True},row_uuid='late-row',shard='verified.lance')
    ledger.finish(pending[1],{'accepted':False})
    summary=ledger.summary()
    assert summary['sources_retained']==1 and summary['sources_excluded']==1
    assert summary['target']==10 and summary['remaining']==8 and summary['excluded_accepted']==2
    for _ in range(5):attempt(ledger,'b',accepted=True)
    summary=ledger.summary()
    # 'a' is excluded with a durable deficit; the donor still owes three rows.
    assert not summary['complete'] and summary['blocked_pairs']==[]
    assert summary['accepted']==7 and summary['remaining']==3
    assert summary['retained_accepted']==5


def test_pre_migration_writer_sql_fails_loudly_on_pair_budget_ledger(ledger,tmp_path):
    old=sqlite3.connect(tmp_path/'quota.sqlite')
    try:
        with pytest.raises(sqlite3.OperationalError):
            old.execute("UPDATE slots SET state='ready' WHERE source_key='a'").fetchall()
    finally:old.close()


def test_replayed_acceptance_is_not_a_new_trial(ledger):
    ledger.configure_success_filter()
    saved=attempt(ledger,accepted=True)
    ledger.finish(saved,{'accepted':True},row_uuid=f"a-{saved['slot']}",shard='verified.lance')
    assert ledger.db.execute('SELECT COUNT(*) FROM attempts WHERE finished IS NOT NULL').fetchone()[0]==1
    assert ledger.summary()['accepted']==1


def test_filter_survives_reopen_and_reinitialization(ledger,tmp_path):
    ledger.configure_success_filter(minimum_attempts=1)
    attempt(ledger)
    other=QuotaLedger(tmp_path/'quota.sqlite')
    try:
        other.initialize({},[{'key':'a','group_key':'a','pair':'cube1:01','identity':'cube1_01_001'},
                             {'key':'b','group_key':'b','pair':'cube1:01','identity':'cube1_01_002'}])
        assert other.configure_success_filter(minimum_attempts=1)==0
        assert other.summary()['skipped_slots']==5 and not other.summary()['complete']
        with pytest.raises(ValueError,match='policy changed'):
            other.configure_success_filter(minimum_rate=.5,minimum_attempts=1)
        assert other.success_policy()['minimum_rate']==.25
    finally:other.close()


def test_new_sources_reopen_filtered_campaign(ledger):
    ledger.configure_success_filter(minimum_attempts=1)
    attempt(ledger)
    assert not ledger.summary()['complete']
    ledger.initialize({},[{'key':'c','group_key':'c','pair':'cube1:03'}])
    assert not ledger.summary()['complete']
    assert ledger.summary()['target']==15  # append-only growth for the new pair
    for _ in range(5):attempt(ledger,'c',accepted=True)
    summary=ledger.summary()
    assert not summary['complete'] and summary['blocked_pairs']==[]
    assert summary['pairs_met']==1 and summary['remaining']==10


def test_summary_preserves_callers_transaction(ledger):
    ledger.db.execute('BEGIN')
    ledger.db.execute("INSERT INTO workers VALUES('test',0,'{}')")
    ledger.summary()
    assert ledger.db.in_transaction
    ledger.db.rollback()
    assert ledger.db.execute("SELECT COUNT(*) FROM workers").fetchone()[0]==0


@pytest.mark.parametrize('rate',[-.1,1.1,float('nan'),float('inf'),True,'0.25'])
def test_invalid_policy_rate_rejected(ledger,rate):
    with pytest.raises(ValueError):ledger.configure_success_filter(minimum_rate=rate)
    assert ledger.success_policy() is None


@pytest.mark.parametrize('count',[0,-1,True,1.5])
def test_invalid_policy_sample_floor_rejected(ledger,count):
    with pytest.raises(ValueError):ledger.configure_success_filter(minimum_attempts=count)
    assert ledger.success_policy() is None


def test_completed_quota_does_not_hide_publication_blockers():
    from tools.watch_manorl_synthesis_campaign import should_continue
    assert should_continue({'complete':True,'blocked':{'g':{'exit_code':1}}},paused=False)
    assert should_continue({'complete':True,'workers':{'0':{'pid':123}}},paused=False)
    assert not should_continue({'complete':True,'blocked':{'g':{}}},paused=True)
