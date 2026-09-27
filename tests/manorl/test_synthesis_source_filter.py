"""Per-source filtering must preserve saved rows, leases and legacy writers."""
import sqlite3
import time

import pytest

from sim.manorl.synthesis_campaign import QuotaLedger


@pytest.fixture
def ledger(tmp_path):
    value=QuotaLedger(tmp_path/'quota.sqlite')
    value.initialize({},[{'key':'a','group_key':'a','identity':'cube1_01_001'}])
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
    assert summary['sources_excluded']==0 and summary['remaining']==5


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
    assert summary['original_target']==5 and summary['target']==0
    assert summary['skipped_slots']==5 and summary['complete']
    assert summary['sources_retained']==0 and summary['sources_excluded']==1
    assert ledger.groups()==[] and ledger.claim('worker','a',5)==[]
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
    assert summary['skipped_slots']==4 and summary['complete']
    row=ledger.db.execute('SELECT row_uuid,shard FROM slots WHERE source_key=? AND slot=?',
                         ('a',saved['slot'])).fetchone()
    assert tuple(row)==(f"a-{saved['slot']}",'verified.lance')
    assert ledger.exclusion_report()['excluded_sources'][0]['success_rate_at_exclusion']==.2


def test_sampling_rejections_are_settled_generation_attempts(ledger):
    ledger.configure_success_filter()
    for i in range(20):
        diagnostic=({'accepted':False,'prefix_sampling_failure':'joint_limits'} if i%2
                    else {'accepted':False,'initial_geometry_failure':{'reason':'penetration'}})
        attempt(ledger,diagnostic=diagnostic)
    assert ledger.summary()['sources_excluded']==1


def test_historical_full_sources_and_rows_are_preserved(ledger):
    for _ in range(20):attempt(ledger)
    for _ in range(5):attempt(ledger,accepted=True)
    assert ledger.configure_success_filter()==0
    summary=ledger.summary()
    assert summary['accepted']==5 and summary['sources_retained']==1
    assert summary['retained_sources_complete']==1 and summary['complete']


def test_historical_filter_preserves_inflight_tokens_and_accepts_late_rows(ledger):
    for _ in range(19):attempt(ledger)
    attempt(ledger,accepted=True)
    pending=ledger.claim('worker','a',2)
    ledger.initialize({},[{'key':'b','group_key':'b'}])
    assert ledger.configure_success_filter()==1
    assert ledger.summary()['states']=={'accepted':1,'leased':2,'ready':5,'skipped':2}
    for claim in pending:
        row=ledger.db.execute('SELECT state,token FROM slots WHERE source_key=? AND slot=?',
                             ('a',claim['slot'])).fetchone()
        assert tuple(row)==('leased',claim['token'])
    assert not ledger.summary()['complete']
    ledger.finish(pending[0],{'accepted':True},row_uuid='late-row',shard='verified.lance')
    ledger.finish(pending[1],{'accepted':False})
    summary=ledger.summary()
    assert summary['sources_retained']==1 and summary['sources_excluded']==1
    assert summary['target']==7 and summary['remaining']==5 and summary['excluded_accepted']==2
    assert [g['group_key'] for g in ledger.groups()]==['b']
    for _ in range(5):attempt(ledger,'b',accepted=True)
    summary=ledger.summary()
    assert summary['complete'] and summary['accepted']==7
    assert summary['retained_accepted']==summary['retained_target']==5
    assert summary['retained_sources_complete']==summary['sources_retained']==1


def test_legacy_sql_finish_and_release_cannot_resurrect_excluded_work(ledger,tmp_path):
    old=sqlite3.connect(tmp_path/'quota.sqlite')
    try:
        pending=ledger.claim('old','a',2)
        ledger.configure_success_filter(minimum_attempts=1)
        # Same two updates as the deployed pre-filter finish implementation.
        with old:
            old.execute("UPDATE slots SET state='ready',owner=NULL,token=NULL WHERE source_key=? AND slot=?",
                        ('a',pending[0]['slot']))
            old.execute('UPDATE attempts SET finished=?,diagnostic=? WHERE token=?',
                        (time.time(),'{}',pending[0]['token']))
        assert ledger.summary()['states']=={'leased':1,'skipped':4}
        with old:
            old.execute("UPDATE slots SET state='ready',owner=NULL,token=NULL WHERE state='leased' AND owner='old'")
        assert old.execute("SELECT COUNT(*) FROM slots WHERE state='ready'").fetchone()[0]==0
        assert ledger.summary()['complete']
        assert ledger.exclusion_report()['excluded_sources'][0]['settled_attempts']==1
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
        other.initialize({},[{'key':'a','group_key':'a','identity':'cube1_01_001'}])
        assert other.configure_success_filter(minimum_attempts=1)==0
        assert other.summary()['skipped_slots']==5 and other.summary()['complete']
        with pytest.raises(ValueError,match='policy changed'):
            other.configure_success_filter(minimum_rate=.5,minimum_attempts=1)
        assert other.success_policy()['minimum_rate']==.25
    finally:other.close()


def test_new_sources_reopen_filtered_campaign(ledger):
    ledger.configure_success_filter(minimum_attempts=1)
    attempt(ledger)
    assert ledger.summary()['complete']
    ledger.initialize({},[{'key':'b','group_key':'b'}])
    assert not ledger.summary()['complete']
    assert ledger.summary()['retained_target']==5
    for _ in range(5):attempt(ledger,'b',accepted=True)
    assert ledger.summary()['complete']


def test_summary_preserves_callers_transaction(ledger):
    ledger.db.execute('BEGIN')
    ledger.db.execute("INSERT INTO workers VALUES('test',0,'{}')")
    ledger.summary()
    assert ledger.db.in_transaction
    ledger.db.rollback()
    assert ledger.db.execute('SELECT COUNT(*) FROM workers').fetchone()[0]==0


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
