"""Explicit one-way migration: legacy per-source-slot ledgers to fixed pair budgets.

The frozen :class:`LegacyQuotaLedger` reproduces the deployed pre-migration
implementation verbatim so migration tests exercise real historical behavior
instead of hand-written SQL. Production code never instantiates it; only the
migration reconciler and tests do. The migration itself is explicit, idempotent
and refuses to run against ledgers with outstanding leases.
"""
from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path

from sim.manorl.synthesis_campaign import (
    PAIR_BUDGET_CONTRACT, apply_pair_budget_schema, atomic_json, canonical_hash,
    ledger_schema_state,
)

MIGRATION_CONTRACT = "manorl_raw120_fixed_pair_budget_migration_v1"
LEGACY_SUCCESS_FILTER_CONTRACT = "manorl_source_durable_rows_per_settled_attempt_v1"

# Frozen copy of the deployed legacy schema (slots CHECK(slot<5), shrinking
# target semantics). Kept as single statements so tests can build real legacy
# ledgers and the migration can verify what it converts.
LEGACY_SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS sources(key TEXT PRIMARY KEY, group_key TEXT NOT NULL, spec TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS slots(
  source_key TEXT NOT NULL REFERENCES sources(key), slot INTEGER NOT NULL CHECK(slot>=0 AND slot<5),
  state TEXT NOT NULL DEFAULT 'ready', attempts INTEGER NOT NULL DEFAULT 0,
  owner TEXT, token TEXT, error TEXT, row_uuid TEXT UNIQUE, shard TEXT,
  PRIMARY KEY(source_key,slot))""",
    """CREATE INDEX IF NOT EXISTS slots_ready ON slots(state,attempts,source_key,slot)""",
    """CREATE TABLE IF NOT EXISTS attempts(
  token TEXT PRIMARY KEY, source_key TEXT, slot INTEGER, attempt INTEGER,
  started REAL, finished REAL, diagnostic TEXT)""",
    """CREATE TABLE IF NOT EXISTS workers(owner TEXT PRIMARY KEY, heartbeat REAL, details TEXT)""",
    """CREATE TABLE IF NOT EXISTS source_success_policy(
  id INTEGER PRIMARY KEY CHECK(id=1), minimum_rate REAL NOT NULL,
  minimum_attempts INTEGER NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS source_exclusions(
  source_key TEXT PRIMARY KEY REFERENCES sources(key), decided REAL NOT NULL,
  settled_attempts INTEGER NOT NULL, accepted_rows INTEGER NOT NULL,
  minimum_rate REAL NOT NULL, minimum_attempts INTEGER NOT NULL)""",
    """CREATE TRIGGER IF NOT EXISTS excluded_source_ready_update
  AFTER UPDATE OF state ON slots
  WHEN NEW.state='ready' AND EXISTS(
    SELECT 1 FROM source_exclusions WHERE source_key=NEW.source_key)
  BEGIN
    UPDATE slots SET state='skipped' WHERE source_key=NEW.source_key AND slot=NEW.slot;
  END""",
    """CREATE TRIGGER IF NOT EXISTS excluded_source_ready_insert
  AFTER INSERT ON slots
  WHEN NEW.state='ready' AND EXISTS(
    SELECT 1 FROM source_exclusions WHERE source_key=NEW.source_key)
  BEGIN
    UPDATE slots SET state='skipped' WHERE source_key=NEW.source_key AND slot=NEW.slot;
  END""",
    """CREATE TRIGGER IF NOT EXISTS source_success_after_settlement
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
      (SELECT COUNT(*) n FROM slots
       WHERE source_key=NEW.source_key AND state='accepted') q
    WHERE p.id=1 AND q.n<5 AND a.n>=p.minimum_attempts
      AND q.n<a.n*p.minimum_rate;
    UPDATE slots SET state='skipped'
      WHERE source_key=NEW.source_key AND state='ready'
        AND EXISTS(SELECT 1 FROM source_exclusions WHERE source_key=NEW.source_key);
  END""",
)


class LegacyQuotaLedger:
    """Frozen pre-migration ledger: five slots per source, shrinking target."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.row_factory = sqlite3.Row
        state = ledger_schema_state(self.db)
        if state == 'pair_budget':
            self.db.close()
            raise RuntimeError('ledger is already migrated to the pair-budget schema')
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        for statement in LEGACY_SCHEMA_STATEMENTS:
            self.db.execute(statement)

    def success_policy(self) -> dict | None:
        row=self.db.execute('SELECT minimum_rate,minimum_attempts FROM source_success_policy WHERE id=1').fetchone()
        if row is None:return None
        return {'contract':LEGACY_SUCCESS_FILTER_CONTRACT,'minimum_rate':row['minimum_rate'],
                'minimum_attempts':row['minimum_attempts'],
                'denominator':'finished_generation_attempts',
                'numerator':'durably_accepted_source_slots'}

    def configure_success_filter(self, *, minimum_rate: float = .25, minimum_attempts: int = 20) -> int:
        if (isinstance(minimum_rate,bool) or not isinstance(minimum_rate,(int,float)) or not math.isfinite(minimum_rate)
                or not 0<=minimum_rate<=1):
            raise ValueError('minimum success rate must be finite and between zero and one')
        if isinstance(minimum_attempts,bool) or not isinstance(minimum_attempts,int) or minimum_attempts<1:
            raise ValueError('minimum attempts must be a positive integer')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            previous=self.success_policy()
            if previous is not None and (previous['minimum_rate']!=minimum_rate
                    or previous['minimum_attempts']!=minimum_attempts):
                raise ValueError('source success policy changed; an explicit exclusion-policy migration is required')
            self.db.execute('CREATE INDEX IF NOT EXISTS attempts_settled_source '
                            'ON attempts(source_key) WHERE finished IS NOT NULL')
            self.db.execute('INSERT OR IGNORE INTO source_success_policy VALUES(1,?,?)',
                            (minimum_rate,minimum_attempts))
            before=self.db.total_changes
            self.db.execute('''INSERT OR IGNORE INTO source_exclusions
                WITH finished AS (
                  SELECT source_key,COUNT(*) n FROM attempts WHERE finished IS NOT NULL GROUP BY source_key),
                accepted AS (
                  SELECT source_key,COUNT(*) n FROM slots WHERE state='accepted' GROUP BY source_key)
                SELECT f.source_key,?,f.n,COALESCE(a.n,0),?,?
                FROM finished f LEFT JOIN accepted a USING(source_key)
                WHERE f.n>=? AND COALESCE(a.n,0)<5 AND COALESCE(a.n,0)<f.n*?''',
                (time.time(),minimum_rate,minimum_attempts,minimum_attempts,minimum_rate))
            added=self.db.total_changes-before
            self.db.execute("UPDATE slots SET state='skipped' WHERE state='ready' "
                            'AND source_key IN (SELECT source_key FROM source_exclusions)')
            self.db.commit()
            return added
        except BaseException:
            self.db.rollback();raise

    def exclusion_report(self) -> dict:
        rows=[]
        for row in self.db.execute('''SELECT e.*,s.group_key,s.spec FROM source_exclusions e
                JOIN sources s ON s.key=e.source_key ORDER BY s.group_key,e.source_key'''):
            item={k:row[k] for k in row.keys() if k!='spec'}
            item['identity']=json.loads(row['spec']).get('identity',row['source_key'])
            item['success_rate_at_exclusion']=row['accepted_rows']/row['settled_attempts']
            rows.append(item)
        return {'time':time.time(),'policy':self.success_policy(),'excluded_sources':rows}

    def initialize(self, config: dict, sources: list[dict]) -> None:
        digest = canonical_hash(config)
        with self.db:
            previous = self.db.execute("SELECT value FROM settings WHERE key='config'").fetchone()
            if previous and previous[0] != digest:
                raise ValueError("campaign configuration changed; use a new campaign directory")
            self.db.execute("INSERT OR IGNORE INTO settings VALUES('config',?)", (digest,))
            for source in sources:
                key, group = source["key"], source["group_key"]
                encoded = json.dumps(source, sort_keys=True)
                old = self.db.execute("SELECT spec FROM sources WHERE key=?", (key,)).fetchone()
                if old and old[0] != encoded:
                    raise ValueError("source/checkpoint provenance changed within campaign")
                self.db.execute("INSERT OR IGNORE INTO sources VALUES(?,?,?)", (key, group, encoded))
                self.db.executemany("INSERT OR IGNORE INTO slots(source_key,slot) VALUES(?,?)", [(key, i) for i in range(5)])

    def claim(self, owner: str, group_key: str, count: int) -> list[dict]:
        if count < 1:
            raise ValueError("batch size must be positive")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            rows = self.db.execute("""SELECT s.spec,q.source_key,q.slot,q.attempts
                FROM slots q JOIN sources s ON s.key=q.source_key
                WHERE s.group_key=? AND q.state='ready'
                ORDER BY q.attempts,q.slot,q.source_key LIMIT ?""", (group_key, count)).fetchall()
            claims = []
            for row in rows:
                token = uuid.uuid4().hex
                attempt = row["attempts"] + 1
                self.db.execute("UPDATE slots SET state='leased',owner=?,token=?,attempts=? WHERE source_key=? AND slot=?",
                                (owner, token, attempt, row["source_key"], row["slot"]))
                self.db.execute("INSERT INTO attempts(token,source_key,slot,attempt,started) VALUES(?,?,?,?,?)",
                                (token, row["source_key"], row["slot"], attempt, time.time()))
                claims.append({"source": json.loads(row["spec"]), "slot": row["slot"], "attempt": attempt, "token": token, "owner": owner})
            self.db.commit()
            return claims
        except BaseException:
            self.db.rollback(); raise

    def finish(self, claim: dict, diagnostic: dict, *, row_uuid: str | None = None, shard: str | None = None) -> None:
        if bool(row_uuid) != bool(shard):
            raise ValueError("accepted rows require both UUID and durable shard")
        with self.db:
            row = self.db.execute("SELECT state,token,row_uuid,shard FROM slots WHERE source_key=? AND slot=?",
                                  (claim["source"]["key"], claim["slot"])).fetchone()
            if row is not None and row["state"] == "accepted" and row["row_uuid"] == row_uuid and row["shard"] == shard:
                return
            if row is None or row["state"] != "leased" or row["token"] != claim["token"]:
                raise RuntimeError("stale/foreign claim cannot finish a slot")
            self.db.execute("UPDATE slots SET state=?,owner=NULL,token=NULL,error=?,row_uuid=?,shard=? WHERE source_key=? AND slot=?",
                            ("accepted" if row_uuid else "ready", json.dumps(diagnostic, sort_keys=True), row_uuid, shard,
                             claim["source"]["key"], claim["slot"]))
            self.db.execute("UPDATE attempts SET finished=?,diagnostic=? WHERE token=?",
                            (time.time(), json.dumps(diagnostic, sort_keys=True), claim["token"]))

    def release_owner(self, owner: str) -> int:
        with self.db:
            result = self.db.execute("UPDATE slots SET state='ready',owner=NULL,token=NULL WHERE state='leased' AND owner=?", (owner,))
            return result.rowcount

    def heartbeat(self, owner: str, details: dict) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO workers VALUES(?,?,?)", (owner, time.time(), json.dumps(details, sort_keys=True)))

    def summary(self) -> dict:
        own_snapshot=not self.db.in_transaction
        if own_snapshot:self.db.execute('BEGIN')
        try:return self._summary_snapshot()
        finally:
            if own_snapshot:self.db.rollback()

    def _summary_snapshot(self) -> dict:
        counts = dict(self.db.execute("SELECT state,COUNT(*) FROM slots GROUP BY state").fetchall())
        sources = int(self.db.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
        complete = int(self.db.execute("SELECT COUNT(*) FROM (SELECT source_key FROM slots WHERE state='accepted' GROUP BY source_key HAVING COUNT(*)=5)").fetchone()[0])
        accepted = counts.get("accepted", 0)
        excluded=int(self.db.execute('SELECT COUNT(*) FROM source_exclusions').fetchone()[0])
        retained=sources-excluded
        retained_complete=int(self.db.execute('''SELECT COUNT(*) FROM (
            SELECT q.source_key FROM slots q LEFT JOIN source_exclusions e ON e.source_key=q.source_key
            WHERE e.source_key IS NULL AND q.state='accepted' GROUP BY q.source_key HAVING COUNT(*)=5)''').fetchone()[0])
        excluded_accepted=int(self.db.execute('''SELECT COUNT(*) FROM slots q
            JOIN source_exclusions e ON e.source_key=q.source_key WHERE q.state='accepted' ''').fetchone()[0])
        skipped=counts.get('skipped',0);original_target=5*sources;target=original_target-skipped
        return {"sources": sources, "sources_complete": complete, "accepted": accepted,
                "original_target":original_target,"target":target,"remaining":target-accepted,"states":counts,
                "sources_excluded":excluded,"sources_retained":retained,"skipped_slots":skipped,
                "retained_sources_complete":retained_complete,"retained_target":5*retained,
                "retained_accepted":accepted-excluded_accepted,"excluded_accepted":excluded_accepted,
                "source_success_policy":self.success_policy(),
                "attempts": int(self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]),
                "complete": sources>0 and retained_complete==retained and accepted+skipped==original_target
                            and counts.get('leased',0)==0 and counts.get('ready',0)==0}

    def groups(self) -> list[dict]:
        return [dict(row) for row in self.db.execute("""SELECT s.group_key,COUNT(*) remaining,
            SUM(q.state='ready') ready,MIN(q.attempts) min_attempts
            FROM slots q JOIN sources s ON s.key=q.source_key WHERE q.state IN ('ready','leased')
            GROUP BY s.group_key ORDER BY min_attempts,group_key""")]

    def close(self) -> None:
        self.db.close()


def _legacy_pairs(db: sqlite3.Connection) -> tuple[dict[str, int], dict[str, str]]:
    """Explicit object:action pair per source; loud when a spec lacks one."""
    pairs: Counter = Counter()
    by_source: dict[str, str] = {}
    for key, spec in db.execute("SELECT key,spec FROM sources"):
        pair = json.loads(spec).get("pair")
        if not isinstance(pair, str) or ":" not in pair:
            raise RuntimeError(f"legacy source spec lacks an explicit object:action pair: {spec[:200]}")
        pairs[pair] += 1
        by_source[key] = pair
    return dict(pairs), by_source


def _snapshot(db: sqlite3.Connection, table: str, index_column: str) -> dict:
    return {
        "state_counts": {row[0]: row[1] for row in db.execute(f"SELECT state,COUNT(*) FROM {table} GROUP BY state")},
        "accepted_identities": [tuple(row) for row in db.execute(
            f"SELECT source_key,{index_column},row_uuid,shard FROM {table} "
            "WHERE state='accepted' ORDER BY source_key," + index_column)],
        "attempts": int(db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]),
        "settled_attempts": int(db.execute("SELECT COUNT(*) FROM attempts WHERE finished IS NOT NULL").fetchone()[0]),
        "exclusions": int(db.execute("SELECT COUNT(*) FROM source_exclusions").fetchone()[0]),
        "sources": int(db.execute("SELECT COUNT(*) FROM sources").fetchone()[0]),
        "settings": {row[0]: row[1] for row in db.execute("SELECT key,value FROM settings")},
    }


def _leased(db: sqlite3.Connection, table: str) -> int:
    return int(db.execute(f"SELECT COUNT(*) FROM {table} WHERE state='leased'").fetchone()[0])


def _reapply_policy(db: sqlite3.Connection, minimum_rate: float, minimum_attempts: int) -> int:
    """Insert corrected-semantics exclusions for stored-policy ledgers.

    Same statement the v2 settlement trigger uses, minus the accepted>=5
    exemption. Returns the number of newly excluded sources; their ready
    obligations are retired to 'skipped'. Historical rows are never changed.
    """
    db.execute("CREATE INDEX IF NOT EXISTS attempts_settled_source "
               "ON attempts(source_key) WHERE finished IS NOT NULL")
    changed = db.total_changes
    db.execute('''INSERT OR IGNORE INTO source_exclusions
        WITH finished AS (
          SELECT source_key,COUNT(*) n FROM attempts WHERE finished IS NOT NULL GROUP BY source_key),
        accepted AS (
          SELECT source_key,COUNT(*) n FROM outputs WHERE state='accepted' GROUP BY source_key)
        SELECT f.source_key,?,f.n,COALESCE(a.n,0),?,?
        FROM finished f LEFT JOIN accepted a USING(source_key)
        WHERE f.n>=? AND COALESCE(a.n,0)<f.n*?''',
        (time.time(), minimum_rate, minimum_attempts, minimum_attempts, minimum_rate))
    newly_excluded = db.total_changes - changed
    db.execute("UPDATE outputs SET state='skipped' WHERE state='ready' "
               "AND source_key IN (SELECT source_key FROM source_exclusions)")
    return newly_excluded


def _policy_audit(db: sqlite3.Connection, exclusions_before: int, newly_excluded: int,
                  policy_row) -> dict:
    exclusions_after = int(db.execute("SELECT COUNT(*) FROM source_exclusions").fetchone()[0])
    return {
        "reapplied": policy_row is not None,
        "minimum_rate": None if policy_row is None else policy_row["minimum_rate"],
        "minimum_attempts": None if policy_row is None else policy_row["minimum_attempts"],
        "exclusions_before": exclusions_before,
        "newly_excluded_formerly_exempt_donors": newly_excluded,
        "exclusions_after": exclusions_after,
        "note": ("stored policy reapplied under corrected semantics before any claim"
                 if policy_row is not None else
                 "no stored source-success policy; run set-source-filter before claiming"),
    }


def _copy_ledger_readonly(source_path: Path, destination_path: Path) -> None:
    """Copy a ledger for projection without any writable connection to it."""
    try:
        source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
        try:
            destination = sqlite3.connect(destination_path)
            try:
                source.backup(destination)
            finally:
                destination.close()
        finally:
            source.close()
    except sqlite3.OperationalError:
        # Quiescent-ledger fallback: copy the file set byte-for-byte; all
        # subsequent writes still happen only on the copy.
        shutil.copy2(source_path, destination_path)
        for suffix in ("-wal", "-shm"):
            if Path(str(source_path) + suffix).exists():
                shutil.copy2(str(source_path) + suffix, str(destination_path) + suffix)


def _migrate_file(db_path: Path, campaign: Path, output_root: Path, *, verify_hashes: bool) -> dict:
    from sim.manorl.synthesis_campaign import QuotaLedger
    from sim.manorl.synthesis_publication import reconcile_shards

    probe = sqlite3.connect(db_path)
    try:
        state = ledger_schema_state(probe)
    finally:
        probe.close()
    if state == 'fresh':
        raise RuntimeError('refusing to migrate an empty ledger; prepare the campaign first')
    if state != 'legacy':
        # Already migrated: verify eligibility is corrected under the stored
        # policy and REPAIR it if an interrupted earlier candidate committed the
        # schema marker without finishing the reapplication. Credit, history
        # and original exclusion rows are preserved either way.
        ledger = QuotaLedger(db_path)
        try:
            policy_row = ledger.db.execute(
                "SELECT minimum_rate,minimum_attempts FROM source_success_policy WHERE id=1").fetchone()
            exclusions_before = int(ledger.db.execute("SELECT COUNT(*) FROM source_exclusions").fetchone()[0])
            repaired = 0
            if policy_row is not None:
                with ledger.db:
                    repaired = _reapply_policy(ledger.db, policy_row["minimum_rate"],
                                               policy_row["minimum_attempts"])
            return {'contract': MIGRATION_CONTRACT, 'campaign': str(campaign),
                    'state': 'already_migrated',
                    'repaired_eligibility_sources': repaired,
                    'policy_reapplication': _policy_audit(ledger.db, exclusions_before, repaired, policy_row),
                    'summary': ledger.summary()}
        finally:
            ledger.close()

    legacy = LegacyQuotaLedger(db_path)
    try:
        if ledger_schema_state(legacy.db) != "legacy":
            raise RuntimeError("unsupported ledger schema state for migration")
        # Promotion-before-ledger crashes are recovered under legacy semantics
        # first, so the copied state includes every durably published row.
        recovered = reconcile_shards(legacy, output_root, verify_hashes=verify_hashes)
        # Quiescence AFTER reconciliation: any lease still outstanding belongs
        # to a live or abandoned worker whose rows are not durable; refusing is
        # the only way not to silently abandon that work.
        if _leased(legacy.db, "slots"):
            raise RuntimeError(
                "outstanding leased slots remain after shard reconciliation: drain workers "
                "and settle pending publications before migrating; leases are never "
                "silently abandoned")
        before = _snapshot(legacy.db, "slots", "slot")
        pairs, pairs_by_source = _legacy_pairs(legacy.db)
        policy_row = legacy.db.execute(
            "SELECT minimum_rate,minimum_attempts FROM source_success_policy WHERE id=1").fetchone()
        exclusions_before = int(legacy.db.execute("SELECT COUNT(*) FROM source_exclusions").fetchone()[0])
        report = {"contract": MIGRATION_CONTRACT, "campaign": str(campaign), "output": str(output_root),
                  "recovered_promoted_rows": recovered,
                  "pair_targets": {pair: 5 * count for pair, count in sorted(pairs.items())},
                  "before": before,
                  "accepted_identity_sha256": canonical_hash(before["accepted_identities"])}
        db = legacy.db
        db.execute("BEGIN IMMEDIATE")
        try:
            if ledger_schema_state(db) != "legacy":
                raise RuntimeError("ledger schema changed during migration")
            if _leased(db, "slots"):
                raise RuntimeError("leases appeared during migration")
            slots = db.execute(
                "SELECT source_key,slot,state,attempts,owner,token,error,row_uuid,shard FROM slots").fetchall()
            db.execute("DROP TABLE slots")  # drops the legacy slot triggers with it
            db.execute("DROP TRIGGER IF EXISTS source_success_after_settlement")
            # The explicit pair becomes a real column; it was only inside specs.
            columns = {row[1] for row in db.execute("PRAGMA table_info(sources)")}
            if "pair" not in columns:
                db.execute("ALTER TABLE sources ADD COLUMN pair TEXT")
            db.executemany("UPDATE sources SET pair=? WHERE key=?",
                           [(pair, key) for key, pair in pairs_by_source.items()])
            if db.execute("SELECT COUNT(*) FROM sources WHERE pair IS NULL").fetchone()[0]:
                raise RuntimeError("legacy source without an explicit object:action pair")
            apply_pair_budget_schema(db)
            db.executemany(
                "INSERT INTO outputs(pair,source_key,variant,state,attempts,owner,token,error,row_uuid,shard) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                [(pairs_by_source[r["source_key"]], r["source_key"], r["slot"], r["state"], r["attempts"],
                  r["owner"], r["token"], r["error"], r["row_uuid"], r["shard"]) for r in slots])
            for pair, count in sorted(pairs.items()):
                db.execute("INSERT INTO pair_targets(pair,original_sources,target_rows) VALUES(?,?,?)",
                           (pair, count, 5 * count))
            # Legacy triggers guarantee no ready rows remain for excluded
            # sources; normalize defensively and verify below either way.
            db.execute("UPDATE outputs SET state='skipped' WHERE state='ready' "
                       "AND source_key IN (SELECT source_key FROM source_exclusions)")
            # Stored-policy reapplication happens INSIDE this transaction: the
            # pair-budget marker below is committed only after eligibility is
            # corrected, so a crash can never expose an uncorrected donor set.
            newly_excluded = 0
            if policy_row is not None:
                newly_excluded = _reapply_policy(db, policy_row["minimum_rate"],
                                                 policy_row["minimum_attempts"])
            db.execute("INSERT OR REPLACE INTO settings VALUES('ledger_schema',?)", (PAIR_BUDGET_CONTRACT,))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        after = _snapshot(db, "outputs", "variant")
        if after["accepted_identities"] != before["accepted_identities"]:
            raise RuntimeError("migration altered accepted-row lineage; the transaction was committed but verification failed")
        for field in ("attempts", "settled_attempts", "sources"):
            if after[field] != before[field]:
                raise RuntimeError(f"migration changed {field}: {before[field]} -> {after[field]}")
        if after["state_counts"].get("accepted") != before["state_counts"].get("accepted"):
            raise RuntimeError("migration changed accepted-row counts")
        if sum(after["state_counts"].values()) != sum(before["state_counts"].values()):
            raise RuntimeError("migration changed the number of output obligations")
        # The only permitted state change beyond the copy is ready -> skipped
        # for sources excluded by the in-transaction policy reapplication.
        permitted = max(0, after["state_counts"].get("skipped", 0) - before["state_counts"].get("skipped", 0))
        if (after["state_counts"].get("ready", 0) != before["state_counts"].get("ready", 0) - permitted
                or after["state_counts"].get("leased", 0) != before["state_counts"].get("leased", 0)):
            raise RuntimeError("migration changed obligation states beyond policy reapplication")
        if after["settings"].get("config") != before["settings"].get("config"):
            raise RuntimeError("migration changed the campaign config digest")
        report["state"] = "migrated"
        report["after"] = after
        report["policy_reapplication"] = _policy_audit(db, exclusions_before, newly_excluded, policy_row)
        return report
    finally:
        legacy.close()


def migrate_ledger(campaign: Path, *, output: Path | None = None, dry_run: bool = False,
                   verify_hashes: bool = True) -> dict:
    """Explicitly convert a legacy per-source-slot ledger to fixed pair budgets.

    Requires a quiescent ledger (no outstanding leases) and reconciles promoted
    shards against the legacy schema BEFORE converting, so no durable published
    row is lost and no lease is silently abandoned. The schema conversion and
    the stored 25%/20 policy reapplication commit in ONE transaction: the
    pair-budget marker is never visible before eligibility is corrected.
    Preserves accepted UUID/source/slot/shard lineage, attempt history, the
    config digest, the exclusion audit and on-disk commit manifests. Idempotent:
    an already-migrated ledger is verified and repaired (an interrupted earlier
    candidate may have left eligibility uncorrected) without touching credit.

    ``dry_run`` never opens the original ledger for writing (no writable
    legacy construction, no ``finish`` on the original, no shard writes): the
    reconciliation and conversion are projected onto an isolated host-local
    temporary SQLite copy and the projected report is returned.
    """
    campaign = Path(campaign).resolve()
    db_path = campaign / "quota.sqlite"
    if not db_path.is_file():
        raise RuntimeError(f"no ledger database at {db_path}")
    config_path = campaign / "config.json"
    if not config_path.is_file():
        raise RuntimeError("legacy campaign directory lacks config.json; publication root is required to reconcile before migration")
    config = json.loads(config_path.read_text())
    output_root = Path(output) if output is not None else Path(config["output"])

    if not dry_run:
        return _migrate_file(db_path, campaign, output_root, verify_hashes=verify_hashes)
    with tempfile.TemporaryDirectory(prefix="manorl-migrate-projection-") as folder:
        projection = Path(folder) / "projection.sqlite"
        _copy_ledger_readonly(db_path, projection)
        report = _migrate_file(projection, campaign, output_root, verify_hashes=verify_hashes)
        projected_state = report["state"]
        report["state"] = "dry_run"
        report["projected_state"] = projected_state
        report["campaign"] = str(campaign)
        report["output"] = str(output_root)
        return report
def write_migration_report(campaign: Path, report: dict) -> None:
    atomic_json(Path(campaign) / "migration-report.json", report)
