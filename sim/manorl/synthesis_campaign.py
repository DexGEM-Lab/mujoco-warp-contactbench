"""Source-relative prefix sampling and durable accepted-row quotas.

This module has no JAX, Torch, Lance or asset imports. Physics and publication
are owned by the runtime; SQLite records accepted rows only AFTER publication.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.approach_prefix import _discrete_c1_prefix, _readonly
from sim.manorl.trajectory import ReferenceTrajectory

CAMPAIGN_CONTRACT = "manorl_raw120_source_prefix_offline_retreat_quota_v1"
PREFIX_CONTRACT = "manorl_source_relative_shell_prefix_preserve_base_early_v1"
SUCCESS_FILTER_CONTRACT = "manorl_source_durable_rows_per_settled_attempt_v2"
PAIR_BUDGET_CONTRACT = "manorl_raw120_fixed_pair_output_budget_v2"


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_key(trajectory: ReferenceTrajectory) -> str:
    i = trajectory.identity
    return canonical_hash([str(i.dataset_path), int(i.dataset_version), str(i.uuid), i.identity])


@dataclass(frozen=True)
class PrefixConfig:
    quota: int = 5
    pre_padding: int = 80
    early_phase_steps: int = 80
    minimum_offset_m: float = 0.05
    maximum_offset_m: float = 0.50
    wrist_rotation_deg: float = 45.0
    finger_offset_rad: float = 0.15
    nominal_speed_m_s: float = 0.30
    nominal_angular_speed_deg_s: float = 90.0
    minimum_prefix_frames: int = 20
    maximum_prefix_frames: int = 300
    vertical_arc_height_m: float = 0.04
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("quota", "pre_padding", "early_phase_steps", "minimum_prefix_frames", "maximum_prefix_frames", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.quota != 5:
            raise ValueError("this campaign requires exactly five accepted slots per source")
        if not (0 < self.minimum_offset_m <= self.maximum_offset_m and
                0 <= self.wrist_rotation_deg <= 90 and 0 <= self.finger_offset_rad <= 0.5 and
                self.nominal_speed_m_s > 0 and self.nominal_angular_speed_deg_s > 0 and
                4 <= self.minimum_prefix_frames <= self.maximum_prefix_frames and
                0 <= self.vertical_arc_height_m <= 0.1):
            raise ValueError("invalid prefix sampling bounds")
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("prefix configuration must be finite")


def attempt_seed(key: str, slot: int, attempt: int, config: PrefixConfig) -> int:
    """Deterministic per-variant seed; slots 0-4 keep the historical derivation.

    Extra donor variants (slot >= quota) derive their base candidate in a
    dedicated positive signed-int64 namespace [2**31, 2**63), disjoint from
    the legacy [0, 2**31-2] range by construction. Production claims allocate
    the durable seed inside the ledger transaction (see QuotaLedger.claim)
    with a UNIQUE check and deterministic collision resolution; this function
    only derives the base candidate for standalone callers.
    """
    if slot < 0 or attempt < 1:
        raise ValueError("slot/attempt is outside the campaign contract")
    if slot < config.quota:
        return int(canonical_hash([config.seed, key, slot, attempt, asdict(config)])[:8], 16) % (2**31 - 1)
    return 2**31 + int(canonical_hash([config.seed, key, "extra-variant", slot, attempt, asdict(config)])[:15], 16) % (2**63 - 2**31)


def _collision_candidate(key: str, slot: int, attempt: int, config: PrefixConfig, retry: int) -> int:
    """Deterministic in-namespace alternative after a durable seed collision."""
    return 2**31 + int(canonical_hash([config.seed, key, "extra-variant-collision", retry, slot, attempt, asdict(config)])[:15], 16) % (2**63 - 2**31)


def augment_source_prefix(
    source: ReferenceTrajectory, *, slot: int, attempt: int,
    joint_lower: np.ndarray, joint_upper: np.ndarray,
    config: PrefixConfig = PrefixConfig(), seed: int | None = None,
) -> tuple[ReferenceTrajectory, dict]:
    """Five radial strata, stratified azimuth and randomized feasible posture.

    Each retry stays in its original radial stratum. Directions span the sphere;
    infeasible starts are rejected, not clipped into a different displacement.
    The base reference is copied bit-for-bit and retains its trained early phase.
    """
    if source.hand_sides != ("right",) or source.control_fps != 120 or source.reference_fps != 120:
        raise ValueError("source-prefix campaign requires a right-only 120Hz source")
    if source.augmentation_prefix_frames or source.augmentation_suffix_frames:
        raise ValueError("source is already augmented")
    if source.movement_start_step is None or not 0 <= source.movement_start_step <= config.pre_padding:
        raise ValueError("source was not cropped with the campaign pre-padding")
    lower, upper = np.asarray(joint_lower), np.asarray(joint_upper)
    if lower.shape != (28,) or upper.shape != (28,) or np.any(lower >= upper):
        raise ValueError("expected 28 ordered joint limits")
    key = source_key(source)
    if seed is None:
        seed = attempt_seed(key, slot, attempt, config)
    elif (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63):
        raise ValueError("allocated seed must be a nonnegative signed int64")
    rng = np.random.default_rng(seed)
    # Extra donor variants (slot >= quota) cycle the same five radial/azimuth
    # strata as slots 0-4 instead of widening the sampled displacement range.
    stratum = slot if slot < config.quota else slot % config.quota
    width = (config.maximum_offset_m - config.minimum_offset_m) / config.quota
    low = config.minimum_offset_m + stratum * width
    high = low + width
    radius = float(rng.uniform(low, high))
    # A per-source permutation keeps all five azimuth strata represented without
    # correlating large radii with one direction across the corpus.
    order = np.random.default_rng(int(key[:16], 16) ^ config.seed).permutation(config.quota)
    azimuth = 2 * math.pi * (int(order[stratum]) + rng.random()) / config.quota
    cosine = float(rng.uniform(-1, 1))
    direction = np.array([math.sqrt(1-cosine*cosine)*math.cos(azimuth),
                          math.sqrt(1-cosine*cosine)*math.sin(azimuth), cosine])
    start = np.array(source.q_ref[0], copy=True)
    start[:3] += radius * direction
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = float(rng.uniform(0, math.radians(config.wrist_rotation_deg)))
    rotation = Rotation.from_rotvec(axis * angle) * Rotation.from_euler("XYZ", start[3:6])
    euler = rotation.as_euler("XYZ")
    start[3:6] += (euler - start[3:6] + math.pi) % (2 * math.pi) - math.pi
    # Raw MANO references may exceed physical finger limits (e.g. negative
    # PIP flexion). Training's native command_target clamps them. Perturb the
    # reachable baseline, not an unattainable raw value that erases variation.
    start[6:] = np.clip(np.clip(start[6:], lower[6:], upper[6:])
                        + rng.uniform(-config.finger_offset_rad, config.finger_offset_rad, 22),
                        lower[6:], upper[6:])
    if np.any(start < lower) or np.any(start > upper):
        raise ValueError("sampled_start_joint_limits")
    duration = max(radius / config.nominal_speed_m_s,
                   math.degrees(angle) / config.nominal_angular_speed_deg_s,
                   math.degrees(float(np.max(np.abs(start[6:] - source.q_ref[0, 6:])))) / config.nominal_angular_speed_deg_s)
    frames = int(np.clip(math.ceil(duration * 120), config.minimum_prefix_frames, config.maximum_prefix_frames))
    dt = 1 / 120
    reachable = np.array(source.q_ref[:2], copy=True)
    reachable[:, 6:] = np.clip(reachable[:, 6:], lower[6:], upper[6:])
    q = _discrete_c1_prefix(start, reachable, frames=frames, dt=dt)
    q[:, :3] = _discrete_c1_prefix(start[:3], np.asarray(source.q_ref[:2, :3]), frames=frames, dt=dt,
                                  vertical_arc_height=config.vertical_arc_height_m)
    # Endpoint derivatives can overshoot a bound; the native servo makes this
    # same projection. The unchanged base remains raw, and will be clamped by
    # the existing controller at execution as it was during training.
    q[:, 6:] = np.clip(q[:, 6:], lower[6:], upper[6:])
    if np.any(q < lower[None] - 1e-6) or np.any(q > upper[None] + 1e-6):
        raise ValueError("sampled_prefix_joint_limits")
    q_all = _readonly(np.concatenate([q, source.q_ref]))
    def prepend(value: np.ndarray) -> np.ndarray:
        return _readonly(np.concatenate([np.repeat(value[:1], frames, axis=0), value]))
    augmented = replace(
        source, q_ref=q_all, q_ref_by_side={"right": q_all},
        source_indices=_readonly(np.r_[np.repeat(source.source_indices[:1], frames), source.source_indices], dtype=np.int64),
        timestamps=_readonly(np.r_[source.timestamps[0] - dt * np.arange(frames, 0, -1), source.timestamps]),
        object_pos=prepend(source.object_pos), object_pos_raw=prepend(source.object_pos_raw),
        object_quat_xyzw=prepend(source.object_quat_xyzw),
        movement_start_step=source.movement_start_step + frames,
        movement_end_step=source.movement_end_step + frames,
        augmentation_prefix_frames=frames,
        augmentation_base_early_phase_steps=config.early_phase_steps,
    )
    return augmented, {
        "contract": PREFIX_CONTRACT, "source_key": key, "seed": seed,
        "slot": slot, "attempt": attempt, "radius_stratum_m": [low, high],
        "displacement_m": radius, "direction": direction.tolist(),
        "azimuth_stratum": int(order[stratum]), "wrist_rotation_deg": math.degrees(angle),
        "start_q": start.tolist(), "prefix_frames": frames,
        "base_pre_padding": source.movement_start_step,
        "base_early_phase_steps": config.early_phase_steps,
        "finger_template": "native_servo_clipped_source_frame0",
        "source_finger_limit_excess": np.flatnonzero(
            (source.q_ref[0, 6:] < lower[6:]) | (source.q_ref[0, 6:] > upper[6:])
        ).tolist(),
        "policy_enable_step": frames + config.early_phase_steps,
        "config": asdict(config),
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def acquire_owner_lock(root: Path, owner: str):
    """Hold across GPU and inherited CPU-publisher lifetimes, including SIGKILL."""
    if not owner or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._' for c in owner):
        raise ValueError('invalid worker owner name')
    path=root/'leases'/(owner+'.lock');path.parent.mkdir(parents=True,exist_ok=True)
    handle=path.open('a+b')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:handle.close();raise
    return handle


def owner_lock_busy(root: Path, owner: str) -> bool:
    try:handle=acquire_owner_lock(root,owner)
    except BlockingIOError:return True
    handle.close();return False



def ledger_schema_state(db: sqlite3.Connection) -> str:
    """Classify a ledger file without mutating it: legacy, pair_budget or fresh."""
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'outputs' in tables and 'slots' in tables:
        raise RuntimeError('ledger contains both legacy slots and pair-budget outputs tables')
    if 'outputs' in tables:
        return 'pair_budget'
    if 'slots' in tables:
        return 'legacy'
    if tables & {'sources', 'attempts', 'workers', 'source_exclusions', 'source_success_policy'}:
        raise RuntimeError('unrecognized ledger schema')
    return 'fresh'


class LegacyLedgerError(RuntimeError):
    """A legacy per-source-slot ledger was opened by pair-budget code."""


# One statement per entry: executescript() would commit, so migrations apply
# these inside a single transaction with execute().
PAIR_BUDGET_SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS sources(key TEXT PRIMARY KEY, group_key TEXT NOT NULL, pair TEXT NOT NULL, spec TEXT NOT NULL)""",
    """CREATE INDEX IF NOT EXISTS sources_pair ON sources(pair)""",
    """CREATE TABLE IF NOT EXISTS pair_targets(
  pair TEXT PRIMARY KEY, original_sources INTEGER NOT NULL CHECK(original_sources>0),
  target_rows INTEGER NOT NULL CHECK(target_rows>=0))""",
    """CREATE TABLE IF NOT EXISTS outputs(
  pair TEXT NOT NULL, source_key TEXT NOT NULL REFERENCES sources(key),
  variant INTEGER NOT NULL CHECK(variant>=0), state TEXT NOT NULL DEFAULT 'ready',
  attempts INTEGER NOT NULL DEFAULT 0, owner TEXT, token TEXT, error TEXT,
  row_uuid TEXT UNIQUE, shard TEXT, PRIMARY KEY(source_key,variant))""",
    """CREATE INDEX IF NOT EXISTS outputs_ready ON outputs(state,attempts,source_key,variant)""",
    """CREATE INDEX IF NOT EXISTS outputs_pair ON outputs(pair,state)""",
    """CREATE TABLE IF NOT EXISTS attempts(
  token TEXT PRIMARY KEY, source_key TEXT, slot INTEGER, attempt INTEGER,
  started REAL, finished REAL, diagnostic TEXT, allocated_seed INTEGER)""",
    """CREATE UNIQUE INDEX IF NOT EXISTS attempts_allocated_seed
  ON attempts(allocated_seed) WHERE allocated_seed IS NOT NULL""",
    """CREATE TABLE IF NOT EXISTS workers(owner TEXT PRIMARY KEY, heartbeat REAL, details TEXT)""",
    """CREATE TABLE IF NOT EXISTS source_success_policy(
  id INTEGER PRIMARY KEY CHECK(id=1), minimum_rate REAL NOT NULL,
  minimum_attempts INTEGER NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS source_exclusions(
  source_key TEXT PRIMARY KEY REFERENCES sources(key), decided REAL NOT NULL,
  settled_attempts INTEGER NOT NULL, accepted_rows INTEGER NOT NULL,
  minimum_rate REAL NOT NULL, minimum_attempts INTEGER NOT NULL)""",
    """CREATE TRIGGER IF NOT EXISTS excluded_output_ready_update
  AFTER UPDATE OF state ON outputs
  WHEN NEW.state='ready' AND EXISTS(
    SELECT 1 FROM source_exclusions WHERE source_key=NEW.source_key)
  BEGIN
    UPDATE outputs SET state='skipped' WHERE source_key=NEW.source_key AND variant=NEW.variant;
  END""",
    """CREATE TRIGGER IF NOT EXISTS excluded_output_ready_insert
  AFTER INSERT ON outputs
  WHEN NEW.state='ready' AND EXISTS(
    SELECT 1 FROM source_exclusions WHERE source_key=NEW.source_key)
  BEGIN
    UPDATE outputs SET state='skipped' WHERE source_key=NEW.source_key AND variant=NEW.variant;
  END""",
    # Eligibility no longer exempts sources that already reached five accepted
    # rows: durable successes beyond five count toward the success-rate numerator.
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
      (SELECT COUNT(*) n FROM outputs
       WHERE source_key=NEW.source_key AND state='accepted') q
    WHERE p.id=1 AND a.n>=p.minimum_attempts AND q.n<a.n*p.minimum_rate;
    UPDATE outputs SET state='skipped'
      WHERE source_key=NEW.source_key AND state='ready'
        AND EXISTS(SELECT 1 FROM source_exclusions WHERE source_key=NEW.source_key);
  END""",
)


def apply_pair_budget_schema(db: sqlite3.Connection) -> None:
    # Ledgers created by an earlier pair-budget candidate lack the durable
    # seed-allocation column; add it before the index statements and without
    # touching historical rows (they keep NULL and remain fully readable).
    existing = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
    if existing and "allocated_seed" not in existing:
        db.execute("ALTER TABLE attempts ADD COLUMN allocated_seed INTEGER")
    for statement in PAIR_BUDGET_SCHEMA_STATEMENTS:
        db.execute(statement)


class QuotaLedger:
    """Single-host fixed pair-budget coordination; never on NAS/CIFS.

    Each object:action pair owes exactly five accepted rows per originally
    admitted training source, independent of source eligibility. Obligations
    are claimable ``outputs`` rows; excluded or exhausted donor work is
    reassigned by creating further variants (index >= 5) on eligible donors of
    the SAME pair, so ``accepted + ready + leased`` never exceeds the pair
    target and no pair can be overpublished. Legacy per-source-slot ledgers
    must be migrated explicitly first; opening one here fails loudly.
    """
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.row_factory = sqlite3.Row
        state = ledger_schema_state(self.db)
        if state == 'legacy':
            self.db.close()
            raise LegacyLedgerError(
                f'{path} still uses the legacy per-source-slot schema; run the explicit '
                'migration first: python -m tools.run_manorl_synthesis_campaign migrate '
                '--campaign <campaign-dir>')
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        apply_pair_budget_schema(self.db)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO settings VALUES('ledger_schema',?)", (PAIR_BUDGET_CONTRACT,))
        row = self.db.execute("SELECT value FROM settings WHERE key='ledger_schema'").fetchone()
        if row is None or row[0] != PAIR_BUDGET_CONTRACT:
            raise RuntimeError('unknown pair-budget ledger schema contract')

    def success_policy(self) -> dict | None:
        row=self.db.execute('SELECT minimum_rate,minimum_attempts FROM source_success_policy WHERE id=1').fetchone()
        if row is None:return None
        return {'contract':SUCCESS_FILTER_CONTRACT,'minimum_rate':row['minimum_rate'],
                'minimum_attempts':row['minimum_attempts'],
                'denominator':'finished_generation_attempts',
                'numerator':'durably_accepted_source_rows',
                'accepted_ceiling_exemption':False}

    def configure_success_filter(self, *, minimum_rate: float = .25, minimum_attempts: int = 20) -> int:
        """Activate a persistent per-source donor filter; targets never shrink.

        Finished attempts include sampler/geometry rejections and solved GPU
        candidates. Abandoned/in-flight attempts have no finished timestamp and
        do not count. Sources strictly below the rate at the attempt floor stop
        being donors; their accepted rows and pair credit are preserved and the
        obligations they still owed are reassigned to same-pair donors.
        """
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
            # No accepted>=5 exemption: five-row sources remain evaluable and
            # their later successes still count toward the numerator.
            self.db.execute('''INSERT OR IGNORE INTO source_exclusions
                WITH finished AS (
                  SELECT source_key,COUNT(*) n FROM attempts WHERE finished IS NOT NULL GROUP BY source_key),
                accepted AS (
                  SELECT source_key,COUNT(*) n FROM outputs WHERE state='accepted' GROUP BY source_key)
                SELECT f.source_key,?,f.n,COALESCE(a.n,0),?,?
                FROM finished f LEFT JOIN accepted a USING(source_key)
                WHERE f.n>=? AND COALESCE(a.n,0)<f.n*?''',
                (time.time(),minimum_rate,minimum_attempts,minimum_attempts,minimum_rate))
            added=self.db.total_changes-before
            self.db.execute("UPDATE outputs SET state='skipped' WHERE state='ready' "
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
        pairs: dict[str,int] = {}
        with self.db:
            previous = self.db.execute("SELECT value FROM settings WHERE key='config'").fetchone()
            if previous and previous[0] != digest:
                raise ValueError("campaign configuration changed; use a new campaign directory")
            self.db.execute("INSERT OR IGNORE INTO settings VALUES('config',?)", (digest,))
            self.db.execute("INSERT OR REPLACE INTO settings VALUES('campaign_config',?)",
                            (json.dumps(config, sort_keys=True),))
            for source in sources:
                key, group, pair = source["key"], source["group_key"], source.get("pair")
                if not isinstance(pair, str) or ":" not in pair:
                    raise ValueError("pair-budget sources require an explicit object:action pair")
                encoded = json.dumps(source, sort_keys=True)
                old = self.db.execute("SELECT spec FROM sources WHERE key=?", (key,)).fetchone()
                if old and old[0] != encoded:
                    raise ValueError("source/checkpoint provenance changed within campaign")
                # Explicit column names: a migrated legacy table appends `pair`
                # after `spec`, unlike the fresh schema's column order.
                self.db.execute("INSERT OR IGNORE INTO sources(key,group_key,pair,spec) VALUES(?,?,?,?)",
                                (key, group, pair, encoded))
                self.db.executemany("INSERT OR IGNORE INTO outputs(pair,source_key,variant) VALUES(?,?,?)",
                                    [(pair, key, i) for i in range(5)])
                pairs[pair] = pairs.get(pair, 0) + 1
            # Targets are fixed at 5x the pair's originally admitted source count,
            # computed from the table (old + newly appended sources), never from
            # the admission batch: append-only when sources arrive, immutable
            # otherwise, and never modified by eligibility decisions.
            for pair in sorted(pairs):
                total = int(self.db.execute("SELECT COUNT(*) FROM sources WHERE pair=?", (pair,)).fetchone()[0])
                row = self.db.execute("SELECT original_sources,target_rows FROM pair_targets WHERE pair=?", (pair,)).fetchone()
                if row is None:
                    self.db.execute("INSERT INTO pair_targets VALUES(?,?,?)", (pair, total, 5*total))
                elif total > row["original_sources"]:
                    self.db.execute("UPDATE pair_targets SET original_sources=?,target_rows=? WHERE pair=?",
                                    (total, 5*total, pair))
                elif 5*total != row["target_rows"]:
                    raise ValueError("pair output target is immutable for already admitted sources")

    def _allocate_attempt_seed(self, row, attempt: int, token: str) -> dict:
        """Durably register this attempt's seed inside the claim transaction.

        Variants 0-4 keep the historical derivation and store NULL: legacy
        31-bit seeds may legitimately repeat across sources, so uniqueness is
        enforced only for the new int64 namespace. Extra variants allocate a
        candidate from that namespace and resolve collisions deterministically
        against the partial UNIQUE index until the insert succeeds.
        """
        config = self.seed_config
        if row["variant"] < config.quota:
            seed = attempt_seed(row["source_key"], row["variant"], attempt, config)
            self.db.execute(
                "INSERT INTO attempts(token,source_key,slot,attempt,started) VALUES(?,?,?,?,?)",
                (token, row["source_key"], row["variant"], attempt, time.time()))
            return {"seed": seed}
        candidate = attempt_seed(row["source_key"], row["variant"], attempt, config)
        for retry in range(64):
            try:
                self.db.execute(
                    "INSERT INTO attempts(token,source_key,slot,attempt,started,allocated_seed) VALUES(?,?,?,?,?,?)",
                    (token, row["source_key"], row["variant"], attempt, time.time(), candidate))
                return {"seed": candidate}
            except sqlite3.IntegrityError:
                if self.db.execute(
                    "SELECT 1 FROM attempts WHERE allocated_seed=?", (candidate,)).fetchone() is None:
                    raise  # not a seed collision; do not mask other failures
                candidate = _collision_candidate(row["source_key"], row["variant"], attempt, config, retry + 1)
        raise RuntimeError("could not allocate a unique extra-variant seed after 64 collisions")

    @property
    def seed_config(self) -> PrefixConfig:
        """The campaign's prefix sampling contract, for deterministic seeds."""
        row = self.db.execute("SELECT value FROM settings WHERE key='campaign_config'").fetchone()
        if row is not None:
            stored = json.loads(row[0])
            if 'prefix' in stored:
                return PrefixConfig(**stored['prefix'])
        return PrefixConfig()

    def _ready_rows(self, group_key: str, count: int) -> list:
        return self.db.execute("""SELECT s.spec,q.source_key,q.pair,q.variant,q.attempts
            FROM outputs q JOIN sources s ON s.key=q.source_key
            WHERE s.group_key=? AND q.state='ready'
            ORDER BY q.attempts,q.variant,q.source_key LIMIT ?""", (group_key, count)).fetchall()

    def _grow_group_obligations(self, group_key: str, needed: int) -> int:
        """Reassign retired obligations to eligible same-pair donors.

        Only the deficit below the pair target is created, so a pair can never
        hold more claimable obligations than its fixed budget; cross-pair
        borrowing is impossible because donors come from the same pair.
        """
        created = 0
        for row in self.db.execute("SELECT DISTINCT pair FROM sources WHERE group_key=?", (group_key,)).fetchall():
            if needed <= 0:
                break
            pair = row["pair"]
            target = self.db.execute("SELECT target_rows FROM pair_targets WHERE pair=?", (pair,)).fetchone()
            if target is None:
                continue
            outstanding = self.db.execute(
                "SELECT COUNT(*) FROM outputs WHERE pair=? AND state IN ('accepted','ready','leased')",
                (pair,)).fetchone()[0]
            deficit = int(target["target_rows"]) - int(outstanding)
            if deficit <= 0:
                continue
            donors = self.db.execute("""SELECT s.key FROM sources s
                WHERE s.pair=? AND NOT EXISTS(
                    SELECT 1 FROM source_exclusions e WHERE e.source_key=s.key)
                ORDER BY (SELECT COUNT(*) FROM outputs o WHERE o.source_key=s.key), s.key""",
                (pair,)).fetchall()
            if not donors:
                continue  # blocked pair: deficit without eligible donors
            for i in range(min(deficit, needed)):
                donor = donors[i % len(donors)]["key"]
                variant = self.db.execute(
                    "SELECT COALESCE(MAX(variant)+1,0) FROM outputs WHERE source_key=?", (donor,)).fetchone()[0]
                self.db.execute("INSERT INTO outputs(pair,source_key,variant) VALUES(?,?,?)", (pair, donor, variant))
                created += 1
                needed -= 1
        return created

    def claim(self, owner: str, group_key: str, count: int) -> list[dict]:
        if count < 1:
            raise ValueError("batch size must be positive")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            rows = self._ready_rows(group_key, count)
            if len(rows) < count:
                self._grow_group_obligations(group_key, count - len(rows))
                rows = self._ready_rows(group_key, count)
            claims = []
            for row in rows:
                token = uuid.uuid4().hex
                attempt = row["attempts"] + 1
                self.db.execute(
                    "UPDATE outputs SET state='leased',owner=?,token=?,attempts=? WHERE source_key=? AND variant=?",
                    (owner, token, attempt, row["source_key"], row["variant"]))
                claims.append({"source": json.loads(row["spec"]), "slot": row["variant"],
                               "pair": row["pair"], "attempt": attempt, "token": token, "owner": owner,
                               **self._allocate_attempt_seed(row, attempt, token)})
            self.db.commit()
            return claims
        except BaseException:
            self.db.rollback(); raise

    def finish(self, claim: dict, diagnostic: dict, *, row_uuid: str | None = None, shard: str | None = None) -> None:
        if bool(row_uuid) != bool(shard):
            raise ValueError("accepted rows require both UUID and durable shard")
        variant = int(claim["slot"])
        with self.db:
            row = self.db.execute("SELECT state,token,row_uuid,shard FROM outputs WHERE source_key=? AND variant=?",
                                  (claim["source"]["key"], variant)).fetchone()
            if row is not None and row["state"] == "accepted" and row["row_uuid"] == row_uuid and row["shard"] == shard:
                return
            if row is None or row["state"] != "leased" or row["token"] != claim["token"]:
                raise RuntimeError("stale/foreign claim cannot finish a slot")
            self.db.execute(
                "UPDATE outputs SET state=?,owner=NULL,token=NULL,error=?,row_uuid=?,shard=? WHERE source_key=? AND variant=?",
                ("accepted" if row_uuid else "ready", json.dumps(diagnostic, sort_keys=True), row_uuid, shard,
                 claim["source"]["key"], variant))
            self.db.execute("UPDATE attempts SET finished=?,diagnostic=? WHERE token=?",
                            (time.time(), json.dumps(diagnostic, sort_keys=True), claim["token"]))

    def release_owner(self, owner: str) -> int:
        # Caller is responsible for proving that owner PID+start-time is dead.
        with self.db:
            result = self.db.execute(
                "UPDATE outputs SET state='ready',owner=NULL,token=NULL WHERE state='leased' AND owner=?", (owner,))
            return result.rowcount

    def heartbeat(self, owner: str, details: dict) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO workers VALUES(?,?,?)", (owner, time.time(), json.dumps(details, sort_keys=True)))

    def summary(self) -> dict:
        # Exclusion decisions and in-flight publications can change together;
        # report all counters from one snapshot, not several unrelated reads.
        own_snapshot=not self.db.in_transaction
        if own_snapshot:self.db.execute('BEGIN')
        try:return self._summary_snapshot()
        finally:
            if own_snapshot:self.db.rollback()

    def _summary_snapshot(self) -> dict:
        counts = dict(self.db.execute("SELECT state,COUNT(*) FROM outputs GROUP BY state").fetchall())
        sources = int(self.db.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
        accepted = counts.get("accepted", 0)
        target = int(self.db.execute("SELECT COALESCE(SUM(target_rows),0) FROM pair_targets").fetchone()[0])
        excluded = int(self.db.execute('SELECT COUNT(*) FROM source_exclusions').fetchone()[0])
        excluded_accepted = int(self.db.execute('''SELECT COUNT(*) FROM outputs o
            JOIN source_exclusions e ON e.source_key=o.source_key WHERE o.state='accepted' ''').fetchone()[0])
        pairs = []
        for row in self.db.execute("""
            SELECT t.pair,t.target_rows,
              (SELECT COUNT(*) FROM outputs o WHERE o.pair=t.pair AND o.state='accepted') accepted,
              (SELECT COUNT(*) FROM outputs o WHERE o.pair=t.pair
                 AND o.state IN ('accepted','ready','leased')) outstanding,
              (SELECT COUNT(*) FROM sources s WHERE s.pair=t.pair AND NOT EXISTS(
                 SELECT 1 FROM source_exclusions e WHERE e.source_key=s.key)) donors
            FROM pair_targets t ORDER BY t.pair"""):
            remaining = max(0, int(row["target_rows"]) - int(row["accepted"]))
            pairs.append({"pair": row["pair"], "target": int(row["target_rows"]), "accepted": int(row["accepted"]),
                          "remaining": remaining, "donors": int(row["donors"]), "met": remaining == 0,
                          "blocked": remaining > 0 and int(row["donors"]) == 0,
                          "runnable": remaining > 0 and int(row["donors"]) > 0})
        blocked = [p["pair"] for p in pairs if p["blocked"]]
        met = sum(p["met"] for p in pairs)
        remaining = max(0, target - accepted)
        # Debt the fleet can still generate is reported separately from debt
        # stranded on no-donor pairs: blocked debt cannot shrink without new
        # donors, while runnable debt is ordinary outstanding work.
        blocked_debt = sum(p["remaining"] for p in pairs if p["blocked"])
        return {"contract": PAIR_BUDGET_CONTRACT, "sources": sources, "pairs": len(pairs),
                "accepted": accepted, "target": target, "original_target": target,
                "remaining": remaining, "pair_debt": remaining, "states": counts,
                "blocked_pair_debt": blocked_debt, "runnable_pair_debt": remaining - blocked_debt,
                "skipped_slots": counts.get('skipped', 0),
                "sources_excluded": excluded, "sources_retained": sources - excluded,
                "retained_accepted": accepted - excluded_accepted, "excluded_accepted": excluded_accepted,
                "pairs_met": met, "pairs_runnable": len(pairs) - met - len(blocked),
                "pairs_blocked": len(blocked), "blocked_pairs": blocked,
                "source_success_policy": self.success_policy(),
                "attempts": int(self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]),
                "complete": bool(pairs) and met == len(pairs)
                            and counts.get('leased', 0) == 0 and counts.get('ready', 0) == 0}

    def pair_status(self) -> list[dict]:
        """Per-pair fixed target, accepted, remaining and blocked/runnable state."""
        own_snapshot = not self.db.in_transaction
        if own_snapshot:
            self.db.execute('BEGIN')
        try:
            rows = []
            for row in self.db.execute("""
                SELECT t.pair,t.target_rows,
                  (SELECT COUNT(*) FROM outputs o WHERE o.pair=t.pair AND o.state='accepted') accepted,
                  (SELECT COUNT(*) FROM sources s WHERE s.pair=t.pair AND NOT EXISTS(
                     SELECT 1 FROM source_exclusions e WHERE e.source_key=s.key)) donors
                FROM pair_targets t ORDER BY t.pair"""):
                remaining = max(0, int(row["target_rows"]) - int(row["accepted"]))
                rows.append({"pair": row["pair"], "target": int(row["target_rows"]),
                             "accepted": int(row["accepted"]), "remaining": remaining,
                             "eligible_donors": int(row["donors"]),
                             "met": remaining == 0,
                             "blocked": remaining > 0 and int(row["donors"]) == 0})
            return rows
        finally:
            if own_snapshot:
                self.db.rollback()

    def _creatable_by_group(self) -> dict:
        creatable = {}
        for row in self.db.execute("""
            SELECT DISTINCT s.group_key g,t.pair,t.target_rows,
              (SELECT COUNT(*) FROM outputs o WHERE o.pair=t.pair
                 AND o.state IN ('accepted','ready','leased')) outstanding,
              (SELECT COUNT(*) FROM sources d WHERE d.pair=t.pair AND NOT EXISTS(
                 SELECT 1 FROM source_exclusions e WHERE e.source_key=d.key)) donors
            FROM pair_targets t JOIN sources s ON s.pair=t.pair"""):
            deficit = max(0, int(row["target_rows"]) - int(row["outstanding"]))
            if deficit > 0 and int(row["donors"]) > 0:
                creatable[row["g"]] = creatable.get(row["g"], 0) + deficit
        return creatable

    def groups(self) -> list[dict]:
        rows = {}
        for row in self.db.execute("""SELECT s.group_key g,
            SUM(o.state='ready') ready,SUM(o.state='leased') leased,
            MIN(CASE WHEN o.state='ready' THEN o.attempts END) min_attempts
            FROM outputs o JOIN sources s ON s.key=o.source_key
            WHERE o.state IN ('ready','leased') GROUP BY s.group_key"""):
            rows[row["g"]] = {"ready": int(row["ready"] or 0), "leased": int(row["leased"] or 0),
                              "min_attempts": row["min_attempts"]}
        creatable = self._creatable_by_group()
        groups = []
        for group, info in rows.items():
            extra = creatable.get(group, 0)
            groups.append({"group_key": group, "remaining": info["ready"] + info["leased"] + extra,
                           "ready": info["ready"] + extra, "leased": info["leased"],
                           "min_attempts": 0 if info["min_attempts"] is None else int(info["min_attempts"])})
        for group, extra in creatable.items():
            if group not in rows and extra > 0:
                groups.append({"group_key": group, "remaining": extra, "ready": extra, "leased": 0, "min_attempts": 0})
        return sorted(groups, key=lambda g: (g["min_attempts"], g["group_key"]))

    def close(self) -> None:
        self.db.close()
