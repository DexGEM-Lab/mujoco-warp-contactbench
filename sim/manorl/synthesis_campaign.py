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
    if not 0 <= slot < config.quota or attempt < 1:
        raise ValueError("slot/attempt is outside the campaign contract")
    return int(canonical_hash([config.seed, key, slot, attempt, asdict(config)])[:8], 16) % (2**31 - 1)


def augment_source_prefix(
    source: ReferenceTrajectory, *, slot: int, attempt: int,
    joint_lower: np.ndarray, joint_upper: np.ndarray,
    config: PrefixConfig = PrefixConfig(),
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
    seed = attempt_seed(key, slot, attempt, config)
    rng = np.random.default_rng(seed)
    width = (config.maximum_offset_m - config.minimum_offset_m) / config.quota
    low = config.minimum_offset_m + slot * width
    high = low + width
    radius = float(rng.uniform(low, high))
    # A per-source permutation keeps all five azimuth strata represented without
    # correlating large radii with one direction across the corpus.
    order = np.random.default_rng(int(key[:16], 16) ^ config.seed).permutation(config.quota)
    azimuth = 2 * math.pi * (int(order[slot]) + rng.random()) / config.quota
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
        "azimuth_stratum": int(order[slot]), "wrist_rotation_deg": math.degrees(angle),
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


class QuotaLedger:
    """Single-host SQLite coordination; never place this database on NAS/CIFS.

    Leases do not expire solely by wall time: the supervisor must establish that
    the owning process died before calling release_owner. Accepted slots cannot
    be claimed again, even if a worker restarts or a shard is replayed.
    """
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sources(key TEXT PRIMARY KEY, group_key TEXT NOT NULL, spec TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS slots(
          source_key TEXT NOT NULL REFERENCES sources(key), slot INTEGER NOT NULL CHECK(slot>=0 AND slot<5),
          state TEXT NOT NULL DEFAULT 'ready', attempts INTEGER NOT NULL DEFAULT 0,
          owner TEXT, token TEXT, error TEXT, row_uuid TEXT UNIQUE, shard TEXT,
          PRIMARY KEY(source_key,slot));
        CREATE INDEX IF NOT EXISTS slots_ready ON slots(state,attempts,source_key,slot);
        CREATE TABLE IF NOT EXISTS attempts(
          token TEXT PRIMARY KEY, source_key TEXT, slot INTEGER, attempt INTEGER,
          started REAL, finished REAL, diagnostic TEXT);
        CREATE TABLE IF NOT EXISTS workers(owner TEXT PRIMARY KEY, heartbeat REAL, details TEXT);
        """)

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
        # Caller is responsible for proving that owner PID+start-time is dead.
        with self.db:
            result = self.db.execute("UPDATE slots SET state='ready',owner=NULL,token=NULL WHERE state='leased' AND owner=?", (owner,))
            return result.rowcount

    def heartbeat(self, owner: str, details: dict) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO workers VALUES(?,?,?)", (owner, time.time(), json.dumps(details, sort_keys=True)))

    def summary(self) -> dict:
        counts = dict(self.db.execute("SELECT state,COUNT(*) FROM slots GROUP BY state").fetchall())
        sources = int(self.db.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
        complete = int(self.db.execute("SELECT COUNT(*) FROM (SELECT source_key FROM slots WHERE state='accepted' GROUP BY source_key HAVING COUNT(*)=5)").fetchone()[0])
        accepted = counts.get("accepted", 0)
        return {"sources": sources, "sources_complete": complete, "accepted": accepted,
                "target": 5*sources, "remaining": 5*sources-accepted, "states": counts,
                "attempts": int(self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]),
                "complete": sources > 0 and sources == complete and accepted == 5*sources}

    def groups(self) -> list[dict]:
        return [dict(row) for row in self.db.execute("""SELECT s.group_key,COUNT(*) remaining,MIN(q.attempts) min_attempts
            FROM slots q JOIN sources s ON s.key=q.source_key WHERE q.state!='accepted'
            GROUP BY s.group_key ORDER BY min_attempts,group_key""")]

    def close(self) -> None:
        self.db.close()
