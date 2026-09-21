"""Immutable, balanced U1 campaign identities and crash-conservative ledgers."""
from pathlib import Path
import hashlib
import itertools
import json
import os
import uuid
import numpy as np

VERSION = 'u1-5x160-v1'
SETTING = 'c6db552f105d4de493880a52518772dbacad9496b105d2f4e43e87a907e692cd'
ACTIONS = ('003', '005', '006', '007', '009')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

def plan(registry_digest, seed=20260921, reserves=16):
    if not 1 <= reserves <= 64: raise ValueError('finite reserve count must be 1..64')
    slots = []
    orientations = [(0, .5), (0, -.5), (1, .5), (1, -.5)]
    for radius, signs, (axis, degrees) in itertools.product((5,10,15,20,30), itertools.product((-1,1), repeat=3), orientations):
        slot = len(slots)
        rng = np.random.default_rng(np.random.SeedSequence([seed, slot]))
        candidates = []
        for ordinal in range(reserves):
            direction = np.asarray(signs) * (np.ones(3) if ordinal == 0 else rng.uniform(.25, 1.75, 3))
            delta = np.zeros(6); delta[:3] = radius/1000 * direction/np.linalg.norm(direction)
            delta[3+axis] = np.deg2rad(degrees)
            candidates.append(dict(ordinal=ordinal, delta=delta.tolist()))
        slots.append(dict(slot=slot, radius_mm=radius, octant=list(signs), candidates=candidates))
    p = dict(version=VERSION, registry_digest=registry_digest, seed=seed, reserves=reserves,
             rotation='additive intrinsic-XYZ joint coordinates, radians', slots=slots)
    return dict(p, digest=digest(p))

def verify_signed(document):
    if document['digest'] != digest({k:v for k,v in document.items() if k != 'digest'}):
        raise ValueError('document digest mismatch')

def perturb(target, qpos, delta, C):
    target = np.asarray(target); delta = np.asarray(delta, dtype=float)
    if target.dtype.kind != 'f' or target.ndim != 2 or target.shape[1] != 28:
        raise ValueError('floating [N,28] target required')
    if not 3 <= C <= len(target) or delta.shape != (6,) or not np.isfinite(delta).all():
        raise ValueError('insufficient taper window or invalid delta')
    u = np.arange(C-1, dtype=float)/(C-1)
    envelope = 1-10*u**3+15*u**4-6*u**5
    out = target.copy(); out[:C-1,:6] += (envelope[:,None]*delta).astype(target.dtype)
    initial = np.asarray(qpos).copy(); initial[:6] += delta
    assert out[C-1:].tobytes() == target[C-1:].tobytes()
    assert out[:,6:].tobytes() == target[:,6:].tobytes()
    return out, initial

def child_uuid(registry, action, plan_digest, slot, candidate):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest(dict(version=VERSION, registry=registry,
        action=action, plan=plan_digest, slot=slot, candidate=candidate))))

class Ledger:
    """Single writer. An interrupted candidate is failed, never rerolled for luck."""
    def __init__(self, path, identity):
        self.path = Path(path); self.identity = identity; self.rows = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                if row['identity'] != identity: raise ValueError('ledger identity mismatch')
                self.rows.append(row)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.reconcile()

    def append(self, **row):
        row['identity'] = self.identity
        with self.path.open('a') as f:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
        self.rows.append(row)

    def reconcile(self):
        latest = {}
        for row in self.rows:
            if 'uuid' in row: latest[row['uuid']] = row
        selected = set()
        for row in list(latest.values()):
            if row['status'] == 'started':
                self.append(uuid=row['uuid'], slot=row['slot'], status='rejected', reason='interrupted; no lucky retry')
            elif row['status'] == 'selected':
                if row['slot'] in selected: raise ValueError('duplicate selected slot')
                selected.add(row['slot'])
                for path, expected in row['artifacts'].items():
                    if file_sha(path) != expected: raise ValueError('selected artifact changed')

    def selected(self):
        return {r['slot']:r for r in self.rows if r['status'] == 'selected'}
