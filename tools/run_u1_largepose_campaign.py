#!/usr/bin/env python3
"""Large-pose U1 augmentation runner for actions 003/006/007/009.

Historical approach-prefix mechanism: prepend a quintic offset prefix to the
accepted U1 parent, then append the complete parent target byte-exactly.
Each child is freshly integrated from its perturbed frame0 and independently
replayed from frozen controls in a separate process.

5/10/15 cm translation × 8 XYZ octants × 6 signed 30° wrist axes = 144 base
cells, plus 16 deterministic extra spatial draws = 160 slots/action. Each slot
has 16 fixed same-cell reserve candidates; no silent radius/orientation shrink.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.manorl.u1_campaign import Ledger, array_sha, child_uuid, digest, file_sha, write_json
from tools.u1_campaign_registry import load_parent
from tools.u1_interactive_session import Session

VERSION = 'u1-four-action-largepose-v1'
ACTIONS = ('003', '006', '007', '009')
RADII_MM = (50, 100, 150)
ROT_AXES = (0, 1, 2)  # floating-root intrinsic XYZ: roll, pitch, yaw
ROT_DEG = 30.0
PREFIX_FRAMES = 120  # 1.0 s at 120 Hz; peak residual speed for 15 cm is 0.28 m/s.
RESERVES = 16
SEED = 20260922
SOURCE_REGISTRY = Path('/mnt/nas-222-project/mocap_v2/lance_datasets/.manorl_u1_5x160_20260921_staging/contracts/registry/registry.json')
SETTING = 'c6db552f105d4de493880a52518772dbacad9496b105d2f4e43e87a907e692cd'


def build_plan(seed=SEED):
    """160 slots: 144 base cells + 16 extra spatial draws."""
    slots = []
    for radius, signs, (axis, sign) in itertools.product(
            RADII_MM, itertools.product((-1, 1), repeat=3),
            itertools.product(ROT_AXES, (1, -1))):
        sid = len(slots)
        rng = np.random.default_rng(np.random.SeedSequence([seed, sid]))
        candidates = []
        for ordinal in range(RESERVES):
            direction = np.asarray(signs) * (np.ones(3) if ordinal == 0 else rng.uniform(.25, 1.75, 3))
            trans = radius / 1000.0 * direction / np.linalg.norm(direction)
            rot = np.zeros(3); rot[axis] = sign * np.deg2rad(ROT_DEG)
            candidates.append(dict(ordinal=ordinal, delta=np.r_[trans, rot].tolist()))
        slots.append(dict(slot=sid, radius_mm=radius, octant=list(signs),
                          rotation_axis=axis, rotation_sign=sign, candidates=candidates,
                          kind='base'))
    for k in range(16):
        radius = RADII_MM[k % 3]; octant = k // 2
        signs = [1 if octant & (1 << ax) else -1 for ax in range(3)]
        axis = k % 3
        sign = 1 if (k // 3) % 2 == 0 else -1
        sid = len(slots)
        rng = np.random.default_rng(np.random.SeedSequence([seed, sid, 987654321]))
        candidates = []
        for ordinal in range(RESERVES):
            direction = np.asarray(signs) * rng.uniform(.5, 1.5, 3)
            trans = radius / 1000.0 * direction / np.linalg.norm(direction)
            rot = np.zeros(3); rot[axis] = sign * np.deg2rad(ROT_DEG)
            candidates.append(dict(ordinal=ordinal, delta=np.r_[trans, rot].tolist()))
        slots.append(dict(slot=sid, radius_mm=radius, octant=list(signs),
                          rotation_axis=axis, rotation_sign=sign, candidates=candidates,
                          kind='extra'))
    return slots


def prefix_envelope(frames, delta):
    phase = np.arange(frames, dtype=float) / float(frames - 1)
    envelope = 1.0 - 10.0 * phase**3 + 15.0 * phase**4 - 6.0 * phase**5
    return envelope[:, None] * np.asarray(delta, dtype=float)[None, :]


def make_target(base, delta, prefix_frames):
    prefix = np.repeat(base[:1], prefix_frames, axis=0)
    prefix[:, :6] += prefix_envelope(prefix_frames, delta)
    target = np.concatenate([prefix, base], axis=0)
    assert target[prefix_frames:].tobytes() == base.tobytes()
    return target


class LargePoseRuntime:
    """Single-world persistent U1 runtime; complete reset between candidates."""

    def __init__(self, registry, action):
        self.r, self.p, self.I, self.m, self.base, self.teacher = load_parent(registry, action)
        self.action = action

    def _replay(self, target, q0, prefix_frames, folder):
        from sim.manorl.mjx_sim import command_target
        from sim.manorl.start_augmentation import scene_contacts, unsupported_intervals
        from sim.manorl.local_contact_repair import evaluate
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
        arr = dict(self.I.arrays)
        arr['source_object_pos'] = np.concatenate(
            [np.repeat(arr['source_object_pos'][:1], prefix_frames, axis=0), arr['source_object_pos']], axis=0)
        arr['source_object_quat_xyzw'] = np.concatenate(
            [np.repeat(arr['source_object_quat_xyzw'][:1], prefix_frames, axis=0), arr['source_object_quat_xyzw']], axis=0)
        teacher_ext = np.concatenate(
            [np.repeat(self.teacher['qpos'][:1], prefix_frames, axis=0), self.teacher['qpos']], axis=0)
        inp = SimpleNamespace(row_id=self.I.row_id, frames=len(target), hz=120,
                              metrics=self.I.metrics, arrays=arr,
                              initial=dict(qpos=q0, qvel=np.asarray(self.I.initial['qvel']).copy()),
                              manifest=self.I.manifest, provenance=getattr(self.I, 'provenance', None))
        session = Session(inp, self.m, target, {'qpos': teacher_ext}, self.p)
        s = session
        m = self.m
        s.data.qpos.assign(q0[None].astype(np.float32))
        s.data.ctrl.assign(command_target(target[0], q0[:28], *m.jnt_range[:28].T)[None].astype(np.float32))
        s.mw.forward(s.wm, s.data); s.wp.synchronize()
        names = self.p['names']
        adrs = [int(m.joint(n + '_free').qposadr[0]) for n in names]
        q, v, controls, precontact = [], [], [], []
        gaps, radii, heights, pair_force = [], [], [], []
        native_path = folder / 'native_contacts.jsonl'
        high = {'contacts': 0, 'constraints': 0}
        with native_path.open('w') as evidence:
            for f in range(len(target)):
                if f:
                    s.frame += 1
                    control = command_target(target[f], s.data.qpos.numpy()[0, :28], *m.jnt_range[:28].T)
                    s.data.ctrl.assign(control[None].astype(np.float32))
                    for substep in range(4):
                        s.mw.step(s.wm, s.data); s.wp.synchronize()
                        nsub = int(s.data.nacon.numpy().ravel()[0]); efcsub = int(np.max(s.data.nefc.numpy()))
                        high['contacts'] = max(high['contacts'], nsub); high['constraints'] = max(high['constraints'], efcsub)
                        if nsub >= 1024 or efcsub >= 4096:
                            raise ValueError(f'native capacity reached at frame{f} substep{substep}')
                pose = s.data.qpos.numpy()[0].copy(); vel = s.data.qvel.numpy()[0].copy()
                if not np.isfinite(pose).all() or not np.isfinite(vel).all():
                    raise ValueError('nonfinite state')
                q.append(pose); v.append(vel); controls.append(s.data.ctrl.numpy()[0].copy())
                n = int(s.data.nacon.numpy().ravel()[0]); nefc = int(np.max(s.data.nefc.numpy()))
                high['contacts'] = max(high['contacts'], n); high['constraints'] = max(high['constraints'], nefc)
                if n >= 1024 or nefc >= 4096:
                    raise ValueError('native capacity reached')
                pairs = s.data.contact.geom.numpy()[:n]
                ids = s.wp.array(np.arange(n, dtype=np.int32), dtype=s.wp.int32)
                forces = s.wp.zeros(n, dtype=s.wp.spatial_vector)
                if n:
                    s.mw.contact_force(s.wm, s.data, ids, False, forces)
                F = forces.numpy()
                if self.action == '006':
                    d = s.cpu; d.qpos[:] = pose; d.qvel[:] = vel; s.mj.mj_forward(m, d)
                    tg = m.geom('thumb_ip_collision').id; ig = m.geom('index_dip_collision').id
                    gaps.append(float(s.mj.mj_geomDistance(m, d, tg, ig, .15, np.zeros(6))))
                    ob = m.body('pitcherbase').id; bb = m.body('bowl').id
                    sp = d.xpos[ob] + d.xmat[ob].reshape(3, 3) @ np.array([-.081061, -.0049836, .08246349])
                    radii.append(float(np.linalg.norm(sp[:2] - d.xpos[bb, :2])))
                    heights.append(float(sp[2] - d.xpos[bb, 2]))
                    mask = ((pairs[:, 0] == tg) & (pairs[:, 1] == ig)) | ((pairs[:, 0] == ig) & (pairs[:, 1] == tg))
                    pair_force.append(float(F[mask, 0].sum()))
                evidence.write(json.dumps(dict(frame=f, geom_pairs=pairs.tolist(),
                    force_contact_frame=F.tolist(), position=s.data.contact.pos.numpy()[:n].tolist(),
                    contact_frame=s.data.contact.frame.numpy()[:n].tolist(),
                    friction=s.data.contact.friction.numpy()[:n].tolist(),
                    dimension=s.data.contact.dim.numpy()[:n].tolist(),
                    efc_address=s.data.contact.efc_address.numpy()[:n].tolist(),
                    worldid=s.data.contact.worldid.numpy()[:n].tolist())) + '\n')
                if f < prefix_frames and scene_contacts(m, pose, vel, names):
                    precontact.append(f)
        q = np.asarray(q); v = np.asarray(v); controls = np.asarray(controls)
        np.save(folder / 'target.npy', controls)
        np.savez_compressed(folder / 'trace.npz', qpos=q, qvel=v, ctrl=controls)
        trace = dict(qpos=q, qvel=v, desired=target,
                     actual_object_pos=np.stack([q[:, a:a + 3] for a in adrs], 1),
                     actual_object_quat_xyzw=np.stack([q[:, a + 3:a + 7][:, [1, 2, 3, 0]] for a in adrs], 1))
        physical, diag = evaluate(m, inp, trace)
        a = diag['arrays']
        gates = dict(physical['gates'])
        error = controls.astype(float) - target
        error[:, 3:6] = np.arctan2(np.sin(error[:, 3:6]), np.cos(error[:, 3:6]))
        gates.update(no_prefix_scene_contact=not precontact,
                     no_uncontrolled_flight=not unsupported_intervals(a, 120),
                     controls_canonical=float(np.max(np.abs(error))) < 1e-6,
                     initial_pose=bool(np.allclose(q[0], q0, atol=1e-7, rtol=0)),
                     initial_velocity=bool(np.allclose(v[0], self.I.initial['qvel'], atol=1e-7, rtol=0)),
                     executed_suffix_exact=controls[prefix_frames:].tobytes() == self.base.astype(np.float32).tobytes())
        if self.action == '009':
            gates['terminal_100ms_unsupported_multifinger'] = bool(
                np.all(a['finger_ray_count'][-12:] >= 2) and not np.any(a['has_support'][-12:]))
        if self.action == '006':
            shift = prefix_frames
            f = np.arange(len(q))
            pour = (f >= 560 + shift) & (f <= 840 + shift) & (a['world_tilt_deg'] >= 70)
            da = int(m.joint('pitcherbase_free').dofadr[0])
            # Per-frame native evidence already recorded; rebuild only the 006-specific
            # geometric proxies needed for release ordering.
            tg = m.geom('thumb_ip_collision').id; ig = m.geom('index_dip_collision').id
            gaps, radii, heights, pair_force = [], [], [], []
            for f2 in range(len(q)):
                d = s.cpu
                d.qpos[:] = q[f2]; d.qvel[:] = v[f2]; s.mj.mj_forward(m, d)
                gaps.append(float(s.mj.mj_geomDistance(m, d, tg, ig, .15, np.zeros(6))))
                ob = m.body('pitcherbase').id; bb = m.body('bowl').id
                sp = d.xpos[ob] + d.xmat[ob].reshape(3, 3) @ np.array([-.081061, -.0049836, .08246349])
                radii.append(float(np.linalg.norm(sp[:2] - d.xpos[bb, :2]))); heights.append(float(sp[2] - d.xpos[bb, 2]))
            gaps = np.asarray(gaps); radii = np.asarray(radii); heights = np.asarray(heights); pair_force = np.asarray(pair_force)
            gates.update(pour_present=bool(np.any(pour)),
                         final_source_orientation_under15deg=physical['object_orientation_final_deg'] < 15,
                         pour_spout_proxy_inside_bowl=bool(np.any(pour) and np.all(radii[pour] < .077101396)),
                         pour_above_bowl=bool(np.any(pour) and np.all(heights[pour] > .025)),
                         distal_surfaces_touch_during_pour=bool(np.any(pour) and np.mean(gaps[pour] < .0005) > .95),
                         native_tip_pair_force_during_pour=bool(np.any(pour & (f % 12 == 0)) and np.all(pair_force[pour & (f % 12 == 0)] > .1)),
                         settled_final_object=bool(np.linalg.norm(v[-1, da:da + 6]) < .01),
                         release_upright=bool(np.max(a['world_tilt_deg'][1130 + shift:]) < 10))
        np.savez_compressed(folder / 'contact_validation.npz', **a)
        write_json(folder / 'geometry_contacts.json', diag['contacts'])
        report = dict(accepted=all(gates.values()), gates=gates, physical=physical,
                      prefix_frames=prefix_frames, high_water=high,
                      contract=s.contract, semantics=self.p['semantics'],
                      pid=os.getpid(), requested_target_sha256=array_sha(target),
                      executed_target_sha256=array_sha(controls),
                      native_contact_capture='native geom pairs, contact-frame wrench, position, frame basis, friction, dimension, efc address and world id')
        write_json(folder / 'result.json', report)
        return report

    def replay(self, delta, prefix_frames, folder, frozen=None):
        folder = Path(folder)
        target = make_target(self.base, delta, prefix_frames)
        if frozen is not None:
            target = np.load(frozen, allow_pickle=False)
        q0 = self.I.initial['qpos'].copy(); q0[:6] += delta
        return self._replay(target, q0, prefix_frames, folder)


def run_action(args):
    slots = build_plan()
    staging = args.staging
    staging.mkdir(parents=True, exist_ok=True)
    ledger_path = staging / f'{args.action}.jsonl'
    identity = digest(dict(version=VERSION, action=args.action,
                           registry_sha=file_sha(SOURCE_REGISTRY),
                           implementation={f: file_sha(ROOT / f) for f in (
                               'sim/manorl/u1_largepose.py', 'tools/run_u1_largepose_campaign.py')}))
    ledger = Ledger(ledger_path, identity)
    runtime = LargePoseRuntime(SOURCE_REGISTRY, args.action)
    for slot in slots:
        sid = slot['slot']
        if sid in ledger.selected():
            continue
        for cand in slot['candidates']:
            delta = np.asarray(cand['delta'], dtype=float)
            uid = child_uuid(dict(version=VERSION, registry=file_sha(SOURCE_REGISTRY)),
                             args.action, digest(slot), sid, cand)
            if any(row.get('uuid') == uid for row in ledger.rows):
                continue
            folder = staging / args.action / uid
            ledger.append(uuid=uid, slot=sid, candidate=cand, status='started')
            try:
                first = runtime.replay(delta, PREFIX_FRAMES, folder / 'first')
                if not first['accepted']:
                    raise ValueError('first-pass gates failed')
                cmd = [sys.executable, '-m', 'tools.run_u1_largepose_campaign', 'second-pass',
                       '--action', args.action, '--staging', str(staging),
                       '--uuid', uid, '--frozen', str(folder / 'first/target.npy')]
                with (folder / 'second-process.log').open('w') as log:
                    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
                second = json.loads((folder / 'second/result.json').read_text())
                if not second['accepted'] or second['pid'] == first['pid']:
                    raise ValueError('independent gates failed')
                artifacts = {str(x.resolve()): file_sha(x) for x in folder.rglob('*') if x.is_file()}
                ledger.append(uuid=uid, slot=sid, status='selected', artifacts=artifacts)
                break
            except Exception as e:
                ledger.append(uuid=uid, slot=sid, status='rejected', reason=repr(e))
        if sid not in ledger.selected():
            ledger.append(slot=sid, status='exhausted')
            raise RuntimeError(f'exhausted slot {sid}')
    print(json.dumps(dict(action=args.action, selected=len(ledger.selected()), quota=160)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run-action', 'second-pass', 'status'])
    parser.add_argument('--action', choices=ACTIONS, required=True)
    parser.add_argument('--staging', type=Path, required=True)
    parser.add_argument('--uuid', default=None)
    parser.add_argument('--frozen', type=Path, default=None)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.mode == 'status':
        ledger_path = args.staging / f'{args.action}.jsonl'
        rows = [json.loads(x) for x in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
        print(json.dumps(dict(selected=len({x['slot'] for x in rows if x['status'] == 'selected'}),
                              quota=160,
                              rejected=sum(1 for x in rows if x['status'] == 'rejected'),
                              exhausted=sum(1 for x in rows if x['status'] == 'exhausted'))))
    elif args.mode == 'run-action':
        if not args.execute:
            raise ValueError('production requires --execute')
        run_action(args)
    elif args.mode == 'second-pass':
        if not args.uuid or not args.frozen:
            raise ValueError('second-pass requires --uuid and --frozen')
        runtime = LargePoseRuntime(SOURCE_REGISTRY, args.action)
        folder = args.staging / args.action / args.uuid / 'second'
        # Reconstruct the delta from the first-pass manifest to keep frame0 consistent.
        first_folder = args.staging / args.action / args.uuid / 'first'
        first_report = json.loads((first_folder / 'result.json').read_text())
        # The first pass stored its delta inside target reconstruction; read the
        # candidate from the ledger entry for this UUID.
        ledger_path = args.staging / f'{args.action}.jsonl'
        rows = [json.loads(x) for x in ledger_path.read_text().splitlines()]
        started = next(r for r in rows if r.get('uuid') == args.uuid and r['status'] == 'started')
        delta = np.asarray(started['candidate']['delta'], dtype=float)
        runtime.replay(delta, PREFIX_FRAMES, folder, frozen=args.frozen)


if __name__ == '__main__':
    main()
