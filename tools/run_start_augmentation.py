#!/usr/bin/env python3
"""Deterministic42-parent positional campaign: plan, independent worlds, ledger."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from sim.manorl.local_contact_repair import MODEL_FIELDS, dump, evaluate, sha
from sim.manorl.start_augmentation import augment, scene_contacts, unsupported_intervals
from tools.pilot_start_augmentation import reconstruct
from tools.repair_local_contacts import bind_model

RADII = (.03, .05, .075, .10)
ROOT = Path(__file__).resolve().parents[1]
CODE = ['sim/manorl/start_augmentation.py', 'sim/manorl/start_batch_dynamics.py',
        'sim/manorl/local_contact_repair.py', 'sim/manorl/local_contact_dynamics.py',
        'tools/pilot_start_augmentation.py', 'tools/repair_local_contacts.py', 'tools/run_start_augmentation.py']


def code_pin():
    return {p: sha(ROOT/p) for p in CODE}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def compile_parent(a, inp, assets=None):
    if assets is None:
        return bind_model(SimpleNamespace(manifest=a.bundle/'asset_manifest.json', asset_root=a.asset_root), inp)
    mj, model = assets.compile_unified_model(object_types=inp.arrays['scene_object_names'].tolist(),
                                             object_collisions=True, physics_timestep=1/480)
    model.opt.cone = mj.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = 100.
    for key in MODEL_FIELDS:
        np.testing.assert_array_equal(getattr(model, key), np.asarray(inp.manifest['native'][key]))
    return assets, model


def merge_frame(inp, baseline_contacts, guard):
    if not np.isfinite(guard) or guard <= 0:
        raise ValueError('guard must be finite and positive')
    indices = np.flatnonzero(baseline_contacts['finger_ray_count'])
    if not len(indices):
        raise ValueError('accepted seed has no measured active contact')
    first = int(indices[0])
    move = min(x['start_frame'] for x in inp.metrics['source_metadata']['object_move'])/100
    C = int(np.floor(min(first/120-guard, max(.30, move-guard))*120+1e-9))
    if not 3 <= C < inp.frames-1:
        raise ValueError('no supported precontact blending interval')
    return C, first


def plan_campaign(a):
    if a.plan.exists():
        raise ValueError('plan already exists; never resample a production plan silently')
    ids = (a.bundle/'accepted_full_pose_ids.txt').read_text().split()
    comparison = json.loads((a.bundle/'comparison.json').read_text())
    if len(ids) != 42 or len(set(ids)) != 42 or set(ids) != {r['id'] for r in comparison['rows'] if r['final120_full_pose']}:
        raise ValueError('expected exactly42 strict seeds')
    if 'B_row035' in ids:
        raise ValueError('functional-only B035 is not an accepted seed')
    assets, parents = None, []
    for number, row in enumerate(sorted(ids)):
        inp, parent, fingers, baseline, provenance = reconstruct(a.bundle, row)
        assets, model = compile_parent(a, inp, assets)
        with np.load(a.bundle/'trajectories'/row/'replay/contact_validation.npz') as contacts:
            C, first = merge_frame(inp, contacts, a.guard)
        names = inp.arrays['scene_object_names'].tolist()
        slots = []
        for radial, radius in enumerate(RADII):
            for octant in range(8):
                rng = np.random.default_rng(np.random.SeedSequence([a.seed, number, radial, octant]))
                signs = np.array([1 if octant & (1 << axis) else -1 for axis in range(3)])
                rejected = []
                offset = None
                for draw in range(16):
                    direction = abs(rng.normal(size=3))*signs
                    candidate = radius*direction/np.linalg.norm(direction)
                    initial = inp.initial['qpos'].copy(); initial[:3] += candidate
                    contacts = scene_contacts(model, initial, inp.initial['qvel'], names)
                    penetrating = [c for c in contacts if c['distance_m'] < 0]
                    inside = bool(np.all(initial[:3] >= model.jnt_range[:3, 0]) and np.all(initial[:3] <= model.jnt_range[:3, 1]))
                    if not penetrating and inside:
                        offset = candidate
                        break
                    rejected.append(dict(draw=draw, delta_xyz_m=candidate.tolist(), contacts=penetrating, wrist_in_range=inside))
                slot = dict(slot=f'r{radial+1}_o{octant}', radius_m=radius, octant=octant,
                            geometric_rejections=rejected, sampling='one feasible draw, max16 per fixed radial/octant cell')
                if offset is None:
                    slot.update(state='geometry_excluded', delta_xyz_m=None)
                else:
                    _, _, _, info = augment(inp, parent, offset, C)
                    slot.update(state='planned', **info)
                identity = json.dumps(dict(parent=row, parent_trace=provenance['parent_trace_sha256'],
                                           seed=a.seed, slot=slot), sort_keys=True).encode()
                slot['variant_id'] = row+'-'+slot['slot']+'-'+hashlib.sha256(identity).hexdigest()[:12]
                slots.append(slot)
        parents.append(dict(row_id=row, merge_frame=C, first_contact_frame=first, frames=inp.frames,
                            source=inp.manifest['source'], provenance=provenance,
                            initial_qpos_sha256=hashlib.sha256(inp.initial['qpos'].tobytes()).hexdigest(), slots=slots))
        print('PLANNED', row, sum(s['state']=='planned' for s in slots), '/32', flush=True)
    result = dict(schema='manorl.position-augmentation-plan.v1', seed=a.seed, guard_s=a.guard,
                  radii_m=RADII, octants=8, hand='cheyingtong', control_hz=120, physics_hz=480,
                  expected_slots=1344, parents=parents, bundle_comparison_sha256=sha(a.bundle/'comparison.json'),
                  catalog_sha256=sha(a.bundle/'recipes/catalog.json'), manifest_sha256=sha(a.bundle/'asset_manifest.json'),
                  created=timestamp())
    a.plan.parent.mkdir(parents=True, exist_ok=True)
    dump(a.plan, result)


def write_run(folder, inp, parent, fingers, trace, runtime, plan, provenance, model, C):
    folder.mkdir(parents=True, exist_ok=False)
    np.testing.assert_array_equal(trace['desired'][C-1:], parent[C-1:])
    np.testing.assert_array_equal(trace['finger_target_delta'], fingers)
    np.testing.assert_array_equal(trace['qpos'][0], inp.initial['qpos'])
    np.testing.assert_array_equal(trace['qvel'][0], inp.initial['qvel'])
    assert trace['ctrl_substeps'].shape == (inp.frames-1, 4, 28)
    trace['parent_desired'] = parent
    np.savez_compressed(folder/'trace.npz', **trace)
    physical, diagnostic = evaluate(model, inp, trace)
    early = []
    for frame in range(C):
        contacts = scene_contacts(model, trace['qpos'][frame], trace['qvel'][frame], inp.arrays['scene_object_names'].tolist())
        if contacts:
            early.append(dict(frame=frame, contacts=contacts))
    gaps = unsupported_intervals(diagnostic['arrays'], inp.hz)
    gates = dict(physical_pose=physical['physical_pose_pass'], no_premerge_scene_contact=not early,
                 no_unsupported_interval=not gaps)
    result = dict(variant_id=plan['variant_id'], row_id=inp.row_id, state='complete', accepted=all(gates.values()),
                  gates=gates, physical=physical, plan=plan, trace_sha256=sha(folder/'trace.npz'),
                  premerge_scene_contacts=early, unsupported_intervals=gaps,
                  contact_semantics='native geometry at120Hz, not480Hz force telemetry')
    np.savez_compressed(folder/'contact_validation.npz', **diagnostic['arrays'])
    dump(folder/'result.json', result)
    dump(folder/'provenance.json', dict(provenance, augmentation=plan, frame_zero_integrated=False,
                                      physical_hand='cheyingtong', predecessor_replay_hand='sunke'))
    dump(folder/'metrics.json', dict(runtime=runtime, source=inp.manifest['source'],
                                     source_metadata=inp.metrics['source_metadata'], clock=inp.provenance,
                                     asset_provenance=inp.provenance['asset_provenance']))
    dump(folder/'active_contact_chronology.json', diagnostic['contacts'])
    return result


def run_parent(a):
    plan = json.loads(a.plan.read_text())
    if sha(a.bundle/'comparison.json') != plan['bundle_comparison_sha256'] or sha(a.bundle/'recipes/catalog.json') != plan['catalog_sha256'] or sha(a.bundle/'asset_manifest.json') != plan['manifest_sha256']:
        raise ValueError('frozen bundle identity changed')
    entry = next(p for p in plan['parents'] if p['row_id']==a.row)
    inp, parent, fingers, baseline, provenance = reconstruct(a.bundle, a.row)
    if provenance != entry['provenance']:
        raise ValueError('parent source/recipe identity changed')
    _, model = compile_parent(a, inp)
    folder = a.output/a.row
    folder.mkdir(parents=True, exist_ok=False)
    slots = entry['slots']
    if a.slot:
        slots = [s for s in slots if s['slot'] in a.slot]
        if len(slots) != len(set(a.slot)):
            raise ValueError('unknown slot')
    valid, results = [], []
    for slot in slots:
        if slot['state']=='geometry_excluded':
            result = dict(variant_id=slot['variant_id'], row_id=a.row, state='geometry_excluded', accepted=False, plan=slot)
            out = folder/slot['slot']; out.mkdir(); dump(out/'result.json', result); results.append(result)
        else:
            valid.append(slot)
    from sim.manorl.start_batch_dynamics import replay_batch
    from sim.manorl.local_contact_dynamics import replay
    for begin in range(0, len(valid), a.batch_size):
        selected = valid[begin:begin+a.batch_size]
        inputs, targets = [], []
        for slot in selected:
            shifted, desired, _, info = augment(inp, parent, slot['delta_xyz_m'], entry['merge_frame'])
            np.testing.assert_array_equal(shifted.initial['qpos'][3:], inp.initial['qpos'][3:])
            np.testing.assert_array_equal(shifted.initial['qvel'], inp.initial['qvel'])
            inputs.append(shifted); targets.append(desired)
        start = time.monotonic()
        if a.scalar:
            if len(selected) != 1:
                raise ValueError('--scalar requires batch-size1')
            trace, runtime = replay(inputs[0], model, targets[0], fingers); traces = [trace]
        else:
            traces, runtime = replay_batch(inputs, model, targets, fingers)
        for slot, shifted, trace in zip(selected, inputs, traces):
            result = write_run(folder/slot['slot'], shifted, parent, fingers, trace, runtime, slot, provenance, model, entry['merge_frame'])
            results.append(result)
            print('DONE', a.row, slot['slot'], result['accepted'], result['physical']['failed_gates'], flush=True)
        dump(folder/'progress.json', dict(completed=len(results), planned=len(slots), last_batch_elapsed_s=time.monotonic()-start))
    dump(folder/'summary.json', dict(row_id=a.row, results=results, count=len(results), accepted=sum(r['accepted'] for r in results)))


def run_queue(a):
    plan = json.loads(a.plan.read_text())
    if a.output.exists():
        raise ValueError('new campaign output required; partial runs are never silently retried')
    a.output.mkdir(parents=True)
    pin = code_pin()
    plan_hash = sha(a.plan)
    dump(a.output/'code_pin.json', pin)
    dump(a.output/'job.json', dict(pid=os.getpid(), plan=str(a.plan.resolve()), plan_sha256=plan_hash,
                                  bundle=str(a.bundle.resolve()), batch_size=a.batch_size, started=timestamp()))
    parents = plan['parents'] if not a.rows else [p for p in plan['parents'] if p['row_id'] in a.rows]
    if a.rows and len(parents) != len(set(a.rows)):
        raise ValueError('unknown or duplicate row selector')
    complete, results = [], []
    for parent in parents:
        if code_pin() != pin or sha(a.plan) != plan_hash:
            raise RuntimeError('execution code or frozen plan changed during campaign')
        name = parent['row_id']
        cmd = [sys.executable, '-m', 'tools.run_start_augmentation', '--run-parent', '--row', name,
               '--bundle', str(a.bundle.resolve()), '--asset-root', str(a.asset_root.resolve()),
               '--output', str(a.output.resolve()), '--plan', str(a.plan.resolve()), '--batch-size', str(a.batch_size)]
        if a.slot: cmd += ['--slot', *a.slot]
        if a.scalar: cmd += ['--scalar']
        with (a.output/f'{name}.log').open('x') as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            dump(a.output/'status.json', dict(state='running', row_id=name, child_pid=child.pid, completed=complete,
                                             attempted=len(results), accepted=sum(r['accepted'] for r in results), updated=timestamp()))
            code = child.wait()
        if code:
            dump(a.output/'status.json', dict(state='failed', row_id=name, exit_code=code, completed=complete, updated=timestamp()))
            raise RuntimeError(f'execution failure {name}; inspect its log, do not count as physical rejection')
        summary = json.loads((a.output/name/'summary.json').read_text())
        results.extend(summary['results']); complete.append(name)
        dump(a.output/'summary.json', dict(parents_completed=complete, attempted=len(results),
                                           accepted=sum(r['accepted'] for r in results), results=results))
    dump(a.output/'status.json', dict(state='complete', completed=complete, attempted=len(results),
                                     accepted=sum(r['accepted'] for r in results), updated=timestamp()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('bundle','asset-root','plan','output'):
        p.add_argument('--'+field, type=Path, required=field!='output')
    p.add_argument('--make-plan', action='store_true')
    p.add_argument('--run-parent', action='store_true')
    p.add_argument('--row'); p.add_argument('--rows', nargs='+'); p.add_argument('--slot', nargs='+')
    p.add_argument('--guard', type=float, default=.1); p.add_argument('--seed', type=int, default=2026091701)
    p.add_argument('--batch-size', type=int, default=32); p.add_argument('--scalar', action='store_true')
    a = p.parse_args()
    if not 1 <= a.batch_size <= 32 or (a.scalar and a.batch_size != 1):
        p.error('batch-size1..32; scalar requires1')
    if a.make_plan: plan_campaign(a)
    elif a.output is None: p.error('--output required for execution')
    elif a.run_parent: run_parent(a)
    else: run_queue(a)


if __name__=='__main__':
    main()
