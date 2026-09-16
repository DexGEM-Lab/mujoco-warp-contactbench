#!/usr/bin/env python3
"""Six scalar start-position trials. Stops on the first contradictory zero replay."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
import numpy as np

from sim.manorl.local_contact_repair import dump, edit_targets, evaluate, load_catalog, load_input, sha
from sim.manorl.start_augmentation import augment, scene_contacts, unsupported_intervals
from tools.repair_local_contacts import bind_model


def reconstruct(bundle, row):
    inp = load_input(bundle/'inputs100', row, 120)
    catalog = load_catalog(bundle/'recipes/catalog.json', bundle/'inputs100', 120)
    recipe = (json.loads(catalog[row].read_text()) if catalog[row] else
              dict(schema='manorl.local-contact-repair.v1', row_id=row, phases=[]))
    desired, fingers, _ = edit_targets(inp, recipe, tuple(inp.manifest['joint_names']))
    path = bundle/'trajectories'/row/'replay/trace.npz'
    expected = next(r['trace_sha256'] for r in json.loads((bundle/'comparison.json').read_text())['rows'] if r['id']==row)
    if sha(path) != expected:
        raise ValueError('published trace hash mismatch')
    with np.load(path, allow_pickle=False) as z:
        trace = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(desired, trace['desired'])
    np.testing.assert_array_equal(fingers, trace['finger_target_delta'])
    return inp, desired, fingers, trace, dict(parent_trace_sha256=expected,
        recipe_sha256=sha(catalog[row]) if catalog[row] else None, source=inp.provenance)


def one(a):
    out = a.output/a.row/a.variant
    out.mkdir(parents=True, exist_ok=False)
    try:
        inp, parent, fingers, baseline, provenance = reconstruct(a.bundle, a.row)
        _, model = bind_model(SimpleNamespace(manifest=a.bundle/'asset_manifest.json', asset_root=a.asset_root), inp)
        with np.load(a.bundle/'trajectories'/a.row/'replay/contact_validation.npz') as z:
            first = int(np.flatnonzero(z['finger_ray_count'])[0])
        move = min(x['start_frame'] for x in inp.metrics['source_metadata']['object_move'])/100
        C = int(np.floor(min(first/120-a.guard, max(.30, move-a.guard))*120+1e-9))
        names = inp.arrays['scene_object_names'].tolist()
        approach = parent[C, :2]-parent[0, :2]
        norm = np.linalg.norm(approach)
        tangent = np.array([-approach[1], approach[0], 0.])/norm if norm > 1e-10 else np.array([1.,0.,0.])
        # A single, common collision-free direction is used for both signs.
        choices = [np.array([1.,0.,0.]), np.array([0.,1.,0.]), tangent]
        exclusions = []
        direction = None
        for label, candidate in zip(('world_x','world_y','approach_tangent'), choices):
            collisions = []
            for sign in (-1,1):
                q = inp.initial['qpos'].copy(); q[:3] += sign*a.radius*candidate
                contacts = scene_contacts(model, q, inp.initial['qvel'], names)
                collisions += [dict(sign=sign, **c) for c in contacts if c['distance_m'] < 0]
            if not collisions:
                direction = candidate; direction_label = label; break
            exclusions.append(dict(direction=label, penetration=collisions))
        if a.variant != 'zero' and direction is None:
            dump(out/'result.json', dict(state='geometry_excluded', exclusions=exclusions, accepted=False))
            return
        delta = np.zeros(3) if a.variant == 'zero' else a.radius*direction*(1 if a.variant == 'plus' else -1)
        shifted, desired, envelope, plan = augment(inp, parent, delta, C)
        preflight = scene_contacts(model, shifted.initial['qpos'], shifted.initial['qvel'], names)
        plan.update(guard_s=a.guard, parent_first_contact_frame=first, movement_start_s=move,
                    direction='zero' if a.variant=='zero' else direction_label,
                    geometry_exclusions=exclusions, initial_scene_contacts=preflight,
                    variant_id=f'{a.row}-{a.variant}-'+__import__('hashlib').sha256(desired.tobytes()).hexdigest()[:16],
                    planning_caps=dict(speed_m_s=.5, acceleration_m_s2=5.), provenance=provenance)
        dump(out/'plan.json', plan)
        if any(c['distance_m'] < 0 for c in preflight):
            dump(out/'result.json', dict(state='geometry_excluded', contacts=preflight, accepted=False))
            return
        np.savez_compressed(out/'input.npz', desired=desired, parent_desired=parent, envelope=envelope,
                            finger_target_delta=fingers, **{'initial_'+k:v for k,v in shifted.initial.items()})
        from sim.manorl.local_contact_dynamics import replay
        trace, runtime = replay(shifted, model, desired, fingers)
        trace['parent_desired'] = parent
        np.testing.assert_array_equal(trace['qpos'][0], shifted.initial['qpos'])
        np.testing.assert_array_equal(trace['qvel'][0], inp.initial['qvel'])
        assert trace['ctrl_substeps'].shape[0] == inp.frames-1
        np.savez_compressed(out/'trace.npz', **trace)
        physical, diagnostic = evaluate(model, shifted, trace)
        np.savez_compressed(out/'contact_validation.npz', **diagnostic['arrays'])
        dump(out/'active_contact_chronology.json', diagnostic['contacts'])
        early = []
        for frame in range(C):
            contacts = scene_contacts(model, trace['qpos'][frame], trace['qvel'][frame], names)
            if contacts:
                early.append(dict(frame=frame, contacts=contacts))
        gaps = unsupported_intervals(diagnostic['arrays'], 120)
        first_actual = np.flatnonzero(diagnostic['arrays']['finger_ray_count'])
        max_qpos = float(np.max(abs(trace['qpos']-baseline['qpos'])))
        if a.variant == 'zero':
            for key in ('desired', 'finger_target_delta', 'physics_time'):
                np.testing.assert_array_equal(trace[key], baseline[key])
            for key in ('qpos', 'qvel', 'ctrl'):
                np.testing.assert_array_equal(trace[key][0], baseline[key][0])
            parent_runtime = json.loads((a.bundle/'trajectories'/a.row/'replay/metrics.json').read_text())['runtime']
            for key in ('model_array_hashes', 'controller', 'control_hz', 'physics_hz', 'substeps', 'frame_zero_integrated'):
                if runtime[key] != parent_runtime[key]:
                    raise ValueError(f'zero replay runtime changed: {key}')
        merge = dict(wrist_position_error_m=float(np.linalg.norm(trace['qpos'][C,:3]-baseline['qpos'][C,:3])),
                     wrist_velocity_error_m_s=float(np.linalg.norm(trace['qvel'][C,:3]-baseline['qvel'][C,:3])),
                     qpos_delta=(trace['qpos'][C]-baseline['qpos'][C]).tolist(),
                     qvel_delta=(trace['qvel'][C]-baseline['qvel'][C]).tolist(),
                     integral_delta=(trace['wrist_integral'][C]-baseline['wrist_integral'][C]).tolist(),
                     object_position_delta_m=(trace['actual_object_pos'][C]-baseline['actual_object_pos'][C]).tolist())
        gates = dict(physical_pose=physical['physical_pose_pass'], no_premerge_scene_contact=not early,
                     no_unsupported_interval=not gaps)
        result = dict(state='complete', accepted=all(gates.values()), gates=gates, physical=physical,
                      initial_hand_xyz_m=trace['qpos'][0,:3].tolist(), merge=merge,
                      parent_max_qpos_difference=max_qpos, first_active_contact_frame=int(first_actual[0]) if len(first_actual) else None,
                      premerge_scene_contacts=early, unsupported_intervals=gaps,
                      zero_contradiction=a.variant=='zero' and not all(gates.values()),
                      sampling_boundary='contacts reconstructed at120Hz; no480Hz contact/force telemetry', runtime=runtime)
        dump(out/'result.json', result)
        print(json.dumps(dict(row=a.row, variant=a.variant, accepted=result['accepted'], merge=merge)), flush=True)
        if result['zero_contradiction']:
            raise RuntimeError('zero replay fails physical/augmentation gates; inspect before continuing')
    except Exception as exc:
        dump(out/'failure.json', dict(error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','asset-root','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--guard', type=float, default=.1)
    p.add_argument('--radius', type=float, default=.1)
    p.add_argument('--row', choices=('A_row035','B_row043'))
    p.add_argument('--variant', choices=('zero','plus','minus'))
    a = p.parse_args()
    if not np.isfinite([a.guard,a.radius]).all() or not 0<a.guard or not .03<=a.radius<=.1:
        p.error('positive guard and radius in[.03,.1] required')
    if a.row:
        if not a.variant: p.error('--row requires --variant')
        one(a); return
    a.output.mkdir(parents=True, exist_ok=False)
    ids = (a.bundle/'accepted_full_pose_ids.txt').read_text().split()
    rows = json.loads((a.bundle/'comparison.json').read_text())['rows']
    if len(ids)!=42 or len(set(ids))!=42 or set(ids)!={r['id'] for r in rows if r['final120_full_pose']} or 'B_row035' in ids:
        raise ValueError('expected exactly the42 published strict seeds')
    verified = {row: reconstruct(a.bundle,row)[4] for row in ids}
    root = Path(__file__).resolve().parents[1]
    files = ['sim/manorl/start_augmentation.py','sim/manorl/local_contact_repair.py',
             'sim/manorl/local_contact_dynamics.py','tools/repair_local_contacts.py','tools/pilot_start_augmentation.py']
    pin = {f:sha(root/f) for f in files}
    dump(a.output/'provenance.json', dict(verified_seeds=verified, code_sha256=pin,
        catalog_sha256=sha(a.bundle/'recipes/catalog.json'), manifest_sha256=sha(a.bundle/'asset_manifest.json')))
    results = []
    for row in ('A_row035','B_row043'):
        for variant in ('zero','plus','minus'):
            if {f:sha(root/f) for f in files} != pin: raise RuntimeError('pilot code changed')
            cmd = [sys.executable, '-m', 'tools.pilot_start_augmentation','--bundle',str(a.bundle),
                   '--asset-root',str(a.asset_root),'--output',str(a.output),'--guard',str(a.guard),
                   '--radius',str(a.radius),'--row',row,'--variant',variant]
            with (a.output/f'{row}-{variant}.log').open('x') as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                dump(a.output/'active_process.json', dict(pid=child.pid, row=row, variant=variant, command=cmd))
                code = child.wait()
            if code:
                dump(a.output/'status.json', dict(state='stopped_for_inspection', row=row, variant=variant, exit_code=code))
                raise RuntimeError(f'{row}/{variant} failed; inspect its log and failure.json')
            results.append(dict(row=row,variant=variant,**json.loads((a.output/row/variant/'result.json').read_text())))
            dump(a.output/'summary.json', dict(results=results, complete=len(results)==6))
    dump(a.output/'status.json', dict(state='complete'))


if __name__ == '__main__':
    main()
