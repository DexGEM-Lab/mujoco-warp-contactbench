#!/usr/bin/env python3
"""Terminal-only planted-state U1 load diagnosis, never an accepted trajectory.

Initial hand/object states come from one source frame. Both bodies then move
freely under native U1; only a constant position-actuator target is applied.
Compare no compensation with the target offset required to support object weight.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from tools.u1_raw_source import load_raw_source
from tools.u1_interactive_session import Session, native_object_forces


def run(args):
    import mujoco as mj

    inp, model, base, teacher, provenance = load_raw_source(
        args.dataset, args.version, args.row, args.asset_root, args.asset_manifest)
    if not 0 <= args.frame < len(base):
        raise ValueError('frame outside source')
    args.output.mkdir(parents=True, exist_ok=False)
    initial = teacher['qpos'][args.frame].copy()
    body = model.body(inp.metrics['source_metadata']['active_object']).id
    d = mj.MjData(model); d.qpos[:] = initial; mj.mj_forward(model, d)
    point = d.xipos[body].copy()
    jacp = np.zeros((3, model.nv)); jacr = np.zeros_like(jacp)
    mj.mj_jac(model, d, jacp, jacr, point, int(model.jnt_bodyid[5]))
    weight = -float(model.body_mass[body]) * np.asarray(model.opt.gravity)
    delta = (jacp[:, :6].T @ weight) / model.actuator_gainprm[:6, 0]
    frames = int(round(args.seconds * 120)) + 1
    diagnostic = deepcopy(inp)
    diagnostic.frames = frames
    diagnostic.initial = {'qpos': initial.copy(), 'qvel': np.zeros(model.nv)}
    static_teacher = {'qpos': np.repeat(initial[None], frames, axis=0)}
    print(f'DIAGNOSTIC source frame={args.frame}; not a frame0 success or acquisition test', flush=True)
    print(f'U1 CCD={model.opt.ccd_iterations} cone={model.opt.cone} impratio={model.opt.impratio} '
          f'weight={weight[2]:.4f}N analytic wrist-target shift={delta.tolist()}', flush=True)
    results = []
    for label, correction in [('source_target', np.zeros(6)), ('weight_compensated_target', delta)]:
        ctrl = initial[:28].copy()
        # Source may slightly exceed native limits; report clipping of the INITIAL
        # target explicitly. Diagnostic state geometry itself stays as supplied.
        ctrl = np.clip(ctrl, model.jnt_range[:28, 0], model.jnt_range[:28, 1])
        ctrl[:6] += correction
        if np.any(ctrl < model.jnt_range[:28, 0]) or np.any(ctrl > model.jnt_range[:28, 1]):
            raise ValueError('calculated compensation exceeds native control limits')
        target = np.repeat(ctrl[None], frames, axis=0)
        s = Session(diagnostic, model, target, static_teacher, provenance)
        samples, positions, velocities = [], [], []
        baseline = s.state()
        print(f'BEGIN {label}; ctrl_minus_source_xyz_mm={(ctrl[:3]-initial[:3])*1000}', flush=True)
        for f in range(frames):
            if f:
                s.step()
            positions.append(s.data.qpos.numpy()[0].copy())
            velocities.append(s.data.qvel.numpy()[0].copy())
            if f % 12 and f != frames-1:
                continue
            state = s.state()
            force_by_body, total = native_object_forces(s) if f else ({}, [0., 0., 0.])
            hand_force = np.sum([v for k,v in force_by_body.items() if k not in s.names+['world']], axis=0) if force_by_body else np.zeros(3)
            if np.shape(hand_force) != (3,):
                hand_force = np.zeros(3)
            support_force = np.sum([v for k,v in force_by_body.items() if k in s.names+['world']], axis=0) if force_by_body else np.zeros(3)
            if np.shape(support_force) != (3,):
                support_force = np.zeros(3)
            relative = np.linalg.norm(np.asarray(state['hand_object_transform']['position']) - np.asarray(baseline['hand_object_transform']['position'])) * 1000
            sample = {'step': f, 'time_s': f/120,
                      'hand_dz_mm': (state['current_28d'][2]-baseline['current_28d'][2])*1000,
                      'object_dz_mm': (state['object_position'][2]-baseline['object_position'][2])*1000,
                      'relative_translation_drift_mm': relative,
                      'tilt_deg': state['object_tilt_deg'],
                      'native_hand_force_on_object_N': hand_force.tolist(),
                      'native_other_force_on_object_N': support_force.tolist(),
                      'native_total_constraint_object_force_N': total,
                      'native_force_by_other_body_N': force_by_body,
                      'geometry_links': state['hand_links']}
            samples.append(sample)
            print(f"{label} t={f/120:.2f} hand_dz={sample['hand_dz_mm']:+.1f}mm "
                  f"object_dz={sample['object_dz_mm']:+.1f}mm relative_drift={relative:.1f}mm "
                  f"hand_Fz={hand_force[2]:+.2f}N other_Fz={support_force[2]:+.2f}N "
                  f"tilt={state['object_tilt_deg']:.1f}deg links={','.join(state['hand_links'])}", flush=True)
        results.append({'mode': label, 'accepted_trajectory': False, 'samples': samples})
        np.savez_compressed(args.output/(label+'.npz'), qpos=np.asarray(positions),
                            qvel=np.asarray(velocities), targets=target)
    report = {'diagnostic_only': True, 'source': provenance, 'source_frame': args.frame,
              'contract': s.contract, 'analytic_wrist_delta': delta.tolist(), 'results': results}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','asset-root','asset-manifest','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--version',type=int,required=True)
    p.add_argument('--row',type=int,required=True)
    p.add_argument('--frame',type=int,required=True)
    p.add_argument('--seconds',type=float,default=1.)
    a=p.parse_args()
    if not np.isfinite(a.seconds) or not 0 < a.seconds <= 10:
        p.error('seconds must be finite in (0,10]')
    run(a)
