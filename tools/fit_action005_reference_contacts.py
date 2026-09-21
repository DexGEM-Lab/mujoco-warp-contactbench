"""Bounded A030 object-local contact fit; frozen native-U1 diagnostic only."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
BOUNDS = np.r_[np.full(3, .015/np.sqrt(3)), np.full(3, np.deg2rad(8)/np.sqrt(3)), np.ones(22)]


def smooth(x):
    u = np.clip(x, 0, 1)
    return u*u*(3-2*u)


def phase_weights(frames):
    """Four fingers close first; thumb rejoins reference first on release."""
    f = np.arange(frames)
    nonthumb = smooth((f-225)/55)*smooth((1050-f)/65)
    thumb = smooth((f-260)/55)*smooth((1010-f)/55)
    wrist = smooth((f-225)/90)*smooth((1050-f)/65)
    return np.column_stack([np.repeat(wrist[:, None], 6, axis=1),
                            np.repeat(thumb[:, None], 6, axis=1),
                            np.repeat(nonthumb[:, None], 16, axis=1)])


def extract_patches(mj, model, donor, telemetry):
    # Native last-substep contacts precede the final integration by 1/480s;
    # the trace has 120Hz poses only. Preserve this <=1/120s epoch uncertainty.
    data = mj.MjData(model); data.qpos[:] = donor['qpos'][90]
    mj.mj_forward(model, data)
    ob = model.body('mayonnaisebottle').id
    rotation = data.xmat[ob].reshape(3, 3)
    patches = []
    for finger in FINGERS:
        c = max((c for c in telemetry['contacts'] if c['finger'] == finger),
                key=lambda c: c['local_wrench'][0])
        body = model.body(c['link']).id
        point = np.asarray(c['position'])
        normal = c['object_sign']*np.asarray(c['basis']).reshape(3, 3)[0]
        patches.append(dict(finger=finger, body=body, link=c['link'],
            object_point=(rotation.T@(point-data.xpos[ob])).tolist(),
            hand_point=(data.xmat[body].reshape(3, 3).T@(point-data.xpos[body])).tolist(),
            object_normal=(rotation.T@normal).tolist(),
            hand_normal=(data.xmat[body].reshape(3, 3).T@normal).tolist()))
    return patches


def fit(args, mj, inp, model, teacher):
    donor = np.load(args.donor/'trace.npz')
    with gzip.open(args.donor/'telemetry.jsonl.gz', 'rt') as stream:
        telemetry = next(row for row in map(json.loads, stream) if row['frame'] == 90)
    patches = extract_patches(mj, model, donor, telemetry)
    reference = teacher['qpos'].copy()
    reference[:, 3:6] = np.unwrap(reference[:, 3:6], axis=0)
    ob = model.body('mayonnaisebottle').id
    data = mj.MjData(model)
    knots = np.unique(np.r_[np.arange(265, 986, 8), 985])
    corrections, evidence = [], []
    previous = np.zeros(28)
    for frame in knots:
        base = reference[frame].copy()
        lower = np.maximum(-BOUNDS, model.jnt_range[:28, 0]-base[:28]+1e-5)
        upper = np.minimum(BOUNDS, model.jnt_range[:28, 1]-base[:28]-1e-5)
        lower[:6], upper[:6] = -BOUNDS[:6], BOUNDS[:6]
        def geometry(delta):
            data.qpos[:] = base; data.qpos[:28] += delta
            mj.mj_forward(model, data)
            rot = data.xmat[ob].reshape(3, 3)
            errors, normals = [], []
            for patch in patches:
                br = data.xmat[patch['body']].reshape(3, 3)
                actual = data.xpos[patch['body']]+br@patch['hand_point']
                goal = data.xpos[ob]+rot@patch['object_point']
                errors.append(actual-goal)
                normals.append(br@patch['hand_normal']-rot@patch['object_normal'])
            return np.asarray(errors), np.asarray(normals)
        def residual(delta):
            errors, normals = geometry(delta)
            return np.r_[errors.ravel()/.002, normals.ravel()/.25,
                         .06*delta/BOUNDS, .08*(delta-previous)/BOUNDS]
        result = least_squares(residual, np.clip(previous, lower+1e-9, upper-1e-9),
                               bounds=(lower, upper), diff_step=1e-4, max_nfev=100,
                               ftol=1e-6, xtol=1e-6, gtol=1e-6)
        previous = result.x.copy(); corrections.append(previous)
        error, normal = geometry(previous)
        record = dict(frame=int(frame), point_error_mm=(np.linalg.norm(error, axis=1)*1000).tolist(),
                      normal_error=np.linalg.norm(normal, axis=1).tolist(), cost=float(result.cost),
                      bound_fraction=float(np.max(abs(previous)/BOUNDS)))
        evidence.append(record)
        if frame == knots[0] or frame % 40 == 25: print('FIT', json.dumps(record), flush=True)
    interpolated = PchipInterpolator(knots, corrections)(np.clip(np.arange(inp.frames), knots[0], knots[-1]))
    weights = phase_weights(inp.frames)
    base_target = np.load(args.base)
    # Keep existing wrist damping/gravity lead; measured opposed-contact finger
    # servo deflection supplies preload without an additional squeeze sweep.
    finger_lead = np.clip(donor['ctrl'][90, 6:28]-donor['qpos'][90, 6:28], -.3, .3)
    desired = base_target.copy()
    desired[:, :6] += weights[:, :6]*interpolated[:, :6]
    fitted_fingers = reference[:, 6:28]+interpolated[:, 6:28]+finger_lead
    desired[:, 6:28] += weights[:, 6:28]*(fitted_fingers-base_target[:, 6:28])
    desired = np.clip(desired, *model.actuator_ctrlrange.T)
    np.save(args.output/'target.npy', desired)
    np.savez_compressed(args.output/'fit.npz', knots=knots, corrections=corrections,
                        reference_qpos=reference, weights=weights, baseline_target=base_target)
    report = dict(reference='A_row030', frame_count=inp.frames, patches=patches,
        donor=str(args.donor), donor_frame=90, contact_pose_epoch_uncertainty_s=1/480,
        wrist_translation_bound_m=.015, wrist_euler_norm_bound_deg=8,
        finger_correction_bound_rad=1., finger_lead_rad=finger_lead.tolist(),
        wrist_peak_translation_m=float(np.linalg.norm(desired[:, :3]-base_target[:, :3], axis=1).max()),
        wrist_peak_euler_norm_deg=float(np.rad2deg(np.linalg.norm(desired[:, 3:6]-base_target[:, 3:6], axis=1).max())),
        max_target_step=np.max(abs(np.diff(desired, axis=0)), axis=0).tolist(),
        prefix_exact=bool(np.array_equal(desired[:225], base_target[:225])),
        release_tail_exact=bool(np.array_equal(desired[1050:], base_target[1050:])),
        evidence=evidence, accepted=False)
    (args.output/'fit.json').write_text(json.dumps(report, indent=2)+'\n')
    return desired


def replay(args, mj, inp, model, teacher, provenance, target):
    from tools.u1_interactive_session import Session
    from tools.search_action005_critical_grip import observe, critical_frames
    s = Session(inp, model, target, teacher, provenance)
    ob, hb = s.object_body, s.hand_body
    adr = int(model.joint('mayonnaisebottle_free').qposadr[0])
    end = next(x['frame'] for x in critical_frames(teacher['qpos'], adr) if x['requested_tilt_deg'] == 90)
    q0 = inp.initial['qpos'].copy(); d = s.cpu
    baseline = None; previous = None
    poses, velocities, controls = [q0], [inp.initial['qvel'].copy()], [s.data.ctrl.numpy()[0].copy()]
    high = dict(nacon=0, ncollision=0, nefc=0)
    rows = []
    with gzip.open(args.output/'telemetry.jsonl.gz', 'wt') as stream:
        for frame in range(1, end+1):
            s.frame = frame
            s.data.ctrl.assign(target[frame][None].astype(np.float32))
            for _ in range(4):
                s.mw.step(s.wm, s.data); s.wp.synchronize()
                for field, limit in (('nacon', 1024), ('ncollision', 256), ('nefc', 4096)):
                    value = int(getattr(s.data, field).numpy().max())
                    high[field] = max(high[field], value)
                    if value >= limit: raise RuntimeError(f'{field} capacity reached: {value}')
                if np.any(s.data.xfrc_applied.numpy()) or np.any(s.data.qfrc_applied.numpy()):
                    raise RuntimeError('nonzero applied external force')
            q = s.data.qpos.numpy()[0].copy(); v = s.data.qvel.numpy()[0].copy()
            poses.append(q); velocities.append(v); controls.append(s.data.ctrl.numpy()[0].copy())
            if not np.isfinite(q).all() or not np.isfinite(v).all(): raise RuntimeError('nonfinite state')
            if frame < 265: continue
            d.qpos[:] = q; d.qvel[:] = v; mj.mj_forward(model, d)
            if baseline is None or frame == 380:
                hr = d.xmat[hb].reshape(3, 3)
                baseline = (hr.T@(d.xpos[ob]-d.xpos[hb]), hr.T@d.xmat[ob].reshape(3, 3), q.copy())
                previous = baseline[:2]
            row, previous, _, _, _ = observe(s, baseline, previous, target[frame], frame)
            nr = s.data.xmat.numpy()[0, ob].reshape(3, 3)
            np_ = s.data.xpos.numpy()[0, ob]
            for c in row['contacts']:
                c['object_local_position'] = (nr.T@(np.asarray(c['position'])-np_)).tolist()
                c['object_local_normal'] = (nr.T@(c['object_sign']*np.asarray(c['basis']).reshape(3, 3)[0])).tolist()
            row['actual_tilt_deg'] = float(np.rad2deg(np.arccos(np.clip(d.xmat[ob].reshape(3, 3)[2, 2], -1, 1))))
            row['reference_tilt_deg'] = float(np.rad2deg(np.arccos(np.clip(R.from_quat(teacher['qpos'][frame, adr+3:adr+7], scalar_first=True).as_matrix()[2, 2], -1, 1))))
            row['baseline_frame'] = 380 if frame >= 380 else 265
            stream.write(json.dumps(row)+'\n'); rows.append(row)
            if frame % 20 == 0: print('NATIVE', frame, row['reference_tilt_deg'], row['actual_tilt_deg'], row['drift_deg'], row['bearing_normal_N'], flush=True)
    np.savez_compressed(args.output/'trace.npz', qpos=poses, qvel=velocities, ctrl=controls)
    interval = [r for r in rows if 50 <= r['reference_tilt_deg'] <= 90]
    maintained = bool(interval) and all(r['bearing_normal_N']['ring']>.2 and r['bearing_normal_N']['pinky']>.2 and r['axial_contact_span_m']>.07 and r['drift_deg']<5 for r in interval)
    result = dict(accepted=False, full_action_passed_twice=False, focused_mechanism_passed=maintained,
                  end_frame=end, capacity_high_water=high, external_forces_zero=True,
                  post_frame0_state_writes=False, finite=True,
                  target_max_error=float(np.max(abs(np.asarray(controls)-target[:len(controls)]))),
                  interval_frames=[r['frame'] for r in interval],
                  first_missing_lower_contact=next((r['frame'] for r in interval if min(r['bearing_normal_N'][f] for f in ('ring', 'pinky'))<=.2), None),
                  peak_interval_drift_deg=max((r['drift_deg'] for r in interval), default=None))
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('RESULT', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'donor', 'base', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--fit-only', action='store_true')
    parser.add_argument('--replay-existing', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    import mujoco as mj
    from run_teacher import load_teacher
    from run_reference import setting
    from tools.search_action005_critical_grip import U1
    inp, model, _, teacher, provenance, _ = load_teacher('A_row030')
    digest = hashlib.sha256(json.dumps(setting(model, inp.arrays['scene_object_names'].tolist()), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != U1: raise RuntimeError('nominal U1 mismatch')
    args.output.mkdir(parents=True, exist_ok=args.replay_existing)
    if args.replay_existing:
        if (args.output/'trace.npz').exists(): raise FileExistsError('preserve previous replay')
        target = np.load(args.output/'target.npy')
    else:
        target = fit(args, mj, inp, model, teacher)
    (args.output/'contract.json').write_text(json.dumps(dict(setting_sha256=digest, reference='A_row030', initial_qpos=inp.initial['qpos'].tolist(), initial_qvel=inp.initial['qvel'].tolist(), source_base=str(args.base), source_target_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest()), indent=2)+'\n')
    if not args.fit_only: replay(args, mj, inp, model, teacher, provenance, target)


if __name__ == '__main__':
    main()
