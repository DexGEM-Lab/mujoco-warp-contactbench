"""Bounded A030 sliding-contact continuation; frozen native-U1 diagnostic only."""
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
BOUNDS = np.r_[np.full(3, .020/np.sqrt(3)), np.full(3, np.deg2rad(15)/np.sqrt(3)), np.ones(22)]


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


def topology_metrics(points, normals):
    """Bottle-local ordering/leverage, independent of donor axial coordinates."""
    points, normals = np.asarray(points), np.asarray(normals)
    return dict(opposed_sides=bool(points[0, 1] < 0 and np.all(points[1:, 1] > 0)),
                axial_order=bool(np.all(np.diff(points[1:, 2]) < 0)),
                index_pinky_span_m=float(points[1, 2]-points[4, 2]),
                opposed_normal_cosines=(-normals[1:]@normals[0]).tolist())


class SurfaceGeometry:
    def __init__(self, mj, model, patches):
        self.mj, self.model = mj, model
        self.data = mj.MjData(model)
        self.ob = model.body('mayonnaisebottle').id
        self.objects = [g for g in range(model.ngeom) if model.geom_bodyid[g] == self.ob]
        self.geoms = [model.geom(p['link']+'_collision').id for p in patches]
        self.hand = [g for g in range(model.ngeom)
                     if model.body(int(model.geom_bodyid[g])).name.split('_')[0] in (*FINGERS, 'palm')]

    def __call__(self, q):
        m, d, mj = self.model, self.data, self.mj
        d.qpos[:] = q; mj.mj_forward(m, d)
        rot = d.xmat[self.ob].reshape(3, 3)
        distances, points, normals = [], [], []
        for g in self.geoms:
            options = []
            for o in self.objects:
                segment = np.zeros(6)
                distance = mj.mj_geomDistance(m, d, g, o, .2, segment)
                options.append((distance, segment))
            distance, segment = min(options, key=lambda v: v[0])
            normal = (segment[3:]-segment[:3]) / (distance if abs(distance) > 1e-10 else 1e-10)
            normal /= max(np.linalg.norm(normal), 1e-12)
            distances.append(distance); points.append(rot.T@(segment[3:]-d.xpos[self.ob]))
            normals.append(rot.T@normal)
        penetration = np.array([min(mj.mj_geomDistance(m, d, g, o, .1, np.zeros(6))
                                   for o in self.objects) for g in self.hand])
        return np.array(distances), np.array(points), np.array(normals), penetration


def fit(args, mj, inp, model, teacher):
    """Continue one sliding-contact branch from the preserved 70-degree solution."""
    prior = args.seed_root
    patches = json.loads((prior/'fit.json').read_text())['patches']
    seed = json.loads((prior/'surface_feasibility_70_neighbor.json').read_text())[0]
    reference = teacher['qpos'].copy()
    reference[:, 3:6] = np.unwrap(reference[:, 3:6], axis=0)
    geometry = SurfaceGeometry(mj, model, patches)
    anchor = seed['frame']
    knots = np.unique(np.r_[np.arange(265, 986, 4), anchor, 985])
    solutions, evidence = {}, {}

    def solve(frame, previous):
        base = reference[frame].copy()
        lower = np.maximum(-BOUNDS, model.jnt_range[:28, 0]-base[:28]+1e-5)
        upper = np.minimum(BOUNDS, model.jnt_range[:28, 1]-base[:28]-1e-5)
        lower[:6], upper[:6] = -BOUNDS[:6], BOUNDS[:6]
        def measure(delta):
            q = base.copy(); q[:28] += delta
            return geometry(q)
        def residual(delta):
            distance, points, normals, penetration = measure(delta)
            # No donor axial/azimuth matching: lower placement is useful leverage.
            order = np.maximum(np.diff(points[1:, 2])+.003, 0)/.001
            span = max(.082-(points[1, 2]-points[4, 2]), 0)/.001
            side = np.maximum(.020-points[:, 1]*[-1, 1, 1, 1, 1], 0)/.001
            opposed = np.maximum(normals[1:]@normals[0]+.5, 0)
            return np.r_[(distance+.0004)/.0005, order, span, side, opposed,
                         10*np.minimum(penetration+.001, 0)/.001,
                         .04*(delta-previous)/BOUNDS, .005*delta/BOUNDS]
        sol = least_squares(residual, np.clip(previous, lower+1e-8, upper-1e-8),
                            bounds=(lower, upper), diff_step=1e-4, max_nfev=100,
                            ftol=1e-7, xtol=1e-7, gtol=1e-7)
        distance, points, normals, pen = measure(sol.x)
        record = dict(frame=int(frame), distance_mm=(distance*1000).tolist(),
                      object_local_points_mm=(points*1000).tolist(),
                      object_local_normals=normals.tolist(), **topology_metrics(points, normals),
                      max_penetration_mm=float(max(0, -pen.min())*1000),
                      wrist_translation_mm=float(np.linalg.norm(sol.x[:3])*1000),
                      wrist_euler_norm_deg=float(np.rad2deg(np.linalg.norm(sol.x[3:6]))),
                      nfev=sol.nfev, cost=float(sol.cost))
        solutions[frame], evidence[frame] = sol.x.copy(), record
        print('SURFACE', json.dumps(record), flush=True)
        return sol.x

    center = solve(anchor, np.asarray(seed['delta']))
    for frames in (knots[knots < anchor][::-1], knots[knots > anchor]):
        previous = center.copy()
        for frame in frames:
            previous = solve(int(frame), previous)
    corrections = np.array([solutions[int(f)] for f in knots])
    interpolated = PchipInterpolator(knots, corrections)(np.clip(np.arange(inp.frames), knots[0], knots[-1]))
    weights = phase_weights(inp.frames)
    base_target = np.load(args.base)
    donor = np.load(args.donor/'trace.npz')
    finger_lead = np.clip(donor['ctrl'][90, 6:28]-donor['qpos'][90, 6:28], -.3, .3)
    desired = base_target.copy()
    desired[:, :6] += weights[:, :6]*interpolated[:, :6]
    desired[:, 6:28] += weights[:, 6:28]*(reference[:, 6:28]+interpolated[:, 6:28]+finger_lead-base_target[:, 6:28])
    desired = np.clip(desired, *model.actuator_ctrlrange.T)
    np.save(args.output/'target.npy', desired)
    np.savez_compressed(args.output/'fit.npz', knots=knots, corrections=corrections,
                        reference_qpos=reference, weights=weights, baseline_target=base_target)
    # Audit interpolation rather than interpreting knot success as path success.
    path = []
    for f in range(315, 986):
        q = reference[f].copy(); q[:28] += interpolated[f]
        dist, points, normals, pen = geometry(q)
        metric = topology_metrics(points, normals)
        passed = (metric['opposed_sides'] and metric['axial_order'] and metric['index_pinky_span_m'] >= .080
                  and min(metric['opposed_normal_cosines']) >= .5 and dist.max() <= .0001
                  and pen.min() >= -.00105)
        path.append(dict(frame=f, passed=bool(passed), max_gap_mm=float(dist.max()*1000),
                         max_penetration_mm=float(-pen.min()*1000), **metric))
    report = dict(reference='A_row030', frame_count=inp.frames, patches=patches,
                  seed_root=str(prior), anchor_frame=anchor, evidence=[evidence[int(f)] for f in knots],
                  interpolated_geometry=path, accepted=False,
                  wrist_translation_bound_m=.020, wrist_euler_norm_bound_deg=15,
                  max_correction_step=np.max(abs(np.diff(interpolated, axis=0)), axis=0).tolist(),
                  max_target_step=np.max(abs(np.diff(desired, axis=0)), axis=0).tolist(),
                  prefix_exact=bool(np.array_equal(desired[:225], base_target[:225])),
                  release_tail_exact=bool(np.array_equal(desired[1050:], base_target[1050:])))
    (args.output/'fit.json').write_text(json.dumps(report, indent=2)+'\n')
    return desired


def bearing_topology(contacts):
    # Missing middle contact must not erase measured index-to-pinky leverage.
    points, normals, missing = {}, {}, []
    for finger in FINGERS:
        active = [c for c in contacts if c['finger'] == finger and c['local_wrench'][0] > .01]
        if not active:
            missing.append(finger)
            continue
        weights = np.array([c['local_wrench'][0] for c in active])
        points[finger] = np.average([c['object_local_position'] for c in active], weights=weights, axis=0)
        normal = np.average([c['object_local_normal'] for c in active], weights=weights, axis=0)
        normals[finger] = normal/max(np.linalg.norm(normal), 1e-12)
    others = [f for f in FINGERS[1:] if f in points]
    opposition = [float(-normals[f]@normals['thumb']) for f in others] if 'thumb' in normals else []
    return dict(complete=not missing, missing_fingers=missing,
                bearing_points_m={f: p.tolist() for f, p in points.items()},
                bearing_normals={f: n.tolist() for f, n in normals.items()},
                opposed_sides=bool('thumb' in points and others and points['thumb'][1] < 0 and all(points[f][1] > 0 for f in others)),
                axial_order=bool(np.all(np.diff([points[f][2] for f in others]) < 0)) if len(others) == 4 else None,
                index_pinky_span_m=float(points['index'][2]-points['pinky'][2]) if 'index' in points and 'pinky' in points else 0.,
                opposed_normal_cosines=opposition)


def retention_failures(row):
    topology = row['topology']
    failures = []
    if min(row['bearing_normal_N'][f] for f in ('ring', 'pinky')) <= .2:
        failures.append('lower_finger_bearing')
    if not topology['opposed_sides'] or not topology['opposed_normal_cosines'] or min(topology['opposed_normal_cosines']) < .5:
        failures.append('opposed_contacts')
    if not topology['complete']:
        failures.append('missing_desired_contact')
    elif not topology['axial_order']:
        failures.append('axial_order')
    if topology['index_pinky_span_m'] < .080:
        failures.append('index_pinky_span')
    if row['max_penetration_m'] > .0015:
        failures.append('deep_contact')
    if row['drift_deg'] > 5 and row['relative_angular_acceleration_deg_s2'] > 20:
        failures.append('accelerating_relative_rotation')
    return failures


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
    rows = []; first_failure = None
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
            row['topology'] = bearing_topology(row['contacts'])
            window = (rows+[row])[-12:]
            row['relative_angular_acceleration_deg_s2'] = float(np.polyfit(
                np.arange(len(window))/120, [r['speed_deg_s'] for r in window], 1)[0]) if len(window) == 12 else 0.
            stream.write(json.dumps(row)+'\n'); rows.append(row)
            if 50 <= row['reference_tilt_deg'] <= 90:
                failures = retention_failures(row)
                if failures:
                    first_failure = dict(frame=frame, reasons=failures, observation=row)
                    print('FIRST_FAILURE', frame, failures, flush=True)
                    break
            if frame % 20 == 0: print('NATIVE', frame, row['reference_tilt_deg'], row['actual_tilt_deg'], row['drift_deg'], row['bearing_normal_N'], flush=True)
    np.savez_compressed(args.output/'trace.npz', qpos=poses, qvel=velocities, ctrl=controls)
    interval = [r for r in rows if 50 <= r['reference_tilt_deg'] <= 90]
    maintained = bool(interval) and first_failure is None and len(poses)-1 == end
    result = dict(accepted=False, full_action_passed_twice=False, focused_mechanism_passed=maintained,
                  end_frame=len(poses)-1, intended_end_frame=end, first_failure=first_failure, capacity_high_water=high, external_forces_zero=True,
                  post_frame0_state_writes=False, finite=True,
                  target_max_error=float(np.max(abs(np.asarray(controls)-target[:len(controls)]))),
                  interval_frames=[r['frame'] for r in interval],
                  first_missing_lower_contact=next((r['frame'] for r in interval if min(r['bearing_normal_N'][f] for f in ('ring', 'pinky'))<=.2), None),
                  peak_interval_drift_deg=max((r['drift_deg'] for r in interval), default=None))
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('RESULT', json.dumps(result), flush=True)


def audit_saved(args, mj, model, teacher):
    geometry = SurfaceGeometry(mj, model, json.loads((args.output/'fit.json').read_text())['patches'])
    trace = np.load(args.output/'trace.npz')
    with gzip.open(args.output/'telemetry.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        row['topology'] = bearing_topology(row['contacts'])
    end = rows[-1]; frame = end['frame']
    distance, points, normals, pen = geometry(trace['qpos'][frame])
    result = dict(raw_native_result_preserved=True, physical_rerun=False,
                  end_frame=frame, corrected_end_topology=end['topology'],
                  corrected_end_failures=retention_failures(end),
                  first_missing_middle_after380=next((r['frame'] for r in rows if r['frame'] >= 380 and r['bearing_normal_N']['middle'] <= .2), None),
                  first_middle_zero_after380=next((r['frame'] for r in rows if r['frame'] >= 380 and r['bearing_normal_N']['middle'] <= .01), None),
                  actual_end_surface_gaps_mm=(distance*1000).tolist(),
                  actual_end_surface_points_mm=(points*1000).tolist(),
                  end_joint_target_error_rad=(trace['ctrl'][frame, 6:28]-trace['qpos'][frame, 6:28]).tolist(),
                  reference_interval_sample_count=sum(50 <= r['reference_tilt_deg'] <= 90 for r in rows))
    (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def rebalance(args, mj, model, teacher):
    """One measured-geometry correction; only thumb/middle actuator targets."""
    source = args.balance_from
    archive = np.load(source/'trace.npz')
    q = archive['qpos'][-1].copy()
    patches = json.loads((source/'fit.json').read_text())['patches']
    geometry = SurfaceGeometry(mj, model, patches)
    joints = np.r_[np.arange(6, 12), np.arange(16, 20)]
    before = geometry(q)[0]
    goal = np.array([-.0009, -.0004])
    def residual(delta):
        trial = q.copy(); trial[joints] += delta
        distance = geometry(trial)[0][[0, 2]]
        return np.r_[(distance-goal)/.0001, .01*delta/.05]
    lower = np.maximum(-.05, model.jnt_range[joints, 0]-q[joints]+1e-6)
    upper = np.minimum(.05, model.jnt_range[joints, 1]-q[joints]-1e-6)
    sol = least_squares(residual, np.zeros(len(joints)), bounds=(lower, upper),
                        diff_step=1e-4, max_nfev=100, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    trial = q.copy(); trial[joints] += sol.x
    after = geometry(trial)[0]
    target = np.load(source/'target.npy').copy()
    f = np.arange(len(target)); onset = smooth((f-350)/30)
    fade = phase_weights(len(target))[:, joints]
    target[:, joints] += onset[:, None]*fade*sol.x
    target = np.clip(target, *model.actuator_ctrlrange.T)
    unchanged = np.r_[np.arange(6), np.arange(12, 16), np.arange(20, 28)]
    assert np.array_equal(target[:, unchanged], np.load(source/'target.npy')[:, unchanged])
    np.save(args.output/'target.npy', target)
    (args.output/'fit.json').write_text((source/'fit.json').read_text())
    report = dict(source=str(source), measured_frame=len(archive['qpos'])-1,
                  joints=joints.tolist(), delta_rad=sol.x.tolist(), before_gap_mm=(before*1000).tolist(),
                  after_gap_mm=(after*1000).tolist(), onset_frames=[350, 380],
                  unchanged_wrist_index_ring_pinky=True, parameter_sweep=False)
    (args.output/'balance.json').write_text(json.dumps(report, indent=2)+'\n')
    print('BALANCE', json.dumps(report), flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'donor', 'base', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed-root', type=Path, default=Path('outputs/action005_reference_contacts_v1'))
    parser.add_argument('--balance-from', type=Path)
    parser.add_argument('--audit-existing', action='store_true')
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
    if args.audit_existing:
        audit_saved(args, mj, model, teacher)
        return
    args.output.mkdir(parents=True, exist_ok=args.replay_existing)
    if args.replay_existing:
        if (args.output/'trace.npz').exists(): raise FileExistsError('preserve previous replay')
        target = np.load(args.output/'target.npy')
    elif args.balance_from:
        target = rebalance(args, mj, model, teacher)
    else:
        target = fit(args, mj, inp, model, teacher)
    (args.output/'contract.json').write_text(json.dumps(dict(setting_sha256=digest, reference='A_row030', initial_qpos=inp.initial['qpos'].tolist(), initial_qvel=inp.initial['qvel'].tolist(), source_base=str(args.base), source_target_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest()), indent=2)+'\n')
    if not args.fit_only: replay(args, mj, inp, model, teacher, provenance, target)


if __name__ == '__main__':
    main()
