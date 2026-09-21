"""Bounded action005 planted-grip diagnostic; never an accepted action replay.

Contact/actuator telemetry is sampled at 120 Hz (the last native 480 Hz
substep), with post-integration pose and finite-difference rates. Capacity
checks run at every substep. No claim is made about inter-sample force peaks.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation as R
from scipy.stats import qmc

U1 = 'c6db552f105d4de493880a52518772dbacad9496b105d2f4e43e87a907e692cd'
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
TILTS = (50., 70., 90., 110., 122.5)


def candidates():
    """Baseline plus 31 fixed low-discrepancy geometries, not squeeze multiples.

    Translation and rotation bounds are Euclidean; every finger joint gets an
    independent coordinate. The first seven samples explicitly favor axial
    shifts/rocking; the remaining coordinates still change fingers independently.
    """
    values = 2*qmc.Sobol(28, scramble=True, seed=847).random_base2(5)-1
    values[0] = 0
    out = []
    for i, v in enumerate(values):
        translation = v[:3]*.020/max(1., np.linalg.norm(v[:3]))
        rotation = v[3:6]*np.deg2rad(15)/max(1., np.linalg.norm(v[3:6]))
        if 0 < i < 8:
            translation = np.array([0., 0., (-1)**i * .005*((i+1)//2)])
            rotation = np.array([(-1)**i*np.deg2rad(3*(i//2)), 0., 0.])
        out.append(dict(id=i, translation_m=translation.tolist(),
                        rotation_vector_rad=rotation.tolist(), finger_delta_rad=(v[6:]*.3).tolist()))
    return out


def map_grip(grip, destination, adr, candidate):
    """Keep seed hand pose object-relative; rotate offsets in object frame."""
    q = np.array(destination, dtype=float, copy=True)
    source_R = R.from_quat(grip[adr+3:adr+7], scalar_first=True)
    dest_R = R.from_quat(q[adr+3:adr+7], scalar_first=True)
    relative_p = source_R.inv().apply(grip[:3]-grip[adr:adr+3])
    relative_R = source_R.inv()*R.from_euler('XYZ', grip[3:6])
    q[:3] = q[adr:adr+3]+dest_R.apply(relative_p+candidate['translation_m'])
    q[3:6] = (dest_R*R.from_rotvec(candidate['rotation_vector_rad'])*relative_R).as_euler('XYZ')
    q[6:28] = grip[6:28]
    return q


def critical_frames(qpos, adr):
    rotations = R.from_quat(qpos[:, adr+3:adr+7], scalar_first=True)
    tilts = np.rad2deg(np.arccos(np.clip(rotations.as_matrix()[:, 2, 2], -1, 1)))
    # First inversion branch only, not return-upright duplicates.
    peak = int(np.argmax(tilts))
    return [dict(requested_tilt_deg=t, frame=int(np.argmin(abs(tilts[:peak+1]-t))),
                 actual_tilt_deg=float(tilts[np.argmin(abs(tilts[:peak+1]-t))]),
                 quaternion_wxyz=qpos[np.argmin(abs(tilts[:peak+1]-t)), adr+3:adr+7].tolist())
            for t in TILTS]


def contact_wrench(position, basis, local, sign, com):
    force = sign*np.asarray(basis).reshape(3, 3).T@np.asarray(local)[:3]
    torque = np.cross(np.asarray(position)-com, force)+sign*np.asarray(basis).reshape(3, 3).T@np.asarray(local)[3:]
    return np.r_[force, torque]


def pyramid_utilisation(local, friction, dimension):
    local = np.asarray(local)
    if local[0] <= 1e-8:
        return 0.
    return float(np.sum(abs(local[1:dimension])/np.asarray(friction)[:dimension-1])/local[0])


def wrench_reserve(contacts, com, gravity_force):
    """Maximum gravity load factor using measured normal-force ceilings.

    Exact pyramidal generator LP including torsional/rolling dimensions. This
    is a conservative instantaneous static capacity, not an actuator capability
    extrapolation. Reserve >0 means capacity above one bottle weight.
    """
    columns, ceilings, groups = [], [], []
    for c in contacts:
        if c['finger'] is None or c['local_wrench'][0] <= 1e-8:
            continue
        start = len(columns)
        for axis in range(1, c['dimension']):
            for sign in (-1, 1):
                local = np.zeros(6); local[0] = 1
                local[axis] = sign*c['friction'][axis-1]
                columns.append(contact_wrench(c['position'], c['basis'], local, c['object_sign'], com))
        groups.append((start, len(columns))); ceilings.append(c['local_wrench'][0])
    if not columns:
        return -1.
    matrix = np.stack(columns, axis=1)
    gravity = np.r_[gravity_force, np.zeros(3)]
    inequality = np.zeros((len(groups), len(columns)+1))
    for row, (start, end) in enumerate(groups):
        inequality[row, start:end] = 1
    objective = np.zeros(len(columns)+1); objective[-1] = -1
    result = linprog(objective, A_ub=inequality, b_ub=ceilings,
                     A_eq=np.column_stack([matrix, gravity]), b_eq=np.zeros(6),
                     bounds=[(0, None)]*len(columns)+[(0, 10)], method='highs')
    if not result.success:
        raise RuntimeError(f'wrench LP failed: {result.message}')
    return float(result.x[-1]-1)


def score(rows, stop):
    window = [r for r in rows if r['frame'] > 120]
    complete = len(rows) == 240 and stop == 'duration'
    five_fraction = float(np.mean([r['five_bearing'] for r in window])) if window else 0.
    terminal = rows[-1]
    gates = dict(full_two_seconds=complete, five_bearing_fraction=five_fraction >= .95,
                 drift=bool(window) and max(r['drift_m'] for r in window) < .003 and max(r['drift_deg'] for r in window) < 3,
                 terminal_rate=terminal['speed_m_s'] < .001 and terminal['speed_deg_s'] < 1,
                 no_saturation=not any(r['saturated'] for r in rows),
                 no_support=not any(r['support'] for r in rows),
                 friction_reserve=bool(window) and min(r['friction_reserve'] for r in window) > 1e-6,
                 wrench_reserve=bool(window) and min(r['wrench_reserve'] for r in window) > 1e-6)
    return dict(passed=all(gates.values()), gates=gates, five_fraction=five_fraction,
                establishment_fraction=float(np.mean([r['five_bearing'] for r in rows])),
                survived_s=len(rows)/120, terminal_drift_mm=terminal['drift_m']*1000,
                terminal_drift_deg=terminal['drift_deg'],
                terminal_net_torque_Nm=terminal['net_wrench'][3:],
                terminal_wrench_reserve=terminal['wrench_reserve'], stop_reason=stop)


def observe(s, baseline, previous, target, frame):
    m, d = s.model, s.cpu
    q = s.data.qpos.numpy()[0].copy(); v = s.data.qvel.numpy()[0].copy()
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise RuntimeError('nonfinite native state')
    d.qpos[:] = q; d.qvel[:] = v; s.mj.mj_forward(m, d)
    hr = d.xmat[s.hand_body].reshape(3, 3)
    relative_p = hr.T@(d.xpos[s.object_body]-d.xpos[s.hand_body])
    relative_R = hr.T@d.xmat[s.object_body].reshape(3, 3)
    drift = np.linalg.norm(relative_p-baseline[0])
    angle = np.rad2deg(R.from_matrix(relative_R@baseline[1].T).magnitude())
    speed = np.linalg.norm(relative_p-previous[0])*120
    angular_speed = np.rad2deg(R.from_matrix(relative_R@previous[1].T).magnitude())*120
    n = int(s.data.nacon.numpy().ravel()[0])
    ids = s.wp.array(np.arange(n, dtype=np.int32), dtype=s.wp.int32)
    forces = s.wp.zeros(n, dtype=s.wp.spatial_vector)
    if n: s.mw.contact_force(s.wm, s.data, ids, False, forces)
    local = forces.numpy(); native = s.data.contact
    arrays = {name: getattr(native, name).numpy()[:n] for name in ('geom', 'pos', 'frame', 'friction', 'dim', 'dist', 'efc_address', 'worldid')}
    # Native contact solve and native xipos share the pre-integration epoch.
    com = s.data.xipos.numpy()[0, s.object_body].copy()
    contacts = []; by_link = {}; bearing = dict.fromkeys(FINGERS, 0.); hand = np.zeros(6); other = np.zeros(6)
    for i, pair in enumerate(arrays['geom']):
        b1, b2 = [int(m.geom_bodyid[g]) for g in pair]
        if s.object_body not in (b1, b2): continue
        sign = 1 if b2 == s.object_body else -1
        body = b1 if sign == 1 else b2; name = m.body(body).name
        finger = name.split('_')[0] if name.split('_')[0] in FINGERS else None
        is_hand = body != 0 and name not in s.names
        c = dict(link=name, finger=finger, hand=is_hand, object_sign=sign,
                 geom_ids=pair.tolist(), position=arrays['pos'][i].tolist(), basis=arrays['frame'][i].tolist(),
                 friction=arrays['friction'][i].tolist(), dimension=int(arrays['dim'][i]),
                 distance_m=float(arrays['dist'][i]), efc_address=np.asarray(arrays['efc_address'][i]).tolist(),
                 worldid=int(arrays['worldid'][i]), local_wrench=local[i].tolist())
        c['utilisation'] = pyramid_utilisation(local[i], c['friction'], c['dimension'])
        wrench = contact_wrench(c['position'], c['basis'], local[i], sign, com)
        c['world_wrench_about_com'] = wrench.tolist()
        by_link.setdefault(name, np.zeros(6)); by_link[name] += wrench
        if is_hand: hand += wrench
        else: other += wrench
        if finger: bearing[finger] += max(0., float(local[i, 0]))
        contacts.append(c)
    gravity = m.body_mass[s.object_body]*m.opt.gravity
    actuator = s.data.actuator_force.numpy()[0]
    limited = m.actuator_forcelimited.astype(bool)
    saturation = np.any(limited & ((actuator <= m.actuator_forcerange[:, 0]+1e-5) | (actuator >= m.actuator_forcerange[:, 1]-1e-5)))
    ctrl = s.data.ctrl.numpy()[0].copy()
    error = ctrl-target; error[3:6] = (error[3:6]+np.pi)%(2*np.pi)-np.pi
    control_limited = m.actuator_ctrllimited.astype(bool)
    saturation = bool(saturation or np.max(abs(error)) > 1e-5 or
                      np.any(control_limited & ((ctrl <= m.actuator_ctrlrange[:, 0]+1e-6) |
                                                (ctrl >= m.actuator_ctrlrange[:, 1]-1e-6))))
    if np.max(abs(error)) > 1e-5: raise RuntimeError('applied control differs from requested fixed target')
    bearing_contacts = [c for c in contacts if c['hand'] and c['local_wrench'][0] > .01]
    friction_reserve = min([1-c['utilisation'] for c in bearing_contacts], default=-1.)
    support = any(not c['hand'] and c['local_wrench'][0] > 1e-6 for c in contacts)
    axial = [float((np.asarray(c['position'])-com)@d.xmat[s.object_body].reshape(3, 3)[:, 2]) for c in bearing_contacts]
    thumb = sum((value[:3] for name, value in by_link.items() if name.startswith('thumb_')), np.zeros(3))
    fingers = sum((value[:3] for name, value in by_link.items() if name.split('_')[0] in FINGERS[1:]), np.zeros(3))
    opposition = float(-thumb@fingers/max(1e-12, np.linalg.norm(thumb)*np.linalg.norm(fingers)))
    row = dict(frame=frame, drift_m=float(drift), drift_deg=float(angle), speed_m_s=float(speed), speed_deg_s=float(angular_speed),
               relative_position=relative_p.tolist(), relative_rotation=relative_R.tolist(),
               wrist_translation_from_initial=(q[:3]-baseline[2][:3]).tolist(),
               wrist_rotation_from_initial_deg=float(np.rad2deg((R.from_euler('XYZ', q[3:6])*R.from_euler('XYZ', baseline[2][3:6]).inv()).magnitude())),
               bearing_normal_N=bearing, five_bearing=all(value > .2 for value in bearing.values()),
               force_free=sum(bearing.values()) <= 1e-6, contacts=contacts,
               per_link_wrench={k: val.tolist() for k, val in by_link.items()},
               hand_wrench=hand.tolist(), other_wrench=other.tolist(), net_wrench=(hand+other+np.r_[gravity, np.zeros(3)]).tolist(),
               native_com=com.tolist(), friction_reserve=friction_reserve,
               wrench_reserve=wrench_reserve(contacts, com, gravity), support=support, saturated=saturation,
               actuator_force=actuator.tolist(), applied_control_error=float(max(abs(error))),
               max_penetration_m=max([max(0., -c['distance_m']) for c in contacts], default=0.),
               axial_contact_span_m=max(axial)-min(axial) if axial else 0., thumb_opposition_cosine=opposition)
    return row, (relative_p, relative_R), q, v, ctrl


def run(args):
    import mujoco as mj
    from tools.u1_interactive_session import Session
    from sim.manorl.mjx_sim import command_target
    sys.path.insert(0, str(args.source_root.resolve()))
    from run_teacher import load_teacher
    from run_reference import setting
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    I, m, _, teacher, provenance, _ = load_teacher('A_row030')
    common = setting(m, I.arrays['scene_object_names'].tolist())
    digest = hashlib.sha256(json.dumps(common, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != U1: raise RuntimeError(f'U1 mismatch: {digest}')
    seed_root = args.source_root/'action005_row847_full_U1/independent01'
    trace_path = seed_root/'trace.npz'; target_path = seed_root/'target.npy'
    with np.load(trace_path) as trace: grip = trace['qpos'][300].copy()
    seed_target = np.load(target_path)[300].copy()
    adr = int(m.joint('mayonnaisebottle_free').qposadr[0]); ob = m.body('mayonnaisebottle').id
    orientations = critical_frames(teacher['qpos'], adr)
    plan = candidates()
    metadata = dict(diagnostic_only=True, accepted_action005=False, setting_sha256=digest, setting=common,
                    orientations=orientations, candidates=plan, seed_frame=300,
                    source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (trace_path, target_path)},
                    telemetry_boundary=__doc__, controller='constant native target plus initial analytical payload gravity lead; no feedback')
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('SEARCH_READY', out, json.dumps(orientations), flush=True)
    summaries = []
    for candidate in plan:
        results = []
        for orientation in orientations:
            folder = out/f"candidate_{candidate['id']:02d}"/f"tilt_{orientation['requested_tilt_deg']:05.1f}"
            folder.mkdir(parents=True, exist_ok=False)
            q0 = map_grip(grip, teacher['qpos'][orientation['frame']], adr, candidate)
            target = q0[:28].copy()
            target[6:] = np.clip(seed_target[6:]+candidate['finger_delta_rad'], m.jnt_range[6:28, 0], m.jnt_range[6:28, 1])
            # Preserve realized seed pose at initialization; independent changes
            # act through native finger springs, not planted fingertip contacts.
            d = mj.MjData(m); d.qpos[:] = q0; mj.mj_forward(m, d)
            jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
            mj.mj_jac(m, d, jp, jr, d.xipos[ob], int(m.jnt_bodyid[5]))
            target[:6] += jp[:, :6].T@(-m.body_mass[ob]*m.opt.gravity)/m.actuator_gainprm[:6, 0]
            resolved = command_target(target, q0[:28], *m.jnt_range[:28].T)
            if np.max(abs(resolved[6:]-seed_target[6:])) > .300001:
                raise RuntimeError('native clipping exceeds seed-relative finger bound')
            target = resolved
            targets = np.repeat(target[None], 241, axis=0)
            inp = SimpleNamespace(row_id='005_critical_diagnostic', frames=241, hz=120,
                                  arrays=deepcopy(I.arrays), metrics=deepcopy(I.metrics), initial={'qpos': q0, 'qvel': np.zeros(m.nv)},
                                  manifest=I.manifest, provenance=I.provenance)
            s = Session(inp, m, targets, {'qpos': np.repeat(q0[None], 241, axis=0)}, provenance)
            hr = d.xmat[s.hand_body].reshape(3, 3)
            baseline = (hr.T@(d.xpos[ob]-d.xpos[s.hand_body]), hr.T@d.xmat[ob].reshape(3, 3), q0)
            previous = baseline[:2]; rows = []; poses = [q0]; velocities = [np.zeros(m.nv)]; controls = [target]
            stop = 'duration'; free_frames = 0
            np.save(folder/'target.npy', targets)
            with gzip.open(folder/'telemetry.jsonl.gz', 'wt') as stream:
                for frame in range(1, 241):
                    s.frame = frame
                    # One absolute target, unchanged across all four substeps.
                    s.data.ctrl.assign(target[None].astype(np.float32))
                    for _ in range(4):
                        s.mw.step(s.wm, s.data); s.wp.synchronize()
                        if int(s.data.nacon.numpy().max()) >= 1024 or int(s.data.nefc.numpy().max()) >= 4096:
                            raise RuntimeError('native capacity reached')
                    row, previous, q, v, ctrl = observe(s, baseline, previous, target, frame)
                    rows.append(row); poses.append(q); velocities.append(v); controls.append(ctrl)
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                    free_frames = free_frames+1 if row['force_free'] else 0
                    if row['drift_m'] > .020 or row['drift_deg'] > 20: stop = 'relative_drift'
                    elif row['support']: stop = 'unintended_support'
                    elif free_frames >= 12: stop = '12_force_free_frames'
                    if stop != 'duration': break
            np.savez_compressed(folder/'trace.npz', qpos=poses, qvel=velocities, ctrl=controls)
            result = dict(orientation=orientation, **score(rows, stop))
            (folder/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            results.append(result)
            print('ATTEMPT', candidate['id'], orientation['actual_tilt_deg'], json.dumps(result), flush=True)
            del s
        summary = dict(candidate_id=candidate['id'], passed=all(r['passed'] for r in results), orientations=results,
                       worst_five_fraction=min(r['five_fraction'] for r in results),
                       worst_establishment_fraction=min(r['establishment_fraction'] for r in results),
                       failing_orientations=[r['orientation']['actual_tilt_deg'] for r in results if not r['passed']],
                       worst_survived_s=min(r['survived_s'] for r in results),
                       worst_drift_mm=max(r['terminal_drift_mm'] for r in results))
        summaries.append(summary)
        ranked = sorted(summaries, key=lambda x: (x['passed'], x['worst_five_fraction'], x['worst_establishment_fraction'], x['worst_survived_s'], -x['worst_drift_mm']), reverse=True)
        (out/'ranking.json').write_text(json.dumps(dict(complete=len(summaries)==32, diagnostic_only=True, ranked=ranked), indent=2)+'\n')
    print('SEARCH_COMPLETE', out/'ranking.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('outputs/four_action_single_v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.is_dir():
            (args.output/'ERROR.json').write_text(json.dumps(dict(error=repr(error), stopped=True), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
