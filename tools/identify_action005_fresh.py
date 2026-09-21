"""One fresh U1 baseline and same-process full-buffer central sensitivities."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from tools.reconstruct_action005_native import advance, baseline, dump, FRAMES
from tools.fit_action005_reference_contacts import FINGERS, SurfaceGeometry, bearing_topology

# Output units: N, mm, N/Nm, deg/s². Scaling is fixed before observing probes.
SCALES = np.r_[np.ones(5), np.ones(5), np.ones(3), np.full(3, .02), np.full(3, 100.)]
LABELS = ([f'force_{f}_N' for f in FINGERS] + [f'gap_{f}_mm' for f in FINGERS]
          + ['net_Fx_N', 'net_Fy_N', 'net_Fz_N', 'net_Tx_Nm', 'net_Ty_Nm', 'net_Tz_Nm']
          + ['relative_alpha_x_deg_s2', 'relative_alpha_y_deg_s2', 'relative_alpha_z_deg_s2'])


def qualify_basin(rows, archived):
    """Compare the qualitative basin, not historical native-state identity."""
    selected = [r for r in rows if r['frame'] >= 380]
    endpoint = selected[-1]
    spans = [r['topology']['index_pinky_span_m'] for r in selected]
    lower = all(min(r['bearing_normal_N'][f] for f in ('index', 'ring', 'pinky')) > .2
                and r['topology']['opposed_sides']
                and min(r['topology']['opposed_normal_cosines'], default=0) > .5 for r in selected)
    middle = endpoint['bearing_normal_N']['middle'] < .2 and selected[0]['bearing_normal_N']['middle'] > .5
    # Explicit supervisor clarification: historical checkpoint spans are <80mm.
    # Require no collapse and <=1mm change vs same archived checkpoint geometry.
    differences = {str(r['frame']): 1000*(r['topology']['index_pinky_span_m']-
                    archived[r['frame']]['topology']['index_pinky_span_m'])
                   for r in selected if r['frame'] in FRAMES}
    passed = lower and middle and min(spans) >= .078 and all(abs(v) <= 1 for v in differences.values())
    return dict(passed=bool(passed), lower_opposed_retained=bool(lower), middle_unloads=bool(middle),
                min_span_mm=min(spans)*1000, endpoint_span_mm=spans[-1]*1000,
                checkpoint_span_difference_mm=differences,
                all_spans_ge80mm=bool(min(spans) >= .080))


def consistency(zero, plus4, minus4, plus2, minus2):
    """Reject curvature or scale-dependent slopes above30%; no fitted tolerance."""
    d4 = (plus4-minus4)/.008; d2 = (plus2-minus2)/.004
    norm = lambda x: float(np.linalg.norm(x/SCALES))
    scale = norm(d4-d2)/max(norm(d2), 1e-12)
    symmetry4 = norm(plus4+minus4-2*zero)/max(norm(plus4-minus4), 1e-12)
    symmetry2 = norm(plus2+minus2-2*zero)/max(norm(plus2-minus2), 1e-12)
    # Also reject task-critical force/gap curvature obscured by other outputs.
    critical = []
    for indices in ([0, 2], [5, 7]):
        ix = np.asarray(indices)
        n = lambda x: float(np.linalg.norm(x[ix]/SCALES[ix]))
        critical.append(max(n(d4-d2)/max(n(d2), 1e-12),
                            n(plus4+minus4-2*zero)/max(n(plus4-minus4), 1e-12),
                            n(plus2+minus2-2*zero)/max(n(plus2-minus2), 1e-12)))
    return dict(scale_error=scale, symmetry_error_004=symmetry4, symmetry_error_002=symmetry2,
                force_gap_errors=critical, passed=max(scale, symmetry4, symmetry2, *critical) <= .30,
                derivative_per_rad=d2.tolist())


def normal_basis(geometry, q):
    columns = []
    for finger, joints in ((0, np.arange(6, 12)), (2, np.arange(16, 20))):
        gradient = []
        for j in joints:
            plus, minus = q.copy(), q.copy(); plus[j] += 1e-4; minus[j] -= 1e-4
            gradient.append((geometry(plus)[0][finger]-geometry(minus)[0][finger])/2e-4)
        gradient = np.asarray(gradient)
        if np.linalg.norm(gradient) < 1e-5: raise ValueError('normal direction degenerate')
        column = np.zeros(28); column[joints] = gradient/np.linalg.norm(gradient)
        columns.append(column)
    return np.column_stack(columns)


def record(s, geometry, origin, previous, target):
    from tools.search_action005_critical_grip import observe
    row, previous, q, v, ctrl = observe(s, origin, previous, target, s.frame)
    rot = s.data.xmat.numpy()[0, s.object_body].reshape(3, 3)
    pos = s.data.xpos.numpy()[0, s.object_body]
    for c in row['contacts']:
        c['object_local_position'] = (rot.T@(np.asarray(c['position'])-pos)).tolist()
        c['object_local_normal'] = (rot.T@(c['object_sign']*np.asarray(c['basis']).reshape(3, 3)[0])).tolist()
    row['topology'] = bearing_topology(row['contacts'])
    row['surface_gaps_mm'] = (geometry(q)[0]*1000).tolist()
    return row, previous, q, v, ctrl


def response(rows, origin):
    """Last4 mean outputs; quadratic SO(3) displacement fit over8 frames."""
    rotations = np.array([R.from_matrix(np.asarray(r['relative_rotation'])@origin[1].T).as_rotvec()
                          for r in rows])
    # Include exact initial zero to anchor finite-horizon acceleration.
    alpha = 2*np.polyfit(np.arange(9)/120, np.vstack([np.zeros(3), rotations]), 2)[0]*180/np.pi
    means = np.mean([np.r_[[r['bearing_normal_N'][f] for f in FINGERS],
                          r['surface_gaps_mm'], r['net_wrench']] for r in rows[-4:]], axis=0)
    return np.r_[means, alpha]


def run(args):
    import mujoco as mj
    from tools.u1_interactive_session import Session, device_arrays
    from tools.search_action005_critical_grip import U1
    sys.path.insert(0, str(args.source_root.resolve()))
    from run_teacher import load_teacher
    from run_reference import setting
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    inp, model, _, teacher, provenance, _ = load_teacher('A_row030')
    target = np.load(args.source/'target.npy')
    contract = json.loads((args.source/'contract.json').read_text())
    digest = hashlib.sha256(json.dumps(setting(model, inp.arrays['scene_object_names'].tolist()), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == U1 == contract['setting_sha256']
    for k in ('qpos', 'qvel'): np.testing.assert_array_equal(inp.initial[k], contract['initial_'+k])
    mj.mj_saveModel(model, str(out/'model.mjb'), None)
    model_hash = hashlib.sha256((out/'model.mjb').read_bytes()).hexdigest()
    assert model_hash == json.loads((Path('outputs/action005_reference_contacts_v4')/'manifest.json').read_text())['model_sha256']
    manifest = dict(source=str(args.source), setting_sha256=digest, model_sha256=model_hash,
                    target_sha256=hashlib.sha256((args.source/'target.npy').read_bytes()).hexdigest(),
                    provenance=provenance, realization='new frame0, no historical identity requirement',
                    prediction='Normal thumb opening reduces thumb force/depth; middle closure increases middle force. Contact-mode transitions may invalidate an8-frame local inverse.',
                    frames=FRAMES, amplitudes_rad=[.004, .002], horizon_frames=8,
                    basis='two unit joint-space surface-gap gradients; positive opens',
                    output_labels=LABELS, output_scales=SCALES.tolist(), nonlinearity_limit=.30,
                    condition_limit=1000, complete_passes=0)
    dump(out/'manifest.json', manifest)
    geometry = SurfaceGeometry(mj, model, json.loads((args.source/'fit.json').read_text())['patches'])
    s = Session(inp, model, target, teacher, provenance)
    origin = baseline(s); previous = origin[:2]
    rows = []; snapshots = {}; traces = {k: [getattr(s.data, k).numpy()[0].copy()] for k in ('qpos', 'qvel', 'ctrl')}
    for frame in range(1, 419):
        advance(s, target[frame])
        for k in traces: traces[k].append(getattr(s.data, k).numpy()[0].copy())
        if frame == 380: origin = baseline(s); previous = origin[:2]
        row, previous, _, _, _ = record(s, geometry, origin, previous, target[frame]); rows.append(row)
        if frame in FRAMES: snapshots[frame] = s.save_checkpoint(out/f'checkpoint_{frame}')
        if frame % 40 == 0 or frame in FRAMES: print('BASELINE', frame, row['bearing_normal_N'], row['topology']['index_pinky_span_m'], flush=True)
    np.savez_compressed(out/'baseline_trace.npz', **{k: np.array(v) for k, v in traces.items()})
    with gzip.open(out/'baseline_telemetry.jsonl.gz', 'wt') as stream:
        for row in rows: stream.write(json.dumps(row)+'\n')
    with gzip.open(args.source/'telemetry.jsonl.gz', 'rt') as stream: archived = {r['frame']: r for r in map(json.loads, stream)}
    qualification = qualify_basin(rows, archived)
    qualification['first_bearing_frame'] = {f: next((r['frame'] for r in rows if r['bearing_normal_N'][f] > .2), None) for f in FINGERS}
    qualification['thumb_first_gt1N'] = next((r['frame'] for r in rows if r['bearing_normal_N']['thumb'] > 1), None)
    qualification['external_forces_zero'] = True; qualification['post_frame0_episode_state_writes'] = False
    dump(out/'baseline.json', qualification)
    if not qualification['passed']:
        dump(out/'result.json', dict(stopped='fresh_basin_absent', accepted=False, complete_passes=0, sensitivity_probes=0)); return
    reports = []
    for frame in FRAMES:
        snapshot = snapshots[frame]
        basis = normal_basis(geometry, snapshot['buffers']['qpos'][0])
        folder = out/f'probe_{frame}'; folder.mkdir()
        np.save(folder/'basis.npy', basis)
        def probe(name, delta):
            s.restore(snapshot)
            for key, array in device_arrays(s.data):
                np.testing.assert_array_equal(array.numpy(), snapshot['buffers'][key])
            origin = baseline(s); previous = origin[:2]; samples = []; states = []
            for step in range(1, 9):
                control = target[frame+step]+delta
                if np.any(control < model.actuator_ctrlrange[:, 0]) or np.any(control > model.actuator_ctrlrange[:, 1]):
                    raise ValueError('probe exceeds canonical controls')
                advance(s, control)
                row, previous, q, v, _ = record(s, geometry, origin, previous, control)
                samples.append(row); states.append(np.r_[q, v])
            with gzip.open(folder/f'{name}.jsonl.gz', 'wt') as stream:
                for row in samples: stream.write(json.dumps(row)+'\n')
            np.save(folder/f'{name}_states.npy', states)
            return response(samples, origin), np.asarray(states)
        zero, a = probe('zero', np.zeros(28))
        repeat, b = probe('zero_repeat', np.zeros(28))
        restore = dict(qpos_max=float(abs(a[:, :model.nq]-b[:, :model.nq]).max()),
                       qvel_max=float(abs(a[:, model.nq:]-b[:, model.nq:]).max()),
                       response_error=(repeat-zero).tolist(), buffer_count=len(snapshot['buffers']))
        restore['passed'] = restore['qpos_max'] <= 1e-6 and restore['qvel_max'] <= 1e-4
        directions = []
        for j, finger in enumerate(('thumb', 'middle')):
            values = [probe(f'{finger}_{amplitude:+.3f}', amplitude*basis[:, j])[0]
                      for amplitude in (.004, -.004, .002, -.002)]
            check = consistency(zero, *values); check['finger'] = finger
            check['responses'] = [v.tolist() for v in values]; directions.append(check)
        jac = np.array([d['derivative_per_rad'] for d in directions]).T
        singular = np.linalg.svd(jac/SCALES[:, None], compute_uv=False)
        # Two independent task-critical outputs must be controllable.
        task_singular = np.linalg.svd(jac[[2, 5]], compute_uv=False)
        condition = float(singular[0]/max(singular[-1], 1e-15))
        task_condition = float(task_singular[0]/max(task_singular[-1], 1e-15))
        signs = jac[0, 0] < 0 and jac[5, 0] > 0 and jac[2, 1] < 0
        passed = restore['passed'] and all(d['passed'] for d in directions) and condition < 1000 and task_condition < 1000 and signs
        report = dict(frame=frame, zero=zero.tolist(), restore=restore, directions=directions,
                      singular_values=singular.tolist(), condition=condition, task_singular_values=task_singular.tolist(),
                      task_condition=task_condition, expected_signs=bool(signs), passed=bool(passed))
        dump(folder/'sensitivity.json', report); reports.append(report)
        print('SENSITIVITY', frame, 'passed', passed, 'consistency', [(d['scale_error'], d['symmetry_error_004'], d['symmetry_error_002'], d['force_gap_errors']) for d in directions], flush=True)
    passed = all(r['passed'] for r in reports)
    dump(out/'sensitivity.json', dict(passed=passed, checkpoints=reports))
    dump(out/'result.json', dict(stopped='sensitivity_qualified' if passed else 'sensitivity_rejected',
         accepted=False, complete_passes=0, sensitivity_probes=24, zero_continuations=6,
         inverse_corrections=0, transfer_runs=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path('outputs/four_action_single_v1'))
    p.add_argument('--source', type=Path, default=Path('outputs/action005_reference_contacts_v3'))
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__': main()
