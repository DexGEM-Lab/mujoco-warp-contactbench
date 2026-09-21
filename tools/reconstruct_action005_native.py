"""Fail-closed native-state reconstruction prerequisite for A030 wrench identification."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from tools.fit_action005_reference_contacts import FINGERS

FRAMES = (380, 400, 409)
TOLERANCES = dict(qpos=1e-6, qvel=1e-4, ctrl=0.)


def compare_trace(actual, archived):
    errors = {k: float(np.max(np.abs(actual[k]-archived[k][:len(actual[k])]))) for k in TOLERANCES}
    return dict(max_errors=errors, tolerances=TOLERANCES,
                bitwise={k: bool(np.array_equal(actual[k], archived[k][:len(actual[k])])) for k in TOLERANCES},
                passed=all(errors[k] <= TOLERANCES[k] for k in TOLERANCES))


def compare_forces(actual, archived):
    a = np.array([actual[f] for f in FINGERS]); b = np.array([archived[f] for f in FINGERS])
    return dict(max_error_N=float(abs(a-b).max()),
                passed=bool(np.all(abs(a-b) <= .02+.01*abs(b)) and np.array_equal(a > .2, b > .2)),
                actual_N=a.tolist(), archived_N=b.tolist())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def advance(s, target):
    s.frame += 1
    s.data.ctrl.assign(target[None].astype(np.float32))
    for _ in range(4):
        s.mw.step(s.wm, s.data); s.wp.synchronize()
        for key, limit in (('nacon', 1024), ('ncollision', 256), ('nefc', 4096)):
            if int(getattr(s.data, key).numpy().max()) >= limit:
                raise RuntimeError('capacity exceeded: '+key)
        if np.any(s.data.xfrc_applied.numpy()) or np.any(s.data.qfrc_applied.numpy()):
            raise RuntimeError('external force detected')
    if not np.isfinite(s.data.qpos.numpy()).all() or not np.isfinite(s.data.qvel.numpy()).all():
        raise RuntimeError('nonfinite native state')


def baseline(s):
    d = s.cpu; d.qpos[:] = s.data.qpos.numpy()[0]; d.qvel[:] = s.data.qvel.numpy()[0]
    s.mj.mj_forward(s.model, d)
    hr = d.xmat[s.hand_body].reshape(3, 3)
    return (hr.T@(d.xpos[s.object_body]-d.xpos[s.hand_body]),
            hr.T@d.xmat[s.object_body].reshape(3, 3), d.qpos.copy())


def run(args):
    from tools.u1_interactive_session import Session
    from tools.search_action005_critical_grip import U1, observe
    import mujoco as mj
    sys.path.insert(0, str(args.source_root.resolve()))
    from run_teacher import load_teacher
    from run_reference import setting
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    inp, model, _, teacher, provenance, _ = load_teacher('A_row030')
    target = np.load(args.source/'target.npy'); archived = np.load(args.source/'trace.npz')
    contract = json.loads((args.source/'contract.json').read_text())
    digest = hashlib.sha256(json.dumps(setting(model, inp.arrays['scene_object_names'].tolist()), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != U1 or digest != contract['setting_sha256']:
        raise RuntimeError('nominal U1 setting mismatch')
    if not all(np.array_equal(inp.initial[k], contract['initial_'+k]) for k in ('qpos', 'qvel')):
        raise RuntimeError('archived initial state mismatch')
    mj.mj_saveModel(model, str(out/'model.mjb'), None)
    manifest = dict(setting_sha256=digest, model_sha256=hashlib.sha256((out/'model.mjb').read_bytes()).hexdigest(),
                    source=str(args.source), target_sha256=hashlib.sha256((args.source/'target.npy').read_bytes()).hexdigest(),
                    trace_sha256=hashlib.sha256((args.source/'trace.npz').read_bytes()).hexdigest(),
                    historical_model_hash_available=False,
                    checkpoint_identity='frozen v3 prefix reconstruction; identity unqualified until reconstruction.json passes',
                    frames=FRAMES, tolerances=TOLERANCES, force_tolerance='0.02 N + 1% per finger; same >0.2N bearing mask',
                    prediction='Small thumb unloading and middle normal loading should be locally controllable if servo compliance is the cause; mixed-sign or scale-dependent derivatives refute a smooth local inverse.',
                    probe_horizon_frames=8, probe_amplitudes_rad=[.004, .002],
                    accepted=False, complete_passes=0)
    dump(out/'manifest.json', manifest)
    s = Session(inp, model, target, teacher, provenance)
    poses = [s.data.qpos.numpy()[0].copy()]; velocities = [s.data.qvel.numpy()[0].copy()]; controls = [s.data.ctrl.numpy()[0].copy()]
    with gzip.open(args.source/'telemetry.jsonl.gz', 'rt') as stream:
        oldrows = {r['frame']: r for r in map(json.loads, stream)}
    force_checks = {}; rows = []
    for frame in range(1, max(FRAMES)+1):
        advance(s, target[frame])
        poses.append(s.data.qpos.numpy()[0].copy()); velocities.append(s.data.qvel.numpy()[0].copy()); controls.append(s.data.ctrl.numpy()[0].copy())
        if frame in oldrows:
            b = baseline(s); row, _, _, _, _ = observe(s, b, b[:2], target[frame], frame)
            rows.append(row); force_checks[frame] = compare_forces(row['bearing_normal_N'], oldrows[frame]['bearing_normal_N'])
        if frame in FRAMES:
            s.save_checkpoint(out/f'checkpoint_{frame}')
            print('CHECKPOINT', frame, force_checks[frame], flush=True)
    actual = dict(qpos=np.array(poses), qvel=np.array(velocities), ctrl=np.array(controls))
    np.savez_compressed(out/'reconstruction_trace.npz', **actual)
    with gzip.open(out/'reconstruction_telemetry.jsonl.gz', 'wt') as stream:
        for row in rows: stream.write(json.dumps(row)+'\n')
    report = compare_trace(actual, archived)
    report.update(force_checks=force_checks, external_forces_zero=True, post_frame0_state_writes=False)
    report['passed'] = report['passed'] and all(c['passed'] for c in force_checks.values())
    dump(out/'reconstruction.json', report)
    if not report['passed']:
        dump(out/'result.json', dict(stopped='reconstruction_mismatch', accepted=False, complete_passes=0, sensitivity_probes=0, max_errors=report['max_errors']))
        print('STOP reconstruction_mismatch', report['max_errors'], flush=True)
        return
    dump(out/'result.json', dict(stopped='reconstruction_qualified', accepted=False,
         complete_passes=0, sensitivity_probes=0, checkpoint_frames=FRAMES))
    print('RECONSTRUCTION_QUALIFIED; diagnostic probes require a separate invocation', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path('outputs/four_action_single_v1'))
    p.add_argument('--source', type=Path, default=Path('outputs/action005_reference_contacts_v3'))
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__': main()
