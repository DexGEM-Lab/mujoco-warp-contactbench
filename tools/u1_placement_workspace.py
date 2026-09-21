#!/usr/bin/env python3
"""Bounded absolute-target edits with a locked prefix and fresh U1 frame0 replays.

Recipes contain inclusive window [start, end] and full 28D target knots. A knot
at start restores the original target and the end knot rejoins the original
stream. Checkpoints are diagnostic snapshots, never injected into acceptance
replays. Existing attempt directories cannot be overwritten.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sim.manorl.local_contact_repair import MODEL_FIELDS, array_sha, dump, sha
from tools.fit_uniform_direct_parent import metrics_for
from tools.pilot_start_augmentation import reconstruct
from tools.run_uniform_direct_parent_canary import (
    compile_model, replay_direct, DIRECT_CCD_ITERATIONS, DIRECT_WARP_CAPACITY,
)


def edit_window(base, recipe):
    base = np.asarray(base, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 28 or not np.isfinite(base).all():
        raise ValueError('base must be finite [N,28]')
    start, end = recipe['window']
    if not isinstance(start, int) or not isinstance(end, int) or not 0 < start < end < len(base):
        raise ValueError('invalid bounded window')
    frames = np.asarray([k['frame'] for k in recipe['knots']])
    values = np.asarray([k['target'] for k in recipe['knots']], dtype=np.float64)
    if (frames.ndim != 1 or len(frames) < 2 or not np.issubdtype(frames.dtype, np.integer)
            or values.shape != (len(frames), 28) or not np.isfinite(values).all()
            or frames[0] != start or frames[-1] != end or np.any(np.diff(frames) <= 0)):
        raise ValueError('invalid target knots')
    if not np.array_equal(values[0], base[start]) or not np.array_equal(values[-1], base[end]):
        raise ValueError('window endpoints must rejoin base exactly')
    out = base.copy()
    for i, (left, right) in enumerate(zip(frames[:-1], frames[1:])):
        u = np.linspace(0, 1, right-left+1)[:, None]
        # Smoothstep has zero endpoint speed: release before wrist withdrawal.
        u = u*u*(3-2*u)
        out[left:right+1] = values[i] + u*(values[i+1]-values[i])
    out[start] = base[start]
    out[end] = base[end]
    np.testing.assert_array_equal(out[:start+1], base[:start+1])
    np.testing.assert_array_equal(out[end:], base[end:])
    return out


def assert_u1(model, inp):
    import mujoco as mj
    assert inp.hz == 120 and model.nu == 28
    assert model.opt.timestep == 1/480
    assert model.opt.cone == mj.mjtCone.mjCONE_PYRAMIDAL and model.opt.impratio == 1
    assert model.opt.ccd_iterations == DIRECT_CCD_ITERATIONS, 'U1 requires the formal CCD16 contract'
    assert model.na == 0
    assert np.all(model.actuator_dyntype == mj.mjtDyn.mjDYN_NONE)
    assert np.all(model.actuator_gaintype == mj.mjtGain.mjGAIN_FIXED)
    assert np.all(model.actuator_biastype == mj.mjtBias.mjBIAS_AFFINE)
    np.testing.assert_array_equal(model.actuator_gainprm[:, 0], -model.actuator_biasprm[:, 1])
    for field in MODEL_FIELDS:
        np.testing.assert_array_equal(getattr(model, field), inp.manifest['native'][field])
    return dict(setting='U1', control_hz=120, physics_hz=480, substeps=4,
                cone='pyramidal', impratio=1, indexing='arrival', frame0='prestep',
                actuator='native_position', post_frame0_state_writes=False,
                hidden_controller_state=False, ccd_iterations=DIRECT_CCD_ITERATIONS,
                single_world_warp_capacity=dict(DIRECT_WARP_CAPACITY))


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    inp, _, _, teacher, provenance = reconstruct(a.bundle, 'A_row035')
    _, model = compile_model(a.bundle, a.asset_root, inp, 'U1')
    contract = assert_u1(model, inp)
    base = np.load(a.base, allow_pickle=False)
    recipe = json.loads(a.recipe.read_text()) if a.recipe else None
    target = edit_window(base, recipe) if recipe else base.copy()
    assert target.shape == (inp.frames, 28) and np.isfinite(target).all()
    assert np.all(target >= model.jnt_range[:28, 0]) and np.all(target <= model.jnt_range[:28, 1])
    if a.verify:
        prior = json.loads((a.verify/'result.json').read_text())
        assert prior['physical_pose_pass'], 'verification requires a passing candidate'
        np.testing.assert_array_equal(target, np.load(a.verify/'target.npy'))
    np.save(a.output/'target.npy', target)
    dump(a.output/'recipe.json', recipe)
    dump(a.output/'provenance.json', dict(contract=contract, base_sha256=sha(a.base),
         source=provenance, model_array_hashes={k: array_sha(getattr(model,k)) for k in MODEL_FIELDS},
         verification_of=str(a.verify) if a.verify else None))
    trace = replay_direct(inp, model, target)
    np.testing.assert_array_equal(trace['ctrl_substeps'], np.repeat(trace['ctrl'][1:,None,:],4,axis=1))
    # command_target may pick equivalent Euler branches; no extra control offsets.
    delta = trace['ctrl'] - target
    delta[:,3:6] = np.arctan2(np.sin(delta[:,3:6]), np.cos(delta[:,3:6]))
    np.testing.assert_allclose(delta, 0, atol=1e-6)
    assert not np.any(trace['wrist_integral']) and not np.any(trace['finger_target_delta'])
    result, diagnostic = metrics_for(inp, model, trace, teacher, 0, target)
    result['contract'] = contract
    count = diagnostic['arrays']['finger_ray_count']
    checkpoint = recipe['window'][0] if recipe else 0
    clear = np.flatnonzero((np.arange(len(count)) > checkpoint) & (count == 0))
    result['first_post_checkpoint_zero_contact_frame'] = int(clear[0]) if len(clear) else None
    result['permanent_contact_clear_frame'] = next(
        (int(f) for f in clear if not np.any(count[f:])), None)
    if recipe:
        result['edit_window'] = recipe['window']
        result['locked_target_prefix_sha256'] = array_sha(target[:checkpoint+1])
        np.savez_compressed(a.output/'checkpoint_diagnostic_only.npz',
                            frame=checkpoint, qpos=trace['qpos'][checkpoint],
                            qvel=trace['qvel'][checkpoint])
    if a.prefix_trace:
        assert recipe is not None
        prefix = np.load(a.prefix_trace)['qpos'][:recipe['window'][0]+1]
        error = float(np.max(np.abs(trace['qpos'][:len(prefix)]-prefix)))
        result['locked_prefix_max_qpos_error'] = error
        result['locked_prefix_reproduced'] = error < 1e-6
    np.savez_compressed(a.output/'trace.npz', **trace)
    np.savez_compressed(a.output/'contact_validation.npz', **diagnostic['arrays'])
    dump(a.output/'contacts.json', diagnostic['contacts'])
    dump(a.output/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--asset-root', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--recipe', type=Path)
    p.add_argument('--prefix-trace', type=Path)
    p.add_argument('--verify', type=Path)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
