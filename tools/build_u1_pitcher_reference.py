#!/usr/bin/env python3
"""Rebuild a frozen row82 U1 target recipe; never edit assets or force rollout states.

The recipe records the measured fingertip-preload vector. It is data, not a
runtime feedback controller. Frozen-target replay/acceptance is a separate step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


def cubic(value):
    u = np.clip(value, 0, 1)
    return u**2 * (3 - 2*u)


def quintic(value):
    u = np.clip(value, 0, 1)
    return u**3 * (10 + u*(-15 + 6*u))


def hold_fingers(target, recipe):
    """Keep the acquired actuator targets, with explicit thumb-first release."""
    out = np.asarray(target, dtype=float).copy()
    preload = np.asarray(recipe['loop_preload_target_delta'], dtype=float)
    frame = recipe['grasp_frame']
    if out.ndim != 2 or out.shape[1] != 28 or not np.isfinite(out).all():
        raise ValueError('target must be finite [frames,28]')
    if preload.shape != (28,) or not np.isfinite(preload).all() or np.any(preload[:6]):
        raise ValueError('preload must be finite28D and cannot alter wrist targets')
    if not 0 < frame < len(out):
        raise ValueError('grasp frame outside target')
    f = np.arange(len(out))
    engage = cubic((f-frame)/recipe['engage_frames'])
    for start, stop, release in [(6,12,'thumb_release'), (12,28,'four_finger_release')]:
        end, ramp = recipe[release]
        if not frame < end-ramp < end < len(out):
            raise ValueError('release window must follow acquisition')
        weight = engage*cubic((end-f)/ramp)
        held = target[frame,start:stop] + preload[start:stop]
        out[:,start:stop] = target[:,start:stop] + weight[:,None]*(held-target[:,start:stop])
    return out


def position_registration(frames, recipe):
    """Smooth task-space registration of pour position and final placement."""
    f = np.arange(frames)
    pour = np.asarray(recipe['pour_offset_m'], dtype=float)
    land = np.asarray(recipe['landing_offset_m'], dtype=float)
    if pour.shape != (3,) or land.shape != (3,) or not np.isfinite([pour,land]).all():
        raise ValueError('offsets must be finite XYZ triples')
    enter, enter_length = recipe['pour_ramp']
    back, back_length = recipe['return_ramp']
    leave, leave_length = recipe['offset_release_ramp']
    if not (0 <= enter < enter+enter_length <= back < back+back_length <= leave < leave+leave_length < frames):
        raise ValueError('registration windows must be ordered and non-overlapping')
    offset = quintic((f-enter)/enter_length)[:,None]*pour
    offset += quintic((f-back)/back_length)[:,None]*(land-pour)
    offset *= 1-quintic((f-leave)/leave_length)[:,None]
    return offset


def build_target(model, reference, teacher, recipe):
    import mujoco as mj
    q = reference.copy()
    q[:,3:6] = np.unwrap(q[:,3:6], axis=0)
    velocity = savgol_filter(q,11,3,deriv=1,delta=1/120,axis=0,mode='interp')
    kp = model.actuator_gainprm[:,0]
    kv = -model.actuator_biasprm[:,2]
    raw = q + (kv/kp)*velocity + (model.dof_frictionloss[:28]/kp)*np.tanh(velocity/.03)
    body = model.body('pitcherbase').id
    f = np.arange(len(q))
    start, end, ramp = recipe['load_envelope']
    envelope = cubic((f-start)/ramp)*cubic((end-f)/ramp)
    d = mj.MjData(model)
    jp = np.zeros((3,model.nv)); jr = np.zeros_like(jp)
    for k in range(1,len(q)):
        if envelope[k] == 0:
            continue
        d.qpos[:] = teacher['qpos'][k]
        mj.mj_forward(model,d)
        mj.mj_jac(model,d,jp,jr,d.xipos[body],int(model.jnt_bodyid[5]))
        raw[k,:6] += (jp[:,:6].T@(-model.body_mass[body]*model.opt.gravity))*envelope[k]/kp[:6]
    lo, hi = model.actuator_ctrlrange.T
    baseline = np.clip(raw,lo,hi)
    baseline[0] = np.clip(reference[0],lo,hi)
    held = np.clip(hold_fingers(baseline,recipe),lo,hi)
    offset = position_registration(len(q),recipe)
    target = held.copy()
    target[:,:3] += offset + (kv[:3]/kp[:3])*np.gradient(offset,1/120,axis=0)
    return np.clip(target,lo,hi)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','asset-root','asset-manifest','recipe','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    recipe = json.loads(a.recipe.read_text())
    if hashlib.sha256(a.asset_manifest.read_bytes()).hexdigest() != recipe['asset_manifest_sha256']:
        raise ValueError('recipe asset manifest mismatch')
    from tools.u1_raw_source import load_raw_source
    inp,model,reference,teacher,source = load_raw_source(a.dataset,recipe['dataset_version'],
        recipe['source_row'],a.asset_root,a.asset_manifest)
    if source['index']['uuid'] != recipe['source_uuid'] or inp.frames != recipe['frames']:
        raise ValueError('recipe source identity/frame count mismatch')
    target = build_target(model,reference,teacher,recipe)
    digest = hashlib.sha256(target.tobytes()).hexdigest()
    if digest != recipe['expected_target_sha256']:
        raise ValueError(f'target reconstruction changed: {digest}; do not silently substitute it')
    a.output.mkdir(parents=True,exist_ok=False)
    np.save(a.output/'target.npy',target)
    from tools.u1_placement_workspace import assert_u1
    from sim.manorl.local_contact_repair import MODEL_FIELDS,array_sha
    provenance = dict(source=source,contract=assert_u1(model,inp),recipe=recipe,
                      native_model_field_hashes={k:array_sha(getattr(model,k)) for k in MODEL_FIELDS},
                      target_sha256=digest,status='Reconstructed frozen targets; independent physics validation required')
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output),target_sha256=digest,frames=len(target))))


if __name__ == '__main__':
    main()
