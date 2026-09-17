"""Independent same-parent MJX-Warp worlds; identical scalar control equations."""
from __future__ import annotations

import numpy as np
import warp as wp
from mujoco.mjx.third_party import mujoco_warp as mw

from sim.manorl.local_contact_repair import MODEL_FIELDS, array_sha, finger_control


@wp.kernel
def batch_wrist_targets(
    qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32), desired: wp.array2d(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32), native_kp: wp.array(dtype=wp.float32),
    native_kv: wp.array(dtype=wp.float32), integral: wp.array2d(dtype=wp.float32),
    hand_weight: float, physics_dt: float,
):
    world, j = wp.tid()
    err = desired[world, j] - qpos[world, j]
    if j >= 3:
        err = wp.atan2(wp.sin(err), wp.cos(err))
    kp = float(1200.0)
    kd = float(40.0)
    ki = float(250.0)
    gravity = float(0.0)
    if j == 2:
        gravity = hand_weight
    if j >= 3:
        kp = 100.0
        kd = 3.5
        ki = 8.0
    accum = wp.clamp(integral[world, j] + ki * err * physics_dt, -15.0, 15.0)
    integral[world, j] = accum
    force = kp * err + kd * (velocity[world, j] - qvel[world, j]) + accum + gravity
    raw = qpos[world, j] + (force + native_kv[j] * qvel[world, j]) / native_kp[j]
    lower = float(-2.0)
    upper = float(2.0)
    if j >= 3:
        lower = -9.42478
        upper = 9.42478
    ctrl[world, j] = wp.clamp(raw, lower, upper)


def replay_batch(inputs, model, targets, finger_delta):
    """One model, W separate states/controllers; returns W full continuous traces."""
    import mujoco as mj
    from sim.manorl.mjx_sim import command_target
    if not inputs or len(inputs) != len(targets):
        raise ValueError("one input per target world required")
    inp = inputs[0]
    worlds, frames = len(inputs), inp.frames
    if any(x.frames != frames or x.row_id != inp.row_id or x.hz != 120 for x in inputs):
        raise ValueError("batch must share a120Hz parent and frame count")
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (worlds, frames, 28) or finger_delta.shape != (frames, 22):
        raise ValueError("invalid target/finger shape")
    if abs(model.opt.timestep - 1/480) > 1e-14:
        raise ValueError("expected480Hz physics")
    names = inp.arrays['scene_object_names'].tolist()
    addresses = [int(model.joint(n+'_free').qposadr[0]) for n in names]
    np.testing.assert_array_equal(addresses, inp.arrays['scene_object_qpos_addresses'])
    lo, hi = model.jnt_range[:28].T
    q0 = np.stack([x.initial['qpos'] for x in inputs])
    v0 = np.stack([x.initial['qvel'] for x in inputs])
    c0 = np.stack([command_target(targets[w, 0], q0[w, :28], lo, hi) for w in range(worlds)])
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.ctrl[:] = q0[0], v0[0], c0[0]
    mj.mj_forward(model, data)
    wp.init(); wp.set_device('cuda:0')
    wm = mw.put_model(model)
    wd = mw.put_data(model, data, nworld=worlds, nconmax=512, nccdmax=512, njmax=4000)
    target = wp.array(targets[:, 0, :6].astype(np.float32), device='cuda:0')
    velocity = wp.zeros((worlds, 6), dtype=wp.float32, device='cuda:0')
    kp = wp.array(model.actuator_gainprm[:6, 0].astype(np.float32), device='cuda:0')
    kv = wp.array((-model.actuator_biasprm[:6, 2]).astype(np.float32), device='cuda:0')
    integral = wp.zeros((worlds, 6), dtype=wp.float32, device='cuda:0')
    subctrl = wp.zeros((4, worlds, model.nu), dtype=wp.float32, device='cuda:0')
    bodies = [model.body(n).id for n in names]
    weight = float(-model.opt.gravity[2]*(model.body_mass.sum()-model.body_mass[bodies].sum()))
    flags = inp.manifest['flags']
    wd.qpos.assign(q0.astype(np.float32)); wd.qvel.assign(v0.astype(np.float32)); wd.ctrl.assign(c0.astype(np.float32))
    def step():
        for sub in range(4):
            if flags['wrist_tracking']:
                wp.launch(batch_wrist_targets, dim=(worlds, 6), inputs=[wd.qpos, wd.qvel, wd.ctrl,
                          target, velocity, kp, kv, integral, weight, float(model.opt.timestep)])
            wp.copy(subctrl, wd.ctrl, dest_offset=sub*worlds*model.nu, count=worlds*model.nu)
            mw.step(wm, wd)
    step()
    with wp.ScopedCapture() as capture:
        step()
    wd.qpos.assign(q0.astype(np.float32)); wd.qvel.assign(v0.astype(np.float32)); wd.ctrl.assign(c0.astype(np.float32))
    wd.qacc_warmstart.zero_(); wd.time.zero_(); integral.zero_()
    wrist_speed = np.gradient(targets[:, :, :6], 1/120, axis=1)
    finger_speed = np.gradient(np.clip(targets[:, :, 6:], lo[6:], hi[6:]), 1/120, axis=1)
    positions, velocities, controls = [q0], [v0], [c0]
    integrals, warmstarts, times = [np.zeros((worlds, 6))], [np.zeros((worlds, model.nv))], [np.zeros(worlds)]
    substeps = []
    for frame in range(1, frames):
        current = wd.qpos.numpy()
        control = np.stack([finger_control(model, current[w], targets[w, frame], finger_speed[w, frame],
                    inp.arrays['finger_offset_envelope'][frame], inp.arrays['grip_envelope'][frame],
                    inp.initial['donor_preload'], finger_delta[frame], flags['finger_feedforward']) for w in range(worlds)])
        wd.ctrl.assign(control.astype(np.float32))
        target.assign(targets[:, frame, :6].astype(np.float32))
        velocity.assign(wrist_speed[:, frame].astype(np.float32))
        wp.capture_launch(capture.graph)
        positions.append(wd.qpos.numpy().copy()); velocities.append(wd.qvel.numpy().copy())
        controls.append(wd.ctrl.numpy().copy()); integrals.append(integral.numpy().copy())
        warmstarts.append(wd.qacc_warmstart.numpy().copy()); times.append(wd.time.numpy().copy())
        substeps.append(subctrl.numpy().copy())
    qpos, qvel = np.asarray(positions), np.asarray(velocities)
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise RuntimeError('nonfinite batched dynamics')
    times = np.asarray(times)
    if np.max(abs(times[-1]-(frames-1)/120)) > .002:
        raise RuntimeError('physical clock mismatch')
    controls, integrals, warmstarts = np.asarray(controls), np.asarray(integrals), np.asarray(warmstarts)
    substeps = np.asarray(substeps)
    traces = []
    for w in range(worlds):
        result = dict(inp.arrays)
        result.update(base_desired=inp.arrays['desired'], desired=targets[w], finger_target_delta=finger_delta,
                      qpos=qpos[:, w], qvel=qvel[:, w], ctrl=controls[:, w], ctrl_substeps=substeps[:, :, w],
                      wrist_integral=integrals[:, w], qacc_warmstart=warmstarts[:, w], physics_time=times[:, w],
                      desired_wrist=targets[w, :, :6], actual_wrist=qpos[:, w, :6],
                      actual_object_pos=np.stack([qpos[:, w, a:a+3] for a in addresses], axis=1),
                      actual_object_quat_xyzw=np.stack([qpos[:, w, a+3:a+7][:, [1, 2, 3, 0]] for a in addresses], axis=1))
        traces.append(result)
    runtime = dict(backend='MJX-Warp', device='cuda:0', worlds=worlds, control_hz=120, physics_hz=480,
                   substeps=4, frame_zero_integrated=False, normal_frames=frames,
                   physical_timestep_s=float(model.opt.timestep), observed_final_time_s=times[-1].tolist(),
                   solver=dict(cone='elliptic', impratio=float(model.opt.impratio)),
                   controller=dict(flags=flags, translation_kp=1200., translation_kd=40., translation_ki=250.,
                                   rotation_kp=100., rotation_kd=3.5, rotation_ki=8., integral_limit=15.,
                                   integral_timestep_s=float(model.opt.timestep), hand_weight_N=weight),
                   model_array_hashes={k:array_sha(getattr(model,k)) for k in MODEL_FIELDS},
                   no_object_forcing_after_reset=True, merge_state_reset=False)
    return traces, runtime
