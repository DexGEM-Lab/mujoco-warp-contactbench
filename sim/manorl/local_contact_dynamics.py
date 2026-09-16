"""MJX-Warp dynamics for local contact repair; all objects free after reset."""
from __future__ import annotations

import numpy as np
import warp as wp
from mujoco.mjx.third_party import mujoco_warp as mw

from sim.manorl.local_contact_repair import MODEL_FIELDS, array_sha, finger_control


@wp.kernel
def wrist_targets(
    qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32), desired: wp.array(dtype=wp.float32),
    velocity: wp.array(dtype=wp.float32), native_kp: wp.array(dtype=wp.float32),
    native_kv: wp.array(dtype=wp.float32), integral: wp.array(dtype=wp.float32),
    hand_weight: float, physics_dt: float,
):
    j = wp.tid()
    err = desired[j] - qpos[0, j]
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
    accum = wp.clamp(integral[j] + ki * err * physics_dt, -15.0, 15.0)
    integral[j] = accum
    force = kp * err + kd * (velocity[j] - qvel[0, j]) + accum + gravity
    raw = qpos[0, j] + (force + native_kv[j] * qvel[0, j]) / native_kp[j]
    lower = float(-2.0)
    upper = float(2.0)
    if j >= 3:
        lower = -9.42478
        upper = 9.42478
    ctrl[0, j] = wp.clamp(raw, lower, upper)


def replay(inp, model, desired, finger_delta):
    import mujoco as mj
    from sim.manorl.mjx_sim import command_target
    names = inp.arrays["scene_object_names"].tolist()
    addresses = [int(model.joint(name + "_free").qposadr[0]) for name in names]
    np.testing.assert_array_equal(addresses, inp.arrays["scene_object_qpos_addresses"])
    if model.nq != len(inp.initial["qpos"]) or model.nv != len(inp.initial["qvel"]):
        raise ValueError("model/init shape mismatch")
    if abs(model.opt.timestep - 1 / (4 * inp.hz)) > 1e-14:
        raise ValueError("physics clock mismatch")
    lower, upper = model.jnt_range[:28].T
    data = mj.MjData(model)
    data.qpos[:] = inp.initial["qpos"]
    data.qvel[:] = inp.initial["qvel"]
    data.ctrl[:] = command_target(desired[0], data.qpos[:28], lower, upper)
    mj.mj_forward(model, data)
    initial_qpos, initial_qvel, initial_ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    wp.init()
    wp.set_device("cuda:0")
    wm = mw.put_model(model)
    wd = mw.put_data(model, data, nworld=1, nconmax=512, nccdmax=512, njmax=4000)
    target = wp.array(desired[0, :6].astype(np.float32), device="cuda:0")
    velocity = wp.zeros(6, dtype=wp.float32, device="cuda:0")
    native_kp = wp.array(model.actuator_gainprm[:6, 0].astype(np.float32), device="cuda:0")
    native_kv = wp.array((-model.actuator_biasprm[:6, 2]).astype(np.float32), device="cuda:0")
    integral = wp.zeros(6, dtype=wp.float32, device="cuda:0")
    substep_controls = wp.zeros((4, model.nu), dtype=wp.float32, device="cuda:0")
    object_bodies = [model.body(name).id for name in names]
    weight = float(-model.opt.gravity[2] * (model.body_mass.sum() - model.body_mass[object_bodies].sum()))
    flags = inp.manifest["flags"]
    def step():
        for substep in range(4):
            if flags["wrist_tracking"]:
                wp.launch(wrist_targets, dim=6, inputs=[wd.qpos, wd.qvel, wd.ctrl, target, velocity,
                          native_kp, native_kv, integral, weight, float(model.opt.timestep)])
            wp.copy(substep_controls, wd.ctrl, dest_offset=substep * model.nu, count=model.nu)
            mw.step(wm, wd)
    step()
    with wp.ScopedCapture() as capture:
        step()
    # Warmup/capture are disposable, matching the archived controller's reset.
    wd.qpos.assign(initial_qpos[None].astype(np.float32))
    wd.qvel.assign(initial_qvel[None].astype(np.float32))
    wd.ctrl.assign(initial_ctrl[None].astype(np.float32))
    wd.qacc_warmstart.zero_()
    wd.time.zero_()
    integral.zero_()
    source_velocity = np.gradient(desired[:, :6], 1 / inp.hz, axis=0)
    finger_velocity = np.gradient(np.clip(desired[:, 6:], lower[6:], upper[6:]), 1 / inp.hz, axis=0)
    positions, velocities, controls = [initial_qpos], [initial_qvel], [initial_ctrl]
    integrals, warmstarts, times = [np.zeros(6)], [np.zeros(model.nv)], [0.]
    applied_substeps = []
    # Every trial reexecutes its unmodified prefix in dynamics. No qpos-only jump.
    for frame in range(1, inp.frames):
        current = wd.qpos.numpy()[0]
        control = finger_control(model, current, desired[frame], finger_velocity[frame],
                                 inp.arrays["finger_offset_envelope"][frame], inp.arrays["grip_envelope"][frame],
                                 inp.initial["donor_preload"], finger_delta[frame], flags["finger_feedforward"])
        wd.ctrl.assign(control[None].astype(np.float32))
        target.assign(desired[frame, :6].astype(np.float32))
        velocity.assign(source_velocity[frame].astype(np.float32))
        wp.capture_launch(capture.graph)
        positions.append(wd.qpos.numpy()[0].copy())
        velocities.append(wd.qvel.numpy()[0].copy())
        controls.append(wd.ctrl.numpy()[0].copy())
        applied_substeps.append(substep_controls.numpy().copy())
        integrals.append(integral.numpy().copy())
        warmstarts.append(wd.qacc_warmstart.numpy()[0].copy())
        times.append(float(wd.time.numpy()[0]))
    qpos = np.asarray(positions)
    qvel = np.asarray(velocities)
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise RuntimeError("nonfinite dynamics; candidate is invalid")
    object_pos = np.stack([qpos[:, adr:adr + 3] for adr in addresses], axis=1)
    object_quat = np.stack([qpos[:, adr + 3:adr + 7][:, [1, 2, 3, 0]] for adr in addresses], axis=1)
    result = dict(inp.arrays)
    result.update(base_desired=inp.arrays["desired"], desired=desired, finger_target_delta=finger_delta,
                  qpos=qpos, qvel=qvel, ctrl=np.asarray(controls), ctrl_substeps=np.asarray(applied_substeps),
                  wrist_integral=np.asarray(integrals), qacc_warmstart=np.asarray(warmstarts),
                  physics_time=np.asarray(times), desired_wrist=desired[:, :6], actual_wrist=qpos[:, :6],
                  actual_object_pos=object_pos, actual_object_quat_xyzw=object_quat)
    if abs(times[-1] - (inp.frames - 1) / inp.hz) > 2e-3:
        raise RuntimeError("observed physics time does not match the requested clock")
    runtime = dict(backend="MJX-Warp", device="cuda:0", control_hz=inp.hz, physics_hz=4 * inp.hz,
                   physics_timestep_s=float(model.opt.timestep), substeps=4,
                   observed_final_time_s=times[-1], frame_zero_integrated=False,
                   solver=dict(cone="elliptic", impratio=float(model.opt.impratio)),
                   controller=dict(flags=flags, translation_kp=1200., translation_kd=40., translation_ki=250.,
                                   rotation_kp=100., rotation_kd=3.5, rotation_ki=8., integral_limit=15.,
                                   integral_timestep_s=float(model.opt.timestep), hand_weight_N=weight),
                   model_array_hashes={k: array_sha(getattr(model, k)) for k in MODEL_FIELDS},
                   model_nq=model.nq, model_nv=model.nv, object_qpos_addresses=addresses,
                   no_object_forcing_after_reset=True, restart="physical replay of original initialization and unchanged prefix")
    return result, runtime
