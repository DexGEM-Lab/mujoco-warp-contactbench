#!/usr/bin/env python3
"""Canary: compress a successful historical PID trace into one direct target per 120Hz frame.

The direct target is held unchanged for all four 480Hz substeps.  This tool never
forces hand/object state after frame0 and disables all historical wrist PID,
finger feedforward, donor preload, and correction state.  U1/U2 differ only in
contact cone/impratio so the experiment discriminates target compression from
solver-contact semantics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from sim.manorl.local_contact_repair import MODEL_FIELDS, array_sha, dump
from tools.pilot_start_augmentation import reconstruct


SETTINGS = {
    "U1": {"cone": "pyramidal", "impratio": 1.0},
    "U2": {"cone": "elliptic", "impratio": 100.0},
}
TARGET_KINDS = ("first", "mean", "last", "teacher", "teacher-lead1")


def compile_model(bundle: Path, asset_root: Path, inp, setting: str):
    import mujoco as mj

    manifest = bundle / "asset_manifest.json"
    os.environ["MANORL_ASSET_MANIFEST"] = str(manifest.resolve())
    from sim.manorl import assets

    assets.DEXSTREAM_ROOT = asset_root.resolve()
    names = inp.arrays["scene_object_names"].tolist()
    _, model = assets.compile_unified_model(
        object_types=names, object_collisions=True, physics_timestep=1 / 480
    )
    if setting == "U1":
        model.opt.cone = mj.mjtCone.mjCONE_PYRAMIDAL
        model.opt.impratio = 1.0
    else:
        model.opt.cone = mj.mjtCone.mjCONE_ELLIPTIC
        model.opt.impratio = 100.0
    for field in MODEL_FIELDS:
        np.testing.assert_array_equal(
            getattr(model, field),
            np.asarray(inp.manifest["native"][field]),
            err_msg=f"physical model changed: {field}",
        )
    return assets, model


def target_stream(baseline: dict[str, np.ndarray], kind: str) -> np.ndarray:
    q = np.asarray(baseline["qpos"], dtype=np.float64)[:, :28]
    ctrl0 = np.asarray(baseline["ctrl"], dtype=np.float64)[0]
    sub = np.asarray(baseline["ctrl_substeps"], dtype=np.float64)
    if sub.shape != (len(q) - 1, 4, 28):
        raise ValueError(f"unexpected substep controls: {sub.shape}, q={q.shape}")
    target = np.empty_like(q)
    target[0] = ctrl0
    if kind == "first":
        target[1:] = sub[:, 0]
    elif kind == "mean":
        target[1:] = sub.mean(axis=1)
    elif kind == "last":
        target[1:] = sub[:, -1]
    elif kind == "teacher":
        target[1:] = q[1:]
    elif kind == "teacher-lead1":
        target[1:-1] = q[2:]
        target[-1] = q[-1]
    else:
        raise ValueError(kind)
    return target


def replay_direct(inp, model, target: np.ndarray) -> dict[str, np.ndarray]:
    import mujoco as mj
    import warp as wp
    from mujoco.mjx.third_party import mujoco_warp as mw
    from sim.manorl.mjx_sim import command_target

    names = inp.arrays["scene_object_names"].tolist()
    addresses = [int(model.joint(name + "_free").qposadr[0]) for name in names]
    lower, upper = model.jnt_range[:28].T
    if target.shape != (inp.frames, 28):
        raise ValueError("target shape mismatch")

    data = mj.MjData(model)
    data.qpos[:] = inp.initial["qpos"]
    data.qvel[:] = inp.initial["qvel"]
    data.ctrl[:] = command_target(target[0], data.qpos[:28], lower, upper)
    mj.mj_forward(model, data)
    q0, v0, c0 = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()

    wp.init()
    wp.set_device("cuda:0")
    wm = mw.put_model(model)
    wd = mw.put_data(model, data, nworld=1, nconmax=512, nccdmax=512, njmax=4000)

    def step4():
        for _ in range(4):
            mw.step(wm, wd)

    step4()
    with wp.ScopedCapture() as capture:
        step4()
    wd.qpos.assign(q0[None].astype(np.float32))
    wd.qvel.assign(v0[None].astype(np.float32))
    wd.ctrl.assign(c0[None].astype(np.float32))
    wd.qacc_warmstart.zero_()
    wd.time.zero_()

    qpos, qvel, controls, times = [q0], [v0], [c0], [0.0]
    substeps = []
    for frame in range(1, inp.frames):
        current = wd.qpos.numpy()[0, :28]
        control = command_target(target[frame], current, lower, upper)
        wd.ctrl.assign(control[None].astype(np.float32))
        wp.capture_launch(capture.graph)
        qpos.append(wd.qpos.numpy()[0].copy())
        qvel.append(wd.qvel.numpy()[0].copy())
        actual_control = wd.ctrl.numpy()[0].copy()
        controls.append(actual_control)
        substeps.append(np.broadcast_to(actual_control, (4, 28)).copy())
        times.append(float(wd.time.numpy()[0]))

    qpos = np.asarray(qpos)
    qvel = np.asarray(qvel)
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise RuntimeError("nonfinite direct replay")
    result = dict(inp.arrays)
    result.update(
        base_desired=target,
        desired=target,
        finger_target_delta=np.zeros((inp.frames, 22), dtype=np.float64),
        qpos=qpos,
        qvel=qvel,
        ctrl=np.asarray(controls),
        ctrl_substeps=np.asarray(substeps),
        wrist_integral=np.zeros((inp.frames, 6), dtype=np.float64),
        qacc_warmstart=np.zeros((inp.frames, model.nv), dtype=np.float64),
        physics_time=np.asarray(times),
        desired_wrist=target[:, :6],
        actual_wrist=qpos[:, :6],
        actual_object_pos=np.stack([qpos[:, adr : adr + 3] for adr in addresses], axis=1),
        actual_object_quat_xyzw=np.stack(
            [qpos[:, adr + 3 : adr + 7][:, [1, 2, 3, 0]] for adr in addresses], axis=1
        ),
    )
    return result


def run(args: argparse.Namespace) -> None:
    out = args.output / args.row / args.setting / args.target_kind
    out.mkdir(parents=True, exist_ok=False)
    inp, _parent, _fingers, teacher, provenance = reconstruct(args.bundle, args.row)
    _assets, model = compile_model(args.bundle, args.asset_root, inp, args.setting)
    targets = target_stream(teacher, args.target_kind)
    start = time.monotonic()
    trace = replay_direct(inp, model, targets)
    from sim.manorl.local_contact_repair import evaluate

    physical, diagnostic = evaluate(model, inp, trace)
    teacher_qpos = np.asarray(teacher["qpos"], dtype=np.float64)
    teacher_obj = np.asarray(teacher["actual_object_pos"], dtype=np.float64)
    hand_error = np.linalg.norm(trace["qpos"][:, :3] - teacher_qpos[:, :3], axis=1)
    object_error = np.linalg.norm(trace["actual_object_pos"] - teacher_obj, axis=2)
    setting = SETTINGS[args.setting]
    metrics = {
        "row": args.row,
        "setting": args.setting,
        "setting_config": setting,
        "target_kind": args.target_kind,
        "frames": inp.frames,
        "max_hand_root_position_error_m": float(hand_error.max()),
        "final_hand_root_position_error_m": float(hand_error[-1]),
        "max_object_position_error_m": float(object_error.max()),
        "final_object_position_error_m": float(object_error[-1].max()),
        "first_object_error_over_2cm_frame": (
            int(np.flatnonzero(object_error.max(axis=1) > 0.02)[0])
            if np.any(object_error.max(axis=1) > 0.02)
            else None
        ),
        "physical_pose_pass": bool(physical["physical_pose_pass"]),
        "functional_pass": bool(physical["functional_pass"]),
        "failed_gates": list(physical["failed_gates"]),
        "physical": physical,
        "elapsed_s": time.monotonic() - start,
        "model_array_hashes": {k: array_sha(getattr(model, k)) for k in MODEL_FIELDS},
        "source_provenance": provenance,
    }
    np.savez_compressed(out / "trace.npz", **trace)
    np.savez_compressed(out / "contact_validation.npz", **diagnostic["arrays"])
    dump(out / "result.json", metrics)
    print(json.dumps({k: metrics[k] for k in (
        "row", "setting", "target_kind", "max_hand_root_position_error_m",
        "max_object_position_error_m", "physical_pose_pass", "functional_pass",
        "failed_gates", "elapsed_s")}, ensure_ascii=False), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--row", required=True)
    p.add_argument("--setting", choices=tuple(SETTINGS), required=True)
    p.add_argument("--target-kind", choices=TARGET_KINDS, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
