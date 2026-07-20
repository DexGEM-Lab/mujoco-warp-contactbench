#!/usr/bin/env python3
"""Measure MuJoCo base-rotation position-drive step responses."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

# Allow the script to be run directly from the repository without requiring
# callers to set PYTHONPATH explicitly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.dexhandrl.constants import DEFAULT_ISAAC_SOURCE_ROOT
from sim.dexhandrl.constants import DEXHAND021PRO_FULL_DOF_NAMES
from sim.dexhandrl.scene import HAND_XML_RELATIVE, _patch_hand_xml, _patch_pd_gains


AXES = ("ARRx", "ARRy", "ARRz")
JOINTS = DEXHAND021PRO_FULL_DOF_NAMES[3:]
FINGER_JOINTS = DEXHAND021PRO_FULL_DOF_NAMES[6:]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-source-root", type=Path, default=DEFAULT_ISAAC_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=Path("/tmp/dexhandrl_outputs/pd_mujoco.json"))
    parser.add_argument("--axis", choices=("all", *AXES), default="all")
    parser.add_argument("--joint", choices=JOINTS, default=None)
    parser.add_argument("--all-joints", action="store_true")
    parser.add_argument("--all-fingers", action="store_true")
    parser.add_argument("--step-rad", type=float, default=0.05)
    parser.add_argument("--step-frame", type=int, default=10)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--base-rot-kp", type=float, default=10000.0)
    parser.add_argument("--base-rot-kv", type=float, default=125.0)
    parser.add_argument("--base-rot-force", type=float, default=100.0)
    parser.add_argument("--finger-kp", type=float, default=15.0)
    parser.add_argument("--finger-kv", type=float, default=0.0)
    parser.add_argument("--finger-force", type=float, default=50.0)
    return parser


def _build_model(args: argparse.Namespace) -> mujoco.MjModel:
    hand_xml = args.isaac_source_root / HAND_XML_RELATIVE
    xml = hand_xml.read_text(encoding="utf-8")
    xml = _patch_hand_xml(xml, args.isaac_source_root, args.dt, args.substeps)
    xml = _patch_pd_gains(
        xml,
        base_pos_kp=5000,
        base_pos_kv=500,
        base_pos_force=500,
        base_rot_kp=args.base_rot_kp,
        base_rot_kv=args.base_rot_kv,
        base_rot_force=args.base_rot_force,
        finger_kp=args.finger_kp,
        finger_kv=args.finger_kv,
        finger_force=args.finger_force,
    )
    xml = re.sub(
        r'<option[^>]*/>',
        f'<option gravity="0 0 0" iterations="80" solver="Newton" timestep="{args.dt / args.substeps:g}" />',
        xml,
        count=1,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    # Calibration isolates actuator response from all collision constraints.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    return model


def _run_axis(args: argparse.Namespace, axis: str) -> dict[str, object]:
    model = _build_model(args)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, axis)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{axis}")
    if joint_id < 0 or actuator_id < 0:
        raise RuntimeError(f"missing calibration joint/actuator for {axis}")
    qpos_addr = int(model.jnt_qposadr[joint_id])
    qvel_addr = int(model.jnt_dofadr[joint_id])
    initial = float(data.qpos[qpos_addr])
    controls = np.zeros(model.nu, dtype=np.float64)
    for actuator in range(model.nu):
        actuator_joint = int(model.actuator_trnid[actuator, 0])
        if actuator_joint >= 0:
            controls[actuator] = data.qpos[int(model.jnt_qposadr[actuator_joint])]
    target = initial + float(args.step_rad)
    times: list[float] = []
    targets: list[float] = []
    positions: list[float] = []
    velocities: list[float] = []
    actuator_forces: list[float] = []
    qfrc_actuator: list[float] = []
    limit = float(model.actuator_forcerange[actuator_id, 1])

    for frame in range(max(args.frames, 1)):
        controls[actuator_id] = target if frame >= args.step_frame else initial
        data.ctrl[:] = controls
        for _ in range(max(args.substeps, 1)):
            mujoco.mj_step(model, data)
        times.append((frame + 1) * args.dt)
        targets.append(float(controls[actuator_id]))
        positions.append(float(data.qpos[qpos_addr]))
        velocities.append(float(data.qvel[qvel_addr]))
        actuator_forces.append(float(data.actuator_force[actuator_id]))
        qfrc_actuator.append(float(data.qfrc_actuator[qvel_addr]))

    post = np.asarray(positions[args.step_frame :], dtype=np.float64)
    post_target = target
    reached = np.flatnonzero(
        np.sign(args.step_rad) * (post - initial) >= abs(args.step_rad) * 0.9
    )
    rise_frame = int(args.step_frame + reached[0] + 1) if reached.size else None
    peak = float(np.max(np.sign(args.step_rad) * (post - initial))) if post.size else 0.0
    overshoot = max(0.0, peak / max(abs(args.step_rad), 1e-9) - 1.0)
    force_arr = np.asarray(actuator_forces, dtype=np.float64)
    return {
        "axis": axis,
        "joint": axis,
        "step_rad": float(args.step_rad),
        "step_frame": int(args.step_frame),
        "rise_frame_90pct": rise_frame,
        "overshoot_ratio": float(overshoot),
        "final_position_error_rad": float(positions[-1] - post_target),
        "max_abs_actuator_force": float(np.max(np.abs(force_arr))),
        "force_saturation_ratio": float(np.mean(np.abs(force_arr) >= max(abs(limit) * 0.999, 1e-9))),
        "time_s": times,
        "target": targets,
        "position": positions,
        "velocity": velocities,
        "actuator_force": actuator_forces,
        "qfrc_actuator": qfrc_actuator,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.joint is not None:
        axes = (args.joint,)
    elif args.all_fingers:
        axes = FINGER_JOINTS
    elif args.all_joints:
        axes = JOINTS
    else:
        axes = AXES if args.axis == "all" else (args.axis,)
    result = {
        "schema": "dexhandrl.pd_calibration.mujoco.v1",
        "backend": "mujoco",
        "control": {
            "dt": args.dt,
            "substeps": args.substeps,
            "base_rot_kp": args.base_rot_kp,
            "base_rot_kv": args.base_rot_kv,
            "base_rot_force": args.base_rot_force,
            "finger_kp": args.finger_kp,
            "finger_kv": args.finger_kv,
            "finger_force": args.finger_force,
        },
        "axes": [_run_axis(args, axis) for axis in axes],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "axes": axes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
