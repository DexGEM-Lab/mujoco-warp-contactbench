#!/usr/bin/env python3
"""Replay DexHandRL Lance trajectories with MuJoCo MJX-Warp reference PD control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.dexhandrl.constants import (  # noqa: E402
    DEFAULT_ISAAC_SOURCE_ROOT,
    DEFAULT_LANCE_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEXHAND021PRO_CONTACT_BODY_NAMES,
)
from sim.dexhandrl.contact import collect_contact_summary  # noqa: E402
from sim.dexhandrl.hand_mapping import (  # noqa: E402
    active_to_full_dof,
    dexhand021pro_route_joint_upper_limits,
    full_to_active_dof,
    full_dof_from_qpos,
    make_ctrl_vector,
    raw_finger_to_full_dof,
    resolve_mujoco_ids,
    set_hand_qpos,
    set_object_freejoint,
    wxyz_to_xyzw,
)
from sim.dexhandrl.lance_loader import load_trajectory  # noqa: E402
from sim.dexhandrl.parity import rotation_angle_error_deg  # noqa: E402
from sim.dexhandrl.scene import build_dexhandrl_scene_xml  # noqa: E402


FINGER_CONTACT_GROUPS = {
    "thumb": ("RH0_0", "RH0_2", "RH0_3"),
    "index": ("RH1_0", "RH1_2", "RH1_3"),
    "middle": ("RH2_0", "RH2_2", "RH2_3"),
    "ring": ("RH3_0", "RH3_2", "RH3_3"),
    "pinky": ("RH4_0", "RH4_2", "RH4_3"),
}


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _device_get(jax: Any, value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _object_pose_from_qpos(qpos: np.ndarray, object_qpos_addr: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(qpos[object_qpos_addr : object_qpos_addr + 3], dtype=np.float32)
    quat_wxyz = np.asarray(qpos[object_qpos_addr + 3 : object_qpos_addr + 7], dtype=np.float32)
    return pos, wxyz_to_xyzw(quat_wxyz)


def _reference_full_targets(
    traj: Any,
    frame: int,
    source: str,
    current_full: np.ndarray | None,
    route_joint_upper_limits: dict[int, tuple[float, float]],
) -> np.ndarray:
    if source == "active":
        return active_to_full_dof(
            traj.hand_dof[frame],
            current_full=current_full,
            route_joint_upper_limits=route_joint_upper_limits,
        )
    if source == "full_raw":
        if traj.hand_finger_dof_full_raw is None:
            raise RuntimeError("Lance row does not contain dexhand_joint_angles_full_raw")
        return raw_finger_to_full_dof(traj.hand_dof[frame], traj.hand_finger_dof_full_raw[frame])
    raise ValueError(f"unsupported reference joint source {source!r}")


def _geom_body_names(mujoco: Any, model: Any) -> dict[int, str]:
    body_names = {
        body_id: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        for body_id in range(model.nbody)
    }
    return {geom_id: body_names[int(model.geom_bodyid[geom_id])] for geom_id in range(model.ngeom)}


def _geom_names(mujoco: Any, model: Any) -> dict[int, str]:
    return {
        geom_id: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        for geom_id in range(model.ngeom)
    }


def _contact_summary(
    jax: Any,
    dx: Any,
    geom_body_names: dict[int, str],
    geom_names: dict[int, str],
    object_name: str,
    contact_body_names: tuple[str, ...],
    pyramidal: bool,
) -> dict[str, Any]:
    impl = dx._impl
    nacon = _device_get(jax, impl.nacon)
    active_contacts = int(nacon[0]) if nacon.shape else int(nacon)
    return collect_contact_summary(
        active_contacts=active_contacts,
        contact_geom=_device_get(jax, impl.contact__geom),
        contact_frame=_device_get(jax, impl.contact__frame),
        contact_friction=_device_get(jax, impl.contact__friction),
        contact_dim=_device_get(jax, impl.contact__dim),
        contact_efc_address=_device_get(jax, impl.contact__efc_address),
        efc_force=_device_get(jax, impl.efc__force),
        pyramidal=pyramidal,
        geom_body_names=geom_body_names,
        geom_names=geom_names,
        contact_body_names=contact_body_names,
        object_name=object_name,
    )


def _finger_force_from_body_force(body_force: dict[str, float]) -> dict[str, float]:
    return {
        finger: float(sum(body_force.get(body, 0.0) for body in bodies))
        for finger, bodies in FINGER_CONTACT_GROUPS.items()
    }


def replay(args: argparse.Namespace) -> dict[str, Any]:
    requested_device = str(args.device)
    if requested_device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif requested_device == "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    else:
        raise ValueError(f"unsupported MJX-Warp device: {requested_device}")

    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    actual_backend = jax.default_backend()
    if requested_device == "gpu" and actual_backend != "gpu":
        raise RuntimeError(f"expected JAX GPU backend, got {actual_backend}; devices={jax.devices()}")
    if requested_device == "cpu" and actual_backend != "cpu":
        raise RuntimeError(f"expected JAX CPU backend, got {actual_backend}; devices={jax.devices()}")

    traj = load_trajectory(
        lance_path=args.lance,
        object_name=args.object,
        action=args.action,
        sequence_index=args.sequence_index,
        uuid=args.uuid,
    )
    if args.start_frame is None:
        start_frame = max(0, traj.movement_start_frame - args.pre_contact_frames)
    else:
        start_frame = int(args.start_frame)
    if start_frame < 0 or start_frame >= traj.frames:
        raise ValueError(f"start_frame must be in [0, {traj.frames - 1}], got {start_frame}")
    available_frames = traj.frames - start_frame
    frame_count = available_frames if args.max_frames <= 0 else min(available_frames, args.max_frames)
    end_frame = start_frame + frame_count - 1
    scene_path = build_dexhandrl_scene_xml(
        output_path=Path(tempfile.mkdtemp(prefix="dexhandrl_mjx_scene_")) / "dexhandrl.xml",
        object_name=traj.object_name,
        isaac_source_root=args.isaac_source_root,
        dt=args.dt,
        substeps=args.substeps,
        base_pos_kp=args.base_pos_kp,
        base_pos_kv=args.base_pos_kv,
        base_pos_force=args.base_pos_force,
        base_rot_kp=args.base_rot_kp,
        base_rot_kv=args.base_rot_kv,
        base_rot_force=args.base_rot_force,
        finger_kp=args.finger_kp,
        finger_kv=args.finger_kv,
        finger_force=args.finger_force,
        object_friction=args.object_friction,
        ground_friction=args.ground_friction,
    )
    args.scene_copy.parent.mkdir(parents=True, exist_ok=True)
    args.scene_copy.write_text(scene_path.read_text(encoding="utf-8"), encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    ids = resolve_mujoco_ids(mujoco, model, object_name=traj.object_name)
    route_joint_upper_limits = dexhand021pro_route_joint_upper_limits(mujoco, model)
    geom_body_names = _geom_body_names(mujoco, model)
    geom_names = _geom_names(mujoco, model)
    pyramidal = int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    mx = mjx.put_model(model, impl="warp")
    dx = mjx.make_data(model, impl="warp", naconmax=args.naconmax, njmax=args.njmax)
    forward = jax.jit(mjx.forward)
    step = jax.jit(mjx.step)

    initial_full = _reference_full_targets(
        traj=traj,
        frame=start_frame,
        source=args.reference_joint_source,
        current_full=None,
        route_joint_upper_limits=route_joint_upper_limits,
    )
    initial_object_pos = np.asarray(traj.object_pos[start_frame], dtype=np.float32).copy()
    initial_object_pos[2] += float(args.object_reset_z_offset)
    dx = set_hand_qpos(jnp, dx, ids, initial_full)
    dx = set_object_freejoint(jnp, dx, ids, initial_object_pos, traj.object_quat_xyzw[start_frame])

    ctrl = make_ctrl_vector(initial_full, ids.actuator_ids, model.nu)
    dx = dx.replace(ctrl=jnp.asarray(ctrl, dtype=dx.ctrl.dtype))
    dx = forward(mx, dx)
    for _ in range(args.settle_frames * args.substeps):
        dx = step(mx, dx)
    jax.block_until_ready(dx.qpos)

    pos_errors = []
    pos_errors_xyz = []
    rot_errors_deg = []
    active_l2_errors = []
    full_l2_errors = []
    base_position_l2_errors = []
    base_rotation_l2_errors = []
    finger_active_mean_abs_errors = []
    finger_active_max_abs_errors = []
    hand_tracking_records = []
    hand_object_contact_counts = []
    hand_object_contact_substep_counts = []
    hand_object_contact_pairs_by_frame = []
    contact_force_body_max_by_frame = []
    contact_force_body_sum_by_frame = []
    contact_force_finger_max_by_frame = []
    contact_force_finger_sum_by_frame = []
    object_contact_force_max_by_frame = []
    object_contact_force_sum_by_frame = []
    expected_contact_by_frame = []
    unexpected_contact_by_frame = []
    object_positions = []
    start = time.perf_counter()
    for local_frame in range(frame_count):
        lance_frame = start_frame + local_frame
        current_qpos = _device_get(jax, dx.qpos)
        current_full = (
            full_dof_from_qpos(current_qpos, ids)
            if args.reference_joint_source == "active"
            else None
        )
        full = _reference_full_targets(
            traj=traj,
            frame=lance_frame,
            source=args.reference_joint_source,
            current_full=current_full,
            route_joint_upper_limits=route_joint_upper_limits,
        )
        ctrl = make_ctrl_vector(full, ids.actuator_ids, model.nu)
        dx = dx.replace(ctrl=jnp.asarray(ctrl, dtype=dx.ctrl.dtype))
        frame_contact_count_max = 0
        frame_contact_substep_count = 0
        frame_contact_pairs: dict[tuple[int, int, str, str], dict[str, Any]] = {}
        frame_body_force_sum = {name: 0.0 for name in DEXHAND021PRO_CONTACT_BODY_NAMES}
        frame_body_force_max = {name: 0.0 for name in DEXHAND021PRO_CONTACT_BODY_NAMES}
        frame_object_force_sum = 0.0
        frame_object_force_max = 0.0
        for _ in range(args.substeps):
            dx = step(mx, dx)
            substep_summary = _contact_summary(
                jax,
                dx,
                geom_body_names,
                geom_names,
                traj.object_name,
                tuple(DEXHAND021PRO_CONTACT_BODY_NAMES),
                pyramidal,
            )
            substep_pairs = substep_summary["hand_object_pairs"]
            body_vectors = np.asarray(substep_summary["contact_force_vectors"], dtype=np.float32)
            for body_index, body_name in enumerate(DEXHAND021PRO_CONTACT_BODY_NAMES):
                body_force = float(np.linalg.norm(body_vectors[body_index]))
                frame_body_force_sum[body_name] += body_force
                frame_body_force_max[body_name] = max(frame_body_force_max[body_name], body_force)
            object_force = float(np.linalg.norm(substep_summary["object_force_vector"]))
            frame_object_force_sum += object_force
            frame_object_force_max = max(frame_object_force_max, object_force)
            if substep_pairs:
                frame_contact_substep_count += 1
                frame_contact_count_max = max(frame_contact_count_max, len(substep_pairs))
                for pair in substep_pairs:
                    key = (pair["geom1"], pair["geom2"], pair["body1"], pair["body2"])
                    if key not in frame_contact_pairs:
                        pair["substep_force_sum_N"] = float(pair["force_magnitude_N"])
                        frame_contact_pairs[key] = pair
                    else:
                        existing = frame_contact_pairs[key]
                        existing["force_magnitude_N"] = max(
                            float(existing.get("force_magnitude_N") or 0.0),
                            float(pair["force_magnitude_N"]),
                        )
                        existing["force_normal_N"] = max(
                            float(existing.get("force_normal_N") or 0.0),
                            float(pair["force_normal_N"]),
                        )
                        existing["substep_force_sum_N"] = (
                            float(existing["substep_force_sum_N"]) + float(pair["force_magnitude_N"])
                        )
        if args.progress_interval and local_frame % args.progress_interval == 0:
            jax.block_until_ready(dx.qpos)
            print(
                f"frame={local_frame}/{frame_count} lance_frame={lance_frame} "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )

        qpos = _device_get(jax, dx.qpos)
        sim_pos, sim_quat_xyzw = _object_pose_from_qpos(qpos, ids.object_free_qpos_addr)
        target_pos = traj.object_pos[lance_frame]
        target_quat = traj.object_quat_xyzw[lance_frame]
        diff = sim_pos - target_pos
        pos_errors_xyz.append(diff.tolist())
        pos_errors.append(float(np.linalg.norm(diff)))
        rot_errors_deg.append(rotation_angle_error_deg(sim_quat_xyzw, target_quat))

        sim_full = full_dof_from_qpos(qpos, ids)
        sim_active = full_to_active_dof(sim_full)
        target_active = full_to_active_dof(full)
        active_diff = sim_active - target_active
        full_diff = sim_full - full
        finger_active_abs = np.abs(active_diff[6:])
        active_l2_errors.append(float(np.linalg.norm(active_diff)))
        full_l2_errors.append(float(np.linalg.norm(full_diff)))
        base_position_l2_errors.append(float(np.linalg.norm(active_diff[:3])))
        base_rotation_l2_errors.append(float(np.linalg.norm(active_diff[3:6])))
        finger_active_mean_abs_errors.append(float(finger_active_abs.mean()))
        finger_active_max_abs_errors.append(float(finger_active_abs.max()))
        if args.keep_trajectory:
            hand_tracking_records.append(
                {
                    "step": int(local_frame),
                    "lance_frame": int(lance_frame),
                    "active_l2_error": active_l2_errors[-1],
                    "full_l2_error": full_l2_errors[-1],
                    "base_position_l2_error_m": base_position_l2_errors[-1],
                    "base_rotation_l2_error_rad": base_rotation_l2_errors[-1],
                    "finger_active_mean_abs_error_rad": finger_active_mean_abs_errors[-1],
                    "finger_active_max_abs_error_rad": finger_active_max_abs_errors[-1],
                    "actual_active": sim_active.tolist(),
                    "target_active": target_active.tolist(),
                    "actual_full": sim_full.tolist(),
                    "target_full": full.tolist(),
                }
            )
        expected_contact = bool(np.any(traj.expected_contact_mask[lance_frame] > 0.5))
        hand_object_contact_counts.append(frame_contact_count_max)
        hand_object_contact_substep_counts.append(frame_contact_substep_count)
        body_force_max = [float(frame_body_force_max[name]) for name in DEXHAND021PRO_CONTACT_BODY_NAMES]
        body_force_sum = [float(frame_body_force_sum[name]) for name in DEXHAND021PRO_CONTACT_BODY_NAMES]
        finger_force_max_dict = _finger_force_from_body_force(frame_body_force_max)
        finger_force_sum_dict = _finger_force_from_body_force(frame_body_force_sum)
        finger_names = list(FINGER_CONTACT_GROUPS)
        finger_force_max = [float(finger_force_max_dict[name]) for name in finger_names]
        finger_force_sum = [float(finger_force_sum_dict[name]) for name in finger_names]
        contact_force_body_max_by_frame.append(body_force_max)
        contact_force_body_sum_by_frame.append(body_force_sum)
        contact_force_finger_max_by_frame.append(finger_force_max)
        contact_force_finger_sum_by_frame.append(finger_force_sum)
        object_contact_force_max_by_frame.append(float(frame_object_force_max))
        object_contact_force_sum_by_frame.append(float(frame_object_force_sum))
        frame_pairs = list(frame_contact_pairs.values())
        if args.keep_trajectory:
            hand_object_contact_pairs_by_frame.append(frame_pairs)
        expected_contact_by_frame.append(expected_contact)
        unexpected_contact_by_frame.append(frame_contact_count_max > 0 and not expected_contact)
        object_positions.append(sim_pos.tolist())

    jax.block_until_ready(dx.qpos)
    pos_errors_arr = np.asarray(pos_errors, dtype=np.float64)
    rot_errors_arr = np.asarray(rot_errors_deg, dtype=np.float64)
    object_positions_arr = np.asarray(object_positions, dtype=np.float64)
    target_positions_arr = traj.object_pos[start_frame : start_frame + frame_count].astype(np.float64)
    initial_object_z = float(object_positions_arr[0, 2])
    initial_target_z = float(target_positions_arr[0, 2])
    max_object_z = float(object_positions_arr[:, 2].max())
    max_target_z = float(target_positions_arr[:, 2].max())
    first_error_gt_10cm = next(
        (idx + 1 for idx, err in enumerate(pos_errors) if err > 0.10),
        -1,
    )
    contact_counts_arr = np.asarray(hand_object_contact_counts, dtype=np.int32)
    contact_substep_counts_arr = np.asarray(hand_object_contact_substep_counts, dtype=np.int32)
    contact_body_max_arr = np.asarray(contact_force_body_max_by_frame, dtype=np.float64)
    contact_body_sum_arr = np.asarray(contact_force_body_sum_by_frame, dtype=np.float64)
    contact_finger_max_arr = np.asarray(contact_force_finger_max_by_frame, dtype=np.float64)
    contact_finger_sum_arr = np.asarray(contact_force_finger_sum_by_frame, dtype=np.float64)
    object_contact_max_arr = np.asarray(object_contact_force_max_by_frame, dtype=np.float64)
    object_contact_sum_arr = np.asarray(object_contact_force_sum_by_frame, dtype=np.float64)
    expected_contact_arr = np.asarray(expected_contact_by_frame, dtype=bool)
    unexpected_contact_arr = np.asarray(unexpected_contact_by_frame, dtype=bool)
    first_hand_object_contact_step = next(
        (idx for idx, count in enumerate(hand_object_contact_counts) if count > 0),
        -1,
    )
    first_expected_contact_step = next(
        (idx for idx, expected in enumerate(expected_contact_by_frame) if expected),
        -1,
    )
    first_unexpected_contact_step = next(
        (idx for idx, unexpected in enumerate(unexpected_contact_by_frame) if unexpected),
        -1,
    )
    result = {
        "schema": "dexhandrl.reference_pd_replay.v1",
        "backend": "mjx_warp",
        "device": requested_device,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "mujoco_version": mujoco.__version__,
        "lance_path": str(args.lance),
        "uuid": traj.uuid,
        "row_id": int(traj.row_id),
        "object_name": traj.object_name,
        "action": traj.action,
        "frames": int(frame_count),
        "trajectory_frames": int(traj.frames),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "movement_start_frame": int(traj.movement_start_frame),
        "movement_end_frame": int(traj.movement_end_frame),
        "use_residual": False,
        "control": {
            "mode": "reference_pd",
            "reference_joint_source": args.reference_joint_source,
            "dt": args.dt,
            "substeps": args.substeps,
            "settle_frames": args.settle_frames,
            "pre_contact_frames": args.pre_contact_frames,
            "object_reset_z_offset": args.object_reset_z_offset,
            "base_pos_kp": args.base_pos_kp,
            "base_pos_kv": args.base_pos_kv,
            "base_pos_force": args.base_pos_force,
            "base_rot_kp": args.base_rot_kp,
            "base_rot_kv": args.base_rot_kv,
            "base_rot_force": args.base_rot_force,
            "finger_kp": args.finger_kp,
            "finger_kv": args.finger_kv,
            "finger_force": args.finger_force,
        },
        "scene_xml": str(scene_path),
        "scene_copy": str(args.scene_copy),
        "object_position_error_m": _stats(pos_errors_arr),
        "object_tracking": {
            "initial_object_z": initial_object_z,
            "max_object_z": max_object_z,
            "actual_lift_m": max_object_z - initial_object_z,
            "initial_target_z": initial_target_z,
            "max_target_z": max_target_z,
            "target_lift_m": max_target_z - initial_target_z,
            "max_error_m": float(pos_errors_arr.max()) if pos_errors_arr.size else None,
            "mean_error_m": float(pos_errors_arr.mean()) if pos_errors_arr.size else None,
            "first_error_gt_10cm_step": int(first_error_gt_10cm),
        },
        "object_position_error_m_by_frame": pos_errors if args.keep_trajectory else None,
        "object_position_error_xyz_m": {
            axis: _stats(np.asarray(pos_errors_xyz, dtype=np.float64)[:, idx])
            for idx, axis in enumerate(("x", "y", "z"))
        },
        "object_rotation_error_deg": _stats(rot_errors_arr),
        "object_rotation_error_deg_by_frame": rot_errors_deg if args.keep_trajectory else None,
        "hand_target_tracking": {
            "active_l2_error": _stats(np.asarray(active_l2_errors, dtype=np.float64)),
            "full_l2_error": _stats(np.asarray(full_l2_errors, dtype=np.float64)),
            "base_position_l2_error_m": _stats(np.asarray(base_position_l2_errors, dtype=np.float64)),
            "base_rotation_l2_error_rad": _stats(np.asarray(base_rotation_l2_errors, dtype=np.float64)),
            "finger_active_mean_abs_error_rad": _stats(
                np.asarray(finger_active_mean_abs_errors, dtype=np.float64)
            ),
            "finger_active_max_abs_error_rad": _stats(
                np.asarray(finger_active_max_abs_errors, dtype=np.float64)
            ),
        },
        "hand_target_tracking_by_frame": hand_tracking_records if args.keep_trajectory else None,
        "contact_diagnostics": {
            "mode": "substep_aggregated",
            "expected_contact_source": "lance.contact",
            "force_source": "mjx_warp_decoded_contact_force_world",
            "force_note": (
                "Contact cone rows are decoded into world-frame 3D force, including "
                "tangential components, then aggregated by rigid body."
            ),
            "hand_object_contact_frames": int(np.count_nonzero(contact_counts_arr)),
            "hand_object_contact_count_max": int(contact_counts_arr.max()) if contact_counts_arr.size else 0,
            "hand_object_contact_substep_frames": int(np.count_nonzero(contact_substep_counts_arr)),
            "hand_object_contact_substep_count_max": (
                int(contact_substep_counts_arr.max()) if contact_substep_counts_arr.size else 0
            ),
            "expected_contact_frames": int(np.count_nonzero(expected_contact_arr)),
            "unexpected_contact_frames": int(np.count_nonzero(unexpected_contact_arr)),
            "first_hand_object_contact_step": int(first_hand_object_contact_step),
            "first_expected_contact_step": int(first_expected_contact_step),
            "first_unexpected_contact_step": int(first_unexpected_contact_step),
            "first_hand_object_contact_lance_frame": (
                int(start_frame + first_hand_object_contact_step)
                if first_hand_object_contact_step >= 0
                else None
            ),
            "first_expected_contact_lance_frame": (
                int(start_frame + first_expected_contact_step)
                if first_expected_contact_step >= 0
                else None
            ),
            "first_unexpected_contact_lance_frame": (
                int(start_frame + first_unexpected_contact_step)
                if first_unexpected_contact_step >= 0
                else None
            ),
            "object_contact_force_max_N": _stats(object_contact_max_arr),
            "object_contact_force_substep_sum_N": _stats(object_contact_sum_arr),
            "contact_force_body_max_N": {
                name: _stats(contact_body_max_arr[:, idx])
                for idx, name in enumerate(DEXHAND021PRO_CONTACT_BODY_NAMES)
            }
            if contact_body_max_arr.size
            else {},
            "contact_force_finger_max_N": {
                name: _stats(contact_finger_max_arr[:, idx])
                for idx, name in enumerate(FINGER_CONTACT_GROUPS)
            }
            if contact_finger_max_arr.size
            else {},
        },
        "contact_force_body_names": list(DEXHAND021PRO_CONTACT_BODY_NAMES),
        "contact_force_finger_names": list(FINGER_CONTACT_GROUPS),
        "contact_force_body_max_N_by_frame": (
            contact_force_body_max_by_frame if args.keep_trajectory else None
        ),
        "contact_force_body_substep_sum_N_by_frame": (
            contact_force_body_sum_by_frame if args.keep_trajectory else None
        ),
        "contact_force_finger_max_N_by_frame": (
            contact_force_finger_max_by_frame if args.keep_trajectory else None
        ),
        "contact_force_finger_substep_sum_N_by_frame": (
            contact_force_finger_sum_by_frame if args.keep_trajectory else None
        ),
        "object_contact_force_max_N_by_frame": (
            object_contact_force_max_by_frame if args.keep_trajectory else None
        ),
        "object_contact_force_substep_sum_N_by_frame": (
            object_contact_force_sum_by_frame if args.keep_trajectory else None
        ),
        "hand_object_contact_count_by_frame": hand_object_contact_counts if args.keep_trajectory else None,
        "hand_object_contact_substep_count_by_frame": (
            hand_object_contact_substep_counts if args.keep_trajectory else None
        ),
        "expected_contact_by_frame": expected_contact_by_frame if args.keep_trajectory else None,
        "unexpected_contact_by_frame": unexpected_contact_by_frame if args.keep_trajectory else None,
        "hand_object_contact_pairs_by_frame": hand_object_contact_pairs_by_frame if args.keep_trajectory else None,
        "hand_object_contact_frames": int(np.count_nonzero(hand_object_contact_counts)),
        "hand_object_contact_count_max": int(max(hand_object_contact_counts) if hand_object_contact_counts else 0),
        "object_positions": object_positions if args.keep_trajectory else None,
        "target_object_positions": traj.object_pos[start_frame : start_frame + frame_count].tolist()
        if args.keep_trajectory
        else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay DexHandRL Lance trajectory with MJX-Warp reference PD.")
    parser.add_argument("--lance", type=Path, default=DEFAULT_LANCE_PATH)
    parser.add_argument("--object", default="cube1")
    parser.add_argument("--action", default="01")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--uuid", default=None)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full trajectory")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Exact Lance frame to start replay from. Defaults to movement_start_frame - pre_contact_frames.",
    )
    parser.add_argument(
        "--pre-contact-frames",
        type=int,
        default=200,
        help="Isaac-style contact-window prefix used when --start-frame is omitted.",
    )
    parser.add_argument(
        "--object-reset-z-offset",
        type=float,
        default=0.01,
        help="Isaac reset_task_state adds this z clearance to the object at episode reset.",
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--settle-frames", type=int, default=0)
    parser.add_argument("--base-pos-kp", type=float, default=5000)
    parser.add_argument("--base-pos-kv", type=float, default=500)
    parser.add_argument("--base-pos-force", type=float, default=500)
    parser.add_argument("--base-rot-kp", type=float, default=10000)
    parser.add_argument("--base-rot-kv", type=float, default=125)
    parser.add_argument("--base-rot-force", type=float, default=100)
    parser.add_argument("--finger-kp", type=float, default=15)
    parser.add_argument("--finger-kv", type=float, default=0)
    parser.add_argument("--finger-force", type=float, default=50)
    parser.add_argument("--object-friction", default="0.5 0.01 0.001")
    parser.add_argument("--ground-friction", default="0.5 0.5 0")
    parser.add_argument(
        "--reference-joint-source",
        choices=("active", "full_raw"),
        default="active",
        help="active matches IsaacGym 22D targets; full_raw uses Lance's 20 raw MuJoCo finger joints.",
    )
    parser.add_argument("--naconmax", type=int, default=8192)
    parser.add_argument("--njmax", type=int, default=8192)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--isaac-source-root", type=Path, default=DEFAULT_ISAAC_SOURCE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "dexhandrl_reference_pd_metrics.json",
    )
    parser.add_argument(
        "--scene-copy",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "dexhandrl_reference_pd_scene.xml",
    )
    parser.add_argument("--keep-trajectory", action="store_true")
    args = parser.parse_args()
    result = replay(args)
    print(
        json.dumps(
            {
                "metrics": str(args.output),
                "scene_copy": str(args.scene_copy),
                "object_position_error_m": result["object_position_error_m"],
                "object_tracking": result["object_tracking"],
                "object_rotation_error_deg": result["object_rotation_error_deg"],
                "hand_target_tracking": result["hand_target_tracking"],
                "contact_diagnostics": result["contact_diagnostics"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
