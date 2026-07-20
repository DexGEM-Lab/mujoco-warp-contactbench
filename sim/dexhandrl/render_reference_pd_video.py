#!/usr/bin/env python3
"""Render DexHandRL MuJoCo/MJX reference-PD replay videos."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.dexhandrl.constants import DEFAULT_ISAAC_SOURCE_ROOT, DEFAULT_LANCE_PATH  # noqa: E402
from sim.dexhandrl.hand_mapping import (  # noqa: E402
    active_to_full_dof,
    dexhand021pro_route_joint_upper_limits,
    full_dof_from_qpos,
    make_ctrl_vector,
    raw_finger_to_full_dof,
    set_hand_qpos,
    set_object_freejoint,
)
from sim.dexhandrl.lance_loader import load_trajectory  # noqa: E402
from sim.dexhandrl.scene import build_dexhandrl_scene_xml  # noqa: E402


def _device_get(jax: Any, value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


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


def _make_camera(mujoco: Any, args: argparse.Namespace, default_lookat: np.ndarray) -> Any:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    lookat = (
        np.asarray(args.camera_target, dtype=np.float64)
        if args.camera_target is not None
        else default_lookat
    )
    camera.lookat[:] = lookat
    if args.camera_pos is not None:
        camera_pos = np.asarray(args.camera_pos, dtype=np.float64)
        delta = camera_pos - lookat
        distance = float(np.linalg.norm(delta))
        if distance <= 0:
            raise ValueError("--camera-pos must be different from --camera-target")
        horizontal_distance = float(np.linalg.norm(delta[:2]))
        camera.distance = distance
        camera.azimuth = float(np.degrees(np.arctan2(delta[1], delta[0])))
        # MuJoCo's free camera uses negative elevation for a camera above lookat.
        camera.elevation = -float(np.degrees(np.arctan2(delta[2], horizontal_distance)))
    else:
        camera.distance = float(args.camera_distance)
        camera.azimuth = float(args.camera_azimuth)
        camera.elevation = float(args.camera_elevation)
    return camera


def render_video(args: argparse.Namespace) -> None:
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
    frame_count = traj.frames - start_frame if args.max_frames <= 0 else min(traj.frames - start_frame, args.max_frames)

    scene_path = build_dexhandrl_scene_xml(
        output_path=Path(tempfile.mkdtemp(prefix="dexhandrl_mjx_render_")) / "dexhandrl.xml",
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
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(args.width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(args.height))
    render_data = mujoco.MjData(model)
    ids = __import__("sim.dexhandrl.hand_mapping", fromlist=["resolve_mujoco_ids"]).resolve_mujoco_ids(
        mujoco,
        model,
        object_name=traj.object_name,
    )
    route_joint_upper_limits = dexhand021pro_route_joint_upper_limits(mujoco, model)

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
    dx = dx.replace(ctrl=jnp.asarray(make_ctrl_vector(initial_full, ids.actuator_ids, model.nu), dtype=dx.ctrl.dtype))
    dx = forward(mx, dx)

    lookat = np.asarray(traj.object_pos[min(traj.movement_start_frame, traj.frames - 1)], dtype=np.float64)
    lookat[2] += 0.08
    camera = _make_camera(mujoco, args, lookat)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.output), fps=args.fps, macro_block_size=1)
    written = 0
    try:
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
            dx = dx.replace(ctrl=jnp.asarray(make_ctrl_vector(full, ids.actuator_ids, model.nu), dtype=dx.ctrl.dtype))
            for _ in range(args.substeps):
                dx = step(mx, dx)

            if local_frame % args.render_every != 0:
                continue
            qpos = _device_get(jax, dx.qpos)
            render_data.qpos[:] = qpos
            render_data.qvel[:] = _device_get(jax, dx.qvel)
            mujoco.mj_forward(model, render_data)
            renderer.update_scene(render_data, camera=camera)
            writer.append_data(renderer.render())
            written += 1
            if args.progress_interval and local_frame % args.progress_interval == 0:
                print(
                    f"rendered frame local={local_frame}/{frame_count} "
                    f"lance={lance_frame} written={written}",
                    flush=True,
                )
    finally:
        writer.close()
        renderer.close()

    print(
        f"saved video {args.output} ({written} frames, seq={args.sequence_index}, uuid={traj.uuid})",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a DexHandRL MJX-Warp reference-PD replay.")
    parser.add_argument("--lance", type=Path, default=DEFAULT_LANCE_PATH)
    parser.add_argument("--object", default="cube1")
    parser.add_argument("--action", default="01")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--uuid", default=None)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--pre-contact-frames", type=int, default=200)
    parser.add_argument("--object-reset-z-offset", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--base-pos-kp", type=float, default=5000)
    parser.add_argument("--base-pos-kv", type=float, default=500)
    parser.add_argument("--base-pos-force", type=float, default=500)
    parser.add_argument("--base-rot-kp", type=float, default=10000)
    parser.add_argument("--base-rot-kv", type=float, default=125)
    parser.add_argument("--base-rot-force", type=float, default=100)
    parser.add_argument("--finger-kp", type=float, default=15)
    parser.add_argument("--finger-kv", type=float, default=0)
    parser.add_argument("--finger-force", type=float, default=50)
    parser.add_argument("--reference-joint-source", choices=("active", "full_raw"), default="active")
    parser.add_argument("--naconmax", type=int, default=8192)
    parser.add_argument("--njmax", type=int, default=8192)
    parser.add_argument("--isaac-source-root", type=Path, default=DEFAULT_ISAAC_SOURCE_ROOT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--camera-pos", type=float, nargs=3, default=None)
    parser.add_argument("--camera-target", type=float, nargs=3, default=None)
    parser.add_argument("--camera-distance", type=float, default=0.42)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-23.0)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_video(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
