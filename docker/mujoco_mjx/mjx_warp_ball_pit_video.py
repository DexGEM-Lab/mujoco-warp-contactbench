#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "docker/mujoco_mjx"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.ball_pit.common import VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH, hand_pose_at, physics_dt, spec_from_args, video_frame_indices, write_video  # noqa: E402
from docker.mujoco_mjx.ball_pit_contact import build_ball_pit_scene_xml, hand_joint_addresses  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "logs/mjx_warp_ball_pit_rollout.json"
DEFAULT_VIDEO = REPO_ROOT / "logs/mjx_warp_ball_pit.mp4"
DEFAULT_SCENE_COPY = REPO_ROOT / "logs/mjx_warp_ball_pit_scene.xml"


def timed(label: str, func):
    start = time.perf_counter()
    out = func()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s", flush=True)
    return out, elapsed


def set_hand_pose_mjx(jnp: Any, dx: Any, frame: int, spec: Any, addresses: dict[str, tuple[int, int]]) -> Any:
    pos, euler = hand_pose_at(frame, spec)
    values = {
        "ARTx": pos[0],
        "ARTy": pos[1],
        "ARTz": pos[2],
        "ARRx": euler[0],
        "ARRy": euler[1],
        "ARRz": euler[2],
    }
    qpos = dx.qpos
    qvel = dx.qvel
    qacc = dx.qacc
    for name, (qadr, dadr) in addresses.items():
        qpos = qpos.at[qadr].set(float(values.get(name, 0.0)))
        qvel = qvel.at[dadr].set(0.0)
        qacc = qacc.at[dadr].set(0.0)
    return dx.replace(qpos=qpos, qvel=qvel, qacc=qacc)


def ball_position_stats(mujoco: Any, model: Any, qpos_frames: list[Any], ball_count: int) -> dict[str, Any]:
    import numpy as np

    data = mujoco.MjData(model)
    sample_indices = sorted({0, len(qpos_frames) // 2, max(0, len(qpos_frames) - 1)})
    samples = []
    for frame_idx in sample_indices:
        if not qpos_frames:
            continue
        data.qpos[:] = qpos_frames[frame_idx]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        positions = []
        for i in range(ball_count):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ball_{i:03d}")
            if bid >= 0:
                positions.append([float(v) for v in data.xpos[bid]])
        arr = np.asarray(positions, dtype=float)
        samples.append(
            {
                "frame": int(frame_idx),
                "count": int(len(positions)),
                "min_xyz": arr.min(axis=0).tolist() if len(positions) else [],
                "max_xyz": arr.max(axis=0).tolist() if len(positions) else [],
                "mean_xyz": arr.mean(axis=0).tolist() if len(positions) else [],
                "positions": positions[: min(32, len(positions))],
            }
        )
    return {"samples": samples}


def render_video(mujoco: Any, model: Any, qpos_frames: list[Any], video: Path) -> None:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    frames = []
    for qpos in qpos_frames:
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="ball_pit_cam")
        frames.append(renderer.render())
    try:
        write_video(video, frames, fps=VIDEO_FPS)
    finally:
        renderer.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import mujoco
    from mujoco import mjx

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected gpu backend, got {jax.default_backend()} devices={jax.devices()}")
    spec = spec_from_args(args)
    scene_path = build_ball_pit_scene_xml(Path(tempfile.mkdtemp(prefix="mjx_warp_video_")) / "ball_pit.xml", spec)
    args.scene_copy.parent.mkdir(parents=True, exist_ok=True)
    args.scene_copy.write_text(scene_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"scene={scene_path}", flush=True)
    print(f"scene_copy={args.scene_copy}", flush=True)
    model, t_model = timed("load_mj_model", lambda: mujoco.MjModel.from_xml_path(str(scene_path)))
    addresses = hand_joint_addresses(mujoco, model)
    locked_joint_names = sorted(addresses)
    if any(name.startswith("ball_") for name in locked_joint_names):
        raise RuntimeError(f"ball freejoint leaked into hand lock set: {locked_joint_names}")
    print(f"locked_hand_joints={locked_joint_names}", flush=True)
    mx, t_put = timed("mjx_put_model_warp", lambda: mjx.put_model(model, impl="warp"))
    dx, t_data = timed("mjx_make_data_warp", lambda: mjx.make_data(model, impl="warp", naconmax=args.naconmax, njmax=args.njmax))
    step = jax.jit(mjx.step)

    def rollout() -> tuple[Any, list[Any]]:
        nonlocal dx
        qpos_samples = []
        dx = set_hand_pose_mjx(jnp, dx, 0, spec, addresses)
        for _ in range(spec.settle_frames * spec.substeps):
            dx = set_hand_pose_mjx(jnp, dx, 0, spec, addresses)
            dx = step(mx, dx)
            dx = set_hand_pose_mjx(jnp, dx, 0, spec, addresses)
        render_indices = video_frame_indices(spec)
        for frame in range(spec.rollout_frames):
            dx = set_hand_pose_mjx(jnp, dx, frame, spec, addresses)
            for _ in range(spec.substeps):
                dx = set_hand_pose_mjx(jnp, dx, frame, spec, addresses)
                dx = step(mx, dx)
                dx = set_hand_pose_mjx(jnp, dx, frame, spec, addresses)
            if frame in render_indices:
                qpos_samples.append(dx.qpos)
        jax.block_until_ready(dx.qpos)
        return dx, qpos_samples

    def transfer_sampled_qpos(qpos_samples: list[Any]) -> list[Any]:
        if not qpos_samples:
            return []
        qpos_batch = jnp.stack(qpos_samples)
        qpos_batch = np.asarray(jax.device_get(qpos_batch))
        return [qpos_batch[i] for i in range(qpos_batch.shape[0])]

    (final_dx, qpos_samples), t_rollout = timed("mjx_warp_rollout", rollout)
    qpos_frames, t_transfer = timed("transfer_sampled_qpos", lambda: transfer_sampled_qpos(qpos_samples))
    stats = ball_position_stats(mujoco, model, qpos_frames, spec.ball_count)
    _, t_render = timed("render_video", lambda: render_video(mujoco, model, qpos_frames, args.video))
    payload = {
        "backend": "mjx_warp",
        "status": "ok",
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "mujoco_version": mujoco.__version__,
        "scene_xml": str(scene_path),
        "scene_copy": str(args.scene_copy),
        "video": str(args.video),
        "scenario": spec.scenario,
        "ball_count": spec.ball_count,
        "locked_hand_joints": locked_joint_names,
        "hand_roll_amplitude": spec.hand_roll_amplitude,
        "hand_pitch_amplitude": spec.hand_pitch_amplitude,
        "hand_yaw_amplitude": spec.hand_yaw_amplitude,
        "fps": spec.fps,
        "substeps": spec.substeps,
        "physics_dt": physics_dt(spec),
        "settle_frames": spec.settle_frames,
        "rollout_frames": spec.rollout_frames,
        "duration_seconds": spec.duration_seconds,
        "rendered_video_frames": len(qpos_frames),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "njnt": int(model.njnt),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "naconmax": args.naconmax,
        "njmax": args.njmax,
        "ball_position_stats": stats,
        "timings_seconds": {
            "load_mj_model": t_model,
            "mjx_put_model_warp": t_put,
            "mjx_make_data_warp": t_data,
            "mjx_warp_rollout": t_rollout,
            "transfer_sampled_qpos": t_transfer,
            "render_video": t_render,
        },
        "note": "Physics rollout uses MJX-Warp/JAX on GPU. Sampled qpos stays on device during rollout and is transferred to CPU only after trajectory generation for MuJoCo offscreen camera replay.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MJX-Warp ball-pit rollout and export camera MP4.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--scene-copy", type=Path, default=DEFAULT_SCENE_COPY)
    parser.add_argument("--scenario", default="filled_tank", choices=("sparse_debug", "filled_tank"))
    parser.add_argument("--ball-count", type=int, default=120)
    parser.add_argument("--ball-radius", type=float, default=0.03)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--settle-frames", type=int, default=10)
    parser.add_argument("--rollout-frames", type=int, default=1000)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--naconmax", type=int, default=8192)
    parser.add_argument("--njmax", type=int, default=8192)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
