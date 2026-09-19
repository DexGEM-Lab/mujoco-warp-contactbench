#!/usr/bin/env python
"""Render current ManoRL frozen-evaluation NPZ as actual/reference MP4."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.manorl.assets import compile_model, object_runtime
from sim.manorl.contracts import JOINT_DOF

REQUIRED_ARRAYS = (
    "actual_qpos",
    "reference_qpos_feasible",
    "actual_object_position",
    "reference_object_position",
    "actual_object_quaternion_xyzw",
    "reference_object_quaternion_xyzw",
)


def load_evaluation_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        missing = [
            name for name in REQUIRED_ARRAYS if name not in archive
        ]
        if missing:
            raise ValueError(
                "evaluation artifact is missing: " + ", ".join(missing)
            )
        arrays = {
            name: np.asarray(archive[name]) for name in archive.files
        }
    frames = len(arrays["actual_qpos"])
    if frames < 1:
        raise ValueError("evaluation artifact must contain at least one frame")
    if any(len(arrays[name]) != frames for name in REQUIRED_ARRAYS):
        raise ValueError("evaluation artifact arrays must share one frame count")
    if not all(np.isfinite(arrays[name]).all() for name in REQUIRED_ARRAYS):
        raise ValueError("evaluation artifact pose arrays must be finite")
    return arrays


def _set_pose(
    mujoco,
    model,
    data,
    *,
    hand_qpos: np.ndarray,
    object_position: np.ndarray,
    object_quaternion_xyzw: np.ndarray,
    object_qpos_address: int,
) -> None:
    data.qpos[:JOINT_DOF] = hand_qpos[:JOINT_DOF]
    data.qpos[
        object_qpos_address : object_qpos_address + 3
    ] = object_position
    data.qpos[
        object_qpos_address + 3 : object_qpos_address + 7
    ] = object_quaternion_xyzw[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)


def render_evaluation_artifact(
    artifact: dict[str, np.ndarray],
    output: Path,
    *,
    source_fps: int = 120,
    output_fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> Path:
    if source_fps < 1 or output_fps < 1 or source_fps % output_fps:
        raise ValueError(
            "source-fps must be a positive multiple of output-fps"
        )
    import imageio.v2 as imageio
    import mujoco

    mujoco, model = compile_model(
        object_type="cube2",
        hand_side="right",
        physics_timestep=1 / 480,
    )
    object_joint = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        object_runtime("cube2").free_joint_name,
    )
    object_qpos_address = int(model.jnt_qposadr[object_joint])
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.07, -0.20, 0.15)
    camera.distance = 0.72
    camera.azimuth = 145.0
    camera.elevation = -18.0
    model.vis.headlight.ambient = 0.4
    stride = source_fps // output_fps
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        Image = ImageDraw = None
    with imageio.get_writer(
        output,
        fps=output_fps,
        codec="libx264",
        macro_block_size=1,
    ) as writer:
        for source_index in range(0, len(artifact["actual_qpos"]), stride):
            _set_pose(
                mujoco,
                model,
                data,
                hand_qpos=artifact["actual_qpos"][source_index],
                object_position=artifact[
                    "actual_object_position"
                ][source_index],
                object_quaternion_xyzw=artifact[
                    "actual_object_quaternion_xyzw"
                ][source_index],
                object_qpos_address=object_qpos_address,
            )
            renderer.update_scene(data, camera=camera)
            actual = renderer.render().copy()
            _set_pose(
                mujoco,
                model,
                data,
                hand_qpos=artifact[
                    "reference_qpos_feasible"
                ][source_index],
                object_position=artifact[
                    "reference_object_position"
                ][source_index],
                object_quaternion_xyzw=artifact[
                    "reference_object_quaternion_xyzw"
                ][source_index],
                object_qpos_address=object_qpos_address,
            )
            renderer.update_scene(data, camera=camera)
            reference = renderer.render().copy()
            frame = np.concatenate((actual, reference), axis=1)
            if Image is not None:
                canvas = Image.fromarray(frame)
                draw = ImageDraw.Draw(canvas)
                elapsed = source_index / source_fps
                draw.rectangle((0, 0, 210, 30), fill=(0, 0, 0))
                draw.rectangle(
                    (width, 0, width + 230, 30), fill=(0, 0, 0)
                )
                draw.text(
                    (8, 8),
                    f"Policy actual  t={elapsed:.2f}s",
                    fill=(255, 255, 255),
                )
                draw.text(
                    (width + 8, 8),
                    f"Reference  t={elapsed:.2f}s",
                    fill=(255, 255, 255),
                )
                frame = np.asarray(canvas)
            writer.append_data(frame)
    renderer.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=int, default=120)
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    artifact = load_evaluation_artifact(args.artifact)
    output = render_evaluation_artifact(
        artifact,
        args.output,
        source_fps=args.source_fps,
        output_fps=args.output_fps,
        width=args.width,
        height=args.height,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
