#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parents[1]
for path in (REPO_ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.ball_pit.common import (  # noqa: E402
    BallPitSpec,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    ball_color_rgba,
    ball_initial_positions,
    camera_pose,
    container_wall_boxes,
    hand_pose_at,
    physics_dt,
    render_svg,
    spec_from_args,
    summarize_contact_payload,
    video_frame_indices,
    write_video,
)
from common.contact_schema import LANCE_GENERATED_SCHEMA, validate_contact_sequence  # noqa: E402
from export_contacts import (  # noqa: E402
    body_frame,
    body_id_by_name,
    build_live_scene_xml,
    extract_contacts_from_mujoco,
    import_mujoco,
    resolve_mano_asset,
)

DEFAULT_OUTPUT = REPO_ROOT / "logs/mujoco_mjx_ball_pit_contact.json"
DEFAULT_IMAGE = REPO_ROOT / "logs/mujoco_mjx_ball_pit.svg"
DEFAULT_VIDEO = REPO_ROOT / "logs/mujoco_mjx_ball_pit.mp4"
HAND_LINK_NAMES = {
    "palm",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
}


def _add_xml_before_worldbody_close(xml: str, block: str) -> str:
    return xml.replace("  </worldbody>", block + "  </worldbody>")


def build_ball_pit_scene_xml(scene_path: Path, spec: BallPitSpec) -> Path:
    build_live_scene_xml(scene_path)
    xml = scene_path.read_text(encoding="utf-8")
    # Remove the single-object smoke sphere from the older live scene.
    start = xml.find('    <body name="contact_object"')
    if start >= 0:
        end = xml.find('    </body>\n', start)
        if end >= 0:
            xml = xml[:start] + xml[end + len('    </body>\n') :]
    xml = xml.replace('contype="0" conaffinity="1"', 'contype="1" conaffinity="1"')
    if "<visual>" not in xml:
        xml = xml.replace(
            "  <asset>",
            "  <visual><global offwidth=\"960\" offheight=\"544\" />"
            "<headlight ambient=\"0.55 0.55 0.55\" diffuse=\"0.75 0.75 0.75\" specular=\"0.2 0.2 0.2\" />"
            "</visual>\n\n  <asset>",
        )
    dt = physics_dt(spec)
    option = f'<option timestep="{dt}" iterations="80" solver="Newton" />'
    if "<option" in xml:
        start = xml.find("<option")
        end = xml.find("/>", start)
        if start >= 0 and end >= 0:
            xml = xml[:start] + option + xml[end + 2 :]
    else:
        xml = xml.replace("<compiler angle=\"radian\" />", f"<compiler angle=\"radian\" />\n  {option}")
    cam = camera_pose(spec)
    cpx, cpy, cpz = cam["pos"]
    geoms = [
        '<light name="ball_pit_key" pos="0.34 -0.44 0.78" dir="-0.55 0.60 -0.75" diffuse="1.0 1.0 1.0" ambient="0.35 0.35 0.35" />',
        '<light name="ball_pit_fill" pos="-0.42 0.32 0.62" dir="0.50 -0.35 -0.65" diffuse="0.55 0.55 0.55" ambient="0.20 0.20 0.20" />',
        f'<camera name="ball_pit_cam" pos="{cpx} {cpy} {cpz}" xyaxes="0.723356 0.690476 0 -0.365832 0.383252 0.848106" fovy="{cam["fovy"]}" />',
    ]
    for wall_box in container_wall_boxes(spec):
        px, py, pz = wall_box["pos"]
        sx, sy, sz = wall_box["size"]
        rgba = " ".join(str(v) for v in wall_box["rgba"])
        geoms.append(
            f'<geom name="{wall_box["name"]}" type="box" pos="{px} {py} {pz}" size="{sx} {sy} {sz}" '
            f'contype="1" conaffinity="1" friction="0.8 0.02 0.01" rgba="{rgba}" />'
        )
    balls = []
    for i, pos in enumerate(ball_initial_positions(spec)):
        rgba = " ".join(str(v) for v in ball_color_rgba(i))
        balls.append(
            f'<body name="ball_{i:03d}" pos="{pos[0]} {pos[1]} {pos[2]}">\n'
            f'  <freejoint name="ball_{i:03d}_free" />\n'
            f'  <inertial pos="0 0 0" mass="{spec.ball_mass}" diaginertia="1e-6 1e-6 1e-6" />\n'
            f'  <geom name="ball_{i:03d}_collision" type="sphere" size="{spec.ball_radius}" contype="1" conaffinity="1" condim="3" friction="0.7 0.02 0.01" solref="0.006 1" solimp="0.95 0.995 0.0001" rgba="{rgba}" />\n'
            f'</body>'
        )
    block = "\n    " + "\n    ".join(geoms + balls) + "\n"
    xml = _add_xml_before_worldbody_close(xml, block)
    scene_path.write_text(xml, encoding="utf-8")
    return scene_path


HAND_BASE_JOINT_NAMES = {"ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"}


def is_hand_joint_name(name: str) -> bool:
    return name in HAND_BASE_JOINT_NAMES or name.startswith("j")


def hand_joint_addresses(mujoco: Any, model: Any) -> dict[str, tuple[int, int]]:
    addresses: dict[str, tuple[int, int]] = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not name or not is_hand_joint_name(name):
            continue
        addresses[name] = (int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]))
    return addresses


def set_hand_pose(mujoco: Any, model: Any, data: Any, frame: int, spec: BallPitSpec, addresses: dict[str, tuple[int, int]]) -> None:
    pos, euler = hand_pose_at(frame, spec)
    base_joint_values = {
        "ARTx": pos[0],
        "ARTy": pos[1],
        "ARTz": pos[2],
        "ARRx": euler[0],
        "ARRy": euler[1],
        "ARRz": euler[2],
    }
    for name, (qadr, dadr) in addresses.items():
        data.qpos[qadr] = float(base_joint_values.get(name, 0.0))
        data.qvel[dadr] = 0.0
        data.qacc[dadr] = 0.0
    mujoco.mj_forward(model, data)


def floating_base_offset(mujoco: Any, model: Any) -> list[float]:
    try:
        body_id = body_id_by_name(mujoco, model, "floating_base")
    except ValueError:
        return [0.0, 0.0, 0.0]
    return [float(v) for v in model.body_pos[body_id]]


def hand_qpos_vector(data: Any, addresses: dict[str, tuple[int, int]], base_offset: list[float]) -> list[float]:
    ordered_names = ("ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz")
    qpos = [0.0] * 26
    for idx, name in enumerate(ordered_names):
        if name in addresses:
            qpos[idx] = float(data.qpos[addresses[name][0]])
    for idx, value in enumerate(base_offset[:3]):
        qpos[idx] += float(value)
    finger_names = sorted(name for name in addresses if name not in set(ordered_names))
    for out_idx, name in enumerate(finger_names[:20], start=6):
        qpos[out_idx] = float(data.qpos[addresses[name][0]])
    return qpos


def hand_trajectory_frame(mujoco: Any, model: Any, data: Any, addresses: dict[str, tuple[int, int]], base_offset: list[float]) -> dict[str, Any]:
    palm_id = body_id_by_name(mujoco, model, "palm")
    palm_pos, _ = body_frame(model, data, palm_id)
    qpos = hand_qpos_vector(data, addresses, base_offset)
    return {
        "mano_global_pos": palm_pos,
        "mano_global_rot_aa": qpos[3:6],
        "mano_hand_pose": qpos[3:6] + [0.0] * 45,
        "mano_joint_pos": [palm_pos for _ in range(21)],
        "urdf_dof": qpos,
    }


def render_camera_video(mujoco: Any, model: Any, data: Any, frames: list[Any], video: Path) -> None:
    write_video(video, frames, fps=VIDEO_FPS)


def run_ball_pit(spec: BallPitSpec, output: Path | None = None, image: Path | None = None, video: Path | None = None) -> dict[str, Any]:
    mujoco = import_mujoco()
    tmp_dir = Path(tempfile.mkdtemp(prefix="contactbench_ball_pit_mujoco_"))
    scene_path = build_ball_pit_scene_xml(tmp_dir / "ball_pit.xml", spec)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    object_names = tuple(f"ball_{i:03d}" for i in range(spec.ball_count))
    hand_addresses = hand_joint_addresses(mujoco, model)
    hand_base_offset = floating_base_offset(mujoco, model)
    renderer = None
    rendered_frames: list[Any] = []
    render_indices = video_frame_indices(spec) if video is not None else set()
    if video is not None:
        renderer = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0
    for _ in range(spec.settle_frames * spec.substeps):
        mujoco.mj_step(model, data)

    contact: list[list[dict[str, Any]]] = []
    hand_trace: list[tuple[float, float, float]] = []
    contact_points: list[list[float]] = []
    final_ball_positions: list[list[float]] = []
    ball_positions_by_frame: list[list[list[float]]] = []
    contact_points_by_frame: list[list[list[float]]] = []
    hand_trajectory_frames: list[dict[str, Any]] = []
    for frame in range(spec.rollout_frames):
        set_hand_pose(mujoco, model, data, frame, spec, hand_addresses)
        hand_pos, _ = hand_pose_at(frame, spec)
        hand_trace.append(hand_pos)
        for _ in range(spec.substeps):
            set_hand_pose(mujoco, model, data, frame, spec, hand_addresses)
            mujoco.mj_step(model, data)
            set_hand_pose(mujoco, model, data, frame, spec, hand_addresses)
        if renderer is not None and frame in render_indices:
            renderer.update_scene(data, camera="ball_pit_cam")
            rendered_frames.append(renderer.render())
        hand_trajectory_frames.append(hand_trajectory_frame(mujoco, model, data, hand_addresses, hand_base_offset))
        entries = extract_contacts_from_mujoco(
            mujoco,
            model,
            data,
            wrist_body_name="palm",
            object_body_names=object_names,
            object_surface_radius=spec.ball_radius,
            project_contact_to_object_surface=False,
        )
        entries = [entry for entry in entries if entry["joint_name"] in HAND_LINK_NAMES]
        contact.append(entries)
        frame_points: list[list[float]] = []
        for entry in entries:
            for pair in entry["contact_pairs"]:
                contact_points.append(pair["pos_world"])
                frame_points.append(pair["pos_world"])
        contact_points_by_frame.append(frame_points)
        ball_positions_by_frame.append(
            [
                [float(v) for v in data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]]
                for name in object_names
            ]
        )

    for name in object_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            final_ball_positions.append([float(v) for v in data.xpos[bid]])

    entry_count = validate_contact_sequence(contact)
    if entry_count < 1:
        raise RuntimeError("MuJoCo ball-pit benchmark produced no hand-ball contacts")
    summary = summarize_contact_payload(contact)
    metadata = {
        "backend": "mujoco_mjx",
        "benchmark_case": spec.case_name,
        "source": "MuJoCo ball-pit rigid hand sweep using mj_contactForce",
        "hand_asset": str(resolve_mano_asset().relative_to(REPO_ROOT)),
        "force_convention": "force_on_hand_link_by_ball",
        "ball_count": spec.ball_count,
        "ball_radius": spec.ball_radius,
        "fps": spec.fps,
        "substeps": spec.substeps,
        "physics_dt": physics_dt(spec),
        "settle_frames": spec.settle_frames,
        "rollout_frames": spec.rollout_frames,
        "duration_seconds": spec.duration_seconds,
        "scene_xml": str(scene_path),
        "object_trajectory_note": "All balls are exported. rot_aa is zero because sphere orientation is contact-invariant here.",
        "hand_trajectory_note": "hands fields are exported from MuJoCo state. urdf_dof translation includes the MJCF floating_base offset so URDF mesh replay aligns with MuJoCo world contacts.",
        "hand_base_offset": hand_base_offset,
        "contact_position_note": "pos_world is the raw MuJoCo solver contact.pos value. It is not projected onto the contacted ball surface.",
        **summary,
    }
    object_trajectories = [
        {
            "object_name": name,
            "pos": [ball_positions_by_frame[frame][idx] for frame in range(len(ball_positions_by_frame))],
            "rot_aa": [[0.0, 0.0, 0.0] for _ in range(len(ball_positions_by_frame))],
        }
        for idx, name in enumerate(object_names)
    ]
    hand_trajectory = {
        key: [frame[key] for frame in hand_trajectory_frames]
        for key in ("mano_global_pos", "mano_global_rot_aa", "mano_hand_pose", "mano_joint_pos", "urdf_dof")
    }
    payload = {
        "schema": LANCE_GENERATED_SCHEMA,
        "note": "Real MuJoCo ball-pit hand-ball contact benchmark output.",
        "metadata": metadata,
        "hand_trajectory": hand_trajectory,
        "object_trajectories": object_trajectories,
        "contact": contact,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if image is not None:
        render_svg(
            image,
            spec=spec,
            backend="mujoco_mjx",
            ball_positions=final_ball_positions or ball_initial_positions(spec),
            hand_trace=hand_trace,
            contact_points=contact_points,
            summary=summary,
        )
    if video is not None:
        render_camera_video(mujoco, model, data, rendered_frames, video)
    return {
        "payload": payload,
        "metadata": metadata,
        "output": str(output) if output is not None else None,
        "image": str(image) if image is not None else None,
        "video": str(video) if video is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MuJoCo ball-pit contact benchmark.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--scenario", default="filled_tank", choices=("sparse_debug", "filled_tank"))
    parser.add_argument("--ball-count", type=int, default=120)
    parser.add_argument("--ball-radius", type=float, default=0.03)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--settle-frames", type=int, default=60)
    parser.add_argument("--rollout-frames", type=int, default=1000)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    args = parser.parse_args()
    result = run_ball_pit(spec_from_args(args), args.output, args.image, None if args.no_video else args.video)
    print(json.dumps({"metadata": result["metadata"], "output": result["output"], "image": result["image"], "video": result["video"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
