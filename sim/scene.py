#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.benchmarks.ball_pit.common import (  # noqa: E402
    BallPitSpec,
    ball_color_rgba,
    ball_initial_positions,
    camera_pose,
    container_wall_boxes,
    hand_pose_at,
    physics_dt,
)

SIM_DIR = Path(__file__).resolve().parent
PRIMARY_HAND_ASSET = Path("../assets/mano_hand_s02/mjcf/mano_hand_s02_full_convex.xml")
FALLBACK_HAND_ASSET = Path("../assets/mano_hand_s02/mjcf/mano_hand_s02.xml")

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
HAND_BASE_JOINT_NAMES = {"ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"}


def import_mujoco() -> Any:
    import mujoco  # type: ignore[import-not-found]

    return mujoco


def resolve_mano_asset() -> Path:
    primary = (SIM_DIR / PRIMARY_HAND_ASSET).resolve()
    if primary.exists():
        return primary
    fallback = (SIM_DIR / FALLBACK_HAND_ASSET).resolve()
    if fallback.exists():
        return fallback
    return primary


def backend_relative_path(path: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), SIM_DIR)).as_posix()


def build_live_scene_xml(output_path: Path) -> Path:
    source = resolve_mano_asset()
    xml = source.read_text(encoding="utf-8")
    asset_dir = source.parent
    xml = xml.replace('file="../meshes/', f'file="{asset_dir.parent.as_posix()}/meshes/')
    if "<option" not in xml:
        xml = xml.replace("<compiler angle=\"radian\" />", "<compiler angle=\"radian\" />\n  <option timestep=\"0.002\" iterations=\"80\" solver=\"Newton\" />")
    output_path.write_text(xml, encoding="utf-8")
    return output_path


def _add_xml_before_worldbody_close(xml: str, block: str) -> str:
    return xml.replace("  </worldbody>", block + "  </worldbody>")


def build_ball_pit_scene_xml(scene_path: Path, spec: BallPitSpec) -> Path:
    build_live_scene_xml(scene_path)
    xml = scene_path.read_text(encoding="utf-8")
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


def set_hand_pose_mjx(jnp: Any, dx: Any, frame: int, spec: BallPitSpec, addresses: dict[str, tuple[int, int]]) -> Any:
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
