#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.contact_schema import (  # noqa: E402
    make_contact_entry,
    make_contact_pair,
    transform_force_to_local,
    transform_point_to_local,
    validate_contact_sequence,
    write_contact_fixture,
)

PRIMARY_HAND_ASSET = Path("../../assets/mano_hand_s02/mjcf/mano_hand_s02_full_convex.xml")
FALLBACK_HAND_ASSET = Path("../../assets/mano_hand_s02/mjcf/mano_hand_s02.xml")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "mujoco_mjx_live_contact.json"


def import_numpy() -> Any:
    import numpy as np  # type: ignore[import-not-found]

    return np


def backend_relative_path(path: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), BACKEND_DIR)).as_posix()


def resolve_mano_asset() -> Path:
    primary = (BACKEND_DIR / PRIMARY_HAND_ASSET).resolve()
    if primary.exists():
        return primary
    fallback = (BACKEND_DIR / FALLBACK_HAND_ASSET).resolve()
    if fallback.exists():
        return fallback
    return primary


def build_sample_contact_sequence() -> list[list[dict[str, Any]]]:
    """Return a deterministic fixture shaped like Lance generated_data_schema contact."""
    force_normal_world = [0.0, 0.0, 4.0]
    force_tangential_world = [0.2, -0.1, 0.0]
    total_force_world = [
        force_normal_world[i] + force_tangential_world[i] for i in range(3)
    ]
    pos_world = [-0.136, 0.002, 0.031]

    pair = make_contact_pair(
        force_normal=force_normal_world,
        force_tangential=force_tangential_world,
        pos_world=pos_world,
        pos_wrist=pos_world,
        pos_joint=[-0.004, 0.001, 0.006],
        pos_object=[0.011, -0.003, -0.025],
    )
    entry = make_contact_entry(
        joint_name="index_dip",
        object_name="contact_object",
        total_force_world=total_force_world,
        total_force_wrist=total_force_world,
        total_force_joint=total_force_world,
        total_force_object=total_force_world,
        contact_pairs=[pair],
    )
    contact = [[], [entry]]
    validate_contact_sequence(contact)
    return contact


def write_sample_fixture(output_path: Path = DEFAULT_OUTPUT) -> Path:
    asset_path = resolve_mano_asset()
    contact = build_sample_contact_sequence()
    metadata = {
        "backend": "mujoco_mjx",
        "hand_asset": backend_relative_path(asset_path),
        "scene": "MANO hand with contact_object pressed into index_dip",
        "force_convention": "force_on_hand_link_by_object",
    }
    write_contact_fixture(output_path, contact, metadata=metadata)
    return output_path


def import_mujoco() -> Any:
    import mujoco  # type: ignore[import-not-found]

    return mujoco


def load_mujoco_model(asset_path: Path | None = None) -> tuple[Any, Any, Any]:
    """Load a MuJoCo model and allocate data."""
    mujoco = import_mujoco()
    path = asset_path or resolve_mano_asset()
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    return mujoco, model, data


def build_live_scene_xml(output_path: Path) -> Path:
    """Create a temporary MANO scene with a static object intersecting the hand.

    The object is intentionally simple and partly penetrates the palm/index side so
    the first Docker smoke test produces non-empty contact data deterministically.
    """
    source = resolve_mano_asset()
    xml = source.read_text(encoding="utf-8")
    asset_dir = source.parent
    xml = xml.replace('file="../meshes/', f'file="{asset_dir.parent.as_posix()}/meshes/')
    if "<option" not in xml:
        xml = xml.replace("<compiler angle=\"radian\" />", "<compiler angle=\"radian\" />\n  <option timestep=\"0.002\" iterations=\"80\" solver=\"Newton\" />")
    object_xml = """
    <body name="contact_object" pos="-0.070 0.000 0.075" quat="1 0 0 0">
      <geom name="contact_object_collision" type="sphere" size="0.075" material="default_material" contype="1" conaffinity="1" condim="3" friction="1 0.01 0.01" solref="0.004 1" solimp="0.99 0.999 0.00001" />
    </body>
"""
    if "name=\"contact_object\"" not in xml:
        xml = xml.replace("  </worldbody>", object_xml + "  </worldbody>")
    output_path.write_text(xml, encoding="utf-8")
    return output_path


def load_live_scene() -> tuple[Any, Any, Any, Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="contactbench_mujoco_"))
    scene_path = tmp_dir / "mano_contact_scene.xml"
    build_live_scene_xml(scene_path)
    mujoco = import_mujoco()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    return mujoco, model, data, scene_path


def mujoco_name(mujoco: Any, model: Any, object_type: Any, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, int(object_id))
    return name or f"unnamed_{int(object_id)}"


def geom_and_body_names(mujoco: Any, model: Any, geom_id: int) -> tuple[str, str, int]:
    geom_name = mujoco_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    body_id = int(model.geom_bodyid[int(geom_id)])
    body_name = mujoco_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return geom_name, body_name, body_id


def body_frame(model: Any, data: Any, body_id: int) -> tuple[list[float], list[list[float]]]:
    np = import_numpy()
    origin = np.asarray(data.xpos[int(body_id)], dtype=float).tolist()
    rotation = np.asarray(data.xmat[int(body_id)], dtype=float).reshape(3, 3).tolist()
    return origin, rotation


def body_id_by_name(mujoco: Any, model: Any, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {body_name}")
    return int(body_id)


def contact_force_on_geom1_world(mujoco: Any, model: Any, data: Any, contact_index: int) -> Any:
    """Return MuJoCo contact force in world frame using MuJoCo's geom1 convention.

    mj_contactForce returns a local contact-frame wrench. MuJoCo stores contact.frame
    as the contact basis in world coordinates; frame.T maps local force to world.
    The caller flips the sign when the hand link is geom2 so exported force always
    means force_on_hand_link_by_object.
    """
    np = import_numpy()
    wrench_contact = np.zeros(6, dtype=float)
    mujoco.mj_contactForce(model, data, int(contact_index), wrench_contact)
    frame_world_from_contact = np.asarray(
        data.contact[int(contact_index)].frame, dtype=float
    ).reshape(3, 3)
    return frame_world_from_contact.T @ wrench_contact[:3]


def normal_tangential_components(contact: Any, force_world: Iterable[float]) -> tuple[list[float], list[float]]:
    np = import_numpy()
    force = np.asarray(force_world, dtype=float)
    frame_world_from_contact = np.asarray(contact.frame, dtype=float).reshape(3, 3)
    normal_axis_world = frame_world_from_contact[0]
    normal = float(np.dot(force, normal_axis_world)) * normal_axis_world
    tangential = force - normal
    return normal.tolist(), tangential.tolist()


def classify_hand_object_contact(
    mujoco: Any,
    model: Any,
    contact: Any,
    object_body_names: set[str],
) -> tuple[int, int, str, str] | None:
    """Identify hand-vs-object contacts and return hand/object geom ids and body names."""
    geom1 = int(contact.geom1)
    geom2 = int(contact.geom2)
    _, body1_name, _ = geom_and_body_names(mujoco, model, geom1)
    _, body2_name, _ = geom_and_body_names(mujoco, model, geom2)

    body1_is_object = body1_name in object_body_names
    body2_is_object = body2_name in object_body_names
    if body1_is_object == body2_is_object:
        return None
    if body1_is_object:
        return geom2, geom1, body2_name, body1_name
    return geom1, geom2, body1_name, body2_name


def extract_contacts_from_mujoco(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    wrist_body_name: str = "palm",
    object_body_names: Iterable[str] = ("contact_object",),
    object_surface_radius: float | None = None,
    project_contact_to_object_surface: bool = False,
) -> list[dict[str, Any]]:
    """Convert the current MuJoCo data.contact array into Lance contact entries.

    This is the real extraction hook for a scene that includes the MANO hand and one
    or more object bodies. It uses mj_contactForce, contact.pos, geom/body name
    mapping, and body xpose/xmat transforms for wrist, joint, and object frames.
    """
    wrist_body_id = body_id_by_name(mujoco, model, wrist_body_name)
    wrist_origin, wrist_rotation = body_frame(model, data, wrist_body_id)
    object_names = {str(name) for name in object_body_names}
    np = import_numpy()
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"pairs": [], "total_world": np.zeros(3, dtype=float)}
    )

    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        classified = classify_hand_object_contact(mujoco, model, contact, object_names)
        if classified is None:
            continue

        hand_geom, object_geom, joint_name, object_name = classified
        _, _, joint_body_id = geom_and_body_names(mujoco, model, hand_geom)
        _, _, object_body_id = geom_and_body_names(mujoco, model, object_geom)
        joint_origin, joint_rotation = body_frame(model, data, joint_body_id)
        object_origin, object_rotation = body_frame(model, data, object_body_id)

        force_world_geom1 = contact_force_on_geom1_world(mujoco, model, data, contact_index)
        force_world = force_world_geom1 if hand_geom == int(contact.geom1) else -force_world_geom1
        raw_pos_world = np.asarray(contact.pos, dtype=float)
        pos_world_arr = raw_pos_world
        if project_contact_to_object_surface and object_surface_radius is not None:
            object_origin_arr = np.asarray(object_origin, dtype=float)
            object_rotation_arr = np.asarray(object_rotation, dtype=float)
            local = object_rotation_arr.T @ (raw_pos_world - object_origin_arr)
            local_norm = float(np.linalg.norm(local))
            if local_norm > 1.0e-12:
                local = local * (float(object_surface_radius) / local_norm)
                pos_world_arr = object_origin_arr + object_rotation_arr @ local
        pos_world = pos_world_arr.tolist()
        force_normal, force_tangential = normal_tangential_components(contact, force_world)

        pair = make_contact_pair(
            force_normal=force_normal,
            force_tangential=force_tangential,
            pos_world=pos_world,
            pos_wrist=transform_point_to_local(pos_world, wrist_origin, wrist_rotation),
            pos_joint=transform_point_to_local(pos_world, joint_origin, joint_rotation),
            pos_object=transform_point_to_local(pos_world, object_origin, object_rotation),
        )
        key = (joint_name, object_name)
        grouped[key]["pairs"].append(pair)
        grouped[key]["total_world"] += force_world
        grouped[key]["joint_frame"] = (joint_origin, joint_rotation)
        grouped[key]["object_frame"] = (object_origin, object_rotation)

    entries = []
    for (joint_name, object_name), values in sorted(grouped.items()):
        total_world = values["total_world"].tolist()
        _, joint_rotation = values["joint_frame"]
        _, object_rotation = values["object_frame"]
        entries.append(
            make_contact_entry(
                joint_name=joint_name,
                object_name=object_name,
                total_force_world=total_world,
                total_force_wrist=transform_force_to_local(total_world, wrist_rotation),
                total_force_joint=transform_force_to_local(total_world, joint_rotation),
                total_force_object=transform_force_to_local(total_world, object_rotation),
                contact_pairs=values["pairs"],
            )
        )
    return entries


def run_live_rollout(output_path: Path, steps: int = 3) -> Path:
    mujoco, model, data, scene_path = load_live_scene()
    contact_sequence: list[list[dict[str, Any]]] = []
    for _ in range(int(steps)):
        mujoco.mj_step(model, data)
        entries = extract_contacts_from_mujoco(
            mujoco,
            model,
            data,
            wrist_body_name="palm",
            object_body_names=("contact_object",),
        )
        contact_sequence.append(entries)
    contact_count = validate_contact_sequence(contact_sequence)
    if contact_count < 1:
        raise RuntimeError(
            f"MuJoCo live rollout produced no contacts. Scene: {scene_path}"
        )
    metadata = {
        "backend": "mujoco_mjx",
        "hand_asset": backend_relative_path(resolve_mano_asset()),
        "scene": "temporary_mano_hand_contact_object_scene",
        "force_convention": "force_on_hand_link_by_object",
        "source": "mujoco_live_rollout_mj_contactForce",
        "steps": int(steps),
        "contact_entries": contact_count,
    }
    write_contact_fixture(output_path, contact_sequence, metadata=metadata)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MuJoCo/MJX Lance contact data for ContactBench."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the generated MuJoCo contact JSON."
    )
    parser.add_argument("--steps", type=int, default=3, help="Live MuJoCo rollout steps.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Write the deterministic schema fixture instead of running MuJoCo. This is only for offline schema checks.",
    )
    args = parser.parse_args()

    output_path = write_sample_fixture(args.output) if args.synthetic else run_live_rollout(args.output, args.steps)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
