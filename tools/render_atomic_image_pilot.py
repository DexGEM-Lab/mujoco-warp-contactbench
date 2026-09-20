#!/usr/bin/env python3
"""Render atomic-task replay pilots from saved Cheyingtong compact states.

This is a kinematic render mirror: it restores the saved right-hand and scene
object poses for every selected frame, calls ``mj_forward`` and renders.  It
never invokes a policy or advances physics, so visual-only background objects
cannot perturb the saved motion.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ACTIONS = ("001", "002", "003", "004", "005", "006", "007", "009")
SCENE_OBJECTS = (
    "egg_ellipsoid",
    "egg_stick_rack",
    "cylinder7",
    "egg_cup",
    "trash_bin",
    "bowl",
    "cuboid1",
    "mayonnaisebottle",
    "pitcherbase",
)
CONTROL_HZ = 120
PHYSICS_HZ = 480
HEAD_CAMERA = {
    "name": "atomic_head_camera",
    "position": (0.0, -1.2, 0.8),
    "target": (0.0, -0.1, 0.1),
    "horizontal_fov": 65.0,
}
WRIST_CAMERA = {
    "name": "atomic_right_wrist_camera",
    "position": (-0.08, 0.0, -0.08),
    "target": (0.06, 0.0, -0.05),
    "horizontal_fov": 95.0,
    "parent_body": "ARRz_link",
}
CONTRACT = "manorl.atomic-image-replay-pilot.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def camera_xyaxes(position: Sequence[float], target: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    eye = np.asarray(position, dtype=np.float64)
    lookat = np.asarray(target, dtype=np.float64)
    if eye.shape != (3,) or lookat.shape != (3,) or not np.isfinite([*eye, *lookat]).all():
        raise ValueError("camera position and target must be finite XYZ vectors")
    forward = lookat - eye
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-12:
        raise ValueError("camera position and target must differ")
    forward /= norm
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) <= 1e-12:
        right = np.cross(forward, np.asarray([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up


def horizontal_to_vertical_fov(horizontal_fov_deg: float, width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        raise ValueError("render width and height must be positive")
    return float(
        np.rad2deg(
            2
            * np.arctan(
                np.tan(np.deg2rad(float(horizontal_fov_deg)) / 2) * height / width
            )
        )
    )


def add_cameras(xml: str, *, width: int, height: int) -> str:
    root = ET.fromstring(xml)
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(width))
    global_visual.set("offheight", str(height))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("scene XML has no worldbody")

    def add(spec: Mapping[str, object], parent: ET.Element) -> None:
        right, up = camera_xyaxes(spec["position"], spec["target"])  # type: ignore[arg-type]
        ET.SubElement(
            parent,
            "camera",
            {
                "name": str(spec["name"]),
                "pos": " ".join(f"{float(value):.9g}" for value in spec["position"]),  # type: ignore[union-attr]
                "xyaxes": " ".join(f"{float(value):.9g}" for value in (*right, *up)),
                "fovy": f"{horizontal_to_vertical_fov(float(spec['horizontal_fov']), width, height):.9g}",
            },
        )

    add(HEAD_CAMERA, worldbody)
    wrist = next(
        (
            body
            for body in worldbody.iter("body")
            if body.get("name") == WRIST_CAMERA["parent_body"]
        ),
        None,
    )
    if wrist is None:
        raise ValueError(f"missing wrist-camera parent {WRIST_CAMERA['parent_body']!r}")
    add(WRIST_CAMERA, wrist)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def select_rows_by_action(
    metadata_rows: Sequence[Mapping[str, object]],
    actions: Sequence[str],
    overrides: Mapping[str, int] | None = None,
) -> dict[str, int]:
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(actions)
    if unknown:
        raise ValueError(f"row overrides name unrequested actions: {sorted(unknown)}")
    selected: dict[str, int] = {}
    for action in actions:
        if action in overrides:
            index = int(overrides[action])
            if not 0 <= index < len(metadata_rows):
                raise ValueError(f"row override for action {action} is out of range: {index}")
            gesture = str(metadata_rows[index]["trajectory_metadata"]["gesture"])  # type: ignore[index]
            if not gesture.startswith(action + "-"):
                raise ValueError(
                    f"row {index} gesture {gesture!r} does not match action {action}"
                )
            selected[action] = index
            continue
        for index, row in enumerate(metadata_rows):
            gesture = str(row["trajectory_metadata"]["gesture"])  # type: ignore[index]
            if gesture.startswith(action + "-"):
                selected[action] = index
                break
        else:
            raise ValueError(f"dataset has no row for action {action}")
    return selected


def layout_by_uuid(payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError("scene layout has no trajectories list")
    result: dict[str, Mapping[str, object]] = {}
    for entry in trajectories:
        if not isinstance(entry, dict) or not isinstance(entry.get("uuid"), str):
            raise ValueError("scene layout contains an invalid trajectory entry")
        uuid = str(entry["uuid"])
        if uuid in result:
            raise ValueError(f"duplicate scene-layout UUID {uuid}")
        result[uuid] = entry
    return result


def row_pose_sources(
    row: Mapping[str, object],
    layout_entry: Mapping[str, object],
    *,
    scene_objects: Sequence[str] = SCENE_OBJECTS,
) -> dict[str, dict[str, object]]:
    metadata = row["trajectory_metadata"]  # type: ignore[index]
    names = list(metadata["object_names"])  # type: ignore[index]
    objects = list(row["objects"])  # type: ignore[arg-type]
    if len(names) != len(objects) or len(set(names)) != len(names):
        raise ValueError("row scene object names and arrays do not align")
    layout_objects = layout_entry.get("objects")
    if not isinstance(layout_objects, dict):
        raise ValueError("scene-layout trajectory has no objects mapping")
    result: dict[str, dict[str, object]] = {}
    by_name = dict(zip(names, objects, strict=True))
    frame_count = int(metadata["total_frames"])  # type: ignore[index]
    for name in scene_objects:
        layout = layout_objects.get(name)
        if not isinstance(layout, dict):
            raise ValueError(f"scene layout is missing {name!r}")
        if name in by_name:
            record = by_name[name]
            pos = np.asarray(record["pos"], dtype=np.float64)  # type: ignore[index]
            rot_aa = np.asarray(record["rot_aa"], dtype=np.float64)  # type: ignore[index]
            if pos.shape != (frame_count, 3) or rot_aa.shape != (frame_count, 3):
                raise ValueError(f"recorded pose arrays for {name!r} are malformed")
            quat = Rotation.from_rotvec(rot_aa).as_quat()
            source = "recorded"
        else:
            position = np.asarray(layout.get("pos"), dtype=np.float64)
            rotation = np.asarray(layout.get("rot_aa", (0.0, 0.0, 0.0)), dtype=np.float64)
            if position.shape != (3,) or rotation.shape != (3,) or not np.isfinite([*position, *rotation]).all():
                raise ValueError(f"background pose for {name!r} is malformed")
            pos = np.broadcast_to(position, (frame_count, 3)).copy()
            quat = np.broadcast_to(Rotation.from_rotvec(rotation).as_quat(), (frame_count, 4)).copy()
            source = "layout_static"
        if not np.isfinite(pos).all() or not np.isfinite(quat).all():
            raise ValueError(f"scene poses for {name!r} contain nonfinite values")
        result[name] = {"position": pos, "quaternion_xyzw": quat, "source": source}
    return result


def parse_overrides(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        action, separator, row = value.partition("=")
        if not separator or action in result:
            raise ValueError(f"invalid row override {value!r}; use ACTION=ROW once")
        result[action] = int(row)
    return result


def storyboard(frames: Sequence[tuple[int, np.ndarray, np.ndarray]], output: Path) -> None:
    from PIL import Image, ImageDraw

    if not frames:
        raise ValueError("storyboard requires at least one frame")
    panels = []
    for frame, head, wrist in frames:
        joined = np.concatenate([head, wrist], axis=1)
        panel = Image.fromarray(joined)
        canvas = Image.new("RGB", (panel.width, panel.height + 26), "white")
        canvas.paste(panel, (0, 0))
        ImageDraw.Draw(canvas).text(
            (6, panel.height + 6), f"frame {frame} | head / right wrist", fill="black"
        )
        panels.append(canvas)
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (panels[0].width * columns, panels[0].height * rows), "white"
    )
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * panel.width, (index // columns) * panel.height))
    sheet.save(output, quality=90)


def render_row(
    *,
    mujoco: object,
    model: object,
    row_index: int,
    row: Mapping[str, object],
    layout_entry: Mapping[str, object],
    output: Path,
    width: int,
    height: int,
    video_stride: int,
    max_frames: int | None,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    import imageio.v2 as imageio

    metadata = row["trajectory_metadata"]  # type: ignore[index]
    action = str(metadata["gesture"]).split("-", 1)[0]  # type: ignore[index]
    hand = np.asarray(row["hands"][0]["urdf_dof"], dtype=np.float64)  # type: ignore[index]
    total_frames = int(metadata["total_frames"])  # type: ignore[index]
    if hand.shape != (total_frames, 28) or not np.isfinite(hand).all():
        raise ValueError("right-hand saved qpos is malformed")
    rendered_frames = total_frames if max_frames is None else min(total_frames, max_frames)
    if rendered_frames < 1:
        raise ValueError("no frames selected for rendering")
    poses = row_pose_sources(row, layout_entry)
    data = mujoco.MjData(model)  # type: ignore[attr-defined]
    addresses = {
        name: int(model.joint(name + "_free").qposadr[0]) for name in SCENE_OBJECTS  # type: ignore[attr-defined]
    }
    selected_frames = list(range(0, rendered_frames, video_stride))
    if selected_frames[-1] != rendered_frames - 1:
        selected_frames.append(rendered_frames - 1)
    story_targets = set(
        int(value)
        for value in np.linspace(0, rendered_frames - 1, min(12, rendered_frames)).round()
    )
    story_selected = {
        min(selected_frames, key=lambda candidate: abs(candidate - target))
        for target in story_targets
    }
    output.mkdir(parents=True, exist_ok=False)
    head_path, wrist_path = output / "head.mp4", output / "right_wrist.mp4"
    renderer = mujoco.Renderer(model, width=width, height=height)  # type: ignore[attr-defined]
    scene_option = mujoco.MjvOption()  # type: ignore[attr-defined]
    scene_option.geomgroup[3] = 0
    saved_story: list[tuple[int, np.ndarray, np.ndarray]] = []
    fps = CONTROL_HZ / video_stride
    with imageio.get_writer(
        head_path, fps=fps, codec="libx264", quality=8, macro_block_size=2
    ) as head_writer, imageio.get_writer(
        wrist_path, fps=fps, codec="libx264", quality=8, macro_block_size=2
    ) as wrist_writer:
        for frame in selected_frames:
            data.qpos[:28] = hand[frame]
            data.qvel[:] = 0.0
            data.time = frame / CONTROL_HZ
            for name in SCENE_OBJECTS:
                address = addresses[name]
                pose = poses[name]
                data.qpos[address : address + 3] = pose["position"][frame]  # type: ignore[index]
                xyzw = pose["quaternion_xyzw"][frame]  # type: ignore[index]
                data.qpos[address + 3 : address + 7] = xyzw[[3, 0, 1, 2]]
            mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
            renderer.update_scene(
                data, camera=HEAD_CAMERA["name"], scene_option=scene_option
            )
            head = renderer.render().copy()
            renderer.update_scene(
                data, camera=WRIST_CAMERA["name"], scene_option=scene_option
            )
            wrist = renderer.render().copy()
            head_writer.append_data(head)
            wrist_writer.append_data(wrist)
            if frame in story_selected:
                saved_story.append((frame, head, wrist))
    renderer.close()
    saved_story.sort(key=lambda item: item[0])
    storyboard(saved_story, output / "storyboard.jpg")
    result = {
        "contract": CONTRACT,
        "action": action,
        "row_index": row_index,
        "uuid": str(row["index"]["uuid"]),  # type: ignore[index]
        "seed_uuid": str(row["index"]["seed_uuid"]),  # type: ignore[index]
        "gesture": str(metadata["gesture"]),  # type: ignore[index]
        "source_total_frames": total_frames,
        "rendered_source_frames": rendered_frames,
        "video_stride": video_stride,
        "video_fps": fps,
        "video_frame_count": len(selected_frames),
        "selected_source_frames": selected_frames,
        "storyboard_source_frames": [item[0] for item in saved_story],
        "width": width,
        "height": height,
        "cameras": {"head": HEAD_CAMERA, "right_wrist": WRIST_CAMERA},
        "scene_objects": {
            name: {
                "source": poses[name]["source"],
                "layout_role": layout_entry["objects"][name]["role"],  # type: ignore[index]
                "initial_position": poses[name]["position"][0].tolist(),  # type: ignore[index]
            }
            for name in SCENE_OBJECTS
        },
        "kinematic_replay": True,
        "policy_inference": False,
        "physics_integration": False,
        "render_model_collision_bits_zero": bool(
            np.all(np.asarray(model.geom_contype) == 0)  # type: ignore[attr-defined]
            and np.all(np.asarray(model.geom_conaffinity) == 0)  # type: ignore[attr-defined]
        ),
        "head_sha256": sha256(head_path),
        "right_wrist_sha256": sha256(wrist_path),
        "storyboard_sha256": sha256(output / "storyboard.jpg"),
        "provenance": dict(provenance),
    }
    dump(output / "metadata.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-version", type=int, default=2)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actions", nargs="+", default=list(ACTIONS))
    parser.add_argument("--row", action="append", default=[], metavar="ACTION=ROW")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output must be a fresh path: {args.output}")
    if args.width <= 0 or args.height <= 0 or args.video_stride <= 0:
        raise ValueError("render dimensions and video stride must be positive")
    if len(set(args.actions)) != len(args.actions) or not set(args.actions) <= set(ACTIONS):
        raise ValueError(f"actions must be unique members of {ACTIONS}")
    os.environ["MANORL_ASSET_MANIFEST"] = str(args.asset_manifest.resolve())
    from sim.manorl import assets
    from sim.manorl.contracts import ServoConfig
    import lance
    import mujoco

    assets.DEXSTREAM_ROOT = args.asset_root.resolve()
    dataset = lance.dataset(str(args.dataset), version=args.dataset_version)
    metadata_rows = dataset.to_table(columns=["index", "trajectory_metadata"]).to_pylist()
    selected = select_rows_by_action(metadata_rows, args.actions, parse_overrides(args.row))
    layout_payload = json.loads(args.layout.read_text(encoding="utf-8"))
    layouts = layout_by_uuid(layout_payload)
    if int(layout_payload.get("total_trajectories", -1)) != dataset.count_rows():
        raise ValueError("scene-layout row count differs from Lance")
    xml = assets.build_unified_scene_xml(
        object_types=SCENE_OBJECTS,
        object_collisions=False,
        visual_meshes=True,
        hand_side="right",
        physics_timestep=1 / PHYSICS_HZ,
    )
    model = mujoco.MjModel.from_xml_string(
        add_cameras(xml, width=args.width, height=args.height)
    )
    assets.validate_unified_compiled_model(
        mujoco,
        model,
        ServoConfig(),
        object_types=SCENE_OBJECTS,
        object_collisions=False,
        hand_side="right",
        physics_timestep=1 / PHYSICS_HZ,
    )
    if model.nskin != 1:
        raise ValueError(f"expected exactly one Cheyingtong skin, got {model.nskin}")
    collision_mask = (np.asarray(model.geom_contype) != 0) | (
        np.asarray(model.geom_conaffinity) != 0
    )
    collision_geoms_disabled = int(collision_mask.sum())
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    if collision_geoms_disabled == 0:
        raise ValueError("render model unexpectedly had no collision geoms to disable")
    asset_commit = assets.asset_provenance()["asset_source_commit"]
    if asset_commit != json.loads(args.asset_manifest.read_text())["source_commit"]:
        raise ValueError("asset root and dataset manifest name different commits")
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "dataset": str(args.dataset.resolve()),
        "dataset_version": args.dataset_version,
        "dataset_rows": dataset.count_rows(),
        "layout": str(args.layout.resolve()),
        "layout_sha256": sha256(args.layout),
        "asset_manifest": str(args.asset_manifest.resolve()),
        "asset_manifest_sha256": sha256(args.asset_manifest),
        "asset_root": str(args.asset_root.resolve()),
        "asset_commit": asset_commit,
        "control_hz": CONTROL_HZ,
        "physics_hz_provenance_only": PHYSICS_HZ,
        "scene_contract": "checker-floor, nine DexStream object visuals, no contact bits",
        "hand_contract": "Cheyingtong right MANO URDF plus manifest skin",
        "collision_geoms_disabled": collision_geoms_disabled,
        "model_nskin": int(model.nskin),
        "model_ncam": int(model.ncam),
    }
    results = []
    for action in args.actions:
        row_index = selected[action]
        row = dataset.take([row_index]).to_pylist()[0]
        uuid = str(row["index"]["uuid"])
        if uuid not in layouts:
            raise ValueError(f"scene layout has no entry for row UUID {uuid}")
        result = render_row(
            mujoco=mujoco,
            model=model,
            row_index=row_index,
            row=row,
            layout_entry=layouts[uuid],
            output=args.output / action,
            width=args.width,
            height=args.height,
            video_stride=args.video_stride,
            max_frames=args.max_frames,
            provenance=provenance,
        )
        results.append(result)
        print(
            "RENDERED",
            action,
            "row",
            row_index,
            "frames",
            result["video_frame_count"],
            flush=True,
        )
    dump(
        args.output / "manifest.json",
        {
            "contract": CONTRACT,
            "actions": list(args.actions),
            "selected_rows": selected,
            "rows": results,
            "provenance": provenance,
            "action008_present": False,
        },
    )


if __name__ == "__main__":
    main()
