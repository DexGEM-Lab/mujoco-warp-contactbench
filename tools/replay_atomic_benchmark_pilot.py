#!/usr/bin/env python3
"""Replay five compact trajectories with MJX-Warp and render one benchmark video.

Physics executes saved absolute ``urdf_dof_target`` commands at 120 Hz with four
480 Hz MJX-Warp substeps.  Every object named by the compact row participates in
physics.  The host-only benchmark render mirror contains all nine task objects;
objects absent from the physical row remain static at the per-trajectory layout
position and every render-mirror contact bit is disabled.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

CONTROL_HZ = 120
PHYSICS_HZ = 480
VIDEO_STRIDE = 4
VIDEO_FPS = CONTROL_HZ // VIDEO_STRIDE
WIDTH = 640
HEIGHT = 360
ALL_OBJECTS = (
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
CONTRACT = "manorl.atomic-benchmark-target-dof-pilot.v1"


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


def wxyz(rot_aa: Sequence[float]) -> np.ndarray:
    xyzw = Rotation.from_rotvec(np.asarray(rot_aa, dtype=np.float64)).as_quat()
    return xyzw[[3, 0, 1, 2]]


def validate_gpu_binding(gpu: int) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    egl = os.environ.get("MUJOCO_EGL_DEVICE_ID", "").strip()
    if visible != str(gpu) or egl != str(gpu):
        raise RuntimeError(
            "GPU binding must use one identical physical index for compute and EGL: "
            f"--gpu={gpu}, CUDA_VISIBLE_DEVICES={visible!r}, "
            f"MUJOCO_EGL_DEVICE_ID={egl!r}"
        )
    return {
        "physical_gpu": gpu,
        "cuda_visible_devices": visible,
        "mujoco_egl_device_id": egl,
        "contract": "one_worker_one_matching_compute_egl_gpu_v1",
    }


def configure_modules(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    os.environ["MANORL_REPO_ROOT"] = str(args.manorl_root.resolve())
    os.environ["MANORL_EXPECTED_COMMIT"] = args.manorl_commit
    os.environ["MANORL_ASSET_MANIFEST"] = str(args.asset_manifest.resolve())
    os.environ["MANORL_DEXSTREAM_SKIN_FRAGMENT"] = str(
        args.asset_root
        / "hand/mano/cheyingtong/right/skin/mano_skin_mjcf_fragment.xml"
    )
    os.environ["BENCHMARK_SCENE_LIBRARY_ROOT"] = str(args.benchmark_root.resolve())
    client = str(args.client_root.resolve())
    manorl = str(args.manorl_root.resolve())
    for root in (client, manorl):
        if root not in sys.path:
            sys.path.insert(0, root)

    from scripts.eval import mjx_skin as mjx_skin
    import scripts.eval.mjx_skin.manorl_native_physics as native
    import scripts.eval.mjx_skin.mano_visual_contract as visual
    import scripts.eval.mjx_skin.scene_asset_catalog as scene_catalog

    sys.modules["scripts.eval.manorl_native_physics"] = native
    sys.modules["scripts.eval.mano_visual_contract"] = visual
    sys.modules["scripts.eval.scene_asset_catalog"] = scene_catalog
    from scripts.eval.mjx_skin import consumer_visual

    from sim.manorl import assets, contracts

    commit = subprocess.check_output(
        ["git", "-C", str(args.manorl_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != args.manorl_commit:
        raise RuntimeError(f"ManoRL source {commit} != requested {args.manorl_commit}")
    status = subprocess.check_output(
        ["git", "-C", str(args.manorl_root), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise RuntimeError(f"ManoRL source is dirty: {status.splitlines()[:8]}")
    if contracts.JOINT_DOF != 28 or tuple(contracts.JOINT_NAMES_28) != tuple(
        contracts.JOINT_NAMES
    ):
        raise RuntimeError("atomic replay requires the canonical right-hand 28D ABI")
    # The general ManoRL branch retains its older default clock. This dataset
    # and the production State45 runtime are explicitly 480 Hz / four substeps;
    # every compilation below therefore receives 1/480 rather than a default.
    native._RUNTIME = {
        "root": args.manorl_root.resolve(),
        "assets": assets,
        "contracts": contracts,
        "commit": commit,
    }
    assets.DEXSTREAM_ROOT = args.asset_root.resolve()

    # The benchmark renderer was written against the older Client keyword while
    # ManoRL dev uses the shorter ``object_collisions`` name.  Translate only at
    # this adapter boundary; the underlying implementation remains unchanged.
    if not getattr(assets, "_atomic_replay_keyword_compat", False):
        original_compile = assets.compile_unified_model
        original_build = assets.build_unified_scene_xml
        original_validate = assets.validate_unified_compiled_model

        def compile_compat(*positional: Any, object_object_collisions: bool = False, **kwargs: Any):
            kwargs.setdefault("physics_timestep", 1 / PHYSICS_HZ)
            collisions = bool(kwargs.pop("object_collisions", object_object_collisions))
            return original_compile(
                *positional, object_collisions=collisions, **kwargs
            )

        def build_compat(*positional: Any, object_object_collisions: bool = False, **kwargs: Any):
            kwargs.setdefault("physics_timestep", 1 / PHYSICS_HZ)
            collisions = bool(kwargs.pop("object_collisions", object_object_collisions))
            return original_build(
                *positional, object_collisions=collisions, **kwargs
            )

        def validate_compat(*positional: Any, object_object_collisions: bool = False, **kwargs: Any):
            kwargs.setdefault("physics_timestep", 1 / PHYSICS_HZ)
            collisions = bool(kwargs.pop("object_collisions", object_object_collisions))
            return original_validate(
                *positional, object_collisions=collisions, **kwargs
            )

        assets.compile_unified_model = compile_compat
        assets.build_unified_scene_xml = build_compat
        assets.validate_unified_compiled_model = validate_compat
        assets._atomic_replay_keyword_compat = True

    # New ManoRL already places the manifest-bound Cheyingtong skin in the
    # visual XML.  The historical Client injector is only for segmented s02.
    def existing_skin(root: Any) -> dict[str, Any]:
        skins = root.findall("./deformable/skin")
        if len(skins) != 1:
            raise ValueError(f"expected one manifest skin before benchmark decoration, got {len(skins)}")
        return {
            "contract": "manifest_bound_cheyingtong_skin_v1",
            "path": os.environ["MANORL_DEXSTREAM_SKIN_FRAGMENT"],
            "skin_name": skins[0].get("name"),
        }

    consumer_visual._inject_consumer_skin = existing_skin
    return native, visual, consumer_visual, assets, contracts


def selected_rows(dataset: Any, selection: Mapping[str, object], action: str) -> list[tuple[int, dict]]:
    values = selection.get("actions", {}).get(action)  # type: ignore[union-attr]
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError(f"selection must contain exactly five rows for action {action}")
    indices = [int(value) for value in values]
    rows = dataset.take(indices).to_pylist()
    result = []
    for index, row in zip(indices, rows, strict=True):
        gesture = str(row["trajectory_metadata"]["gesture"])
        if not gesture.startswith(action + "-"):
            raise ValueError(f"selected row {index} does not belong to action {action}: {gesture}")
        result.append((index, row))
    return result


def decode_row(row_index: int, row: dict) -> dict[str, Any]:
    metadata = row["trajectory_metadata"]
    provenance = row["provenance"]
    if (
        provenance["contract"] != "synthetic_mano_target_replay_visual_v2_contact"
        or metadata["data_fps"] != CONTROL_HZ
        or provenance["control_fps"] != CONTROL_HZ
        or provenance["physics_fps"] != PHYSICS_HZ
        or provenance["physics_substeps_per_control"] != 4
    ):
        raise ValueError(f"row {row_index} has an incompatible replay contract")
    names = tuple(metadata["object_names"])
    objects = row["objects"]
    if len(names) != len(objects) or len(set(names)) != len(names):
        raise ValueError(f"row {row_index} object slots are malformed")
    moves = metadata["trajectory_info"]["object_move"]
    if len(moves) != 1 or moves[0]["object_name"] not in names:
        raise ValueError(f"row {row_index} has no unique manipulated object")
    target = moves[0]["object_name"]
    frames = int(metadata["total_frames"])
    hand = row["hands"][0]
    hand_qpos = np.asarray(hand["urdf_dof"], dtype=np.float64)
    commands = np.asarray(hand["urdf_dof_target"], dtype=np.float64)
    object_pos = {
        name: np.asarray(record["pos"], dtype=np.float64)
        for name, record in zip(names, objects, strict=True)
    }
    object_quat = {
        name: Rotation.from_rotvec(np.asarray(record["rot_aa"], dtype=np.float64))
        .as_quat()[:, [3, 0, 1, 2]]
        for name, record in zip(names, objects, strict=True)
    }
    if hand_qpos.shape != (frames, 28) or commands.shape != (frames, 28):
        raise ValueError(f"row {row_index} hand arrays are malformed")
    if any(array.shape != (frames, 3) for array in object_pos.values()):
        raise ValueError(f"row {row_index} object position arrays are malformed")
    if any(array.shape != (frames, 4) for array in object_quat.values()):
        raise ValueError(f"row {row_index} object quaternion arrays are malformed")
    return {
        "row_index": row_index,
        "uuid": row["index"]["uuid"],
        "seed_uuid": row["index"]["seed_uuid"],
        "gesture": metadata["gesture"],
        "frames": frames,
        "names": names,
        "target": target,
        "movement": dict(moves[0]),
        "hand_recorded": hand_qpos,
        "commands": commands,
        "object_recorded_pos": object_pos,
        "object_recorded_quat_wxyz": object_quat,
        "provenance": provenance,
    }


def benchmark_invariance(mujoco: Any, physics_model: Any, visual_model: Any, table: Mapping[str, Any]) -> dict[str, Any]:
    if (physics_model.nq, physics_model.nv, physics_model.nu) != (
        visual_model.nq,
        visual_model.nv,
        visual_model.nu,
    ):
        raise ValueError("benchmark decoration changed nq/nv/nu")
    exact = (
        "jnt_type",
        "jnt_bodyid",
        "jnt_qposadr",
        "jnt_dofadr",
        "body_parentid",
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "actuator_trnid",
        "actuator_gainprm",
        "actuator_biasprm",
    )
    for name in exact:
        if not np.array_equal(np.asarray(getattr(physics_model, name)), np.asarray(getattr(visual_model, name))):
            raise ValueError(f"benchmark decoration changed {name}")
    if not np.isclose(physics_model.opt.timestep, visual_model.opt.timestep, rtol=0, atol=1e-15):
        raise ValueError("benchmark decoration changed physics timestep")
    for geom_id in range(physics_model.ngeom):
        name = mujoco.mj_id2name(physics_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "floor":
            continue
        if int(physics_model.geom_contype[geom_id]) == 0 and int(physics_model.geom_conaffinity[geom_id]) == 0:
            continue
        other = mujoco.mj_name2id(visual_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if other < 0:
            raise ValueError(f"benchmark decoration removed collision geom {name}")
        for field in ("geom_type", "geom_bodyid", "geom_contype", "geom_conaffinity"):
            if int(getattr(physics_model, field)[geom_id]) != int(getattr(visual_model, field)[other]):
                raise ValueError(f"benchmark decoration changed {field} for {name}")
        for field in ("geom_pos", "geom_quat", "geom_size", "geom_friction"):
            if not np.array_equal(np.asarray(getattr(physics_model, field)[geom_id]), np.asarray(getattr(visual_model, field)[other])):
                raise ValueError(f"benchmark decoration changed {field} for {name}")
    floor = mujoco.mj_name2id(visual_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0 or int(visual_model.geom_type[floor]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError("benchmark collidable tabletop is absent")
    if not np.allclose(visual_model.geom_pos[floor], table["top_pos"], atol=1e-12, rtol=0):
        raise ValueError("benchmark tabletop position changed")
    if not np.allclose(visual_model.geom_size[floor], table["top_size"], atol=1e-12, rtol=0):
        raise ValueError("benchmark tabletop size changed")
    return {
        "verified": True,
        "joint_actuator_body_contract_unchanged": True,
        "task_collision_geoms_unchanged": True,
        "intentional_table_delta": table["contract_id"],
    }


def build_benchmark_model(
    *,
    consumer_visual: Any,
    assets: Any,
    contracts: Any,
    target: str,
    object_types: Sequence[str],
    decorative_scene_spec: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    return consumer_visual.build_consumer_visual_model(
        target,
        WIDTH,
        HEIGHT,
        object_types=tuple(object_types),
        head_camera_preset="current",
        scene_style="collidable_table",
        background_style="photoreal_room",
        decorative_scene_spec=str(decorative_scene_spec),
        closed_ceiling=True,
        runtime={"assets": assets, "contracts": contracts},
        apply_visual_lod=lambda _root, _name: None,
        collidable_invariance_check=benchmark_invariance,
        visual_invariance_check=lambda _mujoco, _physics, _visual: {
            "verified": True
        },
        object_body_name=assets.object_runtime(target).body_name,
    )


def make_scene(
    consumer_visual: Any,
    assets: Any,
    contracts: Any,
    *,
    target: str,
    object_types: Sequence[str],
    decorative_scene_spec: Path,
    create_renderer: bool,
) -> tuple[Any, ...]:
    mujoco, model, invariance = build_benchmark_model(
        consumer_visual=consumer_visual,
        assets=assets,
        contracts=contracts,
        target=target,
        object_types=object_types,
        decorative_scene_spec=decorative_scene_spec,
    )
    data = mujoco.MjData(model)
    renderer = (
        mujoco.Renderer(model, width=WIDTH, height=HEIGHT) if create_renderer else None
    )
    target_address = int(model.joint(target + "_free").qposadr[0])
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in contracts.JOINT_NAMES_28
    ]
    hand_addresses = np.asarray(
        [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids], dtype=np.int64
    )
    return (
        mujoco,
        model,
        data,
        renderer,
        target_address,
        0,
        hand_addresses,
        np.arange(28),
        None,
        invariance,
    )


def run_physics(
    *,
    dataset_path: Path,
    dataset_version: int,
    decoded: list[dict[str, Any]],
    consumer_visual: Any,
    assets: Any,
    contracts: Any,
    decorative_scene_spec: Path,
    constraint_capacity: int = 4096,
    ccd_contacts_per_world: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import sim.manorl.environment as environment_module
    from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
    from sim.manorl.observations import SOURCE_ALIGNED_COMPATIBILITY
    from sim.manorl.trajectory import ReferenceTrajectory, TrajectoryBatch
    from sim.manorl.contracts import TrajectoryIdentity

    row_names = decoded[0]["names"]
    target = decoded[0]["target"]
    if any(item["names"] != row_names or item["target"] != target for item in decoded):
        raise ValueError("one action batch must have one physical object topology and target")
    trajectories = []
    for item in decoded:
        names = item["names"]
        active_index = names.index(target)
        movement = item["movement"]
        identity = TrajectoryIdentity(
            str(dataset_path),
            dataset_version,
            item["row_index"],
            active_index,
            item["uuid"],
            item["uuid"],
            item["provenance"]["source_identity"],
            0,
            item["frames"],
            int(movement["start_frame"]),
            int(movement["end_frame"]),
        )
        scene_pos = np.stack([item["object_recorded_pos"][name][0] for name in names])
        scene_quat_xyzw = np.stack(
            [item["object_recorded_quat_wxyz"][name][0][[1, 2, 3, 0]] for name in names]
        )
        active_pos = item["object_recorded_pos"][target]
        active_quat_xyzw = item["object_recorded_quat_wxyz"][target][:, [1, 2, 3, 0]]
        trajectories.append(
            ReferenceTrajectory(
                identity,
                1,
                np.arange(item["frames"], dtype=np.int64),
                np.arange(item["frames"], dtype=np.float64) / CONTROL_HZ,
                item["hand_recorded"],
                active_pos,
                active_pos,
                active_quat_xyzw,
                0.0,
                reference_fps=CONTROL_HZ,
                control_fps=CONTROL_HZ,
                movement_start_step=int(movement["start_frame"]),
                movement_end_step=int(movement["end_frame"]),
                scene_object_types=names,
                scene_object_initial_pos=scene_pos,
                scene_object_initial_quat_xyzw=scene_quat_xyzw,
            )
        )
    environment_module.compile_unified_model = lambda servo, *, object_types, **_kwargs: build_benchmark_model(
        consumer_visual=consumer_visual,
        assets=assets,
        contracts=contracts,
        target=target,
        object_types=tuple(object_types),
        decorative_scene_spec=decorative_scene_spec,
    )[:2]
    if constraint_capacity < 1 or ccd_contacts_per_world < 1:
        raise ValueError("solver and CCD capacities must be positive")
    config = EnvironmentConfig(
        num_envs=len(decoded),
        device="gpu",
        hand_side="right",
        reference_fps=CONTROL_HZ,
        control_fps=CONTROL_HZ,
        post_padding=0,
        compatibility=SOURCE_ALIGNED_COMPATIBILITY,
        residual_enabled=False,
        expected_contact_mode="five_fingertips",
        contact_capacity=1024 * len(decoded),
        constraint_capacity=constraint_capacity,
        unified_object_batch=True,
        warp_ccd_iterations=16,
        warp_ccd_contacts_per_world=ccd_contacts_per_world,
        warp_persistent_ccd_workspace=True,
        device_resident_controls=True,
        capture_transition_diagnostics=True,
        point_sampling_backend="numpy_per_env",
    )
    environment = MujocoManoEnvironment(TrajectoryBatch(tuple(trajectories)), config)
    model = environment.model
    names = tuple(environment._unified_object_types)
    if not np.isclose(model.opt.timestep, 1 / PHYSICS_HZ, rtol=0, atol=1e-15):
        raise RuntimeError(f"compiled physics timestep {model.opt.timestep} is not 1/480")
    if config.physics_substeps_per_control != 4:
        raise RuntimeError("120 Hz replay did not resolve four physics substeps")
    addresses = {name: int(model.joint(name + "_free").qposadr[0]) for name in names}
    qpos = np.asarray(environment.data.qpos).copy()
    qvel = np.zeros_like(np.asarray(environment.data.qvel))
    ctrl = np.zeros_like(np.asarray(environment.data.ctrl))
    for world, item in enumerate(decoded):
        qpos[world, :28] = item["hand_recorded"][0]
        ctrl[world] = item["commands"][0]
        for name in item["names"]:
            address = addresses[name]
            qpos[world, address : address + 3] = item["object_recorded_pos"][name][0]
            qpos[world, address + 3 : address + 7] = item[
                "object_recorded_quat_wxyz"
            ][name][0]
    device = lambda value: environment.jax.device_put(
        environment.jp.asarray(value), environment.device
    )
    environment.data = environment.data.replace(
        qpos=device(qpos),
        qvel=device(qvel),
        ctrl=device(ctrl),
        qacc_warmstart=device(np.zeros_like(qvel)),
        time=device(np.zeros(len(decoded))),
    )
    environment.data = environment._forward_fn(environment.data)
    outputs = []
    for item in decoded:
        frames = item["frames"]
        outputs.append(
            {
                **item,
                "hand_simulated": np.empty((frames, 28), dtype=np.float32),
                "object_simulated_pos": {
                    name: np.empty((frames, 3), dtype=np.float32) for name in item["names"]
                },
                "object_simulated_quat_wxyz": {
                    name: np.empty((frames, 4), dtype=np.float32) for name in item["names"]
                },
                "sim_time": np.arange(frames, dtype=np.float64) / CONTROL_HZ,
            }
        )
    max_frames = max(item["frames"] for item in decoded)
    for frame in range(max_frames):
        state = environment.producer.materialize_state(environment.data)
        for world, output in enumerate(outputs):
            if frame >= output["frames"]:
                continue
            output["hand_simulated"][frame] = state.qpos[world, :28]
            for name in output["names"]:
                address = addresses[name]
                output["object_simulated_pos"][name][frame] = state.qpos[
                    world, address : address + 3
                ]
                output["object_simulated_quat_wxyz"][name][frame] = state.qpos[
                    world, address + 3 : address + 7
                ]
        if frame == max_frames - 1:
            break
        next_targets = np.stack(
            [
                item["commands"][min(frame + 1, item["frames"] - 1)]
                for item in decoded
            ]
        )
        environment.data = environment.data.replace(ctrl=device(next_targets))
        for _ in range(config.physics_substeps_per_control):
            environment.data = environment._step_fn(environment.data)
            environment._check_warp_ccd_overflow()
        if frame and frame % 120 == 0:
            print("PHYSICS_FRAME", frame, "/", max_frames - 1, flush=True)
    metrics = []
    for output in outputs:
        target_pos_error = np.linalg.norm(
            output["object_simulated_pos"][target]
            - output["object_recorded_pos"][target],
            axis=1,
        )
        hand_error = np.abs(output["hand_simulated"] - output["hand_recorded"])
        context = {}
        for name in output["names"]:
            error = np.linalg.norm(
                output["object_simulated_pos"][name]
                - output["object_recorded_pos"][name],
                axis=1,
            )
            context[name] = {
                "max_position_error_m": float(error.max()),
                "final_position_error_m": float(error[-1]),
            }
        metric = {
            "row_index": output["row_index"],
            "uuid": output["uuid"],
            "frames": output["frames"],
            "target": target,
            "max_target_position_error_m": float(target_pos_error.max()),
            "final_target_position_error_m": float(target_pos_error[-1]),
            "max_hand_qpos_error": float(hand_error.max()),
            "objects": context,
        }
        if not all(
            np.isfinite(value).all()
            for value in (
                output["hand_simulated"],
                *output["object_simulated_pos"].values(),
                *output["object_simulated_quat_wxyz"].values(),
            )
        ):
            raise FloatingPointError(f"row {output['row_index']} dynamic replay is nonfinite")
        output["metrics"] = metric
        metrics.append(metric)
    return outputs, {
        "runtime": {
            "physics_backend": "mjx_warp",
            "right_hand_only": True,
            "control_hz": CONTROL_HZ,
            "physics_hz": PHYSICS_HZ,
            "physics_substeps_per_control": config.physics_substeps_per_control,
            "num_worlds": len(decoded),
            "object_types": list(names),
            "target_object": target,
            "object_object_collisions": True,
            "ccd_iterations": 16,
            "ccd_contacts_per_world": ccd_contacts_per_world,
            "constraint_capacity": constraint_capacity,
            "environment_clock": {
                "control_timestep": config.control_timestep,
                "physics_timestep": config.physics_timestep,
            },
        },
        "ccd": environment.warp_ccd_metadata(),
        "rows": metrics,
    }


def static_layout(layout_entry: Mapping[str, Any], names: Sequence[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    objects = layout_entry["objects"]
    for name in ALL_OBJECTS:
        if name in names:
            continue
        entry = objects[name]
        position = np.asarray(entry["pos"], dtype=np.float64)
        rot_aa = np.asarray(entry.get("rot_aa", (0.0, 0.0, 0.0)), dtype=np.float64)
        result[name] = (position, wxyz(rot_aa))
    return result


def label(frame: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(image)


def render_action(
    *,
    action: str,
    outputs: list[dict[str, Any]],
    layouts: Mapping[str, Mapping[str, Any]],
    visual: Any,
    consumer_visual: Any,
    assets: Any,
    contracts: Any,
    decorative_scene_spec: Path,
    output_dir: Path,
    gpu: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw

    scene = make_scene(
        consumer_visual,
        assets,
        contracts,
        target=outputs[0]["target"],
        object_types=ALL_OBJECTS,
        decorative_scene_spec=decorative_scene_spec,
        create_renderer=True,
    )
    mujoco, model, data, renderer = scene[:4]
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        )
        if body_name in ALL_OBJECTS:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
    addresses = {
        name: int(model.joint(name + "_free").qposadr[0]) for name in ALL_OBJECTS
    }
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    output_dir.mkdir(parents=True, exist_ok=False)
    video = output_dir / f"action{action}_five_trajectories.mp4"
    story_frames: list[Image.Image] = []
    row_reports = []
    with imageio.get_writer(
        video,
        fps=VIDEO_FPS,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    ) as writer:
        for sequence, trace in enumerate(outputs, 1):
            layout = layouts[trace["uuid"]]
            backgrounds = static_layout(layout, trace["names"])
            selected = list(range(0, trace["frames"], VIDEO_STRIDE))
            if selected[-1] != trace["frames"] - 1:
                selected.append(trace["frames"] - 1)
            story_targets = {
                min(selected, key=lambda value: abs(value - target))
                for target in np.linspace(0, trace["frames"] - 1, 6).round().astype(int)
            }
            first_combined = None
            for frame in selected:
                data.qpos[:28] = trace["hand_simulated"][frame]
                data.qvel[:] = 0.0
                data.time = frame / CONTROL_HZ
                for name in trace["names"]:
                    address = addresses[name]
                    data.qpos[address : address + 3] = trace["object_simulated_pos"][name][frame]
                    data.qpos[address + 3 : address + 7] = trace[
                        "object_simulated_quat_wxyz"
                    ][name][frame]
                for name, (position, quaternion) in backgrounds.items():
                    address = addresses[name]
                    data.qpos[address : address + 3] = position
                    data.qpos[address + 3 : address + 7] = quaternion
                mujoco.mj_forward(model, data)
                renderer.update_scene(
                    data, camera=visual.HEAD_CAMERA_NAME, scene_option=option
                )
                head = renderer.render().copy()
                renderer.update_scene(
                    data, camera=visual.WRIST_CAMERA_NAME, scene_option=option
                )
                wrist = renderer.render().copy()
                combined = np.concatenate([head, wrist], axis=1)
                combined = label(
                    combined,
                    f"action {action} | trajectory {sequence}/5 | row {trace['row_index']} | "
                    f"t={frame / CONTROL_HZ:.2f}s | head / right wrist",
                )
                if first_combined is None:
                    first_combined = combined
                    for _ in range(15):
                        writer.append_data(combined)
                writer.append_data(combined)
                if frame in story_targets:
                    story_frames.append(Image.fromarray(combined))
            row_dir = output_dir / f"row{trace['row_index']:04d}"
            row_dir.mkdir()
            arrays = {
                "hand_qpos": trace["hand_simulated"],
                "sim_time": trace["sim_time"],
            }
            for name in trace["names"]:
                arrays[f"{name}_position"] = trace["object_simulated_pos"][name]
                arrays[f"{name}_quaternion_wxyz"] = trace[
                    "object_simulated_quat_wxyz"
                ][name]
            np.savez_compressed(row_dir / "dynamic_trace.npz", **arrays)
            report = {
                **trace["metrics"],
                "seed_uuid": trace["seed_uuid"],
                "source_identity": trace["provenance"]["source_identity"],
                "gesture": trace["gesture"],
                "physical_objects": list(trace["names"]),
                "visual_only_objects": sorted(set(ALL_OBJECTS) - set(trace["names"])),
                "selected_video_frames": selected,
                "trace_sha256": sha256(row_dir / "dynamic_trace.npz"),
            }
            dump(row_dir / "report.json", report)
            row_reports.append(report)
    renderer.close()
    columns = 3
    panel_width, panel_height = WIDTH * 2, HEIGHT
    sheet = Image.new(
        "RGB",
        (panel_width * columns, panel_height * ((len(story_frames) + 2) // 3)),
        "white",
    )
    for index, image in enumerate(story_frames):
        sheet.paste(image, ((index % columns) * panel_width, (index // columns) * panel_height))
    storyboard = output_dir / "storyboard.jpg"
    sheet.save(storyboard, quality=90)
    result = {
        "contract": CONTRACT,
        "action": action,
        "video": str(video),
        "video_sha256": sha256(video),
        "storyboard": str(storyboard),
        "storyboard_sha256": sha256(storyboard),
        "rows": row_reports,
        "video_stride": VIDEO_STRIDE,
        "video_fps": VIDEO_FPS,
        "camera_layout": "head/right-wrist side-by-side",
        "benchmark_scene": str(decorative_scene_spec),
        "render_model_all_nine_objects": True,
        "render_mirror_never_stepped": True,
        "visual_only_object_collision_bits_zero": True,
        "gpu": gpu,
        "provenance": dict(provenance),
    }
    dump(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-version", type=int, default=2)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manorl-root", type=Path, required=True)
    parser.add_argument("--manorl-commit", required=True)
    parser.add_argument("--client-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output must be fresh: {args.output}")
    gpu_binding = validate_gpu_binding(args.gpu)
    selection = json.loads(args.selection.read_text())
    import lance

    dataset = lance.dataset(str(args.dataset), version=args.dataset_version)
    rows = selected_rows(dataset, selection, args.action)
    decoded = [decode_row(index, row) for index, row in rows]
    layout_payload = json.loads(args.layout.read_text())
    layouts = {entry["uuid"]: entry for entry in layout_payload["trajectories"]}
    if any(item["uuid"] not in layouts for item in decoded):
        raise ValueError("one or more selected rows have no scene layout")
    native, visual, consumer_visual, assets, contracts = configure_modules(args)
    outputs, physics_report = run_physics(
        dataset_path=args.dataset,
        dataset_version=args.dataset_version,
        decoded=decoded,
        assets=assets,
        contracts=contracts,
        consumer_visual=consumer_visual,
        decorative_scene_spec=args.scene,
    )
    provenance = {
        "dataset": str(args.dataset),
        "dataset_version": args.dataset_version,
        "dataset_rows": dataset.count_rows(),
        "layout": str(args.layout),
        "layout_sha256": sha256(args.layout),
        "asset_manifest": str(args.asset_manifest),
        "asset_manifest_sha256": sha256(args.asset_manifest),
        "asset_root": str(args.asset_root),
        "asset_commit": assets.asset_provenance()["asset_source_commit"],
        "manorl_root": str(args.manorl_root),
        "manorl_commit": args.manorl_commit,
        "client_root": str(args.client_root),
        "client_commit": os.popen(f"git -C {args.client_root} rev-parse HEAD").read().strip(),
        "selection": str(args.selection),
        "selection_sha256": sha256(args.selection),
        "gpu_binding": gpu_binding,
        "physics": physics_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = render_action(
        action=args.action,
        outputs=outputs,
        layouts=layouts,
        visual=visual,
        consumer_visual=consumer_visual,
        assets=assets,
        contracts=contracts,
        decorative_scene_spec=args.scene,
        output_dir=args.output,
        gpu=args.gpu,
        provenance=provenance,
    )
    print(json.dumps({"action": args.action, "video": result["video"]}, indent=2))


if __name__ == "__main__":
    main()
