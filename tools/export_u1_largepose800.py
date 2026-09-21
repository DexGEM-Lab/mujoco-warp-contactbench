#!/usr/bin/env python3
"""Export a new exact800 by interleaving immutable exact640 rows with action005."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

import numpy as np

from sim.manorl.u1_action005 import (
    PARENT_ROW_BY_SLOT,
    SETTING,
    VERSION as ACTION005_VERSION,
    action005_child_uuid,
    load_action005_parent,
)
from sim.manorl.u1_campaign import array_sha, digest, file_sha, write_json
from sim.manorl.u1_export import compact_contacts, native_frame, require
from tools.export_u1_largepose640 import (
    ACTIONS as SOURCE_ACTIONS,
    CONTRACT as SOURCE_CONTRACT,
    schema as source_schema,
)
from tools.run_u1_action005_largepose_campaign import (
    campaign_identity,
    campaign_plan,
)
from tools.run_u1_largepose_campaign import PREFIX_FRAMES, make_target

CONTRACT = "u1_five_action_largepose800_c1_native_v1"
AUGMENTATION_CONTRACT = "u1-five-action-largepose-v1-exact-discrete-c1"
ACTIONS = ("003", "005", "006", "007", "009")
EXPECTED_COUNTS = {action: 160 for action in ACTIONS}
SOURCE_COUNTS = {action: 160 for action in SOURCE_ACTIONS}
_PARENT_CACHE: dict[tuple[str, int], tuple] = {}


def schema():
    source = source_schema()
    metadata = dict(source.metadata or {})
    metadata[b"schema_version"] = CONTRACT.encode()
    metadata[b"augmentation_contract"] = AUGMENTATION_CONTRACT.encode()
    return source.with_metadata(metadata)


def source640_index_for_output(output_index: int) -> int | None:
    """Map exact800 row order 003,005,006,007,009 onto exact640 rows."""

    if not 0 <= output_index < 800:
        raise ValueError("output row must lie in0..799")
    if output_index < 160:
        return output_index
    if output_index < 320:
        return None
    return output_index - 160


def _load_parent_cached(registry: Path, row: int):
    key = (str(registry.resolve()), row)
    if key not in _PARENT_CACHE:
        _PARENT_CACHE[key] = load_action005_parent(registry, row)
    return _PARENT_CACHE[key]


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit_source640(source: Path, *, verify_payload_hashes: bool) -> dict:
    import lance

    require(source.is_dir(), "exact640 source directory is missing")
    hash_path = source / "sha256.json"
    hash_sidecar = source / "sha256.json.sha256"
    require(hash_path.is_file() and hash_sidecar.is_file(), "exact640 hash evidence")
    require(
        hash_sidecar.read_text().strip() == file_sha(hash_path),
        "exact640 hash index digest",
    )
    hashes = json.loads(hash_path.read_text())
    expected_files = {
        str(path.relative_to(source))
        for path in source.rglob("*")
        if path.is_file() and path.name not in ("sha256.json", "sha256.json.sha256")
    }
    require(set(hashes) == expected_files, "exact640 hash coverage")
    if verify_payload_hashes:
        for name, expected in hashes.items():
            require(file_sha(source / name) == expected, "exact640 file changed: " + name)
    dataset = lance.dataset(str(source / "compact.lance"))
    require(dataset.version == 1 and dataset.count_rows() == 640, "exact640 Lance")
    require(
        dataset.schema.equals(source_schema(), check_metadata=True),
        "exact640 schema",
    )
    rows = dataset.to_table(columns=["index", "trajectory_metadata"]).to_pylist()
    uuids = [row["index"]["uuid"] for row in rows]
    actions = [row["trajectory_metadata"]["gesture"].split("-")[0] for row in rows]
    require(len(set(uuids)) == 640, "exact640 UUID uniqueness")
    require(Counter(actions) == Counter(SOURCE_COUNTS), "exact640 quotas")
    require(
        actions == [action for action in SOURCE_ACTIONS for _ in range(160)],
        "exact640 order",
    )
    manifest = json.loads((source / "manifest.json").read_text())
    validation = json.loads((source / "validation.json").read_text())
    require(manifest["contract"] == SOURCE_CONTRACT, "exact640 contract")
    require(manifest["ordered_uuids"] == uuids, "exact640 manifest order")
    require(validation["validated"] is True, "exact640 source validation")
    return {
        "path": str(source.resolve()),
        "rows": 640,
        "ordered_uuids": uuids,
        "hash_index_sha256": file_sha(hash_path),
        "manifest_sha256": file_sha(source / "manifest.json"),
        "validation_sha256": file_sha(source / "validation.json"),
    }


def collect_action005(registry: Path, plan_path: Path, attempts: Path):
    plan = json.loads(plan_path.read_text())
    require(plan == campaign_plan(registry), "saved action005 plan changed")
    rows = _read_rows(attempts / "005.jsonl")
    identity = campaign_identity(registry, plan)
    require(all(row["identity"] == identity for row in rows), "action005 ledger identity")
    require(
        all(row["status"] in ("started", "selected", "rejected", "exhausted") for row in rows),
        "action005 ledger status",
    )
    require(not any(row["status"] == "exhausted" for row in rows), "action005 exhausted slot")
    starts: dict[str, dict] = {}
    selected: dict[int, dict] = {}
    latest: dict[str, dict] = {}
    audit = []
    for row in rows:
        uid = row.get("uuid")
        if uid is not None:
            latest[uid] = row
        if row["status"] == "started":
            require(uid not in starts, "duplicate action005 start")
            starts[uid] = row
        elif row["status"] == "selected":
            require(row["slot"] not in selected, "duplicate action005 slot")
            selected[row["slot"]] = row
        elif row["status"] == "rejected":
            audit.append(row)
    require(set(selected) == set(range(160)), "action005 exact160 slots")
    require(
        all(row["status"] in ("selected", "rejected") for row in latest.values()),
        "unfinished action005 attempt",
    )
    registry_sha = file_sha(registry)
    records = []
    for slot in range(160):
        chosen = selected[slot]
        uid = chosen["uuid"]
        require(latest[uid]["status"] == "selected", "superseded action005 selection")
        require(uid in starts, "missing action005 start")
        started = starts[uid]
        parent_row = PARENT_ROW_BY_SLOT[slot]
        require(
            chosen["parent_row"] == started["parent_row"] == parent_row,
            "action005 parent assignment",
        )
        _registry, parent, inp, model, base, teacher = _load_parent_cached(
            registry, parent_row
        )
        require(
            chosen["parent_uuid"] == started["parent_uuid"] == parent["parent_uuid"],
            "action005 parent UUID",
        )
        candidate = started["candidate"]
        require(candidate in plan["slots"][slot]["candidates"], "candidate outside plan")
        expected_uuid = action005_child_uuid(
            registry_sha256=registry_sha,
            plan_digest=plan["digest"],
            slot=slot,
            candidate=candidate,
            parent_uuid=parent["parent_uuid"],
        )
        require(uid == expected_uuid, "action005 deterministic UUID")
        folder = attempts / "005" / uid
        actual = {
            str(path.resolve()) for path in folder.rglob("*") if path.is_file()
        }
        require(actual == set(chosen["artifacts"]), "action005 artifact coverage")
        for path, expected in chosen["artifacts"].items():
            require(file_sha(path) == expected, "action005 artifact changed: " + path)
        reports = []
        targets = []
        expected_frames = parent["frames"] + PREFIX_FRAMES
        for phase in ("first", "second"):
            replay = folder / phase
            required = {
                "result.json",
                "trace.npz",
                "target.npy",
                "native_contacts.jsonl",
                "contact_validation.npz",
                "geometry_contacts.json",
            }
            require(
                all(str((replay / name).resolve()) in chosen["artifacts"] for name in required),
                "missing action005 replay artifact",
            )
            report = json.loads((replay / "result.json").read_text())
            require(report["accepted"] and all(report["gates"].values()), "failed action005 replay")
            require(
                report["contract"] == parent["contract"]
                and report["semantics"] == parent["semantics"]
                and report["augmentation_contract"] == ACTION005_VERSION
                and report["parent_row"] == parent_row
                and report["parent_uuid"] == parent["parent_uuid"],
                "action005 replay lineage",
            )
            require(report["prefix_frames"] == PREFIX_FRAMES, "action005 prefix")
            require(not report["prefix_force_contacts"], "action005 prefix force contact")
            target = np.load(replay / "target.npy", allow_pickle=False)
            require(
                target.shape == (expected_frames, 28) and np.isfinite(target).all(),
                "action005 target shape",
            )
            require(
                array_sha(target) == report["executed_target_sha256"],
                "action005 executed target hash",
            )
            with np.load(replay / "trace.npz", allow_pickle=False) as trace:
                require(
                    trace["qpos"].shape == (expected_frames, model.nq)
                    and trace["qvel"].shape == (expected_frames, model.nv),
                    "action005 trace shape",
                )
                require(
                    np.isfinite(trace["qpos"]).all()
                    and np.isfinite(trace["qvel"]).all(),
                    "action005 finite trace",
                )
                require(np.array_equal(trace["ctrl"], target), "action005 control trace")
                q0 = inp.initial["qpos"].copy()
                q0[:6] += np.asarray(candidate["delta"], dtype=np.float64)
                require(
                    np.allclose(trace["qpos"][0], q0, atol=1e-7, rtol=0),
                    "action005 initial pose",
                )
                require(
                    np.allclose(
                        trace["qvel"][0], inp.initial["qvel"], atol=1e-7, rtol=0
                    ),
                    "action005 initial velocity",
                )
            lines = (replay / "native_contacts.jsonl").read_text().splitlines()
            require(len(lines) == expected_frames, "action005 native frame count")
            for frame, line in enumerate(lines):
                native_frame(json.loads(line), frame, model.ngeom)
            reports.append(report)
            targets.append(target)
        require(
            all(isinstance(report["pid"], int) and report["pid"] > 0 for report in reports)
            and reports[0]["pid"] != reports[1]["pid"],
            "action005 distinct replay PIDs",
        )
        require(np.array_equal(targets[0], targets[1]), "action005 frozen target")
        requested = make_target(
            base,
            np.asarray(candidate["delta"], dtype=np.float64),
            PREFIX_FRAMES,
            teacher["qpos"][:2, :28],
        )
        require(
            reports[0]["requested_target_sha256"] == array_sha(requested),
            "action005 first requested target",
        )
        require(
            reports[1]["requested_target_sha256"] == array_sha(targets[0]),
            "action005 second requested target",
        )
        require(
            np.max(np.abs(targets[0].astype(float) - requested)) < 1e-6,
            "action005 requested target mismatch",
        )
        require(
            targets[0][PREFIX_FRAMES:].tobytes() == base.astype(np.float32).tobytes(),
            "action005 parent suffix changed",
        )
        records.append(
            {
                "action": "005",
                "slot": slot,
                "uuid": uid,
                "candidate": candidate,
                "parent_row": parent_row,
                "parent_uuid": parent["parent_uuid"],
                "folder": str(folder.resolve()),
            }
        )
        if (slot + 1) % 20 == 0:
            print(json.dumps({"phase": "action005_audit", "selected": slot + 1}), flush=True)
    require(len({record["uuid"] for record in records}) == 160, "action005 UUID uniqueness")
    require(
        Counter(record["parent_row"] for record in records)
        == Counter({row: 16 for row in range(10)}),
        "action005 per-parent quotas",
    )
    return plan, records, audit


def movement_record(parent: dict) -> dict:
    move = dict(parent["movement"])
    require(move["object_name"] == "mayonnaisebottle", "action005 movement object")
    start = int(move["start_frame"])
    end = int(move["end_frame"])
    require(0 <= start <= end < parent["frames"], "action005 movement interval")
    return {
        "object_name": move["object_name"],
        "start_frame": start + PREFIX_FRAMES,
        "end_frame": end + PREFIX_FRAMES,
    }


def _bind_mano_assets(registry: dict, parent: dict) -> None:
    from sim.manorl import assets
    from sim.manorl import mano_pose

    manifest = Path(registry["assets"]["manifest"])
    require(file_sha(manifest) == registry["assets"]["manifest_sha256"], "asset manifest")
    require(
        parent["setting"]["asset_manifest_sha256"] == registry["assets"]["manifest_sha256"],
        "parent asset identity",
    )
    os.environ["MANORL_ASSET_MANIFEST"] = str(manifest)
    assets.DEXSTREAM_ROOT = Path(registry["assets"]["root"])
    assets.ASSET_MANIFEST = manifest
    assets._asset_manifest.cache_clear()
    require(json.loads(manifest.read_text())["hand_operator"] == "cheyingtong", "hand operator")
    mano_pose._source_axes.cache_clear()


def make_action005_row(registry_path: Path, record: dict, plan: dict) -> dict:
    import mujoco as mj
    from scipy.spatial.transform import Rotation as Rotation

    from sim.manorl.contracts import KEYPOINT_NAMES
    from sim.manorl.environment import _FINGERTIP_LOCAL_OFFSETS, _FINGERTIP_NAMES
    from sim.manorl.lance_v2 import FORCE_DIRECTION_CONTRACT, SYNTHETIC_LANCE_CONTRACT
    from sim.manorl.mano_pose import right_urdf_trajectory_to_mano_48d

    registry, parent, inp, model, _base, teacher = _load_parent_cached(
        registry_path, record["parent_row"]
    )
    _bind_mano_assets(registry, parent)
    replay = Path(record["folder"]) / "second"
    frames = parent["frames"] + PREFIX_FRAMES
    with np.load(replay / "trace.npz", allow_pickle=False) as trace:
        physical = {key: trace[key].copy() for key in ("qpos", "qvel", "ctrl")}
    qpos = physical["qpos"]
    qvel = physical["qvel"]
    ctrl = physical["ctrl"]
    require(qpos.shape == (frames, model.nq), "action005 export qpos")
    require(qvel.shape == (frames, model.nv), "action005 export qvel")
    require(ctrl.shape == (frames, 28), "action005 export ctrl")
    lines = (replay / "native_contacts.jsonl").read_text().splitlines()
    require(len(lines) == frames, "action005 export native contacts")
    data = mj.MjData(model)
    contacts = []
    joints = []
    keypoint_ids = [model.body(name).id for name in KEYPOINT_NAMES]
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        mj.mj_kinematics(model, data)
        tips = [
            data.xpos[model.body(name).id]
            + data.xmat[model.body(name).id].reshape(3, 3) @ offset
            for name, offset in zip(
                _FINGERTIP_NAMES, _FINGERTIP_LOCAL_OFFSETS, strict=True
            )
        ]
        joints.append(np.concatenate([data.xpos[keypoint_ids], tips], axis=0))
        contacts.append(
            compact_contacts(
                native_frame(json.loads(lines[frame]), frame, model.ngeom),
                model,
                data,
                parent["names"],
            )
        )
    hand_qpos = qpos[:, :28]
    hand = {
        "hand_name": "right",
        "mano_global_pos": hand_qpos[:, :3].tolist(),
        "mano_global_rot_aa": Rotation.from_euler(
            "XYZ", hand_qpos[:, 3:6]
        ).as_rotvec().tolist(),
        "mano_hand_pose": right_urdf_trajectory_to_mano_48d(hand_qpos).tolist(),
        "mano_joint_pos": np.asarray(joints).tolist(),
        "urdf_dof": hand_qpos.tolist(),
        "urdf_dof_target": ctrl.tolist(),
    }
    empty_hand = {
        "hand_name": None,
        "mano_global_pos": [],
        "mano_global_rot_aa": [],
        "mano_hand_pose": [],
        "mano_joint_pos": [],
        "urdf_dof": [],
        "urdf_dof_target": [],
    }
    objects = []
    for name in parent["names"]:
        address = int(model.joint(name + "_free").qposadr[0])
        objects.append(
            {
                "pos": qpos[:, address : address + 3].tolist(),
                "rot_aa": Rotation.from_quat(
                    qpos[:, address + 3 : address + 7][:, [1, 2, 3, 0]]
                ).as_rotvec().tolist(),
            }
        )
    active = "mayonnaisebottle"
    active_index = parent["names"].index(active)
    teacher_extended = np.concatenate(
        [np.repeat(teacher["qpos"][:1], PREFIX_FRAMES, axis=0), teacher["qpos"]],
        axis=0,
    )
    source_position = np.concatenate(
        [
            np.repeat(inp.arrays["source_object_pos"][:1], PREFIX_FRAMES, axis=0),
            inp.arrays["source_object_pos"],
        ],
        axis=0,
    )
    source_quaternion = np.concatenate(
        [
            np.repeat(
                inp.arrays["source_object_quat_xyzw"][:1], PREFIX_FRAMES, axis=0
            ),
            inp.arrays["source_object_quat_xyzw"],
        ],
        axis=0,
    )
    source_frames = [0] * PREFIX_FRAMES + list(range(parent["frames"]))
    lineage = {
        "contract": CONTRACT,
        "augmentation_contract": ACTION005_VERSION,
        "action": "005",
        "slot": record["slot"],
        "candidate": record["candidate"],
        "setting_sha256": SETTING,
        "parent_row": record["parent_row"],
        "parent_uuid": parent["parent_uuid"],
        "prefix_frames": PREFIX_FRAMES,
        "join": "physical_reference_discrete_c1",
        "source_registry_sha256": file_sha(registry_path),
        "plan_digest": plan["digest"],
    }
    return {
        "index": {
            "uuid": record["uuid"],
            "seed_uuid": parent["parent_uuid"],
            "capMachine": "native-U1",
            "operator": "cheyingtong",
            "scene": ",".join(parent["names"]),
            "is_generated": True,
        },
        "trajectory_metadata": {
            "data_fps": 120,
            "total_frames": frames,
            "gesture": "005-" + parent["semantics"],
            "hand_names": ["right"],
            "hand_slots": ["right", "left"],
            "object_names": parent["names"],
            "mano_hand_shapes": [parent["mano_shape"]],
            "trajectory_info": {"object_move": [movement_record(parent)]},
        },
        "timestamp": (np.arange(frames) / 120).tolist(),
        "hands": [hand, empty_hand],
        "objects": objects,
        "contact": contacts,
        "reference": {
            "source_frame_index": source_frames,
            "hand_urdf_dof": teacher_extended[:, :28].tolist(),
            "object_pos": source_position[:, active_index].tolist(),
            "object_rot_aa": Rotation.from_quat(
                source_quaternion[:, active_index]
            ).as_rotvec().tolist(),
        },
        "command_reference_index": list(range(1, frames)),
        "command_source_frame_index": source_frames[1:],
        "provenance": {
            "contract": CONTRACT,
            "source_contract": SYNTHETIC_LANCE_CONTRACT,
            "force_contract": FORCE_DIRECTION_CONTRACT,
            "reference_fps": 120,
            "control_fps": 120,
            "control_timestep_seconds": 1 / 120,
            "physics_fps": 480,
            "physics_timestep_seconds": 1 / 480,
            "physics_substeps_per_control": 4,
            "policy_mode": "frozen_native_position",
            "checkpoint_metadata_sha256": digest(lineage),
            "warp_ccd_iterations": 16,
            "warp_ccd_contacts_per_world": 256,
            "seed": plan["source_plan_seed"],
            "episode_index": record["slot"],
            "generation_attempt": record["candidate"]["ordinal"] + 1,
            "augmentation_identity": record["uuid"],
        },
        "physical": {key: value.tolist() for key, value in physical.items()},
        "native_contacts": lines,
        "lineage_json": json.dumps(lineage, sort_keys=True),
        "teacher_qpos": teacher_extended.tolist(),
        "reference_objects_json": json.dumps(
            {
                "scene_object_names": inp.arrays["scene_object_names"].tolist(),
                "source_object_pos": source_position.tolist(),
                "source_object_quat_xyzw": source_quaternion.tolist(),
            },
            sort_keys=True,
        ),
    }


def _with_output_schema(batch, output_schema):
    import pyarrow as pa

    require(
        batch.schema.remove_metadata().equals(output_schema.remove_metadata()),
        "source/output field schema mismatch",
    )
    return pa.RecordBatch.from_arrays(batch.columns, schema=output_schema)


def export(args) -> dict:
    import lance
    import pyarrow as pa

    require(args.output.name.startswith("."), "output staging directory must be hidden")
    require(not args.output.exists(), "output already exists")
    source = audit_source640(args.exact640, verify_payload_hashes=True)
    plan, records, audit = collect_action005(
        args.action005_registry, args.action005_plan, args.action005_attempts
    )
    require(
        not (set(source["ordered_uuids"]) & {record["uuid"] for record in records}),
        "UUID collision between exact640 and action005",
    )
    source_dataset = lance.dataset(str(args.exact640 / "compact.lance"))
    old_rows = source_dataset.to_batches(batch_size=1)
    output_schema = schema()

    def batches():
        for old_index, batch in enumerate(old_rows):
            require(batch.num_rows == 1, "exact640 source batch size")
            if old_index == 160:
                for index, record in enumerate(records, start=1):
                    yield pa.RecordBatch.from_pylist(
                        [make_action005_row(args.action005_registry, record, plan)],
                        schema=output_schema,
                    )
                    if index % 20 == 0:
                        print(
                            json.dumps(
                                {"phase": "export_action005", "rows_built": index}
                            ),
                            flush=True,
                        )
            yield _with_output_schema(batch, output_schema)
        require(old_index == 639, "exact640 stream ended early")

    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.action005_registry, args.output / "action005_registry.json")
    shutil.copyfile(args.action005_plan, args.output / "action005_plan.json")
    parent_validation = args.action005_registry.parent / "parent_registry_validation.json"
    require(parent_validation.is_file(), "missing action005 parent validation")
    shutil.copyfile(
        parent_validation, args.output / "action005_parent_validation.json"
    )
    shard_merge = args.action005_attempts / "action005_shard_merge.json"
    require(shard_merge.is_file(), "missing action005 shard merge report")
    shutil.copyfile(shard_merge, args.output / "action005_shard_merge.json")
    for name in ("manifest.json", "validation.json", "publication.json"):
        shutil.copyfile(args.exact640 / name, args.output / ("source640_" + name))
    write_json(args.output / "action005_rejections.json", audit)
    ordered_uuids = (
        source["ordered_uuids"][:160]
        + [record["uuid"] for record in records]
        + source["ordered_uuids"][160:]
    )
    manifest = {
        "contract": CONTRACT,
        "augmentation_contract": AUGMENTATION_CONTRACT,
        "rows": 800,
        "counts": EXPECTED_COUNTS,
        "setting_sha256": SETTING,
        "ordered_uuids": ordered_uuids,
        "source_exact640": source,
        "action005_registry_sha256": file_sha(args.action005_registry),
        "action005_registry_digest": json.loads(args.action005_registry.read_text())["digest"],
        "action005_plan_sha256": file_sha(args.action005_plan),
        "action005_plan_digest": plan["digest"],
        "action005_parent_validation_sha256": file_sha(parent_validation),
        "action005_shard_merge_sha256": file_sha(shard_merge),
    }
    write_json(args.output / "manifest.json", manifest)
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(output_schema, batches()),
        str(args.output / "compact.lance"),
        mode="create",
    )
    write_json(
        args.output / "sha256.json",
        {
            str(path.relative_to(args.output)): file_sha(path)
            for path in sorted(args.output.rglob("*"))
            if path.is_file()
        },
    )
    return {"rows": 800, "output": str(args.output), "counts": EXPECTED_COUNTS}


def validate(args) -> dict:
    import lance
    import pyarrow as pa

    source = audit_source640(args.exact640, verify_payload_hashes=True)
    plan, records, audit = collect_action005(
        args.action005_registry, args.action005_plan, args.action005_attempts
    )
    dataset = lance.dataset(str(args.output / "compact.lance"))
    require(dataset.version == 1 and dataset.count_rows() == 800, "exact800 Lance")
    require(dataset.schema.equals(schema(), check_metadata=True), "exact800 schema")
    compact = dataset.to_table(columns=["index", "trajectory_metadata"]).to_pylist()
    uuids = [row["index"]["uuid"] for row in compact]
    actions = [row["trajectory_metadata"]["gesture"].split("-")[0] for row in compact]
    require(len(set(uuids)) == 800, "exact800 UUID uniqueness")
    require(Counter(actions) == Counter(EXPECTED_COUNTS), "exact800 action quotas")
    require(actions == [action for action in ACTIONS for _ in range(160)], "exact800 order")
    manifest = json.loads((args.output / "manifest.json").read_text())
    require(manifest["ordered_uuids"] == uuids, "exact800 manifest order")
    require(manifest["source_exact640"] == source, "exact640 source binding")
    require(
        json.loads((args.output / "action005_rejections.json").read_text()) == audit,
        "action005 rejection audit",
    )
    require(
        file_sha(args.output / "action005_parent_validation.json")
        == manifest["action005_parent_validation_sha256"],
        "action005 parent validation snapshot",
    )
    require(
        file_sha(args.output / "action005_shard_merge.json")
        == manifest["action005_shard_merge_sha256"],
        "action005 shard merge snapshot",
    )
    source_dataset = lance.dataset(str(args.exact640 / "compact.lance"))
    for output_index in range(800):
        source_index = source640_index_for_output(output_index)
        actual = dataset.take([output_index]).to_pylist()[0]
        if source_index is None:
            record = records[output_index - 160]
            expected = pa.Table.from_pylist(
                [make_action005_row(args.action005_registry, record, plan)],
                schema=schema(),
            ).to_pylist()[0]
        else:
            expected = source_dataset.take([source_index]).to_pylist()[0]
        require(actual == expected, "exact800 row mismatch: " + str(output_index))
        if (output_index + 1) % 20 == 0:
            print(
                json.dumps(
                    {
                        "phase": "readback",
                        "rows": output_index + 1,
                        "action": actions[output_index],
                    }
                ),
                flush=True,
            )
    report = {
        "contract": CONTRACT,
        "validated": True,
        "rows": 800,
        "counts": EXPECTED_COUNTS,
        "unique_uuids": 800,
        "setting_sha256": SETTING,
        "source_exact640_hash_index_sha256": source["hash_index_sha256"],
        "action005_registry_sha256": file_sha(args.action005_registry),
        "action005_plan_digest": plan["digest"],
        "lance_version": dataset.version,
        "checks": [
            "immutable exact640 payload hashes and source publication",
            "ten qualified action005 parents with exact16 children each",
            "two accepted frozen-control action005 replays with distinct PIDs",
            "zero solved native prefix-force contacts",
            "strict physical-reference discrete-C1 target",
            "byte-exact parent control suffix",
            "complete native contact frames and full two-object scene",
            "full Arrow-normalized Lance readback equality",
        ],
    }
    write_json(args.output / "validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "export", "validate"))
    parser.add_argument("--exact640", type=Path, required=True)
    parser.add_argument("--action005-registry", type=Path, required=True)
    parser.add_argument("--action005-plan", type=Path, required=True)
    parser.add_argument("--action005-attempts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute-export", action="store_true")
    args = parser.parse_args()
    args.exact640 = args.exact640.expanduser().resolve(strict=True)
    args.action005_registry = args.action005_registry.expanduser().resolve(strict=True)
    args.action005_plan = args.action005_plan.expanduser().resolve(strict=True)
    args.action005_attempts = args.action005_attempts.expanduser().resolve(strict=True)
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
    if args.mode == "audit":
        source = audit_source640(args.exact640, verify_payload_hashes=True)
        plan, records, audit = collect_action005(
            args.action005_registry, args.action005_plan, args.action005_attempts
        )
        print(
            json.dumps(
                {
                    "source_rows": source["rows"],
                    "action005_rows": len(records),
                    "action005_rejections": len(audit),
                    "plan_digest": plan["digest"],
                }
            )
        )
        return
    require(args.output is not None, "export/validate require output")
    if args.mode == "export":
        require(args.execute_export, "export requires --execute-export")
        result = export(args)
    else:
        result = validate(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
