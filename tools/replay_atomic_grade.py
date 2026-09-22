#!/usr/bin/env python3
"""Grade compact Cheyintong target-DOF replays without rendering images.

The physics path is exactly the atomic benchmark pilot: saved absolute right-hand
commands at 120 Hz, four 480 Hz MJX-Warp substeps, every object named by the row
physical, and the fixed benchmark scene/table.  Grade is the established State45
quality grade over the full target-object trajectory:

    A: max position error < 0.03 m
    B: 0.03 m <= max position error < 0.08 m
    C: max position error >= 0.08 m

Only compact JSON evidence is persisted.  No image, video, or dynamic trace array
is written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from tools.replay_atomic_benchmark_pilot import (
    CONTRACT as PILOT_CONTRACT,
    configure_modules,
    decode_row,
    resolved_replay_identity,
    run_physics,
    sha256,
    validate_gpu_binding,
)

PLAN_CONTRACT = "manorl.atomic-grade-replay-plan.v1"
RUN_CONTRACT = "manorl.atomic-grade-replay-shard.v1"
GRADE_CONTRACT = "mano_target_replay_max_position_error_grade_v1"
GRADE_A_MAX_M = 0.03
GRADE_B_MAX_M = 0.08
# Multi-object rows increase only allocation demand. These capacities exceed the
# observed action001 requirements (contact5648 total / CCD1874 total /
# constraints5302 for 5 worlds) without changing equations, model parameters,
# controls, or grade thresholds.
GRADE_CONTACT_CAPACITY_PER_WORLD = 2048
GRADE_CONSTRAINT_CAPACITY = 8192
GRADE_CCD_CONTACTS_PER_WORLD = 512
U1_CONTACT_CAPACITY_PER_WORLD = 1024
U1_CONSTRAINT_CAPACITY = 4096
U1_CCD_CONTACTS_PER_WORLD = 256


CAPACITY_PROFILES = ("expanded", "u1")


def capacities_for_profile(profile: str) -> tuple[int, int, int]:
    if profile == "u1":
        return (
            U1_CONTACT_CAPACITY_PER_WORLD,
            U1_CONSTRAINT_CAPACITY,
            U1_CCD_CONTACTS_PER_WORLD,
        )
    if profile == "expanded":
        return (
            GRADE_CONTACT_CAPACITY_PER_WORLD,
            GRADE_CONSTRAINT_CAPACITY,
            GRADE_CCD_CONTACTS_PER_WORLD,
        )
    raise ValueError(f"unsupported physics profile {profile!r}")


def dump_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def grade_from_max_error(value: float) -> str:
    if not isinstance(value, (int, float)) or not float("-inf") < float(value) < float("inf"):
        raise ValueError("max position error must be finite")
    if value < GRADE_A_MAX_M:
        return "A"
    if value < GRADE_B_MAX_M:
        return "B"
    return "C"


def parse_rows(text: str, count: int) -> list[int]:
    if not text.strip():
        return list(range(count))
    rows: list[int] = []
    for raw in text.split(","):
        part = raw.strip()
        fields = part.split(":")
        if len(fields) == 1:
            rows.append(int(fields[0]))
        elif len(fields) == 2 and int(fields[1]) > int(fields[0]):
            rows.extend(range(int(fields[0]), int(fields[1])))
        else:
            raise ValueError(f"invalid end-exclusive row range: {part!r}")
    rows = sorted(set(rows))
    invalid = [row for row in rows if row < 0 or row >= count]
    if invalid:
        raise ValueError(f"rows outside [0,{count}): {invalid[:20]}")
    return rows


def _metadata_rows(dataset: Any, selected: set[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    all_indices = sorted(selected)
    for start in range(0, len(all_indices), 128):
        indices = all_indices[start : start + 128]
        payload = dataset.take(
            indices, columns=["index", "trajectory_metadata", "provenance"]
        ).to_pylist()
        for row_index, row in zip(indices, payload, strict=True):
            metadata = row["trajectory_metadata"]
            movement = metadata["trajectory_info"]["object_move"]
            if len(movement) != 1:
                raise ValueError(f"row {row_index} does not have one active object")
            action = str(metadata["gesture"])[:3]
            if not action.isdigit():
                raise ValueError(f"row {row_index} has malformed action prefix {action!r}")
            names = list(metadata["object_names"])
            target = str(movement[0]["object_name"])
            if target not in names:
                raise ValueError(f"row {row_index} target {target!r} is absent")
            uuid = str(row["index"]["uuid"])
            records.append(
                {
                    "row_index": row_index,
                    "uuid": uuid,
                    "action": action,
                    "frames": int(metadata["total_frames"]),
                    "object_names": names,
                    "target": target,
                    "source_identity": resolved_replay_identity(
                        row["provenance"].get("source_identity"),
                        target=target,
                        action=action,
                        uuid=uuid,
                    ),
                }
            )
    if [record["row_index"] for record in records] != all_indices:
        raise RuntimeError("metadata scan changed selected row ordering")
    uuids = [record["uuid"] for record in records]
    if len(set(uuids)) != len(uuids):
        raise ValueError("selected rows contain duplicate UUIDs")
    return records


def make_balanced_plan(
    records: Sequence[Mapping[str, Any]], *, dataset_rows: int, dataset_version: int, shard_count: int
) -> dict[str, Any]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shards = [
        {"shard_id": index, "frame_count": 0, "rows": []}
        for index in range(shard_count)
    ]
    # Longest-processing-time scheduling makes six shards comparable in actual
    # physics work even though trajectory lengths differ by more than 5x.
    for record in sorted(records, key=lambda value: (-int(value["frames"]), int(value["row_index"]))):
        shard = min(
            shards,
            key=lambda value: (
                int(value["frame_count"]), len(value["rows"]), int(value["shard_id"])
            ),
        )
        shard["rows"].append(dict(record))
        shard["frame_count"] += int(record["frames"])
    for shard in shards:
        shard["rows"].sort(key=lambda value: int(value["row_index"]))
        shard["row_count"] = len(shard["rows"])
        shard["row_indices_sha256"] = canonical_sha256(
            [value["row_index"] for value in shard["rows"]]
        )
    selected = sorted(int(value["row_index"]) for value in records)
    plan: dict[str, Any] = {
        "contract": PLAN_CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(dataset_rows),
        "dataset_version": int(dataset_version),
        "selected_row_count": len(records),
        "selected_row_indices_sha256": canonical_sha256(selected),
        "selected_uuid_sha256": canonical_sha256(
            [value["uuid"] for value in sorted(records, key=lambda value: int(value["row_index"]))]
        ),
        "shard_count": shard_count,
        "grade": {
            "contract": GRADE_CONTRACT,
            "quantity": "full_trajectory_max_target_position_error_m",
            "A": "error < 0.03",
            "B": "0.03 <= error < 0.08",
            "C": "error >= 0.08",
        },
        "physics": {
            "source_contract": PILOT_CONTRACT,
            "right_hand_only": True,
            "control_hz": 120,
            "physics_hz": 480,
            "substeps": 4,
            "rendering": False,
        },
        "shards": shards,
    }
    plan["plan_payload_sha256"] = canonical_sha256(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("contract") != PLAN_CONTRACT:
        raise ValueError("unknown replay plan contract")
    payload = dict(plan)
    expected_payload_sha = payload.pop("plan_payload_sha256", None)
    if canonical_sha256(payload) != expected_payload_sha:
        raise ValueError("plan payload SHA mismatch")
    shards = plan.get("shards")
    if not isinstance(shards, list) or len(shards) != int(plan.get("shard_count", -1)):
        raise ValueError("plan shard count mismatch")
    for shard in shards:
        shard_rows = shard.get("rows", [])
        shard_indices = [int(row["row_index"]) for row in shard_rows]
        if int(shard.get("row_count", -1)) != len(shard_rows):
            raise ValueError("plan shard row count mismatch")
        if int(shard.get("frame_count", -1)) != sum(
            int(row["frames"]) for row in shard_rows
        ):
            raise ValueError("plan shard frame count mismatch")
        if canonical_sha256(shard_indices) != shard.get("row_indices_sha256"):
            raise ValueError("plan shard row identity mismatch")
    rows = [row for shard in shards for row in shard.get("rows", [])]
    indices = [int(row["row_index"]) for row in rows]
    uuids = [str(row["uuid"]) for row in rows]
    if len(indices) != int(plan.get("selected_row_count", -1)):
        raise ValueError("plan selected row count mismatch")
    if len(set(indices)) != len(indices) or len(set(uuids)) != len(uuids):
        raise ValueError("plan rows or UUIDs are not unique")
    if canonical_sha256(sorted(indices)) != plan.get("selected_row_indices_sha256"):
        raise ValueError("plan row-index identity mismatch")
    ordered_uuid = [
        row["uuid"] for row in sorted(rows, key=lambda value: int(value["row_index"]))
    ]
    if canonical_sha256(ordered_uuid) != plan.get("selected_uuid_sha256"):
        raise ValueError("plan UUID identity mismatch")


def make_plan(args: argparse.Namespace) -> None:
    import lance

    dataset = lance.dataset(str(args.dataset), version=args.dataset_version)
    count = dataset.count_rows()
    selected = set(parse_rows(args.rows, count))
    records = _metadata_rows(dataset, selected)
    plan = make_balanced_plan(
        records,
        dataset_rows=count,
        dataset_version=args.dataset_version,
        shard_count=args.shard_count,
    )
    if args.plan.exists():
        raise FileExistsError(f"plan output already exists: {args.plan}")
    dump_atomic(args.plan, plan)
    print(
        json.dumps(
            {
                "plan": str(args.plan),
                "rows": len(records),
                "frames": sum(int(record["frames"]) for record in records),
                "shards": [
                    {"id": shard["shard_id"], "rows": shard["row_count"], "frames": shard["frame_count"]}
                    for shard in plan["shards"]
                ],
            },
            indent=2,
        )
    )


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _plan_rows(plan: Mapping[str, Any], shard_id: int) -> list[dict[str, Any]]:
    if shard_id < 0 or shard_id >= int(plan["shard_count"]):
        raise ValueError(f"shard_id {shard_id} outside plan")
    shard = plan["shards"][shard_id]
    if int(shard["shard_id"]) != shard_id:
        raise ValueError("plan shard ordering differs from IDs")
    return [dict(row) for row in shard["rows"]]


def _group_batches(rows: Sequence[Mapping[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["action"], tuple(row["object_names"]), row["target"])
        groups[key].append(dict(row))
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda value: int(value["row_index"]))
        for start in range(0, len(group), batch_size):
            yield group[start : start + batch_size]


def _take_payload(dataset: Any, indices: Sequence[int], mode: str) -> list[dict[str, Any]]:
    if mode == "batch":
        return dataset.take(list(indices)).to_pylist()
    if mode == "individual":
        return [dataset.take([row_index]).to_pylist()[0] for row_index in indices]
    raise ValueError(f"unsupported Lance read mode {mode!r}")


def _pad_batch(
    rows: Sequence[Mapping[str, Any]], batch_size: int
) -> tuple[list[dict[str, Any]], int]:
    """Pad one homogeneous tail batch to a fixed MJX-Warp static shape."""

    if not rows or len(rows) > batch_size:
        raise ValueError("one non-empty batch no larger than batch_size is required")
    result = [dict(row) for row in rows]
    real_count = len(result)
    source = list(result)
    while len(result) < batch_size:
        result.append(dict(source[(len(result) - real_count) % real_count]))
    return result, real_count


def _row_record_valid(path: Path, expected: Mapping[str, Any], run_identity: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
        return (
            value.get("status") == "ok"
            and int(value.get("row_index", -1)) == int(expected["row_index"])
            and value.get("uuid") == expected["uuid"]
            and value.get("run_identity") == run_identity
            and value.get("grade") in {"A", "B", "C"}
        )
    except Exception:
        return False


def run_shard(args: argparse.Namespace) -> None:
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"output exists without --resume: {args.output}")
        if not (args.output / "run_manifest.json").is_file():
            raise ValueError("resume output has no run manifest")
        existing = json.loads((args.output / "run_manifest.json").read_text())
    else:
        existing = None
    gpu_binding = validate_gpu_binding(args.gpu)
    import lance

    dataset = lance.dataset(str(args.dataset), version=args.dataset_version)
    if dataset.count_rows() != int(plan["dataset_rows"]):
        raise ValueError("dataset row count differs from plan")
    if int(args.dataset_version) != int(plan["dataset_version"]):
        raise ValueError("dataset version differs from plan")
    rows = _plan_rows(plan, args.shard_id)
    if args.action:
        rows = [row for row in rows if row["action"] == args.action]
        if not rows:
            raise ValueError(
                f"shard {args.shard_id} has no rows for action {args.action}"
            )
    if existing is not None:
        expected = {
            "contract": "one_worker_one_matching_compute_egl_gpu_v1",
            "physical_gpu": int(args.gpu),
            "cuda_visible_devices": str(args.gpu),
            "mujoco_egl_device_id": str(args.gpu),
        }
        existing_binding = existing.get("gpu_binding") or {}
        if (
            existing.get("contract") != RUN_CONTRACT
            or any(existing_binding.get(key) != value for key, value in expected.items())
        ):
            raise ValueError("resume run identity differs before GPU binding")
    _native, _visual, consumer_visual, assets, contracts = configure_modules(args)
    contact_capacity, constraint_capacity, ccd_capacity = capacities_for_profile(
        args.capacity_profile
    )
    run_identity = {
        "contract": RUN_CONTRACT,
        "plan_sha256": sha256(args.plan),
        "plan_payload_sha256": plan["plan_payload_sha256"],
        "dataset_version": args.dataset_version,
        "dataset_rows": dataset.count_rows(),
        "asset_manifest_sha256": sha256(args.asset_manifest),
        "asset_commit": assets.asset_provenance()["asset_source_commit"],
        "manorl_commit": args.manorl_commit,
        "client_commit": _git_head(args.client_root),
        "scene_sha256": sha256(args.scene),
        "grade_contract": GRADE_CONTRACT,
        "contact_capacity_per_world": contact_capacity,
        "constraint_capacity": constraint_capacity,
        "ccd_contacts_per_world": ccd_capacity,
        "batch_size": args.batch_size,
        "shard_id": args.shard_id,
        "action": args.action or None,
        "physics_profile": args.physics_profile,
        "capacity_profile": args.capacity_profile,
        "gpu_binding": gpu_binding,
        "rendering": False,
        "persisted_payload": "per-row JSON metrics only; no images, videos, or trace arrays",
    }
    if existing is not None:
        if existing != run_identity:
            raise ValueError("resume run identity differs")
    else:
        args.output.mkdir(parents=True)
        dump_atomic(args.output / "run_manifest.json", run_identity)
    row_dir = args.output / "rows"
    row_dir.mkdir(exist_ok=True)
    pending = [
        row
        for row in rows
        if not _row_record_valid(
            row_dir / f"row{int(row['row_index']):04d}.json", row, run_identity
        )
    ]
    completed_before = len(rows) - len(pending)
    status = {
        "contract": RUN_CONTRACT,
        "state": "running",
        "shard_id": args.shard_id,
        "action": args.action or None,
        "row_count": len(rows),
        "pending": len(pending),
        "completed_before_resume": completed_before,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    dump_atomic(args.output / "status.json", status)
    processed = completed_before
    try:
        for batch_number, real_batch in enumerate(
            _group_batches(pending, args.batch_size), 1
        ):
            batch, real_count = _pad_batch(real_batch, args.batch_size)
            indices = [int(value["row_index"]) for value in batch]
            payload = _take_payload(dataset, indices, args.lance_read_mode)
            decoded = [
                decode_row(row_index, row)
                for row_index, row in zip(indices, payload, strict=True)
            ]
            for expected, actual in zip(batch, decoded, strict=True):
                if (
                    actual["uuid"] != expected["uuid"]
                    or actual["frames"] != int(expected["frames"])
                    or list(actual["names"]) != list(expected["object_names"])
                    or actual["target"] != expected["target"]
                ):
                    raise ValueError(f"row {expected['row_index']} differs from replay plan")
            outputs, physics = run_physics(
                dataset_path=args.dataset,
                dataset_version=args.dataset_version,
                decoded=decoded,
                assets=assets,
                contracts=contracts,
                consumer_visual=consumer_visual,
                decorative_scene_spec=args.scene,
                contact_capacity_per_world=contact_capacity,
                constraint_capacity=constraint_capacity,
                ccd_contacts_per_world=ccd_capacity,
                physics_profile=args.physics_profile,
            )
            for expected, output in zip(
                real_batch, outputs[:real_count], strict=True
            ):
                maximum = float(output["metrics"]["max_target_position_error_m"])
                record = {
                    "status": "ok",
                    "row_index": int(expected["row_index"]),
                    "uuid": expected["uuid"],
                    "action": expected["action"],
                    "frames": int(expected["frames"]),
                    "target": expected["target"],
                    "object_names": list(expected["object_names"]),
                    "grade": grade_from_max_error(maximum),
                    "grade_contract": GRADE_CONTRACT,
                    "max_target_position_error_m": maximum,
                    "final_target_position_error_m": float(
                        output["metrics"]["final_target_position_error_m"]
                    ),
                    "max_hand_qpos_error": float(output["metrics"]["max_hand_qpos_error"]),
                    "objects": output["metrics"]["objects"],
                    "run_identity": run_identity,
                }
                dump_atomic(
                    row_dir / f"row{int(expected['row_index']):04d}.json", record
                )
            processed += real_count
            print(
                json.dumps(
                    {
                        "shard": args.shard_id,
                        "action": args.action or None,
                        "batch": batch_number,
                        "processed": processed,
                        "total": len(rows),
                        "rows": [int(value["row_index"]) for value in real_batch],
                        "padding_worlds": len(batch) - real_count,
                        "batch_grade_counts": dict(
                            Counter(
                                grade_from_max_error(
                                    float(output["metrics"]["max_target_position_error_m"])
                                )
                                for output in outputs[:real_count]
                            )
                        ),
                        "physics": physics["runtime"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del outputs, decoded, payload
            gc.collect()
    except Exception as error:
        status.update(
            state="failed",
            failed_at=datetime.now(timezone.utc).isoformat(),
            processed=processed,
            error=f"{type(error).__name__}: {error}",
        )
        dump_atomic(args.output / "status.json", status)
        raise
    records = [
        json.loads((row_dir / f"row{int(row['row_index']):04d}.json").read_text())
        for row in rows
    ]
    indices = [int(record["row_index"]) for record in records]
    if indices != [int(row["row_index"]) for row in rows] or len(set(indices)) != len(rows):
        raise RuntimeError("finished shard result population differs from plan")
    counts = Counter(record["grade"] for record in records)
    summary = {
        "contract": RUN_CONTRACT,
        "state": "complete",
        "shard_id": args.shard_id,
        "action": args.action or None,
        "row_count": len(rows),
        "frame_count": sum(int(row["frames"]) for row in rows),
        "grade_counts": {grade: int(counts.get(grade, 0)) for grade in "ABC"},
        "row_indices_sha256": canonical_sha256(indices),
        "run_identity": run_identity,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    dump_atomic(args.output / "summary.json", summary)
    dump_atomic(args.output / "status.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    roots = [Path(value) for value in args.shard_outputs.split(",") if value]
    if len(roots) != int(plan["shard_count"]):
        raise ValueError("one shard output root is required per plan shard")
    overflow = re.compile(
        r"\b(?:CCD|nefc|nacon|narrowphase|constraint|contact) "
        r"overflow - please increase\b",
        re.IGNORECASE,
    )
    forbidden_suffixes = {".png", ".jpg", ".jpeg", ".mp4", ".npy", ".npz"}
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    provenance: dict[str, set[Any]] = defaultdict(set)
    for shard_id, root in enumerate(roots):
        if not root.is_dir():
            raise FileNotFoundError(f"shard root is absent: {root}")
        forbidden = [
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in forbidden_suffixes
        ]
        if forbidden:
            raise ValueError(f"shard {shard_id} contains forbidden image/trace artifacts")
        expected_shard = plan["shards"][shard_id]
        actions = sorted({str(row["action"]) for row in expected_shard["rows"]})
        for action in actions:
            expected_action = [
                row for row in expected_shard["rows"] if row["action"] == action
            ]
            action_root = root / f"action{action}"
            summary = json.loads((action_root / "summary.json").read_text())
            if (
                summary.get("state") != "complete"
                or int(summary["shard_id"]) != shard_id
                or summary.get("action") != action
                or int(summary["row_count"]) != len(expected_action)
            ):
                raise ValueError(f"shard {shard_id} action {action} is not complete")
            expected_indices = [int(row["row_index"]) for row in expected_action]
            if summary.get("row_indices_sha256") != canonical_sha256(expected_indices):
                raise ValueError(f"shard {shard_id} action {action} row identity differs")
            log = root / f"action{action}.log"
            if not log.is_file():
                raise FileNotFoundError(f"missing action log: {log}")
            if overflow.search(log.read_text(errors="replace")):
                raise ValueError(f"capacity overflow in shard {shard_id} action {action}")
            identity = summary.get("run_identity") or {}
            expected_capacities = capacities_for_profile(
                identity.get("capacity_profile", "expanded")
            )
            if (
                identity.get("grade_contract") != GRADE_CONTRACT
                or identity.get("contact_capacity_per_world")
                != expected_capacities[0]
                or identity.get("constraint_capacity") != expected_capacities[1]
                or identity.get("ccd_contacts_per_world") != expected_capacities[2]
                or identity.get("batch_size") != 5
                or identity.get("rendering") is not False
                or identity.get("physics_profile") not in {"atomic-benchmark", "u1-table"}
                or identity.get("capacity_profile") not in CAPACITY_PROFILES
            ):
                raise ValueError(f"shard {shard_id} action {action} runtime differs")
            for key in (
                "asset_commit",
                "asset_manifest_sha256",
                "client_commit",
                "manorl_commit",
                "scene_sha256",
                "physics_profile",
                "capacity_profile",
            ):
                provenance[key].add(identity.get(key))
            action_rows = [
                json.loads(path.read_text())
                for path in sorted((action_root / "rows").glob("row*.json"))
            ]
            if [int(row["row_index"]) for row in action_rows] != expected_indices:
                raise ValueError(f"shard {shard_id} action {action} records differ")
            if any(row.get("status") != "ok" for row in action_rows):
                raise ValueError(f"shard {shard_id} action {action} has non-ok rows")
            rows.extend(action_rows)
            summaries.append(summary)
    if any(len(values) != 1 or None in values for values in provenance.values()):
        raise ValueError(f"cross-shard provenance differs: {dict(provenance)}")
    rows.sort(key=lambda value: int(value["row_index"]))
    expected = sorted(
        [row for shard in plan["shards"] for row in shard["rows"]],
        key=lambda value: int(value["row_index"]),
    )
    if [row["row_index"] for row in rows] != [row["row_index"] for row in expected]:
        raise ValueError("aggregate row population differs from plan")
    if [row["uuid"] for row in rows] != [row["uuid"] for row in expected]:
        raise ValueError("aggregate UUID population differs from plan")
    overall = Counter(row["grade"] for row in rows)
    by_action: dict[str, Counter[str]] = defaultdict(Counter)
    error_by_action: dict[str, list[float]] = defaultdict(list)
    boundary_rows = []
    for row in rows:
        error = float(row["max_target_position_error_m"])
        if not float("-inf") < error < float("inf"):
            raise ValueError(f"row {row['row_index']} has nonfinite error")
        expected_grade = grade_from_max_error(error)
        if row["grade"] != expected_grade:
            raise ValueError(f"row {row['row_index']} grade does not match error")
        by_action[row["action"]][row["grade"]] += 1
        error_by_action[row["action"]].append(error)
        margin = min(abs(error - GRADE_A_MAX_M), abs(error - GRADE_B_MAX_M))
        if margin <= 0.001:
            boundary_rows.append(
                {
                    "row_index": int(row["row_index"]),
                    "uuid": row["uuid"],
                    "action": row["action"],
                    "grade": row["grade"],
                    "max_target_position_error_m": error,
                    "threshold_margin_m": margin,
                }
            )
    import numpy as np

    result = {
        "contract": "manorl.atomic-grade-replay-result.v1",
        "plan_sha256": sha256(args.plan),
        "rows": len(rows),
        "grade_contract": GRADE_CONTRACT,
        "grade_counts": {grade: int(overall.get(grade, 0)) for grade in "ABC"},
        "grade_rates": {
            grade: float(overall.get(grade, 0) / len(rows)) for grade in "ABC"
        },
        "by_action": {
            action: {
                "rows": sum(counts.values()),
                "grade_counts": {grade: int(counts.get(grade, 0)) for grade in "ABC"},
                "grade_rates": {
                    grade: float(counts.get(grade, 0) / sum(counts.values()))
                    for grade in "ABC"
                },
                "max_error_m": {
                    "median": float(np.median(error_by_action[action])),
                    "p95": float(np.quantile(error_by_action[action], 0.95)),
                    "maximum": float(np.max(error_by_action[action])),
                },
            }
            for action, counts in sorted(by_action.items())
        },
        "boundary_rows_within_1mm": boundary_rows,
        "runtime_provenance": {
            key: next(iter(values)) for key, values in sorted(provenance.items())
        },
        "action_shards": summaries,
        "rendered_images": 0,
        "rendered_videos": 0,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.output.exists():
        raise FileExistsError(f"aggregate output exists: {args.output}")
    dump_atomic(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-plan", action="store_true")
    mode.add_argument("--run-shard", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    value.add_argument("--dataset", type=Path)
    value.add_argument("--dataset-version", type=int, default=2)
    value.add_argument("--rows", default="")
    value.add_argument("--shard-count", type=int, default=6)
    value.add_argument("--plan", type=Path, required=True)
    value.add_argument("--shard-id", type=int)
    value.add_argument("--action", default="")
    value.add_argument("--batch-size", type=int, default=5)
    value.add_argument("--output", type=Path)
    value.add_argument("--resume", action="store_true")
    value.add_argument(
        "--lance-read-mode", choices=("batch", "individual"), default="batch"
    )
    value.add_argument("--shard-outputs", default="")
    value.add_argument("--asset-manifest", type=Path)
    value.add_argument("--asset-root", type=Path)
    value.add_argument("--manorl-root", type=Path)
    value.add_argument("--manorl-commit")
    value.add_argument("--client-root", type=Path)
    value.add_argument("--benchmark-root", type=Path)
    value.add_argument("--scene", type=Path)
    value.add_argument("--gpu", type=int)
    value.add_argument(
        "--physics-profile",
        choices=("atomic-benchmark", "u1-table"),
        default="atomic-benchmark",
    )
    value.add_argument(
        "--capacity-profile", choices=CAPACITY_PROFILES, default="expanded"
    )
    return value


def main() -> None:
    args = parser().parse_args()
    if args.make_plan:
        if args.dataset is None:
            raise ValueError("--dataset is required for --make-plan")
        make_plan(args)
        return
    if args.aggregate:
        if args.output is None:
            raise ValueError("--output is required for --aggregate")
        aggregate(args)
        return
    required = (
        "dataset",
        "output",
        "shard_id",
        "asset_manifest",
        "asset_root",
        "manorl_root",
        "manorl_commit",
        "client_root",
        "benchmark_root",
        "scene",
        "gpu",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing worker arguments: {missing}")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    run_shard(args)


if __name__ == "__main__":
    main()
