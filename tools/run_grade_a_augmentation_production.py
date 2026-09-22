#!/usr/bin/env python3
"""Production candidate planning and two-pass replay for current Grade-A parents."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from tools.pilot_atomic_grade_a_augmentation import Observer, hold_extend, prefix_target
from tools.replay_atomic_benchmark_pilot import (
    configure_modules,
    decode_row,
    run_physics,
    sha256,
    validate_gpu_binding,
)

PREFIX_FRAMES = 120
CONTRACT = "manorl.current-grade-a-parent-approach-production.v1"
ACTIONS = ("001", "002", "004", "005", "006")
TARGET_PER_ACTION = 160
RESERVES = 16
COMPACT_DATASET = "/mnt/podshare/all/sjq-pi05-tmptest/lora/datasets/manorl_cheyingtong_unified_compact_20260918/compact.lance"
LARGE_DATASET = "/mnt/podshare/all/sjq-pi05-tmptest/lora/datasets/manorl_cheyingtong_u1_largepose_800.lance"
COMPACT_RESULTS = (
    "/mnt/podshare/all/chensiyuan-pi05Mint/lora/results/compact534_u1_table_grade_replay_v3",
    "/mnt/podshare/all/sjq-pi05-tmptest/lora/results/compact534_u1_table_grade_replay_v3",
)
LARGE_RESULTS = (
    "/mnt/podshare/all/chensiyuan-pi05Mint/lora/results/largepose800_u1_table_grade_replay_v1",
    "/mnt/podshare/all/sjq-pi05-tmptest/lora/results/largepose800_u1_table_grade_replay_v1",
)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    payload = value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()
    return hashlib.sha256(payload).hexdigest()


def build_slots(seed: int = 20260922) -> list[dict]:
    """160 slots: 144 base cells plus16 deterministic extras."""
    slots: list[dict] = []
    for radius in (0.05, 0.10, 0.15):
        for sector in range(8):
            for axis in range(3):
                for sign in (1, -1):
                    sid = len(slots)
                    azimuth = sector * np.pi / 4.0
                    elevation = np.pi / 6.0
                    delta = np.zeros(6, dtype=np.float64)
                    delta[:3] = radius * np.array(
                        [np.cos(elevation) * np.cos(azimuth),
                         np.cos(elevation) * np.sin(azimuth),
                         np.sin(elevation)]
                    )
                    delta[3 + axis] = sign * np.pi / 6.0
                    rng = np.random.default_rng(np.random.SeedSequence([seed, sid]))
                    candidates = []
                    for ordinal in range(RESERVES):
                        candidate = delta.copy()
                        if ordinal:
                            angle = azimuth + rng.uniform(-np.pi / 8, np.pi / 8)
                            elev = np.deg2rad(rng.uniform(15.0, 60.0))
                            candidate[:3] = radius * np.array(
                                [np.cos(elev) * np.cos(angle),
                                 np.cos(elev) * np.sin(angle),
                                 np.sin(elev)]
                            )
                        candidates.append({"ordinal": ordinal, "delta": candidate.tolist()})
                    slots.append({
                        "slot": sid, "kind": "base", "radius_m": radius,
                        "sector": sector, "rotation_axis": axis,
                        "rotation_sign": sign, "candidates": candidates,
                    })
    for extra in range(16):
        sid = len(slots)
        radius = (0.05, 0.10, 0.15)[extra % 3]
        sector = (extra // 2) % 8
        axis = extra % 3
        sign = 1 if (extra // 3) % 2 == 0 else -1
        azimuth = sector * np.pi / 4.0
        rng = np.random.default_rng(np.random.SeedSequence([seed, sid, 987654321]))
        candidates = []
        for ordinal in range(RESERVES):
            angle = azimuth + rng.uniform(-np.pi / 8, np.pi / 8)
            elev = np.deg2rad(rng.uniform(15.0, 60.0))
            delta = np.zeros(6, dtype=np.float64)
            delta[:3] = radius * np.array(
                [np.cos(elev) * np.cos(angle),
                 np.cos(elev) * np.sin(angle),
                 np.sin(elev)]
            )
            delta[3 + axis] = sign * np.pi / 6.0
            candidates.append({"ordinal": ordinal, "delta": delta.tolist()})
        slots.append({
            "slot": sid, "kind": "extra", "radius_m": radius,
            "sector": sector, "rotation_axis": axis,
            "rotation_sign": sign, "candidates": candidates,
        })
    assert len(slots) == 160
    return slots


def grade_rows(roots: tuple[str, ...], action: str) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    overflow = __import__("re").compile(
        r"\b(?:CCD|nefc|nacon|narrowphase|constraint|contact) overflow - please increase\b",
        __import__("re").IGNORECASE,
    )
    for root in roots:
        root_path = Path(root)
        for log in root_path.glob(f"shard*/action{action}.log"):
            if overflow.search(log.read_text(errors="replace")):
                raise ValueError(f"parent result has overflow log: {log}")
        for path in root_path.glob(f"shard*/action{action}/rows/row*.json"):
            record = json.loads(path.read_text())
            if record["grade"] != "A" or record["uuid"] in seen:
                continue
            seen.add(record["uuid"])
            rows.append({
                "row_index": int(record["row_index"]),
                "uuid": record["uuid"],
                "error_m": float(record["max_target_position_error_m"]),
                "record": str(path),
            })
    return sorted(rows, key=lambda x: x["row_index"])


def source_for_action(action: str) -> str:
    return COMPACT_DATASET if action in ("001", "002", "004") else LARGE_DATASET


def roots_for_action(action: str) -> tuple[str, ...]:
    return COMPACT_RESULTS if action in ("001", "002", "004") else LARGE_RESULTS


def prepare(args: argparse.Namespace) -> None:
    import lance

    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {action: source_for_action(action) for action in ACTIONS}
    parent_manifest: dict[str, list[dict]] = {}
    datasets = {path: lance.dataset(path, version=1) for path in set(sources.values())}
    for action in ACTIONS:
        source = sources[action]
        dataset = datasets[source]
        parents = grade_rows(roots_for_action(action), action)
        clean: list[dict] = []
        seen: set[str] = set()
        for parent in parents:
            if parent["uuid"] in seen:
                continue
            row = dataset.take([parent["row_index"]]).to_pylist()[0]
            assert row["index"]["uuid"] == parent["uuid"]
            assert row["trajectory_metadata"]["gesture"][:3] == action
            seen.add(parent["uuid"])
            clean.append({
                **parent,
                "dataset": source,
                "dataset_version": 1,
                "action": action,
                "frames": int(row["trajectory_metadata"]["total_frames"]),
                "object_names": row["trajectory_metadata"]["object_names"],
                "target": row["trajectory_metadata"]["trajectory_info"]["object_move"][0]["object_name"],
            })
        if not clean:
            raise ValueError(f"no Grade-A parents for action {action}")
        parent_manifest[action] = clean

    slots = build_slots()
    candidates: list[dict] = []
    phase0_batches: list[dict] = []
    for action in ACTIONS:
        current = len(parent_manifest[action])
        need = max(0, TARGET_PER_ACTION - current)
        if not need:
            continue
        phase_candidates: list[dict] = []
        parents = parent_manifest[action]
        for slot in slots:
            parent = parents[slot["slot"] % len(parents)]
            for candidate in slot["candidates"]:
                signature = {
                    "contract": CONTRACT,
                    "action": action,
                    "slot": slot["slot"],
                    "candidate": candidate["ordinal"],
                    "parent": parent["uuid"],
                    "delta": candidate["delta"],
                }
                child_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(signature, sort_keys=True)))
                item = {
                    **signature,
                    "uuid": child_uuid,
                    "parent": parent,
                    "radius_m": slot["radius_m"],
                    "sector": slot["sector"],
                    "rotation_axis": slot["rotation_axis"],
                    "rotation_sign": slot["rotation_sign"],
                }
                candidates.append(item)
                if candidate["ordinal"] == 0:
                    phase_candidates.append(item)
        for start in range(0, len(phase_candidates), 5):
            chunk = phase_candidates[start:start + 5]
            phase0_batches.append({
                "batch": len(phase0_batches),
                "action": action,
                "candidate_uuids": [item["uuid"] for item in chunk],
                "real_count": len(chunk),
            })

    manifest = {
        "contract": CONTRACT,
        "dataset_inputs": sources,
        "parent_results": {action: list(roots_for_action(action)) for action in ACTIONS},
        "parents": parent_manifest,
        "target_per_action": TARGET_PER_ACTION,
        "current_A_counts": {action: len(value) for action, value in parent_manifest.items()},
        "deficits": {action: max(0, TARGET_PER_ACTION - len(parent_manifest[action])) for action in ACTIONS},
        "slot_count_per_action": 160,
        "reserve_count": RESERVES,
        "slots": slots,
        "candidates": candidates,
        "phase0_batches": phase0_batches,
    }
    args.output.mkdir(parents=True)
    dump(args.output / "plan.json", manifest)
    print(json.dumps({
        "current_A_counts": manifest["current_A_counts"],
        "deficits": manifest["deficits"],
        "candidate_count": len(candidates),
        "phase0_batches": len(phase0_batches),
        "plan_sha256": file_sha(args.output / "plan.json"),
    }, indent=2))


def load_candidate(dataset, metadata: dict) -> tuple[dict, dict, dict]:
    row = dataset.take([metadata["parent"]["row_index"]]).to_pylist()[0]
    hand = row["hands"][0]
    base = np.asarray(hand["urdf_dof_target"], dtype=np.float32)
    teacher = np.asarray(hand["urdf_dof"], dtype=np.float32)
    target, initial, splice = prefix_target(base, teacher, metadata["delta"])
    item = decode_row(metadata["parent"]["row_index"], row)
    item["uuid"] = metadata["uuid"]
    item["seed_uuid"] = metadata["parent"]["uuid"]
    item["replay_identity"] = f"{item['target']}_{item['action']}_{metadata['uuid']}"
    item["hand_recorded"] = hold_extend(item["hand_recorded"])
    item["hand_recorded"][0] = initial
    item["commands"] = target
    item["object_recorded_pos"] = {key: hold_extend(value) for key, value in item["object_recorded_pos"].items()}
    item["object_recorded_quat_wxyz"] = {key: hold_extend(value) for key, value in item["object_recorded_quat_wxyz"].items()}
    item["frames"] = len(target)
    item["movement"] = {
        **item["movement"],
        "start_frame": item["movement"]["start_frame"] + PREFIX_FRAMES,
        "end_frame": item["movement"]["end_frame"] + PREFIX_FRAMES,
    }
    return row, item, splice


def run(args: argparse.Namespace) -> None:
    import lance

    validate_gpu_binding(args.gpu)
    plan = json.loads(args.plan.read_text())
    batch = plan["phase0_batches"][args.batch]
    candidate_by_uuid = {item["uuid"]: item for item in plan["candidates"]}
    metadata = [candidate_by_uuid[uid] for uid in batch["candidate_uuids"]]
    datasets = {}
    items = []
    for candidate in metadata:
        source = candidate["parent"]["dataset"]
        datasets.setdefault(source, lance.dataset(source, version=1))
        _, item, splice = load_candidate(datasets[source], candidate)
        candidate["splice"] = splice
        items.append(item)
    real_count = len(items)
    if not real_count:
        raise ValueError("empty production batch")
    padded = items + [copy.deepcopy(items[index % real_count]) for index in range(5 - real_count)]
    for index in range(real_count, 5):
        padded[index]["uuid"] = f"padding-{args.batch}-{index}"
        padded[index]["replay_identity"] = f"padding_{args.batch}_{index}"

    args.output.mkdir(parents=True)
    _native, _visual, consumer, assets, contracts = configure_modules(args)
    observer = Observer(padded)
    dump(args.output / "status.json", {
        "state": "running", "batch": args.batch, "pass": args.pass_name,
        "pid": os.getpid(), "real_count": real_count,
    })
    outputs, physics = run_physics(
        dataset_path=Path(metadata[0]["parent"]["dataset"]), dataset_version=1,
        decoded=padded, consumer_visual=consumer, assets=assets, contracts=contracts,
        decorative_scene_spec=args.scene, physics_profile="u1-table", observer=observer,
    )
    reports = []
    for world, candidate in enumerate(metadata):
        folder = args.output / f"world{world}"
        observer.save(folder, world)
        output = outputs[world]
        target = candidate["parent"]["target"]
        error = float(output["metrics"]["max_target_position_error_m"])
        grade = "A" if error < 0.03 else "B" if error < 0.08 else "C"
        angle = float(np.degrees(
            (Rotation.from_quat(output["object_recorded_quat_wxyz"][target][-1][[1, 2, 3, 0]]).inv()
             * Rotation.from_quat(output["object_simulated_quat_wxyz"][target][-1][[1, 2, 3, 0]])).magnitude()
        ))
        report = {
            "uuid": candidate["uuid"], "action": candidate["action"],
            "slot": candidate["slot"], "candidate": candidate["candidate"],
            "parent_uuid": candidate["parent"]["uuid"], "parent_row": candidate["parent"]["row_index"],
            "radius_m": candidate["radius_m"], "max_error_m": error, "grade": grade,
            "prefix_max_force_N": float(observer.prefix_max[world]),
            "contact_frames": int(observer.target_frames[world]),
            "final_rotation_deg": angle,
            "splice": candidate.get("splice"),
            "files": {path.name: file_sha(path) for path in folder.iterdir() if path.is_file()},
        }
        dump(folder / "report.json", report)
        reports.append(report)
    dump(args.output / "manifest.json", {
        "state": "complete", "contract": CONTRACT, "batch": args.batch,
        "pass": args.pass_name, "pid": os.getpid(), "manorl_commit": args.manorl_commit,
        "plan_sha256": file_sha(args.plan), "physics": physics, "reports": reports,
        "real_count": real_count,
    })
    dump(args.output / "status.json", {"state": "complete", "batch": args.batch,
                                        "pass": args.pass_name, "pid": os.getpid()})
    print(json.dumps({"batch": args.batch, "pass": args.pass_name,
                      "grades": dict(Counter(report["grade"] for report in reports))}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--pass-name", choices=("generation", "verification"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manorl-root", type=Path, required=True)
    parser.add_argument("--manorl-commit", required=True)
    parser.add_argument("--client-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        if args.plan is None or args.batch is None or args.pass_name is None or args.gpu is None:
            raise ValueError("run requires plan,batch,pass-name,gpu")
        run(args)


if __name__ == "__main__":
    main()
