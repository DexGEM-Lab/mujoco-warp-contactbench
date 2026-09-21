#!/usr/bin/env python3
"""Run the balanced ten-parent, exact160 action005 strict-C1 campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.manorl.start_augmentation import scene_contacts
from sim.manorl.u1_action005 import (
    PARENT_ROW_BY_SLOT,
    VERSION,
    action005_child_uuid,
    action005_trace_gates,
    load_action005_parent,
    validate_parent_assignment,
)
from sim.manorl.u1_campaign import Ledger, digest, file_sha, write_json
from tools.run_u1_largepose_campaign import (
    PREFIX_FRAMES,
    LargePoseRuntime,
    build_plan,
)


class Action005LargePoseRuntime(LargePoseRuntime):
    def __init__(self, registry: Path, parent_row: int):
        (
            self.r,
            self.p,
            self.I,
            self.m,
            self.base,
            self.teacher,
        ) = load_action005_parent(registry, parent_row)
        self.action = "005"
        self.parent_row = parent_row

    def _replay(self, target, q0, prefix_frames, folder):
        report = super()._replay(target, q0, prefix_frames, folder)
        gates, metrics = action005_trace_gates(
            self.m, self.I, Path(folder) / "trace.npz"
        )
        report["gates"].update(gates)
        report["physical"]["action005"] = metrics
        report["accepted"] = all(report["gates"].values())
        report["parent_row"] = self.parent_row
        report["parent_uuid"] = self.p["parent_uuid"]
        report["augmentation_contract"] = VERSION
        write_json(Path(folder) / "result.json", report)
        return report


def campaign_plan(registry: Path) -> dict:
    slots = build_plan()
    assigned = validate_parent_assignment(slots)
    registry_sha = file_sha(registry)
    payload = {
        "version": VERSION,
        "source_plan_version": "u1-four-action-largepose-v3-exact-discrete-c1",
        "source_plan_seed": 20260922,
        "prefix_frames": PREFIX_FRAMES,
        "registry_sha256": registry_sha,
        "parent_row_by_slot": list(PARENT_ROW_BY_SLOT),
        "parent_slot_counts": {str(row): len(values) for row, values in assigned.items()},
        "slots": slots,
    }
    return dict(payload, digest=digest(payload))


def _identity(registry: Path, plan: dict) -> str:
    return digest(
        {
            "version": VERSION,
            "registry_sha256": file_sha(registry),
            "plan_digest": plan["digest"],
            "implementation": {
                path: file_sha(ROOT / path)
                for path in (
                    "sim/manorl/u1_action005.py",
                    "tools/run_u1_action005_largepose_campaign.py",
                    "tools/run_u1_largepose_campaign.py",
                )
            },
        }
    )


def _runtime(cache: dict[int, Action005LargePoseRuntime], registry: Path, row: int):
    if row not in cache:
        cache[row] = Action005LargePoseRuntime(registry, row)
    return cache[row]


def run(args) -> None:
    plan = campaign_plan(args.registry)
    args.staging.mkdir(parents=True, exist_ok=True)
    plan_path = args.staging / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("saved action005 plan differs")
    else:
        write_json(plan_path, plan)
    ledger = Ledger(args.staging / "005.jsonl", _identity(args.registry, plan))
    registry_sha = file_sha(args.registry)
    runtimes: dict[int, Action005LargePoseRuntime] = {}
    for slot in plan["slots"][args.slot_start : args.slot_stop]:
        slot_id = slot["slot"]
        if slot_id in ledger.selected():
            continue
        parent_row = PARENT_ROW_BY_SLOT[slot_id]
        runtime = _runtime(runtimes, args.registry, parent_row)
        parent_uuid = runtime.p["parent_uuid"]
        for candidate in slot["candidates"]:
            uid = action005_child_uuid(
                registry_sha256=registry_sha,
                plan_digest=plan["digest"],
                slot=slot_id,
                candidate=candidate,
                parent_uuid=parent_uuid,
            )
            if any(row.get("uuid") == uid for row in ledger.rows):
                continue
            folder = args.staging / "005" / uid
            ledger.append(
                uuid=uid,
                slot=slot_id,
                parent_row=parent_row,
                parent_uuid=parent_uuid,
                candidate=candidate,
                status="started",
            )
            try:
                delta = np.asarray(candidate["delta"], dtype=np.float64)
                qpos0 = runtime.I.initial["qpos"].copy()
                qpos0[:6] += delta
                contacts = scene_contacts(
                    runtime.m,
                    qpos0,
                    runtime.I.initial["qvel"],
                    runtime.p["names"],
                )
                penetrating = [contact for contact in contacts if contact["distance_m"] < 0]
                inside = bool(
                    np.all(qpos0[:6] >= runtime.m.jnt_range[:6, 0])
                    and np.all(qpos0[:6] <= runtime.m.jnt_range[:6, 1])
                )
                if penetrating or not inside:
                    raise ValueError(
                        "initial geometry preflight failed: "
                        f"penetration={penetrating}, wrist_in_range={inside}"
                    )
                first = runtime.replay(delta, PREFIX_FRAMES, folder / "first")
                if not first["accepted"]:
                    raise ValueError("first-pass gates failed")
                command = [
                    sys.executable,
                    "-m",
                    "tools.run_u1_action005_largepose_campaign",
                    "second-pass",
                    "--registry",
                    str(args.registry),
                    "--staging",
                    str(args.staging),
                    "--uuid",
                    uid,
                    "--parent-row",
                    str(parent_row),
                    "--frozen",
                    str(folder / "first/target.npy"),
                ]
                with (folder / "second-process.log").open("w") as log:
                    subprocess.run(
                        command,
                        check=True,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                second = json.loads((folder / "second/result.json").read_text())
                if not second["accepted"] or second["pid"] == first["pid"]:
                    raise ValueError("independent gates failed")
                artifacts = {
                    str(path.resolve()): file_sha(path)
                    for path in folder.rglob("*")
                    if path.is_file()
                }
                ledger.append(
                    uuid=uid,
                    slot=slot_id,
                    parent_row=parent_row,
                    parent_uuid=parent_uuid,
                    status="selected",
                    artifacts=artifacts,
                )
                break
            except Exception as exc:
                ledger.append(
                    uuid=uid,
                    slot=slot_id,
                    parent_row=parent_row,
                    parent_uuid=parent_uuid,
                    status="rejected",
                    reason=repr(exc),
                )
        if slot_id not in ledger.selected():
            ledger.append(
                slot=slot_id,
                parent_row=parent_row,
                parent_uuid=parent_uuid,
                status="exhausted",
            )
            raise RuntimeError(f"exhausted slot {slot_id}")
        print(
            json.dumps(
                {
                    "slot": slot_id,
                    "parent_row": parent_row,
                    "selected_total": len(ledger.selected()),
                    "quota": 160,
                }
            ),
            flush=True,
        )


def status(args) -> None:
    path = args.staging / "005.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    selected = [row for row in rows if row["status"] == "selected"]
    per_parent = {
        str(parent): sum(row.get("parent_row") == parent for row in selected)
        for parent in range(10)
    }
    print(
        json.dumps(
            {
                "selected": len({row["slot"] for row in selected}),
                "quota": 160,
                "per_parent": per_parent,
                "rejected": sum(row["status"] == "rejected" for row in rows),
                "exhausted": sum(row["status"] == "exhausted" for row in rows),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "second-pass", "status"))
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--slot-start", type=int, default=0)
    parser.add_argument("--slot-stop", type=int, default=160)
    parser.add_argument("--uuid")
    parser.add_argument("--parent-row", type=int)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.registry = args.registry.expanduser().resolve(strict=True)
    args.staging = args.staging.expanduser().resolve()
    if args.mode == "status":
        status(args)
        return
    if args.mode == "second-pass":
        if (
            not args.uuid
            or args.parent_row not in range(10)
            or args.frozen is None
        ):
            raise ValueError("second-pass requires uuid/parent-row/frozen")
        runtime = Action005LargePoseRuntime(args.registry, args.parent_row)
        rows = [
            json.loads(line)
            for line in (args.staging / "005.jsonl").read_text().splitlines()
        ]
        started = next(
            row
            for row in rows
            if row.get("uuid") == args.uuid and row["status"] == "started"
        )
        if started["parent_row"] != args.parent_row:
            raise ValueError("second-pass parent assignment changed")
        delta = np.asarray(started["candidate"]["delta"], dtype=np.float64)
        runtime.replay(
            delta,
            PREFIX_FRAMES,
            args.staging / "005" / args.uuid / "second",
            frozen=args.frozen,
        )
        return
    if not args.execute:
        raise ValueError("production requires --execute")
    if not 0 <= args.slot_start < args.slot_stop <= 160:
        raise ValueError("slot interval must lie in0..160")
    run(args)


if __name__ == "__main__":
    main()
