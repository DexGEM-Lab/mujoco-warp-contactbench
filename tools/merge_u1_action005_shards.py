#!/usr/bin/env python3
"""Validate and atomically merge disjoint action005 shard ledgers/artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

from sim.manorl.u1_action005 import PARENT_ROW_BY_SLOT
from sim.manorl.u1_campaign import file_sha, write_json
from tools.run_u1_action005_largepose_campaign import campaign_identity, campaign_plan


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_ledger(path: Path) -> list[dict]:
    require(path.is_file(), "missing ledger: " + str(path))
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_rows(
    rows: list[dict],
    identity: str,
    root: Path,
    alternate_root: Path | None = None,
) -> set[int]:
    require(rows, "empty ledger: " + str(root))
    require(all(row["identity"] == identity for row in rows), "ledger identity")
    require(not any(row["status"] == "exhausted" for row in rows), "exhausted slot")
    latest = {row["uuid"]: row for row in rows if "uuid" in row}
    require(
        all(row["status"] in ("selected", "rejected") for row in latest.values()),
        "unfinished shard candidate",
    )
    selected = [row for row in rows if row["status"] == "selected"]
    slots = {row["slot"] for row in selected}
    require(len(slots) == len(selected), "duplicate selected shard slot")
    for row in selected:
        folder = root / "005" / row["uuid"]
        storage_root = root
        if not folder.exists() and alternate_root is not None:
            folder = alternate_root / "005" / row["uuid"]
            storage_root = alternate_root
        expected_relative = {
            str(Path(path).resolve().relative_to(root))
            for path in row["artifacts"]
        }
        actual_relative = {
            str(path.resolve().relative_to(storage_root))
            for path in folder.rglob("*")
            if path.is_file()
        }
        require(actual_relative == expected_relative, "selected artifact coverage")
        for path, expected in row["artifacts"].items():
            relative = Path(path).resolve().relative_to(root)
            actual_path = storage_root / relative
            require(
                file_sha(actual_path) == expected,
                "selected artifact changed: " + str(actual_path),
            )
    return slots


def remap_row(row: dict, source: Path, destination: Path) -> dict:
    remapped = dict(row)
    if "artifacts" in row:
        remapped["artifacts"] = {
            str((destination / Path(path).resolve().relative_to(source)).resolve()): sha
            for path, sha in row["artifacts"].items()
        }
    return remapped


def move_tree_children(source: Path, destination: Path) -> int:
    if not source.exists():
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    moved = 0
    for child in sorted(source.iterdir()):
        target = destination / child.name
        if target.exists():
            source_files = {
                str(path.relative_to(child)): file_sha(path)
                for path in child.rglob("*")
                if path.is_file()
            }
            target_files = {
                str(path.relative_to(target)): file_sha(path)
                for path in target.rglob("*")
                if path.is_file()
            }
            require(source_files == target_files, "conflicting resumed artifact tree")
            shutil.rmtree(child)
        else:
            require(
                child.stat().st_dev == destination.stat().st_dev,
                "shard merge must use one filesystem",
            )
            os.rename(child, target)
        moved += 1
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(args.execute, "merge requires --execute")
    canonical = args.canonical.resolve(strict=True)
    registry = args.registry.resolve(strict=True)
    shards = [path.resolve(strict=True) for path in args.shard]
    require(len(set(shards)) == len(shards), "duplicate shard root")
    plan = campaign_plan(registry)
    require(json.loads((canonical / "plan.json").read_text()) == plan, "canonical plan")
    identity = campaign_identity(registry, plan)
    canonical_rows = read_ledger(canonical / "005.jsonl")
    canonical_slots = validate_rows(canonical_rows, identity, canonical)
    all_rows = list(canonical_rows)
    selected_slots = set(canonical_slots)
    merge_sources = []
    for shard in shards:
        require(json.loads((shard / "plan.json").read_text()) == plan, "shard plan")
        ledger_path = shard / "005.jsonl"
        rows = read_ledger(ledger_path)
        slots = validate_rows(rows, identity, shard, canonical)
        require(not (slots & selected_slots), "overlapping selected shard slots")
        selected_slots.update(slots)
        merge_sources.append(
            {
                "root": str(shard),
                "ledger_sha256": file_sha(ledger_path),
                "selected_slots": sorted(slots),
                "rows": len(rows),
                "run_log_sha256": file_sha(shard / "run.log")
                if (shard / "run.log").is_file()
                else None,
            }
        )
        all_rows.extend(remap_row(row, shard, canonical) for row in rows)
    require(selected_slots == set(range(160)), "merged selection is not exact160")
    selected = [row for row in all_rows if row["status"] == "selected"]
    require(len(selected) == 160, "merged selected event count")
    require(
        Counter(row["parent_row"] for row in selected)
        == Counter({row: 16 for row in range(10)}),
        "merged per-parent quotas",
    )
    require(
        all(row["parent_row"] == PARENT_ROW_BY_SLOT[row["slot"]] for row in selected),
        "merged parent assignment",
    )
    moved = 0
    for shard in shards:
        moved += move_tree_children(shard / "005", canonical / "005")
    for row in selected:
        folder = canonical / "005" / row["uuid"]
        actual = {str(path.resolve()) for path in folder.rglob("*") if path.is_file()}
        require(actual == set(row["artifacts"]), "remapped artifact coverage")
        for path, expected in row["artifacts"].items():
            require(file_sha(path) == expected, "remapped artifact changed: " + path)
    ledger = canonical / "005.jsonl"
    temporary = ledger.with_suffix(".jsonl.merge.tmp")
    with temporary.open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, ledger)
    report = {
        "contract": "u1-action005-shard-merge-v1",
        "identity": identity,
        "plan_digest": plan["digest"],
        "canonical_premerge_slots": sorted(canonical_slots),
        "sources": merge_sources,
        "selected": 160,
        "per_parent": {str(row): 16 for row in range(10)},
        "artifact_directories_moved": moved,
        "canonical_ledger_sha256": file_sha(ledger),
    }
    write_json(canonical / "action005_shard_merge.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
