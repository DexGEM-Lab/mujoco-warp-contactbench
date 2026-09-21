#!/usr/bin/env python3
"""Merge large-pose shards into one canonical ledger with safe resumption.

Shard ledgers already bind every artifact to a SHA-256. Artifacts are moved by
same-filesystem rename and their hashes are carried to the canonical paths; the
final campaign audit rereads every selected artifact once. This avoids hashing
large traces twice over CIFS while preserving fail-closed recovery semantics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def latest(rows: list[dict]) -> dict[str, dict]:
    states: dict[str, dict] = {}
    for row in rows:
        if "uuid" in row:
            states[row["uuid"]] = row
    return states


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()


def remap_artifacts(artifacts: dict[str, str], source: Path, destination: Path) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for raw_path, sha256 in artifacts.items():
        try:
            relative = Path(raw_path).relative_to(source)
        except ValueError as exc:
            raise RuntimeError(f"artifact outside shard directory: {raw_path}") from exc
        mapped[str(destination / relative)] = sha256
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    args = parser.parse_args()

    main_path = args.attempts / f"{args.action}.jsonl"
    main_rows = read_jsonl(main_path)
    identity = main_rows[0]["identity"]
    states = latest(main_rows)

    # A canonical attempt interrupted before sharding cannot later be accepted as
    # the same attempt. Close it explicitly before importing shard attempts.
    for uuid, row in list(states.items()):
        if row["status"] == "started":
            closed = {
                "uuid": uuid,
                "slot": row["slot"],
                "status": "rejected",
                "reason": "interrupted; no lucky retry",
                "identity": identity,
            }
            append_jsonl(main_path, closed)
            states[uuid] = closed

    selected = {
        row["slot"]: row
        for row in read_jsonl(main_path)
        if row["status"] == "selected"
    }
    merged_slots: list[int] = []

    for shard_index in range(args.shard_count):
        shard = args.shard_root / f"shard{shard_index}"
        completion = json.loads((shard / "complete.json").read_text())
        shard_rows = read_jsonl(shard / f"{args.action}.jsonl")
        if completion["action"] != args.action or completion["shard_index"] != shard_index:
            raise RuntimeError(f"shard completion identity mismatch: {shard}")
        if not all(row["identity"] == identity for row in shard_rows):
            raise RuntimeError(f"ledger identity mismatch: {shard}")
        if any(row["status"] == "exhausted" for row in shard_rows):
            raise RuntimeError(f"exhausted shard: {shard}")
        final = latest(shard_rows)
        final_selected = [row for row in final.values() if row["status"] == "selected"]
        if completion["selected"] != len(final_selected):
            raise RuntimeError(f"shard selected count mismatch: {shard}")
        if set(completion["slots"]) != {row["slot"] for row in final_selected}:
            raise RuntimeError(f"shard slot coverage mismatch: {shard}")

        for row in shard_rows:
            uuid = row.get("uuid")
            if row["status"] == "started" and uuid not in states:
                append_jsonl(main_path, row)
                states[uuid] = row
            elif row["status"] == "rejected" and states.get(uuid, {}).get("status") != "rejected":
                if uuid not in states:
                    raise RuntimeError(f"rejection missing start: {uuid}")
                rejected = {
                    "uuid": uuid,
                    "slot": row["slot"],
                    "status": "rejected",
                    "reason": f"shard{shard_index}: {row['reason']}",
                    "identity": identity,
                }
                append_jsonl(main_path, rejected)
                states[uuid] = rejected

        for uuid, row in sorted(final.items(), key=lambda item: item[1]["slot"]):
            if row["status"] != "selected":
                continue
            slot = row["slot"]
            source = shard / args.action / uuid
            destination = args.attempts / args.action / uuid
            if slot in selected:
                if selected[slot]["uuid"] != uuid or not destination.exists():
                    raise RuntimeError(f"partial merge mismatch for slot {slot}")
                continue
            if source.exists() and destination.exists():
                raise RuntimeError(f"both source and destination exist: {uuid}")
            if not source.exists() and not destination.exists():
                raise RuntimeError(f"artifact directory missing: {uuid}")
            mapped = remap_artifacts(row["artifacts"], source, destination)
            if source.exists():
                source.rename(destination)
            # destination-only is the recoverable state after rename but before
            # ledger append. Full content hashes are checked by final audit.
            selected_row = {
                "uuid": uuid,
                "slot": slot,
                "status": "selected",
                "artifacts": mapped,
                "identity": identity,
            }
            append_jsonl(main_path, selected_row)
            states[uuid] = selected_row
            selected[slot] = selected_row
            merged_slots.append(slot)

    if set(selected) != set(range(160)):
        missing = sorted(set(range(160)) - set(selected))
        raise RuntimeError(f"canonical quota incomplete: {len(selected)}, missing={missing}")
    # One durability boundary after the resumable per-row operations.
    with main_path.open("a") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    report = {
        "action": args.action,
        "selected": 160,
        "merged_slots": sorted(merged_slots),
        "shards": args.shard_count,
        "identity": identity,
    }
    (args.shard_root / "merge.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
