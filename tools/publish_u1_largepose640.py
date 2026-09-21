#!/usr/bin/env python3
"""Fail-closed finalization and atomic publication of exact640 U1 Lance."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

from sim.manorl.u1_campaign import digest, file_sha, write_json
from tools.export_u1_largepose640 import ACTIONS, CONTRACT, SETTING, SOURCE_REGISTRY, schema
from tools.build_u1_largepose640_scene_layout import CONTRACT as LAYOUT_CONTRACT

EXPECTED_COUNTS = {action: 160 for action in ACTIONS}
PUBLICATION_CONTRACT = "manorl.u1-largepose640.publication.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-output", type=Path, required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    args = parser.parse_args()

    staging = args.staging_output
    final = args.final_output
    require(staging.is_dir() and staging.name.startswith("."), "staging output must be hidden")
    require(not final.exists(), "final output already exists")
    require(staging.stat().st_dev == final.parent.stat().st_dev, "publication must stay on one filesystem")

    import lance

    dataset = lance.dataset(str(staging / "compact.lance"))
    require(dataset.version == 1, "unexpected Lance version")
    require(dataset.count_rows() == 640, "Lance row count")
    require(dataset.schema.equals(schema(), check_metadata=True), "Lance schema")
    compact = dataset.to_table(columns=["index", "trajectory_metadata"]).to_pylist()
    uuids = [row["index"]["uuid"] for row in compact]
    counts = Counter(row["trajectory_metadata"]["gesture"].split("-")[0] for row in compact)
    require(len(set(uuids)) == 640, "UUID uniqueness")
    require(dict(counts) == EXPECTED_COUNTS, f"action counts: {dict(counts)}")

    manifest = json.loads((staging / "manifest.json").read_text())
    validation = json.loads((staging / "validation.json").read_text())
    offsets = json.loads((staging / "background_offsets.json").read_text())
    layout = json.loads((staging / "visualization_layout.json").read_text())
    require(manifest["contract"] == CONTRACT, "manifest contract")
    require(manifest["setting_sha256"] == SETTING, "setting hash")
    require(manifest["ordered_uuids"] == uuids, "manifest UUID order")
    registry = json.loads(SOURCE_REGISTRY.read_text())
    settings = [registry["parents"][action]["setting"] for action in ACTIONS]
    require(all(setting == settings[0] for setting in settings), "registry settings differ")
    require(digest(settings[0]) == SETTING, "registry setting digest")
    shutil.copyfile(SOURCE_REGISTRY, staging / "registry.json")
    write_json(staging / "setting.json", settings[0])
    manifest.update(
        registry_digest=registry["digest"],
        registry_sha256=file_sha(staging / "registry.json"),
        plan_sha256=file_sha(staging / "plan.json"),
    )
    write_json(staging / "manifest.json", manifest)
    require(validation["validated"] is True and validation["rows"] == 640, "source-bound validation")
    require(validation["counts"] == EXPECTED_COUNTS, "validated action counts")
    require(offsets["lance"] == "compact.lance", "offset Lance path must be relative")
    require(layout["contract"] == LAYOUT_CONTRACT, "layout contract")
    require(layout["lance"] == "compact.lance", "layout Lance path must be relative")
    require(layout["total_trajectories"] == 640, "layout row count")
    require(layout["counts_by_action"] == EXPECTED_COUNTS, "layout action counts")
    require([row["uuid"] for row in layout["trajectories"]] == uuids, "layout UUID order")
    require(
        all(stat["rows"] == 160 and stat["unresolved_pairs"] == 0 for stat in offsets["report_by_action"].values()),
        "background layout has unresolved clearances",
    )
    require(set(offsets["report_by_action"]) == set(ACTIONS), "offset action coverage")

    readme = f"""# ManoRL strict-C1 U1 large-pose exact640

This directory contains 640 newly generated Cheyingtong right-hand trajectories:
160 each for actions 003, 006, 007, and 009. Action005 is intentionally excluded.

- U1 setting SHA-256: `{SETTING}`
- augmentation contract: `u1-four-action-largepose-v3-exact-discrete-c1`
- dataset contract: `{CONTRACT}`
- Lance: `compact.lance` (version 1, 640 rows)
- source-bound validation: `validation.json`
- frozen source registry and U1 setting: `registry.json`, `setting.json`
- deterministic perturbation plan: `plan.json`
- rejected attempts retained as evidence: `rejections.json`
- per-UUID visualization layout: `visualization_layout.json`
- visualization clearance solver output: `background_offsets.json`
- initial seven unresolved clearance cases: `background_offsets_initial.json`
- file hashes: `sha256.json`

Every accepted child has two accepted frozen-control U1 replays with distinct
process IDs. The exported row is reconstructed from the second replay and was
compared in full against Lance readback. Prefix contacts use solved native
normal force, and the complete native contact-frame evidence is retained in each
row. The 120-frame approach joins the physical teacher reference by the frozen
discrete-C1 rule; the complete parent control suffix is byte-exact.

Background objects in `visualization_layout.json` are visualization-only and are
not part of the recorded physics.
"""
    (staging / "README.md").write_text(readme)
    publication = {
        "contract": PUBLICATION_CONTRACT,
        "dataset_directory": final.name,
        "dataset_contract": CONTRACT,
        "layout_contract": LAYOUT_CONTRACT,
        "lance_version": 1,
        "rows": 640,
        "counts": EXPECTED_COUNTS,
        "unique_uuids": 640,
        "setting_sha256": SETTING,
        "plan_digest": manifest["plan_digest"],
        "source_bound_readback": True,
        "background_clearances_resolved": True,
    }
    write_json(staging / "publication.json", publication)

    hash_path = staging / "sha256.json"
    hash_sidecar = staging / "sha256.json.sha256"
    for old in (hash_path, hash_sidecar):
        if old.exists():
            old.unlink()
    files = [
        path
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path not in (hash_path, hash_sidecar)
    ]
    print(json.dumps({"phase": "publish", "step": "hash_start", "files": len(files)}), flush=True)
    hashes = {str(path.relative_to(staging)): file_sha(path) for path in files}
    write_json(hash_path, hashes)
    hash_sidecar.write_text(file_sha(hash_path) + "\n")
    print(json.dumps({"phase": "publish", "step": "hash_complete"}), flush=True)

    os.rename(staging, final)
    print(
        json.dumps(
            {
                "published": str(final),
                "rows": 640,
                "counts": EXPECTED_COUNTS,
                "hashed_files": len(hashes),
            }
        )
    )


if __name__ == "__main__":
    main()
