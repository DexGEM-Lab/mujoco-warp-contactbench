#!/usr/bin/env python3
"""Fail-closed validation and atomic publication of exact800 U1 Lance."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

from sim.manorl.u1_action005 import load_action005_parent
from sim.manorl.u1_campaign import digest, file_sha, write_json
from tools.build_u1_largepose800_scene_layout import CONTRACT as LAYOUT_CONTRACT
from tools.compute_u1_exact800_background_offsets import ALL_NAMES
from tools.export_u1_largepose800 import (
    ACTIONS,
    AUGMENTATION_CONTRACT,
    CONTRACT,
    EXPECTED_COUNTS,
    SETTING,
    schema,
)

PUBLICATION_CONTRACT = "manorl.u1-largepose800.publication.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _copy_parent_bundles(registry_path: Path, output: Path) -> None:
    registry = json.loads(registry_path.read_text())
    destination_root = output / "parents"
    require(not destination_root.exists(), "parent destination already exists")
    for row in range(10):
        parent = registry["parents"][str(row)]
        source = registry_path.parent / parent["folder"]
        destination = output / parent["folder"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, copy_function=shutil.copyfile)
    for row in range(10):
        load_action005_parent(output / "action005_registry.json", row)


def _validate_action005_rows(dataset) -> Counter:
    parent_counts: Counter[int] = Counter()
    columns = [
        "index",
        "trajectory_metadata",
        "objects",
        "physical",
        "native_contacts",
        "lineage_json",
        "teacher_qpos",
        "reference_objects_json",
    ]
    for row_index in range(160, 320):
        row = dataset.take([row_index], columns=columns).to_pylist()[0]
        metadata = row["trajectory_metadata"]
        frames = metadata["total_frames"]
        require(
            metadata["object_names"] == ["mayonnaisebottle", "bowl"],
            "action005 complete scene",
        )
        require(len(row["objects"]) == 2, "action005 object tracks")
        physical = row["physical"]
        require(
            all(len(physical[key]) == frames for key in ("qpos", "qvel", "ctrl")),
            "action005 physical frames",
        )
        require(len(row["native_contacts"]) == frames, "action005 native frames")
        require(len(row["teacher_qpos"]) == frames, "action005 teacher frames")
        reference_objects = json.loads(row["reference_objects_json"])
        require(
            reference_objects["scene_object_names"] == ["mayonnaisebottle", "bowl"],
            "action005 reference scene",
        )
        require(
            len(reference_objects["source_object_pos"]) == frames
            and len(reference_objects["source_object_quat_xyzw"]) == frames,
            "action005 reference frames",
        )
        lineage = json.loads(row["lineage_json"])
        require(
            lineage["contract"] == CONTRACT
            and lineage["action"] == "005"
            and lineage["parent_uuid"] == row["index"]["seed_uuid"],
            "action005 row lineage",
        )
        parent_counts[int(lineage["parent_row"])] += 1
    require(parent_counts == Counter({row: 16 for row in range(10)}), "parent quotas")
    return parent_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-output", type=Path, required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    parser.add_argument("--action005-registry", type=Path, required=True)
    args = parser.parse_args()
    staging = args.staging_output.resolve(strict=True)
    final = args.final_output.resolve()
    registry_path = args.action005_registry.resolve(strict=True)
    require(staging.is_dir() and staging.name.startswith("."), "hidden staging output")
    require(not final.exists(), "final output already exists")
    require(
        staging.stat().st_dev == final.parent.stat().st_dev,
        "publication must stay on one filesystem",
    )
    require(
        file_sha(staging / "action005_registry.json") == file_sha(registry_path),
        "registry snapshot differs",
    )

    import lance

    dataset = lance.dataset(str(staging / "compact.lance"))
    require(dataset.version == 1 and dataset.count_rows() == 800, "Lance version/count")
    require(dataset.schema.equals(schema(), check_metadata=True), "Lance schema")
    compact = dataset.to_table(columns=["index", "trajectory_metadata"]).to_pylist()
    uuids = [row["index"]["uuid"] for row in compact]
    actions = [row["trajectory_metadata"]["gesture"].split("-")[0] for row in compact]
    require(len(set(uuids)) == 800, "UUID uniqueness")
    require(Counter(actions) == Counter(EXPECTED_COUNTS), "action quotas")
    require(actions == [action for action in ACTIONS for _ in range(160)], "action order")
    parent_counts = _validate_action005_rows(dataset)

    manifest = json.loads((staging / "manifest.json").read_text())
    validation = json.loads((staging / "validation.json").read_text())
    parent_validation = json.loads(
        (staging / "action005_parent_validation.json").read_text()
    )
    shard_merge = json.loads((staging / "action005_shard_merge.json").read_text())
    offsets = json.loads((staging / "background_offsets.json").read_text())
    layout = json.loads((staging / "visualization_layout.json").read_text())
    require(manifest["contract"] == CONTRACT, "manifest contract")
    require(manifest["setting_sha256"] == SETTING, "manifest setting")
    require(manifest["ordered_uuids"] == uuids, "manifest UUID order")
    require(validation["validated"] is True and validation["rows"] == 800, "validation")
    require(validation["counts"] == EXPECTED_COUNTS, "validated action quotas")
    require(
        parent_validation["validated"] is True
        and parent_validation["parents"] == 10
        and parent_validation["registry_digest"]
        == manifest["action005_registry_digest"],
        "action005 parent validation",
    )
    require(
        shard_merge["selected"] == 160
        and shard_merge["per_parent"] == {str(row): 16 for row in range(10)}
        and shard_merge["plan_digest"] == manifest["action005_plan_digest"],
        "action005 shard merge",
    )
    require(offsets["lance"] == "compact.lance", "offset Lance path")
    require(set(offsets["report_by_action"]) == set(ACTIONS), "offset action coverage")
    require(
        all(
            report["rows"] == 160 and report["unresolved_pairs"] == 0
            for report in offsets["report_by_action"].values()
        ),
        "unresolved background clearances",
    )
    require(layout["contract"] == LAYOUT_CONTRACT, "layout contract")
    require(layout["lance"] == "compact.lance", "layout Lance path")
    require(layout["total_trajectories"] == 800, "layout row count")
    require(layout["counts_by_action"] == EXPECTED_COUNTS, "layout action counts")
    require([row["uuid"] for row in layout["trajectories"]] == uuids, "layout UUID order")
    for row in layout["trajectories"]:
        require(set(row["objects"]) == set(ALL_NAMES), "layout complete scene")
        if row["action"] == "005":
            require(
                row["active_objects"] == ["mayonnaisebottle", "bowl"],
                "layout action005 active scene",
            )

    source640 = Path(manifest["source_exact640"]["path"])
    setting = json.loads((source640 / "setting.json").read_text())
    require(digest(setting) == SETTING, "source setting digest")
    write_json(staging / "setting.json", setting)
    registry = json.loads(registry_path.read_text())
    require(
        all(
            parent["setting_sha256"] == SETTING
            and digest(parent["setting"]) == SETTING
            for parent in registry["parents"].values()
        ),
        "action005 parent settings",
    )
    shutil.copyfile(
        registry["assets"]["manifest"], staging / "action005_asset_manifest.json"
    )
    _copy_parent_bundles(registry_path, staging)

    readme = f"""# ManoRL strict-C1 U1 large-pose exact800

This directory contains 800 newly generated Cheyingtong right-hand trajectories:
160 each for actions 003, 005, 006, 007, and 009.

- U1 setting SHA-256: `{SETTING}`
- dataset contract: `{CONTRACT}`
- augmentation contract: `{AUGMENTATION_CONTRACT}`
- Lance: `compact.lance` (version 1, 800 rows)
- source-bound validation: `validation.json`
- deterministic action005 plan/registry: `action005_plan.json`, `action005_registry.json`
- parent qualification and shard provenance: `action005_parent_validation.json`,
  `action005_shard_merge.json`
- runnable action005 parent bundles: `parents/row00` through `parents/row09`
- visualization layout and clearances: `visualization_layout.json`, `background_offsets.json`
- file hashes: `sha256.json`

The published exact640 source is unchanged. Its rows are interleaved read-only with
160 new action005 rows. Every action005 child retains mayonnaisebottle and bowl,
uses a strict 120-frame physical-teacher discrete-C1 prefix, has zero solved-force
prefix contact, and passed two frozen-control physical replays with distinct PIDs.
The exported physical state, native contact-frame evidence, teacher state, and
complete-scene reference were reconstructed from the second replay and compared
exactly against Lance readback.

Background objects in `visualization_layout.json` are visualization-only and are
outside recorded physics.
"""
    (staging / "README.md").write_text(readme)
    publication = {
        "contract": PUBLICATION_CONTRACT,
        "dataset_directory": final.name,
        "dataset_contract": CONTRACT,
        "layout_contract": LAYOUT_CONTRACT,
        "lance_version": 1,
        "rows": 800,
        "counts": EXPECTED_COUNTS,
        "action005_parent_counts": dict(sorted(parent_counts.items())),
        "unique_uuids": 800,
        "setting_sha256": SETTING,
        "source_exact640_hash_index_sha256": manifest["source_exact640"][
            "hash_index_sha256"
        ],
        "action005_registry_digest": manifest["action005_registry_digest"],
        "action005_plan_digest": manifest["action005_plan_digest"],
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
    final_dataset = lance.dataset(str(final / "compact.lance"))
    require(final_dataset.version == 1 and final_dataset.count_rows() == 800, "final Lance")
    require(
        (final / "sha256.json.sha256").read_text().strip()
        == file_sha(final / "sha256.json"),
        "final hash index",
    )
    print(
        json.dumps(
            {
                "published": str(final),
                "rows": 800,
                "counts": EXPECTED_COUNTS,
                "hashed_files": len(hashes),
            }
        )
    )


if __name__ == "__main__":
    main()
