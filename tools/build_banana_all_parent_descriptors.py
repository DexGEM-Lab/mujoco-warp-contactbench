#!/usr/bin/env python3
"""Build accepted-parent v3 descriptors for every strict banana source.

Reads the strict one-parent-per-source ledger (263 banana identities) plus the
pinned production Lance and the raw source Lance lineage, and writes one
descriptor per source. The 56 ineligible banana sources are recorded in an
unavailable manifest and are not part of the strict augmented production.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.manorl.synthetic_parent import (
    ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
    AcceptedSyntheticParent,
    load_accepted_synthetic_parent,
    write_accepted_synthetic_parent,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(
            "/home/jay/data/manorl_banana_vla_parent_screen_20260820/"
            "one_parent_per_source.json"
        ),
    )
    parser.add_argument(
        "--production-dataset",
        type=Path,
        default=Path(
            "/mnt/nas-222-projects/sunjieqiang/mujoco_synthetic/"
            "gg_f120_xy02cm_8obj_contact_20260811.lance"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-version", type=int, default=295)
    args = parser.parse_args(argv)

    import lance

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    eligible = [item for item in ledger if item.get("eligible") and str(item.get("source_identity", "")).startswith("banana_")]
    eligible_by_identity = {item["source_identity"]: item for item in eligible}
    if len(eligible_by_identity) != len(eligible):
        raise ValueError("ledger contains duplicate eligible banana sources")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    production = lance.dataset(str(args.production_dataset))
    row_indices = sorted({item["row_index"] for item in eligible})
    provenance_rows = production.take(row_indices, columns=["provenance"]).to_pylist()
    provenance_by_index = {
        row_index: row["provenance"]
        for row_index, row in zip(row_indices, provenance_rows, strict=True)
    }
    source = lance.dataset(str(args.source_dataset), version=args.source_version)
    source_row_indices = sorted(
        {int(prov["row_index"]) for prov in provenance_by_index.values()}
    )
    source_rows = source.take(
        source_row_indices, columns=["hands", "trajectory_metadata"]
    ).to_pylist()
    source_by_index = {
        row_index: row
        for row_index, row in zip(source_row_indices, source_rows, strict=True)
    }

    written = []
    for item in sorted(eligible, key=lambda row: row["source_identity"]):
        prov = provenance_by_index[item["row_index"]]
        raw = source_by_index[int(prov["row_index"])]
        names = raw["trajectory_metadata"]["hand_names"]
        slot = names.index("right")
        frame0 = raw["hands"][slot]["urdf_dof"][0][3:28]
        parent = AcceptedSyntheticParent(
            contract=ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
            parent_dataset_path=str(args.production_dataset.resolve()),
            parent_dataset_version=int(production.version),
            parent_row_index=item["row_index"],
            parent_row_uuid=item["parent_uuid"],
            parent_row_contract=prov["contract"],
            source_identity=item["source_identity"],
            source_dataset_path=prov["dataset_path"],
            source_dataset_version=int(prov["dataset_version"]),
            source_row_index=int(prov["row_index"]),
            checkpoint_sha256=prov["checkpoint_sha256"],
            checkpoint_update=int(prov["checkpoint_update"]),
            parent_seed=int(prov["seed"]),
            parent_episode_index=int(prov["episode_index"]),
            parent_generation_attempt=int(prov["generation_attempt"]),
            object_init_xy_offset_m=(
                item["object_offset_x_m"],
                item["object_offset_y_m"],
            ),
            reference_fps=int(prov["reference_fps"]),
            retreat_last_contact_state_index=item["last_contact_state"],
            retreat_anchor_state_index=item["anchor_state"],
            retreat_anchor_source_frame_index=item["anchor_source_frame"],
            retreat_anchor_horizontal_distance_m=item["horizontal_retreat_m"],
            retreat_anchor_offset_frames=15,
            parent_movement_end_state_index=item["movement_end"],
            source_row_frame0_right_q_ref_3_28=tuple(float(v) for v in frame0),
        )
        path = output / f"{item['source_identity']}.json"
        write_accepted_synthetic_parent(parent, path, replace=True)
        assert load_accepted_synthetic_parent(path) == parent
        written.append(str(path))

    manifest = {
        "contract": "manorl_banana_all_strict_accepted_parents_v1",
        "eligible_sources": len(written),
        "production_dataset": str(args.production_dataset.resolve()),
        "source_dataset": str(args.source_dataset.resolve()),
        "source_version": args.source_version,
        "descriptors": written,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"descriptors": len(written), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
