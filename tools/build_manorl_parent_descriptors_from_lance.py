#!/usr/bin/env python3
"""Build accepted-parent descriptors from every eligible row in one Lance dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim.manorl.synthetic_parent import load_accepted_synthetic_parent
from tools.select_manorl_synthetic_parent import select_parent


def build_descriptors(
    dataset_path: Path,
    *,
    output_dir: Path,
    predecoded_manifest: Path,
    replace: bool = False,
) -> Path:
    import lance

    dataset_path = dataset_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not replace:
        raise FileExistsError(f"accepted-parent output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    predecoded = json.loads(predecoded_manifest.read_text(encoding="utf-8"))
    records = {
        str(record.get("identity")): record
        for record in predecoded.get("valid_records", [])
    }
    dataset = lance.dataset(str(dataset_path))
    index_rows = dataset.scanner(columns=["index", "provenance"]).to_table().to_pylist()
    written: list[str] = []
    unavailable: list[dict[str, object]] = []
    seen_identity: set[str] = set()
    for row_index, row in enumerate(index_rows):
        index = row.get("index") or {}
        provenance = row.get("provenance") or {}
        row_uuid = index.get("uuid")
        identity = provenance.get("source_identity")
        if not isinstance(row_uuid, str) or not isinstance(identity, str):
            unavailable.append(
                {
                    "row_index": row_index,
                    "row_uuid": row_uuid,
                    "source_identity": identity,
                    "reason": "missing_uuid_or_source_identity",
                }
            )
            continue
        if identity in seen_identity:
            unavailable.append(
                {
                    "row_index": row_index,
                    "row_uuid": row_uuid,
                    "source_identity": identity,
                    "reason": "duplicate_source_identity_after_first_eligible_row",
                }
            )
            continue
        record = records.get(identity)
        if record is None:
            unavailable.append(
                {
                    "row_index": row_index,
                    "row_uuid": row_uuid,
                    "source_identity": identity,
                    "reason": "missing_pre60_source",
                }
            )
            continue
        descriptor_path = output_dir / f"{identity}.json"
        try:
            select_parent(
                dataset_path,
                row_uuid=row_uuid,
                output=descriptor_path,
                dataset_version=int(dataset.version),
                replace=replace,
            )
            parent = load_accepted_synthetic_parent(descriptor_path)
            # Confirm the accepted movement-end+15 source frame against the
            # production pre60 bundle that final augmentation will consume.
            import pickle

            with (predecoded_manifest.parent / f"{identity}.pkl").open("rb") as stream:
                trajectory = pickle.load(stream)
            anchor = int(trajectory.movement_end_step) + parent.retreat_anchor_offset_frames
            if (
                not 0 <= anchor < len(trajectory.source_indices)
                or int(trajectory.source_indices[anchor])
                != int(parent.retreat_anchor_source_frame_index)
            ):
                raise ValueError("pre60 movement_end+15 source mapping differs")
        except Exception as exc:
            descriptor_path.unlink(missing_ok=True)
            unavailable.append(
                {
                    "row_index": row_index,
                    "row_uuid": row_uuid,
                    "source_identity": identity,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        seen_identity.add(identity)
        written.append(str(descriptor_path))
    by_action: dict[str, dict[str, str]] = {}
    for raw in written:
        path = Path(raw)
        identity = path.stem
        _, action, _ = identity.split("_")
        by_action.setdefault(action, {})[identity] = str(path)
    manifest = {
        "contract": "manorl_lance_accepted_parent_descriptors_v1",
        "production_dataset": str(dataset_path),
        "production_dataset_version": int(dataset.version),
        "predecoded_manifest": str(predecoded_manifest.resolve()),
        "rows": len(index_rows),
        "verified_sources": len(written),
        "unavailable_sources": len(unavailable),
        "descriptors": written,
        "unavailable": unavailable,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "parents_by_action.json").write_text(
        json.dumps(
            {
                "contract": "manorl_synthesis_accepted_parents_by_action_v1",
                "actions": by_action,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predecoded-manifest", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = build_descriptors(
        args.input,
        output_dir=args.output_dir,
        predecoded_manifest=args.predecoded_manifest.resolve(),
        replace=args.replace,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
