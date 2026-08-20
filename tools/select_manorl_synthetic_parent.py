#!/usr/bin/env python3
"""Select one accepted scalable-synthesis row for approach augmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from sim.manorl.synthetic_parent import (
    ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
    AcceptedSyntheticParent,
    write_accepted_synthetic_parent,
)


def select_parent(
    dataset_path: str | Path,
    *,
    row_uuid: str,
    output: str | Path,
    dataset_version: int | None = None,
    replace: bool = False,
) -> Path:
    import lance

    dataset = lance.dataset(str(Path(dataset_path).expanduser().resolve()), version=dataset_version)
    # These four columns are stable across the historical compact fragments.
    # trajectory_metadata had an old nested object_move schema split and is not
    # required to bind an accepted parent.
    table = dataset.scanner(
        columns=["index", "objects", "reference", "provenance"]
    ).to_table()
    matched = [
        (row_index, row)
        for row_index, row in enumerate(table.to_pylist())
        if (row.get("index") or {}).get("uuid") == row_uuid
    ]
    if len(matched) != 1:
        raise LookupError(
            f"expected exactly one accepted parent row UUID {row_uuid!r}, got {len(matched)}"
        )
    parent_row_index, row = matched[0]
    provenance = dict(row.get("provenance") or {})
    objects = row.get("objects") or []
    reference = dict(row.get("reference") or {})
    if len(objects) != 1 or not isinstance(objects[0], dict):
        raise ValueError("accepted parent must contain exactly one object")
    actual_object = np.asarray(objects[0].get("pos") or (), dtype=np.float64)
    reference_object = np.asarray(reference.get("object_pos") or (), dtype=np.float64)
    if actual_object.ndim != 2 or reference_object.shape != actual_object.shape or actual_object.shape[1:] != (3,):
        raise ValueError("accepted parent object/reference positions are not frame-aligned")
    offset = actual_object[0, :2] - reference_object[0, :2]
    source_identity = provenance.get("source_identity")
    if not isinstance(source_identity, str):
        raise ValueError("accepted parent omits source_identity")
    parent = AcceptedSyntheticParent(
        contract=ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
        parent_dataset_path=str(Path(dataset_path).expanduser().resolve()),
        parent_dataset_version=int(dataset.version),
        parent_row_index=parent_row_index,
        parent_row_uuid=row_uuid,
        parent_row_contract=str(provenance.get("contract") or ""),
        source_identity=source_identity,
        source_dataset_path=str(provenance.get("dataset_path") or ""),
        source_dataset_version=int(provenance.get("dataset_version")),
        source_row_index=int(provenance.get("row_index")),
        checkpoint_sha256=str(provenance.get("checkpoint_sha256") or ""),
        checkpoint_update=int(provenance.get("checkpoint_update")),
        parent_seed=int(provenance.get("seed")),
        parent_episode_index=int(provenance.get("episode_index")),
        parent_generation_attempt=int(provenance.get("generation_attempt")),
        object_init_xy_offset_m=(float(offset[0]), float(offset[1])),
        reference_fps=int(provenance.get("reference_fps")),
    )
    return write_accepted_synthetic_parent(parent, output, replace=replace)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="prior scalable synthetic Lance")
    parser.add_argument("--row-uuid", required=True, help="accepted generated row UUID")
    parser.add_argument("--output", type=Path, required=True, help="accepted-parent JSON descriptor")
    parser.add_argument("--dataset-version", type=int)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = select_parent(
        args.input,
        row_uuid=args.row_uuid,
        output=args.output,
        dataset_version=args.dataset_version,
        replace=args.replace,
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
