#!/usr/bin/env python3
"""Export one Lance row into a standalone target-DOF replay package.

Run this exporter on a healthy machine with the Lance source available.  Copy
the resulting ``.npz`` and adjacent ``.json`` to a replay host; the replay CLI
never opens Lance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import lance
import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.contracts import CONTROL_TIMESTEP, PHYSICS_SUBSTEPS_PER_TARGET

from sim.manorl.target_replay import (
    TargetReplayPackageError,
    write_target_replay_package,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _row_object(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    index = row.get("index")
    metadata = row.get("trajectory_metadata")
    objects = row.get("objects")
    if (
        not isinstance(index, dict)
        or not isinstance(metadata, dict)
        or not isinstance(objects, list)
    ):
        raise ValueError("Lance row lacks index, trajectory_metadata, or objects")
    object_name = str(index.get("scene") or "")
    names = metadata.get("object_names")
    if isinstance(names, list) and object_name in names:
        object_index = names.index(object_name)
    else:
        object_index = 0
    if not 0 <= object_index < len(objects) or not isinstance(
        objects[object_index], dict
    ):
        raise ValueError("Lance row does not contain the selected object state")
    return object_name, objects[object_index]


def _right_hand(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("trajectory_metadata")
    hands = row.get("hands")
    if not isinstance(metadata, dict) or not isinstance(hands, list):
        raise ValueError("Lance row lacks trajectory hand data")
    names = metadata.get("hand_names")
    if isinstance(names, list) and "right" in names:
        hand_index = names.index("right")
    else:
        hand_index = 0
    if not 0 <= hand_index < len(hands) or not isinstance(hands[hand_index], dict):
        raise ValueError("Lance row lacks a right-hand slot")
    return hands[hand_index]


def _source_identity(row: dict[str, Any], object_name: str, row_index: int) -> str:
    provenance = row.get("provenance")
    if isinstance(provenance, dict) and isinstance(
        provenance.get("source_identity"), str
    ):
        return str(provenance["source_identity"])
    metadata = row.get("trajectory_metadata")
    gesture = "01"
    raw_id = row_index
    if isinstance(metadata, dict):
        gesture = str(metadata.get("gesture") or gesture)
        raw_info = metadata.get("raw_data_info")
        if isinstance(raw_info, dict) and isinstance(raw_info.get("id"), int):
            raw_id = int(raw_info["id"])
    digits = "".join(character for character in gesture if character.isdigit()) or "1"
    return f"{object_name}_{int(digits):02d}_{raw_id}"


def _checkpoint_warp_ccd(provenance: dict[str, Any]) -> tuple[int | None, int | None]:
    """Extract recorded CCD settings without inventing a runtime override."""

    candidates: list[dict[str, Any]] = []
    direct = provenance.get("warp_ccd")
    if isinstance(direct, dict):
        candidates.append(direct)
    raw_metadata = provenance.get("checkpoint_metadata_json")
    if isinstance(raw_metadata, str) and raw_metadata:
        try:
            decoded = json.loads(raw_metadata)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            runtime = decoded.get("runtime_config")
            environment = (
                runtime.get("environment") if isinstance(runtime, dict) else None
            )
            warp = (
                environment.get("warp_ccd") if isinstance(environment, dict) else None
            )
            if isinstance(warp, dict):
                candidates.append(warp)
    candidates.append(provenance)
    for candidate in candidates:
        iterations = candidate.get(
            "ccd_iterations", candidate.get("warp_ccd_iterations")
        )
        contacts = candidate.get(
            "contacts_per_world", candidate.get("warp_ccd_contacts_per_world")
        )
        if (
            isinstance(iterations, int)
            and iterations >= 1
            and isinstance(contacts, int)
            and contacts >= 1
        ):
            return int(iterations), int(contacts)
    return None, None


def export_package(
    *,
    dataset: Path,
    dataset_version: int,
    row_index: int,
    output: Path,
    replace: bool,
) -> dict[str, Any]:
    lance_dataset = lance.dataset(str(dataset), version=dataset_version)
    rows = lance_dataset.take([row_index]).to_pylist()
    if len(rows) != 1:
        raise ValueError(
            f"dataset.take returned {len(rows)} rows for row index {row_index}"
        )
    row = rows[0]
    hand = _right_hand(row)
    object_name, object_state = _row_object(row)
    timestamps = np.asarray(row.get("timestamp", ()), dtype=np.float64)
    recorded_qpos = np.asarray(hand.get("urdf_dof", ()), dtype=np.float64)
    target_qpos = np.asarray(hand.get("urdf_dof_target", ()), dtype=np.float64)
    object_position = np.asarray(object_state.get("pos", ()), dtype=np.float64)
    object_quaternion_xyzw = Rotation.from_rotvec(
        np.asarray(object_state.get("rot_aa", ()), dtype=np.float64)
    ).as_quat()
    if timestamps.ndim != 1 or recorded_qpos.ndim != 2 or target_qpos.ndim != 2:
        raise ValueError("Lance row target replay fields have invalid ranks")
    if len(timestamps) < 2:
        raise ValueError("Lance row must contain at least two frames")
    provenance = (
        row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    )
    index = row.get("index") if isinstance(row.get("index"), dict) else {}
    trajectory_metadata = (
        row.get("trajectory_metadata")
        if isinstance(row.get("trajectory_metadata"), dict)
        else {}
    )
    movement = trajectory_metadata.get("trajectory_info", {}).get("object_move", [])
    movement_entry = (
        next(
            (entry for entry in movement if entry.get("object_name") == object_name),
            None,
        )
        if isinstance(movement, list)
        else None
    )
    ccd_iterations, ccd_contacts = _checkpoint_warp_ccd(provenance)
    metadata: dict[str, Any] = {
        "object": object_name,
        "source_identity": _source_identity(row, object_name, row_index),
        "source_uuid": str(
            provenance.get("source_uuid")
            or index.get("seed_uuid")
            or index.get("uuid")
            or ""
        ),
        "source_file_uuid": str(index.get("file_uuid") or ""),
        "source_dataset": str(dataset),
        "source_dataset_version": dataset_version,
        "source_row_index": row_index,
        "checkpoint_update": (
            int(provenance["checkpoint_update"])
            if isinstance(provenance.get("checkpoint_update"), int)
            else None
        ),
        "checkpoint_sha256": str(provenance.get("checkpoint_sha256") or ""),
        "control_timestep_seconds": CONTROL_TIMESTEP,
        "physics_substeps_per_target": PHYSICS_SUBSTEPS_PER_TARGET,
        "warp_ccd_iterations": ccd_iterations,
        "warp_ccd_contacts_per_world": ccd_contacts,
        "movement_start_raw": (
            int(movement_entry.get("start_frame", 0)) if movement_entry else 0
        ),
        "movement_end_raw": (
            int(movement_entry.get("end_frame", len(timestamps) - 1))
            if movement_entry
            else len(timestamps) - 1
        ),
    }
    package = write_target_replay_package(
        output,
        metadata=metadata,
        timestamps=timestamps,
        recorded_qpos=recorded_qpos,
        target_qpos=target_qpos,
        object_position=object_position,
        object_quaternion_xyzw=object_quaternion_xyzw,
        replace=replace,
    )
    return {
        "schema": "manorl.target_dof_replay_export_result.v1",
        "package": str(package.path),
        "metadata": str(package.path.with_suffix(".json")),
        "npz_sha256": package.metadata["npz_sha256"],
        "frames": package.frames,
        "transitions": package.transitions,
        "object": package.object_type,
        "source_identity": package.source_identity,
        "source_uuid": package.metadata["source_uuid"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-version", type=_positive_int, required=True)
    parser.add_argument("--row-index", type=_positive_int, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="destination .npz package"
    )
    parser.add_argument(
        "--replace", action="store_true", help="replace an existing package explicitly"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_package(
            dataset=args.dataset,
            dataset_version=args.dataset_version,
            row_index=args.row_index,
            output=args.output,
            replace=args.replace,
        )
    except (TargetReplayPackageError, OSError, ValueError, KeyError) as exc:
        print(f"target-DOF package export error: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
