#!/usr/bin/env python3
"""Select five farther, spatially diverse accepted parents per ManoRL pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.lance_v2 import file_sha256
from sim.manorl.synthetic_parent import load_accepted_synthetic_parent
from sim.manorl.synthesis_acceptance import (
    count_hand_object_contact_frames,
    evaluate_synthesis_acceptance,
)
from sim.manorl.trajectory import ReferenceTrajectory

CONFIG_CONTRACT = "manorl_targeted_parent_selection_config_v1"
SELECTION_CONTRACT = "manorl_targeted_parent_selection_v1"


def _current_parent_acceptance(
    parent: Any,
    *,
    object_type: str,
    dataset_cache: dict[tuple[str, int], Any],
) -> dict[str, object]:
    """Recompute the current production gate from one persisted parent row."""

    import lance

    key = (parent.parent_dataset_path, parent.parent_dataset_version)
    dataset = dataset_cache.get(key)
    if dataset is None:
        dataset = lance.dataset(
            parent.parent_dataset_path, version=parent.parent_dataset_version
        )
        dataset_cache[key] = dataset
    row = dataset.take(
        [parent.parent_row_index],
        columns=["index", "objects", "contact", "reference", "provenance"],
    ).to_pylist()[0]
    if (
        (row.get("index") or {}).get("uuid") != parent.parent_row_uuid
        or (row.get("provenance") or {}).get("source_identity")
        != parent.source_identity
    ):
        raise RuntimeError("accepted-parent descriptor no longer binds its Lance row")
    objects = row.get("objects") or []
    reference = row.get("reference") or {}
    if len(objects) != 1:
        raise ValueError("accepted parent must persist exactly one object")
    simulated_rotvec = np.asarray(
        objects[0].get("rot_aa")[-1], dtype=np.float32
    ).astype(np.float64)
    reference_rotvec = np.asarray(
        reference.get("object_rot_aa")[-1], dtype=np.float32
    ).astype(np.float64)
    contact_frames = count_hand_object_contact_frames(
        row.get("contact") or [], target_object_name=object_type
    )
    # Historical v3 descriptors come from success-only parent rows. They do
    # not persist termination codes, so descriptor eligibility supplies the
    # completion predicate while rotation/contact are independently recomputed
    # from the exact persisted row representation.
    return evaluate_synthesis_acceptance(
        trajectory_complete=True,
        termination_reason_code=1,
        simulated_final_object_quaternion_xyzw=Rotation.from_rotvec(
            simulated_rotvec
        ).as_quat(),
        reference_final_object_quaternion_xyzw=Rotation.from_rotvec(
            reference_rotvec
        ).as_quat(),
        hand_object_contact_frames=contact_frames,
    ).to_dict()


def _feature(
    record: dict[str, Any], bounds: dict[str, tuple[float, float]]
) -> np.ndarray:
    delta = np.asarray(record["pre60_wrist_object_delta_m"], dtype=np.float64)
    xy = float(np.linalg.norm(delta[:2]))
    azimuth = math.atan2(float(delta[1]), float(delta[0]))

    def normalize(name: str, value: float) -> float:
        low, high = bounds[name]
        return 0.5 if high - low <= 1e-12 else (value - low) / (high - low)

    return np.asarray(
        (
            math.cos(azimuth),
            math.sin(azimuth),
            normalize("xy", xy),
            normalize("z", float(delta[2])),
        ),
        dtype=np.float64,
    )


def select_records(
    records: list[dict[str, Any]], *, count: int = 5
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count < 1 or len(records) < count:
        raise ValueError("targeted parent selection lacks enough candidates")
    identities = [str(record["identity"]) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("targeted parent candidates repeat an identity")
    distances = np.asarray(
        [float(record["pre60_wrist_object_distance_m"]) for record in records],
        dtype=np.float64,
    )
    threshold = float(np.quantile(distances, 0.5))
    pool = [
        record
        for record in records
        if float(record["pre60_wrist_object_distance_m"]) >= threshold
    ]
    if len(pool) < count:
        raise RuntimeError("farther half contains fewer candidates than requested")
    xy = [
        float(
            np.linalg.norm(
                np.asarray(record["pre60_wrist_object_delta_m"], dtype=np.float64)[:2]
            )
        )
        for record in pool
    ]
    z = [
        float(np.asarray(record["pre60_wrist_object_delta_m"], dtype=np.float64)[2])
        for record in pool
    ]
    bounds = {"xy": (min(xy), max(xy)), "z": (min(z), max(z))}
    features = {
        str(record["identity"]): _feature(record, bounds) for record in pool
    }
    pool_distances = np.asarray(
        [float(record["pre60_wrist_object_distance_m"]) for record in pool],
        dtype=np.float64,
    )
    distance_low = float(pool_distances.min())
    distance_high = float(pool_distances.max())
    distance_preference = {
        str(record["identity"]): (
            float(record["pre60_wrist_object_distance_m"]) - distance_low
        )
        / (distance_high - distance_low if distance_high > distance_low else 1.0)
        for record in pool
    }
    selected = [
        max(
            pool,
            key=lambda record: (
                float(record["pre60_wrist_object_distance_m"]),
                str(record["identity"]),
            ),
        )
    ]
    trace: list[dict[str, Any]] = [
        {
            "rank": 1,
            "selected_identity": selected[0]["identity"],
            "reason": "maximum_pre60_wrist_object_distance",
        }
    ]
    while len(selected) < count:
        scored = []
        for record in pool:
            if record in selected:
                continue
            identity = str(record["identity"])
            minimum_feature_distance = min(
                float(
                    np.linalg.norm(
                        features[identity] - features[str(chosen["identity"])]
                    )
                )
                for chosen in selected
            )
            score = (
                0.75 * minimum_feature_distance
                + 0.25 * distance_preference[identity]
            )
            scored.append(
                (
                    score,
                    minimum_feature_distance,
                    distance_preference[identity],
                    identity,
                    record,
                )
            )
        score, feature_distance, radial_score, identity, winner = max(
            scored, key=lambda value: (value[0], value[1], value[2], value[3])
        )
        selected.append(winner)
        trace.append(
            {
                "rank": len(selected),
                "selected_identity": identity,
                "score": score,
                "minimum_feature_distance": feature_distance,
                "distance_preference": radial_score,
            }
        )
    return selected, {
        "candidate_count": len(records),
        "farther_pool_rule": (
            "pre60 frame0 right-wrist/initial-object 3D distance >= pair median"
        ),
        "farther_pool_threshold_m": threshold,
        "farther_pool_count": len(pool),
        "selection_rule": (
            "first maximum distance; then greedy 0.75 normalized relative "
            "XYZ/azimuth coverage + 0.25 normalized distance preference"
        ),
        "trace": trace,
    }


def _candidate_records(
    *,
    pair: str,
    parents_by_action: Path,
    predecoded_manifest: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    object_type, action = pair.split(":", maxsplit=1)
    parents = json.loads(parents_by_action.read_text(encoding="utf-8"))
    mapping = (parents.get("actions") or {}).get(action)
    if not isinstance(mapping, dict) or not mapping:
        raise LookupError(f"accepted-parent manifest omits {pair}")
    predecoded = json.loads(predecoded_manifest.read_text(encoding="utf-8"))
    records = {
        str(record.get("identity")): record
        for record in predecoded.get("valid_records", [])
        if record.get("pair") == pair
    }
    output = []
    rejected_by_gate = []
    dataset_cache: dict[tuple[str, int], Any] = {}
    for identity, descriptor_raw in sorted(mapping.items()):
        record = records.get(identity)
        if record is None:
            continue
        descriptor_path = Path(str(descriptor_raw))
        parent = load_accepted_synthetic_parent(descriptor_path)
        trajectory_path = predecoded_manifest.parent / f"{identity}.pkl"
        if file_sha256(trajectory_path) != record.get("pickle_sha256"):
            raise RuntimeError(f"predecoded trajectory hash changed: {identity}")
        with trajectory_path.open("rb") as stream:
            trajectory = pickle.load(stream)
        if not isinstance(trajectory, ReferenceTrajectory):
            raise TypeError("predecoded candidate is not ReferenceTrajectory")
        if (
            trajectory.identity.identity != identity
            or trajectory.movement_start_step != 60
            or parent.source_identity != identity
            or parent.source_dataset_version != trajectory.identity.dataset_version
            or parent.source_row_index != trajectory.identity.row_index
        ):
            raise RuntimeError(f"parent/pre60 identity binding changed: {identity}")
        acceptance = _current_parent_acceptance(
            parent,
            object_type=object_type,
            dataset_cache=dataset_cache,
        )
        candidate_base = {
            "pair": pair,
            "identity": identity,
            "descriptor_path": str(descriptor_path.resolve()),
            "parent_uuid": parent.parent_row_uuid,
            "parent_acceptance": acceptance,
        }
        if not bool(acceptance["accepted"]):
            rejected_by_gate.append(candidate_base)
            continue
        delta = np.asarray(trajectory.q_ref[0, :3], dtype=np.float64) - np.asarray(
            trajectory.object_pos[0], dtype=np.float64
        )
        output.append(
            {
                **candidate_base,
                "pair": pair,
                "identity": identity,
                "descriptor_path": str(descriptor_path.resolve()),
                "source_row_index": int(trajectory.identity.row_index),
                "pre60_wrist_position_m": np.asarray(
                    trajectory.q_ref[0, :3], dtype=np.float64
                ).tolist(),
                "initial_object_position_m": np.asarray(
                    trajectory.object_pos[0], dtype=np.float64
                ).tolist(),
                "pre60_wrist_object_delta_m": delta.tolist(),
                "pre60_wrist_object_xy_distance_m": float(
                    np.linalg.norm(delta[:2])
                ),
                "pre60_wrist_object_distance_m": float(np.linalg.norm(delta)),
                "pre60_azimuth_deg": math.degrees(
                    math.atan2(float(delta[1]), float(delta[0]))
                ),
            }
        )
    if len(output) < 5:
        raise LookupError(
            f"{pair} has only {len(output)} current-gate-qualified "
            "parent/pre60 candidates"
        )
    return output, rejected_by_gate


def select_from_config(config_path: Path, *, output_dir: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("contract") != CONFIG_CONTRACT:
        raise ValueError("unsupported targeted-parent selection config")
    pairs = config.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("targeted-parent selection config has no pairs")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "selection.json").exists():
        raise FileExistsError(f"targeted selection already exists: {output_dir}")
    selected_parents: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for item in pairs:
        pair = str(item["pair"])
        local_manifest = Path(str(item["predecoded_manifest"])).resolve()
        runtime_manifest = str(
            item.get("runtime_predecoded_manifest") or local_manifest
        )
        candidates, rejected_by_gate = _candidate_records(
            pair=pair,
            parents_by_action=Path(str(item["parents_by_action"])),
            predecoded_manifest=local_manifest,
        )
        selected, method = select_records(candidates, count=5)
        method["current_gate_qualified_count"] = len(candidates)
        method["current_gate_rejected_count"] = len(rejected_by_gate)
        method["qualification_rule"] = (
            "manorl_accepted_synthetic_parent_v3 completion evidence AND "
            "current persisted-float32 final rotation/contact gate"
        )
        pair_slug = pair.replace(":", "_")
        sidecar = output_dir / f"{pair_slug}.json"
        sidecar.write_text(
            json.dumps(
                {
                    "contract": SELECTION_CONTRACT,
                    "pair": pair,
                    "method": method,
                    "selected": selected,
                    "all_qualified_candidates": candidates,
                    "rejected_by_current_gate": rejected_by_gate,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for record in selected:
            selected_parents.append(
                {
                    "pair": pair,
                    "source_identity": record["identity"],
                    "descriptor_path": record["descriptor_path"],
                    "planning_predecoded_manifest": str(local_manifest),
                    "predecoded_manifest": runtime_manifest,
                    "selection_metrics": {
                        key: record[key]
                        for key in (
                            "pre60_wrist_object_delta_m",
                            "pre60_wrist_object_xy_distance_m",
                            "pre60_wrist_object_distance_m",
                            "pre60_azimuth_deg",
                            "parent_acceptance",
                        )
                    },
                }
            )
        summaries[pair] = {
            "sidecar": str(sidecar.resolve()),
            "selected": [record["identity"] for record in selected],
            "method": method,
        }
    result = {
        "contract": SELECTION_CONTRACT,
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "pairs": summaries,
        "parents": selected_parents,
        "parents_per_pair": 5,
    }
    path = output_dir / "selection.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = select_from_config(
        args.config.resolve(), output_dir=args.output_dir.expanduser().resolve()
    )
    values = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(path),
                "pairs": len(values["pairs"]),
                "parents": len(values["parents"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
