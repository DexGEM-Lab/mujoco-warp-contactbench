#!/usr/bin/env python3
"""Build deterministic spatial-coverage slots for prefix-only ManoRL synthesis.

The input selection contains accepted-parent descriptors and the matching pre60
predecoded manifest for each source. Every parent receives 50 Far slots
(5 radius x 5 azimuth x 2 height) and 30 Near slots
(5 empirical distance quantiles x 3 azimuth x 2 height). Each slot owns twelve
fallback episode seeds from the same spatial cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from sim.manorl.approach_prefix import (
    ApproachPrefixConfig,
    RetreatSuffixConfig,
    _identity_seed,
    _map_endpoint_to_initial_object,
    _sample_retreat_endpoint,
    augmentation_stream_seed,
)
from sim.manorl.lance_v2 import file_sha256
from sim.manorl.synthetic_parent import load_accepted_synthetic_parent
from sim.manorl.trajectory import ReferenceTrajectory

PLAN_CONTRACT = "manorl_prefix_only_spatial_coverage_plan_v1"
SELECTION_CONTRACT = "manorl_targeted_parent_selection_v1"
FALLBACKS_PER_SLOT = 12
FAR_SHAPE = (5, 5, 2)
NEAR_SHAPE = (5, 3, 2)
CANDIDATE_POOL_PER_TASK = 6000
SEED_STRIDE_PER_TASK = 100_000


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_source(
    manifest_path: Path, identity: str
) -> tuple[ReferenceTrajectory, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        str(record.get("identity")): record
        for record in manifest.get("valid_records", [])
    }
    record = records.get(identity)
    if record is None:
        raise LookupError(f"predecoded manifest omits {identity}")
    trajectory_path = manifest_path.parent / f"{identity}.pkl"
    if file_sha256(trajectory_path) != record.get("pickle_sha256"):
        raise RuntimeError(f"predecoded trajectory hash changed: {trajectory_path}")
    with trajectory_path.open("rb") as stream:
        trajectory = pickle.load(stream)
    if not isinstance(trajectory, ReferenceTrajectory):
        raise TypeError("predecoded source is not a ReferenceTrajectory")
    if trajectory.identity.identity != identity:
        raise RuntimeError("predecoded source identity changed")
    if trajectory.movement_start_step != 60:
        raise ValueError(f"{identity} is not canonical pre60")
    return trajectory, record


def _far_geometry(
    trajectory: ReferenceTrajectory, episode_seed: int
) -> dict[str, float]:
    config = ApproachPrefixConfig(mode="far")
    rng = np.random.default_rng(
        _identity_seed(episode_seed, trajectory.identity.identity)
    )
    radius = float(
        rng.uniform(config.minimum_xy_radius_m, config.maximum_xy_radius_m)
    )
    angle = float(
        rng.uniform(-config.maximum_xy_offset_deg, config.maximum_xy_offset_deg)
    )
    z_offset = float(
        rng.uniform(config.minimum_z_offset_m, config.maximum_z_offset_m)
    )
    return {
        "radius_m": radius,
        "azimuth_offset_deg": angle,
        "z_offset_m": z_offset,
    }


def _near_geometry(
    trajectory: ReferenceTrajectory,
    *,
    anchor_reference_index: int,
    episode_seed: int,
) -> dict[str, float]:
    approach_seed = augmentation_stream_seed(episode_seed, "near-approach")
    endpoint = _sample_retreat_endpoint(
        trajectory,
        seed=approach_seed,
        anchor_reference_index=anchor_reference_index,
        config=RetreatSuffixConfig(),
    )
    start = _map_endpoint_to_initial_object(
        trajectory, np.asarray(endpoint.end_position_m, dtype=np.float64)
    )
    relative = start - np.asarray(trajectory.object_pos[0], dtype=np.float64)
    return {
        "radius_m": float(np.linalg.norm(relative[:2])),
        "azimuth_offset_deg": float(endpoint.xy_offset_deg),
        "z_offset_m": float(relative[2]),
    }


def _normalized_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = (np.arange(len(values), dtype=np.float64) + 0.5) / len(values)
    return ranks


def _cell_candidates(
    geometries: list[dict[str, float]],
    *,
    shape: tuple[int, int, int],
    empirical: bool,
) -> dict[tuple[int, int, int], list[tuple[float, int]]]:
    matrix = np.asarray(
        [
            (
                item["radius_m"],
                item["azimuth_offset_deg"],
                item["z_offset_m"],
            )
            for item in geometries
        ],
        dtype=np.float64,
    )
    if empirical:
        normalized = np.column_stack(
            [_normalized_ranks(matrix[:, axis]) for axis in range(3)]
        )
    else:
        normalized = np.column_stack(
            (
                (matrix[:, 0] - 0.30) / (1.00 - 0.30),
                (matrix[:, 1] + 30.0) / 60.0,
                (matrix[:, 2] - 0.08) / (0.30 - 0.08),
            )
        )
    normalized = np.clip(normalized, 0.0, np.nextafter(1.0, 0.0))
    buckets: dict[tuple[int, int, int], list[tuple[float, int]]] = {
        (a, b, c): []
        for a in range(shape[0])
        for b in range(shape[1])
        for c in range(shape[2])
    }
    for candidate_index, point in enumerate(normalized):
        cell = tuple(int(point[axis] * shape[axis]) for axis in range(3))
        center = np.asarray(
            [(cell[axis] + 0.5) / shape[axis] for axis in range(3)],
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(point - center))
        buckets[cell].append((distance, candidate_index))
    for cell, candidates in buckets.items():
        candidates.sort(key=lambda item: (item[0], item[1]))
        if len(candidates) < FALLBACKS_PER_SLOT:
            raise RuntimeError(
                f"coverage cell {cell} has {len(candidates)} candidates; "
                f"need {FALLBACKS_PER_SLOT}"
            )
    return buckets


def _build_task(
    *,
    task_index: int,
    pair: str,
    identity: str,
    descriptor_path: Path,
    predecoded_manifest: Path,
    mode: str,
    seed_base: int,
    candidate_pool: int,
) -> dict[str, Any]:
    parent = load_accepted_synthetic_parent(descriptor_path)
    if parent.source_identity != identity:
        raise ValueError("selected parent descriptor identity changed")
    trajectory, record = _load_source(predecoded_manifest, identity)
    source_path_alias = parent.source_dataset_path != str(
        trajectory.identity.dataset_path
    )
    if parent.source_dataset_version != trajectory.identity.dataset_version:
        raise ValueError("parent and predecoded source dataset versions differ")
    if parent.source_row_index != trajectory.identity.row_index:
        raise ValueError("parent and predecoded source row identities differ")
    anchor = int(trajectory.movement_end_step) + parent.retreat_anchor_offset_frames
    if mode == "near":
        if (
            not 0 <= anchor < len(trajectory.q_ref)
            or int(trajectory.source_indices[anchor])
            != int(parent.retreat_anchor_source_frame_index)
        ):
            raise ValueError("Near movement_end+15 anchor changed")
    task_seed_base = seed_base + task_index * SEED_STRIDE_PER_TASK
    if task_seed_base + candidate_pool >= 2**32:
        raise ValueError("coverage seed range exceeds numpy seed ABI")
    seeds = [task_seed_base + index for index in range(candidate_pool)]
    geometries = [
        (
            _far_geometry(trajectory, seed)
            if mode == "far"
            else _near_geometry(
                trajectory, anchor_reference_index=anchor, episode_seed=seed
            )
        )
        for seed in seeds
    ]
    shape = FAR_SHAPE if mode == "far" else NEAR_SHAPE
    buckets = _cell_candidates(
        geometries, shape=shape, empirical=(mode == "near")
    )
    slots = []
    for slot_index, cell in enumerate(sorted(buckets)):
        candidates = []
        for fallback_rank, (_, candidate_index) in enumerate(
            buckets[cell][:FALLBACKS_PER_SLOT]
        ):
            candidates.append(
                {
                    "fallback_rank": fallback_rank,
                    "episode_seed": seeds[candidate_index],
                    "sampled_start": geometries[candidate_index],
                }
            )
        slots.append(
            {
                "slot_index": slot_index,
                "cell": {
                    "distance_bin": cell[0],
                    "azimuth_bin": cell[1],
                    "height_bin": cell[2],
                    "shape": list(shape),
                    "normalization": (
                        "empirical_per_parent_mode_quantiles"
                        if mode == "near"
                        else "fixed_far_bounds_0p30_1p00m_pm30deg_0p08_0p30m"
                    ),
                },
                "candidates": candidates,
            }
        )
    return {
        "task_index": task_index,
        "pair": pair,
        "source_identity": identity,
        "parent_uuid": parent.parent_row_uuid,
        "descriptor_path": str(descriptor_path.resolve()),
        "descriptor_sha256": file_sha256(descriptor_path),
        "parent_object_init_xy_offset_m": list(parent.object_init_xy_offset_m),
        "formal_random_object_xy_offset_range_m": 0.0,
        "predecoded_manifest": str(predecoded_manifest.resolve()),
        "predecoded_record_sha256": record.get("pickle_sha256"),
        "source_dataset_path_alias": (
            {
                "parent": parent.source_dataset_path,
                "predecoded": str(trajectory.identity.dataset_path),
                "binding": "same_version_and_source_row_identity",
            }
            if source_path_alias
            else None
        ),
        "mode": mode,
        "target_rows": len(slots),
        "fallbacks_per_slot": FALLBACKS_PER_SLOT,
        "candidate_pool": candidate_pool,
        "seed_range": [seeds[0], seeds[-1]],
        "slots": slots,
    }


def build_plan(
    selection_path: Path,
    *,
    output: Path,
    seed_base: int = 10_000_000,
    candidate_pool: int = CANDIDATE_POOL_PER_TASK,
) -> Path:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("contract") != SELECTION_CONTRACT:
        raise ValueError("unsupported targeted-parent selection contract")
    parents = selection.get("parents")
    if not isinstance(parents, list) or not parents:
        raise ValueError("targeted-parent selection has no parents")
    ordered = sorted(
        parents,
        key=lambda item: (str(item["pair"]), str(item["source_identity"])),
    )
    tasks = []
    for parent_index, item in enumerate(ordered):
        for mode_offset, mode in enumerate(("far", "near")):
            tasks.append(
                _build_task(
                    task_index=parent_index * 2 + mode_offset,
                    pair=str(item["pair"]),
                    identity=str(item["source_identity"]),
                    descriptor_path=Path(str(item["descriptor_path"])),
                    predecoded_manifest=Path(
                        str(
                            item.get("planning_predecoded_manifest")
                            or item["predecoded_manifest"]
                        )
                    ),
                    mode=mode,
                    seed_base=seed_base,
                    candidate_pool=candidate_pool,
                )
            )
            tasks[-1]["predecoded_manifest"] = str(
                item["predecoded_manifest"]
            )
            tasks[-1]["planning_predecoded_manifest"] = str(
                item.get("planning_predecoded_manifest")
                or item["predecoded_manifest"]
            )
    all_seeds = [
        candidate["episode_seed"]
        for task in tasks
        for slot in task["slots"]
        for candidate in slot["candidates"]
    ]
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("coverage plan repeats an episode seed")
    payload: dict[str, Any] = {
        "contract": PLAN_CONTRACT,
        "selection": str(selection_path.resolve()),
        "selection_sha256": file_sha256(selection_path),
        "seed_base": seed_base,
        "seed_stride_per_task": SEED_STRIDE_PER_TASK,
        "candidate_pool_per_task": candidate_pool,
        "fallbacks_per_slot": FALLBACKS_PER_SLOT,
        "far_grid_shape": list(FAR_SHAPE),
        "near_grid_shape": list(NEAR_SHAPE),
        "parents": len(ordered),
        "tasks": tasks,
        "target_rows": sum(int(task["target_rows"]) for task in tasks),
    }
    payload["plan_digest"] = _canonical_digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"coverage plan exists: {output}")
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=10_000_000)
    parser.add_argument(
        "--candidate-pool", type=int, default=CANDIDATE_POOL_PER_TASK
    )
    args = parser.parse_args(argv)
    if args.seed_base < 0:
        parser.error("--seed-base must be non-negative")
    if args.candidate_pool < FALLBACKS_PER_SLOT * max(np.prod(FAR_SHAPE), np.prod(NEAR_SHAPE)):
        parser.error("--candidate-pool is too small for every fallback slot")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = build_plan(
        args.selection.resolve(),
        output=args.output.expanduser().resolve(),
        seed_base=args.seed_base,
        candidate_pool=args.candidate_pool,
    )
    values = json.loads(output.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(output),
                "plan_digest": values["plan_digest"],
                "parents": values["parents"],
                "tasks": len(values["tasks"]),
                "target_rows": values["target_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
