"""Measure zero-residual physical replay success for every row in an RL MTP."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


def _pair_from_identity(identity: str) -> str:
    fields = identity.rsplit("_", 2)
    if len(fields) != 3 or not fields[2].isdigit():
        raise ValueError(f"invalid trajectory identity: {identity!r}")
    return f"{fields[0]}:{fields[1]}"


def _selected_catalog_rows(catalog: Any, pair_names: tuple[str, ...]) -> tuple[Any, ...]:
    requested = set(pair_names)
    selected = tuple(
        trajectory
        for trajectory in catalog.trajectories
        if _pair_from_identity(trajectory.identity.identity) in requested
    )
    present = {_pair_from_identity(item.identity.identity) for item in selected}
    missing = requested - present
    if missing:
        raise LookupError("package lacks requested pair(s): " + ", ".join(sorted(missing)))
    return selected


def run_replay(
    *,
    package_path: Path,
    pair_names: tuple[str, ...],
    output_path: Path,
    device: str,
    contacts_per_world: int,
    constraint_capacity: int,
    expected_contact_mode: str,
    joint_scale_multiplier: float,
    joint_max_offset_multiplier: float,
) -> dict[str, object]:
    from sim.manorl.abi import ResidualActionConfig, TARGET_MAX_DEVIATION_DISTANCE
    from sim.manorl import assets
    from sim.manorl.environment import (
        EnvironmentConfig,
        MujocoManoEnvironment,
        recommended_warp_contact_capacity,
    )
    from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv
    from sim.manorl.observations import SOURCE_ALIGNED_COMPATIBILITY
    from sim.manorl.trajectory import (
        ObjectActionPair,
        RL_EPISODE_REFERENCE_CONTRACT,
        TrajectoryBatch,
    )
    from sim.manorl.trajectory_package import load_trajectory_package

    catalog = load_trajectory_package(package_path)
    manifest = catalog.manifest
    package_asset_profile = manifest.get("source_hand_asset_profile")
    active_asset_profile = assets.asset_provenance()
    if package_asset_profile != active_asset_profile:
        raise ValueError(
            "trajectory package fixed-hand asset profile differs from active manifest"
        )
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("trajectory package selection contract is missing")
    expected_selection = {
        "source_reference_contract": RL_EPISODE_REFERENCE_CONTRACT,
        "hand_side": "right",
        "pre_padding": 0,
        "post_padding": 0,
        "reference_fps": 120,
        "control_fps": 120,
    }
    mismatches = {
        name: (selection.get(name), expected)
        for name, expected in expected_selection.items()
        if selection.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"RL replay package contract mismatch: {mismatches}")

    selected = _selected_catalog_rows(catalog, pair_names)
    resolved_pairs = tuple(
        ObjectActionPair(*name.split(":", maxsplit=1)) for name in pair_names
    )
    batch = TrajectoryBatch(
        selected,
        resolved_pairs=resolved_pairs,
        selection_mode="pairs",
        trajectory_package=catalog.checkpoint_metadata,
    )
    num_envs = batch.num_envs
    residual_action = ResidualActionConfig(
        joint_scale_multiplier=joint_scale_multiplier,
        joint_max_offset_multiplier=joint_max_offset_multiplier,
    )
    contact_capacity = max(
        recommended_warp_contact_capacity(num_envs, batch.hand_sides),
        contacts_per_world * num_envs,
    )
    environment = MujocoManoEnvironment(
        batch,
        EnvironmentConfig(
            device=device,
            num_envs=num_envs,
            residual_enabled=True,
            residual_action=residual_action,
            expected_contact_mode=expected_contact_mode,
            compatibility=replace(
                SOURCE_ALIGNED_COMPATIBILITY,
                movement_pre_padding=0,
            ),
            max_deviation_distance=TARGET_MAX_DEVIATION_DISTANCE,
            contact_capacity=contact_capacity,
            constraint_capacity=constraint_capacity,
            reference_fps=120,
            control_fps=120,
            post_padding=0,
            hand_side="right",
        ),
    )
    gym_env = ManoGymnasiumVectorEnv(environment)
    gym_env.reset(seed=42)
    actions = np.zeros((num_envs, environment.action_dim), dtype=np.float64)
    active = np.ones(num_envs, dtype=bool)
    success = np.zeros(num_envs, dtype=bool)
    failure = np.zeros(num_envs, dtype=bool)
    reason_codes = np.zeros(num_envs, dtype=np.int32)
    completion_calls = np.full(num_envs, -1, dtype=np.int64)
    final_distances = np.zeros(num_envs, dtype=np.float64)
    max_distances = np.zeros(num_envs, dtype=np.float64)
    started = time.monotonic()
    max_calls = int(np.max(environment.trajectory_lengths)) + 2
    for call in range(max_calls):
        _, _, terminated, truncated, info = gym_env.step(actions)
        done = np.asarray(terminated | truncated, dtype=bool)
        active_before = active.copy()
        physical = environment.last_physical
        if physical is None:
            raise RuntimeError("physical replay omitted state diagnostics")
        indices = environment.trajectory_steps
        target = environment.reference_object_pos[np.arange(num_envs), indices]
        distances = np.linalg.norm(physical.object_position - target, axis=1)
        max_distances[active_before] = np.maximum(
            max_distances[active_before], distances[active_before]
        )
        newly_done = active_before & done
        if np.any(newly_done):
            step_success = np.asarray(info["termination_success"], dtype=bool)
            step_failure = np.asarray(info["termination_failure"], dtype=bool)
            step_reasons = np.asarray(info["termination_reason_code"], dtype=np.int32)
            success[newly_done] = step_success[newly_done]
            failure[newly_done] = step_failure[newly_done]
            reason_codes[newly_done] = step_reasons[newly_done]
            completion_calls[newly_done] = call + 1
            final_distances[newly_done] = distances[newly_done]
            active[newly_done] = False
        if not np.any(active):
            break
    if np.any(active):
        remaining = np.flatnonzero(active).tolist()
        raise RuntimeError(
            "zero-residual replay did not terminate every row; "
            f"remaining env_ids={remaining[:32]}"
        )
    if np.any(success & failure) or np.any((success | failure) == 0):
        raise RuntimeError("replay terminal classification is incomplete")

    source_records = {
        (int(item["row_index"]), str(item["identity"])): item
        for item in manifest["source_catalog"]["candidates"]
        if isinstance(item, Mapping)
    }
    rows: list[dict[str, object]] = []
    for env_id, trajectory in enumerate(selected):
        identity = trajectory.identity
        source_record = source_records[(identity.row_index, identity.identity)]
        rows.append(
            {
                "env_id": env_id,
                "identity": identity.identity,
                "row_index": identity.row_index,
                "uuid": identity.uuid,
                "pair": _pair_from_identity(identity.identity),
                "success": bool(success[env_id]),
                "failure": bool(failure[env_id]),
                "termination_reason_code": int(reason_codes[env_id]),
                "completion_calls": int(completion_calls[env_id]),
                "final_object_target_distance_m": float(final_distances[env_id]),
                "max_object_target_distance_m": float(max_distances[env_id]),
                "source_provenance": source_record.get("provenance"),
            }
        )

    by_pair: dict[str, dict[str, object]] = {}
    for pair in pair_names:
        pair_rows = [item for item in rows if item["pair"] == pair]
        pair_successes = sum(bool(item["success"]) for item in pair_rows)
        by_pair[pair] = {
            "rows": len(pair_rows),
            "successes": pair_successes,
            "failures": len(pair_rows) - pair_successes,
            "success_rate": pair_successes / len(pair_rows),
            "max_object_target_distance_m": max(
                float(item["max_object_target_distance_m"]) for item in pair_rows
            ),
        }
    success_count = int(success.sum())
    result: dict[str, object] = {
        "schema": "manorl.rl_episode_zero_replay.v1",
        "trajectory_package": str(package_path),
        "package_digest": catalog.package_digest,
        "catalog_digest": catalog.catalog_digest,
        "asset_profile": manifest.get("source_hand_asset_profile"),
        "fixed_hand_operator": "cheyingtong",
        "reference_fps": 120,
        "control_fps": 120,
        "physics_fps": 480,
        "zero_residual_action": True,
        "joint_scale_multiplier": joint_scale_multiplier,
        "joint_max_offset_multiplier": joint_max_offset_multiplier,
        "num_envs": num_envs,
        "elapsed_seconds": time.monotonic() - started,
        "success_count": success_count,
        "failure_count": num_envs - success_count,
        "success_rate": success_count / num_envs,
        "by_pair": by_pair,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to replace replay result: {output_path}")
    temporary = output_path.with_name(output_path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-package", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pairs",
        default=(
            "mayonnaisebottle:01,mayonnaisebottle:02,mayonnaisebottle:03,"
            "mayonnaisebottle:04,mayonnaisebottle:05,mayonnaisebottle:08"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--contacts-per-world", type=int, default=128)
    parser.add_argument("--constraint-capacity", type=int, default=512)
    parser.add_argument(
        "--expected-contact-mode",
        choices=("source_mapping", "five_fingertips"),
        default="five_fingertips",
    )
    parser.add_argument("--joint-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--joint-max-offset-multiplier", type=float, default=1.0)
    parser.add_argument("--fixed-hand-operator", default="cheyingtong")
    args = parser.parse_args(argv)
    if args.contacts_per_world < 1 or args.constraint_capacity < 1:
        parser.error("contact and constraint capacities must be positive")
    if args.joint_scale_multiplier <= 0 or args.joint_max_offset_multiplier <= 0:
        parser.error("joint multipliers must be positive")
    package = args.trajectory_package.expanduser().resolve()
    asset_manifest = args.asset_manifest.expanduser().resolve()
    if not package.is_dir():
        parser.error("--trajectory-package must name a package directory")
    if not asset_manifest.is_file():
        parser.error("--asset-manifest must name a file")
    if "sim.manorl.assets" in sys.modules:
        parser.error("asset module loaded before fixed hand manifest selection")
    os.environ["MANORL_ASSET_MANIFEST"] = str(asset_manifest)
    from sim.manorl import assets
    from sim.manorl.trajectory import parse_trajectory_selector

    if assets.MANO_OPERATOR != args.fixed_hand_operator:
        parser.error(
            f"asset manifest hand_operator {assets.MANO_OPERATOR!r} != "
            f"required {args.fixed_hand_operator!r}"
        )
    requested = parse_trajectory_selector(args.pairs)
    if requested is None:
        parser.error("--pairs must be an explicit object:action list")
    pair_names = tuple(pair.canonical for pair in requested)
    result = run_replay(
        package_path=package,
        pair_names=pair_names,
        output_path=args.output.expanduser().resolve(),
        device=args.device,
        contacts_per_world=args.contacts_per_world,
        constraint_capacity=args.constraint_capacity,
        expected_contact_mode=args.expected_contact_mode,
        joint_scale_multiplier=args.joint_scale_multiplier,
        joint_max_offset_multiplier=args.joint_max_offset_multiplier,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "num_envs": result["num_envs"],
                "success_count": result["success_count"],
                "failure_count": result["failure_count"],
                "success_rate": result["success_rate"],
                "by_pair": result["by_pair"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
