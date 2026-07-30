#!/usr/bin/env python3
"""Validate corrected v2 Lance rows in isolated decoder subprocesses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.lance_v2 import FORCE_DIRECTION_CONTRACT, SYNTHETIC_LANCE_V2_CONTRACT


def _schema_metadata(dataset: Any) -> dict[str, str]:
    return {
        key.decode(): value.decode()
        for key, value in (dataset.schema.metadata or {}).items()
    }


def validate_row(path: Path, row_index: int) -> dict[str, Any]:
    import lance

    dataset = lance.dataset(str(path))
    rows = dataset.take([row_index]).to_pylist()
    if len(rows) != 1:
        raise RuntimeError(f"Lance row {row_index} did not decode exactly once")
    row = rows[0]
    metadata = row["trajectory_metadata"]
    total_frames = int(metadata["total_frames"])
    rollout = row["rollout"]
    hand = row["hands"][0]
    obj = row["objects"][0]
    if metadata["data_fps"] != 200 or len(row["timestamp"]) != total_frames:
        raise ValueError(f"row {row_index} has invalid 200 Hz frame metadata")
    timestamp = np.asarray(row["timestamp"], dtype=np.float64)
    if not np.allclose(np.diff(timestamp), 0.005, rtol=0.0, atol=1e-12):
        raise ValueError(f"row {row_index} timestamp spacing differs from 0.005 s")
    expected_shapes = {
        "urdf_dof": (total_frames, 28),
        "urdf_dof_target": (total_frames, 28),
        "mano_joint_pos": (total_frames, 21, 3),
    }
    for name, shape in expected_shapes.items():
        values = np.asarray(hand[name])
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"row {row_index} hands.{name} has invalid shape/values")
    transitions = total_frames - 1
    rollout_shapes = {
        "observation_t": (transitions, 480),
        "next_observation": (transitions, 480),
        "policy_mean_action": (transitions, 28),
        "processed_action": (transitions, 28),
        "controller_target": (transitions, 28),
        "cumulative_joint_residual": (transitions, 22),
    }
    for name, shape in rollout_shapes.items():
        values = np.asarray(rollout[name])
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"row {row_index} rollout.{name} has invalid shape/values")
    if int(rollout["transition_count"]) != transitions:
        raise ValueError(f"row {row_index} transition count differs from T-1")
    if (
        sum(bool(value) for value in rollout["terminated"]) != 1
        or not bool(rollout["terminated"][-1])
        or int(rollout["termination_reason_code"][-1]) != 1
        or any(int(value) != 0 for value in rollout["termination_reason_code"][:-1])
    ):
        raise ValueError(f"row {row_index} does not end at one complete episode boundary")
    if row["provenance"]["force_contract"] != FORCE_DIRECTION_CONTRACT:
        raise ValueError(f"row {row_index} force contract changed")

    contact_frames = 0
    contact_pairs = 0
    max_pos_object_error = 0.0
    max_force_sum_error = 0.0
    max_object_force_error = 0.0
    max_joint_force_norm_error = 0.0
    for frame_index, entries in enumerate(row["contact"]):
        if entries:
            contact_frames += 1
        object_rotation = Rotation.from_rotvec(obj["rot_aa"][frame_index]).as_matrix()
        object_position = np.asarray(obj["pos"][frame_index], dtype=np.float64)
        for entry in entries:
            pairs = entry["contact_pairs"]
            contact_pairs += len(pairs)
            total_world = np.asarray(entry["total_force_world"], dtype=np.float64)
            pair_sum = np.sum(
                np.asarray([pair["force_normal"] for pair in pairs], dtype=np.float64),
                axis=0,
            )
            max_force_sum_error = max(
                max_force_sum_error, float(np.linalg.norm(total_world - pair_sum))
            )
            max_object_force_error = max(
                max_object_force_error,
                float(
                    np.linalg.norm(
                        np.asarray(entry["total_force_object"], dtype=np.float64)
                        - object_rotation.T @ total_world
                    )
                ),
            )
            max_joint_force_norm_error = max(
                max_joint_force_norm_error,
                abs(
                    float(np.linalg.norm(total_world))
                    - float(np.linalg.norm(entry["total_force_joint"]))
                ),
            )
            for pair in pairs:
                reconstructed = object_position + object_rotation @ np.asarray(
                    pair["pos_object"], dtype=np.float64
                )
                max_pos_object_error = max(
                    max_pos_object_error,
                    float(
                        np.linalg.norm(
                            np.asarray(pair["pos_world"], dtype=np.float64)
                            - reconstructed
                        )
                    ),
                )
    result = {
        "row_index": row_index,
        "uuid": row["index"]["uuid"],
        "source_identity": row["provenance"]["source_identity"],
        "source_row_index": int(row["provenance"]["row_index"]),
        "frames": total_frames,
        "transitions": transitions,
        "contact_frames": contact_frames,
        "contact_pairs": contact_pairs,
        "nonzero_policy_action_values": int(
            np.count_nonzero(np.abs(np.asarray(rollout["policy_mean_action"])) > 1e-8)
        ),
        "reward_sum": float(metadata["train_info"]["reward_value"]),
        "checkpoint_sha256": row["provenance"]["checkpoint_sha256"],
        "max_object_position_reconstruction_m": max_pos_object_error,
        "max_total_force_sum_error_N": max_force_sum_error,
        "max_object_force_rotation_error_N": max_object_force_error,
        "max_joint_force_norm_error_N": max_joint_force_norm_error,
    }
    if row_index == 0:
        checkpoint = json.loads(row["provenance"]["checkpoint_metadata_json"])
        result["checkpoint_environment_contract"] = checkpoint.get("environment_contract")
        result["checkpoint_controlled_hand_sides"] = (
            checkpoint.get("runtime_config", {})
            .get("environment", {})
            .get("controlled_hand_sides")
        )
        result["checkpoint_reference_following_hand_sides"] = (
            checkpoint.get("runtime_config", {})
            .get("environment", {})
            .get("reference_following_hand_sides")
        )
    return result


def validate_dataset(
    path: Path, output: Path | None = None, *, max_attempts: int = 5
) -> dict[str, Any]:
    import lance

    dataset = lance.dataset(str(path))
    metadata = _schema_metadata(dataset)
    if metadata.get("schema_version") != SYNTHETIC_LANCE_V2_CONTRACT:
        raise ValueError("dataset schema_version is not the corrected v2 contract")
    if metadata.get("force_contract") != FORCE_DIRECTION_CONTRACT:
        raise ValueError("dataset force contract is not normal-only scale 1.0")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    row_count = int(dataset.count_rows())
    rows = []
    for row_index in range(row_count):
        failures: list[tuple[int, str]] = []
        for attempt in range(1, max_attempts + 1):
            child = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(path),
                    "--row-index",
                    str(row_index),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if child.returncode == 0:
                decoded = json.loads(child.stdout)
                decoded["decoder_attempts"] = attempt
                rows.append(decoded)
                break
            failures.append((child.returncode, child.stderr.strip()))
            if child.returncode >= 0 and child.returncode not in {134, 139}:
                break
        else:
            child = None
        if len(rows) != row_index + 1:
            code, error = failures[-1]
            raise RuntimeError(
                f"isolated Lance validation failed for row {row_index} after "
                f"{len(failures)} attempt(s), final exit {code}: {error}"
            )
    identities = [row["source_identity"] for row in rows]
    uuids = [row["uuid"] for row in rows]
    source_rows = [row["source_row_index"] for row in rows]
    if len(set(identities)) != row_count or len(set(uuids)) != row_count or len(set(source_rows)) != row_count:
        raise ValueError("dataset contains duplicate source identities, UUIDs, or row indices")
    checkpoint_hashes = {row["checkpoint_sha256"] for row in rows}
    if len(checkpoint_hashes) != 1:
        raise ValueError("dataset rows do not share one checkpoint SHA256")
    summary = {
        "schema": SYNTHETIC_LANCE_V2_CONTRACT,
        "schema_metadata": metadata,
        "rows": row_count,
        "unique_identities": len(set(identities)),
        "source_row_range": [min(source_rows), max(source_rows)] if rows else None,
        "frame_count_min_max_sum": [
            min(row["frames"] for row in rows),
            max(row["frames"] for row in rows),
            sum(row["frames"] for row in rows),
        ] if rows else [0, 0, 0],
        "transition_count_sum": sum(row["transitions"] for row in rows),
        "contact_frames": sum(row["contact_frames"] for row in rows),
        "contact_pairs": sum(row["contact_pairs"] for row in rows),
        "nonzero_policy_action_values": sum(
            row["nonzero_policy_action_values"] for row in rows
        ),
        "isolated_decoder_attempts": sum(row["decoder_attempts"] for row in rows),
        "retried_row_count": sum(row["decoder_attempts"] > 1 for row in rows),
        "max_decoder_attempts_for_one_row": max(
            (row["decoder_attempts"] for row in rows), default=0
        ),
        "reward_sum_all_rows": sum(row["reward_sum"] for row in rows),
        "checkpoint_sha256": next(iter(checkpoint_hashes)) if rows else None,
        "checkpoint_environment_contract": (
            rows[0].get("checkpoint_environment_contract") if rows else None
        ),
        "checkpoint_controlled_hand_sides": (
            rows[0].get("checkpoint_controlled_hand_sides") if rows else None
        ),
        "checkpoint_reference_following_hand_sides": (
            rows[0].get("checkpoint_reference_following_hand_sides") if rows else None
        ),
        "max_object_position_reconstruction_m": max(
            (row["max_object_position_reconstruction_m"] for row in rows), default=0.0
        ),
        "max_total_force_sum_error_N": max(
            (row["max_total_force_sum_error_N"] for row in rows), default=0.0
        ),
        "max_object_force_rotation_error_N": max(
            (row["max_object_force_rotation_error_N"] for row in rows), default=0.0
        ),
        "max_joint_force_norm_error_N": max(
            (row["max_joint_force_norm_error_N"] for row in rows), default=0.0
        ),
    }
    target = output or path.parent / f"{path.name}.validation.json"
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**summary, "validation_output": str(target)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--row-index", type=int, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.row_index is not None:
        print(json.dumps(validate_row(args.dataset, args.row_index), sort_keys=True))
        return 0
    print(
        json.dumps(
            validate_dataset(
                args.dataset, args.output, max_attempts=args.max_attempts
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
