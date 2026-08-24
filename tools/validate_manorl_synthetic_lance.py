#!/usr/bin/env python3
"""Validate full v2.2/v2.3 or compact replay Lance rows in subprocesses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.approach_prefix import (
    APPROACH_PREFIX_ONLY_PRODUCTION_CONTRACT,
    PREFIX_ONLY_AUGMENTATION_IDENTITY_CONTRACT,
)
from sim.manorl.contracts import simulation_clock
from sim.manorl.lance_v2 import (
    FORCE_DIRECTION_CONTRACT,
    MANO_GLOBAL_FRAME_CONTRACT,
    SYNTHETIC_LANCE_COMPACT_CONTRACTS,
    SYNTHETIC_LANCE_COMPACT_V1_CONTRACT,
    SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT,
    SYNTHETIC_LANCE_CONTRACT,
    SYNTHETIC_LANCE_SOURCE_CONTRACTS,
    SYNTHETIC_LANCE_V22_CONTRACT,
)
from sim.manorl.rewards import PPO_REWARD_CONTRACT_ID, REWARD_CONTRACT_ID
from sim.manorl.synthesis_acceptance import (
    SYNTHESIS_ACCEPTANCE_CONTRACT,
    count_hand_object_contact_frames,
    evaluate_synthesis_acceptance,
    synthesis_acceptance_manifest,
)
from sim.manorl.target_replay import target_replay_source_from_row


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
    data_fps = metadata["data_fps"]
    if (
        not isinstance(data_fps, int)
        or isinstance(data_fps, bool)
        or data_fps not in (100, 120, 200)
        or len(row["timestamp"]) != total_frames
    ):
        raise ValueError(f"row {row_index} has invalid frame-clock metadata")
    expected_timestep = simulation_clock(data_fps).control_timestep
    timestamp = np.asarray(row["timestamp"], dtype=np.float64)
    if not np.allclose(np.diff(timestamp), expected_timestep, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"row {row_index} timestamp spacing differs from {expected_timestep} s"
        )
    expected_shapes = {
        "urdf_dof": (total_frames, 28),
        "urdf_dof_target": (total_frames, 28),
        "mano_joint_pos": (total_frames, 21, 3),
    }
    for name, shape in expected_shapes.items():
        values = np.asarray(hand[name])
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"row {row_index} hands.{name} has invalid shape/values")
    urdf_dof = np.asarray(hand["urdf_dof"], dtype=np.float64)
    mano_global_pos = np.asarray(hand["mano_global_pos"], dtype=np.float64)
    mano_global_rot = Rotation.from_rotvec(
        np.asarray(hand["mano_global_rot_aa"], dtype=np.float64)
    )
    expected_global_rot = Rotation.from_euler("XYZ", urdf_dof[:, 3:6])
    global_position_error = float(
        np.max(np.linalg.norm(mano_global_pos - urdf_dof[:, :3], axis=1))
    )
    global_rotation_error = float(
        np.max((mano_global_rot.inv() * expected_global_rot).magnitude())
    )
    if global_position_error > 1e-12 or global_rotation_error > 1e-6:
        raise ValueError(
            f"row {row_index} MANO global pose is not the URDF floating root"
        )
    mano_pose = np.asarray(hand["mano_hand_pose"], dtype=np.float64)
    if mano_pose.shape != (total_frames, 48) or not np.all(np.isfinite(mano_pose)):
        raise ValueError(f"row {row_index} MANO hand pose has invalid shape/values")
    hand_shapes = np.asarray(metadata["mano_hand_shapes"], dtype=np.float64)
    if metadata["hand_names"] != ["right"] or hand_shapes.shape != (1, 10):
        raise ValueError(f"row {row_index} must contain one active right-hand shape")
    provenance = row["provenance"]
    dataset_contract = _schema_metadata(dataset).get("schema_version")
    if dataset_contract not in (SYNTHETIC_LANCE_V22_CONTRACT, SYNTHETIC_LANCE_CONTRACT):
        raise ValueError(f"row {row_index} belongs to an unsupported schema contract")
    if provenance["contract"] != dataset_contract:
        raise ValueError(f"row {row_index} provenance contract differs from its schema")
    if dataset_contract == SYNTHETIC_LANCE_V22_CONTRACT:
        if data_fps != 200:
            raise ValueError(f"legacy v2.2 row {row_index} must use 200 Hz")
    else:
        clock = simulation_clock(data_fps)
        reference_fps = provenance["reference_fps"]
        if reference_fps not in (None, 100, 120):
            raise ValueError(f"row {row_index} provenance reference_fps is invalid")
        if data_fps in (100, 120) and reference_fps != data_fps:
            raise ValueError(
                f"row {row_index} public reference/control clocks are not coupled"
            )
        expected_clock_fields = {
            "control_fps": clock.policy_fps,
            "physics_fps": clock.physics_fps,
            "physics_substeps_per_control": clock.physics_substeps_per_control,
        }
        for field, expected in expected_clock_fields.items():
            if provenance[field] != expected:
                raise ValueError(f"row {row_index} provenance {field} is inconsistent")
        for field, expected in (
            ("control_timestep_seconds", clock.control_timestep),
            ("physics_timestep_seconds", clock.physics_timestep),
        ):
            if not np.isclose(float(provenance[field]), expected, rtol=0.0, atol=1e-15):
                raise ValueError(f"row {row_index} provenance {field} is inconsistent")
    source_dataset = lance.dataset(
        provenance["dataset_path"], version=int(provenance["dataset_version"])
    )
    source_row = source_dataset.take(
        [int(provenance["row_index"])], columns=["trajectory_metadata"]
    ).to_pylist()[0]["trajectory_metadata"]
    source_hand_names = source_row.get("hand_names") or []
    source_hand_shapes = source_row.get("mano_hand_shapes") or []
    if "right" not in source_hand_names:
        raise ValueError(f"source row for export row {row_index} has no right hand")
    source_right_index = source_hand_names.index("right")
    if source_right_index >= len(source_hand_shapes):
        raise ValueError(f"source row for export row {row_index} omits right shape")
    right_shape_error = float(
        np.max(
            np.abs(
                hand_shapes[0]
                - np.asarray(source_hand_shapes[source_right_index], dtype=np.float64)
            )
        )
    )
    if right_shape_error > 1e-7:
        raise ValueError(f"row {row_index} right MANO shape differs from raw Lance")
    if row["hands"][1]["hand_name"] is not None:
        raise ValueError(f"row {row_index} left fixed hand slot must remain empty")
    transitions = total_frames - 1
    rollout_shapes = {
        "observation_t": (transitions, 480),
        "next_observation": (transitions, 480),
        "policy_mean_action": (transitions, 28),
        "processed_action": (transitions, 28),
        "controller_target": (transitions, 28),
        "cumulative_joint_residual": (transitions, 22),
        "raw_contact_reward": (transitions,),
        "contact_reward": (transitions,),
    }
    for name, shape in rollout_shapes.items():
        values = np.asarray(rollout[name])
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"row {row_index} rollout.{name} has invalid shape/values")
    if int(rollout["transition_count"]) != transitions:
        raise ValueError(f"row {row_index} transition count differs from T-1")
    raw_contact_reward = np.asarray(rollout["raw_contact_reward"], dtype=np.float64)
    contact_reward = np.asarray(rollout["contact_reward"], dtype=np.float64)
    if (
        np.any(raw_contact_reward < -1e-7)
        or np.any(raw_contact_reward > 0.4 + 1e-6)
        or np.any(contact_reward < -1.2 - 1e-6)
        or np.any(contact_reward > 0.4 + 1e-6)
    ):
        raise ValueError(
            f"row {row_index} contact rewards violate the synthetic bounds"
        )
    movement = metadata["trajectory_info"]["object_move"][0]
    contact_start = int(movement["start_frame"])
    contact_end = int(movement["end_frame"])
    negative_indices = np.flatnonzero(contact_reward < 0.0)
    positive_indices = np.flatnonzero(contact_reward > 0.0)
    if (
        np.any(negative_indices <= contact_end + 10)
        or np.any(positive_indices < contact_start)
        or np.any(positive_indices > contact_end)
        or not np.allclose(contact_reward[negative_indices], -1.2, rtol=0.0, atol=1e-6)
        or not np.allclose(
            contact_reward[contact_end + 1 : min(contact_end + 11, transitions)],
            0.0,
            rtol=0.0,
            atol=1e-7,
        )
    ):
        raise ValueError(f"row {row_index} violates the late-contact timing contract")
    negative_contact_steps = int(len(negative_indices))
    if (
        sum(bool(value) for value in rollout["terminated"]) != 1
        or not bool(rollout["terminated"][-1])
        or int(rollout["termination_reason_code"][-1]) != 1
        or any(int(value) != 0 for value in rollout["termination_reason_code"][:-1])
    ):
        raise ValueError(
            f"row {row_index} does not end at one complete episode boundary"
        )
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
    hand_object_contact_frames = count_hand_object_contact_frames(
        row["contact"], target_object_name=str(row["index"]["scene"])
    )
    final_acceptance = evaluate_synthesis_acceptance(
        trajectory_complete=True,
        termination_reason_code=int(rollout["termination_reason_code"][-1]),
        simulated_final_object_quaternion_xyzw=Rotation.from_rotvec(
            np.asarray(obj["rot_aa"][-1], dtype=np.float64)
        ).as_quat(),
        reference_final_object_quaternion_xyzw=Rotation.from_rotvec(
            np.asarray(row["reference"]["object_rot_aa"][-1], dtype=np.float64)
        ).as_quat(),
        hand_object_contact_frames=hand_object_contact_frames,
    )
    result = {
        "row_index": row_index,
        "uuid": row["index"]["uuid"],
        "source_identity": row["provenance"]["source_identity"],
        "source_row_index": int(row["provenance"]["row_index"]),
        "schema": dataset_contract,
        "reference_fps": provenance.get("reference_fps"),
        "data_fps": data_fps,
        "frames": total_frames,
        "transitions": transitions,
        "contact_frames": contact_frames,
        "contact_pairs": contact_pairs,
        "nonzero_policy_action_values": int(
            np.count_nonzero(np.abs(np.asarray(rollout["policy_mean_action"])) > 1e-8)
        ),
        "reward_sum": float(metadata["train_info"]["reward_value"]),
        "checkpoint_sha256": row["provenance"]["checkpoint_sha256"],
        "seed": int(row["provenance"]["seed"]),
        "episode_index": int(row["provenance"]["episode_index"]),
        "generation_attempt": int(row["provenance"]["generation_attempt"]),
        "augmentation_identity": row["provenance"].get("augmentation_identity"),
        "synthesis_acceptance": final_acceptance.to_dict(),
        "negative_contact_reward_steps": negative_contact_steps,
        "minimum_contact_reward": float(np.min(contact_reward, initial=0.0)),
        "max_mano_global_position_error_m": global_position_error,
        "max_mano_global_rotation_error_rad": global_rotation_error,
        "max_right_hand_shape_error": right_shape_error,
        "max_object_position_reconstruction_m": max_pos_object_error,
        "max_total_force_sum_error_N": max_force_sum_error,
        "max_object_force_rotation_error_N": max_object_force_error,
        "max_joint_force_norm_error_N": max_joint_force_norm_error,
    }
    if row_index == 0:
        checkpoint = json.loads(row["provenance"]["checkpoint_metadata_json"])
        result["checkpoint_environment_contract"] = checkpoint.get(
            "environment_contract"
        )
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


def validate_compact_row(path: Path, row_index: int) -> dict[str, Any]:
    """Validate one compact replay/visual row and its clock contract."""

    import lance

    dataset = lance.dataset(str(path))
    schema_contract = _schema_metadata(dataset).get("schema_version")
    columns = [
        "index",
        "trajectory_metadata",
        "timestamp",
        "hands",
        "objects",
        "provenance",
    ]
    if schema_contract == SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT:
        columns[5:5] = [
            "contact",
            "reference",
            "command_reference_index",
            "command_source_frame_index",
        ]
    rows = dataset.take([row_index], columns=columns).to_pylist()
    if len(rows) != 1:
        raise RuntimeError(f"Lance row {row_index} did not decode exactly once")
    row = rows[0]
    provenance = row.get("provenance") or {}
    if provenance.get("contract") != schema_contract:
        raise ValueError(f"row {row_index} compact provenance contract changed")
    if provenance.get("force_contract") != FORCE_DIRECTION_CONTRACT:
        raise ValueError(f"row {row_index} compact force contract changed")
    metadata_hash = provenance.get("checkpoint_metadata_sha256")
    if (
        not isinstance(metadata_hash, str)
        or len(metadata_hash) != 64
        or any(character not in "0123456789abcdef" for character in metadata_hash)
    ):
        raise ValueError(f"row {row_index} compact metadata SHA256 is invalid")
    source = target_replay_source_from_row(
        row,
        dataset_path=path,
        dataset_version=int(dataset.version),
        row_index=row_index,
    )
    if source.warp_ccd_iterations is None or source.warp_ccd_contacts_per_world is None:
        raise ValueError(
            f"row {row_index} compact provenance lacks explicit Warp CCD settings"
        )
    metadata = row["trajectory_metadata"]
    if metadata.get("hand_slots") != ["right", "left"] or len(row["hands"]) != 2:
        raise ValueError(f"row {row_index} compact hand-slot contract changed")
    hand = row["hands"][metadata["hand_slots"].index("right")]
    for name, shape in {
        "mano_global_pos": (source.frames, 3),
        "mano_global_rot_aa": (source.frames, 3),
        "mano_hand_pose": (source.frames, 48),
        "mano_joint_pos": (source.frames, 21, 3),
    }.items():
        values = np.asarray(hand[name], dtype=np.float64)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(
                f"row {row_index} compact hands.{name} has invalid shape/values"
            )
    mano_global_pos = np.asarray(hand["mano_global_pos"], dtype=np.float64)
    mano_global_rot = Rotation.from_rotvec(
        np.asarray(hand["mano_global_rot_aa"], dtype=np.float64)
    )
    expected_global_rot = Rotation.from_euler("XYZ", source.recorded_qpos[:, 3:6])
    if (
        np.max(np.linalg.norm(mano_global_pos - source.recorded_qpos[:, :3], axis=1))
        > 1e-12
        or np.max((mano_global_rot.inv() * expected_global_rot).magnitude()) > 1e-6
    ):
        raise ValueError(f"row {row_index} compact MANO global pose changed")
    if schema_contract == SYNTHETIC_LANCE_COMPACT_V1_CONTRACT:
        if any(name in row for name in ("contact", "reference", "rollout")):
            raise ValueError("compact v1 row contains full/audit top-level fields")
    elif schema_contract == SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT:
        contacts = row.get("contact")
        reference = row.get("reference") or {}
        command_reference = np.asarray(
            row.get("command_reference_index") or (), dtype=np.int64
        )
        command_source = np.asarray(
            row.get("command_source_frame_index") or (), dtype=np.int64
        )
        reference_source = np.asarray(
            reference.get("source_frame_index") or (), dtype=np.int64
        )
        reference_hand = np.asarray(
            reference.get("hand_urdf_dof") or (), dtype=np.float64
        )
        reference_object = np.asarray(
            reference.get("object_pos") or (), dtype=np.float64
        )
        reference_object_rot = np.asarray(
            reference.get("object_rot_aa") or (), dtype=np.float64
        )
        if not isinstance(contacts, list) or len(contacts) != source.frames:
            raise ValueError(f"row {row_index} compact contact frames differ from T")
        if (
            reference_source.shape != (source.frames,)
            or reference_hand.shape != (source.frames, 28)
            or reference_object.shape != (source.frames, 3)
            or reference_object_rot.shape != (source.frames, 3)
            or not np.all(np.isfinite(reference_hand))
            or not np.all(np.isfinite(reference_object))
            or not np.all(np.isfinite(reference_object_rot))
        ):
            raise ValueError(f"row {row_index} compact reference mapping is invalid")
        if (
            command_reference.shape != (source.transitions,)
            or command_source.shape != (source.transitions,)
            or np.any(command_reference < 0)
            or np.any(command_reference >= source.frames)
            or np.any(np.diff(command_reference) < 0)
            or not np.array_equal(
                command_source, reference_source[command_reference]
            )
        ):
            raise ValueError(f"row {row_index} compact command mapping is invalid")
        max_force_sum_error = 0.0
        max_object_force_error = 0.0
        object_rotations = Rotation.from_rotvec(
            np.asarray(row["objects"][0]["rot_aa"], dtype=np.float64)
        )
        for frame_index, entries in enumerate(contacts):
            for entry in entries:
                total_world = np.asarray(
                    entry.get("total_force_world") or (), dtype=np.float64
                )
                pairs = entry.get("contact_pairs") or []
                pair_forces = np.asarray(
                    [pair.get("force_normal") for pair in pairs], dtype=np.float64
                )
                if (
                    total_world.shape != (3,)
                    or pair_forces.ndim != 2
                    or pair_forces.shape[1:] != (3,)
                    or not np.all(np.isfinite(pair_forces))
                ):
                    raise ValueError(
                        f"row {row_index} compact contact force shape is invalid"
                    )
                pair_sum = np.sum(pair_forces, axis=0)
                max_force_sum_error = max(
                    max_force_sum_error,
                    float(np.linalg.norm(total_world - pair_sum)),
                )
                total_object = np.asarray(
                    entry.get("total_force_object") or (), dtype=np.float64
                )
                expected_object = object_rotations[frame_index].inv().apply(
                    total_world
                )
                if total_object.shape != (3,) or not np.all(np.isfinite(total_object)):
                    raise ValueError(
                        f"row {row_index} compact object force shape is invalid"
                    )
                max_object_force_error = max(
                    max_object_force_error,
                    float(np.linalg.norm(total_object - expected_object)),
                )
        if max_force_sum_error > 1e-5 or max_object_force_error > 1e-5:
            raise ValueError(
                f"row {row_index} compact contact force frames are inconsistent"
            )
    else:
        raise ValueError(f"row {row_index} compact schema contract is unsupported")
    final_acceptance = None
    if schema_contract == SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT:
        hand_object_contact_frames = count_hand_object_contact_frames(
            row.get("contact") or [],
            target_object_name=str(row["index"]["scene"]),
        )
        final_acceptance = evaluate_synthesis_acceptance(
            trajectory_complete=True,
            termination_reason_code=1,
            simulated_final_object_quaternion_xyzw=Rotation.from_rotvec(
                np.asarray(row["objects"][0]["rot_aa"][-1], dtype=np.float64)
            ).as_quat(),
            reference_final_object_quaternion_xyzw=Rotation.from_rotvec(
                np.asarray(
                    (row.get("reference") or {})["object_rot_aa"][-1],
                    dtype=np.float64,
                )
            ).as_quat(),
            hand_object_contact_frames=hand_object_contact_frames,
        )
    return {
        "row_index": row_index,
        "uuid": row["index"]["uuid"],
        "source_identity": source.source_identity,
        "source_row_index": source.source_row_index,
        "source_contract": source.source_contract,
        "reference_fps": source.reference_fps,
        "data_fps": source.control_fps,
        "frames": source.frames,
        "transitions": source.transitions,
        "object": row["index"]["scene"],
        "checkpoint_sha256": source.checkpoint_sha256,
        "checkpoint_metadata_sha256": metadata_hash,
        "checkpoint_update": source.checkpoint_update,
        "seed": int(provenance["seed"]),
        "episode_index": int(provenance["episode_index"]),
        "generation_attempt": int(provenance["generation_attempt"]),
        "augmentation_identity": provenance.get("augmentation_identity"),
        "synthesis_acceptance": (
            None if final_acceptance is None else final_acceptance.to_dict()
        ),
        "warp_ccd_iterations": source.warp_ccd_iterations,
        "warp_ccd_contacts_per_world": source.warp_ccd_contacts_per_world,
    }


def _validate_prefix_only_contract(
    *, manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    synthesis = manifest.get("synthesis") or {}
    production_contract = synthesis.get("production_contract")
    if production_contract is None:
        return None
    if production_contract != APPROACH_PREFIX_ONLY_PRODUCTION_CONTRACT:
        raise ValueError("prefix-only production contract changed")
    if synthesis.get("augmentation_identity_contract") != (
        PREFIX_ONLY_AUGMENTATION_IDENTITY_CONTRACT
    ):
        raise ValueError("prefix-only augmentation identity contract changed")
    if synthesis.get("retreat_suffix") is not None:
        raise ValueError("prefix-only production manifest contains retreat suffix")
    if manifest.get("retreat_suffixes"):
        raise ValueError("prefix-only production records accepted retreat suffixes")
    approach = synthesis.get("approach_prefix") or {}
    config = approach.get("config") or {}
    if (
        int(config.get("required_base_pre_padding", -1)) != 60
        or not np.isclose(
            float(config.get("vertical_arc_height_m", np.nan)),
            0.04,
            rtol=0.0,
            atol=1e-12,
        )
        or config.get("mode") not in ("far", "near")
    ):
        raise ValueError("prefix-only pre60/Far-Near/4cm contract changed")
    augmentation_values = [row.get("augmentation_identity") for row in rows]
    if any(
        not isinstance(value, str)
        or not value.startswith(PREFIX_ONLY_AUGMENTATION_IDENTITY_CONTRACT + ":")
        for value in augmentation_values
    ):
        raise ValueError("prefix-only rows lack v4 augmentation identity")
    return {
        "contract": production_contract,
        "mode": config["mode"],
        "base_pre_padding": 60,
        "vertical_arc_height_m": 0.04,
        "retreat_suffix": None,
    }


def _validate_manifest_acceptance(
    *,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Recompute and bind every saved row to one accepted attempt diagnostic."""

    synthesis = manifest.get("synthesis") or {}
    gate = synthesis.get("acceptance_gate")
    if gate is None:
        return None
    expected_gate = synthesis_acceptance_manifest()
    if gate != expected_gate or gate.get("contract") != SYNTHESIS_ACCEPTANCE_CONTRACT:
        raise ValueError("synthesis acceptance gate manifest changed")
    attempts = synthesis.get("acceptance_attempts")
    if not isinstance(attempts, list):
        raise ValueError("synthesis manifest omits acceptance_attempts")

    accepted_attempts: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    all_attempt_keys: set[tuple[str, int, int, int]] = set()
    for attempt in attempts:
        acceptance = attempt.get("acceptance") or {}
        if acceptance.get("contract") != SYNTHESIS_ACCEPTANCE_CONTRACT:
            raise ValueError("attempt acceptance contract changed")
        key = (
            str(attempt.get("source_identity")),
            int(attempt.get("seed")),
            int(attempt.get("attempt_number")),
            int(attempt.get("episode_index")),
        )
        if key in all_attempt_keys:
            raise ValueError("synthesis acceptance attempt keys are not unique")
        all_attempt_keys.add(key)
        outer_accepted = bool(attempt.get("accepted"))
        inner_accepted = bool(acceptance.get("accepted"))
        failure_reasons = acceptance.get("failure_reasons")
        additional_failures = attempt.get("additional_failure_reasons", [])
        if (
            not isinstance(failure_reasons, list)
            or not isinstance(additional_failures, list)
            or any(
                not isinstance(reason, str) or not reason
                for reason in failure_reasons + additional_failures
            )
            or (inner_accepted and failure_reasons)
            or (not inner_accepted and not failure_reasons)
            or outer_accepted != (inner_accepted and not additional_failures)
        ):
            raise ValueError("attempt acceptance diagnostics are inconsistent")
        if outer_accepted:
            accepted_attempts[key] = attempt

    row_keys: set[tuple[str, int, int, int]] = set()
    for row in rows:
        acceptance = row.get("synthesis_acceptance") or {}
        if not bool(acceptance.get("accepted")):
            raise ValueError(
                f"saved row {row.get('row_index')} fails synthesis acceptance"
            )
        key = (
            str(row["source_identity"]),
            int(row["seed"]),
            int(row["generation_attempt"]),
            int(row["episode_index"]),
        )
        recorded_attempt = accepted_attempts.get(key)
        if recorded_attempt is None:
            raise ValueError("saved row has no accepted attempt diagnostic")
        recorded = recorded_attempt.get("acceptance") or {}
        exact_fields = (
            "accepted",
            "trajectory_complete",
            "termination_reason_code",
            "hand_object_contact_frames",
            "failure_reasons",
        )
        if any(recorded.get(field) != acceptance.get(field) for field in exact_fields):
            raise ValueError(
                "saved row acceptance differs from manifest diagnostics"
            )
        recorded_xyz = np.asarray(
            recorded.get("final_rotation_xyz_abs_error_deg"), dtype=np.float64
        )
        recomputed_xyz = np.asarray(
            acceptance.get("final_rotation_xyz_abs_error_deg"), dtype=np.float64
        )
        if (
            recorded_xyz.shape != (3,)
            or recomputed_xyz.shape != (3,)
            or not np.allclose(recorded_xyz, recomputed_xyz, rtol=0.0, atol=1e-4)
            or not np.isclose(
                float(recorded.get("final_rotation_xyz_mean_error_deg")),
                float(acceptance.get("final_rotation_xyz_mean_error_deg")),
                rtol=0.0,
                atol=1e-4,
            )
        ):
            raise ValueError(
                "saved row rotation acceptance differs from manifest diagnostics"
            )
        if key in row_keys:
            raise ValueError("saved synthesis acceptance keys are not unique")
        row_keys.add(key)
    if row_keys != set(accepted_attempts):
        raise ValueError("saved rows differ from accepted attempt diagnostics")

    contact_frames = [
        int(row["synthesis_acceptance"]["hand_object_contact_frames"])
        for row in rows
    ]
    rotation_errors = [
        float(row["synthesis_acceptance"]["final_rotation_xyz_mean_error_deg"])
        for row in rows
    ]
    return {
        "contract": SYNTHESIS_ACCEPTANCE_CONTRACT,
        "accepted_rows": len(rows),
        "attempts": len(attempts),
        "rejected_attempts": len(attempts) - len(rows),
        "hand_object_contact_frames_min_max": (
            [min(contact_frames), max(contact_frames)] if contact_frames else [0, 0]
        ),
        "final_rotation_xyz_mean_error_deg_max": (
            max(rotation_errors, default=0.0)
        ),
    }


def validate_compact_dataset(
    path: Path,
    output: Path | None = None,
    *,
    max_attempts: int = 5,
) -> dict[str, Any]:
    """Validate compact schema, external metadata, and replay clocks."""

    import lance

    dataset = lance.dataset(str(path))
    metadata = _schema_metadata(dataset)
    if metadata.get("schema_version") not in SYNTHETIC_LANCE_COMPACT_CONTRACTS:
        raise ValueError(
            "dataset schema_version is not the compact replay/visual contract"
        )
    source_contract = metadata.get("source_contract")
    if source_contract not in SYNTHETIC_LANCE_SOURCE_CONTRACTS:
        raise ValueError("compact source_contract is not v2.2 or v2.3")
    try:
        control_fps = int(metadata["control_fps"])
        clock = simulation_clock(control_fps)
        raw_reference = metadata["reference_fps"]
        reference_fps = None if raw_reference == "none" else int(raw_reference)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("compact schema has invalid clock metadata") from exc
    if source_contract == SYNTHETIC_LANCE_V22_CONTRACT:
        if control_fps != 200 or reference_fps is not None:
            raise ValueError("v2.2 compact schema must use legacy 200 Hz clock")
    elif control_fps in (100, 120) and reference_fps != control_fps:
        raise ValueError("v2.3 compact reference/control clocks are not coupled")
    for field, expected in {
        "physics_fps": clock.physics_fps,
        "physics_substeps_per_control": clock.physics_substeps_per_control,
    }.items():
        if int(metadata.get(field, -1)) != expected:
            raise ValueError(f"compact schema {field} is inconsistent")
    for field, expected in (
        ("control_timestep_seconds", clock.control_timestep),
        ("physics_timestep_seconds", clock.physics_timestep),
    ):
        if not np.isclose(
            float(metadata.get(field, "nan")), expected, rtol=0.0, atol=1e-15
        ):
            raise ValueError(f"compact schema {field} is inconsistent")
    expected_fields = [
        "index",
        "trajectory_metadata",
        "timestamp",
        "hands",
        "objects",
    ]
    if metadata.get("schema_version") == SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT:
        expected_fields.extend(
            [
                "contact",
                "reference",
                "command_reference_index",
                "command_source_frame_index",
            ]
        )
    expected_fields.append("provenance")
    if dataset.schema.names != expected_fields:
        raise ValueError(f"compact schema fields differ: {dataset.schema.names}")
    row_count = int(dataset.count_rows())
    rows: list[dict[str, Any]] = []
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
        else:
            child = None
        if len(rows) != row_index + 1:
            code, error = failures[-1]
            raise RuntimeError(
                f"isolated compact validation failed for row {row_index} after "
                f"{len(failures)} attempt(s), final exit {code}: {error}"
            )
    if any(
        row["source_contract"] != source_contract
        or row["data_fps"] != control_fps
        or row["reference_fps"] != reference_fps
        for row in rows
    ):
        raise ValueError("compact rows differ from schema clock/source contract")
    uuids = [row["uuid"] for row in rows]
    identities = [row["source_identity"] for row in rows]
    checkpoints = {row["checkpoint_sha256"] for row in rows}
    metadata_hashes = {row["checkpoint_metadata_sha256"] for row in rows}
    if len(set(uuids)) != row_count:
        raise ValueError("compact dataset contains duplicate generated UUIDs")
    if len(checkpoints) > 1:
        raise ValueError("compact dataset rows do not share one checkpoint SHA256")
    catalog_path = path.parent / f"{path.name}.checkpoint-metadata.json"
    manifest_path = path.parent / f"{path.name}.manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        entries = catalog.get("entries")
        if (
            catalog.get("schema") != "manorl.synthetic_checkpoint_metadata_catalog.v1"
            or catalog.get("source_contract") != source_contract
            or not isinstance(entries, dict)
        ):
            raise ValueError("compact checkpoint metadata catalog has invalid schema")
        if not metadata_hashes.issubset(entries):
            raise ValueError("compact rows reference metadata absent from catalog")
        for metadata_hash in metadata_hashes:
            calculated = hashlib.sha256(
                json.dumps(
                    entries[metadata_hash], sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if calculated != metadata_hash:
                raise ValueError("compact checkpoint metadata catalog hash mismatch")
        metadata_source = str(catalog_path)
    else:
        checkpoint_metadata = (manifest.get("checkpoint") or {}).get(
            "checkpoint_metadata"
        )
        if not isinstance(checkpoint_metadata, dict):
            raise ValueError(
                "compact dataset lacks checkpoint metadata catalog/manifest"
            )
        calculated = hashlib.sha256(
            json.dumps(
                checkpoint_metadata, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if metadata_hashes != {calculated}:
            raise ValueError("compact manifest checkpoint metadata hash mismatch")
        metadata_source = str(manifest_path)
    summary = {
        "schema": str(metadata.get("schema_version")),
        "source_contract": source_contract,
        "schema_metadata": metadata,
        "rows": row_count,
        "unique_source_identities": len(set(identities)),
        "frame_count_min_max_sum": [
            min((row["frames"] for row in rows), default=0),
            max((row["frames"] for row in rows), default=0),
            sum(row["frames"] for row in rows),
        ],
        "transition_count_sum": sum(row["transitions"] for row in rows),
        "checkpoint_sha256": next(iter(checkpoints)) if checkpoints else None,
        "checkpoint_metadata_sha256": sorted(metadata_hashes),
        "checkpoint_metadata_source": metadata_source,
        "clock": {
            "reference_fps": reference_fps,
            "control_fps": control_fps,
            "physics_fps": clock.physics_fps,
            "physics_substeps_per_control": clock.physics_substeps_per_control,
        },
        "warp_ccd": sorted(
            {
                (row["warp_ccd_iterations"], row["warp_ccd_contacts_per_world"])
                for row in rows
            }
        ),
        "isolated_decoder_attempts": sum(row["decoder_attempts"] for row in rows),
        "retried_row_count": sum(row["decoder_attempts"] > 1 for row in rows),
        "prefix_only_production": _validate_prefix_only_contract(
            manifest=manifest, rows=rows
        ),
        "synthesis_acceptance": _validate_manifest_acceptance(
            manifest=manifest, rows=rows
        ),
    }
    target = output or path.parent / f"{path.name}.validation.json"
    target.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**summary, "validation_output": str(target)}


def _is_retryable_nested_decode_failure(returncode: int, stderr: str) -> bool:
    """Retry every isolated child failure; deterministic failures remain bounded."""

    del stderr
    return returncode != 0


def validate_dataset(
    path: Path,
    output: Path | None = None,
    *,
    max_attempts: int = 5,
    expected_episodes_per_identity: int = 5,
    max_generation_attempts_per_identity: int = 10,
) -> dict[str, Any]:
    import lance

    dataset = lance.dataset(str(path))
    metadata = _schema_metadata(dataset)
    schema_contract = metadata.get("schema_version")
    if schema_contract in SYNTHETIC_LANCE_COMPACT_CONTRACTS:
        return validate_compact_dataset(path, output, max_attempts=max_attempts)
    if schema_contract not in (SYNTHETIC_LANCE_V22_CONTRACT, SYNTHETIC_LANCE_CONTRACT):
        raise ValueError("dataset schema_version is not a supported synthetic contract")
    if schema_contract == SYNTHETIC_LANCE_V22_CONTRACT:
        schema_clock = simulation_clock(200)
        schema_reference_fps = None
    else:
        try:
            schema_control_fps = int(metadata["control_fps"])
            schema_clock = simulation_clock(schema_control_fps)
            recorded_reference_fps = metadata["reference_fps"]
            schema_reference_fps = (
                None
                if recorded_reference_fps == "none"
                else int(recorded_reference_fps)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("v2.3 schema has invalid clock metadata") from exc
        if schema_reference_fps not in (None, 100, 120):
            raise ValueError("v2.3 schema reference_fps is invalid")
        if (
            schema_clock.policy_fps in (100, 120)
            and schema_reference_fps != schema_clock.policy_fps
        ):
            raise ValueError("v2.3 public reference/control clocks are not coupled")
        expected_integer_metadata = {
            "physics_fps": schema_clock.physics_fps,
            "physics_substeps_per_control": schema_clock.physics_substeps_per_control,
        }
        for field, expected in expected_integer_metadata.items():
            if int(metadata.get(field, -1)) != expected:
                raise ValueError(f"v2.3 schema {field} is inconsistent")
        for field, expected in (
            ("control_timestep_seconds", schema_clock.control_timestep),
            ("physics_timestep_seconds", schema_clock.physics_timestep),
        ):
            try:
                recorded = float(metadata[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"v2.3 schema has invalid {field}") from exc
            if not np.isclose(recorded, expected, rtol=0.0, atol=1e-15):
                raise ValueError(f"v2.3 schema {field} is inconsistent")
    if metadata.get("mano_global_frame_contract") != MANO_GLOBAL_FRAME_CONTRACT:
        raise ValueError("dataset MANO global-frame contract changed")
    if metadata.get("force_contract") != FORCE_DIRECTION_CONTRACT:
        raise ValueError("dataset force contract is not normal-only scale 1.0")
    if metadata.get("reward_contract") != REWARD_CONTRACT_ID:
        raise ValueError("dataset reward contract changed")
    if metadata.get("ppo_reward_contract") != PPO_REWARD_CONTRACT_ID:
        raise ValueError("dataset PPO reward contract changed")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if expected_episodes_per_identity < 1:
        raise ValueError("expected episodes per identity must be positive")
    if max_generation_attempts_per_identity < expected_episodes_per_identity:
        raise ValueError("generation attempt budget is smaller than episode target")
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
            stderr = child.stderr.strip()
            failures.append((child.returncode, stderr))
            if not _is_retryable_nested_decode_failure(child.returncode, stderr):
                break
        else:
            child = None
        if len(rows) != row_index + 1:
            code, error = failures[-1]
            raise RuntimeError(
                f"isolated Lance validation failed for row {row_index} after "
                f"{len(failures)} attempt(s), final exit {code}: {error}"
            )
    if any(row["data_fps"] != schema_clock.policy_fps for row in rows):
        raise ValueError("row clocks differ from the dataset schema clock")
    if schema_contract == SYNTHETIC_LANCE_CONTRACT and any(
        row["reference_fps"] != schema_reference_fps for row in rows
    ):
        raise ValueError("row reference clocks differ from the dataset schema clock")
    identities = [row["source_identity"] for row in rows]
    uuids = [row["uuid"] for row in rows]
    source_rows = [row["source_row_index"] for row in rows]
    if len(set(uuids)) != row_count:
        raise ValueError("dataset contains duplicate generated UUIDs")
    grouped: dict[str, list[dict[str, Any]]] = {}
    source_row_by_identity: dict[str, int] = {}
    for row in rows:
        identity = row["source_identity"]
        grouped.setdefault(identity, []).append(row)
        source_row = row["source_row_index"]
        if (
            identity in source_row_by_identity
            and source_row_by_identity[identity] != source_row
        ):
            raise ValueError("one source identity maps to multiple raw rows")
        source_row_by_identity[identity] = source_row
    if len(set(source_row_by_identity.values())) != len(grouped):
        raise ValueError("multiple source identities map to one raw row")
    expected_episode_indices = list(range(expected_episodes_per_identity))
    for identity, identity_rows in grouped.items():
        episodes = sorted(row["episode_index"] for row in identity_rows)
        attempts = [row["generation_attempt"] for row in identity_rows]
        if episodes != expected_episode_indices:
            raise ValueError(
                f"identity {identity} episodes {episodes} != {expected_episode_indices}"
            )
        if len(set(attempts)) != len(attempts) or any(
            attempt < 1 or attempt > max_generation_attempts_per_identity
            for attempt in attempts
        ):
            raise ValueError(f"identity {identity} has invalid generation attempts")
    checkpoint_hashes = {row["checkpoint_sha256"] for row in rows}
    seeds = {row["seed"] for row in rows}
    if len(checkpoint_hashes) != 1:
        raise ValueError("dataset rows do not share one checkpoint SHA256")
    manifest_path = path.parent / f"{path.name}.manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    summary = {
        "schema": schema_contract,
        "schema_metadata": metadata,
        "rows": row_count,
        "unique_identities": len(grouped),
        "episodes_per_identity": expected_episodes_per_identity,
        "max_generation_attempts_per_identity": max_generation_attempts_per_identity,
        "generation_attempt_range": (
            [
                min(row["generation_attempt"] for row in rows),
                max(row["generation_attempt"] for row in rows),
            ]
            if rows
            else None
        ),
        "episode_seed_range": [min(seeds), max(seeds)] if rows else None,
        "source_row_range": [min(source_rows), max(source_rows)] if rows else None,
        "frame_count_min_max_sum": (
            [
                min(row["frames"] for row in rows),
                max(row["frames"] for row in rows),
                sum(row["frames"] for row in rows),
            ]
            if rows
            else [0, 0, 0]
        ),
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
        "negative_contact_reward_steps": sum(
            row["negative_contact_reward_steps"] for row in rows
        ),
        "minimum_contact_reward": min(
            (row["minimum_contact_reward"] for row in rows), default=0.0
        ),
        "prefix_only_production": _validate_prefix_only_contract(
            manifest=manifest, rows=rows
        ),
        "synthesis_acceptance": _validate_manifest_acceptance(
            manifest=manifest, rows=rows
        ),
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
        "max_mano_global_position_error_m": max(
            (row["max_mano_global_position_error_m"] for row in rows), default=0.0
        ),
        "max_mano_global_rotation_error_rad": max(
            (row["max_mano_global_rotation_error_rad"] for row in rows), default=0.0
        ),
        "max_right_hand_shape_error": max(
            (row["max_right_hand_shape_error"] for row in rows), default=0.0
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
    target.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**summary, "validation_output": str(target)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--expected-episodes-per-identity", type=int, default=5)
    parser.add_argument("--max-generation-attempts-per-identity", type=int, default=10)
    parser.add_argument("--row-index", type=int, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.row_index is not None:
        import lance

        schema = _schema_metadata(lance.dataset(str(args.dataset))).get(
            "schema_version"
        )
        row_result = (
            validate_compact_row(args.dataset, args.row_index)
            if schema in SYNTHETIC_LANCE_COMPACT_CONTRACTS
            else validate_row(args.dataset, args.row_index)
        )
        print(json.dumps(row_result, sort_keys=True))
        return 0
    print(
        json.dumps(
            validate_dataset(
                args.dataset,
                args.output,
                max_attempts=args.max_attempts,
                expected_episodes_per_identity=args.expected_episodes_per_identity,
                max_generation_attempts_per_identity=(
                    args.max_generation_attempts_per_identity
                ),
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
