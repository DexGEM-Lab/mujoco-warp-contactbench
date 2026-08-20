#!/usr/bin/env python3
"""Export bounded 1:N ManoRL checkpoint episodes to clock-aware v2.3 Lance."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace as dataclass_replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from sim.manorl.abi import TARGET_MAX_DEVIATION_DISTANCE
from sim.manorl.approach_prefix import (
    APPROACH_PREFIX_CONTRACT,
    ApproachPrefixConfig,
    ApproachPrefixSample,
    augment_trajectory_with_approach_prefix,
)
from sim.manorl.checkpoint import checkpoint_runtime_metadata
from sim.manorl.contracts import JOINT_DOF, simulation_clock
from sim.manorl.environment import (
    EnvironmentConfig,
    MujocoManoEnvironment,
    recommended_warp_contact_capacity,
)
from sim.manorl.lance_v2 import (
    FORCE_DIRECTION_CONTRACT,
    SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT,
    SYNTHETIC_LANCE_CONTRACT,
    SYNTHETIC_LANCE_OUTPUT_FORMAT_COMPACT,
    SYNTHETIC_LANCE_OUTPUT_FORMAT_FULL,
    SYNTHETIC_LANCE_OUTPUT_FORMATS,
    build_compact_row,
    build_v2_row,
    corrected_contact_frames,
    file_sha256,
    write_compact_lance,
    write_v2_lance,
)
from sim.manorl.observations import SOURCE_ALIGNED_COMPATIBILITY
from sim.manorl.rewards import (
    CONTACT_FORCE_THRESHOLD,
    PPO_REWARD_CONTRACT_ID,
    REWARD_CONTRACT_ID,
)
from sim.manorl.synthetic_parent import (
    AcceptedSyntheticParent,
    load_accepted_synthetic_parent,
)
from sim.manorl.trajectory import (
    SUPPORTED_REFERENCE_FPS,
    TrajectoryBatch,
    TrajectorySelection,
    load_assigned_trajectory_batch,
    resample_reference_trajectory,
)
from sim.manorl.view_environment import (
    _build_checkpoint_stepper,
    _checkpoint_environment_options,
    _resolve_reference_fps,
    _validate_checkpoint_path,
)

DEFAULT_DATASET = Path(
    "/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_guangguan/"
    "human_p1_guangguan_clean.lance"
)


def _validate_output_format(output_format: str) -> str:
    if output_format not in SYNTHETIC_LANCE_OUTPUT_FORMATS:
        choices = ", ".join(SYNTHETIC_LANCE_OUTPUT_FORMATS)
        raise ValueError(
            f"output_format must be one of {choices}, got {output_format!r}"
        )
    return output_format


def _projection_manifest(output_format: str) -> dict[str, Any]:
    if output_format == SYNTHETIC_LANCE_OUTPUT_FORMAT_COMPACT:
        return {
            "kept_top_level": [
                "index",
                "trajectory_metadata",
                "timestamp",
                "hands",
                "objects",
                "contact",
                "reference",
                "command_reference_index",
                "command_source_frame_index",
                "provenance",
            ],
            "dropped_top_level": ["rollout"],
            "checkpoint_metadata": "embedded_once_in_manifest",
        }
    return {
        "kept_top_level": "all_v2_3_fields",
        "dropped_top_level": [],
        "checkpoint_metadata": "embedded_per_row",
    }


def _software_commit() -> str:
    override = os.environ.get("MANORL_SOFTWARE_COMMIT")
    if override:
        return override
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _checkpoint_update(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def _load_predecoded_batch(
    selection: TrajectorySelection,
    *,
    num_envs: int,
    manifest_path: Path,
    exact_identities: tuple[str, ...] | None = None,
) -> TrajectoryBatch:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(manifest.get("dataset_path", "")) != selection.dataset_path:
        raise ValueError(
            "predecoded manifest source dataset differs from the requested dataset"
        )
    if int(manifest.get("dataset_version", -1)) != selection.expected_dataset_version:
        raise ValueError(
            "predecoded manifest dataset version differs from the requested version"
        )
    if manifest.get("hand_side") != "right":
        raise ValueError("predecoded manifest must use right-hand selection")
    pairs = selection.requested_pairs
    if not pairs:
        raise ValueError("predecoded v2 export requires explicit object/action pairs")
    object_types = {pair.object_type for pair in pairs}
    if len(object_types) != 1:
        raise ValueError("predecoded v2 export requires one homogeneous object type")
    requested = {pair.canonical for pair in pairs}
    records = [
        record
        for record in manifest.get("valid_records", [])
        if record.get("pair") in requested
    ]
    if exact_identities is not None:
        wanted_identities = set(exact_identities)
        records = [
            record for record in records if record.get("identity") in wanted_identities
        ]
        found_identities = {str(record.get("identity")) for record in records}
        missing_identities = sorted(wanted_identities - found_identities)
        if missing_identities:
            raise LookupError(
                f"predecoded manifest omits requested identities: {missing_identities}"
            )
    available = {record["pair"] for record in records}
    missing = sorted(requested - available)
    if missing:
        raise LookupError(f"predecoded manifest omits requested pairs: {missing}")
    if num_envs > len(records):
        raise ValueError(
            f"num-envs {num_envs} exceeds {len(records)} distinct predecoded identities "
            f"for {selection.canonical_selector}"
        )
    offset = (selection.pair_assignment_cycle * num_envs) % len(records)
    records = (records[offset:] + records[:offset])[:num_envs]
    trajectories = []
    for record in records:
        path = manifest_path.parent / f"{record['identity']}.pkl"
        if file_sha256(path) != record.get("pickle_sha256"):
            raise RuntimeError(f"predecoded trajectory hash changed: {path}")
        with path.open("rb") as stream:
            trajectory = pickle.load(stream)
        if trajectory.identity.identity != record["identity"]:
            raise RuntimeError(f"predecoded trajectory identity changed: {path}")
        if selection.reference_fps is not None:
            trajectory = resample_reference_trajectory(
                trajectory,
                reference_fps=selection.reference_fps,
                control_fps=selection.resolved_control_fps,
            )
        trajectories.append(trajectory)
    trajectory_package = manifest.get("trajectory_package")
    if trajectory_package is not None and not isinstance(trajectory_package, dict):
        raise ValueError("predecoded manifest trajectory_package must be an object")
    return TrajectoryBatch(
        tuple(trajectories),
        resolved_pairs=pairs,
        selection_mode=selection.mode,
        pair_assignment_cycle=selection.pair_assignment_cycle,
        trajectory_package=trajectory_package,
    )


def _manifest_lineage_by_row(
    predecoded_manifest: Path,
) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
    """Extract source lineage embedded in a predecoded manifest, keyed by row index."""

    manifest = json.loads(predecoded_manifest.read_text(encoding="utf-8"))
    records = manifest.get("valid_records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("predecoded manifest has no valid_records")
    lineage: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for record in records:
        row_index = record.get("row_index")
        if not isinstance(row_index, int):
            raise ValueError("predecoded manifest record omits integer row_index")
        lineage[row_index] = (
            dict(record.get("source_index") or {}),
            dict(record.get("trajectory_metadata") or {}),
        )
    return lineage


def _source_metadata(
    trajectories: TrajectoryBatch,
    *,
    lineage_by_row: dict[int, tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
    """Read only lightweight lineage columns from the pinned source version."""

    first = trajectories.trajectories[0]
    indices = sorted({item.identity.row_index for item in trajectories.trajectories})
    if lineage_by_row is not None:
        missing = sorted(set(indices) - set(lineage_by_row))
        if missing:
            raise RuntimeError(f"predecoded manifest omits lineage for rows: {missing}")
        return {row_index: lineage_by_row[row_index] for row_index in indices}

    import lance

    path = first.identity.dataset_path
    version = first.identity.dataset_version
    dataset = lance.dataset(path, version=version)
    rows = dataset.take(indices, columns=["index", "trajectory_metadata"]).to_pylist()
    if len(rows) != len(indices):
        raise RuntimeError("source Lance did not return every requested lineage row")
    return {
        row_index: (
            dict(row.get("index") or {}),
            dict(row.get("trajectory_metadata") or {}),
        )
        for row_index, row in zip(indices, rows, strict=True)
    }


def _materialized_contacts(
    environment: MujocoManoEnvironment,
) -> tuple[list[list[dict[str, Any]]], np.ndarray]:
    state = environment.producer.materialize_state(environment.data)
    buffers = environment.producer.materialize_contact_buffers(
        environment.data, environment.config.num_envs
    )
    producer = environment.producer
    if producer.active_object_geom_ids is not None:
        raise RuntimeError(
            "v2 checkpoint export currently requires one homogeneous object type"
        )
    contacts = corrected_contact_frames(
        buffers=buffers,
        state=state,
        model=environment.model,
        keypoint_geom_ids=producer.keypoint_geom_ids_by_side["right"],
        object_geom_ids=tuple(sorted(producer.object_geom_ids)),
        object_body_id=producer.object_body_id,
        wrist_body_id=producer.keypoint_body_ids[0],
        object_name=environment.object_type,
    )
    floor_geom_id = int(
        environment.mujoco.mj_name2id(
            environment.model,
            environment.mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )
    )
    if floor_geom_id < 0:
        raise RuntimeError("synthesis scene has no resolvable floor geom")
    # The left hand may be compiled for reference following. Augmentation owns
    # only the controlled right hand, so unrelated left-hand floor candidates
    # must not reject an otherwise valid sample.
    hand_geom_ids = {
        int(geom_id)
        for geom_id in producer.keypoint_geom_ids_by_side["right"]
    }
    hand_floor = np.zeros(environment.config.num_envs, dtype=bool)
    for contact_id in range(buffers.count):
        first, second = map(int, buffers.geom[contact_id])
        if (first == floor_geom_id and second in hand_geom_ids) or (
            second == floor_geom_id and first in hand_geom_ids
        ):
            world_id = int(buffers.world[contact_id])
            solved = buffers.constraint_force[
                world_id, np.asarray(buffers.addresses[contact_id], dtype=np.int64)
            ]
            if float(np.sum(solved)) > CONTACT_FORCE_THRESHOLD:
                hand_floor[world_id] = True
    return contacts, hand_floor


def _state_row(
    environment: MujocoManoEnvironment, env_id: int
) -> dict[str, np.ndarray]:
    physical = environment.last_physical
    observation = environment.last_observation
    if physical is None or observation is None:
        raise RuntimeError("environment omitted physical/observation state")
    return {
        "mano_joint_pos": np.concatenate(
            [
                np.asarray(physical.hand_keypoint_positions[env_id], dtype=np.float64),
                np.asarray(physical.fingertip_positions[env_id], dtype=np.float64),
            ],
            axis=0,
        ),
        "urdf_dof": np.asarray(physical.mano_dof_pos[env_id], dtype=np.float64),
        "object_position": np.asarray(
            physical.object_position[env_id], dtype=np.float64
        ),
        "object_orientation_xyzw": np.asarray(
            physical.object_orientation_xyzw[env_id], dtype=np.float64
        ),
        "observation": np.asarray(observation.policy_input[env_id], dtype=np.float64),
    }


def _append_state(
    storage: dict[str, list[np.ndarray]], row: dict[str, np.ndarray]
) -> None:
    for name, value in row.items():
        storage[name].append(value.copy())


def _seed_attempt(seed: int, device: str) -> None:
    np.random.seed(seed)
    if device == "gpu":
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _trajectory_subset(
    batch: TrajectoryBatch, trajectories: list[Any]
) -> TrajectoryBatch:
    return TrajectoryBatch(
        tuple(trajectories),
        resolved_pairs=batch.resolved_pairs,
        selection_mode=batch.selection_mode,
        pair_assignment_cycle=batch.pair_assignment_cycle,
        trajectory_package=batch.trajectory_package,
    )


def _augment_attempt_trajectories(
    trajectories: TrajectoryBatch,
    *,
    attempt_seed: int,
    config: ApproachPrefixConfig,
) -> tuple[TrajectoryBatch, dict[str, ApproachPrefixSample]]:
    augmented = []
    samples: dict[str, ApproachPrefixSample] = {}
    for trajectory in trajectories.trajectories:
        resolved, sample = augment_trajectory_with_approach_prefix(
            trajectory,
            seed=attempt_seed,
            config=config,
        )
        augmented.append(resolved)
        samples[trajectory.identity.identity] = sample
    return _trajectory_subset(trajectories, augmented), samples


def _run_attempt_batch(
    *,
    checkpoint: Path,
    checkpoint_options: Any,
    trajectories: TrajectoryBatch,
    metadata_by_row: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    device: str,
    output_format: str,
    allow_deviation_termination: bool,
    attempt_seed: int,
    attempt_numbers: dict[str, int],
    episode_indices: dict[str, int],
    provenance_base: dict[str, Any],
    object_xy_offset_m: float = 0.0,
    approach_prefix_config: ApproachPrefixConfig | None = None,
    accepted_parent: AcceptedSyntheticParent | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    int,
    int,
    dict[str, dict[str, object]],
]:
    """Run one candidate episode for each identity in one attempt round."""

    _seed_attempt(attempt_seed, device)
    if accepted_parent is not None:
        if trajectories.num_envs != 1 or (
            trajectories.trajectories[0].identity.identity
            != accepted_parent.source_identity
        ):
            raise ValueError(
                "accepted-parent synthesis requires exactly its bound source identity"
            )
        if approach_prefix_config is None:
            raise ValueError("accepted-parent synthesis requires approach-prefix augmentation")
    resolved_deviation_termination = (
        allow_deviation_termination or approach_prefix_config is not None
    )
    approach_samples: dict[str, ApproachPrefixSample] = {}
    if approach_prefix_config is not None:
        trajectories, approach_samples = _augment_attempt_trajectories(
            trajectories,
            attempt_seed=attempt_seed,
            config=approach_prefix_config,
        )
    num_envs = trajectories.num_envs
    environment = MujocoManoEnvironment(
        trajectories,
        EnvironmentConfig(
            device=device,
            num_envs=num_envs,
            residual_enabled=True,
            residual_action=checkpoint_options.residual_action,
            compatibility=dataclass_replace(
                SOURCE_ALIGNED_COMPATIBILITY,
                movement_pre_padding=(
                    approach_prefix_config.required_base_pre_padding
                    if approach_prefix_config is not None
                    else checkpoint_options.pre_padding
                ),
            ),
            max_deviation_distance=(
                TARGET_MAX_DEVIATION_DISTANCE
                if resolved_deviation_termination
                else 1_000_000.0
            ),
            contact_capacity=recommended_warp_contact_capacity(
                num_envs, trajectories.hand_sides
            ),
            reference_fps=checkpoint_options.reference_fps,
            control_fps=checkpoint_options.control_fps,
            post_padding=checkpoint_options.post_padding,
            warp_ccd_iterations=checkpoint_options.warp_ccd_iterations,
            warp_ccd_contacts_per_world=checkpoint_options.warp_ccd_contacts_per_world,
            hand_side="right",
            object_init_xy_offset_range_m=(
                0.0 if accepted_parent is not None else object_xy_offset_m
            ),
            object_init_xy_offsets_m=(
                None
                if accepted_parent is None
                else (accepted_parent.object_init_xy_offset_m,)
            ),
        ),
    )
    stepper = _build_checkpoint_stepper(
        environment,
        checkpoint,
        policy_transfer=approach_prefix_config is not None,
        policy_enable_steps=(
            environment.early_phase_lengths
            if approach_prefix_config is not None
            else None
        ),
    )
    right_model_index = environment.model_hand_sides.index("right")
    right_model_slice = slice(
        right_model_index * JOINT_DOF, (right_model_index + 1) * JOINT_DOF
    )
    state_storage = [defaultdict(list) for _ in range(num_envs)]
    rollout_storage = [defaultdict(list) for _ in range(num_envs)]
    contact_storage: list[list[list[dict[str, Any]]]] = [[] for _ in range(num_envs)]
    active = np.ones(num_envs, dtype=bool)
    succeeded = np.zeros(num_envs, dtype=bool)
    failure_reasons: dict[str, str] = {}

    initial_contacts, initial_hand_floor = _materialized_contacts(environment)
    prefix_valid = np.ones(num_envs, dtype=bool)
    prefix_invalid_reasons: list[str | None] = [None] * num_envs
    if approach_prefix_config is not None:
        for env_id in range(num_envs):
            if bool(initial_hand_floor[env_id]):
                prefix_valid[env_id] = False
                prefix_invalid_reasons[env_id] = "augmentation_prefix_hand_table_clearance"
    for env_id in range(num_envs):
        initial = _state_row(environment, env_id)
        _append_state(state_storage[env_id], initial)
        state_storage[env_id]["urdf_dof_target"].append(
            np.asarray(
                environment.reference_q_model[env_id, 0, right_model_slice],
                dtype=np.float64,
            )
        )
        contact_storage[env_id].append(initial_contacts[env_id])

    max_steps = int(environment.trajectory_lengths.max())
    for step_index in range(max_steps):
        if not np.any(active):
            break
        previous_observation = np.asarray(
            environment.last_observation.policy_input, dtype=np.float64
        ).copy()
        stepper.step()
        transition = environment.last_transition
        if transition is None:
            raise RuntimeError("checkpoint exporter requires transition diagnostics")
        frame_contacts, frame_hand_floor = _materialized_contacts(environment)
        for env_id in np.flatnonzero(active):
            if bool(transition.reset_applied[env_id]):
                raise RuntimeError(
                    f"active env {env_id} reset before its candidate episode terminated"
                )
            post = _state_row(environment, int(env_id))
            _append_state(state_storage[env_id], post)
            state_storage[env_id]["urdf_dof_target"].append(
                np.asarray(
                    transition.controller_targets[env_id, right_model_slice],
                    dtype=np.float64,
                )
            )
            contact_storage[env_id].append(frame_contacts[env_id])
            if (
                approach_prefix_config is not None
                and int(transition.command_reference_indices[env_id])
                < int(environment.early_phase_lengths[env_id])
            ):
                if bool(frame_hand_floor[env_id]):
                    prefix_valid[env_id] = False
                    prefix_invalid_reasons[env_id] = "augmentation_prefix_hand_table_clearance"
            command_index = int(transition.command_reference_indices[env_id])
            rollout = rollout_storage[env_id]
            rollout["observation_t"].append(previous_observation[env_id])
            rollout["next_observation"].append(
                np.asarray(
                    transition.observation.policy_input[env_id], dtype=np.float64
                )
            )
            rollout["policy_mean_action"].append(
                np.asarray(transition.raw_actions[env_id], dtype=np.float64)
            )
            rollout["processed_action"].append(
                np.asarray(transition.processed_actions[env_id], dtype=np.float64)
            )
            rollout["cumulative_position_residual"].append(
                np.asarray(
                    environment.cumulative_offset_by_side["right"][env_id],
                    dtype=np.float64,
                )
            )
            rollout["cumulative_joint_residual"].append(
                np.asarray(
                    environment.cumulative_joint_offset[env_id], dtype=np.float64
                ).copy()
            )
            rollout["command_reference_index"].append(command_index)
            rollout["command_source_frame_index"].append(
                int(environment.reference_source_indices[env_id, command_index])
            )
            rollout["reference_target"].append(
                np.asarray(
                    transition.command_targets[env_id, right_model_slice],
                    dtype=np.float64,
                )
            )
            rollout["processed_target"].append(
                np.asarray(
                    transition.processed_targets[env_id, right_model_slice],
                    dtype=np.float64,
                )
            )
            rollout["controller_target"].append(
                np.asarray(
                    transition.controller_targets[env_id, right_model_slice],
                    dtype=np.float64,
                )
            )
            rollout["reward"].append(float(transition.reward.total[env_id]))
            rollout["raw_contact_reward"].append(
                float(transition.reward.raw_contact[env_id])
            )
            rollout["contact_reward"].append(float(transition.reward.contact[env_id]))
            terminal = bool(transition.termination.reset[env_id])
            rollout["terminated"].append(terminal)
            rollout["termination_reason_code"].append(
                int(transition.termination.reason_code[env_id])
            )
            if terminal:
                identity = environment.trajectories[env_id].identity.identity
                active[env_id] = False
                if bool(transition.termination.success[env_id]) and prefix_valid[env_id]:
                    succeeded[env_id] = True
                elif not prefix_valid[env_id]:
                    failure_reasons[identity] = str(prefix_invalid_reasons[env_id])
                else:
                    failure_reasons[identity] = "deviation_before_source_completion"
        if step_index % 100 == 0 or not np.any(active):
            print(
                json.dumps(
                    {
                        "event": "rollout_attempt_progress",
                        "seed": attempt_seed,
                        "step": step_index + 1,
                        "active": int(np.count_nonzero(active)),
                        "succeeded": int(np.count_nonzero(succeeded)),
                        "failed": len(failure_reasons),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if np.any(active):
        unresolved = [
            environment.trajectories[index].identity.identity
            for index in np.flatnonzero(active)
        ]
        raise RuntimeError(
            f"rollout exceeded source horizon without terminal: {unresolved}"
        )

    rows: dict[str, dict[str, Any]] = {}
    for env_id in np.flatnonzero(succeeded):
        trajectory = environment.trajectories[int(env_id)]
        identity = trajectory.identity.identity
        states = {
            name: np.asarray(values) for name, values in state_storage[env_id].items()
        }
        rollout = {
            name: np.asarray(values) for name, values in rollout_storage[env_id].items()
        }
        source_index, source_metadata = metadata_by_row[trajectory.identity.row_index]
        provenance = {
            **provenance_base,
            "seed": attempt_seed,
            "episode_index": episode_indices[identity],
            "generation_attempt": attempt_numbers[identity],
            "augmentation_identity": (
                None
                if accepted_parent is None
                else (
                    f"{accepted_parent.parent_row_uuid}:seed={attempt_seed}:"
                    f"attempt={attempt_numbers[identity]}"
                )
            ),
        }
        full_row = build_v2_row(
            trajectory=trajectory,
            source_index=source_index,
            source_metadata=source_metadata,
            states=states,
            contacts=contact_storage[env_id],
            rollout=rollout,
            provenance=provenance,
            include_checkpoint_metadata_json=(
                output_format == SYNTHETIC_LANCE_OUTPUT_FORMAT_FULL
            ),
        )
        rows[identity] = (
            full_row
            if output_format == SYNTHETIC_LANCE_OUTPUT_FORMAT_FULL
            else build_compact_row(
                full_row,
                checkpoint_metadata=provenance["checkpoint_metadata"],
                warp_ccd_iterations=checkpoint_options.warp_ccd_iterations,
                warp_ccd_contacts_per_world=checkpoint_options.warp_ccd_contacts_per_world,
            )
        )
    accepted_augmentations = {
        identity: approach_samples[identity].to_manifest()
        for identity in rows
        if identity in approach_samples
    }
    return (
        rows,
        failure_reasons,
        environment.observation_dim,
        environment.action_dim,
        accepted_augmentations,
    )


def _export_isolated_repeated_rollouts(
    *,
    checkpoint: Path,
    output: Path,
    selection: TrajectorySelection,
    num_envs: int,
    device: str,
    output_format: str,
    replace: bool,
    allow_deviation_termination: bool,
    predecoded_manifest: Path | None,
    seed: int,
    episodes_per_identity: int,
    max_attempts_per_identity: int,
    object_xy_offset_m: float = 0.0,
    approach_prefix_config: ApproachPrefixConfig | None = None,
    accepted_parent: AcceptedSyntheticParent | None = None,
) -> dict[str, Any]:
    """Run each attempt round in a fresh process and append accepted rows."""

    checkpoint = _validate_checkpoint_path(checkpoint)
    clock = simulation_clock(selection.resolved_control_fps)
    trajectories = (
        load_assigned_trajectory_batch(selection, num_envs=num_envs)
        if predecoded_manifest is None
        else _load_predecoded_batch(
            selection,
            num_envs=num_envs,
            manifest_path=predecoded_manifest,
            exact_identities=(
                None
                if accepted_parent is None
                else (accepted_parent.source_identity,)
            ),
        )
    )
    identities = [item.identity.identity for item in trajectories.trajectories]
    if len(set(identities)) != len(identities):
        raise ValueError("isolated synthesis requires distinct source identities")
    if trajectories.action_dim != JOINT_DOF or any(
        item.action_layout.controlled_sides != ("right",)
        for item in trajectories.trajectories
    ):
        raise ValueError("checkpoint export requires one controlled right 28D hand")
    if output.exists():
        if not replace:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    building = output.parent / f".{output.name}.building"
    partial = output.parent / f"{output.name}.partial"
    for stale in (building, partial):
        if stale.exists():
            if not replace:
                raise FileExistsError(f"stale synthesis output exists: {stale}")
            shutil.rmtree(stale)
    output.parent.mkdir(parents=True, exist_ok=True)
    control_path = output.parent / f".{output.name}.attempt-control.json"
    accepted_parent_path = output.parent / f".{output.name}.accepted-parent.json"
    if accepted_parent is not None:
        accepted_parent_path.write_text(
            json.dumps(accepted_parent.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    child_manifest = building.parent / f"{building.name}.manifest.json"
    counters = {
        identity: {"attempts": 0, "saved": 0, "failures": []} for identity in identities
    }
    generated_uuids: list[str] = []
    row_source_identities: list[str] = []
    accepted_approach_prefixes: list[dict[str, object]] = []

    for attempt_round in range(1, max_attempts_per_identity + 1):
        pending = [
            identity
            for identity in identities
            if counters[identity]["saved"] < episodes_per_identity
            and counters[identity]["attempts"] < max_attempts_per_identity
        ]
        if not pending:
            break
        for identity in pending:
            counters[identity]["attempts"] += 1
        attempt_seed = seed + attempt_round - 1
        control = {
            "pending_identities": pending,
            "attempt_numbers": {
                identity: counters[identity]["attempts"] for identity in pending
            },
            "episode_indices": {
                identity: counters[identity]["saved"] for identity in pending
            },
            "attempt_seed": attempt_seed,
            "append": building.exists(),
        }
        control_path.write_text(
            json.dumps(control, indent=2, sort_keys=True), encoding="utf-8"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(building),
            "--object",
            selection.object_type,
            "--gesture",
            selection.action_id,
            "--dataset-path",
            str(selection.dataset_path),
            "--dataset-version",
            str(selection.expected_dataset_version),
            "--num-envs",
            str(num_envs),
            "--pair-assignment-cycle",
            str(selection.pair_assignment_cycle),
            "--device",
            device,
            "--output-format",
            output_format,
            "--seed",
            str(attempt_seed),
            "--episodes-per-identity",
            "1",
            "--max-attempts-per-identity",
            "1",
            "--internal-attempt-control",
            str(control_path),
        ]
        if selection.reference_fps is not None:
            command.extend(["--reference-fps", str(selection.reference_fps)])
        if selection.selector is not None:
            command.extend(["--pairs", selection.canonical_selector])
        if predecoded_manifest is not None:
            command.extend(["--predecoded-manifest", str(predecoded_manifest)])
        if object_xy_offset_m > 0.0:
            command.extend(["--object-xy-offset-m", str(object_xy_offset_m)])
        if accepted_parent is not None:
            command.extend(["--accepted-parent", str(accepted_parent_path)])
        if approach_prefix_config is not None:
            command.extend(
                [
                    "--approach-prefix",
                    "--approach-xy-radius-min-m",
                    str(approach_prefix_config.minimum_xy_radius_m),
                    "--approach-xy-radius-max-m",
                    str(approach_prefix_config.maximum_xy_radius_m),
                    "--approach-xy-deg",
                    str(approach_prefix_config.maximum_xy_offset_deg),
                    "--approach-z-offset-min-m",
                    str(approach_prefix_config.minimum_z_offset_m),
                    "--approach-z-offset-max-m",
                    str(approach_prefix_config.maximum_z_offset_m),
                    "--approach-total-pre-min",
                    str(approach_prefix_config.minimum_total_pre_padding),
                    "--approach-total-pre-max",
                    str(approach_prefix_config.maximum_total_pre_padding),
                    "--approach-nominal-speed-m-s",
                    str(approach_prefix_config.nominal_speed_m_s),
                    "--approach-nominal-angular-speed-deg-s",
                    str(approach_prefix_config.nominal_angular_speed_deg_s),
                    "--approach-duration-jitter-fraction",
                    str(approach_prefix_config.duration_jitter_fraction),
                    "--approach-orientation-perturbation-deg",
                    str(approach_prefix_config.maximum_orientation_perturbation_deg),
                    "--approach-vertical-arc-height-m",
                    str(approach_prefix_config.vertical_arc_height_m),
                ]
            )
        if allow_deviation_termination:
            command.append("--allow-deviation-termination")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"isolated synthesis attempt round {attempt_round} exited "
                f"{completed.returncode}; control={control_path}"
            )
        child = json.loads(child_manifest.read_text(encoding="utf-8"))
        child_counters = child["synthesis"]["counters"]
        for identity in pending:
            child_counter = child_counters[identity]
            if child_counter["saved"] == 1:
                counters[identity]["saved"] += 1
            else:
                counters[identity]["failures"].extend(child_counter["failures"])
        generated_uuids.extend(child["generated_uuids"])
        row_source_identities.extend(child["row_source_identities"])
        accepted_approach_prefixes.extend(child.get("approach_prefixes", []))
        print(
            json.dumps(
                {
                    "event": "isolated_synthesis_progress",
                    "attempt_round": attempt_round,
                    "attempt_seed": attempt_seed,
                    "saved_total": len(generated_uuids),
                    "target_total": len(identities) * episodes_per_identity,
                    "completed_identities": sum(
                        counters[identity]["saved"] >= episodes_per_identity
                        for identity in identities
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    complete = all(
        counter["saved"] == episodes_per_identity for counter in counters.values()
    )
    published = output if complete else partial
    if building.exists():
        building.replace(published)
    child_manifest.unlink(missing_ok=True)
    control_path.unlink(missing_ok=True)
    accepted_parent_path.unlink(missing_ok=True)
    checkpoint_sha = file_sha256(checkpoint)
    provenance_base = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_update": _checkpoint_update(checkpoint),
        "checkpoint_metadata": checkpoint_runtime_metadata(checkpoint),
        "software_commit": _software_commit(),
    }
    manifest_path = published.parent / f"{published.name}.manifest.json"
    manifest = {
        "schema": (
            SYNTHETIC_LANCE_CONTRACT
            if output_format == SYNTHETIC_LANCE_OUTPUT_FORMAT_FULL
            else SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT
        ),
        "output_format": output_format,
        "source_schema": SYNTHETIC_LANCE_CONTRACT,
        "force_contract": FORCE_DIRECTION_CONTRACT,
        "reward_contract": REWARD_CONTRACT_ID,
        "ppo_reward_contract": PPO_REWARD_CONTRACT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(published.resolve()),
        "complete": complete,
        "rows": len(generated_uuids),
        "projection": _projection_manifest(output_format),
        "source_identities": identities,
        "row_source_identities": row_source_identities,
        "generated_uuids": generated_uuids,
        "approach_prefixes": accepted_approach_prefixes,
        "accepted_parent": (
            None if accepted_parent is None else accepted_parent.to_dict()
        ),
        "checkpoint": provenance_base,
        "selection": {
            "object": selection.object_type,
            "gesture": selection.action_id,
            "selector": selection.canonical_selector,
            "dataset_path": str(selection.dataset_path),
            "dataset_version": selection.expected_dataset_version,
            "num_source_identities": num_envs,
            "pair_assignment_cycle": selection.pair_assignment_cycle,
            "predecoded_manifest": (
                None if predecoded_manifest is None else str(predecoded_manifest)
            ),
        },
        "synthesis": {
            "episodes_per_identity": episodes_per_identity,
            "max_attempts_per_identity": max_attempts_per_identity,
            "base_seed": seed,
            "counters": counters,
            "attempt_isolation": "one_fresh_process_per_attempt_round",
            "approach_prefix": (
                None
                if approach_prefix_config is None
                else {
                    "contract": APPROACH_PREFIX_CONTRACT,
                    "config": asdict(approach_prefix_config),
                    "reset_semantics": "fresh_seeded_reference_per_isolated_attempt",
                    "residual_gate": "zero_through_prefix_then_checkpoint_policy",
                    "deviation_terminal_gate": "disabled_through_prefix_then_0p10m_at_original_pre60_start",
                    "accepted_rows": accepted_approach_prefixes,
                }
            ),
        },
        "runtime": {
            "device": device,
            "policy_mode": "deterministic_mean",
            "reference_fps": selection.reference_fps,
            "control_fps": clock.policy_fps,
            "control_timestep_seconds": clock.control_timestep,
            "physics_fps": clock.physics_fps,
            "physics_timestep_seconds": clock.physics_timestep,
            "physics_substeps_per_control": clock.physics_substeps_per_control,
            "normal_force_scale": 1.0,
            "deviation_termination": (
                allow_deviation_termination or approach_prefix_config is not None
            ),
            "late_contact_grace_frames": 10,
            "late_contact_penalty_multiplier": 3.0,
            "late_contact_scope": "any_16_keypoint_hand_object_contact_above_0p2N",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if complete:
        import lance

        if lance.dataset(str(output)).count_rows() != len(generated_uuids):
            raise RuntimeError("published repeated Lance row count is incorrect")
    else:
        incomplete = {
            identity: counter
            for identity, counter in counters.items()
            if counter["saved"] < episodes_per_identity
        }
        raise RuntimeError(
            "synthesis target incomplete after bounded isolated attempts; "
            f"partial={published}, manifest={manifest_path}, incomplete={incomplete}"
        )
    return {
        "output": str(output),
        "manifest": str(manifest_path),
        "rows": len(generated_uuids),
        "source_identities": identities,
        "episodes_per_identity": episodes_per_identity,
        "max_attempts_per_identity": max_attempts_per_identity,
        "checkpoint_sha256": checkpoint_sha,
        "approach_prefixes": accepted_approach_prefixes,
    }


def export_checkpoint_rollouts(
    *,
    checkpoint: Path,
    output: Path,
    selection: TrajectorySelection,
    num_envs: int,
    device: str,
    replace: bool = False,
    allow_deviation_termination: bool = False,
    predecoded_manifest: Path | None = None,
    seed: int = 42,
    episodes_per_identity: int = 5,
    max_attempts_per_identity: int = 10,
    output_format: str = SYNTHETIC_LANCE_OUTPUT_FORMAT_FULL,
    internal_attempt_control: dict[str, Any] | None = None,
    object_xy_offset_m: float = 0.0,
    approach_prefix_config: ApproachPrefixConfig | None = None,
    accepted_parent: AcceptedSyntheticParent | None = None,
) -> dict[str, Any]:
    """Generate an accepted 1:N checkpoint rollout dataset per raw identity."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if (
        not isinstance(object_xy_offset_m, (int, float))
        or isinstance(object_xy_offset_m, bool)
        or not np.isfinite(object_xy_offset_m)
        or object_xy_offset_m < 0.0
    ):
        raise ValueError("object_xy_offset_m must be a finite non-negative float")
    if (approach_prefix_config is None) != (accepted_parent is None):
        raise ValueError(
            "approach-prefix synthesis requires an accepted-parent descriptor"
        )
    if accepted_parent is not None:
        if num_envs != 1:
            raise ValueError("accepted-parent synthesis requires num-envs=1")
        if object_xy_offset_m != 0.0:
            raise ValueError(
                "accepted-parent synthesis reuses its exact object XY offset"
            )
        if selection.expected_dataset_version != accepted_parent.source_dataset_version:
            raise ValueError("accepted parent source dataset version differs from selection")
        if selection.dataset_path != Path(accepted_parent.source_dataset_path):
            raise ValueError("accepted parent source dataset path differs from selection")
        pair = f"{accepted_parent.object_type}:{accepted_parent.action_id}"
        requested = selection.requested_pairs
        if len(requested) != 1 or requested[0].canonical != pair:
            raise ValueError("accepted parent source pair differs from selection")
    if episodes_per_identity < 1:
        raise ValueError("episodes_per_identity must be positive")
    if max_attempts_per_identity < episodes_per_identity:
        raise ValueError(
            "max attempts must be at least the target episodes per identity"
        )
    output_format = _validate_output_format(output_format)
    checkpoint = _validate_checkpoint_path(checkpoint)
    checkpoint_options = _checkpoint_environment_options(checkpoint)
    if accepted_parent is not None:
        if file_sha256(checkpoint) != accepted_parent.checkpoint_sha256:
            raise ValueError(
                "accepted parent checkpoint SHA256 differs from requested checkpoint"
            )
        if checkpoint_options.reference_fps != accepted_parent.reference_fps:
            raise ValueError("accepted parent reference FPS differs from checkpoint")
    reference_fps = _resolve_reference_fps(
        selection.reference_fps,
        checkpoint_options=checkpoint_options,
        has_checkpoint=True,
    )
    reference_pre_padding = (
        approach_prefix_config.required_base_pre_padding
        if approach_prefix_config is not None
        else checkpoint_options.pre_padding
    )
    selection = dataclass_replace(
        selection,
        reference_fps=reference_fps,
        control_fps=checkpoint_options.control_fps,
        pre_padding=reference_pre_padding,
        post_padding=checkpoint_options.post_padding,
    )
    clock = simulation_clock(selection.resolved_control_fps)
    if episodes_per_identity > 1 and internal_attempt_control is None:
        return _export_isolated_repeated_rollouts(
            checkpoint=checkpoint,
            output=output,
            selection=selection,
            num_envs=num_envs,
            device=device,
            output_format=output_format,
            replace=replace,
            allow_deviation_termination=allow_deviation_termination,
            predecoded_manifest=predecoded_manifest,
            seed=seed,
            episodes_per_identity=episodes_per_identity,
            max_attempts_per_identity=max_attempts_per_identity,
            object_xy_offset_m=object_xy_offset_m,
            approach_prefix_config=approach_prefix_config,
            accepted_parent=accepted_parent,
        )
    trajectories = (
        load_assigned_trajectory_batch(selection, num_envs=num_envs)
        if predecoded_manifest is None
        else _load_predecoded_batch(
            selection,
            num_envs=num_envs,
            manifest_path=predecoded_manifest,
            exact_identities=(
                None
                if accepted_parent is None
                else (accepted_parent.source_identity,)
            ),
        )
    )
    identities = [item.identity.identity for item in trajectories.trajectories]
    if internal_attempt_control is not None:
        requested = list(internal_attempt_control["pending_identities"])
        available = {item.identity.identity: item for item in trajectories.trajectories}
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(
                f"internal attempt references unknown identities: {missing}"
            )
        trajectories = _trajectory_subset(
            trajectories, [available[identity] for identity in requested]
        )
        identities = requested
    if len(set(identities)) != len(identities):
        raise ValueError(
            "num-envs exceeds the distinct valid identities in this assignment window; "
            "reduce it so each synthesis counter owns one raw trajectory"
        )
    if trajectories.action_dim != JOINT_DOF or any(
        item.action_layout.controlled_sides != ("right",)
        for item in trajectories.trajectories
    ):
        raise ValueError("checkpoint export requires one controlled right 28D hand")
    metadata_by_row = _source_metadata(
        trajectories,
        lineage_by_row=(
            _manifest_lineage_by_row(predecoded_manifest)
            if predecoded_manifest is not None
            else None
        ),
    )
    checkpoint_sha = file_sha256(checkpoint)
    provenance_base = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_update": _checkpoint_update(checkpoint),
        "checkpoint_metadata": checkpoint_runtime_metadata(checkpoint),
        "software_commit": _software_commit(),
    }
    counters = {
        identity: {"attempts": 0, "saved": 0, "failures": []} for identity in identities
    }
    trajectory_by_identity = {
        item.identity.identity: item for item in trajectories.trajectories
    }
    rows: list[dict[str, Any]] = []
    accepted_approach_prefixes: list[dict[str, object]] = []
    observation_dim: int | None = None
    action_dim: int | None = None

    for attempt_round in range(1, max_attempts_per_identity + 1):
        pending = [
            identity
            for identity in identities
            if counters[identity]["saved"] < episodes_per_identity
            and counters[identity]["attempts"] < max_attempts_per_identity
        ]
        if not pending:
            break
        for identity in pending:
            counters[identity]["attempts"] += 1
        attempt_numbers = {
            identity: int(counters[identity]["attempts"]) for identity in pending
        }
        episode_indices = {
            identity: int(counters[identity]["saved"]) for identity in pending
        }
        attempt_seed = seed + attempt_round - 1
        if internal_attempt_control is not None:
            attempt_numbers = {
                identity: int(internal_attempt_control["attempt_numbers"][identity])
                for identity in pending
            }
            episode_indices = {
                identity: int(internal_attempt_control["episode_indices"][identity])
                for identity in pending
            }
            attempt_seed = int(internal_attempt_control["attempt_seed"])
        (
            attempt_rows,
            failures,
            observed_dim,
            acted_dim,
            attempt_augmentations,
        ) = _run_attempt_batch(
            checkpoint=checkpoint,
            checkpoint_options=checkpoint_options,
            trajectories=_trajectory_subset(
                trajectories, [trajectory_by_identity[identity] for identity in pending]
            ),
            metadata_by_row=metadata_by_row,
            device=device,
            output_format=output_format,
            allow_deviation_termination=allow_deviation_termination,
            attempt_seed=attempt_seed,
            attempt_numbers=attempt_numbers,
            episode_indices=episode_indices,
            provenance_base=provenance_base,
            object_xy_offset_m=object_xy_offset_m,
            approach_prefix_config=approach_prefix_config,
            accepted_parent=accepted_parent,
        )
        observation_dim = observed_dim
        action_dim = acted_dim
        accepted_approach_prefixes.extend(attempt_augmentations.values())
        for identity in pending:
            row = attempt_rows.get(identity)
            if row is not None:
                rows.append(row)
                counters[identity]["saved"] += 1
            else:
                counters[identity]["failures"].append(
                    failures.get(identity, "candidate_not_accepted")
                )
        print(
            json.dumps(
                {
                    "event": "synthesis_progress",
                    "attempt_round": attempt_round,
                    "attempt_seed": attempt_seed,
                    "saved_total": len(rows),
                    "target_total": len(identities) * episodes_per_identity,
                    "completed_identities": sum(
                        counters[identity]["saved"] >= episodes_per_identity
                        for identity in identities
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    complete = all(
        counter["saved"] == episodes_per_identity for counter in counters.values()
    )
    target_output = (
        output
        if internal_attempt_control is not None or complete
        else output.parent / f"{output.name}.partial"
    )
    append_output = bool(
        internal_attempt_control is not None
        and internal_attempt_control.get("append", False)
    )
    if target_output.exists() and not replace and not append_output:
        raise FileExistsError(f"output already exists: {target_output}")
    previous_rows = 0
    if append_output:
        import lance

        previous_rows = int(lance.dataset(str(target_output)).count_rows())
    if rows:
        if output_format == SYNTHETIC_LANCE_OUTPUT_FORMAT_FULL:
            assert observation_dim is not None and action_dim is not None
            write_v2_lance(
                rows,
                output=target_output,
                observation_dim=observation_dim,
                action_dim=action_dim,
                replace=replace,
                append=append_output,
            )
        else:
            write_compact_lance(
                rows,
                output=target_output,
                replace=replace,
                append=append_output,
            )
        import lance

        dataset = lance.dataset(str(target_output))
        if dataset.count_rows() != previous_rows + len(rows):
            raise RuntimeError(
                "written Lance row count differs from accepted rollout count"
            )
    manifest_path = target_output.parent / f"{target_output.name}.manifest.json"
    manifest = {
        "schema": (
            SYNTHETIC_LANCE_CONTRACT
            if output_format == SYNTHETIC_LANCE_OUTPUT_FORMAT_FULL
            else SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT
        ),
        "output_format": output_format,
        "source_schema": SYNTHETIC_LANCE_CONTRACT,
        "force_contract": FORCE_DIRECTION_CONTRACT,
        "reward_contract": REWARD_CONTRACT_ID,
        "ppo_reward_contract": PPO_REWARD_CONTRACT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(target_output.resolve()),
        "complete": complete,
        "rows": len(rows),
        "projection": _projection_manifest(output_format),
        "source_identities": identities,
        "row_source_identities": [row["provenance"]["source_identity"] for row in rows],
        "generated_uuids": [row["index"]["uuid"] for row in rows],
        "approach_prefixes": accepted_approach_prefixes,
        "accepted_parent": (
            None if accepted_parent is None else accepted_parent.to_dict()
        ),
        "checkpoint": provenance_base,
        "selection": {
            "object": selection.object_type,
            "gesture": selection.action_id,
            "selector": selection.canonical_selector,
            "dataset_path": str(selection.dataset_path),
            "dataset_version": selection.expected_dataset_version,
            "num_source_identities": num_envs,
            "pair_assignment_cycle": selection.pair_assignment_cycle,
            "predecoded_manifest": (
                None if predecoded_manifest is None else str(predecoded_manifest)
            ),
        },
        "synthesis": {
            "episodes_per_identity": episodes_per_identity,
            "max_attempts_per_identity": max_attempts_per_identity,
            "base_seed": seed,
            "counters": counters,
            "approach_prefix": (
                None
                if approach_prefix_config is None
                else {
                    "contract": APPROACH_PREFIX_CONTRACT,
                    "config": asdict(approach_prefix_config),
                    "reset_semantics": "fresh_seeded_reference_per_attempt",
                    "residual_gate": "zero_through_prefix_then_checkpoint_policy",
                    "deviation_terminal_gate": "disabled_through_prefix_then_0p10m_at_original_pre60_start",
                    "accepted_rows": accepted_approach_prefixes,
                }
            ),
        },
        "runtime": {
            "device": device,
            "policy_mode": "deterministic_mean",
            "reference_fps": selection.reference_fps,
            "control_fps": clock.policy_fps,
            "control_timestep_seconds": clock.control_timestep,
            "physics_fps": clock.physics_fps,
            "physics_timestep_seconds": clock.physics_timestep,
            "physics_substeps_per_control": clock.physics_substeps_per_control,
            "normal_force_scale": 1.0,
            "deviation_termination": (
                allow_deviation_termination or approach_prefix_config is not None
            ),
            "late_contact_grace_frames": 10,
            "late_contact_penalty_multiplier": 3.0,
            "late_contact_scope": "any_16_keypoint_hand_object_contact_above_0p2N",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not complete and internal_attempt_control is None:
        incomplete = {
            identity: counter
            for identity, counter in counters.items()
            if counter["saved"] < episodes_per_identity
        }
        raise RuntimeError(
            "synthesis target incomplete after bounded attempts; "
            f"partial={target_output}, manifest={manifest_path}, incomplete={incomplete}"
        )
    return {
        "output": str(target_output),
        "manifest": str(manifest_path),
        "rows": len(rows),
        "source_identities": identities,
        "accepted_source_identities": [
            row["provenance"]["source_identity"] for row in rows
        ],
        "generated_uuids": [row["index"]["uuid"] for row in rows],
        "episodes_per_identity": episodes_per_identity,
        "max_attempts_per_identity": max_attempts_per_identity,
        "checkpoint_sha256": checkpoint_sha,
        "approach_prefixes": accepted_approach_prefixes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object", dest="object_type", default="cube2")
    parser.add_argument("--gesture", default="02")
    parser.add_argument(
        "--pairs",
        help="optional comma-separated homogeneous object:action selector",
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-version", type=int, default=295)
    parser.add_argument(
        "--reference-fps",
        type=int,
        choices=SUPPORTED_REFERENCE_FPS,
        help="source trajectory clock; omitted values restore the checkpoint sidecar",
    )
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--pair-assignment-cycle", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--output-format",
        choices=SYNTHETIC_LANCE_OUTPUT_FORMATS,
        default=SYNTHETIC_LANCE_OUTPUT_FORMAT_FULL,
        help="full v2.3 audit rows or compact replay/visual rows",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes-per-identity", type=int, default=5)
    parser.add_argument("--max-attempts-per-identity", type=int, default=10)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--internal-attempt-control", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--predecoded-manifest",
        type=Path,
        help="optional isolated-predecode manifest for native-Lance-unstable hosts",
    )
    parser.add_argument(
        "--allow-deviation-termination",
        action="store_true",
        help="retain the training 0.10 m deviation terminal instead of requiring full source-length episodes",
    )
    parser.add_argument(
        "--object-xy-offset-m",
        type=float,
        default=0.0,
        help="per-attempt uniform XY object initial-position offset range in meters (default: 0)",
    )
    parser.add_argument(
        "--approach-prefix",
        action="store_true",
        help="prepend a fresh seeded human-like approach reference for every synthesis attempt",
    )
    parser.add_argument(
        "--accepted-parent",
        type=Path,
        help="JSON descriptor of one accepted prior scalable-synthesis row",
    )
    parser.add_argument("--approach-xy-radius-min-m", type=float, default=0.30)
    parser.add_argument("--approach-xy-radius-max-m", type=float, default=0.70)
    parser.add_argument("--approach-xy-deg", type=float, default=30.0)
    parser.add_argument("--approach-z-offset-min-m", type=float, default=0.08)
    parser.add_argument("--approach-z-offset-max-m", type=float, default=0.30)
    parser.add_argument("--approach-total-pre-min", type=int, default=100)
    parser.add_argument("--approach-total-pre-max", type=int, default=360)
    parser.add_argument("--approach-nominal-speed-m-s", type=float, default=0.30)
    parser.add_argument("--approach-nominal-angular-speed-deg-s", type=float, default=90.0)
    parser.add_argument("--approach-duration-jitter-fraction", type=float, default=0.10)
    parser.add_argument("--approach-orientation-perturbation-deg", type=float, default=12.0)
    parser.add_argument("--approach-vertical-arc-height-m", type=float, default=0.04)
    args = parser.parse_args(argv)
    if args.num_envs < 1:
        parser.error("num-envs must be positive")
    if args.dataset_version < 1:
        parser.error("dataset-version must be positive")
    if args.pair_assignment_cycle < 0:
        parser.error("pair-assignment-cycle must be non-negative")
    if args.seed < 0:
        parser.error("seed must be non-negative")
    if args.episodes_per_identity < 1:
        parser.error("episodes-per-identity must be positive")
    if args.max_attempts_per_identity < args.episodes_per_identity:
        parser.error("max-attempts-per-identity must be at least episodes-per-identity")
    if (
        not np.isfinite(args.object_xy_offset_m) or args.object_xy_offset_m < 0.0
    ):
        parser.error("object-xy-offset-m must be a finite non-negative float")
    args.accepted_parent_descriptor = (
        None
        if args.accepted_parent is None
        else load_accepted_synthetic_parent(args.accepted_parent)
    )
    if args.approach_prefix != (args.accepted_parent_descriptor is not None):
        parser.error("--approach-prefix and --accepted-parent must be supplied together")
    try:
        args.approach_prefix_config = (
            ApproachPrefixConfig(
                minimum_xy_radius_m=args.approach_xy_radius_min_m,
                maximum_xy_radius_m=args.approach_xy_radius_max_m,
                maximum_xy_offset_deg=args.approach_xy_deg,
                minimum_z_offset_m=args.approach_z_offset_min_m,
                maximum_z_offset_m=args.approach_z_offset_max_m,
                minimum_total_pre_padding=args.approach_total_pre_min,
                maximum_total_pre_padding=args.approach_total_pre_max,
                nominal_speed_m_s=args.approach_nominal_speed_m_s,
                nominal_angular_speed_deg_s=args.approach_nominal_angular_speed_deg_s,
                duration_jitter_fraction=args.approach_duration_jitter_fraction,
                maximum_orientation_perturbation_deg=args.approach_orientation_perturbation_deg,
                vertical_arc_height_m=args.approach_vertical_arc_height_m,
            )
            if args.approach_prefix
            else None
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    internal_control = (
        None
        if args.internal_attempt_control is None
        else json.loads(args.internal_attempt_control.read_text(encoding="utf-8"))
    )
    parent = args.accepted_parent_descriptor
    selection = TrajectorySelection(
        object_type=(parent.object_type if parent is not None else args.object_type),
        gesture=(parent.action_id if parent is not None else args.gesture),
        selector=(
            f"{parent.object_type}:{parent.action_id}"
            if parent is not None
            else args.pairs
        ),
        dataset_path=(
            Path(parent.source_dataset_path)
            if parent is not None
            else args.dataset_path
        ),
        expected_dataset_version=(
            parent.source_dataset_version
            if parent is not None
            else args.dataset_version
        ),
        hand_side="right",
        reference_fps=(parent.reference_fps if parent is not None else args.reference_fps),
        pair_assignment_cycle=args.pair_assignment_cycle,
    )
    result = export_checkpoint_rollouts(
        checkpoint=args.checkpoint,
        output=args.output,
        selection=selection,
        num_envs=args.num_envs,
        device=args.device,
        output_format=args.output_format,
        replace=args.replace,
        allow_deviation_termination=args.allow_deviation_termination,
        predecoded_manifest=args.predecoded_manifest,
        seed=args.seed,
        episodes_per_identity=args.episodes_per_identity,
        max_attempts_per_identity=args.max_attempts_per_identity,
        internal_attempt_control=internal_control,
        object_xy_offset_m=args.object_xy_offset_m,
        approach_prefix_config=args.approach_prefix_config,
        accepted_parent=args.accepted_parent_descriptor,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
