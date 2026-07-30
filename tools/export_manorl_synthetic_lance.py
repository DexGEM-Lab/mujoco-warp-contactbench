#!/usr/bin/env python3
"""Export complete deterministic ManoRL checkpoint episodes to corrected v2 Lance."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
from typing import Any

import numpy as np

from sim.manorl.abi import TARGET_MAX_DEVIATION_DISTANCE
from sim.manorl.checkpoint import checkpoint_runtime_metadata
from sim.manorl.contracts import JOINT_DOF
from sim.manorl.environment import (
    EnvironmentConfig,
    MujocoManoEnvironment,
    recommended_warp_contact_capacity,
)
from sim.manorl.lance_v2 import (
    FORCE_DIRECTION_CONTRACT,
    SYNTHETIC_LANCE_V2_CONTRACT,
    build_v2_row,
    corrected_contact_frames,
    file_sha256,
    write_v2_lance,
)
from sim.manorl.trajectory import (
    TrajectoryBatch,
    TrajectorySelection,
    load_assigned_trajectory_batch,
)
from sim.manorl.view_environment import (
    _build_checkpoint_stepper,
    _checkpoint_environment_options,
    _validate_checkpoint_path,
)

DEFAULT_DATASET = Path(
    "/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_guangguan/"
    "human_p1_guangguan_clean.lance"
)


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
    selection: TrajectorySelection, *, num_envs: int, manifest_path: Path
) -> TrajectoryBatch:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(manifest.get("dataset_path", "")) != selection.dataset_path:
        raise ValueError("predecoded manifest source dataset differs from the requested dataset")
    if int(manifest.get("dataset_version", -1)) != selection.expected_dataset_version:
        raise ValueError("predecoded manifest dataset version differs from the requested version")
    if manifest.get("hand_side") != "right":
        raise ValueError("predecoded manifest must use right-hand selection")
    pairs = selection.requested_pairs
    if pairs is None or len(pairs) != 1:
        raise ValueError("predecoded v2 export requires one explicit object/action pair")
    pair = pairs[0]
    records = [
        record for record in manifest.get("valid_records", [])
        if record.get("pair") == pair.canonical
    ]
    if not records:
        raise LookupError(f"predecoded manifest has no valid {pair.canonical} trajectories")
    if num_envs > len(records):
        raise ValueError(
            f"num-envs {num_envs} exceeds {len(records)} distinct predecoded identities "
            f"for {pair.canonical}"
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
        trajectories.append(trajectory)
    return TrajectoryBatch(
        tuple(trajectories),
        resolved_pairs=pairs,
        selection_mode=selection.mode,
        pair_assignment_cycle=selection.pair_assignment_cycle,
    )


def _source_metadata(trajectories: TrajectoryBatch) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
    """Read only lightweight lineage columns from the pinned source version."""

    import lance

    first = trajectories.trajectories[0]
    path = first.identity.dataset_path
    version = first.identity.dataset_version
    indices = sorted({item.identity.row_index for item in trajectories.trajectories})
    dataset = lance.dataset(path, version=version)
    rows = dataset.take(indices, columns=["index", "trajectory_metadata"]).to_pylist()
    if len(rows) != len(indices):
        raise RuntimeError("source Lance did not return every requested lineage row")
    return {
        row_index: (dict(row.get("index") or {}), dict(row.get("trajectory_metadata") or {}))
        for row_index, row in zip(indices, rows, strict=True)
    }


def _materialized_contacts(environment: MujocoManoEnvironment) -> list[list[dict[str, Any]]]:
    state = environment.producer.materialize_state(environment.data)
    buffers = environment.producer.materialize_contact_buffers(
        environment.data, environment.config.num_envs
    )
    producer = environment.producer
    if producer.active_object_geom_ids is not None:
        raise RuntimeError("v2 checkpoint export currently requires one homogeneous object type")
    return corrected_contact_frames(
        buffers=buffers,
        state=state,
        model=environment.model,
        keypoint_geom_ids=producer.keypoint_geom_ids_by_side["right"],
        object_geom_ids=tuple(sorted(producer.object_geom_ids)),
        object_body_id=producer.object_body_id,
        wrist_body_id=producer.keypoint_body_ids[0],
        object_name=environment.object_type,
    )


def _state_row(environment: MujocoManoEnvironment, env_id: int) -> dict[str, np.ndarray]:
    physical = environment.last_physical
    observation = environment.last_observation
    if physical is None or observation is None:
        raise RuntimeError("environment omitted physical/observation state")
    return {
        "hand_position": np.asarray(physical.hand_position[env_id], dtype=np.float64),
        "hand_orientation_xyzw": np.asarray(
            physical.hand_orientation_xyzw[env_id], dtype=np.float64
        ),
        "mano_joint_pos": np.concatenate(
            [
                np.asarray(physical.hand_keypoint_positions[env_id], dtype=np.float64),
                np.asarray(physical.fingertip_positions[env_id], dtype=np.float64),
            ],
            axis=0,
        ),
        "urdf_dof": np.asarray(physical.mano_dof_pos[env_id], dtype=np.float64),
        "object_position": np.asarray(physical.object_position[env_id], dtype=np.float64),
        "object_orientation_xyzw": np.asarray(
            physical.object_orientation_xyzw[env_id], dtype=np.float64
        ),
        "observation": np.asarray(observation.policy_input[env_id], dtype=np.float64),
    }


def _append_state(storage: dict[str, list[np.ndarray]], row: dict[str, np.ndarray]) -> None:
    for name, value in row.items():
        storage[name].append(value.copy())


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
) -> dict[str, Any]:
    """Run one independent source-length episode per assigned trajectory."""

    checkpoint = _validate_checkpoint_path(checkpoint)
    checkpoint_options = _checkpoint_environment_options(checkpoint)
    trajectories = (
        load_assigned_trajectory_batch(selection, num_envs=num_envs)
        if predecoded_manifest is None
        else _load_predecoded_batch(
            selection, num_envs=num_envs, manifest_path=predecoded_manifest
        )
    )
    identities = [item.identity.identity for item in trajectories.trajectories]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "num-envs exceeds the distinct valid identities in this assignment window; "
            "reduce it so every exported Lance row has a unique source trajectory"
        )
    if trajectories.action_dim != JOINT_DOF or any(
        item.action_layout.controlled_sides != ("right",)
        for item in trajectories.trajectories
    ):
        raise ValueError("v2 checkpoint export currently requires one controlled right 28D hand")
    metadata_by_row = _source_metadata(trajectories)
    environment = MujocoManoEnvironment(
        trajectories,
        EnvironmentConfig(
            device=device,
            num_envs=num_envs,
            residual_enabled=True,
            residual_action=checkpoint_options.residual_action,
            max_deviation_distance=(
                TARGET_MAX_DEVIATION_DISTANCE
                if allow_deviation_termination
                else 1_000_000.0
            ),
            contact_capacity=recommended_warp_contact_capacity(
                num_envs, trajectories.hand_sides
            ),
            warp_ccd_iterations=checkpoint_options.warp_ccd_iterations,
            warp_ccd_contacts_per_world=checkpoint_options.warp_ccd_contacts_per_world,
            hand_side="right",
        ),
    )
    stepper = _build_checkpoint_stepper(environment, checkpoint)
    checkpoint_sha = file_sha256(checkpoint)
    checkpoint_metadata = checkpoint_runtime_metadata(checkpoint)
    provenance = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_update": _checkpoint_update(checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "software_commit": _software_commit(),
    }
    right_model_index = environment.model_hand_sides.index("right")
    right_model_slice = slice(
        right_model_index * JOINT_DOF, (right_model_index + 1) * JOINT_DOF
    )
    state_storage = [defaultdict(list) for _ in range(num_envs)]
    rollout_storage = [defaultdict(list) for _ in range(num_envs)]
    contact_storage: list[list[list[dict[str, Any]]]] = [[] for _ in range(num_envs)]
    active = np.ones(num_envs, dtype=bool)

    initial_contacts = _materialized_contacts(environment)
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
        frame_contacts = _materialized_contacts(environment)
        for env_id in np.flatnonzero(active):
            if bool(transition.reset_applied[env_id]):
                raise RuntimeError(
                    f"active env {env_id} reset before its first episode was complete"
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
            command_index = int(transition.command_reference_indices[env_id])
            rollout = rollout_storage[env_id]
            rollout["observation_t"].append(previous_observation[env_id])
            rollout["next_observation"].append(
                np.asarray(transition.observation.policy_input[env_id], dtype=np.float64)
            )
            rollout["policy_mean_action"].append(
                np.asarray(transition.raw_actions[env_id], dtype=np.float64)
            )
            rollout["processed_action"].append(
                np.asarray(transition.processed_actions[env_id], dtype=np.float64)
            )
            rollout["cumulative_position_residual"].append(
                np.asarray(environment.cumulative_offset_by_side["right"][env_id], dtype=np.float64)
            )
            rollout["cumulative_joint_residual"].append(
                np.asarray(environment.cumulative_joint_offset[env_id], dtype=np.float64)
            )
            rollout["command_reference_index"].append(command_index)
            rollout["command_source_frame_index"].append(
                int(environment.reference_source_indices[env_id, command_index])
            )
            rollout["reference_target"].append(
                np.asarray(
                    transition.command_targets[env_id, right_model_slice], dtype=np.float64
                )
            )
            rollout["processed_target"].append(
                np.asarray(
                    transition.processed_targets[env_id, right_model_slice], dtype=np.float64
                )
            )
            rollout["controller_target"].append(
                np.asarray(
                    transition.controller_targets[env_id, right_model_slice], dtype=np.float64
                )
            )
            rollout["reward"].append(float(transition.reward.total[env_id]))
            terminal = bool(transition.termination.reset[env_id])
            rollout["terminated"].append(terminal)
            rollout["termination_reason_code"].append(
                int(transition.termination.reason_code[env_id])
            )
            if terminal:
                if not bool(transition.termination.success[env_id]):
                    raise RuntimeError(
                        f"trajectory {environment.trajectories[env_id].identity.identity} "
                        "terminated by deviation before its complete source episode"
                    )
                active[env_id] = False
        if step_index % 100 == 0 or not np.any(active):
            print(
                json.dumps(
                    {
                        "event": "rollout_progress",
                        "step": step_index + 1,
                        "active": int(np.count_nonzero(active)),
                        "completed": int(num_envs - np.count_nonzero(active)),
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
        raise RuntimeError(f"rollout exceeded source horizon without terminal: {unresolved}")

    rows: list[dict[str, Any]] = []
    for env_id, trajectory in enumerate(environment.trajectories):
        states = {
            name: np.asarray(values)
            for name, values in state_storage[env_id].items()
        }
        rollout = {
            name: np.asarray(values)
            for name, values in rollout_storage[env_id].items()
        }
        source_index, source_metadata = metadata_by_row[trajectory.identity.row_index]
        rows.append(
            build_v2_row(
                trajectory=trajectory,
                source_index=source_index,
                source_metadata=source_metadata,
                states=states,
                contacts=contact_storage[env_id],
                rollout=rollout,
                provenance=provenance,
            )
        )
    write_v2_lance(
        rows,
        output=output,
        observation_dim=environment.observation_dim,
        action_dim=environment.action_dim,
        replace=replace,
    )
    import lance

    dataset = lance.dataset(str(output))
    if dataset.count_rows() != len(rows):
        raise RuntimeError("written Lance row count differs from completed rollout count")
    manifest_path = output.parent / f"{output.name}.manifest.json"
    manifest = {
        "schema": SYNTHETIC_LANCE_V2_CONTRACT,
        "force_contract": FORCE_DIRECTION_CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output.resolve()),
        "rows": len(rows),
        "identities": [item.identity.identity for item in environment.trajectories],
        "generated_uuids": [row["index"]["uuid"] for row in rows],
        "checkpoint": provenance,
        "selection": {
            "object": selection.object_type,
            "gesture": selection.action_id,
            "dataset_path": str(selection.dataset_path),
            "dataset_version": selection.expected_dataset_version,
            "num_envs": num_envs,
            "pair_assignment_cycle": selection.pair_assignment_cycle,
            "predecoded_manifest": (
                None if predecoded_manifest is None else str(predecoded_manifest)
            ),
        },
        "runtime": {
            "device": device,
            "policy_mode": "deterministic_mean",
            "control_timestep_seconds": 0.005,
            "normal_force_scale": 1.0,
            "deviation_termination": allow_deviation_termination,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output": str(output),
        "manifest": str(manifest_path),
        "rows": len(rows),
        "identities": manifest["identities"],
        "checkpoint_sha256": checkpoint_sha,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object", dest="object_type", default="cube2")
    parser.add_argument("--gesture", default="02")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-version", type=int, default=295)
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--pair-assignment-cycle", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--replace", action="store_true")
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
    args = parser.parse_args(argv)
    if args.num_envs < 1:
        parser.error("num-envs must be positive")
    if args.dataset_version < 1:
        parser.error("dataset-version must be positive")
    if args.pair_assignment_cycle < 0:
        parser.error("pair-assignment-cycle must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = export_checkpoint_rollouts(
        checkpoint=args.checkpoint,
        output=args.output,
        selection=TrajectorySelection(
            object_type=args.object_type,
            gesture=args.gesture,
            dataset_path=args.dataset_path,
            expected_dataset_version=args.dataset_version,
            hand_side="right",
            pair_assignment_cycle=args.pair_assignment_cycle,
        ),
        num_envs=args.num_envs,
        device=args.device,
        replace=args.replace,
        allow_deviation_termination=args.allow_deviation_termination,
        predecoded_manifest=args.predecoded_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
