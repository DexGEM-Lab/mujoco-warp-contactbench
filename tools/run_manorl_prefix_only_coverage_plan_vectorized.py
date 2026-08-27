#!/usr/bin/env python3
"""Run prefix-only coverage candidates in homogeneous-object vector waves.

Each wave assigns at most one candidate from each task to one MJX world. Tasks
for different actions, parents, and approach modes may share a wave when their
object type matches. Candidate seeds, parent offsets, collision checks, atomic
acceptance, task journals, UUIDs, and per-task Lance outputs remain independent.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from sim.manorl.abi import TARGET_MAX_DEVIATION_DISTANCE
from sim.manorl.approach_prefix import RetreatSuffixConfig
from sim.manorl.checkpoint import checkpoint_runtime_metadata
from sim.manorl.contracts import JOINT_DOF
from sim.manorl.environment import (
    EnvironmentConfig,
    MujocoManoEnvironment,
    recommended_warp_contact_capacity,
)
from sim.manorl.lance_v2 import (
    build_compact_row,
    build_v2_row,
    file_sha256,
    write_compact_lance,
)
from sim.manorl.observations import SOURCE_ALIGNED_COMPATIBILITY
from sim.manorl.synthetic_parent import (
    ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
    AcceptedSyntheticParent,
    load_accepted_synthetic_parent,
)
from sim.manorl.trajectory import ReferenceTrajectory, TrajectoryBatch
from sim.manorl.view_environment import (
    _build_checkpoint_stepper,
    _checkpoint_environment_options,
    _validate_checkpoint_path,
)
from tools.export_manorl_synthetic_lance import (
    _append_state,
    _augmentation_identity,
    _checkpoint_update,
    _evaluate_collected_candidate,
    _materialized_contacts,
    _seed_attempt,
    _software_commit,
    _state_row,
)
from tools.run_manorl_prefix_only_coverage_plan import (
    RUN_CONTRACT,
    STATUS_CONTRACT,
    _atomic_json,
    _finalize_status,
    _predecoded_source,
    _recover_pending_attempt,
    _sample_trajectory,
    _selected_indices,
    _status_template,
    _task_manifest,
    _task_paths,
    load_plan,
)

VECTOR_RUN_CONTRACT = "manorl_prefix_only_spatial_coverage_vector_run_v1"
WAVE_CONTRACT = "manorl_prefix_only_spatial_coverage_vector_wave_v1"
DEFAULT_BATCH_SIZE = 32


@dataclass
class TaskState:
    task: dict[str, Any]
    parent: AcceptedSyntheticParent
    source: ReferenceTrajectory
    source_index: dict[str, Any]
    source_metadata: dict[str, Any]
    output: Path
    status_path: Path
    manifest_path: Path
    status: dict[str, Any]

    @property
    def task_index(self) -> int:
        return int(self.task["task_index"])

    @property
    def object_type(self) -> str:
        return str(self.task["pair"]).split(":", maxsplit=1)[0]


@dataclass(frozen=True)
class CandidateAssignment:
    task_index: int
    slot_index: int
    fallback_rank: int
    seed: int
    attempt_number: int
    episode_index: int

    def to_dict(self) -> dict[str, int]:
        return {
            "task_index": self.task_index,
            "slot_index": self.slot_index,
            "fallback_rank": self.fallback_rank,
            "seed": self.seed,
            "attempt_number": self.attempt_number,
            "episode_index": self.episode_index,
        }


def _next_candidate(state: TaskState) -> CandidateAssignment | None:
    """Return one task-local candidate, terminalizing exhausted slots in place."""

    for slot in state.task["slots"]:
        slot_index = int(slot["slot_index"])
        key = str(slot_index)
        if key in state.status["accepted"] or key in state.status["exhausted"]:
            continue
        failures = state.status["failures"].setdefault(key, [])
        tried = {int(item["fallback_rank"]) for item in failures}
        candidate = next(
            (
                item
                for item in slot["candidates"]
                if int(item["fallback_rank"]) not in tried
            ),
            None,
        )
        if candidate is None:
            state.status["exhausted"][key] = {
                "slot_index": slot_index,
                "candidate_attempts": len(failures),
                "coverage_cell": slot["cell"],
            }
            _atomic_json(state.status_path, state.status)
            continue
        return CandidateAssignment(
            task_index=state.task_index,
            slot_index=slot_index,
            fallback_rank=int(candidate["fallback_rank"]),
            seed=int(candidate["episode_seed"]),
            attempt_number=int(state.status["attempts_total"]) + 1,
            episode_index=len(state.status["accepted"]),
        )
    _finalize_status(state.status, len(state.task["slots"]))
    _atomic_json(state.status_path, state.status)
    return None


def _planned_candidate(
    task: dict[str, Any], assignment: CandidateAssignment
) -> tuple[dict[str, Any], dict[str, Any]]:
    slot = task["slots"][assignment.slot_index]
    candidate = slot["candidates"][assignment.fallback_rank]
    if (
        int(slot["slot_index"]) != assignment.slot_index
        or int(candidate["fallback_rank"]) != assignment.fallback_rank
        or int(candidate["episode_seed"]) != assignment.seed
    ):
        raise RuntimeError("vector wave assignment no longer matches coverage plan")
    return slot, candidate


def _mark_running(state: TaskState, assignment: CandidateAssignment) -> None:
    if state.status.get("pending_attempt") is not None:
        raise RuntimeError(f"task {state.task_index} already has a pending attempt")
    state.status["attempts_total"] = assignment.attempt_number
    state.status["pending_attempt"] = {
        "phase": "running",
        "slot_index": assignment.slot_index,
        "fallback_rank": assignment.fallback_rank,
        "episode_seed": assignment.seed,
        "episode_index": assignment.episode_index,
        "attempt_number": assignment.attempt_number,
    }
    _atomic_json(state.status_path, state.status)


def _prepare_states(
    plan: dict[str, Any],
    *,
    indices: list[int],
    excluded: set[int],
    checkpoint: Path,
    output_dir: Path,
) -> dict[int, TaskState]:
    states: dict[int, TaskState] = {}
    checkpoint_sha = file_sha256(checkpoint)
    for task_index in indices:
        if task_index in excluded:
            continue
        task = plan["tasks"][task_index]
        parent = load_accepted_synthetic_parent(Path(task["descriptor_path"]))
        if (
            parent.contract != ACCEPTED_SYNTHETIC_PARENT_CONTRACT
            or parent.source_identity != task["source_identity"]
            or parent.checkpoint_sha256 != checkpoint_sha
        ):
            raise ValueError(f"coverage task {task_index} parent binding changed")
        source, source_index, source_metadata = _predecoded_source(
            Path(task["predecoded_manifest"]), str(task["source_identity"])
        )
        output, status_path, manifest_path = _task_paths(output_dir, task)
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file()
            else _status_template(task)
        )
        if (
            status.get("contract") != STATUS_CONTRACT
            or int(status.get("task_index", -1)) != task_index
        ):
            raise ValueError(f"coverage task {task_index} status contract changed")
        status.setdefault("attempts_total", 0)
        status.setdefault("pending_attempt", None)
        _recover_pending_attempt(output=output, status=status)
        _finalize_status(status, len(task["slots"]))
        _atomic_json(status_path, status)
        if not output.exists() and status["accepted"]:
            raise RuntimeError(
                f"coverage task {task_index} records rows but Lance is absent"
            )
        states[task_index] = TaskState(
            task=task,
            parent=parent,
            source=source,
            source_index=source_index,
            source_metadata=source_metadata,
            output=output,
            status_path=status_path,
            manifest_path=manifest_path,
            status=status,
        )
    return states


def _new_batch_runtime(
    trajectories: list[ReferenceTrajectory],
    parents: list[AcceptedSyntheticParent],
    checkpoint: Path,
    *,
    seeds: list[int],
    device: str,
) -> tuple[MujocoManoEnvironment, Any, Any]:
    if not trajectories or len(trajectories) != len(parents) or len(parents) != len(seeds):
        raise ValueError("vector wave trajectories/parents/seeds must align")
    object_types = {item.identity.identity.split("_", maxsplit=1)[0] for item in trajectories}
    if len(object_types) != 1:
        raise ValueError("one vector wave must contain one object type")
    options = _checkpoint_environment_options(checkpoint)
    _seed_attempt(seeds[0], device)
    batch = TrajectoryBatch(tuple(trajectories))
    environment = MujocoManoEnvironment(
        batch,
        EnvironmentConfig(
            device=device,
            num_envs=len(trajectories),
            residual_enabled=True,
            residual_action=options.residual_action,
            compatibility=replace(
                SOURCE_ALIGNED_COMPATIBILITY, movement_pre_padding=60
            ),
            max_deviation_distance=TARGET_MAX_DEVIATION_DISTANCE,
            contact_capacity=recommended_warp_contact_capacity(
                len(trajectories), batch.hand_sides
            ),
            reference_fps=options.reference_fps,
            control_fps=options.control_fps,
            post_padding=options.post_padding,
            warp_ccd_iterations=options.warp_ccd_iterations,
            warp_ccd_contacts_per_world=options.warp_ccd_contacts_per_world,
            hand_side="right",
            object_init_xy_offsets_m=tuple(
                parent.object_init_xy_offset_m for parent in parents
            ),
        ),
    )
    environment.reseed_point_templates_per_env(np.asarray(seeds, dtype=np.int64))
    stepper = _build_checkpoint_stepper(
        environment,
        checkpoint,
        policy_transfer=True,
        policy_enable_steps=environment.early_phase_lengths,
    )
    return environment, stepper, options


def _collect_batch(
    *,
    environment: MujocoManoEnvironment,
    stepper: Any,
    trajectories: list[ReferenceTrajectory],
    parents: list[AcceptedSyntheticParent],
    assignments: list[CandidateAssignment],
    source_indices: list[dict[str, Any]],
    source_metadata: list[dict[str, Any]],
    prefixes: list[Any],
    configs: list[Any],
    modes: list[str],
    provenance_base: dict[str, Any],
    checkpoint_options: Any,
) -> list[dict[str, Any]]:
    count = len(assignments)
    right_model_index = environment.model_hand_sides.index("right")
    right_model_slice = slice(
        right_model_index * JOINT_DOF, (right_model_index + 1) * JOINT_DOF
    )
    states = [defaultdict(list) for _ in range(count)]
    rollouts = [defaultdict(list) for _ in range(count)]
    contact_frames: list[list[list[dict[str, Any]]]] = [[] for _ in range(count)]
    active = np.ones(count, dtype=bool)
    terminal_success = np.zeros(count, dtype=bool)
    termination_reason = np.zeros(count, dtype=np.int64)
    initial_contacts, initial_floor, initial_object = _materialized_contacts(environment)
    prefix_valid = ~(initial_floor | initial_object)
    prefix_reason: list[str | None] = [
        "augmentation_prefix_hand_table_clearance"
        if bool(initial_floor[index])
        else (
            "augmentation_prefix_hand_object_contact"
            if bool(initial_object[index])
            else None
        )
        for index in range(count)
    ]
    for env_id in range(count):
        _append_state(states[env_id], _state_row(environment, env_id))
        states[env_id]["urdf_dof_target"].append(
            np.asarray(
                environment.reference_q_model[env_id, 0, right_model_slice],
                dtype=np.float64,
            )
        )
        contact_frames[env_id].append(initial_contacts[env_id])

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
            raise RuntimeError("vector coverage rollout omitted transition diagnostics")
        frame_contacts, frame_floor, frame_object = _materialized_contacts(environment)
        for raw_env_id in np.flatnonzero(active):
            env_id = int(raw_env_id)
            if bool(transition.reset_applied[env_id]):
                raise RuntimeError(
                    f"active vector coverage env {env_id} reset before terminal"
                )
            _append_state(states[env_id], _state_row(environment, env_id))
            states[env_id]["urdf_dof_target"].append(
                np.asarray(
                    transition.controller_targets[env_id, right_model_slice],
                    dtype=np.float64,
                )
            )
            contact_frames[env_id].append(frame_contacts[env_id])
            if int(transition.command_reference_indices[env_id]) < int(
                environment.early_phase_lengths[env_id]
            ):
                if bool(frame_floor[env_id]):
                    prefix_valid[env_id] = False
                    prefix_reason[env_id] = "augmentation_prefix_hand_table_clearance"
                elif bool(frame_object[env_id]):
                    prefix_valid[env_id] = False
                    prefix_reason[env_id] = "augmentation_prefix_hand_object_contact"
            command_index = int(transition.command_reference_indices[env_id])
            rollout = rollouts[env_id]
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
            rollout["contact_reward"].append(
                float(transition.reward.contact[env_id])
            )
            terminal = bool(transition.termination.reset[env_id])
            rollout["terminated"].append(terminal)
            termination_reason[env_id] = int(
                transition.termination.reason_code[env_id]
            )
            rollout["termination_reason_code"].append(
                int(termination_reason[env_id])
            )
            if terminal:
                terminal_success[env_id] = bool(
                    transition.termination.success[env_id]
                )
                active[env_id] = False
        if step_index % 100 == 0 or not np.any(active):
            print(
                json.dumps(
                    {
                        "event": "vector_coverage_wave_progress",
                        "step": step_index + 1,
                        "worlds": count,
                        "active": int(np.count_nonzero(active)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if np.any(active):
        raise RuntimeError(
            "vector coverage rollout exceeded horizons for envs "
            + repr(np.flatnonzero(active).tolist())
        )

    results: list[dict[str, Any]] = []
    for env_id, assignment in enumerate(assignments):
        trajectory = trajectories[env_id]
        acceptance = _evaluate_collected_candidate(
            trajectory=trajectory,
            state_storage=states[env_id],
            trajectory_complete=bool(terminal_success[env_id]),
            termination_reason_code=int(termination_reason[env_id]),
            contact_frames=contact_frames[env_id],
        )
        additional_failures = (
            [] if bool(prefix_valid[env_id]) else [str(prefix_reason[env_id])]
        )
        accepted = bool(acceptance.accepted and bool(prefix_valid[env_id]))
        diagnostic = {
            "source_identity": trajectory.identity.identity,
            "mode": modes[env_id],
            "seed": assignment.seed,
            "slot_index": assignment.slot_index,
            "episode_index": assignment.episode_index,
            "attempt_number": assignment.attempt_number,
            "fallback_rank": assignment.fallback_rank,
            "accepted": accepted,
            "acceptance": acceptance.to_dict(),
            "additional_failure_reasons": additional_failures,
            "approach_prefix": prefixes[env_id].to_manifest(),
            "execution": {
                "contract": VECTOR_RUN_CONTRACT,
                "wave_size": count,
                "wave_env_id": env_id,
                "point_template_seed": assignment.seed,
            },
        }
        row = None
        if accepted:
            augmentation_identity = _augmentation_identity(
                accepted_parent=parents[env_id],
                episode_seed=assignment.seed,
                attempt_number=assignment.attempt_number,
                episode_index=assignment.episode_index,
                approach_config=configs[env_id],
                approach_sample=prefixes[env_id],
                near_endpoint_config=RetreatSuffixConfig(),
                retreat_config=None,
                retreat_sample=None,
            )
            provenance = {
                **provenance_base,
                "seed": assignment.seed,
                "episode_index": assignment.episode_index,
                "generation_attempt": assignment.attempt_number,
                "augmentation_identity": augmentation_identity,
            }
            full_row = build_v2_row(
                trajectory=trajectory,
                source_index=source_indices[env_id],
                source_metadata=source_metadata[env_id],
                states={
                    name: np.asarray(values)
                    for name, values in states[env_id].items()
                },
                contacts=contact_frames[env_id],
                rollout={
                    name: np.asarray(values)
                    for name, values in rollouts[env_id].items()
                },
                provenance=provenance,
                include_checkpoint_metadata_json=False,
            )
            row = build_compact_row(
                full_row,
                checkpoint_metadata=provenance["checkpoint_metadata"],
                warp_ccd_iterations=checkpoint_options.warp_ccd_iterations,
                warp_ccd_contacts_per_world=(
                    checkpoint_options.warp_ccd_contacts_per_world
                ),
            )
        results.append(
            {
                "assignment": assignment.to_dict(),
                "diagnostic": diagnostic,
                "row": row,
            }
        )
    return results


def _run_internal_wave(control_path: Path) -> int:
    control = json.loads(control_path.read_text(encoding="utf-8"))
    if control.get("contract") != WAVE_CONTRACT:
        raise ValueError("unsupported vector coverage wave control")
    plan_path = Path(control["plan"])
    plan = load_plan(plan_path)
    checkpoint = _validate_checkpoint_path(Path(control["checkpoint"]))
    device = str(control["device"])
    assignments = [CandidateAssignment(**item) for item in control["assignments"]]
    if not assignments:
        raise ValueError("vector coverage wave is empty")
    tasks = [plan["tasks"][item.task_index] for item in assignments]
    object_types = {str(task["pair"]).split(":", maxsplit=1)[0] for task in tasks}
    if len(object_types) != 1 or next(iter(object_types)) != control["object_type"]:
        raise ValueError("vector coverage wave object grouping changed")

    parents: list[AcceptedSyntheticParent] = []
    sources: list[ReferenceTrajectory] = []
    source_indices: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    trajectories: list[ReferenceTrajectory] = []
    prefixes: list[Any] = []
    configs: list[Any] = []
    modes: list[str] = []
    for task, assignment in zip(tasks, assignments, strict=True):
        parent = load_accepted_synthetic_parent(Path(task["descriptor_path"]))
        source, source_index, metadata = _predecoded_source(
            Path(task["predecoded_manifest"]), str(task["source_identity"])
        )
        trajectory, prefix, config = _sample_trajectory(
            source, parent, mode=str(task["mode"]), seed=assignment.seed
        )
        slot, candidate = _planned_candidate(task, assignment)
        planned = candidate["sampled_start"]
        if not (
            np.isclose(prefix.start_xy_radius_m, planned["radius_m"], rtol=0.0, atol=1e-12)
            and np.isclose(prefix.xy_offset_deg, planned["azimuth_offset_deg"], rtol=0.0, atol=1e-12)
            and np.isclose(prefix.start_z_offset_m, planned["z_offset_m"], rtol=0.0, atol=1e-12)
        ):
            raise RuntimeError("coverage-plan sampled start changed")
        del slot
        parents.append(parent)
        sources.append(source)
        source_indices.append(source_index)
        source_metadata.append(metadata)
        trajectories.append(trajectory)
        prefixes.append(prefix)
        configs.append(config)
        modes.append(str(task["mode"]))

    provenance_base = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_update": _checkpoint_update(checkpoint),
        "checkpoint_metadata": checkpoint_runtime_metadata(checkpoint),
        "software_commit": _software_commit(),
    }
    environment, stepper, options = _new_batch_runtime(
        trajectories,
        parents,
        checkpoint,
        seeds=[item.seed for item in assignments],
        device=device,
    )
    results = _collect_batch(
        environment=environment,
        stepper=stepper,
        trajectories=trajectories,
        parents=parents,
        assignments=assignments,
        source_indices=source_indices,
        source_metadata=source_metadata,
        prefixes=prefixes,
        configs=configs,
        modes=modes,
        provenance_base=provenance_base,
        checkpoint_options=options,
    )
    result_path = Path(control["result"])
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(results, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, result_path)
    summary_path = result_path.with_suffix(".json")
    _atomic_json(
        summary_path,
        {
            "contract": WAVE_CONTRACT,
            "object_type": control["object_type"],
            "worlds": len(results),
            "accepted": sum(item["row"] is not None for item in results),
            "assignments": [item.to_dict() for item in assignments],
        },
    )
    return 0


def _apply_wave_results(
    *,
    results: list[dict[str, Any]],
    states: dict[int, TaskState],
) -> set[int]:
    changed: set[int] = set()
    for result in results:
        assignment = CandidateAssignment(**result["assignment"])
        state = states[assignment.task_index]
        pending = state.status.get("pending_attempt") or {}
        if any(
            int(pending.get(name, -1)) != value
            for name, value in (
                ("slot_index", assignment.slot_index),
                ("fallback_rank", assignment.fallback_rank),
                ("episode_seed", assignment.seed),
                ("episode_index", assignment.episode_index),
                ("attempt_number", assignment.attempt_number),
            )
        ):
            raise RuntimeError(
                f"task {assignment.task_index} pending journal changed during wave"
            )
        slot, candidate = _planned_candidate(state.task, assignment)
        diagnostic = result["diagnostic"]
        diagnostic["coverage_cell"] = slot["cell"]
        diagnostic["planned_start"] = candidate["sampled_start"]
        key = str(assignment.slot_index)
        row = result["row"]
        if row is None:
            state.status["failures"].setdefault(key, []).append(diagnostic)
            state.status["pending_attempt"] = None
            _atomic_json(state.status_path, state.status)
            changed.add(assignment.task_index)
            continue
        prefix_manifest = diagnostic["approach_prefix"]
        accepted_record = {
            "slot_index": assignment.slot_index,
            "seed": assignment.seed,
            "fallback_rank": assignment.fallback_rank,
            "uuid": row["index"]["uuid"],
            "augmentation_identity": row["provenance"]["augmentation_identity"],
            "coverage_cell": slot["cell"],
            "planned_start": candidate["sampled_start"],
            "approach_prefix": prefix_manifest,
            "diagnostic": diagnostic,
        }
        state.status["pending_attempt"] = {
            "phase": "accepted_ready",
            "slot_index": assignment.slot_index,
            "fallback_rank": assignment.fallback_rank,
            "episode_seed": assignment.seed,
            "episode_index": assignment.episode_index,
            "attempt_number": assignment.attempt_number,
            "uuid": row["index"]["uuid"],
            "accepted_record": accepted_record,
        }
        _atomic_json(state.status_path, state.status)
        write_compact_lance(
            [row],
            output=state.output,
            replace=False,
            append=state.output.exists(),
        )
        state.status["accepted"][key] = accepted_record
        state.status["pending_attempt"] = None
        _atomic_json(state.status_path, state.status)
        changed.add(assignment.task_index)
        print(
            json.dumps(
                {
                    "event": "vector_coverage_slot_accepted",
                    "task_index": assignment.task_index,
                    "source_identity": state.task["source_identity"],
                    "mode": state.task["mode"],
                    "slot_index": assignment.slot_index,
                    "seed": assignment.seed,
                    "fallback_rank": assignment.fallback_rank,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return changed


def _write_manifests(
    *,
    changed: set[int],
    states: dict[int, TaskState],
    plan_path: Path,
    checkpoint: Path,
    provenance_base: dict[str, Any],
    device: str,
) -> None:
    for task_index in sorted(changed):
        state = states[task_index]
        _finalize_status(state.status, len(state.task["slots"]))
        _atomic_json(state.status_path, state.status)
        manifest = _task_manifest(
            plan_path=plan_path,
            task=state.task,
            parent=state.parent,
            output=state.output,
            status=state.status,
            checkpoint=checkpoint,
            provenance_base=provenance_base,
            device=device,
        )
        manifest["runtime"]["vector_execution_contract"] = VECTOR_RUN_CONTRACT
        _atomic_json(state.manifest_path, manifest)


def _next_wave_index(wave_root: Path) -> int:
    indices = []
    for path in wave_root.glob("wave-*-control.json"):
        try:
            indices.append(int(path.name.split("-")[1]))
        except (IndexError, ValueError):
            continue
    return max(indices, default=0) + 1


def _run_vectorized(
    *,
    plan_path: Path,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    task_indices: str | None,
    excluded: set[int],
    batch_size: int,
    max_waves: int | None,
) -> dict[str, Any]:
    plan = load_plan(plan_path)
    indices = _selected_indices(task_indices, len(plan["tasks"]))
    checkpoint = _validate_checkpoint_path(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = _prepare_states(
        plan,
        indices=indices,
        excluded=excluded,
        checkpoint=checkpoint,
        output_dir=output_dir,
    )
    provenance_base = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_update": _checkpoint_update(checkpoint),
        "checkpoint_metadata": checkpoint_runtime_metadata(checkpoint),
        "software_commit": _software_commit(),
    }
    wave_root = output_dir / "_vector_waves"
    wave_root.mkdir(parents=True, exist_ok=True)
    wave_index = _next_wave_index(wave_root)
    object_order = list(
        dict.fromkeys(states[index].object_type for index in sorted(states))
    )
    waves_run = 0
    stop = False
    while not stop:
        any_wave = False
        for object_type in object_order:
            if max_waves is not None and waves_run >= max_waves:
                stop = True
                break
            assignments: list[CandidateAssignment] = []
            for task_index in sorted(states):
                state = states[task_index]
                if state.object_type != object_type or state.status["attempts_complete"]:
                    continue
                assignment = _next_candidate(state)
                if assignment is not None:
                    assignments.append(assignment)
                if len(assignments) >= batch_size:
                    break
            if not assignments:
                continue
            any_wave = True
            for assignment in assignments:
                _mark_running(states[assignment.task_index], assignment)
            stem = f"wave-{wave_index:06d}-{object_type}"
            control_path = wave_root / f"{stem}-control.json"
            result_path = wave_root / f"{stem}-result.pkl"
            control = {
                "contract": WAVE_CONTRACT,
                "plan": str(plan_path),
                "checkpoint": str(checkpoint),
                "device": device,
                "object_type": object_type,
                "assignments": [item.to_dict() for item in assignments],
                "result": str(result_path),
            }
            _atomic_json(control_path, control)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--internal-wave-control",
                str(control_path),
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0 or not result_path.is_file():
                raise RuntimeError(
                    f"vector coverage wave {wave_index} exited {completed.returncode}"
                )
            with result_path.open("rb") as stream:
                results = pickle.load(stream)
            changed = _apply_wave_results(results=results, states=states)
            _write_manifests(
                changed=changed,
                states=states,
                plan_path=plan_path,
                checkpoint=checkpoint,
                provenance_base=provenance_base,
                device=device,
            )
            result_path.unlink()
            wave_index += 1
            waves_run += 1
        if not any_wave:
            break
    _write_manifests(
        changed=set(states),
        states=states,
        plan_path=plan_path,
        checkpoint=checkpoint,
        provenance_base=provenance_base,
        device=device,
    )
    results = [
        {
            "task_index": state.task_index,
            "pair": state.task["pair"],
            "source_identity": state.task["source_identity"],
            "mode": state.task["mode"],
            "accepted_slots": len(state.status["accepted"]),
            "exhausted_slots": len(state.status["exhausted"]),
            "attempts_complete": bool(state.status["attempts_complete"]),
        }
        for state in sorted(states.values(), key=lambda item: item.task_index)
    ]
    summary = {
        "contract": VECTOR_RUN_CONTRACT,
        "plan": str(plan_path),
        "checkpoint": str(checkpoint),
        "batch_size": batch_size,
        "waves_run": waves_run,
        "stopped_at_max_waves": bool(
            max_waves is not None and waves_run >= max_waves
        ),
        "selected_task_indices": indices,
        "excluded_task_indices": sorted(excluded),
        "results": results,
    }
    _atomic_json(output_dir / "vector-run-summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--task-indices")
    parser.add_argument("--exclude-task-indices", default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-waves",
        type=int,
        help="diagnostic bound; omit for full resumable production",
    )
    parser.add_argument("--internal-wave-control", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.internal_wave_control is None and any(
        value is None for value in (args.plan, args.checkpoint, args.output_dir)
    ):
        parser.error("--plan, --checkpoint, and --output-dir are required")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_waves is not None and args.max_waves < 1:
        parser.error("--max-waves must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.internal_wave_control is not None:
        return _run_internal_wave(args.internal_wave_control.resolve())
    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)
    excluded = set(
        _selected_indices(args.exclude_task_indices, len(plan["tasks"]))
        if args.exclude_task_indices.strip()
        else []
    )
    summary = _run_vectorized(
        plan_path=plan_path,
        checkpoint=args.checkpoint.resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        device=args.device,
        task_indices=args.task_indices,
        excluded=excluded,
        batch_size=args.batch_size,
        max_waves=args.max_waves,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
