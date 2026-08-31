#!/usr/bin/env python3
"""Run a resumable prefix-only ManoRL spatial-coverage plan.

One task is one accepted parent and one mode. It owns one MJX/checkpoint runtime,
50 Far or 30 Near coverage slots, and up to twelve same-cell fallback seeds per
slot. Every candidate passes the production prefix-collision and three-rule
acceptance gate before it is appended to compact v2_contact Lance.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from sim.manorl.abi import TARGET_MAX_DEVIATION_DISTANCE
from sim.manorl.approach_prefix import (
    APPROACH_PREFIX_CONTRACT,
    APPROACH_PREFIX_ONLY_PRODUCTION_CONTRACT,
    PREFIX_ONLY_AUGMENTATION_IDENTITY_CONTRACT,
    ApproachPrefixConfig,
    ApproachPrefixSample,
    RetreatSuffixConfig,
    augment_trajectory_with_approach_prefix,
    augmentation_stream_seed,
)
from sim.manorl.checkpoint import checkpoint_runtime_metadata
from sim.manorl.contracts import JOINT_DOF
from sim.manorl.environment import (
    EnvironmentConfig,
    MujocoManoEnvironment,
    recommended_warp_contact_capacity,
)
from sim.manorl.lance_v2 import (
    FORCE_DIRECTION_CONTRACT,
    SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT,
    build_compact_row,
    build_v2_row,
    file_sha256,
    write_compact_lance,
)
from sim.manorl.observations import SOURCE_ALIGNED_COMPATIBILITY
from sim.manorl.rewards import PPO_REWARD_CONTRACT_ID, REWARD_CONTRACT_ID
from sim.manorl.synthetic_parent import (
    ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
    AcceptedSyntheticParent,
    load_accepted_synthetic_parent,
)
from sim.manorl.synthesis_acceptance import synthesis_acceptance_manifest
from sim.manorl.trajectory import ReferenceTrajectory, TrajectoryBatch
from sim.manorl.view_environment import (
    _build_checkpoint_stepper,
    _checkpoint_environment_options,
    _reinstall_approach_prefix_reference,
    _validate_checkpoint_path,
)
from tools.build_manorl_prefix_only_coverage_plan import (
    PLAN_CONTRACT,
    _canonical_digest,
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

RUN_CONTRACT = "manorl_prefix_only_spatial_coverage_run_v1"
STATUS_CONTRACT = "manorl_prefix_only_spatial_coverage_task_status_v1"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_plan(path: Path) -> dict[str, Any]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if values.get("contract") != PLAN_CONTRACT:
        raise ValueError(f"unsupported prefix-only coverage plan: {values.get('contract')!r}")
    recorded_digest = values.get("plan_digest")
    digest_payload = dict(values)
    digest_payload.pop("plan_digest", None)
    if recorded_digest != _canonical_digest(digest_payload):
        raise ValueError("coverage plan digest changed")
    tasks = values.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("coverage plan has no tasks")
    seen_indices: set[int] = set()
    seen_seeds: set[int] = set()
    for expected_index, task in enumerate(tasks):
        task_index = task.get("task_index")
        if task_index != expected_index or task_index in seen_indices:
            raise ValueError("coverage task indices must be unique and contiguous")
        seen_indices.add(task_index)
        mode = task.get("mode")
        expected_slots = 50 if mode == "far" else 30 if mode == "near" else None
        slots = task.get("slots")
        if expected_slots is None or not isinstance(slots, list) or len(slots) != expected_slots:
            raise ValueError("coverage task mode/slot count changed")
        if task.get("target_rows") != expected_slots:
            raise ValueError("coverage task target differs from slot count")
        if task.get("formal_random_object_xy_offset_range_m") != 0.0:
            raise ValueError("formal coverage task must disable random object XY sampling")
        for slot_index, slot in enumerate(slots):
            if slot.get("slot_index") != slot_index:
                raise ValueError("coverage slot indices must be contiguous")
            candidates = slot.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 12:
                raise ValueError("coverage slot must have twelve fallback candidates")
            for rank, candidate in enumerate(candidates):
                seed = candidate.get("episode_seed")
                if candidate.get("fallback_rank") != rank:
                    raise ValueError("coverage fallback ranks must be contiguous")
                if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                    raise ValueError("coverage seed must be a non-negative integer")
                if seed in seen_seeds:
                    raise ValueError(f"coverage plan repeats episode seed {seed}")
                seen_seeds.add(seed)
        descriptor_path = Path(str(task.get("descriptor_path") or ""))
        if not descriptor_path.is_file():
            raise FileNotFoundError(f"accepted-parent descriptor is absent for task {task_index}")
        if file_sha256(descriptor_path) != task.get("descriptor_sha256"):
            raise RuntimeError(
                f"accepted-parent descriptor hash changed for task {task_index}"
            )
        if not Path(str(task.get("predecoded_manifest") or "")).is_file():
            raise FileNotFoundError(f"predecoded manifest is absent for task {task_index}")
    return values


def _predecoded_source(
    manifest_path: Path, identity: str, *, required_pre_padding: int = 60
) -> tuple[ReferenceTrajectory, dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        str(record.get("identity")): record
        for record in manifest.get("valid_records", [])
    }
    record = records.get(identity)
    if record is None:
        raise LookupError(f"predecoded manifest omits {identity}")
    path = manifest_path.parent / f"{identity}.pkl"
    if file_sha256(path) != record.get("pickle_sha256"):
        raise RuntimeError(f"predecoded trajectory hash changed: {path}")
    with path.open("rb") as stream:
        source = pickle.load(stream)
    if not isinstance(source, ReferenceTrajectory):
        raise TypeError("predecoded source is not ReferenceTrajectory")
    if (
        source.identity.identity != identity
        or source.movement_start_step != required_pre_padding
    ):
        raise RuntimeError(
            f"predecoded identity or pre{required_pre_padding} contract changed"
        )
    return (
        source,
        dict(record.get("source_index") or {}),
        dict(record.get("trajectory_metadata") or {}),
    )


def _sample_trajectory(
    source: ReferenceTrajectory,
    parent: AcceptedSyntheticParent,
    *,
    mode: str,
    seed: int,
    required_pre_padding: int = 60,
) -> tuple[ReferenceTrajectory, ApproachPrefixSample, ApproachPrefixConfig]:
    config = ApproachPrefixConfig(
        mode=mode, required_base_pre_padding=required_pre_padding
    )
    anchor = int(source.movement_end_step) + parent.retreat_anchor_offset_frames
    if mode == "near":
        if (
            not 0 <= anchor < len(source.q_ref)
            or int(source.source_indices[anchor])
            != int(parent.retreat_anchor_source_frame_index)
        ):
            raise ValueError("Near movement_end+15 anchor disagrees with parent")
    approach_seed = seed if mode == "far" else augmentation_stream_seed(seed, "near-approach")
    trajectory, prefix = augment_trajectory_with_approach_prefix(
        source,
        seed=approach_seed,
        config=config,
        near_anchor_reference_index=anchor if mode == "near" else None,
        near_endpoint_config=RetreatSuffixConfig(),
        start_q_ref_3_28=parent.source_row_frame0_right_q_ref_3_28,
    )
    if trajectory.augmentation_suffix_frames != 0:
        raise RuntimeError("prefix-only trajectory unexpectedly contains a suffix")
    prefix_frames = prefix.prefix_frames
    np.testing.assert_array_equal(trajectory.q_ref[prefix_frames:], source.q_ref)
    np.testing.assert_array_equal(
        trajectory.source_indices[prefix_frames:], source.source_indices
    )
    return trajectory, prefix, config


def _new_runtime(
    trajectory: ReferenceTrajectory,
    parent: AcceptedSyntheticParent,
    checkpoint: Path,
    *,
    seed: int,
    device: str,
    required_pre_padding: int = 60,
) -> tuple[MujocoManoEnvironment, Any, Any]:
    options = _checkpoint_environment_options(checkpoint)
    _seed_attempt(seed, device)
    batch = TrajectoryBatch((trajectory,))
    environment = MujocoManoEnvironment(
        batch,
        EnvironmentConfig(
            device=device,
            num_envs=1,
            residual_enabled=True,
            residual_action=options.residual_action,
            compatibility=replace(
                SOURCE_ALIGNED_COMPATIBILITY,
                movement_pre_padding=required_pre_padding,
            ),
            max_deviation_distance=TARGET_MAX_DEVIATION_DISTANCE,
            contact_capacity=recommended_warp_contact_capacity(1, batch.hand_sides),
            reference_fps=options.reference_fps,
            control_fps=options.control_fps,
            post_padding=options.post_padding,
            warp_ccd_iterations=options.warp_ccd_iterations,
            warp_ccd_contacts_per_world=options.warp_ccd_contacts_per_world,
            hand_side="right",
            object_init_xy_offsets_m=(parent.object_init_xy_offset_m,),
        ),
    )
    stepper = _build_checkpoint_stepper(
        environment,
        checkpoint,
        policy_transfer=True,
        policy_enable_steps=environment.early_phase_lengths,
    )
    _reinstall_approach_prefix_reference(
        environment, stepper, trajectory, seed=seed
    )
    return environment, stepper, options


def _collect_episode(
    environment: MujocoManoEnvironment,
    stepper: Any,
    trajectory: ReferenceTrajectory,
    *,
    parent: AcceptedSyntheticParent,
    checkpoint_options: Any,
    source_index: dict[str, Any],
    source_metadata: dict[str, Any],
    provenance_base: dict[str, Any],
    mode: str,
    seed: int,
    slot_index: int,
    fallback_rank: int,
    episode_index: int,
    attempt_number: int,
    prefix: ApproachPrefixSample,
    config: ApproachPrefixConfig,
    reinstall: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if reinstall:
        _reinstall_approach_prefix_reference(
            environment, stepper, trajectory, seed=seed
        )
    right_model_index = environment.model_hand_sides.index("right")
    right_model_slice = slice(
        right_model_index * JOINT_DOF, (right_model_index + 1) * JOINT_DOF
    )
    state_storage: dict[str, list[np.ndarray]] = defaultdict(list)
    rollout_storage: dict[str, list[Any]] = defaultdict(list)
    contact_storage: list[list[dict[str, Any]]] = []
    contacts, hand_floor, hand_object = _materialized_contacts(environment)
    prefix_valid = not bool(hand_floor[0]) and not bool(hand_object[0])
    prefix_reason = (
        "augmentation_prefix_hand_table_clearance"
        if bool(hand_floor[0])
        else (
            "augmentation_prefix_hand_object_contact"
            if bool(hand_object[0])
            else None
        )
    )
    initial = _state_row(environment, 0)
    _append_state(state_storage, initial)
    state_storage["urdf_dof_target"].append(
        np.asarray(environment.reference_q_model[0, 0, right_model_slice], dtype=np.float64)
    )
    contact_storage.append(contacts[0])
    terminal_seen = False
    terminal_success = False
    termination_reason = 0
    for step_index in range(int(environment.trajectory_lengths[0])):
        previous_observation = np.asarray(
            environment.last_observation.policy_input, dtype=np.float64
        ).copy()
        stepper.step()
        transition = environment.last_transition
        if transition is None or bool(transition.reset_applied[0]):
            raise RuntimeError("coverage rollout reset before explicit terminal")
        frame_contacts, frame_floor, frame_object = _materialized_contacts(environment)
        post = _state_row(environment, 0)
        _append_state(state_storage, post)
        state_storage["urdf_dof_target"].append(
            np.asarray(transition.controller_targets[0, right_model_slice], dtype=np.float64)
        )
        contact_storage.append(frame_contacts[0])
        if int(transition.command_reference_indices[0]) < int(environment.early_phase_lengths[0]):
            if bool(frame_floor[0]):
                prefix_valid = False
                prefix_reason = "augmentation_prefix_hand_table_clearance"
            elif bool(frame_object[0]):
                prefix_valid = False
                prefix_reason = "augmentation_prefix_hand_object_contact"
        command_index = int(transition.command_reference_indices[0])
        rollout_storage["observation_t"].append(previous_observation[0])
        rollout_storage["next_observation"].append(
            np.asarray(transition.observation.policy_input[0], dtype=np.float64)
        )
        rollout_storage["policy_mean_action"].append(
            np.asarray(transition.raw_actions[0], dtype=np.float64)
        )
        rollout_storage["processed_action"].append(
            np.asarray(transition.processed_actions[0], dtype=np.float64)
        )
        rollout_storage["cumulative_position_residual"].append(
            np.asarray(environment.cumulative_offset_by_side["right"][0], dtype=np.float64)
        )
        rollout_storage["cumulative_joint_residual"].append(
            np.asarray(environment.cumulative_joint_offset[0], dtype=np.float64).copy()
        )
        rollout_storage["command_reference_index"].append(command_index)
        rollout_storage["command_source_frame_index"].append(
            int(environment.reference_source_indices[0, command_index])
        )
        rollout_storage["reference_target"].append(
            np.asarray(transition.command_targets[0, right_model_slice], dtype=np.float64)
        )
        rollout_storage["processed_target"].append(
            np.asarray(transition.processed_targets[0, right_model_slice], dtype=np.float64)
        )
        rollout_storage["controller_target"].append(
            np.asarray(transition.controller_targets[0, right_model_slice], dtype=np.float64)
        )
        rollout_storage["reward"].append(float(transition.reward.total[0]))
        rollout_storage["raw_contact_reward"].append(float(transition.reward.raw_contact[0]))
        rollout_storage["contact_reward"].append(float(transition.reward.contact[0]))
        terminal = bool(transition.termination.reset[0])
        rollout_storage["terminated"].append(terminal)
        termination_reason = int(transition.termination.reason_code[0])
        rollout_storage["termination_reason_code"].append(termination_reason)
        if terminal:
            terminal_seen = True
            terminal_success = bool(transition.termination.success[0])
            break
        if step_index % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "coverage_episode_progress",
                        "source_identity": trajectory.identity.identity,
                        "mode": mode,
                        "slot_index": slot_index,
                        "fallback_rank": fallback_rank,
                        "seed": seed,
                        "step": step_index + 1,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not terminal_seen:
        raise RuntimeError("coverage rollout exceeded source horizon without terminal")
    acceptance = _evaluate_collected_candidate(
        trajectory=trajectory,
        state_storage=state_storage,
        trajectory_complete=terminal_success,
        termination_reason_code=termination_reason,
        contact_frames=contact_storage,
    )
    additional_failures = [] if prefix_valid else [str(prefix_reason)]
    accepted = bool(acceptance.accepted and prefix_valid)
    diagnostic = {
        "source_identity": trajectory.identity.identity,
        "mode": mode,
        "seed": seed,
        "slot_index": slot_index,
        "episode_index": episode_index,
        "attempt_number": attempt_number,
        "fallback_rank": fallback_rank,
        "accepted": accepted,
        "acceptance": acceptance.to_dict(),
        "additional_failure_reasons": additional_failures,
        "approach_prefix": prefix.to_manifest(),
    }
    if not accepted:
        return None, diagnostic
    states = {name: np.asarray(values) for name, values in state_storage.items()}
    rollout = {name: np.asarray(values) for name, values in rollout_storage.items()}
    augmentation_identity = _augmentation_identity(
        accepted_parent=parent,
        episode_seed=seed,
        attempt_number=attempt_number,
        episode_index=episode_index,
        approach_config=config,
        approach_sample=prefix,
        near_endpoint_config=RetreatSuffixConfig(),
        retreat_config=None,
        retreat_sample=None,
    )
    provenance = {
        **provenance_base,
        "seed": seed,
        "episode_index": episode_index,
        "generation_attempt": attempt_number,
        "augmentation_identity": augmentation_identity,
    }
    full_row = build_v2_row(
        trajectory=trajectory,
        source_index=source_index,
        source_metadata=source_metadata,
        states=states,
        contacts=contact_storage,
        rollout=rollout,
        provenance=provenance,
        include_checkpoint_metadata_json=False,
    )
    compact = build_compact_row(
        full_row,
        checkpoint_metadata=provenance["checkpoint_metadata"],
        warp_ccd_iterations=checkpoint_options.warp_ccd_iterations,
        warp_ccd_contacts_per_world=checkpoint_options.warp_ccd_contacts_per_world,
    )
    return compact, diagnostic


def _status_template(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": STATUS_CONTRACT,
        "task_index": task["task_index"],
        "pair": task["pair"],
        "source_identity": task["source_identity"],
        "mode": task["mode"],
        "attempts_total": 0,
        "pending_attempt": None,
        "accepted": {},
        "failures": {},
        "exhausted": {},
        "attempts_complete": False,
        "all_slots_succeeded": False,
        "complete": False,
    }


def _finalize_status(status: dict[str, Any], slot_count: int) -> None:
    accepted = set(status["accepted"])
    exhausted = set(status["exhausted"])
    if accepted & exhausted:
        raise ValueError("one coverage slot is both accepted and exhausted")
    expected = {str(index) for index in range(slot_count)}
    if not (accepted | exhausted).issubset(expected):
        raise ValueError("coverage status contains an unknown slot")
    status["attempts_complete"] = accepted | exhausted == expected
    status["all_slots_succeeded"] = accepted == expected
    status["complete"] = status["all_slots_succeeded"]


def _task_paths(output_dir: Path, task: dict[str, Any]) -> tuple[Path, Path, Path]:
    pair_slug = str(task["pair"]).replace(":", "_")
    root = output_dir / "tasks" / pair_slug
    name = f"{task['source_identity']}_{task['mode']}"
    return root / f"{name}.lance", root / f"{name}.status.json", root / f"{name}.lance.manifest.json"


def _task_manifest(
    *,
    plan_path: Path,
    task: dict[str, Any],
    parent: AcceptedSyntheticParent,
    output: Path,
    status: dict[str, Any],
    checkpoint: Path,
    provenance_base: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    attempts = [
        item
        for slot in sorted(status["failures"], key=int)
        for item in status["failures"][slot]
    ] + [
        item["diagnostic"]
        for _, item in sorted(status["accepted"].items(), key=lambda value: int(value[0]))
    ]
    attempts.sort(key=lambda item: (int(item["slot_index"]), int(item["fallback_rank"])))
    accepted = [
        value for _, value in sorted(status["accepted"].items(), key=lambda item: int(item[0]))
    ]
    config = ApproachPrefixConfig(mode=str(task["mode"]))
    return {
        "contract": RUN_CONTRACT,
        "schema": SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT,
        "output_format": "compact-replay-visual",
        "output": str(output.resolve()),
        "complete": bool(status["complete"]),
        "attempts_complete": bool(status["attempts_complete"]),
        "rows": len(accepted),
        "source_identities": [task["source_identity"]],
        "row_source_identities": [task["source_identity"]] * len(accepted),
        "generated_uuids": [item["uuid"] for item in accepted],
        "approach_prefixes": [item["approach_prefix"] for item in accepted],
        "retreat_suffixes": [],
        "accepted_parent": parent.to_dict(),
        "checkpoint": provenance_base,
        "selection": {
            "pair": task["pair"],
            "source_identity": task["source_identity"],
            "predecoded_manifest": task["predecoded_manifest"],
            "coverage_plan": str(plan_path.resolve()),
            "coverage_task_index": task["task_index"],
        },
        "synthesis": {
            "production_contract": APPROACH_PREFIX_ONLY_PRODUCTION_CONTRACT,
            "augmentation_identity_contract": PREFIX_ONLY_AUGMENTATION_IDENTITY_CONTRACT,
            "checkpoint_loading": "policy_transfer",
            "target_rows": task["target_rows"],
            "max_attempts_per_slot": task["fallbacks_per_slot"],
            "counters": {
                task["source_identity"]: {
                    "attempts": int(status["attempts_total"]),
                    "saved": len(accepted),
                    "failures": [
                        reason
                        for item in attempts
                        if not item["accepted"]
                        for reason in (
                            item["additional_failure_reasons"]
                            + item["acceptance"]["failure_reasons"]
                        )[:1]
                    ],
                }
            },
            "object_xy": {
                "formal_random_offset_range_m": 0.0,
                "parent_fixed_offset_m": task["parent_object_init_xy_offset_m"],
                "semantics": "no formal resampling; reuse accepted-parent ABI offset",
            },
            "coverage": {
                "plan_contract": PLAN_CONTRACT,
                "task_index": task["task_index"],
                "mode": task["mode"],
                "slots": task["slots"],
                "accepted_slots": [item["slot_index"] for item in accepted],
                "exhausted_slots": [int(key) for key in sorted(status["exhausted"], key=int)],
            },
            "acceptance_gate": synthesis_acceptance_manifest(),
            "acceptance_attempts": attempts,
            "approach_prefix": {
                "contract": APPROACH_PREFIX_CONTRACT,
                "config": asdict(config),
                "near_endpoint_config": asdict(RetreatSuffixConfig()) if task["mode"] == "near" else None,
                "seed_stream": "episode_seed" if task["mode"] == "far" else "hash(episode_seed,near-approach)",
                "residual_gate": "zero_through_prefix_then_checkpoint_policy",
                "accepted_rows": [item["approach_prefix"] for item in accepted],
            },
            "retreat_suffix": None,
        },
        "runtime": {
            "device": device,
            "policy_mode": "deterministic_mean",
            "reference_fps": 120,
            "control_fps": 120,
            "physics_fps": 480,
            "physics_substeps_per_control": 4,
        },
        "force_contract": FORCE_DIRECTION_CONTRACT,
        "reward_contract": REWARD_CONTRACT_ID,
        "ppo_reward_contract": PPO_REWARD_CONTRACT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _recover_pending_attempt(
    *,
    output: Path,
    status: dict[str, Any],
) -> None:
    pending = status.get("pending_attempt")
    accepted_count = len(status["accepted"])
    output_count = 0
    last_uuid = None
    if output.exists():
        import lance

        dataset = lance.dataset(str(output))
        output_count = int(dataset.count_rows())
        if output_count:
            row = dataset.take([output_count - 1], columns=["index"]).to_pylist()[0]
            last_uuid = (row.get("index") or {}).get("uuid")
    if pending is None:
        if output_count != accepted_count:
            raise RuntimeError("coverage Lance row count disagrees with status")
        return
    if not isinstance(pending, dict):
        raise ValueError("coverage pending attempt must be an object or null")
    phase = pending.get("phase")
    if output_count == accepted_count:
        # No durable row exists. Re-run this deterministic candidate under the
        # same globally unique attempt number.
        status["attempts_total"] = int(pending["attempt_number"]) - 1
        status["pending_attempt"] = None
        return
    if (
        output_count == accepted_count + 1
        and phase == "accepted_ready"
        and last_uuid == pending.get("uuid")
    ):
        key = str(pending["slot_index"])
        if key in status["accepted"] or key in status["exhausted"]:
            raise RuntimeError("pending coverage slot is already terminal")
        status["accepted"][key] = pending["accepted_record"]
        status["pending_attempt"] = None
        return
    raise RuntimeError(
        "coverage pending journal cannot reconcile Lance/status state"
    )


def _run_task(
    plan_path: Path,
    plan: dict[str, Any],
    *,
    task_index: int,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    max_slots: int | None,
    replace_output: bool,
) -> dict[str, Any]:
    task = plan["tasks"][task_index]
    identity = str(task["source_identity"])
    mode = str(task["mode"])
    parent = load_accepted_synthetic_parent(Path(task["descriptor_path"]))
    if parent.contract != ACCEPTED_SYNTHETIC_PARENT_CONTRACT or parent.source_identity != identity:
        raise ValueError("coverage task accepted-parent contract changed")
    checkpoint = _validate_checkpoint_path(checkpoint)
    if file_sha256(checkpoint) != parent.checkpoint_sha256:
        raise ValueError("coverage checkpoint differs from accepted parent")
    source, source_index, source_metadata = _predecoded_source(
        Path(task["predecoded_manifest"]), identity,
        required_pre_padding=int(task.get("required_pre_padding", 60)),
    )
    output, status_path, manifest_path = _task_paths(output_dir, task)
    if replace_output:
        for path in (output, status_path, manifest_path):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    status = json.loads(status_path.read_text()) if status_path.is_file() else _status_template(task)
    if status.get("contract") != STATUS_CONTRACT or status.get("task_index") != task_index:
        raise ValueError("coverage status contract changed")
    status.setdefault("attempts_total", 0)
    status.setdefault("pending_attempt", None)
    _recover_pending_attempt(output=output, status=status)
    _atomic_json(status_path, status)
    if not output.exists() and status["accepted"]:
        raise RuntimeError("coverage status records rows but Lance output is absent")
    options = _checkpoint_environment_options(checkpoint)
    provenance_base = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_update": _checkpoint_update(checkpoint),
        "checkpoint_metadata": checkpoint_runtime_metadata(checkpoint),
        "software_commit": _software_commit(),
    }
    environment = None
    stepper = None
    slots = task["slots"][:max_slots] if max_slots is not None else task["slots"]
    for slot in slots:
        slot_index = int(slot["slot_index"])
        key = str(slot_index)
        if key in status["accepted"] or key in status["exhausted"]:
            continue
        failures = status["failures"].setdefault(key, [])
        tried = {int(item["fallback_rank"]) for item in failures}
        accepted = False
        for candidate in slot["candidates"]:
            fallback_rank = int(candidate["fallback_rank"])
            if fallback_rank in tried:
                continue
            seed = int(candidate["episode_seed"])
            attempt_number = int(status["attempts_total"]) + 1
            episode_index = len(status["accepted"])
            status["attempts_total"] = attempt_number
            status["pending_attempt"] = {
                "phase": "running",
                "slot_index": slot_index,
                "fallback_rank": fallback_rank,
                "episode_seed": seed,
                "episode_index": episode_index,
                "attempt_number": attempt_number,
            }
            _atomic_json(status_path, status)
            trajectory, prefix, config = _sample_trajectory(
                source, parent, mode=mode, seed=seed,
                required_pre_padding=int(task.get("required_pre_padding", 60)),
            )
            planned = candidate["sampled_start"]
            if not (
                np.isclose(prefix.start_xy_radius_m, planned["radius_m"], rtol=0.0, atol=1e-12)
                and np.isclose(prefix.xy_offset_deg, planned["azimuth_offset_deg"], rtol=0.0, atol=1e-12)
                and np.isclose(prefix.start_z_offset_m, planned["z_offset_m"], rtol=0.0, atol=1e-12)
            ):
                raise RuntimeError("coverage-plan sampled start changed")
            if environment is None:
                environment, stepper, options = _new_runtime(
                    trajectory, parent, checkpoint, seed=seed, device=device,
                    required_pre_padding=int(task.get("required_pre_padding", 60)),
                )
                reinstall = False
            else:
                reinstall = True
            row, diagnostic = _collect_episode(
                environment,
                stepper,
                trajectory,
                parent=parent,
                checkpoint_options=options,
                source_index=source_index,
                source_metadata=source_metadata,
                provenance_base=provenance_base,
                mode=mode,
                seed=seed,
                slot_index=slot_index,
                fallback_rank=fallback_rank,
                episode_index=episode_index,
                attempt_number=attempt_number,
                prefix=prefix,
                config=config,
                reinstall=reinstall,
            )
            diagnostic["coverage_cell"] = slot["cell"]
            diagnostic["planned_start"] = candidate["sampled_start"]
            if row is None:
                failures.append(diagnostic)
                status["pending_attempt"] = None
                _atomic_json(status_path, status)
                continue
            accepted_record = {
                "slot_index": slot_index,
                "seed": seed,
                "fallback_rank": fallback_rank,
                "uuid": row["index"]["uuid"],
                "augmentation_identity": row["provenance"]["augmentation_identity"],
                "coverage_cell": slot["cell"],
                "planned_start": candidate["sampled_start"],
                "approach_prefix": prefix.to_manifest(),
                "diagnostic": diagnostic,
            }
            status["pending_attempt"] = {
                "phase": "accepted_ready",
                "slot_index": slot_index,
                "fallback_rank": fallback_rank,
                "episode_seed": seed,
                "episode_index": episode_index,
                "attempt_number": attempt_number,
                "uuid": row["index"]["uuid"],
                "accepted_record": accepted_record,
            }
            _atomic_json(status_path, status)
            write_compact_lance(
                [row], output=output, replace=False, append=output.exists()
            )
            status["accepted"][key] = accepted_record
            status["pending_attempt"] = None
            _atomic_json(status_path, status)
            accepted = True
            print(
                json.dumps(
                    {
                        "event": "coverage_slot_accepted",
                        "task_index": task_index,
                        "source_identity": identity,
                        "mode": mode,
                        "slot_index": slot_index,
                        "seed": seed,
                        "fallback_rank": fallback_rank,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            break
        if not accepted:
            status["exhausted"][key] = {
                "slot_index": slot_index,
                "candidate_attempts": len(failures),
                "coverage_cell": slot["cell"],
            }
            _atomic_json(status_path, status)
    _finalize_status(status, len(task["slots"]))
    _atomic_json(status_path, status)
    manifest = _task_manifest(
        plan_path=plan_path,
        task=task,
        parent=parent,
        output=output,
        status=status,
        checkpoint=checkpoint,
        provenance_base=provenance_base,
        device=device,
    )
    _atomic_json(manifest_path, manifest)
    return {
        "task_index": task_index,
        "pair": task["pair"],
        "source_identity": identity,
        "mode": mode,
        "output": str(output),
        "manifest": str(manifest_path),
        "status": str(status_path),
        "accepted_slots": len(status["accepted"]),
        "exhausted_slots": len(status["exhausted"]),
        "complete": bool(status["complete"]),
    }


def _selected_indices(raw: str | None, total: int) -> list[int]:
    if raw is None:
        return list(range(total))
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if len(values) != len(set(values)) or any(not 0 <= value < total for value in values):
        raise ValueError("task index selection is duplicate or out of range")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--task-indices", help="comma-separated zero-based task indices")
    parser.add_argument("--max-slots", type=int)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--internal-task-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.max_slots is not None and args.max_slots < 1:
        parser.error("--max-slots must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)
    output_dir = args.output_dir.expanduser().resolve()
    if args.internal_task_index is not None:
        result = _run_task(
            plan_path,
            plan,
            task_index=args.internal_task_index,
            checkpoint=args.checkpoint,
            output_dir=output_dir,
            device=args.device,
            max_slots=args.max_slots,
            replace_output=args.replace,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["complete"] else 3
    indices = _selected_indices(args.task_indices, len(plan["tasks"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for task_index in indices:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--plan",
            str(plan_path),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--output-dir",
            str(output_dir),
            "--device",
            args.device,
            "--internal-task-index",
            str(task_index),
        ]
        if args.max_slots is not None:
            command.extend(["--max-slots", str(args.max_slots)])
        if args.replace:
            command.append("--replace")
        completed = subprocess.run(command, check=False)
        if completed.returncode not in (0, 3):
            raise RuntimeError(
                f"coverage task {task_index} exited {completed.returncode}"
            )
        task = plan["tasks"][task_index]
        _, status_path, _ = _task_paths(output_dir, task)
        status = json.loads(status_path.read_text())
        results.append(
            {
                "task_index": task_index,
                "pair": task["pair"],
                "source_identity": task["source_identity"],
                "mode": task["mode"],
                "accepted_slots": len(status["accepted"]),
                "exhausted_slots": len(status["exhausted"]),
                "complete": bool(status["complete"]),
            }
        )
    summary = {
        "contract": RUN_CONTRACT,
        "plan": str(plan_path),
        "checkpoint": str(args.checkpoint.resolve()),
        "selected_task_indices": indices,
        "results": results,
    }
    _atomic_json(output_dir / "run-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
