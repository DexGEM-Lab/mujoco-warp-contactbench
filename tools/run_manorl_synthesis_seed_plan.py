#!/usr/bin/env python3
"""Run an explicit paired near/far ManoRL synthesis seed plan.

One parent worker owns one single-environment MJX/checkpoint runtime. Planned
seeds are installed sequentially without rebuilding Warp. A seed is accepted
only when both its near and far episodes succeed; failed pairs advance to the
next candidate from the same precomputed distance cell.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
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
    RETREAT_SUFFIX_CONTRACT,
    ApproachPrefixConfig,
    ApproachPrefixSample,
    RetreatSuffixConfig,
    RetreatSuffixSample,
    augment_trajectory_with_approach_prefix,
    augment_trajectory_with_retreat_suffix,
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
from sim.manorl.trajectory import ReferenceTrajectory, TrajectoryBatch
from sim.manorl.view_environment import (
    _build_checkpoint_stepper,
    _checkpoint_environment_options,
    _reinstall_approach_prefix_reference,
    _validate_checkpoint_path,
)
from tools.export_manorl_synthetic_lance import (
    AUGMENTATION_IDENTITY_CONTRACT,
    _append_state,
    _augmentation_identity,
    _checkpoint_update,
    _materialized_contacts,
    _seed_attempt,
    _software_commit,
    _state_row,
)

PLAN_CONTRACT = "manorl_banana_vla_paired_seed_fallback_plan_v1"
RUN_CONTRACT = "manorl_synthesis_paired_seed_plan_run_v3"
STATUS_CONTRACT = "manorl_synthesis_parent_seed_status_v3"
# Preserve the established synthesis setting. Yield is measured under this
# fixed 4 cm endpoint-smooth arc; failed candidates consume their bounded
# attempt budget rather than changing physical parameters to meet a quota.
PRODUCTION_APPROACH_VERTICAL_ARC_HEIGHT_M = ApproachPrefixConfig().vertical_arc_height_m


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
        raise ValueError(f"unsupported paired seed plan: {values.get('contract')!r}")
    parents = values.get("parents_plan")
    if not isinstance(parents, list) or not parents:
        raise ValueError("paired seed plan has no parents_plan")
    seen_parent: set[str] = set()
    seen_seed: set[int] = set()
    for parent in parents:
        identity = parent.get("source_identity")
        slots = parent.get("slots")
        if not isinstance(identity, str) or not identity or identity in seen_parent:
            raise ValueError("paired seed plan parent identities must be unique")
        seen_parent.add(identity)
        if not isinstance(slots, list) or len(slots) != 10:
            raise ValueError(f"paired seed plan parent {identity} must have ten slots")
        slot_ranks = []
        for slot in slots:
            slot_ranks.append(slot.get("slot_rank_by_far_distance"))
            candidates = slot.get("candidates")
            if not isinstance(candidates, list) or len(candidates) < 1:
                raise ValueError(f"paired seed plan slot for {identity} has no candidates")
            far_decile = slot.get("target_far_distance_decile")
            near_decile = slot.get("target_near_distance_decile")
            for rank, candidate in enumerate(candidates):
                seed = candidate.get("episode_seed")
                if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                    raise ValueError("paired seed candidate seed must be non-negative int")
                if seed in seen_seed:
                    raise ValueError(f"paired seed plan repeats episode seed {seed}")
                seen_seed.add(seed)
                if candidate.get("fallback_rank") != rank:
                    raise ValueError("paired seed fallback ranks must be contiguous")
                if candidate.get("far_distance_decile") != far_decile or candidate.get(
                    "near_distance_decile"
                ) != near_decile:
                    raise ValueError("fallback candidate left its target distance cell")
        if sorted(slot_ranks) != list(range(1, 11)):
            raise ValueError("paired seed slot ranks must be 1..10")
        descriptor = Path(str(parent.get("descriptor_path") or ""))
        if not descriptor.is_file():
            raise FileNotFoundError(f"accepted-parent descriptor is absent: {descriptor}")
    return values


def _predecoded_source(
    manifest_path: Path, identity: str
) -> tuple[ReferenceTrajectory, dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        str(record.get("identity")): record
        for record in manifest.get("valid_records", [])
    }
    if identity not in records:
        raise LookupError(f"predecoded manifest omits {identity}")
    record = records[identity]
    trajectory_path = manifest_path.parent / f"{identity}.pkl"
    if file_sha256(trajectory_path) != record.get("pickle_sha256"):
        raise RuntimeError(f"predecoded trajectory hash changed: {trajectory_path}")
    with trajectory_path.open("rb") as stream:
        trajectory = pickle.load(stream)
    if not isinstance(trajectory, ReferenceTrajectory):
        raise TypeError("predecoded trajectory is not ReferenceTrajectory")
    if trajectory.identity.identity != identity:
        raise RuntimeError("predecoded trajectory identity changed")
    return (
        trajectory,
        dict(record.get("source_index") or {}),
        dict(record.get("trajectory_metadata") or {}),
    )


def _sample_trajectory(
    source: ReferenceTrajectory,
    parent: AcceptedSyntheticParent,
    *,
    mode: str,
    seed: int,
    approach_config: ApproachPrefixConfig,
) -> tuple[ReferenceTrajectory, ApproachPrefixSample, RetreatSuffixSample]:
    anchor = int(source.movement_end_step) + parent.retreat_anchor_offset_frames
    if (
        not 0 <= anchor < len(source.q_ref)
        or int(source.source_indices[anchor])
        != int(parent.retreat_anchor_source_frame_index)
    ):
        raise ValueError("source movement-end anchor disagrees with accepted parent")
    approach_seed = seed if mode == "far" else augmentation_stream_seed(seed, "near-approach")
    if approach_config.mode != mode:
        raise ValueError("approach configuration mode does not match planned mode")
    augmented, prefix = augment_trajectory_with_approach_prefix(
        source,
        seed=approach_seed,
        config=approach_config,
        near_anchor_reference_index=anchor if mode == "near" else None,
        near_endpoint_config=RetreatSuffixConfig(),
        start_q_ref_3_28=parent.source_row_frame0_right_q_ref_3_28,
    )
    augmented_anchor = (
        int(augmented.movement_end_step) + parent.retreat_anchor_offset_frames
    )
    augmented, suffix = augment_trajectory_with_retreat_suffix(
        augmented,
        seed=seed,
        anchor_reference_index=augmented_anchor,
        config=RetreatSuffixConfig(),
    )
    return augmented, prefix, suffix


def _new_runtime(
    trajectory: ReferenceTrajectory,
    parent: AcceptedSyntheticParent,
    checkpoint: Path,
    *,
    seed: int,
    device: str,
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
                SOURCE_ALIGNED_COMPATIBILITY, movement_pre_padding=60
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
        policy_disable_steps=(
            environment.trajectory_lengths - environment.augmentation_suffix_frames
        ),
    )
    # _build_checkpoint_stepper performs an unseeded bootstrap reset. Replace
    # it with the explicit planned seed before the first recorded transition,
    # matching every later in-place reference reinstall.
    _reinstall_approach_prefix_reference(
        environment,
        stepper,
        trajectory,
        seed=seed,
        policy_disable_steps=(
            environment.trajectory_lengths - environment.augmentation_suffix_frames
        ),
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
    prefix: ApproachPrefixSample,
    suffix: RetreatSuffixSample,
    approach_config: ApproachPrefixConfig,
    reinstall: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    if reinstall:
        _reinstall_approach_prefix_reference(
            environment,
            stepper,
            trajectory,
            seed=seed,
            policy_disable_steps=(
                np.asarray([len(trajectory.q_ref)], dtype=np.int64)
                - np.asarray([trajectory.augmentation_suffix_frames], dtype=np.int64)
            ),
        )
    right_model_index = environment.model_hand_sides.index("right")
    right_model_slice = slice(
        right_model_index * JOINT_DOF, (right_model_index + 1) * JOINT_DOF
    )
    state_storage: dict[str, list[np.ndarray]] = defaultdict(list)
    rollout_storage: dict[str, list[Any]] = defaultdict(list)
    contact_storage: list[list[dict[str, Any]]] = []
    initial_contacts, initial_hand_floor, initial_hand_object = _materialized_contacts(
        environment
    )
    prefix_valid = not bool(initial_hand_floor[0]) and not bool(initial_hand_object[0])
    prefix_reason = (
        "augmentation_prefix_hand_table_clearance"
        if bool(initial_hand_floor[0])
        else (
            "augmentation_prefix_hand_object_contact"
            if bool(initial_hand_object[0])
            else None
        )
    )
    initial = _state_row(environment, 0)
    _append_state(state_storage, initial)
    state_storage["urdf_dof_target"].append(
        np.asarray(environment.reference_q_model[0, 0, right_model_slice], dtype=np.float64)
    )
    contact_storage.append(initial_contacts[0])
    terminal_success = False
    terminal_seen = False
    failure: str | None = None
    for step_index in range(int(environment.trajectory_lengths[0])):
        previous_observation = np.asarray(
            environment.last_observation.policy_input, dtype=np.float64
        ).copy()
        stepper.step()
        transition = environment.last_transition
        if transition is None:
            raise RuntimeError("planned synthesis requires transition diagnostics")
        if bool(transition.reset_applied[0]):
            raise RuntimeError("planned synthesis reset before episode terminal")
        frame_contacts, frame_hand_floor, frame_hand_object = _materialized_contacts(
            environment
        )
        post = _state_row(environment, 0)
        _append_state(state_storage, post)
        state_storage["urdf_dof_target"].append(
            np.asarray(
                transition.controller_targets[0, right_model_slice], dtype=np.float64
            )
        )
        contact_storage.append(frame_contacts[0])
        if int(transition.command_reference_indices[0]) < int(
            environment.early_phase_lengths[0]
        ):
            if bool(frame_hand_floor[0]):
                prefix_valid = False
                prefix_reason = "augmentation_prefix_hand_table_clearance"
            elif bool(frame_hand_object[0]):
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
            np.asarray(
                environment.cumulative_offset_by_side["right"][0], dtype=np.float64
            )
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
        rollout_storage["raw_contact_reward"].append(
            float(transition.reward.raw_contact[0])
        )
        rollout_storage["contact_reward"].append(float(transition.reward.contact[0]))
        terminal = bool(transition.termination.reset[0])
        rollout_storage["terminated"].append(terminal)
        rollout_storage["termination_reason_code"].append(
            int(transition.termination.reason_code[0])
        )
        if terminal:
            terminal_seen = True
            terminal_success = bool(transition.termination.success[0]) and prefix_valid
            if not terminal_success:
                failure = (
                    str(prefix_reason)
                    if not prefix_valid
                    else "deviation_before_source_completion"
                )
            break
        if step_index % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "planned_episode_progress",
                        "source_identity": trajectory.identity.identity,
                        "mode": mode,
                        "seed": seed,
                        "slot": slot_index,
                        "fallback_rank": fallback_rank,
                        "step": step_index + 1,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not terminal_seen:
        raise RuntimeError("planned rollout exceeded source horizon without terminal")
    if not terminal_success:
        return None, failure or "candidate_not_accepted"
    states = {name: np.asarray(values) for name, values in state_storage.items()}
    rollout = {name: np.asarray(values) for name, values in rollout_storage.items()}
    augmentation_identity = _augmentation_identity(
        accepted_parent=parent,
        episode_seed=seed,
        attempt_number=fallback_rank + 1,
        episode_index=slot_index,
        approach_config=approach_config,
        approach_sample=prefix,
        near_endpoint_config=RetreatSuffixConfig(),
        retreat_config=RetreatSuffixConfig(),
        retreat_sample=suffix,
    )
    provenance = {
        **provenance_base,
        "seed": seed,
        "episode_index": slot_index,
        "generation_attempt": fallback_rank + 1,
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
    return compact, None


def _status_template(parent_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": STATUS_CONTRACT,
        "source_identity": parent_plan["source_identity"],
        "parent_uuid": parent_plan["parent_uuid"],
        "accepted": {},
        "failures": {},
        "exhausted": {},
        "attempts_complete": False,
        "all_slots_succeeded": False,
        "complete": False,
    }


def _finalize_status(status: dict[str, Any], *, slot_count: int) -> None:
    if slot_count < 1:
        raise ValueError("slot_count must be positive")
    accepted = set(status.get("accepted", {}))
    exhausted = set(status.get("exhausted", {}))
    if accepted & exhausted:
        raise ValueError("one paired slot cannot be both accepted and exhausted")
    terminal_slots = accepted | exhausted
    expected_slots = {str(index) for index in range(slot_count)}
    if not terminal_slots.issubset(expected_slots):
        raise ValueError("paired status contains an unknown slot")
    status["attempts_complete"] = terminal_slots == expected_slots
    status["all_slots_succeeded"] = accepted == expected_slots
    # Preserve exporter semantics: complete means the requested target was
    # fully met. Bounded attempts finishing with partial yield are represented
    # separately by attempts_complete=true and complete=false.
    status["complete"] = status["all_slots_succeeded"]


def _parent_manifest(
    *,
    output: Path,
    parent: AcceptedSyntheticParent,
    checkpoint: Path,
    provenance_base: dict[str, Any],
    status: dict[str, Any],
    approach_vertical_arc_height_m: float,
) -> dict[str, Any]:
    accepted = sorted(
        status["accepted"].values(), key=lambda item: int(item["slot_index"])
    )
    return {
        "contract": RUN_CONTRACT,
        "schema": SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT,
        "output_format": "compact-replay-visual",
        "output": str(output.resolve()),
        "complete": bool(status["complete"]),
        "attempts_complete": bool(status.get("attempts_complete", status["complete"])),
        "all_slots_succeeded": bool(
            status.get("all_slots_succeeded", status["complete"])
        ),
        "accepted_pairs": len(accepted),
        "exhausted_slots": len(status.get("exhausted", {})),
        "rows": len(accepted) * 2,
        "generated_uuids": [
            uuid for item in accepted for uuid in (item["near_uuid"], item["far_uuid"])
        ],
        "row_plan": [
            {
                "slot_index": item["slot_index"],
                "seed": item["seed"],
                "fallback_rank": item["fallback_rank"],
                "near_uuid": item["near_uuid"],
                "far_uuid": item["far_uuid"],
            }
            for item in accepted
        ],
        "accepted_parent": parent.to_dict(),
        "checkpoint": {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": provenance_base["checkpoint_sha256"],
            "checkpoint_update": provenance_base["checkpoint_update"],
            "checkpoint_metadata": provenance_base["checkpoint_metadata"],
        },
        "force_contract": FORCE_DIRECTION_CONTRACT,
        "reward_contract": REWARD_CONTRACT_ID,
        "ppo_reward_contract": PPO_REWARD_CONTRACT_ID,
        "approach_contract": APPROACH_PREFIX_CONTRACT,
        "retreat_contract": RETREAT_SUFFIX_CONTRACT,
        "augmentation_identity_contract": AUGMENTATION_IDENTITY_CONTRACT,
        "approach_vertical_arc_height_m": approach_vertical_arc_height_m,
        "retreat_suffix_required": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _run_parent(
    plan: dict[str, Any],
    *,
    parent_index: int,
    checkpoint: Path,
    predecoded_manifest: Path,
    output_dir: Path,
    device: str,
    max_slots: int | None,
    replace_output: bool,
    approach_vertical_arc_height_m: float,
) -> dict[str, Any]:
    parent_plan = plan["parents_plan"][parent_index]
    identity = str(parent_plan["source_identity"])
    parent = load_accepted_synthetic_parent(parent_plan["descriptor_path"])
    if parent.contract != ACCEPTED_SYNTHETIC_PARENT_CONTRACT:
        raise ValueError("seed plan parent is not accepted-parent v3")
    if parent.source_identity != identity or parent.parent_row_uuid != parent_plan["parent_uuid"]:
        raise ValueError("seed plan parent descriptor identity changed")
    checkpoint = _validate_checkpoint_path(checkpoint)
    if file_sha256(checkpoint) != parent.checkpoint_sha256:
        raise ValueError("seed plan checkpoint differs from accepted parent")
    source, source_index, source_metadata = _predecoded_source(
        predecoded_manifest, identity
    )
    parent_dir = output_dir / "parents"
    parent_dir.mkdir(parents=True, exist_ok=True)
    output = parent_dir / f"{identity}.lance"
    status_path = parent_dir / f"{identity}.status.json"
    manifest_path = output.parent / f"{output.name}.manifest.json"
    if replace_output:
        for path in (output, status_path, manifest_path):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else _status_template(parent_plan)
    )
    if status.get("contract") != STATUS_CONTRACT:
        raise ValueError("parent status contract changed")
    # Status v2 predates partial-yield delivery. Additive fields let an
    # interrupted strict-quota run resume without rewriting accepted rows.
    status.setdefault("exhausted", {})
    status.setdefault("attempts_complete", False)
    status.setdefault("all_slots_succeeded", False)
    if output.exists():
        import lance

        expected_rows = 2 * len(status["accepted"])
        if lance.dataset(str(output)).count_rows() != expected_rows:
            raise RuntimeError("parent output row count disagrees with status")
    elif status["accepted"]:
        raise RuntimeError("parent status records rows but Lance output is absent")
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
    slots = sorted(
        parent_plan["slots"], key=lambda item: int(item["slot_rank_by_far_distance"])
    )
    if max_slots is not None:
        slots = slots[:max_slots]
    for slot in slots:
        slot_index = int(slot["slot_rank_by_far_distance"]) - 1
        key = str(slot_index)
        if key in status["accepted"] or key in status["exhausted"]:
            continue
        slot_failures = status["failures"].setdefault(key, [])
        tried = {int(item["fallback_rank"]) for item in slot_failures}
        accepted = False
        for candidate in slot["candidates"]:
            fallback_rank = int(candidate["fallback_rank"])
            if fallback_rank in tried:
                continue
            seed = int(candidate["episode_seed"])
            pair_rows: dict[str, dict[str, Any]] = {}
            pair_failure: dict[str, str] = {}
            for mode in ("near", "far"):
                approach_config = ApproachPrefixConfig(
                    mode=mode,
                    vertical_arc_height_m=approach_vertical_arc_height_m,
                )
                trajectory, prefix, suffix = _sample_trajectory(
                    source,
                    parent,
                    mode=mode,
                    seed=seed,
                    approach_config=approach_config,
                )
                if environment is None:
                    environment, stepper, options = _new_runtime(
                        trajectory, parent, checkpoint, seed=seed, device=device
                    )
                    reinstall = False
                else:
                    reinstall = True
                row, failure = _collect_episode(
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
                    prefix=prefix,
                    suffix=suffix,
                    approach_config=approach_config,
                    reinstall=reinstall,
                )
                if row is None:
                    pair_failure[mode] = str(failure)
                    break
                pair_rows[mode] = row
            if len(pair_rows) != 2:
                slot_failures.append(
                    {
                        "fallback_rank": fallback_rank,
                        "seed": seed,
                        "failure": pair_failure,
                    }
                )
                _atomic_json(status_path, status)
                continue
            rows = [pair_rows["near"], pair_rows["far"]]
            write_compact_lance(
                rows,
                output=output,
                replace=False,
                append=output.exists(),
            )
            status["accepted"][key] = {
                "slot_index": slot_index,
                "seed": seed,
                "fallback_rank": fallback_rank,
                "far_distance_decile": slot["target_far_distance_decile"],
                "near_distance_decile": slot["target_near_distance_decile"],
                "near_uuid": pair_rows["near"]["index"]["uuid"],
                "far_uuid": pair_rows["far"]["index"]["uuid"],
            }
            _atomic_json(status_path, status)
            accepted = True
            print(
                json.dumps(
                    {
                        "event": "paired_slot_accepted",
                        "parent_index": parent_index,
                        "source_identity": identity,
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
                "candidate_attempts": len(slot_failures),
                "far_distance_decile": slot["target_far_distance_decile"],
                "near_distance_decile": slot["target_near_distance_decile"],
            }
            _atomic_json(status_path, status)
            print(
                json.dumps(
                    {
                        "event": "paired_slot_exhausted",
                        "parent_index": parent_index,
                        "source_identity": identity,
                        "slot_index": slot_index,
                        "candidate_attempts": len(slot_failures),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    _finalize_status(status, slot_count=len(slots))
    _atomic_json(status_path, status)
    manifest = _parent_manifest(
        output=output,
        parent=parent,
        checkpoint=checkpoint,
        provenance_base=provenance_base,
        status=status,
        approach_vertical_arc_height_m=approach_vertical_arc_height_m,
    )
    _atomic_json(manifest_path, manifest)
    return {
        "parent_index": parent_index,
        "source_identity": identity,
        "output": str(output),
        "manifest": str(manifest_path),
        "status": str(status_path),
        "accepted_slots": len(status["accepted"]),
        "exhausted_slots": len(status["exhausted"]),
        "rows": 2 * len(status["accepted"]),
        "complete": status["complete"],
        "all_slots_succeeded": status["all_slots_succeeded"],
    }


def _selected_parent_indices(raw: str | None, total: int) -> list[int]:
    if raw is None:
        return list(range(total))
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            raise ValueError("parent index list contains an empty token")
        value = int(token)
        if not 0 <= value < total:
            raise ValueError(f"parent index {value} is outside [0, {total - 1}]")
        values.append(value)
    if len(values) != len(set(values)):
        raise ValueError("parent index list contains duplicates")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predecoded-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--parent-indices", help="comma-separated zero-based parent indices")
    parser.add_argument("--max-slots", type=int)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--approach-vertical-arc-height-m",
        type=float,
        default=PRODUCTION_APPROACH_VERTICAL_ARC_HEIGHT_M,
        help="collision-avoidance Z arc used for both near and far prefixes",
    )
    parser.add_argument("--internal-parent-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.max_slots is not None and args.max_slots < 1:
        parser.error("--max-slots must be positive")
    if not np.isfinite(args.approach_vertical_arc_height_m) or not (
        0.0 <= args.approach_vertical_arc_height_m <= 0.10
    ):
        parser.error("--approach-vertical-arc-height-m must be in [0, 0.10]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = load_plan(args.plan.resolve())
    if args.internal_parent_index is not None:
        result = _run_parent(
            plan,
            parent_index=args.internal_parent_index,
            checkpoint=args.checkpoint,
            predecoded_manifest=args.predecoded_manifest,
            output_dir=args.output_dir,
            device=args.device,
            max_slots=args.max_slots,
            replace_output=args.replace,
            approach_vertical_arc_height_m=args.approach_vertical_arc_height_m,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        # Match the ordinary exporter: preserve bounded partial yield, but do
        # not report target completion when one or more slots exhausted all
        # candidates. The outer shard records this nonzero result and continues.
        return 0 if result["complete"] else 3
    indices = _selected_parent_indices(args.parent_indices, len(plan["parents_plan"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for parent_index in indices:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--plan",
            str(args.plan.resolve()),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--predecoded-manifest",
            str(args.predecoded_manifest.resolve()),
            "--output-dir",
            str(args.output_dir.resolve()),
            "--device",
            args.device,
            "--internal-parent-index",
            str(parent_index),
        ]
        if args.max_slots is not None:
            command.extend(["--max-slots", str(args.max_slots)])
        if args.replace:
            command.append("--replace")
        command.extend(
            [
                "--approach-vertical-arc-height-m",
                str(args.approach_vertical_arc_height_m),
            ]
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"planned synthesis parent worker {parent_index} exited {completed.returncode}"
            )
        parent = plan["parents_plan"][parent_index]
        status_path = args.output_dir / "parents" / f"{parent['source_identity']}.status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        results.append(
            {
                "parent_index": parent_index,
                "source_identity": parent["source_identity"],
                "accepted_slots": len(status["accepted"]),
                "exhausted_slots": len(status.get("exhausted", {})),
                "complete": bool(status["complete"]),
                "all_slots_succeeded": bool(
                    status.get("all_slots_succeeded", status["complete"])
                ),
            }
        )
    summary = {
        "contract": RUN_CONTRACT,
        "plan": str(args.plan.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "predecoded_manifest": str(args.predecoded_manifest.resolve()),
        "selected_parent_indices": indices,
        "results": results,
    }
    _atomic_json(args.output_dir / "run-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
