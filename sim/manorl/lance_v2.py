"""Clock-aware synthetic Lance contracts for repeated checkpoint rollouts."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from sim.manorl.contracts import (
    JOINT_DOF,
    KEYPOINT_NAMES,
    SimulationClock,
    simulation_clock,
)
from sim.manorl.environment import MaterializedContactBuffers, MaterializedState
from sim.manorl.mano_pose import right_urdf_trajectory_to_mano_48d
from sim.manorl.rewards import PPO_REWARD_CONTRACT_ID, REWARD_CONTRACT_ID
from sim.manorl.trajectory import ReferenceTrajectory, wxyz_to_xyzw

SYNTHETIC_LANCE_V22_CONTRACT = "synthetic_mano_28d_checkpoint_rollout_v2_2"
SYNTHETIC_LANCE_V23_CONTRACT = "synthetic_mano_28d_checkpoint_rollout_v2_3"
SYNTHETIC_LANCE_CONTRACT = SYNTHETIC_LANCE_V23_CONTRACT
SYNTHETIC_LANCE_SOURCE_CONTRACTS = (
    SYNTHETIC_LANCE_V22_CONTRACT,
    SYNTHETIC_LANCE_V23_CONTRACT,
)
SYNTHETIC_LANCE_COMPACT_V1_CONTRACT = "synthetic_mano_target_replay_visual_v1"
SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT = (
    "synthetic_mano_target_replay_visual_v2_contact"
)
SYNTHETIC_LANCE_COMPACT_CONTRACTS = (
    SYNTHETIC_LANCE_COMPACT_V1_CONTRACT,
    SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT,
)
SYNTHETIC_LANCE_OUTPUT_FORMAT_FULL = "full"
SYNTHETIC_LANCE_OUTPUT_FORMAT_COMPACT = "compact-replay-visual"
SYNTHETIC_LANCE_OUTPUT_FORMATS = (
    SYNTHETIC_LANCE_OUTPUT_FORMAT_FULL,
    SYNTHETIC_LANCE_OUTPUT_FORMAT_COMPACT,
)
FORCE_DIRECTION_CONTRACT = "normal_only_hand_to_object_world_joint_object_scale_1p0"
MANO_GLOBAL_FRAME_CONTRACT = "urdf_floating_root_translation_intrinsic_XYZ_to_rotvec_v1"
HAND_SLOT_ORDER = ("right", "left")


def _source_clock(
    source_contract: str,
    control_fps: Any,
    provenance: Mapping[str, Any],
) -> tuple[SimulationClock, int | None]:
    if source_contract not in SYNTHETIC_LANCE_SOURCE_CONTRACTS:
        raise ValueError(f"unsupported compact source contract: {source_contract!r}")
    if isinstance(control_fps, bool) or not isinstance(control_fps, int):
        raise ValueError("synthetic rows must record integer data_fps")
    clock = simulation_clock(control_fps)
    if source_contract == SYNTHETIC_LANCE_V22_CONTRACT:
        if clock.policy_fps != 200:
            raise ValueError("legacy v2.2 compact sources must use 200 Hz")
        if provenance.get("reference_fps") is not None:
            raise ValueError("legacy v2.2 compact sources cannot record reference_fps")
        for name, expected in (
            ("control_fps", clock.policy_fps),
            ("physics_fps", clock.physics_fps),
            ("physics_substeps_per_control", clock.physics_substeps_per_control),
        ):
            if name in provenance and provenance.get(name) != expected:
                raise ValueError(f"legacy v2.2 compact source has inconsistent {name}")
        for name, expected in (
            ("control_timestep_seconds", clock.control_timestep),
            ("physics_timestep_seconds", clock.physics_timestep),
        ):
            value = provenance.get(name)
            if name in provenance and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isclose(float(value), expected, rtol=0.0, atol=1e-15)
            ):
                raise ValueError(f"legacy v2.2 compact source has inconsistent {name}")
        return clock, None
    reference_fps = provenance.get("reference_fps")
    if reference_fps not in (None, 100, 120):
        raise ValueError("v2.3 compact source has invalid reference_fps")
    if clock.policy_fps in (100, 120) and reference_fps != clock.policy_fps:
        raise ValueError("v2.3 public reference/control clocks must be coupled")
    for name, expected in (
        ("control_fps", clock.policy_fps),
        ("physics_fps", clock.physics_fps),
        ("physics_substeps_per_control", clock.physics_substeps_per_control),
    ):
        if provenance.get(name) != expected:
            raise ValueError(f"v2.3 compact source has inconsistent {name}")
    for name, expected in (
        ("control_timestep_seconds", clock.control_timestep),
        ("physics_timestep_seconds", clock.physics_timestep),
    ):
        value = provenance.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isclose(float(value), expected, rtol=0.0, atol=1e-15)
        ):
            raise ValueError(f"v2.3 compact source has inconsistent {name}")
    return clock, reference_fps


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rotation_wxyz(quaternion: NDArray[object]) -> NDArray[np.float64]:
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("body quaternion must be one finite WXYZ value")
    return Rotation.from_quat(wxyz_to_xyzw(values)).as_matrix()


def _vec3(value: NDArray[object]) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("v2 export requires finite 3-vectors")
    return [float(item) for item in array]


def corrected_contact_frames(
    *,
    buffers: MaterializedContactBuffers,
    state: MaterializedState,
    model: Any,
    keypoint_geom_ids: Sequence[int],
    object_geom_ids: Sequence[int],
    object_body_id: int,
    wrist_body_id: int,
    object_name: str,
    hand_name: str = "right",
) -> list[list[dict[str, Any]]]:
    """Decode one batched state into corrected normal-only contact entries.

    Every exported vector is the force exerted by the named hand link on the
    object.  Joint and object values rotate that same vector into their actual
    live MuJoCo body frames; neither changes its sign or applies a force scale.
    """

    batch = int(state.qpos.shape[0])
    if len(keypoint_geom_ids) != len(KEYPOINT_NAMES):
        raise ValueError("v2 contact export requires one geom per MANO keypoint")
    keypoint_lookup = {
        int(geom_id): (index, KEYPOINT_NAMES[index])
        for index, geom_id in enumerate(keypoint_geom_ids)
    }
    object_ids = {int(value) for value in object_geom_ids}
    if not object_ids or object_ids & set(keypoint_lookup):
        raise ValueError("v2 contact export requires disjoint hand/object geoms")
    if buffers.position.shape != (buffers.capacity, 3):
        raise ValueError("v2 contact export requires raw MJX-Warp contact positions")

    grouped: list[dict[tuple[int, str], list[dict[str, Any]]]] = [
        defaultdict(list) for _ in range(batch)
    ]
    joint_rotations: list[dict[str, NDArray[np.float64]]] = [
        dict() for _ in range(batch)
    ]
    wrist_rotations = [
        _rotation_wxyz(state.xquat[world_id, wrist_body_id])
        for world_id in range(batch)
    ]
    object_rotations = [
        _rotation_wxyz(state.xquat[world_id, object_body_id])
        for world_id in range(batch)
    ]

    for contact_id in range(buffers.count):
        world_id = int(buffers.world[contact_id])
        first_geom, second_geom = map(int, buffers.geom[contact_id])
        first_hand = keypoint_lookup.get(first_geom)
        second_hand = keypoint_lookup.get(second_geom)
        if first_hand is not None and second_geom in object_ids and second_hand is None:
            keypoint_index, joint_name = first_hand
            sign = 1.0
            hand_geom_id = first_geom
        elif (
            second_hand is not None and first_geom in object_ids and first_hand is None
        ):
            keypoint_index, joint_name = second_hand
            sign = -1.0
            hand_geom_id = second_geom
        else:
            continue
        if not 0 <= world_id < batch:
            raise RuntimeError(
                f"contact {contact_id} references invalid world {world_id}"
            )
        if int(buffers.dimension[contact_id]) != 3:
            raise RuntimeError("v2 contact export requires condim=3 solved contacts")
        address = np.asarray(buffers.addresses[contact_id], dtype=np.int64)
        if np.any(address < 0) or np.any(address >= buffers.nefc[world_id]):
            raise RuntimeError(
                f"contact {contact_id} has invalid solved-force addresses"
            )
        pyramid = buffers.constraint_force[world_id, address]
        normal_magnitude = float(np.sum(pyramid))
        normal_world = (
            sign
            * normal_magnitude
            * np.asarray(buffers.frame[contact_id, 0], dtype=np.float64)
        )
        hand_body_id = int(model.geom_bodyid[hand_geom_id])
        joint_rotation = _rotation_wxyz(state.xquat[world_id, hand_body_id])
        joint_rotations[world_id][joint_name] = joint_rotation
        pos_world = np.asarray(buffers.position[contact_id], dtype=np.float64)
        pos_joint = joint_rotation.T @ (pos_world - state.xpos[world_id, hand_body_id])
        pos_object = object_rotations[world_id].T @ (
            pos_world - state.xpos[world_id, object_body_id]
        )
        pos_wrist = wrist_rotations[world_id].T @ (
            pos_world - state.xpos[world_id, wrist_body_id]
        )
        grouped[world_id][(keypoint_index, joint_name)].append(
            {
                "force_normal": _vec3(normal_world),
                "pos_world": _vec3(pos_world),
                "pos_wrist": _vec3(pos_wrist),
                "pos_joint": _vec3(pos_joint),
                "pos_object": _vec3(pos_object),
            }
        )

    output: list[list[dict[str, Any]]] = []
    for world_id, groups in enumerate(grouped):
        entries: list[dict[str, Any]] = []
        for (_keypoint_index, joint_name), pairs in sorted(groups.items()):
            total_world = np.sum(
                np.asarray([pair["force_normal"] for pair in pairs], dtype=np.float64),
                axis=0,
            )
            joint_rotation = joint_rotations[world_id][joint_name]
            entries.append(
                {
                    "hand_name": hand_name,
                    "joint_name": joint_name,
                    "object_name": object_name,
                    "total_force_world": _vec3(total_world),
                    "total_force_wrist": _vec3(
                        wrist_rotations[world_id].T @ total_world
                    ),
                    "total_force_joint": _vec3(joint_rotation.T @ total_world),
                    "total_force_object": _vec3(
                        object_rotations[world_id].T @ total_world
                    ),
                    "contact_pairs": pairs,
                }
            )
        output.append(entries)
    return output


def _contact_pair_schema(pa: Any) -> Any:
    """Arrow struct for one contact pair within a hand-object contact entry."""

    fixed = lambda size: pa.list_(pa.float32(), size)
    return pa.struct(
        [
            ("force_normal", fixed(3)),
            ("pos_world", fixed(3)),
            ("pos_wrist", fixed(3)),
            ("pos_joint", fixed(3)),
            ("pos_object", fixed(3)),
        ]
    )


def _contact_entry_schema(pa: Any) -> Any:
    """Arrow struct for one hand-object contact entry (per keypoint per frame)."""

    fixed = lambda size: pa.list_(pa.float32(), size)
    return pa.struct(
        [
            ("hand_name", pa.string()),
            ("joint_name", pa.string()),
            ("object_name", pa.string()),
            ("total_force_world", fixed(3)),
            ("total_force_wrist", fixed(3)),
            ("total_force_joint", fixed(3)),
            ("total_force_object", fixed(3)),
            ("contact_pairs", pa.list_(_contact_pair_schema(pa))),
        ]
    )


def _reference_schema(pa: Any) -> Any:
    """Arrow struct for the source reference trajectory columns."""

    fixed = lambda size: pa.list_(pa.float32(), size)
    return pa.struct(
        [
            ("source_frame_index", pa.list_(pa.int64())),
            ("hand_urdf_dof", pa.list_(fixed(JOINT_DOF))),
            ("object_pos", pa.list_(fixed(3))),
            ("object_rot_aa", pa.list_(fixed(3))),
        ]
    )


def build_v2_schema(
    *,
    observation_dim: int,
    action_dim: int,
    control_fps: int = 200,
    reference_fps: int | None = None,
) -> Any:
    """Build the explicit Arrow schema; no empty-list field is inferred as null."""

    import pyarrow as pa

    if observation_dim < 1 or action_dim < 1:
        raise ValueError("v2 schema dimensions must be positive")
    clock = simulation_clock(control_fps)
    if reference_fps not in (None, 100, 120):
        raise ValueError("synthetic reference_fps must be 100, 120, or None")
    if control_fps in (100, 120) and reference_fps != control_fps:
        raise ValueError("public synthetic clocks require reference_fps == control_fps")
    fixed = lambda size: pa.list_(pa.float32(), size)
    contact_pair = _contact_pair_schema(pa)
    contact_entry = _contact_entry_schema(pa)
    hand = pa.struct(
        [
            ("hand_name", pa.string()),
            ("mano_global_pos", pa.list_(fixed(3))),
            ("mano_global_rot_aa", pa.list_(fixed(3))),
            ("mano_hand_pose", pa.list_(fixed(48))),
            ("mano_joint_pos", pa.list_(pa.list_(fixed(3), 21))),
            ("urdf_dof", pa.list_(fixed(JOINT_DOF))),
            ("urdf_dof_target", pa.list_(fixed(JOINT_DOF))),
        ]
    )
    object_move = pa.struct(
        [
            ("object_name", pa.string()),
            ("start_frame", pa.int64()),
            ("end_frame", pa.int64()),
        ]
    )
    metadata = {
        b"schema": b"synthetic",
        b"schema_version": SYNTHETIC_LANCE_CONTRACT.encode(),
        b"mano_global_frame_contract": MANO_GLOBAL_FRAME_CONTRACT.encode(),
        b"reward_contract": REWARD_CONTRACT_ID.encode(),
        b"ppo_reward_contract": PPO_REWARD_CONTRACT_ID.encode(),
        b"default_episodes_per_identity": b"5",
        b"default_max_attempts_per_identity": b"10",
        b"hand_slot_order": b"right,left",
        b"mano_dof_dim": b"28",
        b"reference_fps": (
            b"none" if reference_fps is None else str(reference_fps).encode()
        ),
        b"control_fps": str(clock.policy_fps).encode(),
        b"control_timestep_seconds": str(clock.control_timestep).encode(),
        b"physics_fps": str(clock.physics_fps).encode(),
        b"physics_timestep_seconds": str(clock.physics_timestep).encode(),
        b"physics_substeps_per_control": str(
            clock.physics_substeps_per_control
        ).encode(),
        b"force_contract": FORCE_DIRECTION_CONTRACT.encode(),
    }
    return pa.schema(
        [
            (
                "index",
                pa.struct(
                    [
                        ("uuid", pa.string()),
                        ("seed_uuid", pa.string()),
                        ("capMachine", pa.string()),
                        ("operator", pa.string()),
                        ("scene", pa.string()),
                        ("is_generated", pa.bool_()),
                    ]
                ),
            ),
            (
                "trajectory_metadata",
                pa.struct(
                    [
                        ("data_fps", pa.int64()),
                        ("total_frames", pa.int64()),
                        ("gesture", pa.string()),
                        ("hand_names", pa.list_(pa.string())),
                        ("hand_slots", pa.list_(pa.string())),
                        ("object_names", pa.list_(pa.string())),
                        ("mano_hand_shapes", pa.list_(fixed(10))),
                        (
                            "raw_data_info",
                            pa.struct(
                                [
                                    ("capMachine", pa.string()),
                                    ("operator", pa.string()),
                                    ("scene", pa.string()),
                                    ("id", pa.int64()),
                                ]
                            ),
                        ),
                        (
                            "trajectory_info",
                            pa.struct([("object_move", pa.list_(object_move))]),
                        ),
                        ("capture_info", pa.null()),
                        (
                            "train_info",
                            pa.struct(
                                [
                                    ("commit_hash", pa.string()),
                                    ("reward_value", pa.float64()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("timestamp", pa.list_(pa.float64())),
            ("hands", pa.list_(hand)),
            (
                "objects",
                pa.list_(
                    pa.struct(
                        [("rot_aa", pa.list_(fixed(3))), ("pos", pa.list_(fixed(3)))]
                    )
                ),
            ),
            ("contact", pa.list_(pa.list_(contact_entry))),
            (
                "reference",
                pa.struct(
                    [
                        ("source_frame_index", pa.list_(pa.int64())),
                        ("hand_urdf_dof", pa.list_(fixed(JOINT_DOF))),
                        ("object_pos", pa.list_(fixed(3))),
                        ("object_rot_aa", pa.list_(fixed(3))),
                    ]
                ),
            ),
            (
                "rollout",
                pa.struct(
                    [
                        ("transition_count", pa.int64()),
                        ("observation_t", pa.list_(fixed(observation_dim))),
                        ("next_observation", pa.list_(fixed(observation_dim))),
                        ("policy_mean_action", pa.list_(fixed(action_dim))),
                        ("processed_action", pa.list_(fixed(action_dim))),
                        ("cumulative_position_residual", pa.list_(fixed(3))),
                        ("cumulative_joint_residual", pa.list_(fixed(action_dim - 6))),
                        ("command_reference_index", pa.list_(pa.int64())),
                        ("command_source_frame_index", pa.list_(pa.int64())),
                        ("reference_target", pa.list_(fixed(JOINT_DOF))),
                        ("processed_target", pa.list_(fixed(JOINT_DOF))),
                        ("controller_target", pa.list_(fixed(JOINT_DOF))),
                        ("reward", pa.list_(pa.float32())),
                        ("raw_contact_reward", pa.list_(pa.float32())),
                        ("contact_reward", pa.list_(pa.float32())),
                        ("terminated", pa.list_(pa.bool_())),
                        ("termination_reason_code", pa.list_(pa.int32())),
                    ]
                ),
            ),
            (
                "provenance",
                pa.struct(
                    [
                        ("contract", pa.string()),
                        ("force_contract", pa.string()),
                        ("reference_fps", pa.int64()),
                        ("control_fps", pa.int64()),
                        ("control_timestep_seconds", pa.float64()),
                        ("physics_fps", pa.int64()),
                        ("physics_timestep_seconds", pa.float64()),
                        ("physics_substeps_per_control", pa.int64()),
                        ("policy_mode", pa.string()),
                        ("checkpoint_path", pa.string()),
                        ("checkpoint_sha256", pa.string()),
                        ("checkpoint_update", pa.int64()),
                        ("checkpoint_metadata_json", pa.string()),
                        ("dataset_path", pa.string()),
                        ("dataset_version", pa.int64()),
                        ("row_index", pa.int64()),
                        ("source_identity", pa.string()),
                        ("software_commit", pa.string()),
                        ("seed", pa.int64()),
                        ("episode_index", pa.int64()),
                        ("generation_attempt", pa.int64()),
                    ]
                ),
            ),
        ],
        metadata=metadata,
    )


def build_compact_schema(
    *,
    source_contract: str = SYNTHETIC_LANCE_CONTRACT,
    control_fps: int = 200,
    reference_fps: int | None = None,
) -> Any:
    """Build a clock-explicit replay/visual schema without audit intermediates."""

    import pyarrow as pa

    clock = simulation_clock(control_fps)
    source_provenance = {
        "reference_fps": reference_fps,
        "control_fps": clock.policy_fps,
        "control_timestep_seconds": clock.control_timestep,
        "physics_fps": clock.physics_fps,
        "physics_timestep_seconds": clock.physics_timestep,
        "physics_substeps_per_control": clock.physics_substeps_per_control,
    }
    clock, reference_fps = _source_clock(
        source_contract, control_fps, source_provenance
    )
    fixed = lambda size: pa.list_(pa.float32(), size)
    hand = pa.struct(
        [
            ("hand_name", pa.string()),
            ("mano_global_pos", pa.list_(fixed(3))),
            ("mano_global_rot_aa", pa.list_(fixed(3))),
            ("mano_hand_pose", pa.list_(fixed(48))),
            ("mano_joint_pos", pa.list_(pa.list_(fixed(3), 21))),
            ("urdf_dof", pa.list_(fixed(JOINT_DOF))),
            ("urdf_dof_target", pa.list_(fixed(JOINT_DOF))),
        ]
    )
    object_move = pa.struct(
        [
            ("object_name", pa.string()),
            ("start_frame", pa.int64()),
            ("end_frame", pa.int64()),
        ]
    )
    metadata = {
        b"schema": b"synthetic",
        b"schema_version": SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT.encode(),
        b"source_contract": source_contract.encode(),
        b"mano_global_frame_contract": MANO_GLOBAL_FRAME_CONTRACT.encode(),
        b"hand_slot_order": b"right,left",
        b"mano_dof_dim": b"28",
        b"reference_fps": (
            b"none" if reference_fps is None else str(reference_fps).encode()
        ),
        b"control_fps": str(clock.policy_fps).encode(),
        b"control_timestep_seconds": str(clock.control_timestep).encode(),
        b"physics_fps": str(clock.physics_fps).encode(),
        b"physics_timestep_seconds": str(clock.physics_timestep).encode(),
        b"physics_substeps_per_control": str(
            clock.physics_substeps_per_control
        ).encode(),
        b"compact_projection": b"replay_visual_v2_contact",
        b"full_checkpoint_metadata": b"external_manifest_only",
    }
    return pa.schema(
        [
            (
                "index",
                pa.struct(
                    [
                        ("uuid", pa.string()),
                        ("seed_uuid", pa.string()),
                        ("capMachine", pa.string()),
                        ("operator", pa.string()),
                        ("scene", pa.string()),
                        ("is_generated", pa.bool_()),
                    ]
                ),
            ),
            (
                "trajectory_metadata",
                pa.struct(
                    [
                        ("data_fps", pa.int64()),
                        ("total_frames", pa.int64()),
                        ("gesture", pa.string()),
                        ("hand_names", pa.list_(pa.string())),
                        ("hand_slots", pa.list_(pa.string())),
                        ("object_names", pa.list_(pa.string())),
                        ("mano_hand_shapes", pa.list_(fixed(10))),
                        (
                            "trajectory_info",
                            pa.struct([("object_move", pa.list_(object_move))]),
                        ),
                    ]
                ),
            ),
            ("timestamp", pa.list_(pa.float64())),
            ("hands", pa.list_(hand)),
            (
                "objects",
                pa.list_(
                    pa.struct(
                        [("rot_aa", pa.list_(fixed(3))), ("pos", pa.list_(fixed(3)))]
                    )
                ),
            ),
            ("contact", pa.list_(pa.list_(_contact_entry_schema(pa)))),
            ("reference", _reference_schema(pa)),
            ("command_reference_index", pa.list_(pa.int64())),
            ("command_source_frame_index", pa.list_(pa.int64())),
            (
                "provenance",
                pa.struct(
                    [
                        ("contract", pa.string()),
                        ("source_contract", pa.string()),
                        ("force_contract", pa.string()),
                        ("reference_fps", pa.int64()),
                        ("control_fps", pa.int64()),
                        ("control_timestep_seconds", pa.float64()),
                        ("physics_fps", pa.int64()),
                        ("physics_timestep_seconds", pa.float64()),
                        ("physics_substeps_per_control", pa.int64()),
                        ("policy_mode", pa.string()),
                        ("checkpoint_path", pa.string()),
                        ("checkpoint_sha256", pa.string()),
                        ("checkpoint_update", pa.int64()),
                        ("checkpoint_metadata_sha256", pa.string()),
                        ("warp_ccd_iterations", pa.int64()),
                        ("warp_ccd_contacts_per_world", pa.int64()),
                        ("dataset_path", pa.string()),
                        ("dataset_version", pa.int64()),
                        ("row_index", pa.int64()),
                        ("source_identity", pa.string()),
                        ("software_commit", pa.string()),
                        ("seed", pa.int64()),
                        ("episode_index", pa.int64()),
                        ("generation_attempt", pa.int64()),
                    ]
                ),
            ),
        ],
        metadata=metadata,
    )


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_compact_row(
    full_row: Mapping[str, Any],
    *,
    checkpoint_metadata: Mapping[str, Any],
    warp_ccd_iterations: int,
    warp_ccd_contacts_per_world: int,
) -> dict[str, Any]:
    """Project one v2.2/v2.3 full row into the compact replay/visual contract."""

    if warp_ccd_iterations is None or warp_ccd_contacts_per_world is None:
        raise ValueError("compact projection requires explicit Warp CCD settings")
    provenance = dict(full_row.get("provenance") or {})
    source_contract = provenance.get("contract")
    metadata = dict(full_row.get("trajectory_metadata") or {})
    clock, reference_fps = _source_clock(
        str(source_contract), metadata.get("data_fps"), provenance
    )
    compact_metadata = {
        name: metadata[name]
        for name in (
            "data_fps",
            "total_frames",
            "gesture",
            "hand_names",
            "hand_slots",
            "object_names",
            "mano_hand_shapes",
            "trajectory_info",
        )
    }
    hands = []
    for hand in full_row.get("hands") or []:
        hand = dict(hand)
        hands.append(
            {
                "hand_name": hand.get("hand_name"),
                "mano_global_pos": hand.get("mano_global_pos") or [],
                "mano_global_rot_aa": hand.get("mano_global_rot_aa") or [],
                "mano_hand_pose": hand.get("mano_hand_pose") or [],
                "mano_joint_pos": hand.get("mano_joint_pos") or [],
                "urdf_dof": hand.get("urdf_dof") or [],
                "urdf_dof_target": hand.get("urdf_dof_target") or [],
            }
        )
    compact_provenance = {
        "contract": SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT,
        "source_contract": source_contract,
        "force_contract": provenance.get("force_contract"),
        "reference_fps": reference_fps,
        "control_fps": clock.policy_fps,
        "control_timestep_seconds": clock.control_timestep,
        "physics_fps": clock.physics_fps,
        "physics_timestep_seconds": clock.physics_timestep,
        "physics_substeps_per_control": clock.physics_substeps_per_control,
        "policy_mode": provenance.get("policy_mode"),
        "checkpoint_path": provenance.get("checkpoint_path"),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "checkpoint_update": provenance.get("checkpoint_update"),
        "checkpoint_metadata_sha256": _canonical_json_sha256(checkpoint_metadata),
        "warp_ccd_iterations": warp_ccd_iterations,
        "warp_ccd_contacts_per_world": warp_ccd_contacts_per_world,
        "dataset_path": provenance.get("dataset_path"),
        "dataset_version": provenance.get("dataset_version"),
        "row_index": provenance.get("row_index"),
        "source_identity": provenance.get("source_identity"),
        "software_commit": provenance.get("software_commit"),
        "seed": provenance.get("seed"),
        "episode_index": provenance.get("episode_index"),
        "generation_attempt": provenance.get("generation_attempt"),
        "augmentation_identity": provenance.get("augmentation_identity"),
    }
    return {
        "index": dict(full_row["index"]),
        "trajectory_metadata": compact_metadata,
        "timestamp": list(full_row["timestamp"]),
        "hands": hands,
        "objects": list(full_row["objects"]),
        "contact": list(full_row.get("contact") or []),
        "reference": dict(full_row.get("reference") or {}),
        "command_reference_index": list(
            (full_row.get("rollout") or {}).get("command_reference_index") or []
        ),
        "command_source_frame_index": list(
            (full_row.get("rollout") or {}).get("command_source_frame_index") or []
        ),
        "provenance": compact_provenance,
    }


def _compact_row_clock(
    row: Mapping[str, Any]
) -> tuple[str, SimulationClock, int | None]:
    provenance = dict(row.get("provenance") or {})
    if provenance.get("contract") not in SYNTHETIC_LANCE_COMPACT_CONTRACTS:
        raise ValueError("compact writer requires compact replay/visual rows")
    metadata = dict(row.get("trajectory_metadata") or {})
    source_contract = str(provenance.get("source_contract"))
    clock, reference_fps = _source_clock(
        source_contract, metadata.get("data_fps"), provenance
    )
    return source_contract, clock, reference_fps


def write_compact_lance_stream(
    rows: Iterable[dict[str, Any]],
    *,
    output: str | Path,
    replace: bool = False,
    append: bool = False,
    batch_size: int = 16,
) -> Path:
    """Stream homogeneous compact rows without materializing the episode set."""

    import lance
    import pyarrow as pa

    if batch_size < 1:
        raise ValueError("compact writer batch_size must be positive")
    output_path = Path(output)
    if append and replace:
        raise ValueError("append and replace are mutually exclusive")
    if output_path.exists() and not append:
        if not replace:
            raise FileExistsError(f"output already exists: {output_path}")
        shutil.rmtree(output_path)
    if append and not output_path.exists():
        raise FileNotFoundError(f"append target does not exist: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(rows)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("cannot write an empty compact rollout dataset") from exc
    source_contract, clock, reference_fps = _compact_row_clock(first)
    schema = build_compact_schema(
        source_contract=source_contract,
        control_fps=clock.policy_fps,
        reference_fps=reference_fps,
    )
    expected_identity = (source_contract, clock.policy_fps, reference_fps)
    if append:
        existing_metadata = {
            key.decode(): value.decode()
            for key, value in (
                lance.dataset(str(output_path)).schema.metadata or {}
            ).items()
        }
        expected_reference = "none" if reference_fps is None else str(reference_fps)
        if (
            existing_metadata.get("schema_version")
            != SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT
            or existing_metadata.get("source_contract") != source_contract
            or int(existing_metadata.get("control_fps", -1)) != clock.policy_fps
            or existing_metadata.get("reference_fps") != expected_reference
        ):
            raise ValueError("append target uses a different compact clock/contract")

    def batches():
        pending = [first]
        for row in iterator:
            row_source, row_clock, row_reference = _compact_row_clock(row)
            if (row_source, row_clock.policy_fps, row_reference) != expected_identity:
                raise ValueError(
                    "one compact Lance dataset cannot mix clocks/contracts"
                )
            pending.append(row)
            if len(pending) >= batch_size:
                yield pa.RecordBatch.from_pylist(pending, schema=schema)
                pending = []
        if pending:
            yield pa.RecordBatch.from_pylist(pending, schema=schema)

    reader = pa.RecordBatchReader.from_batches(schema, batches())
    lance.write_dataset(reader, str(output_path), mode="append" if append else "create")
    return output_path


def write_compact_lance(
    rows: Sequence[dict[str, Any]],
    *,
    output: str | Path,
    replace: bool = False,
    append: bool = False,
) -> Path:
    """Write compact replay/visual rows with a clock-explicit schema."""

    return write_compact_lance_stream(
        rows,
        output=output,
        replace=replace,
        append=append,
        batch_size=max(1, len(rows)),
    )


def _float_rows(values: NDArray[object]) -> list[list[float]]:
    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("v2 output contains non-finite floating values")
    return array.tolist()


def generated_rollout_uuid(
    source_uuid: str,
    checkpoint_sha256: str,
    episode: int = 0,
    *,
    augmentation_identity: str | None = None,
) -> str:
    try:
        namespace = uuid.UUID(source_uuid)
    except ValueError:
        namespace = uuid.NAMESPACE_URL
    return str(
        uuid.uuid5(
            namespace,
            (
                f"{SYNTHETIC_LANCE_CONTRACT}:{source_uuid}:{checkpoint_sha256}:episode={episode}"
                + (
                    ""
                    if augmentation_identity is None
                    else f":augmentation={augmentation_identity}"
                )
            ),
        )
    )


def build_v2_row(
    *,
    trajectory: ReferenceTrajectory,
    source_index: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    states: Mapping[str, NDArray[object]],
    contacts: list[list[dict[str, Any]]],
    rollout: Mapping[str, NDArray[object]],
    provenance: Mapping[str, Any],
    include_checkpoint_metadata_json: bool = True,
) -> dict[str, Any]:
    """Assemble one independent complete trajectory row."""

    clock = simulation_clock(getattr(trajectory, "control_fps", None))
    urdf_dof = np.asarray(states["urdf_dof"], dtype=np.float64)
    total_frames = len(urdf_dof)
    if total_frames != len(trajectory.q_ref) or len(contacts) != total_frames:
        raise ValueError(
            "v2 row must contain exactly one complete source-length episode"
        )
    for name in (
        "mano_joint_pos",
        "urdf_dof_target",
        "object_position",
        "object_orientation_xyzw",
    ):
        if len(np.asarray(states[name])) != total_frames:
            raise ValueError(f"state field {name} is not frame-aligned")
    transitions = total_frames - 1
    if any(len(np.asarray(value)) != transitions for value in rollout.values()):
        raise ValueError("every rollout field must contain exactly T-1 transitions")
    object_rot_aa = Rotation.from_quat(
        np.asarray(states["object_orientation_xyzw"], dtype=np.float64)
    ).as_rotvec()
    hand_global_pos = urdf_dof[:, :3]
    hand_rot_aa = Rotation.from_euler("XYZ", urdf_dof[:, 3:6]).as_rotvec()
    reference_object_rot_aa = Rotation.from_quat(
        trajectory.object_quat_xyzw
    ).as_rotvec()
    mano_pose = right_urdf_trajectory_to_mano_48d(urdf_dof)
    checkpoint_sha = str(provenance["checkpoint_sha256"])
    episode_index = int(provenance["episode_index"])
    generation_attempt = int(provenance["generation_attempt"])
    augmentation_identity = provenance.get("augmentation_identity")
    if augmentation_identity is not None and (
        not isinstance(augmentation_identity, str) or not augmentation_identity
    ):
        raise ValueError("augmentation_identity must be a non-empty string or null")
    if episode_index < 0 or generation_attempt < 1:
        raise ValueError(
            "episode index must be non-negative and generation attempt positive"
        )
    source_uuid = str(trajectory.identity.uuid)
    cap_machine = str(source_index.get("capMachine") or "manorl-mjx-warp")
    operator = str(source_index.get("operator") or "manorl")
    raw_hand_shapes = source_metadata.get("mano_hand_shapes") or []
    raw_hand_names = source_metadata.get("hand_names") or []
    if "right" in raw_hand_names:
        right_index = raw_hand_names.index("right")
        if right_index >= len(raw_hand_shapes):
            raise ValueError("source metadata omits the named right-hand MANO shape")
        right_shape = np.asarray(raw_hand_shapes[right_index], dtype=np.float64)
        if right_shape.shape != (10,) or not np.all(np.isfinite(right_shape)):
            raise ValueError("source right-hand MANO shape must be one finite 10D row")
        hand_shapes = [right_shape]
    else:
        hand_shapes = [np.zeros(10, dtype=np.float64)]
    start_frame = int(
        trajectory.movement_start_step
        if trajectory.movement_start_step is not None
        else trajectory.identity.movement_start_raw - trajectory.identity.source_start
    )
    end_frame = int(
        trajectory.movement_end_step
        if trajectory.movement_end_step is not None
        else trajectory.identity.movement_end_raw - trajectory.identity.source_start
    )
    empty_hand = {
        "hand_name": None,
        "mano_global_pos": [],
        "mano_global_rot_aa": [],
        "mano_hand_pose": [],
        "mano_joint_pos": [],
        "urdf_dof": [],
        "urdf_dof_target": [],
    }
    row = {
        "index": {
            "uuid": generated_rollout_uuid(
                source_uuid,
                checkpoint_sha,
                episode_index,
                augmentation_identity=augmentation_identity,
            ),
            "seed_uuid": source_uuid,
            "capMachine": cap_machine,
            "operator": operator,
            "scene": trajectory.identity.identity.split("_", 1)[0],
            "is_generated": True,
        },
        "trajectory_metadata": {
            "data_fps": clock.policy_fps,
            "total_frames": total_frames,
            "gesture": str(
                source_metadata.get("gesture")
                or trajectory.identity.identity.split("_")[1]
            ),
            "hand_names": ["right"],
            "hand_slots": list(HAND_SLOT_ORDER),
            "object_names": [trajectory.identity.identity.split("_", 1)[0]],
            "mano_hand_shapes": _float_rows(hand_shapes),
            "raw_data_info": {
                "capMachine": cap_machine,
                "operator": operator,
                "scene": trajectory.identity.identity.split("_", 1)[0],
                "id": int(trajectory.identity.row_index),
            },
            "trajectory_info": {
                "object_move": [
                    {
                        "object_name": trajectory.identity.identity.split("_", 1)[0],
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                ]
            },
            "capture_info": None,
            "train_info": {
                "commit_hash": str(provenance["software_commit"]),
                "reward_value": float(
                    np.sum(np.asarray(rollout["reward"], dtype=np.float64))
                ),
            },
        },
        "timestamp": (
            np.arange(total_frames, dtype=np.float64) * clock.control_timestep
        ).tolist(),
        "hands": [
            {
                "hand_name": "right",
                "mano_global_pos": _float_rows(hand_global_pos),
                "mano_global_rot_aa": _float_rows(hand_rot_aa),
                "mano_hand_pose": _float_rows(mano_pose),
                "mano_joint_pos": _float_rows(states["mano_joint_pos"]),
                "urdf_dof": _float_rows(urdf_dof),
                "urdf_dof_target": _float_rows(states["urdf_dof_target"]),
            },
            empty_hand,
        ],
        "objects": [
            {
                "rot_aa": _float_rows(object_rot_aa),
                "pos": _float_rows(states["object_position"]),
            }
        ],
        "contact": contacts,
        "reference": {
            "source_frame_index": trajectory.source_indices.astype(np.int64).tolist(),
            "hand_urdf_dof": _float_rows(trajectory.q_ref),
            "object_pos": _float_rows(trajectory.object_pos),
            "object_rot_aa": _float_rows(reference_object_rot_aa),
        },
        "rollout": {
            "transition_count": transitions,
            **{
                name: (
                    np.asarray(value, dtype=np.int64).tolist()
                    if name in {"command_reference_index", "command_source_frame_index"}
                    else (
                        np.asarray(value, dtype=np.int32).tolist()
                        if name == "termination_reason_code"
                        else (
                            np.asarray(value, dtype=bool).tolist()
                            if name == "terminated"
                            else _float_rows(value)
                        )
                    )
                )
                for name, value in rollout.items()
            },
        },
        "provenance": {
            "contract": SYNTHETIC_LANCE_CONTRACT,
            "force_contract": FORCE_DIRECTION_CONTRACT,
            "reference_fps": getattr(trajectory, "reference_fps", None),
            "control_fps": clock.policy_fps,
            "control_timestep_seconds": clock.control_timestep,
            "physics_fps": clock.physics_fps,
            "physics_timestep_seconds": clock.physics_timestep,
            "physics_substeps_per_control": clock.physics_substeps_per_control,
            "policy_mode": "deterministic_mean",
            "checkpoint_path": str(provenance["checkpoint_path"]),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_update": int(provenance["checkpoint_update"]),
            "checkpoint_metadata_json": (
                json.dumps(
                    provenance.get("checkpoint_metadata", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if include_checkpoint_metadata_json
                else ""
            ),
            "dataset_path": str(trajectory.identity.dataset_path),
            "dataset_version": int(trajectory.identity.dataset_version),
            "row_index": int(trajectory.identity.row_index),
            "source_identity": trajectory.identity.identity,
            "software_commit": str(provenance["software_commit"]),
            "seed": int(provenance["seed"]),
            "episode_index": episode_index,
            "generation_attempt": generation_attempt,
        },
    }
    return row


def write_v2_lance(
    rows: Sequence[dict[str, Any]],
    *,
    output: str | Path,
    observation_dim: int,
    action_dim: int,
    replace: bool = False,
    append: bool = False,
) -> Path:
    import lance
    import pyarrow as pa

    output_path = Path(output)
    if not rows:
        raise ValueError("cannot write an empty v2 rollout dataset")
    control_fps_values = {
        row.get("trajectory_metadata", {}).get("data_fps") for row in rows
    }
    if len(control_fps_values) != 1:
        raise ValueError("one synthetic Lance dataset cannot mix control clocks")
    control_fps = next(iter(control_fps_values))
    if not isinstance(control_fps, int) or isinstance(control_fps, bool):
        raise ValueError("synthetic Lance rows must record integer data_fps")
    try:
        clock = simulation_clock(control_fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("synthetic Lance rows record an invalid data_fps") from exc
    reference_fps_values = {
        row.get("provenance", {}).get("reference_fps") for row in rows
    }
    if len(reference_fps_values) != 1:
        raise ValueError("one synthetic Lance dataset cannot mix reference clocks")
    reference_fps = next(iter(reference_fps_values))
    if reference_fps not in (None, 100, 120):
        raise ValueError("synthetic Lance rows record an invalid reference_fps")
    if control_fps in (100, 120) and reference_fps != control_fps:
        raise ValueError("public synthetic clocks require reference_fps == control_fps")
    if append and replace:
        raise ValueError("append and replace are mutually exclusive")
    if output_path.exists() and not append:
        if not replace:
            raise FileExistsError(f"output already exists: {output_path}")
        shutil.rmtree(output_path)
    if append and not output_path.exists():
        raise FileNotFoundError(f"append target does not exist: {output_path}")
    if append:
        existing_metadata = {
            key.decode(): value.decode()
            for key, value in (
                lance.dataset(str(output_path)).schema.metadata or {}
            ).items()
        }
        if existing_metadata.get("schema_version") != SYNTHETIC_LANCE_CONTRACT:
            raise ValueError(
                "append target does not use the current synthetic Lance contract"
            )
        if int(existing_metadata.get("control_fps", -1)) != clock.policy_fps:
            raise ValueError("append target uses a different control clock")
        recorded_reference_fps = existing_metadata.get("reference_fps")
        expected_reference_fps = "none" if reference_fps is None else str(reference_fps)
        if recorded_reference_fps != expected_reference_fps:
            raise ValueError("append target uses a different reference clock")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        list(rows),
        schema=build_v2_schema(
            observation_dim=observation_dim,
            action_dim=action_dim,
            control_fps=clock.policy_fps,
            reference_fps=reference_fps,
        ),
    )
    lance.write_dataset(table, str(output_path), mode="append" if append else "create")
    return output_path
