"""Corrected v2.1 synthetic Lance contract for checkpoint rollouts."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from sim.manorl.contracts import CONTROL_TIMESTEP, JOINT_DOF, KEYPOINT_NAMES
from sim.manorl.environment import MaterializedContactBuffers, MaterializedState
from sim.manorl.mano_pose import right_urdf_trajectory_to_mano_48d
from sim.manorl.trajectory import ReferenceTrajectory, wxyz_to_xyzw

SYNTHETIC_LANCE_V21_CONTRACT = "synthetic_mano_28d_checkpoint_rollout_v2_1"
FORCE_DIRECTION_CONTRACT = "normal_only_hand_to_object_world_joint_object_scale_1p0"
MANO_GLOBAL_FRAME_CONTRACT = (
    "urdf_floating_root_translation_intrinsic_XYZ_to_rotvec_v1"
)
HAND_SLOT_ORDER = ("right", "left")


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
    joint_rotations: list[dict[str, NDArray[np.float64]]] = [dict() for _ in range(batch)]
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
        elif second_hand is not None and first_geom in object_ids and first_hand is None:
            keypoint_index, joint_name = second_hand
            sign = -1.0
            hand_geom_id = second_geom
        else:
            continue
        if not 0 <= world_id < batch:
            raise RuntimeError(f"contact {contact_id} references invalid world {world_id}")
        if int(buffers.dimension[contact_id]) != 3:
            raise RuntimeError("v2 contact export requires condim=3 solved contacts")
        address = np.asarray(buffers.addresses[contact_id], dtype=np.int64)
        if np.any(address < 0) or np.any(address >= buffers.nefc[world_id]):
            raise RuntimeError(f"contact {contact_id} has invalid solved-force addresses")
        pyramid = buffers.constraint_force[world_id, address]
        normal_magnitude = float(np.sum(pyramid))
        normal_world = sign * normal_magnitude * np.asarray(
            buffers.frame[contact_id, 0], dtype=np.float64
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
                    "total_force_wrist": _vec3(wrist_rotations[world_id].T @ total_world),
                    "total_force_joint": _vec3(joint_rotation.T @ total_world),
                    "total_force_object": _vec3(object_rotations[world_id].T @ total_world),
                    "contact_pairs": pairs,
                }
            )
        output.append(entries)
    return output


def build_v2_schema(*, observation_dim: int, action_dim: int) -> Any:
    """Build the explicit Arrow schema; no empty-list field is inferred as null."""

    import pyarrow as pa

    if observation_dim < 1 or action_dim < 1:
        raise ValueError("v2 schema dimensions must be positive")
    fixed = lambda size: pa.list_(pa.float32(), size)
    contact_pair = pa.struct(
        [
            ("force_normal", fixed(3)),
            ("pos_world", fixed(3)),
            ("pos_wrist", fixed(3)),
            ("pos_joint", fixed(3)),
            ("pos_object", fixed(3)),
        ]
    )
    contact_entry = pa.struct(
        [
            ("hand_name", pa.string()),
            ("joint_name", pa.string()),
            ("object_name", pa.string()),
            ("total_force_world", fixed(3)),
            ("total_force_wrist", fixed(3)),
            ("total_force_joint", fixed(3)),
            ("total_force_object", fixed(3)),
            ("contact_pairs", pa.list_(contact_pair)),
        ]
    )
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
        [("object_name", pa.string()), ("start_frame", pa.int64()), ("end_frame", pa.int64())]
    )
    metadata = {
        b"schema": b"synthetic",
        b"schema_version": SYNTHETIC_LANCE_V21_CONTRACT.encode(),
        b"mano_global_frame_contract": MANO_GLOBAL_FRAME_CONTRACT.encode(),
        b"hand_slot_order": b"right,left",
        b"mano_dof_dim": b"28",
        b"control_timestep_seconds": str(CONTROL_TIMESTEP).encode(),
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
                        ("trajectory_info", pa.struct([("object_move", pa.list_(object_move))])),
                        ("capture_info", pa.null()),
                        (
                            "train_info",
                            pa.struct([("commit_hash", pa.string()), ("reward_value", pa.float64())]),
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
                    ]
                ),
            ),
        ],
        metadata=metadata,
    )


def _float_rows(values: NDArray[object]) -> list[list[float]]:
    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("v2 output contains non-finite floating values")
    return array.tolist()


def generated_rollout_uuid(source_uuid: str, checkpoint_sha256: str, episode: int = 0) -> str:
    try:
        namespace = uuid.UUID(source_uuid)
    except ValueError:
        namespace = uuid.NAMESPACE_URL
    return str(
        uuid.uuid5(
            namespace,
            f"{SYNTHETIC_LANCE_V21_CONTRACT}:{source_uuid}:{checkpoint_sha256}:episode={episode}",
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
) -> dict[str, Any]:
    """Assemble one independent complete trajectory row."""

    urdf_dof = np.asarray(states["urdf_dof"], dtype=np.float64)
    total_frames = len(urdf_dof)
    if total_frames != len(trajectory.q_ref) or len(contacts) != total_frames:
        raise ValueError("v2 row must contain exactly one complete source-length episode")
    for name in (
        "mano_joint_pos", "urdf_dof_target", "object_position",
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
    reference_object_rot_aa = Rotation.from_quat(trajectory.object_quat_xyzw).as_rotvec()
    mano_pose = right_urdf_trajectory_to_mano_48d(urdf_dof)
    checkpoint_sha = str(provenance["checkpoint_sha256"])
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
    start_frame = int(trajectory.identity.movement_start_raw - trajectory.identity.source_start)
    end_frame = int(trajectory.identity.movement_end_raw - trajectory.identity.source_start)
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
            "uuid": generated_rollout_uuid(source_uuid, checkpoint_sha),
            "seed_uuid": source_uuid,
            "capMachine": cap_machine,
            "operator": operator,
            "scene": trajectory.identity.identity.split("_", 1)[0],
            "is_generated": True,
        },
        "trajectory_metadata": {
            "data_fps": int(round(1.0 / CONTROL_TIMESTEP)),
            "total_frames": total_frames,
            "gesture": str(source_metadata.get("gesture") or trajectory.identity.identity.split("_")[1]),
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
                "reward_value": float(np.sum(np.asarray(rollout["reward"], dtype=np.float64))),
            },
        },
        "timestamp": (np.arange(total_frames, dtype=np.float64) * CONTROL_TIMESTEP).tolist(),
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
                    else np.asarray(value, dtype=np.int32).tolist()
                    if name == "termination_reason_code"
                    else np.asarray(value, dtype=bool).tolist()
                    if name == "terminated"
                    else _float_rows(value)
                )
                for name, value in rollout.items()
            },
        },
        "provenance": {
            "contract": SYNTHETIC_LANCE_V21_CONTRACT,
            "force_contract": FORCE_DIRECTION_CONTRACT,
            "policy_mode": "deterministic_mean",
            "checkpoint_path": str(provenance["checkpoint_path"]),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_update": int(provenance["checkpoint_update"]),
            "checkpoint_metadata_json": json.dumps(
                provenance.get("checkpoint_metadata", {}), sort_keys=True, separators=(",", ":")
            ),
            "dataset_path": str(trajectory.identity.dataset_path),
            "dataset_version": int(trajectory.identity.dataset_version),
            "row_index": int(trajectory.identity.row_index),
            "source_identity": trajectory.identity.identity,
            "software_commit": str(provenance["software_commit"]),
            "seed": int(provenance["seed"]),
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
) -> Path:
    import lance
    import pyarrow as pa

    output_path = Path(output)
    if not rows:
        raise ValueError("cannot write an empty v2 rollout dataset")
    if output_path.exists():
        if not replace:
            raise FileExistsError(f"output already exists: {output_path}")
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        list(rows), schema=build_v2_schema(observation_dim=observation_dim, action_dim=action_dim)
    )
    lance.write_dataset(table, str(output_path), mode="create")
    return output_path
