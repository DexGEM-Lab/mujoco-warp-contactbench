"""DexHand021Pro active-target and MuJoCo joint mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.dexhandrl.constants import (
    DEXHAND021PRO_ACTIVE_DOF_NAMES,
    DEXHAND021PRO_ACTUATOR_NAMES,
    DEXHAND021PRO_FULL_DOF_NAMES,
)


DIRECT_FINGER_CONTROL_TO_FULL_DOF = {
    0: 6,
    1: 7,
    2: 8,
    3: 9,
    4: 10,
    5: 11,
    7: 14,
    8: 15,
    10: 18,
    11: 19,
    13: 22,
    14: 23,
}

ROUTED_FINGER_CONTROL_TO_FULL_DOF = {
    6: (12, 13),
    9: (16, 17),
    12: (20, 21),
    15: (24, 25),
}

ROUTED_JOINT3_FORWARD_LIMIT = 1.2


@dataclass(frozen=True)
class MujocoDexHandIds:
    """MuJoCo ids and qpos/dof addresses needed by replay and env code."""

    joint_qpos_addr: dict[str, int]
    joint_dof_addr: dict[str, int]
    actuator_ids: dict[str, int]
    object_free_qpos_addr: int
    object_free_dof_addr: int
    object_body_id: int
    hand_body_ids: dict[str, int]


def active_to_full_dof(
    active: np.ndarray,
    current_full: np.ndarray | None = None,
    route_joint_upper_limits: dict[int, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Expand 22D DexHand021Pro active controls to 26 MuJoCo joint targets.

    This mirrors the IsaacGym task-specific 021pro rule. Thumb joints and the
    first two joints of other fingers are direct controls. For index/middle/
    ring/pinky distal controls, one active value represents joint3 + joint4;
    the target preserves the current joint3/joint4 distribution when possible.
    """

    arr = np.asarray(active, dtype=np.float32)
    if arr.shape[-1] != len(DEXHAND021PRO_ACTIVE_DOF_NAMES):
        raise ValueError(
            f"DexHand021Pro active target must end with dim "
            f"{len(DEXHAND021PRO_ACTIVE_DOF_NAMES)}, got {arr.shape}"
        )
    out_shape = arr.shape[:-1] + (len(DEXHAND021PRO_FULL_DOF_NAMES),)
    if current_full is None:
        full = np.zeros(out_shape, dtype=np.float32)
    else:
        current = np.asarray(current_full, dtype=np.float32)
        if current.shape[-1] != len(DEXHAND021PRO_FULL_DOF_NAMES):
            raise ValueError(f"current_full must end with dim 26, got {current.shape}")
        full = np.broadcast_to(current, out_shape).copy()
    full[..., :6] = arr[..., :6]
    finger_targets = arr[..., 6:]
    for finger_idx, full_idx in DIRECT_FINGER_CONTROL_TO_FULL_DOF.items():
        full[..., full_idx] = finger_targets[..., finger_idx]

    route_joint_upper_limits = route_joint_upper_limits or {}
    for finger_idx, (joint3_idx, joint4_idx) in ROUTED_FINGER_CONTROL_TO_FULL_DOF.items():
        joint3_limit, joint4_limit = route_joint_upper_limits.get(
            finger_idx,
            (np.inf, np.inf),
        )
        joint3_limit = min(float(joint3_limit), ROUTED_JOINT3_FORWARD_LIMIT)
        joint4_limit = float(joint4_limit)

        desired_total = np.clip(finger_targets[..., finger_idx], a_min=0.0, a_max=None)
        desired_total = np.minimum(desired_total, joint3_limit + joint4_limit)
        joint3_current = np.clip(full[..., joint3_idx], a_min=0.0, a_max=None)
        joint4_current = np.clip(full[..., joint4_idx], a_min=0.0, a_max=None)
        current_total = joint3_current + joint4_current
        delta_total = desired_total - current_total

        positive_delta = np.clip(delta_total, a_min=0.0, a_max=None)
        joint3_room = np.clip(joint3_limit - joint3_current, a_min=0.0, a_max=None)
        joint3_increase = np.minimum(positive_delta, joint3_room)
        joint3_next = joint3_current + joint3_increase
        positive_remaining = positive_delta - joint3_increase
        joint4_room = np.clip(joint4_limit - joint4_current, a_min=0.0, a_max=None)
        joint4_increase = np.minimum(positive_remaining, joint4_room)
        joint4_next = joint4_current + joint4_increase

        negative_delta = np.clip(-delta_total, a_min=0.0, a_max=None)
        joint3_decrease = np.minimum(negative_delta, np.clip(joint3_next, a_min=0.0, a_max=None))
        joint3_next = joint3_next - joint3_decrease
        negative_remaining = negative_delta - joint3_decrease
        joint4_decrease = np.minimum(negative_remaining, np.clip(joint4_next, a_min=0.0, a_max=None))
        joint4_next = joint4_next - joint4_decrease

        full[..., joint3_idx] = joint3_next
        full[..., joint4_idx] = joint4_next
    return full


def raw_finger_to_full_dof(active_base: np.ndarray, raw_finger_dof: np.ndarray) -> np.ndarray:
    """Combine Lance 6D world base targets with 20D raw MuJoCo finger joints."""

    active_arr = np.asarray(active_base, dtype=np.float32)
    raw = np.asarray(raw_finger_dof, dtype=np.float32)
    if active_arr.shape[-1] != len(DEXHAND021PRO_ACTIVE_DOF_NAMES):
        raise ValueError(f"active_base must end with dim 22, got {active_arr.shape}")
    if raw.shape[-1] != 20:
        raise ValueError(f"raw_finger_dof must end with dim 20, got {raw.shape}")
    out_shape = np.broadcast_shapes(active_arr.shape[:-1], raw.shape[:-1]) + (
        len(DEXHAND021PRO_FULL_DOF_NAMES),
    )
    full = np.zeros(out_shape, dtype=np.float32)
    full[..., :6] = np.broadcast_to(active_arr[..., :6], out_shape[:-1] + (6,))
    full[..., 6:] = np.broadcast_to(raw, out_shape[:-1] + (20,))
    return full


def full_to_active_dof(full: np.ndarray) -> np.ndarray:
    """Extract the 22D active target vector from 26D MuJoCo joint positions."""

    arr = np.asarray(full, dtype=np.float32)
    if arr.shape[-1] != len(DEXHAND021PRO_FULL_DOF_NAMES):
        raise ValueError(
            f"DexHand021Pro full target must end with dim "
            f"{len(DEXHAND021PRO_FULL_DOF_NAMES)}, got {arr.shape}"
        )
    active_shape = arr.shape[:-1] + (len(DEXHAND021PRO_ACTIVE_DOF_NAMES),)
    active = np.zeros(active_shape, dtype=np.float32)
    active[..., :6] = arr[..., :6]
    finger_targets = active[..., 6:]
    for finger_idx, full_idx in DIRECT_FINGER_CONTROL_TO_FULL_DOF.items():
        finger_targets[..., finger_idx] = arr[..., full_idx]
    for finger_idx, (joint3_idx, joint4_idx) in ROUTED_FINGER_CONTROL_TO_FULL_DOF.items():
        finger_targets[..., finger_idx] = (
            np.clip(arr[..., joint3_idx], a_min=0.0, a_max=None)
            + np.clip(arr[..., joint4_idx], a_min=0.0, a_max=None)
        )
    return active


def _name_to_joint_id(mujoco: Any, model: Any, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"MuJoCo model is missing joint {name!r}")
    return int(joint_id)


def _name_to_actuator_id(mujoco: Any, model: Any, name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        raise RuntimeError(f"MuJoCo model is missing actuator {name!r}")
    return int(actuator_id)


def resolve_mujoco_ids(mujoco: Any, model: Any, object_name: str) -> MujocoDexHandIds:
    """Resolve all MuJoCo ids used by DexHandRL replay/env code."""

    joint_qpos_addr = {}
    joint_dof_addr = {}
    for name in DEXHAND021PRO_FULL_DOF_NAMES:
        joint_id = _name_to_joint_id(mujoco, model, name)
        joint_qpos_addr[name] = int(model.jnt_qposadr[joint_id])
        joint_dof_addr[name] = int(model.jnt_dofadr[joint_id])

    actuator_ids = {
        name: _name_to_actuator_id(mujoco, model, name)
        for name in DEXHAND021PRO_ACTUATOR_NAMES
    }

    object_joint = _name_to_joint_id(mujoco, model, f"{object_name}_free")
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if object_body_id < 0:
        raise RuntimeError(f"MuJoCo model is missing object body {object_name!r}")

    hand_body_ids = {}
    for body_name in ("right_hand_base", "RFH1"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            hand_body_ids[body_name] = int(body_id)

    return MujocoDexHandIds(
        joint_qpos_addr=joint_qpos_addr,
        joint_dof_addr=joint_dof_addr,
        actuator_ids=actuator_ids,
        object_free_qpos_addr=int(model.jnt_qposadr[object_joint]),
        object_free_dof_addr=int(model.jnt_dofadr[object_joint]),
        object_body_id=int(object_body_id),
        hand_body_ids=hand_body_ids,
    )


def make_ctrl_vector(full_targets: np.ndarray, actuator_ids: dict[str, int], nu: int) -> np.ndarray:
    """Build a MuJoCo ctrl vector ordered by actuator id."""

    full = np.asarray(full_targets, dtype=np.float32)
    if full.shape[-1] != len(DEXHAND021PRO_FULL_DOF_NAMES):
        raise ValueError(f"expected 26D full target, got {full.shape}")
    ctrl = np.zeros((nu,), dtype=np.float32)
    for idx, dof_name in enumerate(DEXHAND021PRO_FULL_DOF_NAMES):
        ctrl[actuator_ids[f"act_{dof_name}"]] = full[idx]
    return ctrl


def full_dof_from_qpos(qpos: np.ndarray, ids: MujocoDexHandIds) -> np.ndarray:
    """Extract 26 ordered DexHand021Pro DOF positions from a MuJoCo qpos vector."""

    qpos_arr = np.asarray(qpos, dtype=np.float32)
    full = np.zeros((len(DEXHAND021PRO_FULL_DOF_NAMES),), dtype=np.float32)
    for idx, name in enumerate(DEXHAND021PRO_FULL_DOF_NAMES):
        full[idx] = qpos_arr[ids.joint_qpos_addr[name]]
    return full


def dexhand021pro_route_joint_upper_limits(mujoco: Any, model: Any) -> dict[int, tuple[float, float]]:
    """Return routed distal joint upper limits keyed by 16D finger-control index."""

    limits: dict[int, tuple[float, float]] = {}
    for finger_idx, (joint3_full_idx, joint4_full_idx) in ROUTED_FINGER_CONTROL_TO_FULL_DOF.items():
        joint3_name = DEXHAND021PRO_FULL_DOF_NAMES[joint3_full_idx]
        joint4_name = DEXHAND021PRO_FULL_DOF_NAMES[joint4_full_idx]
        joint3_id = _name_to_joint_id(mujoco, model, joint3_name)
        joint4_id = _name_to_joint_id(mujoco, model, joint4_name)
        limits[finger_idx] = (
            float(model.jnt_range[joint3_id][1]),
            float(model.jnt_range[joint4_id][1]),
        )
    return limits


def set_hand_qpos(jnp: Any, dx: Any, ids: MujocoDexHandIds, full_targets: np.ndarray) -> Any:
    """Teleport hand joints to a target pose and zero their velocities."""

    qpos = dx.qpos
    qvel = dx.qvel
    qacc = getattr(dx, "qacc", None)
    full = np.asarray(full_targets, dtype=np.float32)
    for idx, name in enumerate(DEXHAND021PRO_FULL_DOF_NAMES):
        qpos = qpos.at[ids.joint_qpos_addr[name]].set(float(full[idx]))
        qvel = qvel.at[ids.joint_dof_addr[name]].set(0.0)
        if qacc is not None:
            qacc = qacc.at[ids.joint_dof_addr[name]].set(0.0)
    if qacc is None:
        return dx.replace(qpos=qpos, qvel=qvel)
    return dx.replace(qpos=qpos, qvel=qvel, qacc=qacc)


def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)


def set_object_freejoint(jnp: Any, dx: Any, ids: MujocoDexHandIds, pos: np.ndarray, quat_xyzw: np.ndarray) -> Any:
    """Set object freejoint pose from Lance position and xyzw quaternion."""

    qpos = dx.qpos
    qvel = dx.qvel
    qacc = getattr(dx, "qacc", None)
    qadr = ids.object_free_qpos_addr
    dadr = ids.object_free_dof_addr
    qpos = qpos.at[qadr : qadr + 3].set(jnp.asarray(pos, dtype=qpos.dtype))
    qpos = qpos.at[qadr + 3 : qadr + 7].set(jnp.asarray(xyzw_to_wxyz(quat_xyzw), dtype=qpos.dtype))
    qvel = qvel.at[dadr : dadr + 6].set(0.0)
    if qacc is not None:
        qacc = qacc.at[dadr : dadr + 6].set(0.0)
    if qacc is None:
        return dx.replace(qpos=qpos, qvel=qvel)
    return dx.replace(qpos=qpos, qvel=qvel, qacc=qacc)
