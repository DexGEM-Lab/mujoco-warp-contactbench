"""Source-compatible ManoRL raw observation and contact transformations.

The module accepts resolved simulator state instead of inspecting MuJoCo data
directly. Mapping MuJoCo bodies and contacts into the source MANO keypoint
order remains an explicit environment-owner responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from sim.manorl.contracts import KEYPOINT_NAMES


OBSERVATION_KEYS: Final[tuple[str, ...]] = (
    "wrist_pos",
    "finger_pos",
    "hand_orientation",
    "object_position",
    "object_orientation",
    "hand_position",
    "finger_tip_position",
    "target_object_position",
    "target_object_orientation",
    "cumulative_offset",
    "target_object_pos_next_5",
    "table_clearance",
    "object_point_cloud_raw",
    "action_types",
    "object_geometry",
    "hand_keypoints",
    "contact_forces",
    "cumulative_joint_offset",
    "contact_force_directions",
    "expected_contact_mask",
)
OBSERVATION_SLICES: Final[dict[str, slice]] = {
    "wrist_pos": slice(0, 6),
    "finger_pos": slice(6, 26),
    "hand_orientation": slice(26, 30),
    "object_position": slice(30, 33),
    "object_orientation": slice(33, 37),
    "hand_position": slice(37, 40),
    "finger_tip_position": slice(40, 55),
    "target_object_position": slice(55, 58),
    "target_object_orientation": slice(58, 62),
    "cumulative_offset": slice(62, 65),
    "target_object_pos_next_5": slice(65, 68),
    "table_clearance": slice(68, 74),
    "object_point_cloud_raw": slice(74, 266),
    "action_types": slice(266, 316),
    "object_geometry": slice(316, 328),
    "hand_keypoints": slice(328, 376),
    "contact_forces": slice(376, 392),
    "cumulative_joint_offset": slice(392, 412),
    "contact_force_directions": slice(412, 460),
    "expected_contact_mask": slice(460, 476),
}
RAW_OBSERVATION_DIM: Final[int] = 476
POINT_COUNT: Final[int] = 64
POLICY_OBSERVATION_CLIP: Final[float] = 5.0
CONTACT_FORCE_THRESHOLD: Final[float] = 2.0


@dataclass(frozen=True)
class ObservationCompatibility:
    """A selected current-source or checkpoint-sidecar environment contract."""

    name: str
    early_phase_steps: int
    movement_pre_padding: int
    point_template_mode: str

    def __post_init__(self) -> None:
        if self.point_template_mode not in {"static_seed_42", "dynamic_reset"}:
            raise ValueError("point_template_mode must be static_seed_42 or dynamic_reset")
        if self.early_phase_steps < 0 or self.movement_pre_padding < 0:
            raise ValueError("compatibility step and padding values must be non-negative")


CURRENT_SOURCE_COMPATIBILITY: Final = ObservationCompatibility(
    name="current_source", early_phase_steps=100, movement_pre_padding=250,
    point_template_mode="static_seed_42",
)
CHECKPOINT_SIDECAR_COMPATIBILITY: Final = ObservationCompatibility(
    name="checkpoint_sidecar", early_phase_steps=50, movement_pre_padding=200,
    point_template_mode="dynamic_reset",
)


@dataclass(frozen=True)
class PointCloudTemplate:
    """Object-local templates selected by the source static/dynamic policy."""

    local_points: NDArray[np.float64]
    mode: str
    normalized: bool = False
    scale: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class ContactFeatures:
    """Ordered source keypoint force representation used by observations/rewards."""

    force_xyz: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    normalized_magnitude: NDArray[np.float64]
    direction: NDArray[np.float64]


@dataclass(frozen=True)
class ObservationState:
    """All resolved source-state fields required for the raw 476D ABI."""

    mano_dof_pos: NDArray[np.float64]
    mano_dof_lower: NDArray[np.float64]
    mano_dof_upper: NDArray[np.float64]
    hand_position: NDArray[np.float64]
    hand_orientation_xyzw: NDArray[np.float64]
    object_position: NDArray[np.float64]
    object_orientation_xyzw: NDArray[np.float64]
    target_object_position: NDArray[np.float64]
    target_object_orientation_xyzw: NDArray[np.float64]
    target_object_pos_next_5: NDArray[np.float64]
    cumulative_offset: NDArray[np.float64]
    cumulative_joint_offset: NDArray[np.float64]
    point_cloud: PointCloudTemplate
    object_geometry: NDArray[np.float64]
    hand_keypoint_positions: NDArray[np.float64]
    fingertip_positions: NDArray[np.float64]
    hand_keypoint_contact_forces: NDArray[np.float64]
    expected_contact_mask: NDArray[np.float64]
    action_ids: NDArray[np.int64]
    object_support_points: NDArray[np.float64]
    table_surface_height: float = -0.001
    table_clearance_support_point_cap: int = 256


@dataclass(frozen=True)
class ObservationResult:
    """Raw encoder output and the separately clipped policy-normalizer input."""

    raw: NDArray[np.float64]
    policy_input: NDArray[np.float64]
    contacts: ContactFeatures


def _finite_batch(name: str, values: NDArray[object], width: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite (batch, {width}) array")
    return array


def _finite_tensor(
    name: str, values: NDArray[object], trailing_shape: tuple[int, ...], batch: int
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (batch, *trailing_shape) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite (batch, {', '.join(map(str, trailing_shape))}) array")
    return array


def _batch_vector(name: str, values: NDArray[object], batch: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (batch,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite (batch,) array")
    return array


def quat_rotate_xyzw(
    quaternion_xyzw: NDArray[object], vectors: NDArray[object]
) -> NDArray[np.float64]:
    """Match the source quaternion rotation expression without renormalizing it."""

    quat = np.asarray(quaternion_xyzw, dtype=np.float64)
    vector = np.asarray(vectors, dtype=np.float64)
    if quat.shape != vector.shape[:-1] + (4,) or vector.shape[-1:] != (3,):
        raise ValueError("quaternion/vector shapes must be (..., 4) and (..., 3)")
    if not np.all(np.isfinite(quat)) or not np.all(np.isfinite(vector)):
        raise ValueError("quaternions and vectors must be finite")
    q_vec, q_w = quat[..., :3], quat[..., 3:4]
    return (
        vector * (2.0 * q_w * q_w - 1.0)
        + np.cross(q_vec, vector) * q_w * 2.0
        + q_vec * np.sum(q_vec * vector, axis=-1, keepdims=True) * 2.0
    )


def action_type_one_hot(action_ids: NDArray[object], batch: int) -> NDArray[np.float64]:
    """Encode validated 1-based source action IDs in the source 50D order."""

    ids = np.asarray(action_ids)
    if ids.shape != (batch,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("action_ids must be an integer (batch,) array")
    if np.any(ids < 1) or np.any(ids > 50):
        raise ValueError("action_ids must be in the validated source range [1, 50]")
    encoded = np.zeros((batch, 50), dtype=np.float64)
    encoded[np.arange(batch), ids.astype(np.int64) - 1] = 1.0
    return encoded


def expected_contact_mask_from_keypoint_ids(
    keypoint_ids: NDArray[object], batch: int
) -> NDArray[np.float64]:
    """Build the source-order 16D expected-contact mask from validated IDs."""

    ids = np.asarray(keypoint_ids)
    if ids.ndim != 2 or ids.shape[0] != batch or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("keypoint_ids must be an integer (batch, count) array")
    if np.any(ids < 0) or np.any(ids >= len(KEYPOINT_NAMES)):
        raise ValueError("keypoint_ids must use the declared 16-keypoint source order")
    mask = np.zeros((batch, len(KEYPOINT_NAMES)), dtype=np.float64)
    mask[np.arange(batch)[:, None], ids] = 1.0
    return mask


def extract_contact_features(
    hand_keypoint_contact_forces: NDArray[object], *, threshold: float = CONTACT_FORCE_THRESHOLD
) -> ContactFeatures:
    """Compute source observation contact magnitude and direction fields."""

    forces = np.asarray(hand_keypoint_contact_forces, dtype=np.float64)
    if forces.ndim != 3 or forces.shape[1:] != (len(KEYPOINT_NAMES), 3) or not np.all(np.isfinite(forces)):
        raise ValueError("hand_keypoint_contact_forces must be finite (batch, 16, 3)")
    if threshold < 0:
        raise ValueError("contact threshold must be non-negative")
    magnitude = np.linalg.norm(forces, axis=-1)
    direction = forces / np.maximum(magnitude[..., None], 1e-6)
    direction *= (magnitude > threshold)[..., None]
    return ContactFeatures(
        force_xyz=forces,
        magnitude=magnitude,
        normalized_magnitude=np.tanh(magnitude * 0.025),
        direction=direction,
    )


def geometry_encoding(
    *, object_name: str, geometry_type: str, dimensions: NDArray[object], max_size: float = 0.2
) -> NDArray[np.float64]:
    """Implement the source 12D geometry encoding after URDF parsing."""

    values = np.asarray(dimensions, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("geometry dimensions must be a finite non-negative vector")
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    encoded = np.zeros(12, dtype=np.float64)
    category = object_category(object_name)
    if geometry_type == "box" and category == "box" and values.shape == (3,):
        encoded[0:3] = values
    elif geometry_type == "cylinder" and category == "cylinder" and values.shape == (2,):
        encoded[3:6] = (2.0 * values[0], 2.0 * values[0], values[1])
    elif geometry_type == "sphere" and category == "sphere" and values.shape == (1,):
        encoded[6:9] = 2.0 * values[0]
    elif geometry_type == "box" and category == "cylinder" and values.shape == (3,):
        encoded[3:6] = (max(values[0], values[1]), max(values[0], values[1]), values[2])
    elif geometry_type == "box" and category == "sphere" and values.shape == (3,):
        encoded[6:9] = max(values)
    elif category == "irregular" and geometry_type == "box" and values.shape == (3,):
        encoded[9:12] = values
    elif geometry_type == "box" and values.shape == (3,):
        encoded[0:3] = values
    elif geometry_type == "cylinder" and values.shape == (2,):
        encoded[3:6] = (2.0 * values[0], 2.0 * values[0], values[1])
    elif geometry_type == "sphere" and values.shape == (1,):
        encoded[6:9] = 2.0 * values[0]
    # The source emits a warning then retains the zero vector for an unknown
    # parser result. The caller owns parser diagnostics; this pure layer keeps
    # the source numerical fallback for a supplied parsed geometry.
    return np.clip(encoded / max_size, 0.0, 1.0)


def object_category(object_name: str) -> str:
    """Return the source URDFGeometryParser category for an object asset name."""

    if not isinstance(object_name, str) or not object_name.strip():
        raise ValueError("object_name must be a non-empty string")
    lowered = object_name.lower()
    if "cube" in lowered or "cuboid" in lowered:
        return "box"
    if "cylinder" in lowered:
        return "cylinder"
    if "sphere" in lowered:
        return "sphere"
    return "irregular"


def _reduce_support_points(points: NDArray[np.float64], cap: int) -> NDArray[np.float64]:
    if cap < 0:
        raise ValueError("table_clearance_support_point_cap must be non-negative")
    if cap == 0 or len(points) <= cap:
        return points
    extrema = np.unique(np.concatenate((np.argmin(points, axis=0), np.argmax(points, axis=0))))
    remaining = cap - len(extrema)
    if remaining > 0:
        sampled = np.linspace(0, len(points) - 1, remaining).astype(np.int64)
        indices = np.unique(np.concatenate((extrema, sampled)))
    else:
        indices = extrema
    return points[indices[:cap]]


def _point_cloud_observation(
    point_cloud: PointCloudTemplate,
    object_position: NDArray[np.float64],
    object_orientation_xyzw: NDArray[np.float64],
    hand_position: NDArray[np.float64],
    compatibility: ObservationCompatibility,
) -> NDArray[np.float64]:
    if point_cloud.mode != compatibility.point_template_mode:
        raise ValueError("point-cloud template mode must match the selected compatibility")
    batch = len(object_position)
    points = np.asarray(point_cloud.local_points, dtype=np.float64)
    if points.shape == (POINT_COUNT, 3):
        if point_cloud.mode == "dynamic_reset":
            raise ValueError("dynamic_reset compatibility requires a per-environment template")
        points = np.broadcast_to(points, (batch, POINT_COUNT, 3))
    elif points.shape != (batch, POINT_COUNT, 3):
        raise ValueError("point-cloud local_points must be (64, 3) or (batch, 64, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("point-cloud local_points must be finite")
    scale: NDArray[np.float64] | None = None
    if point_cloud.normalized:
        if point_cloud.scale is None:
            raise ValueError("normalized point-cloud templates require an object scale")
        scale_values = np.asarray(point_cloud.scale, dtype=np.float64)
        if scale_values.shape == (3,):
            scale = np.broadcast_to(scale_values, (batch, 3))
        elif scale_values.shape == (batch, 3):
            scale = scale_values
        else:
            raise ValueError("point-cloud scale must be (3,) or (batch, 3)")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("point-cloud scale must be finite and positive")
        points = points * scale[:, None, :]
    quaternions = np.broadcast_to(object_orientation_xyzw[:, None, :], (batch, POINT_COUNT, 4))
    world = quat_rotate_xyzw(quaternions, points) + object_position[:, None, :]
    output = world - hand_position[:, None, :]
    if point_cloud.normalized:
        assert scale is not None
        output = output / scale[:, None, :]
    return output.reshape(batch, POINT_COUNT * 3)


def build_observation(
    state: ObservationState, *, compatibility: ObservationCompatibility
) -> ObservationResult:
    """Assemble source-order raw and policy-clipped observations from supplied state."""

    if not isinstance(state, ObservationState):
        raise TypeError("state must be an ObservationState")
    if not isinstance(compatibility, ObservationCompatibility):
        raise TypeError("compatibility must be an explicit ObservationCompatibility")
    mano = _finite_batch("mano_dof_pos", state.mano_dof_pos, 26)
    batch = len(mano)
    lower = np.asarray(state.mano_dof_lower, dtype=np.float64)
    upper = np.asarray(state.mano_dof_upper, dtype=np.float64)
    if lower.shape != (26,) or upper.shape != (26,) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(upper <= lower):
        raise ValueError("mano DOF limits must be finite ordered (26,) arrays")
    hand_pos = _finite_batch("hand_position", state.hand_position, 3)
    hand_quat = _finite_batch("hand_orientation_xyzw", state.hand_orientation_xyzw, 4)
    obj_pos = _finite_batch("object_position", state.object_position, 3)
    obj_quat = _finite_batch("object_orientation_xyzw", state.object_orientation_xyzw, 4)
    target_pos = _finite_batch("target_object_position", state.target_object_position, 3)
    target_quat = _finite_batch("target_object_orientation_xyzw", state.target_object_orientation_xyzw, 4)
    target_next = _finite_batch("target_object_pos_next_5", state.target_object_pos_next_5, 3)
    cumulative_offset = _finite_batch("cumulative_offset", state.cumulative_offset, 3)
    cumulative_joint = _finite_batch("cumulative_joint_offset", state.cumulative_joint_offset, 20)
    geometry = _finite_batch("object_geometry", state.object_geometry, 12)
    keypoints = _finite_tensor("hand_keypoint_positions", state.hand_keypoint_positions, (16, 3), batch)
    fingertips = _finite_tensor("fingertip_positions", state.fingertip_positions, (5, 3), batch)
    mask = _finite_batch("expected_contact_mask", state.expected_contact_mask, 16)
    if np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("expected_contact_mask must be binary")
    for name, values in (
        ("hand_position", hand_pos), ("hand_orientation_xyzw", hand_quat),
        ("object_position", obj_pos), ("object_orientation_xyzw", obj_quat),
        ("target_object_position", target_pos), ("target_object_orientation_xyzw", target_quat),
        ("target_object_pos_next_5", target_next), ("cumulative_offset", cumulative_offset),
        ("cumulative_joint_offset", cumulative_joint), ("object_geometry", geometry),
    ):
        if len(values) != batch:
            raise ValueError(f"{name} batch size must match mano_dof_pos")
    contact = extract_contact_features(state.hand_keypoint_contact_forces)
    if len(contact.magnitude) != batch:
        raise ValueError("hand_keypoint_contact_forces batch size must match mano_dof_pos")
    support = np.asarray(state.object_support_points, dtype=np.float64)
    if support.ndim != 2 or support.shape[1:] != (3,) or len(support) == 0 or not np.all(np.isfinite(support)):
        raise ValueError("object_support_points must be a non-empty finite (points, 3) array")
    support = _reduce_support_points(support, state.table_clearance_support_point_cap)
    support_quat = obj_quat / np.maximum(np.linalg.norm(obj_quat, axis=1, keepdims=True), 1e-9)
    target_support_quat = target_quat / np.maximum(np.linalg.norm(target_quat, axis=1, keepdims=True), 1e-9)
    object_min_z = (quat_rotate_xyzw(
        np.broadcast_to(support_quat[:, None, :], (batch, len(support), 4)),
        np.broadcast_to(support[None, :, :], (batch, len(support), 3)),
    )[:, :, 2] + obj_pos[:, None, 2]).min(axis=1)
    target_min_z = (quat_rotate_xyzw(
        np.broadcast_to(target_support_quat[:, None, :], (batch, len(support), 4)),
        np.broadcast_to(support[None, :, :], (batch, len(support), 3)),
    )[:, :, 2] + target_pos[:, None, 2]).min(axis=1)
    if not np.isfinite(state.table_surface_height):
        raise ValueError("table_surface_height must be finite")
    table = float(state.table_surface_height)
    clearance = np.stack((
        object_min_z - table,
        obj_pos[:, 2] - table,
        keypoints[:, :, 2].min(axis=1) - table,
        fingertips[:, :, 2].min(axis=1) - table,
        target_min_z - table,
        target_pos[:, 2] - table,
    ), axis=1)
    observation = np.empty((batch, RAW_OBSERVATION_DIM), dtype=np.float64)
    observation[:, OBSERVATION_SLICES["wrist_pos"]] = mano[:, :6]
    observation[:, OBSERVATION_SLICES["finger_pos"]] = 2.0 * (mano[:, 6:] - lower[6:]) / (upper[6:] - lower[6:]) - 1.0
    observation[:, OBSERVATION_SLICES["hand_orientation"]] = hand_quat
    observation[:, OBSERVATION_SLICES["object_position"]] = obj_pos
    observation[:, OBSERVATION_SLICES["object_orientation"]] = obj_quat
    observation[:, OBSERVATION_SLICES["hand_position"]] = hand_pos
    observation[:, OBSERVATION_SLICES["finger_tip_position"]] = (fingertips - obj_pos[:, None, :]).reshape(batch, -1)
    observation[:, OBSERVATION_SLICES["target_object_position"]] = target_pos
    observation[:, OBSERVATION_SLICES["target_object_orientation"]] = target_quat
    observation[:, OBSERVATION_SLICES["cumulative_offset"]] = cumulative_offset
    observation[:, OBSERVATION_SLICES["target_object_pos_next_5"]] = target_next
    observation[:, OBSERVATION_SLICES["table_clearance"]] = clearance
    observation[:, OBSERVATION_SLICES["object_point_cloud_raw"]] = _point_cloud_observation(
        state.point_cloud, obj_pos, obj_quat, hand_pos, compatibility
    )
    observation[:, OBSERVATION_SLICES["action_types"]] = action_type_one_hot(state.action_ids, batch)
    observation[:, OBSERVATION_SLICES["object_geometry"]] = geometry
    observation[:, OBSERVATION_SLICES["hand_keypoints"]] = (keypoints - obj_pos[:, None, :]).reshape(batch, -1)
    observation[:, OBSERVATION_SLICES["contact_forces"]] = contact.normalized_magnitude
    observation[:, OBSERVATION_SLICES["cumulative_joint_offset"]] = cumulative_joint
    observation[:, OBSERVATION_SLICES["contact_force_directions"]] = contact.direction.reshape(batch, -1)
    observation[:, OBSERVATION_SLICES["expected_contact_mask"]] = mask
    if not np.all(np.isfinite(observation)):
        raise ValueError("assembled raw observation is non-finite")
    return ObservationResult(
        raw=observation,
        policy_input=np.clip(observation, -POLICY_OBSERVATION_CLIP, POLICY_OBSERVATION_CLIP),
        contacts=contact,
    )
