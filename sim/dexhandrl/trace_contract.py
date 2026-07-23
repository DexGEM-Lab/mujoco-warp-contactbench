"""Offline-safe contract for DexHand Gym↔MuJoCo first-divergence traces."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from sim.dexhandrl.obs_layout import ACTION_DIM_021PRO, OBS_DIM_021PRO

SCHEMA = "dexhandrl.first_divergence_trace.v3"
# Gym initializes physics from frame 61, then its two reset observation passes expose
# frame 62 to the policy. Trace rows name the policy decision, never the initializer.
PHYSICAL_RESET_FRAME = 61
RESET_DECISION_FRAME = 62
FINAL_STATE_FRAME = 693
MAX_ACTIONS = FINAL_STATE_FRAME - RESET_DECISION_FRAME
EPISODE_TIME_FRAMES = FINAL_STATE_FRAME - PHYSICAL_RESET_FRAME + 1
EXACT_IDENTITY = {
    "row_id": 22,
    "uuid": "c21d4fa5-185a-5ea7-aa22-e1d34beae42f",
    "object_name": "cube1", "action": "01", "sequence_index": 1,
    "total_frames": 694, "movement_frames": [261, 575], "effective_window": [PHYSICAL_RESET_FRAME, FINAL_STATE_FRAME],
}
VECTOR_SPECS = {
    "processed_observation": OBS_DIM_021PRO, "rms_observation": OBS_DIM_021PRO,
    "policy_mean": ACTION_DIM_021PRO, "final_action": ACTION_DIM_021PRO,
    "base_offset_before": 3, "base_offset_after": 3, "joint_offset_before": 16,
    "joint_offset_after": 16, "reference_active_target": ACTION_DIM_021PRO,
    "corrected_active_target": ACTION_DIM_021PRO, "applied_active_target": ACTION_DIM_021PRO,
    "full_target": 26, "hand_qpos": 26, "hand_qvel": 26,
    "object_position": 3, "object_quaternion_xyzw": 4, "object_linear_velocity": 3,
    "target_object_position": 3, "target_object_quaternion_xyzw": 4,
}
OPTIONAL_VECTOR_SPECS = {
    # Observation-intervention traces retain the pre-replacement live input.
    "live_processed_observation": OBS_DIM_021PRO,
}
SCALAR_SPECS = {
    "decision_frame": np.dtype(np.int64), "state_frame": np.dtype(np.int64),
    "reset": np.dtype(np.bool_), "terminated": np.dtype(np.bool_), "truncated": np.dtype(np.bool_),
}


def validate_trace(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> int:
    """Validate the v3 Gym-authoritative policy decision/post-state schedule."""
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"unexpected trace schema {metadata.get('schema')!r}")
    if metadata.get("backend") not in {"gym", "mujoco"}:
        raise ValueError("trace metadata backend must be 'gym' or 'mujoco'")
    if not isinstance(metadata.get("post_filters"), list) or not all(isinstance(x, str) for x in metadata["post_filters"]):
        raise ValueError("trace metadata post_filters must be an ordered list of strings")
    identity = metadata.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("trace metadata lacks identity")
    for key, expected in EXACT_IDENTITY.items():
        if identity.get(key) != expected:
            raise ValueError(f"exact identity mismatch for {key}: {identity.get(key)!r} != {expected!r}")
    count: int | None = None
    for name, width in VECTOR_SPECS.items():
        value = arrays.get(name)
        if value is None:
            raise ValueError(f"trace lacks required array {name!r}")
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f"{name} must have shape (frames, {width}), got {array.shape}")
        if array.dtype != np.dtype(np.float32) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite float32 values")
        count = array.shape[0] if count is None else count
        if array.shape[0] != count:
            raise ValueError("trace arrays have inconsistent frame counts")
    assert count is not None
    for name, width in OPTIONAL_VECTOR_SPECS.items():
        if name not in arrays:
            continue
        array = np.asarray(arrays[name])
        if array.ndim != 2 or array.shape != (count, width):
            raise ValueError(f"{name} must have shape ({count}, {width}), got {array.shape}")
        if array.dtype != np.dtype(np.float32) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite float32 values")
    for name, dtype in SCALAR_SPECS.items():
        value = arrays.get(name)
        array = None if value is None else np.asarray(value)
        if array is None or array.ndim != 1 or array.shape[0] != count or array.dtype != dtype:
            raise ValueError(f"{name} must have shape ({count},) and dtype {dtype.name}")
    decision = np.asarray(arrays["decision_frame"])
    state = np.asarray(arrays["state_frame"])
    reset = np.asarray(arrays["reset"])
    if not 1 <= count <= MAX_ACTIONS:
        raise ValueError(f"trace must contain 1..{MAX_ACTIONS} actions")
    expected_decision = RESET_DECISION_FRAME + np.arange(count, dtype=np.int64)
    if not np.array_equal(decision, expected_decision):
        raise ValueError(
            f"decision_frame must exactly equal {RESET_DECISION_FRAME} + arange(actions)"
        )
    if not np.array_equal(state, decision + 1) or int(state[-1]) > FINAL_STATE_FRAME:
        raise ValueError(
            "state_frame must exactly equal decision_frame + 1 and be at most "
            f"{FINAL_STATE_FRAME}"
        )
    expected_reset = decision == RESET_DECISION_FRAME
    if not np.array_equal(reset, expected_reset):
        raise ValueError(
            f"reset marker must be true only for decision_frame {RESET_DECISION_FRAME}"
        )
    return count


def load_trace(path: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load a compressed NPZ containing JSON metadata under ``metadata``."""
    import json
    with np.load(path, allow_pickle=False) as payload:
        if "metadata" not in payload:
            raise ValueError("trace NPZ lacks metadata")
        metadata = json.loads(str(payload["metadata"].item()))
        arrays = {name: np.asarray(payload[name]) for name in payload.files if name != "metadata"}
    validate_trace(metadata, arrays)
    return metadata, arrays
