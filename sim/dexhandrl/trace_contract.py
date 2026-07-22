"""Offline-safe contract for DexHand Gym↔MuJoCo first-divergence traces."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from sim.dexhandrl.obs_layout import ACTION_DIM_021PRO, OBS_DIM_021PRO

SCHEMA = "dexhandrl.first_divergence_trace.v1"
EXACT_IDENTITY = {
    "row_id": 22,
    "uuid": "c21d4fa5-185a-5ea7-aa22-e1d34beae42f",
    "object_name": "cube1", "action": "01", "sequence_index": 1,
    "total_frames": 694, "movement_frames": [261, 575], "effective_window": [61, 693],
}
VECTOR_SPECS = {
    "processed_observation": OBS_DIM_021PRO, "rms_observation": OBS_DIM_021PRO,
    "policy_mean": ACTION_DIM_021PRO, "final_action": ACTION_DIM_021PRO,
    "base_offset_before": 3, "base_offset_after": 3, "joint_offset_before": 16,
    "joint_offset_after": 16, "reference_active_target": ACTION_DIM_021PRO,
    "corrected_active_target": ACTION_DIM_021PRO, "filtered_active_target": ACTION_DIM_021PRO,
    "full_target": 26, "hand_qpos": 26, "hand_qvel": 26,
    "object_position": 3, "object_quaternion_xyzw": 4, "object_linear_velocity": 3,
    "target_object_position": 3, "target_object_quaternion_xyzw": 4,
}
SCALAR_SPECS = {"lance_frame": "i", "reset": "b", "terminated": "b", "truncated": "b"}


def validate_trace(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> int:
    """Validate exact trace alignment and finite numeric stage arrays without runtimes."""
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"unexpected trace schema {metadata.get('schema')!r}")
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
        if not np.issubdtype(array.dtype, np.floating) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite floating values")
        count = array.shape[0] if count is None else count
        if array.shape[0] != count:
            raise ValueError("trace arrays have inconsistent frame counts")
    for name in SCALAR_SPECS:
        value = arrays.get(name)
        if value is None or np.asarray(value).ndim != 1 or np.asarray(value).shape[0] != count:
            raise ValueError(f"{name} must have shape ({count},)")
    frames = np.asarray(arrays["lance_frame"])
    if frames.size == 0 or np.any(np.diff(frames) < 0):
        raise ValueError("lance_frame must be non-empty and non-decreasing")
    return int(count)


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
