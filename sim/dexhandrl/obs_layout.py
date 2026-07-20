"""Canonical 021pro observation layout shared by MuJoCo SKRL components."""

from __future__ import annotations

OBS_DIM_021PRO = 461
ACTION_DIM_021PRO = 22

OBS_COMPONENT_SLICES: dict[str, tuple[int, int]] = {
    "action_type_onehot": (0, 50),
    "object_geometry_encoding": (50, 62),
    "active_finger_dof_pos": (62, 78),
    "hand_pose": (78, 85),
    "fingertip_positions_world": (85, 100),
    "16_body_pose_world": (100, 148),
    "episode_time": (148, 149),
    "object_position": (149, 152),
    "object_orientation": (152, 156),
    "object_linear_velocity": (156, 159),
    "target_object_position": (159, 162),
    "target_object_orientation": (162, 166),
    "target_object_pos_next_5": (166, 169),
    "expected_contact_mask": (169, 185),
    "contact_force_magnitude": (185, 201),
    "contact_force_directions": (201, 249),
    "object_contact_force_magnitude": (249, 250),
    "cumulative_offset": (250, 253),
    "cumulative_joint_offset": (253, 269),
    "object_point_cloud_hand_frame": (269, 461),
}

OBS_COMPONENT_ORDER: tuple[str, ...] = tuple(OBS_COMPONENT_SLICES.keys())
FILM_CONDITION_KEYS: tuple[str, ...] = ("action_type_onehot", "object_geometry_encoding")
POINT_CLOUD_KEY = "object_point_cloud_hand_frame"
POINT_CLOUD_SLICE = OBS_COMPONENT_SLICES[POINT_CLOUD_KEY]
POINT_CLOUD_DIM = POINT_CLOUD_SLICE[1] - POINT_CLOUD_SLICE[0]
NON_POINT_CLOUD_SLICE = (0, POINT_CLOUD_SLICE[0])
FILM_CONDITION_SLICE = (OBS_COMPONENT_SLICES[FILM_CONDITION_KEYS[0]][0], OBS_COMPONENT_SLICES[FILM_CONDITION_KEYS[-1]][1])

DEFAULT_POINTNET_FEATURE_DIM = 64
DEFAULT_POINTNET_NUM_POINTS = 64
DEFAULT_MLP_UNITS: tuple[int, ...] = (512, 256, 128)
DEFAULT_FILM_GENERATOR_HIDDEN_DIM = 256


def slice_length(name: str) -> int:
    start, end = OBS_COMPONENT_SLICES[name]
    return end - start


def observation_layout_summary() -> dict[str, object]:
    """Return a compact summary of the 021pro observation contract."""

    return {
        "obs_dim": OBS_DIM_021PRO,
        "action_dim": ACTION_DIM_021PRO,
        "film_condition_keys": FILM_CONDITION_KEYS,
        "film_condition_slice": FILM_CONDITION_SLICE,
        "point_cloud_key": POINT_CLOUD_KEY,
        "point_cloud_slice": POINT_CLOUD_SLICE,
        "component_slices": OBS_COMPONENT_SLICES,
    }
