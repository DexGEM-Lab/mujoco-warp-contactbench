"""Shared DexHandRL constants mirrored from the IsaacGym project."""

from __future__ import annotations

import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_ASSET_CACHE_ROOT = Path("/tmp/dexhandrl_assets")
_LOCAL_OUTPUT_ROOT = _REPO_ROOT / "outputs"
_VENDORED_ISAAC_SOURCE_ROOT = _REPO_ROOT / "assets/isaac_source_root"


def _env_path(name: str, default: str) -> Path:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return Path(default)
    return Path(value).expanduser()


DEFAULT_ISAAC_SOURCE_ROOT = _env_path(
    "DEXHANDRL_ISAAC_SOURCE_ROOT",
    str(
        _VENDORED_ISAAC_SOURCE_ROOT
        if _VENDORED_ISAAC_SOURCE_ROOT.exists()
        else _LOCAL_ASSET_CACHE_ROOT
        if _LOCAL_ASSET_CACHE_ROOT.exists()
        else _VENDORED_ISAAC_SOURCE_ROOT
    ),
)
DEFAULT_LANCE_PATH = _env_path(
    "DEXHANDRL_LANCE_PATH",
    "/mnt/nas-222-project/mocap/for_retargeting/dexhand021pro/results_all_mano/all/"
    "lance_new_all_generated_mano/lance_result_archive_full/"
    "dexhand021pro_lance_new_all_generated_mano_retarget_result_archive.lance",
)
DEFAULT_OUTPUT_ROOT = _env_path("DEXHANDRL_OUTPUT_ROOT", str(_LOCAL_OUTPUT_ROOT))

DEXHAND021PRO_ACTIVE_DOF_NAMES = (
    "tx_world",
    "ty_world",
    "tz_world",
    "euler_rx_world",
    "euler_ry_world",
    "euler_rz_world",
    "ctrl_thumb_joint1",
    "ctrl_thumb_joint2",
    "ctrl_thumb_joint3",
    "ctrl_thumb_joint4",
    "ctrl_index_joint1",
    "ctrl_index_joint2",
    "ctrl_index_joint3",
    "ctrl_middle_joint1",
    "ctrl_middle_joint2",
    "ctrl_middle_joint3",
    "ctrl_ring_joint1",
    "ctrl_ring_joint2",
    "ctrl_ring_joint3",
    "ctrl_pinky_joint1",
    "ctrl_pinky_joint2",
    "ctrl_pinky_joint3",
)

DEXHAND021PRO_FULL_DOF_NAMES = (
    "ARTx",
    "ARTy",
    "ARTz",
    "ARRx",
    "ARRy",
    "ARRz",
    "r_f_jiont_1_1",
    "r_f_jiont_1_2",
    "r_f_jiont_1_3",
    "r_f_jiont_1_4",
    "r_f_jiont_2_1",
    "r_f_jiont_2_2",
    "r_f_jiont_2_3",
    "r_f_jiont_2_4",
    "r_f_jiont_3_1",
    "r_f_jiont_3_2",
    "r_f_jiont_3_3",
    "r_f_jiont_3_4",
    "r_f_jiont_4_1",
    "r_f_jiont_4_2",
    "r_f_jiont_4_3",
    "r_f_jiont_4_4",
    "r_f_jiont_5_1",
    "r_f_jiont_5_2",
    "r_f_jiont_5_3",
    "r_f_jiont_5_4",
)

DEXHAND021PRO_ACTUATOR_NAMES = tuple(f"act_{name}" for name in DEXHAND021PRO_FULL_DOF_NAMES)

DEXHAND021PRO_CONTACT_BODY_NAMES = (
    "RFH1",
    "RH0_0",
    "RH0_2",
    "RH0_3",
    "RH1_0",
    "RH1_2",
    "RH1_3",
    "RH2_0",
    "RH2_2",
    "RH2_3",
    "RH3_0",
    "RH3_2",
    "RH3_3",
    "RH4_0",
    "RH4_2",
    "RH4_3",
)

DEXHAND021PRO_FINGERTIP_BODY_NAMES = (
    "RH0_tip_visual",
    "RH1_tip_visual",
    "RH2_tip_visual",
    "RH3_tip_visual",
    "RH4_tip_visual",
)

POLICY_OBSERVATION_KEYS_021PRO = (
    "action_type_onehot",
    "object_geometry_encoding",
    "active_finger_dof_pos",
    "hand_pose",
    "fingertip_positions_world",
    "16_body_pose_world",
    "episode_time",
    "object_position",
    "object_orientation",
    "object_linear_velocity",
    "target_object_position",
    "target_object_orientation",
    "target_object_pos_next_5",
    "expected_contact_mask",
    "contact_force_magnitude",
    "contact_force_directions",
    "object_contact_force_magnitude",
    "cumulative_offset",
    "cumulative_joint_offset",
    "object_point_cloud_hand_frame",
)

ACTION_DIM_021PRO = len(DEXHAND021PRO_ACTIVE_DOF_NAMES)
FULL_DOF_DIM_021PRO = len(DEXHAND021PRO_FULL_DOF_NAMES)
NUM_ACTION_TYPES_021PRO = 50
OBJECT_GEOMETRY_ENCODING_DIM = 12
POINT_CLOUD_POINTS_021PRO = 64
POINT_CLOUD_DIM_021PRO = POINT_CLOUD_POINTS_021PRO * 3
