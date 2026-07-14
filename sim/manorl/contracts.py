"""Immutable contracts for the single accepted ManoRL replay trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DATASET_PATH = "/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_remake/npy_s02_v3.lance"
EXPECTED_DATASET_VERSION = 132
DATASET_ROW_INDEX = 1
OBJECT_INDEX = 0
OBJECT_TYPE = "cube1"
OBJECT_LINK_NAME = "cube1_link"
OBJECT_BODY_NAME = "cube1"
OBJECT_FREE_JOINT_NAME = "cube1_free"
OBJECT_COLLISION_GEOM_COUNT = 1
SOURCE_FRAME_COUNT = 1373
SOURCE_DATA_FPS = 111
SOURCE_SLICE = (440, 1232)
REFERENCE_FRAME_COUNT = SOURCE_SLICE[1] - SOURCE_SLICE[0]
# Isaac source holds ref[0] for two commands and terminates after 791 calls:
# command indices 0, 0, 1, ..., 789; post-step references 0, 1, ..., 790.
# Slice ref[791] is therefore intentionally never consumed.
CONTROL_STEP_COUNT = REFERENCE_FRAME_COUNT - 1
MOVEMENT_RAW_RANGE = (690, 982)

JOINT_NAMES = (
    "ARTx",
    "ARTy",
    "ARTz",
    "ARRx",
    "ARRy",
    "ARRz",
    "j1_thumb_cmc_abd",
    "j1_thumb_cmc_flex",
    "j1_thumb_mcp",
    "j1_thumb_ip",
    "j2_index_mcp_abd",
    "j2_index_mcp_flex",
    "j2_index_pip",
    "j2_index_dip",
    "j3_middle_mcp_abd",
    "j3_middle_mcp_flex",
    "j3_middle_pip",
    "j3_middle_dip",
    "j4_ring_mcp_abd",
    "j4_ring_mcp_flex",
    "j4_ring_pip",
    "j4_ring_dip",
    "j5_pinky_mcp_abd",
    "j5_pinky_mcp_flex",
    "j5_pinky_pip",
    "j5_pinky_dip",
)

# Observation/contact ABI from MANOHand.hand_keypoint_names. The thumb is last.
KEYPOINT_NAMES = (
    "palm",
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
)

SOURCE_OBJECT_ROTATION = "axis_angle_xyz_radians"
TASK_QUATERNION_ORDER = "xyzw"
MUJOCO_QUATERNION_ORDER = "wxyz"
PHYSICS_TIMESTEP = 0.0025
PHYSICS_SUBSTEPS_PER_TARGET = 2
CONTROL_TIMESTEP = PHYSICS_TIMESTEP * PHYSICS_SUBSTEPS_PER_TARGET
RESIDUAL_ENABLED = False
FLOOR_TOP_Z = -0.001
OBJECT_CLEARANCE = 0.001
JOINT_FRICTIONLOSS = 0.1
JOINT_ARMATURE = 0.01


@dataclass(frozen=True)
class TrajectoryIdentity:
    dataset_path: str
    dataset_version: int
    row_index: int
    object_index: int
    uuid: str
    file_uuid: str
    identity: str
    source_start: int
    source_stop: int
    movement_start_raw: int
    movement_end_raw: int


TRAJECTORY_IDENTITY = TrajectoryIdentity(
    dataset_path=DATASET_PATH,
    dataset_version=EXPECTED_DATASET_VERSION,
    row_index=DATASET_ROW_INDEX,
    object_index=OBJECT_INDEX,
    uuid="d5bc2bc6-9458-52d0-bccc-66c9ec21bae3",
    file_uuid="e6fe4732-72cd-5ab7-93e6-2e62dc0263a5",
    identity="cube1_01_009",
    source_start=SOURCE_SLICE[0],
    source_stop=SOURCE_SLICE[1],
    movement_start_raw=MOVEMENT_RAW_RANGE[0],
    movement_end_raw=MOVEMENT_RAW_RANGE[1],
)


def _readonly(values: list[float]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    array.setflags(write=False)
    return array


_DIGIT_KP = [6.0, 4.0, 3.0, 3.0]
_DIGIT_KD = [0.6, 0.4, 0.3, 0.3]
_DIGIT_EFFORT = [3.0, 2.0, 1.2, 1.2]

# These reproduce the source PhysX position-drive configuration. Directly using
# them in an external MuJoCo torque law was falsified by the free-space replay.
SOURCE_PHYSX_KP = _readonly([5000.0] * 6 + _DIGIT_KP * 5)
SOURCE_PHYSX_KD = _readonly([500.0] * 6 + _DIGIT_KD * 5)
EFFORT = _readonly([5000.0] * 6 + _DIGIT_EFFORT * 5)

WRIST_KP_GRID = (100.0, 200.0, 400.0)
WRIST_DAMPRATIO_GRID = (0.7, 1.0, 1.4)
FINGER_SERVO_KP = _readonly(_DIGIT_KP * 5)
FINGER_SERVO_DAMPRATIO = 1.0


@dataclass(frozen=True)
class ServoConfig:
    """Bounded MuJoCo-native position-servo experiment configuration."""

    wrist_kp: float = 200.0
    wrist_dampratio: float = 1.0
    hand_contacts_enabled: bool = True

    def __post_init__(self) -> None:
        if float(self.wrist_kp) not in WRIST_KP_GRID:
            raise ValueError(f"wrist_kp must be one of {WRIST_KP_GRID}")
        if float(self.wrist_dampratio) not in WRIST_DAMPRATIO_GRID:
            raise ValueError(f"wrist_dampratio must be one of {WRIST_DAMPRATIO_GRID}")
        if not isinstance(self.hand_contacts_enabled, bool):
            raise TypeError("hand_contacts_enabled must be bool")

    @property
    def kp(self) -> NDArray[np.float64]:
        return _readonly([float(self.wrist_kp)] * 6 + FINGER_SERVO_KP.tolist())

    @property
    def dampratio(self) -> NDArray[np.float64]:
        return _readonly(
            [float(self.wrist_dampratio)] * 6 + [FINGER_SERVO_DAMPRATIO] * 20
        )


def validate_contract() -> None:
    """Fail immediately if a code edit makes the settled contract inconsistent."""

    if len(JOINT_NAMES) != 26 or len(set(JOINT_NAMES)) != 26:
        raise ValueError("JOINT_NAMES must contain 26 unique entries in source order")
    if len(KEYPOINT_NAMES) != 16 or len(set(KEYPOINT_NAMES)) != 16:
        raise ValueError("KEYPOINT_NAMES must contain 16 unique entries in source order")
    if SOURCE_SLICE != (440, 1232) or REFERENCE_FRAME_COUNT != 792:
        raise ValueError("the accepted source slice must remain [440, 1232)")
    if (
        OBJECT_TYPE != "cube1"
        or OBJECT_LINK_NAME != "cube1_link"
        or OBJECT_BODY_NAME != "cube1"
        or OBJECT_FREE_JOINT_NAME != "cube1_free"
        or OBJECT_COLLISION_GEOM_COUNT != 1
    ):
        raise ValueError("the accepted runtime object must remain the one-geom cube1")
    for name, values in (
        ("source_physx_kp", SOURCE_PHYSX_KP),
        ("source_physx_kd", SOURCE_PHYSX_KD),
        ("effort", EFFORT),
    ):
        if values.shape != (26,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name} must contain 26 finite positive values")
        if values.flags.writeable:
            raise ValueError(f"{name} must be immutable")
    servo = ServoConfig()
    for name, values in (("servo kp", servo.kp), ("servo dampratio", servo.dampratio)):
        if values.shape != (26,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name} must contain 26 finite positive values")
        if values.flags.writeable:
            raise ValueError(f"{name} must be immutable")
    if CONTROL_STEP_COUNT != 791:
        raise ValueError("source counter schedule must produce 791 replay steps")
    if CONTROL_TIMESTEP != 0.005 or PHYSICS_SUBSTEPS_PER_TARGET != 2:
        raise ValueError("each target must execute exactly two 0.0025 second substeps")
    if RESIDUAL_ENABLED:
        raise ValueError("this migration slice is residual-off")


validate_contract()
