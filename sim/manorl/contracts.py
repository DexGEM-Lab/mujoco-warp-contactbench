"""Immutable contracts for the single accepted ManoRL replay trajectory."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

# The historical source-aligned replay remains available through the legacy
# loader.  New training jobs should pass their Lance path explicitly (the
# 2026-07-22 capture is exposed as ``DEFAULT_HAND_DATASET_PATH`` below).
LEGACY_DATASET_PATH = "/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_remake/npy_s02_v3.lance"
DEFAULT_HAND_DATASET_PATH = "/mnt/nas-222-project/mocap_v2/lance_datasets/human_guangguan_p1_cma2lance_20260722_025541.lance"
DATASET_PATH = LEGACY_DATASET_PATH
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

LEGACY_JOINT_NAMES: Final[tuple[str, ...]] = (
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

# S02's revised URDF keeps the six floating-base coordinates and adds two
# explicit thumb axes. Keep the legacy tuple only for decoding historical
# Lance rows and read-only semantic fixtures; live policy/checkpoint contracts
# use this authoritative 28-wide order.
JOINT_NAMES_28: Final[tuple[str, ...]] = (
    "ARTx",
    "ARTy",
    "ARTz",
    "ARRx",
    "ARRy",
    "ARRz",
    "j1_thumb_cmc_abd",
    "j1_thumb_cmc_flex",
    "j1_thumb_cmc_twist",
    "j1_thumb_mcp_flex",
    "j1_thumb_mcp_abd",
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

# Public canonical name.  Consumers which must decode the pre-migration
# dataset should use ``LEGACY_JOINT_NAMES`` explicitly.
JOINT_NAMES: Final[tuple[str, ...]] = JOINT_NAMES_28
JOINT_DOF: Final[int] = len(JOINT_NAMES_28)
LEGACY_JOINT_DOF: Final[int] = len(LEGACY_JOINT_NAMES)
FINGER_DOF: Final[int] = JOINT_DOF - 6
LEGACY_FINGER_DOF: Final[int] = LEGACY_JOINT_DOF - 6

HandSide = Literal["left", "right"]
HAND_SIDES: Final[tuple[HandSide, HandSide]] = ("left", "right")
ACTION_SIDE_ORDER: Final[tuple[HandSide, HandSide]] = ("right", "left")


def normalize_hand_side(value: str, *, allow_auto: bool = True, allow_both: bool = True) -> str:
    """Normalize user/dataset side labels to ``left``/``right``/``both``.

    Capture pipelines have historically emitted ``lhand``, ``r_hand`` and
    capitalized variants.  Keeping normalization in one place prevents a
    list-order assumption from leaking into trajectory or Gym code.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("hand side must be a non-empty string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "l": "left",
        "lh": "left",
        "lhand": "left",
        "l_hand": "left",
        "left_hand": "left",
        "hand_left": "left",
        "r": "right",
        "rh": "right",
        "rhand": "right",
        "r_hand": "right",
        "right_hand": "right",
        "hand_right": "right",
        "all": "both",
        "dual": "both",
        "bimanual": "both",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"left", "right"}
    if allow_auto:
        allowed.add("auto")
    if allow_both:
        allowed.add("both")
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"unsupported hand side {value!r}; expected one of {choices}")
    return normalized


def canonical_hand_sides(values: object) -> tuple[HandSide, ...]:
    """Return a de-duplicated side tuple in stable left/right order."""

    if isinstance(values, str):
        values = [values]
    if isinstance(values, (bytes, bytearray)) or not isinstance(values, Iterable):
        raise TypeError("hand sides must be a string or sequence of strings")
    normalized = {
        normalize_hand_side(str(value), allow_auto=False, allow_both=False)
        for value in values
    }
    if not normalized:
        raise ValueError("at least one hand side is required")
    return tuple(side for side in HAND_SIDES if side in normalized)  # type: ignore[return-value]

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
# Historical accepted replay clock. Production 100/120 Hz policy runs use
# ``simulation_clock`` below; legacy replay modules retain these aliases.
PHYSICS_TIMESTEP = 0.0025
PHYSICS_SUBSTEPS_PER_TARGET = 2
CONTROL_TIMESTEP = PHYSICS_TIMESTEP * PHYSICS_SUBSTEPS_PER_TARGET
SUPPORTED_POLICY_FPS: Final = (100, 120)
DEFAULT_POLICY_FPS: Final = 120
SELECTED_POLICY_PHYSICS_SUBSTEPS: Final = 4
RESIDUAL_ENABLED = False
FLOOR_TOP_Z = -0.001
OBJECT_CLEARANCE = 0.001
JOINT_FRICTIONLOSS = 0.1
JOINT_ARMATURE = 0.01


@dataclass(frozen=True)
class SimulationClock:
    """Uniform policy clock and its exact integer physics subdivision."""

    policy_fps: int
    control_timestep: float
    physics_fps: int
    physics_timestep: float
    physics_substeps_per_control: int
    legacy: bool = False

    def __post_init__(self) -> None:
        if self.policy_fps < 1 or self.physics_fps < 1:
            raise ValueError("simulation clock frequencies must be positive")
        if self.physics_substeps_per_control < 1:
            raise ValueError("simulation clock substeps must be positive")
        if not np.isclose(
            self.control_timestep,
            self.physics_timestep * self.physics_substeps_per_control,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("simulation clock physics steps do not equal one control period")
        if self.physics_fps != self.policy_fps * self.physics_substeps_per_control:
            raise ValueError("simulation clock frequencies do not match its substep ratio")


def simulation_clock(policy_fps: int | None) -> SimulationClock:
    """Resolve new coupled source/policy modes or the internal legacy clock."""

    if policy_fps is None or policy_fps == 200:
        return SimulationClock(
            policy_fps=200,
            control_timestep=CONTROL_TIMESTEP,
            physics_fps=400,
            physics_timestep=PHYSICS_TIMESTEP,
            physics_substeps_per_control=PHYSICS_SUBSTEPS_PER_TARGET,
            legacy=True,
        )
    if (
        not isinstance(policy_fps, int)
        or isinstance(policy_fps, bool)
        or policy_fps not in SUPPORTED_POLICY_FPS
    ):
        raise ValueError(f"policy FPS must be one of {SUPPORTED_POLICY_FPS}")
    substeps = SELECTED_POLICY_PHYSICS_SUBSTEPS
    physics_fps = policy_fps * substeps
    return SimulationClock(
        policy_fps=policy_fps,
        control_timestep=1.0 / policy_fps,
        physics_fps=physics_fps,
        physics_timestep=1.0 / physics_fps,
        physics_substeps_per_control=substeps,
    )


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

# The two added thumb axes use the same conservative source finger gains as
# the neighboring thumb coordinates.  The exact source PhysX values are not
# applied by the native MuJoCo controller, but keeping a complete immutable
# vector is important for actuator ordering and checkpoint manifests.
_THUMB_KP_28 = [6.0, 4.0, 3.0, 3.0, 3.0, 3.0]
_THUMB_KD_28 = [0.6, 0.4, 0.3, 0.3, 0.3, 0.3]
_THUMB_EFFORT_28 = [3.0, 2.0, 1.2, 1.2, 1.2, 1.2]

# These reproduce the source PhysX position-drive configuration. Directly using
# them in an external MuJoCo torque law was falsified by the free-space replay.
SOURCE_PHYSX_KP = _readonly([5000.0] * 6 + _THUMB_KP_28 + _DIGIT_KP * 4)
SOURCE_PHYSX_KD = _readonly([500.0] * 6 + _THUMB_KD_28 + _DIGIT_KD * 4)
EFFORT = _readonly([5000.0] * 6 + _THUMB_EFFORT_28 + _DIGIT_EFFORT * 4)

WRIST_KP_GRID = (100.0, 200.0, 400.0)
WRIST_DAMPRATIO_GRID = (0.7, 1.0, 1.4)
FINGER_SERVO_KP = _readonly(_THUMB_KP_28 + _DIGIT_KP * 4)
FINGER_SERVO_DAMPRATIO = 1.0


@dataclass(frozen=True)
class ServoConfig:
    """Bounded MuJoCo-native position-servo experiment configuration."""

    wrist_kp: float = 100.0
    wrist_dampratio: float = 1.4
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
            [float(self.wrist_dampratio)] * 6 + [FINGER_SERVO_DAMPRATIO] * FINGER_DOF
        )


def validate_contract() -> None:
    """Fail immediately if a code edit makes the settled contract inconsistent."""

    if len(JOINT_NAMES) != JOINT_DOF or len(set(JOINT_NAMES)) != JOINT_DOF:
        raise ValueError("JOINT_NAMES must contain unique entries in source order")
    if len(LEGACY_JOINT_NAMES) != LEGACY_JOINT_DOF:
        raise ValueError("legacy joint contract is malformed")
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
        if values.shape != (JOINT_DOF,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name} must contain {JOINT_DOF} finite positive values")
        if values.flags.writeable:
            raise ValueError(f"{name} must be immutable")
    servo = ServoConfig()
    for name, values in (("servo kp", servo.kp), ("servo dampratio", servo.dampratio)):
        if values.shape != (JOINT_DOF,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name} must contain {JOINT_DOF} finite positive values")
        if values.flags.writeable:
            raise ValueError(f"{name} must be immutable")
    if CONTROL_STEP_COUNT != 791:
        raise ValueError("source counter schedule must produce 791 replay steps")
    if CONTROL_TIMESTEP != 0.005 or PHYSICS_SUBSTEPS_PER_TARGET != 2:
        raise ValueError("each target must execute exactly two 0.0025 second substeps")
    if RESIDUAL_ENABLED:
        raise ValueError("this migration slice is residual-off")


validate_contract()
