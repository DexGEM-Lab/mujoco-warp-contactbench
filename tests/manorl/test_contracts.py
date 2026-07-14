from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.manorl.contracts import (
    CONTROL_STEP_COUNT,
    EFFORT,
    JOINT_NAMES,
    KEYPOINT_NAMES,
    MUJOCO_QUATERNION_ORDER,
    REFERENCE_FRAME_COUNT,
    SOURCE_PHYSX_KD,
    SOURCE_PHYSX_KP,
    SOURCE_SLICE,
    ServoConfig,
    TASK_QUATERNION_ORDER,
    TRAJECTORY_IDENTITY,
    validate_contract,
)
from sim.manorl.trajectory import rotvec_to_xyzw, wxyz_to_xyzw, xyzw_to_wxyz


def test_exact_identity_and_slice_contract() -> None:
    validate_contract()
    assert SOURCE_SLICE == (440, 1232)
    assert REFERENCE_FRAME_COUNT == 792
    assert CONTROL_STEP_COUNT == 791
    assert TRAJECTORY_IDENTITY.uuid == "d5bc2bc6-9458-52d0-bccc-66c9ec21bae3"
    assert TRAJECTORY_IDENTITY.file_uuid == "e6fe4732-72cd-5ab7-93e6-2e62dc0263a5"
    assert TRAJECTORY_IDENTITY.identity == "cube1_01_009"
    assert (TRAJECTORY_IDENTITY.movement_start_raw, TRAJECTORY_IDENTITY.movement_end_raw) == (
        690,
        982,
    )
    with pytest.raises(FrozenInstanceError):
        TRAJECTORY_IDENTITY.row_index = 2  # type: ignore[misc]


def test_joint_and_keypoint_order_is_explicit_not_sorted() -> None:
    assert JOINT_NAMES[:6] == ("ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz")
    assert JOINT_NAMES[6:10] == (
        "j1_thumb_cmc_abd",
        "j1_thumb_cmc_flex",
        "j1_thumb_mcp",
        "j1_thumb_ip",
    )
    assert JOINT_NAMES[-4:] == (
        "j5_pinky_mcp_abd",
        "j5_pinky_mcp_flex",
        "j5_pinky_pip",
        "j5_pinky_dip",
    )
    assert KEYPOINT_NAMES == (
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
    assert JOINT_NAMES != tuple(sorted(JOINT_NAMES))


def test_source_and_mujoco_gains_have_distinct_explicit_contracts() -> None:
    expected_kp = np.array([6.0, 4.0, 3.0, 3.0])
    expected_kd = np.array([0.6, 0.4, 0.3, 0.3])
    expected_effort = np.array([3.0, 2.0, 1.2, 1.2])
    np.testing.assert_array_equal(SOURCE_PHYSX_KP[:6], 5000.0)
    np.testing.assert_array_equal(SOURCE_PHYSX_KD[:6], 500.0)
    np.testing.assert_array_equal(EFFORT[:6], 5000.0)
    servo = ServoConfig()
    np.testing.assert_array_equal(servo.kp[:6], 100.0)
    np.testing.assert_array_equal(servo.dampratio[:6], 1.4)
    np.testing.assert_array_equal(servo.dampratio[6:], 1.0)
    for start in range(6, 26, 4):
        np.testing.assert_array_equal(SOURCE_PHYSX_KP[start : start + 4], expected_kp)
        np.testing.assert_array_equal(SOURCE_PHYSX_KD[start : start + 4], expected_kd)
        np.testing.assert_array_equal(servo.kp[start : start + 4], expected_kp)
        np.testing.assert_array_equal(EFFORT[start : start + 4], expected_effort)
    assert not SOURCE_PHYSX_KP.flags.writeable
    assert not SOURCE_PHYSX_KD.flags.writeable
    assert not EFFORT.flags.writeable


def test_quaternion_roundtrip_and_axis_angle_order() -> None:
    assert TASK_QUATERNION_ORDER == "xyzw"
    assert MUJOCO_QUATERNION_ORDER == "wxyz"
    rotvec = np.array([[0.0, 0.0, np.pi / 2], [0.2, -0.1, 0.3]])
    xyzw = rotvec_to_xyzw(rotvec)
    np.testing.assert_allclose(xyzw[0], [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    np.testing.assert_allclose(wxyz_to_xyzw(xyzw_to_wxyz(xyzw)), xyzw)
    np.testing.assert_allclose(Rotation.from_quat(xyzw).as_rotvec(), rotvec, atol=1e-12)
