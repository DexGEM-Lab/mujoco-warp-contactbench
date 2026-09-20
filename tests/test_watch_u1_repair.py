from copy import deepcopy

import numpy as np
import pytest

from tools.watch_u1_repair import format_state, relative_error


def sample(frame=320):
    transform = {"position": [0.1, 0.0, 0.0], "rotation": np.eye(3).tolist()}
    return {
        "frame": frame, "physics_seconds": frame / 120,
        "running": False, "remaining_steps": 0, "checkpoint_frame": 320,
        "object_position": [0.55, 0.1, 0.14], "object_tilt_deg": 14.45,
        "hand_links": ["palm", "index_dip", "thumb_ip"],
        "hand_contact_count": 3,
        "contacts": [{"link": "index_dip", "distance": -0.0042},
                     {"link": "thumb_ip", "distance": -0.001},
                     {"link": "palm", "distance": -0.003},
                     {"link": "world", "distance": -0.0001}],
        "teacher_hand_links": ["index_dip", "middle_dip"],
        "hand_object_transform": transform,
        "teacher_hand_object_transform": deepcopy(transform),
        "applied_ctrl": [0.0] * 28, "current_28d": [0.0] * 28, "edits": [],
    }


def test_terminal_separates_contact_from_grasp_and_servo_from_source_error():
    state = sample()
    state["applied_ctrl"][2] = 0.1256
    text = format_state(state)
    assert "nonthumb_groups=1/4" in text
    assert "thumb=yes palm=yes" in text
    assert "penetration_max=4.20mm" in text
    assert "floor_geometry=touch" in text
    assert "relative_pose_vs_source=0.00mm/0.00deg" in text
    assert "servo_ctrl_minus_q_xyz=(+0.0,+0.0,+125.6)mm" in text
    assert "pass" not in text.lower()


def test_same_frame_does_not_invent_object_motion_and_rewind_is_explicit():
    previous = sample(332)
    text = format_state(sample(320), previous)
    assert "REWIND" in text
    assert "delta_since_sample=--mm" in text
    assert "delta_since_sample=--mm" in format_state(previous, previous)


def test_relative_transform_error_and_forward_sample_motion():
    before = sample(320)
    after = sample(332)
    after["hand_object_transform"]["position"][1] = 0.01
    after["object_position"][0] += 0.002
    assert relative_error(after) == pytest.approx((10, 0))
    assert "delta_since_sample=2.00mm" in format_state(after, before)


def test_edits_are_printed_only_when_changed():
    before = sample()
    after = sample()
    after["edits"] = [{"joint": 6, "value": np.pi / 18,
                       "start": 320, "end": 380, "ramp": 12}]
    assert "j6=+10.00deg@320:380/r12" in format_state(after, before)
    assert "edits=" not in format_state(after, after)
