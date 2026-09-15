from __future__ import annotations

import numpy as np
import pytest

from tools.replay_capture_no_policy import longest_duration, source_arrays
from sim.manorl import assets


def _row():
    q = np.arange(4 * 28, dtype=float).reshape(4, 28) / 100
    row = {
        "index": {"scene": "egg_cup,bowl"},
        "trajectory_metadata": {
            "total_frames": 4, "hand_names": ["left", "right"],
            "mano_hand_shapes": [[1.] * 10, [0.] * 10],
            "trajectory_info": {"object_move": [{"object_name": "egg_cup", "start_frame": 1, "end_frame": 2}]},
        },
        "timestamp": (np.arange(4) / 120).tolist(),
        "hands": [{"urdf_dof": (q + 100).tolist()}, {"urdf_dof": q.tolist()}],
        "objects": [{"pos": [[0., 0., .2]] * 4, "rot_aa": [[0., 0., 0.]] * 4},
                    {"pos": [[.3, 0., .3]] * 4, "rot_aa": [[0., 0., 0.]] * 4}],
    }
    return row, q


def test_raw_replay_keeps_full_frames_right_slot_and_scene_objects(monkeypatch):
    row, q = _row()
    vertices = np.array([[0., 0., -.02], [0., 0., .02]])
    vertices.setflags(write=False)
    monkeypatch.setattr(assets, "object_collision_vertices", lambda name: vertices)
    manifest = {"hands": {"right": {"betas": [0.] * 10}}}
    result = source_arrays(row, manifest, 120)
    assert result["active"] == "egg_cup"
    assert result["names"] == ("egg_cup", "bowl")
    assert len(result["commands"]) == 4  # includes approach and withdrawal
    np.testing.assert_array_equal(result["original_commands"], q)
    expected = q.copy(); expected[:, 2] += result["shift"]
    np.testing.assert_array_equal(result["commands"], expected)
    np.testing.assert_allclose(result["positions"][:, :, 2], [[.2 + result["shift"], .3 + result["shift"]]] * 4)


def test_raw_replay_rejects_wrong_hand_shape():
    row, _ = _row()
    with pytest.raises(ValueError, match="shape differs"):
        source_arrays(row, {"hands": {"right": {"betas": [2.] * 10}}}, 120)


def test_raw_replay_rejects_retiming_from_rounded_metadata():
    row, _ = _row()
    with pytest.raises(ValueError, match="clock"):
        source_arrays(row, {"hands": {"right": {"betas": [0.] * 10}}}, 100)


@pytest.mark.parametrize(("mask", "duration"), (([False] * 4, 0), ([True] * 4, .4), ([True, True, False, True], .2)))
def test_contiguous_grip_duration(mask, duration):
    assert longest_duration(np.array(mask), 10) == pytest.approx(duration)


def test_invalid_collision_capacity_is_rejected_before_loading():
    from argparse import Namespace
    from tools.replay_capture_no_policy import replay
    with pytest.raises(ValueError, match="CCD capacity"):
        replay(Namespace(ccd=64, contacts=32))
