from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from tools.render_atomic_image_pilot import (
    SCENE_OBJECTS,
    add_cameras,
    camera_xyaxes,
    horizontal_to_vertical_fov,
    layout_by_uuid,
    parse_overrides,
    row_pose_sources,
    select_rows_by_action,
)


def metadata(gesture: str) -> dict:
    return {"trajectory_metadata": {"gesture": gesture}}


def test_action_selection_is_deterministic_and_validates_overrides() -> None:
    rows = [metadata("002-stick"), metadata("001-egg-a"), metadata("001-egg-b")]
    assert select_rows_by_action(rows, ("001", "002")) == {"001": 1, "002": 0}
    assert select_rows_by_action(rows, ("001",), {"001": 2}) == {"001": 2}
    with pytest.raises(ValueError, match="does not match"):
        select_rows_by_action(rows, ("001",), {"001": 0})
    with pytest.raises(ValueError, match="no row"):
        select_rows_by_action(rows, ("009",))


def test_layout_index_rejects_duplicate_uuid() -> None:
    assert layout_by_uuid({"trajectories": [{"uuid": "a"}]}) == {
        "a": {"uuid": "a"}
    }
    with pytest.raises(ValueError, match="duplicate"):
        layout_by_uuid({"trajectories": [{"uuid": "a"}, {"uuid": "a"}]})


def test_recorded_objects_move_and_backgrounds_stay_static() -> None:
    names = ["egg_ellipsoid", "egg_cup"]
    frames = 3
    row = {
        "trajectory_metadata": {
            "gesture": "001-egg",
            "total_frames": frames,
            "object_names": names,
        },
        "objects": [
            {
                "pos": [[0, 0, 0.1], [0.1, 0, 0.2], [0.2, 0, 0.3]],
                "rot_aa": [[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2]],
            },
            {
                "pos": [[0.4, 0, 0.1]] * frames,
                "rot_aa": [[0, 0, 0]] * frames,
            },
        ],
    }
    layout = {
        "objects": {
            name: {
                "role": "active" if name in names else "background",
                "pos": [index, index + 1, index + 2],
            }
            for index, name in enumerate(SCENE_OBJECTS)
        }
    }
    before = deepcopy(row)
    poses = row_pose_sources(row, layout)
    assert row == before
    np.testing.assert_array_equal(
        poses["egg_ellipsoid"]["position"], np.asarray(row["objects"][0]["pos"])
    )
    assert poses["egg_ellipsoid"]["source"] == "recorded"
    background = poses["bowl"]
    assert background["source"] == "layout_static"
    np.testing.assert_array_equal(
        background["position"], np.repeat([[5, 6, 7]], frames, axis=0)
    )
    np.testing.assert_array_equal(
        background["quaternion_xyzw"], np.repeat([[0, 0, 0, 1]], frames, axis=0)
    )


def test_camera_contract_is_orthonormal_and_xml_attaches_wrist() -> None:
    right, up = camera_xyaxes((0, -1, 1), (0, 0, 0))
    np.testing.assert_allclose(np.linalg.norm(right), 1)
    np.testing.assert_allclose(np.linalg.norm(up), 1)
    np.testing.assert_allclose(np.dot(right, up), 0, atol=1e-12)
    assert 0 < horizontal_to_vertical_fov(65, 640, 360) < 65
    xml = """<mujoco><worldbody><body name='ARRz_link'/></worldbody></mujoco>"""
    output = add_cameras(xml, width=640, height=360)
    assert 'name="atomic_head_camera"' in output
    assert 'name="atomic_right_wrist_camera"' in output
    assert 'offwidth="640"' in output and 'offheight="360"' in output


def test_override_parser() -> None:
    assert parse_overrides(["001=3", "002=7"]) == {"001": 3, "002": 7}
    with pytest.raises(ValueError):
        parse_overrides(["001=3", "001=4"])
