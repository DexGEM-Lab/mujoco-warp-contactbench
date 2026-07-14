from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sim.manorl.assets import (
    HAND_SELF_COLLISION_GROUPS,
    build_scene_xml,
    compile_model,
    urdf_zero_fk,
    validate_asset_manifest,
)
from sim.manorl.contracts import EFFORT, JOINT_NAMES, ServoConfig


def test_manifest_and_generated_scene_preserve_authoritative_semantics() -> None:
    manifest = validate_asset_manifest()
    assert manifest["source_repository"] == "sibling:manohand_reconstruction/model/all_assets"
    assert manifest["source_commit"] == "ead79126589d1abf2362ea30b9d674d9e675a2f9"
    assert len(manifest["files"]) == 25
    assert all("powerdrill.obj" not in entry["curated_path"] for entry in manifest["files"])

    root = ET.fromstring(build_scene_xml())
    compiled_order = tuple(
        joint.get("name", "") for joint in root.findall(".//joint") if joint.get("name")
    )
    assert compiled_order == JOINT_NAMES
    positions = root.findall("./actuator/position")
    actuator_order = tuple(actuator.get("name", "") for actuator in positions)
    assert actuator_order == JOINT_NAMES
    assert len(positions) == 26
    assert all(actuator.get("inheritrange") == "1" for actuator in positions)
    assert all(actuator.get("dampratio") == "1" for actuator in positions)
    assert len(root.findall(".//body[@name='powerdrill']/geom")) == 5
    assert root.find(".//body[@name='powerdrill']/freejoint").get("name") == "powerdrill_free"
    assert root.find("./option").get("timestep") == "0.0025"
    assert root.find("./worldbody/geom[@name='floor']").get("pos") == "0 0 -0.001"

    palm = root.find(".//body[@name='palm']")
    assert palm is not None
    palm_quat = np.fromstring(palm.get("quat", ""), sep=" ")
    expected = np.array(
        [-2.59734347e-06, -2.59735301e-06, 0.707108080, 0.707105483]
    )
    np.testing.assert_allclose(palm_quat, expected, atol=5e-10)
    palm_mesh = root.find("./asset/mesh[@name='hand_palm']")
    assert palm_mesh is not None and palm_mesh.get("scale") == "0.7 0.7 0.7"


def test_independent_zero_fk_contains_fixed_palm_rotation() -> None:
    transforms = urdf_zero_fk()
    assert len(transforms) == 28
    palm = transforms["palm"]
    assert not np.allclose(palm[:3, :3], np.eye(3))
    np.testing.assert_allclose(palm[:3, 3], 0.0, atol=0.0)


def test_compiled_native_position_actuator_semantics_and_static_fk() -> None:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco is not installed in this environment")
    mujoco, model = compile_model()
    servo = ServoConfig()
    assert model.nq == 33
    assert model.nv == 32
    assert model.nu == 26
    for actuator_id, (name, kp, effort) in enumerate(
        zip(JOINT_NAMES, servo.kp, EFFORT, strict=True)
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert model.actuator_trnid[actuator_id, 0] == joint_id
        np.testing.assert_allclose(model.actuator_ctrlrange[actuator_id], model.jnt_range[joint_id])
        np.testing.assert_allclose(model.actuator_forcerange[actuator_id], [-effort, effort])
        assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(kp)
        assert model.actuator_biasprm[actuator_id, 1] == pytest.approx(-kp)
        assert model.actuator_biasprm[actuator_id, 2] < 0


def test_compiled_collision_masks_match_source_disable_within_finger_mode() -> None:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco is not installed in this environment")
    mujoco, model = compile_model()

    def geom_id(name: str) -> int:
        value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert value >= 0
        return value

    def can_collide(first: int, second: int) -> bool:
        return bool(
            (model.geom_contype[first] & model.geom_conaffinity[second])
            or (model.geom_contype[second] & model.geom_conaffinity[first])
        )

    hand_ids = [
        geom_id(f"{name}_collision")
        for name in (
            "palm",
            "thumb_cmc",
            "thumb_mcp",
            "thumb_ip",
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
        )
    ]
    object_ids = [geom_id(f"powerdrill_collision_{index}") for index in range(5)]
    floor_id = geom_id("floor")

    assert all((model.geom_contype[geom], model.geom_conaffinity[geom]) == (1, 7) for geom in hand_ids)
    assert all((model.geom_contype[geom], model.geom_conaffinity[geom]) == (2, 5) for geom in object_ids)
    assert (model.geom_contype[floor_id], model.geom_conaffinity[floor_id]) == (4, 1)
    assert all(can_collide(hand, obj) for hand in hand_ids for obj in object_ids)
    assert all(can_collide(hand, floor_id) for hand in hand_ids)
    assert all(can_collide(obj, floor_id) for obj in object_ids)

    def signature(first_name: str, second_name: str) -> int:
        first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, first_name)
        second = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, second_name)
        return (min(first, second) << 16) + max(first, second)

    actual_excludes = {int(value) for value in model.exclude_signature}
    expected_excludes = {
        signature(first, second)
        for names in HAND_SELF_COLLISION_GROUPS.values()
        for index, first in enumerate(names)
        for second in names[index + 1 :]
    }
    assert actual_excludes == expected_excludes
    assert signature("thumb_ip", "index_dip") not in actual_excludes
    assert signature("palm", "thumb_ip") not in actual_excludes


def test_free_space_configuration_disables_only_hand_contacts() -> None:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco is not installed in this environment")
    mujoco, model = compile_model(ServoConfig(hand_contacts_enabled=False))
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "powerdrill")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    hand_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] not in (0, object_id)
    ]
    object_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"powerdrill_collision_{index}")
        for index in range(5)
    ]
    assert all((model.geom_contype[geom], model.geom_conaffinity[geom]) == (0, 0) for geom in hand_ids)
    assert all((model.geom_contype[geom], model.geom_conaffinity[geom]) == (2, 5) for geom in object_ids)
    assert (model.geom_contype[floor_id], model.geom_conaffinity[floor_id]) == (4, 1)
    assert model.nexclude == 15
