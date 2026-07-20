from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sim.manorl.assets import (
    HAND_SELF_COLLISION_GROUPS,
    build_scene_xml,
    compile_model,
    object_collision_vertices,
    object_runtime,
    supported_object_types,
    urdf_zero_fk,
    validate_asset_manifest,
)
from sim.manorl.contracts import EFFORT, JOINT_NAMES, ServoConfig


def test_manifest_and_generated_scene_preserve_authoritative_semantics() -> None:
    manifest = validate_asset_manifest()
    assert manifest["source_repository"] == "sibling:manohand_reconstruction/model/all_assets"
    assert manifest["source_commit"] == "ead79126589d1abf2362ea30b9d674d9e675a2f9"
    assert len(manifest["files"]) == 21
    assert all(
        entry["curated_path"].startswith(("hand/", "cube1/"))
        for entry in manifest["files"]
    )

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
    assert all(actuator.get("dampratio") == "1.3999999999999999" for actuator in positions[:6])
    assert all(actuator.get("dampratio") == "1" for actuator in positions[6:])
    assert len(root.findall(".//body[@name='cube1']/geom")) == 1
    assert root.find(".//body[@name='cube1']/freejoint").get("name") == "cube1_free"
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


def test_s02_object_registry_is_closed_materialized_and_digest_checked() -> None:
    expected = (
        "cube1",
        "cube2",
        "cuboid1",
        "cuboid2",
        "cylinder1",
        "cylinder2",
        "cylinder3",
        "cylinder4",
        "cylinder5",
        "cylinder6",
        "sphere1",
        "sphere2",
        "sphere3",
    )
    assert supported_object_types() == expected
    validate_asset_manifest()

    bounds = {}
    for object_type in expected:
        runtime = object_runtime(object_type)
        assert runtime.urdf_path.is_file()
        assert runtime.collision_mesh_path.is_file()
        assert runtime.grasp_mapping_path.is_file()
        vertices = object_collision_vertices(object_type)
        dimensions = np.ptp(vertices, axis=0)
        assert np.all(dimensions > 0.0)
        bounds[object_type] = tuple(np.round(dimensions, decimals=6))

    assert bounds["cube1"] != bounds["cube2"]
    assert bounds["cuboid1"] != bounds["sphere1"]
    with pytest.raises(ValueError, match="unsupported ManoRL object runtime"):
        object_runtime("unknown_object")


def test_all_s02_object_models_compile_with_object_specific_mass_and_geometry() -> None:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco is not installed in this environment")

    signatures = {}
    for object_type in supported_object_types():
        mujoco, model = compile_model(object_type=object_type)
        runtime = object_runtime(object_type)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.body_name)
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{object_type}_collision"
        )
        assert body_id >= 0 and geom_id >= 0
        assert float(model.body_mass[body_id]) > 0.0
        signatures[object_type] = (
            round(float(model.body_mass[body_id]), 6),
            tuple(np.round(np.ptp(object_collision_vertices(object_type), axis=0), 6)),
        )

    assert len(set(signatures.values())) == len(signatures)


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
    object_ids = [geom_id("cube1_collision")]
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
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube1")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    hand_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] not in (0, object_id)
    ]
    object_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube1_collision")]
    assert all((model.geom_contype[geom], model.geom_conaffinity[geom]) == (0, 0) for geom in hand_ids)
    assert all((model.geom_contype[geom], model.geom_conaffinity[geom]) == (2, 5) for geom in object_ids)
    assert (model.geom_contype[floor_id], model.geom_conaffinity[floor_id]) == (4, 1)
    assert model.nexclude == 15
