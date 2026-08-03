from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sim.manorl.assets import (
    HAND_SELF_COLLISION_GROUPS,
    build_scene_xml,
    build_unified_scene_xml,
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
    assert manifest["source_commit"] == "033b358b73c57e5f437f6582b6a9b0d4add7f9ee"
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
    assert len(positions) == len(JOINT_NAMES) == 28
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


@pytest.mark.parametrize(
    ("policy_fps", "physics_timestep"),
    ((100, 1.0 / 400.0), (120, 1.0 / 480.0)),
)
def test_selected_clock_compiles_exact_physics_timestep(
    policy_fps: int, physics_timestep: float
) -> None:
    assert physics_timestep == pytest.approx(1.0 / (4 * policy_fps))
    root = ET.fromstring(build_scene_xml(physics_timestep=physics_timestep))
    assert float(root.find("./option").get("timestep")) == pytest.approx(
        physics_timestep
    )
    mujoco, model = compile_model(physics_timestep=physics_timestep)
    assert model.opt.timestep == pytest.approx(physics_timestep)
    data = mujoco.MjData(model)
    for _ in range(4):
        mujoco.mj_step(model, data)
    assert data.time == pytest.approx(1.0 / policy_fps)


@pytest.mark.parametrize("unified", (False, True))
def test_generated_scene_uses_checkerboard_floor_material(unified: bool) -> None:
    xml = (
        build_unified_scene_xml(object_types=("cube1",))
        if unified
        else build_scene_xml()
    )
    root = ET.fromstring(xml)
    texture = root.find("./asset/texture[@name='floor_checker']")
    material = root.find("./asset/material[@name='floor_checker']")
    floor = root.find("./worldbody/geom[@name='floor']")

    assert texture is not None
    assert texture.attrib == {
        "name": "floor_checker",
        "type": "2d",
        "builtin": "checker",
        "rgb1": "0.18 0.20 0.22",
        "rgb2": "0.72 0.74 0.76",
        "width": "512",
        "height": "512",
    }
    assert material is not None
    assert material.attrib == {
        "name": "floor_checker",
        "texture": "floor_checker",
        "texrepeat": "8 8",
        "texuniform": "true",
        "reflectance": "0.08",
    }
    assert floor is not None
    assert floor.get("material") == "floor_checker"
    assert floor.get("rgba") is None
    assert floor.get("pos") == "0 0 -0.001"
    assert floor.get("size") == "1 1 0.01"
    assert floor.get("contype") == "4"
    assert floor.get("conaffinity") == "1"
    assert floor.get("friction") == "1 0.01 0.001"


def test_visual_scene_uses_urdf_visual_meshes_without_collision_bits() -> None:
    root = ET.fromstring(build_scene_xml(visual_meshes=True))

    hand_visual_assets = [
        mesh
        for mesh in root.findall("./asset/mesh")
        if mesh.get("name", "").startswith("hand_visual_")
    ]
    assert len(hand_visual_assets) == 16
    palm_asset = root.find("./asset/mesh[@name='hand_visual_palm_visual']")
    assert palm_asset is not None
    assert palm_asset.get("file", "").endswith("/visual_palm.stl")

    palm_visual = root.find(".//body[@name='palm']/geom[@name='palm_visual']")
    assert palm_visual is not None
    assert palm_visual.get("mesh") == "hand_visual_palm_visual"
    assert palm_visual.get("rgba") == "0.95 0.75 0.65 1.0"
    assert palm_visual.get("contype") == "0"
    assert palm_visual.get("conaffinity") == "0"
    assert palm_visual.get("group") == "2"

    fingertip_marker = root.find(
        ".//body[@name='thumb_ip']/geom[@name='thumb_ip_visual_1']"
    )
    assert fingertip_marker is not None
    assert fingertip_marker.get("type") == "sphere"
    assert fingertip_marker.get("size") == "0.0025"

    object_asset = root.find("./asset/mesh[@name='cube1_visual_mesh']")
    object_visual = root.find(".//body[@name='cube1']/geom[@name='cube1_visual']")
    object_collision = root.find(".//body[@name='cube1']/geom[@name='cube1_collision']")
    assert object_asset is not None
    assert object_asset.get("file", "").endswith("/objects/cube1/cube1.obj")
    assert object_asset.get("scale") == "0.001 0.001 0.001"
    assert object_visual is not None
    assert (object_visual.get("contype"), object_visual.get("conaffinity")) == ("0", "0")
    assert object_visual.get("group") == "2"
    assert object_collision is not None
    assert (object_collision.get("contype"), object_collision.get("conaffinity")) == ("2", "5")
    assert object_collision.get("group") == "3"


def test_compiled_visual_model_retains_collision_only_physics_selection() -> None:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco is not installed in this environment")

    mujoco, collision_model = compile_model()
    _, visual_model = compile_model(visual_meshes=True)
    assert collision_model.ngeom == 18
    assert visual_model.ngeom == 40
    assert (
        mujoco.mj_name2id(
            collision_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "palm_visual",
        )
        == -1
    )
    assert (
        mujoco.mj_name2id(
            collision_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "cube1_visual",
        )
        == -1
    )

    for name in ("palm_visual", "cube1_visual"):
        geom_id = mujoco.mj_name2id(visual_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert geom_id >= 0
        assert (visual_model.geom_contype[geom_id], visual_model.geom_conaffinity[geom_id]) == (
            0,
            0,
        )
    object_collision = mujoco.mj_name2id(
        visual_model, mujoco.mjtObj.mjOBJ_GEOM, "cube1_collision"
    )
    assert (visual_model.geom_contype[object_collision], visual_model.geom_conaffinity[object_collision]) == (
        2,
        5,
    )


def test_unified_visual_scene_contains_each_object_visual_mesh() -> None:
    root = ET.fromstring(
        build_unified_scene_xml(
            object_types=("cube1", "cube2"),
            visual_meshes=True,
        )
    )
    for object_type in ("cube1", "cube2"):
        visual = root.find(
            f".//body[@name='{object_type}']/geom[@name='{object_type}_visual']"
        )
        assert visual is not None
        assert visual.get("mesh") == f"{object_type}_visual_mesh"
        assert (visual.get("contype"), visual.get("conaffinity")) == ("0", "0")


def test_s02_object_registry_is_closed_materialized_and_digest_checked() -> None:
    expected = (
        "banana",
        "bottlewithcap",
        "bowl",
        "cube1",
        "cube2",
        "cuboid1",
        "cuboid2",
        "cuboid3",
        "cylinder1",
        "cylinder2",
        "cylinder3",
        "cylinder4",
        "cylinder5",
        "cylinder6",
        "cylinder7",
        "iphone",
        "largeclamp",
        "mayonnaisebottle",
        "pitcherbase",
        "powerdrill",
        "scissor",
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
        assert all(path.is_file() for path in runtime.collision_mesh_paths)
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
        geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
            and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").endswith(
                "_collision"
            )
        ]
        assert body_id >= 0 and len(geom_ids) == runtime.collision_geom_count
        assert float(model.body_mass[body_id]) > 0.0
        signatures[object_type] = (
            round(float(model.body_mass[body_id]), 6),
            tuple(np.round(np.ptp(object_collision_vertices(object_type), axis=0), 6)),
        )

    assert len(set(signatures.values())) == len(signatures)


def test_independent_zero_fk_contains_fixed_palm_rotation() -> None:
    transforms = urdf_zero_fk()
    # 28 actuated joints plus the fixed palm joint produce 30 link frames.
    assert len(transforms) == len(JOINT_NAMES) + 2
    palm = transforms["palm"]
    assert not np.allclose(palm[:3, :3], np.eye(3))
    np.testing.assert_allclose(palm[:3, 3], 0.0, atol=0.0)


def test_compiled_native_position_actuator_semantics_and_static_fk() -> None:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco is not installed in this environment")
    mujoco, model = compile_model()
    servo = ServoConfig()
    assert model.nq == 35
    assert model.nv == 34
    assert model.nu == len(JOINT_NAMES) == 28
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
