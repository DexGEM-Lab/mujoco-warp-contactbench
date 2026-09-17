from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sim.manorl.assets import (
    HAND_SELF_COLLISION_GROUPS,
    asset_provenance,
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


def test_asset_provenance_matches_pinned_manifest() -> None:
    manifest = validate_asset_manifest("cube1")
    provenance = asset_provenance()
    assert provenance["asset_source_repository"] == manifest["source_repository"]
    assert provenance["asset_source_commit"] == manifest["source_commit"]
    assert len(provenance["asset_manifest_sha256"]) == 64
    assert all(
        character in "0123456789abcdef"
        for character in provenance["asset_manifest_sha256"]
    )


def test_manifest_and_generated_scene_preserve_authoritative_semantics() -> None:
    manifest = validate_asset_manifest("cube1")
    assert (
        manifest["source_repository"]
        == "git@github.com:DexGEM-Lab/dexstream_digital-assets.git"
    )
    assert manifest["source_commit"] == "f98da997f316c8a6b4bc2931cabed19e831ef163"
    assert manifest["hand_operator"] == "sunke"
    assert set(manifest["hands"]) == {"left", "right"}
    assert len(manifest["objects"]) == 28
    assert manifest["task_metadata"]["grasp_mapping"]["storage"] == "project"

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
    assert palm_mesh is not None and palm_mesh.get("scale") is None


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


@pytest.mark.parametrize(
    ("hand_side", "expected_nq", "expected_nv", "expected_nu", "expected_hand_geoms"),
    (
        ("right", 35, 34, 28, 16),
        ("left", 35, 34, 28, 16),
        ("both", 63, 62, 56, 32),
    ),
)
def test_dexstream_single_and_bimanual_models_compile(
    hand_side: str,
    expected_nq: int,
    expected_nv: int,
    expected_nu: int,
    expected_hand_geoms: int,
) -> None:
    mujoco, model = compile_model(object_type="cube1", hand_side=hand_side)
    assert (model.nq, model.nv, model.nu) == (expected_nq, expected_nv, expected_nu)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube1")
    hand_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) not in (0, object_id)
    ]
    assert len(hand_geoms) == expected_hand_geoms
    if hand_side == "both":
        actuator_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(model.nu)
        )
        assert all(name.startswith("right_") for name in actuator_names[:28])
        assert all(name.startswith("left_") for name in actuator_names[28:])


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


def test_visual_scene_replaces_rigid_partition_with_mano_skin() -> None:
    root = ET.fromstring(build_scene_xml(visual_meshes=True))

    # The rigid per-link visual partition is gone; the MANO skin replaces it.
    hand_visual_assets = [
        mesh
        for mesh in root.findall("./asset/mesh")
        if mesh.get("name", "").startswith("hand_visual_")
    ]
    assert hand_visual_assets == []
    assert root.find(".//body[@name='palm']/geom[@name='palm_visual']") is None

    skin = root.find("./deformable/skin")
    assert skin is not None
    assert skin.get("name") == "mano_skin"
    assert skin.get("rgba") == "0.95 0.75 0.65 1"
    bones = skin.findall("bone")
    assert len(bones) == 16
    assert {bone.get("body") for bone in bones} == {
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
    }

    object_asset = root.find("./asset/mesh[@name='cube1_visual_mesh']")
    object_visual = root.find(".//body[@name='cube1']/geom[@name='cube1_visual']")
    object_collision = root.find(".//body[@name='cube1']/geom[@name='cube1_collision']")
    assert object_asset is not None
    assert object_asset.get("file", "").endswith("/objects/DexGEM/cube1/cube1.obj")
    assert object_asset.get("scale") == "0.001 0.001 0.001"
    assert object_visual is not None
    assert (object_visual.get("contype"), object_visual.get("conaffinity")) == ("0", "0")
    assert object_visual.get("group") == "2"
    assert object_collision is not None
    assert (object_collision.get("contype"), object_collision.get("conaffinity")) == ("2", "5")
    assert object_collision.get("group") == "3"


def test_bimanual_visual_scene_binds_one_skin_per_hand() -> None:
    root = ET.fromstring(build_scene_xml(hand_side="both", visual_meshes=True))
    skins = root.findall("./deformable/skin")
    assert [skin.get("name") for skin in skins] == ["right_mano_skin", "left_mano_skin"]
    for skin, prefix in zip(skins, ("right_", "left_"), strict=True):
        bones = skin.findall("bone")
        assert len(bones) == 16
        assert all(bone.get("body", "").startswith(prefix) for bone in bones)


def test_compiled_visual_model_retains_collision_only_physics_selection() -> None:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco is not installed in this environment")

    mujoco, collision_model = compile_model()
    _, visual_model = compile_model(visual_meshes=True)
    assert collision_model.ngeom == 18
    # 16 hand collision geoms + cube1 collision + floor + cube1 visual; hand
    # visuals are the skinned MANO surface and add no geoms.
    assert visual_model.ngeom == 19
    assert visual_model.nskin == 1
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

    assert (
        mujoco.mj_name2id(visual_model, mujoco.mjtObj.mjOBJ_GEOM, "palm_visual")
        == -1
    )
    cube1_visual = mujoco.mj_name2id(
        visual_model, mujoco.mjtObj.mjOBJ_GEOM, "cube1_visual"
    )
    assert cube1_visual >= 0
    assert (
        visual_model.geom_contype[cube1_visual],
        visual_model.geom_conaffinity[cube1_visual],
    ) == (0, 0)
    object_collision = mujoco.mj_name2id(
        visual_model, mujoco.mjtObj.mjOBJ_GEOM, "cube1_collision"
    )
    assert (
        visual_model.geom_contype[object_collision],
        visual_model.geom_conaffinity[object_collision],
    ) == (2, 5)


def test_mano_skin_bind_frames_match_compiled_zero_pose() -> None:
    if importlib.util.find_spec("mujoco") is None:
        pytest.skip("mujoco is not installed in this environment")

    import sim.manorl.assets as assets

    for side in ("right", "left"):
        mujoco, model = compile_model(object_type="cube1", hand_side=side, visual_meshes=True)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        fragment = ET.parse(assets._hand_skin_path(side)).getroot()
        skin = fragment.find("skin")
        assert model.nskin == 1
        max_pos_error = 0.0
        max_quat_error = 0.0
        for bone in skin.findall("bone"):
            body_name = bone.get("body")
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            assert body_id >= 0, body_name
            bindpos = np.fromstring(bone.get("bindpos", ""), sep=" ")
            bindquat = np.fromstring(bone.get("bindquat", ""), sep=" ")
            max_pos_error = max(
                max_pos_error, float(np.linalg.norm(data.xpos[body_id] - bindpos))
            )
            max_quat_error = max(
                max_quat_error,
                min(
                    float(np.linalg.norm(data.xquat[body_id] - bindquat)),
                    float(np.linalg.norm(data.xquat[body_id] + bindquat)),
                ),
            )
        assert max_pos_error < 5e-8, f"{side} skin bind drift {max_pos_error}"
        assert max_quat_error < 1e-9, f"{side} skin bind quat drift {max_quat_error}"

    mujoco, bimanual = compile_model(
        object_type="cube1", hand_side="both", visual_meshes=True
    )
    assert bimanual.nskin == 2
    right_skin = ET.parse(assets._hand_skin_path("right")).getroot().find("skin")
    for bone in right_skin.findall("bone"):
        assert (
            mujoco.mj_name2id(
                bimanual, mujoco.mjtObj.mjOBJ_BODY, f"right_{bone.get('body')}"
            )
            >= 0
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


def test_dexstream_object_registry_is_discovered_materialized_and_digest_checked() -> None:
    manifest = validate_asset_manifest()
    expected = tuple(sorted(manifest["objects"]))
    assert len(expected) == 28
    assert supported_object_types() == expected
    assert {"banana", "cube2", "largeclamp", "mayonnaisebottle"} <= set(expected)
    assert not {"bottlewithcap", "scissor", "iphone17", "iphone17_T", "phone_sunke"} & set(expected)
    assert object_runtime("camera1").urdf_path.name == "CAMERA1.urdf"
    assert object_runtime("camera2").urdf_path.name == "CAMERA2.urdf"

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


def test_all_dexstream_object_models_compile_with_object_specific_mass_and_geometry() -> None:
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

    # Every DexStream object has its own authoritative URDF path, even when
    # source meshes intentionally share physical geometry.
    assert len(signatures) == len(supported_object_types())
    assert len({object_runtime(name).urdf_path for name in signatures}) == len(signatures)


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
