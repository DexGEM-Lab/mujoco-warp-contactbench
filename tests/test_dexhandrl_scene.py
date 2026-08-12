import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from sim.dexhandrl.constants import DEFAULT_ISAAC_SOURCE_ROOT
from sim.dexhandrl.contracts import DexHandMaterialConfig, DexHandPhysicsConfig, DexHandServoConfig
from sim.dexhandrl.scene import (
    _compiled_exclude_pairs,
    _hand_self_collision_exclude_pairs,
    _object_contact_xml,
    _parse_object_asset,
    _parse_urdf_origin,
    _patch_hand_xml,
    _rpy_to_wxyz,
    build_dexhandrl_scene_xml,
    compile_dexhandrl_model,
    validate_dexhandrl_compiled_model,
)


_HAND_XML = '''<mujoco>
  <compiler meshdir="old" />
  <default><geom friction="3 2 2" solref="0.01 0.9" /></default>
  <size njmax="1" nconmax="1" />
</mujoco>'''


def test_rpy_to_wxyz_matches_urdf_fixed_axis_rotations() -> None:
    assert _rpy_to_wxyz((0.0, 0.0, 0.0)) == (1.0, 0.0, 0.0, 0.0)
    assert _rpy_to_wxyz((math.pi, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0, 0.0), abs=1e-15)
    assert _rpy_to_wxyz((0.0, math.pi / 2.0, math.pi / 2.0)) == pytest.approx(
        (0.5, -0.5, 0.5, 0.5), abs=1e-15
    )


def test_parse_cube1_source_origins_exact() -> None:
    asset = _parse_object_asset("cube1", DEFAULT_ISAAC_SOURCE_ROOT)

    assert asset["inertial_pos"] == pytest.approx(
        (0.005175592236, -0.005187185076, 0.001516276923), abs=0.0
    )
    assert asset["inertial_quat"] == (1.0, 0.0, 0.0, 0.0)
    assert asset["urdf_path"].as_posix().endswith("Assets/sim/decomposed/cube1/cube1.urdf")
    assert len(asset["collisions"]) == 1
    assert asset["collisions"][0]["position"] == (0.0, 0.0, 0.0)
    assert asset["collisions"][0]["quaternion"] == (1.0, 0.0, 0.0, 0.0)
    assert asset["collisions"][0]["scale"] == (1.0, 1.0, 1.0)
    assert asset["collisions"][0]["mesh_path"].name == "coacd_convex_piece_0.obj"
    assert asset["visual"]["scale"] == (0.001, 0.001, 0.001)
    assert asset["visual"]["mesh_path"].name == "cube1.obj"


def test_origin_parser_defaults_only_absent_fields_and_rejects_nonfinite() -> None:
    assert _parse_urdf_origin(None) == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    origin = ET.fromstring("<origin xyz='1 2 3'/>")
    assert _parse_urdf_origin(origin) == ((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        _parse_urdf_origin(ET.fromstring("<origin xyz='nan 0 0'/>"))


def test_object_parser_rejects_nonfinite_inertia(tmp_path: Path) -> None:
    urdf_dir = tmp_path / "assets/all_assets/Assets/sim/decomposed/bad"
    urdf_dir.mkdir(parents=True)
    (urdf_dir / "mesh.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    (urdf_dir / "bad.urdf").write_text("""<robot><link><visual><geometry><mesh filename='mesh.obj'/></geometry></visual><collision><geometry><mesh filename='mesh.obj'/></geometry></collision><inertial><mass value='1'/><inertia ixx='nan' iyy='1' izz='1' ixy='0' ixz='0' iyz='0'/></inertial></link></robot>""")
    with pytest.raises(ValueError, match="inertia ixx"):
        _parse_object_asset("bad", tmp_path)


def test_bowl_parser_and_scene_keep_all_decomposed_collision_pieces(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    asset = _parse_object_asset("bowl", DEFAULT_ISAAC_SOURCE_ROOT)
    assert len(asset["collisions"]) == 28
    assert [collision["mesh_path"].name for collision in asset["collisions"]] == [
        f"coacd_convex_piece_{index}.obj" for index in range(28)
    ]

    xml_path = build_dexhandrl_scene_xml(tmp_path / "bowl.xml", "bowl")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    validate_dexhandrl_compiled_model(mujoco, model, object_name="bowl")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bowl")
    object_geoms = [index for index in range(model.ngeom) if int(model.geom_bodyid[index]) == body_id]
    assert len(object_geoms) == 29
    assert sum(int(model.geom_contype[index]) != 0 for index in object_geoms) == 28


def test_generated_cube1_xml_and_compiled_inertial_frame(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    xml_path = build_dexhandrl_scene_xml(tmp_path / "cube1.xml", "cube1")
    root = ET.parse(xml_path).getroot()
    inertial = root.find(".//body[@name='cube1']/inertial")
    geom = root.find(".//geom[@name='cube1_collision']")
    visual = root.find(".//geom[@name='cube1_visual']")
    mesh = root.find(".//asset/mesh[@name='cube1_mesh']")
    visual_mesh = root.find(".//asset/mesh[@name='cube1_visual_mesh']")
    assert inertial is not None and geom is not None and visual is not None
    assert mesh is not None and visual_mesh is not None
    assert tuple(float(value) for value in inertial.attrib["pos"].split()) == pytest.approx(
        (0.005175592236, -0.005187185076, 0.001516276923), abs=0.0
    )
    assert "quat" not in inertial.attrib
    assert tuple(float(value) for value in geom.attrib["pos"].split()) == (0.0, 0.0, 0.0)
    assert tuple(float(value) for value in geom.attrib["quat"].split()) == (1.0, 0.0, 0.0, 0.0)
    assert mesh.attrib["scale"] == "1 1 1"
    assert mesh.attrib["file"].endswith("Assets/sim/decomposed/cube1/coacd/coacd_convex_piece_0.obj")
    assert visual_mesh.attrib["scale"] == "0.001 0.001 0.001"
    assert visual_mesh.attrib["file"].endswith("Assets/sim/mano_assets/objects/cube1/cube1.obj")
    assert (geom.attrib["contype"], geom.attrib["conaffinity"], geom.attrib["group"]) == ("1", "1", "3")
    assert (visual.attrib["contype"], visual.attrib["conaffinity"], visual.attrib["group"]) == ("0", "0", "2")
    assert inertial.attrib["mass"] == "0.125"
    assert inertial.attrib["fullinertia"] == "0.0001333333352 0.0001333333353 0.0001333333366 2.97890911e-13 -6.782344705e-13 -2.157039074e-13"
    assert geom.attrib["friction"] == "0.5 0.01 0.001"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube1")
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube1_collision")
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "cube1_mesh")
    assert model.body_ipos[body_id] == pytest.approx((0.005175592236, -0.005187185076, 0.001516276923), abs=1e-14)
    assert model.body_mass[body_id] == pytest.approx(0.125, abs=1e-15)
    assert sorted(model.body_inertia[body_id]) == pytest.approx(sorted((0.0001333333352, 0.0001333333353, 0.0001333333366)), abs=1e-15)
    # MuJoCo recenters imported mesh vertices and stores the compensating
    # transform in mesh_pos. The CoACD piece center differs from the source
    # inertial COM by ~1e-8 m, so the compiled mesh transform is authoritative.
    assert model.geom_pos[geom_id] == pytest.approx(model.mesh_pos[mesh_id], abs=2e-12)
    assert model.geom_pos[geom_id] == pytest.approx(model.body_ipos[body_id], abs=2e-8)
    assert model.geom_friction[geom_id] == pytest.approx((0.5, 0.01, 0.001), abs=1e-15)


def test_default_xml_matches_frozen_oracle_and_physics_legacy_resolution(tmp_path: Path) -> None:
    frozen = Path("outputs/dexhandrl_refactor/manorl_style_baseline_20260723/cube1_scene.xml")
    default = build_dexhandrl_scene_xml(tmp_path / "default.xml", "cube1")
    canonical = build_dexhandrl_scene_xml(tmp_path / "canonical.xml", "cube1", physics=DexHandPhysicsConfig())
    legacy = build_dexhandrl_scene_xml(tmp_path / "legacy.xml", "cube1", dt=0.01, substeps=10, finger_kp=15)
    assert default.read_bytes() == canonical.read_bytes() == legacy.read_bytes()
    assert hashlib.sha256(default.read_bytes()).hexdigest() == "2291e5d9fb3cae6233da643acfea8804b833f9a951d439e984db0ce871732497"
    if frozen.exists():
        assert default.read_bytes() != frozen.read_bytes()
    with pytest.raises(ValueError, match="cannot be combined"):
        build_dexhandrl_scene_xml(tmp_path / "bad.xml", "cube1", physics=DexHandPhysicsConfig(), dt=0.01)


def test_compiled_validator_rejects_option_actuator_and_material_mutations() -> None:
    mujoco = pytest.importorskip("mujoco")
    mujoco, model = compile_dexhandrl_model()
    validate_dexhandrl_compiled_model(mujoco, model)
    model.opt.iterations += 1
    with pytest.raises(ValueError, match="solver"):
        validate_dexhandrl_compiled_model(mujoco, model)
    model.opt.iterations -= 1
    model.actuator_gainprm[0, 0] += 1
    with pytest.raises(ValueError, match="gain"):
        validate_dexhandrl_compiled_model(mujoco, model)
    model.actuator_gainprm[0, 0] -= 1
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube1_collision")
    model.geom_friction[object_geom, 0] += 1
    with pytest.raises(ValueError, match="object friction"):
        validate_dexhandrl_compiled_model(mujoco, model)


def test_object_contact_xml_preserves_default_bytes_and_emits_custom_tails() -> None:
    assert _object_contact_xml(DexHandPhysicsConfig()) == ("0.006 1", "0.95 0.995 0.0001")
    custom = DexHandPhysicsConfig(material=DexHandMaterialConfig(object_solimp=(0.95, 0.995, 0.0001, 0.75, 3.0)))
    assert _object_contact_xml(custom) == ("0.006 1", "0.95 0.995 0.0001 0.75 3")
    with pytest.raises(ValueError, match="time constant"):
        DexHandMaterialConfig(object_solref=(0.0, 1.0))


def test_hand_friction_patch_changes_only_canonical_hand_default() -> None:
    patched = _patch_hand_xml(_HAND_XML, DEFAULT_ISAAC_SOURCE_ROOT, dt=0.01, substeps=10, hand_friction="3 2 2")
    with pytest.raises(ValueError, match="cannot be combined"):
        _patch_hand_xml(_HAND_XML, DEFAULT_ISAAC_SOURCE_ROOT, dt=0.01, physics=DexHandPhysicsConfig())
    assert 'friction="3 2 2" solref=' in patched
    assert patched.count('friction="3 2 2"') == 1


@pytest.mark.parametrize("xml", [
    _HAND_XML.replace('friction="3 2 2"', 'friction="0.5 0.01 0.001"'),
    _HAND_XML.replace('</default>', '<geom friction="3 2 2" solref="0.01 0.9" /></default>'),
])
def test_hand_friction_patch_fails_closed_without_one_canonical_default(xml: str) -> None:
    with pytest.raises(RuntimeError, match="exactly one canonical hand default"):
        _patch_hand_xml(xml, DEFAULT_ISAAC_SOURCE_ROOT, dt=0.01, substeps=10, hand_friction="3 2 2")


@pytest.mark.parametrize("mode, expected_count", [("none", 0), ("disable_within_finger", 40), ("all_disabled", 210)])
def test_compiled_self_collision_exclude_signatures_exact(mode: str, expected_count: int) -> None:
    mujoco = pytest.importorskip("mujoco")
    physics = DexHandPhysicsConfig(hand_self_collision_mode=mode)
    mujoco, model = compile_dexhandrl_model(physics=physics)
    assert model.nexclude == expected_count
    assert _compiled_exclude_pairs(mujoco, model) == _hand_self_collision_exclude_pairs(mode)
    if expected_count:
        model.exclude_signature[0] = model.exclude_signature[1]
        with pytest.raises(ValueError, match="self-collision"):
            validate_dexhandrl_compiled_model(mujoco, model, physics)


def test_high_precision_scene_contract_compiles(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    physics = DexHandPhysicsConfig(
        control_dt=0.010000000000001,
        servo=DexHandServoConfig(finger_kp=15.123456789012345),
        material=DexHandMaterialConfig(object_friction=(0.5000000000000001, 0.010000000000001, 0.001000000000001)),
    )
    path = build_dexhandrl_scene_xml(tmp_path / "precise.xml", "cube1", physics=physics)
    assert 'kp="15.123456789012344"' in path.read_text()
    model = mujoco.MjModel.from_xml_path(str(path))
    validate_dexhandrl_compiled_model(mujoco, model, physics)
