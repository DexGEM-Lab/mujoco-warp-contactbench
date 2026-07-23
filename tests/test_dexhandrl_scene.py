import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from sim.dexhandrl.constants import DEFAULT_ISAAC_SOURCE_ROOT
from sim.dexhandrl.scene import (
    _parse_object_asset,
    _parse_urdf_origin,
    _patch_hand_xml,
    _rpy_to_wxyz,
    build_dexhandrl_scene_xml,
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
    assert asset["collision_pos"] == (0.0, 0.0, 0.0)
    assert asset["collision_quat"] == (1.0, 0.0, 0.0, 0.0)


def test_origin_parser_defaults_only_absent_fields_and_rejects_nonfinite() -> None:
    assert _parse_urdf_origin(None) == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    origin = ET.fromstring("<origin xyz='1 2 3'/>")
    assert _parse_urdf_origin(origin) == ((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        _parse_urdf_origin(ET.fromstring("<origin xyz='nan 0 0'/>"))


def test_object_parser_rejects_nonfinite_inertia(tmp_path: Path) -> None:
    urdf_dir = tmp_path / "assets/all_assets/Assets/sim/mano_objects_urdf"
    urdf_dir.mkdir(parents=True)
    (urdf_dir / "bad.urdf").write_text("""<robot><link><collision><geometry><mesh filename='mesh.obj'/></geometry></collision><inertial><mass value='1'/><inertia ixx='nan' iyy='1' izz='1' ixy='0' ixz='0' iyz='0'/></inertial></link></robot>""")
    with pytest.raises(ValueError, match="inertia ixx"):
        _parse_object_asset("bad", tmp_path)


def test_generated_cube1_xml_and_compiled_inertial_frame(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    xml_path = build_dexhandrl_scene_xml(tmp_path / "cube1.xml", "cube1")
    root = ET.parse(xml_path).getroot()
    inertial = root.find(".//body[@name='cube1']/inertial")
    geom = root.find(".//geom[@name='cube1_collision']")
    mesh = root.find(".//asset/mesh[@name='cube1_mesh']")
    assert inertial is not None and geom is not None and mesh is not None
    assert tuple(float(value) for value in inertial.attrib["pos"].split()) == pytest.approx(
        (0.005175592236, -0.005187185076, 0.001516276923), abs=0.0
    )
    # MuJoCo forbids an inertial orientation alongside fullinertia; identity is implicit.
    assert "quat" not in inertial.attrib
    assert tuple(float(value) for value in geom.attrib["pos"].split()) == (0.0, 0.0, 0.0)
    assert tuple(float(value) for value in geom.attrib["quat"].split()) == (1.0, 0.0, 0.0, 0.0)
    assert mesh.attrib["scale"] == "0.001 0.001 0.001"
    assert inertial.attrib["mass"] == "0.125"
    assert inertial.attrib["fullinertia"] == "0.0001333333352 0.0001333333353 0.0001333333366 2.97890911e-13 -6.782344705e-13 -2.157039074e-13"
    assert geom.attrib["friction"] == "0.5 0.01 0.001"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube1")
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube1_collision")
    assert model.body_ipos[body_id] == pytest.approx(
        (0.005175592236, -0.005187185076, 0.001516276923), abs=1e-14
    )
    assert model.body_mass[body_id] == pytest.approx(0.125, abs=1e-15)
    assert sorted(model.body_inertia[body_id]) == pytest.approx(
        sorted((0.0001333333352, 0.0001333333353, 0.0001333333366)), abs=1e-15
    )
    # The XML collision origin is zero; compilation relocates the free body to its COM.
    assert model.geom_pos[geom_id] == pytest.approx(model.body_ipos[body_id], abs=2e-12)
    assert model.geom_friction[geom_id] == pytest.approx((0.5, 0.01, 0.001), abs=1e-15)


def test_hand_friction_patch_changes_only_canonical_hand_default(tmp_path: Path) -> None:
    patched = _patch_hand_xml(_HAND_XML, tmp_path, dt=0.01, substeps=10, hand_friction="3 2 2")

    assert 'friction="3 2 2" solref=' in patched
    assert patched.count('friction="3 2 2"') == 1


@pytest.mark.parametrize("xml", [
    _HAND_XML.replace('friction="3 2 2"', 'friction="0.5 0.01 0.001"'),
    _HAND_XML.replace('</default>', '<geom friction="3 2 2" solref="0.01 0.9" /></default>'),
])
def test_hand_friction_patch_fails_closed_without_one_canonical_default(xml: str, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exactly one canonical hand default"):
        _patch_hand_xml(xml, tmp_path, dt=0.01, substeps=10, hand_friction="3 2 2")
