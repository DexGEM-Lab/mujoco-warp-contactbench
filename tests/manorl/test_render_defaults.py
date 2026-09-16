"""Shared lighting is visible in native models and cannot change physics."""
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from sim.manorl.assets import build_scene_xml, build_unified_scene_xml
from sim.manorl.view_environment import _configure_tiled_visuals


@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("visual_meshes", [False, True])
def test_scene_headlight_defaults_are_render_only(unified, visual_meshes):
    import mujoco

    xml = (
        build_unified_scene_xml(object_types=("cube1",), visual_meshes=visual_meshes)
        if unified else build_scene_xml(object_type="cube1", visual_meshes=visual_meshes)
    )
    root = ET.fromstring(xml)
    headlight = root.find("./visual/headlight")
    assert headlight is not None
    np.testing.assert_allclose(np.fromstring(headlight.get("ambient"), sep=" "), [0.4] * 3)
    np.testing.assert_allclose(np.fromstring(headlight.get("diffuse"), sep=" "), [0.65] * 3)
    model = mujoco.MjModel.from_xml_string(xml)
    np.testing.assert_allclose(model.vis.headlight.ambient, [0.4] * 3)
    np.testing.assert_allclose(model.vis.headlight.diffuse, [0.65] * 3)

    root.remove(root.find("visual"))
    previous = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for field in (
        "body_mass", "body_inertia", "dof_damping", "dof_frictionloss", "jnt_range",
        "geom_friction", "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp",
        "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange",
    ):
        np.testing.assert_array_equal(getattr(model, field), getattr(previous, field))
    np.testing.assert_array_equal(model.opt.gravity, previous.opt.gravity)
    assert model.opt.timestep == previous.opt.timestep


def test_tiled_viewer_preserves_default_headlight():
    model = SimpleNamespace(vis=SimpleNamespace(
        rgba=SimpleNamespace(fog=np.zeros(4), haze=np.zeros(4)),
        headlight=SimpleNamespace(ambient=np.zeros(3), diffuse=np.zeros(3), specular=np.zeros(3)),
    ))
    _configure_tiled_visuals(model)
    np.testing.assert_array_equal(model.vis.headlight.ambient, [0.4] * 3)
    np.testing.assert_array_equal(model.vis.headlight.diffuse, [0.65] * 3)
