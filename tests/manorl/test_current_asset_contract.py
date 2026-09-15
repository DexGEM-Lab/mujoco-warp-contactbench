"""Current DexStream root entries, marker frames, and visual resource closure."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim.manorl import assets
from tools.generate_manorl_asset_manifest import _all_dexstream_objects, generate


def test_generator_ignores_calibration_snapshots_and_reproduces_manifest() -> None:
    objects = _all_dexstream_objects(assets.DEXSTREAM_ROOT)
    assert len(objects) == 32
    for _, source_name, path in objects:
        assert path == f"objects/DexGEM/{source_name}/{source_name}.urdf"
    assert generate(assets.DEXSTREAM_ROOT, assets.REPOSITORY_ROOT) == json.loads(
        assets.ASSET_MANIFEST.read_text()
    )


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=np.float64))
    return result.reshape(3, 3)


@pytest.mark.parametrize("name", ("egg_cup", "egg_ellipsoid", "egg_stick_rack", "trash_bin"))
def test_collision_bounds_use_compiled_object_body_frame(name: str) -> None:
    _, model = assets.compile_model(object_type=name)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    compiled = []
    for geom_id in np.flatnonzero(model.geom_bodyid == body_id):
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
        compiled.append(vertices @ _rotation(model.geom_quat[geom_id]).T + model.geom_pos[geom_id])
    actual = assets.object_collision_vertices(name)
    expected = np.concatenate(compiled)
    np.testing.assert_allclose(actual.min(axis=0), expected.min(axis=0), atol=2e-7, rtol=0)
    np.testing.assert_allclose(actual.max(axis=0), expected.max(axis=0), atol=2e-7, rtol=0)


def test_urdf_physics_matches_authoritative_root_mjcf() -> None:
    """Guard the compatibility adapter against future URDF/MJCF physical drift."""
    manifest = assets.validate_asset_manifest()
    for name, record in manifest["objects"].items():
        source = ET.parse(assets.DEXSTREAM_ROOT / record["mjcf"]["path"]).getroot()
        generated = ET.fromstring(assets.build_scene_xml(object_type=name))
        source_body = source.find("./worldbody/body")
        actual_body = generated.find(f"./worldbody/body[@name='{name}']")
        assert source_body is not None and actual_body is not None
        for key in ("pos", "mass", "fullinertia"):
            np.testing.assert_allclose(
                np.fromstring(actual_body.find("inertial").get(key), sep=" "),
                np.fromstring(source_body.find("inertial").get(key), sep=" "),
                atol=1e-10, rtol=1e-8, err_msg=f"{name}: {key}",
            )
        expected_geoms = [g for g in source_body.findall("geom") if g.get("contype") != "0"]
        actual_geoms = actual_body.findall("geom")
        assert len(actual_geoms) == len(expected_geoms)
        source_meshes = {m.get("name"): m for m in source.findall("./asset/mesh")}
        generated_meshes = {m.get("name"): m for m in generated.findall("./asset/mesh")}
        for actual, expected in zip(actual_geoms, expected_geoms, strict=True):
            am, em = generated_meshes[actual.get("mesh")], source_meshes[expected.get("mesh")]
            assert Path(am.get("file")) == (assets.DEXSTREAM_ROOT / record["root"] / em.get("file")).resolve()
            np.testing.assert_allclose(np.fromstring(am.get("scale", "1 1 1"), sep=" "), np.fromstring(em.get("scale", "1 1 1"), sep=" "))
            np.testing.assert_allclose(np.fromstring(actual.get("pos", "0 0 0"), sep=" "), np.fromstring(expected.get("pos", "0 0 0"), sep=" "), atol=1e-10)
            np.testing.assert_allclose(_rotation(np.fromstring(actual.get("quat", "1 0 0 0"), sep=" ")), _rotation(np.fromstring(expected.get("quat", "1 0 0 0"), sep=" ")), atol=1e-10)


@pytest.mark.parametrize("name", ("bowl", "egg_ellipsoid", "trash_bin"))
def test_selected_source_texture_is_loaded_without_changing_collision_physics(name: str) -> None:
    _, model = assets.compile_model(object_type=name, visual_meshes=True)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_visual")
    texture = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, f"{name}_visual_texture")
    material = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, f"{name}_visual_material")
    assert texture >= 0 and material >= 0
    assert model.geom_matid[geom] == material
    assert texture in model.mat_texid[material]
    assert (model.geom_contype[geom], model.geom_conaffinity[geom]) == (0, 0)
    assert model.nskin == 1


def test_unified_visual_materials_are_namespaced_per_object() -> None:
    _, model = assets.compile_unified_model(
        object_types=("cube1", "cube2", "bowl", "cylinder7"), visual_meshes=True,
        object_collisions=True,
    )
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, f"{name}_visual_material")
           for name in ("cube1", "cube2", "bowl", "cylinder7")]
    assert min(ids) >= 0 and len(set(ids)) == 4


@pytest.mark.parametrize("failure", ("missing", "pointer", "digest"))
def test_texture_integrity_is_checked_before_scene_loading(tmp_path, monkeypatch, failure) -> None:
    record = assets._asset_manifest()["objects"]["bowl"]["textures"][0]
    actual_path = assets._manifest_record_path(record)
    replacement = tmp_path / "texture.png"
    if failure == "pointer":
        replacement.write_bytes(assets._LFS_POINTER_PREFIX)
    elif failure == "digest":
        content = bytearray(actual_path.read_bytes())
        content[-1] ^= 1
        replacement.write_bytes(content)
    original = assets._manifest_record_path
    monkeypatch.setattr(assets, "_manifest_record_path", lambda r: replacement if r["path"] == record["path"] else original(r))
    with pytest.raises((FileNotFoundError, ValueError), match="absent|not materialized|digest mismatch"):
        assets.validate_asset_manifest("bowl")


def test_uninitialized_unified_scene_fk_does_not_solve_overlapping_objects(monkeypatch) -> None:
    def forbidden_forward(*args, **kwargs):
        raise AssertionError("static FK must not solve contacts at uninitialized object poses")
    monkeypatch.setattr(mujoco, "mj_forward", forbidden_forward)
    _, model = assets.compile_unified_model(
        object_types=("cylinder7", "egg_cup", "egg_stick_rack"), object_collisions=True,
    )
    assert model.nq == 28 + 3 * 7
    assert model.nu == 28
