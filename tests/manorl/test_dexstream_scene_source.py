from __future__ import annotations

import importlib.util

import pytest

from sim.scene import build_live_scene_xml, resolve_mano_asset


def test_generic_scene_resolves_only_pinned_dexstream_mano(tmp_path) -> None:
    path = resolve_mano_asset()
    assert "assets/dexstream_digital_assets/hand/mano/sunke/right" in path.as_posix()
    output = tmp_path / "mano.xml"
    build_live_scene_xml(output)
    xml = output.read_text(encoding="utf-8")
    assert "mano_hand_s02" not in xml
    assert "../meshes/" not in xml
    assert "../skin/" not in xml
    assert "dexstream_digital_assets" in xml


@pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco is not installed")
def test_generic_scene_compiles_with_dexstream_skin(tmp_path) -> None:
    import mujoco

    output = tmp_path / "mano.xml"
    build_live_scene_xml(output)
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.nq == 28
    assert model.nv == 28
    assert model.nskin == 1


def test_mano_pose_axes_are_read_from_the_pinned_urdf() -> None:
    import xml.etree.ElementTree as ET
    import numpy as np

    from sim.manorl.assets import hand_urdf_path
    from sim.manorl.mano_pose import _AXIS_JOINTS, _source_axes

    root = ET.parse(hand_urdf_path("right")).getroot()
    source = {
        joint.get("name"): np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        for joint in root.findall("joint")
        if joint.find("axis") is not None
    }
    for logical, joint_name in _AXIS_JOINTS.items():
        np.testing.assert_allclose(_source_axes()[logical], source[joint_name])


def test_dataset_manifest_asset_provenance_is_checked(monkeypatch) -> None:
    from tools import validate_manorl_synthetic_lance as validator

    identity = {
        "asset_source_repository": "repo",
        "asset_source_commit": "commit",
        "asset_manifest_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        "sim.manorl.assets.asset_provenance", lambda: dict(identity)
    )
    assert validator._validate_asset_provenance({"asset_provenance": identity}) == identity
    with pytest.raises(ValueError, match="asset provenance"):
        validator._validate_asset_provenance(
            {"asset_provenance": {**identity, "asset_source_commit": "other"}}
        )
    assert validator._validate_asset_provenance({}) is None


def test_manifest_closes_direct_mano_urdf_visual_dependencies() -> None:
    import json
    from sim.manorl.assets import ASSET_MANIFEST

    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    for side in ("right", "left"):
        hand = manifest["hands"][side]
        assert len(hand["visual_files"]) == 16
        paths = {record["path"] for record in hand["files"]}
        assert set(hand["visual_files"]) <= paths
        assert set(hand["skin_files"]) <= paths
