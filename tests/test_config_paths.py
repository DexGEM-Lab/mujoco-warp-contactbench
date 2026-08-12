from pathlib import Path

from sim.dexhandrl.asset_paths import (
    PINNED_ALL_ASSETS_COMMIT,
    PINNED_MANO_ASSETS_COMMIT,
    asset_provenance,
    contact_bodies_path,
    object_mesh_provenance,
    object_urdf_path,
    task_config_path,
)
from sim.dexhandrl.constants import DEFAULT_ISAAC_SOURCE_ROOT
from sim.dexhandrl.skrl_env import DexHandRLMJXEnvConfig


def test_default_asset_root_resolves_pinned_decomposed_runtime() -> None:
    cfg = DexHandRLMJXEnvConfig()
    provenance = asset_provenance("cube1", DEFAULT_ISAAC_SOURCE_ROOT)

    assert cfg.isaac_source_root == DEFAULT_ISAAC_SOURCE_ROOT
    assert cfg.contact_bodies_file == contact_bodies_path(DEFAULT_ISAAC_SOURCE_ROOT)
    assert cfg.task_config_path == task_config_path(DEFAULT_ISAAC_SOURCE_ROOT)
    assert object_urdf_path("cube1", DEFAULT_ISAAC_SOURCE_ROOT).as_posix().endswith(
        "assets/all_assets/Assets/sim/decomposed/cube1/cube1.urdf"
    )
    assert provenance["all_assets_commit"] == PINNED_ALL_ASSETS_COMMIT
    assert provenance["mano_assets_commit"] == PINNED_MANO_ASSETS_COMMIT
    assert provenance["schema"] == "dexhandrl.asset_provenance.v2"
    assert provenance["object_urdf_sha256"] == (
        "b9685e086c79cb6d5198793fcdd89ec8772f1b5d5a73658d94ce0c574d72e9f8"
    )
    assert provenance["visual_mesh"]["sha256"] == (
        "d33f5fdf5276d2e8117cbf58deebd6f353e23dbb86a5dfb0475f9a523d2b54b1"
    )
    assert provenance["collision_mesh_count"] == 1
    assert provenance["collision_meshes"][0]["sha256"] == (
        "edf46dc464fbffd2619ef2f0a0450db44577820af60440fe8e6fefc2eaa1be61"
    )


def test_multi_piece_asset_provenance_keeps_every_collision_mesh() -> None:
    provenance = object_mesh_provenance("bowl", DEFAULT_ISAAC_SOURCE_ROOT)

    assert provenance["collision_mesh_count"] == 28
    assert len(provenance["collision_meshes"]) == 28
    assert [piece["index"] for piece in provenance["collision_meshes"]] == list(range(28))
    assert len({piece["path"] for piece in provenance["collision_meshes"]}) == 28
    assert all(len(piece["sha256"]) == 64 for piece in provenance["collision_meshes"])
    assert provenance["visual_mesh"]["path"].endswith("/mano_assets/objects/bowl/bowl.obj")


def test_explicit_outer_source_root_derives_assets_and_task_config(tmp_path: Path) -> None:
    all_assets = tmp_path / "assets" / "all_assets" / "Assets"
    all_assets.mkdir(parents=True)
    contact = all_assets / "object_grasps_simple.yaml"
    contact.write_text("cube1: {}\n", encoding="utf-8")
    task = tmp_path / "dexhand_env" / "cfg" / "task" / "Dexhand021proReconstruction.yaml"
    task.parent.mkdir(parents=True)
    task.write_text("task: {}\n", encoding="utf-8")

    cfg = DexHandRLMJXEnvConfig(isaac_source_root=tmp_path)

    assert cfg.isaac_source_root == tmp_path
    assert cfg.contact_bodies_file == contact
    assert cfg.task_config_path == task
