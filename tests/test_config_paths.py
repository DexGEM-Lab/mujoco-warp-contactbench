from pathlib import Path

from sim.dexhandrl.skrl_env import DexHandRLMJXEnvConfig


def test_env_config_derives_isaac_side_paths_from_root() -> None:
    isaac_root = Path("/tmp/isaac_source_root_test")
    cfg = DexHandRLMJXEnvConfig(isaac_source_root=isaac_root)

    assert cfg.isaac_source_root == isaac_root
    assert cfg.contact_bodies_file == isaac_root / "assets/all_assets/Assets/object_grasps_simple.yaml"
    assert cfg.task_config_path == isaac_root / "dexhand_env/cfg/task/Dexhand021proReconstruction.yaml"
