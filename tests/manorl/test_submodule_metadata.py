from configparser import ConfigParser
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_PATH = "assets/dexstream_digital_assets"
ASSET_URL = "git@github.com:DexGEM-Lab/dexstream_digital-assets.git"
ASSET_BRANCH = "main"
ASSET_COMMIT = "f98da997f316c8a6b4bc2931cabed19e831ef163"


def test_dexstream_is_the_only_physical_asset_submodule() -> None:
    modules = ConfigParser()
    modules.read(REPO_ROOT / ".gitmodules", encoding="utf-8")

    sections = set(modules.sections())
    assert sections == {
        'submodule "assets/dexstream_digital_assets"',
        'submodule "lance_manager"',
    }
    section = 'submodule "assets/dexstream_digital_assets"'
    assert modules[section]["path"] == ASSET_PATH
    assert modules[section]["url"] == ASSET_URL
    assert modules[section]["branch"] == ASSET_BRANCH
    assert 'submodule "assets/all_assets"' not in sections
    assert 'submodule "assets/mano_hand_s02"' not in sections


def test_dexstream_submodule_pin_and_required_manorl_paths() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", ASSET_PATH],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    mode, commit, stage_and_path = result.stdout.strip().split(maxsplit=2)
    assert mode == "160000"
    assert commit == ASSET_COMMIT
    assert stage_and_path == f"0\t{ASSET_PATH}"

    source_root = REPO_ROOT / ASSET_PATH
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == ASSET_COMMIT
    for relative in (
        "hand/mano/sunke/right/urdf/mano_right_hand_floating.urdf",
        "hand/mano/sunke/left/urdf/mano_left_hand_floating.urdf",
        "hand/mano/sunke/right/skin/mano_skin_mjcf_fragment.xml",
        "objects/DexGEM/cube1/cube1.urdf",
        "objects/DexGEM/banana/banana.urdf",
    ):
        assert (source_root / relative).is_file()


def test_old_physical_asset_paths_are_not_tracked() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    assert not any(path.startswith("assets/all_assets") for path in tracked)
    assert not any(path.startswith("assets/mano_hand_s02") for path in tracked)
    assert not (REPO_ROOT / "sim/manorl/runtime_assets").exists()
