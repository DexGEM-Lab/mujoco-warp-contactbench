from configparser import ConfigParser
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
ALL_ASSETS_PATH = "assets/all_assets"
ALL_ASSETS_URL = "git@192.168.10.116:jieqiangsun/all_assets.git"
ALL_ASSETS_BRANCH = "main"
ALL_ASSETS_COMMIT = "033b358b73c57e5f437f6582b6a9b0d4add7f9ee"
MANO_ASSETS_PATH = "Assets/sim/mano_assets"
MANO_ASSETS_COMMIT = "e25f2ef0c0c81f6ceaf6befd107f0f2744d41a85"
MANO_HAND_PATH = "assets/mano_hand_s02"
MANO_HAND_BRANCH = "main"
MANO_HAND_COMMIT = "d98a9423e297f155d6dfaa7e4d563e54c7d67cbb"


def test_all_assets_submodule_url_and_pin() -> None:
    modules = ConfigParser()
    modules.read(REPO_ROOT / ".gitmodules", encoding="utf-8")

    section = 'submodule "assets/all_assets"'
    assert modules[section]["path"] == ALL_ASSETS_PATH
    assert modules[section]["url"] == ALL_ASSETS_URL
    assert modules[section]["branch"] == ALL_ASSETS_BRANCH

    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", ALL_ASSETS_PATH],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    mode, commit, stage_and_path = result.stdout.strip().split(maxsplit=2)
    assert mode == "160000"
    assert commit == ALL_ASSETS_COMMIT
    assert stage_and_path == f"0\t{ALL_ASSETS_PATH}"


def test_mano_hand_submodule_branch_and_pin() -> None:
    modules = ConfigParser()
    modules.read(REPO_ROOT / ".gitmodules", encoding="utf-8")

    section = 'submodule "assets/mano_hand_s02"'
    assert modules[section]["path"] == MANO_HAND_PATH
    assert modules[section]["branch"] == MANO_HAND_BRANCH

    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", MANO_HAND_PATH],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    mode, commit, stage_and_path = result.stdout.strip().split(maxsplit=2)
    assert mode == "160000"
    assert commit == MANO_HAND_COMMIT
    assert stage_and_path == f"0\t{MANO_HAND_PATH}"


def test_nested_mano_assets_pin_contains_decomposed_runtime_assets() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", MANO_ASSETS_PATH],
        cwd=REPO_ROOT / ALL_ASSETS_PATH,
        check=True,
        text=True,
        capture_output=True,
    )
    mode, commit, stage_and_path = result.stdout.strip().split(maxsplit=2)
    assert mode == "160000"
    assert commit == MANO_ASSETS_COMMIT
    assert stage_and_path == f"0\t{MANO_ASSETS_PATH}"
