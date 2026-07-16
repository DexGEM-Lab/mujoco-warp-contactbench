from configparser import ConfigParser
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
ALL_ASSETS_PATH = "assets/all_assets"
ALL_ASSETS_URL = "git@192.168.10.116:jieqiangsun/all_assets.git"
ALL_ASSETS_COMMIT = "ead79126589d1abf2362ea30b9d674d9e675a2f9"


def test_all_assets_submodule_url_and_pin() -> None:
    modules = ConfigParser()
    modules.read(REPO_ROOT / ".gitmodules", encoding="utf-8")

    section = 'submodule "assets/all_assets"'
    assert modules[section]["path"] == ALL_ASSETS_PATH
    assert modules[section]["url"] == ALL_ASSETS_URL

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
