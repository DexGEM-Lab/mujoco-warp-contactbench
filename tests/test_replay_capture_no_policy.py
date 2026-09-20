"""Explicit replay profile must fail closed before loading physical assets."""
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tools import replay_capture_no_policy as replay


@pytest.fixture
def profile(tmp_path, monkeypatch):
    root = tmp_path / "assets"
    root.mkdir()
    path = tmp_path / "manifest.json"
    manifest = {"hand_operator": "cheyingtong", "source_commit": replay.PINNED_ASSET_COMMIT}
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(replay, "PINNED_MANIFEST_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(replay.subprocess, "run", Mock(return_value=SimpleNamespace(stdout=replay.PINNED_ASSET_COMMIT)))
    monkeypatch.setenv("MANORL_ASSET_MANIFEST", "prior-profile")
    return root, path


def test_explicit_binding_reuses_manifest(profile, monkeypatch):
    root, path = profile
    before = path.read_bytes()
    import sim.manorl
    fake = SimpleNamespace(_asset_manifest=Mock(), object_collision_vertices=Mock(), validate_asset_manifest=Mock())
    monkeypatch.setitem(sys.modules, "sim.manorl.assets", fake)
    monkeypatch.setattr(sim.manorl, "assets", fake, raising=False)
    replay.activate_hand_profile("cheyingtong", root, path)
    assert replay.os.environ["MANORL_ASSET_MANIFEST"] == str(path)
    assert fake.DEXSTREAM_ROOT == root
    assert fake.ASSET_MANIFEST == path
    assert fake.MANO_OPERATOR == "cheyingtong"
    fake.validate_asset_manifest.assert_called_once_with(hand_side="right")
    assert path.read_bytes() == before
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("failure", ["sha", "operator", "manifest_operator", "manifest_commit", "root_commit", "missing_manifest", "missing_root"])
def test_invalid_profile_does_not_bind(profile, monkeypatch, failure):
    root, path = profile
    operator = "cheyingtong"
    if failure == "sha":
        path.write_text("{}")
    elif failure == "operator":
        operator = "sunke"
    elif failure.startswith("manifest_"):
        manifest = json.loads(path.read_text())
        manifest["hand_operator" if failure == "manifest_operator" else "source_commit"] = "wrong"
        path.write_text(json.dumps(manifest))
        monkeypatch.setattr(replay, "PINNED_MANIFEST_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    elif failure == "root_commit":
        replay.subprocess.run.return_value.stdout = "wrong"
    elif failure == "missing_manifest":
        path.unlink()
    elif failure == "missing_root":
        root.rmdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        replay.activate_hand_profile(operator, root, path)
    assert replay.os.environ["MANORL_ASSET_MANIFEST"] == "prior-profile"


def test_cli_requires_explicit_asset_paths(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["replay", "--dataset", "source", "--version", "4", "--row", "82", "--output", "out"])
    with pytest.raises(SystemExit) as exc:
        replay.main()
    assert exc.value.code == 2
