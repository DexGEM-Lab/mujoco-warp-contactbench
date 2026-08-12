from __future__ import annotations

from pathlib import Path

from sim.dexhandrl.provenance import REPOSITORY_ROOT, code_provenance


def test_code_provenance_binds_repository_revision_and_dirty_source() -> None:
    record = code_provenance()

    assert record["schema"] == "dexhandrl.code_provenance.v1"
    assert Path(record["repository_root"]).resolve() == REPOSITORY_ROOT.resolve()
    assert isinstance(record["revision"], str) and len(record["revision"]) == 40
    assert record["branch"] is None or isinstance(record["branch"], str)
    assert isinstance(record["dirty"], bool)
    if record["dirty"]:
        assert isinstance(record["source_diff_sha256"], str)
        assert len(record["source_diff_sha256"]) == 64
    else:
        assert record["source_diff_sha256"] is None
