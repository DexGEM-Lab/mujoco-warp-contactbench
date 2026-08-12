"""Reproducible implementation provenance for DexHandRL artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str, binary: bool = False) -> bytes | str | None:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *args],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="replace").strip()


def _dirty_source_digest(status: str) -> str:
    """Hash tracked diffs and untracked files visible to Git.

    A HEAD alone cannot reproduce a run from a dirty worktree. The digest is
    deterministic for the same tracked diff plus untracked file bytes; ignored
    outputs and caches are excluded by Git and therefore do not perturb it.
    """

    digest = hashlib.sha256()
    digest.update(status.encode("utf-8"))
    tracked = _git("diff", "--binary", "--no-ext-diff", "HEAD", "--", binary=True)
    digest.update(tracked if isinstance(tracked, bytes) else b"")
    untracked = _git("ls-files", "--others", "--exclude-standard", "-z", binary=True)
    if isinstance(untracked, bytes):
        for raw_name in sorted(name for name in untracked.split(b"\0") if name):
            digest.update(b"\0path\0" + raw_name + b"\0")
            path = REPOSITORY_ROOT / raw_name.decode("utf-8", errors="surrogateescape")
            if path.is_symlink():
                digest.update(b"symlink\0" + str(path.readlink()).encode("utf-8"))
            elif path.is_file():
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
            else:
                digest.update(b"non-file")
    return digest.hexdigest()


def code_provenance() -> dict[str, Any]:
    """Return commit/branch/dirty identity for run and checkpoint records."""

    revision = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    dirty = status is None or bool(status)
    return {
        "schema": "dexhandrl.code_provenance.v1",
        "repository_root": str(REPOSITORY_ROOT.resolve()),
        "revision": revision,
        "branch": branch or None,
        "dirty": dirty,
        "source_diff_sha256": _dirty_source_digest(status or "") if dirty else None,
    }
