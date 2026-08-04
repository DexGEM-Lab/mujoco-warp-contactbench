"""Publish a verified local MTP to a filesystem without directory renames."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from sim.manorl.trajectory_package import file_sha256, load_trajectory_package


def _fsync(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(source: Path, destination_parent: Path, *, name_prefix: str) -> Path:
    """Copy an MTP and expose READY only after destination-side verification."""

    catalog = load_trajectory_package(source)
    destination_parent.mkdir(parents=True, exist_ok=True)
    destination = destination_parent / f"{name_prefix}-{catalog.package_digest}"
    if destination.exists():
        raise FileExistsError(f"refusing to replace published package: {destination}")
    destination.mkdir()
    ready = destination / "READY"
    try:
        manifest = catalog.manifest
        array_files = [metadata["file"] for metadata in manifest["arrays"].values()]
        filenames = ["manifest.json", *array_files]
        for filename in filenames:
            source_file = source / filename
            destination_file = destination / filename
            shutil.copyfile(source_file, destination_file)
            _fsync(destination_file)
        if file_sha256(destination / "manifest.json") != catalog.manifest_sha256:
            raise RuntimeError("published manifest SHA256 differs from verified source")
        for metadata in manifest["arrays"].values():
            destination_file = destination / metadata["file"]
            if file_sha256(destination_file) != metadata["sha256"]:
                raise RuntimeError(
                    f"published array SHA256 mismatch: {metadata['file']}"
                )
        _fsync_directory(destination)
        ready.write_text(catalog.package_digest + "\n", encoding="ascii")
        _fsync(ready)
        _fsync_directory(destination)
        published = load_trajectory_package(destination)
        if published.package_digest != catalog.package_digest:
            raise RuntimeError("published package identity changed after verification")
        _fsync_directory(destination_parent)
        return destination
    except BaseException:
        # An incomplete upload remains inspectable but never consumable.
        ready.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination-parent", type=Path, required=True)
    parser.add_argument(
        "--name-prefix",
        default="mtp-v1-v295-all75-right-f120-pre180-post250",
    )
    args = parser.parse_args(argv)
    if not args.name_prefix or "/" in args.name_prefix:
        parser.error("name-prefix must be one path component")
    destination = publish(
        args.source.expanduser().resolve(),
        args.destination_parent.expanduser().resolve(),
        name_prefix=args.name_prefix,
    )
    catalog = load_trajectory_package(destination)
    print(
        json.dumps(
            {
                "trajectory_package": str(destination),
                "package_digest": catalog.package_digest,
                "catalog_digest": catalog.catalog_digest,
                "trajectory_count": len(catalog.trajectories),
                "pair_count": len(catalog.resolved_pairs),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
