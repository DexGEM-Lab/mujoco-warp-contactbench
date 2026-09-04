#!/usr/bin/env python3
"""Generate ManoRL's integrity manifest from a pinned DexStream checkout.

Git LFS pointer OIDs are the SHA-256 digests of the materialized files. Text
assets use the SHA-256 of the Git blob itself. The resulting manifest therefore
validates a normal LFS-smudged checkout without committing asset copies here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_REPOSITORY = "git@github.com:DexGEM-Lab/dexstream_digital-assets.git"
HAND_OPERATOR = "sunke"
SCHEMA = "manorl.dexstream-assets.v1"

# The runtime encodes every object's metric collision AABB.  The semantic
# category is derived from its canonical object name by observations.py; colors
# come from the source URDF rather than from a second asset copy.


def _profile(name: str, urdf: ET.Element) -> tuple[str, str]:
    color = urdf.find("./link/visual/material/color")
    rgba = color.get("rgba", "0.65 0.65 0.65 1") if color is not None else "0.65 0.65 0.65 1"
    values = tuple(float(value) for value in rgba.split())
    if len(values) != 4 or not all(value == value and abs(value) != float("inf") for value in values):
        raise ValueError(f"{name} visual material has invalid rgba {rgba!r}")
    # The observation layer chooses the semantic box/cylinder/sphere/irregular
    # block from the object name.  This field means that the supplied dimensions
    # are the collision AABB, not that the source mesh is a primitive.
    return "box", rgba


def _all_dexstream_objects(asset_root: Path) -> tuple[tuple[str, str, str], ...]:
    """Return (canonical name, source directory, URDF path) for DexGEM objects.

    The source repository contains two camera directories whose names are
    uppercase while trajectory identities are lowercase.  Canonicalizing here
    keeps the runtime API stable without renaming or copying source assets.
    """

    try:
        entries = _run_git(asset_root, "ls-tree", "-d", "--name-only", "HEAD:objects/DexGEM")
    except subprocess.CalledProcessError as exc:
        raise ValueError("DexStream commit has no objects/DexGEM tree") from exc
    objects: list[tuple[str, str, str]] = []
    seen: dict[str, str] = {}
    for source_name in entries.decode("utf-8").splitlines():
        if not source_name:
            continue
        prefix = f"objects/DexGEM/{source_name}/"
        files = _run_git(asset_root, "ls-tree", "-r", "--name-only", "HEAD", prefix)
        urdfs = []
        for relative in files.decode("utf-8").splitlines():
            path = PurePosixPath(relative)
            if path.suffix.lower() == ".urdf" and path.stem.casefold() == source_name.casefold():
                urdfs.append(relative)
        if len(urdfs) != 1:
            continue
        canonical = source_name.casefold()
        previous = seen.get(canonical)
        if previous is not None:
            raise ValueError(f"DexStream object names collide after normalization: {previous}, {source_name}")
        seen[canonical] = source_name
        objects.append((canonical, source_name, urdfs[0]))
    if not objects:
        raise ValueError("DexStream contains no same-name DexGEM object URDFs")
    return tuple(sorted(objects))

_LFS_POINTER = re.compile(
    rb"\Aversion https://git-lfs\.github\.com/spec/v1\n"
    rb"oid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?\Z"
)


def _run_git(asset_root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(asset_root), *args],
        check=True,
        capture_output=True,
    ).stdout


def _git_blob(asset_root: Path, relative: str) -> bytes:
    try:
        return _run_git(asset_root, "show", f"HEAD:{relative}")
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"DexStream commit omits required asset {relative}") from exc


def _record(asset_root: Path, relative: str) -> dict[str, Any]:
    blob = _git_blob(asset_root, relative)
    pointer = _LFS_POINTER.fullmatch(blob)
    if pointer is not None:
        return {
            "path": relative,
            "sha256": pointer.group(1).decode("ascii"),
            "size": int(pointer.group(2)),
            "storage": "lfs",
        }
    return {
        "path": relative,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size": len(blob),
        "storage": "git",
    }


def _resolve_urdf_reference(urdf_path: str, filename: str) -> str:
    candidate = posixpath.normpath(
        str(PurePosixPath(urdf_path).parent / PurePosixPath(filename))
    )
    if candidate.startswith("../") or candidate.startswith("/"):
        raise ValueError(
            f"URDF asset reference escapes DexStream root: {urdf_path} -> {filename}"
        )
    return candidate


def _mesh_paths(urdf_path: str, root: ET.Element, query: str) -> list[str]:
    paths: list[str] = []
    for mesh in root.findall(query):
        filename = mesh.get("filename")
        if not filename:
            raise ValueError(f"mesh in {urdf_path} has no filename")
        paths.append(_resolve_urdf_reference(urdf_path, filename))
    return paths


def _hand_manifest(asset_root: Path, side: str) -> dict[str, Any]:
    root_path = f"hand/mano/{HAND_OPERATOR}/{side}"
    urdf_name = f"mano_{side}_hand_floating.urdf"
    urdf_path = f"{root_path}/urdf/{urdf_name}"
    metadata_path = f"{root_path}/metadata.json"
    metadata = json.loads(_git_blob(asset_root, metadata_path).decode("utf-8"))
    if (
        metadata.get("subject") != HAND_OPERATOR
        or metadata.get("handedness") != side
        or metadata.get("num_dofs") != 28
        or metadata.get("dof_layout") != "cmc3_mcp2_28dof"
        or metadata.get("num_links") != 16
        or metadata.get("num_collision_links") != 16
        or metadata.get("num_visual_links") != 16
    ):
        raise ValueError(f"{side} MANO metadata does not satisfy the ManoRL contract")
    urdf = ET.fromstring(_git_blob(asset_root, urdf_path))
    joint_names = [
        joint.get("name", "")
        for joint in urdf.findall("joint")
        if joint.get("type") != "fixed"
    ]
    if tuple(joint_names) != tuple(metadata.get("joint_names", ())):
        raise ValueError(f"{side} MANO metadata/URDF joint order differs")
    collision_paths = _mesh_paths(
        urdf_path, urdf, ".//collision/geometry/mesh"
    )
    visual_paths = _mesh_paths(urdf_path, urdf, ".//visual/geometry/mesh")
    if len(joint_names) != 28 or len(collision_paths) != 16 or len(visual_paths) != 16:
        raise ValueError(
            f"{side} MANO contract drifted: joints={len(joint_names)}, "
            f"collision meshes={len(collision_paths)}, visual meshes={len(visual_paths)}"
        )
    # ManoRL renders the continuous MANO skin, but the source URDF still owns
    # 16 visual STL references and external consumers may load that URDF
    # directly. Keep the full URDF visual closure in the manifest rather than
    # leaving raw LFS pointers in a fresh skip-smudge checkout.
    # The skin bundle is also complete: its XML fragment references the binary
    # .skn file, and the bind .npz is part of the source hand contract.
    file_paths = [urdf_path, metadata_path, *collision_paths, *visual_paths]
    skin_metadata = metadata.get("skin")
    skin_files = skin_metadata.get("files") if isinstance(skin_metadata, dict) else None
    if not isinstance(skin_files, list) or not skin_files:
        raise ValueError(f"{side} MANO metadata has no skin file list")
    skin_paths: list[str] = []
    for relative in skin_files:
        if not isinstance(relative, str) or not relative or relative.startswith(("/", "../")):
            raise ValueError(f"{side} MANO metadata has an invalid skin path: {relative!r}")
        path = posixpath.normpath(f"{root_path}/{relative}")
        if not path.startswith(f"{root_path}/"):
            raise ValueError(f"{side} MANO skin path escapes its bundle: {relative!r}")
        _git_blob(asset_root, path)
        skin_paths.append(path)
    skin_path = next(
        (path for path in skin_paths if path.endswith("/mano_skin_mjcf_fragment.xml")),
        None,
    )
    if skin_path is None:
        raise ValueError(f"{side} MANO metadata skin bundle has no XML fragment")
    file_paths.extend(skin_paths)
    if len(set(file_paths)) != len(file_paths):
        raise ValueError(f"{side} MANO manifest contains duplicate files")
    return {
        "root": root_path,
        "urdf": urdf_path,
        "metadata": metadata_path,
        "skin": skin_path,
        "skin_files": skin_paths,
        "visual_files": visual_paths,
        "joint_names": joint_names,
        "betas": metadata.get("betas"),
        "palm_collision_scale": metadata.get("palm_collision_scale"),
        "files": [_record(asset_root, path) for path in file_paths],
    }


def _mesh_scale(mesh: ET.Element) -> list[float]:
    raw = mesh.get("scale", "1 1 1").split()
    values = [float(value) for value in raw]
    if len(values) != 3:
        raise ValueError(f"mesh scale must contain three values: {raw}")
    return values


def _object_manifest(
    asset_root: Path, canonical_name: str, source_name: str, urdf_path: str
) -> dict[str, Any]:
    object_root = f"objects/DexGEM/{source_name}"
    urdf = ET.fromstring(_git_blob(asset_root, urdf_path))
    links = urdf.findall("link")
    if len(links) != 1 or urdf.findall("joint"):
        raise ValueError(f"{canonical_name} URDF must contain exactly one rigid link and no joints")
    link_name = links[0].get("name")
    if not link_name:
        raise ValueError(f"{canonical_name} URDF link has no name")
    visual_meshes = urdf.findall("./link/visual/geometry/mesh")
    collision_meshes = urdf.findall("./link/collision/geometry/mesh")
    if len(visual_meshes) != 1 or not collision_meshes:
        raise ValueError(
            f"{canonical_name} URDF requires one visual and at least one collision mesh"
        )
    visual_filename = visual_meshes[0].get("filename")
    if not visual_filename:
        raise ValueError(f"{canonical_name} visual mesh has no filename")
    visual_path = _resolve_urdf_reference(urdf_path, visual_filename)
    collisions: list[dict[str, Any]] = []
    for mesh in collision_meshes:
        filename = mesh.get("filename")
        if not filename:
            raise ValueError(f"{canonical_name} collision mesh has no filename")
        record = _record(asset_root, _resolve_urdf_reference(urdf_path, filename))
        record["scale"] = _mesh_scale(mesh)
        collisions.append(record)
    geometry_type, rgba = _profile(canonical_name, urdf)
    visual = _record(asset_root, visual_path)
    visual["scale"] = _mesh_scale(visual_meshes[0])
    return {
        "root": object_root,
        "source_name": source_name,
        "link_name": link_name,
        "geometry_type": geometry_type,
        "rgba": rgba,
        "urdf": _record(asset_root, urdf_path),
        "visual": visual,
        "collisions": collisions,
    }

def generate(asset_root: Path, repository_root: Path) -> dict[str, Any]:
    commit = _run_git(asset_root, "rev-parse", "HEAD").decode("ascii").strip()
    mapping_path = repository_root / "sim/manorl/task_assets/object_grasps_simple.yaml"
    mapping = mapping_path.read_bytes()
    return {
        "schema": SCHEMA,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": commit,
        "hand_operator": HAND_OPERATOR,
        "hands": {
            side: _hand_manifest(asset_root, side) for side in ("right", "left")
        },
        "objects": {
            canonical: _object_manifest(asset_root, canonical, source_name, urdf_path)
            for canonical, source_name, urdf_path in _all_dexstream_objects(asset_root)
        },
        "task_metadata": {
            "grasp_mapping": {
                "path": mapping_path.relative_to(repository_root).as_posix(),
                "sha256": hashlib.sha256(mapping).hexdigest(),
                "size": len(mapping),
                "storage": "project",
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=Path("assets/dexstream_digital_assets"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sim/manorl/task_assets/dexstream_manifest.json"),
    )
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    asset_root = (repository_root / args.asset_root).resolve()
    output = (repository_root / args.output).resolve()
    manifest = generate(asset_root, repository_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output} for {manifest['source_commit']}")


if __name__ == "__main__":
    main()
