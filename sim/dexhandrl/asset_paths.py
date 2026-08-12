"""Authoritative DexHandRL asset paths and revision provenance.

The object/hand source is the root repository's pinned ``assets/all_assets``
submodule.  DexHand task configuration is not part of all_assets, so the small
checked-in Isaac-compatible bundle remains its independent default source.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any
import xml.etree.ElementTree as ET


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPOSITORY_ROOT
DEFAULT_ALL_ASSETS_ROOT = REPOSITORY_ROOT / "assets" / "all_assets"
DEFAULT_TASK_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "assets"
    / "isaac_source_root"
    / "dexhand_env"
    / "cfg"
    / "task"
    / "Dexhand021proReconstruction.yaml"
)
PINNED_ALL_ASSETS_COMMIT = "7228b5cfce8d9a072ed4bded7a489cf73d521b68"
PINNED_MANO_ASSETS_COMMIT = "cde03ef94816b589f574ca6f358695005d3d1a3f"


def resolve_all_assets_root(source_root: Path = DEFAULT_SOURCE_ROOT) -> Path:
    """Resolve either an outer repository/source root or all_assets itself."""

    root = Path(source_root).expanduser()
    candidates = (root, root / "assets" / "all_assets")
    for candidate in candidates:
        if (candidate / "Assets").is_dir():
            return candidate
    expected = " or ".join(str(candidate / "Assets") for candidate in candidates)
    raise FileNotFoundError(f"all_assets checkout is missing; expected {expected}")


def object_urdf_path(object_name: str, source_root: Path = DEFAULT_SOURCE_ROOT) -> Path:
    """Return ``Assets/sim/decomposed/<object>/<object>.urdf`` fail-closed."""

    if not object_name or Path(object_name).name != object_name or object_name in {".", ".."}:
        raise ValueError(f"invalid object name: {object_name!r}")
    path = (
        resolve_all_assets_root(source_root)
        / "Assets"
        / "sim"
        / "decomposed"
        / object_name
        / f"{object_name}.urdf"
    )
    if not path.is_file():
        raise FileNotFoundError(f"decomposed object URDF not found: {path}")
    return path


def hand_xml_path(source_root: Path = DEFAULT_SOURCE_ROOT) -> Path:
    path = (
        resolve_all_assets_root(source_root)
        / "Assets"
        / "HAND"
        / "dexhand021pro"
        / "marvin_021pro_mujoco"
        / "marvin_021pro_right_hand_floating_simplified.xml"
    )
    if not path.is_file():
        raise FileNotFoundError(f"DexHand021Pro hand XML not found: {path}")
    return path


def contact_bodies_path(source_root: Path = DEFAULT_SOURCE_ROOT) -> Path:
    path = resolve_all_assets_root(source_root) / "Assets" / "object_grasps_simple.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"object contact mapping not found: {path}")
    return path


def task_config_path(source_root: Path = DEFAULT_SOURCE_ROOT) -> Path:
    """Resolve task YAML independently from all_assets.

    An explicit outer Isaac source root may own ``dexhand_env``.  The repository
    default and direct all_assets roots use the checked-in task configuration.
    """

    root = Path(source_root).expanduser()
    candidate = root / "dexhand_env" / "cfg" / "task" / "Dexhand021proReconstruction.yaml"
    if candidate.is_file():
        return candidate
    try:
        all_assets = resolve_all_assets_root(root)
    except FileNotFoundError:
        return candidate
    if all_assets.resolve() == DEFAULT_ALL_ASSETS_ROOT.resolve() and DEFAULT_TASK_CONFIG_PATH.is_file():
        return DEFAULT_TASK_CONFIG_PATH
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mesh_record(urdf: Path, owner: ET.Element, *, role: str, index: int) -> dict[str, Any]:
    mesh = owner.find("geometry/mesh")
    filename = None if mesh is None else mesh.get("filename")
    if not filename:
        raise ValueError(f"{role} {index} in {urdf} has no mesh filename")
    path = (urdf.parent / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{role} mesh not found: {path}")
    origin = owner.find("origin")
    return {
        "index": index,
        "urdf_name": owner.get("name"),
        "path": str(path),
        "sha256": _sha256(path),
        "scale": mesh.get("scale", "1 1 1"),
        "origin_xyz": "0 0 0" if origin is None else origin.get("xyz", "0 0 0"),
        "origin_rpy": "0 0 0" if origin is None else origin.get("rpy", "0 0 0"),
    }


def object_mesh_provenance(
    object_name: str,
    source_root: Path = DEFAULT_SOURCE_ROOT,
) -> dict[str, Any]:
    """Record the ordered visual and complete collision-mesh inputs."""

    urdf = object_urdf_path(object_name, source_root)
    root = ET.parse(urdf).getroot()
    links = root.findall("link")
    if len(links) != 1:
        raise ValueError(f"object URDF must contain exactly one link: {urdf}")
    visuals = links[0].findall("visual")
    collisions = links[0].findall("collision")
    if len(visuals) != 1:
        raise ValueError(f"object URDF must contain exactly one visual: {urdf}")
    if not collisions:
        raise ValueError(f"object URDF has no collision meshes: {urdf}")
    return {
        "visual_mesh": _mesh_record(urdf, visuals[0], role="visual", index=0),
        "collision_mesh_count": len(collisions),
        "collision_meshes": [
            _mesh_record(urdf, collision, role="collision", index=index)
            for index, collision in enumerate(collisions)
        ],
    }


def _git_head(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def asset_provenance(
    object_name: str,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    *,
    require_pinned_revisions: bool = True,
) -> dict[str, Any]:
    """Record the exact asset revisions and materialized object inputs."""

    all_assets = resolve_all_assets_root(source_root)
    mano_assets = all_assets / "Assets" / "sim" / "mano_assets"
    urdf = object_urdf_path(object_name, source_root)
    all_assets_head = _git_head(all_assets)
    mano_assets_head = _git_head(mano_assets)
    if require_pinned_revisions:
        if all_assets_head != PINNED_ALL_ASSETS_COMMIT:
            raise RuntimeError(
                f"all_assets revision {all_assets_head!r} != pinned {PINNED_ALL_ASSETS_COMMIT}"
            )
        if mano_assets_head != PINNED_MANO_ASSETS_COMMIT:
            raise RuntimeError(
                f"mano_assets revision {mano_assets_head!r} != pinned {PINNED_MANO_ASSETS_COMMIT}"
            )
    meshes = object_mesh_provenance(object_name, source_root)
    return {
        "schema": "dexhandrl.asset_provenance.v2",
        "all_assets_root": str(all_assets.resolve()),
        "all_assets_commit": all_assets_head,
        "mano_assets_commit": mano_assets_head,
        "object_urdf": str(urdf.resolve()),
        "object_urdf_sha256": _sha256(urdf),
        **meshes,
        "hand_xml": str(hand_xml_path(source_root).resolve()),
        "hand_xml_sha256": _sha256(hand_xml_path(source_root)),
        "contact_bodies_file": str(contact_bodies_path(source_root).resolve()),
        "contact_bodies_sha256": _sha256(contact_bodies_path(source_root)),
        "task_config_path": str(task_config_path(source_root).resolve()),
        "task_config_sha256": _sha256(task_config_path(source_root)),
    }
