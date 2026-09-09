"""Deterministic MuJoCo scene construction from pinned DexStream assets."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from sim.manorl.contracts import (
    ACTION_SIDE_ORDER,
    JOINT_DOF,
    EFFORT,
    FLOOR_TOP_Z,
    JOINT_ARMATURE,
    JOINT_FRICTIONLOSS,
    normalize_hand_side,
    OBJECT_TYPE,
    PHYSICS_TIMESTEP,
    ServoConfig,
)

TASK_ASSET_ROOT = Path(__file__).resolve().parent / "task_assets"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEXSTREAM_ROOT = REPOSITORY_ROOT / "assets" / "dexstream_digital_assets"
DEXSTREAM_REPOSITORY = "git@github.com:DexGEM-Lab/dexstream_digital-assets.git"
MANO_OPERATOR = "sunke"
ASSET_MANIFEST = TASK_ASSET_ROOT / "dexstream_manifest.json"
GRASP_MAPPING = TASK_ASSET_ROOT / "object_grasps_simple.yaml"
VISUAL_GEOM_GROUP = 2
COLLISION_GEOM_GROUP = 3
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


@lru_cache(maxsize=1)
def _asset_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ManoRL DexStream manifest: {ASSET_MANIFEST}") from exc
    if manifest.get("schema") != "manorl.dexstream-assets.v1":
        raise ValueError("unsupported ManoRL DexStream asset manifest schema")
    return manifest


def asset_provenance() -> dict[str, str]:
    """Return the physical asset identity recorded in the ManoRL manifest."""

    manifest = _asset_manifest()
    repository = manifest.get("source_repository")
    commit = manifest.get("source_commit")
    if repository != DEXSTREAM_REPOSITORY:
        raise ValueError(
            f"ManoRL asset manifest names unexpected source repository: {repository!r}"
        )
    if not isinstance(repository, str) or not repository:
        raise ValueError("ManoRL DexStream manifest has no source repository")
    if not isinstance(commit, str) or not commit:
        raise ValueError("ManoRL DexStream manifest has no source commit")
    return {
        "asset_source_repository": repository,
        "asset_source_commit": commit,
        "asset_manifest_sha256": hashlib.sha256(ASSET_MANIFEST.read_bytes()).hexdigest(),
    }


def _source_path(relative: str) -> Path:
    return DEXSTREAM_ROOT / relative


def _manifest_record_path(record: dict[str, Any]) -> Path:
    relative = record.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid DexStream manifest path: {relative!r}")
    path = _source_path(relative).resolve()
    try:
        path.relative_to(DEXSTREAM_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"DexStream manifest path escapes source root: {relative!r}") from exc
    return path


def hand_asset_root(hand_side: str = "right") -> Path:
    """Return the pinned DexStream root for one normalized MANO side."""

    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
    hand = _asset_manifest().get("hands", {}).get(side)
    if not isinstance(hand, dict) or not isinstance(hand.get("root"), str):
        raise ValueError(f"DexStream manifest has no {side}-hand contract")
    return _source_path(hand["root"])


def hand_urdf_path(hand_side: str = "right") -> Path:
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
    hand = _asset_manifest()["hands"][side]
    path = _source_path(hand["urdf"])
    if not path.is_file():
        raise FileNotFoundError(f"DexStream {side}-hand URDF is absent: {path}")
    return path


def hand_joint_names(hand_side: str = "right") -> tuple[str, ...]:
    """Read and cross-check the pinned MANO joint order and metadata."""

    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
    hand = _asset_manifest().get("hands", {}).get(side)
    if not isinstance(hand, dict):
        raise ValueError(f"DexStream manifest has no {side}-hand contract")
    names = tuple(hand.get("joint_names", ()))
    if len(names) != JOINT_DOF or len(set(names)) != JOINT_DOF:
        raise ValueError(
            f"{side} hand manifest must declare {JOINT_DOF} unique joints"
        )
    metadata_path = hand.get("metadata")
    if not isinstance(metadata_path, str):
        raise ValueError(f"DexStream manifest has no {side}-hand metadata path")
    try:
        metadata = json.loads(_source_path(metadata_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid DexStream {side}-hand metadata") from exc
    if (
        metadata.get("subject") != MANO_OPERATOR
        or metadata.get("handedness") != side
        or metadata.get("num_dofs") != JOINT_DOF
        or tuple(metadata.get("joint_names", ())) != names
    ):
        raise ValueError(f"DexStream {side}-hand metadata does not match the manifest")
    return names


def _hand_skin_path(hand_side: str) -> Path:
    """Return the pinned DexStream MANO skin fragment for one hand side."""

    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
    skin = _asset_manifest().get("hands", {}).get(side, {}).get("skin")
    if not isinstance(skin, str) or not skin:
        raise ValueError(
            f"DexStream {side}-hand manifest has no skin contract; "
            "regenerate the asset manifest"
        )
    return _source_path(skin)


def _hand_skin_element(hand_side: str, *, name_prefix: str = "") -> ET.Element:
    """Load the MANO skin fragment and rebind it to one compiled hand tree.

    DexStream exports the full 778-vertex MANO surface with per-vertex LBS
    weights as a MuJoCo ``<deformable><skin>`` fragment. The 16 rigid visual
    STLs are an argmax partition of the same surface and tear at joints by
    construction, so viewer scenes use the skin instead. Bones reference body
    names, which receive the same side prefix as the compiled tree.
    """

    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
    path = _hand_skin_path(side)
    if not path.is_file():
        raise FileNotFoundError(f"DexStream {side}-hand skin fragment is absent: {path}")
    try:
        fragment = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid DexStream {side}-hand skin fragment: {path}") from exc
    if fragment.tag != "deformable":
        raise ValueError(
            f"DexStream {side}-hand skin fragment root must be <deformable>"
        )
    skin = fragment.find("skin")
    if skin is None or not skin.get("name"):
        raise ValueError(f"DexStream {side}-hand skin fragment has no <skin>")
    bones = skin.findall("bone")
    if len(bones) != 16:
        raise ValueError(
            f"DexStream {side}-hand skin fragment must declare 16 bones, got {len(bones)}"
        )
    skin.set("name", f"{name_prefix}mano_skin")
    for bone in bones:
        body = bone.get("body")
        if not body:
            raise ValueError(f"DexStream {side}-hand skin bone has no body")
        bone.set("body", f"{name_prefix}{body}")
    return fragment


@dataclass(frozen=True)
class ObjectRuntime:
    """Materialized DexStream object inputs for one homogeneous MJX model."""

    object_type: str
    link_name: str
    body_name: str
    free_joint_name: str
    source_mesh_filename: str
    urdf_path: Path
    collision_mesh_paths: tuple[Path, ...]
    grasp_mapping_path: Path
    source_mesh_scale: tuple[float, float, float]
    collision_mesh_scales: tuple[tuple[float, float, float], ...]
    rgba: str
    geometry_type: str = "box"
    expected_sha256: tuple[tuple[Path, str], ...] = ()

    @property
    def collision_geom_count(self) -> int:
        return len(self.collision_mesh_paths)

    @property
    def collision_mesh_path(self) -> Path:
        """Compatibility accessor for legacy single-piece callers."""

        return self.collision_mesh_paths[0]

    @property
    def collision_mesh_scale(self) -> tuple[float, float, float]:
        """Compatibility accessor for the first collision piece."""

        return self.collision_mesh_scales[0]

    def __post_init__(self) -> None:
        if not self.collision_mesh_paths:
            raise ValueError(f"{self.object_type} runtime requires collision meshes")
        if len(self.collision_mesh_scales) != len(self.collision_mesh_paths):
            raise ValueError(
                f"{self.object_type} collision mesh paths/scales must have equal lengths"
            )


def _canonical_object_type(object_type: str) -> str:
    if not isinstance(object_type, str) or not object_type.strip():
        raise ValueError(f"unsupported ManoRL object runtime {object_type!r}")
    return object_type.strip().casefold()


def _object_record(object_type: str) -> dict[str, Any]:
    objects = _asset_manifest().get("objects", {})
    canonical = _canonical_object_type(object_type)
    if canonical not in objects:
        available = ", ".join(sorted(objects))
        raise ValueError(
            f"unsupported ManoRL object runtime {object_type!r}; "
            f"materialized runtimes: {available}"
        )
    record = objects[canonical]
    if not isinstance(record, dict):
        raise ValueError(f"invalid DexStream object manifest for {canonical!r}")
    return record


def supported_object_types() -> tuple[str, ...]:
    """Return the closed set of object types pinned in the source manifest."""

    objects = _asset_manifest().get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError("DexStream manifest objects must be a mapping")
    return tuple(sorted(objects))


def object_runtime(object_type: str = OBJECT_TYPE) -> ObjectRuntime:
    """Resolve one DexStream object runtime or fail before compilation."""

    canonical = _canonical_object_type(object_type)
    record = _object_record(canonical)
    urdf = record.get("urdf")
    visual = record.get("visual")
    collisions = record.get("collisions")
    if not isinstance(urdf, dict) or not isinstance(visual, dict) or not isinstance(
        collisions, list
    ):
        raise ValueError(f"incomplete DexStream object manifest for {object_type!r}")
    collision_records = tuple(collisions)
    collision_paths = tuple(_manifest_record_path(item) for item in collision_records)
    collision_scales = tuple(
        tuple(float(value) for value in item.get("scale", (1.0, 1.0, 1.0)))
        for item in collision_records
    )
    source_scale = tuple(float(value) for value in visual.get("scale", (1.0,) * 3))
    if len(source_scale) != 3 or any(len(scale) != 3 for scale in collision_scales):
        raise ValueError(f"invalid DexStream mesh scale for {object_type!r}")
    expected = tuple(
        (_manifest_record_path(item), str(item["sha256"]))
        for item in (urdf, visual, *collision_records)
    ) + ((GRASP_MAPPING, str(_asset_manifest()["task_metadata"]["grasp_mapping"]["sha256"])),)
    visual_path = _manifest_record_path(visual)
    return ObjectRuntime(
        object_type=canonical,
        link_name=str(record.get("link_name", f"{canonical}_link")),
        body_name=canonical,
        free_joint_name=f"{canonical}_free",
        source_mesh_filename=visual_path.name,
        urdf_path=_manifest_record_path(urdf),
        collision_mesh_paths=collision_paths,
        grasp_mapping_path=GRASP_MAPPING,
        source_mesh_scale=source_scale,  # type: ignore[arg-type]
        collision_mesh_scales=collision_scales,  # type: ignore[arg-type]
        rgba=str(record.get("rgba", "0.65 0.65 0.65 1")),
        geometry_type=str(record.get("geometry_type", "box")),
        expected_sha256=expected,
    )


# Public compatibility constants resolve to the pinned DexStream cube1 bundle.
OBJECT_URDF = _manifest_record_path(_object_record("cube1")["urdf"])
OBJECT_MESH = _manifest_record_path(_object_record("cube1")["visual"])


# Source shape groups name four rigid bodies per finger, but each abduction
# body has no collision shape in the authoritative URDF. These are the three
# actual collision-bearing bodies selected for each source filter group.
HAND_SELF_COLLISION_GROUPS = {
    "thumb": ("thumb_cmc", "thumb_mcp", "thumb_ip"),
    "index": ("index_mcp", "index_pip", "index_dip"),
    "middle": ("middle_mcp", "middle_pip", "middle_dip"),
    "ring": ("ring_mcp", "ring_pip", "ring_dip"),
    "pinky": ("pinky_mcp", "pinky_pip", "pinky_dip"),
}


def _numbers(
    value: str | None, count: int, default: Iterable[float]
) -> tuple[float, ...]:
    values = (
        tuple(default)
        if value is None
        else tuple(float(item) for item in value.split())
    )
    if len(values) != count or not np.all(np.isfinite(values)):
        raise ValueError(f"expected {count} finite values, got {value!r}")
    return values


def _format(values: Iterable[float]) -> str:
    return " ".join(f"{value:.17g}" for value in values)


def _rpy_to_wxyz(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _origin(element: ET.Element | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if element is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    xyz = _numbers(element.get("xyz"), 3, (0.0, 0.0, 0.0))
    rpy = _numbers(element.get("rpy"), 3, (0.0, 0.0, 0.0))
    return xyz, _rpy_to_wxyz(rpy)


def _resolve_mesh_path(
    urdf_path: Path, filename: str, *, fallback_roots: Iterable[Path] = ()
) -> Path:
    """Resolve a mesh relative to its pinned DexStream URDF."""

    candidates = [(urdf_path.parent / filename).resolve()]
    basename = Path(filename).name
    candidates.extend((root / basename).resolve() for root in fallback_roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"URDF mesh is absent: {filename!r}; checked "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _visual_rgba(
    visual: ET.Element,
    global_materials: dict[str, str],
    fallback: str,
) -> str:
    material = visual.find("material")
    if material is None:
        return fallback
    color = material.find("color")
    if color is not None and color.get("rgba"):
        return color.get("rgba", fallback)
    material_name = material.get("name")
    return global_materials.get(material_name or "", fallback)


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _verify_record(record: dict[str, Any], *, project_owned: bool = False) -> Path:
    path = (
        REPOSITORY_ROOT / str(record.get("path"))
        if project_owned
        else _manifest_record_path(record)
    )
    if not path.is_file():
        raise FileNotFoundError(f"manifest asset is absent: {path}")
    if record.get("storage") == "lfs" and _is_lfs_pointer(path):
        raise FileNotFoundError(
            f"DexStream LFS asset is not materialized: {path}; run "
            "`git submodule update --init assets/dexstream_digital_assets` followed by "
            "`git -C assets/dexstream_digital_assets lfs pull`"
        )
    size = path.stat().st_size
    if size != record.get("size"):
        raise ValueError(f"asset size mismatch for {path}: {size}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record.get("sha256"):
        raise ValueError(f"asset digest mismatch for {path}: {digest}")
    return path


def _required_paths(
    object_type: str | None = None, *, hand_side: str = "right"
) -> tuple[Path, ...]:
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
    manifest = _asset_manifest()
    hand = manifest["hands"].get(side)
    if not isinstance(hand, dict) or not isinstance(hand.get("files"), list):
        raise ValueError(f"DexStream manifest has no {side}-hand files")
    hand_paths = tuple(_verify_record(record) for record in hand["files"])
    object_names = (object_type,) if object_type is not None else supported_object_types()
    object_paths: list[Path] = []
    for name in object_names:
        record = _object_record(name)
        for item in (record["urdf"], record["visual"], *record["collisions"]):
            object_paths.append(_verify_record(item))
    mapping = manifest["task_metadata"].get("grasp_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("ManoRL manifest has no grasp mapping contract")
    mapping_path = _verify_record(mapping, project_owned=True)
    return (*hand_paths, *object_paths, mapping_path, ASSET_MANIFEST)


def validate_asset_manifest(
    object_type: str | None = None, *, hand_side: str = "right"
) -> dict[str, Any]:
    """Verify the pinned DexStream checkout and ManoRL task metadata."""

    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
    manifest = _asset_manifest()
    try:
        import subprocess

        source_commit = subprocess.run(
            ["git", "-C", str(DEXSTREAM_ROOT), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FileNotFoundError(
            f"DexStream submodule is not initialized: {DEXSTREAM_ROOT}"
        ) from exc
    if manifest.get("source_repository") != DEXSTREAM_REPOSITORY:
        raise ValueError(
            f"DexStream manifest source repository mismatch: "
            f"{manifest.get('source_repository')!r}"
        )
    if manifest.get("hand_operator") != MANO_OPERATOR:
        raise ValueError(
            f"DexStream manifest hand operator mismatch: "
            f"{manifest.get('hand_operator')!r}"
        )
    if source_commit != manifest.get("source_commit"):
        raise ValueError(
            f"DexStream submodule pin mismatch: {source_commit} != "
            f"{manifest.get('source_commit')}"
        )
    names = tuple(manifest["hands"][side].get("joint_names", ()))
    if names != hand_joint_names(side):
        raise ValueError(f"DexStream {side}-hand joint order drifted")
    _required_paths(object_type, hand_side=side)
    return manifest


def _link_inertial(body: ET.Element, link: ET.Element) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError(f"authoritative link {link.get('name')} has no inertial")
    position, quaternion = _origin(inertial.find("origin"))
    if not np.allclose(quaternion, (1.0, 0.0, 0.0, 0.0), atol=1e-15, rtol=0):
        raise ValueError(
            f"rotated full inertia is unsupported for link {link.get('name')}"
        )
    mass_element = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass_element is None or inertia is None:
        raise ValueError(
            f"authoritative link {link.get('name')} has incomplete inertial"
        )
    full_inertia = (
        float(inertia.get("ixx", "nan")),
        float(inertia.get("iyy", "nan")),
        float(inertia.get("izz", "nan")),
        float(inertia.get("ixy", "nan")),
        float(inertia.get("ixz", "nan")),
        float(inertia.get("iyz", "nan")),
    )
    ET.SubElement(
        body,
        "inertial",
        pos=_format(position),
        mass=mass_element.get("value", ""),
        fullinertia=_format(full_inertia),
    )


def _link_collision(
    body: ET.Element,
    link: ET.Element,
    *,
    hand_contacts_enabled: bool,
    viewer_visuals: bool,
    hand_root: Path,
    name_prefix: str = "",
    asset_prefix: str = "",
) -> None:
    collisions = link.findall("collision")
    if len(collisions) > 1:
        raise ValueError(
            f"hand link {link.get('name')} unexpectedly has multiple collisions"
        )
    if not collisions:
        return
    collision = collisions[0]
    geometry = collision.find("geometry")
    mesh = None if geometry is None else geometry.find("mesh")
    if mesh is None:
        raise ValueError(f"hand collision for {link.get('name')} is not a mesh")
    filename = Path(mesh.get("filename", "")).name
    source_path = hand_root / "meshes" / filename
    if not source_path.is_file():
        raise FileNotFoundError(f"DexStream hand collision mesh is absent: {source_path}")
    position, quaternion = _origin(collision.find("origin"))
    attributes = {
        "name": f"{name_prefix}{link.get('name')}_collision",
        "type": "mesh",
        "mesh": f"{asset_prefix}hand_{Path(filename).stem}",
        "pos": _format(position),
        "quat": _format(quaternion),
        "rgba": "0.88 0.58 0.46 1",
        # Enabled masks allow palm/cross-finger, hand-object, and hand-floor
        # contacts. Explicit body-pair excludes below disable only collisions
        # within each finger, matching the source disable_within_finger mode.
        # The free-space diagnostic zeros both bits only on hand geoms.
        "contype": "1" if hand_contacts_enabled else "0",
        "conaffinity": "7" if hand_contacts_enabled else "0",
        "condim": "3",
        "friction": "1 0.01 0.001",
    }
    if viewer_visuals:
        attributes["group"] = str(COLLISION_GEOM_GROUP)
    ET.SubElement(body, "geom", attributes)


def _hand_tree(
    worldbody: ET.Element,
    asset: ET.Element,
    urdf_root: ET.Element,
    *,
    hand_contacts_enabled: bool,
    visual_meshes: bool = False,
    hand_side: str = "right",
    name_prefix: str = "",
    asset_prefix: str = "",
) -> None:
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
    hand_root = hand_asset_root(side)
    links = {element.get("name", ""): element for element in urdf_root.findall("link")}
    joints = list(urdf_root.findall("joint"))
    children: dict[str, list[ET.Element]] = {}
    child_links: set[str] = set()
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {joint.get('name')} has incomplete topology")
        parent_name, child_name = parent.get("link", ""), child.get("link", "")
        children.setdefault(parent_name, []).append(joint)
        child_links.add(child_name)
    roots = set(links) - child_links
    if roots != {"base_link"}:
        raise ValueError(f"expected base_link as sole hand root, got {roots}")

    mesh_files: list[str] = []
    for link in links.values():
        collision = link.find("collision/geometry/mesh")
        if collision is not None:
            mesh_files.append(Path(collision.get("filename", "")).name)
    if len(mesh_files) != 16 or len(set(mesh_files)) != 16:
        raise ValueError(
            "authoritative hand URDF must reference 16 unique collision meshes"
        )
    for filename in mesh_files:
        attributes = {
            "name": f"{asset_prefix}hand_{Path(filename).stem}",
            "file": str((hand_root / "meshes" / filename).resolve()),
        }
        source_mesh = next(
            link.find("collision/geometry/mesh")
            for link in links.values()
            if link.find("collision/geometry/mesh") is not None
            and Path(link.find("collision/geometry/mesh").get("filename", "")).name
            == filename
        )
        scale = source_mesh.get("scale")
        if scale is not None:
            attributes["scale"] = scale
        ET.SubElement(asset, "mesh", attributes)

    if visual_meshes and not _asset_manifest().get("hands", {}).get(side, {}).get("skin"):
        raise ValueError(
            f"DexStream {side}-hand manifest has no skin contract; "
            "regenerate the asset manifest"
        )

    def append_link(
        parent_xml: ET.Element, link_name: str, source_joint: ET.Element | None
    ) -> None:
        attributes = {"name": f"{name_prefix}{link_name}", "gravcomp": "1"}
        if source_joint is not None:
            position, quaternion = _origin(source_joint.find("origin"))
            attributes.update(pos=_format(position), quat=_format(quaternion))
        body = ET.SubElement(parent_xml, "body", attributes)
        if source_joint is not None and source_joint.get("type") != "fixed":
            joint_type = source_joint.get("type")
            if joint_type not in {"prismatic", "revolute"}:
                raise ValueError(f"unsupported hand joint type: {joint_type}")
            axis = source_joint.find("axis")
            limit = source_joint.find("limit")
            if axis is None or limit is None:
                raise ValueError(
                    f"joint {source_joint.get('name')} has no axis or limit"
                )
            ET.SubElement(
                body,
                "joint",
                name=f"{name_prefix}{source_joint.get('name', '')}",
                type="slide" if joint_type == "prismatic" else "hinge",
                axis=axis.get("xyz", ""),
                range=f"{limit.get('lower')} {limit.get('upper')}",
                limited="true",
                damping="0",
                frictionloss=str(JOINT_FRICTIONLOSS),
                armature=str(JOINT_ARMATURE),
            )
        _link_inertial(body, links[link_name])
        _link_collision(
            body,
            links[link_name],
            hand_contacts_enabled=hand_contacts_enabled,
            viewer_visuals=visual_meshes,
            hand_root=hand_root,
            name_prefix=name_prefix,
            asset_prefix=asset_prefix,
        )
        if visual_meshes:
            # The MANO skin replaces the rigid per-link visual partition;
            # the scene builder appends the skinned surface to the model.
            pass
        for joint in children.get(link_name, []):
            child = joint.find("child")
            assert child is not None
            append_link(body, child.get("link", ""), joint)

    append_link(worldbody, "base_link", None)


def _add_hand_self_collision_excludes(
    contact: ET.Element, *, name_prefix: str = ""
) -> None:
    for group_name, body_names in HAND_SELF_COLLISION_GROUPS.items():
        for pair_index, (first, second) in enumerate(combinations(body_names, 2)):
            ET.SubElement(
                contact,
                "exclude",
                name=f"{name_prefix}{group_name}_internal_{pair_index}",
                body1=f"{name_prefix}{first}",
                body2=f"{name_prefix}{second}",
            )


def _object_body(
    worldbody: ET.Element,
    asset: ET.Element,
    urdf_root: ET.Element,
    runtime: ObjectRuntime,
    *,
    gravity_compensated: bool = False,
    visual_meshes: bool = False,
    object_collisions: bool = False,
) -> None:
    object_link = urdf_root.find("link")
    if object_link is None or object_link.get("name") != runtime.link_name:
        raise ValueError(f"{runtime.object_type} URDF must contain {runtime.link_name}")
    collisions = object_link.findall("collision")
    if len(collisions) != runtime.collision_geom_count:
        raise ValueError(
            f"{runtime.object_type} URDF must contain exactly "
            f"{runtime.collision_geom_count} collision mesh(es)"
        )
    collision_assets: list[tuple[ET.Element, str]] = []
    for collision_index, (collision, collision_path, collision_scale) in enumerate(
        zip(
            collisions,
            runtime.collision_mesh_paths,
            runtime.collision_mesh_scales,
            strict=True,
        )
    ):
        source_mesh = collision.find("geometry/mesh")
        expected_filename = collision_path.name
        if (
            source_mesh is None
            or Path(source_mesh.get("filename", "")).name != expected_filename
        ):
            raise ValueError(
                f"{runtime.object_type} URDF collision {collision_index} must reference "
                f"{expected_filename}"
            )
        expected_source_scale = collision_scale
        mesh_scale = _numbers(source_mesh.get("scale"), 3, (1.0, 1.0, 1.0))
        if mesh_scale != expected_source_scale:
            raise ValueError(
                f"{runtime.object_type} URDF collision {collision_index} scale must remain "
                f"{expected_source_scale}"
            )
        asset_name = (
            f"{runtime.object_type}_mesh"
            if runtime.collision_geom_count == 1
            else f"{runtime.object_type}_collision_mesh_{collision_index}"
        )
        ET.SubElement(
            asset,
            "mesh",
            name=asset_name,
            file=str(collision_path.resolve()),
            scale=_format(collision_scale),
        )
        collision_assets.append((collision, asset_name))

    body = ET.SubElement(
        worldbody,
        "body",
        name=runtime.body_name,
        # Unified scenes park inactive objects outside the bounded workspace;
        # they retain native gravity so the active object uses the exact same
        # generalized dynamics as the homogeneous model.
        gravcomp="1" if gravity_compensated else "0",
    )
    ET.SubElement(body, "freejoint", name=runtime.free_joint_name)
    _link_inertial(body, object_link)
    for collision_index, (collision, asset_name) in enumerate(collision_assets):
        position, quaternion = _origin(collision.find("origin"))
        geom_name = (
            f"{runtime.object_type}_collision"
            if collision_index == 0
            else f"{runtime.object_type}_piece_{collision_index}_collision"
        )
        collision_geom = ET.SubElement(
            body,
            "geom",
            name=geom_name,
            type="mesh",
            mesh=asset_name,
            pos=_format(position),
            quat=_format(quaternion),
            rgba=runtime.rgba,
            contype="2",
            conaffinity="7" if object_collisions else "5",
            condim="3",
            friction="0.9 0.01 0.001",
        )
        if visual_meshes:
            collision_geom.set("group", str(COLLISION_GEOM_GROUP))
    if visual_meshes:
        visuals = object_link.findall("visual")
        if len(visuals) != 1:
            raise ValueError(
                f"{runtime.object_type} URDF must contain exactly one visual mesh"
            )
        visual = visuals[0]
        visual_mesh = visual.find("geometry/mesh")
        if visual_mesh is None:
            raise ValueError(f"{runtime.object_type} URDF visual is not a mesh")
        visual_filename = Path(visual_mesh.get("filename", "")).name
        if visual_filename != runtime.source_mesh_filename:
            raise ValueError(
                f"{runtime.object_type} URDF visual must reference {runtime.source_mesh_filename}"
            )
        visual_path = _resolve_mesh_path(
            runtime.urdf_path,
            visual_mesh.get("filename", ""),
        )
        visual_scale = _numbers(visual_mesh.get("scale"), 3, runtime.source_mesh_scale)
        ET.SubElement(
            asset,
            "mesh",
            name=f"{runtime.object_type}_visual_mesh",
            file=str(visual_path),
            scale=_format(visual_scale),
        )
        position, quaternion = _origin(visual.find("origin"))
        ET.SubElement(
            body,
            "geom",
            name=f"{runtime.object_type}_visual",
            type="mesh",
            mesh=f"{runtime.object_type}_visual_mesh",
            pos=_format(position),
            quat=_format(quaternion),
            rgba=_visual_rgba(visual, {}, runtime.rgba),
            contype="0",
            conaffinity="0",
            group=str(VISUAL_GEOM_GROUP),
        )


def _add_scene_visual_assets(asset: ET.Element) -> None:
    """Add viewer-only sky and floor assets shared by every scene topology."""

    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.07 0.12 0.18",
        rgb2="0.35 0.48 0.62",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "texture",
        name="floor_checker",
        type="2d",
        builtin="checker",
        rgb1="0.18 0.20 0.22",
        rgb2="0.72 0.74 0.76",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor_checker",
        texture="floor_checker",
        texrepeat="8 8",
        texuniform="true",
        reflectance="0.08",
    )


def build_scene_xml(
    servo: ServoConfig = ServoConfig(),
    *,
    object_type: str = OBJECT_TYPE,
    visual_meshes: bool = False,
    hand_side: str = "right",
    physics_timestep: float = PHYSICS_TIMESTEP,
) -> str:
    """Build one homogeneous scene from a materialized object runtime."""

    runtime = object_runtime(object_type)
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=True)
    scene_sides = ACTION_SIDE_ORDER if side == "both" else (side,)
    for scene_side in scene_sides:
        validate_asset_manifest(object_type, hand_side=scene_side)
    object_root = ET.parse(runtime.urdf_path).getroot()
    root = ET.Element("mujoco", model=f"manorl_{runtime.object_type}_reference")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep=str(physics_timestep),
        gravity="0 0 -9.81",
        integrator="implicitfast",
    )
    asset = ET.SubElement(root, "asset")
    _add_scene_visual_assets(asset)
    worldbody = ET.SubElement(root, "worldbody")
    contact = ET.SubElement(root, "contact")
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        _add_hand_self_collision_excludes(contact, name_prefix=prefix)
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        pos=f"0 0 {FLOOR_TOP_Z}",
        size="1 1 0.01",
        material="floor_checker",
        contype="4",
        conaffinity="1",
        friction="1 0.01 0.001",
    )
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        _hand_tree(
            worldbody,
            asset,
            ET.parse(hand_urdf_path(scene_side)).getroot(),
            hand_contacts_enabled=servo.hand_contacts_enabled,
            visual_meshes=visual_meshes,
            hand_side=scene_side,
            name_prefix=prefix,
            asset_prefix=prefix,
        )
    if visual_meshes:
        for scene_side in scene_sides:
            prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
            root.append(_hand_skin_element(scene_side, name_prefix=prefix))
    _object_body(worldbody, asset, object_root, runtime, visual_meshes=visual_meshes)

    actuator = ET.SubElement(root, "actuator")
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        for name, effort, kp, dampratio in zip(
            hand_joint_names(scene_side), EFFORT, servo.kp, servo.dampratio, strict=True
        ):
            ET.SubElement(
                actuator,
                "position",
                name=f"{prefix}{name}",
                joint=f"{prefix}{name}",
                kp=f"{kp:.17g}",
                dampratio=f"{dampratio:.17g}",
                inheritrange="1",
                forcelimited="true",
                forcerange=f"{-effort:.17g} {effort:.17g}",
            )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def build_unified_scene_xml(
    servo: ServoConfig = ServoConfig(),
    *,
    object_types: Iterable[str],
    object_collisions: bool = False,
    visual_meshes: bool = False,
    hand_side: str = "right",
    physics_timestep: float = PHYSICS_TIMESTEP,
) -> str:
    """Build one fixed-topology scene containing several real object meshes.

    Each world places its scene bodies in the interaction workspace. With
    ``object_collisions=True``, these bodies also collide with each other.
    Object types absent from that world are parked far
    above the floor and retain their native gravity/contact properties; their
    trajectories cannot interact with the active workspace during a bounded
    episode.  No collision geometry is approximated or replaced.
    """

    names = tuple(dict.fromkeys(object_types))
    if not names:
        raise ValueError("unified scene requires at least one object type")
    runtimes = tuple(object_runtime(name) for name in names)
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=True)
    scene_sides = ACTION_SIDE_ORDER if side == "both" else (side,)
    for name in names:
        for scene_side in scene_sides:
            validate_asset_manifest(name, hand_side=scene_side)
    root = ET.Element("mujoco", model="manorl_unified")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep=str(physics_timestep),
        gravity="0 0 -9.81",
        integrator="implicitfast",
    )
    asset = ET.SubElement(root, "asset")
    _add_scene_visual_assets(asset)
    worldbody = ET.SubElement(root, "worldbody")
    contact = ET.SubElement(root, "contact")
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        _add_hand_self_collision_excludes(contact, name_prefix=prefix)
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        pos=f"0 0 {FLOOR_TOP_Z}",
        size="1 1 0.01",
        material="floor_checker",
        contype="4",
        conaffinity="1",
        friction="1 0.01 0.001",
    )
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        _hand_tree(
            worldbody,
            asset,
            ET.parse(hand_urdf_path(scene_side)).getroot(),
            hand_contacts_enabled=servo.hand_contacts_enabled,
            visual_meshes=visual_meshes,
            hand_side=scene_side,
            name_prefix=prefix,
            asset_prefix=prefix,
        )
    if visual_meshes:
        for scene_side in scene_sides:
            prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
            root.append(_hand_skin_element(scene_side, name_prefix=prefix))
    for runtime in runtimes:
        object_root = ET.parse(runtime.urdf_path).getroot()
        _object_body(
            worldbody,
            asset,
            object_root,
            runtime,
            visual_meshes=visual_meshes,
            object_collisions=object_collisions,
        )

    actuator = ET.SubElement(root, "actuator")
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        for name, effort, kp, dampratio in zip(
            hand_joint_names(scene_side), EFFORT, servo.kp, servo.dampratio, strict=True
        ):
            ET.SubElement(
                actuator,
                "position",
                name=f"{prefix}{name}",
                joint=f"{prefix}{name}",
                kp=f"{kp:.17g}",
                dampratio=f"{dampratio:.17g}",
                inheritrange="1",
                forcelimited="true",
                forcerange=f"{-effort:.17g} {effort:.17g}",
            )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _model_names(
    mujoco: Any, model: Any, object_type: Any, count: int
) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)
    )


def _is_collision_geom(mujoco: Any, model: Any, geom_id: int) -> bool:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    return name.endswith("_collision")


def validate_compiled_model(
    mujoco: Any,
    model: Any,
    servo: ServoConfig = ServoConfig(),
    *,
    object_type: str = OBJECT_TYPE,
    hand_side: str = "right",
    physics_timestep: float = PHYSICS_TIMESTEP,
) -> None:
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=True)
    scene_sides = ACTION_SIDE_ORDER if side == "both" else (side,)
    runtime = object_runtime(object_type)
    expected_joint_names = tuple(
        f"{scene_side}_{name}" if len(scene_sides) > 1 else name
        for scene_side in scene_sides
        for name in hand_joint_names(scene_side)
    )
    joint_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    if joint_names[: len(expected_joint_names)] != expected_joint_names or joint_names[
        len(expected_joint_names) :
    ] != (runtime.free_joint_name,):
        raise ValueError(f"compiled joint order mismatch: {joint_names}")
    actuator_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    if actuator_names != expected_joint_names:
        raise ValueError(f"compiled actuator order mismatch: {actuator_names}")
    expected_dof = len(expected_joint_names)
    if (
        model.nu != expected_dof
        or model.nq != expected_dof + 7
        or model.nv != expected_dof + 6
    ):
        raise ValueError(
            f"compiled dimensions mismatch: nq={model.nq}, nv={model.nv}, nu={model.nu}"
        )
    if not np.isclose(model.opt.timestep, physics_timestep):
        raise ValueError(f"compiled timestep mismatch: {model.opt.timestep}")
    for actuator_id, (name, effort, kp, dampratio) in enumerate(
        zip(
            expected_joint_names,
            tuple(EFFORT) * len(scene_sides),
            tuple(servo.kp) * len(scene_sides),
            tuple(servo.dampratio) * len(scene_sides),
            strict=True,
        )
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_address = int(model.jnt_dofadr[joint_id])
        if not np.isclose(model.dof_frictionloss[dof_address], JOINT_FRICTIONLOSS):
            raise ValueError(f"compiled frictionloss mismatch for {name}")
        if not np.isclose(model.dof_armature[dof_address], JOINT_ARMATURE):
            raise ValueError(f"compiled armature mismatch for {name}")
        if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise ValueError(f"actuator {name} is attached to the wrong joint")
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError(f"actuator {name} must use a direct joint transmission")
        if not model.actuator_ctrllimited[actuator_id] or not np.allclose(
            model.actuator_ctrlrange[actuator_id], model.jnt_range[joint_id]
        ):
            raise ValueError(f"compiled actuator control range mismatch for {name}")
        if not model.actuator_forcelimited[actuator_id] or not np.allclose(
            model.actuator_forcerange[actuator_id], (-effort, effort)
        ):
            raise ValueError(f"compiled actuator effort limit mismatch for {name}")
        if not np.isclose(model.actuator_gainprm[actuator_id, 0], kp):
            raise ValueError(f"compiled position gain mismatch for {name}")
        if not np.isclose(model.actuator_biasprm[actuator_id, 1], -kp):
            raise ValueError(f"compiled position bias mismatch for {name}")
        kv = -float(model.actuator_biasprm[actuator_id, 2])
        if not np.isfinite(kv) or kv <= 0:
            raise ValueError(
                f"compiled dampratio did not produce positive damping for {name}"
            )
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.body_name)
    if object_id < 0:
        raise ValueError(f"compiled object body is absent: {runtime.body_name}")
    if model.body_gravcomp[object_id] != 0:
        raise ValueError(f"{runtime.object_type} gravity must remain active")
    object_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == object_id
        and _is_collision_geom(mujoco, model, geom_id)
    ]
    if len(object_geom_ids) != runtime.collision_geom_count:
        raise ValueError(
            f"expected {runtime.collision_geom_count} {runtime.object_type} collision geoms, "
            f"got {len(object_geom_ids)}"
        )
    if not np.all(model.geom_contype[object_geom_ids] == 2) or not np.all(
        model.geom_conaffinity[object_geom_ids] == 5
    ):
        raise ValueError(
            "compiled object collision masks mismatch source configuration"
        )
    hand_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] not in (0, object_id)
        and _is_collision_geom(mujoco, model, geom_id)
    ]
    if len(hand_geom_ids) != 16 * len(scene_sides):
        raise ValueError(
            f"expected {16 * len(scene_sides)} hand collision geoms, got {len(hand_geom_ids)}"
        )
    expected_hand_bits = (1, 7) if servo.hand_contacts_enabled else (0, 0)
    if not np.all(
        model.geom_contype[hand_geom_ids] == expected_hand_bits[0]
    ) or not np.all(model.geom_conaffinity[hand_geom_ids] == expected_hand_bits[1]):
        raise ValueError("compiled hand collision masks mismatch servo configuration")
    expected_exclude_signatures = {
        (min(first_id, second_id) << 16) + max(first_id, second_id)
        for scene_side in scene_sides
        for prefix in [f"{scene_side}_" if len(scene_sides) > 1 else ""]
        for body_names in HAND_SELF_COLLISION_GROUPS.values()
        for first_name, second_name in combinations(body_names, 2)
        for first_id, second_id in [
            (
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{first_name}"
                ),
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{second_name}"
                ),
            )
        ]
    }
    actual_exclude_signatures = {int(value) for value in model.exclude_signature}
    if actual_exclude_signatures != expected_exclude_signatures:
        raise ValueError(
            "compiled within-finger collision excludes mismatch source groups"
        )
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name != runtime.body_name and model.body_gravcomp[body_id] != 1:
            raise ValueError(f"hand body gravity compensation missing: {name}")


def compile_model(
    servo: ServoConfig = ServoConfig(),
    *,
    object_type: str = OBJECT_TYPE,
    visual_meshes: bool = False,
    hand_side: str = "right",
    physics_timestep: float = PHYSICS_TIMESTEP,
) -> tuple[Any, Any]:
    """Compile and validate one bounded native-servo scene."""

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("mujoco is required to compile the ManoRL scene") from exc
    model = mujoco.MjModel.from_xml_string(
        build_scene_xml(
            servo,
            object_type=object_type,
            visual_meshes=visual_meshes,
            hand_side=hand_side,
            physics_timestep=physics_timestep,
        )
    )
    validate_compiled_model(
        mujoco,
        model,
        servo,
        object_type=object_type,
        hand_side=hand_side,
        physics_timestep=physics_timestep,
    )
    validate_static_fk(mujoco, model, object_type=object_type, hand_side=hand_side)
    return mujoco, model


def validate_unified_compiled_model(
    mujoco: Any,
    model: Any,
    servo: ServoConfig = ServoConfig(),
    *,
    object_types: Iterable[str],
    object_collisions: bool = False,
    hand_side: str = "right",
    physics_timestep: float = PHYSICS_TIMESTEP,
) -> None:
    """Validate the fixed hand topology and every real object in a superset scene."""

    names = tuple(dict.fromkeys(object_types))
    if not names:
        raise ValueError("unified model validation requires at least one object type")
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=True)
    scene_sides = ACTION_SIDE_ORDER if side == "both" else (side,)
    runtimes = tuple(object_runtime(name) for name in names)
    expected_hand_joint_names = tuple(
        f"{scene_side}_{name}" if len(scene_sides) > 1 else name
        for scene_side in scene_sides
        for name in hand_joint_names(scene_side)
    )
    joint_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    expected_joints = expected_hand_joint_names + tuple(
        runtime.free_joint_name for runtime in runtimes
    )
    if joint_names != expected_joints:
        raise ValueError(f"unified joint order mismatch: {joint_names}")
    actuator_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    if actuator_names != expected_hand_joint_names:
        raise ValueError(f"unified actuator order mismatch: {actuator_names}")
    expected_dof = len(expected_hand_joint_names)
    expected_nq = expected_dof + 7 * len(runtimes)
    expected_nv = expected_dof + 6 * len(runtimes)
    if model.nu != expected_dof or model.nq != expected_nq or model.nv != expected_nv:
        raise ValueError(
            f"unified dimensions mismatch: nq={model.nq}, nv={model.nv}, nu={model.nu}; "
            f"expected nq={expected_nq}, nv={expected_nv}, nu={expected_dof}"
        )
    if not np.isclose(model.opt.timestep, physics_timestep):
        raise ValueError(f"unified timestep mismatch: {model.opt.timestep}")
    for actuator_id, (name, effort, kp, dampratio) in enumerate(
        zip(
            expected_hand_joint_names,
            tuple(EFFORT) * len(scene_sides),
            tuple(servo.kp) * len(scene_sides),
            tuple(servo.dampratio) * len(scene_sides),
            strict=True,
        )
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_address = int(model.jnt_dofadr[joint_id])
        if not np.isclose(model.dof_frictionloss[dof_address], JOINT_FRICTIONLOSS):
            raise ValueError(f"compiled frictionloss mismatch for {name}")
        if not np.isclose(model.dof_armature[dof_address], JOINT_ARMATURE):
            raise ValueError(f"compiled armature mismatch for {name}")
        if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise ValueError(f"actuator {name} is attached to the wrong joint")
        if not model.actuator_ctrllimited[actuator_id] or not np.allclose(
            model.actuator_ctrlrange[actuator_id], model.jnt_range[joint_id]
        ):
            raise ValueError(f"compiled actuator control range mismatch for {name}")
        if not model.actuator_forcelimited[actuator_id] or not np.allclose(
            model.actuator_forcerange[actuator_id], (-effort, effort)
        ):
            raise ValueError(f"compiled actuator effort limit mismatch for {name}")
        if not np.isclose(model.actuator_gainprm[actuator_id, 0], kp):
            raise ValueError(f"compiled position gain mismatch for {name}")
        if not np.isclose(model.actuator_biasprm[actuator_id, 1], -kp):
            raise ValueError(f"compiled position bias mismatch for {name}")
        kv = -float(model.actuator_biasprm[actuator_id, 2])
        if not np.isfinite(kv) or kv <= 0:
            raise ValueError(
                f"compiled dampratio did not produce positive damping for {name}"
            )
    hand_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id]
        not in {
            0,
            *(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.body_name)
                for runtime in runtimes
            ),
        }
        and _is_collision_geom(mujoco, model, geom_id)
    ]
    if len(hand_geom_ids) != 16 * len(scene_sides):
        raise ValueError(
            f"expected {16 * len(scene_sides)} hand collision geoms, got {len(hand_geom_ids)}"
        )
    expected_hand_bits = (1, 7) if servo.hand_contacts_enabled else (0, 0)
    if not np.all(
        model.geom_contype[hand_geom_ids] == expected_hand_bits[0]
    ) or not np.all(model.geom_conaffinity[hand_geom_ids] == expected_hand_bits[1]):
        raise ValueError("compiled hand collision masks mismatch servo configuration")
    for runtime in runtimes:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.body_name)
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, runtime.free_joint_name
        )
        if body_id < 0 or joint_id < 0:
            raise ValueError(
                f"compiled unified object is absent: {runtime.object_type}"
            )
        if model.body_gravcomp[body_id] != 0:
            raise ValueError(
                f"unified object {runtime.object_type} must retain native gravity"
            )
        object_geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
            and _is_collision_geom(mujoco, model, geom_id)
        ]
        if len(object_geom_ids) != runtime.collision_geom_count:
            raise ValueError(
                f"expected {runtime.collision_geom_count} {runtime.object_type} collision geoms, "
                f"got {len(object_geom_ids)}"
            )
        if not np.all(model.geom_contype[object_geom_ids] == 2) or not np.all(
            model.geom_conaffinity[object_geom_ids] == (7 if object_collisions else 5)
        ):
            raise ValueError(f"compiled {runtime.object_type} collision masks mismatch")
    validate_static_fk(mujoco, model, object_type=names[0], hand_side=side)


def compile_unified_model(
    servo: ServoConfig = ServoConfig(),
    *,
    object_types: Iterable[str],
    object_collisions: bool = False,
    visual_meshes: bool = False,
    hand_side: str = "right",
    physics_timestep: float = PHYSICS_TIMESTEP,
) -> tuple[Any, Any]:
    """Compile one fixed-topology model containing the requested real objects."""

    names = tuple(dict.fromkeys(object_types))
    if not names:
        raise ValueError("unified model requires at least one object type")
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("mujoco is required to compile the ManoRL scene") from exc
    model = mujoco.MjModel.from_xml_string(
        build_unified_scene_xml(
            servo,
            object_types=names,
            object_collisions=object_collisions,
            visual_meshes=visual_meshes,
            hand_side=hand_side,
            physics_timestep=physics_timestep,
        )
    )
    validate_unified_compiled_model(
        mujoco,
        model,
        servo,
        object_types=names,
        object_collisions=object_collisions,
        hand_side=hand_side,
        physics_timestep=physics_timestep,
    )
    return mujoco, model


def _matrix_from_pose(
    position: tuple[float, ...], quaternion_wxyz: tuple[float, ...]
) -> NDArray[np.float64]:
    w, x, y, z = quaternion_wxyz
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def urdf_zero_fk(hand_side: str = "right") -> dict[str, NDArray[np.float64]]:
    """Compute zero-joint link transforms independently from the MJCF builder."""

    root = ET.parse(hand_urdf_path(hand_side)).getroot()
    transforms = {"base_link": np.eye(4, dtype=np.float64)}
    remaining = list(root.findall("joint"))
    while remaining:
        progressed = False
        for joint in remaining.copy():
            parent = joint.find("parent")
            child = joint.find("child")
            assert parent is not None and child is not None
            parent_name = parent.get("link", "")
            if parent_name not in transforms:
                continue
            position, quaternion = _origin(joint.find("origin"))
            transforms[child.get("link", "")] = transforms[
                parent_name
            ] @ _matrix_from_pose(position, quaternion)
            remaining.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("hand URDF joint graph is disconnected or cyclic")
    return transforms


def validate_static_fk(
    mujoco: Any,
    model: Any,
    *,
    object_type: str = OBJECT_TYPE,
    hand_side: str = "right",
    atol: float = 1e-10,
) -> None:
    """Check compiled body frames against independent zero-pose URDF FK."""

    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=True)
    scene_sides = ACTION_SIDE_ORDER if side == "both" else (side,)
    data = mujoco.MjData(model)
    hand_width = sum(len(hand_joint_names(scene_side)) for scene_side in scene_sides)
    data.qpos[:hand_width] = 0.0
    runtime = object_runtime(object_type)
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, runtime.free_joint_name
    )
    object_qpos_adr = model.jnt_qposadr[object_joint]
    data.qpos[object_qpos_adr + 3] = 1.0
    mujoco.mj_forward(model, data)
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        for name, expected in urdf_zero_fk(scene_side).items():
            compiled_name = f"{prefix}{name}"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, compiled_name)
            if body_id < 0:
                raise ValueError(f"compiled hand body is absent: {compiled_name}")
            if not np.allclose(data.xpos[body_id], expected[:3, 3], atol=atol, rtol=0):
                raise ValueError(
                    f"compiled static FK position mismatch at {compiled_name}"
                )
            if not np.allclose(
                data.xmat[body_id].reshape(3, 3), expected[:3, :3], atol=atol, rtol=0
            ):
                raise ValueError(
                    f"compiled static FK orientation mismatch at {compiled_name}"
                )


@lru_cache(maxsize=None)
def object_collision_vertices(object_type: str = OBJECT_TYPE) -> NDArray[np.float64]:
    """Return pinned DexStream collision triangles in object-local metric coordinates."""

    runtime = object_runtime(object_type)
    validate_asset_manifest(object_type)
    pieces = [
        _collision_mesh_vertices(object_type, path, scale)
        for path, scale in zip(
            runtime.collision_mesh_paths,
            runtime.collision_mesh_scales,
            strict=True,
        )
    ]
    result = np.concatenate(pieces, axis=0)
    if (
        result.ndim != 2
        or result.shape[1] != 3
        or len(result) < 12
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"unexpected {object_type} collision vertices: {result.shape}")
    result.setflags(write=False)
    return result


def _collision_mesh_vertices(
    object_type: str,
    path: Path,
    scale: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Decode one digest-checked collision piece into metric triangles."""

    if path.suffix.lower() == ".stl":
        payload = path.read_bytes()
        if len(payload) < 84:
            raise ValueError(f"{object_type} collision STL is too short")
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        expected_length = 84 + triangle_count * 50
        if len(payload) != expected_length:
            raise ValueError(
                f"{object_type} collision STL has an invalid binary length"
            )
        triangles = np.frombuffer(
            payload,
            dtype=np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]
            ),
            count=triangle_count,
            offset=84,
        )
        result = np.asarray(triangles["vertices"], dtype=np.float64).reshape(-1, 3)
    elif path.suffix.lower() == ".obj":
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts or parts[0] == "#":
                continue
            if parts[0] == "v" and len(parts) == 4:
                vertices.append(tuple(float(value) for value in parts[1:4]))
            elif parts[0] == "f" and len(parts) == 4:
                faces.append(
                    tuple(
                        int(value.split("/", maxsplit=1)[0]) - 1 for value in parts[1:4]
                    )
                )
        vertex_array = np.asarray(vertices, dtype=np.float64)
        face_array = np.asarray(faces, dtype=np.int64)
        if (
            face_array.ndim != 2
            or face_array.shape[1:] != (3,)
            or np.any(face_array < 0)
            or np.any(face_array >= len(vertex_array))
        ):
            raise ValueError(
                f"{object_type} collision OBJ must contain indexed triangles"
            )
        result = vertex_array[face_array].reshape(-1, 3)
    else:
        raise ValueError(
            f"unsupported collision mesh format for {object_type}: {path.suffix}"
        )
    result = result * np.asarray(scale, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[1] != 3
        or len(result) < 12
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(
            f"unexpected {object_type} collision piece vertices: {result.shape}"
        )
    return result
