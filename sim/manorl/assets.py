"""Deterministic MuJoCo scene construction from curated authoritative URDF assets."""

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
    EFFORT,
    FLOOR_TOP_Z,
    JOINT_ARMATURE,
    JOINT_FRICTIONLOSS,
    JOINT_NAMES,
    OBJECT_TYPE,
    PHYSICS_TIMESTEP,
    ServoConfig,
)

ASSET_ROOT = Path(__file__).resolve().parent / "runtime_assets"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ALL_ASSETS_ROOT = REPOSITORY_ROOT / "assets" / "all_assets"
ALL_ASSETS_SIM_ROOT = ALL_ASSETS_ROOT / "Assets" / "sim"
ALL_OBJECT_GRASPS = ALL_ASSETS_ROOT / "Assets" / "object_grasps_simple.yaml"
ALL_OBJECT_GRASPS_SHA256 = "97293b96a3015173bf2fa875d165b04c925953147ebc0f4c0935ed1215558a9d"
HAND_URDF = ASSET_ROOT / "hand" / "mano_hand.urdf"
OBJECT_URDF = ASSET_ROOT / "cube1" / "cube1.urdf"
ASSET_MANIFEST = ASSET_ROOT / "manifest.json"
OBJECT_MESH = ASSET_ROOT / "cube1" / "cube1_aligned.stl"


@dataclass(frozen=True)
class ObjectRuntime:
    """Materialized object inputs for one homogeneous MJX model."""

    object_type: str
    link_name: str
    body_name: str
    free_joint_name: str
    source_mesh_filename: str
    urdf_path: Path
    collision_mesh_path: Path
    grasp_mapping_path: Path
    source_mesh_scale: tuple[float, float, float]
    collision_mesh_scale: tuple[float, float, float]
    rgba: str
    geometry_type: str = "box"
    collision_geom_count: int = 1
    expected_sha256: tuple[tuple[Path, str], ...] = ()


def _all_assets_runtime(
    object_type: str,
    *,
    urdf_sha256: str,
    collision_sha256: str,
    rgba: str,
) -> ObjectRuntime:
    urdf_path = ALL_ASSETS_SIM_ROOT / "mano_objects_urdf" / f"{object_type}.urdf"
    collision_path = (
        ALL_ASSETS_SIM_ROOT
        / "for_math_retaregeting"
        / object_type
        / "coacd"
        / "coacd_convex_piece_0.obj"
    )
    return ObjectRuntime(
        object_type=object_type,
        link_name=f"{object_type}_link",
        body_name=object_type,
        free_joint_name=f"{object_type}_free",
        source_mesh_filename=f"{object_type}.obj",
        urdf_path=urdf_path,
        collision_mesh_path=collision_path,
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        collision_mesh_scale=(1.0, 1.0, 1.0),
        rgba=rgba,
        expected_sha256=(
            (urdf_path, urdf_sha256),
            (collision_path, collision_sha256),
            (ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),
        ),
    )


# This registry is deliberately closed. Adding a name requires materialized,
# digest-checked URDF, collision, and grasp inputs; arbitrary Lance object names
# must never fall through to cube1 geometry. Each profile compiles a separate
# static MJX model and heterogeneous vector batches are routed by object type.
_OBJECT_RUNTIMES = {
    "cube1": ObjectRuntime(
        object_type="cube1",
        link_name="cube1_link",
        body_name="cube1",
        free_joint_name="cube1_free",
        source_mesh_filename="cube1.obj",
        urdf_path=ASSET_ROOT / "cube1" / "cube1.urdf",
        collision_mesh_path=OBJECT_MESH,
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        collision_mesh_scale=(0.001, 0.001, 0.001),
        rgba="0.8 0.18 0.16 1",
        expected_sha256=((ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),),
    ),
    "cube2": _all_assets_runtime(
        "cube2",
        urdf_sha256="334fb68ecf7eca5a860b72f10b556d37af6045e18cfc2049f9f2490463abdce2",
        collision_sha256="018616c33d159ca5246da8fdc923579c009b900e56a1997e9e8a34463d069d6f",
        rgba="0.15 0.45 0.85 1",
    ),
    "cuboid1": _all_assets_runtime(
        "cuboid1",
        urdf_sha256="311d3d33c4afc1b7e0c715520592b73a2752160a8c91ebe5985eb799b3b927ec",
        collision_sha256="60bae7c578e404d09b942373b66958ce873642cc2b741000d618ea5852729b29",
        rgba="0.72 0.45 0.12 1",
    ),
    "cuboid2": _all_assets_runtime(
        "cuboid2",
        urdf_sha256="b1578b13d594cbced941a479fd4d29c7c408844152daaf1215eee42ec7c07351",
        collision_sha256="ce0e9e194c7bf7798ea300702ee82de2b85c7e823d3c9ca394c5e1b85a1c412d",
        rgba="0.36 0.62 0.18 1",
    ),
    "cylinder1": _all_assets_runtime(
        "cylinder1",
        urdf_sha256="804894a8406a3a9de2cb377af3927a1cdcd9ddd49e26cce4f9f36940833bd37a",
        collision_sha256="104df8077aa9009571f25d5637a2b8aa36fc78e29479c5155aefe60a5c178509",
        rgba="0.62 0.25 0.55 1",
    ),
    "cylinder2": _all_assets_runtime(
        "cylinder2",
        urdf_sha256="1e33837f669d0c69c3c155aacc541acd8717649b35eea261f8f3efea0a5743c9",
        collision_sha256="14a264cf597bdfabc662834fa8c750c7b277eb3c19903a6b04af8041d01c189e",
        rgba="0.16 0.62 0.58 1",
    ),
    "cylinder3": _all_assets_runtime(
        "cylinder3",
        urdf_sha256="0ee0d19fe45f11792172db042d66dc1ae6dad2c9ea3c1dd96669a4ba35974630",
        collision_sha256="a955748b8cc1d85e7c4ca9a63a6ed334558149994cea1d176f1296e51bdfc3ee",
        rgba="0.75 0.22 0.25 1",
    ),
    "cylinder4": _all_assets_runtime(
        "cylinder4",
        urdf_sha256="35e83a639fee1dfdb44c7c77a65955665465afc6955a53e7b5a22e15e9d16fe0",
        collision_sha256="1700c2f9528711ebff23c0d42a311a8d3f00356802f64339340ea1e6db26bfb4",
        rgba="0.30 0.45 0.78 1",
    ),
    "cylinder5": _all_assets_runtime(
        "cylinder5",
        urdf_sha256="a026bf1762370f8471e73d20c8f0ad27e4213a654bb19b7d91197c557f0f1478",
        collision_sha256="8c0cbbe5ac1fcc5827d0983454db0d47e7adcb12b7c0c53f6565b5977cb234f7",
        rgba="0.72 0.54 0.12 1",
    ),
    "cylinder6": _all_assets_runtime(
        "cylinder6",
        urdf_sha256="b1105b23619ce7697cddce6d18ec51ed7740e76e5795a63fcf22d2737f8b4d17",
        collision_sha256="be1134bdce8bc270ec2b5f64d03c7adb650f8c429555c5d465818ecdd671bec8",
        rgba="0.30 0.68 0.32 1",
    ),
    "sphere1": _all_assets_runtime(
        "sphere1",
        urdf_sha256="5a6560e0c32d99c580cda2932e6563edc3406473fc8a0f2f780f8664bc23e983",
        collision_sha256="3ed3410d72eb6ee6915d6530f5544921d32eaa47261bfc20b2d3101b14526c8b",
        rgba="0.85 0.30 0.18 1",
    ),
    "sphere2": _all_assets_runtime(
        "sphere2",
        urdf_sha256="5c934be6a43e7efda8338998aeb8e6dceae1935bfcb74ea5ef2a65aa5a0abf7e",
        collision_sha256="7ab5563b5883a61b9ed55143b07cf582c0b3ab2fdbaea24340320fc81e112460",
        rgba="0.18 0.58 0.82 1",
    ),
    "sphere3": _all_assets_runtime(
        "sphere3",
        urdf_sha256="d79e32663a1f065dd4855297e182b745262615329176e7bb314b5de8b75e2e04",
        collision_sha256="da32bee52e7841bb2e1313ed9752623c351f34e0023d0f7a7a276c5323835601",
        rgba="0.46 0.68 0.18 1",
    ),
}


def supported_object_types() -> tuple[str, ...]:
    """Return the closed set of object types with complete runtime assets."""

    return tuple(sorted(_OBJECT_RUNTIMES))


def object_runtime(object_type: str = OBJECT_TYPE) -> ObjectRuntime:
    """Resolve a materialized runtime or fail before model compilation."""

    if not isinstance(object_type, str) or object_type not in _OBJECT_RUNTIMES:
        available = ", ".join(supported_object_types())
        raise ValueError(
            f"unsupported ManoRL object runtime {object_type!r}; materialized runtimes: {available}"
        )
    runtime = _OBJECT_RUNTIMES[object_type]
    missing = [
        str(path)
        for path in (runtime.urdf_path, runtime.collision_mesh_path, runtime.grasp_mapping_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"ManoRL object runtime {object_type!r} is registered but not materialized: {missing}"
        )
    return runtime
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


def _numbers(value: str | None, count: int, default: Iterable[float]) -> tuple[float, ...]:
    values = tuple(default) if value is None else tuple(float(item) for item in value.split())
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


def _required_paths(object_type: str | None = None) -> tuple[Path, ...]:
    hand_meshes = tuple((ASSET_ROOT / "hand" / "meshes").glob("*.stl"))
    if len(hand_meshes) != 16:
        raise FileNotFoundError(f"expected exactly 16 curated hand collision meshes, got {len(hand_meshes)}")
    runtimes = (object_runtime(object_type),) if object_type is not None else tuple(
        object_runtime(name) for name in supported_object_types()
    )
    object_paths = tuple(
        path
        for runtime in runtimes
        for path in (runtime.urdf_path, runtime.collision_mesh_path, runtime.grasp_mapping_path)
    )
    paths = (HAND_URDF, *hand_meshes, ASSET_MANIFEST, *object_paths)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"curated ManoRL assets are incomplete: {missing}")
    return tuple(paths)


def validate_asset_manifest(object_type: str | None = None) -> dict[str, Any]:
    """Verify every curated file against its committed provenance digest."""

    required_paths = _required_paths(object_type)
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("source_commit") != "ead79126589d1abf2362ea30b9d674d9e675a2f9":
        raise ValueError("asset manifest does not name the settled all_assets commit")
    entries = manifest.get("files", [])
    if not isinstance(entries, list):
        raise ValueError("asset manifest files must be a list")
    digests_by_path = {entry.get("curated_path"): entry for entry in entries}
    runtimes = (object_runtime(object_type),) if object_type is not None else tuple(
        object_runtime(name) for name in supported_object_types()
    )
    external_digests = {
        path.resolve(): digest
        for runtime in runtimes
        for path, digest in runtime.expected_sha256
    }
    for path in required_paths:
        if path == ASSET_MANIFEST:
            continue
        try:
            relative = path.relative_to(ASSET_ROOT).as_posix()
        except ValueError:
            expected_digest = external_digests.get(path.resolve())
            if expected_digest is None:
                raise ValueError(f"unpinned runtime asset outside curated root: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_digest:
                raise ValueError(f"pinned all_assets digest mismatch for {path}: {digest}")
        else:
            if relative not in digests_by_path:
                raise ValueError(f"asset manifest has no digest entry for {relative}")
    for entry in entries:
        path = ASSET_ROOT / entry["curated_path"]
        if not path.is_file():
            raise FileNotFoundError(f"manifest asset is absent: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["curated_sha256"]:
            raise ValueError(f"asset digest mismatch for {path}: {digest}")
    return manifest


def _link_inertial(body: ET.Element, link: ET.Element) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError(f"authoritative link {link.get('name')} has no inertial")
    position, quaternion = _origin(inertial.find("origin"))
    if not np.allclose(quaternion, (1.0, 0.0, 0.0, 0.0), atol=1e-15, rtol=0):
        raise ValueError(f"rotated full inertia is unsupported for link {link.get('name')}")
    mass_element = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass_element is None or inertia is None:
        raise ValueError(f"authoritative link {link.get('name')} has incomplete inertial")
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
    body: ET.Element, link: ET.Element, *, hand_contacts_enabled: bool
) -> None:
    collisions = link.findall("collision")
    if len(collisions) > 1:
        raise ValueError(f"hand link {link.get('name')} unexpectedly has multiple collisions")
    if not collisions:
        return
    collision = collisions[0]
    geometry = collision.find("geometry")
    mesh = None if geometry is None else geometry.find("mesh")
    if mesh is None:
        raise ValueError(f"hand collision for {link.get('name')} is not a mesh")
    filename = Path(mesh.get("filename", "")).name
    source_path = ASSET_ROOT / "hand" / "meshes" / filename
    if not source_path.is_file():
        raise FileNotFoundError(f"curated hand collision mesh is absent: {source_path}")
    position, quaternion = _origin(collision.find("origin"))
    attributes = {
        "name": f"{link.get('name')}_collision",
        "type": "mesh",
        "mesh": f"hand_{Path(filename).stem}",
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
    ET.SubElement(body, "geom", attributes)


def _hand_tree(
    worldbody: ET.Element,
    asset: ET.Element,
    urdf_root: ET.Element,
    *,
    hand_contacts_enabled: bool,
) -> None:
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
        raise ValueError("authoritative hand URDF must reference 16 unique collision meshes")
    for filename in mesh_files:
        attributes = {
            "name": f"hand_{Path(filename).stem}",
            "file": str((ASSET_ROOT / "hand" / "meshes" / filename).resolve()),
        }
        source_mesh = next(
            link.find("collision/geometry/mesh")
            for link in links.values()
            if link.find("collision/geometry/mesh") is not None
            and Path(link.find("collision/geometry/mesh").get("filename", "")).name == filename
        )
        scale = source_mesh.get("scale")
        if scale is not None:
            attributes["scale"] = scale
        ET.SubElement(asset, "mesh", attributes)

    def append_link(parent_xml: ET.Element, link_name: str, source_joint: ET.Element | None) -> None:
        attributes = {"name": link_name, "gravcomp": "1"}
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
                raise ValueError(f"joint {source_joint.get('name')} has no axis or limit")
            ET.SubElement(
                body,
                "joint",
                name=source_joint.get("name", ""),
                type="slide" if joint_type == "prismatic" else "hinge",
                axis=axis.get("xyz", ""),
                range=f"{limit.get('lower')} {limit.get('upper')}",
                limited="true",
                damping="0",
                frictionloss=str(JOINT_FRICTIONLOSS),
                armature=str(JOINT_ARMATURE),
            )
        _link_inertial(body, links[link_name])
        _link_collision(body, links[link_name], hand_contacts_enabled=hand_contacts_enabled)
        for joint in children.get(link_name, []):
            child = joint.find("child")
            assert child is not None
            append_link(body, child.get("link", ""), joint)

    append_link(worldbody, "base_link", None)


def _add_hand_self_collision_excludes(contact: ET.Element) -> None:
    for group_name, body_names in HAND_SELF_COLLISION_GROUPS.items():
        for pair_index, (first, second) in enumerate(combinations(body_names, 2)):
            ET.SubElement(
                contact,
                "exclude",
                name=f"{group_name}_internal_{pair_index}",
                body1=first,
                body2=second,
            )


def _object_body(
    worldbody: ET.Element,
    asset: ET.Element,
    urdf_root: ET.Element,
    runtime: ObjectRuntime,
    *,
    gravity_compensated: bool = False,
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
    collision = collisions[0]
    source_mesh = collision.find("geometry/mesh")
    if (
        source_mesh is None
        or Path(source_mesh.get("filename", "")).name != runtime.source_mesh_filename
    ):
        raise ValueError(
            f"{runtime.object_type} URDF collision must reference "
            f"{runtime.source_mesh_filename}"
        )
    mesh_scale = _numbers(source_mesh.get("scale"), 3, (1.0, 1.0, 1.0))
    if mesh_scale != runtime.source_mesh_scale:
        raise ValueError(
            f"{runtime.object_type} URDF collision scale must remain {runtime.source_mesh_scale}"
        )
    ET.SubElement(
        asset,
        "mesh",
        name=f"{runtime.object_type}_mesh",
        file=str(runtime.collision_mesh_path.resolve()),
        scale=_format(runtime.collision_mesh_scale),
    )

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
    position, quaternion = _origin(collision.find("origin"))
    ET.SubElement(
        body,
        "geom",
        name=f"{runtime.object_type}_collision",
        type="mesh",
        mesh=f"{runtime.object_type}_mesh",
        pos=_format(position),
        quat=_format(quaternion),
        rgba=runtime.rgba,
        contype="2",
        conaffinity="5",
        condim="3",
        friction="0.9 0.01 0.001",
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
    servo: ServoConfig = ServoConfig(), *, object_type: str = OBJECT_TYPE
) -> str:
    """Build one homogeneous scene from a materialized object runtime."""

    runtime = object_runtime(object_type)
    validate_asset_manifest(object_type)
    hand_root = ET.parse(HAND_URDF).getroot()
    object_root = ET.parse(runtime.urdf_path).getroot()
    root = ET.Element("mujoco", model=f"manorl_{runtime.object_type}_reference")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep=str(PHYSICS_TIMESTEP),
        gravity="0 0 -9.81",
        integrator="implicitfast",
    )
    asset = ET.SubElement(root, "asset")
    _add_scene_visual_assets(asset)
    worldbody = ET.SubElement(root, "worldbody")
    contact = ET.SubElement(root, "contact")
    _add_hand_self_collision_excludes(contact)
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
    _hand_tree(
        worldbody,
        asset,
        hand_root,
        hand_contacts_enabled=servo.hand_contacts_enabled,
    )
    _object_body(worldbody, asset, object_root, runtime)

    actuator = ET.SubElement(root, "actuator")
    for name, effort, kp, dampratio in zip(
        JOINT_NAMES, EFFORT, servo.kp, servo.dampratio, strict=True
    ):
        ET.SubElement(
            actuator,
            "position",
            name=name,
            joint=name,
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
) -> str:
    """Build one fixed-topology scene containing several real object meshes.

    Each world in an MJX data batch selects one object by placing that object's
    free body in the interaction workspace.  Inactive objects are parked far
    above the floor and retain their native gravity/contact properties; their
    trajectories cannot interact with the active workspace during a bounded
    episode.  No collision geometry is approximated or replaced.
    """

    names = tuple(dict.fromkeys(object_types))
    if not names:
        raise ValueError("unified scene requires at least one object type")
    runtimes = tuple(object_runtime(name) for name in names)
    for name in names:
        validate_asset_manifest(name)
    hand_root = ET.parse(HAND_URDF).getroot()
    root = ET.Element("mujoco", model="manorl_unified")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep=str(PHYSICS_TIMESTEP),
        gravity="0 0 -9.81",
        integrator="implicitfast",
    )
    asset = ET.SubElement(root, "asset")
    _add_scene_visual_assets(asset)
    worldbody = ET.SubElement(root, "worldbody")
    contact = ET.SubElement(root, "contact")
    _add_hand_self_collision_excludes(contact)
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
    _hand_tree(
        worldbody,
        asset,
        hand_root,
        hand_contacts_enabled=servo.hand_contacts_enabled,
    )
    for runtime in runtimes:
        object_root = ET.parse(runtime.urdf_path).getroot()
        _object_body(
            worldbody,
            asset,
            object_root,
            runtime,
        )

    actuator = ET.SubElement(root, "actuator")
    for name, effort, kp, dampratio in zip(
        JOINT_NAMES, EFFORT, servo.kp, servo.dampratio, strict=True
    ):
        ET.SubElement(
            actuator,
            "position",
            name=name,
            joint=name,
            kp=f"{kp:.17g}",
            dampratio=f"{dampratio:.17g}",
            inheritrange="1",
            forcelimited="true",
            forcerange=f"{-effort:.17g} {effort:.17g}",
        )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _model_names(mujoco: Any, model: Any, object_type: Any, count: int) -> tuple[str, ...]:
    return tuple(mujoco.mj_id2name(model, object_type, index) or "" for index in range(count))


def validate_compiled_model(
    mujoco: Any,
    model: Any,
    servo: ServoConfig = ServoConfig(),
    *,
    object_type: str = OBJECT_TYPE,
) -> None:
    runtime = object_runtime(object_type)
    joint_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    if joint_names[: len(JOINT_NAMES)] != JOINT_NAMES or joint_names[len(JOINT_NAMES) :] != (
        runtime.free_joint_name,
    ):
        raise ValueError(f"compiled joint order mismatch: {joint_names}")
    actuator_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    if actuator_names != JOINT_NAMES:
        raise ValueError(f"compiled actuator order mismatch: {actuator_names}")
    if model.nu != 26 or model.nq != 33 or model.nv != 32:
        raise ValueError(f"compiled dimensions mismatch: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    if not np.isclose(model.opt.timestep, PHYSICS_TIMESTEP):
        raise ValueError(f"compiled timestep mismatch: {model.opt.timestep}")
    for actuator_id, (name, effort, kp, dampratio) in enumerate(
        zip(JOINT_NAMES, EFFORT, servo.kp, servo.dampratio, strict=True)
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
            raise ValueError(f"compiled dampratio did not produce positive damping for {name}")
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.body_name)
    if object_id < 0:
        raise ValueError(f"compiled object body is absent: {runtime.body_name}")
    if model.body_gravcomp[object_id] != 0:
        raise ValueError(f"{runtime.object_type} gravity must remain active")
    object_geom_ids = [
        geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == object_id
    ]
    if len(object_geom_ids) != runtime.collision_geom_count:
        raise ValueError(
            f"expected {runtime.collision_geom_count} {runtime.object_type} collision geoms, "
            f"got {len(object_geom_ids)}"
        )
    if not np.all(model.geom_contype[object_geom_ids] == 2) or not np.all(
        model.geom_conaffinity[object_geom_ids] == 5
    ):
        raise ValueError("compiled object collision masks mismatch source configuration")
    hand_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] not in (0, object_id)
    ]
    if len(hand_geom_ids) != 16:
        raise ValueError(f"expected 16 hand collision geoms, got {len(hand_geom_ids)}")
    expected_hand_bits = (1, 7) if servo.hand_contacts_enabled else (0, 0)
    if not np.all(model.geom_contype[hand_geom_ids] == expected_hand_bits[0]) or not np.all(
        model.geom_conaffinity[hand_geom_ids] == expected_hand_bits[1]
    ):
        raise ValueError("compiled hand collision masks mismatch servo configuration")
    expected_exclude_signatures = {
        (min(first_id, second_id) << 16) + max(first_id, second_id)
        for body_names in HAND_SELF_COLLISION_GROUPS.values()
        for first_name, second_name in combinations(body_names, 2)
        for first_id, second_id in [
            (
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, first_name),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, second_name),
            )
        ]
    }
    actual_exclude_signatures = {int(value) for value in model.exclude_signature}
    if actual_exclude_signatures != expected_exclude_signatures:
        raise ValueError("compiled within-finger collision excludes mismatch source groups")
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name != runtime.body_name and model.body_gravcomp[body_id] != 1:
            raise ValueError(f"hand body gravity compensation missing: {name}")


def compile_model(
    servo: ServoConfig = ServoConfig(), *, object_type: str = OBJECT_TYPE
) -> tuple[Any, Any]:
    """Compile and validate one bounded native-servo scene."""

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("mujoco is required to compile the ManoRL scene") from exc
    model = mujoco.MjModel.from_xml_string(build_scene_xml(servo, object_type=object_type))
    validate_compiled_model(mujoco, model, servo, object_type=object_type)
    validate_static_fk(mujoco, model, object_type=object_type)
    return mujoco, model


def validate_unified_compiled_model(
    mujoco: Any,
    model: Any,
    servo: ServoConfig = ServoConfig(),
    *,
    object_types: Iterable[str],
) -> None:
    """Validate the fixed hand topology and every real object in a superset scene."""

    names = tuple(dict.fromkeys(object_types))
    if not names:
        raise ValueError("unified model validation requires at least one object type")
    runtimes = tuple(object_runtime(name) for name in names)
    joint_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    expected_joints = JOINT_NAMES + tuple(runtime.free_joint_name for runtime in runtimes)
    if joint_names != expected_joints:
        raise ValueError(f"unified joint order mismatch: {joint_names}")
    actuator_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    if actuator_names != JOINT_NAMES:
        raise ValueError(f"unified actuator order mismatch: {actuator_names}")
    expected_nq = 26 + 7 * len(runtimes)
    expected_nv = 26 + 6 * len(runtimes)
    if model.nu != 26 or model.nq != expected_nq or model.nv != expected_nv:
        raise ValueError(
            f"unified dimensions mismatch: nq={model.nq}, nv={model.nv}, nu={model.nu}; "
            f"expected nq={expected_nq}, nv={expected_nv}, nu=26"
        )
    if not np.isclose(model.opt.timestep, PHYSICS_TIMESTEP):
        raise ValueError(f"unified timestep mismatch: {model.opt.timestep}")
    for actuator_id, (name, effort, kp, dampratio) in enumerate(
        zip(JOINT_NAMES, EFFORT, servo.kp, servo.dampratio, strict=True)
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
            raise ValueError(f"compiled dampratio did not produce positive damping for {name}")
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
    ]
    if len(hand_geom_ids) != 16:
        raise ValueError(f"expected 16 hand collision geoms, got {len(hand_geom_ids)}")
    expected_hand_bits = (1, 7) if servo.hand_contacts_enabled else (0, 0)
    if not np.all(model.geom_contype[hand_geom_ids] == expected_hand_bits[0]) or not np.all(
        model.geom_conaffinity[hand_geom_ids] == expected_hand_bits[1]
    ):
        raise ValueError("compiled hand collision masks mismatch servo configuration")
    for runtime in runtimes:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.body_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, runtime.free_joint_name)
        if body_id < 0 or joint_id < 0:
            raise ValueError(f"compiled unified object is absent: {runtime.object_type}")
        if model.body_gravcomp[body_id] != 0:
            raise ValueError(f"unified object {runtime.object_type} must retain native gravity")
        object_geom_ids = [
            geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == body_id
        ]
        if len(object_geom_ids) != runtime.collision_geom_count:
            raise ValueError(
                f"expected {runtime.collision_geom_count} {runtime.object_type} collision geoms, "
                f"got {len(object_geom_ids)}"
            )
        if not np.all(model.geom_contype[object_geom_ids] == 2) or not np.all(
            model.geom_conaffinity[object_geom_ids] == 5
        ):
            raise ValueError(f"compiled {runtime.object_type} collision masks mismatch")
    validate_static_fk(mujoco, model, object_type=names[0])


def compile_unified_model(
    servo: ServoConfig = ServoConfig(), *, object_types: Iterable[str]
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
        build_unified_scene_xml(servo, object_types=names)
    )
    validate_unified_compiled_model(
        mujoco, model, servo, object_types=names
    )
    return mujoco, model


def _matrix_from_pose(position: tuple[float, ...], quaternion_wxyz: tuple[float, ...]) -> NDArray[np.float64]:
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


def urdf_zero_fk() -> dict[str, NDArray[np.float64]]:
    """Compute zero-joint link transforms independently from the MJCF builder."""

    root = ET.parse(HAND_URDF).getroot()
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
            transforms[child.get("link", "")] = transforms[parent_name] @ _matrix_from_pose(
                position, quaternion
            )
            remaining.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("hand URDF joint graph is disconnected or cyclic")
    return transforms


def validate_static_fk(
    mujoco: Any, model: Any, *, object_type: str = OBJECT_TYPE, atol: float = 1e-10
) -> None:
    """Check compiled body frames against independent zero-pose URDF FK."""

    data = mujoco.MjData(model)
    data.qpos[:26] = 0.0
    runtime = object_runtime(object_type)
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, runtime.free_joint_name
    )
    object_qpos_adr = model.jnt_qposadr[object_joint]
    data.qpos[object_qpos_adr + 3] = 1.0
    mujoco.mj_forward(model, data)
    for name, expected in urdf_zero_fk().items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"compiled hand body is absent: {name}")
        if not np.allclose(data.xpos[body_id], expected[:3, 3], atol=atol, rtol=0):
            raise ValueError(f"compiled static FK position mismatch at {name}")
        if not np.allclose(data.xmat[body_id].reshape(3, 3), expected[:3, :3], atol=atol, rtol=0):
            raise ValueError(f"compiled static FK orientation mismatch at {name}")


@lru_cache(maxsize=None)
def object_collision_vertices(object_type: str = OBJECT_TYPE) -> NDArray[np.float64]:
    """Return curated collision triangles in object-local metric coordinates."""

    runtime = object_runtime(object_type)
    validate_asset_manifest(object_type)
    path = runtime.collision_mesh_path
    if path.suffix.lower() == ".stl":
        payload = path.read_bytes()
        if len(payload) < 84:
            raise ValueError(f"{object_type} collision STL is too short")
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        expected_length = 84 + triangle_count * 50
        if len(payload) != expected_length:
            raise ValueError(f"{object_type} collision STL has an invalid binary length")
        triangles = np.frombuffer(
            payload,
            dtype=np.dtype(
                [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
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
                faces.append(tuple(int(value.split("/", maxsplit=1)[0]) - 1 for value in parts[1:4]))
        vertex_array = np.asarray(vertices, dtype=np.float64)
        face_array = np.asarray(faces, dtype=np.int64)
        if face_array.ndim != 2 or face_array.shape[1:] != (3,) or np.any(face_array < 0) or np.any(face_array >= len(vertex_array)):
            raise ValueError(f"{object_type} collision OBJ must contain indexed triangles")
        result = vertex_array[face_array].reshape(-1, 3)
    else:
        raise ValueError(f"unsupported collision mesh format for {object_type}: {path.suffix}")
    result = result * np.asarray(runtime.collision_mesh_scale, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 12 or not np.all(np.isfinite(result)):
        raise ValueError(f"unexpected {object_type} collision vertices: {result.shape}")
    result.setflags(write=False)
    return result
