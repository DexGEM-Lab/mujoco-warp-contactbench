"""Deterministic MuJoCo scene construction from curated authoritative URDF assets."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
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
    OBJECT_BODY_NAME,
    OBJECT_COLLISION_GEOM_COUNT,
    OBJECT_FREE_JOINT_NAME,
    OBJECT_LINK_NAME,
    PHYSICS_TIMESTEP,
    ServoConfig,
)

ASSET_ROOT = Path(__file__).resolve().parent / "runtime_assets"
HAND_URDF = ASSET_ROOT / "hand" / "mano_hand.urdf"
OBJECT_URDF = ASSET_ROOT / "cube1" / "cube1.urdf"
ASSET_MANIFEST = ASSET_ROOT / "manifest.json"
OBJECT_MESH = ASSET_ROOT / "cube1" / "cube1_aligned.stl"
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


def _required_paths() -> tuple[Path, ...]:
    hand_meshes = tuple((ASSET_ROOT / "hand" / "meshes").glob("*.stl"))
    if len(hand_meshes) != 16:
        raise FileNotFoundError(f"expected exactly 16 curated hand collision meshes, got {len(hand_meshes)}")
    paths = (HAND_URDF, OBJECT_URDF, *hand_meshes, OBJECT_MESH, ASSET_MANIFEST)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"curated ManoRL assets are incomplete: {missing}")
    return tuple(paths)


def validate_asset_manifest() -> dict[str, Any]:
    """Verify every curated file against its committed provenance digest."""

    _required_paths()
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("source_commit") != "ead79126589d1abf2362ea30b9d674d9e675a2f9":
        raise ValueError("asset manifest does not name the settled all_assets commit")
    for entry in manifest.get("files", []):
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


def _object_body(worldbody: ET.Element, asset: ET.Element, urdf_root: ET.Element) -> None:
    object_link = urdf_root.find("link")
    if object_link is None or object_link.get("name") != OBJECT_LINK_NAME:
        raise ValueError(f"cube URDF must contain {OBJECT_LINK_NAME}")
    collisions = object_link.findall("collision")
    if len(collisions) != OBJECT_COLLISION_GEOM_COUNT:
        raise ValueError("cube URDF must contain exactly one collision mesh")
    collision = collisions[0]
    source_mesh = collision.find("geometry/mesh")
    if source_mesh is None or Path(source_mesh.get("filename", "")).name != "cube1.obj":
        raise ValueError("cube URDF collision must reference cube1.obj")
    mesh_scale = _numbers(source_mesh.get("scale"), 3, (1.0, 1.0, 1.0))
    if mesh_scale != (0.001, 0.001, 0.001):
        raise ValueError("cube URDF collision scale must remain 0.001")
    ET.SubElement(
        asset,
        "mesh",
        name="cube1_mesh",
        file=str(OBJECT_MESH.resolve()),
        scale=_format(mesh_scale),
    )

    body = ET.SubElement(worldbody, "body", name=OBJECT_BODY_NAME, gravcomp="0")
    ET.SubElement(body, "freejoint", name=OBJECT_FREE_JOINT_NAME)
    _link_inertial(body, object_link)
    position, quaternion = _origin(collision.find("origin"))
    ET.SubElement(
        body,
        "geom",
        name="cube1_collision",
        type="mesh",
        mesh="cube1_mesh",
        pos=_format(position),
        quat=_format(quaternion),
        rgba="0.8 0.18 0.16 1",
        contype="2",
        conaffinity="5",
        condim="3",
        friction="0.9 0.01 0.001",
    )


def build_scene_xml(servo: ServoConfig = ServoConfig()) -> str:
    """Build the curated scene for one bounded native-servo configuration."""

    validate_asset_manifest()
    hand_root = ET.parse(HAND_URDF).getroot()
    object_root = ET.parse(OBJECT_URDF).getroot()
    root = ET.Element("mujoco", model="manorl_cube1_reference")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep=str(PHYSICS_TIMESTEP),
        gravity="0 0 -9.81",
        integrator="implicitfast",
    )
    asset = ET.SubElement(root, "asset")
    # Visual-only skybox: it supplies a readable horizon in MuJoCo viewers and
    # is not referenced by any physical geom, contact, or actuator.
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
        rgba="0.58 0.61 0.64 1",
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
    _object_body(worldbody, asset, object_root)

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
    mujoco: Any, model: Any, servo: ServoConfig = ServoConfig()
) -> None:
    joint_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    if joint_names[: len(JOINT_NAMES)] != JOINT_NAMES or joint_names[len(JOINT_NAMES) :] != (
        OBJECT_FREE_JOINT_NAME,
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
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
    if model.body_gravcomp[object_id] != 0:
        raise ValueError("cube gravity must remain active")
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
        if name != OBJECT_BODY_NAME and model.body_gravcomp[body_id] != 1:
            raise ValueError(f"hand body gravity compensation missing: {name}")


def compile_model(servo: ServoConfig = ServoConfig()) -> tuple[Any, Any]:
    """Compile and validate one bounded native-servo scene."""

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("mujoco is required to compile the ManoRL scene") from exc
    model = mujoco.MjModel.from_xml_string(build_scene_xml(servo))
    validate_compiled_model(mujoco, model, servo)
    validate_static_fk(mujoco, model)
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


def validate_static_fk(mujoco: Any, model: Any, *, atol: float = 1e-10) -> None:
    """Check compiled body frames against independent zero-pose URDF FK."""

    data = mujoco.MjData(model)
    data.qpos[:26] = 0.0
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_FREE_JOINT_NAME)
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


@lru_cache(maxsize=1)
def object_collision_vertices() -> NDArray[np.float64]:
    """Return authoritative cube collision vertices in object coordinates."""

    validate_asset_manifest()
    payload = OBJECT_MESH.read_bytes()
    if len(payload) < 84:
        raise ValueError("cube collision STL is too short")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    expected_length = 84 + triangle_count * 50
    if len(payload) != expected_length:
        raise ValueError("cube collision STL has an invalid binary length")
    triangles = np.frombuffer(
        payload,
        dtype=np.dtype(
            [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
        ),
        count=triangle_count,
        offset=84,
    )
    result = np.asarray(triangles["vertices"], dtype=np.float64).reshape(-1, 3) * 0.001
    if result.shape != (36, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"unexpected cube collision vertices: {result.shape}")
    result.setflags(write=False)
    return result
