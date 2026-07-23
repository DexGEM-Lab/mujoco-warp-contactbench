"""DexHandRL MuJoCo scene generation."""

from __future__ import annotations

from itertools import combinations
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from sim.dexhandrl.constants import DEFAULT_ISAAC_SOURCE_ROOT


HAND_XML_RELATIVE = Path(
    "assets/all_assets/Assets/HAND/dexhand021pro/marvin_021pro_mujoco/"
    "marvin_021pro_right_hand_floating_simplified.xml"
)
OBJECT_URDF_RELATIVE = Path("assets/all_assets/Assets/sim/mano_objects_urdf")


def _object_urdf_path(object_name: str, isaac_source_root: Path) -> Path:
    path = isaac_source_root / OBJECT_URDF_RELATIVE / f"{object_name}.urdf"
    if not path.exists():
        raise FileNotFoundError(f"object URDF not found: {path}")
    return path


def _finite_numbers(value: str | None, count: int, default: tuple[float, ...]) -> tuple[float, ...]:
    values = default if value is None else tuple(float(item) for item in value.split())
    if len(values) != count or not all(math.isfinite(item) for item in values):
        raise ValueError(f"expected {count} finite values, got {value!r}")
    return values


def _rpy_to_wxyz(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Convert URDF fixed-axis roll, pitch, yaw to a MuJoCo wxyz quaternion."""
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


def _parse_urdf_origin(origin: ET.Element | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if origin is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    xyz = _finite_numbers(origin.get("xyz"), 3, (0.0, 0.0, 0.0))
    rpy = _finite_numbers(origin.get("rpy"), 3, (0.0, 0.0, 0.0))
    return xyz, _rpy_to_wxyz(rpy)


def _parse_finite_float(value: str | None, description: str) -> float:
    try:
        result = float(value) if value is not None else float("nan")
    except ValueError as error:
        raise ValueError(f"invalid {description}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"invalid {description}: {value!r}")
    return result


def _format_numbers(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.17g}" for value in values)


def _parse_object_asset(object_name: str, isaac_source_root: Path) -> dict[str, object]:
    urdf = _object_urdf_path(object_name, isaac_source_root)
    root = ET.parse(urdf).getroot()
    collision = root.find(".//collision")
    mesh = None if collision is None else collision.find("geometry/mesh")
    if mesh is None:
        visual = root.find(".//visual")
        mesh = None if visual is None else visual.find("geometry/mesh")
    if mesh is None:
        raise RuntimeError(f"DexHandRL object {object_name!r} currently expects mesh URDF geometry")
    if "filename" not in mesh.attrib:
        raise ValueError(f"DexHandRL object {object_name!r} mesh has no filename")
    mesh_path = (urdf.parent / mesh.attrib["filename"]).resolve()
    scale = mesh.attrib.get("scale", "1 1 1")
    inertial = root.find(".//inertial")
    mass_elem = None if inertial is None else inertial.find("mass")
    inertia_elem = None if inertial is None else inertial.find("inertia")
    if mass_elem is None or inertia_elem is None:
        raise ValueError(f"DexHandRL object {object_name!r} has incomplete inertial data")
    mass = _parse_finite_float(mass_elem.get("value"), "inertial mass")
    inertia = {
        key: _parse_finite_float(inertia_elem.get(key), f"inertia {key}")
        for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    }
    inertial_pos, inertial_quat = _parse_urdf_origin(inertial.find("origin"))
    collision_pos, collision_quat = _parse_urdf_origin(None if collision is None else collision.find("origin"))
    return {
        "mesh_path": mesh_path,
        "scale": scale,
        "mass": mass,
        "inertia": inertia,
        "inertial_pos": inertial_pos,
        "inertial_quat": inertial_quat,
        "collision_pos": collision_pos,
        "collision_quat": collision_quat,
    }


def _replace_or_insert_option(xml: str, dt: float, substeps: int) -> str:
    timestep = dt / max(int(substeps), 1)
    option = f'<option gravity="0 0 -9.81" iterations="80" solver="Newton" timestep="{timestep}" />'
    if "<option" in xml:
        return re.sub(r"<option[^>]*/>", option, xml, count=1)
    return xml.replace("<compiler", f"{option}\n  <compiler", 1)


def _patch_hand_xml(
    xml: str, isaac_source_root: Path, dt: float, substeps: int, hand_friction: str = "0.5 0.01 0.001"
) -> str:
    """Patch the one canonical hand-geometry default without touching scene geoms."""
    hand_dir = (isaac_source_root / HAND_XML_RELATIVE).parent
    mesh_dir = (hand_dir / "../marvin_021pro_urdf/meshes").resolve()
    xml = re.sub(r'meshdir="[^"]+"', f'meshdir="{mesh_dir.as_posix()}"', xml, count=1)
    canonical_hand_geom = r'(<geom\s+friction=")3 2 2("(?=\s+solref=))'
    xml, replacements = re.subn(canonical_hand_geom, rf'\g<1>{hand_friction}\g<2>', xml)
    if replacements != 1:
        raise RuntimeError(
            "expected exactly one canonical hand default geom friction='3 2 2' replacement, "
            f"found {replacements}"
        )
    xml = re.sub(r'<size[^>]*/>', '<size njmax="2048" nconmax="8192" />', xml, count=1)
    xml = _replace_or_insert_option(xml, dt=dt, substeps=substeps)
    return xml


def _object_asset_xml(object_name: str, asset: dict[str, object]) -> str:
    return (
        f'    <mesh name="{object_name}_mesh" file="{Path(asset["mesh_path"]).as_posix()}" '
        f'scale="{asset["scale"]}" />\n'
    )


def _object_body_xml(object_name: str, asset: dict[str, object], object_friction: str) -> str:
    inertia = asset["inertia"]
    inertial_pos = _format_numbers(asset["inertial_pos"])
    inertial_quat = _format_numbers(asset["inertial_quat"])
    inertial_orientation = "" if inertial_quat == "1 0 0 0" else f' quat="{inertial_quat}"'
    collision_pos = _format_numbers(asset["collision_pos"])
    collision_quat = _format_numbers(asset["collision_quat"])
    return f'''
    <body name="{object_name}" pos="0 0 0">
      <freejoint name="{object_name}_free" />
      <inertial pos="{inertial_pos}"{inertial_orientation} mass="{asset["mass"]}" fullinertia="{inertia["ixx"]} {inertia["iyy"]} {inertia["izz"]} {inertia["ixy"]} {inertia["ixz"]} {inertia["iyz"]}" />
      <geom name="{object_name}_collision" type="mesh" mesh="{object_name}_mesh" pos="{collision_pos}" quat="{collision_quat}" contype="1" conaffinity="1" condim="3" friction="{object_friction}" solref="0.006 1" solimp="0.95 0.995 0.0001" rgba="0.80 0.18 0.16 1" />
    </body>
'''


def _ground_xml(ground_friction: str) -> str:
    return f'    <geom name="ground" type="plane" size="3 3 0.05" pos="0 0 0" contype="1" conaffinity="1" friction="{ground_friction}" rgba="0.55 0.55 0.55 1" />\n'


def _hand_self_collision_excludes_xml(mode: str = "disable_within_finger") -> str:
    valid_modes = {"none", "disable_within_finger", "all_disabled"}
    if mode not in valid_modes:
        raise ValueError(f"hand self-collision mode must be one of {sorted(valid_modes)}, got {mode!r}")
    if mode == "none":
        return ""

    excluded_pairs: set[tuple[str, str]] = set()
    finger_bodies = [f"RH{finger}_{link}" for finger in range(5) for link in range(4)]
    if mode == "all_disabled":
        excluded_pairs.update(tuple(sorted(pair)) for pair in combinations(["RFH1", *finger_bodies], 2))
    else:
        for finger in range(5):
            bodies = [f"RH{finger}_{link}" for link in range(4)]
            excluded_pairs.update(tuple(sorted(pair)) for pair in combinations(bodies, 2))
            for proximal_body in bodies[:2]:
                excluded_pairs.add(tuple(sorted(("RFH1", proximal_body))))
    lines = ["  <contact>\n"]
    for body1, body2 in sorted(excluded_pairs):
        lines.append(f'    <exclude body1="{body1}" body2="{body2}" />\n')
    lines.append("  </contact>\n")
    return "".join(lines)


def _force_range(limit: float) -> str:
    value = abs(float(limit))
    return f"-{value:g} {value:g}"


def _patch_pd_gains(
    xml: str,
    base_pos_kp: float,
    base_pos_kv: float,
    base_pos_force: float,
    base_rot_kp: float,
    base_rot_kv: float,
    base_rot_force: float,
    finger_kp: float,
    finger_kv: float,
    finger_force: float,
) -> str:
    replacements = {
        "act_ARTx": (base_pos_kp, base_pos_kv, _force_range(base_pos_force)),
        "act_ARTy": (base_pos_kp, base_pos_kv, _force_range(base_pos_force)),
        "act_ARTz": (base_pos_kp, base_pos_kv, _force_range(base_pos_force)),
        "act_ARRx": (base_rot_kp, base_rot_kv, _force_range(base_rot_force)),
        "act_ARRy": (base_rot_kp, base_rot_kv, _force_range(base_rot_force)),
        "act_ARRz": (base_rot_kp, base_rot_kv, _force_range(base_rot_force)),
    }
    for name, (kp, kv, force_range) in replacements.items():
        xml = re.sub(rf'(<position name="{name}"[^>]*?)kp="[^"]+"', rf'\1kp="{kp:g}"', xml, count=1)
        xml = re.sub(rf'(<position name="{name}"[^>]*?)kv="[^"]+"', rf'\1kv="{kv:g}"', xml, count=1)
        xml = re.sub(rf'(<position name="{name}"[^>]*?)forcerange="[^"]+"', rf'\1forcerange="{force_range}"', xml, count=1)
    xml = re.sub(
        r'(<position name="act_r_f_jiont_[^"]+"[^>]*?)kp="[^"]+"',
        rf'\1kp="{finger_kp:g}"',
        xml,
    )
    xml = re.sub(
        r'(<position name="act_r_f_jiont_[^"]+"(?![^>]*kv=)[^>]*?)\s*/>',
        rf'\1 kv="{finger_kv:g}" forcelimited="true" forcerange="{_force_range(finger_force)}" />',
        xml,
    )
    xml = re.sub(
        r'(<position name="act_r_f_jiont_[^"]+"[^>]*?)kv="[^"]+"',
        rf'\1kv="{finger_kv:g}"',
        xml,
    )
    xml = re.sub(
        r'(<position name="act_r_f_jiont_[^"]+"[^>]*?)forcerange="[^"]+"',
        rf'\1forcerange="{_force_range(finger_force)}"',
        xml,
    )
    return xml


def build_dexhandrl_scene_xml(
    output_path: Path,
    object_name: str,
    isaac_source_root: Path = DEFAULT_ISAAC_SOURCE_ROOT,
    dt: float = 0.01,
    substeps: int = 10,
    base_pos_kp: float = 5000,
    base_pos_kv: float = 500,
    base_pos_force: float = 500,
    base_rot_kp: float = 10000,
    base_rot_kv: float = 125,
    base_rot_force: float = 100,
    finger_kp: float = 15,
    finger_kv: float = 0,
    finger_force: float = 50,
    hand_self_collision_mode: str = "disable_within_finger",
    hand_friction: str = "0.5 0.01 0.001",
    object_friction: str = "0.5 0.01 0.001",
    ground_friction: str = "0.5 0.5 0",
) -> Path:
    """Create a MuJoCo XML with DexHand021Pro and one Lance object."""

    hand_xml_path = isaac_source_root / HAND_XML_RELATIVE
    if not hand_xml_path.exists():
        raise FileNotFoundError(f"DexHand021Pro hand XML not found: {hand_xml_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    asset = _parse_object_asset(object_name=object_name, isaac_source_root=isaac_source_root)
    xml = hand_xml_path.read_text(encoding="utf-8")
    xml = _patch_hand_xml(
        xml, isaac_source_root=isaac_source_root, dt=dt, substeps=substeps, hand_friction=hand_friction
    )
    xml = _patch_pd_gains(
        xml,
        base_pos_kp=base_pos_kp,
        base_pos_kv=base_pos_kv,
        base_pos_force=base_pos_force,
        base_rot_kp=base_rot_kp,
        base_rot_kv=base_rot_kv,
        base_rot_force=base_rot_force,
        finger_kp=finger_kp,
        finger_kv=finger_kv,
        finger_force=finger_force,
    )
    xml = xml.replace("  </asset>", _object_asset_xml(object_name, asset) + "  </asset>", 1)
    xml = xml.replace(
        "  </worldbody>",
        _ground_xml(ground_friction) + _object_body_xml(object_name, asset, object_friction) + "  </worldbody>",
        1,
    )
    xml = xml.replace(
        "  <actuator>",
        _hand_self_collision_excludes_xml(hand_self_collision_mode) + "  <actuator>",
        1,
    )
    output_path.write_text(xml, encoding="utf-8")
    return output_path
