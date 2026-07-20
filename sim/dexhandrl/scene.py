"""DexHandRL MuJoCo scene generation."""

from __future__ import annotations

from itertools import combinations
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


def _parse_object_asset(object_name: str, isaac_source_root: Path) -> dict[str, object]:
    urdf = _object_urdf_path(object_name, isaac_source_root)
    root = ET.parse(urdf).getroot()
    mesh = root.find(".//collision/geometry/mesh") or root.find(".//visual/geometry/mesh")
    if mesh is None:
        raise RuntimeError(f"DexHandRL object {object_name!r} currently expects mesh URDF geometry")
    mesh_path = (urdf.parent / mesh.attrib["filename"]).resolve()
    scale = mesh.attrib.get("scale", "1 1 1")
    mass_elem = root.find(".//inertial/mass")
    inertia_elem = root.find(".//inertial/inertia")
    mass = float(mass_elem.attrib.get("value", "0.1")) if mass_elem is not None else 0.1
    if inertia_elem is not None:
        inertia = {
            key: float(inertia_elem.attrib.get(key, "0"))
            for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        }
    else:
        inertia = {"ixx": 1e-4, "iyy": 1e-4, "izz": 1e-4, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0}
    return {"mesh_path": mesh_path, "scale": scale, "mass": mass, "inertia": inertia}


def _replace_or_insert_option(xml: str, dt: float, substeps: int) -> str:
    timestep = dt / max(int(substeps), 1)
    option = f'<option gravity="0 0 -9.81" iterations="80" solver="Newton" timestep="{timestep}" />'
    if "<option" in xml:
        return re.sub(r"<option[^>]*/>", option, xml, count=1)
    return xml.replace("<compiler", f"{option}\n  <compiler", 1)


def _patch_hand_xml(xml: str, isaac_source_root: Path, dt: float, substeps: int) -> str:
    hand_dir = (isaac_source_root / HAND_XML_RELATIVE).parent
    mesh_dir = (hand_dir / "../marvin_021pro_urdf/meshes").resolve()
    xml = re.sub(r'meshdir="[^"]+"', f'meshdir="{mesh_dir.as_posix()}"', xml, count=1)
    xml = re.sub(r'<geom friction="[^"]+" solref=', '<geom friction="0.5 0.01 0.001" solref=', xml, count=1)
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
    return f'''
    <body name="{object_name}" pos="0 0 0">
      <freejoint name="{object_name}_free" />
      <inertial pos="0 0 0" mass="{asset["mass"]}" fullinertia="{inertia["ixx"]} {inertia["iyy"]} {inertia["izz"]} {inertia["ixy"]} {inertia["ixz"]} {inertia["iyz"]}" />
      <geom name="{object_name}_collision" type="mesh" mesh="{object_name}_mesh" contype="1" conaffinity="1" condim="3" friction="{object_friction}" solref="0.006 1" solimp="0.95 0.995 0.0001" rgba="0.80 0.18 0.16 1" />
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
    xml = _patch_hand_xml(xml, isaac_source_root=isaac_source_root, dt=dt, substeps=substeps)
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
