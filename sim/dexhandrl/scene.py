"""DexHandRL MuJoCo scene generation."""

from __future__ import annotations

from itertools import combinations
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any

from sim.dexhandrl.asset_paths import hand_xml_path, object_urdf_path
from sim.dexhandrl.contracts import DexHandPhysicsConfig, format_friction, legacy_physics_config
from sim.dexhandrl.constants import (
    DEFAULT_ISAAC_SOURCE_ROOT,
    DEXHAND021PRO_CONTACT_BODY_NAMES,
    DEXHAND021PRO_FINGERTIP_BODY_NAMES,
)


_UNSET = object()

# These spellings are part of the frozen legacy XML oracle.  All non-legacy
# values use 17 significant digits so XML round-trips their floating values.
_LEGACY_FLOAT_TEXT = {
    0.0: "0", 0.0001: "0.0001", 0.001: "0.001", 0.006: "0.006", 0.01: "0.01",
    0.5: "0.5", 0.95: "0.95", 0.995: "0.995", 1.0: "1", 2.0: "2", 3.0: "3",
    9.81: "9.81", 15.0: "15", 50.0: "50", 80.0: "80", 100.0: "100",
    125.0: "125", 500.0: "500", 5000.0: "5000", 10000.0: "10000",
}


def _format_float(value: float) -> str:
    value = float(value)
    text = _LEGACY_FLOAT_TEXT.get(abs(value))
    if text is not None:
        return f"-{text}" if value < 0 else text
    return f"{value:.17g}"


def _format_numbers_legacy_compatible(values: tuple[float, ...]) -> str:
    return " ".join(_format_float(value) for value in values)


def _object_urdf_path(object_name: str, isaac_source_root: Path) -> Path:
    return object_urdf_path(object_name, isaac_source_root)


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


def _parse_mesh_geometry(
    *, urdf: Path, geometry_owner: ET.Element, description: str
) -> dict[str, object]:
    mesh = geometry_owner.find("geometry/mesh")
    if mesh is None or not mesh.get("filename"):
        raise ValueError(f"{description} must contain a mesh filename")
    path = (urdf.parent / mesh.get("filename", "")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} mesh not found: {path}")
    scale = _finite_numbers(mesh.get("scale"), 3, (1.0, 1.0, 1.0))
    position, quaternion = _parse_urdf_origin(geometry_owner.find("origin"))
    return {
        "mesh_path": path,
        "scale": scale,
        "position": position,
        "quaternion": quaternion,
    }


def _parse_object_asset(object_name: str, isaac_source_root: Path) -> dict[str, object]:
    """Parse one authoritative decomposed URDF without dropping collision pieces."""

    urdf = _object_urdf_path(object_name, isaac_source_root)
    root = ET.parse(urdf).getroot()
    links = root.findall("link")
    if len(links) != 1:
        raise ValueError(f"DexHandRL object {object_name!r} requires exactly one URDF link")
    link = links[0]
    visual_elements = link.findall("visual")
    collision_elements = link.findall("collision")
    if len(visual_elements) != 1:
        raise ValueError(f"DexHandRL object {object_name!r} requires exactly one visual mesh")
    if not collision_elements:
        raise ValueError(f"DexHandRL object {object_name!r} requires decomposed collision meshes")
    visual = _parse_mesh_geometry(
        urdf=urdf,
        geometry_owner=visual_elements[0],
        description=f"DexHandRL object {object_name!r} visual",
    )
    collisions = tuple(
        _parse_mesh_geometry(
            urdf=urdf,
            geometry_owner=collision,
            description=f"DexHandRL object {object_name!r} collision {index}",
        )
        for index, collision in enumerate(collision_elements)
    )
    inertial = link.find("inertial")
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
    return {
        "urdf_path": urdf.resolve(),
        "visual": visual,
        "collisions": collisions,
        "mass": mass,
        "inertia": inertia,
        "inertial_pos": inertial_pos,
        "inertial_quat": inertial_quat,
    }


def _replace_exactly_once(xml: str, pattern: str, replacement: str, description: str) -> str:
    result, count = re.subn(pattern, replacement, xml)
    if count != 1:
        raise RuntimeError(f"expected exactly one {description} transformation, found {count}")
    return result


def _replace_or_insert_option(xml: str, physics: DexHandPhysicsConfig) -> str:
    # Euler is MuJoCo's default; serializing it changes the frozen legacy XML bytes.
    gravity = _format_numbers_legacy_compatible(physics.gravity)
    option = (f'<option gravity="{gravity}" iterations="{physics.iterations}" '
              f'solver="{physics.solver}" timestep="{_format_float(physics.physics_dt)}" />')
    if "<option" in xml:
        return _replace_exactly_once(xml, r"<option[^>]*/>", option, "option")
    if xml.count("<compiler") != 1:
        raise RuntimeError("expected exactly one compiler tag when inserting option")
    return xml.replace("<compiler", f"{option}\n  <compiler", 1)


def _patch_hand_xml(
    xml: str,
    isaac_source_root: Path,
    dt: Any = _UNSET,
    substeps: Any = _UNSET,
    hand_friction: Any = _UNSET,
    *,
    physics: DexHandPhysicsConfig | None = None,
) -> str:
    """Patch the one canonical hand-geometry default without touching scene geoms."""
    legacy = {
        name: value
        for name, value in {"dt": dt, "substeps": substeps, "hand_friction": hand_friction}.items()
        if value is not _UNSET
    }
    if physics is not None and legacy:
        raise ValueError("physics cannot be combined with legacy hand XML overrides")
    hand_dir = hand_xml_path(isaac_source_root).parent
    mesh_dir = (hand_dir / "../marvin_021pro_urdf/meshes").resolve()
    physics = physics or legacy_physics_config(**legacy)
    xml = _replace_exactly_once(xml, r'meshdir="[^"]+"', f'meshdir="{mesh_dir.as_posix()}"', "mesh directory")
    canonical_hand_geom = r'(<geom\s+friction=")3 2 2("(?=\s+solref=))'
    try:
        xml = _replace_exactly_once(xml, canonical_hand_geom, rf'\g<1>{format_friction(physics.material.hand_friction)}\g<2>', "canonical hand friction")
    except RuntimeError as error:
        raise RuntimeError("expected exactly one canonical hand default geom friction replacement") from error
    xml = _replace_exactly_once(xml, r'<size[^>]*/>', f'<size njmax="{physics.model_njmax}" nconmax="{physics.model_nconmax}" />', "model capacity")
    xml = _replace_or_insert_option(xml, physics)
    return xml


def _object_asset_xml(object_name: str, asset: dict[str, object]) -> str:
    lines: list[str] = []
    for index, collision in enumerate(asset["collisions"]):
        mesh_name = f"{object_name}_mesh" if index == 0 else f"{object_name}_collision_mesh_{index}"
        lines.append(
            f'    <mesh name="{mesh_name}" file="{Path(collision["mesh_path"]).as_posix()}" '
            f'scale="{_format_numbers(collision["scale"])}" />\n'
        )
    visual = asset["visual"]
    lines.append(
        f'    <mesh name="{object_name}_visual_mesh" file="{Path(visual["mesh_path"]).as_posix()}" '
        f'scale="{_format_numbers(visual["scale"])}" />\n'
    )
    return "".join(lines)


def _object_contact_xml(physics: DexHandPhysicsConfig) -> tuple[str, str]:
    solref = _format_numbers_legacy_compatible(physics.material.object_solref)
    solimp_values = physics.material.object_solimp
    # The legacy three-field form expands to the same compiled five-field values.
    # Serialize tails only when a custom contract changes either tail value.
    solimp = _format_numbers_legacy_compatible(
        solimp_values[:3] if solimp_values[3:] == (0.5, 2.0) else solimp_values
    )
    return solref, solimp


def _object_body_xml(
    object_name: str, asset: dict[str, object], object_friction: str, physics: DexHandPhysicsConfig
) -> str:
    inertia = asset["inertia"]
    solref, solimp = _object_contact_xml(physics)
    inertial_pos = _format_numbers(asset["inertial_pos"])
    inertial_quat = _format_numbers(asset["inertial_quat"])
    inertial_orientation = "" if inertial_quat == "1 0 0 0" else f' quat="{inertial_quat}"'
    collision_lines: list[str] = []
    for index, collision in enumerate(asset["collisions"]):
        mesh_name = f"{object_name}_mesh" if index == 0 else f"{object_name}_collision_mesh_{index}"
        geom_name = f"{object_name}_collision" if index == 0 else f"{object_name}_piece_{index}_collision"
        collision_lines.append(
            f'      <geom name="{geom_name}" type="mesh" mesh="{mesh_name}" '
            f'pos="{_format_numbers(collision["position"])}" '
            f'quat="{_format_numbers(collision["quaternion"])}" contype="1" conaffinity="1" '
            f'condim="3" friction="{object_friction}" solref="{solref}" solimp="{solimp}" '
            f'rgba="0.80 0.18 0.16 1" group="3" />\n'
        )
    visual = asset["visual"]
    visual_line = (
        f'      <geom name="{object_name}_visual" type="mesh" mesh="{object_name}_visual_mesh" '
        f'pos="{_format_numbers(visual["position"])}" quat="{_format_numbers(visual["quaternion"])}" '
        f'contype="0" conaffinity="0" rgba="0.80 0.18 0.16 1" group="2" />\n'
    )
    return f'''
    <body name="{object_name}" pos="0 0 0">
      <freejoint name="{object_name}_free" />
      <inertial pos="{inertial_pos}"{inertial_orientation} mass="{asset["mass"]}" fullinertia="{inertia["ixx"]} {inertia["iyy"]} {inertia["izz"]} {inertia["ixy"]} {inertia["ixz"]} {inertia["iyz"]}" />
{''.join(collision_lines)}{visual_line}    </body>
'''


def _ground_xml(ground_friction: str) -> str:
    return f'    <geom name="ground" type="plane" size="3 3 0.05" pos="0 0 0" contype="1" conaffinity="1" friction="{ground_friction}" rgba="0.55 0.55 0.55 1" />\n'


def _hand_self_collision_exclude_pairs(mode: str = "disable_within_finger") -> set[tuple[str, str]]:
    valid_modes = {"none", "disable_within_finger", "all_disabled"}
    if mode not in valid_modes:
        raise ValueError(f"hand self-collision mode must be one of {sorted(valid_modes)}, got {mode!r}")
    if mode == "none":
        return set()
    finger_bodies = [f"RH{finger}_{link}" for finger in range(5) for link in range(4)]
    if mode == "all_disabled":
        return {tuple(sorted(pair)) for pair in combinations(["RFH1", *finger_bodies], 2)}
    excluded_pairs = set()
    for finger in range(5):
        bodies = [f"RH{finger}_{link}" for link in range(4)]
        excluded_pairs.update(tuple(sorted(pair)) for pair in combinations(bodies, 2))
        excluded_pairs.update(tuple(sorted(("RFH1", body))) for body in bodies[:2])
    return excluded_pairs


def _hand_self_collision_excludes_xml(mode: str = "disable_within_finger") -> str:
    excluded_pairs = _hand_self_collision_exclude_pairs(mode)
    if not excluded_pairs:
        return ""
    lines = ["  <contact>\n"]
    for body1, body2 in sorted(excluded_pairs):
        lines.append(f'    <exclude body1="{body1}" body2="{body2}" />\n')
    lines.append("  </contact>\n")
    return "".join(lines)


def _force_range(limit: float) -> str:
    value = abs(float(limit))
    return f"-{_format_float(value)} {_format_float(value)}"


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
    *,
    strict: bool = False,
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
        xml = re.sub(rf'(<position name="{name}"[^>]*?)kp="[^"]+"', rf'\1kp="{_format_float(kp)}"', xml, count=1)
        xml = re.sub(rf'(<position name="{name}"[^>]*?)kv="[^"]+"', rf'\1kv="{_format_float(kv)}"', xml, count=1)
        xml = re.sub(rf'(<position name="{name}"[^>]*?)forcerange="[^"]+"', rf'\1forcerange="{force_range}"', xml, count=1)
    xml = re.sub(
        r'(<position name="act_r_f_jiont_[^"]+"[^>]*?)kp="[^"]+"',
        rf'\1kp="{_format_float(finger_kp)}"',
        xml,
    )
    xml = re.sub(
        r'(<position name="act_r_f_jiont_[^"]+"(?![^>]*kv=)[^>]*?)\s*/>',
        rf'\1 kv="{_format_float(finger_kv)}" forcelimited="true" forcerange="{_force_range(finger_force)}" />',
        xml,
    )
    xml = re.sub(
        r'(<position name="act_r_f_jiont_[^"]+"[^>]*?)kv="[^"]+"',
        rf'\1kv="{_format_float(finger_kv)}"',
        xml,
    )
    xml = re.sub(
        r'(<position name="act_r_f_jiont_[^"]+"[^>]*?)forcerange="[^"]+"',
        rf'\1forcerange="{_force_range(finger_force)}"',
        xml,
    )
    if strict:
        expected = _ACTUATOR_NAMES
        actual = tuple(re.findall(r'<position name="([^"]+)"', xml))
        if actual != expected:
            raise RuntimeError("PD gain transformation changed actuator names/order")
        for name in expected:
            match = re.search(rf'<position name="{re.escape(name)}"[^>]*/>', xml)
            if match is None or 'kp="' not in match.group(0) or 'forcerange="' not in match.group(0):
                raise RuntimeError(f"PD gain transformation did not apply exactly once: {name}")
    return xml


def build_dexhandrl_scene_xml(
    output_path: Path,
    object_name: str,
    isaac_source_root: Path = DEFAULT_ISAAC_SOURCE_ROOT,
    *,
    physics: DexHandPhysicsConfig | None = None,
    dt: Any = _UNSET, substeps: Any = _UNSET,
    base_pos_kp: Any = _UNSET, base_pos_kv: Any = _UNSET, base_pos_force: Any = _UNSET,
    base_rot_kp: Any = _UNSET, base_rot_kv: Any = _UNSET, base_rot_force: Any = _UNSET,
    finger_kp: Any = _UNSET, finger_kv: Any = _UNSET, finger_force: Any = _UNSET,
    hand_self_collision_mode: Any = _UNSET,
    hand_friction: Any = _UNSET, object_friction: Any = _UNSET, ground_friction: Any = _UNSET,
) -> Path:
    """Create a MuJoCo XML with DexHand021Pro and one Lance object."""

    legacy = {name: value for name, value in {
        "dt": dt, "substeps": substeps, "base_pos_kp": base_pos_kp, "base_pos_kv": base_pos_kv,
        "base_pos_force": base_pos_force, "base_rot_kp": base_rot_kp, "base_rot_kv": base_rot_kv,
        "base_rot_force": base_rot_force, "finger_kp": finger_kp, "finger_kv": finger_kv,
        "finger_force": finger_force, "hand_self_collision_mode": hand_self_collision_mode,
        "hand_friction": hand_friction, "object_friction": object_friction, "ground_friction": ground_friction,
    }.items() if value is not _UNSET}
    if physics is not None and legacy:
        raise ValueError("physics cannot be combined with legacy scene overrides")
    config = physics or legacy_physics_config(**legacy)
    hand_xml_source = hand_xml_path(isaac_source_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    asset = _parse_object_asset(object_name=object_name, isaac_source_root=isaac_source_root)
    xml = hand_xml_source.read_text(encoding="utf-8")
    _validate_hand_source(xml)
    xml = _patch_hand_xml(
        xml, isaac_source_root=isaac_source_root, physics=config
    )
    xml = _patch_pd_gains(
        xml,
        base_pos_kp=config.servo.base_pos_kp,
        base_pos_kv=config.servo.base_pos_kv,
        base_pos_force=config.servo.base_pos_force,
        base_rot_kp=config.servo.base_rot_kp,
        base_rot_kv=config.servo.base_rot_kv,
        base_rot_force=config.servo.base_rot_force,
        finger_kp=config.servo.finger_kp,
        finger_kv=config.servo.finger_kv,
        finger_force=config.servo.finger_force,
        strict=True,
    )
    if xml.count("  </asset>") != 1 or xml.count("  </worldbody>") != 1 or xml.count("  <actuator>") != 1:
        raise RuntimeError("hand source has ambiguous scene insertion anchors")
    xml = xml.replace("  </asset>", _object_asset_xml(object_name, asset) + "  </asset>", 1)
    xml = xml.replace(
        "  </worldbody>",
        _ground_xml(format_friction(config.material.ground_friction)) + _object_body_xml(object_name, asset, format_friction(config.material.object_friction), config) + "  </worldbody>",
        1,
    )
    xml = xml.replace(
        "  <actuator>",
        _hand_self_collision_excludes_xml(config.hand_self_collision_mode) + "  <actuator>",
        1,
    )
    output_path.write_text(xml, encoding="utf-8")
    return output_path


_ACTUATOR_NAMES = (
    "act_ARTx", "act_ARTy", "act_ARTz", "act_ARRx", "act_ARRy", "act_ARRz",
    *(f"act_r_f_jiont_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)),
)
_JOINT_NAMES = tuple(name.removeprefix("act_") for name in _ACTUATOR_NAMES)


def _validate_hand_source(xml: str) -> None:
    """Guard the fragile legacy textual port against a changed hand asset."""
    names = tuple(re.findall(r'<position name="([^"]+)"', xml))
    if names != _ACTUATOR_NAMES:
        raise RuntimeError(f"hand source actuator order/names changed: {names}")
    for name in _ACTUATOR_NAMES:
        if not re.search(rf'<position name="{re.escape(name)}" joint="{re.escape(name.removeprefix("act_"))}"', xml):
            raise RuntimeError(f"hand source actuator transmission changed: {name}")
    required_bodies = ("RFH1", "right_hand_base", *(f"RH{finger}_{link}" for finger in range(5) for link in range(4)))
    missing = [name for name in required_bodies if f'name="{name}"' not in xml]
    if missing:
        raise RuntimeError(f"hand source required bodies absent: {missing}")


def _model_names(mujoco: Any, model: Any, obj: Any, count: int) -> tuple[str, ...]:
    return tuple(mujoco.mj_id2name(model, obj, index) or "" for index in range(count))


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError(f"compiled {label} mismatch: {actual!r} != {expected!r}")


def _compiled_exclude_pairs(mujoco: Any, model: Any) -> set[tuple[str, str]]:
    pairs = set()
    for signature in model.exclude_signature:
        # MuJoCo packs the lower and upper body ids in the two 16-bit halves.
        body1, body2 = int(signature) & 0xFFFF, int(signature) >> 16
        names = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2),
        )
        if None in names:
            raise ValueError(f"compiled exclude has unnamed body signature: {signature}")
        pairs.add(tuple(sorted(names)))
    return pairs


def validate_dexhandrl_compiled_model(
    mujoco: Any,
    model: Any,
    physics: DexHandPhysicsConfig = DexHandPhysicsConfig(),
    *,
    object_name: str = "cube1",
    isaac_source_root: Path = DEFAULT_ISAAC_SOURCE_ROOT,
) -> None:
    """Fail closed unless a compiled model realizes the DexHand legacy profile."""
    if model.nu != 26 or model.nq != 33 or model.nv != 32:
        raise ValueError(f"compiled dimensions mismatch: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    if model.njmax != physics.model_njmax or model.nconmax != physics.model_nconmax:
        raise ValueError("compiled model capacities mismatch")
    _close(model.opt.timestep, physics.physics_dt, "timestep")
    if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_EULER):
        raise ValueError("compiled integrator must be Euler")
    if int(model.opt.solver) != int(mujoco.mjtSolver.mjSOL_NEWTON) or int(model.opt.iterations) != physics.iterations:
        raise ValueError("compiled solver configuration mismatch")
    for actual, expected in zip(model.opt.gravity, physics.gravity, strict=True):
        _close(actual, expected, "gravity")
    joints = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    if joints != _JOINT_NAMES + (f"{object_name}_free",):
        raise ValueError(f"compiled joint order mismatch: {joints}")
    actuators = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    if actuators != _ACTUATOR_NAMES:
        raise ValueError(f"compiled actuator order mismatch: {actuators}")
    group_values = (
        *((physics.servo.base_pos_kp, physics.servo.base_pos_kv, physics.servo.base_pos_force) for _ in range(3)),
        *((physics.servo.base_rot_kp, physics.servo.base_rot_kv, physics.servo.base_rot_force) for _ in range(3)),
        *((physics.servo.finger_kp, physics.servo.finger_kv, physics.servo.finger_force) for _ in range(20)),
    )
    for actuator_id, (name, (kp, kv, force)) in enumerate(zip(_ACTUATOR_NAMES, group_values, strict=True)):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name.removeprefix("act_"))
        if joint_id < 0 or int(model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise ValueError(f"compiled actuator transmission mismatch: {name}")
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError(f"compiled actuator transmission type mismatch: {name}")
        _close(model.actuator_gainprm[actuator_id, 0], kp, f"actuator gain {name}")
        _close(model.actuator_biasprm[actuator_id, 1], -kp, f"actuator bias {name}")
        _close(model.actuator_biasprm[actuator_id, 2], -kv, f"actuator damping {name}")
        if not model.actuator_forcelimited[actuator_id]:
            raise ValueError(f"compiled actuator force limit missing: {name}")
        _close(model.actuator_forcerange[actuator_id, 0], -force, f"actuator lower force {name}")
        _close(model.actuator_forcerange[actuator_id, 1], force, f"actuator upper force {name}")
    ground_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if min(ground_geom, object_body) < 0:
        raise ValueError("compiled object, ground, or object body absent")
    source_asset = _parse_object_asset(object_name, isaac_source_root)
    object_geoms = [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == object_body
    ]
    expected_collision_names = [
        f"{object_name}_collision" if index == 0 else f"{object_name}_piece_{index}_collision"
        for index in range(len(source_asset["collisions"]))
    ]
    collision_geoms = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in expected_collision_names
    ]
    visual_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{object_name}_visual")
    if any(geom_id < 0 for geom_id in collision_geoms) or visual_geom < 0:
        raise ValueError("compiled decomposed object geoms are incomplete")
    if set(object_geoms) != {*collision_geoms, visual_geom}:
        raise ValueError("compiled object geom set differs from decomposed URDF")

    def friction(geom_id: int, expected: tuple[float, float, float], label: str) -> None:
        for actual, value in zip(model.geom_friction[geom_id], expected, strict=True):
            _close(actual, value, f"{label} friction")

    for object_geom in collision_geoms:
        if int(model.geom_contype[object_geom]) != 1 or int(model.geom_conaffinity[object_geom]) != 1:
            raise ValueError("compiled object collision mask mismatch")
        friction(object_geom, physics.material.object_friction, "object")
        for actual, value in zip(model.geom_solref[object_geom], physics.material.object_solref, strict=True):
            _close(actual, value, "object solref")
        for actual, value in zip(model.geom_solimp[object_geom], physics.material.object_solimp, strict=True):
            _close(actual, value, "object solimp")
    if int(model.geom_contype[visual_geom]) != 0 or int(model.geom_conaffinity[visual_geom]) != 0:
        raise ValueError("compiled object visual geom must be non-colliding")
    friction(ground_geom, physics.material.ground_friction, "ground")
    hand_geoms = [
        index for index in range(model.ngeom)
        if index != ground_geom and index not in object_geoms
    ]
    if len(hand_geoms) != 47:
        raise ValueError(f"compiled hand geom count mismatch: {len(hand_geoms)}")
    for geom_id in hand_geoms:
        friction(geom_id, physics.material.hand_friction, "hand")
    expected_excludes = _hand_self_collision_exclude_pairs(physics.hand_self_collision_mode)
    actual_excludes = _compiled_exclude_pairs(mujoco, model)
    if model.nexclude != len(expected_excludes) or actual_excludes != expected_excludes:
        raise ValueError("compiled hand self-collision excludes mismatch")
    required_bodies = (
        "right_hand_base", *DEXHAND021PRO_CONTACT_BODY_NAMES, *DEXHAND021PRO_FINGERTIP_BODY_NAMES, object_name
    )
    for name in (*required_bodies, f"{object_name}_free", "ground"):
        kind = mujoco.mjtObj.mjOBJ_JOINT if name.endswith("_free") else (mujoco.mjtObj.mjOBJ_GEOM if name == "ground" else mujoco.mjtObj.mjOBJ_BODY)
        if mujoco.mj_name2id(model, kind, name) < 0:
            raise ValueError(f"compiled required name absent: {name}")


def compile_dexhandrl_model(
    object_name: str = "cube1", physics: DexHandPhysicsConfig = DexHandPhysicsConfig(), *, isaac_source_root: Path = DEFAULT_ISAAC_SOURCE_ROOT
) -> tuple[Any, Any]:
    """Build, compile, and validate the canonical profile without importing ManoRL."""
    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("mujoco is required to compile DexHandRL") from error
    import tempfile
    with tempfile.TemporaryDirectory(prefix="dexhandrl_contract_") as directory:
        path = build_dexhandrl_scene_xml(Path(directory) / "scene.xml", object_name, isaac_source_root, physics=physics)
        model = mujoco.MjModel.from_xml_path(str(path))
    validate_dexhandrl_compiled_model(
        mujoco,
        model,
        physics,
        object_name=object_name,
        isaac_source_root=isaac_source_root,
    )
    return mujoco, model
