"""Ten-parent action005 strict-C1 large-pose augmentation contracts."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import uuid

import numpy as np

from sim.manorl.local_contact_repair import MODEL_FIELDS
from sim.manorl.mjx_sim import command_target
from sim.manorl.u1_campaign import array_sha, digest, file_sha, verify_signed

SETTING = "c6db552f105d4de493880a52518772dbacad9496b105d2f4e43e87a907e692cd"
VERSION = "u1-action005-ten-parent-largepose-v1-exact-discrete-c1"
PARENT_REGISTRY_SCHEMA = "u1-action005-ten-parent-registry-v1"
PARENT_SEMANTICS = (
    "mayonnaise bottle pickup, bowl-aligned pour beyond 90deg, upright return, "
    "world placement and release"
)

# Frozen optimal assignment of the existing global 160-slot plan to ten parent
# rows. Each parent receives 16 slots, exactly two from every azimuth sector,
# 5-6 from every radius, 2-3 from every signed rotation, and 1-2 extra draws.
PARENT_ROW_BY_SLOT = (
    9, 6, 7, 2, 9, 3, 2, 7, 8, 3, 9, 1, 0, 0, 9, 4, 2, 7, 1, 8,
    4, 8, 3, 2, 9, 9, 5, 5, 7, 3, 6, 8, 0, 1, 1, 4, 3, 4, 2, 0,
    8, 5, 0, 2, 4, 6, 6, 5, 5, 5, 1, 4, 0, 1, 6, 4, 0, 8, 6, 3,
    1, 6, 8, 3, 4, 9, 9, 1, 7, 7, 6, 0, 7, 2, 2, 1, 4, 6, 5, 3,
    7, 9, 2, 0, 5, 0, 9, 4, 3, 8, 8, 2, 1, 7, 8, 5, 3, 0, 6, 8,
    7, 4, 4, 1, 0, 2, 5, 7, 3, 7, 1, 5, 2, 6, 2, 9, 5, 0, 5, 4,
    4, 8, 8, 1, 0, 6, 7, 4, 5, 6, 3, 8, 2, 1, 6, 7, 7, 9, 0, 9,
    3, 3, 1, 9, 8, 2, 9, 5, 5, 8, 6, 3, 0, 3, 9, 2, 1, 6, 4, 7,
)


def validate_parent_assignment(plan: list[dict]) -> dict[int, list[int]]:
    """Prove the frozen ten-parent allocation retains the plan marginals."""

    if len(plan) != 160 or len(PARENT_ROW_BY_SLOT) != len(plan):
        raise ValueError("action005 requires the exact 160-slot large-pose plan")
    assigned = {row: [] for row in range(10)}
    for slot, (entry, row) in enumerate(zip(plan, PARENT_ROW_BY_SLOT, strict=True)):
        if entry.get("slot") != slot or row not in assigned:
            raise ValueError("invalid slot or parent assignment")
        assigned[row].append(slot)
    for row, slots in assigned.items():
        if len(slots) != 16:
            raise ValueError(f"parent row{row} does not receive exactly16 slots")
        radii = [
            sum(plan[slot]["radius_mm"] == value for slot in slots)
            for value in (50, 100, 150)
        ]
        sectors = [
            sum(plan[slot]["azimuth_sector"] == value for slot in slots)
            for value in range(8)
        ]
        rotations = [
            sum(
                plan[slot]["rotation_axis"] == axis
                and plan[slot]["rotation_sign"] == sign
                for slot in slots
            )
            for axis in range(3)
            for sign in (1, -1)
        ]
        extras = sum(plan[slot]["kind"] == "extra" for slot in slots)
        if not all(5 <= count <= 6 for count in radii):
            raise ValueError(f"parent row{row} radius allocation is imbalanced")
        if sectors != [2] * 8:
            raise ValueError(f"parent row{row} sector allocation is imbalanced")
        if not all(2 <= count <= 3 for count in rotations):
            raise ValueError(f"parent row{row} rotation allocation is imbalanced")
        if extras not in (1, 2):
            raise ValueError(f"parent row{row} extra allocation is imbalanced")
    return assigned


def action005_child_uuid(
    *, registry_sha256: str, plan_digest: str, slot: int, candidate: dict, parent_uuid: str
) -> str:
    payload = dict(
        version=VERSION,
        registry_sha256=registry_sha256,
        plan_digest=plan_digest,
        slot=slot,
        candidate=candidate,
        parent_uuid=parent_uuid,
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest(payload)))


def setting_fingerprint(model, object_names: tuple[str, ...], asset_manifest: Path) -> dict:
    """Reproduce the frozen U1 setting identity used by the published actions."""

    options = {}
    for name in dir(model.opt):
        if name.startswith("_"):
            continue
        value = getattr(model.opt, name)
        if isinstance(value, (int, float, np.number)):
            options[name] = float(value) if isinstance(value, (float, np.floating)) else int(value)
        elif isinstance(value, np.ndarray):
            options[name] = value.tolist()
    fields = (
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_gear",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "actuator_dyntype",
        "actuator_gaintype",
        "actuator_biastype",
    )
    hashes = {key: array_sha(getattr(model, key)) for key in fields}
    for key in (
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "jnt_range",
        "jnt_stiffness",
    ):
        hashes[key] = array_sha(getattr(model, key)[:28])
    hand_body_ids = [
        body_id
        for body_id in range(1, model.nbody)
        if model.body(body_id).name not in object_names
    ]
    for key in ("body_mass", "body_inertia", "body_gravcomp"):
        hashes["hand_" + key] = array_sha(getattr(model, key)[hand_body_ids])
    return {
        "physics_options": options,
        "hand_and_actuator_hashes": hashes,
        "control_hz": 120,
        "substeps": 4,
        "frame0": "prestep",
        "indexing": "arrival",
        "capacity": {"nconmax": 1024, "nccdmax": 256, "njmax": 4096},
        "asset_manifest_sha256": hashlib.sha256(asset_manifest.read_bytes()).hexdigest(),
    }


def require_setting(model, object_names: tuple[str, ...], asset_manifest: Path) -> dict:
    fingerprint = setting_fingerprint(model, object_names, asset_manifest)
    if digest(fingerprint) != SETTING:
        raise ValueError("action005 model does not match the frozen U1 setting")
    return fingerprint


def canonical_parent_target(source, model) -> np.ndarray:
    """Canonicalize inert frame0; preserve equivalent transition targets exactly."""

    target = np.asarray(source.target_qpos, dtype=np.float64).copy()
    if target.shape != (source.frames, 28) or not np.isfinite(target).all():
        raise ValueError("parent target must be finite [frames,28]")
    target[0] = command_target(
        target[0], source.recorded_qpos[0], *model.jnt_range[:28].T
    )
    for frame in range(1, source.frames):
        executed = command_target(
            target[frame], source.recorded_qpos[frame], *model.jnt_range[:28].T
        )
        difference = executed - target[frame]
        difference[3:6] = np.arctan2(
            np.sin(difference[3:6]), np.cos(difference[3:6])
        )
        if float(np.max(np.abs(difference))) >= 1e-6:
            raise ValueError(
                f"parent target frame{frame} is not native-canonical within1e-6"
            )
    return target


def source_parent_input(source, model) -> tuple[SimpleNamespace, dict[str, np.ndarray]]:
    """Build the qualification input and complete recorded teacher from one row."""

    names = tuple(source.scene_object_types)
    if names != ("mayonnaisebottle", "bowl") or source.object_type != "mayonnaisebottle":
        raise ValueError("action005 parent must contain ordered mayonnaisebottle,bowl")
    qpos0 = np.asarray(model.qpos0, dtype=np.float64).copy()
    qpos0[:28] = source.recorded_qpos[0]
    teacher_qpos = np.repeat(qpos0[None], source.frames, axis=0)
    teacher_qpos[:, :28] = source.recorded_qpos
    for index, name in enumerate(names):
        address = int(model.joint(name + "_free").qposadr[0])
        position = source.scene_object_position[index]
        quaternion = source.scene_object_quaternion_xyzw[index]
        qpos0[address : address + 3] = position[0]
        qpos0[address + 3 : address + 7] = quaternion[0, [3, 0, 1, 2]]
        teacher_qpos[:, address : address + 3] = position
        teacher_qpos[:, address + 3 : address + 7] = quaternion[:, [3, 0, 1, 2]]
    arrays = {
        "scene_object_names": np.asarray(names),
        "source_object_pos": np.transpose(source.scene_object_position, (1, 0, 2)),
        "source_object_quat_xyzw": np.transpose(
            source.scene_object_quaternion_xyzw, (1, 0, 2)
        ),
    }
    inp = SimpleNamespace(
        row_id=source.generated_uuid,
        frames=source.frames,
        hz=120,
        metrics={
            "source_metadata": {
                "active_object": source.object_type,
                "source_gesture": "005-full-pour",
            }
        },
        arrays=arrays,
        initial={"qpos": qpos0, "qvel": np.zeros(model.nv, dtype=np.float64)},
        manifest={
            "native": {
                field: np.asarray(getattr(model, field)).copy()
                for field in MODEL_FIELDS
            }
        },
        provenance={
            "dataset": str(source.dataset_path),
            "dataset_version": source.dataset_version,
            "row_index": source.row_index,
            "uuid": source.generated_uuid,
        },
    )
    return inp, {"qpos": teacher_qpos}


def action005_gate_values(
    *,
    tilt_deg: np.ndarray,
    bottle_position: np.ndarray,
    bowl_position: np.ndarray,
    bottle_velocity: np.ndarray,
    bowl_velocity: np.ndarray,
) -> tuple[dict[str, bool], dict[str, float | int]]:
    """Action-specific physical invariants shared by parents and children."""

    tilt = np.asarray(tilt_deg, dtype=np.float64)
    bottle = np.asarray(bottle_position, dtype=np.float64)
    bowl = np.asarray(bowl_position, dtype=np.float64)
    bottle_v = np.asarray(bottle_velocity, dtype=np.float64)
    bowl_v = np.asarray(bowl_velocity, dtype=np.float64)
    frames = len(tilt)
    if (
        tilt.shape != (frames,)
        or bottle.shape != (frames, 3)
        or bowl.shape != (frames, 3)
        or bottle_v.shape != (frames, 6)
        or bowl_v.shape != (frames, 6)
        or frames < 24
        or not all(np.isfinite(value).all() for value in (tilt, bottle, bowl, bottle_v, bowl_v))
    ):
        raise ValueError("invalid action005 gate arrays")
    peak = int(np.argmax(tilt))
    deep_frames = int(np.count_nonzero(tilt >= 90.0))
    peak_xy = float(np.linalg.norm(bottle[peak, :2] - bowl[peak, :2]))
    peak_height = float(bottle[peak, 2] - bowl[peak, 2])
    terminal_bottle_linear = float(
        np.max(np.linalg.norm(bottle_v[-24:, :3], axis=1))
    )
    terminal_bottle_angular = float(
        np.max(np.linalg.norm(bottle_v[-24:, 3:], axis=1))
    )
    terminal_bowl_linear = float(
        np.max(np.linalg.norm(bowl_v[-24:, :3], axis=1))
    )
    terminal_bowl_angular = float(
        np.max(np.linalg.norm(bowl_v[-24:, 3:], axis=1))
    )
    terminal_tilt = float(np.max(tilt[-24:]))
    metrics: dict[str, float | int] = {
        "max_world_tilt_deg": float(tilt[peak]),
        "peak_tilt_frame": peak,
        "frames_at_or_above_90deg": deep_frames,
        "peak_bottle_bowl_xy_m": peak_xy,
        "peak_bottle_height_above_bowl_m": peak_height,
        "terminal_200ms_max_tilt_deg": terminal_tilt,
        "terminal_200ms_max_bottle_linear_speed_mps": terminal_bottle_linear,
        "terminal_200ms_max_bottle_angular_speed_radps": terminal_bottle_angular,
        "terminal_200ms_max_bowl_linear_speed_mps": terminal_bowl_linear,
        "terminal_200ms_max_bowl_angular_speed_radps": terminal_bowl_angular,
    }
    gates = {
        "deep_pour_over_90deg": bool(tilt[peak] >= 90.0),
        "deep_pour_duration_100ms": bool(deep_frames >= 12),
        "peak_pour_over_bowl_xy": bool(peak_xy < 0.14),
        "peak_pour_above_bowl": bool(peak_height > 0.05),
        "returned_upright_terminal_200ms": bool(terminal_tilt < 10.0),
        "settled_terminal_bottle_linear": bool(terminal_bottle_linear < 0.002),
        "settled_terminal_bottle_angular": bool(terminal_bottle_angular < 0.02),
        "settled_terminal_bowl_linear": bool(terminal_bowl_linear < 0.002),
        "settled_terminal_bowl_angular": bool(terminal_bowl_angular < 0.02),
    }
    return gates, metrics


def action005_trace_gates(model, inp, trace_path: str | Path) -> tuple[dict, dict]:
    from scipy.spatial.transform import Rotation

    names = inp.arrays["scene_object_names"].tolist()
    if names != ["mayonnaisebottle", "bowl"]:
        raise ValueError("action005 trace scene order changed")
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos = np.asarray(trace["qpos"], dtype=np.float64)
        qvel = np.asarray(trace["qvel"], dtype=np.float64)
    bottle_q = int(model.joint("mayonnaisebottle_free").qposadr[0])
    bowl_q = int(model.joint("bowl_free").qposadr[0])
    bottle_v = int(model.joint("mayonnaisebottle_free").dofadr[0])
    bowl_v = int(model.joint("bowl_free").dofadr[0])
    bottle_quaternion = qpos[:, bottle_q + 3 : bottle_q + 7][:, [1, 2, 3, 0]]
    axis = Rotation.from_quat(bottle_quaternion).apply([0.0, 0.0, 1.0])
    tilt = np.rad2deg(np.arccos(np.clip(axis[:, 2], -1.0, 1.0)))
    return action005_gate_values(
        tilt_deg=tilt,
        bottle_position=qpos[:, bottle_q : bottle_q + 3],
        bowl_position=qpos[:, bowl_q : bowl_q + 3],
        bottle_velocity=qvel[:, bottle_v : bottle_v + 6],
        bowl_velocity=qvel[:, bowl_v : bowl_v + 6],
    )


def load_action005_parent(registry_path: str | Path, parent_row: int):
    import importlib.metadata
    import json
    import mujoco as mj

    registry_path = Path(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    verify_signed(registry)
    if registry.get("schema") != PARENT_REGISTRY_SCHEMA:
        raise ValueError("wrong action005 parent registry schema")
    if registry.get("setting_sha256") != SETTING:
        raise ValueError("wrong action005 U1 setting")
    if parent_row not in range(10):
        raise ValueError("parent row must be0..9")
    parent = registry["parents"][str(parent_row)]
    folder = registry_path.parent / parent["folder"]
    if parent.get("final_registry_status") != "accepted_for_named_semantics":
        raise ValueError("action005 parent is not accepted")
    for name, expected in parent["files"].items():
        if file_sha(folder / name) != expected:
            raise ValueError("action005 parent bundle file changed: " + name)
    for name, version in registry["runtime"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("runtime version changed: " + name)
    model = mj.MjModel.from_binary_path(str(folder / "model.mjb"))
    archive = np.load(folder / "input.npz", allow_pickle=False)
    inp = SimpleNamespace(
        row_id=parent["row_id"],
        frames=parent["frames"],
        hz=120,
        metrics=parent["metrics"],
        arrays={
            key: archive[key]
            for key in archive.files
            if not key.startswith("native_")
            and key not in ("qpos0", "qvel0", "teacher")
        },
        initial={"qpos": archive["qpos0"], "qvel": archive["qvel0"]},
        manifest={
            "native": {
                field: archive["native_" + field] for field in MODEL_FIELDS
            }
        },
    )
    return (
        registry,
        parent,
        inp,
        model,
        np.load(folder / "target.npy", allow_pickle=False),
        {"qpos": archive["teacher"]},
    )
