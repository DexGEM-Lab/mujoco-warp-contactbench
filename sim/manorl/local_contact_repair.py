"""Time-preserving, free-object local repair of archived physical hand replays.

Inputs are archived commands and initialization, never measured hand qpos as a
new control track. Finger corrections act on actuator targets after the existing
feedforward/preload blend; wrist corrections act on the desired wrist pose.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

CHEY_MANIFEST_SHA = "e686d91931979c444c923d4f962ddd135ce8204e0de9df6fd7368bfdb855e29b"
ASSET_COMMIT = "778614d09e917deffed0bff3f357aa237efa762d"
MODEL_FIELDS = (
    "jnt_range", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange",
    "actuator_forcerange", "dof_frictionloss", "dof_damping", "dof_armature",
    "body_mass", "body_inertia", "body_ipos", "body_iquat", "geom_size",
    "geom_pos", "geom_quat", "geom_friction", "geom_solref", "geom_solimp",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def interpolate(values: np.ndarray, old_time: np.ndarray, new_time: np.ndarray,
                *, angles: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite source values")
    if angles:
        values = np.unwrap(values, axis=0, period=2 * np.pi)
    flat = values.reshape(len(values), -1)
    result = np.stack([np.interp(new_time, old_time, col) for col in flat.T], axis=1)
    return result.reshape((len(new_time),) + values.shape[1:])


def resample_hand(values: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    return np.concatenate((interpolate(values[:, :3], old, new),
                           interpolate(values[:, 3:], old, new, angles=True)), axis=1)


@dataclass
class ReplayInput:
    row_id: str
    hz: int
    arrays: dict[str, np.ndarray]
    initial: dict[str, np.ndarray]
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def frames(self) -> int:
        return len(self.arrays["desired"])


def load_input(root: Path, row_id: str, hz: int = 120) -> ReplayInput:
    if hz not in (100, 120):
        raise ValueError("only coupled100/400 and120/480 clocks are supported")
    comparison = json.loads((root / "comparison.json").read_text())
    row = next(x for x in comparison["rows"] if x["id"] == row_id)
    folder = root / row_id
    trace_path = folder / "replay/trace.npz"
    if sha(trace_path) != row["cheyingtong"]["trace_sha256"]:
        raise ValueError("archived trace does not match the baseline comparison")
    manifest = json.loads((folder / "manifest.json").read_text())
    metrics = json.loads((folder / "replay/metrics.json").read_text())
    if manifest["asset_provenance"]["asset_manifest_sha256"] != CHEY_MANIFEST_SHA:
        raise ValueError("input is not the pinned Cheyingtong baseline")
    if manifest["frame_zero_integrated"]:
        raise ValueError("this repair contract requires a prestep frame0")
    if metrics["clock"]["nominal_hz"] != 100:
        raise ValueError("archive is not100Hz")
    with np.load(trace_path, allow_pickle=False) as z:
        source = {k: z[k] for k in z.files}
    with np.load(folder / "initial.npz", allow_pickle=False) as z:
        initial = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(initial["qpos"], source["qpos"][0])
    np.testing.assert_array_equal(initial["desired"], source["desired"])
    np.testing.assert_array_equal(initial["finger_offset_envelope"], source["finger_offset_envelope"])
    old_time = np.arange(len(source["desired"]), dtype=float) / 100
    # End on a regular control/physics boundary, with <= one-frame terminal hold.
    frames = int(np.ceil(old_time[-1] * hz - 1e-10)) + 1
    time = np.arange(frames, dtype=float) / hz
    sample_time = np.minimum(time, old_time[-1])
    out = {"time": time, "source_frame": sample_time * 100}
    for key in ("desired", "source_hand", "reference_hand"):
        out[key] = resample_hand(source[key], old_time, sample_time)
    for key in ("source_timestamp", "source_object_pos", "finger_offset_envelope"):
        out[key] = interpolate(source[key], old_time, sample_time)
    out["source_object_quat_xyzw"] = np.stack([
        Slerp(old_time, Rotation.from_quat(source["source_object_quat_xyzw"][:, i]))(sample_time).as_quat()
        for i in range(len(source["scene_object_names"]))], axis=1)
    for key in ("scene_object_names", "scene_object_qpos_addresses", "source_ground_translation"):
        out[key] = source[key].copy()
    if len(initial["grip_envelope"]):
        out["grip_envelope"] = interpolate(initial["grip_envelope"], old_time, sample_time)
    else:
        out["grip_envelope"] = np.zeros(frames)
    provenance = dict(
        source_hand_profile="sunke", replay_hand_profile="cheyingtong", source_hz=100,
        control_hz=hz, physics_hz=4 * hz, physics_substeps=4,
        original_duration_s=float(old_time[-1]), duration_s=float(time[-1]),
        terminal_grid_hold_s=float(time[-1] - old_time[-1]), frame_zero_integrated=False,
        source=manifest["source"], baseline_trace_sha256=sha(trace_path),
        baseline_initial_sha256=sha(folder / "initial.npz"),
        baseline_manifest_sha256=sha(folder / "manifest.json"),
        baseline_comparison_sha256=sha(root / "comparison.json"),
        baseline_physical_pose_pass=row["cheyingtong"]["physical_pose"],
        asset_provenance=manifest["asset_provenance"],
    )
    return ReplayInput(row_id, hz, out, initial, manifest, metrics, provenance)


def smooth_envelope(time: np.ndarray, phase: dict[str, Any]) -> np.ndarray:
    allowed = {"start_s", "rise_s", "end_s", "fall_s", "finger_deg", "translation_m",
               "rotation_deg", "rotation_frame", "note"}
    if set(phase) - allowed:
        raise ValueError(f"unknown edit fields: {set(phase) - allowed}")
    start = float(phase["start_s"])
    rise = float(phase.get("rise_s", .15))
    end = phase.get("end_s")
    fall = float(phase.get("fall_s", .15))
    if start < 0 or rise <= 0 or fall <= 0 or (end is not None and float(end) < start + rise + fall):
        raise ValueError("edit needs positive smooth ramps and a valid time interval")
    def smooth(x):
        x = np.clip(x, 0, 1)
        return x * x * (3 - 2 * x)
    envelope = smooth((time - start) / rise)
    if end is not None:
        envelope = np.minimum(envelope, smooth((float(end) - time) / fall))
    return envelope


def edit_targets(inp: ReplayInput, recipe: dict[str, Any], joint_names: tuple[str, ...]):
    if set(recipe) - {"schema", "row_id", "note", "phases"}:
        raise ValueError("unrecognized recipe fields")
    if recipe.get("schema") != "manorl.local-contact-repair.v1" or recipe.get("row_id") != inp.row_id:
        raise ValueError("recipe schema or source row mismatch")
    base = inp.arrays["desired"]
    desired = base.copy()
    finger_delta = np.zeros((inp.frames, 22))
    for phase in recipe.get("phases", []):
        env = smooth_envelope(inp.arrays["time"], phase)
        translation = np.asarray(phase.get("translation_m", [0, 0, 0]), dtype=float)
        rotation = np.deg2rad(np.asarray(phase.get("rotation_deg", [0, 0, 0]), dtype=float))
        if translation.shape != (3,) or rotation.shape != (3,):
            raise ValueError("wrist corrections must be3-vectors")
        desired[:, :3] += env[:, None] * translation
        correction = Rotation.from_rotvec(env[:, None] * rotation)
        wrist = Rotation.from_euler("XYZ", desired[:, 3:6])
        frame = phase.get("rotation_frame", "world")
        if frame not in ("world", "wrist"):
            raise ValueError("rotation_frame must be world or wrist")
        wrist = correction * wrist if frame == "world" else wrist * correction
        # Match the original Euler branch at frame0; avoid a representation jump.
        angles = np.unwrap(wrist.as_euler("XYZ"), axis=0, period=2 * np.pi)
        angles += 2 * np.pi * np.round((base[0, 3:6] - angles[0]) / (2 * np.pi))
        if np.any(rotation):
            desired[:, 3:6] = angles
        for name, degrees in phase.get("finger_deg", {}).items():
            if name not in joint_names or joint_names.index(name) < 6:
                raise ValueError(f"unknown finger joint: {name}")
            finger_delta[:, joint_names.index(name) - 6] += env * np.deg2rad(float(degrees))
    if not np.isfinite(desired).all() or not np.isfinite(finger_delta).all():
        raise ValueError("nonfinite repair")
    np.testing.assert_allclose(desired[0], base[0], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(finger_delta[0], np.zeros(22))
    position = np.linalg.norm(desired[:, :3] - base[:, :3], axis=1)
    angles = (Rotation.from_euler("XYZ", base[:, 3:6]).inv() * Rotation.from_euler("XYZ", desired[:, 3:6])).magnitude()
    info = dict(wrist_translation_max_mm=float(position.max() * 1000),
                wrist_rotation_max_deg=float(np.rad2deg(angles.max())),
                finger_target_max_deg=float(np.rad2deg(abs(finger_delta).max())))
    # Explicit local envelope: not a route replacement or arbitrary pose reset.
    if info["wrist_translation_max_mm"] > 15.000001 or info["wrist_rotation_max_deg"] > 10.000001 or info["finger_target_max_deg"] > 10.000001:
        raise ValueError(f"correction exceeds local repair envelope: {info}")
    return desired, finger_delta, info


def finger_control(model, current, desired, velocity, envelope, grip, preload, delta, enabled):
    from sim.manorl.mjx_sim import command_target
    lower, upper = model.jnt_range[:28].T
    ctrl = command_target(desired, current[:28], lower, upper)
    if enabled:
        ff = (-model.actuator_biasprm[6:28, 2] * velocity
              + model.dof_frictionloss[6:28] * np.tanh((ctrl[6:] - current[6:28]) / .025))
        ctrl[6:] = np.clip(ctrl[6:] + envelope * ff / model.actuator_gainprm[6:28, 0], lower[6:], upper[6:])
    if len(preload):
        ctrl[6:] = np.clip((1 - grip) * ctrl[6:] + grip * preload, lower[6:], upper[6:])
    # Single additional target offset, after absolute donor preload, never twice.
    ctrl[6:] = np.clip(ctrl[6:] + delta, lower[6:], upper[6:])
    return ctrl


def evaluate(model, inp: ReplayInput, trace: dict[str, np.ndarray]) -> tuple[dict, dict]:
    import mujoco as mj
    data = mj.MjData(model)
    names = inp.arrays["scene_object_names"].tolist()
    active = inp.metrics["source_metadata"]["active_object"]
    index = names.index(active)
    body = model.body(active).id
    fingers, supports, penetration = [], [], []
    contacts = []
    for frame, (pose, vel) in enumerate(zip(trace["qpos"], trace["qvel"])):
        data.qpos[:] = pose
        data.qvel[:] = vel
        mj.mj_forward(model, data)
        finger, support, depth = set(), set(), 0.
        frame_contacts = []
        for c in data.contact[:data.ncon]:
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if body not in (b1, b2):
                continue
            other = model.body(b2 if b1 == body else b1).name
            if other in names or other == "world":
                support.add(other)
            else:
                finger.add(other.split("_")[0])
                depth = max(depth, -float(c.dist))
            frame_contacts.append(dict(other=other, distance_m=float(c.dist), position=c.pos.tolist(), normal=c.frame[:3].tolist()))
        fingers.append(len(finger)); supports.append(sorted(support)); penetration.append(depth)
        contacts.append(frame_contacts)
    count = np.asarray(fingers)
    supported = np.asarray([bool(s) for s in supports])
    obj = trace["actual_object_pos"][:, index]
    reference = inp.arrays["source_object_pos"][:, index]
    height = obj[:, 2] - obj[0, 2]
    ref_height = reference[:, 2] - reference[0, 2]
    rotation = Rotation.from_quat(trace["actual_object_quat_xyzw"][:, index])
    ref_rot = Rotation.from_quat(inp.arrays["source_object_quat_xyzw"][:, index])
    full_error = np.rad2deg((ref_rot.inv() * rotation).magnitude())
    axis = rotation.apply([0, 0, 1])
    ref_axis = ref_rot.apply([0, 0, 1])
    tilt = np.rad2deg(np.arccos(np.clip(axis[:, 2], -1, 1)))
    axis_error = np.rad2deg(np.arccos(np.clip((axis * ref_axis).sum(axis=1), -1, 1)))
    error = np.linalg.norm(obj - reference, axis=1)
    airborne = ref_height > max(.03, float(ref_height.max()) * .4)
    fraction = float(np.mean(count[airborne] >= 2)) if airborne.any() else 0.
    action = inp.metrics["source_metadata"]["source_gesture"][:3]
    hold = action == "009"
    gates = dict(lift_tracks_reference_scale=float(height.max()) > float(ref_height.max()) * .65,
                 airborne_opposing_contact=fraction > .80)
    if hold:
        gates.update(final_held=count[-1] >= 2 and not supported[-1],
                     final_object_orientation_under_15deg=full_error[-1] < 15)
    else:
        expected = "cuboid1" if action == "007" else "world"
        gates.update(final_supported=expected in supports[-1], final_released=count[-1] == 0,
                     final_object_upright=tilt[-1] < 10,
                     final_object_position_under_8cm=error[-1] < .08)
    gates = {k: bool(v) for k, v in gates.items()}
    core = {k: v for k, v in gates.items() if k not in ("final_object_upright", "final_object_orientation_under_15deg")}
    wrist_error = np.linalg.norm(trace["qpos"][:, :3] - trace["desired"][:, :3], axis=1)
    wrist_angle = np.rad2deg((Rotation.from_euler("XYZ", trace["desired"][:, 3:6]).inv() * Rotation.from_euler("XYZ", trace["qpos"][:, 3:6])).magnitude())
    lifted = (height >= .02) & (count > 0) & ~supported
    boundaries = np.flatnonzero(np.diff(np.r_[False, lifted, False]))
    runs = list(zip(boundaries[::2], boundaries[1::2]))
    longest = max((end - start for start, end in runs), default=0)
    result = dict(row_id=inp.row_id, physical_pose_pass=all(gates.values()), functional_pass=all(core.values()),
                  gates=gates, failed_gates=[k for k, v in gates.items() if not v],
                  category=inp.metrics["source_metadata"]["source_gesture"], hold=hold,
                  max_lift_m=float(height.max()), reference_max_lift_m=float(ref_height.max()),
                  airborne_multi_finger_fraction=fraction, final_support=supports[-1],
                  final_finger_rays=int(count[-1]), object_position_final_cm=float(error[-1] * 100),
                  object_orientation_final_deg=float(full_error[-1]), final_world_tilt_deg=float(tilt[-1]),
                  final_opening_axis_reference_error_deg=float(axis_error[-1]),
                  longest_contact_lift_s=longest / inp.hz, max_hand_penetration_mm=float(max(penetration) * 1000),
                  actual_wrist_position_p95_mm=float(np.quantile(wrist_error, .95) * 1000),
                  actual_wrist_angle_p95_deg=float(np.quantile(wrist_angle, .95)),
                  contact_semantics="native geometry reconstructed from measured qpos; not force closure or GPU force telemetry",
                  terminal_semantics="hold" if hold else "support and release", outlet_alignment="unmeasured; no fluid simulation")
    arrays = dict(finger_ray_count=count, has_support=supported, height_m=height, reference_height_m=ref_height,
                  world_tilt_deg=tilt, full_orientation_error_deg=full_error,
                  opening_axis_reference_error_deg=axis_error, object_position_error_m=error,
                  hand_penetration_m=np.asarray(penetration))
    return result, dict(arrays=arrays, contacts=contacts)
