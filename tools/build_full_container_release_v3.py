#!/usr/bin/env python3
"""Create late-phase v3 repairs from the seven pass-2 physical failure traces.

The v2 command arrays are immutable inputs.  Each v3 candidate copies its v2
prefix byte-for-byte and changes only the observed late failure interval:
measured wrist load is removed before opening, then the hand withdraws along
its physically measured bottle radial or pitcher-handle-local-Y direction.
No simulator stepping, force, object-pose, material, or gain edits occur here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "patches/full_containers_v2"
DEFAULT_OUTPUT = ROOT / "patches/full_containers_v3"
ROWS = (26, 29, 30, 35, 37, 38)
# First command changed; all preceding commands must remain v2-identical.
ANCHOR = {26: 610, 29: 440, 30: 625, 35: 600, 37: 1040, 38: 430}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    return hashlib.sha256(values.dtype.str.encode() + str(values.shape).encode() + values.tobytes()).hexdigest()


def smooth(a: float) -> float:
    a = float(np.clip(a, 0., 1.))
    return a * a * (3. - 2. * a)


def blend_track(base: np.ndarray, start: int, end: int, target: np.ndarray) -> None:
    """Smoothly move the command from its observed v2 target to target."""
    initial = base[start].copy()
    for frame in range(start, len(base)):
        base[frame] = (1. - smooth((frame - start) / (end - start))) * initial + smooth((frame - start) / (end - start)) * target


def apply_slice_ramp(commands: np.ndarray, start: int, end: int, indices: slice, final: np.ndarray) -> None:
    initial = commands[start, indices].copy()
    for frame in range(start, len(commands)):
        commands[frame, indices] = (1. - smooth((frame - start) / (end - start))) * initial + smooth((frame - start) / (end - start)) * final


def yaw_only(rotation: Rotation) -> Rotation:
    forward = rotation.apply([1., 0., 0.])
    return Rotation.from_euler("z", np.arctan2(forward[1], forward[0]))


def target_wrist(trace: dict[str, np.ndarray], frame: int, target_pos: np.ndarray, level: bool) -> np.ndarray:
    """Measured object-relative wrist target, optionally leveled about the object."""
    q = trace["qpos"][frame, :28].copy()
    if not level:
        return q
    names = trace["scene_object_names"].tolist()
    active = "mayonnaisebottle" if "mayonnaisebottle" in names else "pitcherbase"
    index = names.index(active)
    object_pos = trace["scene_object_pos"][frame, index]
    object_rot = Rotation.from_rotvec(trace["scene_object_rot_aa"][frame, index])
    correction = yaw_only(object_rot) * object_rot.inv()
    q[:3] = target_pos + correction.apply(q[:3] - object_pos)
    q[3:6] = (correction * Rotation.from_euler("XYZ", q[3:6])).as_euler("XYZ")
    return q


def build_row(row: int, v2dir: Path, evidence: Path, output: Path) -> dict[str, Any]:
    stem = "bottle05" if row < 31 else "pitcher06"
    v2patch_path = v2dir / f"guangxue_{stem}_row{row}_contact_primitive_transfer_v2.json"
    patch = json.loads(v2patch_path.read_text())
    v2command = v2dir / patch["command_track"]["path"]
    with np.load(v2command, allow_pickle=False) as saved:
        commands = saved["ctrl"].copy()
        reference_index = saved["reference_index"].copy()
        donor_trace_index = saved["donor_trace_index"].copy()
    with np.load(evidence / f"row{row}" / "trajectory.npz", allow_pickle=False) as saved:
        trace = {key: saved[key].copy() for key in saved.files}
    names = tuple(trace["scene_object_names"].tolist())
    expected = ("bowl", "mayonnaisebottle" if row < 31 else "pitcherbase")
    if names != expected:
        raise ValueError(f"row{row}: source scene order {names} != {expected}")
    if not np.array_equal(trace["scene_object_qpos_addresses"], np.array([28, 35], dtype=trace["scene_object_qpos_addresses"].dtype)):
        raise ValueError(f"row{row}: unexpected object qpos ABI")
    if len(trace["qpos"]) != len(commands):
        raise ValueError(f"row{row}: evidence/command lengths differ")

    anchor = ANCHOR[row]
    settle_end = anchor + (90 if row in (29, 35, 38) else 45)
    release_start = settle_end + (30 if row == 35 else 15)
    release_end = release_start + 45
    withdraw_end = release_end + 70
    if withdraw_end >= len(commands):
        raise ValueError(f"row{row}: no tail for safe withdrawal")
    active_index = 1
    observed = trace["qpos"][anchor, :28]
    object_pos = trace["scene_object_pos"][anchor, active_index]
    object_rot = Rotation.from_rotvec(trace["scene_object_rot_aa"][anchor, active_index])
    goal = trace["reference_object_pos"][-1]

    # Carry the object with its measured grasp frame to its receiver goal.  The
    # bottle correction only needs leveling for row29; pitcher35/38 need it to
    # remove the observed tilted wrist load.  Row37 is already upright and only
    # needs a de-loaded release.
    level = row in (29, 35, 38)
    place_pos = goal if row != 37 else object_pos
    wrist = target_wrist(trace, anchor, place_pos, level)
    if row == 38:
        # The raw trace first intersects the bowl while the pitcher is tilted.
        # Lift the hand/object clearance 7cm before flattening; the original
        # tilt prefix remains untouched and is followed by an actual standoff.
        wrist[:3] += np.array([0., 0., 0.07])
    blend_track(commands, anchor, settle_end, wrist)

    # Remove actuator load by using the observed wrist state before any opening.
    # After the load-reduction interval, open thumb before the remaining fingers
    # for handled pitchers; bottles use simultaneous source-like opening.
    open_targets = commands[-1, 6:].copy()
    if row < 31:
        apply_slice_ramp(commands, release_start, release_end, slice(6, 28), open_targets)
    else:
        apply_slice_ramp(commands, release_start, release_start + 20, slice(6, 12), open_targets[:6])
        apply_slice_ramp(commands, release_start + 20, release_end, slice(12, 28), open_targets[6:])

    # Fix a released, geometry-aligned retreat.  Pitcher donor traces leave the
    # handle along +object-local-Y.  Bottles retreat radially from the measured
    # object center once all fingers have opened.
    if row < 31:
        radial = observed[:3] - object_pos
        radial /= np.linalg.norm(radial)
        retreat = radial * 0.18 + np.array([0., 0., 0.06])
        retreat_kind = "measured bottle radial withdrawal after simultaneous opening"
    else:
        retreat = object_rot.apply([0., 0.14, 0.06])
        retreat_kind = "measured pitcher handle +object-local-Y withdrawal after thumb-first opening"
    final = commands[release_end].copy()
    final[:3] = wrist[:3] + retreat
    # Retain the leveled/de-loaded wrist attitude: changing attitude during the
    # retreat reintroduced the row37 palm collision in the physical trace.
    final[3:6] = wrist[3:6]
    blend_track(commands, release_end, withdraw_end, final)

    # Strict ownership invariant: v2 approach/lift/transport prefix unchanged.
    if not np.array_equal(commands[:anchor], np.load(v2command, allow_pickle=False)["ctrl"][:anchor]):
        raise AssertionError(f"row{row}: modified v2 prefix")
    command_name = f"guangxue_{stem}_row{row}_late_release_commands_v3.npz"
    command_path = output / command_name
    np.savez_compressed(command_path, ctrl=commands, reference_index=reference_index, donor_trace_index=donor_trace_index)
    prefix = np.ascontiguousarray(commands[:anchor])
    patch["name"] = f"guangxue_{stem}_row{row}_late_release_v3"
    patch["edit"] = {"name": "v3_measured_late_placement_release", "translation": [0, 0, 0], "release_at_end": True}
    patch["command_track"] = {"path": command_name, "sha256": sha256(command_path), "frames": len(commands)}
    meta = patch["contact_primitive_transfer"]
    meta["method"] = "v3_v2_prefix_preserving_measured_late_placement_release"
    meta["command_sha256_before_npz"] = array_hash(commands)
    meta["v3_late_repair"] = {
        "evidence_trajectory": str(evidence / f"row{row}" / "trajectory.npz"),
        "evidence_sha256": sha256(evidence / f"row{row}" / "trajectory.npz"),
        "scene_object_names": list(names),
        "scene_object_qpos_addresses": trace["scene_object_qpos_addresses"].astype(int).tolist(),
        "prefix_end_exclusive": anchor,
        "prefix_ctrl_sha256": array_hash(prefix),
        "anchor_frame": anchor,
        "settle_end_frame": settle_end,
        "release_start_frame": release_start,
        "release_end_frame": release_end,
        "withdraw_end_frame": withdraw_end,
        "observed_active_pos_m": object_pos.tolist(),
        "observed_hand_to_object_m": (observed[:3] - object_pos).tolist(),
        "observed_active_rotvec_rad": trace["scene_object_rot_aa"][anchor, active_index].tolist(),
        "receiver_goal_object_pos_m": goal.tolist(),
        "level_object_about_measured_grasp": level,
        "bowl_clearance_lift_m": 0.07 if row == 38 else 0.0,
        "retreat_world_m": retreat.tolist(),
        "retreat_policy": retreat_kind,
        "finger_release_policy": "simultaneous bottle opening" if row < 31 else "thumb joints [6:12] open before index/middle/ring/pinky joints [12:28]",
        "reason": {26: "upright floor support at frames618-644 was lost as wrist target continued to load the bottle", 29: "return began at 29deg tilt and never established upright support", 30: "upright floor support at frames633-683 was lost under late wrist rotation/load", 35: "pitcher remained hand-supported at a tilted placement and fell after delayed release", 37: "post-release retreat reintroduced a palm contact after an otherwise upright stable placement", 38: "late tilted descent contacted the bowl before release and toppled the pitcher"}[row],
    }
    patch_path = output / f"guangxue_{stem}_row{row}_late_release_v3.json"
    patch_path.write_text(json.dumps(patch, indent=2, sort_keys=True) + "\n")
    return {"row": row, "uuid": patch["source"]["uuid"], "active_object": patch["active_object"], "patch": patch_path.name, "command_track": command_name, "command_sha256": patch["command_track"]["sha256"], "prefix_end_exclusive": anchor}


def row33_metric_review(evidence: Path, output: Path) -> dict[str, Any]:
    with np.load(evidence / "row33" / "trajectory.npz", allow_pickle=False) as saved:
        names = tuple(saved["scene_object_names"].tolist())
        if names != ("bowl", "pitcherbase"):
            raise ValueError(f"row33 object order: {names}")
        pos = saved["scene_object_pos"]
        rotation = Rotation.from_rotvec(saved["scene_object_rot_aa"][:, 1])
        tilt = np.degrees(np.arccos(np.clip(rotation.apply([0., 0., 1.])[:, 2], -1., 1.)))
        distance = np.linalg.norm(pos[:, 1, :2] - pos[:, 0, :2], axis=1)
        elevation = pos[:, 1, 2] - pos[:, 0, 2]
        frame = int(np.argmin(distance))
        observation = {"nearest_center_frame": frame, "center_distance_m": float(distance[frame]), "tilt_deg": float(tilt[frame]), "pitcher_above_bowl_m": float(elevation[frame]), "max_tilt_deg": float(tilt.max()), "evidence_sha256": sha256(evidence / "row33" / "trajectory.npz")}
    report = {"row": 33, "kind": "metric_review_no_command_candidate", "conclusion": "The pitcher center reaches 5.6cm from the bowl center while elevated 10.2cm and tilted 54.1deg; its spout-side volume is consequently over the bowl even though the evaluator requires 25 frames strictly above 55deg. No fake command correction is supplied.", "observation": observation, "recommendation": "review tilt_over_bowl threshold/boundary against spout geometry; retain accepted physical outcome (upright, released, supported, near goal)."}
    path = output / "row33_tilt_over_bowl_metric_review.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"row": 33, "kind": "metric_review_no_command_candidate", "report": path.name, **observation}


def validate(output: Path, v2dir: Path) -> dict[str, Any]:
    inventory = json.loads((output / "receiver_inventory.json").read_text())["repairs"]
    assert {x["row"] for x in inventory if "patch" in x} == set(ROWS)
    assert any(x["row"] == 33 and x["kind"] == "metric_review_no_command_candidate" for x in inventory)
    for item in inventory:
        if "patch" not in item:
            continue
        patch = json.loads((output / item["patch"]).read_text())
        v3 = patch["contact_primitive_transfer"]["v3_late_repair"]
        with np.load(output / item["command_track"], allow_pickle=False) as saved, np.load(v2dir / f"guangxue_{'bottle05' if item['row'] < 31 else 'pitcher06'}_row{item['row']}_contact_primitive_commands_v2.npz", allow_pickle=False) as old:
            ctrl = saved["ctrl"]
            assert ctrl.shape == old["ctrl"].shape and np.all(np.isfinite(ctrl))
            assert np.array_equal(ctrl[:v3["prefix_end_exclusive"]], old["ctrl"][:v3["prefix_end_exclusive"]])
            assert array_hash(ctrl) == patch["contact_primitive_transfer"]["command_sha256_before_npz"]
        assert sha256(output / item["command_track"]) == patch["command_track"]["sha256"]
    return {"validated_candidates": len(ROWS), "metric_reviews": 1, "files": len(list(output.iterdir()))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, help="Directory of measured pass-2 replay records")
    p.add_argument("--v2", type=Path, default=V2)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        print(json.dumps(validate(a.output, a.v2), sort_keys=True))
        return
    if a.evidence is None:
        p.error("--evidence is required when generating release corrections")
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError(f"refusing nonempty output: {a.output}")
    a.output.mkdir(parents=True, exist_ok=True)
    repairs = [build_row(row, a.v2, a.evidence, a.output) for row in ROWS]
    repairs.append(row33_metric_review(a.evidence, a.output))
    (a.output / "receiver_inventory.json").write_text(json.dumps({"schema": "container_late_release_inventory.v3", "source": "v2 command assets plus measured pass2 physical traces", "repairs": repairs}, indent=2, sort_keys=True) + "\n")
    runners = ["# Six GPU replay candidates; row33 is a metric review, not a replay patch.", "# REPLAY=/mnt/nas-222-project/mocap_v2/lance_datasets/dexgem_vla_demo_guangxue_astra_repair_5samples_20260910/tools/replay_repaired_capture.py"]
    for x in repairs:
        if "patch" in x:
            runners.append(f"$REPLAY --dataset /mnt/nas-222-project/mocap_v2/lance_datasets/dexgem_vla_demo_guangxue_cma2lance_20260909_060700.lance --patch {a.output / x['patch']} --output <parent-output>/row{x['row']}")
    (a.output / "RUNNERS.md").write_text("\n".join(runners) + "\n")
    print(json.dumps(validate(a.output, a.v2), sort_keys=True))

if __name__ == "__main__":
    main()
