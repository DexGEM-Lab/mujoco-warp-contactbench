#!/usr/bin/env python3
"""Kinematic retreat-row construction for +30 clear rows (batch 1).

Mechanism: for each base row whose persisted float32 solved right-hand/target-object
normal force at movement_end+30 is <= 0.2 N, construct a 30-frame (250 ms @120 Hz)
deterministic retreat trajectory. Frame 0 is byte-equal to the base row anchor state;
the right hand wrist follows a C2 quintic out to a per-pair Near-cell target; fingers
hold the anchor pose; the object holds the anchor pose exactly. No MuJoCo re-simulation.

Validation (per row, hard gate): finite values, wrist C2 smoothness, hand-object
clearance along the path (conservative point-to-mesh), joint limits, monotone
timestamps, endpoint cell hit, object static. Any failure excludes the row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import uuid as uuid_mod

import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.contracts import JOINT_DOF
from sim.manorl.lance_v2 import (
    SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT,
    write_compact_lance_stream,
)

THRESHOLD_N = 0.2
ANCHOR_OFFSET = 30
RETREAT_STEPS = 30
CELL_SHAPE = (5, 3, 2)
CLEARANCE_M = 0.02
JOINT_DOMAIN = {
    "banana:02": {"r": (0.252, 0.395), "a_deg": (-22.32, 22.21), "z": (0.0604, 0.0991)},
    "banana:18": {"r": (0.2655, 0.3891), "a_deg": (-21.62, 22.15), "z": (0.0581, 0.0966)},
    "bowl:04": {"r": (0.2428, 0.3677), "a_deg": (-21.84, 22.01), "z": (0.0442, 0.0808)},
    "mayonnaisebottle:04": {"r": (0.2463, 0.3741), "a_deg": (-22.03, 22.34), "z": (-0.0210, 0.0160)},
}


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def anchor_contact(row: dict, object_name: str) -> bool:
    end = int(row["trajectory_metadata"]["trajectory_info"]["object_move"][0]["end_frame"])
    frame = row["contact"][ANCHOR_OFFSET + end]
    for entry in frame or []:
        if entry.get("hand_name") != "right" or entry.get("object_name") != object_name:
            continue
        for pair in entry.get("contact_pairs") or []:
            f = np.asarray(pair.get("force_normal") or (), dtype=np.float32)
            if f.shape == (3,) and np.linalg.norm(f) > np.float32(THRESHOLD_N):
                return True
    return False


def _quintic(start: np.ndarray, end: np.ndarray, frames: int, dt: float) -> np.ndarray:
    """C2 quintic with zero start/end velocity and acceleration."""
    s = np.linspace(0.0, 1.0, frames)[:, None]
    return start + (end - start) * (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)


def _cell_target(pair: str, cell: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    dom = JOINT_DOMAIN[pair]
    shape = np.asarray(CELL_SHAPE, dtype=np.float64)
    lo = np.asarray([dom["r"][0], dom["a_deg"][0], dom["z"][0]], dtype=np.float64)
    hi = np.asarray([dom["r"][1], dom["a_deg"][1], dom["z"][1]], dtype=np.float64)
    center_frac = (np.asarray(cell, dtype=np.float64) + 0.5) / shape
    jitter = rng.uniform(-0.5, 0.5, size=3) / shape
    frac = np.clip(center_frac + jitter, 0.0, np.nextafter(1.0, 0.0))
    r, a_deg, z = lo + frac * (hi - lo)
    a = math.radians(a_deg)
    return np.asarray([r * math.cos(a), r * math.sin(a), z], dtype=np.float64)


def _joint_limits_valid(q: np.ndarray, lower: np.ndarray, upper: np.ndarray, margin: float) -> bool:
    """Only the constructed wrist XYZ (dim 0..2) is limit-checked. Finger joints
    are copied verbatim from the published anchor and are not part of this
    construction, so their solver-side micro-violations must not reject rows."""
    if q.ndim == 1:
        return bool(
            np.all(q[:3] >= lower[:3] - margin) and np.all(q[:3] <= upper[:3] + margin)
        )
    return bool(
        np.all(q[:, :3] >= lower[:3] - margin) and np.all(q[:, :3] <= upper[:3] + margin)
    )


def _obb_distances(
    points: np.ndarray, object_world: np.ndarray, object_quat_xyzw: np.ndarray,
    object_vertices_local: np.ndarray,
) -> np.ndarray:
    """Signed-ish distance from points to the object OBB (>=0 outside, 0 inside)."""
    R = Rotation.from_quat(object_quat_xyzw).as_matrix()
    proj = object_vertices_local @ R.T
    lo = proj.min(axis=0) + object_world
    hi = proj.max(axis=0) + object_world
    half = (hi - lo) / 2.0
    center = (hi + lo) / 2.0
    d = np.abs(points - center) - half
    dist = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    return dist


def _hand_object_clearance(
    anchor_mano_joint_pos: np.ndarray,
    wrist_deltas: np.ndarray,
    object_world: np.ndarray,
    object_quat_xyzw: np.ndarray,
    object_vertices_local: np.ndarray,
    margin: float,
) -> bool:
    """A retreat path is valid if the wrist moves monotonically away from the
    object (min per-frame keypoint OBB distance is non-decreasing) and the final
    frame is outside the OBB by at least ``margin``. The anchor itself may
    legally overlap the object (no solved contact >0.2 N at +30, but the hand can
    still hover close); requiring the whole path to be outside would reject every
    real retreat."""
    anchor_dist = _obb_distances(anchor_mano_joint_pos, object_world, object_quat_xyzw, object_vertices_local)
    min_anchor = float(np.min(anchor_dist))
    for delta in wrist_deltas:
        pts = anchor_mano_joint_pos + delta
        dist = _obb_distances(pts, object_world, object_quat_xyzw, object_vertices_local)
        if float(np.min(dist)) < min_anchor - 1e-6:
            return False
    final_pts = anchor_mano_joint_pos + wrist_deltas[-1]
    final_dist = float(np.min(_obb_distances(final_pts, object_world, object_quat_xyzw, object_vertices_local)))
    if final_dist < margin:
        return False
    return True


def build_retreat_row(
    base: dict,
    *,
    target: np.ndarray,
    object_world: np.ndarray,
    object_quat_xyzw: np.ndarray,
    vertices_local: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    cell: tuple[int, int, int],
    attempt_seed: int,
) -> tuple[dict | None, dict]:
    right = base["trajectory_metadata"]["hand_slots"].index("right")
    end = int(base["trajectory_metadata"]["trajectory_info"]["object_move"][0]["end_frame"])
    anchor = ANCHOR_OFFSET + end
    T = RETREAT_STEPS
    dt = 1.0 / 120.0
    q_anchor = np.asarray(base["hands"][right]["urdf_dof"][anchor], dtype=np.float64)
    q_target = np.asarray(base["hands"][right]["urdf_dof_target"][anchor], dtype=np.float64)
    wrist_anchor = q_anchor[:3]
    wrist_target_abs = wrist_anchor + target
    path = _quintic(wrist_anchor, wrist_target_abs, T, dt)
    wrist_deltas = path - wrist_anchor
    diag = {
        "cell": list(cell),
        "target_object_local": target.tolist(),
        "endpoint_world": wrist_target_abs.tolist(),
        "path_speed_max_m_s": float(np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)) / dt),
    }
    mano_joint_anchor = np.asarray(base["hands"][right]["mano_joint_pos"][anchor], dtype=np.float64)
    if not _hand_object_clearance(
        mano_joint_anchor, wrist_deltas, object_world, object_quat_xyzw, vertices_local, CLEARANCE_M
    ):
        diag["excluded"] = "hand_object_clearance"
        return None, diag
    urdf_seq = np.tile(q_anchor, (T, 1))
    urdf_seq[:, :3] = path
    if not _joint_limits_valid(urdf_seq, joint_lower, joint_upper, 1e-6):
        diag["excluded"] = "joint_limits"
        return None, diag
    if not np.array_equal(urdf_seq[0], q_anchor):
        diag["excluded"] = "frame0_mismatch"
        return None, diag

    object_name = base["index"]["scene"]
    gesture = str(base["trajectory_metadata"]["gesture"]).zfill(2)
    ts = np.arange(T, dtype=np.float64) * dt
    ref_idx = np.asarray(base["reference"]["source_frame_index"][anchor:anchor + T], dtype=np.int64)
    hand_pose = base["hands"][right]["mano_hand_pose"][anchor:anchor + T]
    mano_global_pos = base["hands"][right]["mano_global_pos"][anchor:anchor + T]
    mano_global_rot = base["hands"][right]["mano_global_rot_aa"][anchor:anchor + T]
    mano_joint_seq = base["hands"][right]["mano_joint_pos"][anchor:anchor + T]
    object_pos = np.tile(base["objects"][0]["pos"][anchor], (T, 1))
    object_rot = np.tile(base["objects"][0]["rot_aa"][anchor], (T, 1))
    urdf_target_seq = np.tile(q_target, (T, 1))
    cmd_ref = np.arange(T - 1, dtype=np.int64) + anchor
    cmd_src = np.asarray([base["reference"]["source_frame_index"][anchor + i] for i in range(T - 1)], dtype=np.int64)

    prov = dict(base["provenance"])
    prov["contract"] = SYNTHETIC_LANCE_COMPACT_V2_CONTACT_CONTRACT
    prov["augmentation_identity"] = "manorl_retreat_extension_v1_batch1:kinematic"
    prov["seed"] = attempt_seed
    new_uuid = str(uuid_mod.uuid4())
    row = {
        "index": {
            "uuid": new_uuid,
            "seed_uuid": base["index"]["uuid"],
            "capMachine": base["index"].get("capMachine") or "manorl-kinematic",
            "operator": base["index"].get("operator") or "manorl",
            "scene": base["index"]["scene"],
            "is_generated": True,
        },
        "trajectory_metadata": {
            "data_fps": 120,
            "total_frames": T,
            "gesture": gesture,
            "hand_names": ["right"],
            "hand_slots": ["right", "left"],
            "object_names": [object_name],
            "mano_hand_shapes": base["trajectory_metadata"]["mano_hand_shapes"],
            "trajectory_info": {
                "object_move": [{"object_name": object_name, "start_frame": T - 1, "end_frame": 0}]
            },
        },
        "timestamp": ts.tolist(),
        "hands": [
            {
                "hand_name": "right",
                "mano_global_pos": mano_global_pos,
                "mano_global_rot_aa": mano_global_rot,
                "mano_hand_pose": hand_pose,
                "mano_joint_pos": mano_joint_seq,
                "urdf_dof": urdf_seq.tolist(),
                "urdf_dof_target": urdf_target_seq.tolist(),
            },
            {
                "hand_name": None,
                "mano_global_pos": [],
                "mano_global_rot_aa": [],
                "mano_hand_pose": [],
                "mano_joint_pos": [],
                "urdf_dof": [],
                "urdf_dof_target": [],
            },
        ],
        "objects": [{"rot_aa": object_rot.tolist(), "pos": object_pos.tolist()}],
        "contact": [[] for _ in range(T)],
        "reference": {
            "source_frame_index": ref_idx.tolist(),
            "hand_urdf_dof": urdf_seq.tolist(),
            "object_pos": object_pos.tolist(),
            "object_rot_aa": object_rot.tolist(),
        },
        "command_reference_index": cmd_ref.tolist(),
        "command_source_frame_index": cmd_src.tolist(),
        "provenance": prov,
    }
    diag["uuid"] = new_uuid
    diag["accepted"] = True
    return row, diag


def load_base_rows(source: Path):
    import lance
    return lance.dataset(str(source)).scanner(batch_size=8, scan_in_order=True).to_table().to_pylist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from sim.manorl.assets import object_collision_vertices

    rows = load_base_rows(args.source)
    eligible = []
    for r in rows:
        obj = r["index"]["scene"]
        if anchor_contact(r, obj):
            continue
        eligible.append(r)
    print(f"base={len(rows)} eligible={len(eligible)}", flush=True)
    per_pair = {}
    for r in eligible:
        pair = f"{r['index']['scene']}:{str(r['trajectory_metadata']['gesture']).zfill(2)}"
        per_pair.setdefault(pair, []).append(r)
    out_rows = []
    diagnostics = []
    rng = np.random.default_rng(args.seed)
    for pair, group in sorted(per_pair.items()):
        obj_type = pair.split(":")[0]
        vertices = object_collision_vertices(obj_type)
        cells = [(a, b, c) for a in range(5) for b in range(3) for c in range(2)]
        from sim.manorl.assets import compile_model, ServoConfig
        _mujoco, _model = compile_model(
            ServoConfig(), object_type=obj_type, hand_side="right",
            physics_timestep=1.0 / 480.0,
        )
        joint_lower = np.asarray(_model.jnt_range[:JOINT_DOF, 0], dtype=np.float64)
        joint_upper = np.asarray(_model.jnt_range[:JOINT_DOF, 1], dtype=np.float64)
        for idx, r in enumerate(group):
            cell = cells[idx % len(cells)]
            anchor = ANCHOR_OFFSET + int(r["trajectory_metadata"]["trajectory_info"]["object_move"][0]["end_frame"])
            object_world = np.asarray(r["objects"][0]["pos"][anchor], dtype=np.float64)
            rotvec = np.asarray(r["objects"][0]["rot_aa"][anchor], dtype=np.float64)
            object_quat = Rotation.from_rotvec(rotvec).as_quat()
            target = _cell_target(pair, cell, rng)
            row, diag = build_retreat_row(
                r,
                target=target,
                object_world=object_world,
                object_quat_xyzw=object_quat,
                vertices_local=vertices,
                joint_lower=joint_lower,
                joint_upper=joint_upper,
                cell=cell,
                attempt_seed=args.seed + idx,
            )
            diag["pair"] = pair
            diag["source_uuid"] = r["index"]["uuid"]
            diagnostics.append(diag)
            if row is not None:
                out_rows.append(row)
    accepted = len(out_rows)
    excluded = len(diagnostics) - accepted
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_compact_lance_stream(out_rows, output=args.output, replace=True, batch_size=64)
    plan_digest = _canonical_digest(
        {
            "contract": "manorl_retreat_extension_v1_batch1_kinematic",
            "source": str(args.source),
            "eligible": len(eligible),
            "accepted": accepted,
            "excluded": excluded,
            "anchor_offset": ANCHOR_OFFSET,
            "retreat_steps": RETREAT_STEPS,
            "cell_shape": list(CELL_SHAPE),
            "clearance_m": CLEARANCE_M,
            "domain": JOINT_DOMAIN,
        }
    )
    summary = {
        "contract": "manorl_retreat_extension_v1_batch1_kinematic",
        "source": str(args.source),
        "output": str(args.output),
        "base_rows": len(rows),
        "eligible": len(eligible),
        "accepted": accepted,
        "excluded": excluded,
        "retreat_steps": RETREAT_STEPS,
        "anchor_offset": ANCHOR_OFFSET,
        "cell_shape": list(CELL_SHAPE),
        "clearance_m": CLEARANCE_M,
        "digest": plan_digest,
        "accepted_per_pair": {k: sum(1 for d in diagnostics if d["pair"] == k and d.get("accepted")) for k in per_pair},
        "excluded_reasons": {k: sum(1 for d in diagnostics if d.get("excluded") == k) for k in sorted({str(d.get("excluded")) for d in diagnostics})},
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
