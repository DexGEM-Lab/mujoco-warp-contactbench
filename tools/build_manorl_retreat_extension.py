#!/usr/bin/env python3
"""Kinematic retreat-row construction for +30 clear rows (batch 1) and
per-row first-clear anchor rows (batch 2).

Mechanism: for each base row, select the anchor at the first offset in
{30, 40, 50, 75} after movement_end whose persisted float32 solved
right-hand/target-object normal force is <= 0.2 N. Frame 0 of the retreat is
byte-equal to that anchor state; the right hand wrist follows a C2 quintic out
to a per-pair Near-cell target; fingers and object hold the anchor pose.
No MuJoCo re-simulation.

Validation (per row, hard gate): finite values, wrist XYZ joint limits,
monotone timestamps, monotone hand-object OBB separation, final OBB clearance,
object static, frame0 exact. Clearance failures retry with fresh cell targets
up to --retry-clearance attempts. Any unresolved failure excludes the row.
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
ANCHOR_OFFSETS = (30, 40, 50, 75)
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


def anchor_contact_at(row: dict, object_name: str, index: int) -> bool:
    frame = row["contact"][index]
    for entry in frame or []:
        if entry.get("hand_name") != "right" or entry.get("object_name") != object_name:
            continue
        for pair in entry.get("contact_pairs") or []:
            f = np.asarray(pair.get("force_normal") or (), dtype=np.float32)
            if f.shape == (3,) and np.linalg.norm(f) > np.float32(THRESHOLD_N):
                return True
    return False


def first_clear_offset(row: dict, object_name: str) -> int | None:
    """First offset in ANCHOR_OFFSETS with no solved hand-object contact."""
    end = int(row["trajectory_metadata"]["trajectory_info"]["object_move"][0]["end_frame"])
    total = int(row["trajectory_metadata"]["total_frames"])
    for off in ANCHOR_OFFSETS:
        index = end + off
        if index >= total:
            continue
        if not anchor_contact_at(row, object_name, index):
            return off
    return None


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
    """Retreat path is valid if the wrist moves monotonically away from the
    object (min per-frame keypoint OBB distance non-decreasing) and the final
    frame is outside the OBB by at least ``margin``. The anchor itself may
    legally overlap the object (no solved contact >0.2 N at the anchor, but the
    hand can still hover close)."""
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
    anchor_offset: int,
    target: np.ndarray,
    object_world: np.ndarray,
    object_quat_xyzw: np.ndarray,
    vertices_local: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    cell: tuple[int, int, int],
    attempt_seed: int,
    batch_tag: str,
) -> tuple[dict | None, dict]:
    right = base["trajectory_metadata"]["hand_slots"].index("right")
    end = int(base["trajectory_metadata"]["trajectory_info"]["object_move"][0]["end_frame"])
    anchor = anchor_offset + end
    total = int(base["trajectory_metadata"]["total_frames"])
    if anchor + RETREAT_STEPS > total:
        return None, {"excluded": "insufficient_frames", "anchor_offset": anchor_offset, "remaining_frames": total - anchor}
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
        "anchor_offset": anchor_offset,
        "remaining_frames": total - anchor,
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
    prov["augmentation_identity"] = f"manorl_retreat_extension_v1_{batch_tag}:kinematic"
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
    parser.add_argument("--row-filter", choices=("clear", "contact", "both"), default="both")
    parser.add_argument("--anchor-mode", choices=("first-clear", "fixed"), default="first-clear")
    parser.add_argument("--fixed-offset", type=int, default=30)
    parser.add_argument("--retry-clearance", type=int, default=0)
    parser.add_argument("--include-uuids", type=Path, default=None)
    parser.add_argument("--batch-tag", type=str, default="batch1")
    args = parser.parse_args(argv)

    from sim.manorl.assets import compile_model, object_collision_vertices, ServoConfig

    rows = load_base_rows(args.source)

    def contact_at_30(r: dict) -> bool:
        obj = r["index"]["scene"]
        end = int(r["trajectory_metadata"]["trajectory_info"]["object_move"][0]["end_frame"])
        return anchor_contact_at(r, obj, end + 30)

    if args.row_filter == "clear":
        selected = [r for r in rows if not contact_at_30(r)]
    elif args.row_filter == "contact":
        selected = [r for r in rows if contact_at_30(r)]
    else:
        selected = list(rows)
    if args.include_uuids is not None:
        wanted = set(json.loads(args.include_uuids.read_text()))
        extra = [r for r in rows if r["index"]["uuid"] in wanted]
        selected = list(dict.fromkeys(selected + extra))
    print(f"base={len(rows)} selected={len(selected)}", flush=True)

    per_pair: dict[str, list[dict]] = {}
    for r in selected:
        pair = f"{r['index']['scene']}:{str(r['trajectory_metadata']['gesture']).zfill(2)}"
        per_pair.setdefault(pair, []).append(r)
    out_rows = []
    diagnostics = []
    rng = np.random.default_rng(args.seed)
    for pair, group in sorted(per_pair.items()):
        obj_type = pair.split(":")[0]
        vertices = object_collision_vertices(obj_type)
        _mujoco, _model = compile_model(
            ServoConfig(), object_type=obj_type, hand_side="right",
            physics_timestep=1.0 / 480.0,
        )
        joint_lower = np.asarray(_model.jnt_range[:JOINT_DOF, 0], dtype=np.float64)
        joint_upper = np.asarray(_model.jnt_range[:JOINT_DOF, 1], dtype=np.float64)
        cells = [(a, b, c) for a in range(5) for b in range(3) for c in range(2)]
        for idx, r in enumerate(group):
            obj = r["index"]["scene"]
            if args.anchor_mode == "first-clear":
                off = first_clear_offset(r, obj)
            else:
                off = args.fixed_offset
            if off is None:
                diagnostics.append({"pair": pair, "source_uuid": r["index"]["uuid"], "excluded": "no_clear_anchor"})
                continue
            end = int(r["trajectory_metadata"]["trajectory_info"]["object_move"][0]["end_frame"])
            anchor = end + off
            object_world = np.asarray(r["objects"][0]["pos"][anchor], dtype=np.float64)
            object_quat = Rotation.from_rotvec(
                np.asarray(r["objects"][0]["rot_aa"][anchor], dtype=np.float64)
            ).as_quat()
            row = None
            last_diag: dict = {}
            for attempt in range(1 + max(0, args.retry_clearance)):
                cell = cells[(idx + attempt) % len(cells)]
                target = _cell_target(pair, cell, rng)
                row, last_diag = build_retreat_row(
                    r,
                    anchor_offset=off,
                    target=target,
                    object_world=object_world,
                    object_quat_xyzw=object_quat,
                    vertices_local=vertices,
                    joint_lower=joint_lower,
                    joint_upper=joint_upper,
                    cell=cell,
                    attempt_seed=args.seed + idx * 1000 + attempt,
                    batch_tag=args.batch_tag,
                )
                if row is not None:
                    last_diag["attempts"] = attempt + 1
                    break
                if last_diag.get("excluded") != "hand_object_clearance":
                    break
            last_diag.setdefault("pair", pair)
            last_diag.setdefault("source_uuid", r["index"]["uuid"])
            last_diag.setdefault("anchor_offset", off)
            last_diag.setdefault("attempts", 1)
            diagnostics.append(last_diag)
            if row is not None:
                out_rows.append(row)

    accepted = len(out_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_compact_lance_stream(out_rows, output=args.output, replace=True, batch_size=64)
    plan_digest = _canonical_digest(
        {
            "contract": f"manorl_retreat_extension_v1_{args.batch_tag}_kinematic",
            "source": str(args.source),
            "row_filter": args.row_filter,
            "anchor_mode": args.anchor_mode,
            "anchor_offsets": list(ANCHOR_OFFSETS),
            "selected": len(selected),
            "accepted": accepted,
            "excluded": len(diagnostics) - accepted,
            "retreat_steps": RETREAT_STEPS,
            "cell_shape": list(CELL_SHAPE),
            "clearance_m": CLEARANCE_M,
            "domain": JOINT_DOMAIN,
        }
    )
    summary = {
        "contract": f"manorl_retreat_extension_v1_{args.batch_tag}_kinematic",
        "source": str(args.source),
        "output": str(args.output),
        "base_rows": len(rows),
        "selected": len(selected),
        "accepted": accepted,
        "excluded": len(diagnostics) - accepted,
        "retreat_steps": RETREAT_STEPS,
        "cell_shape": list(CELL_SHAPE),
        "clearance_m": CLEARANCE_M,
        "anchor_offsets_used": {str(k): sum(1 for d in diagnostics if d.get("anchor_offset") == k) for k in ANCHOR_OFFSETS},
        "retry_distribution": {str(k): sum(1 for d in diagnostics if d.get("attempts", 1) == k) for k in sorted({d.get("attempts", 1) for d in diagnostics})},
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
