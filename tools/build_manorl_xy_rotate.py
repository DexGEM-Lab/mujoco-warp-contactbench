#!/usr/bin/env python3
"""Rotate whole-scene rows around the object center about world Z.

For each selected base row, sample one rotation angle theta in
[-range_deg, +range_deg] and apply it about the per-frame object position
about the world Z axis: object keeps its position (it rotates in place),
object rot_aa rotates by theta, right hand rotates around the object center,
reference and contact world-frame positions/forces rotate, local-frame
positions (pos_wrist/pos_joint/pos_object), fingers, timestamps and all
other fields are untouched. Relative hand-object geometry is preserved.

Stratified per object-action pair, same scheme as the XY translate tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.lance_v2 import write_compact_lance_stream


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def pair_of(row: dict) -> str:
    return f"{row['index']['scene']}:{str(row['trajectory_metadata']['gesture']).zfill(2)}"


def _rz(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotate_rotvec(v: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate axis-angle vectors: v' = log(R @ Rodrigues(v)) with R = Rz.

    Matrix composition: new = R @ mats (apply the object's own orientation
    first, then rotate about the world Z axis). Row-vector convention for
    positions uses @ R.T which equals R @ x for column vectors; rotation
    matrices compose as R @ mats, NOT mats @ R.T (= R @ Rz(-theta))."""
    mats = Rotation.from_rotvec(v).as_matrix()
    return Rotation.from_matrix(R @ mats).as_rotvec()


def rotate_row(row: dict, theta_deg: float) -> dict:
    """Rotate the whole scene by theta about world Z around the per-frame
    object center. Representations handled separately:

    - positions (hand XYZ, contact pos_world): p' = Rz(p - o) + o
    - object orientation (objects.rot_aa / reference.object_rot_aa,
      axis-angle rotvec): v' = log(Rz @ Rodrigues(v))
    - hand orientation (urdf_dof[:,3:6] extrinsic-XYZ euler AND its
      mano_global_rot_aa axis-angle mirror): R' = Rz @ R_euler, written back
      in the same extrinsic-XYZ order and as the new rotvec
    - force vectors (force_normal / total_force_world): f' = Rz f
    - local-frame values (pos_wrist/joint/object, pos_joint), fingers,
      timestamps: unchanged
    """
    theta = math.radians(theta_deg)
    R = _rz(theta)
    out = dict(row)

    obj_pos = np.asarray(row["objects"][0]["pos"], dtype=np.float64)  # (T,3)

    hands = []
    for side, h in enumerate(row["hands"]):
        h = dict(h)
        if h.get("hand_name") == "right" or side == 0:
            q = np.asarray(h["urdf_dof"], dtype=np.float64).copy()
            # 1) wrist translation around object center
            q[:, :3] = ((q[:, :3] - obj_pos) @ R.T) + obj_pos
            # 2) wrist orientation: extrinsic-XYZ euler -> matrix -> Rz @ R -> euler (same order)
            euler = q[:, 3:6]
            mats = Rotation.from_euler("XYZ", euler).as_matrix()
            new_mats = R @ mats  # Rz @ R: own orientation then world-Z
            q[:, 3:6] = Rotation.from_matrix(new_mats).as_euler("XYZ")
            h["urdf_dof"] = q.tolist()
            h["mano_global_pos"] = q[:, :3].tolist()
            # 3) axis-angle mirror of the rotated wrist orientation
            h["mano_global_rot_aa"] = Rotation.from_matrix(new_mats).as_rotvec().tolist()
            # 4) controller target 同样旋转 (同构 28D: 位置绕物体中心, 欧拉 Rz@R)
            if h.get("urdf_dof_target"):
                qt = np.asarray(h["urdf_dof_target"], dtype=np.float64).copy()
                qt[:, :3] = ((qt[:, :3] - obj_pos) @ R.T) + obj_pos
                t_mats = Rotation.from_euler("XYZ", qt[:, 3:6]).as_matrix()
                new_t_mats = R @ t_mats
                qt[:, 3:6] = Rotation.from_matrix(new_t_mats).as_euler("XYZ")
                h["urdf_dof_target"] = qt.tolist()
        hands.append(h)
    out["hands"] = hands

    objects = []
    for o in row["objects"]:
        o = dict(o)
        o["rot_aa"] = _rotate_rotvec(np.asarray(o["rot_aa"], dtype=np.float64), R).tolist()
        objects.append(o)
    out["objects"] = objects

    ref = dict(row["reference"])
    ref_obj = np.asarray(ref["object_pos"], dtype=np.float64)
    rh = np.asarray(ref["hand_urdf_dof"], dtype=np.float64).copy()
    rh[:, :3] = ((rh[:, :3] - ref_obj) @ R.T) + ref_obj
    # reference hand orientation is the same extrinsic-XYZ euler contract
    ref_mats = Rotation.from_euler("XYZ", rh[:, 3:6]).as_matrix()
    new_ref_mats = R @ ref_mats  # Rz @ R
    rh[:, 3:6] = Rotation.from_matrix(new_ref_mats).as_euler("XYZ")
    ref["hand_urdf_dof"] = rh.tolist()
    ref["object_rot_aa"] = _rotate_rotvec(np.asarray(ref["object_rot_aa"], dtype=np.float64), R).tolist()
    out["reference"] = ref

    contact = []
    for frame_idx, frame in enumerate(row["contact"]):
        center = obj_pos[frame_idx]
        new_frame = []
        for entry in frame or []:
            entry = dict(entry)
            if entry.get("total_force_world"):
                entry["total_force_world"] = (
                    np.asarray(entry["total_force_world"], dtype=np.float64) @ R.T
                ).tolist()
            pairs = []
            for p_ in entry.get("contact_pairs") or []:
                p_ = dict(p_)
                if p_.get("pos_world"):
                    w = np.asarray(p_["pos_world"], dtype=np.float64)
                    p_["pos_world"] = ((w - center) @ R.T + center).tolist()
                if p_.get("force_normal"):
                    p_["force_normal"] = (
                        np.asarray(p_["force_normal"], dtype=np.float64) @ R.T
                    ).tolist()
                pairs.append(p_)
            entry["contact_pairs"] = pairs
            new_frame.append(entry)
        contact.append(new_frame)
    out["contact"] = contact

    prov = dict(row["provenance"])
    prov["augmentation_identity"] = f"manorl_xy_rotate_v2:theta_deg={theta_deg:.4f}"
    out["provenance"] = prov
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=str, default="banana:02,banana:18")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--range-deg", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    import lance

    wanted = set(args.pairs.split(","))
    rows = lance.dataset(str(args.source)).scanner(batch_size=64).to_table().to_pylist()
    by_pair: dict[str, list[dict]] = {}
    for r in rows:
        p = pair_of(r)
        if p in wanted:
            by_pair.setdefault(p, []).append(r)
    print("available per pair:", {k: len(v) for k, v in by_pair.items()}, flush=True)

    rng = np.random.default_rng(args.seed)
    n = args.count if args.count > 0 else len(rows)
    # 1D Latin hypercube over [-range, +range]
    strata = (np.arange(n, dtype=np.float64) + rng.uniform(0, 1, size=n)) / n
    thetas = -args.range_deg + 2 * args.range_deg * strata[rng.permutation(n)]
    if n <= len(rows):
        row_idx = rng.choice(len(rows), size=n, replace=False)
    else:
        row_idx = np.arange(len(rows)) % len(rows)
    out_rows = []
    diagnostics = []
    for i, idx in enumerate(row_idx):
        theta = float(thetas[i])
        row = rotate_row(rows[int(idx)], theta)
        out_rows.append(row)
        diagnostics.append(
            {
                "pair": pair_of(rows[int(idx)]),
                "source_uuid": rows[int(idx)]["index"]["uuid"],
                "theta_deg": theta,
                "new_uuid": row["index"]["uuid"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_compact_lance_stream(out_rows, output=args.output, replace=True, batch_size=64)
    thetas = np.asarray([d["theta_deg"] for d in diagnostics])
    summary = {
        "contract": "manorl_xy_rotate_v3",
        "source": str(args.source),
        "output": str(args.output),
        "pairs": sorted(wanted),
        "count": len(out_rows),
        "range_deg": args.range_deg,
        "theta_range_actual": [float(thetas.min()), float(thetas.max())],
        "per_pair": {k: sum(1 for d in diagnostics if d["pair"] == k) for k in sorted(set(d["pair"] for d in diagnostics))},
        "seed": args.seed,
        "digest": _canonical_digest(
            {
                "contract": "manorl_xy_rotate_v3",
                "source": str(args.source),
                "count": len(out_rows),
                "range_deg": args.range_deg,
                "per_pair": {k: sum(1 for d in diagnostics if d["pair"] == k) for k in sorted(set(d["pair"] for d in diagnostics))},
            }
        ),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
