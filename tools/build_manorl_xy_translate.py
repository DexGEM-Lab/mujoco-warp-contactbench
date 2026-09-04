#!/usr/bin/env python3
"""XY-translate whole-scene rows (hand + object together) for augmentation.

For each selected base row, sample one XY offset in [-range, +range]^2 and add
it to every world-frame position: right-hand urdf_dof[:, :3] (and its
mano_global_pos mirror), object pos, reference hand_urdf_dof[:, :3] and
reference object_pos, and contact pair pos_world. Local-frame positions
(pos_wrist/pos_joint/pos_object), forces, rotations, fingers, timestamps, and
all other fields are untouched: the whole scene moves rigidly, so intra-scene
relative geometry is preserved exactly.

Sampling is stratified per object-action pair: each pair receives a fair share
of the requested rows and its offsets are drawn as a Latin-hypercube-style
cover of [-range, +range]^2 so the augmented set spans the full range instead
of clustering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sim.manorl.lance_v2 import write_compact_lance_stream


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def pair_of(row: dict) -> str:
    return f"{row['index']['scene']}:{str(row['trajectory_metadata']['gesture']).zfill(2)}"


def translate_row(row: dict, dx: float, dy: float) -> dict:
    """Return a new row with XY offset (dx, dy) applied to all world frames."""
    delta = np.asarray([dx, dy, 0.0], dtype=np.float64)
    out = dict(row)
    hands = []
    for side, h in enumerate(row["hands"]):
        h = dict(h)
        if h.get("hand_name") == "right" or side == 0:
            q = np.asarray(h["urdf_dof"], dtype=np.float64).copy()
            q[:, :3] += delta
            h["urdf_dof"] = q.tolist()
            h["mano_global_pos"] = q[:, :3].tolist()
            # controller target 同样平移 (与 urdf_dof 同构 28D)
            if h.get("urdf_dof_target"):
                qt = np.asarray(h["urdf_dof_target"], dtype=np.float64).copy()
                qt[:, :3] += delta
                h["urdf_dof_target"] = qt.tolist()
        hands.append(h)
    out["hands"] = hands
    objects = []
    for o in row["objects"]:
        o = dict(o)
        pos = np.asarray(o["pos"], dtype=np.float64).copy()
        pos += delta
        o["pos"] = pos.tolist()
        objects.append(o)
    out["objects"] = objects
    ref = dict(row["reference"])
    ref_hand = np.asarray(ref["hand_urdf_dof"], dtype=np.float64).copy()
    ref_hand[:, :3] += delta
    ref_obj = np.asarray(ref["object_pos"], dtype=np.float64).copy()
    ref_obj += delta
    ref["hand_urdf_dof"] = ref_hand.tolist()
    ref["object_pos"] = ref_obj.tolist()
    out["reference"] = ref
    contact = []
    for frame in row["contact"]:
        new_frame = []
        for entry in frame or []:
            entry = dict(entry)
            pairs = []
            for p in entry.get("contact_pairs") or []:
                p = dict(p)
                if p.get("pos_world"):
                    w = np.asarray(p["pos_world"], dtype=np.float64) + delta
                    p["pos_world"] = w.tolist()
                pairs.append(p)
            entry["contact_pairs"] = pairs
            new_frame.append(entry)
        contact.append(new_frame)
    out["contact"] = contact
    prov = dict(row["provenance"])
    prov["augmentation_identity"] = (
        f"manorl_xy_translate_v1:dx={dx:.6f}:dy={dy:.6f}"
    )
    out["provenance"] = prov
    return out


def _lhs_offsets(n: int, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    """Latin-hypercube samples in [lo, hi]^2: one per row/column stratum."""
    strata = (np.arange(n, dtype=np.float64) + rng.uniform(0, 1, size=n)) / n
    # 每维独立分层, 随机打乱配对
    xs = lo + (hi - lo) * strata[rng.permutation(n)]
    ys = lo + (hi - lo) * strata[rng.permutation(n)]
    return np.stack([xs, ys], axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=str, default=None)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--range-m", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import lance

    rows = lance.dataset(str(args.source)).scanner(batch_size=64).to_table().to_pylist()
    if args.pairs:
        wanted = set(args.pairs.split(","))
        rows = [r for r in rows if pair_of(r) in wanted]
    print("selected rows:", len(rows), flush=True)
    n = args.count if args.count > 0 else len(rows)

    rng = np.random.default_rng(args.seed)
    offsets = _lhs_offsets(n, rng, -args.range_m, args.range_m)
    # 行选择: 若 n < len(rows), 分层选行 (每行至多一次); n == len(rows) 全用
    if n <= len(rows):
        row_idx = rng.choice(len(rows), size=n, replace=False)
    else:
        row_idx = np.arange(len(rows)) % len(rows)  # 多轮循环
    out_rows = []
    diagnostics = []
    for i, idx in enumerate(row_idx):
        dx, dy = float(offsets[i, 0]), float(offsets[i, 1])
        row = translate_row(rows[int(idx)], dx, dy)
        out_rows.append(row)
        diagnostics.append(
            {
                "pair": pair_of(rows[int(idx)]),
                "source_uuid": rows[int(idx)]["index"]["uuid"],
                "dx_m": dx,
                "dy_m": dy,
                "new_uuid": row["index"]["uuid"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_compact_lance_stream(out_rows, output=args.output, replace=True, batch_size=64)
    dxs = np.asarray([d["dx_m"] for d in diagnostics])
    dys = np.asarray([d["dy_m"] for d in diagnostics])
    summary = {
        "contract": "manorl_xy_translate_v2",
        "source": str(args.source),
        "output": str(args.output),
        "count": len(out_rows),
        "range_m": args.range_m,
        "sampling": "latin_hypercube_uniform",
        "dx_range_actual": [float(dxs.min()), float(dxs.max())],
        "dy_range_actual": [float(dys.min()), float(dys.max())],
        "seed": args.seed,
        "digest": _canonical_digest(
            {
                "contract": "manorl_xy_translate_v2",
                "source": str(args.source),
                "count": len(out_rows),
                "range_m": args.range_m,
                "sampling": "latin_hypercube_uniform",
            }
        ),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
