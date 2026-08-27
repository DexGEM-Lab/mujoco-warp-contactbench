#!/usr/bin/env python3
"""Merge kinematic retreat rows into their base rows (option A: tail replacement).

For each retreat row, locate its base row by seed_uuid and the anchor frame by
exact frame0 match. The merged row is base[0..anchor] + retreat[1..RETREAT_STEPS).
All fields are stitched consistently: hands, objects, contact, reference,
command mappings, timestamp, trajectory_metadata, provenance. Base rows without
a retreat row pass through unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sim.manorl.lance_v2 import write_compact_lance_stream

RETREAT_STEPS = 30
ANCHOR_OFFSETS = (30, 40, 50, 75)


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _find_anchor(base: dict, frame0: list) -> int:
    right = base["trajectory_metadata"]["hand_slots"].index("right")
    end = int(base["trajectory_metadata"]["trajectory_info"]["object_move"][0]["end_frame"])
    q0 = np.asarray(frame0, dtype=np.float64)
    for off in ANCHOR_OFFSETS:
        i = end + off
        if i >= len(base["hands"][right]["urdf_dof"]):
            continue
        if np.array_equal(np.asarray(base["hands"][right]["urdf_dof"][i], dtype=np.float64), q0):
            return i
    raise ValueError(f"retreat frame0 has no exact base anchor match")


def merge_row(base: dict, retreat: dict, anchor: int) -> dict:
    """base[0..anchor] + retreat[1..29]."""
    base_T = int(base["trajectory_metadata"]["total_frames"])
    if not 0 <= anchor < base_T:
        raise ValueError(f"anchor {anchor} outside base {base_T}")
    ret_T = int(retreat["trajectory_metadata"]["total_frames"])
    if ret_T != RETREAT_STEPS:
        raise ValueError(f"retreat must be {RETREAT_STEPS} frames, got {ret_T}")

    def stitch(base_val, retreat_val):
        # base_val length >= anchor+1 (trajectory arrays); take base[:anchor+1] + retreat[1:]
        b = np.asarray(base_val)
        r = np.asarray(retreat_val)
        return np.concatenate([b[: anchor + 1], r[1:]], axis=0)

    right_b = base["trajectory_metadata"]["hand_slots"].index("right")
    right_r = retreat["trajectory_metadata"]["hand_slots"].index("right")
    merged_T = anchor + 1 + (ret_T - 1)

    # hands (list of 2 dicts: right, left-empty)
    hands = []
    for side in range(2):
        bh = base["hands"][side]
        rh = retreat["hands"][side]
        if side == right_b:
            hands.append(
                {
                    "hand_name": rh.get("hand_name"),
                    "mano_global_pos": stitch(bh["mano_global_pos"], rh["mano_global_pos"]).tolist(),
                    "mano_global_rot_aa": stitch(bh["mano_global_rot_aa"], rh["mano_global_rot_aa"]).tolist(),
                    "mano_hand_pose": stitch(bh["mano_hand_pose"], rh["mano_hand_pose"]).tolist(),
                    "mano_joint_pos": stitch(bh["mano_joint_pos"], rh["mano_joint_pos"]).tolist(),
                    "urdf_dof": stitch(bh["urdf_dof"], rh["urdf_dof"]).tolist(),
                    "urdf_dof_target": stitch(bh["urdf_dof_target"], rh["urdf_dof_target"]).tolist(),
                }
            )
        else:
            hands.append(dict(bh))
    # objects (single object)
    objects = [
        {
            "pos": stitch(base["objects"][0]["pos"], retreat["objects"][0]["pos"]).tolist(),
            "rot_aa": stitch(base["objects"][0]["rot_aa"], retreat["objects"][0]["rot_aa"]).tolist(),
        }
    ]
    # contact: base[:anchor+1] + retreat[1:]
    contact = list(base["contact"][: anchor + 1]) + list(retreat["contact"][1:])
    # reference
    reference = {
        "source_frame_index": stitch(base["reference"]["source_frame_index"], retreat["reference"]["source_frame_index"]).astype(np.int64).tolist(),
        "hand_urdf_dof": stitch(base["reference"]["hand_urdf_dof"], retreat["reference"]["hand_urdf_dof"]).tolist(),
        "object_pos": stitch(base["reference"]["object_pos"], retreat["reference"]["object_pos"]).tolist(),
        "object_rot_aa": stitch(base["reference"]["object_rot_aa"], retreat["reference"]["object_rot_aa"]).tolist(),
    }
    # command mappings: base commands are length base_T-1; retreat commands length ret_T-1.
    # base commands up to transition at anchor: base[:anchor] (transitions 0..anchor-1 lead to frames 1..anchor)
    # then retreat transitions: retreat commands[0:] map to retreat frames 1..29 relative to merged frame anchor+1...
    # Simplify: merged transitions = (anchor) base transitions + (ret_T-1) retreat transitions.
    base_cmd_ref = np.asarray(base["command_reference_index"][:anchor], dtype=np.int64)
    base_cmd_src = np.asarray(base["command_source_frame_index"][:anchor], dtype=np.int64)
    ret_cmd_ref = np.asarray(retreat["command_reference_index"], dtype=np.int64)
    ret_cmd_src = np.asarray(retreat["command_source_frame_index"], dtype=np.int64)
    cmd_ref = np.concatenate([base_cmd_ref, ret_cmd_ref])
    cmd_src = np.concatenate([base_cmd_src, ret_cmd_src])
    # timestamps: base timestamps[:anchor+1] then continue at anchor dt
    base_ts = np.asarray(base["timestamp"], dtype=np.float64)
    ret_ts = np.asarray(retreat["timestamp"], dtype=np.float64)
    dt = base_ts[1] - base_ts[0] if len(base_ts) > 1 else 1.0 / 120.0
    ts = np.concatenate([base_ts[: anchor + 1], base_ts[anchor] + dt * np.arange(1, ret_T, dtype=np.float64)])
    # provenance: base provenance + retreat augmentation marker
    prov = dict(base["provenance"])
    prov["augmentation_identity"] = retreat["provenance"].get("augmentation_identity") or prov.get("augmentation_identity")
    tm = dict(base["trajectory_metadata"])
    tm["total_frames"] = merged_T
    tm["trajectory_info"] = {
        "object_move": [
            {
                "object_name": base["index"]["scene"],
                "start_frame": int(base["trajectory_metadata"]["trajectory_info"]["object_move"][0]["start_frame"]),
                "end_frame": anchor,
            }
        ]
    }
    row = {
        "index": dict(base["index"]),
        "trajectory_metadata": tm,
        "timestamp": ts.tolist(),
        "hands": hands,
        "objects": objects,
        "contact": contact,
        "reference": reference,
        "command_reference_index": cmd_ref.tolist(),
        "command_source_frame_index": cmd_src.tolist(),
        "provenance": prov,
    }
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--retreat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import lance

    base_rows = lance.dataset(str(args.base)).scanner(batch_size=8, scan_in_order=True).to_table().to_pylist()
    ret_rows = lance.dataset(str(args.retreat)).scanner(batch_size=64).to_table().to_pylist()
    base_by_uuid = {r["index"]["uuid"]: r for r in base_rows}
    ret_by_seed: dict[str, list] = {}
    for r in ret_rows:
        ret_by_seed.setdefault(r["index"]["seed_uuid"], []).append(r)
    # validate 1:1
    multi = {k: v for k, v in ret_by_seed.items() if len(v) != 1}
    if multi:
        raise ValueError(f"retreat rows not 1:1 with base: {list(multi)[:5]}")
    orphan = [s for s in ret_by_seed if s not in base_by_uuid]
    if orphan:
        raise ValueError(f"retreat seed_uuid not in base: {orphan[:5]}")

    merged = []
    for base in base_rows:
        if base["index"]["uuid"] in ret_by_seed:
            retreat = ret_by_seed[base["index"]["uuid"]][0]
            anchor = _find_anchor(base, retreat["hands"][retreat["trajectory_metadata"]["hand_slots"].index("right")]["urdf_dof"][0])
            merged.append(merge_row(base, retreat, anchor))
        else:
            merged.append(base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_compact_lance_stream(merged, output=args.output, replace=True, batch_size=64)
    n_merged = sum(1 for r in merged if r["trajectory_metadata"]["total_frames"] > base_rows[0]["trajectory_metadata"]["total_frames"] or True)
    print(json.dumps({"output": str(args.output), "rows": len(merged), "with_retreat": sum(1 for b in base_rows if b["index"]["uuid"] in ret_by_seed), "plain": sum(1 for b in base_rows if b["index"]["uuid"] not in ret_by_seed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
