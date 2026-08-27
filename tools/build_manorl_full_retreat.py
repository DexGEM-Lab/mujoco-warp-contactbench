#!/usr/bin/env python3
"""Build full-length retreat rows offline (replaces the 29-frame batch).

For each base row, find the true last solved hand-object contact frame from
the persisted contact array, then anchor = last_contact + 15. The entire tail
after the anchor (155-251 frames) is regenerated in wrist XYZ only, using the
historical retreat semantics: base XY direction from anchor to the original
final wrist, plus extra horizontal 3-15 cm, +/-30 deg, and Z +4-10 cm, applied
through the smoothstep tail deformation (first two tail frames untouched,
last three reach the endpoint). Length, timestamps, fingers, object, contact
and reference are preserved from the base row. Output rows are base rows with
only right-wrist XYZ replaced after the anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.contracts import JOINT_DOF
from sim.manorl.lance_v2 import write_compact_lance_stream

THRESHOLD_N = 0.2
ANCHOR_OFFSET = 15
MIN_TAIL_FRAMES = 10
RETREAT_CONFIG = {
    "minimum_extra_horizontal_m": 0.03,
    "maximum_extra_horizontal_m": 0.15,
    "maximum_xy_offset_deg": 30.0,
    "minimum_z_offset_m": 0.04,
    "maximum_z_offset_m": 0.10,
}


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _identity_seed(seed: int, identity: str) -> int:
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def last_contact_frame(row: dict, object_name: str) -> int | None:
    """Last frame with solved right-hand/target-object normal force > 0.2 N."""
    for i in range(len(row["contact"]) - 1, -1, -1):
        frame = row["contact"][i]
        for entry in frame or []:
            if entry.get("hand_name") != "right" or entry.get("object_name") != object_name:
                continue
            for pair in entry.get("contact_pairs") or []:
                f = np.asarray(pair.get("force_normal") or (), dtype=np.float32)
                if f.shape == (3,) and np.linalg.norm(f) > np.float32(THRESHOLD_N):
                    return i
    return None


def contact_at(row: dict, object_name: str, index: int) -> bool:
    frame = row["contact"][index]
    for entry in frame or []:
        if entry.get("hand_name") != "right" or entry.get("object_name") != object_name:
            continue
        for pair in entry.get("contact_pairs") or []:
            f = np.asarray(pair.get("force_normal") or (), dtype=np.float32)
            if f.shape == (3,) and np.linalg.norm(f) > np.float32(THRESHOLD_N):
                return True
    return False


def _deform_retreat_tail(source_tail: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Preserve source retreat while smoothly moving its endpoint (historical).

    Weight is zero for the first two frames and one for the last three.
    Endpoint displacement applied via quintic smoothstep.
    """
    source = np.asarray(source_tail, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or len(source) < 6:
        raise ValueError("retreat-tail deformation requires at least six XYZ frames")
    weights = np.zeros(len(source), dtype=np.float64)
    phase = np.linspace(0.0, 1.0, len(source) - 3, dtype=np.float64)
    weights[1:-2] = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    weights[-2:] = 1.0
    endpoint_displacement = np.asarray(end, dtype=np.float64) - source[-1]
    return source + weights[:, None] * endpoint_displacement[None]


def sample_endpoint(
    anchor_position: np.ndarray,
    original_end: np.ndarray,
    seed: int,
    identity: str,
) -> dict:
    """Historical retreat endpoint: original direction + extra horizontal/angle/z."""
    original_delta = original_end - anchor_position
    original_horizontal = float(np.linalg.norm(original_delta[:2]))
    if original_horizontal <= 1e-9:
        raise ValueError("original retreat tail has no nonzero horizontal direction")
    original_direction = math.atan2(float(original_delta[1]), float(original_delta[0]))
    rng = np.random.default_rng(_identity_seed(seed, f"{identity}|retreat"))
    extra_horizontal = float(rng.uniform(
        RETREAT_CONFIG["minimum_extra_horizontal_m"],
        RETREAT_CONFIG["maximum_extra_horizontal_m"],
    ))
    xy_offset_deg = float(rng.uniform(
        -RETREAT_CONFIG["maximum_xy_offset_deg"],
        RETREAT_CONFIG["maximum_xy_offset_deg"],
    ))
    extra_z = float(rng.uniform(
        RETREAT_CONFIG["minimum_z_offset_m"],
        RETREAT_CONFIG["maximum_z_offset_m"],
    ))
    end_horizontal = original_horizontal + extra_horizontal
    end_z_displacement = float(original_delta[2] + extra_z)
    sampled_direction = original_direction + math.radians(xy_offset_deg)
    end_position = anchor_position + np.asarray(
        (
            end_horizontal * math.cos(sampled_direction),
            end_horizontal * math.sin(sampled_direction),
            end_z_displacement,
        ),
        dtype=np.float64,
    )
    return {
        "original_horizontal_distance_m": original_horizontal,
        "extra_horizontal_offset_m": extra_horizontal,
        "end_horizontal_distance_m": end_horizontal,
        "original_z_displacement_m": float(original_delta[2]),
        "extra_z_offset_m": extra_z,
        "end_z_displacement_m": end_z_displacement,
        "original_direction_deg": math.degrees(original_direction),
        "xy_offset_deg": xy_offset_deg,
        "end_position_m": tuple(float(v) for v in end_position),
    }


def build_full_retreat_row(
    base: dict,
    *,
    anchor: int,
    seed: int,
    batch_tag: str,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> tuple[dict | None, dict]:
    right = base["trajectory_metadata"]["hand_slots"].index("right")
    T = int(base["trajectory_metadata"]["total_frames"])
    object_name = base["index"]["scene"]
    if not 2 <= anchor < T - 3:
        return None, {"excluded": "anchor_bounds", "anchor": anchor, "T": T}
    tail_len = T - anchor - 1
    if tail_len < MIN_TAIL_FRAMES:
        return None, {"excluded": "tail_too_short", "anchor": anchor, "tail_len": tail_len}

    q = np.asarray(base["hands"][right]["urdf_dof"], dtype=np.float64)  # (T, 28)
    anchor_position = q[anchor, :3]
    original_end = q[-1, :3]
    endpoint = sample_endpoint(anchor_position, original_end, seed, base["provenance"]["source_identity"])
    source_tail = q[anchor + 1 :, :3]
    retreat_position = _deform_retreat_tail(source_tail, np.asarray(endpoint["end_position_m"], dtype=np.float64))

    # 只改右腕 XYZ (dim 0..2), 其余维度与帧保持 base 原值
    q_new = q.copy()
    q_new[anchor + 1 :, :3] = retreat_position

    # 关节限位: 只查构造的 XYZ (手指维照抄 base, 不判)
    if np.any(q_new[anchor + 1 :, :3] < joint_lower[:3] - 1e-6) or np.any(
        q_new[anchor + 1 :, :3] > joint_upper[:3] + 1e-6
    ):
        return None, {"excluded": "joint_limits", "anchor": anchor}

    # splice 连续性与端点检查 (历史语义)
    dt = 1.0 / 120.0
    splice_velocity_error = float(
        np.linalg.norm((q_new[anchor + 1, :3] - q_new[anchor, :3]) / dt - (q[anchor + 1, :3] - q[anchor, :3]) / dt)
    )
    splice_accel_jump = float(
        np.linalg.norm(
            (q_new[anchor + 2, :3] - 2 * q_new[anchor + 1, :3] + q_new[anchor, :3]) / dt**2
            - (q[anchor + 2, :3] - 2 * q[anchor + 1, :3] + q[anchor, :3]) / dt**2
        )
    )

    row = dict(base)
    hands = []
    for side, h in enumerate(base["hands"]):
        h = dict(h)
        if side == right:
            h["urdf_dof"] = q_new.tolist()
            # mano_global_pos == urdf_dof[:, :3] (compact 语义)
            h["mano_global_pos"] = q_new[:, :3].tolist()
        hands.append(h)
    row["hands"] = hands
    prov = dict(base["provenance"])
    prov["augmentation_identity"] = f"manorl_retreat_extension_v2_full_length:{batch_tag}:kinematic"
    prov["seed"] = seed
    row["provenance"] = prov
    diag = {
        "anchor": anchor,
        "last_contact": anchor - ANCHOR_OFFSET,
        "tail_len": tail_len,
        "endpoint": endpoint,
        "splice_velocity_error_m_s": splice_velocity_error,
        "splice_acceleration_jump_m_s2": splice_accel_jump,
        "accepted": True,
    }
    return row, diag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-tag", type=str, default="v2")
    args = parser.parse_args(argv)

    import lance

    from sim.manorl.assets import compile_model, ServoConfig

    rows = lance.dataset(str(args.source)).scanner(batch_size=8, scan_in_order=True).to_table().to_pylist()
    # 每个物体类型编译一次模型取真实关节限位 (全量 1000+ 行共享同一模型)
    limits_by_object: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out_rows = []
    diagnostics = []
    for r in rows:
        obj = r["index"]["scene"]
        if obj not in limits_by_object:
            _mujoco, _model = compile_model(
                ServoConfig(), object_type=obj, hand_side="right",
                physics_timestep=1.0 / 480.0,
            )
            limits_by_object[obj] = (
                np.asarray(_model.jnt_range[:JOINT_DOF, 0], dtype=np.float64),
                np.asarray(_model.jnt_range[:JOINT_DOF, 1], dtype=np.float64),
            )
        joint_lower, joint_upper = limits_by_object[obj]
        last = last_contact_frame(r, obj)
        if last is None:
            diagnostics.append({"uuid": r["index"]["uuid"], "pair": f"{obj}:{str(r['trajectory_metadata']['gesture']).zfill(2)}", "excluded": "no_contact"})
            out_rows.append(r)
            continue
        anchor = last + ANCHOR_OFFSET
        T = int(r["trajectory_metadata"]["total_frames"])
        if anchor >= T - MIN_TAIL_FRAMES or contact_at(r, obj, anchor):
            diagnostics.append({"uuid": r["index"]["uuid"], "pair": f"{obj}:{str(r['trajectory_metadata']['gesture']).zfill(2)}", "excluded": "anchor_invalid", "last_contact": last, "T": T})
            out_rows.append(r)
            continue
        seed = int(r["provenance"]["seed"])
        row, diag = build_full_retreat_row(
            r, anchor=anchor, seed=seed, batch_tag=args.batch_tag,
            joint_lower=joint_lower, joint_upper=joint_upper,
        )
        diag["uuid"] = r["index"]["uuid"]
        diag["pair"] = f"{obj}:{str(r['trajectory_metadata']['gesture']).zfill(2)}"
        diag["last_contact"] = last
        diag["anchor"] = anchor
        diagnostics.append(diag)
        if row is not None:
            out_rows.append(row)
        else:
            out_rows.append(r)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_compact_lance_stream(out_rows, output=args.output, replace=True, batch_size=64)
    accepted = sum(1 for d in diagnostics if d.get("accepted"))
    plain = sum(1 for d in diagnostics if not d.get("accepted"))
    summary = {
        "contract": f"manorl_retreat_extension_v2_full_length_{args.batch_tag}",
        "source": str(args.source),
        "output": str(args.output),
        "base_rows": len(rows),
        "with_retreat": accepted,
        "plain": plain,
        "anchor_offset": ANCHOR_OFFSET,
        "retreat_config": RETREAT_CONFIG,
        "tail_len_med": float(np.median([d["tail_len"] for d in diagnostics if d.get("accepted")])),
        "tail_len_min": int(min(d["tail_len"] for d in diagnostics if d.get("accepted"))),
        "tail_len_max": int(max(d["tail_len"] for d in diagnostics if d.get("accepted"))),
        "excluded_reasons": {k: sum(1 for d in diagnostics if d.get("excluded") == k) for k in sorted({str(d.get("excluded")) for d in diagnostics})},
        "digest": _canonical_digest({"contract": f"manorl_retreat_extension_v2_full_length_{args.batch_tag}", "base_rows": len(rows), "with_retreat": accepted, "anchor_offset": ANCHOR_OFFSET}),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
