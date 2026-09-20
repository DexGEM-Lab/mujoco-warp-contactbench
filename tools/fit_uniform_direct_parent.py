#!/usr/bin/env python3
"""Iteratively fit one self-contained U1 direct-target stream to a successful teacher trace.

Each iteration performs a fresh frame0 replay under the exact 534/H200 setting
(pyramidal, impratio1, native position actuators, one target held four substeps),
then applies bounded/smoothed iterative-learning corrections from measured hand
qpos error.  Objects are never forced and no state is written after frame0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from sim.manorl.local_contact_repair import MODEL_FIELDS, array_sha, dump, evaluate
from tools.pilot_start_augmentation import reconstruct
from tools.run_uniform_direct_parent_canary import compile_model, replay_direct, target_stream


def wrap_rotations(error: np.ndarray) -> np.ndarray:
    out = np.asarray(error, dtype=np.float64).copy()
    out[:, 3:6] = np.arctan2(np.sin(out[:, 3:6]), np.cos(out[:, 3:6]))
    return out


def smooth_time(values: np.ndarray, passes: int = 2) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    for _ in range(passes):
        padded = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        out = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    return out


def target_correction(
    teacher_qpos: np.ndarray,
    replay_qpos: np.ndarray,
    *,
    translation_gain: float,
    rotation_gain: float,
    finger_gain: float,
) -> np.ndarray:
    teacher = np.asarray(teacher_qpos, dtype=np.float64)[:, :28]
    replay = np.asarray(replay_qpos, dtype=np.float64)[:, :28]
    if teacher.shape != replay.shape or teacher.ndim != 2:
        raise ValueError("teacher/replay hand shape mismatch")
    error = smooth_time(wrap_rotations(teacher - replay))
    scale = np.r_[np.full(3, translation_gain), np.full(3, rotation_gain), np.full(22, finger_gain)]
    correction = error * scale
    correction[:, :3] = np.clip(correction[:, :3], -0.03, 0.03)
    correction[:, 3:6] = np.clip(correction[:, 3:6], -np.deg2rad(5), np.deg2rad(5))
    correction[:, 6:] = np.clip(correction[:, 6:], -np.deg2rad(4), np.deg2rad(4))
    correction[0] = 0.0
    return correction


def metrics_for(inp, model, trace, teacher, iteration: int, target: np.ndarray) -> tuple[dict, dict]:
    physical, diagnostic = evaluate(model, inp, trace)
    tq = np.asarray(teacher["qpos"], dtype=np.float64)
    tobj = np.asarray(teacher["actual_object_pos"], dtype=np.float64)
    hand_root = np.linalg.norm(trace["qpos"][:, :3] - tq[:, :3], axis=1)
    hand_all = np.max(np.abs(wrap_rotations(tq[:, :28] - trace["qpos"][:, :28])), axis=1)
    obj = np.linalg.norm(trace["actual_object_pos"] - tobj, axis=2)
    summary = {
        "iteration": iteration,
        "physical_pose_pass": bool(physical["physical_pose_pass"]),
        "functional_pass": bool(physical["functional_pass"]),
        "failed_gates": list(physical["failed_gates"]),
        "max_hand_root_position_error_m": float(hand_root.max()),
        "max_hand_dof_error": float(hand_all.max()),
        "max_object_position_error_m": float(obj.max()),
        "final_object_position_error_m": float(obj[-1].max()),
        "target_sha256": __import__("hashlib").sha256(target.tobytes()).hexdigest(),
        "physical": physical,
    }
    return summary, diagnostic


def run(args: argparse.Namespace) -> None:
    root = args.output / args.row
    root.mkdir(parents=True, exist_ok=False)
    inp, _parent, _fingers, teacher, provenance = reconstruct(args.bundle, args.row)
    _assets, model = compile_model(args.bundle, args.asset_root, inp, "U1")
    target = target_stream(teacher, args.initial_kind)
    lower, upper = model.jnt_range[:28].T
    history = []
    start = time.monotonic()

    for iteration in range(args.iterations + 1):
        folder = root / f"iter{iteration:02d}"
        folder.mkdir()
        trace = replay_direct(inp, model, target)
        summary, diagnostic = metrics_for(inp, model, trace, teacher, iteration, target)
        history.append(summary)
        np.save(folder / "target.npy", target)
        np.savez_compressed(folder / "trace.npz", **trace)
        np.savez_compressed(folder / "contact_validation.npz", **diagnostic["arrays"])
        dump(folder / "result.json", summary)
        dump(root / "progress.json", {
            "row": args.row,
            "setting": "U1",
            "initial_kind": args.initial_kind,
            "history": history,
            "elapsed_s": time.monotonic() - start,
            "source_provenance": provenance,
            "model_array_hashes": {k: array_sha(getattr(model, k)) for k in MODEL_FIELDS},
        })
        print(json.dumps({
            "row": args.row,
            "iteration": iteration,
            "pass": summary["physical_pose_pass"],
            "max_hand_root_m": summary["max_hand_root_position_error_m"],
            "max_object_m": summary["max_object_position_error_m"],
            "failed": summary["failed_gates"],
        }, ensure_ascii=False), flush=True)
        if summary["physical_pose_pass"]:
            dump(root / "result.json", {
                "state": "accepted",
                "accepted_iteration": iteration,
                "setting": "U1",
                "initial_kind": args.initial_kind,
                "history": history,
                "elapsed_s": time.monotonic() - start,
            })
            return
        if iteration == args.iterations:
            break
        correction = target_correction(
            teacher["qpos"], trace["qpos"],
            translation_gain=args.translation_gain,
            rotation_gain=args.rotation_gain,
            finger_gain=args.finger_gain,
        )
        target = np.clip(target + correction, lower, upper)
        target[0] = target_stream(teacher, args.initial_kind)[0]

    dump(root / "result.json", {
        "state": "not_accepted",
        "setting": "U1",
        "initial_kind": args.initial_kind,
        "history": history,
        "elapsed_s": time.monotonic() - start,
    })
    raise SystemExit(3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--row", required=True)
    p.add_argument("--initial-kind", choices=("first", "mean", "last", "teacher", "teacher-lead1"), default="mean")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--translation-gain", type=float, default=0.8)
    p.add_argument("--rotation-gain", type=float, default=0.5)
    p.add_argument("--finger-gain", type=float, default=0.5)
    args = p.parse_args()
    if args.iterations < 1 or any(x <= 0 for x in (args.translation_gain, args.rotation_gain, args.finger_gain)):
        p.error("positive gains and at least one iteration required")
    run(args)


if __name__ == "__main__":
    main()
