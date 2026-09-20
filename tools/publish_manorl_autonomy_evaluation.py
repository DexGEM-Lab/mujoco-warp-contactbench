#!/usr/bin/env python
"""Publish quantitative evaluation and a real MuJoCo actual/reference video.

This command consumes an already-saved learned rollout trace and checkpoint. It
never runs PPO, edits training history, or substitutes reference poses for
actual measurements. W&B publication is append-only to an existing finished
run selected with ``--wandb-run-id``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sim.manorl.assets import compile_model, object_runtime
from sim.manorl.contracts import JOINT_DOF
from sim.manorl.autonomy_telemetry import genuine_airborne_contact


def _quat_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    actual = actual / max(np.linalg.norm(actual), 1e-12)
    target = target / max(np.linalg.norm(target), 1e-12)
    radians = float(2.0 * np.arccos(np.clip(abs(float(np.dot(actual, target))), 0.0, 1.0)))
    return radians, float(np.degrees(radians))


def _longest(values: Iterable[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _frames(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows = trace.get("trace")
    if not isinstance(rows, list) or not rows:
        raise ValueError("trace must contain a non-empty trace list")
    return [row["info"] if isinstance(row, dict) and isinstance(row.get("info"), dict) else row for row in rows]


def summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Compute metrics with explicit frame denominators from actual trace data."""
    rows = _frames(trace)
    actual = np.asarray([r["object_position"] for r in rows], dtype=float)
    target = np.asarray([r["target_object_position"] for r in rows], dtype=float)
    actual_q = np.asarray([r["object_quaternion_xyzw"] for r in rows], dtype=float)
    target_q = np.asarray([r["target_object_quaternion_xyzw"] for r in rows], dtype=float)
    path = np.linalg.norm(actual - target, axis=1)
    orientation = np.asarray([_quat_error(a, b)[0] for a, b in zip(actual_q, target_q)], dtype=float)
    force_contact = []
    slip = []
    for row in rows:
        force = np.asarray(row.get("hand_object_force", []), dtype=float)
        force_contact.append(bool(force.size and np.linalg.norm(force.reshape(-1, 3), axis=1).max() > 0.02))
        motion = np.asarray(row.get("relative_contact_motion", []), dtype=float)
        slip.append(float(np.linalg.norm(motion.reshape(-1, 3), axis=1).mean()) if motion.size else 0.0)
    airborne = [genuine_airborne_contact(row) for row in rows]
    peak_lift = float(np.max(actual[:, 2]) - actual[0, 2])
    target_lift = float(np.max(target[:, 2]) - target[0, 2])
    reference_contact = np.asarray([np.mean(np.asarray(r.get("reference_surface_proximity", [0.0]))) for r in rows])
    contact_window_end = int(np.flatnonzero(reference_contact > 0.35)[-1] + 1) if np.any(reference_contact > 0.35) else 0
    release = [index >= contact_window_end and not contact for index, contact in enumerate(force_contact)]
    phase = str(rows[-1].get("failure_phase", "unknown"))
    # This predicate is intentionally diagnostic and is never task success.
    max_lift_endpose = bool(peak_lift >= max(0.0, target_lift - 0.02) and path[-1] < 0.03)
    identity = trace.get("identity")
    summary = {
        "schema": "manorl.autonomy.evaluation.v1",
        "identity": identity,
        "split_role": trace.get("split_role"),
        "checkpoint_format": trace.get("checkpoint_format"),
        "frames": len(rows),
        "path_rmse_m": float(np.sqrt(np.mean(path * path))),
        "path_error_final_m": float(path[-1]),
        "orientation_error_rad_mean": float(np.mean(orientation)),
        "orientation_error_rad_final": float(orientation[-1]),
        "orientation_error_degrees_mean": float(np.degrees(np.mean(orientation))),
        "orientation_error_degrees_final": float(np.degrees(orientation[-1])),
        "peak_lift_m": peak_lift,
        "target_peak_lift_m": target_lift,
        "airborne_contact_frames": int(sum(airborne)),
        "airborne_contact_denominator_frames": len(airborne),
        "sustained_airborne_contact_frames": _longest(airborne),
        "hand_object_contact_frames": int(sum(force_contact)),
        "hand_object_contact_denominator_frames": len(force_contact),
        "sustained_hand_object_contact_frames": _longest(force_contact),
        "slip_proxy_mean_mps": float(np.mean(slip)),
        "slip_proxy_contact_frames": float(np.mean([s for s, c in zip(slip, force_contact) if c])) if any(force_contact) else None,
        "release_frames": int(sum(release)),
        "release_denominator_frames": len(release),
        "max_lift_endpose_diagnostic": max_lift_endpose,
        "full_task_success": trace.get("full_task_success"),
        "reference_release_window_contactfree_frames": int(sum(release)),
        "reference_release_window_contactfree_denominator_frames": len(release),
        "failure_phase": phase,
        "episode_length_frames": len(rows),
        "episode_return": float(trace.get("return", np.nan)) if trace.get("return") is not None else None,
        "provenance": {"package": trace.get("package"), "split_role": trace.get("split_role"), "contracts": trace.get("contracts")},
    }
    summary["identity_table"] = [{"identity": identity, **{key: summary[key] for key in (
        "frames", "path_rmse_m", "orientation_error_rad_mean", "peak_lift_m", "target_peak_lift_m",
        "hand_object_contact_frames", "hand_object_contact_denominator_frames",
        "sustained_hand_object_contact_frames", "airborne_contact_frames",
        "airborne_contact_denominator_frames", "slip_proxy_mean_mps", "release_frames",
        "release_denominator_frames", "failure_phase", "full_task_success")}}]
    return summary


def _set_pose(mujoco: Any, model: Any, data: Any, q: np.ndarray, object_position: np.ndarray, quat_xyzw: np.ndarray, object_qpos: int) -> None:
    data.qpos[:JOINT_DOF] = q[:JOINT_DOF]
    data.qpos[object_qpos:object_qpos + 3] = object_position
    data.qpos[object_qpos + 3:object_qpos + 7] = quat_xyzw[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)


def render_video(trace: dict[str, Any], output: Path, *, source_fps: int = 120, fps: int = 30, width: int = 640, height: int = 480) -> Path:
    """Render source-clock poses at matched playback speed through one camera."""
    if source_fps < 1 or fps < 1 or source_fps % fps:
        raise ValueError("source_fps must be a positive multiple of output fps")
    stride = source_fps // fps
    import imageio.v2 as imageio
    import mujoco
    mujoco, model = compile_model(object_type="cube2", hand_side="right", physics_timestep=1 / 480)
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, object_runtime("cube2").free_joint_name)
    object_qpos = int(model.jnt_qposadr[object_joint])
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.07, -0.20, 0.15); camera.distance = 0.72
    camera.azimuth = 145.0; camera.elevation = -18.0
    model.vis.headlight.ambient = 0.4
    rows = _frames(trace)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        Image = ImageDraw = None
    with imageio.get_writer(output, fps=fps, codec="libx264", macro_block_size=1) as writer:
        for frame_index, row in enumerate(rows[::stride]):
            actual_q = np.asarray(row["qpos"], dtype=float)
            actual_pos = np.asarray(row["object_position"], dtype=float)
            actual_quat = np.asarray(row["object_quaternion_xyzw"], dtype=float)
            ref_q = np.asarray(row["reference_hand_q"], dtype=float)
            ref_pos = np.asarray(row["target_object_position"], dtype=float)
            ref_quat = np.asarray(row["target_object_quaternion_xyzw"], dtype=float)
            _set_pose(mujoco, model, data, actual_q, actual_pos, actual_quat, object_qpos)
            renderer.update_scene(data, camera=camera)
            actual_frame = renderer.render().copy()
            _set_pose(mujoco, model, data, ref_q, ref_pos, ref_quat, object_qpos)
            renderer.update_scene(data, camera=camera)
            reference_frame = renderer.render().copy()
            frame = np.concatenate((actual_frame, reference_frame), axis=1)
            if Image is not None:
                canvas = Image.fromarray(frame)
                draw = ImageDraw.Draw(canvas)
                elapsed = frame_index * stride / source_fps
                draw.rectangle((0, 0, 190, 28), fill=(0, 0, 0))
                draw.rectangle((width, 0, width + 210, 28), fill=(0, 0, 0))
                draw.text((8, 7), f"Actual  t={elapsed:.2f}s", fill=(255, 255, 255))
                draw.text((width + 8, 7), f"Reference  t={elapsed:.2f}s", fill=(255, 255, 255))
                frame = np.asarray(canvas)
            writer.append_data(frame)
    renderer.close()
    return output


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "format" not in payload:
        raise ValueError("checkpoint does not contain a recognized autonomy payload")
    return payload


def publish_wandb(summary: dict[str, Any], *, run_id: str, project: str, entity: str | None, json_path: Path, video_path: Path | None) -> None:
    import wandb
    api = wandb.Api()
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    existing = api.run(path)
    if str(getattr(existing, "state", "")).upper() != "FINISHED":
        raise RuntimeError("evaluation publication requires an existing FINISHED W&B run")
    run = wandb.init(project=project, entity=entity, id=run_id, resume="must", reinit=True)
    try:
        payload = {f"evaluation/{key}": value for key, value in summary.items() if isinstance(value, (int, float, bool)) and value is not None}
        # Omit step: W&B's resumed SDK run selects a monotonic history step.
        # The training axis is retained as data, never reused as the history step.
        payload["evaluation/checkpoint_transitions"] = int(summary.get("checkpoint_transitions", 0))
        if video_path is not None and callable(getattr(wandb, "Video", None)):
            payload["evaluation/video"] = wandb.Video(str(video_path), fps=30, format="mp4")
        if callable(getattr(wandb, "Table", None)):
            rows = summary.get("identity_table", [])
            columns = list(rows[0].keys()) if rows else ["identity"]
            payload["evaluation/identity_table"] = wandb.Table(columns=columns, data=[[row.get(column) for column in columns] for row in rows])
        run.log(payload)
        artifact = wandb.Artifact(f"autonomy-evaluation-{run_id}", type="manorl-evaluation")
        artifact.add_file(str(json_path), name=json_path.name)
        if video_path is not None: artifact.add_file(str(video_path), name=video_path.name)
        run.log_artifact(artifact)
    finally:
        run.finish()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manorl/contact_conditioned_autonomy/evaluation"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "mujoco-mano"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "sunjay45711-dexerto"))
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    checkpoint = _load_checkpoint(args.checkpoint)
    if trace.get("checkpoint_format") != checkpoint.get("format"):
        raise ValueError("trace/checkpoint format mismatch")
    summary = summarize_trace(trace)
    summary["checkpoint_transitions"] = int((checkpoint.get("rows") or [{}])[-1].get("transitions", 0))
    summary["trace_path"] = str(args.trace)
    summary["checkpoint_path"] = str(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.trace.stem}.evaluation.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    video_path = None if args.no_video else render_video(trace, args.output_dir / f"{args.trace.stem}.actual-vs-reference.mp4")
    if not args.no_wandb:
        publish_wandb(summary, run_id=args.wandb_run_id, project=args.wandb_project, entity=args.wandb_entity, json_path=json_path, video_path=video_path)
    print(json.dumps({"summary": str(json_path), "video": None if video_path is None else str(video_path), "wandb_run_id": args.wandb_run_id}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
