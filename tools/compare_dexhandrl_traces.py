#!/usr/bin/env python3
"""Compare exact DexHand Gym and MuJoCo boundary traces."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from sim.dexhandrl.obs_layout import OBS_COMPONENT_SLICES
from sim.dexhandrl.trace_contract import load_trace

DEFAULT_TOLERANCE = 1e-6

def _metric(left: np.ndarray, right: np.ndarray, frames: np.ndarray, tolerance: float) -> dict[str, Any]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    absolute = np.abs(delta); per_frame = absolute.max(axis=1) if absolute.ndim == 2 else absolute
    worst = int(np.argmax(per_frame)); crossed = np.flatnonzero(per_frame > tolerance)
    return {"max_abs": float(absolute.max()), "rmse": float(np.sqrt(np.mean(delta ** 2))), "first_frame": None if not len(crossed) else int(frames[crossed[0]]), "worst_frame": int(frames[worst]), "tolerance": tolerance}

def _slice_metrics(left: np.ndarray, right: np.ndarray, frames: np.ndarray, tolerance: float) -> dict[str, Any]:
    return {name: _metric(left[:, begin:end], right[:, begin:end], frames, tolerance) for name, (begin, end) in OBS_COMPONENT_SLICES.items()}

def compare(gym_path: str, mujoco_path: str, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    gym_meta, gym = load_trace(gym_path); mj_meta, mj = load_trace(mujoco_path)
    if gym_meta["identity"] != mj_meta["identity"]: raise ValueError("trace identity differs")
    for column in ("decision_frame", "state_frame"):
        if not np.array_equal(gym[column], mj[column]): raise ValueError(f"trace {column} values are not exactly aligned")
    frames = gym["decision_frame"]
    stages = ["processed_observation", "rms_observation", "policy_mean", "final_action", "base_offset_after", "joint_offset_after", "reference_active_target", "corrected_active_target", "applied_active_target", "full_target"]
    metrics = {stage: _metric(gym[stage], mj[stage], frames, tolerance) for stage in stages}
    metrics["processed_observation_slices"] = _slice_metrics(gym["processed_observation"], mj["processed_observation"], frames, tolerance)
    metrics["rms_observation_slices"] = _slice_metrics(gym["rms_observation"], mj["rms_observation"], frames, tolerance)
    first = next((stage for stage in stages if metrics[stage]["first_frame"] is not None), None)
    physics_keys = ("hand_qpos", "hand_qvel", "object_position", "object_quaternion_xyzw", "object_linear_velocity", "target_object_position", "target_object_quaternion_xyzw")
    physics = {key: _metric(gym[key], mj[key], gym["state_frame"], tolerance) for key in physics_keys}
    first_physics = next((stage for stage in physics_keys if physics[stage]["first_frame"] is not None), None)
    return {"schema": "dexhandrl.first_divergence_comparison.v2", "identity": gym_meta["identity"], "decision_frames": [int(frames[0]), int(frames[-1])], "state_frames": [int(gym["state_frame"][0]), int(gym["state_frame"][-1])], "shared_input": metrics, "physics": physics, "first_divergence": {"shared_input_stage": first, "physics_stage": first_physics}, "stage_semantics": {"gym_post_filters": gym_meta["post_filters"], "mujoco_post_filters": mj_meta["post_filters"], "applied_active_target": "a mismatch is a control-contract divergence"}}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--gym", required=True); parser.add_argument("--mujoco", required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(); result = compare(args.gym, args.mujoco, args.tolerance); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"schema": result["schema"], "decision_frames": result["decision_frames"], "first_divergence": result["first_divergence"]}, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
