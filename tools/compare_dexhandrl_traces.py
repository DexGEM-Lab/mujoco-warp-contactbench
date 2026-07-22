#!/usr/bin/env python3
"""Compare exact DexHand Gym and MuJoCo boundary traces."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sim.dexhandrl.obs_layout import OBS_COMPONENT_SLICES
from sim.dexhandrl.trace_contract import load_trace

DEFAULT_TOLERANCE = 1e-6

def _metric(left: np.ndarray, right: np.ndarray, frames: np.ndarray, tolerance: float) -> dict[str, Any]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    absolute = np.abs(delta)
    per_frame = absolute.max(axis=1) if absolute.ndim == 2 else absolute
    worst = int(np.argmax(per_frame))
    crossed = np.flatnonzero(per_frame > tolerance)
    return {"max_abs": float(absolute.max()), "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "first_frame": None if not len(crossed) else int(frames[crossed[0]]),
            "worst_frame": int(frames[worst]), "tolerance": tolerance}

def compare(gym_path: str, mujoco_path: str, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    gym_meta, gym = load_trace(gym_path); mj_meta, mj = load_trace(mujoco_path)
    if gym_meta["identity"] != mj_meta["identity"]:
        raise ValueError("trace identity differs")
    if not np.array_equal(gym["lance_frame"], mj["lance_frame"]):
        raise ValueError("trace Lance frames are not exactly aligned")
    frames = gym["lance_frame"]
    stages = ["processed_observation", "rms_observation", "policy_mean", "final_action",
              "base_offset_after", "joint_offset_after", "reference_active_target",
              "corrected_active_target", "filtered_active_target", "full_target"]
    metrics = {stage: _metric(gym[stage], mj[stage], frames, tolerance) for stage in stages}
    metrics["observation_slices"] = {
        name: _metric(gym["processed_observation"][:, begin:end], mj["processed_observation"][:, begin:end], frames, tolerance)
        for name, (begin, end) in OBS_COMPONENT_SLICES.items()
    }
    shared = [stage for stage in stages if stage not in {"hand_qpos", "hand_qvel", "object_position"}]
    first = next((stage for stage in shared if metrics[stage]["first_frame"] is not None), None)
    physics = {key: _metric(gym[key], mj[key], frames, tolerance) for key in ("hand_qpos", "hand_qvel", "object_position", "object_quaternion_xyzw", "object_linear_velocity")}
    first_physics = next((stage for stage, value in physics.items() if value["first_frame"] is not None), None)
    return {"schema": "dexhandrl.first_divergence_comparison.v1", "identity": gym_meta["identity"],
            "frames": [int(frames[0]), int(frames[-1])], "shared_input": metrics,
            "physics": physics, "first_divergence": {"shared_input_stage": first, "physics_stage": first_physics},
            "stage_semantics": {"filtered_active_target": {"gym": "Gym ActionRules velocity+position filtered target", "mujoco": "unfiltered; post_filters_absent"}}}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym", required=True); parser.add_argument("--mujoco", required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args(); result = compare(args.gym, args.mujoco, args.tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    first = result["first_divergence"]
    print(json.dumps({"schema": result["schema"], "frames": result["frames"], "first_divergence": first}, sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())
