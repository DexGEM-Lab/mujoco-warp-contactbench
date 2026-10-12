#!/usr/bin/env python3
"""Check measured history and selective reset using real MJX/Warp physics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tools.train_manorl_autonomy import _adapter, _catalog_and_trajectory, parse_args
from sim.manorl.autonomy_contracts import raw_observation_slices_voxmani453


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--identity", default="cube2_02_2833")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cli = parse_args(
        [
            "inspect",
            "--package",
            args.package,
            "--identity",
            args.identity,
            "--policy-version",
            "v8",
            "--reward-version",
            "v11",
            "--device",
            args.device,
            "--num-envs",
            "2",
            *([] if args.device == "gpu" else ["--no-persistentworkspace"]),
        ]
    )
    _, trajectory = _catalog_and_trajectory(cli)
    adapter = _adapter(cli, trajectory)
    obs, _ = adapter.reset()
    assert obs.shape == (2, 453)
    runtime = adapter.runtime
    for _ in range(3):
        adapter.step(torch.full((2, 28), 0.25, device=adapter.device))
    history = np.asarray(runtime.pose_history).copy()
    indices = np.asarray(runtime.indices).copy()
    command = np.asarray(runtime.previous_command).copy()
    states = adapter.policy_states().clone()
    qpos = np.asarray(runtime.data.qpos).copy()
    runtime.reset(np.array([True, False]))
    np.testing.assert_array_equal(np.asarray(runtime.history_counts), [0, 3])
    np.testing.assert_array_equal(np.asarray(runtime.pose_history)[1], history[1])
    np.testing.assert_array_equal(np.asarray(runtime.indices)[1], indices[1])
    np.testing.assert_array_equal(np.asarray(runtime.previous_command)[1], command[1])
    np.testing.assert_array_equal(np.asarray(runtime.data.qpos)[1], qpos[1])
    torch.testing.assert_close(adapter.policy_states()[1], states[1], rtol=0, atol=0)
    slices = raw_observation_slices_voxmani453()
    reset_obs = np.asarray(runtime.observation)
    assert reset_obs[0, slices["obj_past_pad"]].all()
    np.testing.assert_array_equal(reset_obs[0, slices["prev_action"]], 0)
    np.testing.assert_array_equal(reset_obs[1, slices["prev_action"]], 0.25)
    runtime.prepare_action()
    np.testing.assert_array_equal(np.asarray(runtime.history_counts), [0, 3])
    result = {
        "device": args.device,
        "jax_devices": sorted(str(device) for device in runtime.data.qpos.devices()),
        "jax_platforms": sorted(
            {device.platform for device in runtime.data.qpos.devices()}
        ),
        "torch_device": str(adapter.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch": 2,
        "physics_steps": 3,
        "observation_shape": list(reset_obs.shape),
        "structured_state_shape": list(states.shape),
        "subset_reset_preserves_other_world": True,
        "history_advances_only_after_physics": True,
        "all_observations_finite": bool(np.isfinite(reset_obs).all()),
    }
    assert result["all_observations_finite"]
    if args.device == "gpu":
        assert result["jax_platforms"] == ["gpu"] and adapter.device.type == "cuda"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
