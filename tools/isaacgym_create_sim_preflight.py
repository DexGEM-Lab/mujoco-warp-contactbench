#!/usr/bin/env python3
"""Run one bounded, headless Isaac Gym GPU PhysX create_sim preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--max-gpu-contact-pairs", type=int, required=True)
    parser.add_argument("--num-subscenes", type=int, default=0)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _snapshot(*, max_gpu_contact_pairs: int, num_subscenes: int) -> dict[str, Any]:
    # Isaac Gym must be imported before Torch. Importing torch here preserves the
    # same runtime order as the source task while recording its CUDA ABI.
    from isaacgym import gymapi
    import torch

    return {
        "schema": "manorl.isaacgym.create_sim_preflight.v1",
        "status": "started",
        "request": {
            "compute_device_id": 0,
            "graphics_device_id": -1,
            "physics_engine": "physx",
            "use_gpu": True,
            "use_gpu_pipeline": True,
            "dt": 0.005,
            "substeps": 2,
            "solver_type": 1,
            "num_position_iterations": 8,
            "num_velocity_iterations": 4,
            "max_gpu_contact_pairs": max_gpu_contact_pairs,
            "num_subscenes": num_subscenes,
        },
        "runtime": {
            "python": os.sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "isaacgym_binding": gymapi.__file__,
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "PATH",
                    "LD_LIBRARY_PATH",
                    "PYTHONPATH",
                    "CUDA_HOME",
                    "CUDA_VISIBLE_DEVICES",
                    "DISPLAY",
                    "GYM_USD_PLUG_INFO_PATH",
                )
            },
        },
    }


def main() -> int:
    args = parse_args()
    if args.max_gpu_contact_pairs <= 0:
        raise ValueError("--max-gpu-contact-pairs must be positive")
    if args.num_subscenes < 0:
        raise ValueError("--num-subscenes cannot be negative")
    result = args.result.expanduser().resolve()
    if result.exists() and not args.replace:
        raise FileExistsError(f"result exists; pass --replace to replace it: {result}")

    snapshot = _snapshot(
        max_gpu_contact_pairs=args.max_gpu_contact_pairs,
        num_subscenes=args.num_subscenes,
    )
    _write_json(result, snapshot)

    from isaacgym import gymapi

    parameters = gymapi.SimParams()
    parameters.dt = snapshot["request"]["dt"]
    parameters.substeps = snapshot["request"]["substeps"]
    parameters.up_axis = gymapi.UP_AXIS_Z
    parameters.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    parameters.use_gpu_pipeline = True
    parameters.physx.use_gpu = True
    parameters.physx.solver_type = snapshot["request"]["solver_type"]
    parameters.physx.num_position_iterations = snapshot["request"]["num_position_iterations"]
    parameters.physx.num_velocity_iterations = snapshot["request"]["num_velocity_iterations"]
    parameters.physx.max_gpu_contact_pairs = args.max_gpu_contact_pairs
    parameters.physx.num_subscenes = args.num_subscenes

    gym = gymapi.acquire_gym()
    simulation = gym.create_sim(0, -1, gymapi.SIM_PHYSX, parameters)
    if simulation is None:
        snapshot.update(status="failed", error="gym.create_sim returned None")
        _write_json(result, snapshot)
        return 1
    try:
        gym.prepare_sim(simulation)
        snapshot["status"] = "create_and_prepare_returned"
    finally:
        gym.destroy_sim(simulation)
    _write_json(result, snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
