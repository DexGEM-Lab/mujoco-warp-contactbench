#!/usr/bin/env python3
"""Measure ManoRL MJX-Warp physics separately from complete environment steps.

Examples:
  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python tools/benchmark_manorl_mjx_throughput.py \
    --selector cube1:01 --num-envs 4096 --output outputs/manorl/bench_cube1_4096.json
  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python tools/benchmark_manorl_mjx_throughput.py \
    --mode both --selector all --unified-object-batch --num-envs 4096 \
    --warp-ccd-iterations 16 --warp-ccd-contacts-per-world 8 \
    --warp-persistent-ccd-workspace --output outputs/manorl/bench_unified_4096.json

The physics measurement contains only the batched JIT MJX-Warp step function.
Each reported control step performs the selected runtime's exact substep count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.manorl.contracts import DEFAULT_HAND_DATASET_PATH, EXPECTED_DATASET_VERSION
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment, recommended_warp_contact_capacity
from sim.manorl.trajectory import (
    DEFAULT_REFERENCE_FPS,
    SUPPORTED_REFERENCE_FPS,
    TrajectorySelection,
    load_assigned_trajectory_batch,
)

SCHEMA = "manorl.mjx_throughput.v2"
T = TypeVar("T")


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing benchmark artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        # Do not silently replace a concurrent publisher after the first check.
        if path.exists():
            raise FileExistsError(f"refusing to replace existing benchmark artifact: {path}")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _throughput(
    *,
    elapsed_seconds: float,
    control_steps: int,
    num_envs: int,
    physics_substeps_per_control: int,
) -> dict[str, float | int]:
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    if control_steps < 1 or num_envs < 1 or physics_substeps_per_control < 1:
        raise ValueError("control steps, environments, and physics substeps must be positive")
    batch_rate = control_steps / elapsed_seconds
    return {
        "elapsed_seconds": elapsed_seconds,
        "control_steps": control_steps,
        "batch_control_steps_per_second": batch_rate,
        "aggregate_world_control_steps_per_second": batch_rate * num_envs,
        "per_env_control_steps_per_second": batch_rate,
        "aggregate_physics_substeps_per_second": (
            batch_rate * num_envs * physics_substeps_per_control
        ),
        "physics_substeps_per_control_step": physics_substeps_per_control,
    }


def _run_physics_loop(
    data: T,
    *,
    step_fn: Callable[[T], T],
    synchronize: Callable[[T], Any],
    control_steps: int,
    physics_substeps_per_control: int,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[T, float]:
    """Run exactly the configured physics substeps for each control step."""

    if control_steps < 1 or physics_substeps_per_control < 1:
        raise ValueError("control_steps and physics_substeps_per_control must be positive")
    synchronize(data)
    started = clock()
    for _ in range(control_steps):
        for _ in range(physics_substeps_per_control):
            data = step_fn(data)
    synchronize(data)
    return data, clock() - started


def _physics_warmup(environment: MujocoManoEnvironment, control_steps: int) -> None:
    if control_steps == 0:
        environment.jax.block_until_ready(environment.data.qpos)
        return
    environment.data, _ = _run_physics_loop(
        environment.data,
        step_fn=environment._step_fn,
        synchronize=lambda data: environment.jax.block_until_ready(data.qpos),
        control_steps=control_steps,
        physics_substeps_per_control=environment.config.physics_substeps_per_control,
    )


def _finite_state(environment: MujocoManoEnvironment) -> bool:
    qpos = np.asarray(environment.data.qpos)
    qvel = np.asarray(environment.data.qvel)
    return bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))


def _deterministic_actions(num_envs: int, action_dim: int, seed: int) -> np.ndarray:
    # A seeded fixed matrix exercises every action component without an RNG in
    # the timed region. The bound avoids deliberate joint-limit excursions.
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.25, 0.25, size=(num_envs, action_dim)).astype(np.float64)


def _measure_physics(environment: MujocoManoEnvironment, args: argparse.Namespace) -> dict[str, Any]:
    _physics_warmup(environment, args.warmup_steps)
    environment.data, elapsed = _run_physics_loop(
        environment.data,
        step_fn=environment._step_fn,
        synchronize=lambda data: environment.jax.block_until_ready(data.qpos),
        control_steps=args.measurement_steps,
        physics_substeps_per_control=environment.config.physics_substeps_per_control,
    )
    result = _throughput(
        elapsed_seconds=elapsed,
        control_steps=args.measurement_steps,
        num_envs=args.num_envs,
        physics_substeps_per_control=environment.config.physics_substeps_per_control,
    )
    result.update({"mode": "physics", "finite_qpos_qvel": _finite_state(environment)})
    if not result["finite_qpos_qvel"]:
        raise RuntimeError("physics benchmark produced non-finite qpos or qvel")
    return result


def _measure_environment(environment: MujocoManoEnvironment, args: argparse.Namespace) -> dict[str, Any]:
    actions = _deterministic_actions(args.num_envs, environment.action_dim, args.seed)
    for _ in range(args.warmup_steps):
        environment.step(actions)
    environment.jax.block_until_ready(environment.data.qpos)
    if args.profile_phases:
        environment.reset_phase_profile()
    started = time.perf_counter()
    observed_contacts = 0
    observed_resets = 0
    finite_outputs = True
    for _ in range(args.measurement_steps):
        observation, reward, reset, _ = environment.step(actions)
        observed_resets += int(np.count_nonzero(reset))
        finite_outputs = finite_outputs and bool(
            np.all(np.isfinite(observation["obs"])) and np.all(np.isfinite(reward))
        )
        physical = environment.last_physical
        if physical is not None:
            observed_contacts += int(np.sum(physical.contact_count))
    environment.jax.block_until_ready(environment.data.qpos)
    elapsed = time.perf_counter() - started
    result = _throughput(
        elapsed_seconds=elapsed,
        control_steps=args.measurement_steps,
        num_envs=args.num_envs,
        physics_substeps_per_control=environment.config.physics_substeps_per_control,
    )
    result.update(
        {
            "mode": "environment",
            "observed_contact_count": observed_contacts,
            "observed_reset_count": observed_resets,
            "finite_outputs": finite_outputs,
            "finite_qpos_qvel": _finite_state(environment),
        }
    )
    if not finite_outputs or not result["finite_qpos_qvel"]:
        raise RuntimeError("environment benchmark produced non-finite outputs or state")
    if args.profile_phases:
        result["phase_profile"] = environment.phase_profile()
        result["contact_profile"] = environment.contact_profile()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("physics", "environment", "both"), default="both")
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_HAND_DATASET_PATH))
    parser.add_argument("--dataset-version", type=int, default=EXPECTED_DATASET_VERSION)
    parser.add_argument("--selector", default="cube1:01", help="OBJECT:ACTION[,OBJECT:ACTION...] or all")
    parser.add_argument("--hand-side", choices=("auto", "right", "left", "both"), default="auto")
    parser.add_argument(
        "--reference-fps",
        type=int,
        choices=SUPPORTED_REFERENCE_FPS,
        default=DEFAULT_REFERENCE_FPS,
        help="coupled source/policy clock; physics uses four exact substeps",
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--measurement-steps", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--unified-object-batch", action="store_true")
    parser.add_argument("--warp-ccd-iterations", type=int)
    parser.add_argument("--warp-ccd-contacts-per-world", type=int)
    parser.add_argument("--warp-persistent-ccd-workspace", action="store_true")
    parser.add_argument("--device-resident-controls", action="store_true")
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs < 1 or args.measurement_steps < 1 or args.warmup_steps < 0:
        raise ValueError("num-envs and measurement-steps must be positive; warmup-steps must be non-negative")
    if args.dataset_version is not None and args.dataset_version < 1:
        raise ValueError("dataset-version must be positive")
    for name in ("warp_ccd_iterations", "warp_ccd_contacts_per_world"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive when provided")
    if args.warp_persistent_ccd_workspace:
        if not args.unified_object_batch:
            raise ValueError("--warp-persistent-ccd-workspace requires --unified-object-batch")
        if args.warp_ccd_contacts_per_world is None:
            raise ValueError("--warp-persistent-ccd-workspace requires --warp-ccd-contacts-per-world")
        if args.device != "gpu":
            raise ValueError("--warp-persistent-ccd-workspace requires --device gpu")
    TrajectorySelection(
        selector=args.selector,
        dataset_path=args.dataset_path,
        expected_dataset_version=args.dataset_version,
        hand_side=args.hand_side,
        reference_fps=args.reference_fps,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error(f"refusing to replace existing benchmark artifact: {args.output}")

    build_started = time.perf_counter()
    selection = TrajectorySelection(
        selector=args.selector,
        dataset_path=args.dataset_path,
        expected_dataset_version=args.dataset_version,
        hand_side=args.hand_side,
        reference_fps=args.reference_fps,
    )
    trajectories = load_assigned_trajectory_batch(selection, num_envs=args.num_envs)
    contact_capacity = recommended_warp_contact_capacity(args.num_envs, trajectories.hand_sides)
    environment = MujocoManoEnvironment(
        trajectories,
        EnvironmentConfig(
            num_envs=args.num_envs,
            device=args.device,
            hand_side=args.hand_side,
            reference_fps=args.reference_fps,
            contact_capacity=contact_capacity,
            device_resident_controls=args.device_resident_controls,
            capture_transition_diagnostics=False,
            profile_phases=args.profile_phases,
            unified_object_batch=args.unified_object_batch,
            warp_ccd_iterations=args.warp_ccd_iterations,
            warp_ccd_contacts_per_world=args.warp_ccd_contacts_per_world,
            warp_persistent_ccd_workspace=args.warp_persistent_ccd_workspace,
        ),
    )
    environment.jax.block_until_ready(environment.data.qpos)
    build_seconds = time.perf_counter() - build_started

    results: list[dict[str, Any]] = []
    if args.mode in {"physics", "both"}:
        results.append(_measure_physics(environment, args))
    if args.mode == "both":
        # Reset the shared state so the complete environment path starts from a
        # normal captured/reset state rather than from the physics-only tail.
        environment.reset()
        environment.jax.block_until_ready(environment.data.qpos)
    if args.mode in {"environment", "both"}:
        results.append(_measure_environment(environment, args))

    payload = {
        "schema": SCHEMA,
        "config": {
            "mode": args.mode, "dataset_path": str(args.dataset_path),
            "dataset_version": args.dataset_version, "selector": selection.canonical_selector,
            "hand_side": args.hand_side, "num_envs": args.num_envs, "seed": args.seed,
            "reference_fps": args.reference_fps,
            "control_fps": environment.config.clock.policy_fps,
            "control_timestep_seconds": environment.config.control_timestep,
            "physics_fps": environment.config.clock.physics_fps,
            "physics_timestep_seconds": environment.config.physics_timestep,
            "physics_substeps_per_control": environment.config.physics_substeps_per_control,
            "warmup_steps": args.warmup_steps, "measurement_steps": args.measurement_steps,
            "device": args.device, "unified_object_batch": args.unified_object_batch,
            "contact_capacity": contact_capacity, "warp_ccd_iterations": args.warp_ccd_iterations,
            "warp_ccd_contacts_per_world": args.warp_ccd_contacts_per_world,
            "warp_persistent_ccd_workspace": args.warp_persistent_ccd_workspace,
            "device_resident_controls": args.device_resident_controls,
            "profile_phases": args.profile_phases,
        },
        "build_seconds": build_seconds,
        "results": results,
    }
    _write_json_atomically(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
