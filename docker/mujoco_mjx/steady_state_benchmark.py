#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "docker/mujoco_mjx"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.ball_pit.common import hand_pose_at, physics_dt, spec_from_args  # noqa: E402
from docker.mujoco_mjx.ball_pit_contact import build_ball_pit_scene_xml, hand_joint_addresses, set_hand_pose  # noqa: E402
from docker.mujoco_mjx.mjx_warp_ball_pit_video import set_hand_pose_mjx  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "logs/mjx_warp_steady_state_benchmark.json"
DEFAULT_SCENE_COPY = REPO_ROOT / "logs/mjx_warp_steady_state_scene.xml"


def now() -> float:
    return time.perf_counter()


def timed(label: str, fn):
    start = now()
    result = fn()
    elapsed = now() - start
    print(f"{label}: {elapsed:.6f}s", flush=True)
    return result, elapsed


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


class GpuSampler:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(0.02, float(interval_seconds))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2.0)
        except Exception:
            return
        timestamp = now()
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                continue
            self.samples.append(
                {
                    "timestamp": timestamp,
                    "index": parts[0],
                    "uuid": parts[1],
                    "gpu_util_percent": _float_or_none(parts[2]),
                    "mem_util_percent": _float_or_none(parts[3]),
                    "memory_used_mib": _float_or_none(parts[4]),
                    "memory_total_mib": _float_or_none(parts[5]),
                    "power_w": _float_or_none(parts[6]),
                }
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)

    def start(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2.0))
        self._sample_once()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        by_uuid: dict[str, list[dict[str, Any]]] = {}
        for sample in self.samples:
            by_uuid.setdefault(str(sample["uuid"]), []).append(sample)

        def stat(rows: list[dict[str, Any]], key: str, fn) -> float | None:
            values = [row[key] for row in rows if row.get(key) is not None]
            return float(fn(values)) if values else None

        gpus = []
        for uuid, rows in sorted(by_uuid.items()):
            gpus.append(
                {
                    "index": rows[-1]["index"],
                    "uuid": uuid,
                    "sample_count": len(rows),
                    "avg_gpu_util_percent": stat(rows, "gpu_util_percent", lambda v: sum(v) / len(v)),
                    "max_gpu_util_percent": stat(rows, "gpu_util_percent", max),
                    "avg_mem_util_percent": stat(rows, "mem_util_percent", lambda v: sum(v) / len(v)),
                    "max_mem_util_percent": stat(rows, "mem_util_percent", max),
                    "avg_power_w": stat(rows, "power_w", lambda v: sum(v) / len(v)),
                    "max_power_w": stat(rows, "power_w", max),
                    "max_memory_used_mib": stat(rows, "memory_used_mib", max),
                    "memory_total_mib": rows[-1].get("memory_total_mib"),
                }
            )
        return {"sample_count": len(self.samples), "gpus": gpus}


def resolve_start_frame(spec: Any, measure_frames: int, requested_start_frame: int | None) -> int:
    if requested_start_frame is not None and requested_start_frame >= 0:
        return min(requested_start_frame, max(0, spec.rollout_frames - 1))
    return max(0, (spec.rollout_frames - max(1, measure_frames)) // 2)


def run_cpu(args: argparse.Namespace, spec: Any, model: Any, mujoco: Any, addresses: dict[str, tuple[int, int]], start_frame: int, end_frame: int) -> dict[str, Any]:
    data = mujoco.MjData(model)
    for _ in range(spec.settle_frames * spec.substeps):
        mujoco.mj_step(model, data)
    for frame in range(start_frame):
        set_hand_pose(mujoco, model, data, frame, spec, addresses)
        for _ in range(spec.substeps):
            set_hand_pose(mujoco, model, data, frame, spec, addresses)
            mujoco.mj_step(model, data)
            set_hand_pose(mujoco, model, data, frame, spec, addresses)

    measure_frames = end_frame - start_frame
    simulated_seconds = measure_frames / float(spec.fps)
    timings: dict[str, float] = {}
    throughput: dict[str, float | None] = {}

    if args.mode in {"step_only", "both"}:
        print("MUJOCO_CPU_MEASURE_START mode=step_only", flush=True)
        start = now()
        for _ in range(measure_frames * spec.substeps):
            mujoco.mj_step(model, data)
        elapsed = now() - start
        print(f"mujoco_cpu_measure_step_only: {elapsed:.6f}s", flush=True)
        timings["step_only"] = elapsed
        throughput["step_only_steps_per_second"] = measure_frames * spec.substeps / elapsed if elapsed > 0 else None
        throughput["step_only_real_time_factor"] = simulated_seconds / elapsed if elapsed > 0 else None

    if args.mode in {"controlled_frame", "both"}:
        print("MUJOCO_CPU_MEASURE_START mode=controlled_frame", flush=True)
        start = now()
        for frame in range(start_frame, end_frame):
            set_hand_pose(mujoco, model, data, frame, spec, addresses)
            for _ in range(spec.substeps):
                set_hand_pose(mujoco, model, data, frame, spec, addresses)
                mujoco.mj_step(model, data)
                set_hand_pose(mujoco, model, data, frame, spec, addresses)
        elapsed = now() - start
        print(f"mujoco_cpu_measure_controlled_frame: {elapsed:.6f}s", flush=True)
        timings["controlled_frame"] = elapsed
        throughput["controlled_frames_per_second"] = measure_frames / elapsed if elapsed > 0 else None
        throughput["controlled_steps_per_second"] = measure_frames * spec.substeps / elapsed if elapsed > 0 else None
        throughput["controlled_real_time_factor"] = simulated_seconds / elapsed if elapsed > 0 else None

    return {"timings_seconds": timings, "throughput": throughput}


def run_gpu(args: argparse.Namespace, spec: Any, model: Any, mujoco: Any, addresses: dict[str, tuple[int, int]], start_frame: int, end_frame: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mujoco import mjx

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected JAX GPU backend, got {jax.default_backend()} devices={jax.devices()}")

    mx = mjx.put_model(model, impl="warp")
    dx = mjx.make_data(model, impl="warp", naconmax=args.naconmax, njmax=args.njmax)
    step = jax.jit(mjx.step)
    for _ in range(spec.settle_frames * spec.substeps):
        dx = set_hand_pose_mjx(jnp, dx, 0, spec, addresses)
        dx = step(mx, dx)
    for frame in range(start_frame):
        dx = set_hand_pose_mjx(jnp, dx, frame, spec, addresses)
        for _ in range(spec.substeps):
            dx = set_hand_pose_mjx(jnp, dx, frame, spec, addresses)
            dx = step(mx, dx)
            dx = set_hand_pose_mjx(jnp, dx, frame, spec, addresses)
    jax.block_until_ready(dx.qpos)

    measure_frames = end_frame - start_frame
    simulated_seconds = measure_frames / float(spec.fps)
    timings: dict[str, float] = {}
    throughput: dict[str, float | None] = {}
    gpu_monitor: dict[str, Any] = {}
    base_value_index = {"ARTx": 0, "ARTy": 1, "ARTz": 2, "ARRx": 3, "ARRy": 4, "ARRz": 5}
    address_items = tuple((name, int(qadr), int(dadr)) for name, (qadr, dadr) in sorted(addresses.items()))
    control_values = []
    for frame in range(start_frame, end_frame):
        pos, euler = hand_pose_at(frame, spec)
        control_values.append((pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]))
    control_values_jnp = jnp.asarray(np.asarray(control_values, dtype=np.float32))

    def apply_hand_values(dx_in: Any, values: Any) -> Any:
        qpos = dx_in.qpos
        qvel = dx_in.qvel
        qacc = dx_in.qacc
        for name, qadr, dadr in address_items:
            value_idx = base_value_index.get(name)
            value = values[value_idx] if value_idx is not None else 0.0
            qpos = qpos.at[qadr].set(value)
            qvel = qvel.at[dadr].set(0.0)
            qacc = qacc.at[dadr].set(0.0)
        return dx_in.replace(qpos=qpos, qvel=qvel, qacc=qacc)

    def step_only_scan(dx0: Any) -> Any:
        def body(carry: Any, _unused: Any) -> tuple[Any, None]:
            return step(mx, carry), None
        dx_out, _ = jax.lax.scan(body, dx0, None, length=measure_frames * spec.substeps)
        return dx_out

    def controlled_scan(dx0: Any) -> Any:
        def frame_body(carry: Any, values: Any) -> tuple[Any, None]:
            carry = apply_hand_values(carry, values)

            def sub_body(sub_carry: Any, _sub: Any) -> tuple[Any, None]:
                sub_carry = apply_hand_values(sub_carry, values)
                sub_carry = step(mx, sub_carry)
                sub_carry = apply_hand_values(sub_carry, values)
                return sub_carry, None

            carry, _ = jax.lax.scan(sub_body, carry, None, length=spec.substeps)
            return carry, None

        dx_out, _ = jax.lax.scan(frame_body, dx0, control_values_jnp)
        return dx_out

    if args.mode in {"step_only", "both"}:
        compiled_step_only = jax.jit(step_only_scan)
        print("MJX_WARP_WARMUP_COMPILE mode=step_only", flush=True)
        warm = compiled_step_only(dx)
        jax.block_until_ready(warm.qpos)
        print("MJX_WARP_MEASURE_START mode=step_only", flush=True)
        sampler = GpuSampler(args.gpu_sample_interval).start() if args.gpu_sample_interval > 0 else None
        start = now()
        measured = compiled_step_only(dx)
        jax.block_until_ready(measured.qpos)
        elapsed = now() - start
        if sampler is not None:
            gpu_monitor["step_only"] = sampler.stop()
        print(f"mjx_warp_measure_step_only: {elapsed:.6f}s", flush=True)
        timings["step_only"] = elapsed
        throughput["step_only_steps_per_second"] = measure_frames * spec.substeps / elapsed if elapsed > 0 else None
        throughput["step_only_real_time_factor"] = simulated_seconds / elapsed if elapsed > 0 else None

    if args.mode in {"controlled_frame", "both"}:
        compiled_controlled = jax.jit(controlled_scan)
        print("MJX_WARP_WARMUP_COMPILE mode=controlled_frame", flush=True)
        warm = compiled_controlled(dx)
        jax.block_until_ready(warm.qpos)
        print("MJX_WARP_MEASURE_START mode=controlled_frame", flush=True)
        sampler = GpuSampler(args.gpu_sample_interval).start() if args.gpu_sample_interval > 0 else None
        start = now()
        measured = compiled_controlled(dx)
        jax.block_until_ready(measured.qpos)
        elapsed = now() - start
        if sampler is not None:
            gpu_monitor["controlled_frame"] = sampler.stop()
        print(f"mjx_warp_measure_controlled_frame: {elapsed:.6f}s", flush=True)
        timings["controlled_frame"] = elapsed
        throughput["controlled_frames_per_second"] = measure_frames / elapsed if elapsed > 0 else None
        throughput["controlled_steps_per_second"] = measure_frames * spec.substeps / elapsed if elapsed > 0 else None
        throughput["controlled_real_time_factor"] = simulated_seconds / elapsed if elapsed > 0 else None

    return {"timings_seconds": timings, "throughput": throughput, "gpu_monitor": gpu_monitor, "jax_backend": jax.default_backend(), "jax_devices": [str(d) for d in jax.devices()]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco

    spec = spec_from_args(args)
    start_frame = resolve_start_frame(spec, args.measure_frames, args.start_frame)
    end_frame = min(spec.rollout_frames, start_frame + args.measure_frames)
    measure_frames = max(0, end_frame - start_frame)
    if measure_frames <= 0:
        raise ValueError("measure_frames must cover at least one frame")

    scene_path = build_ball_pit_scene_xml(Path(tempfile.mkdtemp(prefix="mjx_steady_state_")) / "ball_pit.xml", spec)
    args.scene_copy.parent.mkdir(parents=True, exist_ok=True)
    args.scene_copy.write_text(scene_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"scene={scene_path}", flush=True)
    print(f"scene_copy={args.scene_copy}", flush=True)
    model, load_seconds = timed("load_mj_model", lambda: mujoco.MjModel.from_xml_path(str(scene_path)))
    addresses = hand_joint_addresses(mujoco, model)
    if any(name.startswith("ball_") for name in addresses):
        raise RuntimeError(f"ball freejoint leaked into hand lock set: {sorted(addresses)}")

    setup_start = now()
    if args.backend == "cpu":
        result = run_cpu(args, spec, model, mujoco, addresses, start_frame, end_frame)
    else:
        result = run_gpu(args, spec, model, mujoco, addresses, start_frame, end_frame)
    total_unscoped_seconds = now() - setup_start

    payload: dict[str, Any] = {
        "backend": "mjx_warp" if args.backend == "gpu" else "mujoco_cpu",
        "physics_backend": args.backend,
        "mujoco_version": mujoco.__version__,
        "mode": args.mode,
        "benchmark_scope": "steady middle-frame physics; excludes Docker startup, XML generation, model load, mjx put/make data, JIT warmup, settle, advance-to-start, contacts, video, and CPU trajectory readback",
        "scene_xml": str(scene_path),
        "scene_copy": str(args.scene_copy),
        "scenario": spec.scenario,
        "ball_count": spec.ball_count,
        "ball_radius": spec.ball_radius,
        "fps": spec.fps,
        "substeps": spec.substeps,
        "physics_dt": physics_dt(spec),
        "settle_frames": spec.settle_frames,
        "rollout_frames": spec.rollout_frames,
        "duration_seconds": spec.duration_seconds,
        "start_frame": start_frame,
        "measure_frames": measure_frames,
        "measured_physics_steps_per_segment": measure_frames * spec.substeps,
        "measured_simulated_seconds": measure_frames / float(spec.fps),
        "load_model_seconds_unmeasured": load_seconds,
        "backend_setup_warmup_seconds_unmeasured": total_unscoped_seconds,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "njnt": int(model.njnt),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "naconmax": args.naconmax,
        "njmax": args.njmax,
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure MuJoCo CPU or MJX-Warp GPU middle-frame steady-state physics throughput.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scene-copy", type=Path, default=DEFAULT_SCENE_COPY)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--mode", choices=("step_only", "controlled_frame", "both"), default="both")
    parser.add_argument("--scenario", default="filled_tank", choices=("sparse_debug", "filled_tank"))
    parser.add_argument("--ball-count", type=int, default=120)
    parser.add_argument("--ball-radius", type=float, default=0.03)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--settle-frames", type=int, default=10)
    parser.add_argument("--rollout-frames", type=int, default=1000)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--start-frame", type=int, default=-1)
    parser.add_argument("--measure-frames", type=int, default=200)
    parser.add_argument("--naconmax", type=int, default=8192)
    parser.add_argument("--njmax", type=int, default=8192)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
