"""Benchmark device-resident cube2 batching.

The benchmark has separate physics and environment/sampling modes. It reports
actual environment transitions, not updates, and synchronizes only at warmup
and measurement boundaries.
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))
import numpy as np
from sim.manorl.autonomy_batch import BatchedAutonomyRuntime
from sim.manorl.trajectory_package import load_trajectory_package


def _sync(value):
    block = getattr(value, "block_until_ready", None)
    if callable(block): block()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--identity", default="cube2_02_2833")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--mode", choices=("physics", "environment", "sampling", "both", "all"), default="both")
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--measurement-steps", "--steps", dest="measurement_steps", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--persistent-ccd-workspace", action="store_true", help="reuse the existing Warp CCD and solver workspaces")
    parser.add_argument("--ccd-contacts-per-world", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.measurement_steps < 1 or args.warmup_steps < 0 or args.num_envs < 1:
        parser.error("batch, warmup-steps, and measurement-steps must be positive/non-negative")
    catalog = load_trajectory_package(Path(args.package))
    trajectory = next((item for item in catalog.trajectories if item.identity.identity == args.identity), None)
    if trajectory is None: parser.error(f"identity absent from package: {args.identity}")
    construct_start = time.perf_counter()
    runtime = BatchedAutonomyRuntime(trajectory, num_envs=args.num_envs, device=args.device, persistent_ccd_workspace=args.persistent_ccd_workspace, ccd_contacts_per_world=args.ccd_contacts_per_world)
    construct_seconds = time.perf_counter() - construct_start
    jp, device = runtime.jp, runtime.device
    actions = jp.zeros((args.num_envs, 28), dtype=jp.float32)
    mode = "both" if args.mode == "all" else args.mode
    actor = None
    if mode in {"sampling", "both"}:
        import gymnasium as gym
        from sim.manorl.autonomy_training import AutonomyActorCritic
        actor = AutonomyActorCritic(gym.spaces.Box(-5., 5., shape=(538,), dtype=np.float32), gym.spaces.Box(-1., 1., shape=(28,), dtype=np.float32), device="cuda" if args.device == "gpu" else "cpu")
        actor.eval()
    def physics_step():
        nonlocal runtime
        runtime.data = runtime.data.replace(ctrl=actions)
        for _ in range(4): runtime.data = runtime.physics_step_fn(runtime.data)
    def environment_step(with_policy: bool):
        nonlocal actions
        if bool(np.asarray(runtime.pending_reset).any()):
            runtime.prepare_action()
        if with_policy:
            import torch
            with torch.no_grad():
                if args.device == "gpu":
                    from sim.manorl.device_runtime import jax_to_torch_cuda, torch_to_jax_cuda
                    obs = jax_to_torch_cuda(runtime.observation)
                else:
                    obs = torch.as_tensor(np.asarray(runtime.observation), dtype=torch.float32)
                mean, _ = actor.compute({"observations": obs}, role="policy")
                policy_actions = torch.tanh(mean)
                actions = torch_to_jax_cuda(policy_actions) if args.device == "gpu" else np.asarray(policy_actions)
        runtime.step(actions)
    # Warmup is excluded from steady-state rates and synchronizes once at end.
    warmup_start = time.perf_counter()
    for _ in range(args.warmup_steps):
        physics_step() if mode == "physics" else environment_step(mode == "sampling" or mode == "both")
    _sync(runtime.data.qpos)
    warmup_seconds = time.perf_counter() - warmup_start
    reset_acc = jp.zeros((args.num_envs,), dtype=jp.int32)
    valid_acc = jp.zeros((args.num_envs,), dtype=jp.int32)
    start = time.perf_counter()
    for _ in range(args.measurement_steps):
        if mode == "physics":
            physics_step()
        else:
            environment_step(mode == "sampling" or mode == "both")
            # Device accumulation; host reads happen once after the window.
            reset_acc = reset_acc + jp.asarray(runtime.pending_reset, dtype=jp.int32)
            valid_acc = valid_acc + jp.asarray(runtime.last_physical.valid, dtype=jp.int32)
    _sync(runtime.data.qpos)
    elapsed = time.perf_counter() - start
    transitions = args.measurement_steps * args.num_envs
    result = {
        "mode": mode, "batch": args.num_envs, "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps, "transitions": transitions,
        "construct_seconds": construct_seconds, "warmup_seconds": warmup_seconds,
        "steady_seconds": elapsed, "environment_transitions_per_second": transitions / max(elapsed, 1e-12),
        "reset_count": int(np.asarray(reset_acc).sum()) if mode != "physics" else 0,
        "validity_count": int(np.asarray(valid_acc).sum()) if mode != "physics" else None,
        "host_transfer_scope": "measurement-boundary qpos synchronization plus compact aggregate reset/valid counters; no per-step host full-state/contact copy",
        "peak_memory_bytes": None,
        "memory_note": "operator captures peak GPU memory with nvidia-smi or equivalent",
        "clock": {"policy_fps": 120, "physics_fps": 480, "substeps": 4},
    }
    payload = json.dumps(result, sort_keys=True, indent=2)
    print(payload)
    if args.output: args.output.write_text(payload + "\n", encoding="utf-8")

if __name__ == "__main__": main()
