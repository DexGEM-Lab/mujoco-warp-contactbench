#!/usr/bin/env python3
"""Record fixed native-MuJoCo versus CPU-MJX DexHandRL rollout evidence.

The comparison deliberately holds the task, immutable physics profile, reset,
zero 22D actions, and controller path fixed.  It reports divergence; it does
not make an equivalence claim for the two simulator implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.dexhandrl.obs_layout import OBS_COMPONENT_SLICES

DEFAULT_STEPS = 50
DEFAULT_OUTPUT = Path("outputs/dexhandrl_backend_comparison/cube1_action02_sequence0.json")
THRESHOLDS = (1e-7, 1e-6, 1e-4, 1e-3, 1e-2)
TRACE_KEYS = ("reference_active_target", "corrected_active_target", "applied_active_target", "full_target")
STATE_KEYS = ("frame", "processed_obs", "qpos", "qvel", "ctrl", "object_pos", "object_quat", "reward", "ncon")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _array(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"nonfinite array: {name}")
    return result.copy()


def _frame_summary(delta: np.ndarray) -> np.ndarray:
    absolute = np.abs(np.asarray(delta, dtype=np.float64))
    return absolute if absolute.ndim == 1 else absolute.reshape(absolute.shape[0], -1).max(axis=1)


def metric(left: np.ndarray, right: np.ndarray, frames: np.ndarray, thresholds: tuple[float, ...] = THRESHOLDS) -> dict[str, Any]:
    """Return numeric array equality, dtypes, magnitude, and first divergence."""
    left = _array(left, name="metric native")
    right = _array(right, name="metric mjx")
    frames = np.asarray(frames, dtype=np.int64)
    if left.shape != right.shape or left.shape[0] != frames.shape[0]:
        raise ValueError(f"metric alignment mismatch: {left.shape}, {right.shape}, {frames.shape}")
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    per_frame = _frame_summary(delta)
    nonzero = np.flatnonzero(per_frame > 0.0)
    first_exceed = {
        f"{threshold:g}": (None if not (indices := np.flatnonzero(per_frame > threshold)).size else int(frames[indices[0]]))
        for threshold in thresholds
    }
    return {
        # np.array_equal compares numeric values across numeric dtypes; a dtype
        # asymmetry is evidence to report, not an automatic exactness failure.
        "exact": bool(np.array_equal(left, right)),
        "native_dtype": str(left.dtype),
        "mjx_dtype": str(right.dtype),
        "dtype_equal": bool(left.dtype == right.dtype),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "first_nonzero": None if not nonzero.size else int(frames[nonzero[0]]),
        "first_exceed": first_exceed,
    }


def _object_state(env: Any) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(env._dx.qpos)
    address = int(env.ids.object_free_qpos_addr)
    pos = qpos[address:address + 3]
    # MuJoCo free joints store quaternion as wxyz; evidence records xyzw to
    # match the public DexHand observation/trajectory convention.
    quat_wxyz = qpos[address + 3:address + 7]
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return pos, quat_xyzw


def _contact_count(env: Any) -> int:
    backend = env.cfg.runtime_backend
    if backend == "mujoco-native":
        value = getattr(env._dx, "ncon", None)
    elif backend == "mjx-warp":
        impl = getattr(env._dx, "_impl", None)
        value = getattr(impl, "nacon", None)
    else:
        raise RuntimeError(f"unsupported runtime backend for contact count: {backend!r}")
    if value is None:
        raise RuntimeError("runtime state does not expose an active contact count")
    scalar = np.asarray(value).reshape(-1)
    if scalar.size != 1:
        raise RuntimeError(f"active contact count must be scalar, got shape {np.asarray(value).shape}")
    return int(scalar[0])


def _capture_state(env: Any, obs: Any, *, reward: float) -> dict[str, np.ndarray]:
    object_pos, object_quat = _object_state(env)
    return {
        "frame": _array([int(env._current_frame)], name="frame"),
        "processed_obs": _array(obs, name="processed_obs"),
        "qpos": _array(env._dx.qpos, name="qpos"),
        "qvel": _array(env._dx.qvel, name="qvel"),
        "ctrl": _array(env._dx.ctrl, name="ctrl"),
        "object_pos": _array(object_pos, name="object_pos"),
        "object_quat": _array(object_quat, name="object_quat"),
        "reward": _array([reward], name="reward"),
        "ncon": _array([_contact_count(env)], name="ncon"),
    }


def collect_rollout(env: Any, steps: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collect reset and post-action state plus the 50 paired controller decisions."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    state_rows: dict[str, list[np.ndarray]] = {key: [] for key in STATE_KEYS}
    trace_rows: dict[str, list[np.ndarray]] = {key: [] for key in TRACE_KEYS}
    info_full_targets: list[np.ndarray] = []
    decision_frames: list[int] = []
    step_times: list[float] = []

    import time
    obs, _ = env.reset()
    reset = _capture_state(env, obs, reward=0.0)
    for key, value in reset.items():
        state_rows[key].append(value)
    zero_action = np.zeros((22,), dtype=np.float32)
    for _ in range(steps):
        started = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(zero_action)
        step_times.append(time.perf_counter() - started)
        trace = env._last_action_trace
        if not isinstance(trace, Mapping):
            raise RuntimeError("environment did not expose _last_action_trace after step")
        if "decision_frame" not in trace:
            raise RuntimeError("_last_action_trace is missing decision_frame")
        decision_frames.append(int(trace["decision_frame"]))
        for key in TRACE_KEYS:
            if key not in trace:
                raise RuntimeError(f"_last_action_trace is missing {key}")
            trace_rows[key].append(_array(trace[key], name=key))
        if "full_targets" not in info:
            raise RuntimeError("environment info is missing full_targets")
        info_full_targets.append(_array(info["full_targets"], name="info_full_targets"))
        state = _capture_state(env, obs, reward=float(reward))
        for key, value in state.items():
            state_rows[key].append(value)
        if terminated or truncated:
            raise RuntimeError(f"rollout ended before requested {steps} steps")

    arrays = {key: np.stack(values, axis=0) for key, values in state_rows.items()}
    arrays["frame"] = arrays["frame"].reshape(-1).astype(np.int64, copy=False)
    arrays.update({key: np.stack(values, axis=0) for key, values in trace_rows.items()})
    # Decision frames are taken from the controller trace rather than inferred
    # from post-physics state frames, making schedule drift an invariant failure.
    arrays["decision_frame"] = np.asarray(decision_frames, dtype=np.int64)
    arrays["info_full_target"] = np.stack(info_full_targets, axis=0)
    arrays["step_seconds"] = np.asarray(step_times, dtype=np.float64)
    identity = {
        "object_name": str(env.cfg.object_name),
        "action": str(env.cfg.action),
        "sequence_index": int(env.cfg.sequence_index),
        "trajectory_uuid": str(getattr(env.trajectory, "uuid", "")),
        "use_residual": bool(env.cfg.use_residual),
        "action_dim": int(env.num_actions),
        "observation_dim": int(env.num_observations),
    }
    start_frame = int(env.start_frame)
    expected_reset = start_frame + 1
    expected_decisions = np.arange(expected_reset, expected_reset + steps, dtype=np.int64)
    expected_states = np.arange(expected_reset, expected_reset + steps + 1, dtype=np.int64)
    if arrays["frame"][0] != expected_reset:
        raise ValueError(f"{env.cfg.runtime_backend} reset frame {arrays['frame'][0]} != start_frame + 1 ({expected_reset})")
    if not np.array_equal(arrays["decision_frame"], expected_decisions):
        raise ValueError(f"{env.cfg.runtime_backend} decision frame schedule does not start at reset frame")
    if not np.array_equal(arrays["frame"], expected_states):
        raise ValueError(f"{env.cfg.runtime_backend} state frame schedule does not span reset through reset + steps")
    metadata = {
        "identity": identity,
        "physics_contract": env.cfg.physics.record(),
        "runtime_backend": str(env.cfg.runtime_backend),
        "frame_contract": {
            "start_frame": start_frame,
            "reset_frame": expected_reset,
            "state_frames": arrays["frame"].astype(int).tolist(),
            "decision_frames": arrays["decision_frame"].astype(int).tolist(),
            "zero_action_dim": 22,
            "validated": "reset=start_frame+1; decisions=reset..reset+steps-1; states=reset..reset+steps",
        },
    }
    return arrays, metadata


def _timing(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    steady = values[min(5, len(values)):]
    summary = lambda sample: {"mean_s": float(np.mean(sample)), "median_s": float(np.median(sample))}
    return {
        "measurement": "full env.step wall time; excludes reset and compilation",
        "all": summary(values),
        "steady_after_first5": summary(steady if len(steady) else values),
    }


def _state_metrics(native: Mapping[str, np.ndarray], mjx: Mapping[str, np.ndarray], key: str) -> dict[str, Any]:
    """Keep reset and post-step rows distinguishable from aggregate state evidence."""
    frames = native["frame"]
    return {
        "all_rows": metric(native[key], mjx[key], frames),
        "reset_row": metric(native[key][:1], mjx[key][:1], frames[:1]),
        "post_step_rows": metric(native[key][1:], mjx[key][1:], frames[1:]),
    }


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _provenance(native_env: Any, mjx_env: Any) -> dict[str, Any]:
    # Unit-test fakes intentionally avoid simulator imports. A real comparison
    # has already constructed both runtimes, so these fields are concrete there.
    try:
        import jax
        jax_version = jax.__version__
        actual_backend = jax.default_backend()
        actual_devices = [str(device) for device in jax.devices()]
    except ImportError:
        jax_version, actual_backend, actual_devices = "unavailable", "unavailable", []
    try:
        import mujoco
        mujoco_version = mujoco.__version__
    except ImportError:
        mujoco_version = "unavailable"
    return {
        "git_head": _git_head(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mujoco": mujoco_version,
        "jax": jax_version,
        "native": {"requested_device": str(native_env.cfg.device), "runtime_backend": str(native_env.cfg.runtime_backend)},
        "mjx": {
            "requested_device": str(mjx_env.cfg.device),
            "requested_platforms": os.environ.get("JAX_PLATFORMS"),
            "actual_default_backend": actual_backend,
            "actual_devices": actual_devices,
            "runtime_backend": str(mjx_env.cfg.runtime_backend),
            "actual_mjx_impl": str(getattr(getattr(mjx_env, "mx", None), "impl", "unknown")),
        },
    }


def compare_rollouts(native_env: Any, mjx_env: Any, *, steps: int = DEFAULT_STEPS) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Collect and compare two constructed environments; suitable for fast fake-env tests."""
    try:
        native, native_meta = collect_rollout(native_env, steps)
        mjx, mjx_meta = collect_rollout(mjx_env, steps)
    finally:
        native_env.close()
        mjx_env.close()
    if native_meta["identity"] != mjx_meta["identity"]:
        raise ValueError("identity mismatch between native and MJX backends")
    if native_meta["physics_contract"] != mjx_meta["physics_contract"]:
        raise ValueError("physics contract mismatch between native and MJX backends")
    if native_meta["frame_contract"] != mjx_meta["frame_contract"]:
        raise ValueError("frame contract mismatch between native and MJX backends")

    arrays = {f"native_{key}": value for key, value in native.items()}
    arrays.update({f"mjx_{key}": value for key, value in mjx.items()})
    for name, value in arrays.items():
        _array(value, name=name)
    comparisons: dict[str, Any] = {}
    for key in STATE_KEYS:
        comparisons[key] = _state_metrics(native, mjx, key)
    for key in TRACE_KEYS + ("info_full_target",):
        comparisons[key] = metric(native[key], mjx[key], native["decision_frame"])
    comparisons["processed_obs_components"] = {
        name: _state_metrics(
            {"frame": native["frame"], "processed_obs": native["processed_obs"][:, begin:end]},
            {"frame": mjx["frame"], "processed_obs": mjx["processed_obs"][:, begin:end]},
            "processed_obs",
        )
        for name, (begin, end) in OBS_COMPONENT_SLICES.items()
    }
    reference = comparisons["reference_active_target"]
    if not reference["exact"]:
        raise ValueError(f"reference-active target must be exact: max_abs={reference['max_abs']:g}")
    applied = comparisons["applied_active_target"]
    if applied["max_abs"] > 1e-7:
        raise ValueError(f"applied-active target mismatch exceeds 1e-7: {applied['max_abs']:g}")
    physics = native_meta["physics_contract"]
    report = {
        "schema": "dexhandrl.native_mjx_backend_evidence.v2",
        "provenance": _provenance(native_env, mjx_env),
        "identity": native_meta["identity"],
        "physics_contract": physics,
        "frame_contract": native_meta["frame_contract"],
        "inputs": {
            "steps": steps,
            "actions": "zero 22D",
            "residual": False,
            "object_position_terminal_threshold": float(native_env.cfg.object_position_terminal_threshold),
            "settle_frames": int(native_env.cfg.settle_frames),
            "substeps": int(physics["substeps"]),
        },
        "comparison": comparisons,
        "preregistered_target_exactness": {
            "reference_active_target": {"required": True, "observed_exact": reference["exact"], "observed_max_abs": reference["max_abs"]},
            "corrected_active_target": {
                "preregistered_exactness": "falsified as a backend-independent requirement",
                "observed_exact": comparisons["corrected_active_target"]["exact"],
                "observed_max_abs": comparisons["corrected_active_target"]["max_abs"],
            },
            "applied_active_target": {
                "preregistered_exactness": "falsified as a backend-independent requirement; retained acceptance bound is max_abs <= 1e-7",
                "observed_exact": applied["exact"],
                "observed_max_abs": applied["max_abs"],
                "required_max_abs": 1e-7,
            },
        },
        "timing": {"native": _timing(native["step_seconds"]), "mjx": _timing(mjx["step_seconds"])},
        "artifact": {},
        "backend_authority": (
            "MJX-Warp remains authoritative for training and exact reference traces; native MuJoCo is a CPU viewer backend. "
            "Reward exactness across these early-phase steps is near-trivial and is not equivalence evidence."
        ),
        "reset_note": "Each backend reset already advances settle_frames * substeps internal physics steps before the reset observation is captured; reset_row is therefore reported separately from post_step_rows.",
        "target_routing_note": "reference_active_target is the exact shared reference. corrected_active_target and applied_active_target are state-dependent: live current_full is routed to reference_full, then full_to_active selects active targets. full_target and ctrl are reported without an exactness requirement because routing reads each backend's live distal qpos.",
        "exactness_note": "exact means numeric np.array_equal on the original captured arrays. NumPy array_equal does not require matching numeric dtypes; native_dtype, mjx_dtype, and dtype_equal record dtype asymmetry explicitly.",
    }
    return report, arrays


def _make_envs(args: argparse.Namespace) -> tuple[Any, Any]:
    # JAX reads platform configuration during import. Set it before importing
    # skrl_env (which can import JAX) and before constructing either backend.
    os.environ["JAX_PLATFORMS"] = "cpu" if args.mjx_device == "cpu" else "cuda,cpu"
    # Keep heavyweight runtime imports inside execution so --help and unit tests
    # remain offline-safe.
    from sim.dexhandrl.contracts import DexHandPhysicsConfig
    from sim.dexhandrl.skrl_env import DexHandRLMJXEnv, DexHandRLMJXEnvConfig

    physics = DexHandPhysicsConfig()
    common: dict[str, Any] = {
        "object_name": args.object_name,
        "action": args.action,
        "sequence_index": args.sequence_index,
        "uuid": args.uuid,
        "use_residual": False,
        "physics": physics,
        "object_position_terminal_threshold": 1e9,
        "isaac_source_root": args.isaac_source_root,
    }
    if args.lance is not None:
        common["lance_path"] = args.lance
    native = DexHandRLMJXEnv(DexHandRLMJXEnvConfig(**common, device="cpu", runtime_backend="mujoco-native"))
    mjx = DexHandRLMJXEnv(DexHandRLMJXEnvConfig(**common, device=args.mjx_device, runtime_backend="mjx-warp"))
    return native, mjx


def write_evidence(output: Path, report: dict[str, Any], arrays: Mapping[str, np.ndarray]) -> tuple[Path, str]:
    output = Path(output)
    npz_path = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    report["artifact"] = {"npz": str(npz_path), "sha256": digest, "compression": "np.savez_compressed"}
    output.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return npz_path, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="required JSON evidence output (NPZ shares its stem)")
    parser.add_argument("--lance", type=Path, default=None)
    parser.add_argument("--isaac-source-root", type=Path, default=None)
    parser.add_argument("--object-name", default="cube1")
    parser.add_argument("--action", default="02")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--uuid", default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--mjx-device", choices=("cpu", "gpu"), default="cpu")
    return parser


def main() -> int:
    args = _parser().parse_args()
    native, mjx = _make_envs(args)
    report, arrays = compare_rollouts(native, mjx, steps=args.steps)
    npz_path, digest = write_evidence(args.output, report, arrays)
    print(json.dumps({"json": str(args.output), "npz": str(npz_path), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
