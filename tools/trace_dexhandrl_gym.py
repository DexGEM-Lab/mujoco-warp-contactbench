#!/usr/bin/env python3
"""Runtime-only exact Gym trace wrapper; never writes the Gym checkout.

Importing this module or requesting --help is offline-safe. ``run`` imports Isaac
Gym first, snapshots the source checkout, and only monkeypatches live objects.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, Iterator
import json
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sim.dexhandrl.obs_layout import OBS_COMPONENT_SLICES
from sim.dexhandrl.trace_contract import EXACT_IDENTITY, SCHEMA, validate_trace

GYM_ROOT = Path("/home/jay/dexrobot/FromSSH/dexhand_reconstruction/dexrobot_isaac")

def _git_snapshot(root: Path) -> tuple[str, str]:
    return (subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1"], text=True),
            subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip())

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--steps", type=int, default=3)
    p.add_argument("--gym-root", type=Path, default=GYM_ROOT); p.add_argument("--checkpoint", type=Path, required=True)
    return p

def _require(instance: Any, *names: str) -> None:
    absent = [name for name in names if not hasattr(instance, name)]
    if absent:
        raise RuntimeError(f"Gym trace hook unavailable: {type(instance).__name__} lacks {absent}")


@contextmanager
def _patch(owner: Any, name: str, replacement: Any) -> Iterator[None]:
    """Temporarily replace one runtime attribute and restore it even on failure."""
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


@contextmanager
def _preserve_bounded_run_play(train: Any, runner_type: Any, bounded_run_play: Callable[..., Any]) -> Iterator[None]:
    """Reinstall the trace hook after Gym replaces ``Runner.run_play`` at startup."""
    original_apply = train.apply_rl_games_patches

    def apply_patches_then_restore_trace_hook(*patch_args: Any, **patch_kwargs: Any) -> Any:
        result = original_apply(*patch_args, **patch_kwargs)
        runner_type.run_play = bounded_run_play
        print("[trace-hook] reinstalled bounded Runner.run_play after Gym patches")
        return result

    with _patch(train, "apply_rl_games_patches", apply_patches_then_restore_trace_hook):
        yield


def _filter_exact_trajectory(result: dict[tuple[str, str], list[dict]]) -> dict[tuple[str, str], list[dict]]:
    """Fail closed: retain the one Lance row used by the checkpoint trace."""
    selected = [item for items in result.values() for item in items if
                item.get("row_id") == EXACT_IDENTITY["row_id"] and
                item.get("uuid") == EXACT_IDENTITY["uuid"] and
                item.get("object_type") == EXACT_IDENTITY["object_name"] and
                item.get("action_id") == EXACT_IDENTITY["action"]]
    if len(selected) != 1:
        raise RuntimeError(f"exact Lance isolation failed: expected one row {EXACT_IDENTITY['row_id']}, got {len(selected)}")
    return {(EXACT_IDENTITY["object_name"], EXACT_IDENTITY["action"]): selected}


def _as_np(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=True)


def _deterministic_action(mu, low, high, clip_actions: bool):
    """The deterministic branch of rl_games PpoPlayerContinuous.get_action."""
    if not clip_actions:
        return mu
    return low + (high - low) * (mu.clamp(-1.0, 1.0) + 1.0) / 2.0


def _actual_env(player: Any) -> Any:
    env = player.env
    # rl_games' registered IVecEnv holds the DexHandBase under `.env`.
    while hasattr(env, "env"):
        env = env.env
    _require(env, "obs_buf", "observation_encoder", "action_processor", "pre_physics_step", "post_physics_step", "dof_pos", "dof_vel")
    return env


@contextmanager
def _close_env_after_trace(env: Any) -> Iterator[None]:
    """Close the live Isaac environment once without masking a trace failure."""
    try:
        yield
    except BaseException:
        try:
            env.close()
        except BaseException:
            pass
        raise
    else:
        env.close()


def _is_relative_to(path: Path, root: Path) -> bool:
    """Python 3.8-compatible containment predicate for resolved paths."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_trace_paths(args: argparse.Namespace, caller_cwd: Path | None = None) -> tuple[Path, Path, Path]:
    """Resolve trace outputs before entering the Gym checkout and reject source writes."""
    caller_cwd = (caller_cwd or Path.cwd()).resolve()
    gym_root = args.gym_root.resolve()
    output = (caller_cwd / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    runtime = (output.parent / "gym_runtime").resolve()
    for name, path in (("output", output), ("runtime", runtime)):
        if _is_relative_to(path, gym_root):
            raise ValueError(f"Gym trace {name} must be outside Gym root {gym_root}: {path}")
    return gym_root, output, runtime


def _validated_component_slices(runtime_slices: Any, policy_observation_keys: Any) -> dict[str, tuple[int, int]]:
    """Fail closed unless Gym's live policy layout exactly matches our contract."""
    try:
        actual = {str(key): tuple(runtime_slices[key]) for key in policy_observation_keys}
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Gym observation encoder did not expose policy component slices") from exc
    expected = {name: tuple(bounds) for name, bounds in OBS_COMPONENT_SLICES.items()}
    if tuple(actual.items()) != tuple(expected.items()):
        raise RuntimeError(f"Gym policy observation slices differ from OBS_COMPONENT_SLICES: {actual!r}")
    return actual


def _pre_action_observations(preprocessed: Any, rms: Any) -> tuple[np.ndarray, np.ndarray]:
    """Copy the checkpoint input before env_step can overwrite its backing buffer."""
    return _as_np(preprocessed[0]), _as_np(rms[0])


def _restore_mapped_checkpoint(player: Any, checkpoint_path: Any, torch_module: Any) -> None:
    """Restore a player checkpoint onto its configured device without rl_games retries."""
    if not checkpoint_path:
        raise RuntimeError("Gym trace requires a checkpoint path")
    try:
        checkpoint = torch_module.load(checkpoint_path, map_location=player.device, weights_only=False)
    except TypeError as exc:
        # PyTorch before weights_only support: retry the same mapped load explicitly.
        if "weights_only" not in str(exc):
            raise
        checkpoint = torch_module.load(checkpoint_path, map_location=player.device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Gym trace checkpoint must be a mapping containing 'model'")
    if "model" not in checkpoint:
        raise RuntimeError("Gym trace checkpoint is missing required 'model' state")
    player.model.load_state_dict(checkpoint["model"], strict=True)
    if getattr(player, "normalize_input", False) and "running_mean_std" in checkpoint:
        player.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
    env_state = checkpoint.get("env_state", None)
    if player.env is not None and env_state is not None:
        player.env.set_env_state(env_state)


def _hydra_overrides(args: argparse.Namespace) -> list[str]:
    """Overrides for the single exact checkpoint trace, independent of Hydra's CLI."""
    return [
        "task=Dexhand021proReconstruction",
        "train.test=true",
        f"train.checkpoint={args.checkpoint}",
        "train.reloadInterval=0",
        "env.numEnvs=1",
        "env.device=cuda:0",
        "env.viewer=false",
        "env.videoRecord=false",
        "env.videoStream=false",
        "task.mocap.format=lance",
        "task.mocap.basePath=/mnt/nas-222-project/mocap/for_retargeting/dexhand021pro/results_all_mano/all/lance_new_all_generated_mano/lance_result_archive_full/dexhand021pro_lance_new_all_generated_mano_retarget_result_archive.lance",
        "task.mocap.objectType=[cube1]",
        "task.mocap.actions=[01]",
        "task.mocap.load_multi_object=false",
        "task.use_residual=true",
        "task.pointCloudEncoding.isDynamic=false",
        "task.contact_force_threshold=1000000000",
        "train.seed=0",
    ]


def _compose_train_config(args: argparse.Namespace, initialize_config_dir: Callable[..., Any],
                          compose: Callable[..., Any]) -> Any:
    """Compose train's config from the checkout, never through an importable package."""
    config_dir = (args.gym_root / "dexhand_env" / "cfg").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="config", overrides=_hydra_overrides(args))


def _write_trace(output: Path, store: dict[str, list[np.ndarray]], checkpoint: Path, source_head: str,
                 component_slices: dict[str, tuple[int, int]]) -> Path:
    scalar = {"decision_frame": np.int64, "state_frame": np.int64, "reset": np.bool_, "terminated": np.bool_, "truncated": np.bool_}
    arrays = {name: np.asarray(values, dtype=scalar.get(name, np.float32)) for name, values in store.items()}
    metadata = {"schema": SCHEMA, "backend": "gym", "identity": dict(EXACT_IDENTITY),
                "post_filters": ["velocity_clamp", "position_clamp"], "checkpoint": str(checkpoint),
                "gym_source_head": source_head, "observation_component_slices": {k: list(v) for k, v in component_slices.items()}}
    validate_trace(metadata, arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, metadata=np.asarray(json.dumps(metadata)), **arrays)
    return output


def run(args: argparse.Namespace) -> None:
    """Use the sibling Hydra entrypoint and patch only live Python objects."""
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("set PYTHONDONTWRITEBYTECODE=1 before a Gym trace")
    if not 1 <= args.steps <= 632:
        raise ValueError("--steps must be in 1..632")
    gym_root, output, runtime = _resolve_trace_paths(args)
    before = _git_snapshot(gym_root)
    old_cwd, old_path, old_argv = Path.cwd(), list(sys.path), list(sys.argv)
    runtime.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(gym_root)
        sys.path.insert(0, str(gym_root))
        import isaacgym  # noqa: F401  # Isaac Gym must precede torch/train import
        import torch
        import train
        from hydra import compose, initialize_config_dir
        from dexhand_env.tasks.dexhand_base import DexHandBase
        from dexhand_env.utils.batch_mocap_loader import BatchMocapLoader
        from dexhand_env.utils.experiment_manager import ExperimentManager
        from rl_games.torch_runner import Runner, _override_sigma

        original_discover = BatchMocapLoader._discover_minimal_lance_trajectories
        def discover_exact(loader, ds):
            return _filter_exact_trajectory(original_discover(loader, ds))

        # Keep train.py's relative runs/ and runs_all/ writes outside the source checkout.
        def external_manager(cfg):
            manager = ExperimentManager(getattr(cfg.train.experiment, "maxTrainRuns", 10), getattr(cfg.train.experiment, "maxTestRuns", 10))
            manager.runs_dir, manager.runs_all_dir = runtime / "runs", runtime / "runs_all"
            manager._ensure_directories()
            return manager

        store: dict[str, list[np.ndarray]] = {name: [] for name in (
            "decision_frame", "state_frame", "reset", "terminated", "truncated", "processed_observation", "rms_observation", "policy_mean", "final_action", "base_offset_before", "base_offset_after", "joint_offset_before", "joint_offset_after", "reference_active_target", "corrected_active_target", "applied_active_target", "full_target", "hand_qpos", "hand_qvel", "object_position", "object_quaternion_xyzw", "object_linear_velocity", "target_object_position", "target_object_quaternion_xyzw")}
        captured: dict[str, np.ndarray] = {}

        original_pre, original_post, original_reset = DexHandBase.pre_physics_step, DexHandBase.post_physics_step, DexHandBase.reset_idx
        def pre_capture(env, actions):
            before_base = env.task.task_states["base_position_offset"].clone()
            before_joint = env.task.task_states["joint_offset"].clone()
            result = original_pre(env, actions)
            ap = env.action_processor
            _require(ap, "active_rule_targets", "active_raw_targets", "active_next_targets", "full_dof_targets")
            captured.update(base_offset_before=_as_np(before_base[0]), joint_offset_before=_as_np(before_joint[0]),
                            reference_active_target=_as_np(ap.active_rule_targets[0]), corrected_active_target=_as_np(ap.active_raw_targets[0]),
                            applied_active_target=_as_np(ap.active_next_targets[0]), full_target=_as_np(ap.full_dof_targets[0]))
            return result
        def reset_deferred(env, env_ids):
            # Preserve the terminal state only at the final trace action; initial reset remains normal.
            if getattr(env, "_trace_defer_reset", False):
                return None
            return original_reset(env, env_ids)
        def post_capture(env):
            return original_post(env)

        def run_play(runner, play_args):
            player = runner.create_player()
            env = _actual_env(player)
            # Gym's patch exposes this for its reload machinery; preserve that compatibility.
            runner.player = player
            with _close_env_after_trace(env):
                _restore_mapped_checkpoint(player, play_args.get("checkpoint"), torch)
                _override_sigma(player, play_args)
                if env.num_envs != 1:
                    raise RuntimeError(f"trace requires one Gym env, got {env.num_envs}")
                try:
                    trajectory = env.task.groups[0]["trajectories"][0]
                except (AttributeError, IndexError, KeyError) as exc:
                    raise RuntimeError("task did not expose the allocated Lance trajectory") from exc
                for key, expected in (("uuid", EXACT_IDENTITY["uuid"]), ("row_id", EXACT_IDENTITY["row_id"])):
                    if trajectory.get(key) != expected:
                        raise RuntimeError(f"task loaded wrong trajectory {key}={trajectory.get(key)!r}")
                if (int(env.task.trajectory_start_frame[0]), int(env.task.trajectory_end_frame[0])) != (61, 693):
                    raise RuntimeError("task trajectory window is not exactly 61..693")
                obs = player.env_reset(player.env)
                if tuple(obs.shape[-1:]) != (461,) or int(env.obs_buf.shape[-1]) != 461:
                    raise RuntimeError(f"Gym observation width is not 461: player={tuple(obs.shape)}, env={tuple(env.obs_buf.shape)}")
                component_slices = _validated_component_slices(
                    env.observation_encoder.component_slice_indices,
                    env.observation_encoder.observation_keys,
                )
                for step in range(args.steps):
                    decision = 61 + step
                    preprocessed = player._preproc_obs(obs)
                    rms = player.model.norm_obs(preprocessed)
                    processed_observation, rms_observation = _pre_action_observations(preprocessed, rms)
                    input_dict = {"is_train": False, "prev_actions": None, "obs": preprocessed, "rnn_states": player.states}
                    with torch.no_grad():
                        model_result = player.model(input_dict)
                    mus = model_result["mus"]
                    player.states = model_result["rnn_states"]
                    action = _deterministic_action(mus, player.actions_low, player.actions_high, player.clip_actions)
                    # Defer the automatic terminal reset before the terminal step, preserving its state frame.
                    env._trace_defer_reset = decision == EXACT_IDENTITY["effective_window"][1] - 1
                    # `env.step` owns the actual action boundary and our pre hook snapshots its authoritative targets.
                    next_obs, _reward, done, _info = player.env_step(player.env, action)
                    state = decision + 1
                    if not captured:
                        raise RuntimeError("pre_physics_step did not expose action stages")
                    rb = env.rigid_body_states[0, env.task.object_local_rigid_body_index]
                    _ = done  # termination is recorded below from reset_buf after post physics.
                    values = dict(decision_frame=np.asarray(decision), state_frame=np.asarray(state), reset=np.asarray(step == 0),
                        terminated=np.asarray(bool(env.reset_buf[0].item())), truncated=np.asarray(False), processed_observation=processed_observation, rms_observation=rms_observation, policy_mean=_as_np(mus[0]), final_action=_as_np(action[0]),
                        base_offset_after=_as_np(env.task.task_states["base_position_offset"][0]), joint_offset_after=_as_np(env.task.task_states["joint_offset"][0]), hand_qpos=_as_np(env.dof_pos[0]), hand_qvel=_as_np(env.dof_vel[0]),
                        object_position=_as_np(rb[:3]), object_quaternion_xyzw=_as_np(rb[3:7]), object_linear_velocity=_as_np(rb[7:10]),
                        target_object_position=_as_np(env.obs_dict["target_object_position"][0]), target_object_quaternion_xyzw=_as_np(env.obs_dict["target_object_orientation"][0]))
                    values.update(captured)
                    for name, value in values.items(): store[name].append(value)
                    obs = next_obs
                _write_trace(output, store, args.checkpoint, before[1], component_slices)

        with _patch(BatchMocapLoader, "_discover_minimal_lance_trajectories", discover_exact), _patch(DexHandBase, "pre_physics_step", pre_capture), _patch(DexHandBase, "post_physics_step", post_capture), _patch(DexHandBase, "reset_idx", reset_deferred), _patch(train, "create_experiment_manager", external_manager), _patch(Runner, "run_play", run_play), _preserve_bounded_run_play(train, Runner, run_play):
            cfg = _compose_train_config(args, initialize_config_dir, compose)
            train_main = getattr(train.main, "__wrapped__", None)
            if train_main is None:
                raise RuntimeError("Gym train.main must expose Hydra's undecorated body via __wrapped__")
            train_main(cfg)
        if not output.exists():
            raise RuntimeError("patched Runner.run_play returned without writing the Gym trace")
    finally:
        sys.argv[:] = old_argv
        os.chdir(old_cwd)
        sys.path[:] = old_path
        after = _git_snapshot(gym_root)
        if after != before:
            raise RuntimeError("Gym source checkout changed during trace wrapper execution")

def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv); run(args); return 0
if __name__ == "__main__": raise SystemExit(main())
