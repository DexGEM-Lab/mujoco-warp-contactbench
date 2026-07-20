"""skrl launcher for the DexHandRL MJX environment."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from sim.dexhandrl.lance_loader import list_trajectories
from sim.dexhandrl.rlgames_checkpoint import RlGamesRunningMeanStd, load_rlgames_checkpoint
from sim.dexhandrl.skrl_env import DexHandRLMJXEnv, DexHandRLMJXEnvConfig
from sim.dexhandrl.constants import DEFAULT_ISAAC_SOURCE_ROOT, DEFAULT_OUTPUT_ROOT

try:
    from sim.dexhandrl.skrl_models import SkrlDexHandModelConfig, build_skrl_models
except ImportError as exc:  # pragma: no cover - optional dependency surface
    SkrlDexHandModelConfig = None
    build_skrl_models = None
    _SKRL_MODELS_IMPORT_ERROR = exc


@dataclass(frozen=True)
class SkrlLauncherConfig:
    device: str = "cpu"
    num_envs: int = 8
    lance_path: Path = DexHandRLMJXEnvConfig().lance_path
    isaac_source_root: Path = DEFAULT_ISAAC_SOURCE_ROOT
    object_name: str = "cube1"
    action: str = "01"
    sequence_index: int = 0
    uuid: str | None = None
    use_residual: bool = True
    early_phase_frames: int = 130
    pre_contact_frames: int = 200
    post_contact_frames: int = 250
    object_position_terminal_threshold: float = 0.1
    substeps: int = 10
    finger_kp: float = 15.0
    finger_kv: float = 0.0
    finger_force: float = 50.0
    hand_self_collision_mode: str = "disable_within_finger"
    object_friction: str = "0.5 0.01 0.001"
    naconmax: int = 8192
    njmax: int = 8192
    seed: int = 0
    timesteps: int = 250_000
    rollouts: int = 48
    learning_epochs: int = 3
    mini_batches: int = 48
    learning_rate: float = 1e-4
    discount_factor: float = 0.995
    gae_lambda: float = 0.95
    random_timesteps: int = 0
    learning_starts: int = 0
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.0
    value_loss_scale: float = 4.0
    grad_norm_clip: float = 1.0
    kl_threshold: float = 0.008
    mixed_precision: bool = False
    time_limit_bootstrap: bool = True
    headless: bool = True
    render_interval: int = 1000
    stochastic_evaluation: bool = False
    disable_progressbar: bool = False
    environment_info: str = "episode"
    output_dir: Path = DEFAULT_OUTPUT_ROOT / "skrl_dexhandrl"
    mode: str = "train"
    rollout_max_steps: int = 0
    rollout_keep_observation: bool = False
    checkpoint: Path | None = None
    rlgames_checkpoint: Path | None = None
    pointnet_enabled: bool = True
    film_enabled: bool = True
    separate: bool = True


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _instantiate_config(default_config: Any) -> Any:
    return default_config() if callable(default_config) else copy.deepcopy(default_config)


def _set_cfg_value(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def _configure_experiment_output(agent_cfg: Any, output_dir: Path) -> None:
    output = Path(output_dir)
    experiment = (
        agent_cfg.get("experiment")
        if isinstance(agent_cfg, dict)
        else getattr(agent_cfg, "experiment", None)
    )
    if experiment is None:
        experiment = {}
        _set_cfg_value(agent_cfg, "experiment", experiment)
    _set_cfg_value(experiment, "directory", str(output.parent))
    _set_cfg_value(experiment, "experiment_name", output.name)


def _stack_vector_states(vector_env: Any):
    import numpy as np

    states = [np.asarray(sub_env.state(), dtype=np.float32) for sub_env in vector_env.envs]
    return np.stack(states, axis=0)


def _require_skrl() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from skrl.agents.torch.ppo import PPO, PPO_CFG
        from skrl.envs.wrappers.torch import wrap_env
        from skrl.memories.torch import RandomMemory
        from skrl.trainers.torch import SequentialTrainer
        return PPO, PPO_CFG, wrap_env, RandomMemory, SequentialTrainer
    except Exception:  # pragma: no cover - dependency-specific failure
        try:
            from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
            from skrl.envs.wrappers.torch import wrap_env
            from skrl.memories.torch import RandomMemory
            from skrl.trainers.torch import SequentialTrainer
            return PPO, PPO_DEFAULT_CONFIG, wrap_env, RandomMemory, SequentialTrainer
        except Exception as inner_exc:  # pragma: no cover - dependency-specific failure
            raise RuntimeError(
                "skrl is required for this launcher. Install skrl, torch, and gymnasium in the MuJoCo env first."
            ) from inner_exc


def _resolve_trajectory_specs(cfg: SkrlLauncherConfig) -> list[dict[str, Any]]:
    rows = list_trajectories(cfg.lance_path, limit=None)
    selected = [row for row in rows if row["object_name"] == cfg.object_name and row["action"] == cfg.action]
    if not selected:
        raise RuntimeError(
            f"no trajectories found for object={cfg.object_name!r}, action={cfg.action!r} in {cfg.lance_path}"
        )
    if cfg.uuid:
        matching = [
            {**row, "sequence_index": index}
            for index, row in enumerate(selected)
            if row["uuid"] == cfg.uuid
        ]
        if not matching:
            raise RuntimeError(f"trajectory uuid {cfg.uuid!r} not found for {cfg.object_name}/{cfg.action}")
        return [dict(matching[0]) for _ in range(max(cfg.num_envs, 1))]

    start = cfg.sequence_index % len(selected)
    sequence_indices = [
        (start + env_index) % len(selected)
        for env_index in range(max(cfg.num_envs, 1))
    ]
    return [
        {**selected[sequence_index], "sequence_index": sequence_index}
        for sequence_index in sequence_indices
    ]


def _make_env_factory(cfg: SkrlLauncherConfig, spec: dict[str, Any], env_index: int):
    env_cfg = DexHandRLMJXEnvConfig(
        lance_path=cfg.lance_path,
        object_name=cfg.object_name,
        action=cfg.action,
        sequence_index=int(spec.get("sequence_index", 0)),
        uuid=spec.get("uuid"),
        device=cfg.device,
        use_residual=cfg.use_residual,
        isaac_source_root=cfg.isaac_source_root,
        early_phase_frames=cfg.early_phase_frames,
        pre_contact_frames=cfg.pre_contact_frames,
        post_contact_frames=cfg.post_contact_frames,
        object_position_terminal_threshold=cfg.object_position_terminal_threshold,
        substeps=cfg.substeps,
        finger_kp=cfg.finger_kp,
        finger_kv=cfg.finger_kv,
        finger_force=cfg.finger_force,
        hand_self_collision_mode=cfg.hand_self_collision_mode,
        object_friction=cfg.object_friction,
        naconmax=cfg.naconmax,
        njmax=cfg.njmax,
        seed=cfg.seed + env_index,
    )

    def _factory():
        return DexHandRLMJXEnv(env_cfg)

    return _factory


def _build_wrapped_env(cfg: SkrlLauncherConfig):
    _, _, wrap_env, _, _ = _require_skrl()
    try:
        from gymnasium.vector import SyncVectorEnv
    except Exception as exc:  # pragma: no cover - dependency-specific failure
        raise RuntimeError("gymnasium is required to build the skrl vectorized environment") from exc

    specs = _resolve_trajectory_specs(cfg)
    env_fns = [_make_env_factory(cfg, spec, idx) for idx, spec in enumerate(specs)]
    env = SyncVectorEnv(env_fns)
    env.state_space = env.single_observation_space
    env.state = lambda: _stack_vector_states(env)
    expected_device = "cuda" if cfg.device == "gpu" else "cpu"
    env.device = expected_device
    wrapped = wrap_env(env)
    if torch.device(wrapped.device).type != torch.device(expected_device).type:
        raise RuntimeError(
            f"skrl wrapper device {wrapped.device} does not match launcher device "
            f"{expected_device}"
        )
    return wrapped, specs


def _build_agent(env, cfg: SkrlLauncherConfig):
    if build_skrl_models is None or SkrlDexHandModelConfig is None:  # pragma: no cover - dependency-specific failure
        raise RuntimeError(
            "skrl model helpers are unavailable. Install skrl, torch, and gymnasium in the MuJoCo env first."
        ) from _SKRL_MODELS_IMPORT_ERROR
    PPO, PPO_DEFAULT_CONFIG, _, RandomMemory, _ = _require_skrl()
    from skrl.resources.schedulers.torch import KLAdaptiveLR
    model_cfg = SkrlDexHandModelConfig(
        pointnet_enabled=cfg.pointnet_enabled,
        film_enabled=cfg.film_enabled,
        separate=cfg.separate,
    )
    device = "cuda" if cfg.device == "gpu" else "cpu"
    state_space = getattr(env, "state_space", None) or env.observation_space
    models = build_skrl_models(env.observation_space, state_space, env.action_space, device, model_cfg)
    memory = RandomMemory(memory_size=max(cfg.rollouts, 1), num_envs=env.num_envs, device="cuda" if cfg.device == "gpu" else "cpu")

    agent_cfg = _instantiate_config(PPO_DEFAULT_CONFIG)
    _configure_experiment_output(agent_cfg, cfg.output_dir)
    _set_cfg_value(agent_cfg, "rollouts", int(cfg.rollouts))
    _set_cfg_value(agent_cfg, "learning_epochs", int(cfg.learning_epochs))
    _set_cfg_value(agent_cfg, "mini_batches", int(cfg.mini_batches))
    _set_cfg_value(agent_cfg, "learning_rate", float(cfg.learning_rate))
    _set_cfg_value(agent_cfg, "learning_rate_scheduler", KLAdaptiveLR)
    _set_cfg_value(
        agent_cfg,
        "learning_rate_scheduler_kwargs",
        {"kl_threshold": float(cfg.kl_threshold)},
    )
    _set_cfg_value(agent_cfg, "discount_factor", float(cfg.discount_factor))
    _set_cfg_value(agent_cfg, "gae_lambda", float(cfg.gae_lambda))
    _set_cfg_value(agent_cfg, "random_timesteps", int(cfg.random_timesteps))
    _set_cfg_value(agent_cfg, "learning_starts", int(cfg.learning_starts))
    _set_cfg_value(agent_cfg, "ratio_clip", float(cfg.ratio_clip))
    _set_cfg_value(agent_cfg, "value_clip", float(cfg.value_clip))
    _set_cfg_value(agent_cfg, "entropy_loss_scale", float(cfg.entropy_loss_scale))
    _set_cfg_value(agent_cfg, "value_loss_scale", float(cfg.value_loss_scale))
    _set_cfg_value(agent_cfg, "grad_norm_clip", float(cfg.grad_norm_clip))
    _set_cfg_value(agent_cfg, "kl_threshold", float(cfg.kl_threshold))
    _set_cfg_value(agent_cfg, "mixed_precision", bool(cfg.mixed_precision))
    _set_cfg_value(agent_cfg, "time_limit_bootstrap", bool(cfg.time_limit_bootstrap))
    _set_cfg_value(agent_cfg, "observation_preprocessor", RlGamesRunningMeanStd)
    _set_cfg_value(
        agent_cfg,
        "observation_preprocessor_kwargs",
        {"size": env.observation_space, "device": device},
    )
    _set_cfg_value(agent_cfg, "state_preprocessor", RlGamesRunningMeanStd)
    _set_cfg_value(
        agent_cfg,
        "state_preprocessor_kwargs",
        {"size": state_space, "device": device},
    )
    _set_cfg_value(agent_cfg, "value_preprocessor", RlGamesRunningMeanStd)
    _set_cfg_value(
        agent_cfg,
        "value_preprocessor_kwargs",
        {"size": 1, "device": device},
    )

    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        state_space=state_space,
        device=device,
    )
    return agent, model_cfg


def _build_trainer_cfg(cfg: SkrlLauncherConfig) -> dict[str, Any]:
    return {
        "timesteps": int(cfg.timesteps),
        "headless": bool(cfg.headless),
        "render_interval": max(int(cfg.render_interval), 1),
        "stochastic_evaluation": bool(cfg.stochastic_evaluation),
        "disable_progressbar": bool(cfg.disable_progressbar),
        "close_environment_at_exit": True,
        "environment_info": cfg.environment_info,
    }


def _write_launcher_config(cfg: SkrlLauncherConfig, specs: list[dict[str, Any]], model_cfg: SkrlDexHandModelConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(asdict(cfg))
    payload["resolved_trajectories"] = _to_jsonable(specs)
    payload["model_config"] = _to_jsonable(asdict(model_cfg))
    with (cfg.output_dir / "launcher_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _inspect(env, agent) -> None:
    obs, info = env.reset()
    logger.info(f"Reset observation shape: {getattr(obs, 'shape', None)}")
    logger.info(f"Reset info keys: {sorted(info.keys())}")
    policy = getattr(agent, "policy", None)
    value = getattr(agent, "value", None)
    if policy is None or value is None:
        raise RuntimeError("skrl agent does not expose policy/value models as expected")
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
    normalized_obs = agent._observation_preprocessor(obs_tensor)
    normalized_state = agent._state_preprocessor(obs_tensor)
    actions, _ = policy.act({"observations": normalized_obs}, role="policy")
    values, _ = value.act({"states": normalized_state}, role="value")
    logger.info(f"Policy action shape: {tuple(actions.shape)}")
    logger.info(f"Value shape: {tuple(values.shape)}")


def _first_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].cpu())
    try:
        import numpy as np

        return float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return float(value)


def _first_vector(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        import numpy as np

        array = np.asarray(value)
    if array.ndim > 1:
        array = array[0]
    return [float(item) for item in array.reshape(-1)]


def _rollout(env, agent, cfg: SkrlLauncherConfig) -> dict[str, Any]:
    """Run one deterministic trajectory and save a compact acceptance report."""

    set_mode = getattr(agent, "set_running_mode", None)
    if callable(set_mode):
        set_mode("eval")
    observations, reset_info = env.reset()
    max_steps = int(cfg.rollout_max_steps) if cfg.rollout_max_steps > 0 else 10_000
    rewards: list[float] = []
    object_errors: list[float] = []
    action_abs_means: list[float] = []
    action_abs_maxes: list[float] = []
    records: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    last_info = reset_info

    for _ in range(max_steps):
        observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=agent.policy.device)
        normalized = agent._observation_preprocessor(observation_tensor)
        with torch.no_grad():
            mean_actions, _ = agent.policy.compute({"observations": normalized}, role="policy")
            actions = torch.clamp(mean_actions, -1.0, 1.0)
        observations, reward, terminated_value, truncated_value, last_info = env.step(actions)
        rewards.append(_first_scalar(reward))
        if "object_position_error_m" in last_info:
            object_errors.append(_first_scalar(last_info["object_position_error_m"]))
        action_abs = actions.detach().abs()
        action_abs_means.append(float(action_abs.mean().cpu()))
        action_abs_maxes.append(float(action_abs.max().cpu()))
        base_offset = (
            _first_vector(last_info["base_position_offset"])
            if "base_position_offset" in last_info
            else []
        )
        joint_offset = (
            _first_vector(last_info["joint_offset"])
            if "joint_offset" in last_info
            else []
        )
        records.append(
            {
                "step": len(rewards) - 1,
                "lance_frame": (
                    int(_first_scalar(last_info["current_frame"]))
                    if "current_frame" in last_info
                    else None
                ),
                "reward": rewards[-1],
                "policy_action": _first_vector(actions),
                "object_position_error_m": object_errors[-1] if object_errors else None,
                "object_position": (
                    _first_vector(last_info["object_position"])
                    if "object_position" in last_info
                    else []
                ),
                "target_object_position": (
                    _first_vector(last_info["target_object_position"])
                    if "target_object_position" in last_info
                    else []
                ),
                "hand_active_l2_error": (
                    _first_scalar(last_info["hand_active_l2_error"])
                    if "hand_active_l2_error" in last_info
                    else None
                ),
                "finger_active_mean_abs_error_rad": (
                    _first_scalar(last_info["finger_active_mean_abs_error_rad"])
                    if "finger_active_mean_abs_error_rad" in last_info
                    else None
                ),
                "finger_active_max_abs_error_rad": (
                    _first_scalar(last_info["finger_active_max_abs_error_rad"])
                    if "finger_active_max_abs_error_rad" in last_info
                    else None
                ),
                "active_contact_count": (
                    int(_first_scalar(last_info["active_contact_count"]))
                    if "active_contact_count" in last_info
                    else None
                ),
                "action_abs_mean": action_abs_means[-1],
                "action_abs_max": action_abs_maxes[-1],
                "base_position_offset": base_offset,
                "joint_offset_abs_max": max((abs(value) for value in joint_offset), default=0.0),
            }
        )
        if cfg.rollout_keep_observation:
            records[-1]["policy_observation"] = _first_vector(observations)
        terminated = bool(_first_scalar(terminated_value))
        truncated = bool(_first_scalar(truncated_value))
        if terminated or truncated:
            break

    if not rewards:
        raise RuntimeError("deterministic rollout produced no steps")
    result = {
        "schema": "dexhandrl.skrl_rollout.v1",
        "object_name": cfg.object_name,
        "action": cfg.action,
        "sequence_index": int(cfg.sequence_index),
        "frames": len(rewards),
        "terminated": terminated,
        "truncated": truncated,
        "reward_sum": float(sum(rewards)),
        "reward_mean": float(sum(rewards) / len(rewards)),
        "object_position_error_m": {
            "mean": float(sum(object_errors) / len(object_errors)) if object_errors else None,
            "max": float(max(object_errors)) if object_errors else None,
            "final": float(object_errors[-1]) if object_errors else None,
        },
        "action_abs": {
            "mean": float(sum(action_abs_means) / len(action_abs_means)),
            "max": float(max(action_abs_maxes)),
        },
        "final_frame": (
            int(_first_scalar(last_info["current_frame"]))
            if "current_frame" in last_info
            else None
        ),
        "records": records,
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.output_dir / "rollout_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    logger.info(f"Deterministic rollout report: {output_path}")
    logger.info(json.dumps({key: value for key, value in result.items() if key != "records"}, sort_keys=True))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DexHandRL skrl launcher")
    parser.add_argument("--mode", choices=("train", "eval", "inspect", "rollout"), default="train")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--lance-path",
        type=Path,
        default=DexHandRLMJXEnvConfig().lance_path,
    )
    parser.add_argument("--isaac-source-root", type=Path, default=DEFAULT_ISAAC_SOURCE_ROOT)
    parser.add_argument("--object-name", default="cube1")
    parser.add_argument("--action", default="01")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--uuid", default=None)
    parser.add_argument("--use-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-phase-frames", type=int, default=130)
    parser.add_argument("--pre-contact-frames", type=int, default=200)
    parser.add_argument("--post-contact-frames", type=int, default=250)
    parser.add_argument("--object-position-terminal-threshold", type=float, default=0.1)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--finger-kp", type=float, default=15.0)
    parser.add_argument("--finger-kv", type=float, default=0.0)
    parser.add_argument("--finger-force", type=float, default=50.0)
    parser.add_argument(
        "--hand-self-collision-mode",
        choices=("none", "disable_within_finger", "all_disabled"),
        default="disable_within_finger",
    )
    parser.add_argument("--object-friction", default="0.5 0.01 0.001")
    parser.add_argument("--naconmax", type=int, default=8192)
    parser.add_argument("--njmax", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument(
        "--rollout-max-steps",
        type=int,
        default=0,
        help="Maximum deterministic rollout steps (0 runs until termination/truncation)",
    )
    parser.add_argument(
        "--rollout-keep-observation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the processed 461D policy observation in every rollout record",
    )
    parser.add_argument("--rollouts", type=int, default=48)
    parser.add_argument("--learning-epochs", type=int, default=3)
    parser.add_argument("--mini-batches", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--discount-factor", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--random-timesteps", type=int, default=0)
    parser.add_argument("--learning-starts", type=int, default=0)
    parser.add_argument("--ratio-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--entropy-loss-scale", type=float, default=0.0)
    parser.add_argument("--value-loss-scale", type=float, default=4.0)
    parser.add_argument("--grad-norm-clip", type=float, default=1.0)
    parser.add_argument("--kl-threshold", type=float, default=0.008)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--time-limit-bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-interval", type=int, default=1000)
    parser.add_argument(
        "--stochastic-evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--disable-progressbar", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--environment-info", default="episode")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "skrl_dexhandrl")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--rlgames-checkpoint",
        type=Path,
        default=None,
        help="Load a compatible IsaacGym rl-games checkpoint (weights and running statistics)",
    )
    parser.add_argument("--pointnet-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--film-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--separate", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.checkpoint is not None and args.rlgames_checkpoint is not None:
        raise SystemExit("use only one of --checkpoint or --rlgames-checkpoint")
    launcher_cfg = SkrlLauncherConfig(
        device=args.device,
        num_envs=max(int(args.num_envs), 1),
        lance_path=args.lance_path,
        isaac_source_root=args.isaac_source_root,
        object_name=args.object_name,
        action=args.action,
        sequence_index=int(args.sequence_index),
        uuid=args.uuid,
        use_residual=bool(args.use_residual),
        early_phase_frames=int(args.early_phase_frames),
        pre_contact_frames=int(args.pre_contact_frames),
        post_contact_frames=int(args.post_contact_frames),
        object_position_terminal_threshold=float(args.object_position_terminal_threshold),
        substeps=int(args.substeps),
        finger_kp=float(args.finger_kp),
        finger_kv=float(args.finger_kv),
        finger_force=float(args.finger_force),
        hand_self_collision_mode=str(args.hand_self_collision_mode),
        object_friction=str(args.object_friction),
        naconmax=max(int(args.naconmax), 1),
        njmax=max(int(args.njmax), 1),
        seed=int(args.seed),
        timesteps=int(args.timesteps),
        rollout_max_steps=max(int(args.rollout_max_steps), 0),
        rollout_keep_observation=bool(args.rollout_keep_observation),
        rollouts=int(args.rollouts),
        learning_epochs=int(args.learning_epochs),
        mini_batches=int(args.mini_batches),
        learning_rate=float(args.learning_rate),
        discount_factor=float(args.discount_factor),
        gae_lambda=float(args.gae_lambda),
        random_timesteps=int(args.random_timesteps),
        learning_starts=int(args.learning_starts),
        ratio_clip=float(args.ratio_clip),
        value_clip=float(args.value_clip),
        entropy_loss_scale=float(args.entropy_loss_scale),
        value_loss_scale=float(args.value_loss_scale),
        grad_norm_clip=float(args.grad_norm_clip),
        kl_threshold=float(args.kl_threshold),
        mixed_precision=bool(args.mixed_precision),
        time_limit_bootstrap=bool(args.time_limit_bootstrap),
        headless=bool(args.headless),
        render_interval=int(args.render_interval),
        stochastic_evaluation=bool(args.stochastic_evaluation),
        disable_progressbar=bool(args.disable_progressbar),
        environment_info=str(args.environment_info),
        output_dir=args.output_dir,
        mode=args.mode,
        checkpoint=args.checkpoint,
        rlgames_checkpoint=args.rlgames_checkpoint,
        pointnet_enabled=bool(args.pointnet_enabled),
        film_enabled=bool(args.film_enabled),
        separate=bool(args.separate),
    )

    env, specs = _build_wrapped_env(launcher_cfg)
    agent, model_cfg = _build_agent(env, launcher_cfg)
    _write_launcher_config(launcher_cfg, specs, model_cfg)

    if launcher_cfg.rlgames_checkpoint is not None:
        report = load_rlgames_checkpoint(agent, launcher_cfg.rlgames_checkpoint)
        logger.info(
            "Loaded rl-games checkpoint: "
            f"{report.path} (epoch={report.epoch}, obs={report.observation_dim}, "
            f"policy_tensors={report.policy_tensors}, value_tensors={report.value_tensors})"
        )
    elif launcher_cfg.checkpoint is not None:
        load_fn = getattr(agent, "load", None)
        if callable(load_fn):
            load_fn(str(launcher_cfg.checkpoint))
        else:  # pragma: no cover - runtime-specific
            logger.warning("Agent checkpoint loading is not supported by this skrl version")

    if launcher_cfg.mode == "inspect":
        try:
            _inspect(env, agent)
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        return

    if launcher_cfg.mode == "rollout":
        try:
            _rollout(env, agent, launcher_cfg)
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        return

    _, _, _, _, SequentialTrainer = _require_skrl()
    trainer = SequentialTrainer(cfg=_build_trainer_cfg(launcher_cfg), env=env, agents=agent)
    if launcher_cfg.mode == "eval":
        run_fn = getattr(trainer, "eval", None)
        if callable(run_fn):
            run_fn()
        else:  # pragma: no cover - runtime-specific
            trainer.train()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
