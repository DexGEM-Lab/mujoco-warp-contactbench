from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def adapter() -> ManoGymnasiumVectorEnv:
    from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
    from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv
    from sim.manorl.trajectory import load_reference_trajectory

    physical = MujocoManoEnvironment(
        load_reference_trajectory(),
        EnvironmentConfig(residual_enabled=False, max_deviation_distance=1_000_000.0),
    )
    return ManoGymnasiumVectorEnv(physical)


def _model(adapter: ManoGymnasiumVectorEnv) -> ManoActorCritic:
    from sim.manorl.model import ManoActorCritic

    return ManoActorCritic(
        adapter.single_observation_space,
        None,
        adapter.single_action_space,
        device="cpu",
    )


def test_model_defaults_to_source_pointnet_film(adapter: ManoGymnasiumVectorEnv) -> None:
    import torch

    from sim.manorl.gymnasium_env import ACTION_DIM, OBSERVATION_DIM

    assert adapter.single_action_space.shape == (ACTION_DIM,)
    np.testing.assert_array_equal(adapter.single_action_space.low, np.full(ACTION_DIM, -1.0, dtype=np.float32))
    np.testing.assert_array_equal(adapter.single_action_space.high, np.full(ACTION_DIM, 1.0, dtype=np.float32))
    model = _model(adapter)
    keys = set(model.state_dict())
    assert model.use_film is True
    assert "actor_backbone.0.film.film_generator.weight" in keys
    assert model.actor_backbone[0].linear.in_features == 286

    observations = torch.zeros((2, OBSERVATION_DIM), dtype=torch.float32)
    observations[:, 266] = 1.0
    policy, policy_outputs = model.compute({"observations": observations}, role="policy")
    value, value_outputs = model.compute({"observations": observations}, role="value")
    assert policy.shape == (2, ACTION_DIM)
    assert policy_outputs["log_std"].shape == (2, ACTION_DIM)
    assert value.shape == (2, 1)
    assert value_outputs == {}


def test_policy_keeps_raw_samples_and_environment_clips_at_boundary(
    adapter: ManoGymnasiumVectorEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    from sim.manorl.gymnasium_env import OBSERVATION_DIM
    from sim.manorl.skrl_runtime import ManoPPOConfig, ManoSkrlRuntime

    runtime = ManoSkrlRuntime(adapter, ManoPPOConfig.optimizer_smoke())
    runtime.agent.enable_training_mode(True)
    with torch.no_grad():
        runtime.model.log_std.fill_(2.0)
    adapter.reset(seed=17)
    observations, _ = runtime.env.reset()
    torch.manual_seed(7)
    with torch.no_grad():
        raw_actions, _ = runtime.agent.act(
            observations, None, timestep=0, timesteps=runtime.config.rollouts
        )
    assert raw_actions.shape == (adapter.num_envs, 26)
    assert torch.any(torch.abs(raw_actions) > 1.0)

    captured: list[np.ndarray] = []
    original_step = adapter.environment.step

    def capture_step(actions: np.ndarray):
        captured.append(np.asarray(actions).copy())
        return original_step(actions)

    monkeypatch.setattr(adapter.environment, "step", capture_step)
    next_observations, rewards, terminated, truncated, infos = runtime.env.step(raw_actions)
    runtime.agent.record_transition(
        observations=observations,
        states=None,
        actions=raw_actions,
        rewards=rewards,
        next_observations=next_observations,
        next_states=None,
        terminated=terminated,
        truncated=truncated,
        infos=infos,
        timestep=0,
        timesteps=runtime.config.rollouts,
    )

    stored_actions = runtime.memory.get_tensor_by_name("actions")[0]
    torch.testing.assert_close(stored_actions, raw_actions)
    assert len(captured) == 1
    assert np.all(captured[0] <= 1.0) and np.all(captured[0] >= -1.0)
    np.testing.assert_allclose(
        captured[0], torch.clamp(raw_actions, -1.0, 1.0).cpu().numpy()
    )
    assert next_observations.shape == (adapter.num_envs, OBSERVATION_DIM)


def test_normalizer_uses_source_named_shared_xyz_statistics() -> None:
    import torch

    from sim.manorl.gymnasium_env import OBSERVATION_DIM
    from sim.manorl.normalization import PointCloudAwareRunningStandardScaler

    scaler = PointCloudAwareRunningStandardScaler()
    observations = torch.zeros((2, OBSERVATION_DIM), dtype=torch.float32)
    points = torch.arange(64 * 3, dtype=torch.float32).reshape(64, 3)
    observations[0, 74:266] = points.flatten()
    observations[1, 74:266] = (points + 100.0).flatten()
    output = scaler(observations, train=True)

    assert set(scaler.state_dict()) == {
        "running_mean", "running_var", "count", "pc_running_mean", "pc_running_var", "pc_count"
    }
    assert scaler.count.item() == 3.0
    assert scaler.pc_count.item() == 129.0
    np.testing.assert_allclose(
        scaler.running_mean[74:266].reshape(64, 3).detach().cpu().numpy(),
        np.broadcast_to(scaler.pc_running_mean.detach().cpu().numpy(), (64, 3)),
    )
    assert output.shape == observations.shape
    assert torch.isfinite(output).all()


def test_adapter_and_skrl_wrapper_preserve_vector_tensor_boundary(adapter: ManoGymnasiumVectorEnv) -> None:
    import torch

    from sim.manorl.gymnasium_env import ACTION_DIM, OBSERVATION_DIM
    from sim.manorl.skrl_runtime import ManoPPOConfig, ManoSkrlRuntime

    observations, info = adapter.reset(seed=17)
    assert observations.shape == (1, OBSERVATION_DIM)
    assert info["time_outs"].shape == (1,)
    assert adapter.metadata["autoreset_mode"].value == "NextStep"

    runtime = ManoSkrlRuntime(adapter, ManoPPOConfig.optimizer_smoke())
    assert runtime.agent.cfg.rewards_shaper is not None
    wrapped_observations, _ = runtime.env.reset()
    actions = runtime.deterministic_actions(wrapped_observations)
    assert actions.shape == (1, ACTION_DIM)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)


def test_trajectory_completion_is_terminated_not_truncated(adapter: ManoGymnasiumVectorEnv) -> None:
    physical = adapter.environment
    physical.reset()
    physical.progress[:] = 790
    physical.trajectory_steps[:] = 789
    _, _, terminated, truncated, info = adapter.step(np.zeros((1, 26), dtype=np.float64))
    np.testing.assert_array_equal(terminated, [True])
    np.testing.assert_array_equal(truncated, [False])
    np.testing.assert_array_equal(info["time_outs"], [False])
    np.testing.assert_array_equal(info["_final_observation"], [True])


def test_cpu_rollout_update_and_native_checkpoint_round_trip(adapter: ManoGymnasiumVectorEnv, tmp_path) -> None:
    import torch

    from sim.manorl.abi import ENVIRONMENT_CONTRACT_ID
    from sim.manorl.checkpoint import load_skrl_checkpoint, save_skrl_checkpoint
    from sim.manorl.rewards import PPO_REWARD_CONTRACT_ID, PPO_REWARD_SCALE, REWARD_CONTRACT_ID
    from sim.manorl.skrl_runtime import ManoPPOConfig, ManoSkrlRuntime

    runtime = ManoSkrlRuntime(adapter, ManoPPOConfig.optimizer_smoke())
    assert runtime.checkpoint_metadata()["reward_contract"] == REWARD_CONTRACT_ID
    assert runtime.checkpoint_metadata()["ppo_reward_contract"] == PPO_REWARD_CONTRACT_ID
    assert runtime.checkpoint_metadata()["ppo_reward_scale"] == PPO_REWARD_SCALE
    assert runtime.checkpoint_metadata()["environment_contract"] == ENVIRONMENT_CONTRACT_ID
    assert runtime.checkpoint_metadata()["environment"]["residual_action"]["position_scale"] == (0.005, 0.005, 0.005)
    assert runtime.checkpoint_metadata()["environment"]["residual_action"]["max_position_offset"] == (0.05, 0.05, 0.05)
    assert runtime.checkpoint_metadata()["environment"]["max_deviation_distance"] == 1_000_000.0
    assert "use_film" not in runtime.checkpoint_metadata()["ppo"]
    assert runtime.checkpoint_metadata()["model"]["use_film"] is True
    rollout = runtime.deterministic_rollout(steps=2)
    assert rollout["steps"] == 2
    assert torch.isfinite(rollout["observations"]).all()
    assert runtime.one_update_smoke()
    assert runtime.agent.tracking_data["Learning / Completed minibatches"][-1] == 1
    assert "policy_mean" in runtime.memory.tensors
    assert "policy_std" in runtime.memory.tensors

    checkpoint = save_skrl_checkpoint(
        runtime.agent, tmp_path / "manorl.pt", runtime_config=runtime.checkpoint_metadata()
    )
    original = runtime.model.actor_head.weight.detach().clone()
    with torch.no_grad():
        runtime.model.actor_head.weight.add_(1.0)
    load_skrl_checkpoint(runtime.agent, checkpoint)
    torch.testing.assert_close(runtime.model.actor_head.weight, original)


def test_checkpoint_rejects_rl_games_and_one_sample_update(adapter: ManoGymnasiumVectorEnv, tmp_path) -> None:
    import torch

    from sim.manorl.checkpoint import CheckpointFormatError, load_skrl_checkpoint
    from sim.manorl.skrl_runtime import ManoPPOConfig, ManoSkrlRuntime

    runtime = ManoSkrlRuntime(adapter, ManoPPOConfig.optimizer_smoke())
    source_checkpoint = tmp_path / "rl_games.pth"
    torch.save({"model": {}, "optimizer": {}}, source_checkpoint)
    with pytest.raises(CheckpointFormatError, match="rl-games"):
        load_skrl_checkpoint(runtime.agent, source_checkpoint)

    insufficient = ManoSkrlRuntime(adapter, ManoPPOConfig(rollouts=1, minibatch_size=1, learning_epochs=1))
    with pytest.raises(ValueError, match="at least two rollout samples"):
        insufficient.one_update_smoke()


def test_source_ppo_batch_divisibility_is_explicit() -> None:
    import torch

    from sim.manorl.rewards import PPO_REWARD_SCALE
    from sim.manorl.skrl_runtime import ManoPPOConfig

    config = ManoPPOConfig().skrl_config(num_envs=2048, device="cpu")
    assert config["rollouts"] == 48
    assert config["mini_batches"] == 24
    assert config["time_limit_bootstrap"] is True
    assert config["value_loss_scale"] == 2.0
    assert config["learning_starts"] == 0
    assert config["rewards_shaper"](torch.ones(1), 0, 1).item() == 0.5
    assert PPO_REWARD_SCALE == 0.5
    with pytest.raises(ValueError, match="must divide"):
        ManoPPOConfig().skrl_config(num_envs=1, device="cpu")
    with pytest.raises(ValueError, match="learning_starts must be non-negative"):
        ManoPPOConfig(learning_starts=-1)


def test_gym_aligned_ppo_completes_every_minibatch_without_kl_early_stop(
    adapter: ManoGymnasiumVectorEnv,
) -> None:
    from sim.manorl.rl_games_ppo import RlGamesPPO
    from sim.manorl.skrl_runtime import ManoPPOConfig, ManoSkrlRuntime

    config = ManoPPOConfig(rollouts=4, minibatch_size=2, learning_epochs=2)
    runtime = ManoSkrlRuntime(adapter, config)

    assert isinstance(runtime.agent, RlGamesPPO)
    assert runtime.one_update_smoke()
    assert runtime.agent.tracking_data["Learning / Completed minibatches"][-1] == 4
    assert len(runtime.agent.tracking_data["Learning / Exact KL mean"]) == 1
