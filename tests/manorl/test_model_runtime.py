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


def test_model_matches_source_pointnet_film_state_namespace(adapter: ManoGymnasiumVectorEnv) -> None:
    import torch

    from sim.manorl.gymnasium_env import ACTION_DIM, OBSERVATION_DIM

    model = _model(adapter)
    keys = set(model.state_dict())
    assert len(keys) == 41
    assert "actor_backbone.0.film.film_generator.weight" in keys
    assert "actor_backbone.1.weight" in keys
    assert "actor_backbone.3.weight" in keys
    assert "actor_backbone.5.weight" in keys
    assert all(".0.weight" not in key for key in keys if key.startswith("actor_backbone."))

    observations = torch.zeros((2, OBSERVATION_DIM), dtype=torch.float32)
    observations[:, 266] = 1.0
    policy, policy_outputs = model.compute({"observations": observations}, role="policy")
    value, value_outputs = model.compute({"observations": observations}, role="value")
    assert policy.shape == (2, ACTION_DIM)
    assert policy_outputs["log_std"].shape == (2, ACTION_DIM)
    assert value.shape == (2, 1)
    assert value_outputs == {}


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
    assert runtime.agent.cfg.rewards_shaper is None
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

    from sim.manorl.checkpoint import load_skrl_checkpoint, save_skrl_checkpoint
    from sim.manorl.rewards import PPO_REWARD_CONTRACT_ID, PPO_REWARD_SCALE, REWARD_CONTRACT_ID
    from sim.manorl.skrl_runtime import ManoPPOConfig, ManoSkrlRuntime

    runtime = ManoSkrlRuntime(adapter, ManoPPOConfig.optimizer_smoke())
    assert runtime.checkpoint_metadata()["reward_contract"] == REWARD_CONTRACT_ID
    assert runtime.checkpoint_metadata()["ppo_reward_contract"] == PPO_REWARD_CONTRACT_ID
    assert runtime.checkpoint_metadata()["ppo_reward_scale"] == PPO_REWARD_SCALE
    rollout = runtime.deterministic_rollout(steps=2)
    assert rollout["steps"] == 2
    assert torch.isfinite(rollout["observations"]).all()
    assert runtime.one_update_smoke()

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
    from sim.manorl.rewards import PPO_REWARD_SCALE
    from sim.manorl.skrl_runtime import ManoPPOConfig

    config = ManoPPOConfig().skrl_config(num_envs=64, device="cpu")
    assert config["rollouts"] == 48
    assert config["mini_batches"] == 3
    assert config["time_limit_bootstrap"] is True
    assert config["value_loss_scale"] == 4.0
    assert "rewards_shaper" not in config
    assert PPO_REWARD_SCALE == 1.0
    with pytest.raises(ValueError, match="must divide"):
        ManoPPOConfig().skrl_config(num_envs=1, device="cpu")
