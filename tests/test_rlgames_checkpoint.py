from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from sim.dexhandrl.rlgames_checkpoint import RlGamesRunningMeanStd, load_rlgames_checkpoint
from sim.dexhandrl.skrl_models import build_skrl_models


def _source_key(target_key: str, *, role: str) -> str | None:
    if target_key == "log_std":
        return "a2c_network.log_std"
    if target_key.startswith("mu."):
        return "a2c_network." + target_key
    if target_key.startswith("value."):
        return "a2c_network." + target_key
    if target_key.startswith("backbone.pointnet_encoder."):
        return target_key.replace("backbone.", "a2c_network.", 1)
    if role == "policy" and target_key.startswith("backbone.actor_trunk."):
        return target_key.replace("backbone.actor_trunk.", "a2c_network.actor_mlp.", 1)
    if role == "value" and target_key.startswith("backbone.critic_trunk."):
        return target_key.replace("backbone.critic_trunk.", "a2c_network.critic_mlp.", 1)
    if role == "policy" and target_key.startswith("backbone.actor_film_generators.0."):
        return target_key.replace(
            "backbone.actor_film_generators.0.",
            "a2c_network.film_generators.0.",
            1,
        )
    if role == "policy" and target_key.startswith("backbone.actor_film_generators.2."):
        return target_key.replace(
            "backbone.actor_film_generators.2.",
            "a2c_network.film_generators.1.",
            1,
        )
    return None


def _make_agent_and_source(observation_dim: int = 461):
    observation_space = Box(-np.inf, np.inf, (461,), dtype=np.float32)
    action_space = Box(-1.0, 1.0, (22,), dtype=np.float32)
    models = build_skrl_models(observation_space, observation_space, action_space, "cpu")
    source: dict[str, torch.Tensor] = {}
    for role in ("policy", "value"):
        for target_key, target_value in models[role].state_dict().items():
            source_key = _source_key(target_key, role=role)
            if source_key is not None:
                source[source_key] = torch.full_like(target_value, 0.25)
    source.update(
        {
            "running_mean_std.running_mean": torch.arange(observation_dim, dtype=torch.float64),
            "running_mean_std.running_var": torch.ones(observation_dim, dtype=torch.float64) * 4.0,
            "running_mean_std.count": torch.tensor(10.0, dtype=torch.float64),
            "value_mean_std.running_mean": torch.tensor([2.0], dtype=torch.float64),
            "value_mean_std.running_var": torch.tensor([9.0], dtype=torch.float64),
            "value_mean_std.count": torch.tensor(10.0, dtype=torch.float64),
        }
    )
    agent = SimpleNamespace(
        policy=models["policy"],
        value=models["value"],
        _observation_preprocessor=RlGamesRunningMeanStd(461),
        _state_preprocessor=RlGamesRunningMeanStd(461),
        _value_preprocessor=RlGamesRunningMeanStd(1),
    )
    return agent, source


def test_load_rlgames_checkpoint_maps_weights_and_statistics(tmp_path) -> None:
    agent, source = _make_agent_and_source()
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"model": source, "epoch": 7, "frame": 11, "last_mean_rewards": 3.5}, checkpoint_path)

    report = load_rlgames_checkpoint(agent, checkpoint_path)

    assert report.epoch == 7
    assert report.observation_dim == 461
    assert torch.all(agent.policy.mu.weight == 0.25)
    assert torch.all(agent.value.value.weight == 0.25)
    assert torch.equal(
        agent._observation_preprocessor.running_mean,
        source["running_mean_std.running_mean"],
    )


def test_load_rlgames_checkpoint_rejects_old_observation_layout(tmp_path) -> None:
    agent, source = _make_agent_and_source(observation_dim=429)
    checkpoint_path = tmp_path / "old_checkpoint.pth"
    torch.save({"model": source}, checkpoint_path)

    with pytest.raises(RuntimeError, match="observation dimension is 429"):
        load_rlgames_checkpoint(agent, checkpoint_path)


def test_rlgames_normalizer_matches_checkpoint_formula() -> None:
    normalizer = RlGamesRunningMeanStd(2)
    normalizer.running_mean.copy_(torch.tensor([1.0, -1.0], dtype=torch.float64))
    normalizer.running_var.copy_(torch.tensor([4.0, 9.0], dtype=torch.float64))
    values = torch.tensor([[3.0, 2.0]], dtype=torch.float32)

    normalized = normalizer(values)
    expected = (values - torch.tensor([1.0, -1.0])) / torch.sqrt(
        torch.tensor([4.0, 9.0]) + 1e-5
    )

    assert torch.allclose(normalized, expected)


def test_rlgames_normalizer_flattens_rollout_and_environment_batch_dims() -> None:
    memory_shaped = RlGamesRunningMeanStd(1)
    flat = RlGamesRunningMeanStd(1)
    values = torch.tensor([[[1.0]], [[3.0]]], dtype=torch.float32)

    memory_shaped(values, train=True)
    flat(values.reshape(-1, 1), train=True)

    assert memory_shaped.running_mean.shape == (1,)
    assert torch.allclose(memory_shaped.running_mean, flat.running_mean)
    assert torch.allclose(memory_shaped.running_var, flat.running_var)
    assert torch.allclose(memory_shaped.count, flat.count)


def test_rlgames_normalizer_single_sample_update_stays_finite() -> None:
    normalizer = RlGamesRunningMeanStd(2)

    normalized = normalizer(
        torch.tensor([[1.0, -1.0]], dtype=torch.float32),
        train=True,
    )

    assert torch.isfinite(normalized).all()
    assert torch.isfinite(normalizer.running_mean).all()
    assert torch.isfinite(normalizer.running_var).all()
