"""v4 actor and contract import regressions."""
from __future__ import annotations

import importlib

import gymnasium as gym
import pytest
import torch

from sim.manorl.autonomy_contracts import ACTION_DIM, RAW_OBSERVATION_DIM
from sim.manorl.autonomy_training import AutonomyActorCritic


def _model() -> AutonomyActorCritic:
    return AutonomyActorCritic(
        gym.spaces.Box(-1.0, 1.0, shape=(RAW_OBSERVATION_DIM,)),
        gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,)),
        device="cpu",
    )


def test_v4_actor_imports_and_produces_all_policy_actions():
    model = _model()
    observations = {"observations": torch.zeros((2, RAW_OBSERVATION_DIM))}
    policy, _ = model.compute(observations, role="policy")
    value, _ = model.compute(observations, role="value")
    assert policy.shape == (2, ACTION_DIM)
    assert value.shape == (2, 1)
    assert model.checkpoint_architecture()["raw_observation_dim"] == RAW_OBSERVATION_DIM


def test_obsolete_v3_contract_names_and_imitation_module_fail_closed():
    contracts = importlib.import_module("sim.manorl.autonomy_contracts")
    for name in (
        "AUTONOMY_V3_VERSION",
        "OBSERVATION_V3_CONTRACT_ID",
        "REWARD_V3_CONTRACT_ID",
        "CHECKPOINT_V3_FORMAT",
        "ACTION_V3_CONTRACT_ID",
        "validate_v3_checkpoint_metadata",
    ):
        assert not hasattr(contracts, name)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sim.manorl.autonomy_imitation")


def test_v4_training_and_ppo_imports_remain_available():
    assert importlib.import_module("sim.manorl.autonomy_training").TRAINING_CONTRACT_ID.endswith("v4.disabled")
    assert callable(importlib.import_module("sim.manorl.autonomy_batch_training").run_batched_ppo)
