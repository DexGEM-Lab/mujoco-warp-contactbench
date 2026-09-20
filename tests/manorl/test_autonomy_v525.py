"""ManoRL v5.25 region-hand encoder and integration contracts."""
from __future__ import annotations

import numpy as np
import pytest

from sim.manorl.autonomy_contracts import (
    ACTION_CONTRACT_ID,
    CHECKPOINT_FORMAT_V525,
    OBSERVATION_CONTRACT_ID_V525,
    RAW_OBSERVATION_DIM_V525,
    REWARD_CONTRACT_ID,
)


def _model(*, separate_critic: bool = True):
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from sim.manorl.autonomy_v5_intermediate_model import (
        AutonomyActorCriticV525,
    )

    return AutonomyActorCriticV525(
        gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(RAW_OBSERVATION_DIM_V525,),
            dtype=np.float32,
        ),
        gym.spaces.Box(-1.0, 1.0, shape=(28,), dtype=np.float32),
        device="cpu",
        separate_critic=separate_critic,
    )


def test_v525_forward_contract_and_parameter_count():
    torch = pytest.importorskip("torch")
    model = _model()
    observations = torch.randn((3, RAW_OBSERVATION_DIM_V525))
    mean, _ = model.compute({"observations": observations}, role="policy")
    value, _ = model.compute({"observations": observations}, role="value")
    (mean.square().mean() + value.square().mean()).backward()
    assert mean.shape == (3, 28)
    assert value.shape == (3, 1)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert model.checkpoint_contracts() == {
        "checkpoint_format": CHECKPOINT_FORMAT_V525,
        "observation_contract": OBSERVATION_CONTRACT_ID_V525,
        "reward_contract": REWARD_CONTRACT_ID,
        "action_contract": ACTION_CONTRACT_ID,
    }


def test_v525_region_identity_changes_the_pooled_hand_embedding():
    torch = pytest.importorskip("torch")
    model = _model()
    baseline = torch.zeros((1, 256, 3))
    first_region = baseline.clone()
    second_region = baseline.clone()
    first_region[:, 0:16] = 1.0
    second_region[:, 16:32] = 1.0
    with torch.no_grad():
        first = model.hand_encoder(first_region)
        second = model.hand_encoder(second_region)
    assert not torch.allclose(first, second)


def test_v525_cli_and_checkpoint_detection():
    pytest.importorskip("torch")
    from tools import train_manorl_autonomy as cli

    args = cli.parse_args(
        ["train", "--policy-version", "v5.25", "--no-wandb"]
    )
    assert args.policy_version == "v5.25"
    payload = _model().checkpoint_contracts()
    assert cli._checkpoint_policy_version(payload) == "v5.25"
    assert cli._checkpoint_separate_critic(
        {
            **payload,
            "model_architecture": _model().checkpoint_architecture(),
        }
    )
