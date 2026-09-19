"""v5 point-cloud observation and model ABI checks."""
from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from sim.manorl.autonomy_contracts import (
    ACTION_CONTRACT_ID,
    CHECKPOINT_FORMAT_V5,
    RAW_OBSERVATION_DIM_V5, ENCODED_OBSERVATION_DIM_V5,
    OBSERVATION_CONTRACT_ID_V5,
    REWARD_CONTRACT_ID,
    raw_observation_slices_v5, encoded_observation_slices_v5,
)
from sim.manorl.autonomy_v4 import (
    ReferenceBankV4, build_raw_observation_v5, hand_cloud_actual,
)
from tests.manorl.test_autonomy_v4 import _cache, _state, _contact


def _v5_cache(T=25):
    cache = _cache(T)
    template = np.tile(np.arange(48, dtype=np.float64).reshape(16, 3) * 0.001, (16, 1, 1)).reshape(256, 3)
    reference = np.broadcast_to(template[None], (T, 256, 3)).copy()
    return replace(cache, hand_cloud_template=template, hand_cloud_reference=reference)


def test_v5_slice_tables_cover_exact_widths():
    assert sum(w for _, w in __import__('sim.manorl.autonomy_contracts', fromlist=['x']).RAW_OBSERVATION_FIELDS_V5) == RAW_OBSERVATION_DIM_V5
    assert sum(w for _, w in __import__('sim.manorl.autonomy_contracts', fromlist=['x']).ENCODED_OBSERVATION_FIELDS_V5) == ENCODED_OBSERVATION_DIM_V5
    rs = raw_observation_slices_v5()
    assert rs["autonomous_actual"].stop - rs["autonomous_actual"].start == 119
    assert rs["contact_intent"].stop - rs["contact_intent"].start == 80
    assert rs["hand_point_cloud_raw"].stop - rs["hand_point_cloud_raw"].start == 768
    assert rs["object_geometry"].stop == RAW_OBSERVATION_DIM_V5


def test_v5_hand_cloud_actual_shape_and_frame():
    jax = pytest.importorskip("jax")
    state = _state(2)
    template = np.zeros((16, 16, 3))
    cloud = np.asarray(hand_cloud_actual(state, template))
    assert cloud.shape == (2, 256 * 3)
    assert np.isfinite(cloud).all()


def test_v5_builder_emits_exact_raw_width_and_blocks():
    jax = pytest.importorskip("jax")
    j = jax.numpy
    cache = _v5_cache()
    raw = np.asarray(build_raw_observation_v5(_state(2), _contact(2), cache, j.asarray([0, 0]), j.zeros((2, 28))))
    assert raw.shape == (2, RAW_OBSERVATION_DIM_V5)
    assert np.isfinite(raw).all()
    rs = raw_observation_slices_v5()
    # Contact intent carries confidence/valid from the synthetic cache plus force.
    assert np.allclose(raw[:, rs["contact_intent"]][:, :32], 1.0)


def test_v5_model_forward_shapes():
    torch = pytest.importorskip("torch")
    from sim.manorl.autonomy_training import AutonomyActorCriticV5, actor_critic_architecture_v5
    model = AutonomyActorCriticV5(
        __import__('gymnasium').spaces.Box(-np.inf, np.inf, shape=(RAW_OBSERVATION_DIM_V5,), dtype=np.float32),
        __import__('gymnasium').spaces.Box(-1.0, 1.0, shape=(28,), dtype=np.float32),
        device="cpu", separate_critic=True,
    )
    obs = torch.zeros((4, RAW_OBSERVATION_DIM_V5))
    mean, info = model.compute({"observations": obs}, role="policy")
    value, _ = model.compute({"observations": obs}, role="value")
    assert mean.shape == (4, 28) and value.shape == (4, 1)
    arch = model.checkpoint_architecture()
    assert arch["id"] == "manorl.autonomy.actor_critic.v5.pointcloud"
    assert arch["raw_observation_dim"] == RAW_OBSERVATION_DIM_V5
    assert arch["encoded_feature_dim"] == ENCODED_OBSERVATION_DIM_V5
    assert model.checkpoint_contracts() == {
        "checkpoint_format": CHECKPOINT_FORMAT_V5,
        "observation_contract": OBSERVATION_CONTRACT_ID_V5,
        "reward_contract": REWARD_CONTRACT_ID,
        "action_contract": ACTION_CONTRACT_ID,
    }


def test_v5_cli_and_checkpoint_version_detection():
    pytest.importorskip("torch")
    from tools import train_manorl_autonomy as cli

    args = cli.parse_args(["train", "--policy-version", "v5", "--no-wandb"])
    assert args.policy_version == "v5"
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT_V5,
        "observation_contract": OBSERVATION_CONTRACT_ID_V5,
        "reward_contract": REWARD_CONTRACT_ID,
        "action_contract": ACTION_CONTRACT_ID,
    }
    assert cli._checkpoint_policy_version(payload) == "v5"


def test_v5_bank_carries_hand_cloud_and_template():
    jax = pytest.importorskip("jax")
    a = _v5_cache(25)
    b = _v5_cache(13)
    bank = ReferenceBankV4([a, b])
    assert bank.hand_cloud_reference.shape == (2, 25, 256, 3)
    assert np.asarray(bank.hand_cloud_template).shape == (256, 3)
