"""ManoRL v5.5 and v5.75 observation/model integration contracts."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sim.manorl.autonomy_contracts import (
    CHECKPOINT_FORMAT_V55,
    CHECKPOINT_FORMAT_V575,
    OBSERVATION_CONTRACT_ID_V55,
    OBSERVATION_CONTRACT_ID_V575,
    RAW_OBSERVATION_DIM_V55,
    RAW_OBSERVATION_DIM_V575,
    RAW_OBSERVATION_FIELDS_V55,
    RAW_OBSERVATION_FIELDS_V575,
    raw_observation_slices_v575,
)
from sim.manorl.autonomy_v4 import (
    build_raw_observation_v55,
    build_raw_observation_v575,
)
from tests.manorl.test_autonomy_v4 import _cache, _contact, _state


def _cache_with_goals(frames: int = 25):
    cache = _cache(frames)
    template = np.zeros((256, 3), dtype=np.float64)
    cloud = np.broadcast_to(
        template[None], (frames, 256, 3)
    ).copy()
    pose = np.zeros((frames, 16, 9), dtype=np.float64)
    pose[..., 3] = 1.0
    pose[..., 7] = 1.0
    return replace(
        cache,
        hand_cloud_template=template,
        hand_cloud_reference=cloud,
        hand_region_pose_reference=pose,
        object_mass=0.2,
        gravity_world=np.asarray([0.0, 0.0, -9.81]),
    )


def _contact_with_torque(batch: int):
    jax = pytest.importorskip("jax")
    return _contact(batch)._replace(
        object_all_torque=jax.numpy.zeros((batch, 3))
    )


def test_intermediate_schema_widths_are_exact():
    assert sum(width for _, width in RAW_OBSERVATION_FIELDS_V55) == 1486
    assert sum(width for _, width in RAW_OBSERVATION_FIELDS_V575) == 1581
    assert RAW_OBSERVATION_DIM_V55 == 1486
    assert RAW_OBSERVATION_DIM_V575 == 1581


def test_v55_and_v575_builders_emit_finite_exact_observations():
    jax = pytest.importorskip("jax")
    j = jax.numpy
    cache = _cache_with_goals()
    contact = _contact_with_torque(2)
    state = _state(2)
    index = j.zeros((2,), dtype=j.int32)
    command = j.zeros((2, 28))
    v55 = np.asarray(
        build_raw_observation_v55(
            state, contact, cache, index, command
        )
    )
    v575 = np.asarray(
        build_raw_observation_v575(
            state, contact, cache, index, command
        )
    )
    assert v55.shape == (2, RAW_OBSERVATION_DIM_V55)
    assert v575.shape == (2, RAW_OBSERVATION_DIM_V575)
    assert np.isfinite(v55).all()
    assert np.isfinite(v575).all()


def test_v575_dynamic_contact_stays_bound_to_one_region():
    jax = pytest.importorskip("jax")
    j = jax.numpy
    contact = _contact_with_torque(1)
    contact = contact._replace(
        paired_count=contact.paired_count.at[0, 3].set(2),
        tangential_slip=contact.tangential_slip.at[0, 3].set(
            j.asarray([0.01, 0.02, 0.03])
        ),
    )
    raw = np.asarray(
        build_raw_observation_v575(
            _state(),
            contact,
            _cache_with_goals(),
            j.asarray([0]),
            j.zeros((1, 28)),
        )
    )
    dynamic = raw[
        :, raw_observation_slices_v575()[
            "hand_region_dynamic_contact"
        ]
    ].reshape(1, 16, 5)
    changed = np.flatnonzero(
        np.linalg.norm(dynamic[0], axis=-1) > 0
    )
    np.testing.assert_array_equal(changed, np.asarray([3]))


@pytest.mark.parametrize(
    ("version", "dimension", "checkpoint_format", "observation_contract"),
    [
        (
            "v5.5",
            RAW_OBSERVATION_DIM_V55,
            CHECKPOINT_FORMAT_V55,
            OBSERVATION_CONTRACT_ID_V55,
        ),
        (
            "v5.75",
            RAW_OBSERVATION_DIM_V575,
            CHECKPOINT_FORMAT_V575,
            OBSERVATION_CONTRACT_ID_V575,
        ),
    ],
)
def test_intermediate_model_forward_and_cli_detection(
    version,
    dimension,
    checkpoint_format,
    observation_contract,
):
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from sim.manorl.autonomy_training import model_for_version
    from tools import train_manorl_autonomy as cli

    observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(dimension,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        -1.0, 1.0, shape=(28,), dtype=np.float32
    )
    model = model_for_version(
        version,
        observation_space,
        action_space,
        device="cpu",
        separate_critic=True,
    )
    observations = torch.randn((2, dimension))
    mean, _ = model.compute(
        {"observations": observations}, role="policy"
    )
    value, _ = model.compute(
        {"observations": observations}, role="value"
    )
    (mean.square().mean() + value.square().mean()).backward()
    assert mean.shape == (2, 28)
    assert value.shape == (2, 1)
    contracts = model.checkpoint_contracts()
    assert contracts["checkpoint_format"] == checkpoint_format
    assert contracts["observation_contract"] == observation_contract
    assert cli._checkpoint_policy_version(contracts) == version
    assert cli.parse_args(
        ["train", "--policy-version", version, "--no-wandb"]
    ).policy_version == version
