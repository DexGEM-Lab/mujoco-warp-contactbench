"""v6 observation binding and token-fusion model checks."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sim.manorl.autonomy_contracts import (
    CURRENT_TOKENS_V6,
    GOAL_TOKENS_V6,
    RAW_OBSERVATION_DIM_V6,
    RAW_OBSERVATION_FIELDS_V6,
    raw_observation_slices_v6,
)
from sim.manorl.autonomy_v4 import build_raw_observation_v6
from tests.manorl.test_autonomy_v4 import _cache, _contact, _state


def _v6_cache(frames: int = 25):
    cache = _cache(frames)
    template = (
        np.arange(16 * 16 * 3, dtype=np.float64).reshape(256, 3)
        * 1.0e-4
    )
    reference = np.broadcast_to(
        template[None], (frames, 256, 3)
    ).copy()
    reference += np.arange(frames)[:, None, None] * 1.0e-5
    return replace(
        cache,
        hand_cloud_template=template,
        hand_cloud_reference=reference,
        object_mass=0.2,
        gravity_world=np.asarray([0.0, 0.0, -9.81]),
    )


def _v6_contact(batch: int):
    jax = pytest.importorskip("jax")
    j = jax.numpy
    return _contact(batch)._replace(
        object_all_torque=j.zeros((batch, 3))
    )


def test_v6_schema_has_one_authoritative_exact_width():
    assert sum(width for _, width in RAW_OBSERVATION_FIELDS_V6) == 2205
    assert RAW_OBSERVATION_DIM_V6 == 2205
    slices = raw_observation_slices_v6()
    assert (
        slices["hand_region_contact"].stop
        - slices["hand_region_contact"].start
        == 160
    )
    assert (
        slices["hand_point_cloud_reference_raw"].stop
        - slices["hand_point_cloud_reference_raw"].start
        == 768
    )
    assert slices["object_geometry"].stop == RAW_OBSERVATION_DIM_V6


def test_v6_builder_binds_force_and_slip_to_one_hand_region():
    jax = pytest.importorskip("jax")
    j = jax.numpy
    cache = _v6_cache()
    contact = _v6_contact(1)
    contact = contact._replace(
        paired_force_on_object=contact.paired_force_on_object.at[0, 5].set(
            j.asarray([1.0, 2.0, 3.0])
        ),
        paired_count=contact.paired_count.at[0, 5].set(2),
        tangential_slip=contact.tangential_slip.at[0, 5].set(
            j.asarray([0.01, 0.02, 0.03])
        ),
        object_all_force=j.asarray([[1.0, 2.0, 3.0]]),
    )
    raw = np.asarray(
        build_raw_observation_v6(
            _state(), contact, cache, j.asarray([0]), j.zeros((1, 28))
        )
    )
    assert raw.shape == (1, RAW_OBSERVATION_DIM_V6)
    assert np.isfinite(raw).all()
    region = raw[
        :, raw_observation_slices_v6()["hand_region_contact"]
    ].reshape(1, 16, 10)
    # Confidence/valid are reference intent and remain present for every region.
    dynamic = region[0, :, 2:]
    changed = np.flatnonzero(np.linalg.norm(dynamic, axis=-1) > 0)
    np.testing.assert_array_equal(changed, np.asarray([5]))


def test_v6_encoder_preserves_region_identity_for_contact_features():
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from sim.manorl.autonomy_v6_model import AutonomyActorCriticV6

    model = AutonomyActorCriticV6(
        gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(RAW_OBSERVATION_DIM_V6,),
            dtype=np.float32,
        ),
        gym.spaces.Box(-1.0, 1.0, shape=(28,), dtype=np.float32),
        device="cpu",
    )
    baseline = torch.zeros((1, RAW_OBSERVATION_DIM_V6))
    changed = baseline.clone()
    region_slice = raw_observation_slices_v6()["hand_region_contact"]
    changed[:, region_slice][:, 7 * 10 + 4] = 1.0
    current_base, _ = model.encoder(baseline)
    current_changed, _ = model.encoder(changed)
    token_delta = torch.linalg.vector_norm(
        current_changed - current_base, dim=-1
    )[0]
    affected = torch.nonzero(token_delta > 0, as_tuple=False).flatten()
    # Object tokens occupy [0,16); hand region 7 is current token 23.
    assert affected.tolist() == [23]


def test_v6_reference_hand_is_cross_attention_memory():
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from sim.manorl.autonomy_v6_model import AutonomyActorCriticV6

    torch.manual_seed(7)
    model = AutonomyActorCriticV6(
        gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(RAW_OBSERVATION_DIM_V6,),
            dtype=np.float32,
        ),
        gym.spaces.Box(-1.0, 1.0, shape=(28,), dtype=np.float32),
        device="cpu",
    )
    first = torch.zeros((2, RAW_OBSERVATION_DIM_V6))
    second = first.clone()
    second[
        :,
        raw_observation_slices_v6()[
            "hand_point_cloud_reference_raw"
        ],
    ] = 0.5
    with torch.no_grad():
        mean_first, _ = model.compute(
            {"observations": first}, role="policy"
        )
        mean_second, _ = model.compute(
            {"observations": second}, role="policy"
        )
        current, goal = model.encoder(second)
    assert current.shape == (2, CURRENT_TOKENS_V6, 128)
    assert goal.shape == (2, GOAL_TOKENS_V6, 128)
    assert not torch.allclose(mean_first, mean_second)


def test_v6_forward_backward_and_architecture_metadata_are_finite():
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from sim.manorl.autonomy_v6_model import AutonomyActorCriticV6

    model = AutonomyActorCriticV6(
        gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(RAW_OBSERVATION_DIM_V6,),
            dtype=np.float32,
        ),
        gym.spaces.Box(-1.0, 1.0, shape=(28,), dtype=np.float32),
        device="cpu",
    )
    observations = torch.randn((3, RAW_OBSERVATION_DIM_V6))
    mean, _ = model.compute({"observations": observations}, role="policy")
    value, _ = model.compute({"observations": observations}, role="value")
    (mean.square().mean() + value.square().mean()).backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert mean.shape == (3, 28)
    assert value.shape == (3, 1)
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    architecture = model.checkpoint_architecture()
    assert architecture["raw_observation_dim"] == RAW_OBSERVATION_DIM_V6
    assert architecture["actor_critic"] == (
        "shared observation token encoder; independent fusion towers"
    )
