from __future__ import annotations

import pytest
import torch

from sim.manorl.autonomy_v4_telemetry import REWARD_NAMES, V4TelemetryAccumulator


def _sample(*, rewards=(1., 2.), reason=(0, 0), clearance=(0., 0.), loaded=((False,) * 16, (False,) * 16), airborne=(False, False), clipped=(0., 0.)):
    b = 2
    terms = torch.tensor([[value] * len(REWARD_NAMES) for value in rewards])
    return {
        "reward_terms": terms, "reward_total": terms.sum(1), "reason": torch.tensor(reason), "valid": torch.ones(b, dtype=torch.bool),
        "position_error_abs": torch.tensor([[1., 2., 3.], [3., 2., 1.]]),
        "object_rotation_error_rad": torch.ones(b), "palm_position_error": torch.ones(b),
        "finger_raw_error": torch.ones(b), "finger_feasible_error": torch.full((b,), 2.),
        "bottom_clearance": torch.tensor(clearance), "reference_bottom_clearance": torch.zeros(b),
        "origin_lift_delta": torch.zeros(b), "paired_loaded": torch.tensor(loaded),
        "paired_active": torch.tensor(loaded), "paired_contact_count": torch.tensor([0., 1.]),
        "paired_force_norm": torch.tensor([0., .04]), "object_all_force_norm": torch.tensor([9., 9.]),
        "paired_torque_com_norm": torch.tensor([0., .1]), "tangential_slip": torch.full((b, 16), .25),
        "airborne": torch.tensor(airborne), "action_raw_abs_sum": torch.tensor([14., 7.]),
        "action_raw_abs_max": torch.tensor([.5, .25]), "action_raw_abs_denominator": torch.full((b,), 28.),
        "action_executed_norm": torch.ones(b), "action_clipped": torch.tensor(clipped) * 28,
        "action_denominator": torch.full((b,), 28.),
        "command_envelope_utilization": torch.tensor([.5, .25]), "antiwindup_active": torch.tensor([False, True]),
        "reference_progress": torch.tensor([.2, .8]),
    }


def test_episode_returns_cross_update_cut_and_emit_only_completed() -> None:
    acc = V4TelemetryAccumulator(2, torch.device("cpu"))
    acc.add(_sample(rewards=(1., 2.)))
    row = acc.reduce(update=1, transitions=2)
    assert row["episodes/completed_count"] == 0
    assert "episodes/return_mean" not in row
    # Env 0 completes after two steps (total=18); env 1 did not complete.
    acc.add(_sample(rewards=(1., 2.), reason=(1, 0)))
    row = acc.reduce(update=2, transitions=4)
    assert row["episodes/completed_count"] == 1
    assert row["episodes/return_denominator"] == 1
    assert row["episodes/return_mean"] == 18
    assert row["reward/episode_severe_return_mean"] == 2


def test_reason_bits_overlap_and_horizon_only_is_distinct() -> None:
    acc = V4TelemetryAccumulator(2, torch.device("cpu"))
    acc.add(_sample(reason=(3, 1)))
    row = acc.reduce(update=1, transitions=2)
    assert row["termination/reference_complete_count"] == 2
    assert row["termination/deviation_count"] == 1
    assert row["termination/horizon_only_count"] == 1
    assert row["termination/reference_complete_rate"] == 1
    assert row["termination/deviation_rate"] == .5


def test_physical_and_action_denominators_keep_table_force_separate() -> None:
    loaded = ((False,) * 16, (True,) + (False,) * 15)
    acc = V4TelemetryAccumulator(2, torch.device("cpu"))
    acc.add(_sample(loaded=loaded, clearance=(.0, .01), airborne=(False, True), clipped=(0., 1.)))
    row = acc.reduce(update=1, transitions=2)
    assert row["physics/object_all_force_norm_mean"] == 9
    assert row["physics/paired_force_norm_mean"] == pytest.approx(.02)
    assert row["physics/paired_loaded_contact_fraction"] == pytest.approx(1 / 32)
    assert row["physics/contact_active_tangential_slip_mean"] == pytest.approx(.25)
    assert row["physics/airborne_5mm_fraction"] == pytest.approx(.5)
    assert row["physics/loaded_airborne_fraction"] == pytest.approx(.5)
    assert row["action/raw_abs_mean"] == pytest.approx(.375)
    assert row["action/raw_abs_max"] == pytest.approx(.5)
    assert row["action/raw_clip_fraction"] == pytest.approx(.5)


def test_region_slip_is_scalar_per_region_for_nonuniform_b2_b3_and_rejects_malformed_shape() -> None:
    acc = V4TelemetryAccumulator(2, torch.device("cpu"))
    sample = _sample(loaded=((True,) * 16, (True,) * 16))
    sample["tangential_slip"] = torch.stack((torch.arange(16), torch.arange(16) + 16.)).float()
    acc.add(sample)
    row = acc.reduce(update=1, transitions=2)
    assert row["physics/contact_active_tangential_slip_mean"] == pytest.approx(15.5)
    acc3 = V4TelemetryAccumulator(3, torch.device("cpu")); sample3 = _sample()
    for name, value in tuple(sample3.items()):
        if isinstance(value, torch.Tensor) and value.shape[0] == 2:
            sample3[name] = torch.cat((value, value[:1]), dim=0)
    sample3["paired_active"] = sample3["paired_loaded"] = torch.ones((3, 16), dtype=torch.bool)
    sample3["tangential_slip"] = torch.arange(48, dtype=torch.float32).reshape(3, 16)
    acc3.add(sample3)
    assert acc3.reduce(update=1, transitions=3)["physics/contact_active_tangential_slip_mean"] == pytest.approx(23.5)
    malformed = _sample(); malformed["tangential_slip"] = torch.zeros((2, 16, 3))
    with pytest.raises(ValueError, match="per-region scalar"):
        V4TelemetryAccumulator(2, torch.device("cpu")).add(malformed)


def test_action_element_max_and_invalid_reconciliation_are_exact() -> None:
    acc = V4TelemetryAccumulator(2, torch.device("cpu")); sample = _sample()
    sample["action_raw_abs_sum"] = torch.tensor([28., 56.]); sample["action_raw_abs_max"] = torch.tensor([1., 7.])
    sample["reward_terms"][0, 0] = float("nan"); sample["valid"] = torch.tensor([False, True])
    acc.add(sample); row = acc.reduce(update=1, transitions=2)
    assert row["action/raw_abs_mean"] == pytest.approx(1.5)
    assert row["action/raw_abs_max"] == 7.
    assert row["reward/component_sum_denominator"] == 1.
    assert row["reward/component_sum_minus_total_abs_mean"] == 0.


def test_dynamic_reward_names_preserve_component_accounting() -> None:
    names = ("tracking", "contact", "severe")
    acc = V4TelemetryAccumulator(
        2, torch.device("cpu"), reward_names=names,
    )
    sample = _sample()
    sample["reward_terms"] = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    sample["reward_total"] = sample["reward_terms"].sum(1)
    acc.add(sample)
    row = acc.reduce(update=1, transitions=2)
    assert row["reward/contact"] == pytest.approx(3.5)
    assert row["reward/component_sum_minus_total_abs_mean"] == 0.0
