from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from sim.manorl.checkpoint import CheckpointFormatError, checkpoint_runtime_metadata
from sim.manorl.environment import _torch_global_surface_templates
from sim.manorl.gym_checkpoint import (
    SOURCE_PREFIX,
    GymCheckpointFormatError,
    convert_gym_checkpoint,
)
from sim.manorl.model import ManoActorCritic
from sim.manorl.normalization import PointCloudAwareRunningStandardScaler, SourceRunningStandardScaler
from sim.manorl.observations import OBSERVATION_SLICES


def _target_model() -> ManoActorCritic:
    return ManoActorCritic(
        gym.spaces.Box(-5.0, 5.0, shape=(476,), dtype=float),
        None,
        gym.spaces.Box(-1.0, 1.0, shape=(26,), dtype=float),
        device="cpu",
    )


def _source_payload() -> dict[str, object]:
    model = _target_model()
    state = model.state_dict()
    source_model: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        source_name = (
            SOURCE_PREFIX + "condition_encoder.encoder.0." + name.rsplit(".", 1)[1]
            if name.startswith("condition_encoder.0.")
            else SOURCE_PREFIX + name
        )
        source_model[source_name] = value.detach().clone()
    running_mean = torch.linspace(-0.5, 0.5, 476, dtype=torch.float64)
    running_var = torch.linspace(0.5, 1.5, 476, dtype=torch.float64)
    pc_mean = torch.tensor([-0.2, 0.1, 0.3], dtype=torch.float64)
    pc_var = torch.tensor([0.7, 0.9, 1.1], dtype=torch.float64)
    pc_slice = OBSERVATION_SLICES["object_point_cloud_raw"]
    running_mean[pc_slice] = pc_mean.repeat(64)
    running_var[pc_slice] = pc_var.repeat(64)
    source_model.update(
        {
            "running_mean_std.running_mean": running_mean,
            "running_mean_std.running_var": running_var,
            "running_mean_std.count": torch.tensor(123.0, dtype=torch.float64),
            "running_mean_std.pc_running_mean": pc_mean,
            "running_mean_std.pc_running_var": pc_var,
            "running_mean_std.pc_count": torch.tensor(7873.0, dtype=torch.float64),
            "value_mean_std.running_mean": torch.tensor([1.25], dtype=torch.float64),
            "value_mean_std.running_var": torch.tensor([2.5], dtype=torch.float64),
            "value_mean_std.count": torch.tensor(123.0, dtype=torch.float64),
        }
    )
    return {"model": source_model, "epoch": 3, "frame": 48}


def _source_tensor(source_model: dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
    return source_model[SOURCE_PREFIX + suffix]


def _source_linear(
    source_model: dict[str, torch.Tensor], suffix: str, values: torch.Tensor
) -> torch.Tensor:
    return F.linear(
        values,
        _source_tensor(source_model, suffix + ".weight"),
        _source_tensor(source_model, suffix + ".bias"),
    )


def _source_layer_norm(
    source_model: dict[str, torch.Tensor], suffix: str, values: torch.Tensor
) -> torch.Tensor:
    return F.layer_norm(
        values,
        (values.shape[-1],),
        _source_tensor(source_model, suffix + ".weight"),
        _source_tensor(source_model, suffix + ".bias"),
    )


def _source_policy_value(
    source_model: dict[str, torch.Tensor], observations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent rl-games-key forward, without constructing the target model."""

    point_slice = OBSERVATION_SLICES["object_point_cloud_raw"]
    points = observations[:, point_slice].reshape(-1, 64, 3)
    point_features = points.reshape(-1, 3)
    for linear_index, norm_index in ((0, 1), (3, 4), (6, 7)):
        point_features = _source_linear(
            source_model, f"pointnet.point_mlp.{linear_index}", point_features
        )
        point_features = F.relu(
            _source_layer_norm(
                source_model, f"pointnet.point_mlp.{norm_index}", point_features
            )
        )
    point_features = point_features.reshape(len(observations), 64, 256).max(dim=1).values
    point_features = _source_linear(source_model, "pointnet.global_mlp.0", point_features)
    point_features = F.relu(
        _source_layer_norm(source_model, "pointnet.global_mlp.1", point_features)
    )

    def section(name: str) -> torch.Tensor:
        return observations[:, OBSERVATION_SLICES[name]]

    hand_features = torch.cat(
        [
            section("hand_keypoints"),
            section("contact_forces"),
            section("contact_force_directions"),
            section("expected_contact_mask"),
        ],
        dim=-1,
    )
    base = torch.cat(
        [
            observations[:, : point_slice.start],
            point_features,
            hand_features,
            section("cumulative_joint_offset"),
        ],
        dim=-1,
    )
    condition_input = torch.cat([section("action_types"), section("object_geometry")], dim=-1)
    condition = F.relu(
        F.linear(
            condition_input,
            _source_tensor(source_model, "condition_encoder.encoder.0.weight"),
            _source_tensor(source_model, "condition_encoder.encoder.0.bias"),
        )
    )

    actor = _source_linear(source_model, "actor_backbone.0.linear", base)
    film = _source_linear(source_model, "actor_backbone.0.film.film_generator", condition)
    delta_gamma, beta = film.chunk(2, dim=-1)
    actor = F.elu((1.0 + delta_gamma) * actor + beta)
    for index in (1, 3, 5):
        actor = F.elu(_source_linear(source_model, f"actor_backbone.{index}", actor))
    policy = _source_linear(source_model, "actor_head", actor)

    value = torch.cat([base, condition], dim=-1)
    for index in (0, 2, 4, 6):
        value = F.elu(_source_linear(source_model, f"critic.{index}", value))
    return policy, _source_linear(source_model, "critic.8", value)


def test_realistic_source_conversion_is_strict_and_provenanced(tmp_path) -> None:
    source = tmp_path / "source.pth"
    output = tmp_path / "converted.pt"
    torch.save(_source_payload(), source)
    convert_gym_checkpoint(source, output)
    modules = torch.load(output, map_location="cpu", weights_only=False)
    assert set(modules) == {"policy", "value", "optimizer", "observation_preprocessor", "value_preprocessor"}
    assert len(modules["policy"]) == 41
    assert modules["policy"].keys() == modules["value"].keys()
    assert modules["observation_preprocessor"]["running_mean"].dtype == torch.float64
    assert modules["value_preprocessor"]["running_variance"].dtype == torch.float64
    for name in modules["policy"]:
        torch.testing.assert_close(modules["policy"][name], modules["value"][name])
    sidecar = json.loads(output.with_suffix(".pt.json").read_text())
    assert checkpoint_runtime_metadata(output) == sidecar
    assert sidecar["runtime_config"]["model"]["use_film"] is True
    assert sidecar["runtime_config"]["point_cloud"]["sampling_backend"] == "torch_cuda_global"
    assert sidecar["runtime_config"]["compatibility"] == {
        "name": "gym_eval_aligned",
        "early_phase_steps": 50,
        "movement_pre_padding": 250,
        "point_template_mode": "dynamic_reset",
    }
    assert sidecar["runtime_config"]["residual_action"]["max_position_offset"] == [0.05, 0.05, 0.05]
    assert sidecar["runtime_config"]["residual_action"]["joint_scale"][:4] == [0.1, 0.12, 0.044, 0.01]
    assert sidecar["runtime_config"]["residual_action"]["max_joint_offset"][:4] == [1.0, 1.2, 0.44, 0.1]
    reward = sidecar["runtime_config"]["reward"]
    assert reward["max_contact_reward"] == 0.4
    assert reward["contact_force_threshold"] == 2.0
    assert reward["direct_contact_reward_scale"] == 1.0
    assert reward["force_basis"] == "mujoco_pair_filtered_hand_object_force"
    assert reward["ppo_reward_scale"] == 0.5
    assert sidecar["conversion"]["migration_profile"] == "isaacgym_mano_v1"
    assert sidecar["conversion"]["dtype_conversion"] == {
        "model": {"source": ["float32"], "target": "float32"},
        "observation_normalizer": {"source": ["float64"], "target": "float64"},
        "value_normalizer": {"source": ["float64"], "target": "float64"},
    }
    with pytest.raises(FileExistsError):
        convert_gym_checkpoint(source, output)


def test_conversion_rejects_missing_source_with_format_error(tmp_path) -> None:
    with pytest.raises(GymCheckpointFormatError, match="does not exist"):
        convert_gym_checkpoint(tmp_path / "missing.pth", tmp_path / "converted.pt")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_model_key",
        "extra_model_key",
        "wrong_model_shape",
        "integer_model_dtype",
        "nonfinite_model",
        "integer_normalizer_dtype",
    ],
)
def test_conversion_rejects_incompatible_source_tensors(tmp_path, mutation: str) -> None:
    payload = _source_payload()
    source_model = payload["model"]
    assert isinstance(source_model, dict)
    model_key = next(key for key in source_model if key.startswith(SOURCE_PREFIX))
    if mutation == "missing_model_key":
        source_model.pop(model_key)
    elif mutation == "extra_model_key":
        source_model[SOURCE_PREFIX + "unexpected.weight"] = torch.zeros(1)
    elif mutation == "wrong_model_shape":
        source_model[model_key] = source_model[model_key].reshape(-1)[:-1]
    elif mutation == "integer_model_dtype":
        source_model[model_key] = source_model[model_key].to(torch.int64)
    elif mutation == "nonfinite_model":
        value = source_model[model_key].clone()
        value.reshape(-1)[0] = torch.nan
        source_model[model_key] = value
    elif mutation == "integer_normalizer_dtype":
        key = "running_mean_std.running_mean"
        source_model[key] = source_model[key].to(torch.int64)
    else:  # pragma: no cover - the parametrization is exhaustive
        raise AssertionError(mutation)

    source = tmp_path / f"{mutation}.pth"
    torch.save(payload, source)
    with pytest.raises(GymCheckpointFormatError):
        convert_gym_checkpoint(source, tmp_path / f"{mutation}.pt")


def test_converter_treats_epoch_and_frame_as_optional_provenance(tmp_path) -> None:
    payload = _source_payload()
    payload.pop("epoch")
    payload.pop("frame")
    source = tmp_path / "source.pth"
    output = tmp_path / "converted.pt"
    torch.save(payload, source)
    convert_gym_checkpoint(source, output)
    provenance = checkpoint_runtime_metadata(output)["source"]
    assert "checkpoint_epoch" not in provenance
    assert "checkpoint_frame" not in provenance


def test_converter_accepts_supported_float_dtypes_and_records_casts(tmp_path) -> None:
    payload = _source_payload()
    source_model = payload["model"]
    assert isinstance(source_model, dict)
    for key, value in source_model.items():
        source_model[key] = value.to(
            torch.float64 if key.startswith(SOURCE_PREFIX) else torch.float32
        )
    source = tmp_path / "source.pth"
    output = tmp_path / "converted.pt"
    torch.save(payload, source)
    convert_gym_checkpoint(source, output)
    modules = torch.load(output, map_location="cpu", weights_only=False)
    assert {value.dtype for value in modules["policy"].values()} == {torch.float32}
    assert {value.dtype for value in modules["observation_preprocessor"].values()} == {
        torch.float64
    }
    metadata = checkpoint_runtime_metadata(output)
    assert metadata["conversion"]["dtype_conversion"] == {
        "model": {"source": ["float64"], "target": "float32"},
        "observation_normalizer": {"source": ["float32"], "target": "float64"},
        "value_normalizer": {"source": ["float32"], "target": "float64"},
    }


def test_converted_sidecar_rejects_invalid_provenance_but_not_versioned_defaults(
    tmp_path,
) -> None:
    source = tmp_path / "source.pth"
    output = tmp_path / "converted.pt"
    torch.save(_source_payload(), source)
    convert_gym_checkpoint(source, output)
    sidecar_path = output.with_suffix(".pt.json")
    metadata = json.loads(sidecar_path.read_text())
    metadata["source"]["sha256"] = "invalid"
    sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(CheckpointFormatError):
        checkpoint_runtime_metadata(output)

    convert_output = tmp_path / "future.pt"
    convert_gym_checkpoint(source, convert_output)
    future_sidecar = convert_output.with_suffix(".pt.json")
    future = json.loads(future_sidecar.read_text())
    future["environment_contract"] = "future_mano_environment_v2"
    future["reward_contract"] = "future_reward_v2"
    future["ppo_reward_contract"] = "future_ppo_v2"
    future["runtime_config"]["point_cloud"].update(
        {"points": 128, "coordinate_frame": "object_relative", "sampling_backend": "future_sampler"}
    )
    future["runtime_config"]["residual_action"]["gamma_xy"] = 0.8
    future["runtime_config"]["deterministic_policy_mode"] = "future_evaluator_choice"
    future_sidecar.write_text(json.dumps(future), encoding="utf-8")
    assert checkpoint_runtime_metadata(convert_output) == future


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for source point-sampler parity")
def test_dynamic_point_sampler_uses_global_cuda_rng() -> None:
    torch.cuda.manual_seed_all(42)
    first = _torch_global_surface_templates(1)
    second = _torch_global_surface_templates(1)

    torch.cuda.manual_seed_all(42)
    repeated = _torch_global_surface_templates(1)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, second)


def test_converted_model_preserves_fixed_observation_policy_output(tmp_path) -> None:
    payload = _source_payload()
    source = tmp_path / "source.pth"
    output = tmp_path / "converted.pt"
    torch.save(payload, source)
    convert_gym_checkpoint(source, output)
    modules = torch.load(output, map_location="cpu", weights_only=False)
    target = _target_model()
    target.load_state_dict(modules["policy"], strict=True)
    torch.manual_seed(42)
    observations = torch.randn(11, 476)
    source_model = payload["model"]
    assert isinstance(source_model, dict)
    source_mean = source_model["running_mean_std.running_mean"].float()
    source_variance = source_model["running_mean_std.running_var"].float()
    expected_normalized = torch.clamp(
        (observations - source_mean) / torch.sqrt(source_variance + 1.0e-5),
        -5.0,
        5.0,
    )
    observation_scaler = PointCloudAwareRunningStandardScaler()
    observation_scaler.load_state_dict(modules["observation_preprocessor"], strict=True)
    actual_normalized = observation_scaler(observations)
    torch.testing.assert_close(actual_normalized, expected_normalized, rtol=0.0, atol=0.0)

    with torch.no_grad():
        actual_policy, _ = target.compute({"observations": actual_normalized}, role="policy")
        actual_value, _ = target.compute({"observations": actual_normalized}, role="value")
        expected_policy, expected_value = _source_policy_value(source_model, expected_normalized)
    torch.testing.assert_close(actual_policy, expected_policy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_value, expected_value, rtol=0.0, atol=0.0)

    value_scaler = SourceRunningStandardScaler()
    value_scaler.load_state_dict(modules["value_preprocessor"], strict=True)
    normalized_values = torch.tensor([[-1.0], [0.0], [1.0]])
    expected_denormalized = (
        torch.sqrt(source_model["value_mean_std.running_var"].float() + 1.0e-5)
        * normalized_values
        + source_model["value_mean_std.running_mean"].float()
    )
    torch.testing.assert_close(
        value_scaler(normalized_values, inverse=True),
        expected_denormalized,
        rtol=0.0,
        atol=0.0,
    )
