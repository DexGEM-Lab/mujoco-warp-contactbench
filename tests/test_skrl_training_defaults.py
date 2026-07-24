from dataclasses import dataclass
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

import sim.dexhandrl.skrl_train as skrl_train
from sim.dexhandrl.contracts import DexHandPhysicsConfig, DexHandServoConfig

from sim.dexhandrl.skrl_train import (
    SkrlLauncherConfig,
    _configure_experiment_output,
    _stack_vector_states,
    build_arg_parser,
)
from sim.dexhandrl.skrl_env import DexHandRLMJXEnvConfig


def test_cli_defaults_match_isaac_ppo_configuration() -> None:
    args = build_arg_parser().parse_args([])
    defaults = SkrlLauncherConfig()

    assert args.rollouts == defaults.rollouts == 48
    assert args.learning_epochs == defaults.learning_epochs == 3
    assert args.mini_batches == defaults.mini_batches == 48
    assert args.learning_rate == defaults.learning_rate == 1e-4
    assert args.discount_factor == defaults.discount_factor == 0.995
    assert args.gae_lambda == defaults.gae_lambda == 0.95
    assert args.value_loss_scale == defaults.value_loss_scale == 4.0
    assert args.kl_threshold == defaults.kl_threshold == 0.008
    assert (
        args.object_position_terminal_threshold
        == defaults.object_position_terminal_threshold
        == 0.1
    )
    assert args.hand_friction == defaults.hand_friction == "0.5 0.01 0.001"
    assert args.object_friction == defaults.object_friction == "0.5 0.01 0.001"
    assert args.settle_frames == defaults.settle_frames == 1
    assert args.finger_kp == defaults.finger_kp == 15.0
    assert args.finger_kv == defaults.finger_kv == 0.0
    assert args.naconmax == defaults.naconmax == 8192
    assert args.njmax == defaults.njmax == 8192


def test_launcher_seed_repeats_host_and_torch_rngs_and_seeds_cuda(monkeypatch) -> None:
    cuda_seeds: list[int] = []
    monkeypatch.setattr(skrl_train.torch.cuda, "manual_seed_all", cuda_seeds.append)

    skrl_train._seed_launcher(17)
    first = (random.random(), float(np.random.random()), torch.rand(4))
    skrl_train._seed_launcher(17)
    second = (random.random(), float(np.random.random()), torch.rand(4))

    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])
    # torch.manual_seed also forwards to CUDA; the explicit CUDA call keeps the
    # launcher contract visible rather than relying on that implementation detail.
    assert cuda_seeds == [17, 17, 17, 17]


def test_env_defaults_match_reference_pd_baseline() -> None:
    cfg = DexHandRLMJXEnvConfig()

    assert cfg.settle_frames == 1
    assert cfg.base_rot_kp == 10000.0
    assert cfg.base_rot_kv == 125.0
    assert cfg.base_rot_force == 100.0
    assert cfg.finger_kp == 15.0
    assert cfg.finger_kv == 0.0
    assert cfg.hand_friction == "0.5 0.01 0.001"
    assert cfg.physics == DexHandPhysicsConfig()


def test_legacy_and_canonical_physics_inputs_resolve_to_read_only_aliases() -> None:
    legacy = DexHandRLMJXEnvConfig(dt=0.02, finger_kp=17, naconmax=64)
    canonical_physics = DexHandPhysicsConfig(
        control_dt=0.02, servo=DexHandServoConfig(finger_kp=17), runtime_naconmax=64
    )
    canonical = DexHandRLMJXEnvConfig(physics=canonical_physics)

    assert legacy.physics == canonical.physics == canonical_physics
    assert (legacy.dt, legacy.finger_kp, legacy.naconmax) == (0.02, 17, 64)
    assert (canonical.dt, canonical.finger_kp, canonical.naconmax) == (0.02, 17, 64)
    with pytest.raises(ValueError, match="legacy physical"):
        DexHandRLMJXEnvConfig(physics=canonical_physics, dt=0.02)


def test_launcher_forwards_one_physics_identity_and_serializes_contract(tmp_path, monkeypatch) -> None:
    physics = DexHandPhysicsConfig(runtime_naconmax=64)
    cfg = SkrlLauncherConfig(physics=physics, output_dir=tmp_path)
    captured = {}

    class FakeEnv:
        def __init__(self, env_cfg):
            captured["physics"] = env_cfg.physics

    monkeypatch.setattr(skrl_train, "DexHandRLMJXEnv", FakeEnv)
    skrl_train._make_env_factory(cfg, {"uuid": "x", "sequence_index": 0}, 0)()
    assert captured["physics"] is physics

    @dataclass
    class ModelCfg:
        enabled: bool = True

    skrl_train._write_launcher_config(cfg, [], ModelCfg())
    import json
    payload = json.loads((tmp_path / "launcher_config.json").read_text())
    assert payload["physics_contract"]["runtime_naconmax"] == 64
    assert payload["physics_contract"]["physics_dt"] == 0.001
    assert payload["physics_contract"]["servo"]["finger_kp"] == 15


def test_parser_exposes_complete_physics_overrides() -> None:
    args = build_arg_parser().parse_args([
        "--control-dt", "0.02", "--ground-friction", "3 2 2",
        "--base-pos-kp", "7", "--base-rot-force", "8",
    ])
    assert (args.control_dt, args.ground_friction, args.base_pos_kp, args.base_rot_force) == (
        0.02, "3 2 2", 7.0, 8.0
    )


def test_experiment_output_uses_launcher_output_directory(tmp_path) -> None:
    agent_cfg = SimpleNamespace(experiment=SimpleNamespace())
    output_dir = tmp_path / "seq1_train"

    _configure_experiment_output(agent_cfg, output_dir)

    assert agent_cfg.experiment.directory == str(tmp_path)
    assert agent_cfg.experiment.experiment_name == "seq1_train"


def test_vector_state_stacks_single_agent_observations() -> None:
    vector_env = SimpleNamespace(
        envs=[
            SimpleNamespace(state=lambda: [1.0, 2.0]),
            SimpleNamespace(state=lambda: [3.0, 4.0]),
        ]
    )

    states = _stack_vector_states(vector_env)

    assert states.shape == (2, 2)
    assert states.tolist() == [[1.0, 2.0], [3.0, 4.0]]
