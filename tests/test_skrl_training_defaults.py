from types import SimpleNamespace

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
    assert args.object_friction == defaults.object_friction == "0.5 0.01 0.001"
    assert args.finger_kp == defaults.finger_kp == 15.0
    assert args.finger_kv == defaults.finger_kv == 0.0
    assert args.naconmax == defaults.naconmax == 8192
    assert args.njmax == defaults.njmax == 8192


def test_env_defaults_match_reference_pd_baseline() -> None:
    cfg = DexHandRLMJXEnvConfig()

    assert cfg.base_rot_kp == 10000.0
    assert cfg.base_rot_kv == 125.0
    assert cfg.base_rot_force == 100.0
    assert cfg.finger_kp == 15.0
    assert cfg.finger_kv == 0.0


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
