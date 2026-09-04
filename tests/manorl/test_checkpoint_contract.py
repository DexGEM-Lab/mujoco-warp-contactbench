from __future__ import annotations

import subprocess
import sys


def test_native_checkpoint_requires_target_reward_contract(tmp_path) -> None:
    script = r'''
import json
from pathlib import Path
import sys

import torch

from sim.manorl.abi import ENVIRONMENT_CONTRACT_ID, LEGACY_ENVIRONMENT_CONTRACT_IDS
from sim.manorl.checkpoint import CheckpointFormatError, load_skrl_checkpoint, load_skrl_checkpoint_for_inference, save_skrl_checkpoint
from sim.manorl.rewards import (
    LEGACY_PPO_REWARD_CONTRACT_IDS,
    LEGACY_REWARD_CONTRACT_IDS,
    PPO_REWARD_CONTRACT_ID,
    REWARD_CONTRACT_ID,
)


class Agent:
    device = "cpu"

    def __init__(self):
        self.loaded = None

    def save(self, path):
        torch.save(
            {
                "policy": {},
                "value": {},
                "optimizer": {},
                "observation_preprocessor": {},
                "value_preprocessor": {},
            },
            path,
        )

    def load(self, path):
        self.loaded = path


def sidecar(path):
    return path.with_suffix(path.suffix + ".json")


checkpoint = Path(sys.argv[1]) / "manorl.pt"
agent = Agent()
save_skrl_checkpoint(agent, checkpoint, runtime_config={"reward_contract": REWARD_CONTRACT_ID})
metadata = json.loads(sidecar(checkpoint).read_text(encoding="utf-8"))
assert metadata["reward_contract"] == REWARD_CONTRACT_ID
assert metadata["ppo_reward_contract"] == PPO_REWARD_CONTRACT_ID
assert metadata["environment_contract"] == ENVIRONMENT_CONTRACT_ID
load_skrl_checkpoint(agent, checkpoint)
assert agent.loaded == str(checkpoint)

legacy_reward = dict(metadata)
legacy_reward["reward_contract"] = LEGACY_REWARD_CONTRACT_IDS[0]
legacy_reward["ppo_reward_contract"] = LEGACY_PPO_REWARD_CONTRACT_IDS[0]
sidecar(checkpoint).write_text(json.dumps(legacy_reward), encoding="utf-8")
agent.loaded = None
load_skrl_checkpoint_for_inference(agent, checkpoint)
assert agent.loaded == str(checkpoint)
agent.loaded = None
try:
    load_skrl_checkpoint(agent, checkpoint)
except CheckpointFormatError as exc:
    assert "reward contract" in str(exc)
    assert agent.loaded is None
else:
    raise AssertionError("training resume accepted a legacy reward contract")
sidecar(checkpoint).write_text(json.dumps(metadata), encoding="utf-8")

for legacy_contract in LEGACY_ENVIRONMENT_CONTRACT_IDS:
    legacy_environment_contract = dict(metadata)
    legacy_environment_contract["environment_contract"] = legacy_contract
    sidecar(checkpoint).write_text(json.dumps(legacy_environment_contract), encoding="utf-8")
    agent.loaded = None
    load_skrl_checkpoint(agent, checkpoint)
    assert agent.loaded == str(checkpoint)

missing = dict(metadata)
missing.pop("reward_contract")
sidecar(checkpoint).write_text(json.dumps(missing), encoding="utf-8")
try:
    load_skrl_checkpoint(agent, checkpoint)
except CheckpointFormatError as exc:
    assert "reward contract is missing" in str(exc)
else:
    raise AssertionError("missing reward contract was accepted")

incompatible = dict(metadata)
incompatible["reward_contract"] = "source_aligned_hand_object_contact_1x_threshold_2n_v1"
sidecar(checkpoint).write_text(json.dumps(incompatible), encoding="utf-8")
try:
    load_skrl_checkpoint(agent, checkpoint)
except CheckpointFormatError as exc:
    assert "reward contract" in str(exc) and "required" in str(exc)
else:
    raise AssertionError("mismatched reward contract was accepted")

incompatible_ppo = dict(metadata)
incompatible_ppo["ppo_reward_contract"] = (
    "source_aligned_hand_object_contact_1x_threshold_2n_shaper_0p5_v1"
)
sidecar(checkpoint).write_text(json.dumps(incompatible_ppo), encoding="utf-8")
try:
    load_skrl_checkpoint(agent, checkpoint)
except CheckpointFormatError as exc:
    assert "PPO reward contract" in str(exc) and "required" in str(exc)
else:
    raise AssertionError("mismatched PPO reward contract was accepted")

legacy = dict(metadata)
legacy["format"] = "manorl.skrl.ppo.v1"
sidecar(checkpoint).write_text(json.dumps(legacy), encoding="utf-8")
try:
    load_skrl_checkpoint(agent, checkpoint)
except CheckpointFormatError as exc:
    assert "native ManoRL skrl v2 checkpoint" in str(exc)
else:
    raise AssertionError("legacy reward-scale checkpoint was accepted")

missing_ppo = dict(metadata)
missing_ppo.pop("ppo_reward_contract")
sidecar(checkpoint).write_text(json.dumps(missing_ppo), encoding="utf-8")
try:
    load_skrl_checkpoint(agent, checkpoint)
except CheckpointFormatError as exc:
    assert "PPO reward contract is missing" in str(exc)
else:
    raise AssertionError("missing PPO reward contract was accepted")

legacy_environment = dict(metadata)
legacy_environment.pop("environment_contract")
sidecar(checkpoint).write_text(json.dumps(legacy_environment), encoding="utf-8")
agent.loaded = None
try:
    load_skrl_checkpoint(agent, checkpoint)
except CheckpointFormatError as exc:
    assert "environment contract is missing" in str(exc)
    assert agent.loaded is None
else:
    raise AssertionError("missing environment contract was accepted")
try:
    load_skrl_checkpoint_for_inference(agent, checkpoint)
except CheckpointFormatError as exc:
    assert "environment contract is missing" in str(exc)
    assert agent.loaded is None
else:
    raise AssertionError("inference accepted a missing environment contract")

mismatched_environment = dict(metadata)
mismatched_environment["environment_contract"] = (
    "source_aligned_film_dynamic_residual_gym_authority_early50_pre250_"
    "deviation_0p10_v1"
)
sidecar(checkpoint).write_text(json.dumps(mismatched_environment), encoding="utf-8")
agent.loaded = None
try:
    load_skrl_checkpoint(agent, checkpoint)
except CheckpointFormatError as exc:
    assert "environment contract" in str(exc) and "required" in str(exc)
    assert agent.loaded is None
else:
    raise AssertionError("mismatched environment contract was accepted")
try:
    load_skrl_checkpoint_for_inference(agent, checkpoint)
except CheckpointFormatError as exc:
    assert "environment contract" in str(exc) and "required" in str(exc)
    assert agent.loaded is None
else:
    raise AssertionError("inference accepted a mismatched environment contract")

'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_native_checkpoint_validates_recorded_hand_signature(tmp_path) -> None:
    import copy
    import json
    from types import SimpleNamespace

    import pytest
    import torch

    from sim.manorl.checkpoint import (
        CheckpointFormatError,
        load_skrl_checkpoint,
        load_skrl_checkpoint_for_inference,
        save_skrl_checkpoint,
    )

    class Agent:
        device = "cpu"

        def __init__(self) -> None:
            self.loaded: str | None = None
            self.policy = SimpleNamespace(
                action_dim=28,
                observation_dim=480,
                use_film=True,
            )

        def save(self, path: str) -> None:
            torch.save(
                {
                    "policy": {},
                    "value": {},
                    "optimizer": {},
                    "observation_preprocessor": {},
                    "value_preprocessor": {},
                },
                path,
            )

        def load(self, path: str) -> None:
            self.loaded = path

    signature = {
        "asset_source_repository": "git@github.com:DexGEM-Lab/dexstream_digital-assets.git",
        "asset_source_commit": "f98da997f316c8a6b4bc2931cabed19e831ef163",
        "asset_manifest_sha256": "a" * 64,
        "resolved_hand_side": "right",
        "available_hand_sides": ["right", "left"],
        "controlled_hand_sides": ["right"],
        "reference_following_hand_sides": ["left"],
        "action_dim": 28,
        "observation_dim": 480,
        "model_action_dim": 56,
        "reference_fps": 120,
        "control_fps": 120,
        "control_timestep_seconds": 1.0 / 120.0,
        "physics_fps": 480,
        "physics_timestep_seconds": 1.0 / 480.0,
        "physics_substeps_per_control": 4,
        "pre_padding": 180,
        "post_padding": 250,
        "warp_ccd": {
            "ccd_iterations": None,
            "contacts_per_world": 16,
            "naccdmax": 32768,
        },
        "residual_action": {
            "joint_scale_multiplier": 1.0,
            "joint_max_offset_multiplier": 1.0,
        },
    }
    agent = Agent()
    agent.manorl_environment_signature = signature.copy()
    checkpoint = save_skrl_checkpoint(
        agent,
        tmp_path / "right.pt",
        runtime_config={"model": {"use_film": True}, "environment": signature},
    )
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))

    # Aggregate CCD scratch scales with num_envs and is not a policy/physics
    # compatibility field. A one-world viewer must load an N2048 checkpoint.
    agent.manorl_environment_signature["warp_ccd"]["naccdmax"] = 16
    load_skrl_checkpoint(agent, checkpoint)
    assert agent.loaded == str(checkpoint)

    for field, value in (
        ("asset_source_repository", "git@github.com:other/assets.git"),
        ("asset_source_commit", "deadbeef" * 8),
        ("asset_manifest_sha256", "b" * 64),
    ):
        mismatched_assets = copy.deepcopy(metadata)
        mismatched_assets["runtime_config"]["environment"][field] = value
        sidecar.write_text(json.dumps(mismatched_assets), encoding="utf-8")
        agent.loaded = None
        with pytest.raises(CheckpointFormatError, match=field):
            load_skrl_checkpoint(agent, checkpoint)
        assert agent.loaded is None
        with pytest.raises(CheckpointFormatError, match=field):
            load_skrl_checkpoint_for_inference(agent, checkpoint)
        assert agent.loaded is None

    mismatched = copy.deepcopy(metadata)
    mismatched["runtime_config"]["environment"]["controlled_hand_sides"] = [
        "left"
    ]
    sidecar.write_text(json.dumps(mismatched), encoding="utf-8")
    agent.loaded = None
    with pytest.raises(CheckpointFormatError, match="controlled_hand_sides"):
        load_skrl_checkpoint(agent, checkpoint)
    assert agent.loaded is None
    with pytest.raises(CheckpointFormatError, match="controlled_hand_sides"):
        load_skrl_checkpoint_for_inference(agent, checkpoint)
    assert agent.loaded is None

    mismatched_reference_fps = copy.deepcopy(metadata)
    mismatched_reference_fps["runtime_config"]["environment"]["reference_fps"] = 100
    sidecar.write_text(json.dumps(mismatched_reference_fps), encoding="utf-8")
    with pytest.raises(CheckpointFormatError, match="reference_fps"):
        load_skrl_checkpoint(agent, checkpoint)

    mismatched_control_fps = copy.deepcopy(metadata)
    mismatched_control_fps["runtime_config"]["environment"]["control_fps"] = 100
    sidecar.write_text(json.dumps(mismatched_control_fps), encoding="utf-8")
    with pytest.raises(CheckpointFormatError, match="control_fps"):
        load_skrl_checkpoint(agent, checkpoint)

    mismatched_pre_padding = copy.deepcopy(metadata)
    mismatched_pre_padding["runtime_config"]["environment"]["pre_padding"] = 100
    sidecar.write_text(json.dumps(mismatched_pre_padding), encoding="utf-8")
    with pytest.raises(CheckpointFormatError, match="pre_padding"):
        load_skrl_checkpoint(agent, checkpoint)

    mismatched_post_padding = copy.deepcopy(metadata)
    mismatched_post_padding["runtime_config"]["environment"]["post_padding"] = 100
    sidecar.write_text(json.dumps(mismatched_post_padding), encoding="utf-8")
    with pytest.raises(CheckpointFormatError, match="post_padding"):
        load_skrl_checkpoint(agent, checkpoint)

    mismatched_ccd = copy.deepcopy(metadata)
    mismatched_ccd["runtime_config"]["environment"]["warp_ccd"][
        "contacts_per_world"
    ] = 8
    sidecar.write_text(json.dumps(mismatched_ccd), encoding="utf-8")
    with pytest.raises(CheckpointFormatError, match="warp_ccd"):
        load_skrl_checkpoint(agent, checkpoint)

    mismatched_residual = copy.deepcopy(metadata)
    mismatched_residual["runtime_config"]["environment"]["residual_action"][
        "joint_scale_multiplier"
    ] = 1.5
    sidecar.write_text(json.dumps(mismatched_residual), encoding="utf-8")
    with pytest.raises(CheckpointFormatError, match="residual_action"):
        load_skrl_checkpoint(agent, checkpoint)

    # Sidecars written before explicit multiplier fields had an implicit 1x
    # contract and remain compatible with a current 1x target runtime.
    implicit_one = copy.deepcopy(metadata)
    implicit_one["runtime_config"]["environment"]["residual_action"] = {}
    sidecar.write_text(json.dumps(implicit_one), encoding="utf-8")
    load_skrl_checkpoint(agent, checkpoint)
    assert agent.loaded == str(checkpoint)

    # The current runtime no longer accepts pre-28-DoF sidecars without a hand
    # signature, even if their tensor shapes happen to load.
    legacy = copy.deepcopy(metadata)
    legacy_environment = legacy["runtime_config"]["environment"]
    for field in (
        "asset_source_repository",
        "asset_source_commit",
        "asset_manifest_sha256",
        "resolved_hand_side",
        "available_hand_sides",
        "controlled_hand_sides",
        "reference_following_hand_sides",
        "action_dim",
        "observation_dim",
        "model_action_dim",
    ):
        legacy_environment.pop(field)
    sidecar.write_text(json.dumps(legacy), encoding="utf-8")
    agent.loaded = None
    with pytest.raises(CheckpointFormatError, match="missing current MuJoCo hand signature"):
        load_skrl_checkpoint(agent, checkpoint)
    assert agent.loaded is None


def test_native_checkpoint_restores_all_required_modules_into_evaluator_and_training_agents(tmp_path) -> None:
    import torch

    from sim.manorl.checkpoint import load_skrl_checkpoint, save_skrl_checkpoint

    class Agent:
        device = "cpu"

        def __init__(self) -> None:
            self.policy = torch.nn.Linear(2, 2)
            self.value = torch.nn.Linear(2, 1)
            self.observation_preprocessor = torch.nn.Linear(2, 2)
            self.value_preprocessor = torch.nn.Linear(1, 1)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.01)
            self.modules = {
                "policy": self.policy,
                "value": self.value,
                "optimizer": self.optimizer,
                "observation_preprocessor": self.observation_preprocessor,
                "value_preprocessor": self.value_preprocessor,
            }
            self.save_calls = 0

        def save(self, path: str) -> None:
            self.save_calls += 1
            torch.save({name: module.state_dict() for name, module in self.modules.items()}, path)

        def load(self, path: str) -> None:
            payload = torch.load(path, map_location=self.device, weights_only=False)
            for name, module in self.modules.items():
                module.load_state_dict(payload[name])

    def assert_state_equal(left: object, right: object) -> None:
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right, strict=True):
                assert_state_equal(left_item, right_item)
        else:
            assert left == right

    source = Agent()
    loss = source.policy(torch.ones((1, 2))).sum()
    loss.backward()
    source.optimizer.step()
    checkpoint = save_skrl_checkpoint(source, tmp_path / "initial.pt", runtime_config={"test": "native"})
    evaluator = Agent()
    training = Agent()

    load_skrl_checkpoint(evaluator, checkpoint)
    load_skrl_checkpoint(training, checkpoint)

    assert source.save_calls == 1
    for destination in (evaluator, training):
        for name, source_module in source.modules.items():
            assert_state_equal(source_module.state_dict(), destination.modules[name].state_dict())


def test_warm_start_transfers_models_and_normalizers_but_not_optimizer(tmp_path) -> None:
    import torch

    from sim.manorl.checkpoint import (
        load_skrl_checkpoint_for_warm_start,
        save_skrl_checkpoint,
    )

    class Agent:
        device = "cpu"

        def __init__(self) -> None:
            self.policy = torch.nn.Linear(2, 2)
            self.value = torch.nn.Linear(2, 1)
            self.observation_preprocessor = torch.nn.Linear(2, 2)
            self.value_preprocessor = torch.nn.Linear(1, 1)
            self.optimizer = torch.optim.Adam(
                [*self.policy.parameters(), *self.value.parameters()], lr=0.01
            )
            self.checkpoint_modules = {
                "policy": self.policy,
                "value": self.value,
                "optimizer": self.optimizer,
                "observation_preprocessor": self.observation_preprocessor,
                "value_preprocessor": self.value_preprocessor,
            }

        def save(self, path: str) -> None:
            torch.save(
                {
                    name: module.state_dict()
                    for name, module in self.checkpoint_modules.items()
                },
                path,
            )

    source = Agent()
    loss = source.policy(torch.ones((1, 2))).sum() + source.value(
        torch.ones((1, 2))
    ).sum()
    loss.backward()
    source.optimizer.step()
    checkpoint = save_skrl_checkpoint(
        source,
        tmp_path / "source.pt",
        runtime_config={"model": {}},
    )
    target = Agent()
    assert not target.optimizer.state_dict()["state"]

    load_skrl_checkpoint_for_warm_start(target, checkpoint)

    for name in (
        "policy",
        "value",
        "observation_preprocessor",
        "value_preprocessor",
    ):
        source_state = source.checkpoint_modules[name].state_dict()
        target_state = target.checkpoint_modules[name].state_dict()
        assert source_state.keys() == target_state.keys()
        for key in source_state:
            torch.testing.assert_close(source_state[key], target_state[key])
    assert source.optimizer.state_dict()["state"]
    assert not target.optimizer.state_dict()["state"]


def test_checkpoint_clock_distinguishes_current_and_legacy_contracts() -> None:
    import pytest

    from sim.manorl.abi import ENVIRONMENT_CONTRACT_ID, LEGACY_ENVIRONMENT_CONTRACT_IDS
    from sim.manorl.checkpoint import CheckpointFormatError, checkpoint_simulation_clock

    current = {
        "environment_contract": ENVIRONMENT_CONTRACT_ID,
        "runtime_config": {
            "environment": {"reference_fps": 120, "control_fps": 120}
        },
    }
    clock = checkpoint_simulation_clock(current)
    assert (clock.policy_fps, clock.physics_fps, clock.physics_substeps_per_control) == (
        120,
        480,
        4,
    )

    mixed = {
        "environment_contract": ENVIRONMENT_CONTRACT_ID,
        "runtime_config": {
            "environment": {"reference_fps": 100, "control_fps": 120}
        },
    }
    with pytest.raises(CheckpointFormatError, match="reference_fps == control_fps"):
        checkpoint_simulation_clock(mixed)

    for contract in LEGACY_ENVIRONMENT_CONTRACT_IDS:
        legacy_clock = checkpoint_simulation_clock(
            {
                "environment_contract": contract,
                "runtime_config": {
                    "environment": {"reference_fps": 100}
                },
            }
        )
        assert (
            legacy_clock.policy_fps,
            legacy_clock.physics_fps,
            legacy_clock.physics_substeps_per_control,
        ) == (200, 400, 2)
