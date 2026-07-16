from __future__ import annotations

import subprocess
import sys


def test_native_checkpoint_requires_target_reward_contract(tmp_path) -> None:
    script = r'''
import json
from pathlib import Path
import sys

import torch

from sim.manorl.abi import ENVIRONMENT_CONTRACT_ID
from sim.manorl.checkpoint import CheckpointFormatError, load_skrl_checkpoint, load_skrl_checkpoint_for_inference, save_skrl_checkpoint
from sim.manorl.rewards import PPO_REWARD_CONTRACT_ID, REWARD_CONTRACT_ID


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
incompatible["reward_contract"] = "legacy_source_contact_v1"
sidecar(checkpoint).write_text(json.dumps(incompatible), encoding="utf-8")
try:
    load_skrl_checkpoint(agent, checkpoint)
except CheckpointFormatError as exc:
    assert "reward contract" in str(exc) and "required" in str(exc)
else:
    raise AssertionError("mismatched reward contract was accepted")

legacy = dict(metadata)
legacy["format"] = "manorl.skrl.ppo.v1"
sidecar(checkpoint).write_text(json.dumps(legacy), encoding="utf-8")
try:
    load_skrl_checkpoint(agent, checkpoint)
except CheckpointFormatError as exc:
    assert "legacy 0.5x-reward checkpoints" in str(exc)
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
mismatched_environment["environment_contract"] = "legacy_residual_xyz_0p005_v1"
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
