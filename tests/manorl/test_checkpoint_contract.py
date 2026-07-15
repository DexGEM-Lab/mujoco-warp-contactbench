from __future__ import annotations

import subprocess
import sys


def test_native_checkpoint_requires_target_reward_contract(tmp_path) -> None:
    script = r'''
import json
from pathlib import Path
import sys

import torch

from sim.manorl.checkpoint import CheckpointFormatError, load_skrl_checkpoint, save_skrl_checkpoint
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
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
