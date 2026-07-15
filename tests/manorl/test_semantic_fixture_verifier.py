from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from sim.manorl.rewards import REWARD_CONTRACT_ID


def _verifier_tool():
    path = Path(__file__).resolve().parents[2] / "tools" / "verify_manorl_semantic_fixture.py"
    spec = importlib.util.spec_from_file_location("verify_manorl_semantic_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(spec.name, None)


def test_semantic_verifier_keeps_source_rewards_as_reference_only() -> None:
    tool = _verifier_tool()
    arrays = {
        name: np.asarray([index], dtype=np.float32)
        for index, name in enumerate(tool._SOURCE_REWARD_REFERENCE_FIELDS)
    }

    reference = tool._source_reward_reference(arrays, 0)

    assert reference == {
        name: float(index) for index, name in enumerate(tool._SOURCE_REWARD_REFERENCE_FIELDS)
    }
    assert tool.REWARD_CONTRACT_ID == REWARD_CONTRACT_ID
    assert "pair-filtered hand-object forces" in tool._SOURCE_REWARD_NONCOMPARABILITY
    assert not hasattr(tool, "_reward_result")
