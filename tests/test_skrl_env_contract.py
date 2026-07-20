from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from sim.dexhandrl.skrl_env import DexHandRLMJXEnv, DexHandRLMJXEnvConfig


@pytest.mark.slow
def test_skrl_env_reset_step_contract_cpu() -> None:
    jax = pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    if jax.default_backend() != "cpu":
        pytest.skip("JAX was already initialized with a non-CPU backend")
    if importlib.util.find_spec("lance") is None:
        pytest.skip("pylance is not installed")

    cfg = DexHandRLMJXEnvConfig(
        device="cpu",
        use_residual=False,
        object_name="cube1",
        action="01",
        sequence_index=0,
        substeps=2,
    )
    if not cfg.lance_path.exists():
        pytest.skip(f"Lance dataset not found: {cfg.lance_path}")

    env = DexHandRLMJXEnv(cfg)
    obs, info = env.reset()
    assert obs.shape == (461,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert env.action_space.shape == (22,)
    assert info["object_name"] == "cube1"

    obs, reward, terminated, truncated, info = env.step(np.zeros(env.num_actions, dtype=np.float32))
    assert obs.shape == (461,)
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)
    assert reward == pytest.approx(info["reward_terms"]["total"])
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_terms" in info
    env.close()
