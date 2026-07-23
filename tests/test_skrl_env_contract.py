from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from sim.dexhandrl import skrl_env as skrl_env_module
from sim.dexhandrl.skrl_env import DexHandRLMJXEnv, DexHandRLMJXEnvConfig


def test_runtime_backend_defaults_to_mjx_warp_and_native_gpu_fails_closed() -> None:
    assert DexHandRLMJXEnvConfig().runtime_backend == "mjx-warp"
    with pytest.raises(ValueError, match="requires device='cpu'"):
        DexHandRLMJXEnvConfig(runtime_backend="mujoco-native", device="gpu")
    with pytest.raises(ValueError, match="unsupported runtime_backend"):
        DexHandRLMJXEnvConfig(runtime_backend="unknown")


def test_native_runtime_uses_exact_reset_and_control_substep_counts(monkeypatch) -> None:
    names = skrl_env_module.DEXHAND021PRO_FULL_DOF_NAMES
    ids = type("Ids", (), {
        "joint_qpos_addr": {name: index for index, name in enumerate(names)},
        "joint_dof_addr": {name: index for index, name in enumerate(names)},
        "object_free_qpos_addr": 26, "object_free_dof_addr": 26,
        "actuator_ids": {f"act_{name}": index for index, name in enumerate(names)},
    })()
    state = type("Data", (), {
        "qpos": np.zeros(33), "qvel": np.zeros(32), "qacc": np.zeros(32),
        "ctrl": np.zeros(26), "time": 0.0,
    })()
    calls = {"forward": 0, "step": 0}
    mujoco = type("MuJoCo", (), {
        "MjData": staticmethod(lambda _model: state),
        "mj_forward": staticmethod(lambda _model, _data: calls.__setitem__("forward", calls["forward"] + 1)),
        "mj_step": staticmethod(lambda _model, _data: calls.__setitem__("step", calls["step"] + 1)),
    })()
    cfg = type("Cfg", (), {
        "runtime_backend": "mujoco-native", "settle_frames": 2, "object_reset_z_offset": 0.01,
        "physics": type("Physics", (), {"substeps": 3, "control_dt": 0.01})(),
        "reference_joint_source": "active", "early_phase_frames": 100,
        "object_position_terminal_threshold": 1.0,
    })()
    env = object.__new__(DexHandRLMJXEnv)
    env.cfg, env.ids, env.model, env.mujoco = cfg, ids, type("Model", (), {"nu": 26})(), mujoco
    env.trajectory = type("Trajectory", (), {"object_pos": np.zeros((4, 3)), "object_quat_xyzw": np.tile([0, 0, 0, 1], (4, 1))})()
    env.start_frame, env.route_joint_upper_limits = 0, {}
    monkeypatch.setattr(skrl_env_module, "_reference_full_targets", lambda *_args, **_kwargs: np.zeros(26, dtype=np.float32))
    env._set_initial_state()
    assert calls == {"forward": 1, "step": 6}

    env._done, env._current_frame, env._episode_step_count, env.end_frame = False, 1, 0, 3
    env.max_episode_time_s = 1.0
    env._apply_residual_action = lambda *_args: np.zeros(26, dtype=np.float32)
    env._build_observation = lambda **_kwargs: (np.zeros(461, dtype=np.float32), {"reward_terms": {"total": 0.0}, "object_position_error_m": 0.0})
    env._last_full_targets = None
    env.base_position_offset, env.joint_offset = np.zeros(3), np.zeros(16)
    env.step(np.zeros(22, dtype=np.float32))
    assert calls == {"forward": 2, "step": 9}


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


@pytest.mark.slow
def test_native_mujoco_env_reset_step_contract_cpu() -> None:
    pytest.importorskip("mujoco")
    if importlib.util.find_spec("lance") is None:
        pytest.skip("pylance is not installed")

    cfg = DexHandRLMJXEnvConfig(
        runtime_backend="mujoco-native", device="cpu", use_residual=False,
        object_name="cube1", action="01", sequence_index=0, substeps=2,
    )
    if not cfg.lance_path.exists():
        pytest.skip(f"Lance dataset not found: {cfg.lance_path}")

    env = DexHandRLMJXEnv(cfg)
    obs, info = env.reset()
    assert obs.shape == (461,) and np.isfinite(obs).all()
    assert info["runtime_backend"] == "mujoco-native"
    assert env.action_space.shape == (22,)
    obs, reward, _, _, _ = env.step(np.zeros(22, dtype=np.float32))
    assert obs.shape == (461,) and np.isfinite(obs).all()
    assert np.isfinite(env._dx.qpos).all() and np.isfinite(reward)
    env.close()
