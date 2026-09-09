from __future__ import annotations
from types import SimpleNamespace
import subprocess
import pytest
import torch
import gymnasium as gym
from sim.manorl.autonomy_training import ACTION_DIM, OBSERVATION_DIM, AutonomyActorCritic, RESERVED_TRAIN_IDENTITIES, canonical_gae, identity_split
from tools.train_manorl_autonomy import _git_revision, formal_rollout_schedule

def fake_catalog():
    ids=[f"cube2_02_{2833+i}" for i in range(50)]
    return SimpleNamespace(trajectories=tuple(SimpleNamespace(identity=SimpleNamespace(identity=x)) for x in ids))

def test_identity_split_is_seeded_disjoint_and_reserves_inspected_ids():
    split=identity_split(fake_catalog(),seed=17)
    assert len(split["train_indices"])==40 and len(split["validation_indices"])==5 and len(split["test_indices"])==5
    assert not set(split["train_indices"]) & set(split["validation_indices"])
    assert not set(split["train_indices"]) & set(split["test_indices"])
    assert set(RESERVED_TRAIN_IDENTITIES) <= set(split["train_identities"])
    assert split["digest"] == identity_split(fake_catalog(),seed=17)["digest"]

def test_canonical_gae_bootstraps_time_limit_but_not_true_termination():
    # skrl's Agent.record_transition adds gamma*next_value to a truncated
    # reward before compute_gae; compute_gae then treats the truncation as a
    # boundary while preserving the bootstrap in the adjusted reward.
    rewards=torch.tensor([[[2.8]],[[1.]]])
    values=torch.tensor([[[0.]],[[0.]]])
    last=torch.tensor([[[2.]]])
    terminated=torch.tensor([[[False]],[[True]]])
    truncated=torch.tensor([[[True]],[[False]]])
    returns, advantages=canonical_gae(rewards,terminated,truncated,values,last,discount_factor=.9,lambda_coefficient=.95,time_limit_bootstrap=True)
    assert torch.isfinite(returns).all() and torch.isfinite(advantages).all()
    assert returns.shape == rewards.shape
    assert returns[0,0,0] > 1.0  # truncated transition receives bootstrap value

def test_actor_provenance_is_all28_and_deterministic_in_eval_mode():
    model=AutonomyActorCritic(gym.spaces.Box(-5.,5.,shape=(OBSERVATION_DIM,),dtype=float),gym.spaces.Box(-1.,1.,shape=(ACTION_DIM,),dtype=float),device="cpu")
    model.eval(); x={"observations":torch.zeros((1,OBSERVATION_DIM))}; first=model.compute(x,role="policy")[0]; second=model.compute(x,role="policy")[0]
    value, _ = model.act(x, role="value")
    assert first.shape == (1,ACTION_DIM) and torch.equal(first,second) and value.shape == (1,1)

def test_split_rejects_wrong_catalog_size():
    catalog=SimpleNamespace(trajectories=tuple(fake_catalog().trajectories[:49]))
    try: identity_split(catalog,seed=0)
    except ValueError as exc: assert "50" in str(exc)
    else: raise AssertionError("split accepted a non-50 catalog")

def test_git_revision_uses_valid_archive_marker_and_rejects_invalid_or_missing_metadata(tmp_path):
    commit = "70ff46a" + "0" * 33
    (tmp_path / "DEPLOYED_COMMIT").write_text(commit + "\n", encoding="utf-8")
    assert _git_revision(tmp_path) == commit

    (tmp_path / "DEPLOYED_COMMIT").write_text("not-a-full-commit\n", encoding="utf-8")
    with pytest.raises(ValueError, match="40-character lowercase hexadecimal"):
        _git_revision(tmp_path)

    (tmp_path / "DEPLOYED_COMMIT").unlink()
    with pytest.raises(subprocess.CalledProcessError):
        _git_revision(tmp_path)


def test_formaltrain_schedule_keeps_episode_across_rollout_cuts():
    records, resets=formal_rollout_schedule(horizon=5,rollouts=3,updates=3)
    assert [phase for phase,_ in records] == [1,2,3,4,5,1,2,3,4]
    assert [phase for phase,terminal in records if terminal] == [5]
    assert resets == 2

def test_ordinary_adapter_requires_explicit_reset_and_marks_horizon_terminated():
    import numpy as np
    from sim.manorl.autonomy import AutonomousStep
    from sim.manorl.autonomy_training import AutonomyVectorEnv
    class Fake:
        def __init__(self): self.resets=0; self.steps=0
        def reset(self): self.resets+=1; return np.zeros(OBSERVATION_DIM,dtype=np.float32)
        def step(self, action):
            assert action.shape == (ACTION_DIM,); self.steps+=1
            phase="horizon_reached" if self.steps == 1 else "drop"
            return AutonomousStep(np.zeros(OBSERVATION_DIM,dtype=np.float32), 1., {}, True, {"failure_phase":phase,"task_success":False})
    env=object.__new__(AutonomyVectorEnv); env.environment=Fake(); env.trajectory=SimpleNamespace(identity=SimpleNamespace(identity="cube2_02_2833")); env.contact_conditioned=True; env.num_envs=1; env._device="cpu"; env.observation_space=gym.spaces.Box(-5.,5.,shape=(OBSERVATION_DIM,),dtype=np.float32); env.action_space=gym.spaces.Box(-1.,1.,shape=(ACTION_DIM,),dtype=np.float32); env._pending=False
    env.reset(); _,_,terminated,truncated,_=env.step(np.zeros(ACTION_DIM)); assert terminated is True and truncated is False
    try: env.step(np.zeros(ACTION_DIM))
    except RuntimeError: pass
    else: raise AssertionError("terminal action was accepted without reset")
    env.reset(); _,_,terminated,truncated,_=env.step(np.zeros(ACTION_DIM)); assert terminated is True and truncated is False and env.environment.resets == 2
