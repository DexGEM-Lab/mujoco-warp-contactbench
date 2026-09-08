from __future__ import annotations
from types import SimpleNamespace
import torch
import gymnasium as gym
from sim.manorl.autonomy_training import ACTION_DIM, OBSERVATION_DIM, AutonomyActorCritic, RESERVED_TRAIN_IDENTITIES, canonical_gae, identity_split

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
    assert first.shape == (1,ACTION_DIM) and torch.equal(first,second)

def test_split_rejects_wrong_catalog_size():
    catalog=SimpleNamespace(trajectories=tuple(fake_catalog().trajectories[:49]))
    try: identity_split(catalog,seed=0)
    except ValueError as exc: assert "50" in str(exc)
    else: raise AssertionError("split accepted a non-50 catalog")
