"""v4 model/adapter integration without running an optimiser or trainer."""
from __future__ import annotations
import pytest, torch, gymnasium as gym
from sim.manorl.autonomy_contracts import RAW_OBSERVATION_DIM,ENCODED_OBSERVATION_DIM,ACTION_DIM,CHECKPOINT_FORMAT,OBSERVATION_CONTRACT_ID,ACTION_CONTRACT_ID,REWARD_CONTRACT_ID,validate_v4_checkpoint_metadata
from sim.manorl.autonomy_training import AutonomyActorCritic

def _model():
    return AutonomyActorCritic(gym.spaces.Box(-1.,1.,shape=(RAW_OBSERVATION_DIM,)),gym.spaces.Box(-1.,1.,shape=(ACTION_DIM,)),device='cpu')

def test_raw957_pointnet829_policy_value_and_encoder_gradients():
    model=_model(); raw=torch.randn(3,RAW_OBSERVATION_DIM)
    policy,_=model.compute({'observations':raw},role='policy'); value,_=model.compute({'observations':raw},role='value')
    assert policy.shape==(3,28) and value.shape==(3,1) and model.checkpoint_architecture()['encoded_feature_dim']==ENCODED_OBSERVATION_DIM
    (policy.square().mean()+value.square().mean()).backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.pointnet.parameters())

def test_v4_checkpoint_roundtrip_restores_pointnet_and_rejects_old_contract():
    model=_model(); payload={'checkpoint_format':CHECKPOINT_FORMAT,'observation_contract':OBSERVATION_CONTRACT_ID,'action_contract':ACTION_CONTRACT_ID,'reward_contract':REWARD_CONTRACT_ID,'model':model.state_dict(),'model_architecture':model.checkpoint_architecture()}
    validate_v4_checkpoint_metadata(payload); restored=_model(); restored.load_state_dict(payload['model'],strict=True)
    for a,b in zip(model.pointnet.parameters(),restored.pointnet.parameters()): assert torch.equal(a,b)
    with pytest.raises(ValueError): validate_v4_checkpoint_metadata({'checkpoint_format':'manorl.autonomy.ppo.v3.1'})
