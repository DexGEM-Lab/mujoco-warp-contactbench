"""v4 model/adapter integration without running an optimiser or trainer."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
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


class _TinyV4Adapter:
    """PPO fixture that distinguishes terminal next state from reset state."""
    def __init__(self):
        self.num_envs=1; self.device=torch.device('cpu'); self.observation_space=gym.spaces.Box(-1.,1.,shape=(RAW_OBSERVATION_DIM,)); self.action_space=gym.spaces.Box(-1.,1.,shape=(ACTION_DIM,)); self.phase=0; self.seen=[]; self.physical=[]
    def reset(self): self.phase=0; return torch.zeros((1,RAW_OBSERVATION_DIM)),{}
    def step(self, actions):
        self.seen.append(actions.detach().clone()); self.physical.append(torch.clamp(actions,-1.,1.).detach().clone()); self.phase+=1
        return torch.full((1,RAW_OBSERVATION_DIM),float(self.phase)),torch.ones((1,1)),torch.tensor([[self.phase==1]]),{'valid':torch.tensor(True)}
    def prepare_action(self):
        if self.phase==1: self.phase=0
        return torch.full((1,RAW_OBSERVATION_DIM),float(self.phase))
    def _to_torch(self, value): return value
    def compact_summary(self): return {'object_motion':torch.tensor(0.),'contact_force':torch.tensor(0.),'path':torch.tensor(0.)}


def test_v4_ppo_raw_action_terminal_reset_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    import sim.manorl.autonomy_batch_training as training
    original=training.build_batched_runtime; before={}
    def saturated(*args, **kwargs):
        model,agent=original(*args,**kwargs)
        before.update({name: value.detach().clone() for name,value in model.pointnet.named_parameters()})
        with torch.no_grad():
            model.mean.bias.fill_(4.); model.log_std.fill_(-4.6051702)
        return model,agent
    monkeypatch.setattr(training,'build_batched_runtime',saturated)
    adapter=_TinyV4Adapter(); checkpoint=tmp_path/'v4.pt'
    model,agent,rows=training.run_batched_ppo(adapter,updates=2,rollouts=2,learning_epochs=1,mini_batches=1,checkpoint=checkpoint,checkpoint_interval=1,provenance={'package_digest':'p'})
    # raw samples reach PPO and the physical boundary alone clips them.
    stored=agent.memory.get_tensor_by_name('actions')[0]
    assert (adapter.seen[0].abs()>1).any() and (stored.abs()>1).any()
    torch.testing.assert_close(adapter.physical[0],torch.clamp(adapter.seen[0],-1.,1.))
    # terminal t+1 was observed before prepare_action returned frame zero.
    assert adapter.phase == 0 and rows[0]['terminations']==2.
    assert agent.optimizer.state and any(torch.count_nonzero(state['exp_avg']) for state in agent.optimizer.state.values())
    assert any(not torch.equal(before[name], value) for name,value in model.pointnet.named_parameters())
    restored=_model(); payload=training.load_frozen_v4(checkpoint,restored,expected_provenance={'package_digest':'p'})
    raw=torch.randn(2,RAW_OBSERVATION_DIM)
    torch.testing.assert_close(model.compute({'observations':raw},role='policy')[0],restored.compute({'observations':raw},role='policy')[0])
    assert payload['optimizer']['state'] and payload['normalizer'] is None


def test_v4_public_cli_lists_train_evaluate_and_b4096_controls():
    root=Path(__file__).resolve().parents[2]
    output=subprocess.check_output([sys.executable,str(root/'tools/train_manorl_autonomy.py'),'train','--help'],text=True)
    for option in ('--num-envs','--persistentworkspace','--ccd-contacts-per-world','--wandb'):
        assert option in output
