"""v4 model/adapter integration without running an optimiser or trainer."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import pytest, torch, gymnasium as gym
from sim.manorl.autonomy_contracts import RAW_OBSERVATION_DIM,ENCODED_OBSERVATION_DIM,ACTION_DIM,CHECKPOINT_FORMAT,OBSERVATION_CONTRACT_ID,ACTION_CONTRACT_ID,REWARD_CONTRACT_ID,validate_v4_checkpoint_metadata
from sim.manorl.autonomy_training import AutonomyActorCritic

def _model(*, separate_critic=False):
    return AutonomyActorCritic(gym.spaces.Box(-1.,1.,shape=(RAW_OBSERVATION_DIM,)),gym.spaces.Box(-1.,1.,shape=(ACTION_DIM,)),device='cpu',separate_critic=separate_critic)

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
    assert rows[0]['config/grad_norm_clip'] == .5
    assert rows[0]['config/normalize_observations'] == 0.
    assert rows[0]['config/normalize_advantages'] == 0.
    assert agent.optimizer.state and any(torch.count_nonzero(state['exp_avg']) for state in agent.optimizer.state.values())
    assert any(not torch.equal(before[name], value) for name,value in model.pointnet.named_parameters())
    restored=_model(); payload=training.load_frozen_v4(checkpoint,restored,expected_provenance={'package_digest':'p'})
    raw=torch.randn(2,RAW_OBSERVATION_DIM)
    torch.testing.assert_close(model.compute({'observations':raw},role='policy')[0],restored.compute({'observations':raw},role='policy')[0])
    assert payload['optimizer']['state'] and payload['normalizer'] is None


def _v4_payload(model, provenance):
    return {'checkpoint_format':CHECKPOINT_FORMAT,'observation_contract':OBSERVATION_CONTRACT_ID,
            'action_contract':ACTION_CONTRACT_ID,'reward_contract':REWARD_CONTRACT_ID,
            'model':model.state_dict(),'model_architecture':model.checkpoint_architecture(),
            'provenance':provenance}


def test_v4_shared_teacher_to_separate_critic_preserves_policy_and_fresh_value_before_rollout(tmp_path, monkeypatch):
    import sim.manorl.autonomy_batch_training as training
    torch.manual_seed(11); teacher=_model()
    warmstart=tmp_path/'teacher.pt'
    provenance={'package_digest':'p','clock':{'control_timestep':1/120,'physics_substeps':4,
                                             'full_horizon_diagnostic':True}}
    # Deliberately omit optimizer: PPO warm-start must never require or restore it.
    torch.save(_v4_payload(teacher,provenance),warmstart)
    original=training.build_batched_runtime; observed={}
    def inspect_first_rollout(*args, **kwargs):
        model,agent=original(*args,**kwargs)
        assert model.separate_critic and model.value_net is not None
        observed['fresh_optimizer_state']=len(agent.optimizer.state)
        observed['initial_value']={name:value.detach().clone() for name,value in model.value.named_parameters()}
        observed['initial_value_net']={name:value.detach().clone() for name,value in model.value_net.named_parameters()}
        original_act=agent.act
        def act(observations, *act_args, **act_kwargs):
            if 'first_mean' not in observed:
                with torch.no_grad():
                    observed['first_mean']=model.compute({'observations':observations},role='policy')[0].clone()
                    observed['teacher_mean']=teacher.compute({'observations':observations},role='policy')[0].clone()
                    observed['loaded_net']={name:value.detach().clone() for name,value in model.net.named_parameters()}
                    observed['loaded_value']={name:value.detach().clone() for name,value in model.value.named_parameters()}
                    observed['loaded_value_net']={name:value.detach().clone() for name,value in model.value_net.named_parameters()}
            return original_act(observations,*act_args,**act_kwargs)
        agent.act=act
        return model,agent
    monkeypatch.setattr(training,'build_batched_runtime',inspect_first_rollout)
    output=tmp_path/'ppo.pt'
    model,agent,_=training.run_batched_ppo(
        _TinyV4Adapter(),updates=1,rollouts=2,learning_epochs=1,mini_batches=1,
        checkpoint=output,separate_critic=True,warmstart=warmstart,
        expected_warmstart_provenance={'package_digest':'p','clock':{'control_timestep':1/120,'physics_substeps':4}},
        provenance={'package_digest':'p'},
    )
    assert observed['fresh_optimizer_state']==0
    torch.testing.assert_close(observed['first_mean'],observed['teacher_mean'],rtol=0,atol=0)
    for name in observed['initial_value']:
        assert torch.equal(observed['initial_value'][name],observed['loaded_value'][name])
    for name in observed['initial_value_net']:
        assert torch.equal(observed['initial_value_net'][name],observed['loaded_value_net'][name])
    assert any(not torch.equal(observed['loaded_net'][name], observed['loaded_value_net'][name])
               for name in observed['loaded_value_net'])
    assert {id(p) for group in agent.optimizer.param_groups for p in group['params']} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    saved=torch.load(output,map_location='cpu',weights_only=False)
    assert saved['model_architecture']['value_trunk']=='separate'
    assert saved['config']['separate_critic'] is True
    assert saved['config']['warmstart_transfer_mode']=='shared_policy_to_separate_critic'
    assert saved['provenance']['warmstart_checkpoint']==str(warmstart.resolve())
    assert saved['provenance']['warmstart_transfer_mode']=='shared_policy_to_separate_critic'
    assert saved['optimizer']['state']


def test_v4_same_architecture_warmstart_remains_exact_and_strict(tmp_path):
    import sim.manorl.autonomy_batch_training as training
    torch.manual_seed(3); teacher=_model(separate_critic=True); path=tmp_path/'same.pt'
    torch.save(_v4_payload(teacher,{'package_digest':'p'}),path)
    torch.manual_seed(4); target=_model(separate_critic=True)
    payload=training.load_v4_warmstart(path,target,expected_provenance={'package_digest':'p'})
    assert payload['warmstart_transfer_mode']=='exact_model'
    for name,value in teacher.state_dict().items(): assert torch.equal(value,target.state_dict()[name])
    malformed=_v4_payload(teacher,{'package_digest':'p'}); malformed['model'].pop('value.bias')
    bad=tmp_path/'malformed.pt'; torch.save(malformed,bad)
    with pytest.raises(RuntimeError,match='Missing key'):
        training.load_v4_warmstart(bad,_model(separate_critic=True))


@pytest.mark.parametrize(('field','incompatible'),(('action_dim',27),('raw_observation_dim',956)))
def test_v4_warmstart_rejects_incompatible_action_and_observation_architecture(tmp_path,field,incompatible):
    import sim.manorl.autonomy_batch_training as training
    source=_model(); payload=_v4_payload(source,{'package_digest':'teacher-package'})
    payload['model_architecture']={**payload['model_architecture'],field:incompatible}
    architecture=tmp_path/f'{field}.pt'; torch.save(payload,architecture)
    with pytest.raises(ValueError,match='architecture'):
        training.load_v4_warmstart(architecture,_model(separate_critic=True))


def test_v4_warmstart_rejects_legacy_reverse_transfer_and_provenance_mismatch(tmp_path):
    import sim.manorl.autonomy_batch_training as training
    legacy=tmp_path/'v3.pt'; torch.save({'checkpoint_format':'manorl.autonomy.ppo.v3.1'},legacy)
    with pytest.raises(ValueError,match='checkpoint_format'):
        training.load_v4_warmstart(legacy,_model())
    separate=_model(separate_critic=True); reverse=tmp_path/'reverse.pt'
    torch.save(_v4_payload(separate,{'package_digest':'teacher-package'}),reverse)
    with pytest.raises(ValueError,match='architecture'):
        training.load_v4_warmstart(reverse,_model())
    payload=_v4_payload(_model(),{'package_digest':'teacher-package'}); provenance=tmp_path/'provenance.pt'; torch.save(payload,provenance)
    with pytest.raises(ValueError,match='provenance.package_digest'):
        training.load_v4_warmstart(provenance,_model(),expected_provenance={'package_digest':'current-package'})


def test_v4_public_cli_lists_and_parses_warmstart_and_separate_critic():
    root=Path(__file__).resolve().parents[2]
    output=subprocess.check_output([sys.executable,str(root/'tools/train_manorl_autonomy.py'),'train','--help'],text=True)
    for option in ('--num-envs','--persistentworkspace','--ccd-contacts-per-world','--warmstart','--separate-critic','--wandb'):
        assert option in output
    from tools import train_manorl_autonomy as cli
    defaults=cli.parse_args(['train'])
    assert defaults.warmstart is None and defaults.separate_critic is False
    selected=cli.parse_args(['train','--warmstart','teacher.pt','--separate-critic'])
    assert selected.warmstart=='teacher.pt' and selected.separate_critic is True
    assert cli.parse_args(['evaluate','--checkpoint','frozen.pt']).num_envs == 1
    with pytest.raises(ValueError, match='num-envs 1'):
        cli.evaluate(cli.parse_args(['evaluate','--checkpoint','frozen.pt','--num-envs','2']))
    with pytest.raises(ValueError, match='steps must be positive'):
        cli.evaluate(cli.parse_args(['evaluate','--checkpoint','frozen.pt','--steps','0']))


def test_frozen_separate_critic_architecture_is_selected_and_strictly_loaded(tmp_path):
    import sim.manorl.autonomy_batch_training as training
    from tools import train_manorl_autonomy as cli
    source=_model(separate_critic=True); path=tmp_path/'separate.pt'
    torch.save(_v4_payload(source,{'package_digest':'p'}),path)
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert cli._checkpoint_separate_critic(payload) is True
    target=_model(separate_critic=True)
    training.load_frozen_v4(path,target,expected_provenance={'package_digest':'p'})
    for name,value in source.state_dict().items(): assert torch.equal(value,target.state_dict()[name])
    with pytest.raises(ValueError,match='architecture'):
        training.load_frozen_v4(path,_model(),expected_provenance={'package_digest':'p'})
