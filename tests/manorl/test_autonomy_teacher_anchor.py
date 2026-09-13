"""Training-only teacher labels and post-PPO mean supervision."""
from types import SimpleNamespace
import jax.numpy as jp
import numpy as np
import pytest
import torch
from sim.manorl import autonomy_batch_training as training
from sim.manorl.autonomy_training import BatchedAutonomyAdapter, TEACHER_FLEX_JOINTS, teacher_anchor_metadata, seed_everything
from tests.manorl.test_autonomy_v4_integration import _TinyV4Adapter, _model


def test_teacher_labels_current_command_gate_limits_and_last_frame():
    adapter = object.__new__(BatchedAutonomyAdapter)
    adapter.device_name = 'cpu'; adapter._device = torch.device('cpu')
    q = np.zeros((202, 28), np.float32); q[:, 0] = np.arange(202) / 1000
    previous = np.full((4, 28), -.02, np.float32)
    adapter.runtime = SimpleNamespace(jp=jp, indices=jp.array([198,199,200,201]), length=202,
        cache=SimpleNamespace(q_feasible=jp.asarray(q), control_timestep=.1),
        previous_command=jp.asarray(previous), lower=jp.full(28,-.1), upper=jp.full(28,.15), rate=jp.full(28,2.))
    before = np.asarray(adapter.runtime.previous_command).copy()
    actual = adapter.teacher_actions()
    target = q[[199,200,201,201]].copy()
    target[1:, list(TEACHER_FLEX_JOINTS)] += .2
    expected = np.clip((np.clip(target,-.1,.15)-previous)/.2,-1,1)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert actual.shape == (4,28)
    np.testing.assert_array_equal(adapter.runtime.previous_command, before)
    adapter.runtime.previous_command = jp.full((4,28), .6)
    assert (adapter.teacher_actions() == -1).all()


def test_anchor_mean_gradient_excludes_value_and_logstd_with_adam_momentum():
    seed_everything(0); model = _model(separate_critic=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs = torch.randn(11,957)
    # Establish Adam momentum for every parameter, as in a preceding PPO update.
    sum(p.square().sum() for p in model.parameters()).backward(); optimizer.step()
    before = {k:v.detach().clone() for k,v in model.named_parameters()}
    with torch.no_grad():
        target = model.compute({'observations':obs},role='policy')[0] + .5
        initial = ((model.compute({'observations':obs},role='policy')[0]-target)**2).mean()
    metrics = training._apply_teacher_anchor(model, optimizer, [(obs[:5],target[:5]),(obs[5:],target[5:])], beta=1., passes=2, mini_batches=3)
    with torch.no_grad(): final = ((model.compute({'observations':obs},role='policy')[0]-target)**2).mean()
    assert final < initial
    for name, p in model.named_parameters():
        if name.startswith(('value.', 'value_net.')) or name == 'log_std':
            assert p.grad is None and torch.equal(p, before[name])
    for prefix in ('pointnet.', 'net.', 'mean.'):
        assert any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if n.startswith(prefix))
    assert metrics['teacher_anchor/samples'] == 11
    assert metrics['teacher_anchor/optimizer_steps'] == 6


class LabelAdapter(_TinyV4Adapter):
    def __init__(self): super().__init__(); self.labels = []
    def teacher_actions(self):
        self.labels.append((len(self.seen), self.phase))
        return torch.full((1,28), .25)


def test_disabled_is_exact_default_without_teacher_calls_or_extra_steps(monkeypatch):
    def forbidden(*a, **k): pytest.fail('disabled anchor called teacher or optimizer helper')
    monkeypatch.setattr(training, '_apply_teacher_anchor', forbidden)
    results=[]
    for kwargs in ({}, {'teacher_anchor_beta':0., 'teacher_anchor_passes':0}):
        seed_everything(3); adapter=LabelAdapter(); adapter.teacher_actions=forbidden
        model, agent, _ = training.run_batched_ppo(adapter, updates=2, rollouts=2, learning_epochs=1, mini_batches=1, **kwargs)
        results.append(({n:p.clone() for n,p in model.state_dict().items()}, adapter.seen, agent.optimizer.state_dict()))
    for name in results[0][0]: assert torch.equal(results[0][0][name], results[1][0][name])
    for a,b in zip(results[0][1],results[1][1]): assert torch.equal(a,b)
    for state in results[1][2]['state'].values(): assert state['step'] == 2


def test_rollout_pairs_pre_step_and_metadata(tmp_path, monkeypatch):
    adapter=LabelAdapter(); observed=[]; apply=training._apply_teacher_anchor
    def capture(model, optimizer, pairs, **kwargs):
        observed.append([(obs.clone(), target.clone()) for obs,target in pairs])
        return apply(model,optimizer,pairs,**kwargs)
    monkeypatch.setattr(training,'_apply_teacher_anchor',capture)
    checkpoint=tmp_path/'anchor.pt'
    _, _, rows = training.run_batched_ppo(adapter, updates=2, rollouts=3, learning_epochs=1, mini_batches=2,
        teacher_anchor_beta=1., checkpoint=checkpoint, separate_critic=True)
    assert adapter.labels == [(i,0) for i in range(6)]
    assert len(observed)==2 and all(len(pairs)==3 for pairs in observed)
    for pairs in observed:
        for obs,target in pairs:
            assert torch.equal(obs,torch.zeros(1,957)) and torch.equal(target,torch.full((1,28),.25))
    saved=torch.load(checkpoint,weights_only=False)
    assert saved['config']['teacher_anchor_beta']==1.
    assert saved['config']['teacher_anchor_passes']==2
    assert saved['config']['teacher_anchor']==saved['provenance']['teacher_anchor']==teacher_anchor_metadata(1.,2)
    assert all(row['teacher_anchor/samples']==3 and row['teacher_anchor/rollout_steps']==3 for row in rows)
    assert rows[0]['config/teacher_squeeze_start']==200


@pytest.mark.parametrize('beta,passes', [(-1.,2),(float('nan'),2),(float('inf'),2),(1.,0),(1.,1.5)])
def test_invalid_anchor_fails_before_runtime(beta,passes):
    with pytest.raises(ValueError,match='teacher-anchor'):
        training.run_batched_ppo(None,updates=1,rollouts=2,learning_epochs=1,mini_batches=1,
                                 teacher_anchor_beta=beta,teacher_anchor_passes=passes)


def test_cli_anchor_defaults_and_options():
    from tools import train_manorl_autonomy as cli
    args=cli.parse_args(['train'])
    assert (args.teacher_anchor_beta,args.teacher_anchor_passes)==(0.,2)
    args=cli.parse_args(['train','--teacher-anchor-beta','1','--teacher-anchor-passes','3'])
    assert (args.teacher_anchor_beta,args.teacher_anchor_passes)==(1.,3)
    with pytest.raises(ValueError,match='teacher-anchor-beta'):
        cli.train(cli.parse_args(['train','--teacher-anchor-beta=-1']))


def test_pre_anchor_checkpoint_resume_defaults_and_fixed_recipe(tmp_path):
    from tests.manorl.test_autonomy_v4_resume import Adapter, run, provenance
    source=tmp_path/'old.pt'
    run(Adapter(),updates=1,checkpoint=source)
    payload=torch.load(source,weights_only=False)
    for key in ('teacher_anchor_beta','teacher_anchor_passes','teacher_anchor'):
        payload['config'].pop(key)
    payload['provenance'].pop('teacher_anchor')
    torch.save(payload,source)
    # An actual old-format checkpoint with no anchor fields continues disabled.
    _,_,rows=run(Adapter(),updates=2,resume_checkpoint=source)
    assert rows[0]['config/teacher_anchor_beta']==0.
    with pytest.raises(ValueError,match='config mismatch'):
        run(Adapter(),updates=2,resume_checkpoint=source,teacher_anchor_beta=1.)


def test_enabled_anchor_optimizer_resume(tmp_path):
    from tests.manorl.test_autonomy_v4_resume import provenance
    kwargs=dict(rollouts=2,learning_epochs=1,mini_batches=1,teacher_anchor_beta=1.,
                separate_critic=True,provenance=provenance())
    seed_everything(6)
    full,_,_=training.run_batched_ppo(LabelAdapter(),updates=3,**kwargs)
    seed_everything(6); source=tmp_path/'enabled.pt'
    training.run_batched_ppo(LabelAdapter(),updates=1,checkpoint=source,**kwargs)
    resumed,_,_=training.run_batched_ppo(LabelAdapter(),updates=3,resume_checkpoint=source,**kwargs)
    for k,v in full.state_dict().items(): assert torch.equal(v,resumed.state_dict()[k])
