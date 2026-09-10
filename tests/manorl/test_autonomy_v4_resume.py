"""Strict optimizer continuation at an explicit fresh-episode boundary; no physics/network."""
from __future__ import annotations

import copy
import random
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

import sim.manorl.autonomy_batch_training as training
from sim.manorl.autonomy_training import AutonomyActorCritic, seed_everything
from tools import train_manorl_autonomy as cli


class Adapter:
    num_envs = 1
    device = torch.device('cpu')
    observation_dim = 957
    action_dim = 28
    observation_space = gym.spaces.Box(-1., 1., shape=(957,))
    action_space = gym.spaces.Box(-1., 1., shape=(28,))

    def __init__(self):
        self.actions = []
        self.runtime = SimpleNamespace(cache=SimpleNamespace(control_timestep=1/120))

    def reset(self):
        # Construction/reset must not shift checkpoint sampling RNG on resume.
        random.random(); np.random.rand(); torch.rand(7)
        return torch.zeros(1, 957), {}

    def step(self, actions):
        self.actions.append(actions.detach().clone())
        return torch.ones(1, 957), torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool), {'valid': torch.tensor(True)}

    def prepare_action(self): return torch.zeros(1, 957)
    def _to_torch(self, x): return x
    def compact_summary(self): return {k: torch.tensor(0.) for k in ('object_motion', 'contact_force', 'path')}


def provenance():
    return dict(asset_pin='a', package_digest='p', manifest_sha256='m', catalog_digest='c',
                identity_split={'seed': 0, 'train_indices': [0]}, contracts={'checkpoint': training.CHECKPOINT_FORMAT},
                identity='cube2_02_2833', clock={'control_timestep': 1/120, 'physics_timestep': 1/480, 'physics_substeps': 4},
                source_commit='old', cache_hash_recorded_not_compared='cache', separate_critic=True,
                warmstart_checkpoint='original-teacher.pt', warmstart_transfer_mode='shared_policy_to_separate_critic')


def run(adapter, **kwargs):
    return training.run_batched_ppo(adapter, rollouts=2, learning_epochs=1, mini_batches=1,
                                    learning_rate=3e-5, separate_critic=True, provenance=provenance(), **kwargs)


@pytest.fixture
def source(tmp_path):
    seed_everything(0)
    path = tmp_path / 'source.pt'
    model, agent, _ = run(Adapter(), updates=2, checkpoint=path, checkpoint_interval=1)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return path, payload, model, agent


def inspect(path, payload, model, agent, **kwargs):
    return training.inspect_v4_resume(path, model, agent.optimizer,
        expected_config=kwargs.pop('config', payload['config']),
        expected_provenance=kwargs.pop('provenance', payload['provenance']),
        updates=kwargs.pop('updates', 4), **kwargs)


def test_resume_matches_uninterrupted_next_sample_gradient_and_counters(tmp_path, monkeypatch):
    original = training.build_batched_runtime
    traces = []
    def builder(*args, **kwargs):
        model, agent = original(*args, **kwargs)
        trace = []; traces.append(trace)
        act, record, update = agent.act, agent.record_transition, agent.update
        def acting(*a, **kw):
            trace.append(('act', kw['timestep'], kw['timesteps'], random.random(), np.random.rand(), torch.rand(1).item(),
                          [s['step'].item() for s in agent.optimizer.state.values()]))
            return act(*a, **kw)
        def recording(**kw):
            trace.append(('record', kw['timestep'], kw['timesteps']))
            return record(**kw)
        def updating(**kw):
            trace.append(('update', kw['timestep'], kw['timesteps']))
            return update(**kw)
        agent.act, agent.record_transition, agent.update = acting, recording, updating
        return model, agent
    monkeypatch.setattr(training, 'build_batched_runtime', builder)
    seed_everything(0)
    full_adapter = Adapter()
    def reset_boundary(row):
        if row['update'] == 2:
            state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            full_adapter.reset()
            random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    full_model, full_agent, full_rows = run(full_adapter, updates=4, on_update=reset_boundary)
    seed_everything(0)
    first = Adapter(); path = tmp_path / 'source.pt'
    run(first, updates=2, checkpoint=path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    # Persisted warm-start metadata is prior lineage, not an instruction to reload it.
    saved['provenance']['warmstart_checkpoint'] = 'original-teacher.pt'
    saved['provenance']['warmstart_transfer_mode'] = 'shared_policy_to_separate_critic'
    torch.save(saved, path)
    resumed_adapter = Adapter(); output = tmp_path / 'continued.pt'
    model, agent, rows = run(resumed_adapter, updates=4, resume_checkpoint=path,
                             checkpoint=output, checkpoint_interval=1)
    assert [r['update'] for r in rows] == [3, 4]
    assert [r['transitions'] for r in rows] == [6, 8]
    assert rows[0]['episodes/length_mean'] == 1
    assert traces[2][0][1:3] == (4, 8)
    assert traces[2][0][3:6] == traces[0][10][3:6]  # python, numpy, Torch next sample
    assert set(traces[2][0][6]) == {2.}
    assert traces[2] == traces[0][10:]
    for actual, expected in zip(resumed_adapter.actions, full_adapter.actions[4:]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for k, v in model.state_dict().items(): torch.testing.assert_close(v, full_model.state_dict()[k], rtol=0, atol=0)
    for p, q in zip(agent.optimizer.state.values(), full_agent.optimizer.state.values()):
        for k in p: torch.testing.assert_close(p[k], q[k], rtol=0, atol=0)
    final = torch.load(output, map_location='cpu', weights_only=False)
    assert final['policy_steps'] == 8 and final['environment_transitions'] == 8
    assert final['config']['updates'] == 4
    assert final['provenance']['warmstart_checkpoint'] == 'original-teacher.pt'
    lineage = final['provenance']['resume']
    assert lineage['previous_updates'] == 2 and lineage['physics_restart'] == 'full_start_new_episodes'
    assert len(lineage['sha256']) == 64
    assert (tmp_path / 'continued.update000003.pt').exists()
    assert not (tmp_path / 'continued.update000001.pt').exists()
    # Repeated continuation accepts its own persisted lineage.
    run(Adapter(), updates=5, resume_checkpoint=output)


@pytest.mark.parametrize('kind', ['teacher_none', 'teacher_partial', 'missing_adam', 'adam_shape', 'model_shape',
    'model_nan', 'adam_inf', 'abi', 'architecture', 'normalizer', 'steps_float', 'steps_partial', 'steps_zero',
    'transitions', 'rng', 'cuda_count', 'lr', 'groups', 'sampling'])
def test_invalid_checkpoints_rejected(source, tmp_path, kind):
    _, original, model, agent = source
    p = copy.deepcopy(original); options = {}
    if kind == 'teacher_none': p['optimizer'] = None
    elif kind == 'teacher_partial': p['optimizer']['param_groups'][0]['params'] = list(range(22))
    elif kind == 'missing_adam': p['optimizer']['state'].pop(0)
    elif kind == 'adam_shape': p['optimizer']['state'][0]['exp_avg'] = torch.zeros(27)
    elif kind == 'model_shape': p['model']['log_std'] = torch.zeros(27)
    elif kind == 'model_nan': p['model']['log_std'][0] = float('nan')
    elif kind == 'adam_inf': p['optimizer']['state'][0]['exp_avg'][0] = float('inf')
    elif kind == 'abi': p['observation_contract'] = 'v3'
    elif kind == 'architecture': p['model_architecture']['value_trunk'] = 'shared'
    elif kind == 'normalizer': p['normalizer'] = {}
    elif kind == 'steps_float': p['policy_steps'] = 4.
    elif kind == 'steps_partial': p['policy_steps'] = 3
    elif kind == 'steps_zero': p['policy_steps'] = 0
    elif kind == 'transitions': p['environment_transitions'] = 999
    elif kind == 'rng': p['torch_rng'] = torch.zeros(2)
    elif kind == 'cuda_count': p['cuda_rng'] = [torch.zeros(16, dtype=torch.uint8)]; options['cuda_device_count'] = 2
    elif kind == 'lr': p['optimizer']['param_groups'][0]['lr'] = 3e-4
    elif kind == 'groups': p['optimizer']['param_groups'].append(p['optimizer']['param_groups'][0])
    elif kind == 'sampling': p.pop('policy_sampling_contract')
    path = tmp_path / 'bad.pt'; torch.save(p, path)
    with pytest.raises(ValueError): inspect(path, original, model, agent, **options)


@pytest.mark.parametrize('key,value', [('num_envs', 2), ('rollouts', 4), ('learning_epochs', 2), ('mini_batches', 2),
    ('learning_rate', 3e-4), ('separate_critic', False), ('seed', 9), ('device', 'gpu'), ('unknown_future_knob', True)])
def test_config_drift_fails_closed(source, key, value):
    path, payload, model, agent = source
    with pytest.raises(ValueError, match='config mismatch'):
        inspect(path, payload, model, agent, config={**payload['config'], key: value})


def test_recorded_critic_lineage_may_be_absent_from_expected_provenance(source):
    path, payload, model, agent = source
    expected_provenance = payload['provenance'].copy()
    expected_provenance.pop('separate_critic')
    inspect(path, payload, model, agent, provenance=expected_provenance)


def test_recorded_critic_lineage_does_not_weaken_fixed_config_gate(source):
    path, payload, model, agent = source
    expected_config = payload['config'].copy()
    expected_config['separate_critic'] = False
    with pytest.raises(ValueError, match='config mismatch'):
        inspect(path, payload, model, agent, config=expected_config)


@pytest.mark.parametrize('key', ['asset_pin', 'package_digest', 'manifest_sha256', 'catalog_digest', 'identity_split', 'contracts', 'identity', 'clock'])
def test_physical_provenance_drift_rejected(source, key):
    path, payload, model, agent = source
    with pytest.raises(ValueError, match='provenance'):
        inspect(path, payload, model, agent, provenance={**payload['provenance'], key: 'different'})


def test_budget_conflict_and_noop_resume(source):
    path, payload, model, agent = source
    for target in [1, 2]:
        with pytest.raises(ValueError, match='greater than completed'): inspect(path, payload, model, agent, updates=target)
    with pytest.raises(SystemExit): cli.parse_args(['train', '--warmstart', 'a', '--resume-checkpoint', 'b'])
    with pytest.raises(ValueError, match='mutually exclusive'):
        run(Adapter(), updates=4, warmstart=path, resume_checkpoint=path)


def test_cli_wandb_resume_must_and_cumulative_history(source, tmp_path, monkeypatch, capsys):
    path, payload, _, _ = source
    adapter = Adapter(); trajectory = object(); catalog = SimpleNamespace(trajectories=[trajectory])
    args = cli.parse_args(['train', '--resume-checkpoint', str(path), '--wandb-run-id', 'old-id', '--device', 'cpu',
        '--num-envs', '1', '--updates', '4', '--rollouts', '2', '--learning-epochs', '1', '--mini-batches', '1',
        '--learning-rate', '3e-5', '--separate-critic', '--checkpoint', str(tmp_path / 'out.pt')])
    # Match the CLI's full source config, as production checkpoints do.
    payload['config'].update({k: v for k, v in vars(args).items() if k not in {'fn', 'wandb'}})
    payload['config']['updates'] = 2
    payload['config']['resume_checkpoint'] = None
    payload['config']['wandb_run_id'] = 'old-id'
    payload['provenance']['cache_hash_recorded_not_compared'] = 'old-cache'
    torch.save(payload, path)
    monkeypatch.setattr(cli, '_catalog_and_trajectory', lambda a: (catalog, trajectory))
    monkeypatch.setattr(cli, 'identity_split', lambda *a, **k: {'train_indices': [0]})
    monkeypatch.setattr(cli, '_adapter', lambda *a: adapter)
    monkeypatch.setattr(cli, '_provenance', lambda *a: payload['provenance'].copy())
    seen = {}
    class Config:
        def update(self, data, **kw): seen['metadata'] = copy.deepcopy(data); seen['config_kw'] = kw
    class Run:
        id = 'old-id'
        config = Config()
        def define_metric(self, *a, **kw): pass
        def log(self, row, step): seen.setdefault('history', []).append((row['update'], step))
        def finish(self, **kw): seen['finish'] = kw
    def init(**kw): seen['init'] = kw; return Run()
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init))
    cli.train(args)
    assert seen['init']['id'] == 'old-id' and seen['init']['resume'] == 'must'
    assert seen['init']['config'] is None and seen['config_kw'] == {'allow_val_change': True}
    assert seen['metadata']['config']['updates'] == 4
    assert seen['metadata']['provenance']['resume']['source_config']['updates'] == 2
    assert seen['history'] == [(3., 6), (4., 8)]
    assert '"start_update": 2' in capsys.readouterr().out
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert saved['config']['wandb_run_id'] == 'old-id'
    seen.clear(); args.learning_rate = 3e-4
    with pytest.raises(ValueError, match='config mismatch'): cli.train(args)
    assert not seen  # invalid checkpoint/config cannot create a writer
    args.learning_rate = 3e-5; args.wandb_run_id = None
    monkeypatch.delenv('WANDB_RUN_ID', raising=False)
    with pytest.raises(ValueError, match='explicit'): cli.train(args)
    assert not seen
    args.wandb_run_id = 'other-id'
    with pytest.raises(ValueError, match='run ID differs'): cli.train(args)
    assert not seen


def test_wandb_environment_id_and_offline_rejection(monkeypatch):
    args = cli.parse_args(['train', '--resume-checkpoint', 'source.pt'])
    monkeypatch.setenv('WANDB_RUN_ID', 'legacy-id')
    seen = {}
    run = SimpleNamespace(id='legacy-id', config=SimpleNamespace(update=lambda *a, **kw: None),
                          define_metric=lambda *a, **kw: None, finish=lambda **kw: None)
    def init(**kw): seen.update(kw); return run
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init))
    cli._wandb(args, {'config': {}})
    assert seen['id'] == 'legacy-id' and seen['resume'] == 'must'
    args.wandb_mode = 'offline'
    with pytest.raises(ValueError, match='online'): cli._wandb(args, {'config': {}})
