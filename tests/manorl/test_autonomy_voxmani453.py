"""453 ABI, learned routing, measured history and PPO snapshot regression."""

from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import jax
import jax.numpy as j
import numpy as np
import pytest
import torch

from sim.manorl.autonomy_contracts import (
    RAW_OBSERVATION_FIELDS_VOXMANI453,
    V11_REWARD_CONTRACT_ID,
    raw_observation_slices_voxmani453,
)
from sim.manorl.autonomy_v4 import ReferenceBankV4
from sim.manorl.autonomy_v5_voxmani453_model import AutonomyActorCriticVoxMani453
from sim.manorl.voxmani453.canonical import build_canonical
from sim.manorl.voxmani453.goal_stream import GoalStream, empty_streams
from sim.manorl.voxmani453.observation import (
    build_raw_observation_voxmani453,
    update_history,
)
from sim.manorl.voxmani453.streams import (
    GOAL_SLOTS,
    STATE_DIM,
    STREAM_SPECS,
    pack_streams,
    reference_states,
    unpack_streams,
)
from tests.manorl.test_autonomy_v4 import _cache, _contact, _state


def cache(frames=25):
    poses = np.zeros((frames, 16, 9))
    poses[..., 3] = poses[..., 7] = 1
    origin = np.zeros((frames, 3))
    origin[:, 0] = np.arange(frames) * 0.001
    return replace(
        _cache(frames),
        object_mass=0.2,
        gravity_world=np.array([0.0, 0.0, -9.81]),
        object_origin=origin,
        hand_region_pose_reference=poses,
    )


def observation(batch=2):
    return torch.tensor(
        np.asarray(
            build_raw_observation_voxmani453(
                _state(batch),
                _contact(batch)._replace(object_all_torque=j.zeros((batch, 3))),
                cache(),
                j.zeros(batch, j.int32),
                j.zeros((batch, 28)),
            )
        )
    )


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(453)
    return AutonomyActorCriticVoxMani453(
        gym.spaces.Box(-np.inf, np.inf, (453,), dtype=np.float32),
        gym.spaces.Box(-1.0, 1.0, (28,), dtype=np.float32),
        canonical=build_canonical(),
    )


def streams(batch=2):
    return {
        name: GoalStream.pack(
            [
                {
                    "raw": torch.randn(3, spec.raw_dim),
                    "dt": [0.0, 0.05, 0.1],
                    "frame_id": [10, 16, 22],
                    "type_id": [kind] * 3,
                }
                for _ in range(batch)
            ],
            raw_dim=spec.raw_dim,
            length=GOAL_SLOTS,
        )
        for kind, (name, spec) in zip((0, 2), STREAM_SPECS.items())
    }


def test_exact_abi_and_force_sign():
    assert sum(width for _, width in RAW_OBSERVATION_FIELDS_VOXMANI453) == 453
    slices = raw_observation_slices_voxmani453()
    contact = _contact(2)._replace(
        object_all_torque=j.zeros((2, 3)),
        paired_force_on_object=j.zeros((2, 16, 3)).at[:, 3, 2].set(1.962),
        object_all_force=j.array([[0.0, 0.0, 1.962]] * 2),
    )
    raw = np.asarray(
        build_raw_observation_voxmani453(
            _state(2),
            contact,
            cache(),
            j.array([0, 24]),
            j.zeros((2, 28)),
        )
    )
    assert raw.shape == (2, 453) and np.isfinite(raw).all()
    force = raw[:, slices["obj_force_obs"]]
    np.testing.assert_allclose(force[:, :3], [[0, 0, -1]] * 2)
    np.testing.assert_allclose(force[:, 3:51].reshape(2, 16, 3)[:, 3, 2], 1.0)
    np.testing.assert_allclose(
        raw[:, slices["bone_contact"]].reshape(2, 16, 3)[:, 3, 2],
        -np.tanh(0.1962),
        atol=1e-7,
    )


def test_measured_history_lags_and_no_advance_on_prepare():
    hist, count = j.zeros((2, 17, 7)), j.zeros(2, j.int32)
    fn = jax.jit(update_history)
    for step in range(18):
        current = j.full((2, 7), step, j.float32)
        hist, count, past, pad = fn(hist, count, current, step > 0)
    np.testing.assert_array_equal(past[0, :, 0], [16, 15, 13, 9, 1])
    assert not np.asarray(pad).any()
    before = np.asarray(hist).copy()
    hist, count, _, _ = fn(hist, count, current, False)
    np.testing.assert_array_equal(hist, before)
    np.testing.assert_array_equal(count, [16, 16])
    # Only reset row 1 loses history. Its unfilled past is its measured current pose.
    hist = hist.at[1].set(0)
    count = count.at[1].set(0)
    hist, count, past, pad = fn(hist, count, current, False)
    assert np.asarray(pad[1]).all() and not np.asarray(pad[0]).any()
    np.testing.assert_array_equal(past[1], np.full((5, 7), 17))


def test_reference_bank_streams_use_per_world_time_and_endpoint():
    bank = ReferenceBankV4([cache(25), cache(13)])
    runtime = SimpleNamespace(
        cache=bank, env_ref=j.array([0, 1]), indices=j.array([0, 11]), num_envs=2
    )
    fn = jax.jit(lambda index: reference_states(runtime, index))
    states = torch.tensor(np.asarray(fn(runtime.indices)))
    assert states.shape == (2, STATE_DIM) == (2, 547)
    decoded = unpack_streams(states)
    for stream in decoded.values():
        stream.validate()
        assert stream.lengths.tolist() == [4, 2]
        torch.testing.assert_close(stream.dt[1, :2], torch.tensor([0.0, 1 / 120]))
        assert stream.frame_id[1, :2].tolist() == [11, 12]
    changed = torch.tensor(np.asarray(fn(j.array([1, 12]))))
    assert not torch.equal(states, changed)
    torch.testing.assert_close(pack_streams(decoded), states, rtol=0, atol=0)
    # One-reference bank and ordinary cache give identical records.
    one = SimpleNamespace(cache=cache(), env_ref=None, indices=j.array([4]), num_envs=1)
    bank_one = SimpleNamespace(
        cache=ReferenceBankV4([one.cache]),
        env_ref=j.array([0]),
        indices=one.indices,
        num_envs=1,
    )
    np.testing.assert_array_equal(reference_states(one), reference_states(bank_one))


def test_actor_privilege_isolation_and_independent_learned_towers(model):
    x = observation()
    changed = x.clone()
    for name in ("priv_13", "critic_keypoint_error"):
        changed[:, raw_observation_slices_voxmani453()[name]] += 10
    with torch.no_grad():
        actor_a = model.compute({"observations": x}, role="policy")[0]
        actor_b = model.compute({"observations": changed}, role="policy")[0]
        critic_a = model.compute({"observations": x}, role="value")[0]
        critic_b = model.compute({"observations": changed}, role="value")[0]
        enc = model.encoder.encode(x)
        tokens = model.actor.current_tokens(enc["actor"])[0]
    torch.testing.assert_close(actor_a, actor_b, rtol=0, atol=0)
    assert not torch.allclose(critic_a, critic_b)
    assert enc["actor"]["x_low"].shape == (2, 314)
    assert enc["critic"]["x_low"].shape == (2, 375)
    assert tokens.shape == (2, 70, 128)
    assert not list(model.encoder.parameters())
    assert not (
        {id(p) for p in model.actor.parameters()}
        & {id(p) for p in model.critic.parameters()}
    )


def test_goal_snapshot_padding_provenance_and_permutation(model):
    x, goals = observation(), streams()
    snapshot = pack_streams(goals)
    with torch.no_grad():
        baseline = model.compute(
            {"observations": x, "goal_streams": goals}, role="policy"
        )[0]
        replay = model.compute({"observations": x, "states": snapshot}, role="policy")[
            0
        ]
        changed = {}
        for name, stream in goals.items():
            payload = stream.raw.clone()
            payload[:, 3:] = torch.nan
            changed[name] = replace(
                stream, raw=payload, frame_id=stream.frame_id + 1000
            )
        masked = model.compute(
            {"observations": x, "goal_streams": changed}, role="policy"
        )[0]
        permuted = {}
        order = torch.tensor([2, 0, 1, 3, 4])
        for name, stream in goals.items():
            permuted[name] = replace(
                stream,
                **{
                    field: getattr(stream, field)[:, order]
                    for field in ("raw", "mask", "dt", "frame_id", "type_id")
                },
            )
        shuffled = model.compute(
            {"observations": x, "goal_streams": permuted}, role="policy"
        )[0]
        no_goals = model.compute({"observations": x}, role="policy")[0]
    torch.testing.assert_close(baseline, replay, rtol=0, atol=0)
    torch.testing.assert_close(baseline, masked, rtol=0, atol=0)
    torch.testing.assert_close(baseline, shuffled, rtol=1e-5, atol=1e-7)
    assert not torch.allclose(baseline, no_goals)
    assert torch.isfinite(snapshot).all()


def test_empty_and_present_goals_backward(model):
    x = observation()
    for goals in (empty_streams(2, STREAM_SPECS), streams()):
        model.zero_grad(set_to_none=True)
        actor = model.compute(
            {"observations": x, "goal_streams": goals}, role="policy"
        )[0]
        value = model.compute({"observations": x, "goal_streams": goals}, role="value")[
            0
        ]
        (actor.square().mean() + value.square().mean()).backward()
        assert actor.shape == (2, 28) and value.shape == (2, 1)
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )


def test_real_ppo_update_stores_goal_snapshots_and_keeps_raw_actions(model):
    from skrl.memories.torch import RandomMemory
    from sim.manorl.rl_games_ppo import RlGamesPPO
    from sim.manorl.autonomy_training import v4_ppo_config

    cfg = v4_ppo_config(
        rollouts=2, learning_epochs=1, mini_batches=1, learning_rate=1e-4
    )
    agent = RlGamesPPO(
        models={"policy": model, "value": model},
        memory=RandomMemory(memory_size=2, num_envs=2, device="cpu"),
        observation_space=model.observation_space,
        state_space=gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), dtype=np.float32),
        action_space=model.action_space,
        device="cpu",
        cfg=cfg,
    )
    agent.init()
    agent.enable_training_mode(True, apply_to_models=True)
    x = observation()
    saved = []
    before = model.actor.wrist_head[-1].weight.detach().clone()
    for step in range(2):
        states = pack_streams(streams())
        saved.append(states.clone())
        with torch.no_grad():
            actions, _ = agent.act(x, states, timestep=step, timesteps=2)
        agent.record_transition(
            observations=x,
            states=states,
            actions=actions,
            rewards=torch.ones(2, 1),
            next_observations=x,
            next_states=states,
            terminated=torch.zeros(2, 1, dtype=torch.bool),
            truncated=torch.zeros(2, 1, dtype=torch.bool),
            infos={},
            timestep=step,
            timesteps=2,
        )
    torch.testing.assert_close(
        agent.memory.get_tensor_by_name("states"), torch.stack(saved)
    )
    agent.update(timestep=2, timesteps=2)
    assert not torch.equal(before, model.actor.wrist_head[-1].weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # Strict resume requires Adam moments for every model parameter. In
    # particular, the critic must not register an unused final theta projection.
    assert set(agent.optimizer.state) == set(model.parameters())


def test_packed_telemetry_reduces_per_bone_torque_to_one_world_scalar():
    from sim.manorl.autonomy_training import BatchedAutonomyAdapter, _telemetry_slices
    from sim.manorl.autonomy_reward_v11 import compute_reward
    from sim.manorl.autonomy_contracts import V11_REWARD_TERM_NAMES

    physical, contacts, reference = _state(2), _contact(2), cache()
    contacts = contacts._replace(
        paired_torque_com=j.zeros((2, 16, 3)).at[:, 0, 2].set(2.0).at[:, 1, 2].set(-0.5)
    )
    index = j.zeros(2, j.int32)
    reward = compute_reward(physical, contacts, reference, index, j.zeros((2, 28)))
    adapter = object.__new__(BatchedAutonomyAdapter)
    adapter.runtime = SimpleNamespace(jp=j, cache=reference, env_ref=None)
    adapter.reward_names = V11_REWARD_TERM_NAMES
    packed = adapter._pack_device_telemetry(physical, contacts, reward, index)
    slices, width = _telemetry_slices(adapter.reward_names)
    assert packed.shape == (2, width)
    np.testing.assert_allclose(packed[:, slices["paired_torque_com_norm"]], 1.5)


@pytest.mark.parametrize("spelling", ["v8", "v5.voxmani453"])
def test_cli_and_checkpoint_contract(spelling):
    from tools.train_manorl_autonomy import (
        parse_args,
        _checkpoint_policy_version,
        _contracts,
    )

    args = parse_args(
        [
            "train",
            "--policy-version",
            spelling,
            "--reward-version",
            "v11",
            "--no-wandb",
        ]
    )
    assert args.policy_version == "v8"
    contracts = _contracts(args.policy_version, V11_REWARD_CONTRACT_ID)
    payload = {
        f"{name}_contract": value
        for name, value in contracts.items()
        if name != "checkpoint"
    }
    payload["checkpoint_format"] = contracts["checkpoint"]
    assert _checkpoint_policy_version(payload) == args.policy_version


def test_v8_factory_keeps_existing_weight_and_checkpoint_layout(model):
    from sim.manorl.autonomy_training import (
        actor_critic_architecture_for_version,
        model_for_version,
    )

    renamed = model_for_version(
        "v8", model.observation_space, model.action_space, device="cpu",
        reward_contract_id=V11_REWARD_CONTRACT_ID,
    )
    assert renamed.architecture_name == "v8"
    assert type(renamed) is type(model)
    assert renamed.checkpoint_architecture() == model.checkpoint_architecture()
    assert (
        actor_critic_architecture_for_version("v5.voxmani453")
        == renamed.checkpoint_architecture()
    )
    assert list(renamed.state_dict()) == list(model.state_dict())
    # This must accept historical weight dictionaries without conversion.
    renamed.load_state_dict(model.state_dict(), strict=True)
