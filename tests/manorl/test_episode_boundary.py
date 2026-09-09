from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from sim.manorl.abi import (
    TERMINATION_REASON_DEVIATION,
    TERMINATION_REASON_NONE,
    TERMINATION_REASON_SUCCESS,
    TerminationResult,
    check_termination,
)


def test_termination_reason_masks_are_position_only_and_failure_wins() -> None:
    result = check_termination(
        object_position=np.array(
            [[0.0, 0.0, 0.0], [0.100001, 0.0, 0.0], [0.100001, 0.0, 0.0]]
        ),
        target_position=np.zeros((3, 3)),
        progress=np.array([0, 0, 9]),
        trajectory_lengths=np.array([10, 10, 10]),
        early_mask=np.zeros(3, dtype=bool),
    )

    np.testing.assert_array_equal(result.reset, [False, True, True])
    np.testing.assert_array_equal(result.success, [False, False, False])
    np.testing.assert_array_equal(result.failure, [False, True, True])
    np.testing.assert_array_equal(
        result.reason_code,
        [TERMINATION_REASON_NONE, TERMINATION_REASON_DEVIATION, TERMINATION_REASON_DEVIATION],
    )
    assert TERMINATION_REASON_SUCCESS == 1


def test_gym_adapter_marks_success_and_failure_terminated_without_truncation() -> None:
    from sim.manorl.gymnasium_env import ACTION_DIM, ManoGymnasiumVectorEnv, OBSERVATION_DIM

    termination = TerminationResult(
        reset=np.array([True, True]),
        deviation_reset=np.array([False, True]),
        deviation_penalty=np.zeros(2),
    )

    class Physical:
        last_termination = termination

        def step(self, actions):
            del actions
            return (
                {"obs": np.zeros((2, OBSERVATION_DIM), dtype=np.float64)},
                np.zeros(2, dtype=np.float64),
                termination.reset.copy(),
                {"time_outs": np.zeros(2, dtype=bool)},
            )

    adapter = object.__new__(ManoGymnasiumVectorEnv)
    adapter.num_envs = 2
    adapter.environment = Physical()
    _, _, terminated, truncated, info = adapter.step(np.zeros((2, ACTION_DIM)))

    np.testing.assert_array_equal(terminated, [True, True])
    np.testing.assert_array_equal(truncated, [False, False])
    np.testing.assert_array_equal(
        info["termination_reason_code"],
        [TERMINATION_REASON_SUCCESS, TERMINATION_REASON_DEVIATION],
    )
    np.testing.assert_array_equal(info["termination_success"], [True, False])
    np.testing.assert_array_equal(info["termination_failure"], [False, True])
    np.testing.assert_array_equal(info["trajectory_complete_reset_mask"], [True, False])
    np.testing.assert_array_equal(info["deviation_reset_mask"], [False, True])


def test_gym_adapter_consumes_pending_terminal_ids_before_direct_step() -> None:
    from sim.manorl.gymnasium_env import ACTION_DIM, ManoGymnasiumVectorEnv, OBSERVATION_DIM

    class Physical:
        def __init__(self) -> None:
            self.events: list[tuple[str, list[int] | None]] = []
            self.step_count = 0
            self.last_termination = TerminationResult(
                reset=np.zeros(2, dtype=bool),
                deviation_reset=np.zeros(2, dtype=bool),
                deviation_penalty=np.zeros(2, dtype=np.float64),
            )

        def reset(self, *, env_ids=None):
            ids = None if env_ids is None else np.asarray(env_ids, dtype=np.int64).tolist()
            self.events.append(("reset", ids))
            return {"obs": np.zeros((2, OBSERVATION_DIM), dtype=np.float64)}

        def step(self, actions):
            del actions
            self.step_count += 1
            self.events.append(("step", None))
            done = np.asarray(
                {
                    1: [True, False],
                    2: [False, False],
                    3: [True, True],
                    4: [False, False],
                    5: [True, False],
                    6: [False, False],
                }[self.step_count],
                dtype=bool,
            )
            deviation = done & np.array([False, True], dtype=bool)
            self.last_termination = TerminationResult(
                reset=done,
                deviation_reset=deviation,
                deviation_penalty=np.zeros(2, dtype=np.float64),
            )
            return (
                {"obs": np.zeros((2, OBSERVATION_DIM), dtype=np.float64)},
                np.zeros(2, dtype=np.float64),
                done,
                {"time_outs": np.zeros(2, dtype=bool)},
            )

    physical = Physical()
    adapter = object.__new__(ManoGymnasiumVectorEnv)
    adapter.num_envs = 2
    adapter.environment = physical
    adapter._last_seed = None
    adapter._pending_reset = np.zeros(2, dtype=bool)

    adapter.step(np.zeros((2, ACTION_DIM), dtype=np.float64))
    np.testing.assert_array_equal(adapter.pending_terminal_env_ids, [0])
    adapter.step(np.zeros((2, ACTION_DIM), dtype=np.float64))
    np.testing.assert_array_equal(adapter.pending_terminal_env_ids, [])
    assert physical.events[:3] == [("step", None), ("reset", [0]), ("step", None)]

    adapter.step(np.zeros((2, ACTION_DIM), dtype=np.float64))
    np.testing.assert_array_equal(adapter.pending_terminal_env_ids, [0, 1])
    adapter.reset(options={"env_ids": np.array([0], dtype=np.int64)})
    np.testing.assert_array_equal(adapter.pending_terminal_env_ids, [1])
    adapter.step(np.zeros((2, ACTION_DIM), dtype=np.float64))
    np.testing.assert_array_equal(adapter.pending_terminal_env_ids, [])
    assert physical.events[-2:] == [("reset", [1]), ("step", None)]

    adapter.step(np.zeros((2, ACTION_DIM), dtype=np.float64))
    np.testing.assert_array_equal(adapter.pending_terminal_env_ids, [0])
    adapter.reset()
    np.testing.assert_array_equal(adapter.pending_terminal_env_ids, [])
    adapter.step(np.zeros((2, ACTION_DIM), dtype=np.float64))
    assert physical.events[-2:] == [("reset", None), ("step", None)]


def test_gym_adapter_device_reset_clears_only_selected_pending_rows() -> None:
    pytest.importorskip("gymnasium")
    from sim.manorl.gymnasium_env import ManoGymnasiumVectorEnv, OBSERVATION_DIM

    class Physical:
        config = SimpleNamespace(device_transition=True)

        def __init__(self) -> None:
            self.device_reset_calls: list[list[int]] = []

        def reset(self, *, env_ids=None):
            raise AssertionError("device reset must not use the host reset contract")

        def device_reset(self, env_ids):
            ids = np.asarray(env_ids, dtype=np.int64)
            self.device_reset_calls.append(ids.tolist())
            return np.full((3, OBSERVATION_DIM), 7.0, dtype=np.float32)

    physical = Physical()
    adapter = object.__new__(ManoGymnasiumVectorEnv)
    adapter.num_envs = 3
    adapter.observation_dim = OBSERVATION_DIM
    adapter.environment = physical
    adapter._pending_reset = np.asarray([True, True, False], dtype=bool)

    observation = adapter.reset_done_device(np.asarray([True, False, True]))

    np.testing.assert_array_equal(observation, np.full((3, OBSERVATION_DIM), 7.0, dtype=np.float32))
    assert physical.device_reset_calls == [[0, 2]]
    np.testing.assert_array_equal(adapter.pending_terminal_env_ids, [1])


def test_runtime_device_reset_done_merges_only_terminal_rows() -> None:
    import torch

    pytest.importorskip("gymnasium")
    pytest.importorskip("skrl")
    from sim.manorl.skrl_runtime import ManoSkrlRuntime

    class DeviceWrapper:
        def __init__(self) -> None:
            self.masks: list[torch.Tensor] = []

        def reset_done_device(self, mask):
            self.masks.append(mask.clone())
            return torch.tensor([[5.0], [10.0], [7.0]])

        def reset_done(self, _mask):
            raise AssertionError("device-transition runtime must not call host reset_done")

    wrapper = DeviceWrapper()
    runtime = object.__new__(ManoSkrlRuntime)
    runtime.gymnasium_env = SimpleNamespace(
        num_envs=3, environment=SimpleNamespace(config=SimpleNamespace(device_transition=True))
    )
    runtime.env = wrapper
    observations = torch.tensor([[1.0], [2.0], [3.0]])

    updated = runtime.reset_done(observations, torch.tensor([[True], [False], [True]]))

    torch.testing.assert_close(updated, torch.tensor([[5.0], [2.0], [7.0]]))
    torch.testing.assert_close(wrapper.masks[0], torch.tensor([True, False, True]))


def test_indexed_wrapper_reset_bypasses_cache_and_preserves_seed() -> None:
    import gymnasium as gym
    import torch
    from gymnasium.vector.utils import batch_space

    from sim.manorl.skrl_runtime import ResettableGymnasiumWrapper

    class IndexedResetVectorEnv(gym.vector.VectorEnv):
        metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}

        def __init__(self) -> None:
            self.num_envs = 3
            self.single_observation_space = gym.spaces.Box(
                low=-10_000.0, high=10_000.0, shape=(1,), dtype=np.float32
            )
            self.single_action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
            self.action_space = batch_space(self.single_action_space, self.num_envs)
            self.device = "cpu"
            self.values = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
            self.reset_calls: list[tuple[int | None, list[int]]] = []

        def reset(self, *, seed=None, options=None):
            ids = (
                np.arange(self.num_envs, dtype=np.int64)
                if options is None
                else np.asarray(options["env_ids"], dtype=np.int64)
            )
            self.reset_calls.append((seed, ids.tolist()))
            if options is None:
                self.values[:, 0] = np.array([100.0, 200.0, 300.0], dtype=np.float32)
            else:
                self.values[ids, 0] = 1_000.0 + ids
            return self.values.copy(), {"env_ids": ids.copy()}

        def step(self, actions):
            del actions
            return (
                self.values.copy(),
                np.zeros(self.num_envs, dtype=np.float32),
                np.zeros(self.num_envs, dtype=bool),
                np.zeros(self.num_envs, dtype=bool),
                {},
            )

    environment = IndexedResetVectorEnv()
    wrapper = ResettableGymnasiumWrapper(environment)
    wrapper._seed = 123

    partial, _ = wrapper.reset(options={"env_ids": np.array([1], dtype=np.int64)})
    torch.testing.assert_close(partial, torch.tensor([[10.0], [1001.0], [30.0]]))
    assert environment.reset_calls == [(None, [1])]
    assert wrapper._seed == 123
    assert wrapper._reset_once is False
    torch.testing.assert_close(wrapper._observation, partial)

    full, _ = wrapper.reset()
    torch.testing.assert_close(full, torch.tensor([[100.0], [200.0], [300.0]]))
    assert environment.reset_calls[-1] == (123, [0, 1, 2])
    assert wrapper._seed is None

    reset_done = wrapper.reset_done(torch.tensor([[True], [False], [True]]))
    torch.testing.assert_close(reset_done, torch.tensor([[1000.0], [200.0], [1002.0]]))
    assert environment.reset_calls[-1] == (None, [0, 2])


def test_checkpoint_viewer_resets_terminal_rows_before_second_action() -> None:
    import torch

    from sim.manorl.view_environment import _CheckpointPolicyStepper

    class WrappedEnv:
        def __init__(self) -> None:
            self.calls = 0

        def step(self, actions):
            del actions
            self.calls += 1
            return (
                torch.tensor([[99.0], [11.0]])
                if self.calls == 1
                else torch.tensor([[6.0], [12.0]]),
                torch.ones((2, 1)),
                torch.tensor([[self.calls == 1], [False]]),
                torch.zeros((2, 1), dtype=torch.bool),
                {},
            )

    class Runtime:
        def __init__(self) -> None:
            self.env = WrappedEnv()
            self.observations_seen: list[torch.Tensor] = []
            self.reset_masks: list[torch.Tensor] = []

        def deterministic_actions(self, observations):
            self.observations_seen.append(observations.clone())
            return torch.zeros_like(observations)

        def reset_done(self, observations, done):
            del observations
            self.reset_masks.append(done.clone())
            return torch.tensor([[5.0], [11.0]])

    runtime = Runtime()
    stepper = _CheckpointPolicyStepper(runtime, torch.tensor([[0.0], [10.0]]))
    stepper.step()
    stepper.step()

    assert len(runtime.reset_masks) == 1
    torch.testing.assert_close(runtime.reset_masks[0], torch.tensor([[True], [False]]))
    torch.testing.assert_close(runtime.observations_seen[1], torch.tensor([[5.0], [11.0]]))


class _BoundaryAgent:
    device = "cpu"

    def __init__(self) -> None:
        self.observations_seen: list[object] = []
        self.recorded_timesteps: list[int] = []
        self.post_interaction_timesteps: list[int] = []
        self.training = True

    def enable_training_mode(self, enabled: bool) -> None:
        self.training = enabled

    def act(self, observations, *_args, **_kwargs):
        import torch

        self.observations_seen.append(observations.detach().clone())
        return torch.zeros_like(observations), None

    def record_transition(self, *, timestep: int, **_kwargs) -> None:
        self.recorded_timesteps.append(timestep)

    def post_interaction(self, *, timestep: int, **_kwargs) -> None:
        self.post_interaction_timesteps.append(timestep)


class _BoundaryEnv:
    def __init__(self, environment: SimpleNamespace, *, terminal_call: int = 1) -> None:
        self.environment = environment
        self.terminal_call = terminal_call
        self.step_calls = 0
        self.reset_calls: list[list[int] | None] = []

    def reset(self, *, options=None):
        import torch

        ids = None if options is None else np.asarray(options["env_ids"], dtype=np.int64).tolist()
        self.reset_calls.append(ids)
        values = (
            torch.tensor([[0.0], [10.0]])
            if ids is None
            else torch.tensor([[5.0], [11.0]])
        )
        return values, {}

    def step(self, actions):
        import torch

        self.step_calls += 1
        batch = actions.shape[0]
        reward_values = {
            "total": 1.0,
            "distance_x": 0.1,
            "distance_y": 0.2,
            "distance_z": 0.3,
            "rotation": 0.4,
            "action_penalty": 0.0,
            "contact": 0.5,
            "object_stability": 0.6,
            "survival": 0.001,
            "deviation_penalty": 0.0,
        }
        self.environment.last_reward = SimpleNamespace(
            **{name: np.full(batch, value, dtype=np.float64) for name, value in reward_values.items()}
        )
        terminal = np.array([self.step_calls == self.terminal_call, False], dtype=bool)
        self.environment.last_termination = TerminationResult(
            reset=terminal,
            deviation_reset=np.zeros(batch, dtype=bool),
            deviation_penalty=np.zeros(batch, dtype=np.float64),
        )
        self.environment.episode_returns = np.array([1.0, float(self.step_calls)], dtype=np.float64)
        return (
            torch.tensor([[99.0], [11.0]]) if self.step_calls == 1 else torch.tensor([[6.0], [12.0]]),
            torch.ones((batch, 1)),
            torch.as_tensor(terminal).reshape(batch, 1),
            torch.zeros((batch, 1), dtype=torch.bool),
            {},
        )


def test_training_resets_done_world_before_next_action_without_extra_transition() -> None:
    import torch

    from tools.train_manorl_cube1 import TrainingBudget, _train

    environment = SimpleNamespace(
        config=SimpleNamespace(num_envs=2),
        last_reward=None,
        last_termination=None,
        last_transition=None,
        episode_returns=np.zeros(2, dtype=np.float64),
    )
    env = _BoundaryEnv(environment)
    agent = _BoundaryAgent()
    runtime = SimpleNamespace(
        agent=agent,
        env=env,
        gymnasium_env=SimpleNamespace(environment=environment),
        config=SimpleNamespace(rollouts=2),
        model=SimpleNamespace(parameters=lambda: [torch.ones(1)]),
    )

    updates, transitions, _ = _train(
        runtime,
        TrainingBudget(num_envs=2, updates=1, wall_clock_seconds=None),
    )

    assert transitions == 4
    assert len(updates) == 1
    assert env.reset_calls == [None, [0]]
    assert env.step_calls == 2
    assert agent.recorded_timesteps == [0, 1]
    torch.testing.assert_close(agent.observations_seen[0], torch.tensor([[0.0], [10.0]]))
    torch.testing.assert_close(agent.observations_seen[1], torch.tensor([[5.0], [11.0]]))


def test_training_forces_terminal_rerun_sample_even_off_stride() -> None:
    import torch

    from tools.train_manorl_cube1 import TrainingBudget, _train

    environment = SimpleNamespace(
        config=SimpleNamespace(num_envs=2),
        last_reward=None,
        last_termination=None,
        last_transition=None,
        episode_returns=np.zeros(2, dtype=np.float64),
    )
    env = _BoundaryEnv(environment, terminal_call=2)
    recorder = SimpleNamespace(calls=0)
    recorder.record_transition = lambda: setattr(recorder, "calls", recorder.calls + 1)
    runtime = SimpleNamespace(
        agent=_BoundaryAgent(),
        env=env,
        gymnasium_env=SimpleNamespace(environment=environment),
        config=SimpleNamespace(rollouts=2),
        model=SimpleNamespace(parameters=lambda: [torch.ones(1)]),
    )

    _train(
        runtime,
        TrainingBudget(num_envs=2, updates=1, rerun_stride=10),
        recorder=recorder,
    )

    # Step 0 is the regular stride sample; terminal step 1 is forced.
    assert recorder.calls == 2


def _bare_rerun_recorder(
    termination: TerminationResult, *, archive: bool
) -> tuple[object, list[object], list[list[int]]]:
    from sim.manorl.rerun_recorder import ManoRerunRecorder

    snapshot = SimpleNamespace(
        termination=termination,
        episode_return=np.array([7.0], dtype=np.float64),
        reward=SimpleNamespace(total=np.array([3.0], dtype=np.float64)),
    )
    recorder = object.__new__(ManoRerunRecorder)
    recorder._closed = False
    recorder.environment = SimpleNamespace(last_transition=snapshot)
    recorder.env_id = 0
    recorder.episode_id = 0
    recorder._episode_return = 4.0
    recorder._pending_high_env_id = None
    recorder.archive_dir = Path("archive") if archive else None
    recorder.archive_threshold = 5.0 if archive else None
    events: list[object] = []
    queued: list[list[int]] = []
    recorder._capture_replay_action = lambda resolved: events.append("capture")
    recorder._queue_high_return_replays = (
        lambda resolved, env_ids: queued.append(np.asarray(env_ids, dtype=np.int64).tolist())
    )
    recorder._record = lambda resolved: events.append("record")
    recorder._publish_episode = lambda *, archive_path=None: events.append(("publish", archive_path))
    recorder._start_episode = lambda: events.append("start")
    recorder._high_return_archive_path = lambda value: events.append(("archive", value)) or Path("high.rrd")
    return recorder, events, queued


def test_rerun_publishes_terminal_snapshot_and_preserves_threshold_archive_semantics() -> None:
    success = TerminationResult(
        reset=np.array([True]),
        deviation_reset=np.array([False]),
        deviation_penalty=np.zeros(1),
    )
    success_recorder, success_events, success_queued = _bare_rerun_recorder(success, archive=True)
    success_recorder.record_transition()
    assert success_queued == [[0]]
    assert success_events == [
        "capture",
        "record",
        ("archive", 7.0),
        ("publish", Path("high.rrd")),
        "start",
    ]

    failure = TerminationResult(
        reset=np.array([True]),
        deviation_reset=np.array([True]),
        deviation_penalty=np.zeros(1),
    )
    failure_recorder, failure_events, failure_queued = _bare_rerun_recorder(failure, archive=True)
    failure_recorder.record_transition()
    assert failure_queued == [[0]]
    assert failure_events == [
        "capture",
        "record",
        ("archive", 7.0),
        ("publish", Path("high.rrd")),
        "start",
    ]
