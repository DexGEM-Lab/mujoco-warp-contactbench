from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from sim.manorl.view_environment import _tile_layout, parse_args


def _load_tool(name: str):
    path = Path(__file__).resolve().parents[2] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_graphics(*, fail_context: bool = False) -> tuple[object, object, dict[str, object]]:
    state: dict[str, object] = {"callbacks": {}, "moves": [], "updates": [], "renders": [], "destroy": 0, "terminate": 0, "closed": False}
    window = object()

    def callback(name: str):
        return lambda _, value: state["callbacks"].__setitem__(name, value)

    glfw = SimpleNamespace(
        PRESS=1,
        MOUSE_BUTTON_LEFT=0,
        MOUSE_BUTTON_MIDDLE=1,
        MOUSE_BUTTON_RIGHT=2,
        KEY_ESCAPE=256,
        KEY_R=82,
        init=lambda: True,
        create_window=lambda *_: window,
        make_context_current=lambda _: None,
        swap_interval=lambda _: None,
        set_mouse_button_callback=callback("mouse"),
        set_cursor_pos_callback=callback("cursor"),
        set_scroll_callback=callback("scroll"),
        set_key_callback=callback("key"),
        get_cursor_pos=lambda _: (10.0, 20.0),
        get_window_size=lambda _: (100, 100),
        get_framebuffer_size=lambda _: (1600, 900),
        window_should_close=lambda _: state["closed"],
        set_window_should_close=lambda _, value: state.__setitem__("closed", value),
        swap_buffers=lambda _: None,
        poll_events=lambda: None,
        destroy_window=lambda _: state.__setitem__("destroy", state["destroy"] + 1),
        terminate=lambda: state.__setitem__("terminate", state["terminate"] + 1),
    )

    class Camera:
        def __init__(self) -> None:
            self.type = None
            self.azimuth = self.elevation = self.distance = 0.0
            self.lookat = np.zeros(3)

    class Option:
        def __init__(self) -> None:
            self.geomgroup = np.ones(6, dtype=np.int32)

    class Scene:
        def __init__(self, *_: object, **__: object) -> None:
            pass

    class Context:
        def __init__(self, *_: object, **__: object) -> None:
            if fail_context:
                raise RuntimeError("context failed")

    class Rect:
        def __init__(self, *values: int) -> None:
            self.values = values

    mujoco = SimpleNamespace(
        MjvCamera=Camera,
        MjvOption=Option,
        MjvScene=Scene,
        MjrContext=Context,
        MjrRect=Rect,
        mjtCamera=SimpleNamespace(mjCAMERA_FREE="free"),
        mjtFontScale=SimpleNamespace(mjFONTSCALE_150="scale"),
        mjtCatBit=SimpleNamespace(mjCAT_ALL="all"),
        mjtFont=SimpleNamespace(mjFONT_NORMAL="font"),
        mjtGridPos=SimpleNamespace(mjGRID_TOPLEFT="grid"),
        mjtMouse=SimpleNamespace(
            mjMOUSE_ROTATE_V="rotate", mjMOUSE_MOVE_H="move_h", mjMOUSE_MOVE_V="move_v", mjMOUSE_ZOOM="zoom"
        ),
        MjData=lambda _: object(),
        mjv_defaultCamera=lambda _: None,
        mjv_defaultOption=lambda _: None,
        mjv_moveCamera=lambda _, action, dx, dy, __, ___: state["moves"].append((action, dx, dy)),
        mjv_updateScene=lambda model, *_: (
            state["updates"].append("scene"), state.setdefault("update_models", []).append(model)
        ),
        mjr_rectangle=lambda *_: None,
        mjr_render=lambda *_: state["renders"].append("render"),
        mjr_overlay=lambda *_: None,
    )
    return glfw, mujoco, state


def _training_environment() -> tuple[object, dict[str, int]]:
    calls = {"host": 0}
    visual = SimpleNamespace(
        rgba=SimpleNamespace(fog=np.zeros(4), haze=np.zeros(4)),
        headlight=SimpleNamespace(ambient=np.zeros(3), diffuse=np.zeros(3), specular=np.zeros(3)),
    )
    environment = SimpleNamespace(
        config=SimpleNamespace(num_envs=2),
        model=SimpleNamespace(vis=visual),
        trajectories=[SimpleNamespace(identity=SimpleNamespace(identity=f"env-{index}")) for index in range(2)],
    )
    def host_data_batch() -> list[object]:
        calls["host"] += 1
        return [object(), object()]
    environment.host_data_batch = host_data_batch
    return environment, calls


def _patch_training_viewer_native_model(monkeypatch, viewer, mujoco, environment):
    visual_model = SimpleNamespace(vis=environment.model.vis)
    mirror_calls = []
    monkeypatch.setattr(
        viewer,
        "_compile_native_viewer_model",
        lambda _: (mujoco, visual_model),
    )
    monkeypatch.setattr(
        viewer,
        "_mirror_native_viewer_data",
        lambda *args: mirror_calls.append(args),
    )
    return visual_model, mirror_calls


def test_training_viewer_renders_states_without_stepping_and_preserves_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    import sim.manorl.view_environment as viewer

    glfw, mujoco, state = _fake_graphics()
    monkeypatch.setitem(sys.modules, "glfw", glfw)
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setattr(viewer, "_require_graphical_session", lambda: None)
    environment, calls = _training_environment()
    visual_model, mirror_calls = _patch_training_viewer_native_model(
        monkeypatch, viewer, mujoco, environment
    )
    training_viewer = viewer.TrainingViewer(environment, tile_envs=2)

    training_viewer.render()
    assert calls == {"host": 1}
    assert state["updates"] == ["scene", "scene"]
    assert state["renders"] == ["render", "render"]
    assert visual_model is not environment.model
    assert training_viewer._viewer_model is visual_model
    assert training_viewer._option.geomgroup[3] == 0
    assert state["update_models"] == [visual_model, visual_model]
    assert [call[1] for call in mirror_calls] == [visual_model, visual_model]

    callbacks = state["callbacks"]
    callbacks["mouse"](None, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
    callbacks["cursor"](None, 20.0, 30.0)
    callbacks["mouse"](None, glfw.MOUSE_BUTTON_LEFT, 0, 0)
    callbacks["mouse"](None, glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0)
    callbacks["cursor"](None, 30.0, 40.0)
    callbacks["mouse"](None, glfw.MOUSE_BUTTON_RIGHT, 0, 0)
    callbacks["mouse"](None, glfw.MOUSE_BUTTON_MIDDLE, glfw.PRESS, 0)
    callbacks["cursor"](None, 40.0, 50.0)
    callbacks["scroll"](None, 0.0, 2.0)
    training_viewer._camera.distance = 9.0
    callbacks["key"](None, glfw.KEY_R, 0, glfw.PRESS, 0)
    callbacks["key"](None, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)

    assert [move[0] for move in state["moves"]] == ["rotate", "move_h", "move_v", "zoom"]
    assert training_viewer._camera.distance == 1.2
    assert training_viewer.close_requested is False
    training_viewer.render()
    assert training_viewer.close_requested is True
    training_viewer.close()
    training_viewer.close()
    assert state["destroy"] == 1 and state["terminate"] == 1


def test_training_viewer_quiet_mode_emits_no_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sim.manorl.view_environment as viewer

    glfw, mujoco, _ = _fake_graphics()
    monkeypatch.setitem(sys.modules, "glfw", glfw)
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setattr(viewer, "_require_graphical_session", lambda: None)
    environment, _ = _training_environment()
    _patch_training_viewer_native_model(monkeypatch, viewer, mujoco, environment)
    training_viewer = viewer.TrainingViewer(environment, tile_envs=2, quiet=True)

    assert capsys.readouterr().out == ""
    training_viewer.close()


def test_training_viewer_window_failure_terminates_glfw_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import sim.manorl.view_environment as viewer

    glfw, mujoco, state = _fake_graphics()
    glfw.create_window = lambda *_: None
    monkeypatch.setitem(sys.modules, "glfw", glfw)
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setattr(viewer, "_require_graphical_session", lambda: None)
    environment, _ = _training_environment()
    _patch_training_viewer_native_model(monkeypatch, viewer, mujoco, environment)

    with pytest.raises(RuntimeError, match="could not create GLFW window"):
        viewer.TrainingViewer(environment, tile_envs=2)
    assert state["destroy"] == 0 and state["terminate"] == 1


def test_training_viewer_constructor_failure_releases_glfw_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import sim.manorl.view_environment as viewer

    glfw, mujoco, state = _fake_graphics(fail_context=True)
    monkeypatch.setitem(sys.modules, "glfw", glfw)
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setattr(viewer, "_require_graphical_session", lambda: None)
    environment, _ = _training_environment()
    _patch_training_viewer_native_model(monkeypatch, viewer, mujoco, environment)

    with pytest.raises(RuntimeError, match="context failed"):
        viewer.TrainingViewer(environment, tile_envs=2)
    assert state["destroy"] == 1 and state["terminate"] == 1


def test_native_visual_model_mirrors_collision_only_state_and_hides_collision_geoms() -> None:
    import mujoco

    from sim.manorl.assets import COLLISION_GEOM_GROUP, compile_model
    from sim.manorl.contracts import ServoConfig
    from sim.manorl.view_environment import (
        _compile_native_viewer_model,
        _mirror_native_viewer_data,
        _validate_native_viewer_abi,
    )

    _, physics_model = compile_model()
    environment = SimpleNamespace(
        config=SimpleNamespace(servo=ServoConfig(), physics_timestep=0.0025),
        is_heterogeneous=False,
        model=physics_model,
        object_type="cube1",
    )
    mujoco, viewer_model = _compile_native_viewer_model(environment)
    _validate_native_viewer_abi(mujoco, physics_model, viewer_model)

    physics_data = mujoco.MjData(physics_model)
    mujoco.mj_resetData(physics_model, physics_data)
    physics_data.time = 1.25
    physics_data.qpos[0] = 0.05
    physics_data.qvel[0] = 0.1
    physics_data.ctrl[:] = 0.2
    mujoco.mj_forward(physics_model, physics_data)

    viewer_data = mujoco.MjData(viewer_model)
    _mirror_native_viewer_data(
        mujoco,
        viewer_model,
        viewer_data,
        physics_data,
    )
    assert viewer_data.time == physics_data.time
    for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat"):
        np.testing.assert_array_equal(
            getattr(viewer_data, name), getattr(physics_data, name)
        )
    np.testing.assert_allclose(viewer_data.xpos, physics_data.xpos, atol=1e-15, rtol=0)

    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[COLLISION_GEOM_GROUP] = 0
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    scene = mujoco.MjvScene(viewer_model, maxgeom=1_000)
    mujoco.mjv_updateScene(
        viewer_model,
        viewer_data,
        option,
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    rendered_geom_names = {
        mujoco.mj_id2name(viewer_model, mujoco.mjtObj.mjOBJ_GEOM, int(geom.objid))
        for geom in scene.geoms[: scene.ngeom]
        if int(geom.objtype) == int(mujoco.mjtObj.mjOBJ_GEOM)
    }
    assert {"palm_visual", "cube1_visual"} <= rendered_geom_names
    assert not any(
        name is not None and name.endswith("_collision")
        for name in rendered_geom_names
    )


def test_passive_viewer_lock_guards_native_mirror_and_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sim.manorl.view_environment as viewer

    events: list[str] = []

    class FakeViewer:
        def __init__(self) -> None:
            self.opt = SimpleNamespace(geomgroup=np.ones(6, dtype=np.int32))
            self._running_calls = 0

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args) -> None:
            events.append("exit")

        def lock(self):
            class Lock:
                def __enter__(self_inner):
                    events.append("lock-enter")

                def __exit__(self_inner, *_args) -> None:
                    events.append("lock-exit")

            return Lock()

        def sync(self) -> None:
            events.append("sync")

        def is_running(self) -> bool:
            self._running_calls += 1
            return self._running_calls == 1

    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.__path__ = []
    fake_mujoco.MjData = lambda _model: object()
    fake_viewer_module = ModuleType("mujoco.viewer")
    passive_viewer = FakeViewer()

    def launch_passive(*_args, **_kwargs):
        events.append("launch")
        return passive_viewer

    fake_viewer_module.launch_passive = launch_passive
    fake_mujoco.viewer = fake_viewer_module
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setitem(sys.modules, "mujoco.viewer", fake_viewer_module)

    visual_model = object()
    monkeypatch.setattr(
        viewer,
        "_compile_native_viewer_model",
        lambda _environment: (fake_mujoco, visual_model),
    )
    monkeypatch.setattr(
        viewer,
        "_mirror_native_viewer_data",
        lambda *_args: events.append("mirror"),
    )
    monkeypatch.setattr(viewer, "_telemetry", lambda *_args: "telemetry")

    environment = SimpleNamespace(
        config=SimpleNamespace(num_envs=1, control_timestep=0.005),
        progress=np.array([1]),
        host_data=lambda _env_id: events.append("host") or object(),
    )

    class Stepper:
        def step(self):
            events.append("step")
            return None, np.zeros(1), np.zeros(1, dtype=bool), None

    viewer._view_single(
        environment,
        stepper=Stepper(),
        render_env=0,
        speed=1.0,
        loop=True,
        print_every=1,
        recorder=None,
    )

    assert events == [
        "host",
        "mirror",
        "launch",
        "enter",
        "lock-enter",
        "sync",
        "lock-exit",
        "step",
        "host",
        "lock-enter",
        "mirror",
        "sync",
        "lock-exit",
        "exit",
    ]
    assert passive_viewer.opt.geomgroup[3] == 0


def test_viewer_cli_defaults_to_residual_and_terminal_modes() -> None:
    args = parse_args([])
    assert args.device == "cpu"
    assert args.speed == 0.25
    assert args.loop is True
    assert args.print_every == 10
    assert args.terminal is True
    assert args.use_residual is True
    assert args.trajectory == "accepted"
    assert args.num_envs == 1
    assert args.render_env == 0
    assert args.tile_envs == 1
    assert args.object_type is None
    assert args.gesture is None
    assert args.checkpoint is None
    assert args.rerun_output is None

    one_shot = parse_args(
        [
            "--device",
            "gpu",
            "--no-loop",
            "--speed",
            "1.0",
            "--terminal",
            "false",
            "--use_residual",
            "false",
            "--trajectory",
            "generated-cube1-row-507",
            "--num-envs",
            "10",
            "--render-env",
            "9",
            "--tile-envs",
            "10",
            "--object",
            "cube1",
            "--gesture",
            "03",
            "--checkpoint",
            "outputs/manorl/policy.pt",
            "--rerun-output",
            "outputs/manorl/viewer.rrd",
        ]
    )
    assert one_shot.device == "gpu"
    assert one_shot.loop is False
    assert one_shot.speed == 1.0
    assert one_shot.terminal is False
    assert one_shot.use_residual is False
    assert one_shot.trajectory == "generated-cube1-row-507"
    assert one_shot.num_envs == 10
    assert one_shot.render_env == 9
    assert one_shot.tile_envs == 10
    assert one_shot.object_type == "cube1"
    assert one_shot.gesture == "03"
    assert str(one_shot.checkpoint) == "outputs/manorl/policy.pt"
    assert str(one_shot.rerun_output) == "outputs/manorl/viewer.rrd"


@pytest.mark.parametrize(
    ("mode_args", "expected_residual_enabled", "expected_terminal"),
    [([], True, True), (["--use_residual", "false", "--terminal", "false"], False, False)],
)
def test_record_rerun_cli_reports_null_without_reset_completed_episode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode_args: list[str],
    expected_residual_enabled: bool,
    expected_terminal: bool,
) -> None:
    record_manorl_rerun = _load_tool("record_manorl_rerun")
    captured_configs = []

    class FakeEnvironment:
        def __init__(self, trajectories, config) -> None:
            captured_configs.append(config)

        def step(self, actions) -> None:
            pass

    class FakeRecorder:
        def __init__(self, environment, output, env_id: int) -> None:
            pass

        def record_transition(self) -> None:
            pass

        def close(self) -> Path | None:
            return None

    monkeypatch.setattr(
        record_manorl_rerun,
        "load_assigned_trajectory_batch",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(record_manorl_rerun, "_assignment_payload", lambda trajectories: [])
    monkeypatch.setattr(record_manorl_rerun, "MujocoManoEnvironment", FakeEnvironment)
    monkeypatch.setattr(record_manorl_rerun, "ManoRerunRecorder", FakeRecorder)

    assert record_manorl_rerun.main(
        ["--output", str(tmp_path / "recording.rrd"), "--steps", "1", *mode_args]
    ) == 0
    assert captured_configs[0].residual_enabled is expected_residual_enabled
    assert captured_configs[0].reference_fps == 120
    assert (captured_configs[0].max_deviation_distance == 0.10) is expected_terminal
    assert '"rerun_artifact": null' in capsys.readouterr().out


def test_viewer_reports_missing_reset_completed_rerun_artifact(capsys: pytest.CaptureFixture[str]) -> None:
    from sim.manorl.view_environment import _close_rerun_recorder

    class FakeRecorder:
        def close(self) -> Path | None:
            return None

    assert _close_rerun_recorder(FakeRecorder()) is None
    assert capsys.readouterr().out == "No terminal-complete Rerun episode was published.\n"


@pytest.mark.parametrize(
    ("mode_args", "expected_residual_enabled", "expected_terminal"),
    [([], True, True), (["--use_residual", "false", "--terminal", "false"], False, False)],
)
def test_train_cube1_cli_parses_runtime_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode_args: list[str],
    expected_residual_enabled: bool,
    expected_terminal: bool,
) -> None:
    train_manorl_cube1 = _load_tool("train_manorl_cube1")
    captured_budgets = []

    def fake_run(output: Path, budget) -> dict[str, object]:
        captured_budgets.append(budget)
        return {}

    monkeypatch.setattr(train_manorl_cube1, "run", fake_run)

    assert train_manorl_cube1.main(
        ["--output", str(tmp_path / "training"), *mode_args]
    ) == 0
    assert captured_budgets[0].residual_enabled is expected_residual_enabled
    assert captured_budgets[0].terminal is expected_terminal


def test_tile_layout_covers_non_overlapping_grid() -> None:
    assert _tile_layout(1, width=100, height=80) == [(0, 0, 100, 80)]
    assert _tile_layout(4, width=100, height=80) == [
        (0, 40, 50, 40),
        (50, 40, 50, 40),
        (0, 0, 50, 40),
        (50, 0, 50, 40),
    ]


def test_zero_stepper_preserves_zero_batch_actions() -> None:
    import numpy as np

    from sim.manorl.view_environment import _ZeroActionStepper

    class FakeEnvironment:
        class config:
            num_envs = 3

        def __init__(self) -> None:
            self.actions = None

        def step(self, actions):
            self.actions = actions
            return "obs", np.zeros(3), np.zeros(3, dtype=bool), {}

    environment = FakeEnvironment()
    stepper = _ZeroActionStepper(environment)
    stepper.step()
    assert environment.actions.shape == (3, 28)
    assert environment.actions.dtype == np.float64
    assert not np.any(environment.actions)


def test_checkpoint_stepper_uses_mean_actions_and_wrapped_done_flags() -> None:
    import torch

    from sim.manorl.view_environment import _CheckpointPolicyStepper

    class FakeWrappedEnv:
        def step(self, actions):
            self.actions = actions
            return torch.tensor([[3.0]]), torch.tensor([1.25]), torch.tensor([False]), torch.tensor([True]), {"x": 1}

    class FakeRuntime:
        def __init__(self) -> None:
            self.env = FakeWrappedEnv()
            self.observations = None

        def deterministic_actions(self, observations):
            self.observations = observations
            return torch.tensor([[0.4]])

    runtime = FakeRuntime()
    stepper = _CheckpointPolicyStepper(runtime, torch.tensor([[2.0]]))
    _, rewards, resets, info = stepper.step()
    assert torch.equal(runtime.observations, torch.tensor([[2.0]]))
    assert torch.equal(runtime.env.actions, torch.tensor([[0.4]]))
    assert rewards.tolist() == [1.25]
    assert resets.tolist() == [True]
    assert info == {"x": 1}


def test_checkpoint_stepper_skips_policy_forward_until_prefix_end() -> None:
    import torch

    from sim.manorl.view_environment import _CheckpointPolicyStepper

    class Physical:
        action_dim = 28
        trajectory_steps = np.asarray([0], dtype=np.int64)

        class config:
            num_envs = 1

    physical = Physical()

    class Wrapped:
        def step(self, actions):
            self.actions = actions.clone()
            physical.trajectory_steps += 1
            return (
                torch.zeros((1, 2)),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {},
            )

    class Runtime:
        def __init__(self) -> None:
            self.env = Wrapped()
            self.gymnasium_env = SimpleNamespace(environment=physical)
            self.calls = 0

        def deterministic_actions(self, observations):
            self.calls += 1
            return torch.ones((1, 28))

    runtime = Runtime()
    stepper = _CheckpointPolicyStepper(
        runtime,
        torch.zeros((1, 2)),
        policy_enable_steps=np.asarray([2]),
    )
    stepper.step()
    assert runtime.calls == 0
    np.testing.assert_array_equal(runtime.env.actions.numpy(), 0.0)
    stepper.step()
    assert runtime.calls == 0
    np.testing.assert_array_equal(runtime.env.actions.numpy(), 0.0)
    stepper.step()
    assert runtime.calls == 1
    np.testing.assert_array_equal(runtime.env.actions.numpy(), 1.0)


def test_checkpoint_stepper_stops_policy_forward_at_suffix_boundary() -> None:
    import torch

    from sim.manorl.view_environment import _CheckpointPolicyStepper

    class Physical:
        action_dim = 28
        trajectory_steps = np.asarray([0], dtype=np.int64)

        class config:
            num_envs = 1

    physical = Physical()

    class Wrapped:
        def step(self, actions):
            self.actions = actions.clone()
            physical.trajectory_steps += 1
            return (
                torch.zeros((1, 2)),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {},
            )

    class Runtime:
        def __init__(self) -> None:
            self.env = Wrapped()
            self.gymnasium_env = SimpleNamespace(environment=physical)
            self.calls = 0

        def deterministic_actions(self, observations):
            self.calls += 1
            return torch.ones((1, 28))

    runtime = Runtime()
    stepper = _CheckpointPolicyStepper(
        runtime,
        torch.zeros((1, 2)),
        policy_enable_steps=np.asarray([1]),
        policy_disable_steps=np.asarray([3]),
    )
    stepper.step()
    assert runtime.calls == 0
    np.testing.assert_array_equal(runtime.env.actions.numpy(), 0.0)
    stepper.step()
    assert runtime.calls == 1
    np.testing.assert_array_equal(runtime.env.actions.numpy(), 1.0)
    stepper.step()
    assert runtime.calls == 2
    np.testing.assert_array_equal(runtime.env.actions.numpy(), 1.0)
    stepper.step()
    assert runtime.calls == 2
    np.testing.assert_array_equal(runtime.env.actions.numpy(), 0.0)


def test_checkpoint_builder_loads_eval_runtime_and_resets_wrapped_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    from sim.manorl import checkpoint as checkpoint_module
    from sim.manorl import gymnasium_env, skrl_runtime
    from sim.manorl.view_environment import _build_checkpoint_stepper

    captured = {}

    class FakeAdapter:
        def __init__(self, environment) -> None:
            captured["adapter_environment"] = environment

    class FakeAgent:
        def enable_training_mode(self, enabled: bool) -> None:
            captured["training_enabled"] = enabled

    class FakeModel:
        def eval(self) -> None:
            captured["eval"] = True

    class FakeWrappedEnv:
        def reset(self):
            captured["reset"] = True
            return torch.zeros((2, 1)), {}

    class FakeRuntime:
        def __init__(self, adapter, config) -> None:
            captured["config"] = config
            self.agent = FakeAgent()
            self.model = FakeModel()
            self.env = FakeWrappedEnv()

        def deterministic_actions(self, observations):
            return observations

    environment = type("Environment", (), {"config": type("Config", (), {"num_envs": 2})()})()
    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    monkeypatch.setattr(gymnasium_env, "ManoGymnasiumVectorEnv", FakeAdapter)
    monkeypatch.setattr(skrl_runtime, "ManoSkrlRuntime", FakeRuntime)
    monkeypatch.setattr(
        checkpoint_module,
        "load_skrl_checkpoint_for_inference",
        lambda agent, path: captured.update(path=path, loader="strict"),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "load_skrl_checkpoint_for_policy_transfer_inference",
        lambda agent, path: captured.update(path=path, loader="policy_transfer"),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "checkpoint_runtime_metadata",
        lambda path: {"runtime_config": {"model": {"use_film": False}}},
    )

    stepper = _build_checkpoint_stepper(environment, checkpoint)
    assert captured["adapter_environment"] is environment
    assert captured["path"] == checkpoint
    assert captured["loader"] == "strict"
    assert captured["config"].rollouts == 1
    assert captured["config"].minibatch_size == 2
    assert captured["config"].use_film is False
    assert captured["training_enabled"] is False
    assert captured["eval"] is True
    assert captured["reset"] is True
    assert stepper is not None

    _build_checkpoint_stepper(environment, checkpoint, policy_transfer=True)
    assert captured["loader"] == "policy_transfer"


def test_inference_config_defaults_to_film_when_sidecar_has_no_model_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sim.manorl import checkpoint as checkpoint_module
    from sim.manorl.view_environment import _inference_ppo_config
    from sim.manorl.view_environment import _checkpoint_use_film

    assert _inference_ppo_config(2).use_film is True
    assert _inference_ppo_config(2, use_film=False).use_film is False
    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        checkpoint_module,
        "checkpoint_runtime_metadata",
        lambda path: {"runtime_config": {}},
    )
    assert _checkpoint_use_film(checkpoint) is True


def test_checkpoint_environment_options_restore_residual_and_per_world_ccd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sim.manorl import checkpoint as checkpoint_module
    from sim.manorl.view_environment import _checkpoint_environment_options

    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        checkpoint_module,
        "checkpoint_runtime_metadata",
        lambda path: {
            "environment_contract": checkpoint_module.ENVIRONMENT_CONTRACT_ID,
            "runtime_config": {
                "environment": {
                    "reference_fps": 100,
                    "control_fps": 100,
                    "pre_padding": 180,
                    "post_padding": 250,
                    "residual_action": {
                        "joint_scale_multiplier": 1.5,
                        "joint_max_offset_multiplier": 1.5,
                    },
                    "warp_ccd": {
                        "ccd_iterations": None,
                        "contacts_per_world": 16,
                        "naccdmax": 32768,
                    },
                }
            }
        },
    )

    options = _checkpoint_environment_options(checkpoint)
    assert options.residual_action.joint_scale_multiplier == 1.5
    assert options.residual_action.joint_max_offset_multiplier == 1.5
    assert options.reference_fps == 100
    assert options.control_fps == 100
    assert options.pre_padding == 180
    assert options.post_padding == 250
    assert options.warp_ccd_iterations is None
    assert options.warp_ccd_contacts_per_world == 16

    monkeypatch.setattr(
        checkpoint_module,
        "checkpoint_runtime_metadata",
        lambda path: {
            "environment_contract": sorted(
                checkpoint_module.LEGACY_ENVIRONMENT_CONTRACT_IDS
            )[-1],
            "runtime_config": {
                "environment": {
                    "reference_fps": 100,
                    "compatibility": {"movement_pre_padding": 100},
                }
            },
        },
    )
    legacy = _checkpoint_environment_options(checkpoint)
    assert legacy.reference_fps == 100
    assert legacy.control_fps == 200
    assert legacy.pre_padding == 100
    assert legacy.post_padding == 250


def test_stochastic_record_stepper_uses_sidecar_model_variant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    record_tool = _load_tool("record_manorl_rerun")
    from sim.manorl import checkpoint as checkpoint_module
    from sim.manorl import gymnasium_env, skrl_runtime

    captured: dict[str, object] = {}

    class FakeAdapter:
        def __init__(self, environment: object) -> None:
            captured["environment"] = environment

    class FakeAgent:
        def enable_training_mode(self, enabled: bool) -> None:
            captured["training"] = enabled

    class FakeWrappedEnv:
        def reset(self):
            return torch.zeros((2, 1)), {}

    class FakeRuntime:
        def __init__(self, adapter: object, config: object) -> None:
            captured["adapter"] = adapter
            captured["config"] = config
            self.agent = FakeAgent()
            self.env = FakeWrappedEnv()

    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    monkeypatch.setattr(gymnasium_env, "ManoGymnasiumVectorEnv", FakeAdapter)
    monkeypatch.setattr(skrl_runtime, "ManoSkrlRuntime", FakeRuntime)
    monkeypatch.setattr(record_tool, "_checkpoint_use_film", lambda path: False)
    monkeypatch.setattr(
        checkpoint_module,
        "load_skrl_checkpoint_for_inference",
        lambda agent, path: captured.update(path=path),
    )

    environment = type("Environment", (), {"config": type("Config", (), {"num_envs": 2})()})()
    stepper = record_tool._StochasticCheckpointStepper(environment, checkpoint)

    assert stepper is not None
    assert captured["config"].use_film is False
    assert captured["path"] == checkpoint
    assert captured["training"] is True


@pytest.mark.parametrize("tile_envs", [1, 2])
def test_viewer_dispatches_shared_stepper_to_each_renderer(
    monkeypatch: pytest.MonkeyPatch, tile_envs: int
) -> None:
    import sim.manorl.view_environment as viewer

    dispatched = {}
    created = {}
    sentinel_stepper = object()

    class FakeEnvironment:
        def __init__(self, trajectory, config) -> None:
            self.config = config
            created["config"] = config

    monkeypatch.setattr(viewer, "_require_graphical_session", lambda: None)
    monkeypatch.setattr(viewer, "load_reference_trajectory", lambda: object())
    monkeypatch.setattr(viewer, "MujocoManoEnvironment", FakeEnvironment)
    monkeypatch.setattr(viewer, "_ZeroActionStepper", lambda environment: sentinel_stepper)
    monkeypatch.setattr(
        viewer,
        "_view_single",
        lambda environment, **kwargs: dispatched.update(renderer="single", **kwargs),
    )
    monkeypatch.setattr(
        viewer,
        "_view_tiled",
        lambda environment, **kwargs: dispatched.update(renderer="tiled", **kwargs),
    )

    viewer.view_environment(
        device="cpu",
        speed=1.0,
        loop=False,
        print_every=1,
        terminal=True,
        trajectory_name="accepted",
        num_envs=2,
        render_env=0,
        tile_envs=tile_envs,
        object_type=None,
        gesture=None,
        rerun_output=None,
        use_residual=True,
        checkpoint=None,
    )
    assert dispatched["renderer"] == ("single" if tile_envs == 1 else "tiled")
    assert dispatched["stepper"] is sentinel_stepper
    assert not hasattr(created["config"], "visual_meshes")


def test_checkpoint_viewer_applies_sidecar_options_and_dataset_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sim.manorl.view_environment as viewer

    created: dict[str, object] = {}
    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    dataset = tmp_path / "dataset.lance"

    class FakeEnvironment:
        def __init__(self, trajectory: object, config: object) -> None:
            self.config = config
            created["config"] = config

    def load(selection: object, *, num_envs: int) -> object:
        created["selection"] = selection
        created["num_envs"] = num_envs
        return SimpleNamespace(hand_sides=("right",))

    options = viewer._CheckpointEnvironmentOptions(
        residual_action=viewer.ResidualActionConfig(
            joint_scale_multiplier=1.5,
            joint_max_offset_multiplier=1.5,
        ),
        reference_fps=100,
        control_fps=100,
        pre_padding=180,
        post_padding=250,
        warp_ccd_contacts_per_world=16,
    )
    monkeypatch.setattr(viewer, "_require_graphical_session", lambda: None)
    monkeypatch.setattr(viewer, "_validate_checkpoint_path", lambda path: path)
    monkeypatch.setattr(viewer, "_checkpoint_environment_options", lambda path: options)
    monkeypatch.setattr(viewer, "load_assigned_trajectory_batch", load)
    monkeypatch.setattr(viewer, "MujocoManoEnvironment", FakeEnvironment)
    monkeypatch.setattr(viewer, "_build_checkpoint_stepper", lambda env, path: object())
    monkeypatch.setattr(viewer, "_view_single", lambda *args, **kwargs: None)

    viewer.view_environment(
        device="cpu",
        speed=1.0,
        loop=False,
        print_every=1,
        terminal=True,
        trajectory_name="accepted",
        num_envs=1,
        render_env=0,
        tile_envs=1,
        object_type="banana",
        gesture="01",
        rerun_output=None,
        use_residual=True,
        checkpoint=checkpoint,
        dataset_path=dataset,
        dataset_version=978,
        hand_side="right",
    )

    selection = created["selection"]
    assert selection.expected_dataset_version == 978
    assert selection.reference_fps == 100
    assert selection.control_fps == 100
    assert selection.pre_padding == 180
    assert selection.post_padding == 250
    config = created["config"]
    assert config.reference_fps == 100
    assert config.control_fps == 100
    assert config.compatibility.movement_pre_padding == 180
    assert config.post_padding == 250
    assert config.residual_action.joint_scale_multiplier == 1.5
    assert config.residual_action.joint_max_offset_multiplier == 1.5
    assert config.warp_ccd_contacts_per_world == 16


def test_checkpoint_rejects_disabled_residual_before_graphics(monkeypatch: pytest.MonkeyPatch) -> None:
    import sim.manorl.view_environment as viewer

    monkeypatch.setattr(viewer, "_require_graphical_session", pytest.fail)
    with pytest.raises(ValueError, match="requires --use_residual true"):
        viewer.view_environment(
            device="cpu",
            speed=1.0,
            loop=False,
            print_every=1,
            terminal=True,
            trajectory_name="accepted",
            num_envs=1,
            render_env=0,
            tile_envs=1,
            object_type=None,
            gesture=None,
            rerun_output=None,
            use_residual=False,
            checkpoint=Path("policy.pt"),
        )


def test_viewer_cli_accepts_pinned_dataset_version_and_reference_fps() -> None:
    args = parse_args(["--dataset-version", "978", "--reference-fps", "100"])
    assert args.dataset_version == 978
    assert args.reference_fps == 100
    assert parse_args(["--reference-fps", "120"]).reference_fps == 120
    with pytest.raises(SystemExit):
        parse_args(["--reference-fps", "200"])


def test_viewer_cli_accepts_explicit_padding() -> None:
    args = parse_args(["--pre-padding", "73", "--post-padding", "91"])
    assert args.pre_padding == 73
    assert args.post_padding == 91
    defaults = parse_args([])
    assert defaults.pre_padding is None
    assert defaults.post_padding is None


def test_padding_resolution_defaults_overrides_and_preserves_checkpoint() -> None:
    import sim.manorl.view_environment as viewer

    assert viewer._resolve_padding(
        None, checkpoint_value=180, has_checkpoint=False, name="pre_padding"
    ) == 180
    assert viewer._resolve_padding(
        73, checkpoint_value=180, has_checkpoint=False, name="pre_padding"
    ) == 73
    assert viewer._resolve_padding(
        None, checkpoint_value=100, has_checkpoint=True, name="pre_padding"
    ) == 100
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        viewer._resolve_padding(
            180, checkpoint_value=100, has_checkpoint=True, name="pre_padding"
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        viewer._resolve_padding(
            -1, checkpoint_value=180, has_checkpoint=False, name="pre_padding"
        )


def test_dataset_viewer_applies_explicit_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sim.manorl.view_environment as viewer

    created: dict[str, object] = {}

    class FakeEnvironment:
        def __init__(self, trajectory: object, config: object) -> None:
            self.config = config
            created["config"] = config

    def load(selection: object, *, num_envs: int) -> object:
        created["selection"] = selection
        created["num_envs"] = num_envs
        return SimpleNamespace(hand_sides=("right",))

    monkeypatch.setattr(viewer, "_require_graphical_session", lambda: None)
    monkeypatch.setattr(viewer, "load_assigned_trajectory_batch", load)
    monkeypatch.setattr(viewer, "MujocoManoEnvironment", FakeEnvironment)
    monkeypatch.setattr(viewer, "_ZeroActionStepper", lambda environment: object())
    monkeypatch.setattr(viewer, "_view_single", lambda *args, **kwargs: None)

    viewer.view_environment(
        device="cpu",
        speed=1.0,
        loop=False,
        print_every=1,
        terminal=True,
        trajectory_name="accepted",
        num_envs=1,
        render_env=0,
        tile_envs=1,
        object_type="banana",
        gesture="01",
        rerun_output=None,
        use_residual=False,
        checkpoint=None,
        dataset_path=tmp_path / "dataset.lance",
        dataset_version=978,
        pre_padding=73,
        post_padding=91,
        hand_side="right",
    )

    selection = created["selection"]
    assert selection.pre_padding == 73
    assert selection.post_padding == 91
    config = created["config"]
    assert config.compatibility.movement_pre_padding == 73
    assert config.post_padding == 91


def test_reference_fps_resolution_defaults_restores_and_preserves_legacy() -> None:
    import sim.manorl.view_environment as viewer

    assert viewer._resolve_reference_fps(
        None,
        checkpoint_options=viewer._CheckpointEnvironmentOptions(),
        has_checkpoint=False,
    ) == 120
    assert viewer._resolve_reference_fps(
        None,
        checkpoint_options=viewer._CheckpointEnvironmentOptions(reference_fps=100),
        has_checkpoint=True,
    ) == 100
    assert viewer._resolve_reference_fps(
        None,
        checkpoint_options=viewer._CheckpointEnvironmentOptions(),
        has_checkpoint=True,
    ) is None
    with pytest.raises(ValueError, match="predates"):
        viewer._resolve_reference_fps(
            100,
            checkpoint_options=viewer._CheckpointEnvironmentOptions(),
            has_checkpoint=True,
        )


def test_checkpoint_viewer_rejects_explicit_reference_fps_conflict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sim.manorl.view_environment as viewer

    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    monkeypatch.setattr(viewer, "_validate_checkpoint_path", lambda path: path)
    monkeypatch.setattr(
        viewer,
        "_checkpoint_environment_options",
        lambda path: viewer._CheckpointEnvironmentOptions(reference_fps=120),
    )
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        viewer.view_environment(
            device="cpu",
            speed=1.0,
            loop=False,
            print_every=1,
            terminal=True,
            trajectory_name="accepted",
            num_envs=1,
            render_env=0,
            tile_envs=1,
            object_type="cube1",
            gesture="01",
            rerun_output=None,
            use_residual=True,
            checkpoint=checkpoint,
            reference_fps=100,
        )


def test_removed_policy_contract_cli_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--policy-contract", "auto"])


def test_viewer_requires_a_graphical_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from sim.manorl.view_environment import _require_graphical_session

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="graphical session"):
        _require_graphical_session()


def test_approach_prefix_reinstall_replaces_reference_and_resets_runtime() -> None:
    from sim.manorl.view_environment import _reinstall_approach_prefix_reference

    calls = {"build_reference_tables": 0, "initial_qpos": 0}
    old_trajectory = SimpleNamespace(
        identity=SimpleNamespace(identity="banana_01_052"),
        hand_sides=("right",),
        reference_fps=120,
    )
    new_trajectory = SimpleNamespace(
        identity=SimpleNamespace(identity="banana_01_052"),
        hand_sides=("right",),
        reference_fps=120,
    )
    environment = SimpleNamespace(
        config=SimpleNamespace(num_envs=1, reference_fps=120),
        is_heterogeneous=False,
        trajectory=old_trajectory,
        trajectories=(old_trajectory,),
        early_phase_lengths=np.asarray([90], dtype=np.int64),
        _build_reference_tables=lambda: calls.__setitem__("build_reference_tables", calls["build_reference_tables"] + 1),
        _initial_qpos=lambda: calls.__setitem__("initial_qpos", calls["initial_qpos"] + 1),
    )
    reset_seeds = []

    def fake_reset(*, seed):
        reset_seeds.append(seed)
        return (np.zeros(3), {"seed": seed})

    runtime = SimpleNamespace(env=SimpleNamespace(reset=fake_reset))
    stepper = SimpleNamespace(
        _runtime=runtime,
        _observations=None,
        _pending_done=object(),
        _policy_enable_steps=None,
    )

    _reinstall_approach_prefix_reference(
        environment, stepper, new_trajectory, seed=63
    )

    assert environment.trajectory is new_trajectory
    assert environment.trajectories == (new_trajectory,)
    assert calls == {"build_reference_tables": 1, "initial_qpos": 1}
    assert reset_seeds == [63]
    np.testing.assert_array_equal(stepper._observations, np.zeros(3))
    assert stepper._pending_done is None
    np.testing.assert_array_equal(stepper._policy_enable_steps, np.asarray([90]))


def test_approach_prefix_reinstall_rejects_identity_change() -> None:
    from sim.manorl.view_environment import _reinstall_approach_prefix_reference

    environment = SimpleNamespace(
        config=SimpleNamespace(num_envs=1, reference_fps=120),
        is_heterogeneous=False,
        trajectory=SimpleNamespace(
            identity=SimpleNamespace(identity="banana_01_052"),
            hand_sides=("right",),
            reference_fps=120,
        ),
    )
    stepper = SimpleNamespace(_runtime=SimpleNamespace(env=SimpleNamespace(reset=lambda **_: (None, None))))
    with pytest.raises(RuntimeError, match="accepted source identity"):
        _reinstall_approach_prefix_reference(
            environment,
            stepper,
            SimpleNamespace(
                identity=SimpleNamespace(identity="cube1_01_0001"),
                hand_sides=("right",),
                reference_fps=120,
            ),
            seed=64,
        )


def test_approach_prefix_viewer_cli_validation(tmp_path: Path) -> None:
    module = _load_tool("view_manorl_approach_prefix")
    predecode = tmp_path / "predecode"
    predecode.mkdir()
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--checkpoint",
                str(tmp_path / "checkpoint.pt"),
                "--accepted-parent",
                str(tmp_path / "parent.json"),
                "--predecode-dir",
                str(predecode),
                "--seed",
                "-1",
            ]
        )
    (predecode / "manifest.json").write_text("{}", encoding="utf-8")
    args = module.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--accepted-parent",
            str(tmp_path / "parent.json"),
            "--predecode-dir",
            str(predecode),
            "--seed",
            "49",
            "--max-episodes",
            "3",
        ]
    )
    assert args.seed == 49
    assert args.max_episodes == 3
    assert args.predecode_dir == predecode
