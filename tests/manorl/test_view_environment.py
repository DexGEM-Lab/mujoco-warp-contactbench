from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

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
        pass

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
        mjv_defaultCamera=lambda _: None,
        mjv_defaultOption=lambda _: None,
        mjv_moveCamera=lambda _, action, dx, dy, __, ___: state["moves"].append((action, dx, dy)),
        mjv_updateScene=lambda *_: state["updates"].append("scene"),
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


def test_training_viewer_renders_states_without_stepping_and_preserves_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    import sim.manorl.view_environment as viewer

    glfw, mujoco, state = _fake_graphics()
    monkeypatch.setitem(sys.modules, "glfw", glfw)
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setattr(viewer, "_require_graphical_session", lambda: None)
    environment, calls = _training_environment()
    training_viewer = viewer.TrainingViewer(environment, tile_envs=2)

    training_viewer.render()
    assert calls == {"host": 1}
    assert state["updates"] == ["scene", "scene"]
    assert state["renders"] == ["render", "render"]

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

    with pytest.raises(RuntimeError, match="context failed"):
        viewer.TrainingViewer(environment, tile_envs=2)
    assert state["destroy"] == 1 and state["terminate"] == 1


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
    assert (captured_configs[0].max_deviation_distance == 0.10) is expected_terminal
    assert '"rerun_artifact": null' in capsys.readouterr().out


def test_viewer_reports_missing_reset_completed_rerun_artifact(capsys: pytest.CaptureFixture[str]) -> None:
    from sim.manorl.view_environment import _close_rerun_recorder

    class FakeRecorder:
        def close(self) -> Path | None:
            return None

    assert _close_rerun_recorder(FakeRecorder()) is None
    assert capsys.readouterr().out == "No reset-complete Rerun episode was published.\n"


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
    assert environment.actions.shape == (3, 26)
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
        lambda agent, path: captured.update(path=path),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "checkpoint_runtime_metadata",
        lambda path: {"runtime_config": {"model": {"use_film": False}}},
    )

    stepper = _build_checkpoint_stepper(environment, checkpoint)
    assert captured["adapter_environment"] is environment
    assert captured["path"] == checkpoint
    assert captured["config"].rollouts == 1
    assert captured["config"].minibatch_size == 2
    assert captured["config"].use_film is False
    assert captured["training_enabled"] is False
    assert captured["eval"] is True
    assert captured["reset"] is True
    assert stepper is not None


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
    sentinel_stepper = object()

    class FakeEnvironment:
        def __init__(self, trajectory, config) -> None:
            self.config = config

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


def test_removed_policy_contract_cli_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--policy-contract", "auto"])


def test_viewer_requires_a_graphical_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from sim.manorl.view_environment import _require_graphical_session

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="graphical session"):
        _require_graphical_session()
