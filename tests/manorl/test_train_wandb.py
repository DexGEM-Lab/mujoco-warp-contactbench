from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_tool():
    path = Path(__file__).parents[2] / "tools" / "train_manorl_cube1.py"
    spec = importlib.util.spec_from_file_location("train_manorl_cube1_wandb_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeArtifact:
    def __init__(self, *, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[tuple[str, str]] = []

    def add_file(self, path: str, name: str) -> None:
        self.files.append((path, name))


class FakeRun:
    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, object], int]] = []
        self.summary: dict[str, object] = {}
        self.finished: list[dict[str, object]] = []
        self.artifacts: list[FakeArtifact] = []
        self.finish_error: BaseException | None = None

    def log(self, values: dict[str, object], *, step: int) -> None:
        self.logs.append((values, step))

    def finish(self, **kwargs: object) -> None:
        self.finished.append(kwargs)
        if self.finish_error is not None:
            raise self.finish_error

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)


class FakeWandb:
    Artifact = FakeArtifact

    @staticmethod
    def Histogram(values: list[float]) -> tuple[str, list[float]]:
        return ("histogram", values)

    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_calls: list[dict[str, object]] = []

    def init(self, **kwargs: object) -> FakeRun:
        self.init_calls.append(kwargs)
        return self.run


def _budget(tool, **wandb_overrides: object):
    return tool.TrainingBudget(wandb=tool.WandbOptions(enabled=True, **wandb_overrides))


def test_wandb_disabled_never_imports_or_initializes(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    with tool._wandb_run(
        output=Path("outputs/manorl/cube1"), budget=tool.TrainingBudget(), config={}
    ) as (run, wandb):
        assert run is None
        assert wandb is None

    assert fake.init_calls == []


def test_wandb_defaults_and_overrides_map_to_init(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    default_budget = _budget(tool)

    with tool._wandb_run(
        output=Path("outputs/manorl/cube1_03"), budget=default_budget, config={"seed": 42}
    ) as (run, _):
        assert run is fake.run

    assert fake.init_calls == [{
        "project": "one_policy",
        "entity": None,
        "group": "s02",
        "name": "cube1_03-cube1-01",
        "tags": ["manorl", "mujoco", "skrl"],
        "config": {"seed": 42},
        "dir": "outputs/manorl",
    }]
    assert fake.run.finished == [{}]

    override_fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", override_fake)
    override_budget = _budget(
        tool,
        project="project",
        group="group",
        entity="entity",
        name="explicit-name",
        tags=("custom",),
    )
    with tool._wandb_run(output=Path("output"), budget=override_budget, config={}) as (run, _):
        assert run is override_fake.run

    assert override_fake.init_calls[0]["project"] == "project"
    assert override_fake.init_calls[0]["group"] == "group"
    assert override_fake.init_calls[0]["entity"] == "entity"
    assert override_fake.init_calls[0]["name"] == "explicit-name"
    assert override_fake.init_calls[0]["tags"] == ["custom"]


def test_wandb_initialization_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()

    def fail_init(**kwargs: object) -> None:
        del kwargs
        raise OSError("authentication unavailable")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fail_init))
    with pytest.raises(RuntimeError, match="W&B initialization failed"):
        with tool._wandb_run(output=Path("output"), budget=_budget(tool), config={}):
            pass


def test_wandb_failure_cleanup_preserves_training_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    fake = FakeWandb()
    fake.run.finish_error = OSError("cleanup unavailable")
    monkeypatch.setitem(sys.modules, "wandb", fake)

    with pytest.raises(RuntimeError, match="training failed") as raised:
        with tool._wandb_run(output=Path("output"), budget=_budget(tool), config={}):
            raise RuntimeError("training failed")

    assert fake.run.finished == [{"exit_code": 1}]
    assert "cleanup unavailable" in "\n".join(raised.value.__notes__)


def test_wandb_successful_body_propagates_cleanup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    fake = FakeWandb()
    fake.run.finish_error = OSError("cleanup unavailable")
    monkeypatch.setitem(sys.modules, "wandb", fake)

    with pytest.raises(RuntimeError, match="cleanup after successful training") as raised:
        with tool._wandb_run(output=Path("output"), budget=_budget(tool), config={}):
            pass

    assert isinstance(raised.value.__cause__, OSError)
    assert fake.run.finished == [{}]


def test_wandb_uses_output_parent_without_root_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tool = _load_tool()
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    output = tmp_path / "training-output" / "cube1"

    with tool._wandb_run(output=output, budget=_budget(tool), config={}):
        pass

    assert output.parent.is_dir()
    assert fake.init_calls[0]["dir"] == str(output.parent)
    assert not (tmp_path / "wandb").exists()


def test_wandb_config_is_complete_and_json_serializable() -> None:
    tool = _load_tool()
    budget = _budget(tool, entity="")
    config = tool._wandb_config(
        budget=budget,
        ppo_config=tool.ManoPPOConfig(),
        trajectory_assignments=[{"env_id": 0, "identity": "cube1_01_009", "row_index": 1}],
        evaluation_ppo_config=tool.ManoPPOConfig(minibatch_size=768),
        evaluation_trajectory_assignments=[{"env_id": 0, "identity": "cube1_01_009", "row_index": 1}],
        device={"torch": "2.13.0", "skrl": "cuda", "jax": "gpu"},
    )

    assert json.loads(json.dumps(config)) == config
    assert config["training_budget"]["planned_transitions"] == budget.transitions
    assert config["training_budget"]["wall_clock_seconds"] is None
    assert config["reward"]["ppo_scale"] == 1.0
    assert config["environment"]["contract"] == "target_residual_xyz_0p003_gamma_0p9_cap_0p03_deviation_0p15_v1"
    assert config["environment"]["residual_action"]["position_scale"] == [0.003, 0.003, 0.003]
    assert config["environment"]["max_deviation_distance"] == 0.15
    assert config["trajectory_assignments"][0]["identity"] == "cube1_01_009"
    assert config["evaluation"]["num_envs"] == 64
    assert config["evaluation"]["ppo_config"]["minibatch_size"] == 768
    assert config["device"]["skrl"] == "cuda"


def test_wandb_update_metrics_use_monotonic_transition_steps() -> None:
    tool = _load_tool()
    run = FakeRun()
    wandb = FakeWandb()
    tool._log_wandb_update(run, wandb, {
        "update": 1.0,
        "environment_transitions": 3072.0,
        "reward_mean": 1.0,
        "action_abs_mean": 0.2,
        "reset_count": 3.0,
        "completed_episode_count": 0.0,
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
        "elapsed_seconds": 1.5,
        "update_environment_transitions_per_second": 2048.0,
        "cumulative_environment_transitions_per_second": 2048.0,
    })
    tool._log_wandb_update(run, wandb, {
        "update": 2.0,
        "environment_transitions": 6144.0,
        "reward_mean": 2.0,
        "action_abs_mean": 0.3,
        "reset_count": 4.0,
        "completed_episode_count": 1.0,
        "episode_return_mean": 12.5,
        "episode_return_values": [9.0, 16.0],
        "total": 2.0,
        "distance_x": 0.2,
        "distance_y": 0.3,
        "distance_z": 0.4,
        "rotation": 0.5,
        "action_penalty": 0.0,
        "contact": 0.6,
        "object_stability": 0.7,
        "survival": 0.001,
        "deviation_penalty": 0.0,
        "elapsed_seconds": 3.0,
        "update_environment_transitions_per_second": 2048.0,
        "cumulative_environment_transitions_per_second": 2048.0,
    })

    assert [step for _, step in run.logs] == [3072, 6144]
    assert [metrics["transitions"] for metrics, _ in run.logs] == [3072, 6144]
    assert all({"reward_mean", "action_abs_mean", "reset_count", "completed_episode_count", "elapsed_seconds", "update", "total", "distance_x", "distance_y", "distance_z", "rotation", "action_penalty", "contact", "object_stability", "survival", "deviation_penalty", "update_environment_transitions_per_second", "cumulative_environment_transitions_per_second"} <= metrics.keys()
               for metrics, _ in run.logs)
    assert "episode_return_mean" not in run.logs[0][0]
    assert run.logs[1][0]["episode_return_mean"] == 12.5
    assert run.logs[1][0]["episode_return_distribution"] == ("histogram", [9.0, 16.0])


def test_wandb_evaluation_summaries_preserve_policy_alias_for_both_comparisons() -> None:
    tool = _load_tool()
    run = FakeRun()

    def result(mode: str, return_mean: float) -> object:
        return tool.EvaluationResult(
            mode=mode,
            calls=10,
            return_mean=return_mean,
            reward_mean=0.1,
            action_abs_mean=0.2,
            final_object_target_distance=0.3,
            max_object_target_distance=0.4,
            contact_reward_mean=0.5,
            reset_seen=False,
            timeout_seen=False,
            completed_horizon=True,
            rewards_by_call=[0.1],
            object_target_distance_by_call=[0.3],
        )

    tool._log_wandb_evaluations(run, [result("zero", 0.0), result("untrained", 1.0)], transitions=0)
    tool._log_wandb_evaluations(run, [result("trained", 2.0)], transitions=6144)

    assert [step for _, step in run.logs] == [0, 6144]
    assert run.logs[0][0]["evaluation/zero/return_mean"] == 0.0
    assert run.logs[0][0]["evaluation/untrained/return_mean"] == 1.0
    assert run.logs[0][0]["evaluation/policy/return_mean"] == 1.0
    assert run.logs[1][0]["evaluation/trained/return_mean"] == 2.0
    assert run.logs[1][0]["evaluation/policy/return_mean"] == 2.0
    assert run.summary["evaluation/policy/return_mean"] == 2.0


def test_wandb_artifact_contains_existing_training_outputs(tmp_path: Path) -> None:
    tool = _load_tool()
    run = FakeRun()
    artifact_paths = [
        tmp_path / "training.pt",
        tmp_path / "training.pt.json",
        tmp_path / "training.json",
        tmp_path / "training.eval.npz",
        tmp_path / "training.episodes.jsonl",
        tmp_path / "training.rrd",
    ]
    for path in artifact_paths:
        path.write_text("artifact", encoding="utf-8")

    tool._log_wandb_artifacts(
        run,
        SimpleNamespace(Artifact=FakeArtifact),
        output=tmp_path / "training",
        paths=artifact_paths,
    )

    assert len(run.artifacts) == 1
    assert run.artifacts[0].name == "training-artifacts"
    assert [name for _, name in run.artifacts[0].files] == [path.name for path in artifact_paths]


def test_wandb_cli_parses_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tool = _load_tool()
    captured = []
    monkeypatch.setattr(tool, "run", lambda output, budget: captured.append((output, budget)) or {})

    assert tool.main(["--output", str(tmp_path / "default")]) == 0
    default = captured[-1][1].wandb
    assert default == tool.WandbOptions()

    assert tool.main([
        "--output", str(tmp_path / "override"),
        "--wandb", "true",
        "--wandb-project", "project",
        "--wandb-group", "group",
        "--wandb-entity", "entity",
        "--wandb-name", "name",
        "--wandb-tags", "first,second",
        "--wandb-tags", "third",
    ]) == 0
    override = captured[-1][1].wandb
    assert override == tool.WandbOptions(
        enabled=True,
        project="project",
        group="group",
        entity="entity",
        name="name",
        tags=("first", "second", "third"),
    )
