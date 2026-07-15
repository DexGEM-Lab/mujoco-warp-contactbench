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
        device={"torch": "2.13.0", "skrl": "cuda", "jax": "gpu"},
    )

    assert json.loads(json.dumps(config)) == config
    assert config["training_budget"]["planned_transitions"] == budget.transitions
    assert config["reward"]["ppo_scale"] == 1.0
    assert config["trajectory_assignments"][0]["identity"] == "cube1_01_009"
    assert config["device"]["skrl"] == "cuda"


def test_wandb_update_metrics_use_monotonic_transition_steps() -> None:
    tool = _load_tool()
    run = FakeRun()
    tool._log_wandb_update(run, {
        "update": 1.0,
        "environment_transitions": 3072.0,
        "reward_mean": 1.0,
        "action_abs_mean": 0.2,
        "reset_count": 3.0,
        "elapsed_seconds": 1.5,
    })
    tool._log_wandb_update(run, {
        "update": 2.0,
        "environment_transitions": 6144.0,
        "reward_mean": 2.0,
        "action_abs_mean": 0.3,
        "reset_count": 4.0,
        "elapsed_seconds": 3.0,
    })

    assert [step for _, step in run.logs] == [3072, 6144]
    assert [metrics["transitions"] for metrics, _ in run.logs] == [3072, 6144]
    assert all({"reward_mean", "action_abs_mean", "reset_count", "elapsed_seconds", "update"} <= metrics.keys()
               for metrics, _ in run.logs)


def test_wandb_evaluation_summaries_include_all_modes() -> None:
    tool = _load_tool()
    run = FakeRun()
    results = [
        tool.EvaluationResult(
            mode=mode,
            calls=10,
            return_mean=1.0,
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
        for mode in ("zero", "untrained", "trained")
    ]

    tool._log_wandb_evaluations(run, results, transitions=6144)

    assert run.logs[0][1] == 6144
    assert run.summary["evaluation/zero/return_mean"] == 1.0
    assert run.summary["evaluation/untrained/return_mean"] == 1.0
    assert run.summary["evaluation/trained/return_mean"] == 1.0


def test_wandb_artifact_contains_existing_training_outputs(tmp_path: Path) -> None:
    tool = _load_tool()
    run = FakeRun()
    artifact_paths = [
        tmp_path / "training.pt",
        tmp_path / "training.pt.json",
        tmp_path / "training.json",
        tmp_path / "training.eval.npz",
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
