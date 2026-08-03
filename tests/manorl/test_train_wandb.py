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
        output=Path("outputs/manorl/cube1"),
        budget=tool.TrainingBudget(wandb=tool.WandbOptions(enabled=False)),
        config={},
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
        "project": "mujoco-mano",
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
    budget = tool.TrainingBudget(
        joint_scale_multiplier=1.5,
        joint_max_offset_multiplier=1.5,
        wandb=tool.WandbOptions(enabled=True, entity=""),
    )
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
    assert config["wandb"]["primary_axis"] == "completed_ppo_updates"
    assert config["wandb"]["secondary_metrics"] == ["transitions"]
    assert config["wandb"]["shared_policy_run"] is True
    assert config["wandb"]["aggregation_levels"] == ["global", "object", "object_action"]
    assert config["wandb"]["object_label"] == "object_{object}"
    assert config["wandb"]["object_action_label"] == "{object}_{action:02d}"
    assert "contact_reward_instant" in config["wandb"]["grouped_instant_metrics"]
    assert "episode_reward" in config["wandb"]["grouped_episode_metrics"]
    assert config["reward"]["ppo_scale"] == 0.5
    assert config["reward"]["contact_force_threshold_N"] == 0.2
    assert config["environment"]["contract"] == tool.ENVIRONMENT_CONTRACT_ID
    assert config["environment"]["reference_fps"] == 120
    assert config["environment"]["control_fps"] == 120
    assert config["environment"]["control_timestep_seconds"] == 1.0 / 120.0
    assert config["environment"]["physics_fps"] == 480
    assert config["environment"]["physics_substeps_per_control"] == 4
    assert config["environment"]["pre_padding"] == 180
    assert config["environment"]["post_padding"] == 250
    assert config["environment"]["observation_contact_threshold_N"] == 0.2
    assert config["environment"]["residual_action"]["position_scale"] == [0.003, 0.003, 0.003]
    assert config["environment"]["residual_action"]["max_position_offset"] == [0.03, 0.03, 0.03]
    assert config["environment"]["residual_action"]["joint_scale"][:6] == [
        0.02,
        0.02,
        0.008,
        0.02,
        0.01,
        0.005,
    ]
    assert config["environment"]["residual_action"]["joint_scale"][7] == 0.0025
    assert config["environment"]["residual_action"]["max_joint_offset"][2] == 0.08
    assert config["environment"]["residual_action"]["max_joint_offset"][7] == 0.025
    assert config["environment"]["residual_action"]["joint_scale_multiplier"] == 1.5
    assert config["environment"]["residual_action"]["joint_max_offset_multiplier"] == 1.5
    assert config["environment"]["residual_action"]["early_phase_steps"] == 30
    assert config["environment"]["max_deviation_distance"] == 0.10
    assert config["trajectory_assignments"][0]["identity"] == "cube1_01_009"
    assert config["evaluation"]["num_envs"] == 1
    assert config["evaluation"]["ppo_config"]["minibatch_size"] == 768
    assert config["device"]["skrl"] == "cuda"


def test_wandb_update_metrics_use_completed_update_steps() -> None:
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
        "manorl/reward_mean": 1.0,
        "manorl/total_mean": 1.0,
        "manorl/distance_x_mean": 0.1,
        "manorl/distance_y_mean": 0.2,
        "manorl/distance_z_mean": 0.3,
        "manorl/rotation_mean": 0.4,
        "manorl/action_penalty_mean": 0.0,
        "manorl/contact_mean": 0.5,
        "manorl/object_stability_mean": 0.6,
        "manorl/survival_mean": 0.001,
        "manorl/deviation_penalty_mean": 0.0,
        "elapsed_seconds": 1.5,
        "update_environment_transitions_per_second": 2048.0,
        "cumulative_environment_transitions_per_second": 2048.0,
        "performance/total_fps": 2048.0,
        "performance/step_fps": 2048.0,
        "performance/update_time": 1.0,
        "performance/play_time": 0.8,
        "info/epochs": 1.0,
        "distance_reward_x_instant/step": 0.1,
        "distance_reward_y_instant/step": 0.2,
        "distance_reward_z_instant/step": 0.3,
        "distance_reward_instant/step": 0.6,
        "rotation_reward_instant/step": 0.4,
        "action_penalty_instant/step": 0.0,
        "contact_reward_instant/step": 0.5,
        "object_stability_reward_instant/step": 0.6,
        "survival_reward_instant/step": 0.001,
    })
    tool._log_wandb_update(run, wandb, {
        "update": 2.0,
        "environment_transitions": 6144.0,
        "reward_mean": 2.0,
        "action_abs_mean": 0.3,
        "reset_count": 4.0,
        "completed_episode_count": 1.0,
        "episode_return_mean": 12.5,
        "episode_total_mean": 12.5,
        "episode_total_min": 9.0,
        "episode_total_max": 16.0,
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
        "manorl/reward_mean": 2.0,
        "manorl/total_mean": 2.0,
        "manorl/distance_x_mean": 0.2,
        "manorl/distance_y_mean": 0.3,
        "manorl/distance_z_mean": 0.4,
        "manorl/rotation_mean": 0.5,
        "manorl/action_penalty_mean": 0.0,
        "manorl/contact_mean": 0.6,
        "manorl/object_stability_mean": 0.7,
        "manorl/survival_mean": 0.001,
        "manorl/deviation_penalty_mean": 0.0,
        "elapsed_seconds": 3.0,
        "update_environment_transitions_per_second": 2048.0,
        "cumulative_environment_transitions_per_second": 2048.0,
        "performance/total_fps": 2048.0,
        "performance/step_fps": 2048.0,
        "performance/update_time": 1.5,
        "performance/play_time": 1.2,
        "info/epochs": 2.0,
        "performance/algorithm_update_time_ms": 42.0,
        "rewards/frame": 12.5,
        "rewards/iter": 12.5,
        "rewards/step": 12.5,
        "rewards/time": 12.5,
        "episode_reward": 12.5,
        "distance_reward": 15.0,
        "distance_reward/cube1_01": 15.0,
        "distance_reward/object_cube1": 15.0,
        "distance_reward_x": 4.0,
        "distance_reward_y": 5.0,
        "distance_reward_z": 6.0,
        "rotation_reward": 7.0,
        "action_penalty": -1.0,
        "contact_reward": 8.0,
        "object_stability_reward": 9.0,
        "survival_reward": 10.0,
        "episode_lengths/frame": 48.0,
        "episode_lengths/iter": 48.0,
        "episode_lengths/step": 48.0,
        "episode_cumulative/distance_reward_x": 4.0,
        "episode_cumulative/distance_reward_y": 5.0,
        "episode_cumulative/distance_reward_z": 6.0,
        "episode_cumulative/rotation_reward": 7.0,
        "episode_cumulative/action_penalty": -1.0,
        "episode_cumulative/contact_reward": 8.0,
        "episode_cumulative/object_stability_reward": 9.0,
        "episode_cumulative/survival_reward": 10.0,
        "episode_cumulative_min/episode_reward_min": 1.0,
        "episode_cumulative_max/episode_reward_max": 24.0,
        "episode_cumulative_min/distance_reward_x_min": 2.0,
        "episode_cumulative_max/distance_reward_x_max": 6.0,
        "episode_reward/cube1_01": 12.5,
        "episode_reward/object_cube1": 12.5,
        "episode_cumulative/distance_reward_x/cube1_01": 4.0,
        "episode_cumulative_min/distance_reward_x/cube1_01_min": 2.0,
        "episode_cumulative_max/distance_reward_x/cube1_01_max": 6.0,
        "distance_reward_x_instant/step": 0.2,
        "distance_reward_y_instant/step": 0.3,
        "distance_reward_z_instant/step": 0.4,
        "distance_reward_instant/step": 0.9,
        "rotation_reward_instant/step": 0.5,
        "action_penalty_instant/step": 0.0,
        "contact_reward_instant/step": 0.6,
        "object_stability_reward_instant/step": 0.7,
        "survival_reward_instant/step": 0.001,
        "losses/a_loss": 0.25,
        "losses/c_loss": 0.5,
        "losses/entropy": -0.01,
        "info/last_lr": 0.0003,
        "info/policy_std": 0.8,
        "grouped_metrics": {
            "reward_mean/object_cube1": 1.5,
            "reward_mean/cube1_01": 2.5,
            "success_rate/cube1_01": 50.0,
        },
    })

    assert [step for _, step in run.logs] == [1, 2]
    assert [metrics["update"] for metrics, _ in run.logs] == [1, 2]
    assert [metrics["global_step"] for metrics, _ in run.logs] == [1, 2]
    assert [metrics["transitions"] for metrics, _ in run.logs] == [3072, 6144]
    assert all({"reward_mean", "manorl/reward_mean", "manorl/total_mean", "manorl/distance_x_mean", "manorl/distance_y_mean", "manorl/distance_z_mean", "manorl/rotation_mean", "manorl/action_penalty_mean", "manorl/contact_mean", "manorl/object_stability_mean", "manorl/survival_mean", "manorl/deviation_penalty_mean", "action_abs_mean", "reset_count", "completed_episode_count", "elapsed_seconds", "update", "performance/total_fps", "performance/step_fps", "performance/update_time", "performance/play_time", "info/epochs", "distance_reward_x_instant/step", "distance_reward_y_instant/step", "distance_reward_z_instant/step", "distance_reward_instant/step", "rotation_reward_instant/step", "action_penalty_instant/step", "contact_reward_instant/step", "object_stability_reward_instant/step", "survival_reward_instant/step"} <= metrics.keys()
               for metrics, _ in run.logs)
    assert not {"rewards/frame", "rewards/iter", "rewards/step", "rewards/time"} & run.logs[0][0].keys()
    assert not {"episode_reward", "distance_reward", "distance_reward_x", "distance_reward/cube1_01"} & run.logs[0][0].keys()
    assert "episode_return_mean" not in run.logs[0][0]
    assert not any(name.startswith("episode_cumulative/") for name in run.logs[0][0])
    assert run.logs[1][0]["episode_return_mean"] == 12.5
    assert run.logs[1][0]["total"] == 2.0
    assert run.logs[1][0]["distance_x"] == 0.2
    assert run.logs[1][0]["action_penalty"] == 0.0
    assert run.logs[1][0]["manorl/reward_mean"] == 2.0
    assert run.logs[1][0]["performance/play_time"] == 1.2
    assert run.logs[1][0]["info/epochs"] == 2.0
    assert run.logs[1][0]["episode_reward"] == 12.5
    assert run.logs[1][0]["distance_reward_x"] == 4.0
    assert run.logs[1][0]["distance_reward_y"] == 5.0
    assert run.logs[1][0]["distance_reward_z"] == 6.0
    assert run.logs[1][0]["rotation_reward"] == 7.0
    assert run.logs[1][0]["contact_reward"] == 8.0
    assert run.logs[1][0]["object_stability_reward"] == 9.0
    assert run.logs[1][0]["survival_reward"] == 10.0
    assert run.logs[1][0]["action_penalty"] == 0.0
    assert run.logs[1][0]["distance_reward"] == 15.0
    assert run.logs[1][0]["distance_reward/cube1_01"] == 15.0
    assert run.logs[1][0]["distance_reward/object_cube1"] == 15.0
    assert run.logs[1][0]["episode_lengths/frame"] == 48.0
    assert run.logs[1][0]["episode_lengths/iter"] == 48.0
    assert run.logs[1][0]["episode_lengths/step"] == 48.0
    assert run.logs[1][0]["rewards/frame"] == 12.5
    assert run.logs[1][0]["rewards/iter"] == 12.5
    assert run.logs[1][0]["rewards/step"] == 12.5
    assert run.logs[1][0]["rewards/time"] == 12.5
    assert run.logs[1][0]["episode_cumulative/distance_reward_x"] == 4.0
    assert run.logs[1][0]["episode_cumulative_min/distance_reward_x_min"] == 2.0
    assert run.logs[1][0]["episode_cumulative_max/distance_reward_x_max"] == 6.0
    assert run.logs[1][0]["episode_reward/cube1_01"] == 12.5
    assert run.logs[1][0]["episode_reward/object_cube1"] == 12.5
    assert run.logs[1][0]["episode_cumulative/distance_reward_x/cube1_01"] == 4.0
    assert run.logs[1][0]["episode_cumulative_min/distance_reward_x/cube1_01_min"] == 2.0
    assert run.logs[1][0]["episode_cumulative_max/distance_reward_x/cube1_01_max"] == 6.0
    assert run.logs[1][0]["episode_cumulative/action_penalty"] == -1.0
    assert run.logs[1][0]["episode_cumulative/total"] == 12.5
    assert run.logs[1][0]["episode_cumulative/episode_reward"] == 12.5
    assert run.logs[1][0]["episode_cumulative/total_mean"] == 12.5
    assert run.logs[1][0]["episode_cumulative/total_min"] == 9.0
    assert run.logs[1][0]["episode_cumulative/total_max"] == 16.0
    assert run.logs[1][0]["episode_return_distribution"] == ("histogram", [9.0, 16.0])
    assert run.logs[1][0]["losses/a_loss"] == 0.25
    assert run.logs[1][0]["losses/c_loss"] == 0.5
    assert run.logs[1][0]["losses/entropy"] == -0.01
    assert run.logs[1][0]["info/last_lr"] == 0.0003
    assert run.logs[1][0]["info/policy_std"] == 0.8
    assert run.logs[1][0]["performance/algorithm_update_time_ms"] == 42.0
    assert not any(name.startswith("mu/") for name in run.logs[1][0])
    assert run.logs[1][0]["reward_mean/object_cube1"] == 1.5
    assert run.logs[1][0]["reward_mean/cube1_01"] == 2.5
    assert run.logs[1][0]["success_rate/cube1_01"] == 50.0


def test_latest_skrl_tracking_metrics_maps_only_available_latest_values() -> None:
    tool = _load_tool()
    agent = SimpleNamespace(tracking_data={
        "Loss / Policy loss": [1.0, 0.25],
        "Loss / Value loss": [0.5],
        "Learning / Learning rate": [0.0003],
        "Loss / Bounds loss": [0.125],
        "Learning / LR multiplier": [0.75],
        "Info / E-clip": [0.15],
        "Stats / Clip fraction": [0.2],
        "Info / KL": [0.01],
        "Unknown / Value": [99.0],
    })

    assert tool._latest_skrl_tracking_metrics(agent) == {
        "losses/a_loss": 0.25,
        "losses/c_loss": 0.5,
        "losses/bounds_loss": 0.125,
        "info/last_lr": 0.0003,
        "info/lr_mul": 0.75,
        "info/e_clip": 0.15,
        "info/clip_frac": 0.2,
        "info/kl": 0.01,
    }


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
            groups=(tool.EvaluationGroupResult(
                label="cube1_01",
                kind="object_action",
                num_envs=1,
                return_mean=return_mean,
                reward_mean=0.1,
                action_abs_mean=0.2,
                final_object_target_distance=0.3,
                max_object_target_distance=0.4,
                contact_reward_mean=0.5,
                success_count=1,
                failure_count=0,
            ),),
        )

    tool._log_wandb_evaluations(
        run, [result("zero", 0.0), result("untrained", 1.0)], update=0, transitions=0
    )
    tool._log_wandb_evaluations(run, [result("trained", 2.0)], update=2, transitions=6144)

    assert [step for _, step in run.logs] == [0, 2]
    assert [metrics["update"] for metrics, _ in run.logs] == [0, 2]
    assert [metrics["global_step"] for metrics, _ in run.logs] == [0, 2]
    assert [metrics["transitions"] for metrics, _ in run.logs] == [0, 6144]
    assert run.logs[0][0]["evaluation/zero/return_mean"] == 0.0
    assert run.logs[0][0]["evaluation/untrained/return_mean"] == 1.0
    assert run.logs[0][0]["evaluation/policy/return_mean"] == 1.0
    assert run.logs[1][0]["evaluation/trained/return_mean"] == 2.0
    assert run.logs[1][0]["evaluation/policy/return_mean"] == 2.0
    assert run.logs[1][0]["evaluation/trained/return_mean/cube1_01"] == 2.0
    assert run.logs[1][0]["evaluation/policy/return_mean/cube1_01"] == 2.0
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
        "--wandb", "false",
        "--wandb-project", "project",
        "--wandb-group", "group",
        "--wandb-entity", "entity",
        "--wandb-name", "name",
        "--wandb-tags", "first,second",
        "--wandb-tags", "third",
    ]) == 0
    override = captured[-1][1].wandb
    assert override == tool.WandbOptions(
        enabled=False,
        project="project",
        group="group",
        entity="entity",
        name="name",
        tags=("first", "second", "third"),
    )
