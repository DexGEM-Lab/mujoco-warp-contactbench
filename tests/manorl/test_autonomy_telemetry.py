from types import SimpleNamespace

import numpy as np

from sim.manorl.autonomy_telemetry import TelemetryAccumulator, configure_wandb_axis, genuine_airborne_contact, latest_ppo_metrics, log_update
from tools.publish_manorl_autonomy_evaluation import summarize_trace


def _info(step, *, contact=False):
    force = np.zeros((16, 3), dtype=float)
    if contact:
        force[0, 2] = 0.3
    return {
        "object_position": np.array([0.0, 0.0, 0.02 + step * 0.01]),
        "target_object_position": np.array([0.0, 0.0, 0.02 + step * 0.02]),
        "object_quaternion_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
        "target_object_quaternion_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
        "hand_object_force": force,
        "relative_contact_motion": np.ones((16, 3)) * 0.1,
        "actual_peak_lift": step * 0.01,
        "target_peak_lift": 0.1,
        "failure_phase": "running",
        "terms": {"object_motion": 1.0 + step},
    }


def test_telemetry_reduces_physics_terms_and_native_ppo_without_inventing_clip_fraction():
    accumulator = TelemetryAccumulator()
    accumulator.add(_info(0, contact=True), 2.0)
    accumulator.add(_info(1, contact=False), 4.0)
    agent = SimpleNamespace(tracking_data={"Loss / Policy loss": [0.25], "Learning / Exact KL mean": [0.03]})
    metrics = accumulator.reduce(update=1, transitions=20, window_transitions=2, update_elapsed_seconds=0.5, agent=agent)
    assert metrics["reward_mean"] == 3.0
    assert metrics["manorl/object_motion_mean"] == 1.5
    assert metrics["performance/step_fps"] == 4.0
    assert metrics["physics/path_rmse"] > 0.0
    assert metrics["physics/contact_frames"] == 1.0
    assert metrics["losses/a_loss"] == 0.25
    assert metrics["info/exact_kl_mean"] == 0.03
    assert "info/clip_fraction" not in metrics


def test_airborne_requires_rotated_collision_bottom_clearance():
    force = np.zeros((16, 3)); force[0, 2] = 0.3
    resting = _info(0, contact=True); resting["hand_object_force"] = force
    assert genuine_airborne_contact(resting) is False
    lifted = dict(resting); lifted["object_position"] = np.array([0.0, 0.0, 0.04])
    assert genuine_airborne_contact(lifted) is True


def test_episode_counters_cross_update_windows_and_only_complete_on_terminal():
    acc = TelemetryAccumulator()
    acc.add(_info(0), 2.0)
    first = acc.reduce(update=1, transitions=1, window_transitions=1, update_elapsed_seconds=1.0)
    assert "episodes/length_mean" not in first
    acc.clear()
    acc.add(_info(1), 3.0, terminated=True)
    second = acc.reduce(update=2, transitions=2, window_transitions=1, update_elapsed_seconds=1.0)
    assert second["episodes/length_mean"] == 2.0
    assert second["episodes/return_mean"] == 5.0


def test_wandb_uses_transition_axis():
    class Run:
        def __init__(self): self.defined = []; self.logged = []
        def define_metric(self, *args, **kwargs): self.defined.append((args, kwargs))
        def log(self, values, *, step): self.logged.append((values, step))
    run = Run(); configure_wandb_axis(run); log_update(run, {"transitions": 8, "reward_mean": 1.0})
    assert run.defined[-1] == (("*",), {"step_metric": "transitions"})
    assert run.logged[0][1] == 8


def test_publication_metrics_are_actual_trace_metrics_with_denominators():
    trace = {"identity": "cube2_02_2833", "checkpoint_format": "x", "trace": [
        {"info": _info(0, contact=True)}, {"info": _info(1, contact=True)}, {"info": _info(2, contact=False)}
    ]}
    metrics = summarize_trace(trace)
    assert metrics["frames"] == 3
    assert metrics["hand_object_contact_frames"] == 2
    assert metrics["hand_object_contact_denominator_frames"] == 3
    assert metrics["sustained_hand_object_contact_frames"] == 2
    assert metrics["full_task_success"] is None
    assert "orientation_error_rad_mean" in metrics
    assert isinstance(metrics["max_lift_endpose_diagnostic"], bool)


def test_resumed_wandb_publication_uses_automatic_step_video_and_table(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    import tools.publish_manorl_autonomy_evaluation as publisher

    class Run:
        def __init__(self): self.logs = []; self.artifacts = []
        def log(self, payload, **kwargs): self.logs.append((payload, kwargs))
        def log_artifact(self, artifact): self.artifacts.append(artifact)
        def finish(self): pass
    run = Run()
    class ApiRun: state = "finished"
    class Api:
        def run(self, path): return ApiRun()
    class Artifact:
        def __init__(self, *args, **kwargs): pass
        def add_file(self, *args, **kwargs): pass
    fake = SimpleNamespace(Api=Api, init=lambda **kwargs: run, Artifact=Artifact,
        Video=lambda *args, **kwargs: ("video", args, kwargs),
        Table=lambda **kwargs: ("table", kwargs))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    path = tmp_path / "x.mp4"; path.write_bytes(b"x")
    publisher.publish_wandb({"frames": 538, "checkpoint_transitions": 8192, "identity_table": [{"identity": "x", "frames": 1}]}, run_id="abc", project="p", entity="e", json_path=path, video_path=path)
    payload, kwargs = run.logs[0]
    assert kwargs == {}
    assert payload["evaluation/checkpoint_transitions"] == 8192
    assert payload["evaluation/video"][0] == "video"
    assert payload["evaluation/identity_table"][0] == "table"
