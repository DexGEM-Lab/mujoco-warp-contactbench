from types import SimpleNamespace

import numpy as np

from sim.manorl.autonomy_telemetry import TelemetryAccumulator, configure_wandb_axis, latest_ppo_metrics, log_update
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
    metrics = accumulator.reduce(update=1, transitions=2, update_elapsed_seconds=0.5, agent=agent)
    assert metrics["reward_mean"] == 3.0
    assert metrics["manorl/object_motion_mean"] == 1.5
    assert metrics["physics/path_rmse"] > 0.0
    assert metrics["physics/contact_frames"] == 1.0
    assert metrics["losses/a_loss"] == 0.25
    assert metrics["info/exact_kl_mean"] == 0.03
    assert "info/clip_fraction" not in metrics


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
    assert metrics["task_success"] is False
    assert isinstance(metrics["max_lift_endpose_diagnostic"], bool)
