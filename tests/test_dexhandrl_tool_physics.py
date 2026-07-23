"""Boundary contracts for DexHand reference-PD and trace tooling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from sim.dexhandrl.contracts import DexHandPhysicsConfig, LEGACY_PHYSICS_FIELDS
from sim.dexhandrl import batch_reference_pd
from sim.dexhandrl.reference_pd_replay import _physics_from_args as replay_physics
from sim.dexhandrl.render_reference_pd_video import _physics_from_args as render_physics
from tools import trace_dexhandrl_mujoco as trace


def _replay_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "dt": 0.01, "substeps": 10,
        "base_pos_kp": 5000, "base_pos_kv": 500, "base_pos_force": 500,
        "base_rot_kp": 10000, "base_rot_kv": 125, "base_rot_force": 100,
        "finger_kp": 15, "finger_kv": 0, "finger_force": 50,
        "hand_friction": "0.5 0.01 0.001", "object_friction": "0.5 0.01 0.001",
        "ground_friction": "0.5 0.5 0", "collision": "disable_within_finger",
        "naconmax": 8192, "njmax": 8192,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_replay_and_render_resolve_complete_matching_default_profile() -> None:
    args = _replay_args()
    assert replay_physics(args) == DexHandPhysicsConfig()
    assert render_physics(args) == DexHandPhysicsConfig()


def test_replay_and_render_propagate_every_scene_and_runtime_override() -> None:
    physics = replay_physics(_replay_args(
        dt=0.02, substeps=4, base_pos_force=123, base_rot_kv=9, finger_kp=7,
        hand_friction="3 2 2", object_friction="1 2 3", ground_friction="4 5 6",
        collision="all_disabled", naconmax=64, njmax=128,
    ))
    assert physics.control_dt == 0.02 and physics.physics_dt == 0.005
    assert physics.servo.base_pos_force == 123 and physics.servo.base_rot_kv == 9
    assert physics.servo.finger_kp == 7
    assert physics.material.hand_friction == (3, 2, 2)
    assert physics.material.object_friction == (1, 2, 3)
    assert physics.material.ground_friction == (4, 5, 6)
    assert physics.hand_self_collision_mode == "all_disabled"
    assert (physics.runtime_naconmax, physics.runtime_njmax) == (64, 128)


def test_trace_parser_resolves_profile_and_source_validates_before_mjx_upload(tmp_path: Path) -> None:
    args = trace._parser().parse_args([
        "--output", str(tmp_path / "trace.npz"), "--dt", "0.02", "--substeps", "4",
        "--object-friction", "1 2 3", "--collision", "all_disabled", "--naconmax", "64",
    ])
    physics = trace.resolve_physics_config(
        dt=args.dt, substeps=args.substeps, base_rot_kp=args.base_rot_kp,
        base_rot_kv=args.base_rot_kv, base_rot_force=args.base_rot_force,
        finger_kp=args.finger_kp, finger_kv=args.finger_kv, finger_force=args.finger_force,
        hand_friction=args.hand_friction, object_friction=args.object_friction,
        ground_friction=args.ground_friction, collision=args.collision,
        naconmax=args.naconmax, njmax=args.njmax,
    )
    assert physics.control_dt == 0.02
    assert physics.material.object_friction == (1, 2, 3)
    assert physics.hand_self_collision_mode == "all_disabled"
    assert physics.runtime_naconmax == 64
    source = Path(trace.__file__).read_text(encoding="utf-8")
    assert "physics=physics" in source
    assert '"physics_contract": physics.record()' in source


def test_replay_rejects_mixed_canonical_and_legacy_namespace() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        replay_physics(_replay_args(physics=DexHandPhysicsConfig()))


def test_batch_forwards_only_canonical_physics_to_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[argparse.Namespace] = []

    def fake_replay(replay_args: argparse.Namespace) -> dict[str, object]:
        captured.append(replay_args)
        return {
            "object_tracking": {
                "actual_lift_m": 0.0, "target_lift_m": 0.0, "max_error_m": 0.0,
                "mean_error_m": 0.0, "first_error_gt_10cm_step": -1,
            },
            "row_id": 0, "uuid": "id", "object_name": "cube1", "action": "01",
            "start_frame": 0, "end_frame": 0, "movement_start_frame": 0, "movement_end_frame": 0,
            "hand_object_contact_frames": 0, "hand_object_contact_count_max": 0,
        }

    monkeypatch.setattr(batch_reference_pd, "replay", fake_replay)
    args = argparse.Namespace(
        lance=tmp_path / "input.lance", object="cube1", action="01", sequence_start=0, max_sequences=1,
        device="cpu", max_frames=1, start_frame=None, pre_contact_frames=0, object_reset_z_offset=0.01,
        dt=0.01, substeps=3, settle_frames=2, base_pos_kp=5000, base_pos_kv=500, base_pos_force=500,
        base_rot_kp=10000, base_rot_kv=125, base_rot_force=100, finger_kp=15, finger_kv=0, finger_force=50,
        hand_friction="0.5 0.01 0.001", object_friction="0.5 0.01 0.001", ground_friction="0.5 0.5 0",
        collision="disable_within_finger", naconmax=8192, njmax=8192, reference_joint_source="active",
        progress_interval=0, isaac_source_root=tmp_path, isaacgym_metrics=None, output_dir=tmp_path / "outputs",
        summary=tmp_path / "summary.json", keep_trajectory=False,
    )
    batch_reference_pd.run_batch(args)

    assert len(captured) == 1
    forwarded = captured[0]
    assert forwarded.physics.substeps == 3
    assert not (set(vars(forwarded)) & LEGACY_PHYSICS_FIELDS)


def test_canonical_substeps_own_replay_and_renderer_runtime_loops() -> None:
    physics = DexHandPhysicsConfig(substeps=3)
    assert replay_physics(argparse.Namespace(physics=physics)) is physics
    replay_source = Path("sim/dexhandrl/reference_pd_replay.py").read_text(encoding="utf-8")
    render_source = Path("sim/dexhandrl/render_reference_pd_video.py").read_text(encoding="utf-8")
    assert "args.settle_frames * physics.substeps" in replay_source
    assert replay_source.count("for _ in range(physics.substeps):") == 1
    assert "for _ in range(physics.substeps):" in render_source
