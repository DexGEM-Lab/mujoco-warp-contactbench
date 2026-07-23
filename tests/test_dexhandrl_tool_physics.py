"""Boundary contracts for DexHand reference-PD and trace tooling."""

from __future__ import annotations

import argparse
from pathlib import Path

from sim.dexhandrl.contracts import DexHandPhysicsConfig
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
