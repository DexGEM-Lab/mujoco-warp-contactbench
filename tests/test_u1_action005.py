from types import SimpleNamespace

import numpy as np

from sim.manorl.u1_action005 import (
    PARENT_ROW_BY_SLOT,
    action005_child_uuid,
    action005_gate_values,
    canonical_parent_target,
    validate_parent_assignment,
)
from tools.export_u1_largepose800 import (
    CONTRACT as EXACT800_CONTRACT,
    movement_record,
    schema as exact800_schema,
    source640_index_for_output,
)
from tools.run_u1_largepose_campaign import build_plan


def test_ten_parent_assignment_is_balanced_and_complete() -> None:
    plan = build_plan()
    assigned = validate_parent_assignment(plan)

    assert len(PARENT_ROW_BY_SLOT) == 160
    assert sorted(slot for slots in assigned.values() for slot in slots) == list(
        range(160)
    )
    assert {row: len(slots) for row, slots in assigned.items()} == {
        row: 16 for row in range(10)
    }


def test_child_uuid_binds_exact_parent() -> None:
    candidate = build_plan()[0]["candidates"][0]
    first = action005_child_uuid(
        registry_sha256="a" * 64,
        plan_digest="b" * 64,
        slot=0,
        candidate=candidate,
        parent_uuid="parent-a",
    )
    second = action005_child_uuid(
        registry_sha256="a" * 64,
        plan_digest="b" * 64,
        slot=0,
        candidate=candidate,
        parent_uuid="parent-b",
    )

    assert first != second
    assert first == action005_child_uuid(
        registry_sha256="a" * 64,
        plan_digest="b" * 64,
        slot=0,
        candidate=candidate,
        parent_uuid="parent-a",
    )


def test_canonical_parent_target_changes_only_prestep_frame() -> None:
    source = SimpleNamespace(
        frames=4,
        target_qpos=np.zeros((4, 28), dtype=np.float64),
        recorded_qpos=np.zeros((4, 28), dtype=np.float64),
    )
    source.target_qpos[0, 14] = -0.2
    model = SimpleNamespace(jnt_range=np.tile([-0.1, 0.1], (28, 1)))

    canonical = canonical_parent_target(source, model)

    assert canonical[0, 14] == -0.1
    np.testing.assert_array_equal(canonical[1:], source.target_qpos[1:])
    np.testing.assert_array_equal(source.target_qpos[0, 14], -0.2)


def test_action005_gate_values_require_real_pour_and_settle() -> None:
    frames = 48
    tilt = np.zeros(frames)
    tilt[4:20] = np.linspace(90.0, 105.0, 16)
    bottle = np.zeros((frames, 3))
    bowl = np.zeros((frames, 3))
    bottle[:, 2] = 0.12
    bowl[:, 2] = 0.02
    bottle[19, :2] = [0.08, 0.04]
    quaternion = np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1))
    bottle_velocity = np.zeros((frames, 6))
    bowl_velocity = np.zeros((frames, 6))

    gates, metrics = action005_gate_values(
        tilt_deg=tilt,
        bottle_position=bottle,
        bowl_position=bowl,
        bottle_quaternion_xyzw=quaternion,
        bowl_quaternion_xyzw=quaternion,
        bottle_velocity=bottle_velocity,
        bowl_velocity=bowl_velocity,
    )

    assert all(gates.values())
    assert metrics["max_world_tilt_deg"] == 105.0
    bottle_velocity[-1, 0] = 0.02
    outlier, _ = action005_gate_values(
        tilt_deg=tilt,
        bottle_position=bottle,
        bowl_position=bowl,
        bottle_quaternion_xyzw=quaternion,
        bowl_quaternion_xyzw=quaternion,
        bottle_velocity=bottle_velocity,
        bowl_velocity=bowl_velocity,
    )
    assert outlier["settled_terminal_bottle_linear"] is True
    bottle[-1, 0] = 0.002
    drifting, _ = action005_gate_values(
        tilt_deg=tilt,
        bottle_position=bottle,
        bowl_position=bowl,
        bottle_quaternion_xyzw=quaternion,
        bowl_quaternion_xyzw=quaternion,
        bottle_velocity=bottle_velocity,
        bowl_velocity=bowl_velocity,
    )
    assert drifting["settled_terminal_bottle_linear"] is False


def test_exact800_interleave_mapping_preserves_source_order() -> None:
    assert source640_index_for_output(0) == 0
    assert source640_index_for_output(159) == 159
    assert source640_index_for_output(160) is None
    assert source640_index_for_output(319) is None
    assert source640_index_for_output(320) == 160
    assert source640_index_for_output(799) == 639


def test_exact800_schema_and_shifted_action005_movement() -> None:
    assert exact800_schema().metadata[b"schema_version"] == EXACT800_CONTRACT.encode()
    assert movement_record(
        {
            "frames": 742,
            "movement": {
                "object_name": "mayonnaisebottle",
                "start_frame": 120,
                "end_frame": 491,
            },
        }
    ) == {
        "object_name": "mayonnaisebottle",
        "start_frame": 240,
        "end_frame": 611,
    }
