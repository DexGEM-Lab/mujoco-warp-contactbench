"""Unit tests for the offline full-length retreat constructor (stage 2)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.build_manorl_full_retreat import (
    ANCHOR_OFFSET,
    contact_at,
    last_contact_frame,
    sample_endpoint,
    _deform_retreat_tail,
    _identity_seed,
)


def _base_row(*, contact_frames: list[int], T: int = 300, gesture: str = "02") -> dict:
    """Minimal base-row dict with contact only in the listed frames."""
    contact = [
        (
            [
                {
                    "hand_name": "right",
                    "object_name": "banana",
                    "contact_pairs": [
                        {"force_normal": [0.0, 0.0, 1.0]}  # 1 N > 0.2 N
                    ],
                }
            ]
            if i in contact_frames
            else []
        )
        for i in range(T)
    ]
    q = np.zeros((T, 28), dtype=np.float64)
    # 手腕沿 +X 移动, 让 retreat 有方向
    q[:, 0] = np.linspace(0.0, 0.3, T)
    q[:, 3:6] = 0.1
    q[:, 6:] = 0.2
    q_target = q.copy()
    return {
        "index": {"uuid": "u1", "scene": "banana", "is_generated": True},
        "trajectory_metadata": {
            "total_frames": T,
            "gesture": gesture,
            "hand_slots": ["right", "left"],
            "trajectory_info": {"object_move": [{"object_name": "banana", "start_frame": 60, "end_frame": 200}]},
        },
        "hands": [
            {"hand_name": "right", "urdf_dof": q.tolist(), "urdf_dof_target": q_target.tolist(), "mano_global_pos": q[:, :3].tolist()},
            {"hand_name": None, "urdf_dof": [], "urdf_dof_target": [], "mano_global_pos": []},
        ],
        "objects": [{"pos": [[0.0, 0.0, 0.0]] * T, "rot_aa": [[0.0, 0.0, 0.0]] * T}],
        "contact": contact,
        "timestamp": [i / 120.0 for i in range(T)],
        "reference": {
            "source_frame_index": list(range(T)),
            "hand_urdf_dof": q.tolist(),
            "object_pos": [[0.0, 0.0, 0.0]] * T,
            "object_rot_aa": [[0.0, 0.0, 0.0]] * T,
        },
        "command_reference_index": list(range(T - 1)),
        "command_source_frame_index": list(range(T - 1)),
        "provenance": {
            "seed": 42,
            "source_identity": "banana_02_001",
            "contract": "synthetic_mano_target_replay_visual_v2_contact",
        },
    }


def test_last_contact_frame_finds_latest_contact() -> None:
    row = _base_row(contact_frames=[10, 50, 100])
    assert last_contact_frame(row, "banana") == 100


def test_last_contact_frame_none_when_no_contact() -> None:
    row = _base_row(contact_frames=[])
    assert last_contact_frame(row, "banana") is None


def test_contact_at_threshold() -> None:
    row = _base_row(contact_frames=[10])
    assert contact_at(row, "banana", 10)
    assert not contact_at(row, "banana", 11)
    # 过滤其他物体
    assert not contact_at(row, "bowl", 10)


def test_identity_seed_deterministic_and_distinct() -> None:
    a = _identity_seed(42, "banana_02_001")
    b = _identity_seed(42, "banana_02_001")
    c = _identity_seed(43, "banana_02_001")
    assert a == b
    assert a != c


def test_deform_tail_boundary_behavior() -> None:
    """前两帧与源相同, 最后三帧到终点 (历史 smoothstep 语义)."""
    source = np.zeros((20, 3), dtype=np.float64)
    end = np.ones(3, dtype=np.float64)
    out = _deform_retreat_tail(source, end)
    assert out.shape == source.shape
    np.testing.assert_array_equal(out[0], source[0])
    np.testing.assert_array_equal(out[1], source[1])
    np.testing.assert_array_equal(out[-1], end)
    # 单调非减 (远离)
    assert np.all(np.linalg.norm(out, axis=1) >= np.linalg.norm(out[0]) - 1e-12)


def test_sample_endpoint_geometry() -> None:
    anchor = np.zeros(3)
    original_end = np.array([0.2, 0.0, 0.0])
    ep = sample_endpoint(anchor, original_end, seed=42, identity="banana_02_001")
    assert 0.2 + 0.03 <= ep["end_horizontal_distance_m"] <= 0.2 + 0.15
    assert -30.0 <= ep["xy_offset_deg"] <= 30.0
    assert 0.04 <= ep["extra_z_offset_m"] <= 0.10
    end = np.asarray(ep["end_position_m"])
    assert np.linalg.norm(end[:2]) == pytest.approx(ep["end_horizontal_distance_m"], abs=1e-9)
    # 确定性
    ep2 = sample_endpoint(anchor, original_end, seed=42, identity="banana_02_001")
    assert ep == ep2
    # 不同 seed 不同
    ep3 = sample_endpoint(anchor, original_end, seed=43, identity="banana_02_001")
    assert ep != ep3


def test_build_full_retreat_row_end_to_end() -> None:
    from tools.build_manorl_full_retreat import build_full_retreat_row

    row = _base_row(contact_frames=[50, 100, 150], T=300)
    joint_lower = np.asarray([-2.0] * 6 + [-3.0] * 22, dtype=np.float64)
    joint_upper = np.asarray([2.0] * 6 + [3.0] * 22, dtype=np.float64)
    anchor = 150 + ANCHOR_OFFSET
    out, diag = build_full_retreat_row(
        row, anchor=anchor, seed=42, batch_tag="test",
        joint_lower=joint_lower, joint_upper=joint_upper,
    )
    assert out is not None, diag
    assert diag["accepted"]
    assert out["trajectory_metadata"]["total_frames"] == 300
    q = np.asarray(out["hands"][0]["urdf_dof"], dtype=np.float64)
    qb = np.asarray(row["hands"][0]["urdf_dof"], dtype=np.float64)
    # anchor 前逐位一致
    np.testing.assert_array_equal(q[: anchor + 1], qb[: anchor + 1])
    # anchor 后手指/旋转不变, 仅 XYZ 变
    np.testing.assert_array_equal(q[anchor + 1 :, 3:], qb[anchor + 1 :, 3:])
    assert not np.array_equal(q[anchor + 1 :, :3], qb[anchor + 1 :, :3])
    # mano_global_pos 与 urdf_dof[:, :3] 一致
    np.testing.assert_array_equal(
        np.asarray(out["hands"][0]["mano_global_pos"], dtype=np.float64),
        q[:, :3],
    )
    # splice 误差为 0 (前两帧不动)
    assert diag["splice_velocity_error_m_s"] == 0.0
    assert diag["splice_acceleration_jump_m_s2"] == 0.0
    # 物体/时间戳/contact/reference 保留
    assert out["objects"] == row["objects"]
    assert out["timestamp"] == row["timestamp"]
    assert out["contact"] == row["contact"]
    assert out["reference"] == row["reference"]
