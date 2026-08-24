from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.manorl.approach_prefix import ApproachPrefixConfig, RetreatSuffixConfig
from sim.manorl.synthesis_acceptance import (
    FAILURE_CONTACT_FRAMES,
    FAILURE_FINAL_ROTATION,
    FAILURE_TRAJECTORY_INCOMPLETE,
    SYNTHESIS_ACCEPTANCE_CONTRACT,
    count_hand_object_contact_frames,
    evaluate_synthesis_acceptance,
    final_rotation_xyz_abs_error_deg,
    persisted_rotation_quaternion_xyzw,
    synthesis_acceptance_manifest,
)
from tools.export_manorl_synthetic_lance import (
    _evaluate_collected_candidate,
    _synthesis_acceptance_gate_enabled,
)


def _quat(xyz_deg: tuple[float, float, float]) -> np.ndarray:
    return Rotation.from_euler("XYZ", xyz_deg, degrees=True).as_quat()


def _evaluate(
    *,
    complete: bool = True,
    simulated_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    reference_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    contact_frames: int = 101,
):
    return evaluate_synthesis_acceptance(
        trajectory_complete=complete,
        termination_reason_code=1 if complete else 2,
        simulated_final_object_quaternion_xyzw=_quat(simulated_xyz),
        reference_final_object_quaternion_xyzw=_quat(reference_xyz),
        hand_object_contact_frames=contact_frames,
    )


def test_acceptance_requires_all_three_predicates() -> None:
    result = _evaluate(simulated_xyz=(12.0, 15.0, 18.0), contact_frames=101)
    assert result.accepted
    assert result.failure_reasons == ()
    assert result.final_rotation_xyz_mean_error_deg == pytest.approx(15.0)

    incomplete = _evaluate(complete=False)
    assert not incomplete.accepted
    assert FAILURE_TRAJECTORY_INCOMPLETE in incomplete.failure_reasons

    rotation = _evaluate(simulated_xyz=(36.0, 36.0, 36.0))
    assert not rotation.accepted
    assert FAILURE_FINAL_ROTATION in rotation.failure_reasons

    contact = _evaluate(contact_frames=100)
    assert not contact.accepted
    assert FAILURE_CONTACT_FRAMES in contact.failure_reasons


def test_acceptance_preserves_strict_boundary_semantics() -> None:
    # The user's wording makes >35 degrees fail, so exactly 35 passes.
    assert _evaluate(simulated_xyz=(35.0, 35.0, 35.0)).accepted
    assert not _evaluate(simulated_xyz=(35.1, 35.1, 35.1)).accepted
    # More than 100 frames means 100 fails and 101 passes.
    assert not _evaluate(contact_frames=100).accepted
    assert _evaluate(contact_frames=101).accepted


def test_rotation_xyz_error_wraps_each_coordinate_to_shortest_branch() -> None:
    actual = final_rotation_xyz_abs_error_deg(
        _quat((-179.0, 0.0, -170.0)),
        _quat((179.0, 0.0, 170.0)),
    )
    np.testing.assert_allclose(actual, (2.0, 0.0, 20.0), atol=1e-10)


def test_rejection_reports_every_failed_predicate() -> None:
    result = _evaluate(
        complete=False,
        simulated_xyz=(60.0, 60.0, 60.0),
        contact_frames=0,
    )
    assert result.failure_reasons == (
        FAILURE_TRAJECTORY_INCOMPLETE,
        FAILURE_FINAL_ROTATION,
        FAILURE_CONTACT_FRAMES,
    )


def test_acceptance_manifest_binds_fixed_thresholds() -> None:
    manifest = synthesis_acceptance_manifest()
    assert manifest["contract"] == SYNTHESIS_ACCEPTANCE_CONTRACT
    assert manifest["operator"] == "all"
    rotation = manifest["rules"]["final_object_rotation"]
    assert rotation["measurement_precision"] == "persisted_float32_rotvec"
    assert rotation["maximum_mean_error_deg"] == 35.0
    contact = manifest["rules"]["hand_object_contact_frames"]
    assert contact["measurement_precision"] == "persisted_float32_force_vectors"
    assert contact["force_magnitude_comparison"] == ">"
    assert contact["force_threshold_N"] == 0.2
    assert contact["frame_count_comparison"] == ">"
    assert contact["frame_count_threshold"] == 100
    assert contact["minimum_passing_frames"] == 101


def test_rotation_measurement_round_trips_lance_float32_rotvec() -> None:
    quaternion = _quat((12.3456789, -23.4567891, 34.5678912))
    persisted = persisted_rotation_quaternion_xyzw(quaternion)
    stored_rotvec = Rotation.from_quat(quaternion).as_rotvec().astype(np.float32)
    np.testing.assert_allclose(
        Rotation.from_quat(persisted).as_rotvec(),
        stored_rotvec.astype(np.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_acceptance_rejects_invalid_measurements() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        _evaluate(contact_frames=-1)
    with pytest.raises(ValueError, match="finite XYZW"):
        evaluate_synthesis_acceptance(
            trajectory_complete=True,
            termination_reason_code=1,
            simulated_final_object_quaternion_xyzw=np.zeros(3),
            reference_final_object_quaternion_xyzw=_quat((0.0, 0.0, 0.0)),
            hand_object_contact_frames=101,
        )


def test_contact_frame_count_is_distinct_strict_and_persisted() -> None:
    frames = [
        [],
        [
            {
                "hand_name": "right",
                "object_name": "cube1",
                "contact_pairs": [
                    {"force_normal": [0.2, 0.0, 0.0]},
                    {"force_normal": [0.0, 0.0, 0.21]},
                ],
            },
            {
                "hand_name": "right",
                "object_name": "cube1",
                "contact_pairs": [{"force_normal": [1.0, 0.0, 0.0]}],
            },
        ],
        [
            {
                "hand_name": "left",
                "object_name": "cube1",
                "contact_pairs": [{"force_normal": [2.0, 0.0, 0.0]}],
            },
            {
                "hand_name": "right",
                "object_name": "cube2",
                "contact_pairs": [{"force_normal": [2.0, 0.0, 0.0]}],
            },
        ],
    ]
    # Frame 1 counts once; other-hand and other-object contacts do not count.
    assert count_hand_object_contact_frames(
        frames, target_object_name="cube1"
    ) == 1
    just_above = float(
        np.nextafter(np.float32(0.2), np.float32(np.inf), dtype=np.float32)
    )
    assert count_hand_object_contact_frames(
        [
            [
                {
                    "hand_name": "right",
                    "object_name": "cube1",
                    "contact_pairs": [
                        {"force_normal": [just_above, 0.0, 0.0]}
                    ],
                }
            ]
        ],
        target_object_name="cube1",
    ) == 1


def test_completion_requires_reason_code_one() -> None:
    result = evaluate_synthesis_acceptance(
        trajectory_complete=True,
        termination_reason_code=2,
        simulated_final_object_quaternion_xyzw=_quat((0.0, 0.0, 0.0)),
        reference_final_object_quaternion_xyzw=_quat((0.0, 0.0, 0.0)),
        hand_object_contact_frames=101,
    )
    assert not result.accepted
    assert FAILURE_TRAJECTORY_INCOMPLETE in result.failure_reasons


def test_gate_scope_is_only_no_prefix_no_retreat() -> None:
    assert _synthesis_acceptance_gate_enabled(
        approach_prefix_config=None, retreat_suffix_config=None
    )
    assert not _synthesis_acceptance_gate_enabled(
        approach_prefix_config=ApproachPrefixConfig(), retreat_suffix_config=None
    )
    assert not _synthesis_acceptance_gate_enabled(
        approach_prefix_config=None, retreat_suffix_config=RetreatSuffixConfig()
    )


def test_collected_candidate_uses_last_state_last_reference_and_saved_contacts() -> None:
    trajectory = type(
        "Trajectory",
        (),
        {
            "identity": type("Identity", (), {"identity": "cube1_01_001"})(),
            "object_quat_xyzw": np.asarray(
                [_quat((90.0, 0.0, 0.0)), _quat((0.0, 0.0, 0.0))]
            ),
        },
    )()
    frames = [
        [
            {
                "hand_name": "right",
                "object_name": "cube1",
                "contact_pairs": [{"force_normal": [0.21, 0.0, 0.0]}],
            }
        ]
        for _ in range(101)
    ]
    result = _evaluate_collected_candidate(
        trajectory=trajectory,
        state_storage={
            "object_orientation_xyzw": [
                _quat((100.0, 0.0, 0.0)),
                _quat((30.0, 30.0, 30.0)),
            ]
        },
        trajectory_complete=True,
        termination_reason_code=1,
        contact_frames=frames,
    )
    assert result.accepted
    np.testing.assert_allclose(
        result.final_rotation_xyz_abs_error_deg, (30.0, 30.0, 30.0), atol=1e-10
    )
    assert result.hand_object_contact_frames == 101


def test_manifest_acceptance_binds_rows_and_rejects_bad_recomputation() -> None:
    from tools.validate_manorl_synthetic_lance import _validate_manifest_acceptance

    accepted = _evaluate().to_dict()
    attempt = {
        "source_identity": "cube1_01_001",
        "seed": 42,
        "attempt_number": 1,
        "episode_index": 0,
        "accepted": True,
        "acceptance": accepted,
    }
    row = {
        "row_index": 0,
        "source_identity": "cube1_01_001",
        "seed": 42,
        "generation_attempt": 1,
        "episode_index": 0,
        "synthesis_acceptance": accepted,
    }
    manifest = {
        "synthesis": {
            "acceptance_gate": synthesis_acceptance_manifest(),
            "acceptance_attempts": [attempt],
        }
    }
    summary = _validate_manifest_acceptance(manifest=manifest, rows=[row])
    assert summary is not None
    assert summary["accepted_rows"] == 1
    assert summary["hand_object_contact_frames_min_max"] == [101, 101]

    drifted_manifest = json.loads(json.dumps(manifest))
    drifted_manifest["synthesis"]["acceptance_attempts"][0]["acceptance"][
        "hand_object_contact_frames"
    ] = 102
    with pytest.raises(ValueError, match="differs from manifest diagnostics"):
        _validate_manifest_acceptance(manifest=drifted_manifest, rows=[row])

    rejected = dict(row)
    rejected["synthesis_acceptance"] = _evaluate(contact_frames=100).to_dict()
    with pytest.raises(ValueError, match="fails synthesis acceptance"):
        _validate_manifest_acceptance(manifest=manifest, rows=[rejected])

    contradictory = json.loads(json.dumps(manifest))
    contradictory["synthesis"]["acceptance_attempts"][0]["accepted"] = False
    with pytest.raises(ValueError, match="diagnostics are inconsistent"):
        _validate_manifest_acceptance(manifest=contradictory, rows=[row])

    duplicate = json.loads(json.dumps(manifest))
    duplicate["synthesis"]["acceptance_attempts"].append(
        duplicate["synthesis"]["acceptance_attempts"][0]
    )
    with pytest.raises(ValueError, match="keys are not unique"):
        _validate_manifest_acceptance(manifest=duplicate, rows=[row])
