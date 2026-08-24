from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sim.manorl.approach_prefix import ApproachPrefixConfig, RetreatSuffixConfig
from sim.manorl.synthetic_parent import (
    ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
    AcceptedSyntheticParent,
    load_accepted_synthetic_parent,
    write_accepted_synthetic_parent,
)
from sim.manorl.approach_prefix import (
    APPROACH_PREFIX_ONLY_PRODUCTION_CONTRACT,
    PREFIX_ONLY_AUGMENTATION_IDENTITY_CONTRACT,
)
from tools.export_manorl_synthetic_lance import (
    AUGMENTATION_IDENTITY_CONTRACT,
    _augmentation_identity,
    _resolve_parents_by_identity,
)
from tools.select_manorl_synthetic_parent import select_parent


def _parent() -> AcceptedSyntheticParent:
    return AcceptedSyntheticParent(
        contract=ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
        parent_dataset_path="/data/prior.lance",
        parent_dataset_version=40,
        parent_row_index=3,
        parent_row_uuid="row-uuid",
        parent_row_contract="synthetic_mano_target_replay_visual_v2_contact",
        source_identity="banana_01_052",
        source_dataset_path="/data/source.lance",
        source_dataset_version=295,
        source_row_index=51,
        checkpoint_sha256="a" * 64,
        checkpoint_update=1000,
        parent_seed=42,
        parent_episode_index=0,
        parent_generation_attempt=1,
        object_init_xy_offset_m=(-0.013, -0.017),
        reference_fps=120,
        retreat_last_contact_state_index=100,
        retreat_anchor_state_index=105,
        retreat_anchor_source_frame_index=200,
        retreat_anchor_horizontal_distance_m=0.1,
        retreat_anchor_offset_frames=15,
        parent_movement_end_state_index=90,
        source_row_frame0_right_q_ref_3_28=tuple(float(i) / 100.0 for i in range(25)),
    )


def test_accepted_parent_round_trip_and_pair_properties(tmp_path) -> None:
    parent = _parent()
    path = write_accepted_synthetic_parent(parent, tmp_path / "parent.json")
    assert load_accepted_synthetic_parent(path) == parent
    assert parent.object_type == "banana"
    assert parent.action_id == "01"
    values = json.loads(path.read_text())
    assert values["object_init_xy_offset_m"] == [-0.013, -0.017]


def test_augmentation_identity_distinguishes_mode_and_config() -> None:
    parent = _parent()
    endpoint_config = RetreatSuffixConfig()

    def identity(
        approach: ApproachPrefixConfig,
        endpoint: RetreatSuffixConfig = endpoint_config,
        *,
        retreat: bool = False,
    ) -> str:
        return _augmentation_identity(
            accepted_parent=parent,
            episode_seed=49,
            attempt_number=1,
            episode_index=0,
            approach_config=approach,
            approach_sample=None,
            near_endpoint_config=endpoint,
            retreat_config=endpoint_config if retreat else None,
            retreat_sample=None,
        )

    far = identity(ApproachPrefixConfig(mode="far"))
    near = identity(ApproachPrefixConfig(mode="near"))
    custom_near = identity(
        ApproachPrefixConfig(mode="near"),
        RetreatSuffixConfig(maximum_extra_horizontal_m=0.20),
    )
    assert far != near
    assert near != custom_near
    assert near == identity(
        ApproachPrefixConfig(mode="near", maximum_xy_radius_m=0.70)
    )
    assert far != identity(
        ApproachPrefixConfig(mode="far", maximum_xy_radius_m=0.70)
    )
    assert ApproachPrefixConfig(mode="near").maximum_xy_radius_m == 0.70
    assert far.startswith(PREFIX_ONLY_AUGMENTATION_IDENTITY_CONTRACT + ":")
    historical = identity(ApproachPrefixConfig(mode="far"), retreat=True)
    assert historical.startswith(AUGMENTATION_IDENTITY_CONTRACT + ":")
    assert historical != far
    assert APPROACH_PREFIX_ONLY_PRODUCTION_CONTRACT in (
        "manorl_pre60_far_near_approach_prefix_only_complete_original_tail_v1",
    )


def test_accepted_parent_rejects_stale_or_missing_v3_fields() -> None:
    values = _parent().to_dict()
    values["contract"] = "manorl_accepted_synthetic_parent_v2"
    with pytest.raises(ValueError, match="unsupported accepted-parent contract"):
        AcceptedSyntheticParent(**values)
    values = _parent().to_dict()
    values["retreat_anchor_offset_frames"] = 5
    values["retreat_anchor_state_index"] = 95
    with pytest.raises(ValueError, match="must equal 15"):
        AcceptedSyntheticParent(**values)
    values = _parent().to_dict()
    values["retreat_anchor_source_frame_index"] = None
    with pytest.raises(ValueError, match="requires all retreat anchor fields"):
        AcceptedSyntheticParent(**values)
    values = _parent().to_dict()
    values["retreat_anchor_horizontal_distance_m"] = 0.0
    with pytest.raises(ValueError, match="positive horizontal retreat"):
        AcceptedSyntheticParent(**values)
    values = _parent().to_dict()
    values["source_row_frame0_right_q_ref_3_28"] = None
    with pytest.raises(ValueError, match="requires source-row frame0"):
        AcceptedSyntheticParent(**values)
    values = _parent().to_dict()
    values["retreat_last_contact_state_index"] = 80
    with pytest.raises(ValueError, match="last contact precedes movement end"):
        AcceptedSyntheticParent(**values)


def _selector_rows(*, last_contact_state: int = 4):
    frames = 25
    contact = [[] for _ in range(frames)]
    contact[last_contact_state] = [
        {
            "hand_name": "right",
            "object_name": "banana",
            "contact_pairs": [{"force_normal": [0.3, 0.0, 0.0]}],
        }
    ]
    reference_hand = np.zeros((frames, 28), dtype=np.float64)
    reference_hand[-1, 0] = 0.1
    parent_row = {
        "index": {"uuid": "parent-uuid"},
        "objects": [{"pos": np.zeros((frames, 3)).tolist()}],
        "reference": {
            "object_pos": np.zeros((frames, 3)).tolist(),
            "source_frame_index": list(range(100, 100 + frames)),
            "hand_urdf_dof": reference_hand.tolist(),
        },
        "contact": contact,
        "command_reference_index": list(range(frames - 1)),
        "command_source_frame_index": list(range(100, 100 + frames - 1)),
        "trajectory_metadata": {
            "trajectory_info": {
                "object_move": [
                    {"object_name": "banana", "start_frame": 1, "end_frame": 3}
                ]
            }
        },
        "provenance": {
            "contract": "synthetic_mano_target_replay_visual_v2_contact",
            "source_identity": "banana_01_052",
            "dataset_path": "/source.lance",
            "dataset_version": 295,
            "row_index": 51,
            "checkpoint_sha256": "a" * 64,
            "checkpoint_update": 1000,
            "seed": 42,
            "episode_index": 0,
            "generation_attempt": 1,
            "reference_fps": 120,
        },
    }
    source_row = {
        "trajectory_metadata": {"hand_names": ["left", "right"]},
        "hands": [
            {"urdf_dof": np.zeros((20, 28)).tolist()},
            {"urdf_dof": np.ones((20, 28)).tolist()},
        ],
    }
    return parent_row, source_row


def _install_fake_lance(monkeypatch, parent_row, source_row) -> None:
    class Table:
        def __init__(self, rows):
            self._rows = rows

        def to_pylist(self):
            return self._rows

    class ParentDataset:
        version = 40

        def scanner(self, *, columns):
            assert columns == ["index"]
            return SimpleNamespace(to_table=lambda: Table([parent_row]))

        def take(self, indices, columns=None):
            assert indices == [0]
            assert columns is None
            return Table([parent_row])

    class SourceDataset:
        def take(self, indices, columns=None):
            assert indices == [51]
            assert columns == ["hands", "trajectory_metadata"]
            return Table([source_row])

    def dataset(path, *, version=None):
        if str(path).endswith("prior.lance"):
            assert version is None
            return ParentDataset()
        assert str(path) == "/source.lance"
        assert version == 295
        return SourceDataset()

    monkeypatch.setitem(sys.modules, "lance", SimpleNamespace(dataset=dataset))


def test_selector_maps_movement_end_plus15_through_state_reference(
    tmp_path, monkeypatch
) -> None:
    parent_row, source_row = _selector_rows(last_contact_state=4)
    _install_fake_lance(monkeypatch, parent_row, source_row)
    output = select_parent(
        "/prior.lance",
        row_uuid="parent-uuid",
        output=tmp_path / "parent.json",
    )
    parent = load_accepted_synthetic_parent(output)
    assert parent.parent_movement_end_state_index == 3
    assert parent.retreat_anchor_offset_frames == 15
    assert parent.retreat_anchor_state_index == 18
    # The state-aligned reference is authoritative. Indexing a transition at
    # state-1 would incorrectly produce source frame 117.
    assert parent.retreat_anchor_source_frame_index == 118
    assert parent.retreat_anchor_horizontal_distance_m == pytest.approx(0.1)
    assert parent.source_row_frame0_right_q_ref_3_28 == (1.0,) * 25


def test_selector_rejects_parent_without_horizontal_retreat_direction(
    tmp_path, monkeypatch
) -> None:
    parent_row, source_row = _selector_rows(last_contact_state=4)
    parent_row["reference"]["hand_urdf_dof"] = np.zeros((25, 28)).tolist()
    _install_fake_lance(monkeypatch, parent_row, source_row)
    with pytest.raises(ValueError, match="no nonzero horizontal retreat direction"):
        select_parent(
            "/prior.lance",
            row_uuid="parent-uuid",
            output=tmp_path / "parent.json",
        )


def test_selector_rejects_parent_contact_before_movement_end(
    tmp_path, monkeypatch
) -> None:
    parent_row, source_row = _selector_rows(last_contact_state=2)
    _install_fake_lance(monkeypatch, parent_row, source_row)
    with pytest.raises(ValueError, match="last contact precedes movement end"):
        select_parent(
            "/prior.lance",
            row_uuid="parent-uuid",
            output=tmp_path / "parent.json",
        )


def test_accepted_parent_rejects_invalid_checkpoint_and_offset() -> None:
    values = _parent().to_dict()
    values["checkpoint_sha256"] = "bad"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        AcceptedSyntheticParent(**values)
    values = _parent().to_dict()
    values["object_init_xy_offset_m"] = [0.0]
    with pytest.raises(ValueError, match="object XY offset"):
        AcceptedSyntheticParent(**values)


def test_single_parent_normalizes_to_one_identity_mapping() -> None:
    from sim.manorl.trajectory import TrajectoryBatch
    from tests.manorl.test_approach_prefix import _trajectory

    trajectory = _trajectory()
    parent = _parent()
    parent = AcceptedSyntheticParent(
        **{**parent.to_dict(), "source_identity": trajectory.identity.identity}
    )
    batch = TrajectoryBatch((trajectory,))
    assert _resolve_parents_by_identity(
        batch, accepted_parent=parent, accepted_parents_by_identity=None
    ) == {trajectory.identity.identity: parent}
    assert _resolve_parents_by_identity(
        batch,
        accepted_parent=None,
        accepted_parents_by_identity={trajectory.identity.identity: parent},
    ) == {trajectory.identity.identity: parent}
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_parents_by_identity(
            batch,
            accepted_parent=parent,
            accepted_parents_by_identity={trajectory.identity.identity: parent},
        )
