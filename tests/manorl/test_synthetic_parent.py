from __future__ import annotations

import json

import pytest

from sim.manorl.synthetic_parent import (
    ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
    AcceptedSyntheticParent,
    load_accepted_synthetic_parent,
    write_accepted_synthetic_parent,
)


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
    )


def test_accepted_parent_round_trip_and_pair_properties(tmp_path) -> None:
    parent = _parent()
    path = write_accepted_synthetic_parent(parent, tmp_path / "parent.json")
    assert load_accepted_synthetic_parent(path) == parent
    assert parent.object_type == "banana"
    assert parent.action_id == "01"
    values = json.loads(path.read_text())
    assert values["object_init_xy_offset_m"] == [-0.013, -0.017]


def test_accepted_parent_rejects_invalid_checkpoint_and_offset() -> None:
    values = _parent().to_dict()
    values["checkpoint_sha256"] = "bad"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        AcceptedSyntheticParent(**values)
    values = _parent().to_dict()
    values["object_init_xy_offset_m"] = [0.0]
    with pytest.raises(ValueError, match="object XY offset"):
        AcceptedSyntheticParent(**values)
