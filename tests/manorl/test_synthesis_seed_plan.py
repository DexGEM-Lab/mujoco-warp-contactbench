from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim.manorl.approach_prefix import ApproachPrefixConfig
from sim.manorl.synthetic_parent import (
    ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
    AcceptedSyntheticParent,
    write_accepted_synthetic_parent,
)
from tools.run_manorl_synthesis_seed_plan import (
    PLAN_CONTRACT,
    PRODUCTION_APPROACH_VERTICAL_ARC_HEIGHT_M,
    RUN_CONTRACT,
    STATUS_CONTRACT,
    _parent_manifest,
    _sample_trajectory,
    load_plan,
    parse_args,
)
from tests.manorl.test_approach_prefix import _trajectory


def _parent(identity: str, *, source: object | None = None) -> AcceptedSyntheticParent:
    trajectory = _trajectory() if source is None else source
    anchor = int(trajectory.movement_end_step) + 15
    return AcceptedSyntheticParent(
        contract=ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
        parent_dataset_path="/data/parents.lance",
        parent_dataset_version=40,
        parent_row_index=1,
        parent_row_uuid=f"parent-{identity}",
        parent_row_contract="synthetic_mano_target_replay_visual_v2_contact",
        source_identity=identity,
        source_dataset_path="/data/source.lance",
        source_dataset_version=295,
        source_row_index=1,
        checkpoint_sha256="a" * 64,
        checkpoint_update=1000,
        parent_seed=42,
        parent_episode_index=0,
        parent_generation_attempt=1,
        object_init_xy_offset_m=(0.0, 0.0),
        reference_fps=120,
        retreat_last_contact_state_index=anchor,
        retreat_anchor_state_index=anchor,
        retreat_anchor_source_frame_index=int(trajectory.source_indices[anchor]),
        retreat_anchor_horizontal_distance_m=0.1,
        retreat_anchor_offset_frames=15,
        parent_movement_end_state_index=int(trajectory.movement_end_step),
        source_row_frame0_right_q_ref_3_28=tuple(
            float(value) for value in trajectory.q_ref[0, 3:28]
        ),
    )


def _plan(tmp_path: Path) -> Path:
    parents = []
    used_seeds = set()
    for parent_index in range(35):
        identity = f"banana_{parent_index % 7 + 1:02d}_{parent_index + 1:03d}"
        descriptor = _parent(identity)
        descriptor_path = tmp_path / f"{identity}.json"
        write_accepted_synthetic_parent(descriptor, descriptor_path)
        slots = []
        for slot_index in range(10):
            candidates = []
            for fallback_rank in range(12):
                seed = parent_index * 10000 + slot_index * 100 + fallback_rank
                assert seed not in used_seeds
                used_seeds.add(seed)
                candidates.append(
                    {
                        "fallback_rank": fallback_rank,
                        "episode_seed": seed,
                        "far_distance_decile": slot_index,
                        "near_distance_decile": 9 - slot_index,
                    }
                )
            slots.append(
                {
                    "slot_rank_by_far_distance": slot_index + 1,
                    "target_far_distance_decile": slot_index,
                    "target_near_distance_decile": 9 - slot_index,
                    "candidates": candidates,
                }
            )
        parents.append(
            {
                "source_identity": identity,
                "parent_uuid": descriptor.parent_row_uuid,
                "descriptor_path": str(descriptor_path),
                "slots": slots,
            }
        )
    path = tmp_path / "paired_seed_fallbacks.json"
    path.write_text(
        json.dumps(
            {
                "contract": PLAN_CONTRACT,
                "parents": 35,
                "target_slots_per_parent": 10,
                "candidates_per_slot": 12,
                "parents_plan": parents,
            }
        )
    )
    return path


def test_plan_schema_accepts_fixed_35_parent_pair_contract(tmp_path: Path) -> None:
    plan = load_plan(_plan(tmp_path))
    assert len(plan["parents_plan"]) == 35
    assert sum(len(parent["slots"]) for parent in plan["parents_plan"]) == 350
    assert sum(
        len(slot["candidates"])
        for parent in plan["parents_plan"]
        for slot in parent["slots"]
    ) == 4200


def test_plan_rejects_duplicate_candidate_seed(tmp_path: Path) -> None:
    path = _plan(tmp_path)
    values = json.loads(path.read_text())
    values["parents_plan"][1]["slots"][0]["candidates"][0]["episode_seed"] = 0
    path.write_text(json.dumps(values))
    try:
        load_plan(path)
    except ValueError as exc:
        assert "repeats episode seed" in str(exc)
    else:  # pragma: no cover - assertion is the test
        raise AssertionError("duplicate planned seed was accepted")


def test_sample_trajectory_uses_ten_cm_arc_and_mandatory_retreat() -> None:
    source = _trajectory()
    parent = _parent("banana_01_001", source=source)
    trajectory, prefix, suffix = _sample_trajectory(
        source,
        parent,
        mode="near",
        seed=49,
        approach_config=ApproachPrefixConfig(
            mode="near",
            vertical_arc_height_m=PRODUCTION_APPROACH_VERTICAL_ARC_HEIGHT_M,
        ),
    )
    assert PRODUCTION_APPROACH_VERTICAL_ARC_HEIGHT_M == 0.10
    assert prefix.approach_mode == "near"
    assert suffix.contract.endswith("_v4")
    assert trajectory.augmentation_suffix_frames == suffix.suffix_frames
    assert suffix.extra_horizontal_offset_m >= 0.03
    assert suffix.extra_z_offset_m >= 0.04


def test_parent_manifest_records_actual_arc_and_required_retreat(tmp_path: Path) -> None:
    source = _trajectory()
    parent = _parent("banana_01_001", source=source)
    status = {
        "accepted": {
            "0": {
                "slot_index": 0,
                "seed": 49,
                "fallback_rank": 0,
                "near_uuid": "near",
                "far_uuid": "far",
            }
        },
        "complete": True,
    }
    manifest = _parent_manifest(
        output=tmp_path / "banana.lance",
        parent=parent,
        checkpoint=tmp_path / "checkpoint-001000.pt",
        provenance_base={
            "checkpoint_sha256": "a" * 64,
            "checkpoint_update": 1000,
            "checkpoint_metadata": {},
        },
        status=status,
        approach_vertical_arc_height_m=0.10,
    )
    assert manifest["contract"] == RUN_CONTRACT
    assert manifest["retreat_suffix_required"] is True
    assert manifest["approach_vertical_arc_height_m"] == 0.10
    assert STATUS_CONTRACT not in manifest


def test_cli_defaults_to_production_arc_and_accepts_parent_subset() -> None:
    args = parse_args(
        [
            "--plan",
            "plan.json",
            "--checkpoint",
            "checkpoint.pt",
            "--predecoded-manifest",
            "manifest.json",
            "--output-dir",
            "out",
            "--parent-indices",
            "0,4",
        ]
    )
    assert args.approach_vertical_arc_height_m == 0.10
    assert args.parent_indices == "0,4"
