from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sim.manorl.approach_prefix import ApproachPrefixConfig
from tools.build_manorl_prefix_only_coverage_plan import (
    PLAN_CONTRACT,
    _canonical_digest,
    _cell_candidates,
)
from tools.run_manorl_prefix_only_coverage_plan import (
    STATUS_CONTRACT,
    _finalize_status,
    _recover_pending_attempt,
    _sample_trajectory,
    load_plan,
)
from tools.select_manorl_targeted_parents import select_records
from tests.manorl.test_approach_prefix import _trajectory
from tests.manorl.test_synthesis_seed_plan import _parent


def _slots(count: int, *, seed_start: int) -> list[dict[str, object]]:
    slots = []
    for slot_index in range(count):
        candidates = [
            {
                "fallback_rank": fallback_rank,
                "episode_seed": seed_start + slot_index * 12 + fallback_rank,
                "sampled_start": {
                    "radius_m": 0.5,
                    "azimuth_offset_deg": 0.0,
                    "z_offset_m": 0.1,
                },
            }
            for fallback_rank in range(12)
        ]
        slots.append(
            {
                "slot_index": slot_index,
                "cell": {
                    "distance_bin": 0,
                    "azimuth_bin": 0,
                    "height_bin": 0,
                    "shape": [1, 1, 1],
                },
                "candidates": candidates,
            }
        )
    return slots


def _plan(tmp_path: Path) -> Path:
    descriptor = tmp_path / "parent.json"
    manifest = tmp_path / "manifest.json"
    descriptor.write_text("{}", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    tasks = []
    for task_index, (mode, count, seed_start) in enumerate(
        (("far", 50, 1_000), ("near", 30, 10_000))
    ):
        tasks.append(
            {
                "task_index": task_index,
                "pair": "banana:02",
                "source_identity": "banana_02_001",
                "parent_uuid": "parent",
                "descriptor_path": str(descriptor),
                "predecoded_manifest": str(manifest),
                "mode": mode,
                "target_rows": count,
                "fallbacks_per_slot": 12,
                "slots": _slots(count, seed_start=seed_start),
            }
        )
    values = {
        "contract": PLAN_CONTRACT,
        "tasks": tasks,
        "target_rows": 80,
    }
    values["plan_digest"] = _canonical_digest(values)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_targeted_selection_prefers_farther_half_and_spatial_coverage() -> None:
    records = []
    for index in range(10):
        angle = -1.0 + 0.2 * index
        distance = 0.10 + 0.01 * index
        records.append(
            {
                "identity": f"banana_02_{index:03d}",
                "pre60_wrist_object_distance_m": distance,
                "pre60_wrist_object_delta_m": [
                    distance * np.cos(angle),
                    distance * np.sin(angle),
                    0.02 * (index % 3),
                ],
            }
        )
    selected, method = select_records(records, count=5)
    assert len(selected) == 5
    assert len({record["identity"] for record in selected}) == 5
    assert all(
        record["pre60_wrist_object_distance_m"]
        >= method["farther_pool_threshold_m"]
        for record in selected
    )
    assert selected[0]["identity"] == "banana_02_009"


def test_coverage_plan_loads_50_far_30_near_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    path = _plan(tmp_path)
    plan = load_plan(path)
    assert [task["target_rows"] for task in plan["tasks"]] == [50, 30]
    seeds = [
        candidate["episode_seed"]
        for task in plan["tasks"]
        for slot in task["slots"]
        for candidate in slot["candidates"]
    ]
    assert len(seeds) == 80 * 12
    assert len(set(seeds)) == len(seeds)

    values = json.loads(path.read_text())
    values["tasks"][0]["slots"][0]["candidates"][0]["episode_seed"] += 1
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="digest changed"):
        load_plan(path)


def test_far_fixed_grid_has_twelve_candidates_in_every_cell() -> None:
    geometries = []
    for distance_bin in range(5):
        for azimuth_bin in range(5):
            for height_bin in range(2):
                for _ in range(12):
                    geometries.append(
                        {
                            "radius_m": 0.30 + (distance_bin + 0.5) / 5 * 0.70,
                            "azimuth_offset_deg": -30.0
                            + (azimuth_bin + 0.5) / 5 * 60.0,
                            "z_offset_m": 0.08
                            + (height_bin + 0.5) / 2 * 0.22,
                        }
                    )
    cells = _cell_candidates(
        geometries, shape=(5, 5, 2), empirical=False
    )
    assert len(cells) == 50
    assert {len(candidates) for candidates in cells.values()} == {12}


def test_coverage_sampling_is_prefix_only_and_preserves_complete_tail() -> None:
    source = _trajectory()
    parent = _parent(source.identity.identity, source=source)
    for mode in ("far", "near"):
        augmented, prefix, config = _sample_trajectory(
            source, parent, mode=mode, seed=49
        )
        assert config == ApproachPrefixConfig(mode=mode)
        assert augmented.augmentation_suffix_frames == 0
        np.testing.assert_array_equal(
            augmented.q_ref[prefix.prefix_frames :], source.q_ref
        )
        np.testing.assert_array_equal(
            augmented.source_indices[prefix.prefix_frames :],
            source.source_indices,
        )


def test_pending_attempt_without_durable_row_rewinds_for_deterministic_retry(
    tmp_path: Path,
) -> None:
    status = {
        "accepted": {},
        "exhausted": {},
        "attempts_total": 4,
        "pending_attempt": {
            "phase": "running",
            "slot_index": 0,
            "fallback_rank": 3,
            "episode_seed": 42,
            "episode_index": 0,
            "attempt_number": 4,
        },
    }
    _recover_pending_attempt(output=tmp_path / "absent.lance", status=status)
    assert status["attempts_total"] == 3
    assert status["pending_attempt"] is None


def test_pending_written_row_recovers_by_uuid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "written.lance"
    output.mkdir()
    accepted_record = {"slot_index": 0, "uuid": "generated-uuid"}
    status = {
        "accepted": {},
        "exhausted": {},
        "attempts_total": 1,
        "pending_attempt": {
            "phase": "accepted_ready",
            "slot_index": 0,
            "fallback_rank": 0,
            "episode_seed": 42,
            "episode_index": 0,
            "attempt_number": 1,
            "uuid": "generated-uuid",
            "accepted_record": accepted_record,
        },
    }
    dataset = SimpleNamespace(
        count_rows=lambda: 1,
        take=lambda indices, columns: SimpleNamespace(
            to_pylist=lambda: [{"index": {"uuid": "generated-uuid"}}]
        ),
    )
    monkeypatch.setitem(
        sys.modules, "lance", SimpleNamespace(dataset=lambda path: dataset)
    )
    _recover_pending_attempt(output=output, status=status)
    assert status["pending_attempt"] is None
    assert status["accepted"] == {"0": accepted_record}
    assert status["attempts_total"] == 1


def test_coverage_status_distinguishes_bounded_completion() -> None:
    status = {
        "contract": STATUS_CONTRACT,
        "accepted": {"0": {}, "1": {}},
        "exhausted": {"2": {}},
    }
    _finalize_status(status, 3)
    assert status["attempts_complete"] is True
    assert status["all_slots_succeeded"] is False
    assert status["complete"] is False

    full = {
        "contract": STATUS_CONTRACT,
        "accepted": {"0": {}, "1": {}, "2": {}},
        "exhausted": {},
    }
    _finalize_status(full, 3)
    assert full["attempts_complete"] is True
    assert full["all_slots_succeeded"] is True
    assert full["complete"] is True
