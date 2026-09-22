from __future__ import annotations

from argparse import Namespace
import json

import pytest

from tools import replay_atomic_grade as grade


def records() -> list[dict[str, object]]:
    values = []
    for row_index, frames, action in (
        (0, 100, "001"),
        (1, 600, "002"),
        (2, 300, "003"),
        (3, 500, "003"),
        (4, 200, "004"),
        (5, 400, "005"),
    ):
        values.append(
            {
                "row_index": row_index,
                "uuid": f"u{row_index}",
                "action": action,
                "frames": frames,
                "object_names": [f"object-{action}"],
                "target": f"object-{action}",
                "source_identity": f"source-{row_index}",
            }
        )
    return values


def test_grade_boundaries_match_state45_quality_contract() -> None:
    assert grade.grade_from_max_error(0.0) == "A"
    assert grade.grade_from_max_error(0.029999) == "A"
    assert grade.grade_from_max_error(0.03) == "B"
    assert grade.grade_from_max_error(0.079999) == "B"
    assert grade.grade_from_max_error(0.08) == "C"
    assert grade.GRADE_CONTACT_CAPACITY_PER_WORLD * 5 > 5648
    assert grade.GRADE_CONSTRAINT_CAPACITY > 5302
    assert 5 * grade.GRADE_CCD_CONTACTS_PER_WORLD > 1874
    assert grade.capacities_for_profile("expanded") == (2048, 8192, 512)
    assert grade.capacities_for_profile("u1") == (1024, 4096, 256)
    with pytest.raises(ValueError, match="physics profile"):
        grade.capacities_for_profile("unknown")
    with pytest.raises(ValueError, match="finite"):
        grade.grade_from_max_error(float("nan"))


def test_plan_is_unique_complete_and_frame_balanced() -> None:
    plan = grade.make_balanced_plan(
        records(), dataset_rows=6, dataset_version=2, shard_count=2
    )
    grade.validate_plan(plan)
    rows = [row for shard in plan["shards"] for row in shard["rows"]]
    assert sorted(row["row_index"] for row in rows) == list(range(6))
    assert len({row["uuid"] for row in rows}) == 6
    frame_counts = [shard["frame_count"] for shard in plan["shards"]]
    assert sum(frame_counts) == 2100
    assert max(frame_counts) - min(frame_counts) <= 100


def test_plan_rejects_duplicate_row_or_uuid() -> None:
    plan = grade.make_balanced_plan(
        records(), dataset_rows=6, dataset_version=2, shard_count=2
    )
    duplicate = json.loads(json.dumps(plan))
    duplicate["shards"][1]["rows"][0]["row_index"] = duplicate["shards"][0]["rows"][0]["row_index"]
    with pytest.raises(ValueError, match="SHA|not unique|identity"):
        grade.validate_plan(duplicate)


def test_group_batches_never_mixes_physical_topology() -> None:
    grouped = list(grade._group_batches(records(), batch_size=2))
    assert sum(map(len, grouped)) == 6
    for batch in grouped:
        keys = {
            (row["action"], tuple(row["object_names"]), row["target"])
            for row in batch
        }
        assert len(keys) == 1
        assert len(batch) <= 2


def test_tail_batch_padding_keeps_fixed_shape_without_new_rows() -> None:
    source = records()[:2]
    padded, real_count = grade._pad_batch(source, 5)
    assert real_count == 2
    assert len(padded) == 5
    assert [row["row_index"] for row in padded] == [0, 1, 0, 1, 0]
    with pytest.raises(ValueError, match="non-empty"):
        grade._pad_batch([], 5)


def test_parse_rows_is_end_exclusive_and_deduplicated() -> None:
    assert grade.parse_rows("2,4:7,5", 10) == [2, 4, 5, 6]
    with pytest.raises(ValueError, match="outside"):
        grade.parse_rows("10", 10)
    with pytest.raises(ValueError, match="invalid"):
        grade.parse_rows("7:7", 10)


def test_row_resume_identity_must_match(tmp_path) -> None:
    expected = {"row_index": 3, "uuid": "u3"}
    run = {"contract": "run"}
    path = tmp_path / "row0003.json"
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "row_index": 3,
                "uuid": "u3",
                "grade": "B",
                "run_identity": run,
            }
        )
    )
    assert grade._row_record_valid(path, expected, run)
    assert not grade._row_record_valid(path, expected, {"contract": "other"})


def test_action_isolated_aggregate_requires_clean_complete_population(tmp_path) -> None:
    plan = grade.make_balanced_plan(
        records(), dataset_rows=6, dataset_version=2, shard_count=2
    )
    plan_path = tmp_path / "plan.json"
    grade.dump_atomic(plan_path, plan)
    roots = [tmp_path / "shard0", tmp_path / "shard1"]
    errors = [0.01, 0.04, 0.09, 0.02, 0.07, 0.11]
    identity_base = {
        "grade_contract": grade.GRADE_CONTRACT,
        "contact_capacity_per_world": grade.GRADE_CONTACT_CAPACITY_PER_WORLD,
        "constraint_capacity": grade.GRADE_CONSTRAINT_CAPACITY,
        "ccd_contacts_per_world": grade.GRADE_CCD_CONTACTS_PER_WORLD,
        "batch_size": 5,
        "rendering": False,
        "physics_profile": "atomic-benchmark",
        "capacity_profile": "expanded",
        "asset_commit": "asset",
        "asset_manifest_sha256": "manifest",
        "client_commit": "client",
        "manorl_commit": "manorl",
        "scene_sha256": "scene",
    }
    for shard_id, root in enumerate(roots):
        for action in sorted({row["action"] for row in plan["shards"][shard_id]["rows"]}):
            expected = [
                row for row in plan["shards"][shard_id]["rows"]
                if row["action"] == action
            ]
            action_root = root / f"action{action}"
            row_root = action_root / "rows"
            row_root.mkdir(parents=True)
            identity = {**identity_base, "shard_id": shard_id, "action": action}
            for row in expected:
                error = errors[row["row_index"]]
                grade.dump_atomic(
                    row_root / f"row{row['row_index']:04d}.json",
                    {
                        "status": "ok",
                        "row_index": row["row_index"],
                        "uuid": row["uuid"],
                        "action": action,
                        "grade": grade.grade_from_max_error(error),
                        "max_target_position_error_m": error,
                    },
                )
            grade.dump_atomic(
                action_root / "summary.json",
                {
                    "state": "complete",
                    "shard_id": shard_id,
                    "action": action,
                    "row_count": len(expected),
                    "row_indices_sha256": grade.canonical_sha256(
                        [row["row_index"] for row in expected]
                    ),
                    "run_identity": identity,
                },
            )
            (root / f"action{action}.log").write_text("clean\n")
    output = tmp_path / "result.json"
    args = Namespace(
        plan=plan_path,
        shard_outputs=",".join(map(str, roots)),
        output=output,
    )
    grade.aggregate(args)
    result = json.loads(output.read_text())
    assert result["rows"] == 6
    assert result["grade_counts"] == {"A": 2, "B": 2, "C": 2}
    first_log = next(roots[0].glob("action*.log"))
    first_log.write_text("nefc overflow - please increase njmax to 9000\n")
    with pytest.raises(ValueError, match="overflow"):
        grade.aggregate(Namespace(**{**vars(args), "output": tmp_path / "bad.json"}))
