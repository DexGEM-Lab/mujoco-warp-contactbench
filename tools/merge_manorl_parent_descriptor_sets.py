#!/usr/bin/env python3
"""Merge accepted-parent descriptor sets without silently overwriting variants.

When one source identity has eligible parents from multiple bootstrap rounds,
choose the variant whose weakest normalized robustness margin is largest:
final-rotation headroom to 35 degrees, contact-frame headroom above 100, and
last-contact headroom through movement_end+15. All alternatives remain in the
merge manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sim.manorl.lance_v2 import file_sha256
from sim.manorl.synthetic_parent import load_accepted_synthetic_parent
from sim.manorl.synthesis_acceptance import (
    SYNTHESIS_FINAL_ROTATION_XYZ_MEAN_ERROR_MAX_DEG,
    SYNTHESIS_HAND_OBJECT_CONTACT_FRAME_COUNT_THRESHOLD,
)
from tools.select_manorl_targeted_parents import _current_parent_acceptance

MERGE_CONTRACT = "manorl_accepted_parent_descriptor_set_merge_v1"
PARENTS_BY_ACTION_CONTRACT = "manorl_synthesis_accepted_parents_by_action_v1"
LATE_CONTACT_REFERENCE_FRAMES = 15


def _variant_record(
    descriptor_path: Path,
    *,
    dataset_cache: dict[tuple[str, int], Any],
) -> dict[str, Any]:
    parent = load_accepted_synthetic_parent(descriptor_path)
    descriptor_sha256 = file_sha256(descriptor_path)
    object_type, action, _ = parent.source_identity.split("_", maxsplit=2)
    acceptance = _current_parent_acceptance(
        parent, object_type=object_type, dataset_cache=dataset_cache
    )
    if not bool(acceptance["accepted"]):
        raise ValueError(
            f"descriptor {descriptor_path} no longer passes current parent gate"
        )
    rotation_margin = (
        SYNTHESIS_FINAL_ROTATION_XYZ_MEAN_ERROR_MAX_DEG
        - float(acceptance["final_rotation_xyz_mean_error_deg"])
    )
    contact_margin = (
        int(acceptance["hand_object_contact_frames"])
        - SYNTHESIS_HAND_OBJECT_CONTACT_FRAME_COUNT_THRESHOLD
    )
    late_contact_margin = (
        parent.retreat_last_contact_state_index
        - parent.parent_movement_end_state_index
    )
    normalized = {
        "final_rotation": rotation_margin
        / SYNTHESIS_FINAL_ROTATION_XYZ_MEAN_ERROR_MAX_DEG,
        "contact_frames": contact_margin
        / SYNTHESIS_HAND_OBJECT_CONTACT_FRAME_COUNT_THRESHOLD,
        "late_contact": late_contact_margin / LATE_CONTACT_REFERENCE_FRAMES,
    }
    return {
        "source_identity": parent.source_identity,
        "action": action,
        "descriptor_path": str(descriptor_path.resolve()),
        "descriptor_sha256": descriptor_sha256,
        "variant_identity": (
            "manorl_accepted_parent_descriptor_variant_v1:" + descriptor_sha256
        ),
        "parent_uuid": parent.parent_row_uuid,
        "parent_dataset_path": parent.parent_dataset_path,
        "parent_dataset_version": parent.parent_dataset_version,
        "object_init_xy_offset_m": list(parent.object_init_xy_offset_m),
        "acceptance": acceptance,
        "late_contact_margin_frames": late_contact_margin,
        "normalized_robustness_margins": normalized,
        "weakest_normalized_margin": min(normalized.values()),
        "sum_normalized_margins": sum(normalized.values()),
    }


def _rank(record: dict[str, Any]) -> tuple[float, float, float, int, int, str]:
    acceptance = record["acceptance"]
    return (
        float(record["weakest_normalized_margin"]),
        float(record["sum_normalized_margins"]),
        -float(acceptance["final_rotation_xyz_mean_error_deg"]),
        int(acceptance["hand_object_contact_frames"]),
        int(record["late_contact_margin_frames"]),
        str(record["parent_uuid"]),
    )


def merge_sets(inputs: list[Path], *, output_dir: Path) -> Path:
    if not inputs:
        raise ValueError("at least one parent descriptor set is required")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "parents_by_action.json"
    manifest_path = output_dir / "merge_manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"parent merge output exists: {output_dir}")
    variants_by_identity: dict[str, list[dict[str, Any]]] = {}
    dataset_cache: dict[tuple[str, int], Any] = {}
    input_records = []
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        values = json.loads(path.read_text(encoding="utf-8"))
        if values.get("contract") != PARENTS_BY_ACTION_CONTRACT:
            raise ValueError(f"unsupported parents-by-action contract: {path}")
        input_records.append({"path": str(path), "sha256": file_sha256(path)})
        for action, mapping in (values.get("actions") or {}).items():
            if not isinstance(mapping, dict):
                raise ValueError(f"action {action} descriptor mapping is invalid")
            for identity, descriptor_raw in mapping.items():
                record = _variant_record(
                    Path(str(descriptor_raw)), dataset_cache=dataset_cache
                )
                if record["source_identity"] != identity or record["action"] != action:
                    raise ValueError("descriptor set identity/action binding changed")
                variants_by_identity.setdefault(identity, []).append(record)
    if not variants_by_identity:
        raise ValueError("parent descriptor sets contain no variants")
    selected: dict[str, dict[str, Any]] = {}
    actions: dict[str, dict[str, str]] = {}
    for identity, variants in sorted(variants_by_identity.items()):
        semantic_ids = [str(record["variant_identity"]) for record in variants]
        if len(set(semantic_ids)) != len(semantic_ids):
            raise ValueError(f"duplicate parent descriptor variant for {identity}")
        # Base rollout UUIDs intentionally predate seed/offset identity and can
        # repeat across bootstrap datasets. The descriptor digest includes the
        # parent dataset/version/row, seed, offset, checkpoint, and anchors, so
        # it is the correct physical parent-variant identity.
        winner = max(variants, key=_rank)
        action = str(winner["action"])
        actions.setdefault(action, {})[identity] = str(winner["descriptor_path"])
        selected[identity] = winner
    output_path.write_text(
        json.dumps(
            {"contract": PARENTS_BY_ACTION_CONTRACT, "actions": actions},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "contract": MERGE_CONTRACT,
        "inputs": input_records,
        "selection_rule": {
            "primary": "maximum weakest normalized robustness margin",
            "normalized_margins": {
                "final_rotation": "(35deg-error)/35deg",
                "contact_frames": "(frames-100)/100",
                "late_contact": "(last_contact-movement_end)/15",
            },
            "variant_identity": (
                "SHA256 of canonical accepted-parent descriptor; base rollout UUID "
                "may repeat across no-prefix seed/offset bootstrap datasets"
            ),
            "ties": (
                "maximum sum margins, lower rotation error, more contact frames, "
                "later contact, lexical parent UUID"
            ),
        },
        "unique_source_identities": len(variants_by_identity),
        "variant_count": sum(len(values) for values in variants_by_identity.values()),
        "selected": selected,
        "variants": variants_by_identity,
        "parents_by_action": str(output_path),
        "parents_by_action_sha256": file_sha256(output_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parents-by-action",
        type=Path,
        action="append",
        required=True,
        help="repeat for every accepted-parent descriptor set",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = merge_sets(args.parents_by_action, output_dir=args.output_dir)
    values = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(path),
                "unique_source_identities": values["unique_source_identities"],
                "variant_count": values["variant_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
