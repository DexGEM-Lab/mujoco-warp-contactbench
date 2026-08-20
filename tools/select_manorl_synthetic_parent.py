#!/usr/bin/env python3
"""Select one accepted scalable-synthesis row for approach augmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim.manorl.synthetic_parent import (
    ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
    RETREAT_ANCHOR_OFFSET_FRAMES,
    AcceptedSyntheticParent,
    write_accepted_synthetic_parent,
)


def select_parent(
    dataset_path: str | Path,
    *,
    row_uuid: str,
    output: str | Path,
    dataset_version: int | None = None,
    replace: bool = False,
) -> Path:
    import lance

    dataset = lance.dataset(str(Path(dataset_path).expanduser().resolve()), version=dataset_version)
    # Locate the UUID through the small index projection, then decode contact,
    # reference, and command mapping for that one row only. Decoding those
    # nested columns for the full production dataset is needlessly expensive.
    index_rows = dataset.scanner(columns=["index"]).to_table().to_pylist()
    matched_indices = [
        row_index
        for row_index, row in enumerate(index_rows)
        if (row.get("index") or {}).get("uuid") == row_uuid
    ]
    if len(matched_indices) != 1:
        raise LookupError(
            f"expected exactly one accepted parent row UUID {row_uuid!r}, got {len(matched_indices)}"
        )
    parent_row_index = matched_indices[0]
    row = dataset.take([parent_row_index]).to_pylist()[0]
    provenance = dict(row.get("provenance") or {})
    objects = row.get("objects") or []
    reference = dict(row.get("reference") or {})
    if len(objects) != 1 or not isinstance(objects[0], dict):
        raise ValueError("accepted parent must contain exactly one object")
    actual_object = np.asarray(objects[0].get("pos") or (), dtype=np.float64)
    reference_object = np.asarray(reference.get("object_pos") or (), dtype=np.float64)
    if actual_object.ndim != 2 or reference_object.shape != actual_object.shape or actual_object.shape[1:] != (3,):
        raise ValueError("accepted parent object/reference positions are not frame-aligned")
    offset = actual_object[0, :2] - reference_object[0, :2]
    source_identity = provenance.get("source_identity")
    if not isinstance(source_identity, str):
        raise ValueError("accepted parent omits source_identity")
    object_type = source_identity.split("_", 1)[0]
    movement_rows = (
        ((row.get("trajectory_metadata") or {}).get("trajectory_info") or {}).get(
            "object_move"
        )
        or []
    )
    movement_entry = next(
        (
            entry
            for entry in movement_rows
            if entry.get("object_name") == object_type
        ),
        None,
    )
    if movement_entry is None:
        raise ValueError("accepted parent omits object movement metadata")
    parent_movement_end = int(movement_entry.get("end_frame"))
    contact_frames = row.get("contact") or []
    last_contact_states = []
    for state_index, frame in enumerate(contact_frames):
        for contact in frame or []:
            if (
                contact.get("hand_name") != "right"
                or contact.get("object_name") != object_type
            ):
                continue
            solved = False
            for pair in contact.get("contact_pairs") or []:
                force = np.asarray(
                    pair.get("force_normal") or (), dtype=np.float64
                )
                if force.shape == (3,) and np.linalg.norm(force) > 0.2:
                    solved = True
                    break
            if solved:
                last_contact_states.append(state_index)
                break
    if not last_contact_states:
        raise ValueError("accepted parent contains no solved right-hand/object contact")
    last_contact_state = int(last_contact_states[-1])
    if last_contact_state < parent_movement_end:
        raise ValueError(
            "accepted parent last contact precedes movement end: "
            f"contact={last_contact_state}, movement_end={parent_movement_end}"
        )
    retreat_offset = RETREAT_ANCHOR_OFFSET_FRAMES
    retreat_anchor_state = parent_movement_end + retreat_offset
    if retreat_anchor_state >= len(contact_frames):
        raise ValueError(
            "accepted parent ends before its movement-end+offset retreat"
        )
    reference_source = np.asarray(
        reference.get("source_frame_index") or (), dtype=np.int64
    )
    if reference_source.shape != (len(contact_frames),):
        raise ValueError(
            "accepted parent reference/source mapping is not state-aligned"
        )
    retreat_anchor_source = int(reference_source[retreat_anchor_state])
    reference_hand = np.asarray(
        reference.get("hand_urdf_dof") or (), dtype=np.float64
    )
    if reference_hand.ndim != 2 or reference_hand.shape != (len(contact_frames), 28):
        raise ValueError(
            "accepted parent reference hand_urdf_dof is not state-aligned (T, 28)"
        )
    retreat_horizontal_distance = float(
        np.linalg.norm(
            reference_hand[-1, :2]
            - reference_hand[retreat_anchor_state, :2]
        )
    )
    if retreat_horizontal_distance <= 1e-9:
        raise ValueError(
            "accepted parent has no nonzero horizontal retreat direction after movement-end+offset"
        )
    command_reference = np.asarray(
        row.get("command_reference_index") or (), dtype=np.int64
    )
    command_source = np.asarray(
        row.get("command_source_frame_index") or (), dtype=np.int64
    )
    matching_transitions = np.flatnonzero(
        command_reference == retreat_anchor_state
    )
    if (
        len(matching_transitions) < 1
        or np.any(command_source[matching_transitions] != retreat_anchor_source)
    ):
        raise ValueError(
            "accepted parent retreat anchor disagrees with command/source mapping"
        )

    source_dataset_path = str(provenance.get("dataset_path") or "")
    source_dataset_version = int(provenance.get("dataset_version"))
    source_row_index = int(provenance.get("row_index"))
    source_dataset = lance.dataset(
        source_dataset_path, version=source_dataset_version
    )
    source_row = source_dataset.take(
        [source_row_index], columns=["hands", "trajectory_metadata"]
    ).to_pylist()[0]
    hand_names = list(
        (source_row.get("trajectory_metadata") or {}).get("hand_names") or []
    )
    if "right" not in hand_names:
        raise ValueError("accepted source row omits a right-hand slot")
    right_slot = hand_names.index("right")
    hands = source_row.get("hands") or []
    if not 0 <= right_slot < len(hands):
        raise ValueError("accepted source row right-hand slot is absent")
    raw_q = np.asarray(hands[right_slot].get("urdf_dof") or (), dtype=np.float64)
    if raw_q.ndim != 2 or raw_q.shape[1] != 28 or len(raw_q) < 1:
        raise ValueError("accepted source row right-hand urdf_dof must be (T, 28)")
    source_frame0_q_ref_3_28 = tuple(float(value) for value in raw_q[0, 3:28])

    parent = AcceptedSyntheticParent(
        contract=ACCEPTED_SYNTHETIC_PARENT_CONTRACT,
        parent_dataset_path=str(Path(dataset_path).expanduser().resolve()),
        parent_dataset_version=int(dataset.version),
        parent_row_index=parent_row_index,
        parent_row_uuid=row_uuid,
        parent_row_contract=str(provenance.get("contract") or ""),
        source_identity=source_identity,
        source_dataset_path=source_dataset_path,
        source_dataset_version=source_dataset_version,
        source_row_index=source_row_index,
        checkpoint_sha256=str(provenance.get("checkpoint_sha256") or ""),
        checkpoint_update=int(provenance.get("checkpoint_update")),
        parent_seed=int(provenance.get("seed")),
        parent_episode_index=int(provenance.get("episode_index")),
        parent_generation_attempt=int(provenance.get("generation_attempt")),
        object_init_xy_offset_m=(float(offset[0]), float(offset[1])),
        reference_fps=int(provenance.get("reference_fps")),
        retreat_last_contact_state_index=last_contact_state,
        retreat_anchor_state_index=retreat_anchor_state,
        retreat_anchor_source_frame_index=retreat_anchor_source,
        retreat_anchor_horizontal_distance_m=retreat_horizontal_distance,
        retreat_anchor_offset_frames=retreat_offset,
        parent_movement_end_state_index=parent_movement_end,
        source_row_frame0_right_q_ref_3_28=source_frame0_q_ref_3_28,
    )
    return write_accepted_synthetic_parent(parent, output, replace=replace)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="prior scalable synthetic Lance")
    parser.add_argument("--row-uuid", required=True, help="accepted generated row UUID")
    parser.add_argument("--output", type=Path, required=True, help="accepted-parent JSON descriptor")
    parser.add_argument("--dataset-version", type=int)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = select_parent(
        args.input,
        row_uuid=args.row_uuid,
        output=args.output,
        dataset_version=args.dataset_version,
        replace=args.replace,
    )
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
