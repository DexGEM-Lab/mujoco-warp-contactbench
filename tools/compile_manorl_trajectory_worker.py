"""Short-lived Lance worker for ManoRL trajectory-package compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sim.manorl.trajectory import (
    LANCE_COLUMNS,
    ObjectActionPair,
    TrajectorySelection,
    _discover_trajectory_candidates,
    _selected_trajectory_from_row,
)
from sim.manorl.trajectory_package import write_trajectory_package


def _selection(path: Path) -> TrajectorySelection:
    values = json.loads(path.read_text(encoding="utf-8"))
    return TrajectorySelection(
        object_type=str(values.get("object_type", "cube1")),
        gesture=str(values.get("gesture", "01")),
        selector=str(values["selector"]),
        dataset_path=Path(values["dataset_path"]),
        expected_dataset_version=int(values["dataset_version"]),
        pre_padding=int(values["pre_padding"]),
        post_padding=int(values["post_padding"]),
        hand_side=str(values["hand_side"]),
        reference_fps=int(values["reference_fps"]),
        control_fps=int(values["control_fps"]),
        pair_assignment_cycle=0,
    )


def _dataset(selection: TrajectorySelection):
    # Lance/PyArrow exist only below this short-lived worker boundary.
    import lance

    dataset = lance.dataset(
        str(selection.dataset_path), version=selection.expected_dataset_version
    )
    if int(dataset.version) != selection.expected_dataset_version:
        raise RuntimeError(
            f"dataset version {dataset.version} != {selection.expected_dataset_version}"
        )
    return dataset


def _discover(selection: TrajectorySelection, output: Path) -> None:
    dataset = _dataset(selection)
    pairs, candidates_by_pair = _discover_trajectory_candidates(dataset, selection)
    candidates = [
        {
            "row_index": candidate.row_index,
            "pair": pair.canonical,
            "identity": candidate.identity,
            "sequence": candidate.sequence,
        }
        for pair in pairs
        for candidate in candidates_by_pair[pair]
    ]
    payload = {
        "dataset_version": int(dataset.version),
        "dataset_schema_digest": hashlib.sha256(str(dataset.schema).encode("utf-8")).hexdigest(),
        "resolved_pairs": [pair.canonical for pair in pairs],
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _decode(
    selection: TrajectorySelection,
    requests_path: Path,
    output: Path,
    rejections_output: Path,
    *,
    dataset_schema_digest: str,
    discovery_digest: str,
) -> None:
    requests = json.loads(requests_path.read_text(encoding="utf-8"))
    if not isinstance(requests, list) or not requests:
        raise ValueError("decode request must be a non-empty candidate list")
    dataset = _dataset(selection)
    rows = dataset.take(
        [int(item["row_index"]) for item in requests], columns=list(LANCE_COLUMNS)
    ).to_pylist()
    if len(rows) != len(requests):
        raise RuntimeError("Lance did not return every requested trajectory row")
    trajectories = []
    rejections: list[dict[str, object]] = []
    for request, row in zip(requests, rows, strict=True):
        pair = ObjectActionPair(*str(request["pair"]).split(":", maxsplit=1))
        try:
            trajectory = _selected_trajectory_from_row(
                row,
                int(dataset.version),
                row_index=int(request["row_index"]),
                selection=selection,
                expected_pair=pair,
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            # Existing direct-Lance selection defines these rows as invalid
            # candidates. Make that exclusion explicit and hash-bound rather
            # than silently skipping it or retrying a deterministic defect.
            rejections.append(
                {
                    "row_index": int(request["row_index"]),
                    "identity": str(request["identity"]),
                    "pair": pair.canonical,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        if trajectory.identity.identity != request["identity"]:
            raise RuntimeError(
                f"decoded row {request['row_index']} identity mismatch: "
                f"{trajectory.identity.identity!r} != {request['identity']!r}"
            )
        trajectories.append(trajectory)
    rejections_output.parent.mkdir(parents=True, exist_ok=True)
    rejections_output.write_text(
        json.dumps(rejections, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if trajectories:
        write_trajectory_package(
            output,
            trajectories,
            selection=selection,
            dataset_schema_digest=dataset_schema_digest,
            discovery_digest=discovery_digest,
            compiler={"kind": "isolated_decode_shard", "request_count": len(requests)},
            source_candidates=requests,
            source_rejections=rejections,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("discover", "decode"))
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests-json", type=Path)
    parser.add_argument("--rejections-json", type=Path)
    parser.add_argument("--dataset-schema-digest")
    parser.add_argument("--discovery-digest")
    args = parser.parse_args(argv)
    selection = _selection(args.selection_json)
    if args.mode == "discover":
        if args.requests_json is not None:
            parser.error("discover does not accept --requests-json")
        _discover(selection, args.output)
    else:
        if args.requests_json is None or args.rejections_json is None:
            parser.error("decode requires --requests-json and --rejections-json")
        if not args.dataset_schema_digest or not args.discovery_digest:
            parser.error("decode requires source digests")
        _decode(
            selection,
            args.requests_json,
            args.output,
            args.rejections_json,
            dataset_schema_digest=args.dataset_schema_digest,
            discovery_digest=args.discovery_digest,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
