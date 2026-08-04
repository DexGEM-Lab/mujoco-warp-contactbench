"""Compare one MTP assignment against its pinned Lance source on a healthy host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.manorl.trajectory import TrajectorySelection, load_assigned_trajectory_batch
from sim.manorl.trajectory_package import assign_trajectory_catalog, load_trajectory_package


def _assert_equal(actual, expected) -> None:
    if actual.identity != expected.identity:
        raise AssertionError(
            f"identity mismatch: package={actual.identity!r}, source={expected.identity!r}"
        )
    scalar_names = (
        "dataset_version",
        "object_z_shift",
        "hand_sides",
        "selected_hand_sides",
        "reference_fps",
        "control_fps",
        "movement_start_step",
        "movement_end_step",
    )
    for name in scalar_names:
        if getattr(actual, name) != getattr(expected, name):
            raise AssertionError(
                f"{actual.identity.identity} {name} mismatch: "
                f"{getattr(actual, name)!r} != {getattr(expected, name)!r}"
            )
    array_names = (
        "source_indices",
        "timestamps",
        "object_pos_raw",
        "object_pos",
        "object_quat_xyzw",
    )
    for name in array_names:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for side in actual.hand_sides:
        np.testing.assert_array_equal(actual.q_ref_for(side), expected.q_ref_for(side))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-package", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.num_envs < 1:
        parser.error("num-envs must be positive")
    package = args.trajectory_package.expanduser().resolve()
    catalog = load_trajectory_package(package)
    manifest_selection = catalog.manifest["selection"]
    dataset = catalog.manifest["dataset"]
    selection = TrajectorySelection(
        selector="all",
        dataset_path=Path(dataset["logical_path"]),
        expected_dataset_version=int(dataset["version"]),
        pre_padding=int(manifest_selection["pre_padding"]),
        post_padding=int(manifest_selection["post_padding"]),
        hand_side=str(manifest_selection["hand_side"]),
        reference_fps=int(manifest_selection["reference_fps"]),
        control_fps=int(manifest_selection["control_fps"]),
    )
    package_batch = assign_trajectory_catalog(
        catalog, selection, num_envs=args.num_envs
    )
    source_batch = load_assigned_trajectory_batch(selection, num_envs=args.num_envs)
    package_keys = [
        (item.identity.row_index, item.identity.identity) for item in package_batch.trajectories
    ]
    source_keys = [
        (item.identity.row_index, item.identity.identity) for item in source_batch.trajectories
    ]
    if package_keys != source_keys:
        mismatch = next(
            index
            for index, (package_key, source_key) in enumerate(
                zip(package_keys, source_keys, strict=True)
            )
            if package_key != source_key
        )
        raise AssertionError(
            f"assignment mismatch at env {mismatch}: "
            f"package={package_keys[mismatch]!r}, source={source_keys[mismatch]!r}"
        )
    package_unique = {
        (item.identity.row_index, item.identity.identity): item
        for item in package_batch.trajectories
    }
    source_unique = {
        (item.identity.row_index, item.identity.identity): item
        for item in source_batch.trajectories
    }
    if set(package_unique) != set(source_unique):
        raise AssertionError("package/source unique trajectory catalogs differ")
    for key in sorted(package_unique):
        _assert_equal(package_unique[key], source_unique[key])
    print(
        json.dumps(
            {
                "trajectory_package": str(package),
                "package_digest": catalog.package_digest,
                "catalog_digest": catalog.catalog_digest,
                "trajectory_count": len(catalog.trajectories),
                "pair_count": len(catalog.resolved_pairs),
                "assigned_envs": args.num_envs,
                "unique_assigned_trajectories": len(package_unique),
                "source_equivalent": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
