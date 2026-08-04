"""Repeatedly verify and assign an MTP catalog without Lance/PyArrow."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

from sim.manorl.trajectory import TrajectorySelection
from sim.manorl.trajectory_package import assign_trajectory_catalog, load_trajectory_package


def _assignment_digest(batch) -> str:
    digest = hashlib.sha256()
    for env_id, trajectory in enumerate(batch.trajectories):
        digest.update(
            json.dumps(
                {
                    "env_id": env_id,
                    "identity": trajectory.identity.identity,
                    "row_index": trajectory.identity.row_index,
                    "uuid": trajectory.identity.uuid,
                    "source_start": int(trajectory.source_indices[0]),
                    "source_stop": int(trajectory.source_indices[-1]) + 1,
                    "movement_start_step": trajectory.movement_start_step,
                    "movement_end_step": trajectory.movement_end_step,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-package", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--selector", default="all")
    parser.add_argument("--pair-assignment-cycle", type=int, default=0)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.iterations < 1:
        parser.error("num-envs and iterations must be positive")
    package = args.trajectory_package.expanduser().resolve()
    assignment_digest: str | None = None
    package_digest: str | None = None
    catalog_digest: str | None = None
    trajectory_count: int | None = None
    pair_count: int | None = None
    for iteration in range(1, args.iterations + 1):
        catalog = load_trajectory_package(package)
        manifest_selection = catalog.manifest["selection"]
        dataset = catalog.manifest["dataset"]
        selection = TrajectorySelection(
            selector=args.selector,
            dataset_path=Path(dataset["logical_path"]),
            expected_dataset_version=int(dataset["version"]),
            pre_padding=int(manifest_selection["pre_padding"]),
            post_padding=int(manifest_selection["post_padding"]),
            hand_side=str(manifest_selection["hand_side"]),
            reference_fps=int(manifest_selection["reference_fps"]),
            control_fps=int(manifest_selection["control_fps"]),
            pair_assignment_cycle=args.pair_assignment_cycle,
        )
        batch = assign_trajectory_catalog(catalog, selection, num_envs=args.num_envs)
        current_assignment_digest = _assignment_digest(batch)
        values = (
            catalog.package_digest,
            catalog.catalog_digest,
            len(catalog.trajectories),
            len(catalog.resolved_pairs),
            current_assignment_digest,
        )
        expected = (
            package_digest,
            catalog_digest,
            trajectory_count,
            pair_count,
            assignment_digest,
        )
        if iteration > 1 and values != expected:
            raise RuntimeError(
                f"trajectory package verification changed at iteration {iteration}: "
                f"{values!r} != {expected!r}"
            )
        (
            package_digest,
            catalog_digest,
            trajectory_count,
            pair_count,
            assignment_digest,
        ) = values
        del batch, catalog
        gc.collect()
    if "lance" in sys.modules or "pyarrow" in sys.modules:
        raise RuntimeError("Lance/PyArrow entered the MTP validator process")
    print(
        json.dumps(
            {
                "trajectory_package": str(package),
                "package_digest": package_digest,
                "catalog_digest": catalog_digest,
                "trajectory_count": trajectory_count,
                "pair_count": pair_count,
                "num_envs": args.num_envs,
                "iterations": args.iterations,
                "assignment_digest": assignment_digest,
                "lance_imported": False,
                "pyarrow_imported": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
