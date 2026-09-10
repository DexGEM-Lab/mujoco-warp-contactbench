#!/usr/bin/env python3
"""Generate and run receiver-bound bowl-hold contact-template transfers.

The donor grasp is stored in donor-object coordinates.  For each receiver this
script applies ``R_receiver_initial^-1 * R_donor_initial`` to that local pose,
so the same world rim side is selected at acquisition while the receiver's own
object-pose time series remains the trajectory reference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import lance
import numpy as np
from scipy.spatial.transform import Rotation

DATASET_NAME = "dexgem_vla_demo_guangxue_cma2lance_20260909_060700.lance"
RECEIVER_ROWS = (50, 51, 54, 55, 58)
DEFAULT_DONOR_PATCH = Path("/mnt/nas-222-project/mocap_v2/lance_datasets/dexgem_vla_demo_guangxue_astra_repair_5samples_20260910/patches/guangxue_bowl09_row49_v3.json")
DEFAULT_REPLAY = Path("/mnt/nas-222-project/mocap_v2/lance_datasets/dexgem_vla_demo_guangxue_astra_repair_5samples_20260910/tools/replay_repaired_capture.py")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("generate", "run"))
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--patch-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--donor-patch", type=Path, default=DEFAULT_DONOR_PATCH)
    p.add_argument("--replay-tool", type=Path, default=DEFAULT_REPLAY)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--post-padding", type=int, default=500,
                   help="100 Hz held tail; 500 is five seconds")
    return p


def _base(row: dict, version: int, row_index: int, post_padding: int):
    # The caller must launch from a neutral cwd: the materialized source sim is
    # intentionally selected by PYTHONPATH rather than this case worktree.
    from sim.manorl.trajectory import trajectory_from_lance_row, resample_reference_trajectory
    return resample_reference_trajectory(
        trajectory_from_lance_row(row, version, row_index=row_index,
                                  hand_side="right", pre_padding=180,
                                  post_padding=post_padding),
        reference_fps=100, control_fps=100,
    )


def _row(dataset, index: int) -> dict:
    return dataset.take([index], columns=[
        "index", "trajectory_metadata", "timestamp", "hands", "objects",
    ]).to_pylist()[0]


def _phase_stage(donor_base, receiver_base) -> tuple[int, int]:
    """Map donor's late leveling correction by lift/hold phase, not frame ID."""
    donor_start, donor_end = donor_base.movement_start_step, donor_base.movement_end_step
    receiver_start, receiver_end = receiver_base.movement_start_step, receiver_base.movement_end_step
    donor_late_start = 310
    donor_hold_tail = 500 - donor_end
    lift_fraction = (donor_late_start - donor_start) / (donor_end - donor_start)
    return (
        int(round(receiver_start + lift_fraction * (receiver_end - receiver_start))),
        int(receiver_end + donor_hold_tail),
    )


def _make_patch(donor: dict, donor_base, receiver: dict, receiver_base) -> dict:
    donor_grasp = donor["edit"]["object_relative_grasp"]
    donor_object_rotation = Rotation.from_quat(
        donor_base.scene_object_initial_quat_xyzw[donor_base.identity.object_index]
    )
    receiver_object_rotation = Rotation.from_quat(
        receiver_base.scene_object_initial_quat_xyzw[receiver_base.identity.object_index]
    )
    receiver_from_donor = receiver_object_rotation.inv() * donor_object_rotation
    donor_relative_rotation = Rotation.from_quat(donor_grasp["rotation_xyzw"])
    corrected_grasp = {
        "translation": receiver_from_donor.apply(donor_grasp["translation"]).tolist(),
        "rotation_xyzw": (receiver_from_donor * donor_relative_rotation).as_quat().tolist(),
    }
    stage_start, stage_end = _phase_stage(donor_base, receiver_base)
    movement_start = int(receiver_base.movement_start_step)
    return {
        "schema": "direct_capture_repair.v1",
        "name": f"guangxue_bowl09_row{receiver['index']['row_index'] if 'row_index' in receiver['index'] else 'receiver'}_donor49_template_v1",
        "source": {
            "dataset_name": DATASET_NAME,
            "version": 5,
            "row": None,  # inserted by caller because Lance rows do not carry index
            "uuid": receiver["index"]["uuid"],
            "frames": int(receiver["trajectory_metadata"]["total_frames"]),
        },
        "solver": {"cone": "elliptic", "impratio": 100.0},
        "active_object": "bowl",
        "transfer": {
            "donor_patch": "guangxue_bowl09_row49_v3.json",
            "donor_source": donor["source"],
            "coordinate_correction": "R_receiver_initial^-1 * R_donor_initial applied to donor object-relative grasp",
            "receiver_from_donor_rotation_xyzw": receiver_from_donor.as_quat().tolist(),
            "timing": {
                "receiver_movement_steps": [movement_start, int(receiver_base.movement_end_step)],
                "prefix_preserved_through_step": movement_start - 80,
                "grasp_blend_steps": [movement_start - 80, movement_start - 20],
                "finger_close_steps": [movement_start - 45, movement_start - 15],
                "late_leveling_steps": [stage_start, stage_end],
                "late_leveling_mapping": "donor lift fraction to receiver lift start; donor hold-tail duration after receiver movement end",
            },
        },
        "edit": {
            "name": "donor49_contact_template_receiver_timing",
            "translation": donor["edit"]["translation"],
            "finger_targets": donor["edit"]["finger_targets"],
            "object_relative_grasp": corrected_grasp,
            "grasp_approach_start_offset": -80,
            "grasp_approach_end_offset": -20,
            "close_start_offset": -45,
            "close_end_offset": -15,
            "hand_offsets": donor["edit"]["hand_offsets"],
            "stages": [{
                "start_step": stage_start,
                "end_step": stage_end,
                "wrist_rotation_world": donor["edit"]["stages"][0]["wrist_rotation_world"],
                "pivot_from_object": donor["edit"]["stages"][0]["pivot_from_object"],
            }],
        },
        "scope": "Receiver-bound first-pass donor49 bowl contact-template transfer. Source hand prefix and source object pose/timing remain reference data; only the bounded hand contact transition and phase-mapped late leveling are edited. Object poses are never forced during replay.",
    }


def generate(args) -> list[Path]:
    if args.dataset.name != DATASET_NAME:
        raise ValueError("unexpected dataset basename")
    donor = json.loads(args.donor_patch.read_text())
    if donor.get("source", {}).get("row") != 49 or donor.get("solver") != {"cone": "elliptic", "impratio": 100.0}:
        raise ValueError("expected immutable row49 elliptic/impratio100 donor patch")
    dataset = lance.dataset(args.dataset, version=5)
    donor_row = _row(dataset, 49)
    donor_base = _base(donor_row, dataset.version, 49, args.post_padding)
    args.patch_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for index in RECEIVER_ROWS:
        receiver = _row(dataset, index)
        receiver_base = _base(receiver, dataset.version, index, args.post_padding)
        if receiver_base.scene_object_types[receiver_base.identity.object_index] != "bowl":
            raise ValueError(f"row {index} does not select bowl")
        patch = _make_patch(donor, donor_base, receiver, receiver_base)
        patch["name"] = f"guangxue_bowl09_row{index}_donor49_template_v1"
        patch["source"]["row"] = index
        path = args.patch_dir / f"guangxue_bowl09_row{index}_donor49_template_v1.json"
        path.write_text(json.dumps(patch, indent=2) + "\n")
        generated.append(path)
    return generated


def run(args) -> None:
    if args.output_root is None:
        raise ValueError("--output-root is required for run")
    patches = generate(args)
    args.output_root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/jay/dexrobot/FromSSH/manoRL_mujoco" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    for patch in patches:
        row = json.loads(patch.read_text())["source"]["row"]
        output = args.output_root / f"row{row}"
        command = [str(args.python), str(args.replay_tool), "--dataset", str(args.dataset),
                   "--patch", str(patch.resolve()), "--post-padding", str(args.post_padding),
                   "--output", str(output), "--solver", "recorded"]
        print(json.dumps({"row": row, "command": command}), flush=True)
        subprocess.run(command, cwd="/tmp", env=env, check=True)


def main() -> None:
    args = parser().parse_args()
    if args.post_padding < 300 or args.post_padding > 500:
        raise ValueError("first-pass held tail must be 3–5 seconds (300–500 frames at 100 Hz)")
    if args.command == "generate":
        for path in generate(args):
            print(path)
    else:
        run(args)


if __name__ == "__main__":
    main()
