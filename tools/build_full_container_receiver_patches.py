#!/usr/bin/env python3
"""Build and validate CPU-only bottle/pitcher receiver repair assets.

The generated files are receiver-bound: every patch identifies its own Lance
UUID and each pitcher command track is a phase-resampled donor residual added
to the receiver's decoded reference.  This utility never opens MuJoCo, Warp,
or a GPU and never writes the source or donor datasets.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lance
import numpy as np
from scipy.spatial.transform import Rotation

from sim.manorl.trajectory import resample_reference_trajectory, trajectory_from_lance_row

SOURCE_NAME = "dexgem_vla_demo_guangxue_cma2lance_20260909_060700.lance"
SOURCE_DEFAULT = Path("/mnt/nas-222-project/mocap_v2/lance_datasets") / SOURCE_NAME
DONOR_DEFAULT = Path("/mnt/nas-222-project/mocap_v2/lance_datasets/dexgem_vla_demo_guangxue_astra_repair_5samples_20260910")
OUTPUT_DEFAULT = Path(__file__).resolve().parents[1] / "patches" / "full_containers"
BOTTLE_RECEIVERS = tuple(range(20, 31))
PITCHER_RECEIVERS = (32, 33, 34, 35, 37, 38, 39)
DONOR_ROWS = {"bottle": 19, "pitcher": 31}
PITCHER_PRESERVED_PREFIX_END = 39


@dataclass(frozen=True)
class Decoded:
    row: int
    uuid: str
    frames: int
    active_object: str
    base: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--donor", type=Path, default=DONOR_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--check", action="store_true", help="validate existing output without writing")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_row(dataset: Any, row_index: int) -> dict[str, Any]:
    return dataset.take(
        [row_index], columns=["index", "trajectory_metadata", "timestamp", "hands", "objects"]
    ).to_pylist()[0]


def decode(dataset: Any, row_index: int) -> Decoded:
    row = dataset_row(dataset, row_index)
    base = resample_reference_trajectory(
        trajectory_from_lance_row(
            row, dataset.version, row_index=row_index, hand_side="right", pre_padding=180, post_padding=180
        ),
        reference_fps=100,
        control_fps=100,
    )
    return Decoded(
        row=row_index,
        uuid=str(row["index"]["uuid"]),
        frames=int(row["trajectory_metadata"]["total_frames"]),
        active_object=base.scene_object_types[base.identity.object_index],
        base=base,
    )


def retime_step(step: int, donor_end: int, receiver_end: int) -> int:
    return int(round(int(step) * receiver_end / donor_end))


def retime_interval(start: int, end: int, donor_end: int, receiver_end: int) -> tuple[int, int]:
    mapped_start = retime_step(start, donor_end, receiver_end)
    mapped_end = max(mapped_start + 1, retime_step(end, donor_end, receiver_end))
    return mapped_start, mapped_end


def yaw(quat_xyzw: np.ndarray) -> float:
    return float(Rotation.from_quat(quat_xyzw).as_euler("ZYX")[0])


def unit(vector: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude == 0.0:
        raise ValueError("approach vector must be nonzero")
    return vector / magnitude


def oriented_grasp(donor_grasp: dict[str, Any], donor: Decoded, receiver: Decoded) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Select donor-aligned or receiver-local grasp using observed receiver approach."""
    donor_rotation = Rotation.from_quat(donor.base.object_quat_xyzw[0])
    receiver_rotation = Rotation.from_quat(receiver.base.object_quat_xyzw[0])
    donor_translation = np.asarray(donor_grasp["translation"], dtype=np.float64)
    donor_wrist_rotation = Rotation.from_quat(donor_grasp["rotation_xyzw"])
    observed = receiver.base.q_ref[receiver.base.movement_start_step, :3] - receiver.base.object_pos[
        receiver.base.movement_start_step
    ]
    alignment = receiver_rotation.inv() * donor_rotation
    candidates = {
        "donor_initial_alignment": (alignment.apply(donor_translation), alignment * donor_wrist_rotation),
        "receiver_local_alignment": (donor_translation, donor_wrist_rotation),
    }
    scored: dict[str, float] = {}
    for name, (local_translation, _) in candidates.items():
        scored[name] = float(np.dot(unit(receiver_rotation.apply(local_translation)), unit(observed)))
    strategy = max(scored, key=scored.get)
    local_translation, local_rotation = candidates[strategy]
    return (
        {"translation": local_translation.tolist(), "rotation_xyzw": local_rotation.as_quat().tolist()},
        {
            "strategy": strategy,
            "candidate_approach_cosines": scored,
            "donor_initial_yaw_rad": yaw(donor.base.object_quat_xyzw[0]),
            "receiver_initial_yaw_rad": yaw(receiver.base.object_quat_xyzw[0]),
            "receiver_observed_approach_world": observed.tolist(),
            "receiver_initial_inverse_donor_initial_xyzw": alignment.as_quat().tolist(),
        },
    )


def retimed_edit(donor_edit: dict[str, Any], donor: Decoded, receiver: Decoded, *, kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map every donor contact phase onto the receiver's retained time interval."""
    edit = copy.deepcopy(donor_edit)
    donor_end = int(donor.base.movement_end_step)
    receiver_end = int(receiver.base.movement_end_step)
    grasp, orientation = oriented_grasp(donor_edit["object_relative_grasp"], donor, receiver)
    edit["object_relative_grasp"] = grasp

    for key in ("grasp_approach_start_offset", "grasp_approach_end_offset", "close_start_offset", "close_end_offset"):
        absolute = donor.base.movement_start_step + int(donor_edit[key])
        edit[key] = retime_step(absolute, donor_end, receiver_end) - receiver.base.movement_start_step
    # Make release timing explicit: replay defaults are donor-specific and would otherwise drift.
    donor_release_start = int(donor_edit.get("release_start_step", donor.base.movement_end_step - 20))
    donor_release_end = int(donor_edit.get("release_end_step", donor.base.movement_end_step + 50))
    edit["release_start_step"], edit["release_end_step"] = retime_interval(
        donor_release_start, donor_release_end, donor_end, receiver_end
    )
    edit["wrist_release_start_step"] = edit["release_start_step"]
    edit["wrist_release_end_step"] = edit["release_end_step"]

    if "acquisition" in edit:
        for key in ("start_step", "pregrasp_step", "arrive_step", "closed_step"):
            edit["acquisition"][key] = retime_step(int(donor_edit["acquisition"][key]), donor_end, receiver_end)
    for stage, source_stage in zip(edit.get("stages", []), donor_edit.get("stages", []), strict=True):
        stage["start_step"], stage["end_step"] = retime_interval(
            int(source_stage["start_step"]), int(source_stage["end_step"]), donor_end, receiver_end
        )

    first_change = min(
        int(edit.get("acquisition", {}).get("start_step", receiver.base.movement_start_step + edit["grasp_approach_start_offset"])),
        receiver.base.movement_start_step + int(edit["grasp_approach_start_offset"]),
        receiver.base.movement_start_step + int(edit["close_start_offset"]),
    )
    transfer = {
        "template_row": donor.row,
        "receiver_row": receiver.row,
        "kind": kind,
        "phase_mapping": "step = round(donor_step * receiver_movement_end / donor_movement_end)",
        "donor_movement_end_step": donor_end,
        "receiver_movement_end_step": receiver_end,
        "preserved_reference_prefix": [0, first_change - 1],
        "orientation": orientation,
    }
    return edit, transfer


def source_record(receiver: Decoded) -> dict[str, Any]:
    return {"dataset_name": SOURCE_NAME, "version": 5, "row": receiver.row, "uuid": receiver.uuid, "frames": receiver.frames}


def bottle_patch(donor_recipe: dict[str, Any], donor: Decoded, receiver: Decoded) -> dict[str, Any]:
    edit, transfer = retimed_edit(donor_recipe["edit"], donor, receiver, kind="bottle")
    return {
        "schema": "direct_capture_repair.v1",
        "name": f"guangxue_bottle05_row{receiver.row}_receiver_template_v1",
        "active_object": receiver.active_object,
        "source": source_record(receiver),
        "solver": {"cone": "elliptic", "impratio": 100.0},
        "edit": edit,
        "template_transfer": transfer,
        "scope": "Receiver-bound bottle contact template: mapped open acquisition, donor-relative grasp/level correction, and explicit release; receiver object path and noncontact prefix are retained. CPU preparation only; physics outcome untested.",
    }


def pitcher_commands(donor_recipe: dict[str, Any], donor: Decoded, receiver: Decoded, donor_root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    command_record = donor_recipe["command_track"]
    donor_path = donor_root / "patches" / command_record["path"]
    if sha256(donor_path) != command_record["sha256"]:
        raise ValueError("immutable pitcher donor command hash changed")
    with np.load(donor_path, allow_pickle=False) as saved:
        donor_commands = np.asarray(saved["ctrl"], dtype=np.float64)
        donor_map = np.asarray(saved["reference_index"], dtype=np.int64)
    if donor_commands.shape != (int(command_record["frames"]), 28):
        raise ValueError("donor command shape differs from recipe")
    donor_edit, _ = retimed_edit(donor_recipe["edit"], donor, donor, kind="pitcher-donor")
    # Import only the donor's pure recipe edit function; no replay/simulation modules are loaded.
    spec = importlib.util.spec_from_file_location("replay_pose_edits", donor_root / "tools" / "replay_pose_edits.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    donor_reference = module.apply_replay_edits(donor.base, donor_edit).q_ref[donor_map]
    residual = donor_commands - donor_reference
    receiver_map = np.rint(donor_map * receiver.base.movement_end_step / donor.base.movement_end_step).astype(np.int64)
    if np.any(receiver_map < 0) or np.any(receiver_map >= len(receiver.base.q_ref)) or np.any(np.diff(receiver_map) < 0):
        raise ValueError("receiver reference mapping is not monotonic and in range")
    commands = receiver.base.q_ref[receiver_map].copy() + residual
    # The donor recipe begins its explicitly repaired acquisition at command 40.
    # Retaining the receiver's stable approach gives an exact source prefix.
    commands[: PITCHER_PRESERVED_PREFIX_END + 1] = receiver.base.q_ref[receiver_map[: PITCHER_PRESERVED_PREFIX_END + 1]]
    metadata = {
        "method": "donor_frozen_command_minus_donor_reference_residual_added_to_receiver_reference",
        "donor_command_sha256": command_record["sha256"],
        "donor_command_frames": int(command_record["frames"]),
        "preserved_receiver_command_prefix": [0, PITCHER_PRESERVED_PREFIX_END],
        "release": "donor residual after mapped receiver endpoint is retained against the receiver endpoint reference, preserving the verified thumb-clear then sideways handle exit primitive",
        "canonical_dof_order": "ARTx,ARTy,ARTz,ARRx,ARRy,ARRz followed by 22 named finger axes; no qpos reordering",
    }
    return commands, receiver_map, metadata


def pitcher_patch(donor_recipe: dict[str, Any], donor: Decoded, receiver: Decoded, command_name: str, command_hash: str, command_frames: int, transfer_meta: dict[str, Any]) -> dict[str, Any]:
    edit, transfer = retimed_edit(donor_recipe["edit"], donor, receiver, kind="pitcher")
    # The frozen command asset overrides recipe edits at replay time.  Its exact
    # receiver-source prefix, not the illustrative recipe anchor, is authoritative.
    transfer["recipe_contact_phase_anchor"] = transfer["preserved_reference_prefix"]
    transfer["preserved_reference_prefix"] = [0, PITCHER_PRESERVED_PREFIX_END]
    transfer["command_transfer"] = transfer_meta
    return {
        "schema": "direct_capture_repair.v1",
        "name": f"guangxue_pitcher06_row{receiver.row}_receiver_template_v1",
        "active_object": receiver.active_object,
        "source": source_record(receiver),
        "solver": {"cone": "elliptic", "impratio": 100.0},
        "edit": edit,
        "template_transfer": transfer,
        "command_track": {"path": command_name, "sha256": command_hash, "frames": command_frames},
        "scope": "Receiver-bound pitcher frozen command residual: receiver reference/object path plus phase-resampled donor compensation and safe release primitive. CPU preparation only; physics outcome untested.",
    }


def dump_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate(source: Path, donor_root: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset = lance.dataset(source, version=5)
    bottle_recipe = read_json(donor_root / "patches" / "guangxue_bottle05_row19_v2.json")
    pitcher_recipe = read_json(donor_root / "patches" / "guangxue_pitcher06_row31_v1.json")
    donors = {"bottle": decode(dataset, DONOR_ROWS["bottle"]), "pitcher": decode(dataset, DONOR_ROWS["pitcher"])}
    inventory: list[dict[str, Any]] = []
    for row in BOTTLE_RECEIVERS:
        receiver = decode(dataset, row)
        patch_name = f"guangxue_bottle05_row{row}_receiver_template_v1.json"
        dump_json(output / patch_name, bottle_patch(bottle_recipe, donors["bottle"], receiver))
        inventory.append({"row": row, "object": receiver.active_object, "patch": patch_name, "command_track": None})
    for row in PITCHER_RECEIVERS:
        receiver = decode(dataset, row)
        command_name = f"guangxue_pitcher06_row{row}_receiver_commands.npz"
        commands, reference_index, metadata = pitcher_commands(pitcher_recipe, donors["pitcher"], receiver, donor_root)
        command_path = output / command_name
        np.savez_compressed(command_path, ctrl=commands, reference_index=reference_index)
        patch_name = f"guangxue_pitcher06_row{row}_receiver_template_v1.json"
        dump_json(output / patch_name, pitcher_patch(pitcher_recipe, donors["pitcher"], receiver, command_name, sha256(command_path), len(commands), metadata))
        inventory.append({"row": row, "object": receiver.active_object, "patch": patch_name, "command_track": command_name})
    dump_json(output / "receiver_inventory.json", {"schema": "container_receiver_inventory.v1", "source": str(source), "donor_bundle": str(donor_root), "receivers": inventory})
    runner = [
        "# CPU-prepared receiver repair runner list",
        "# Parent executes GPU replay; this preparation does not claim physical success.",
        "# Set REPLAY to immutable donor bundle tools/replay_repaired_capture.py.",
    ]
    for item in inventory:
        runner.append(f"$REPLAY --dataset {source} --patch {output / item['patch']} --output <parent-output>/row{item['row']}")
    (output / "RUNNERS.md").write_text("\n".join(runner) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Full container receiver templates\n\n"
        "This directory prepares 18 UUID-bound receiver repairs from immutable row-19 bottle and row-31 pitcher donors. "
        "It contains no measured replay result. Parent GPU execution uses `RUNNERS.md`.\n\n"
        "- **Bottles 20–30:** retain the receiver object trajectory and source prefix, retime the donor acquisition/level/release anchors by receiver movement duration, and choose the object-frame grasp orientation whose world-side best matches the observed receiver approach.\n"
        "- **Pitchers 32, 33, 34, 35, 37, 38, 39:** retain a 40-command receiver prefix, then add the donor frozen-command minus donor-reference residual to a monotonic phase-resampled receiver reference. The held tail retains the donor thumb-clear then sideways release residual against the receiver endpoint.\n\n"
        "## Supported failure modes\n\n"
        "1. The pitcher donor required 10–14 cm load compensation under its 1.28 kg gravity load; receiver geometry/timing can change the needed residual enough to lose or sag the handle grasp.\n"
        "2. Bottle receivers differ in initial yaw. The recorded approach-side comparison selects an orientation mapping, but an asymmetric bottle can still collide or miss contact under physics.\n"
        "3. Phase resampling preserves receiver source/object references, not contact dynamics; GPU replay is the required discriminator for lift, tilt, placement, and release.\n",
        encoding="utf-8",
    )


def validate(source: Path, donor_root: Path, output: Path) -> None:
    dataset = lance.dataset(source, version=5)
    spec = importlib.util.spec_from_file_location("replay_pose_edits", donor_root / "tools" / "replay_pose_edits.py")
    editor = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(editor)
    inventory = read_json(output / "receiver_inventory.json")["receivers"]
    expected_rows = set(BOTTLE_RECEIVERS + PITCHER_RECEIVERS)
    assert {int(item["row"]) for item in inventory} == expected_rows
    assert len(inventory) == 18
    for item in inventory:
        patch_path = output / item["patch"]
        assert patch_path.parent == output and patch_path.is_file()
        patch = read_json(patch_path)
        assert patch["schema"] == "direct_capture_repair.v1"
        assert patch["solver"] == {"cone": "elliptic", "impratio": 100.0}
        receiver = decode(dataset, int(item["row"]))
        assert patch["source"] == source_record(receiver)
        assert patch["active_object"] == receiver.active_object
        prefix_start, prefix_end = patch["template_transfer"]["preserved_reference_prefix"]
        assert prefix_start == 0 and prefix_end >= 0
        if int(item["row"]) in BOTTLE_RECEIVERS:
            assert "command_track" not in patch
            assert int(patch["edit"]["acquisition"]["start_step"]) == prefix_end + 1
            repaired = editor.apply_replay_edits(receiver.base, patch["edit"])
            # Recipe math touches complete float64 vectors with zero blend before
            # acquisition; only roundoff (< 1e-15) is permitted in this prefix.
            assert np.allclose(repaired.q_ref[: prefix_end + 1], receiver.base.q_ref[: prefix_end + 1], rtol=0.0, atol=1e-14)
            assert np.array_equal(repaired.object_pos, receiver.base.object_pos)
            assert np.allclose(repaired.object_quat_xyzw, receiver.base.object_quat_xyzw, rtol=0.0, atol=1e-15)
        else:
            record = patch["command_track"]
            command_path = output / record["path"]
            assert command_path.parent == output and sha256(command_path) == record["sha256"]
            with np.load(command_path, allow_pickle=False) as saved:
                commands, mapping = saved["ctrl"], saved["reference_index"]
            assert commands.shape == (int(record["frames"]), 28) and np.all(np.isfinite(commands))
            assert mapping.shape == (len(commands),) and np.issubdtype(mapping.dtype, np.integer)
            assert np.all((0 <= mapping) & (mapping < len(receiver.base.q_ref))) and np.all(np.diff(mapping) >= 0)
            prefix_end = int(patch["template_transfer"]["command_transfer"]["preserved_receiver_command_prefix"][1])
            assert np.array_equal(commands[: prefix_end + 1], receiver.base.q_ref[mapping[: prefix_end + 1]])
    expected_files = {"receiver_inventory.json", "RUNNERS.md", "README.md"}
    expected_files.update(f"guangxue_bottle05_row{row}_receiver_template_v1.json" for row in BOTTLE_RECEIVERS)
    expected_files.update(f"guangxue_pitcher06_row{row}_receiver_template_v1.json" for row in PITCHER_RECEIVERS)
    expected_files.update(f"guangxue_pitcher06_row{row}_receiver_commands.npz" for row in PITCHER_RECEIVERS)
    actual_files = {path.name for path in output.iterdir() if path.is_file()}
    assert actual_files == expected_files, (actual_files - expected_files, expected_files - actual_files)
    assert (donor_root / "patches" / "guangxue_bottle05_row19_v2.json").is_file()
    print(json.dumps({"validated_receivers": len(inventory), "output": str(output), "files": len(actual_files)}, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.check:
        validate(args.source, args.donor, args.output)
    else:
        generate(args.source, args.donor, args.output)
        validate(args.source, args.donor, args.output)


if __name__ == "__main__":
    main()
