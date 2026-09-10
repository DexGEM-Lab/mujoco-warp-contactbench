#!/usr/bin/env python3
"""CPU-only v2 contact-primitive transfer for bottle/pitcher receiver scenes.

Each candidate reuses the *recorded actual donor command curve*, transformed by
the gravity-preserving yaw/XYZ scene symmetry.  It does not add actuator
residuals to receiver commands.  The only receiver conditioning is a smooth
translation warp near the donor's measured bowl approach and final placement.
No simulation backend is imported or initialized.
"""
from __future__ import annotations

import argparse
import hashlib
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
OUTPUT_DEFAULT = Path(__file__).resolve().parents[1] / "patches" / "full_containers_v2"
RECEIVERS = tuple(range(20, 31)) + (32, 33, 34, 35, 37, 38, 39)
DONORS = {
    "mayonnaisebottle": {"row": 19, "recording": "bottle-pour", "recipe": "guangxue_bottle05_row19_v2.json"},
    "pitcherbase": {"row": 31, "recording": "pitcher-pour", "recipe": "guangxue_pitcher06_row31_v1.json"},
}


@dataclass(frozen=True)
class Decoded:
    row: int
    uuid: str
    frames: int
    active: str
    base: Any


@dataclass(frozen=True)
class Trace:
    path: Path
    sha256: str
    qpos: np.ndarray
    ctrl: np.ndarray
    object_names: tuple[str, ...]
    object_pos: np.ndarray
    object_rot: np.ndarray
    source_frame: np.ndarray
    reference_index: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--donor", type=Path, default=DONOR_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    return digest_bytes(values.dtype.str.encode() + str(values.shape).encode() + values.tobytes())


def dump(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def decode(dataset: Any, row_index: int) -> Decoded:
    row = dataset.take([row_index], columns=["index", "trajectory_metadata", "timestamp", "hands", "objects"]).to_pylist()[0]
    base = resample_reference_trajectory(
        trajectory_from_lance_row(row, dataset.version, row_index=row_index, hand_side="right", pre_padding=180, post_padding=180),
        reference_fps=100, control_fps=100,
    )
    return Decoded(
        row_index, str(row["index"]["uuid"]), int(row["trajectory_metadata"]["total_frames"]),
        base.scene_object_types[base.identity.object_index], base,
    )


def load_trace(path: Path) -> Trace:
    with np.load(path, allow_pickle=False) as saved:
        qpos = np.asarray(saved["qpos"], dtype=np.float64)
        ctrl = np.asarray(saved["ctrl"], dtype=np.float64)
        names = tuple(str(x) for x in saved["scene_object_names"])
        positions = np.asarray(saved["scene_object_pos"], dtype=np.float64)
        # Keep rotation vectors in their (frame, object, XYZ) layout; scipy
        # Rotation is reconstructed only for the requested object/frame.
        rotvec = np.asarray(saved["scene_object_rot_aa"], dtype=np.float64)
        source_frame = np.asarray(saved["source_frame"], dtype=np.int64)
        reference_index = np.asarray(saved["reference_indices"], dtype=np.int64)
    if qpos.ndim != 2 or qpos.shape[1] < 28 or ctrl.shape != (len(qpos), 28):
        raise ValueError(f"bad donor trace command ABI: {path}")
    if positions.shape != (len(qpos), len(names), 3) or rotvec.shape != positions.shape:
        raise ValueError(f"bad donor trace object ABI: {path}")
    # Store rotvec as an array; rotations are constructed at requested frames.
    return Trace(path, sha256(path), qpos, ctrl, names, positions, rotvec, source_frame, reference_index)


def yaw_from_quat(quat_xyzw: np.ndarray) -> float:
    return float(Rotation.from_quat(quat_xyzw).as_euler("ZYX")[0])


def object_rotation(trace: Trace, frame: int, name: str) -> Rotation:
    return Rotation.from_rotvec(trace.object_rot[frame, trace.object_names.index(name)])


def cosine_weight(value: np.ndarray, start: float, end: float) -> np.ndarray:
    if end <= start:
        raise ValueError("non-increasing blend interval")
    x = np.clip((value - start) / (end - start), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def donor_pour_phases(trace: Trace, active: str, donor_end: int) -> dict[str, int]:
    """Derive approach/return from actual object positions, not recipe labels."""
    active_index = trace.object_names.index(active)
    bowl_index = trace.object_names.index("bowl")
    distance = np.linalg.norm(trace.object_pos[: donor_end + 1, active_index, :2] - trace.object_pos[: donor_end + 1, bowl_index, :2], axis=1)
    start = 180
    closest = int(start + np.argmin(distance[start:]))
    threshold = float(distance[closest] + 0.75 * (distance[start] - distance[closest]))
    approaching = np.flatnonzero(distance[start : closest + 1] <= threshold)
    returning = np.flatnonzero(distance[closest : donor_end + 1] >= threshold)
    if len(approaching) == 0 or len(returning) == 0:
        raise ValueError("actual donor trace lacks a returnable bowl approach")
    return {"approach_start": int(start + approaching[0]), "pour_peak": closest, "return_end": int(closest + returning[0])}


def interpolate(track: np.ndarray, times: np.ndarray) -> np.ndarray:
    source = np.arange(len(track), dtype=np.float64)
    return np.stack([np.interp(times, source, track[:, axis]) for axis in range(track.shape[1])], axis=1)


def tilt_only(rotation: Rotation) -> Rotation:
    """Drop residual yaw: tilt correction preserves the gravity symmetry choice."""
    vector = rotation.as_rotvec()
    return Rotation.from_rotvec([vector[0], vector[1], 0.0])


def source_record(decoded: Decoded) -> dict[str, Any]:
    return {"dataset_name": SOURCE_NAME, "version": 5, "row": decoded.row, "uuid": decoded.uuid, "frames": decoded.frames}


def build_track(donor: Decoded, receiver: Decoded, trace: Trace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    active = donor.active
    donor_end = donor.base.movement_end_step
    receiver_end = receiver.base.movement_end_step
    if donor_end >= len(trace.ctrl):
        raise ValueError("recorded donor trace ends before donor movement endpoint")
    active_index = trace.object_names.index(active)
    bowl_index = trace.object_names.index("bowl")
    donor_active_initial_pos = trace.object_pos[0, active_index]
    donor_passive_initial_pos = trace.object_pos[0, bowl_index]
    receiver_active_index = receiver.base.scene_object_types.index(active)
    receiver_bowl_index = receiver.base.scene_object_types.index("bowl")
    receiver_active_initial_pos = np.asarray(receiver.base.scene_object_initial_pos[receiver_active_index], dtype=np.float64)
    receiver_passive_initial_pos = np.asarray(receiver.base.scene_object_initial_pos[receiver_bowl_index], dtype=np.float64)
    donor_active_initial_rot = object_rotation(trace, 0, active)
    receiver_active_initial_rot = Rotation.from_quat(receiver.base.scene_object_initial_quat_xyzw[receiver_active_index])
    donor_yaw = yaw_from_quat(donor_active_initial_rot.as_quat())
    receiver_yaw = yaw_from_quat(receiver_active_initial_rot.as_quat())
    yaw_transform = Rotation.from_euler("z", receiver_yaw - donor_yaw)
    translation = receiver_active_initial_pos - yaw_transform.apply(donor_active_initial_pos)
    transformed_donor_active_rot = yaw_transform * donor_active_initial_rot
    tilt_correction = tilt_only(receiver_active_initial_rot * transformed_donor_active_rot.inv())

    # Keep the verified donor tail duration: it contains the release primitive.
    frame_count = receiver_end + (len(trace.ctrl) - donor_end)
    receiver_time = np.arange(frame_count, dtype=np.float64)
    donor_time = np.where(
        receiver_time <= receiver_end,
        receiver_time * donor_end / receiver_end,
        donor_end + receiver_time - receiver_end,
    )
    donor_time = np.clip(donor_time, 0.0, len(trace.ctrl) - 1.0)
    commands = interpolate(trace.ctrl, donor_time)
    wrist_pos = yaw_transform.apply(commands[:, :3]) + translation
    wrist_rot = Rotation.from_euler("XYZ", commands[:, 3:6])
    # The correction is only roll/pitch, retains the scene's physical floor pose,
    # and closes the small donor/receiver resting-tilt mismatch at the hand.
    wrist_rot = tilt_correction * yaw_transform * wrist_rot
    commands[:, :3] = wrist_pos
    commands[:, 3:6] = wrist_rot.as_euler("XYZ")

    phases = donor_pour_phases(trace, active, donor_end)
    mapped_phases = {key: int(round(value * receiver_end / donor_end)) for key, value in phases.items()}
    pour_weight = np.zeros(frame_count, dtype=np.float64)
    peak = mapped_phases["pour_peak"]
    pour_weight[: peak + 1] = cosine_weight(receiver_time[: peak + 1], mapped_phases["approach_start"], peak)
    pour_weight[peak:] = 1.0 - cosine_weight(receiver_time[peak:], peak, mapped_phases["return_end"])
    transformed_passive = yaw_transform.apply(donor_passive_initial_pos) + translation
    bowl_warp = receiver_passive_initial_pos - transformed_passive
    placement_start = mapped_phases["return_end"]
    placement_weight = cosine_weight(receiver_time, placement_start, receiver_end)
    donor_end_actual = trace.object_pos[donor_end, active_index]
    receiver_final_captured = np.asarray(receiver.base.object_pos[receiver_end], dtype=np.float64)
    placement_warp = receiver_final_captured - (yaw_transform.apply(donor_end_actual) + translation)
    commands[:, :3] += pour_weight[:, None] * bowl_warp + placement_weight[:, None] * placement_warp

    reference_index = np.minimum(np.arange(frame_count, dtype=np.int64), receiver_end)
    trace_index = np.rint(donor_time).astype(np.int64)
    initial_hand = np.asarray(trace.qpos[0, :28], dtype=np.float64).copy()
    initial_hand[:3] = yaw_transform.apply(initial_hand[:3]) + translation
    initial_hand[3:6] = (tilt_correction * yaw_transform * Rotation.from_euler("XYZ", initial_hand[3:6])).as_euler("XYZ")
    metadata = {
        "method": "actual_donor_full_command_track_yaw_xyz_transfer_with_phase_retiming",
        "donor_trace": {"path": str(trace.path), "sha256": trace.sha256, "frames": len(trace.ctrl), "actual_command": "trajectory.npz:ctrl", "actual_hand_state": "trajectory.npz:qpos[0,:28]"},
        "donor_movement_end_step": donor_end,
        "receiver_movement_end_step": receiver_end,
        "frame_map": {"receiver_command_to_donor_trace_time": "donor_t=receiver_t*donor_end/receiver_end through movement end; donor tail then runs one-for-one", "donor_trace_index_sha256": array_sha256(trace_index), "receiver_reference_index_sha256": array_sha256(reference_index)},
        "global_yaw_xyz": {"yaw_rad": receiver_yaw - donor_yaw, "translation_m": translation.tolist(), "donor_active_actual_initial_pos_m": donor_active_initial_pos.tolist(), "receiver_active_physical_initial_pos_m": receiver_active_initial_pos.tolist()},
        "initial_active_tilt_correction": {"policy": "receiver active object qpos is retained; roll/pitch mismatch is corrected in wrist orientation only to preserve floor support", "rotation_xyzw": tilt_correction.as_quat().tolist(), "donor_actual_xyzw": donor_active_initial_rot.as_quat().tolist(), "receiver_physical_xyzw": receiver_active_initial_rot.as_quat().tolist()},
        "pour_bowl_warp": {"donor_actual_phases": phases, "receiver_retimed_phases": mapped_phases, "receiver_passive_bowl_initial_pos_m": receiver_passive_initial_pos.tolist(), "transformed_donor_passive_initial_pos_m": transformed_passive.tolist(), "translation_m": bowl_warp.tolist(), "blend": "cosine approach-to-peak and cosine inverse-return"},
        "final_placement_warp": {"start_step": placement_start, "end_step": receiver_end, "receiver_final_captured_active_pos_m": receiver_final_captured.tolist(), "transformed_donor_actual_end_pos_m": (yaw_transform.apply(donor_end_actual) + translation).tolist(), "translation_m": placement_warp.tolist(), "blend": "cosine"},
        "initial_state_hashes": {
            "donor_actual_initial_qpos_sha256": array_sha256(trace.qpos[0]),
            "donor_actual_initial_hand_sha256": array_sha256(trace.qpos[0, :28]),
            "receiver_replaced_initial_hand_sha256": array_sha256(receiver.base.q_ref[0]),
            "transformed_initial_hand_sha256": array_sha256(initial_hand),
        },
        "command_sha256_before_npz": array_sha256(commands),
        "finger_policy": "donor finger targets are copied as the retimed recorded scalar sequence; no receiver residual, geometric transform, or joint-space addition is applied",
    }
    return commands, reference_index, trace_index, metadata, initial_hand


def candidate(donor: Decoded, receiver: Decoded, command_name: str, command_hash: str, frames: int, initial_hand: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    recipe = DONORS[receiver.active]["recipe"]
    return {
        "schema": "direct_capture_repair.v1",
        "name": f"guangxue_{'bottle05' if receiver.active == 'mayonnaisebottle' else 'pitcher06'}_row{receiver.row}_contact_primitive_transfer_v2",
        "active_object": receiver.active,
        "source": source_record(receiver),
        "solver": {"cone": "elliptic", "impratio": 100.0},
        "initial_hand": initial_hand.tolist(),
        # Kept for the replay schema; frozen command targets are authoritative.
        "edit": {"name": "v2_full_contact_primitive_command_track", "translation": [0, 0, 0], "release_at_end": True},
        "command_track": {"path": command_name, "sha256": command_hash, "frames": frames},
        "contact_primitive_transfer": {"donor_source": source_record(donor), "donor_recipe_path": f"patches/{recipe}", **metadata},
        "scene_initialization": {"active_and_passive_object_poses": "receiver source physical poses are retained by replay; no object position/orientation override, force, mass, or passive-object edit", "replaced_receiver_approach": "entire receiver hand approach and subsequent noncontact motion is replaced by the transformed validated donor command primitive"},
        "scope": "CPU-prepared v2 contact-primitive transfer. It records a candidate command asset only; physical grasp, lift, pour, release, and placement require parent GPU replay.",
    }


def generate(source: Path, donor_root: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset = lance.dataset(source, version=5)
    donor_decoded = {kind: decode(dataset, details["row"]) for kind, details in DONORS.items()}
    traces = {kind: load_trace(donor_root / "recordings" / details["recording"] / "trajectory.npz") for kind, details in DONORS.items()}
    inventory: list[dict[str, Any]] = []
    for row in RECEIVERS:
        receiver = decode(dataset, row)
        donor = donor_decoded[receiver.active]
        trace = traces[receiver.active]
        commands, reference_index, trace_index, metadata, initial_hand = build_track(donor, receiver, trace)
        stem = "bottle05" if receiver.active == "mayonnaisebottle" else "pitcher06"
        command_name = f"guangxue_{stem}_row{row}_contact_primitive_commands_v2.npz"
        command_path = output / command_name
        np.savez_compressed(command_path, ctrl=commands, reference_index=reference_index, donor_trace_index=trace_index)
        patch_name = f"guangxue_{stem}_row{row}_contact_primitive_transfer_v2.json"
        dump(output / patch_name, candidate(donor, receiver, command_name, sha256(command_path), len(commands), initial_hand, metadata))
        inventory.append({"row": row, "uuid": receiver.uuid, "active_object": receiver.active, "patch": patch_name, "command_track": command_name})
    dump(output / "receiver_inventory.json", {"schema": "container_contact_primitive_inventory.v2", "source": str(source), "donor_bundle": str(donor_root), "receivers": inventory})
    runners = ["# Parent GPU replay list — candidates only; no physical success implied.", "# REPLAY=/mnt/nas-222-project/mocap_v2/lance_datasets/dexgem_vla_demo_guangxue_astra_repair_5samples_20260910/tools/replay_repaired_capture.py"]
    for item in inventory:
        runners.append(f"$REPLAY --dataset {source} --patch {output / item['patch']} --output <parent-output>/row{item['row']}")
    (output / "RUNNERS.md").write_text("\n".join(runners) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# v2 container contact-primitive transfer\n\n"
        "Every receiver replaces its full hand motion with a retimed validated donor `trajectory.npz:ctrl` curve. A yaw-plus-XYZ transform maps the donor’s *actual* initial active-object state to the receiver physical active-object start; finger values receive no residual addition. Receiver object and passive-bowl reset poses remain unchanged.\n\n"
        "The hand translation receives two smooth, measured-scene conditions: donor-to-receiver bowl displacement over the actual donor approach/return, then captured receiver-final-active displacement over placement. The small donor/receiver resting roll-pitch mismatch is recorded and applied to wrist orientation rather than tilting an object off its floor support.\n\n"
        "These assets replace receiver noncontact approach as well as contact motion. They are CPU artifacts, not replay outcomes.\n",
        encoding="utf-8",
    )


def validate(source: Path, donor_root: Path, output: Path) -> None:
    dataset = lance.dataset(source, version=5)
    traces = {kind: load_trace(donor_root / "recordings" / details["recording"] / "trajectory.npz") for kind, details in DONORS.items()}
    inventory = json.loads((output / "receiver_inventory.json").read_text(encoding="utf-8"))["receivers"]
    assert len(inventory) == len(RECEIVERS) and {x["row"] for x in inventory} == set(RECEIVERS)
    expected = {"receiver_inventory.json", "RUNNERS.md", "README.md"}
    for item in inventory:
        receiver = decode(dataset, int(item["row"]))
        patch_path = output / item["patch"]
        patch = json.loads(patch_path.read_text(encoding="utf-8"))
        assert patch["schema"] == "direct_capture_repair.v1" and patch["source"] == source_record(receiver)
        assert patch["active_object"] == receiver.active and patch["solver"] == {"cone": "elliptic", "impratio": 100.0}
        initial = np.asarray(patch["initial_hand"], dtype=np.float64)
        assert initial.shape == (28,) and np.all(np.isfinite(initial))
        hashes = patch["contact_primitive_transfer"]["initial_state_hashes"]
        assert array_sha256(initial) == hashes["transformed_initial_hand_sha256"]
        assert len(hashes["donor_actual_initial_qpos_sha256"]) == 64
        assert len(hashes["donor_actual_initial_hand_sha256"]) == 64
        assert len(hashes["receiver_replaced_initial_hand_sha256"]) == 64
        record = patch["command_track"]
        command_path = output / record["path"]
        assert command_path.parent == output and sha256(command_path) == record["sha256"]
        with np.load(command_path, allow_pickle=False) as saved:
            ctrl = saved["ctrl"]; mapping = saved["reference_index"]; donor_indices = saved["donor_trace_index"]
        assert ctrl.shape == (int(record["frames"]), 28) and np.all(np.isfinite(ctrl))
        assert mapping.shape == (len(ctrl),) and np.all((0 <= mapping) & (mapping < len(receiver.base.q_ref))) and np.all(np.diff(mapping) >= 0)
        trace = traces[receiver.active]
        assert donor_indices.shape == mapping.shape and np.all((0 <= donor_indices) & (donor_indices < len(trace.ctrl))) and np.all(np.diff(donor_indices) >= 0)
        meta = patch["contact_primitive_transfer"]
        assert meta["donor_trace"]["sha256"] == trace.sha256
        assert array_sha256(ctrl) == meta["command_sha256_before_npz"]
        assert array_sha256(mapping) == meta["frame_map"]["receiver_reference_index_sha256"]
        assert array_sha256(donor_indices) == meta["frame_map"]["donor_trace_index_sha256"]
        # The donor finger curve is never residual-added or geometry-transformed.
        donor_end = int(meta["donor_movement_end_step"])
        receiver_end = int(meta["receiver_movement_end_step"])
        receiver_time = np.arange(len(ctrl), dtype=np.float64)
        donor_time = np.where(receiver_time <= receiver_end, receiver_time * donor_end / receiver_end, donor_end + receiver_time - receiver_end)
        expected_fingers = interpolate(trace.ctrl, donor_time)[:, 6:]
        assert np.array_equal(ctrl[:, 6:], expected_fingers)
        assert hashes["donor_actual_initial_qpos_sha256"] == array_sha256(trace.qpos[0])
        assert hashes["donor_actual_initial_hand_sha256"] == array_sha256(trace.qpos[0, :28])
        assert hashes["receiver_replaced_initial_hand_sha256"] == array_sha256(receiver.base.q_ref[0])
        expected.add(item["patch"]); expected.add(record["path"])
    actual = {p.name for p in output.iterdir() if p.is_file()}
    assert actual == expected, (actual - expected, expected - actual)
    print(json.dumps({"validated_receivers": len(inventory), "files": len(actual), "output": str(output)}, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.check:
        validate(args.source, args.donor, args.output)
    else:
        generate(args.source, args.donor, args.output)
        validate(args.source, args.donor, args.output)


if __name__ == "__main__":
    main()
