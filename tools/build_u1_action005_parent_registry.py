#!/usr/bin/env python3
"""Qualify all ten compact action005 rows and freeze U1 parent bundles."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.manorl.assets import compile_unified_model
from sim.manorl.u1_action005 import (
    PARENT_REGISTRY_SCHEMA,
    PARENT_SEMANTICS,
    SETTING,
    SETTLED_ANGULAR_P95_RADPS,
    SETTLED_LINEAR_P95_MPS,
    SETTLED_ORIENTATION_EXCURSION_RAD,
    SETTLED_POSITION_EXCURSION_M,
    action005_trace_gates,
    canonical_parent_target,
    require_setting,
    source_parent_input,
)
from sim.manorl.u1_campaign import array_sha, digest, file_sha, write_json
from sim.manorl.target_replay import load_target_replay_source
from tools.replay_manorl_target_dof import _activate_asset_profile
from tools.run_u1_largepose_campaign import LargePoseRuntime
from tools.u1_placement_workspace import assert_u1


class Action005Runtime(LargePoseRuntime):
    """Reuse the proven strict replay loop and add full-pour physical gates."""

    def __init__(self, registry, parent, inp, model, base, teacher):
        self.r = registry
        self.p = parent
        self.I = inp
        self.m = model
        self.base = base
        self.teacher = teacher
        self.action = "005"

    def _replay(self, target, q0, prefix_frames, folder):
        report = super()._replay(target, q0, prefix_frames, folder)
        gates, metrics = action005_trace_gates(
            self.m, self.I, Path(folder) / "trace.npz"
        )
        report["gates"].update(gates)
        report["physical"]["action005"] = metrics
        report["accepted"] = all(report["gates"].values())
        write_json(Path(folder) / "result.json", report)
        return report


def _source_row(dataset: Path, version: int, row_index: int) -> dict:
    import lance

    columns = ["index", "trajectory_metadata", "provenance"]
    values = lance.dataset(str(dataset), version=version).take(
        [row_index], columns=columns
    ).to_pylist()
    if len(values) != 1:
        raise ValueError("source Lance row lookup failed")
    return values[0]


def _candidate(args, row_index: int):
    _activate_asset_profile(args.asset_root, args.asset_manifest)
    source = load_target_replay_source(
        args.dataset, dataset_version=args.dataset_version, row_index=row_index
    )
    _mujoco, model = compile_unified_model(
        object_types=tuple(sorted(source.scene_object_types)),
        object_collisions=True,
        hand_side="right",
        physics_timestep=1 / 480,
    )
    model.opt.ccd_iterations = 16
    inp, source_teacher = source_parent_input(source, model)
    contract = assert_u1(model, inp)
    fingerprint = require_setting(
        model, tuple(source.scene_object_types), args.asset_manifest
    )
    target = canonical_parent_target(source, model)
    row = _source_row(args.dataset, args.dataset_version, row_index)
    metadata = row["trajectory_metadata"]
    provenance = row["provenance"]
    parent = {
        "row_id": source.generated_uuid,
        "parent_row": row_index,
        "parent_uuid": source.generated_uuid,
        "frames": source.frames,
        "names": list(source.scene_object_types),
        "semantics": PARENT_SEMANTICS,
        "contract": contract,
        "setting_sha256": SETTING,
        "setting": fingerprint,
        "metrics": inp.metrics,
        "movement": {
            "object_name": source.object_type,
            "start_frame": source.movement_start,
            "end_frame": source.movement_end,
        },
        "mano_shape": metadata["mano_hand_shapes"][0],
        "source": {
            "compact_dataset": str(args.dataset.resolve()),
            "compact_dataset_version": args.dataset_version,
            "compact_row": row_index,
            "compact_uuid": source.generated_uuid,
            "seed_uuid": row["index"]["seed_uuid"],
            "source_dataset": source.source_dataset_path,
            "source_dataset_version": source.source_dataset_version,
            "source_row_index": source.source_row_index,
            "source_identity": source.source_identity,
            "checkpoint_update": source.checkpoint_update,
            "checkpoint_sha256": source.checkpoint_sha256,
            "seed": provenance["seed"],
            "episode_index": provenance["episode_index"],
            "generation_attempt": provenance["generation_attempt"],
            "row_contract": source.row_contract,
            "source_contract": source.source_contract,
        },
    }
    registry = {"setting_sha256": SETTING, "parents": {str(row_index): parent}}
    runtime = Action005Runtime(
        registry, parent, inp, model, target, source_teacher
    )
    return source, row, runtime


def _artifact_hashes(folder: Path) -> dict[str, str]:
    return {
        str(path.relative_to(folder)): file_sha(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _run_replay(args, row_index: int, output: Path, frozen: Path | None):
    source, _row, runtime = _candidate(args, row_index)
    target = (
        np.load(frozen, allow_pickle=False)
        if frozen is not None
        else runtime.base.copy()
    )
    if target.shape != (source.frames, 28):
        raise ValueError("frozen parent target shape changed")
    q0 = runtime.I.initial["qpos"].copy()
    return runtime._replay(target, q0, 0, output)


def _reassess_report(runtime, folder: Path) -> dict:
    report_path = folder / "result.json"
    report = json.loads(report_path.read_text())
    gates, metrics = action005_trace_gates(
        runtime.m, runtime.I, folder / "trace.npz"
    )
    report["gates"].update(gates)
    report["physical"]["action005"] = metrics
    report["accepted"] = all(report["gates"].values())
    write_json(report_path, report)
    return report


def _bundle_from_qualification(
    args, row_index: int, qualification: Path, bundle: Path
) -> dict:
    import mujoco as mj

    source, _row, runtime = _candidate(args, row_index)
    first_report = _reassess_report(runtime, qualification / "first")
    second_report = _reassess_report(runtime, qualification / "second")
    if not first_report["accepted"] or not second_report["accepted"]:
        raise ValueError(f"parent row{row_index} failed physical qualification")
    if first_report["pid"] == second_report["pid"]:
        raise ValueError("parent replays did not use distinct processes")
    first_target = np.load(qualification / "first/target.npy", allow_pickle=False)
    second_target = np.load(qualification / "second/target.npy", allow_pickle=False)
    if not np.array_equal(first_target, second_target):
        raise ValueError("parent frozen targets differ")
    with np.load(qualification / "first/trace.npz", allow_pickle=False) as trace:
        first_qpos = trace["qpos"].copy()
        first_qvel = trace["qvel"].copy()
        first_ctrl = trace["ctrl"].copy()
    with np.load(qualification / "second/trace.npz", allow_pickle=False) as trace:
        second_qpos = trace["qpos"].copy()
        second_qvel = trace["qvel"].copy()
    if not np.array_equal(first_ctrl, first_target):
        raise ValueError("parent trace controls differ from frozen target")
    reproducibility = {
        "max_qpos_abs_difference": float(np.max(np.abs(first_qpos - second_qpos))),
        "max_qvel_abs_difference": float(np.max(np.abs(first_qvel - second_qvel))),
        "interpretation": (
            "contact trajectories may diverge across processes; acceptance requires "
            "identical frozen controls and both physical gate sets, not state identity"
        ),
    }
    names = runtime.p["names"]
    positions = []
    quaternions = []
    for name in names:
        address = int(runtime.m.joint(name + "_free").qposadr[0])
        positions.append(first_qpos[:, address : address + 3])
        quaternions.append(first_qpos[:, address + 3 : address + 7][:, [1, 2, 3, 0]])
    parent_arrays = {
        "scene_object_names": np.asarray(names),
        "source_object_pos": np.stack(positions, axis=1),
        "source_object_quat_xyzw": np.stack(quaternions, axis=1),
    }
    bundle.mkdir(parents=True, exist_ok=False)
    np.save(bundle / "target.npy", first_target)
    mj.mj_saveModel(runtime.m, str(bundle / "model.mjb"), None)
    np.savez_compressed(
        bundle / "input.npz",
        **parent_arrays,
        qpos0=first_qpos[0],
        qvel0=first_qvel[0],
        teacher=first_qpos,
        **{
            "native_" + field: np.asarray(runtime.I.manifest["native"][field])
            for field in runtime.I.manifest["native"]
        },
    )
    evidence = {
        "schema": "u1-action005-parent-qualification-v1",
        "parent_row": row_index,
        "parent_uuid": source.generated_uuid,
        "first": first_report,
        "second": second_report,
        "reproducibility": reproducibility,
        "terminal_settlement_contract": {
            "window_frames": 24,
            "linear_speed_p95_mps_lt": SETTLED_LINEAR_P95_MPS,
            "angular_speed_p95_radps_lt": SETTLED_ANGULAR_P95_RADPS,
            "position_excursion_m_lt": SETTLED_POSITION_EXCURSION_M,
            "orientation_excursion_rad_lt": SETTLED_ORIENTATION_EXCURSION_RAD,
            "mechanism": (
                "p95 rejects sustained motion while position/orientation excursion "
                "directly rejects drift; single contact-solver qvel spikes are retained "
                "as max metrics but do not mislabel a stationary supported object"
            ),
        },
        "qualification_artifacts": _artifact_hashes(qualification),
        "requested_source_target_sha256": array_sha(source.target_qpos),
        "canonical_parent_target_sha256": array_sha(runtime.base),
        "frozen_parent_target_sha256": array_sha(first_target),
        "frame0_only_canonicalization": bool(
            np.array_equal(runtime.base[1:], source.target_qpos[1:])
        ),
        "frame0_max_change": float(
            np.max(np.abs(runtime.base[0] - source.target_qpos[0]))
        ),
    }
    write_json(bundle / "evidence.json", evidence)
    parent = dict(runtime.p)
    parent.update(
        folder=str(bundle.relative_to(args.staging)),
        final_registry_status="accepted_for_named_semantics",
        initial_qpos_sha256=array_sha(first_qpos[0]),
        initial_qvel_sha256=array_sha(first_qvel[0]),
        target_array_sha256=array_sha(first_target),
        teacher_qpos_sha256=array_sha(first_qpos),
        qualification_folder=str(qualification.relative_to(args.staging)),
        files={
            name: file_sha(bundle / name)
            for name in ("model.mjb", "input.npz", "target.npy", "evidence.json")
        },
    )
    write_json(bundle / "parent.json", parent)
    return parent


def qualify_parent(args, row_index: int) -> dict:
    qualification = args.staging / "qualification" / f"row{row_index:02d}"
    bundle = args.staging / "parents" / f"row{row_index:02d}"
    if bundle.exists():
        parent = json.loads((bundle / "parent.json").read_text())
        for name, expected in parent["files"].items():
            if file_sha(bundle / name) != expected:
                raise ValueError(
                    f"parent row{row_index} bundle file changed: {name}"
                )
        return parent
    if qualification.exists():
        first_result = qualification / "first/result.json"
        second_result = qualification / "second/result.json"
        if first_result.is_file() and second_result.is_file():
            return _bundle_from_qualification(
                args, row_index, qualification, bundle
            )
        raise ValueError(
            f"parent row{row_index} has an incomplete prior qualification; "
            "inspect it before rerun"
        )
    first = qualification / "first"
    report = _run_replay(args, row_index, first, None)
    if not report["accepted"]:
        raise ValueError(f"parent row{row_index} first replay failed")
    command = [
        sys.executable,
        "-m",
        "tools.build_u1_action005_parent_registry",
        "qualify-second",
        "--dataset",
        str(args.dataset),
        "--dataset-version",
        str(args.dataset_version),
        "--asset-root",
        str(args.asset_root),
        "--asset-manifest",
        str(args.asset_manifest),
        "--staging",
        str(args.staging),
        "--row",
        str(row_index),
        "--output",
        str(qualification / "second"),
        "--frozen",
        str(first / "target.npy"),
    ]
    with (qualification / "second-process.log").open("w") as log:
        subprocess.run(command, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return _bundle_from_qualification(args, row_index, qualification, bundle)


def write_registry(args, parents: dict[str, dict]) -> Path:
    source_manifest = args.dataset.with_name(args.dataset.name + ".manifest.json")
    source_validation = args.dataset.with_name(args.dataset.name + ".validation.json")
    if not source_manifest.is_file() or not source_validation.is_file():
        raise ValueError("standardized source manifest/validation sidecars are required")
    payload = {
        "schema": PARENT_REGISTRY_SCHEMA,
        "setting_sha256": SETTING,
        "source": {
            "dataset": str(args.dataset.resolve()),
            "dataset_version": args.dataset_version,
            "manifest": str(source_manifest.resolve()),
            "manifest_sha256": file_sha(source_manifest),
            "validation": str(source_validation.resolve()),
            "validation_sha256": file_sha(source_validation),
        },
        "assets": {
            "root": str(args.asset_root.resolve()),
            "manifest": str(args.asset_manifest.resolve()),
            "manifest_sha256": file_sha(args.asset_manifest),
        },
        "parents": parents,
        "parent_order": [parents[str(row)]["parent_uuid"] for row in range(10)],
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("mujoco", "warp-lang", "numpy")
        },
        "builder_sources": {
            path: file_sha(ROOT / path)
            for path in (
                "sim/manorl/u1_action005.py",
                "tools/build_u1_action005_parent_registry.py",
                "tools/run_u1_largepose_campaign.py",
            )
        },
    }
    registry = args.staging / "registry.json"
    write_json(registry, dict(payload, digest=digest(payload)))
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("qualify-one", "qualify-all", "qualify-second")
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-version", type=int, default=1)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--row", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.dataset = args.dataset.expanduser().resolve(strict=True)
    args.asset_root = args.asset_root.expanduser().resolve(strict=True)
    args.asset_manifest = args.asset_manifest.expanduser().resolve(strict=True)
    args.staging = args.staging.expanduser().resolve()
    if args.mode == "qualify-second":
        if args.row not in range(10) or args.output is None or args.frozen is None:
            raise ValueError("qualify-second requires row/output/frozen")
        report = _run_replay(args, args.row, args.output, args.frozen)
        if not report["accepted"]:
            raise ValueError("independent parent replay failed")
        print(json.dumps({"row": args.row, "accepted": True, "pid": report["pid"]}))
        return
    if not args.execute:
        raise ValueError("parent qualification requires --execute")
    args.staging.mkdir(parents=True, exist_ok=True)
    rows = [args.row] if args.mode == "qualify-one" else list(range(10))
    if any(row not in range(10) for row in rows):
        raise ValueError("row must be0..9")
    parents = {
        str(row): qualify_parent(args, row)
        for row in rows
    }
    if args.mode == "qualify-all":
        registry = write_registry(args, parents)
        print(json.dumps({"parents": 10, "registry": str(registry)}), flush=True)
    else:
        print(
            json.dumps(
                {
                    "row": args.row,
                    "parent_uuid": parents[str(args.row)]["parent_uuid"],
                    "accepted": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
