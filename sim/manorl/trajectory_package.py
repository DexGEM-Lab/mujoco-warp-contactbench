"""Content-addressed, Lance-free ManoRL reference trajectory packages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from sim.manorl.abi import ENVIRONMENT_CONTRACT_ID
from sim.manorl.contracts import TrajectoryIdentity
from sim.manorl.trajectory import (
    ObjectActionPair,
    REFERENCE_RESAMPLING_ID,
    ReferenceTrajectory,
    TrajectoryBatch,
    TrajectorySelection,
    _immutable,
)

TRAJECTORY_PACKAGE_SCHEMA = "manorl.trajectory_package.v1"
_READY_FILE = "READY"
_MANIFEST_FILE = "manifest.json"
_ARRAY_FILES = {
    "offsets": "offsets.npy",
    "source_indices": "source_indices.npy",
    "timestamps": "timestamps.npy",
    "q_ref_by_side": "q_ref_by_side.npy",
    "object_pos_raw": "object_pos_raw.npy",
    "object_pos": "object_pos.npy",
    "object_quat_xyzw": "object_quat_xyzw.npy",
}


class TrajectoryPackageError(ValueError):
    """Raised when an MTP artifact violates its content or semantic contract."""


@dataclass(frozen=True)
class TrajectoryCatalog:
    """A complete decoded catalog loaded without Lance or PyArrow."""

    trajectories: tuple[ReferenceTrajectory, ...]
    resolved_pairs: tuple[ObjectActionPair, ...]
    package_path: Path
    package_digest: str
    manifest_sha256: str
    catalog_digest: str
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise TrajectoryPackageError("trajectory package catalog is empty")
        if not self.resolved_pairs:
            raise TrajectoryPackageError("trajectory package has no object/action pairs")

    @property
    def checkpoint_metadata(self) -> dict[str, str]:
        return {
            "schema": TRAJECTORY_PACKAGE_SCHEMA,
            "package_digest": self.package_digest,
            "manifest_sha256": self.manifest_sha256,
            "catalog_digest": self.catalog_digest,
        }


def file_sha256(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _trajectory_sort_key(trajectory: ReferenceTrajectory) -> tuple[str, str, int, str, int]:
    fields = trajectory.identity.identity.split("_")
    if len(fields) != 3 or not fields[2].isdigit():
        raise TrajectoryPackageError(
            f"trajectory identity is not object_action_sequence: {trajectory.identity.identity!r}"
        )
    return fields[0], fields[1], int(fields[2]), trajectory.identity.identity, trajectory.identity.row_index


def _pair_for(trajectory: ReferenceTrajectory) -> ObjectActionPair:
    fields = trajectory.identity.identity.split("_")
    if len(fields) != 3:
        raise TrajectoryPackageError(
            f"trajectory identity is not object_action_sequence: {trajectory.identity.identity!r}"
        )
    return ObjectActionPair(fields[0], fields[1])


def _uniform_value(
    trajectories: Sequence[ReferenceTrajectory],
    name: str,
) -> object:
    values = {json.dumps(getattr(item, name), sort_keys=True) for item in trajectories}
    if len(values) != 1:
        raise TrajectoryPackageError(f"trajectory catalog has non-uniform {name}")
    return getattr(trajectories[0], name)


def _open_array(path: Path, *, dtype: np.dtype[Any], shape: tuple[int, ...]) -> NDArray[Any]:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_trajectory_package(
    output: str | Path,
    trajectories: Sequence[ReferenceTrajectory],
    *,
    selection: TrajectorySelection,
    dataset_schema_digest: str,
    discovery_digest: str,
    compiler: Mapping[str, object] | None = None,
    source_candidates: Sequence[Mapping[str, object]] | None = None,
    source_rejections: Sequence[Mapping[str, object]] = (),
) -> Path:
    """Atomically publish one complete, mmap-friendly MTP catalog."""

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace trajectory package: {destination}")
    if not trajectories:
        raise TrajectoryPackageError("cannot write an empty trajectory package")
    if not isinstance(selection, TrajectorySelection):
        raise TypeError("selection must be a TrajectorySelection")
    ordered = tuple(sorted(trajectories, key=_trajectory_sort_key))
    if len({(item.identity.row_index, item.identity.identity) for item in ordered}) != len(ordered):
        raise TrajectoryPackageError("trajectory package contains duplicate row identities")

    hand_sides = tuple(_uniform_value(ordered, "hand_sides"))
    selected_hand_sides = tuple(_uniform_value(ordered, "selected_hand_sides"))
    reference_fps = _uniform_value(ordered, "reference_fps")
    control_fps = _uniform_value(ordered, "control_fps")
    dataset_version = _uniform_value(ordered, "dataset_version")
    dof_dims = {item.dof_dim for item in ordered}
    if len(dof_dims) != 1:
        raise TrajectoryPackageError("trajectory package contains mixed hand DOF dimensions")
    dof_dim = next(iter(dof_dims))
    if selection.reference_fps != reference_fps or selection.resolved_control_fps != control_fps:
        raise TrajectoryPackageError("package selection clock does not match decoded trajectories")
    if selection.expected_dataset_version is not None and selection.expected_dataset_version != dataset_version:
        raise TrajectoryPackageError("package selection dataset version does not match trajectories")

    pairs = tuple(sorted({_pair_for(item) for item in ordered}))
    pair_counts = {
        pair.canonical: sum(_pair_for(item) == pair for item in ordered)
        for pair in pairs
    }
    decoded_by_key = {
        (item.identity.row_index, item.identity.identity): item for item in ordered
    }
    rejection_by_key: dict[tuple[int, str], Mapping[str, object]] = {}
    for rejection in source_rejections:
        key = (int(rejection["row_index"]), str(rejection["identity"]))
        if key in rejection_by_key:
            raise TrajectoryPackageError(f"duplicate source rejection: {key!r}")
        rejection_by_key[key] = rejection
    raw_candidates = (
        list(source_candidates)
        if source_candidates is not None
        else [
            {
                "row_index": item.identity.row_index,
                "identity": item.identity.identity,
                "pair": _pair_for(item).canonical,
                "sequence": _trajectory_sort_key(item)[2],
            }
            for item in ordered
        ]
    )
    candidate_keys: set[tuple[int, str]] = set()
    source_catalog: list[dict[str, object]] = []
    for candidate in raw_candidates:
        key = (int(candidate["row_index"]), str(candidate["identity"]))
        if key in candidate_keys:
            raise TrajectoryPackageError(f"duplicate source candidate: {key!r}")
        candidate_keys.add(key)
        decoded = decoded_by_key.get(key)
        rejection = rejection_by_key.get(key)
        if (decoded is None) == (rejection is None):
            raise TrajectoryPackageError(
                f"source candidate must be exactly decoded or rejected: {key!r}"
            )
        pair = str(candidate["pair"])
        if decoded is not None and _pair_for(decoded).canonical != pair:
            raise TrajectoryPackageError(f"source candidate pair mismatch: {key!r}")
        entry = {
            "row_index": key[0],
            "identity": key[1],
            "pair": pair,
            "sequence": int(candidate["sequence"]),
            "status": "decoded" if decoded is not None else "rejected",
        }
        if rejection is not None:
            entry["rejection"] = {
                "error_type": str(rejection["error_type"]),
                "message": str(rejection["message"]),
            }
        source_catalog.append(entry)
    if set(decoded_by_key) | set(rejection_by_key) != candidate_keys:
        raise TrajectoryPackageError("decoded/rejected rows do not exactly partition source candidates")
    offsets = np.zeros(len(ordered) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(item.q_ref) for item in ordered], dtype=np.int64)
    total_frames = int(offsets[-1])
    if total_frames < 2:
        raise TrajectoryPackageError("trajectory package contains too few frames")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        arrays: dict[str, NDArray[Any]] = {
            "offsets": _open_array(
                temporary / _ARRAY_FILES["offsets"], dtype=np.dtype(np.int64), shape=offsets.shape
            ),
            "source_indices": _open_array(
                temporary / _ARRAY_FILES["source_indices"],
                dtype=np.dtype(np.int64),
                shape=(total_frames,),
            ),
            "timestamps": _open_array(
                temporary / _ARRAY_FILES["timestamps"],
                dtype=np.dtype(np.float64),
                shape=(total_frames,),
            ),
            "q_ref_by_side": _open_array(
                temporary / _ARRAY_FILES["q_ref_by_side"],
                dtype=np.dtype(np.float64),
                # Side-major layout keeps each side/trajectory slice contiguous
                # and mmap-backed in the long-lived trainer.
                shape=(len(hand_sides), total_frames, dof_dim),
            ),
            "object_pos_raw": _open_array(
                temporary / _ARRAY_FILES["object_pos_raw"],
                dtype=np.dtype(np.float64),
                shape=(total_frames, 3),
            ),
            "object_pos": _open_array(
                temporary / _ARRAY_FILES["object_pos"],
                dtype=np.dtype(np.float64),
                shape=(total_frames, 3),
            ),
            "object_quat_xyzw": _open_array(
                temporary / _ARRAY_FILES["object_quat_xyzw"],
                dtype=np.dtype(np.float64),
                shape=(total_frames, 4),
            ),
        }
        arrays["offsets"][:] = offsets
        records: list[dict[str, object]] = []
        for index, trajectory in enumerate(ordered):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            arrays["source_indices"][start:stop] = trajectory.source_indices
            arrays["timestamps"][start:stop] = trajectory.timestamps
            for side_index, side in enumerate(hand_sides):
                arrays["q_ref_by_side"][side_index, start:stop] = trajectory.q_ref_for(side)
            arrays["object_pos_raw"][start:stop] = trajectory.object_pos_raw
            arrays["object_pos"][start:stop] = trajectory.object_pos
            arrays["object_quat_xyzw"][start:stop] = trajectory.object_quat_xyzw
            records.append(
                {
                    "identity": asdict(trajectory.identity),
                    "object_z_shift": float(trajectory.object_z_shift),
                    "movement_start_step": trajectory.movement_start_step,
                    "movement_end_step": trajectory.movement_end_step,
                    "offset": [start, stop],
                    "pair": _pair_for(trajectory).canonical,
                }
            )
        for array in arrays.values():
            if hasattr(array, "flush"):
                array.flush()
        del arrays

        array_metadata: dict[str, dict[str, object]] = {}
        for name, filename in _ARRAY_FILES.items():
            path = temporary / filename
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
            array_metadata[name] = {
                "file": filename,
                "dtype": loaded.dtype.str,
                "shape": list(loaded.shape),
                "sha256": file_sha256(path),
            }
            _fsync_file(path)

        dataset_paths = {item.identity.dataset_path for item in ordered}
        if len(dataset_paths) != 1:
            raise TrajectoryPackageError("trajectory package contains mixed dataset paths")
        source_catalog_metadata = {
            "candidate_count": len(source_catalog),
            "decoded_count": len(ordered),
            "rejected_count": len(rejection_by_key),
            "candidates": source_catalog,
        }
        catalog_basis = {
            "records": records,
            "arrays": array_metadata,
            "resolved_pairs": [pair.canonical for pair in pairs],
            "pair_counts": pair_counts,
            "source_catalog": source_catalog_metadata,
        }
        manifest: dict[str, object] = {
            "schema": TRAJECTORY_PACKAGE_SCHEMA,
            "environment_contract": ENVIRONMENT_CONTRACT_ID,
            "reference_resampling": REFERENCE_RESAMPLING_ID,
            "dataset": {
                "logical_path": next(iter(dataset_paths)),
                "version": int(dataset_version),
                "schema_digest": str(dataset_schema_digest),
                "discovery_digest": str(discovery_digest),
            },
            "selection": {
                "catalog_selector": "all",
                "hand_side": selection.hand_side,
                "pre_padding": selection.pre_padding,
                "post_padding": selection.post_padding,
                "reference_fps": selection.reference_fps,
                "control_fps": selection.resolved_control_fps,
            },
            "hand_sides": list(hand_sides),
            "selected_hand_sides": list(selected_hand_sides),
            "dof_dim": dof_dim,
            "trajectory_count": len(ordered),
            "total_frames": total_frames,
            "resolved_pairs": [pair.canonical for pair in pairs],
            "pair_counts": pair_counts,
            "source_catalog": source_catalog_metadata,
            "trajectories": records,
            "arrays": array_metadata,
            "compiler": dict(compiler or {}),
            "catalog_digest": _canonical_digest(catalog_basis),
        }
        manifest["package_digest"] = _canonical_digest(manifest)
        manifest_path = temporary / _MANIFEST_FILE
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _fsync_file(manifest_path)
        ready_path = temporary / _READY_FILE
        ready_path.write_text(str(manifest["package_digest"]) + "\n", encoding="ascii")
        _fsync_file(ready_path)
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return destination
    except BaseException:
        # Keep failed compiler state inspectable when the filesystem permits
        # renames. Never mask the causal publication failure with a second
        # best-effort evidence-rename error.
        failed = temporary.with_name(temporary.name + ".failed")
        if temporary.exists():
            try:
                os.replace(temporary, failed)
            except OSError:
                pass
        raise


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest_path = path / _MANIFEST_FILE
    ready_path = path / _READY_FILE
    if not manifest_path.is_file() or not ready_path.is_file():
        raise TrajectoryPackageError("trajectory package requires manifest.json and READY")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrajectoryPackageError("trajectory package manifest is invalid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != TRAJECTORY_PACKAGE_SCHEMA:
        raise TrajectoryPackageError("unsupported trajectory package schema")
    package_digest = manifest.get("package_digest")
    if not isinstance(package_digest, str) or len(package_digest) != 64:
        raise TrajectoryPackageError("trajectory package digest is missing")
    digest_basis = dict(manifest)
    digest_basis.pop("package_digest", None)
    if _canonical_digest(digest_basis) != package_digest:
        raise TrajectoryPackageError("trajectory package manifest digest mismatch")
    if ready_path.read_text(encoding="ascii").strip() != package_digest:
        raise TrajectoryPackageError("trajectory package READY marker mismatch")
    return manifest, file_sha256(manifest_path)


def load_trajectory_package(
    package: str | Path,
    *,
    verify_hashes: bool = True,
) -> TrajectoryCatalog:
    """Load and validate one MTP catalog using JSON and NumPy only."""

    root = Path(package).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"trajectory package directory is absent: {root}")
    manifest, manifest_sha256 = _load_manifest(root)
    arrays_metadata = manifest.get("arrays")
    if not isinstance(arrays_metadata, dict) or set(arrays_metadata) != set(_ARRAY_FILES):
        raise TrajectoryPackageError("trajectory package array manifest is incomplete")
    arrays: dict[str, NDArray[Any]] = {}
    for name, expected_filename in _ARRAY_FILES.items():
        metadata = arrays_metadata[name]
        if not isinstance(metadata, dict) or metadata.get("file") != expected_filename:
            raise TrajectoryPackageError(f"trajectory package {name} file metadata is invalid")
        array_path = root / expected_filename
        if not array_path.is_file():
            raise TrajectoryPackageError(f"trajectory package array is absent: {expected_filename}")
        if verify_hashes and file_sha256(array_path) != metadata.get("sha256"):
            raise TrajectoryPackageError(f"trajectory package array hash mismatch: {expected_filename}")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != metadata.get("shape") or array.dtype.str != metadata.get("dtype"):
            raise TrajectoryPackageError(f"trajectory package array layout mismatch: {expected_filename}")
        arrays[name] = array

    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    records = manifest.get("trajectories")
    if not isinstance(records, list) or len(offsets) != len(records) + 1:
        raise TrajectoryPackageError("trajectory package offsets do not match trajectory records")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 2):
        raise TrajectoryPackageError("trajectory package offsets are invalid")
    total_frames = int(offsets[-1])
    for name, array in arrays.items():
        if name in {"offsets", "q_ref_by_side"}:
            continue
        if len(array) != total_frames:
            raise TrajectoryPackageError(f"trajectory package {name} frame count mismatch")

    hand_sides_raw = manifest.get("hand_sides")
    selected_sides_raw = manifest.get("selected_hand_sides")
    if not isinstance(hand_sides_raw, list) or not isinstance(selected_sides_raw, list):
        raise TrajectoryPackageError("trajectory package hand-side metadata is invalid")
    hand_sides = tuple(str(value) for value in hand_sides_raw)
    selected_hand_sides = tuple(str(value) for value in selected_sides_raw)
    q_by_side = arrays["q_ref_by_side"]
    if (
        q_by_side.ndim != 3
        or q_by_side.shape[0] != len(hand_sides)
        or q_by_side.shape[1] != total_frames
    ):
        raise TrajectoryPackageError("trajectory package q_ref_by_side layout is invalid")
    primary_side = "right" if "right" in selected_hand_sides else selected_hand_sides[0]
    if primary_side not in hand_sides:
        raise TrajectoryPackageError("trajectory package primary hand is absent")

    selection = manifest.get("selection")
    dataset = manifest.get("dataset")
    if not isinstance(selection, dict) or not isinstance(dataset, dict):
        raise TrajectoryPackageError("trajectory package source contract is missing")
    trajectories: list[ReferenceTrajectory] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("identity"), dict):
            raise TrajectoryPackageError(f"trajectory package record {index} is invalid")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        recorded_offset = record.get("offset")
        if recorded_offset != [start, stop]:
            raise TrajectoryPackageError(f"trajectory package record {index} offset mismatch")
        side_map = {
            side: _immutable(q_by_side[side_index, start:stop])
            for side_index, side in enumerate(hand_sides)
        }
        trajectories.append(
            ReferenceTrajectory(
                identity=TrajectoryIdentity(**record["identity"]),
                dataset_version=int(dataset["version"]),
                source_indices=_immutable(arrays["source_indices"][start:stop], dtype=np.int64),
                timestamps=_immutable(arrays["timestamps"][start:stop]),
                q_ref=side_map[primary_side],
                object_pos_raw=_immutable(arrays["object_pos_raw"][start:stop]),
                object_pos=_immutable(arrays["object_pos"][start:stop]),
                object_quat_xyzw=_immutable(arrays["object_quat_xyzw"][start:stop]),
                object_z_shift=float(record["object_z_shift"]),
                hand_sides=hand_sides,
                q_ref_by_side=side_map,
                selected_hand_sides=selected_hand_sides,
                reference_fps=selection.get("reference_fps"),
                control_fps=selection.get("control_fps"),
                movement_start_step=record.get("movement_start_step"),
                movement_end_step=record.get("movement_end_step"),
            )
        )

    resolved_pairs_raw = manifest.get("resolved_pairs")
    if not isinstance(resolved_pairs_raw, list):
        raise TrajectoryPackageError("trajectory package pair catalog is missing")
    resolved_pairs = tuple(
        ObjectActionPair(*str(value).split(":", maxsplit=1)) for value in resolved_pairs_raw
    )
    actual_counts = {
        pair.canonical: sum(_pair_for(item) == pair for item in trajectories)
        for pair in resolved_pairs
    }
    if actual_counts != manifest.get("pair_counts"):
        raise TrajectoryPackageError("trajectory package pair counts do not match records")
    source_catalog = manifest.get("source_catalog")
    if not isinstance(source_catalog, dict) or not isinstance(source_catalog.get("candidates"), list):
        raise TrajectoryPackageError("trajectory package source candidate catalog is missing")
    candidates = source_catalog["candidates"]
    if source_catalog.get("candidate_count") != len(candidates):
        raise TrajectoryPackageError("trajectory package source candidate count mismatch")
    decoded_keys = {
        (item.identity.row_index, item.identity.identity) for item in trajectories
    }
    catalog_keys: set[tuple[int, str]] = set()
    catalog_decoded_keys: set[tuple[int, str]] = set()
    rejected_count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TrajectoryPackageError("trajectory package source candidate entry is invalid")
        key = (int(candidate["row_index"]), str(candidate["identity"]))
        if key in catalog_keys:
            raise TrajectoryPackageError("trajectory package source candidates are duplicated")
        catalog_keys.add(key)
        status = candidate.get("status")
        if status == "decoded":
            catalog_decoded_keys.add(key)
        elif status == "rejected" and isinstance(candidate.get("rejection"), dict):
            rejected_count += 1
        else:
            raise TrajectoryPackageError("trajectory package source candidate status is invalid")
    if catalog_decoded_keys != decoded_keys:
        raise TrajectoryPackageError("trajectory package decoded rows do not match source catalog")
    if (
        source_catalog.get("decoded_count") != len(decoded_keys)
        or source_catalog.get("rejected_count") != rejected_count
    ):
        raise TrajectoryPackageError("trajectory package source catalog partition counts mismatch")
    catalog_digest = manifest.get("catalog_digest")
    package_digest = manifest.get("package_digest")
    if not isinstance(catalog_digest, str) or not isinstance(package_digest, str):
        raise TrajectoryPackageError("trajectory package content digests are missing")
    return TrajectoryCatalog(
        trajectories=tuple(trajectories),
        resolved_pairs=resolved_pairs,
        package_path=root,
        package_digest=package_digest,
        manifest_sha256=manifest_sha256,
        catalog_digest=catalog_digest,
        manifest=manifest,
    )


def _validate_catalog_selection(catalog: TrajectoryCatalog, selection: TrajectorySelection) -> None:
    manifest_selection = catalog.manifest.get("selection")
    dataset = catalog.manifest.get("dataset")
    if not isinstance(manifest_selection, Mapping) or not isinstance(dataset, Mapping):
        raise TrajectoryPackageError("trajectory package selection contract is missing")
    expected = {
        "hand_side": selection.hand_side,
        "pre_padding": selection.pre_padding,
        "post_padding": selection.post_padding,
        "reference_fps": selection.reference_fps,
        "control_fps": selection.resolved_control_fps,
    }
    mismatches = {
        key: (manifest_selection.get(key), value)
        for key, value in expected.items()
        if manifest_selection.get(key) != value
    }
    if selection.expected_dataset_version is not None and dataset.get("version") != selection.expected_dataset_version:
        mismatches["dataset_version"] = (dataset.get("version"), selection.expected_dataset_version)
    if mismatches:
        raise TrajectoryPackageError(f"trajectory package selection mismatch: {mismatches}")


def assign_trajectory_catalog(
    catalog: TrajectoryCatalog,
    selection: TrajectorySelection,
    *,
    num_envs: int,
) -> TrajectoryBatch:
    """Assign a complete MTP catalog without Lance/PyArrow imports."""

    if not isinstance(catalog, TrajectoryCatalog):
        raise TypeError("catalog must be a TrajectoryCatalog")
    if not isinstance(selection, TrajectorySelection):
        raise TypeError("selection must be a TrajectorySelection")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    _validate_catalog_selection(catalog, selection)
    decoded_by_key = {
        (item.identity.row_index, item.identity.identity): item
        for item in catalog.trajectories
    }
    source_catalog = catalog.manifest.get("source_catalog")
    if not isinstance(source_catalog, Mapping) or not isinstance(source_catalog.get("candidates"), list):
        raise TrajectoryPackageError("trajectory package source candidate catalog is missing")
    raw_by_pair: dict[ObjectActionPair, tuple[Mapping[str, object], ...]] = {}
    by_pair: dict[ObjectActionPair, tuple[ReferenceTrajectory, ...]] = {}
    for pair in catalog.resolved_pairs:
        raw = tuple(
            item
            for item in source_catalog["candidates"]
            if isinstance(item, Mapping) and item.get("pair") == pair.canonical
        )
        values = tuple(
            decoded_by_key[(int(item["row_index"]), str(item["identity"]))]
            for item in raw
            if item.get("status") == "decoded"
        )
        if raw and values:
            raw_by_pair[pair] = raw
            by_pair[pair] = values
    requested = selection.requested_pairs
    resolved_pairs = tuple(by_pair) if requested is None else requested
    missing = tuple(pair for pair in resolved_pairs if pair not in by_pair)
    if missing:
        raise LookupError(
            "trajectory package selector pair(s) are absent: "
            + ", ".join(pair.canonical for pair in missing)
        )
    if not resolved_pairs:
        raise LookupError("trajectory package selection resolved no pairs")
    pair_slot_counts = {pair: 0 for pair in resolved_pairs}
    for env_id in range(num_envs):
        pair_slot_counts[resolved_pairs[env_id % len(resolved_pairs)]] += 1
    rotated: dict[ObjectActionPair, tuple[ReferenceTrajectory, ...]] = {}
    for pair in resolved_pairs:
        raw = raw_by_pair[pair]
        offset = selection.pair_assignment_cycle * pair_slot_counts[pair] % len(raw)
        rotated_raw = raw[offset:] + raw[:offset]
        values = tuple(
            decoded_by_key[(int(item["row_index"]), str(item["identity"]))]
            for item in rotated_raw
            if item.get("status") == "decoded"
        )
        if not values:
            raise TrajectoryPackageError(
                f"trajectory package pair {pair.canonical} has no decoded trajectories"
            )
        rotated[pair] = values
    pair_slots = {pair: 0 for pair in resolved_pairs}
    assignments: list[ReferenceTrajectory] = []
    for env_id in range(num_envs):
        pair = resolved_pairs[env_id % len(resolved_pairs)]
        values = rotated[pair]
        pair_slot = pair_slots[pair]
        assignments.append(values[pair_slot % len(values)])
        pair_slots[pair] += 1
    return TrajectoryBatch(
        tuple(assignments),
        resolved_pairs=resolved_pairs,
        selection_mode=selection.mode,
        pair_assignment_cycle=selection.pair_assignment_cycle,
        trajectory_package=catalog.checkpoint_metadata,
    )


def load_assigned_trajectory_package(
    package: str | Path,
    selection: TrajectorySelection,
    *,
    num_envs: int,
    verify_hashes: bool = True,
) -> TrajectoryBatch:
    catalog = load_trajectory_package(package, verify_hashes=verify_hashes)
    return assign_trajectory_catalog(catalog, selection, num_envs=num_envs)
