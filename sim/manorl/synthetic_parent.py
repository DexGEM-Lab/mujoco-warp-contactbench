"""Accepted parent rows for synthesis-only approach augmentation.

This module contains no Lance/PyArrow dependency. A Lance-healthy selector tool
extracts one previously accepted scalable-synthesis row into this compact JSON
contract; the long-lived exporter consumes only the descriptor and a predecoded
pre60 bundle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Final, Mapping

import numpy as np

ACCEPTED_SYNTHETIC_PARENT_CONTRACT: Final = "manorl_accepted_synthetic_parent_v3"
RETREAT_ANCHOR_OFFSET_FRAMES: Final[int] = 15
RETREAT_ANCHOR_CONTRACT: Final = "parent_movement_end_plus15_v2"
_SYNTHETIC_ROW_CONTRACTS: Final = frozenset(
    {
        "synthetic_mano_target_replay_visual_v2_contact",
        "synthetic_mano_28d_checkpoint_rollout_v2_3",
    }
)
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*_[0-9]{2}_[0-9]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AcceptedSyntheticParent:
    """One success-qualified row selected from a prior synthesis publication."""

    contract: str
    parent_dataset_path: str
    parent_dataset_version: int
    parent_row_index: int
    parent_row_uuid: str
    parent_row_contract: str
    source_identity: str
    source_dataset_path: str
    source_dataset_version: int
    source_row_index: int
    checkpoint_sha256: str
    checkpoint_update: int
    parent_seed: int
    parent_episode_index: int
    parent_generation_attempt: int
    object_init_xy_offset_m: tuple[float, float]
    reference_fps: int
    retreat_last_contact_state_index: int | None = None
    retreat_anchor_state_index: int | None = None
    retreat_anchor_source_frame_index: int | None = None
    retreat_anchor_horizontal_distance_m: float | None = None
    retreat_anchor_contract: str = RETREAT_ANCHOR_CONTRACT
    retreat_anchor_offset_frames: int = RETREAT_ANCHOR_OFFSET_FRAMES
    parent_movement_end_state_index: int | None = None
    source_row_frame0_right_q_ref_3_28: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.contract != ACCEPTED_SYNTHETIC_PARENT_CONTRACT:
            raise ValueError(f"unsupported accepted-parent contract: {self.contract!r}")
        if self.parent_row_contract not in _SYNTHETIC_ROW_CONTRACTS:
            raise ValueError(
                f"unsupported accepted-parent row contract: {self.parent_row_contract!r}"
            )
        if not _IDENTITY_RE.fullmatch(self.source_identity):
            raise ValueError("accepted parent has an invalid source_identity")
        if not _SHA256_RE.fullmatch(self.checkpoint_sha256):
            raise ValueError("accepted parent checkpoint_sha256 is invalid")
        for name in (
            "parent_dataset_version",
            "parent_row_index",
            "source_dataset_version",
            "source_row_index",
            "checkpoint_update",
            "parent_seed",
            "parent_episode_index",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"accepted parent {name} must be a non-negative integer")
        if (
            not isinstance(self.parent_generation_attempt, int)
            or isinstance(self.parent_generation_attempt, bool)
            or self.parent_generation_attempt < 1
        ):
            raise ValueError(
                "accepted parent parent_generation_attempt must be a positive integer"
            )
        if self.reference_fps not in (100, 120):
            raise ValueError("accepted parent reference_fps must be 100 or 120")
        retreat_fields = (
            self.retreat_last_contact_state_index,
            self.retreat_anchor_state_index,
            self.retreat_anchor_source_frame_index,
        )
        if not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for value in retreat_fields
        ):
            raise ValueError(
                "accepted parent v3 requires all retreat anchor fields as non-negative integers"
            )
        if (
            self.retreat_anchor_horizontal_distance_m is None
            or not np.isfinite(self.retreat_anchor_horizontal_distance_m)
            or self.retreat_anchor_horizontal_distance_m <= 1e-9
        ):
            raise ValueError(
                "accepted parent v3 requires a positive horizontal retreat distance"
            )
        object.__setattr__(
            self,
            "retreat_anchor_horizontal_distance_m",
            float(self.retreat_anchor_horizontal_distance_m),
        )
        if self.retreat_anchor_contract != RETREAT_ANCHOR_CONTRACT:
            raise ValueError(
                f"unsupported retreat anchor contract: {self.retreat_anchor_contract!r}"
            )
        if self.retreat_anchor_offset_frames != RETREAT_ANCHOR_OFFSET_FRAMES:
            raise ValueError(
                "accepted parent retreat_anchor_offset_frames must equal "
                f"{RETREAT_ANCHOR_OFFSET_FRAMES}"
            )
        if (
            not isinstance(self.parent_movement_end_state_index, int)
            or isinstance(self.parent_movement_end_state_index, bool)
            or self.parent_movement_end_state_index < 0
        ):
            raise ValueError(
                "accepted parent v3 requires parent_movement_end_state_index"
            )
        if self.retreat_anchor_state_index != (
            self.parent_movement_end_state_index
            + self.retreat_anchor_offset_frames
        ):
            raise ValueError(
                "accepted parent retreat anchor must equal movement end plus offset"
            )
        if self.retreat_last_contact_state_index < self.parent_movement_end_state_index:
            raise ValueError(
                "accepted parent last contact precedes movement end"
            )
        if not self.parent_row_uuid or not self.parent_dataset_path:
            raise ValueError("accepted parent dataset path and row UUID are required")
        if not self.source_dataset_path:
            raise ValueError("accepted parent source dataset path is required")
        if self.source_row_frame0_right_q_ref_3_28 is None:
            raise ValueError(
                "accepted parent v3 requires source-row frame0 right q_ref[3:28]"
            )
        initial_dof = np.asarray(
            self.source_row_frame0_right_q_ref_3_28, dtype=np.float64
        )
        if initial_dof.shape != (25,) or not np.all(np.isfinite(initial_dof)):
            raise ValueError(
                "accepted parent source-row frame0 right q_ref[3:28] must be 25 finite values"
            )
        object.__setattr__(
            self,
            "source_row_frame0_right_q_ref_3_28",
            tuple(float(value) for value in initial_dof),
        )
        offset = np.asarray(self.object_init_xy_offset_m, dtype=np.float64)
        if offset.shape != (2,) or not np.all(np.isfinite(offset)):
            raise ValueError("accepted parent object XY offset must be a finite pair")
        object.__setattr__(
            self,
            "object_init_xy_offset_m",
            (float(offset[0]), float(offset[1])),
        )

    @property
    def object_type(self) -> str:
        return self.source_identity.split("_", 1)[0]

    @property
    def action_id(self) -> str:
        return self.source_identity.split("_")[1]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["object_init_xy_offset_m"] = list(self.object_init_xy_offset_m)
        if self.source_row_frame0_right_q_ref_3_28 is not None:
            result["source_row_frame0_right_q_ref_3_28"] = list(
                self.source_row_frame0_right_q_ref_3_28
            )
        return result


def accepted_parent_from_dict(values: Mapping[str, object]) -> AcceptedSyntheticParent:
    if not isinstance(values, Mapping):
        raise TypeError("accepted parent must be a mapping")
    data = dict(values)
    offset = data.get("object_init_xy_offset_m")
    if isinstance(offset, list):
        data["object_init_xy_offset_m"] = tuple(offset)
    initial_dof = data.get("source_row_frame0_right_q_ref_3_28")
    if isinstance(initial_dof, list):
        data["source_row_frame0_right_q_ref_3_28"] = tuple(initial_dof)
    try:
        return AcceptedSyntheticParent(**data)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"accepted parent fields are invalid: {exc}") from exc


def load_accepted_synthetic_parent(path: str | Path) -> AcceptedSyntheticParent:
    descriptor = Path(path).expanduser().resolve()
    if not descriptor.is_file():
        raise FileNotFoundError(f"accepted-parent descriptor is absent: {descriptor}")
    try:
        values = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("accepted-parent descriptor is invalid JSON") from exc
    return accepted_parent_from_dict(values)


def write_accepted_synthetic_parent(
    parent: AcceptedSyntheticParent,
    path: str | Path,
    *,
    replace: bool = False,
) -> Path:
    if not isinstance(parent, AcceptedSyntheticParent):
        raise TypeError("parent must be AcceptedSyntheticParent")
    output = Path(path).expanduser().resolve()
    if output.exists() and not replace:
        raise FileExistsError(f"accepted-parent descriptor exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(parent.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
