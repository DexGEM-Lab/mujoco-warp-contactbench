"""Deterministic 28D URDF-to-MANO pose conversion for synthetic exports.

The right-hand axes and composition order are read from the pinned DexStream
`hand/mano/sunke/right` URDF. Keeping this small conversion local makes
checkpoint exports self-contained while preserving the source MANO 48D layout.
"""

from __future__ import annotations

from functools import lru_cache
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


_AXIS_JOINTS = {
    "thumb_cmc_abd": "j1_thumb_cmc_abd",
    "thumb_cmc_flex": "j1_thumb_cmc_flex",
    "thumb_cmc_twist": "j1_thumb_cmc_twist",
    "thumb_mcp_flex": "j1_thumb_mcp_flex",
    "thumb_mcp_abd": "j1_thumb_mcp_abd",
    "thumb_ip": "j1_thumb_ip",
    "index_mcp_abd": "j2_index_mcp_abd",
    "index_mcp_flex": "j2_index_mcp_flex",
    "index_pip": "j2_index_pip",
    "index_dip": "j2_index_dip",
    "middle_mcp_abd": "j3_middle_mcp_abd",
    "middle_mcp_flex": "j3_middle_mcp_flex",
    "middle_pip": "j3_middle_pip",
    "middle_dip": "j3_middle_dip",
    "ring_mcp_abd": "j4_ring_mcp_abd",
    "ring_mcp_flex": "j4_ring_mcp_flex",
    "ring_pip": "j4_ring_pip",
    "ring_dip": "j4_ring_dip",
    "pinky_mcp_abd": "j5_pinky_mcp_abd",
    "pinky_mcp_flex": "j5_pinky_mcp_flex",
    "pinky_pip": "j5_pinky_pip",
    "pinky_dip": "j5_pinky_dip",
}

_MANO_SLICES = {
    "index_mcp": slice(0, 3),
    "index_pip": slice(3, 6),
    "index_dip": slice(6, 9),
    "middle_mcp": slice(9, 12),
    "middle_pip": slice(12, 15),
    "middle_dip": slice(15, 18),
    "pinky_mcp": slice(18, 21),
    "pinky_pip": slice(21, 24),
    "pinky_dip": slice(24, 27),
    "ring_mcp": slice(27, 30),
    "ring_pip": slice(30, 33),
    "ring_dip": slice(33, 36),
    "thumb_cmc": slice(36, 39),
    "thumb_mcp": slice(39, 42),
    "thumb_ip": slice(42, 45),
}

_FINGER_LAYOUT = {
    "thumb_cmc": (0, 1, 2),
    "thumb_mcp": (3, 4),
    "thumb_ip": (5,),
    "index": (6, 7, 8, 9),
    "middle": (10, 11, 12, 13),
    "ring": (14, 15, 16, 17),
    "pinky": (18, 19, 20, 21),
}


@lru_cache(maxsize=1)
def _source_axes() -> dict[str, NDArray[np.float64]]:
    """Read the exact axis vectors from the selected DexStream MANO URDF."""

    from sim.manorl.assets import hand_urdf_path

    root = ET.parse(hand_urdf_path("right")).getroot()
    axes: dict[str, NDArray[np.float64]] = {}
    by_joint = {
        joint.get("name", ""): joint
        for joint in root.findall("joint")
    }
    for logical_name, joint_name in _AXIS_JOINTS.items():
        joint = by_joint.get(joint_name)
        axis = None if joint is None else joint.find("axis")
        if axis is None or not axis.get("xyz"):
            raise ValueError(f"DexStream MANO URDF has no axis for {joint_name}")
        value = np.fromstring(axis.get("xyz", ""), sep=" ", dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid DexStream MANO axis for {joint_name}")
        norm = float(np.linalg.norm(value))
        if not np.isclose(norm, 1.0, atol=1e-6, rtol=0):
            raise ValueError(f"DexStream MANO axis is not unit length for {joint_name}: {norm}")
        axes[logical_name] = value
    return axes


def _axis(name: str) -> NDArray[np.float64]:
    try:
        return _source_axes()[name]
    except KeyError as exc:
        raise ValueError(f"unknown MANO axis {name!r}") from exc


def _compose(angles: NDArray[np.float64], axes: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    rotation = Rotation.identity()
    for angle, axis in zip(angles, axes, strict=True):
        rotation = rotation * Rotation.from_rotvec(float(angle) * axis)
    return rotation.as_rotvec()


def right_urdf_fingers_to_mano_45d(finger_dofs: NDArray[object]) -> NDArray[np.float64]:
    """Convert one current right-hand 22D finger vector to MANO's 45D pose."""

    angles = np.asarray(finger_dofs, dtype=np.float64)
    if angles.shape != (22,) or not np.all(np.isfinite(angles)):
        raise ValueError("right MANO conversion requires one finite 22D finger vector")
    mano = np.zeros(45, dtype=np.float64)
    cmc = _FINGER_LAYOUT["thumb_cmc"]
    mano[_MANO_SLICES["thumb_cmc"]] = _compose(
        angles[list(cmc)],
        [_axis("thumb_cmc_abd"), _axis("thumb_cmc_flex"), _axis("thumb_cmc_twist")],
    )
    mcp = _FINGER_LAYOUT["thumb_mcp"]
    mano[_MANO_SLICES["thumb_mcp"]] = _compose(
        angles[list(mcp)], [_axis("thumb_mcp_flex"), _axis("thumb_mcp_abd")]
    )
    mano[_MANO_SLICES["thumb_ip"]] = angles[_FINGER_LAYOUT["thumb_ip"][0]] * _axis("thumb_ip")
    for finger in ("index", "middle", "pinky", "ring"):
        abd, flex, pip, dip = _FINGER_LAYOUT[finger]
        mano[_MANO_SLICES[f"{finger}_mcp"]] = _compose(
            angles[[abd, flex]], [_axis(f"{finger}_mcp_abd"), _axis(f"{finger}_mcp_flex")]
        )
        mano[_MANO_SLICES[f"{finger}_pip"]] = angles[pip] * _axis(f"{finger}_pip")
        mano[_MANO_SLICES[f"{finger}_dip"]] = angles[dip] * _axis(f"{finger}_dip")
    return mano


def right_urdf_trajectory_to_mano_48d(urdf_dof: NDArray[object]) -> NDArray[np.float64]:
    """Convert physical 28D right-hand states to the source 48D MANO rows.

    The first three local-wrist components remain zero because global wrist
    orientation is stored independently in ``mano_global_rot_aa``.
    """

    values = np.asarray(urdf_dof, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 28 or not np.all(np.isfinite(values)):
        raise ValueError("right MANO trajectory conversion requires finite (T, 28) values")
    result = np.zeros((len(values), 48), dtype=np.float64)
    result[:, 3:] = np.stack(
        [right_urdf_fingers_to_mano_45d(row[6:]) for row in values], axis=0
    )
    return result
