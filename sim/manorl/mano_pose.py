"""Deterministic 28D URDF-to-MANO pose conversion for synthetic exports.

The right-hand axes and composition order are the current 22-finger-DOF
contract used by manohand_reconstruction's ``URDFToMANOConverter``.  Keeping
this small conversion local makes checkpoint exports self-contained while
preserving the source MANO 48D layout.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


_RIGHT_AXES = {
    "thumb_cmc_abd": (0.13838553, 0.9562202, 0.25786126),
    "thumb_cmc_flex": (0.983308, -0.16371468, 0.07939028),
    "thumb_cmc_twist": (0.11813027, 0.24257056, -0.9629147),
    "thumb_mcp_flex": (0.3696764866888757, -0.8790409917024125, 0.3010419075414728),
    "thumb_mcp_abd": (-0.5716182768686316, -0.4705844355444832, -0.6721628036220215),
    "thumb_ip": (0.3696764866888757, -0.8790409917024125, 0.3010419075414728),
    "index_mcp_abd": (0.061695436024444154, -0.9980950221165086, 0.0),
    "index_mcp_flex": (-0.007558282177929706, -0.0004672015231318583, 0.999971326635547),
    "index_pip": (-0.007558282177929706, -0.0004672015231318583, 0.999971326635547),
    "index_dip": (-0.007558282177929706, -0.0004672015231318583, 0.999971326635547),
    "middle_mcp_abd": (0.059278674, -0.99806947, -0.018527139),
    "middle_mcp_flex": (-0.16968586, -0.02836441, 0.98508996),
    "middle_pip": (-0.16968586, -0.02836441, 0.98508996),
    "middle_dip": (-0.16968586, -0.02836441, 0.98508996),
    "ring_mcp_abd": (0.03498914, -0.9917128, 0.12361857),
    "ring_mcp_flex": (-0.31276166, 0.10661509, 0.9438292),
    "ring_pip": (-0.31276166, 0.10661509, 0.9438292),
    "ring_dip": (-0.31276166, 0.10661508, 0.9438292),
    "pinky_mcp_abd": (-0.12182937, -0.9582415, 0.25870955),
    "pinky_mcp_flex": (-0.52631825, 0.283357, 0.80168444),
    "pinky_pip": (-0.52631825, 0.283357, 0.80168444),
    "pinky_dip": (-0.52631825, 0.283357, 0.80168444),
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


def _axis(name: str) -> NDArray[np.float64]:
    return np.asarray(_RIGHT_AXES[name], dtype=np.float64)


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
