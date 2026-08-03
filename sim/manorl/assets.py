"""Deterministic MuJoCo scene construction from curated authoritative URDF assets."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from sim.manorl.contracts import (
    ACTION_SIDE_ORDER,
    JOINT_DOF,
    EFFORT,
    FLOOR_TOP_Z,
    JOINT_ARMATURE,
    JOINT_FRICTIONLOSS,
    normalize_hand_side,
    OBJECT_TYPE,
    PHYSICS_TIMESTEP,
    ServoConfig,
)

ASSET_ROOT = Path(__file__).resolve().parent / "runtime_assets"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ALL_ASSETS_ROOT = REPOSITORY_ROOT / "assets" / "all_assets"
ALL_ASSETS_SIM_ROOT = ALL_ASSETS_ROOT / "Assets" / "sim"
ALL_OBJECT_GRASPS = ALL_ASSETS_ROOT / "Assets" / "object_grasps_simple.yaml"
ALL_OBJECT_GRASPS_SHA256 = (
    "97293b96a3015173bf2fa875d165b04c925953147ebc0f4c0935ed1215558a9d"
)
HAND_URDF = ASSET_ROOT / "hand" / "mano_hand.urdf"
HAND_LEFT_ROOT = ASSET_ROOT / "hand_left"
HAND_URDF_BY_SIDE = {
    "right": HAND_URDF,
    "left": HAND_LEFT_ROOT / "mano_hand.urdf",
}
OBJECT_URDF = ASSET_ROOT / "cube1" / "cube1.urdf"
ASSET_MANIFEST = ASSET_ROOT / "manifest.json"
OBJECT_MESH = ASSET_ROOT / "cube1" / "cube1_aligned.stl"
HAND_VISUAL_MESH_ROOT = (
    ALL_ASSETS_ROOT / "Assets" / "HAND" / "s02" / "mano" / "Z_upNew" / "meshes"
)
OBJECT_VISUAL_MESH_ROOT = ALL_ASSETS_SIM_ROOT / "mano_assets" / "objects"
VISUAL_GEOM_GROUP = 2
COLLISION_GEOM_GROUP = 3


def hand_asset_root(hand_side: str = "right") -> Path:
    """Return the curated runtime root for one normalized hand side."""

    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
    root = ASSET_ROOT / ("hand" if side == "right" else "hand_left")
    if not root.is_dir():
        raise FileNotFoundError(
            f"curated {side}-hand runtime assets are absent: {root}"
        )
    return root


def hand_urdf_path(hand_side: str = "right") -> Path:
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
    path = HAND_URDF_BY_SIDE[side]
    if not path.is_file():
        raise FileNotFoundError(f"curated {side}-hand URDF is absent: {path}")
    return path


def hand_joint_names(hand_side: str = "right") -> tuple[str, ...]:
    """Read and validate the authoritative joint order from side metadata."""

    root = hand_asset_root(hand_side)
    metadata_path = root / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {hand_side} hand metadata: {metadata_path}") from exc
    names = tuple(payload.get("joint_names", ()))
    if len(names) != JOINT_DOF or len(set(names)) != JOINT_DOF:
        raise ValueError(
            f"{hand_side} hand metadata must declare {JOINT_DOF} unique joints"
        )
    return names


@dataclass(frozen=True)
class ObjectRuntime:
    """Materialized object inputs for one homogeneous MJX model."""

    object_type: str
    link_name: str
    body_name: str
    free_joint_name: str
    source_mesh_filename: str
    urdf_path: Path
    collision_mesh_paths: tuple[Path, ...]
    grasp_mapping_path: Path
    source_mesh_scale: tuple[float, float, float]
    collision_mesh_scales: tuple[tuple[float, float, float], ...]
    rgba: str
    geometry_type: str = "box"
    expected_sha256: tuple[tuple[Path, str], ...] = ()

    @property
    def collision_geom_count(self) -> int:
        return len(self.collision_mesh_paths)

    @property
    def collision_mesh_path(self) -> Path:
        """Compatibility accessor for legacy single-piece callers.

        Multi-piece runtimes must use :attr:`collision_mesh_paths`; returning
        the first piece here keeps older asset probes useful without silently
        dropping pieces from scene construction or vertex decoding.
        """

        return self.collision_mesh_paths[0]

    @property
    def collision_mesh_scale(self) -> tuple[float, float, float]:
        """Compatibility accessor for the first collision piece."""

        return self.collision_mesh_scales[0]

    def __post_init__(self) -> None:
        if not self.collision_mesh_paths:
            raise ValueError(f"{self.object_type} runtime requires collision meshes")
        if len(self.collision_mesh_scales) != len(self.collision_mesh_paths):
            raise ValueError(
                f"{self.object_type} collision mesh paths/scales must have equal lengths"
            )


def _all_assets_runtime(
    object_type: str,
    *,
    urdf_sha256: str,
    collision_sha256: str,
    rgba: str,
) -> ObjectRuntime:
    urdf_path = ALL_ASSETS_SIM_ROOT / "mano_objects_urdf" / f"{object_type}.urdf"
    collision_path = (
        ALL_ASSETS_SIM_ROOT
        / "for_math_retaregeting"
        / object_type
        / "coacd"
        / "coacd_convex_piece_0.obj"
    )
    return ObjectRuntime(
        object_type=object_type,
        link_name=f"{object_type}_link",
        body_name=object_type,
        free_joint_name=f"{object_type}_free",
        source_mesh_filename=f"{object_type}.obj",
        urdf_path=urdf_path,
        collision_mesh_paths=(collision_path,),
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        collision_mesh_scales=((1.0, 1.0, 1.0),),
        rgba=rgba,
        expected_sha256=(
            (urdf_path, urdf_sha256),
            (collision_path, collision_sha256),
            (ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),
        ),
    )


# This registry is deliberately closed. Adding a name requires materialized,
# digest-checked URDF, collision, and grasp inputs; arbitrary Lance object names
# must never fall through to cube1 geometry. Each profile compiles a separate
# static MJX model and heterogeneous vector batches are routed by object type.
_OBJECT_RUNTIMES = {
    "cube1": ObjectRuntime(
        object_type="cube1",
        link_name="cube1_link",
        body_name="cube1",
        free_joint_name="cube1_free",
        source_mesh_filename="cube1.obj",
        urdf_path=ASSET_ROOT / "cube1" / "cube1.urdf",
        collision_mesh_paths=(OBJECT_MESH,),
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        collision_mesh_scales=((0.001, 0.001, 0.001),),
        rgba="0.8 0.18 0.16 1",
        expected_sha256=((ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),),
    ),
    "cuboid1": _all_assets_runtime(
        "cuboid1",
        urdf_sha256="bf4f6d101acdbac594c2da7c4ea48b577d222738887873a7b8e8cb608fc605db",
        collision_sha256="1d02f428a7c8e0264788dbe3379385002f93bcf38ad19c727850beb099428928",
        rgba="0.72 0.45 0.12 1",
    ),
    "cuboid2": _all_assets_runtime(
        "cuboid2",
        urdf_sha256="93442dc020074eec170ea4aa322a23c4a3784c69af46befa621b8bb256a6e9d2",
        collision_sha256="4ca4388942b10a224c9d5e8fa04da6629e1e07d9379131bc8085f823da03e3a4",
        rgba="0.36 0.62 0.18 1",
    ),
    "cylinder2": _all_assets_runtime(
        "cylinder2",
        urdf_sha256="98ba6559327367aa17e5611407dad4336a832e08ef276fe7425961408183e5a2",
        collision_sha256="f5e70d139ed1209c643a81ba5313aeec301d630b26b17c4c39d9484b3b50cae0",
        rgba="0.16 0.62 0.58 1",
    ),
    "cylinder7": _all_assets_runtime(
        "cylinder7",
        urdf_sha256="8ce7485e27852048d84d35f858f0653c242621d3370a425a8d0414c011d9a630",
        collision_sha256="65dc08550f006c7f380a7094b70eb2e1753a735028445e7c23facc8782078d0a",
        rgba="0.75 0.35 0.12 1",
    ),
    "pitcherbase": ObjectRuntime(
        object_type="pitcherbase",
        link_name="pitcherbase_link",
        body_name="pitcherbase",
        free_joint_name="pitcherbase_free",
        source_mesh_filename="pitcherbase.obj",
        urdf_path=ALL_ASSETS_SIM_ROOT / "decomposed" / "pitcherbase" / "pitcherbase.urdf",
        collision_mesh_paths=tuple(
            ALL_ASSETS_SIM_ROOT
            / "for_math_retaregeting"
            / "pitcherbase"
            / "coacd"
            / f"coacd_convex_piece_{index}.obj"
            for index in range(38)
        ),
        collision_mesh_scales=((1.0, 1.0, 1.0),) * 38,
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        rgba="0.55 0.75 0.92 1",
        geometry_type="irregular",
        expected_sha256=(
            (ALL_ASSETS_SIM_ROOT / "decomposed/pitcherbase/pitcherbase.urdf", "8e78eeefdc603f90cedbc712261c094fbb1a222e8803021e9018902f2f8d50d8"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_0.obj", "fd25f208435c0b97c012fa74811bc414c5d5c771ce027c63c4c4dc4862a37749"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_1.obj", "9a8dd5266f0abc4adb7294443094ab2258b5d4e2b92ad484242391317bea0459"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_2.obj", "fcee120dd23a029153c538e1fcc81b78a24737fea2819c40b9046a5500d2e2ff"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_3.obj", "8bf08922e13ce02c186874148833095496646c6142cb263364c4afd8baef2b54"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_4.obj", "ba8440ad4da829e4a552eb4c229f4254dcd142f89f99206fdbc5f889ed1aa38f"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_5.obj", "b3a1dfb17bef71d6f0d19400ed416c8d92bf94894087e5db24f4c90a969cf5c6"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_6.obj", "579119dff739b4ff59ccbb2ef146f9101a74445fcbde0b3cffdc008c6256156d"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_7.obj", "23ea0f49f3184a2f1d0f1d26ad1ef523d523034f5e9b2be66b070bb5be72a226"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_8.obj", "2f9aff7d751723d6e1c66b1f44f00a40c424273a0cb977e09cd4169aadea3aff"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_9.obj", "4841aff760cb66a8798ab29343e43412437e1111a35561b45fcf3b7ddb8ac615"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_10.obj", "d8b9d4a4f8f4a01f41dd329601dfb427f2328d35c6d80e49edcfb67418373f14"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_11.obj", "ba775fced54e3f528aa4b70a8b70de00ed810372889754b6c8cd217be672a4e2"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_12.obj", "f4a0166471b0e2610616ef776a96575a9f07e55c87e29fbc7a97aaebee831d23"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_13.obj", "fbae5e3b95f6086511353323ef105b0c2056ba79c2fecfa22cb8288fa916eb79"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_14.obj", "e52edb51fe8481ec63ade1e6ce68a66418cc9e9881e991978aa264bb2a9eb5c2"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_15.obj", "6f89488c72c31973c348df55e37f94f126d42290fd8b8c01310fc6a4c5473222"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_16.obj", "d165f16a7d63e1493a40a352689a580f45c97cf64cd66a3f14898a0861edff7f"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_17.obj", "3956b8fc6a349dfda6aa64200f848ac5f1d3ff639af98825066b8b0d4714e568"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_18.obj", "b72a067f0e5d1fffbda8c9e9b78c7724ee0e0436b0b92f66dbc0d5aa6b8fe227"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_19.obj", "b8ac7dcd4b8e66dfac1857506904ffb7b648291bbe2fd397a3725e425fed435e"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_20.obj", "6bc6d6dafaf5df2c6eb915599a071d1e51943ebb97d87805a3c445faa4e0c82d"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_21.obj", "20f46efd222b2b7de681af23f9dd74a6ffee0a567660de5b15f07392e9f1673a"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_22.obj", "a90f763611a774f7c4a41583d43395bc9b4c16c9ef890239038422241955a016"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_23.obj", "5645fbf78e2e060c36fab4e3f4f8dbaca5989ff8dd8d92d9298689db2e5ffc3b"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_24.obj", "ebf6bff0c7ae5d011ae9c2eb25a76f37436487b6328a3be817628617111baef9"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_25.obj", "71d170646fa45042fba1202af4d4950f5bd9c2e58626f0f58dfeccb09538f195"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_26.obj", "bb477ee35fa87c273390b24b65b5ec6c346bd1fb53fa8ed2a41888998e247b11"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_27.obj", "0ad0660f1b682651e0bf7789252b4bf9b325ece28ca9047f17807cdf64d473e0"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_28.obj", "ea4f5821a59b83ec7972ecf41255d9535e0d881630e6ab9621f0a831b179cf4e"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_29.obj", "475c8b2142c3e6a62cb7eac344cc0973a28e9ed199376afe9399559b067bf690"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_30.obj", "2a43863b1bb3063b4b3abddb4c3f3a3f42d05a0aa5d82c947cd25110419e0a17"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_31.obj", "4291edcb7bd0940e7af45f87675a4de8710c88bb5461ec79c65b5d0296084f57"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_32.obj", "e971cb6640c40ddc8aef34381fab85a8a39bf7c7651509d6f220d349cc672813"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_33.obj", "9737ba46d8c941c17f2d27a53c8288482869ec4f1e39793e43848bb7016940ff"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_34.obj", "cc8dff68a33125dcfcca6989a70f4dd9b2a3d755a69e22f8acfa1926d09fe7c2"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_35.obj", "61a76fd9e264e4249003aff22277a4bbcb0cf0545d883e8fc033f8addccbe767"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_36.obj", "afa01daffa53a3b99fbcbe5bfd2d3155a6b4b295bf28cd24b3b304a4d24dfd2c"),
            (ALL_ASSETS_SIM_ROOT / "for_math_retaregeting/pitcherbase/coacd/coacd_convex_piece_37.obj", "e76637d3cf0ced352a382b25964817ebdb18f5472f1f0a6960292668db21874c"),
            (ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),
        ),
    ),
    "bowl": ObjectRuntime(
        object_type="bowl",
        link_name="bowl_link",
        body_name="bowl",
        free_joint_name="bowl_free",
        source_mesh_filename="bowl.obj",
        urdf_path=ALL_ASSETS_SIM_ROOT / "decomposed" / "bowl" / "bowl.urdf",
        collision_mesh_paths=tuple(
            ALL_ASSETS_SIM_ROOT
            / "decomposed"
            / "bowl"
            / "coacd"
            / f"coacd_convex_piece_{index}.obj"
            for index in range(28)
        ),
        collision_mesh_scales=((1.0, 1.0, 1.0),) * 28,
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        rgba="0.68 0.30 0.82 1",
        geometry_type="irregular",
        expected_sha256=(
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/bowl.urdf", "7ec12e21fd66b60e00fab7c41b668a8a0ca4fbc4f096a1e01332c668a772edc7"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_0.obj", "22d286fd17044aa89b75cde59907377ac3161a77cabd61c8f6ed14e769cb9154"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_1.obj", "3bf4afc4c31a207a1c1960e8b6703885e6e25cefe11264c9331900ce54b1dfcc"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_2.obj", "2a85639e8ef1743fb35f87794f8b1d2ea7462bfac5a012d2de716634ba891a9d"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_3.obj", "7f24b172bc02b2c942182f45eca7f850deabe3ff2ed8b5d3e939acff816d9d33"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_4.obj", "0e76c8affd5a12b2317a74f4a48acc73956f1a8c0b2c663f61bd4e141a15baa6"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_5.obj", "038cacb591e818edea2db7728ae34f291eb2580fbe27db04536434ea8ea915c3"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_6.obj", "a7ef87145dd5a1462a04c26a10ac6ffcf1369129a3e75dbd8a4e603bbb82144e"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_7.obj", "bf257d629e1eff9ce5454a4ebe9d03edd1d6ea6e9c376d3ceb7b67055f6142a3"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_8.obj", "db4f018b320f1a65357010f071ad89e2223a83e524af33a90be696e0ea4f5aee"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_9.obj", "d2de97ba87f3055182586b745face9575b21d440159cb0927afc907a1cd15dae"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_10.obj", "5d26956bfcf4a23cdaaa004567709f76539754f7b53e65adff6c3af103289319"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_11.obj", "22c95c956bcd37e90ce89782173be74adfb8f46ebd07976079e63808e79fcd11"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_12.obj", "18b0f78e88c2d5b8e6a9f110be3bcff8b7c924ebcdd7b2d0f0fe4dca3a3e01ae"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_13.obj", "0657ee476213eb569f3fb1ce7b8ef8a8a6e1cd8d986112438d1ac56def70f768"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_14.obj", "769a7ad15affa1fc04bb00e5560986f78a41bf2d94fb7325d15dfe67aec27196"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_15.obj", "d786550327ecb9f6582ccc7fddc43ee37e12466312ef72b299daf92336d35586"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_16.obj", "e6fd0a6790e567f051027d417a349f98e9192507f3a9dca958748b4e09e61ccd"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_17.obj", "3d92cd610ddd61d11114c0167e1429ac597f82fc3784b4ad4d2e88f223403e5e"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_18.obj", "df020c722ece9f1c1147e2b9b6d246be6890e495cc19cb28e6e0fd95c614a7ba"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_19.obj", "4483ae3f36a7463d17c8617fe7998d03862c45938f96227884188e594948bfd8"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_20.obj", "aa8d9af06277d7a8edb5da656f3c026b2000fb70b99342db37e0a575d9c7a84e"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_21.obj", "e8d12c6e8fb94ccd84fd5b76a3b04908928b3c6de7c7259cc0fd653967bb4fcc"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_22.obj", "a9e4ef9dab34f042557d8e669c4682de04c946dc07cf2f37de8163afa63d5884"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_23.obj", "4ef5187b8ff5b07842ecc69cd9080ad7352c0bf2075006bd60de101e0b5ebdae"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_24.obj", "31b2e54eed17af18f343611e5cbdf3b67307b483e78dbd69aab4f51cceb999cc"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_25.obj", "74fb073a2fe25f91fbd1740990428845166fb78c6acd66aa7697768539a921d6"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_26.obj", "d499f7fa02613219d9ed1615f399c7542694a6e8d5fd3efd0ed630879c2ecfc2"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/bowl/coacd/coacd_convex_piece_27.obj", "ee3234805bb8ee22bb4a7739043c6da3bedabc9e3086c8298309f26203b2a5ed"),
            (ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),
        ),
    ),
    "scissor": ObjectRuntime(
        object_type="scissor",
        link_name="scissor_link",
        body_name="scissor",
        free_joint_name="scissor_free",
        source_mesh_filename="scissor.obj",
        urdf_path=ALL_ASSETS_SIM_ROOT / "decomposed" / "scissor" / "scissor.urdf",
        collision_mesh_paths=tuple(
            ALL_ASSETS_SIM_ROOT
            / "decomposed"
            / "scissor"
            / "coacd"
            / f"coacd_convex_piece_{index}.obj"
            for index in range(9)
        ),
        collision_mesh_scales=((1.0, 1.0, 1.0),) * 9,
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        rgba="0.55 0.60 0.65 1",
        geometry_type="irregular",
        expected_sha256=(
            (ALL_ASSETS_SIM_ROOT / "decomposed/scissor/scissor.urdf", "6980b8e963833555a6624e06e1b777c37f0346e90c3f498065b2e001b26f620f"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/scissor/coacd/coacd_convex_piece_0.obj", "e7bd654a3515e0f9d799393b9f6f9005a09dcadb701cd5a7c17a205715017710"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/scissor/coacd/coacd_convex_piece_1.obj", "fb582ff179f04aec8b780fa6d6d6fb717f61d3dfacdb42d8119bf30bb209424b"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/scissor/coacd/coacd_convex_piece_2.obj", "170373ed324a37d1011e6d210215015f7476d024e2e8b47789dfbb6a78385e50"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/scissor/coacd/coacd_convex_piece_3.obj", "0fb9698be029aca59d44d172a9413f87c20b1ca0b709407f92083304c73651b8"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/scissor/coacd/coacd_convex_piece_4.obj", "d08c25cd00280dc984e3bc85dbfac7cafa432bd8d8fdb63bd6434e04d95ad21b"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/scissor/coacd/coacd_convex_piece_5.obj", "d796db438531a48dce3496a0ebf65f2b5148a7fe4edb10c53a6068dbc8f2c0ba"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/scissor/coacd/coacd_convex_piece_6.obj", "8efcd7dd94a3cdf6df0128e21d1c86a0aa8f04d8b16966128b6832ef3488c68b"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/scissor/coacd/coacd_convex_piece_7.obj", "59d69f7a1785315152cdbdbaa36e03995f0078be792979b2e25bdfa9eec0c45a"),
            (ALL_ASSETS_SIM_ROOT / "decomposed/scissor/coacd/coacd_convex_piece_8.obj", "f6c1a88395f62be299dad78098a717a310ffec6d2e176c180554877bf495ad18"),
            (ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),
        ),
    ),
    "cube2": _all_assets_runtime(
        "cube2",
        urdf_sha256="334fb68ecf7eca5a860b72f10b556d37af6045e18cfc2049f9f2490463abdce2",
        collision_sha256="018616c33d159ca5246da8fdc923579c009b900e56a1997e9e8a34463d069d6f",
        rgba="0.15 0.45 0.85 1",
    ),
    "cylinder1": _all_assets_runtime(
        "cylinder1",
        urdf_sha256="804894a8406a3a9de2cb377af3927a1cdcd9ddd49e26cce4f9f36940833bd37a",
        collision_sha256="104df8077aa9009571f25d5637a2b8aa36fc78e29479c5155aefe60a5c178509",
        rgba="0.62 0.25 0.55 1",
    ),
    "cylinder3": _all_assets_runtime(
        "cylinder3",
        urdf_sha256="0ee0d19fe45f11792172db042d66dc1ae6dad2c9ea3c1dd96669a4ba35974630",
        collision_sha256="a955748b8cc1d85e7c4ca9a63a6ed334558149994cea1d176f1296e51bdfc3ee",
        rgba="0.75 0.22 0.25 1",
    ),
    "cylinder4": _all_assets_runtime(
        "cylinder4",
        urdf_sha256="35e83a639fee1dfdb44c7c77a65955665465afc6955a53e7b5a22e15e9d16fe0",
        collision_sha256="1700c2f9528711ebff23c0d42a311a8d3f00356802f64339340ea1e6db26bfb4",
        rgba="0.30 0.45 0.78 1",
    ),
    "cylinder5": _all_assets_runtime(
        "cylinder5",
        urdf_sha256="a026bf1762370f8471e73d20c8f0ad27e4213a654bb19b7d91197c557f0f1478",
        collision_sha256="8c0cbbe5ac1fcc5827d0983454db0d47e7adcb12b7c0c53f6565b5977cb234f7",
        rgba="0.72 0.54 0.12 1",
    ),
    "cylinder6": _all_assets_runtime(
        "cylinder6",
        urdf_sha256="b1105b23619ce7697cddce6d18ec51ed7740e76e5795a63fcf22d2737f8b4d17",
        collision_sha256="be1134bdce8bc270ec2b5f64d03c7adb650f8c429555c5d465818ecdd671bec8",
        rgba="0.30 0.68 0.32 1",
    ),
    "sphere1": _all_assets_runtime(
        "sphere1",
        urdf_sha256="5a6560e0c32d99c580cda2932e6563edc3406473fc8a0f2f780f8664bc23e983",
        collision_sha256="3ed3410d72eb6ee6915d6530f5544921d32eaa47261bfc20b2d3101b14526c8b",
        rgba="0.85 0.30 0.18 1",
    ),
    "sphere2": _all_assets_runtime(
        "sphere2",
        urdf_sha256="5c934be6a43e7efda8338998aeb8e6dceae1935bfcb74ea5ef2a65aa5a0abf7e",
        collision_sha256="7ab5563b5883a61b9ed55143b07cf582c0b3ab2fdbaea24340320fc81e112460",
        rgba="0.18 0.58 0.82 1",
    ),
    "sphere3": _all_assets_runtime(
        "sphere3",
        urdf_sha256="d79e32663a1f065dd4855297e182b745262615329176e7bb314b5de8b75e2e04",
        collision_sha256="da32bee52e7841bb2e1313ed9752623c351f34e0023d0f7a7a276c5323835601",
        rgba="0.46 0.68 0.18 1",
    ),
    "cuboid3": _all_assets_runtime(
        "cuboid3",
        urdf_sha256="abaad82b63c72fae8df11bfec35c0d4462d55f809f503d9477184c3868a58273",
        collision_sha256="6a8e186b9c6be97b0681815ce58a79e7e6b7414a3ecdf5033dc5fec1dc24cc74",
        rgba="0.40 0.70 0.35 1",
    ),
    "iphone": ObjectRuntime(
        object_type="iphone",
        link_name="iphone17_link",
        body_name="iphone17",
        free_joint_name="iphone17_free",
        source_mesh_filename="iphone17.obj",
        urdf_path=ALL_ASSETS_SIM_ROOT / "mano_objects_urdf" / "iphone17.urdf",
        collision_mesh_paths=(
            ALL_ASSETS_SIM_ROOT / "for_math_retaregeting" / "iphone17" / "coacd" / "coacd_convex_piece_0.obj",
        ),
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        collision_mesh_scales=((1.0, 1.0, 1.0),),
        rgba="0.20 0.20 0.25 1",
        geometry_type="box",
        expected_sha256=(
            (
                ALL_ASSETS_SIM_ROOT / "mano_objects_urdf" / "iphone17.urdf",
                "b6cf7a4bc57db9b6733a3d13c981366a49836f2d850f921ff8066ead0d95ff5d",
            ),
            (
                ALL_ASSETS_SIM_ROOT / "for_math_retaregeting" / "iphone17" / "coacd" / "coacd_convex_piece_0.obj",
                "1643c316fa319b34af9849c2db6250d6d08f2d47c2b30ebc4ea852b19a72a605",
            ),
            (ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),
        ),
    ),
    "bottlewithcap": ObjectRuntime(
        object_type="bottlewithcap",
        link_name="bottlewithcap_link",
        body_name="bottlewithcap",
        free_joint_name="bottlewithcap_free",
        source_mesh_filename="bottlewithcap.obj",
        urdf_path=ALL_ASSETS_SIM_ROOT / "decomposed" / "bottlewithcap" / "bottlewithcap.urdf",
        collision_mesh_paths=tuple(
            ALL_ASSETS_SIM_ROOT
            / "decomposed"
            / "bottlewithcap"
            / "coacd"
            / f"coacd_convex_piece_{index}.obj"
            for index in range(2)
        ),
        collision_mesh_scales=((1.0, 1.0, 1.0),) * 2,
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        rgba="0.32 0.78 0.74 1",
        geometry_type="irregular",
        expected_sha256=(
            (
                ALL_ASSETS_SIM_ROOT / "decomposed" / "bottlewithcap" / "bottlewithcap.urdf",
                "ac23ee51dbd0a03cb0cc84895eeb78c2d7bcc3fd032671c92e36b7bd3396f191",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "bottlewithcap"
                / "coacd"
                / "coacd_convex_piece_0.obj",
                "bec56e9e7984db51cf91b3d73154dcbf7c58ac20aa53a921da45f641ae3abd85",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "bottlewithcap"
                / "coacd"
                / "coacd_convex_piece_1.obj",
                "8575fd39eb2969cdf9c67bc54a2702de77efcc4900a6766afe58e5266b3da424",
            ),
            (ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),
        ),
    ),
    "mayonnaisebottle": ObjectRuntime(
        object_type="mayonnaisebottle",
        link_name="mayonnaisebottle_link",
        body_name="mayonnaisebottle",
        free_joint_name="mayonnaisebottle_free",
        source_mesh_filename="mayonnaisebottle.obj",
        urdf_path=ALL_ASSETS_SIM_ROOT / "decomposed" / "mayonnaisebottle" / "mayonnaisebottle.urdf",
        collision_mesh_paths=tuple(
            ALL_ASSETS_SIM_ROOT
            / "decomposed"
            / "mayonnaisebottle"
            / "coacd"
            / f"coacd_convex_piece_{index}.obj"
            for index in range(2)
        ),
        collision_mesh_scales=((1.0, 1.0, 1.0),) * 2,
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        rgba="0.90 0.85 0.25 1",
        geometry_type="irregular",
        expected_sha256=(
            (
                ALL_ASSETS_SIM_ROOT / "decomposed" / "mayonnaisebottle" / "mayonnaisebottle.urdf",
                "d9de5ce1455d359e6ccbe3f1078d6e70058a1c788dd8c05251be54e2656f7032",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "mayonnaisebottle"
                / "coacd"
                / "coacd_convex_piece_0.obj",
                "d4de932b65ca9dc76d101cf58d00e90e562ef183b9f34096bb44b44813e3d048",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "mayonnaisebottle"
                / "coacd"
                / "coacd_convex_piece_1.obj",
                "5c76fee70df8d3776e828e88cc443498a8cb77dd8582b5c7e0114c949a1e0f6e",
            ),
            (ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),
        ),
    ),
    "largeclamp": ObjectRuntime(
        object_type="largeclamp",
        link_name="largeclamp_link",
        body_name="largeclamp",
        free_joint_name="largeclamp_free",
        source_mesh_filename="largeclamp.obj",
        urdf_path=ALL_ASSETS_SIM_ROOT / "decomposed" / "largeclamp" / "largeclamp.urdf",
        collision_mesh_paths=tuple(
            ALL_ASSETS_SIM_ROOT
            / "decomposed"
            / "largeclamp"
            / "coacd"
            / f"coacd_convex_piece_{index}.obj"
            for index in range(30)
        ),
        collision_mesh_scales=((1.0, 1.0, 1.0),) * 30,
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        rgba="0.85 0.35 0.20 1",
        geometry_type="irregular",
        expected_sha256=(
            (
                ALL_ASSETS_SIM_ROOT / "decomposed" / "largeclamp" / "largeclamp.urdf",
                "3f8286e375e040ff2f7bc94f392c6614ecba9d1dd76e840c3859e540d5c5cc00",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_0.obj",
                "9dd65950c8595804cf7d2366b3cecc94ad3631fc5875a95d5edd7a5040477f06",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_1.obj",
                "8647fe0168774bcb965d26e87881bd3582c655d11769129085627eceec4944aa",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_2.obj",
                "72855d260a3dedceb4cbdc33a1aee3d4ae17d35d66a9962cc718c6e819ee142d",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_3.obj",
                "176e045fd7877efb2f6925dccdaa0a489ed83f86abc2ada91abb5387dac0bc17",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_4.obj",
                "03b90b20aea1a407dfb9f835c6bf1d9e5ff851d4cd029baee466ded0749c682f",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_5.obj",
                "5b6fe2fb054d20ca176e53607c1fc9744c4b20aca003ac1497d536846359564e",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_6.obj",
                "b6b174d212e1d20f8bca0ad2461ae81e1dcfefb2d242c2abc86b185362135c6c",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_7.obj",
                "8ecb92fc36319dea3e87967179794c4362ba9578f720e27c5f3884a046592dd7",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_8.obj",
                "906015e698e822332810062b221cf5f717073d8a84b9cebf503f7c115e8b8269",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_9.obj",
                "a6bda64dd53545404b14b521703317a6b796e977748e610f498539d19147349d",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_10.obj",
                "00848951014e1caf349ae9e817e639722538e39f2a49e8b690ccb1a5e9a63a1a",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_11.obj",
                "9c54b78e9a5fdbac2b0f2c8fd0cf712adacd08f991d46cce011f7ac2d26cf528",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_12.obj",
                "81d5c191631d4d959bd424875b8fdc377d59ff25a76102d8742b541f411c2d7d",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_13.obj",
                "25e6c3d420cdd2040ee2d9f13a8a392c497616913f49da0e87ed5a0dc60c365e",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_14.obj",
                "799e33697a8d8019d606d40df5148e7e7ad696d7e9a44e877293348c112bafa5",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_15.obj",
                "57ea2b5870f390cf6e472fea143646175c9d48c2c0f7e6dde90255de88dd1747",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_16.obj",
                "ae36a041969dd564e1a1c8fbf8073d649c65d847461683a1ca2bd92c149d8dc1",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_17.obj",
                "c81749e71a3c3668d348734fcaacf60b7e1f7a493e914fd76988a6fee6e81321",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_18.obj",
                "d79a3046854fbbe33c2fa0d751a0af9a7dac9891b2ec72448d65a51f0a8a683d",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_19.obj",
                "2ec60921880111afed15261edf7b8101f76975fa1740ec595b9b31e701800a19",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_20.obj",
                "903ece086332918a8240f6a0098a1ed1e74a1bdf2f6ffd86ff8ce28005007b06",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_21.obj",
                "e733c387e9a60a8c3a7acd04cf6a9a8ff0dab4fcddab3cc8f27603efc415f255",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_22.obj",
                "ea80b78712049777409596e5e021d62ab2699b09f5d7d5358788bee5e47076de",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_23.obj",
                "4fc2accb357af94a600af731ae407efc38361d14074302cbd059b19a92bc8b75",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_24.obj",
                "baf8679c2fe508c4a80dff7262a0aba370db1179b1274c9f57df1c0e070877cb",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_25.obj",
                "4c80ab86ff03b68af57334047b5d111eefc9f92da28b38d9dc3a946fbe3bb205",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_26.obj",
                "3765b5fca411fd22398874751a738b2bdb00109e2279287e5f3e1b0e2d3511c6",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_27.obj",
                "d612ea0994f5270622f092c4176fe244407a2091ea43197514572c35a7afdffd",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_28.obj",
                "4ccc0f0b3c40b88f5ca75e6dd78fd5b84d18507f88c5e4441936871509694bc6",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "largeclamp"
                / "coacd"
                / "coacd_convex_piece_29.obj",
                "71791a816ea21b889bc8b725d977ea5ec16a3eb9f26328ec22c2f335ab59919d",
            ),
            (ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),
        ),
    ),
    "powerdrill": ObjectRuntime(
        object_type="powerdrill",
        link_name="powerdrill_link",
        body_name="powerdrill",
        free_joint_name="powerdrill_free",
        source_mesh_filename="powerdrill.obj",
        urdf_path=ALL_ASSETS_SIM_ROOT / "decomposed" / "powerdrill" / "powerdrill.urdf",
        collision_mesh_paths=tuple(
            ALL_ASSETS_SIM_ROOT
            / "decomposed"
            / "powerdrill"
            / "coacd"
            / f"coacd_convex_piece_{index}.obj"
            for index in range(5)
        ),
        collision_mesh_scales=((1.0, 1.0, 1.0),) * 5,
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        rgba="0.25 0.35 0.75 1",
        geometry_type="irregular",
        expected_sha256=(
            (
                ALL_ASSETS_SIM_ROOT / "decomposed" / "powerdrill" / "powerdrill.urdf",
                "34f22cfcb519e342715d349b5184cb0b6a1ec4e440c793347762e25313a36105",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "powerdrill"
                / "coacd"
                / "coacd_convex_piece_0.obj",
                "a413634662c84599566012de30198ed27e2bbc423319e87bd9dca0d80f427bac",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "powerdrill"
                / "coacd"
                / "coacd_convex_piece_1.obj",
                "116aa7a9d602dbeb24b03aee4b51872ba91bbc822b713ac1a870dcc5b1ddf42e",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "powerdrill"
                / "coacd"
                / "coacd_convex_piece_2.obj",
                "0b06e92975f0558941069674ea0140ca5f406e9bd8f43f9d58a36589330b648d",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "powerdrill"
                / "coacd"
                / "coacd_convex_piece_3.obj",
                "4d89292d56a7bae8ee3404cc766ade0101705d8bffca6013ea4dd0f39fd861be",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "powerdrill"
                / "coacd"
                / "coacd_convex_piece_4.obj",
                "a7e8aaf43df5bca66a011825bbe545c8d09c93174758a09a57263befb0dd9c99",
            ),
            (ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),
        ),
    ),
    "banana": ObjectRuntime(
        object_type="banana",
        link_name="banana_link",
        body_name="banana",
        free_joint_name="banana_free",
        source_mesh_filename="banana.obj",
        urdf_path=ALL_ASSETS_SIM_ROOT / "decomposed" / "banana" / "banana.urdf",
        collision_mesh_paths=tuple(
            ALL_ASSETS_SIM_ROOT
            / "decomposed"
            / "banana"
            / "coacd"
            / f"coacd_convex_piece_{index}.obj"
            for index in range(3)
        ),
        grasp_mapping_path=ALL_OBJECT_GRASPS,
        source_mesh_scale=(0.001, 0.001, 0.001),
        collision_mesh_scales=((1.0, 1.0, 1.0),) * 3,
        rgba="0.95 0.78 0.16 1.0",
        geometry_type="irregular",
        expected_sha256=(
            (
                ALL_ASSETS_SIM_ROOT / "mano_objects_urdf" / "banana.urdf",
                "9e57e199b7a8815f267a9e08ee7665e3f87dea4392a782d369ca43bd333be70b",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "mano_assets"
                / "objects"
                / "banana"
                / "banana.obj",
                "2a1eb42fe4ff1330cad77b79453894a3c9d36c30b44ec36fc17fdeb92e13e8bb",
            ),
            (
                ALL_ASSETS_SIM_ROOT / "decomposed" / "banana" / "banana.urdf",
                "58a9ce7603e02d8dc257786e44164c0c1f75d58e4b16f61e0affec8cf49adb50",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "banana"
                / "coacd"
                / "coacd_convex_piece_0.obj",
                "b5e054445e1d454b5b0b457883a33ccd188c0d559e554213c0e2639aa1f3dad5",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "banana"
                / "coacd"
                / "coacd_convex_piece_1.obj",
                "1c54bbb35c33b1bf3f8c2c3da1345913b3d6b17902ff45d570468be851647286",
            ),
            (
                ALL_ASSETS_SIM_ROOT
                / "decomposed"
                / "banana"
                / "coacd"
                / "coacd_convex_piece_2.obj",
                "a2978545c35275ceeacd7b132ba6fd64312517ab7a2aed1d580abe632fa8d706",
            ),
            (ALL_OBJECT_GRASPS, ALL_OBJECT_GRASPS_SHA256),
        ),
    ),
}


def supported_object_types() -> tuple[str, ...]:
    """Return the closed set of object types with complete runtime assets."""

    return tuple(sorted(_OBJECT_RUNTIMES))


def object_runtime(object_type: str = OBJECT_TYPE) -> ObjectRuntime:
    """Resolve a materialized runtime or fail before model compilation."""

    if not isinstance(object_type, str) or object_type not in _OBJECT_RUNTIMES:
        available = ", ".join(supported_object_types())
        raise ValueError(
            f"unsupported ManoRL object runtime {object_type!r}; materialized runtimes: {available}"
        )
    runtime = _OBJECT_RUNTIMES[object_type]
    missing = [
        str(path)
        for path in (
            runtime.urdf_path,
            *runtime.collision_mesh_paths,
            runtime.grasp_mapping_path,
            *(path for path, _ in runtime.expected_sha256),
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"ManoRL object runtime {object_type!r} is registered but not materialized: {missing}"
        )
    return runtime


# Source shape groups name four rigid bodies per finger, but each abduction
# body has no collision shape in the authoritative URDF. These are the three
# actual collision-bearing bodies selected for each source filter group.
HAND_SELF_COLLISION_GROUPS = {
    "thumb": ("thumb_cmc", "thumb_mcp", "thumb_ip"),
    "index": ("index_mcp", "index_pip", "index_dip"),
    "middle": ("middle_mcp", "middle_pip", "middle_dip"),
    "ring": ("ring_mcp", "ring_pip", "ring_dip"),
    "pinky": ("pinky_mcp", "pinky_pip", "pinky_dip"),
}


def _numbers(
    value: str | None, count: int, default: Iterable[float]
) -> tuple[float, ...]:
    values = (
        tuple(default)
        if value is None
        else tuple(float(item) for item in value.split())
    )
    if len(values) != count or not np.all(np.isfinite(values)):
        raise ValueError(f"expected {count} finite values, got {value!r}")
    return values


def _format(values: Iterable[float]) -> str:
    return " ".join(f"{value:.17g}" for value in values)


def _rpy_to_wxyz(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _origin(element: ET.Element | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if element is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    xyz = _numbers(element.get("xyz"), 3, (0.0, 0.0, 0.0))
    rpy = _numbers(element.get("rpy"), 3, (0.0, 0.0, 0.0))
    return xyz, _rpy_to_wxyz(rpy)


def _resolve_mesh_path(
    urdf_path: Path, filename: str, *, fallback_roots: Iterable[Path] = ()
) -> Path:
    """Resolve a URDF mesh, allowing the curated runtime to use external visuals."""

    candidates = [(urdf_path.parent / filename).resolve()]
    basename = Path(filename).name
    candidates.extend((root / basename).resolve() for root in fallback_roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"URDF mesh is absent: {filename!r}; checked "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _visual_rgba(
    visual: ET.Element,
    global_materials: dict[str, str],
    fallback: str,
) -> str:
    material = visual.find("material")
    if material is None:
        return fallback
    color = material.find("color")
    if color is not None and color.get("rgba"):
        return color.get("rgba", fallback)
    material_name = material.get("name")
    return global_materials.get(material_name or "", fallback)


def _required_paths(
    object_type: str | None = None, *, hand_side: str = "right"
) -> tuple[Path, ...]:
    hand_root = hand_asset_root(hand_side)
    hand_meshes = tuple(
        path
        for path in (hand_root / "meshes").glob("*.stl")
        if not path.name.startswith("visual_")
    )
    if len(hand_meshes) != 16:
        raise FileNotFoundError(
            f"expected exactly 16 curated {hand_side}-hand collision meshes, got {len(hand_meshes)}"
        )
    runtimes = (
        (object_runtime(object_type),)
        if object_type is not None
        else tuple(object_runtime(name) for name in supported_object_types())
    )
    object_paths = tuple(
        path
        for runtime in runtimes
        for path in (
            runtime.urdf_path,
            *runtime.collision_mesh_paths,
            runtime.grasp_mapping_path,
            *(path for path, _ in runtime.expected_sha256),
        )
    )
    paths = (
        hand_urdf_path(hand_side),
        hand_root / "metadata.json",
        *hand_meshes,
        ASSET_MANIFEST,
        *object_paths,
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"curated ManoRL assets are incomplete: {missing}")
    return tuple(paths)


def validate_asset_manifest(
    object_type: str | None = None, *, hand_side: str = "right"
) -> dict[str, Any]:
    """Verify every curated file against its committed provenance digest."""

    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
    required_paths = _required_paths(object_type, hand_side=side)
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("source_commit") != "033b358b73c57e5f437f6582b6a9b0d4add7f9ee":
        raise ValueError("asset manifest does not name the 28-DoF all_assets commit")
    entries = manifest.get("files", [])
    if not isinstance(entries, list):
        raise ValueError("asset manifest files must be a list")
    digests_by_path = {entry.get("curated_path"): entry for entry in entries}
    bundles = manifest.get("hand_bundles", {})
    bundle = bundles.get(side) if isinstance(bundles, dict) else None
    if (
        not isinstance(bundle, dict)
        or bundle.get("curated_root") != hand_asset_root(side).name
    ):
        raise ValueError(f"asset manifest has no integrity bundle for {side} hand")
    bundle_rows: list[str] = []
    for bundle_path in sorted(hand_asset_root(side).rglob("*")):
        if bundle_path.is_file():
            bundle_rows.append(
                f"{bundle_path.relative_to(hand_asset_root(side)).as_posix()}:{hashlib.sha256(bundle_path.read_bytes()).hexdigest()}"
            )
    bundle_digest = hashlib.sha256("\n".join(bundle_rows).encode()).hexdigest()
    if bundle_digest != bundle.get("sha256"):
        raise ValueError(f"{side} hand asset bundle digest mismatch: {bundle_digest}")
    runtimes = (
        (object_runtime(object_type),)
        if object_type is not None
        else tuple(object_runtime(name) for name in supported_object_types())
    )
    external_digests = {
        path.resolve(): digest
        for runtime in runtimes
        for path, digest in runtime.expected_sha256
    }
    for path in required_paths:
        if path == ASSET_MANIFEST:
            continue
        try:
            relative = path.relative_to(ASSET_ROOT).as_posix()
        except ValueError:
            expected_digest = external_digests.get(path.resolve())
            if expected_digest is None:
                raise ValueError(f"unpinned runtime asset outside curated root: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_digest:
                raise ValueError(
                    f"pinned all_assets digest mismatch for {path}: {digest}"
                )
        else:
            # Side-specific runtime files are checked against the manifest when
            # entries are present.  Older manifests intentionally omitted the
            # left-hand bundle; its metadata/URDF/mesh digests are still
            # verified by the explicit side manifest entries below.
            if relative not in digests_by_path and not relative.startswith(
                "hand_left/"
            ):
                raise ValueError(f"asset manifest has no digest entry for {relative}")
    for entry in entries:
        path = ASSET_ROOT / entry["curated_path"]
        if not path.is_file():
            raise FileNotFoundError(f"manifest asset is absent: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["curated_sha256"]:
            raise ValueError(f"asset digest mismatch for {path}: {digest}")
    return manifest


def _link_inertial(body: ET.Element, link: ET.Element) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError(f"authoritative link {link.get('name')} has no inertial")
    position, quaternion = _origin(inertial.find("origin"))
    if not np.allclose(quaternion, (1.0, 0.0, 0.0, 0.0), atol=1e-15, rtol=0):
        raise ValueError(
            f"rotated full inertia is unsupported for link {link.get('name')}"
        )
    mass_element = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass_element is None or inertia is None:
        raise ValueError(
            f"authoritative link {link.get('name')} has incomplete inertial"
        )
    full_inertia = (
        float(inertia.get("ixx", "nan")),
        float(inertia.get("iyy", "nan")),
        float(inertia.get("izz", "nan")),
        float(inertia.get("ixy", "nan")),
        float(inertia.get("ixz", "nan")),
        float(inertia.get("iyz", "nan")),
    )
    ET.SubElement(
        body,
        "inertial",
        pos=_format(position),
        mass=mass_element.get("value", ""),
        fullinertia=_format(full_inertia),
    )


def _link_collision(
    body: ET.Element,
    link: ET.Element,
    *,
    hand_contacts_enabled: bool,
    viewer_visuals: bool,
    hand_root: Path,
    name_prefix: str = "",
    asset_prefix: str = "",
) -> None:
    collisions = link.findall("collision")
    if len(collisions) > 1:
        raise ValueError(
            f"hand link {link.get('name')} unexpectedly has multiple collisions"
        )
    if not collisions:
        return
    collision = collisions[0]
    geometry = collision.find("geometry")
    mesh = None if geometry is None else geometry.find("mesh")
    if mesh is None:
        raise ValueError(f"hand collision for {link.get('name')} is not a mesh")
    filename = Path(mesh.get("filename", "")).name
    source_path = hand_root / "meshes" / filename
    if not source_path.is_file():
        raise FileNotFoundError(f"curated hand collision mesh is absent: {source_path}")
    position, quaternion = _origin(collision.find("origin"))
    attributes = {
        "name": f"{name_prefix}{link.get('name')}_collision",
        "type": "mesh",
        "mesh": f"{asset_prefix}hand_{Path(filename).stem}",
        "pos": _format(position),
        "quat": _format(quaternion),
        "rgba": "0.88 0.58 0.46 1",
        # Enabled masks allow palm/cross-finger, hand-object, and hand-floor
        # contacts. Explicit body-pair excludes below disable only collisions
        # within each finger, matching the source disable_within_finger mode.
        # The free-space diagnostic zeros both bits only on hand geoms.
        "contype": "1" if hand_contacts_enabled else "0",
        "conaffinity": "7" if hand_contacts_enabled else "0",
        "condim": "3",
        "friction": "1 0.01 0.001",
    }
    if viewer_visuals:
        attributes["group"] = str(COLLISION_GEOM_GROUP)
    ET.SubElement(body, "geom", attributes)


def _link_visuals(
    body: ET.Element,
    link: ET.Element,
    *,
    global_materials: dict[str, str],
    name_prefix: str = "",
    asset_prefix: str = "",
) -> None:
    visuals = link.findall("visual")
    for index, visual in enumerate(visuals):
        geometry = visual.find("geometry")
        mesh = None if geometry is None else geometry.find("mesh")
        sphere = None if geometry is None else geometry.find("sphere")
        position, quaternion = _origin(visual.find("origin"))
        attributes = {
            "name": f"{name_prefix}{link.get('name')}_visual"
            + (f"_{index}" if index else ""),
            "pos": _format(position),
            "quat": _format(quaternion),
            "rgba": _visual_rgba(visual, global_materials, "0.88 0.58 0.46 1"),
            "contype": "0",
            "conaffinity": "0",
            "group": str(VISUAL_GEOM_GROUP),
        }
        if mesh is not None:
            filename = Path(mesh.get("filename", "")).name
            attributes.update(
                type="mesh",
                mesh=f"{asset_prefix}hand_{Path(filename).stem}_visual",
            )
        elif sphere is not None:
            attributes.update(type="sphere", size=sphere.get("radius", ""))
        else:
            raise ValueError(f"unsupported hand visual geometry for {link.get('name')}")
        ET.SubElement(body, "geom", attributes)


def _hand_tree(
    worldbody: ET.Element,
    asset: ET.Element,
    urdf_root: ET.Element,
    *,
    hand_contacts_enabled: bool,
    visual_meshes: bool = False,
    hand_side: str = "right",
    name_prefix: str = "",
    asset_prefix: str = "",
) -> None:
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=False)
    hand_root = hand_asset_root(side)
    links = {element.get("name", ""): element for element in urdf_root.findall("link")}
    joints = list(urdf_root.findall("joint"))
    children: dict[str, list[ET.Element]] = {}
    child_links: set[str] = set()
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {joint.get('name')} has incomplete topology")
        parent_name, child_name = parent.get("link", ""), child.get("link", "")
        children.setdefault(parent_name, []).append(joint)
        child_links.add(child_name)
    roots = set(links) - child_links
    if roots != {"base_link"}:
        raise ValueError(f"expected base_link as sole hand root, got {roots}")

    mesh_files: list[str] = []
    for link in links.values():
        collision = link.find("collision/geometry/mesh")
        if collision is not None:
            mesh_files.append(Path(collision.get("filename", "")).name)
    if len(mesh_files) != 16 or len(set(mesh_files)) != 16:
        raise ValueError(
            "authoritative hand URDF must reference 16 unique collision meshes"
        )
    for filename in mesh_files:
        attributes = {
            "name": f"{asset_prefix}hand_{Path(filename).stem}",
            "file": str((hand_root / "meshes" / filename).resolve()),
        }
        source_mesh = next(
            link.find("collision/geometry/mesh")
            for link in links.values()
            if link.find("collision/geometry/mesh") is not None
            and Path(link.find("collision/geometry/mesh").get("filename", "")).name
            == filename
        )
        scale = source_mesh.get("scale")
        if scale is not None:
            attributes["scale"] = scale
        ET.SubElement(asset, "mesh", attributes)

    global_materials = {}
    for material in urdf_root.findall("material"):
        color = material.find("color")
        if material.get("name") and color is not None and color.get("rgba"):
            global_materials[material.get("name", "")] = color.get("rgba", "")
    if visual_meshes:
        visual_meshes_by_filename: dict[str, ET.Element] = {}
        for link in links.values():
            for visual_mesh in link.findall("visual/geometry/mesh"):
                filename = Path(visual_mesh.get("filename", "")).name
                if filename in visual_meshes_by_filename:
                    raise ValueError(
                        f"authoritative hand URDF repeats visual mesh {filename!r}"
                    )
                visual_meshes_by_filename[filename] = visual_mesh
        if len(visual_meshes_by_filename) != 16:
            raise ValueError(
                "authoritative hand URDF must reference 16 unique visual meshes"
            )
        for filename, source_mesh in visual_meshes_by_filename.items():
            attributes = {
                "name": f"{asset_prefix}hand_{Path(filename).stem}_visual",
                "file": str(
                    _resolve_mesh_path(
                        hand_urdf_path(side),
                        source_mesh.get("filename", ""),
                        fallback_roots=(HAND_VISUAL_MESH_ROOT,),
                    )
                ),
            }
            scale = source_mesh.get("scale")
            if scale is not None:
                attributes["scale"] = scale
            ET.SubElement(asset, "mesh", attributes)

    def append_link(
        parent_xml: ET.Element, link_name: str, source_joint: ET.Element | None
    ) -> None:
        attributes = {"name": f"{name_prefix}{link_name}", "gravcomp": "1"}
        if source_joint is not None:
            position, quaternion = _origin(source_joint.find("origin"))
            attributes.update(pos=_format(position), quat=_format(quaternion))
        body = ET.SubElement(parent_xml, "body", attributes)
        if source_joint is not None and source_joint.get("type") != "fixed":
            joint_type = source_joint.get("type")
            if joint_type not in {"prismatic", "revolute"}:
                raise ValueError(f"unsupported hand joint type: {joint_type}")
            axis = source_joint.find("axis")
            limit = source_joint.find("limit")
            if axis is None or limit is None:
                raise ValueError(
                    f"joint {source_joint.get('name')} has no axis or limit"
                )
            ET.SubElement(
                body,
                "joint",
                name=f"{name_prefix}{source_joint.get('name', '')}",
                type="slide" if joint_type == "prismatic" else "hinge",
                axis=axis.get("xyz", ""),
                range=f"{limit.get('lower')} {limit.get('upper')}",
                limited="true",
                damping="0",
                frictionloss=str(JOINT_FRICTIONLOSS),
                armature=str(JOINT_ARMATURE),
            )
        _link_inertial(body, links[link_name])
        _link_collision(
            body,
            links[link_name],
            hand_contacts_enabled=hand_contacts_enabled,
            viewer_visuals=visual_meshes,
            hand_root=hand_root,
            name_prefix=name_prefix,
            asset_prefix=asset_prefix,
        )
        if visual_meshes:
            _link_visuals(
                body,
                links[link_name],
                global_materials=global_materials,
                name_prefix=name_prefix,
                asset_prefix=asset_prefix,
            )
        for joint in children.get(link_name, []):
            child = joint.find("child")
            assert child is not None
            append_link(body, child.get("link", ""), joint)

    append_link(worldbody, "base_link", None)


def _add_hand_self_collision_excludes(
    contact: ET.Element, *, name_prefix: str = ""
) -> None:
    for group_name, body_names in HAND_SELF_COLLISION_GROUPS.items():
        for pair_index, (first, second) in enumerate(combinations(body_names, 2)):
            ET.SubElement(
                contact,
                "exclude",
                name=f"{name_prefix}{group_name}_internal_{pair_index}",
                body1=f"{name_prefix}{first}",
                body2=f"{name_prefix}{second}",
            )


def _object_body(
    worldbody: ET.Element,
    asset: ET.Element,
    urdf_root: ET.Element,
    runtime: ObjectRuntime,
    *,
    gravity_compensated: bool = False,
    visual_meshes: bool = False,
) -> None:
    object_link = urdf_root.find("link")
    if object_link is None or object_link.get("name") != runtime.link_name:
        raise ValueError(f"{runtime.object_type} URDF must contain {runtime.link_name}")
    collisions = object_link.findall("collision")
    if len(collisions) != runtime.collision_geom_count:
        raise ValueError(
            f"{runtime.object_type} URDF must contain exactly "
            f"{runtime.collision_geom_count} collision mesh(es)"
        )
    collision_assets: list[tuple[ET.Element, str]] = []
    for collision_index, (collision, collision_path, collision_scale) in enumerate(
        zip(
            collisions,
            runtime.collision_mesh_paths,
            runtime.collision_mesh_scales,
            strict=True,
        )
    ):
        source_mesh = collision.find("geometry/mesh")
        expected_filename = (
            runtime.source_mesh_filename
            if runtime.collision_geom_count == 1
            else collision_path.name
        )
        if (
            source_mesh is None
            or Path(source_mesh.get("filename", "")).name != expected_filename
        ):
            raise ValueError(
                f"{runtime.object_type} URDF collision {collision_index} must reference "
                f"{expected_filename}"
            )
        expected_source_scale = (
            runtime.source_mesh_scale
            if runtime.collision_geom_count == 1
            else collision_scale
        )
        mesh_scale = _numbers(source_mesh.get("scale"), 3, (1.0, 1.0, 1.0))
        if mesh_scale != expected_source_scale:
            raise ValueError(
                f"{runtime.object_type} URDF collision {collision_index} scale must remain "
                f"{expected_source_scale}"
            )
        asset_name = (
            f"{runtime.object_type}_mesh"
            if runtime.collision_geom_count == 1
            else f"{runtime.object_type}_collision_mesh_{collision_index}"
        )
        ET.SubElement(
            asset,
            "mesh",
            name=asset_name,
            file=str(collision_path.resolve()),
            scale=_format(collision_scale),
        )
        collision_assets.append((collision, asset_name))

    body = ET.SubElement(
        worldbody,
        "body",
        name=runtime.body_name,
        # Unified scenes park inactive objects outside the bounded workspace;
        # they retain native gravity so the active object uses the exact same
        # generalized dynamics as the homogeneous model.
        gravcomp="1" if gravity_compensated else "0",
    )
    ET.SubElement(body, "freejoint", name=runtime.free_joint_name)
    _link_inertial(body, object_link)
    for collision_index, (collision, asset_name) in enumerate(collision_assets):
        position, quaternion = _origin(collision.find("origin"))
        geom_name = (
            f"{runtime.object_type}_collision"
            if collision_index == 0
            else f"{runtime.object_type}_piece_{collision_index}_collision"
        )
        collision_geom = ET.SubElement(
            body,
            "geom",
            name=geom_name,
            type="mesh",
            mesh=asset_name,
            pos=_format(position),
            quat=_format(quaternion),
            rgba=runtime.rgba,
            contype="2",
            conaffinity="5",
            condim="3",
            friction="0.9 0.01 0.001",
        )
        if visual_meshes:
            collision_geom.set("group", str(COLLISION_GEOM_GROUP))
    if visual_meshes:
        visuals = object_link.findall("visual")
        if len(visuals) != 1:
            raise ValueError(
                f"{runtime.object_type} URDF must contain exactly one visual mesh"
            )
        visual = visuals[0]
        visual_mesh = visual.find("geometry/mesh")
        if visual_mesh is None:
            raise ValueError(f"{runtime.object_type} URDF visual is not a mesh")
        visual_filename = Path(visual_mesh.get("filename", "")).name
        if visual_filename != runtime.source_mesh_filename:
            raise ValueError(
                f"{runtime.object_type} URDF visual must reference {runtime.source_mesh_filename}"
            )
        visual_path = _resolve_mesh_path(
            runtime.urdf_path,
            visual_mesh.get("filename", ""),
            fallback_roots=(OBJECT_VISUAL_MESH_ROOT / runtime.object_type,),
        )
        visual_scale = _numbers(visual_mesh.get("scale"), 3, runtime.source_mesh_scale)
        ET.SubElement(
            asset,
            "mesh",
            name=f"{runtime.object_type}_visual_mesh",
            file=str(visual_path),
            scale=_format(visual_scale),
        )
        position, quaternion = _origin(visual.find("origin"))
        ET.SubElement(
            body,
            "geom",
            name=f"{runtime.object_type}_visual",
            type="mesh",
            mesh=f"{runtime.object_type}_visual_mesh",
            pos=_format(position),
            quat=_format(quaternion),
            rgba=_visual_rgba(visual, {}, runtime.rgba),
            contype="0",
            conaffinity="0",
            group=str(VISUAL_GEOM_GROUP),
        )


def _add_scene_visual_assets(asset: ET.Element) -> None:
    """Add viewer-only sky and floor assets shared by every scene topology."""

    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.07 0.12 0.18",
        rgb2="0.35 0.48 0.62",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "texture",
        name="floor_checker",
        type="2d",
        builtin="checker",
        rgb1="0.18 0.20 0.22",
        rgb2="0.72 0.74 0.76",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor_checker",
        texture="floor_checker",
        texrepeat="8 8",
        texuniform="true",
        reflectance="0.08",
    )


def build_scene_xml(
    servo: ServoConfig = ServoConfig(),
    *,
    object_type: str = OBJECT_TYPE,
    visual_meshes: bool = False,
    hand_side: str = "right",
    physics_timestep: float = PHYSICS_TIMESTEP,
) -> str:
    """Build one homogeneous scene from a materialized object runtime."""

    runtime = object_runtime(object_type)
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=True)
    scene_sides = ACTION_SIDE_ORDER if side == "both" else (side,)
    for scene_side in scene_sides:
        validate_asset_manifest(object_type, hand_side=scene_side)
    object_root = ET.parse(runtime.urdf_path).getroot()
    root = ET.Element("mujoco", model=f"manorl_{runtime.object_type}_reference")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep=str(physics_timestep),
        gravity="0 0 -9.81",
        integrator="implicitfast",
    )
    asset = ET.SubElement(root, "asset")
    _add_scene_visual_assets(asset)
    worldbody = ET.SubElement(root, "worldbody")
    contact = ET.SubElement(root, "contact")
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        _add_hand_self_collision_excludes(contact, name_prefix=prefix)
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        pos=f"0 0 {FLOOR_TOP_Z}",
        size="1 1 0.01",
        material="floor_checker",
        contype="4",
        conaffinity="1",
        friction="1 0.01 0.001",
    )
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        _hand_tree(
            worldbody,
            asset,
            ET.parse(hand_urdf_path(scene_side)).getroot(),
            hand_contacts_enabled=servo.hand_contacts_enabled,
            visual_meshes=visual_meshes,
            hand_side=scene_side,
            name_prefix=prefix,
            asset_prefix=prefix,
        )
    _object_body(worldbody, asset, object_root, runtime, visual_meshes=visual_meshes)

    actuator = ET.SubElement(root, "actuator")
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        for name, effort, kp, dampratio in zip(
            hand_joint_names(scene_side), EFFORT, servo.kp, servo.dampratio, strict=True
        ):
            ET.SubElement(
                actuator,
                "position",
                name=f"{prefix}{name}",
                joint=f"{prefix}{name}",
                kp=f"{kp:.17g}",
                dampratio=f"{dampratio:.17g}",
                inheritrange="1",
                forcelimited="true",
                forcerange=f"{-effort:.17g} {effort:.17g}",
            )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def build_unified_scene_xml(
    servo: ServoConfig = ServoConfig(),
    *,
    object_types: Iterable[str],
    visual_meshes: bool = False,
    hand_side: str = "right",
    physics_timestep: float = PHYSICS_TIMESTEP,
) -> str:
    """Build one fixed-topology scene containing several real object meshes.

    Each world in an MJX data batch selects one object by placing that object's
    free body in the interaction workspace.  Inactive objects are parked far
    above the floor and retain their native gravity/contact properties; their
    trajectories cannot interact with the active workspace during a bounded
    episode.  No collision geometry is approximated or replaced.
    """

    names = tuple(dict.fromkeys(object_types))
    if not names:
        raise ValueError("unified scene requires at least one object type")
    runtimes = tuple(object_runtime(name) for name in names)
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=True)
    scene_sides = ACTION_SIDE_ORDER if side == "both" else (side,)
    for name in names:
        for scene_side in scene_sides:
            validate_asset_manifest(name, hand_side=scene_side)
    root = ET.Element("mujoco", model="manorl_unified")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep=str(physics_timestep),
        gravity="0 0 -9.81",
        integrator="implicitfast",
    )
    asset = ET.SubElement(root, "asset")
    _add_scene_visual_assets(asset)
    worldbody = ET.SubElement(root, "worldbody")
    contact = ET.SubElement(root, "contact")
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        _add_hand_self_collision_excludes(contact, name_prefix=prefix)
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        pos=f"0 0 {FLOOR_TOP_Z}",
        size="1 1 0.01",
        material="floor_checker",
        contype="4",
        conaffinity="1",
        friction="1 0.01 0.001",
    )
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        _hand_tree(
            worldbody,
            asset,
            ET.parse(hand_urdf_path(scene_side)).getroot(),
            hand_contacts_enabled=servo.hand_contacts_enabled,
            visual_meshes=visual_meshes,
            hand_side=scene_side,
            name_prefix=prefix,
            asset_prefix=prefix,
        )
    for runtime in runtimes:
        object_root = ET.parse(runtime.urdf_path).getroot()
        _object_body(
            worldbody,
            asset,
            object_root,
            runtime,
            visual_meshes=visual_meshes,
        )

    actuator = ET.SubElement(root, "actuator")
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        for name, effort, kp, dampratio in zip(
            hand_joint_names(scene_side), EFFORT, servo.kp, servo.dampratio, strict=True
        ):
            ET.SubElement(
                actuator,
                "position",
                name=f"{prefix}{name}",
                joint=f"{prefix}{name}",
                kp=f"{kp:.17g}",
                dampratio=f"{dampratio:.17g}",
                inheritrange="1",
                forcelimited="true",
                forcerange=f"{-effort:.17g} {effort:.17g}",
            )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _model_names(
    mujoco: Any, model: Any, object_type: Any, count: int
) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)
    )


def _is_collision_geom(mujoco: Any, model: Any, geom_id: int) -> bool:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    return name.endswith("_collision")


def validate_compiled_model(
    mujoco: Any,
    model: Any,
    servo: ServoConfig = ServoConfig(),
    *,
    object_type: str = OBJECT_TYPE,
    hand_side: str = "right",
    physics_timestep: float = PHYSICS_TIMESTEP,
) -> None:
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=True)
    scene_sides = ACTION_SIDE_ORDER if side == "both" else (side,)
    runtime = object_runtime(object_type)
    expected_joint_names = tuple(
        f"{scene_side}_{name}" if len(scene_sides) > 1 else name
        for scene_side in scene_sides
        for name in hand_joint_names(scene_side)
    )
    joint_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    if joint_names[: len(expected_joint_names)] != expected_joint_names or joint_names[
        len(expected_joint_names) :
    ] != (runtime.free_joint_name,):
        raise ValueError(f"compiled joint order mismatch: {joint_names}")
    actuator_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    if actuator_names != expected_joint_names:
        raise ValueError(f"compiled actuator order mismatch: {actuator_names}")
    expected_dof = len(expected_joint_names)
    if (
        model.nu != expected_dof
        or model.nq != expected_dof + 7
        or model.nv != expected_dof + 6
    ):
        raise ValueError(
            f"compiled dimensions mismatch: nq={model.nq}, nv={model.nv}, nu={model.nu}"
        )
    if not np.isclose(model.opt.timestep, physics_timestep):
        raise ValueError(f"compiled timestep mismatch: {model.opt.timestep}")
    for actuator_id, (name, effort, kp, dampratio) in enumerate(
        zip(
            expected_joint_names,
            tuple(EFFORT) * len(scene_sides),
            tuple(servo.kp) * len(scene_sides),
            tuple(servo.dampratio) * len(scene_sides),
            strict=True,
        )
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_address = int(model.jnt_dofadr[joint_id])
        if not np.isclose(model.dof_frictionloss[dof_address], JOINT_FRICTIONLOSS):
            raise ValueError(f"compiled frictionloss mismatch for {name}")
        if not np.isclose(model.dof_armature[dof_address], JOINT_ARMATURE):
            raise ValueError(f"compiled armature mismatch for {name}")
        if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise ValueError(f"actuator {name} is attached to the wrong joint")
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError(f"actuator {name} must use a direct joint transmission")
        if not model.actuator_ctrllimited[actuator_id] or not np.allclose(
            model.actuator_ctrlrange[actuator_id], model.jnt_range[joint_id]
        ):
            raise ValueError(f"compiled actuator control range mismatch for {name}")
        if not model.actuator_forcelimited[actuator_id] or not np.allclose(
            model.actuator_forcerange[actuator_id], (-effort, effort)
        ):
            raise ValueError(f"compiled actuator effort limit mismatch for {name}")
        if not np.isclose(model.actuator_gainprm[actuator_id, 0], kp):
            raise ValueError(f"compiled position gain mismatch for {name}")
        if not np.isclose(model.actuator_biasprm[actuator_id, 1], -kp):
            raise ValueError(f"compiled position bias mismatch for {name}")
        kv = -float(model.actuator_biasprm[actuator_id, 2])
        if not np.isfinite(kv) or kv <= 0:
            raise ValueError(
                f"compiled dampratio did not produce positive damping for {name}"
            )
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.body_name)
    if object_id < 0:
        raise ValueError(f"compiled object body is absent: {runtime.body_name}")
    if model.body_gravcomp[object_id] != 0:
        raise ValueError(f"{runtime.object_type} gravity must remain active")
    object_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == object_id
        and _is_collision_geom(mujoco, model, geom_id)
    ]
    if len(object_geom_ids) != runtime.collision_geom_count:
        raise ValueError(
            f"expected {runtime.collision_geom_count} {runtime.object_type} collision geoms, "
            f"got {len(object_geom_ids)}"
        )
    if not np.all(model.geom_contype[object_geom_ids] == 2) or not np.all(
        model.geom_conaffinity[object_geom_ids] == 5
    ):
        raise ValueError(
            "compiled object collision masks mismatch source configuration"
        )
    hand_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] not in (0, object_id)
        and _is_collision_geom(mujoco, model, geom_id)
    ]
    if len(hand_geom_ids) != 16 * len(scene_sides):
        raise ValueError(
            f"expected {16 * len(scene_sides)} hand collision geoms, got {len(hand_geom_ids)}"
        )
    expected_hand_bits = (1, 7) if servo.hand_contacts_enabled else (0, 0)
    if not np.all(
        model.geom_contype[hand_geom_ids] == expected_hand_bits[0]
    ) or not np.all(model.geom_conaffinity[hand_geom_ids] == expected_hand_bits[1]):
        raise ValueError("compiled hand collision masks mismatch servo configuration")
    expected_exclude_signatures = {
        (min(first_id, second_id) << 16) + max(first_id, second_id)
        for scene_side in scene_sides
        for prefix in [f"{scene_side}_" if len(scene_sides) > 1 else ""]
        for body_names in HAND_SELF_COLLISION_GROUPS.values()
        for first_name, second_name in combinations(body_names, 2)
        for first_id, second_id in [
            (
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{first_name}"
                ),
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{second_name}"
                ),
            )
        ]
    }
    actual_exclude_signatures = {int(value) for value in model.exclude_signature}
    if actual_exclude_signatures != expected_exclude_signatures:
        raise ValueError(
            "compiled within-finger collision excludes mismatch source groups"
        )
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name != runtime.body_name and model.body_gravcomp[body_id] != 1:
            raise ValueError(f"hand body gravity compensation missing: {name}")


def compile_model(
    servo: ServoConfig = ServoConfig(),
    *,
    object_type: str = OBJECT_TYPE,
    visual_meshes: bool = False,
    hand_side: str = "right",
    physics_timestep: float = PHYSICS_TIMESTEP,
) -> tuple[Any, Any]:
    """Compile and validate one bounded native-servo scene."""

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("mujoco is required to compile the ManoRL scene") from exc
    model = mujoco.MjModel.from_xml_string(
        build_scene_xml(
            servo,
            object_type=object_type,
            visual_meshes=visual_meshes,
            hand_side=hand_side,
            physics_timestep=physics_timestep,
        )
    )
    validate_compiled_model(
        mujoco,
        model,
        servo,
        object_type=object_type,
        hand_side=hand_side,
        physics_timestep=physics_timestep,
    )
    validate_static_fk(mujoco, model, object_type=object_type, hand_side=hand_side)
    return mujoco, model


def validate_unified_compiled_model(
    mujoco: Any,
    model: Any,
    servo: ServoConfig = ServoConfig(),
    *,
    object_types: Iterable[str],
    hand_side: str = "right",
    physics_timestep: float = PHYSICS_TIMESTEP,
) -> None:
    """Validate the fixed hand topology and every real object in a superset scene."""

    names = tuple(dict.fromkeys(object_types))
    if not names:
        raise ValueError("unified model validation requires at least one object type")
    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=True)
    scene_sides = ACTION_SIDE_ORDER if side == "both" else (side,)
    runtimes = tuple(object_runtime(name) for name in names)
    expected_hand_joint_names = tuple(
        f"{scene_side}_{name}" if len(scene_sides) > 1 else name
        for scene_side in scene_sides
        for name in hand_joint_names(scene_side)
    )
    joint_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    expected_joints = expected_hand_joint_names + tuple(
        runtime.free_joint_name for runtime in runtimes
    )
    if joint_names != expected_joints:
        raise ValueError(f"unified joint order mismatch: {joint_names}")
    actuator_names = _model_names(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    if actuator_names != expected_hand_joint_names:
        raise ValueError(f"unified actuator order mismatch: {actuator_names}")
    expected_dof = len(expected_hand_joint_names)
    expected_nq = expected_dof + 7 * len(runtimes)
    expected_nv = expected_dof + 6 * len(runtimes)
    if model.nu != expected_dof or model.nq != expected_nq or model.nv != expected_nv:
        raise ValueError(
            f"unified dimensions mismatch: nq={model.nq}, nv={model.nv}, nu={model.nu}; "
            f"expected nq={expected_nq}, nv={expected_nv}, nu={expected_dof}"
        )
    if not np.isclose(model.opt.timestep, physics_timestep):
        raise ValueError(f"unified timestep mismatch: {model.opt.timestep}")
    for actuator_id, (name, effort, kp, dampratio) in enumerate(
        zip(
            expected_hand_joint_names,
            tuple(EFFORT) * len(scene_sides),
            tuple(servo.kp) * len(scene_sides),
            tuple(servo.dampratio) * len(scene_sides),
            strict=True,
        )
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_address = int(model.jnt_dofadr[joint_id])
        if not np.isclose(model.dof_frictionloss[dof_address], JOINT_FRICTIONLOSS):
            raise ValueError(f"compiled frictionloss mismatch for {name}")
        if not np.isclose(model.dof_armature[dof_address], JOINT_ARMATURE):
            raise ValueError(f"compiled armature mismatch for {name}")
        if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise ValueError(f"actuator {name} is attached to the wrong joint")
        if not model.actuator_ctrllimited[actuator_id] or not np.allclose(
            model.actuator_ctrlrange[actuator_id], model.jnt_range[joint_id]
        ):
            raise ValueError(f"compiled actuator control range mismatch for {name}")
        if not model.actuator_forcelimited[actuator_id] or not np.allclose(
            model.actuator_forcerange[actuator_id], (-effort, effort)
        ):
            raise ValueError(f"compiled actuator effort limit mismatch for {name}")
        if not np.isclose(model.actuator_gainprm[actuator_id, 0], kp):
            raise ValueError(f"compiled position gain mismatch for {name}")
        if not np.isclose(model.actuator_biasprm[actuator_id, 1], -kp):
            raise ValueError(f"compiled position bias mismatch for {name}")
        kv = -float(model.actuator_biasprm[actuator_id, 2])
        if not np.isfinite(kv) or kv <= 0:
            raise ValueError(
                f"compiled dampratio did not produce positive damping for {name}"
            )
    hand_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id]
        not in {
            0,
            *(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.body_name)
                for runtime in runtimes
            ),
        }
        and _is_collision_geom(mujoco, model, geom_id)
    ]
    if len(hand_geom_ids) != 16 * len(scene_sides):
        raise ValueError(
            f"expected {16 * len(scene_sides)} hand collision geoms, got {len(hand_geom_ids)}"
        )
    expected_hand_bits = (1, 7) if servo.hand_contacts_enabled else (0, 0)
    if not np.all(
        model.geom_contype[hand_geom_ids] == expected_hand_bits[0]
    ) or not np.all(model.geom_conaffinity[hand_geom_ids] == expected_hand_bits[1]):
        raise ValueError("compiled hand collision masks mismatch servo configuration")
    for runtime in runtimes:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.body_name)
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, runtime.free_joint_name
        )
        if body_id < 0 or joint_id < 0:
            raise ValueError(
                f"compiled unified object is absent: {runtime.object_type}"
            )
        if model.body_gravcomp[body_id] != 0:
            raise ValueError(
                f"unified object {runtime.object_type} must retain native gravity"
            )
        object_geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
            and _is_collision_geom(mujoco, model, geom_id)
        ]
        if len(object_geom_ids) != runtime.collision_geom_count:
            raise ValueError(
                f"expected {runtime.collision_geom_count} {runtime.object_type} collision geoms, "
                f"got {len(object_geom_ids)}"
            )
        if not np.all(model.geom_contype[object_geom_ids] == 2) or not np.all(
            model.geom_conaffinity[object_geom_ids] == 5
        ):
            raise ValueError(f"compiled {runtime.object_type} collision masks mismatch")
    validate_static_fk(mujoco, model, object_type=names[0], hand_side=side)


def compile_unified_model(
    servo: ServoConfig = ServoConfig(),
    *,
    object_types: Iterable[str],
    visual_meshes: bool = False,
    hand_side: str = "right",
    physics_timestep: float = PHYSICS_TIMESTEP,
) -> tuple[Any, Any]:
    """Compile one fixed-topology model containing the requested real objects."""

    names = tuple(dict.fromkeys(object_types))
    if not names:
        raise ValueError("unified model requires at least one object type")
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("mujoco is required to compile the ManoRL scene") from exc
    model = mujoco.MjModel.from_xml_string(
        build_unified_scene_xml(
            servo,
            object_types=names,
            visual_meshes=visual_meshes,
            hand_side=hand_side,
            physics_timestep=physics_timestep,
        )
    )
    validate_unified_compiled_model(
        mujoco,
        model,
        servo,
        object_types=names,
        hand_side=hand_side,
        physics_timestep=physics_timestep,
    )
    return mujoco, model


def _matrix_from_pose(
    position: tuple[float, ...], quaternion_wxyz: tuple[float, ...]
) -> NDArray[np.float64]:
    w, x, y, z = quaternion_wxyz
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def urdf_zero_fk(hand_side: str = "right") -> dict[str, NDArray[np.float64]]:
    """Compute zero-joint link transforms independently from the MJCF builder."""

    root = ET.parse(hand_urdf_path(hand_side)).getroot()
    transforms = {"base_link": np.eye(4, dtype=np.float64)}
    remaining = list(root.findall("joint"))
    while remaining:
        progressed = False
        for joint in remaining.copy():
            parent = joint.find("parent")
            child = joint.find("child")
            assert parent is not None and child is not None
            parent_name = parent.get("link", "")
            if parent_name not in transforms:
                continue
            position, quaternion = _origin(joint.find("origin"))
            transforms[child.get("link", "")] = transforms[
                parent_name
            ] @ _matrix_from_pose(position, quaternion)
            remaining.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("hand URDF joint graph is disconnected or cyclic")
    return transforms


def validate_static_fk(
    mujoco: Any,
    model: Any,
    *,
    object_type: str = OBJECT_TYPE,
    hand_side: str = "right",
    atol: float = 1e-10,
) -> None:
    """Check compiled body frames against independent zero-pose URDF FK."""

    side = normalize_hand_side(hand_side, allow_auto=False, allow_both=True)
    scene_sides = ACTION_SIDE_ORDER if side == "both" else (side,)
    data = mujoco.MjData(model)
    hand_width = sum(len(hand_joint_names(scene_side)) for scene_side in scene_sides)
    data.qpos[:hand_width] = 0.0
    runtime = object_runtime(object_type)
    object_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, runtime.free_joint_name
    )
    object_qpos_adr = model.jnt_qposadr[object_joint]
    data.qpos[object_qpos_adr + 3] = 1.0
    mujoco.mj_forward(model, data)
    for scene_side in scene_sides:
        prefix = f"{scene_side}_" if len(scene_sides) > 1 else ""
        for name, expected in urdf_zero_fk(scene_side).items():
            compiled_name = f"{prefix}{name}"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, compiled_name)
            if body_id < 0:
                raise ValueError(f"compiled hand body is absent: {compiled_name}")
            if not np.allclose(data.xpos[body_id], expected[:3, 3], atol=atol, rtol=0):
                raise ValueError(
                    f"compiled static FK position mismatch at {compiled_name}"
                )
            if not np.allclose(
                data.xmat[body_id].reshape(3, 3), expected[:3, :3], atol=atol, rtol=0
            ):
                raise ValueError(
                    f"compiled static FK orientation mismatch at {compiled_name}"
                )


@lru_cache(maxsize=None)
def object_collision_vertices(object_type: str = OBJECT_TYPE) -> NDArray[np.float64]:
    """Return curated collision triangles in object-local metric coordinates."""

    runtime = object_runtime(object_type)
    validate_asset_manifest(object_type)
    pieces = [
        _collision_mesh_vertices(object_type, path, scale)
        for path, scale in zip(
            runtime.collision_mesh_paths,
            runtime.collision_mesh_scales,
            strict=True,
        )
    ]
    result = np.concatenate(pieces, axis=0)
    if (
        result.ndim != 2
        or result.shape[1] != 3
        or len(result) < 12
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"unexpected {object_type} collision vertices: {result.shape}")
    result.setflags(write=False)
    return result


def _collision_mesh_vertices(
    object_type: str,
    path: Path,
    scale: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Decode one digest-checked collision piece into metric triangles."""

    if path.suffix.lower() == ".stl":
        payload = path.read_bytes()
        if len(payload) < 84:
            raise ValueError(f"{object_type} collision STL is too short")
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        expected_length = 84 + triangle_count * 50
        if len(payload) != expected_length:
            raise ValueError(
                f"{object_type} collision STL has an invalid binary length"
            )
        triangles = np.frombuffer(
            payload,
            dtype=np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]
            ),
            count=triangle_count,
            offset=84,
        )
        result = np.asarray(triangles["vertices"], dtype=np.float64).reshape(-1, 3)
    elif path.suffix.lower() == ".obj":
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts or parts[0] == "#":
                continue
            if parts[0] == "v" and len(parts) == 4:
                vertices.append(tuple(float(value) for value in parts[1:4]))
            elif parts[0] == "f" and len(parts) == 4:
                faces.append(
                    tuple(
                        int(value.split("/", maxsplit=1)[0]) - 1 for value in parts[1:4]
                    )
                )
        vertex_array = np.asarray(vertices, dtype=np.float64)
        face_array = np.asarray(faces, dtype=np.int64)
        if (
            face_array.ndim != 2
            or face_array.shape[1:] != (3,)
            or np.any(face_array < 0)
            or np.any(face_array >= len(vertex_array))
        ):
            raise ValueError(
                f"{object_type} collision OBJ must contain indexed triangles"
            )
        result = vertex_array[face_array].reshape(-1, 3)
    else:
        raise ValueError(
            f"unsupported collision mesh format for {object_type}: {path.suffix}"
        )
    result = result * np.asarray(scale, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[1] != 3
        or len(result) < 12
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(
            f"unexpected {object_type} collision piece vertices: {result.shape}"
        )
    return result
