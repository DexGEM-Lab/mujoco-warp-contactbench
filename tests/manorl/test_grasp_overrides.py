from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from sim.manorl.contracts import KEYPOINT_NAMES
from sim.manorl.environment import _expected_keypoint_ids


_CURATED_GRASP_EXPECTATIONS = (
    ("cylinder4", "14", ("thumb_ip", "index_dip", "middle_dip", "ring_dip")),
    ("cylinder4", "16", ("index_pip", "index_dip", "middle_pip", "middle_dip")),
    ("iphone", "02", ("thumb_ip", "index_dip", "middle_dip")),
    ("iphone", "03", ("thumb_ip", "index_dip", "middle_dip", "ring_dip")),
    (
        "iphone",
        "04",
        ("thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip"),
    ),
    (
        "iphone",
        "08",
        ("thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip"),
    ),
    (
        "iphone",
        "14",
        ("thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip"),
    ),
    (
        "iphone",
        "15",
        ("thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip"),
    ),
    ("iphone", "18", ("thumb_ip", "index_dip")),
    (
        "largeclamp",
        "12",
        ("thumb_ip", "index_dip", "middle_dip", "ring_dip", "pinky_dip"),
    ),
    ("powerdrill", "18", ("thumb_ip", "index_dip")),
)


def _install_mapping(tmp_path, monkeypatch, payload: dict[str, object]) -> None:
    mapping = tmp_path / "grasp_mapping.yaml"
    mapping.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "sim.manorl.environment.object_runtime",
        lambda _: SimpleNamespace(grasp_mapping_path=mapping),
    )


@pytest.mark.parametrize(
    ("object_type", "action_id", "expected_names"), _CURATED_GRASP_EXPECTATIONS
)
def test_curated_new_capture_grasp_pairs_resolve_exact_keypoints(
    tmp_path,
    monkeypatch,
    object_type: str,
    action_id: str,
    expected_names: tuple[str, ...],
) -> None:
    _install_mapping(tmp_path, monkeypatch, {object_type: {}})

    actual = _expected_keypoint_ids(object_type, action_id)
    expected = np.asarray(
        [KEYPOINT_NAMES.index(name) for name in expected_names], dtype=np.int64
    )
    np.testing.assert_array_equal(actual, expected)


def test_source_grasp_mapping_remains_authoritative_over_curated_pair(
    tmp_path, monkeypatch
) -> None:
    _install_mapping(
        tmp_path,
        monkeypatch,
        {"iphone": {"02": ["thumb3", "index3"]}},
    )

    np.testing.assert_array_equal(
        _expected_keypoint_ids("iphone", "02"),
        [KEYPOINT_NAMES.index("thumb_ip"), KEYPOINT_NAMES.index("index_dip")],
    )


def test_unknown_grasp_pair_still_fails_without_source_or_curated_mapping(
    tmp_path, monkeypatch
) -> None:
    _install_mapping(tmp_path, monkeypatch, {"iphone": {}})

    with pytest.raises(ValueError, match="no source grasp mapping"):
        _expected_keypoint_ids("iphone", "07")
