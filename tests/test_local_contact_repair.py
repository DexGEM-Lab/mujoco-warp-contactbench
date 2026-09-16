"""Focused contracts for physical-time resampling and local command edits."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.manorl.local_contact_repair import (
    ReplayInput, edit_targets, finger_control, interpolate, load_catalog, resample_hand, sha, smooth_envelope,
)


def make_input():
    time = np.arange(241) / 120
    desired = np.zeros((len(time), 28))
    return ReplayInput("example", 120, {"desired": desired, "time": time}, {}, {}, {}, {})


def test_physical_time_and_angle_unwrap():
    old = np.arange(101) / 100
    new = np.arange(121) / 120
    hand = np.zeros((101, 28))
    hand[:, 0] = old * .1
    hand[:, 3] = np.arctan2(np.sin(3.0 + old), np.cos(3.0 + old))
    out = resample_hand(hand, old, new)
    np.testing.assert_allclose(out[:, 0], new * .1, atol=1e-15)
    np.testing.assert_allclose(out[:, 3], 3.0 + new, atol=1e-15)
    np.testing.assert_allclose(out[::6], np.column_stack((hand[::5, :3], np.unwrap(hand[::5, 3:], axis=0))), atol=1e-15)


def test_smooth_time_window_and_endpoint_hold():
    time = np.arange(241) / 120
    value = smooth_envelope(time, {"start_s": .2, "rise_s": .2, "end_s": 1.8, "fall_s": .2})
    assert value[24] == 0 and value[48] == 1
    assert value[192] == 1 and value[216] == 0
    assert np.all(value[:24] == 0) and np.all(value[216:] == 0)
    held = smooth_envelope(time, {"start_s": .2, "rise_s": .2})
    assert held[-1] == 1
    old = np.arange(104) / 100
    new = np.arange(int(np.ceil(old[-1] * 120)) + 1) / 120
    value = interpolate(old, old, new)
    assert value[-1] == old[-1]
    assert 0 <= new[-1] - old[-1] < 1 / 120


def test_identity_and_selected_finger_edit():
    inp = make_input()
    names = tuple(f"w{i}" for i in range(6)) + tuple(f"f{i}" for i in range(22))
    recipe = dict(schema="manorl.local-contact-repair.v1", row_id="example", phases=[])
    desired, delta, info = edit_targets(inp, recipe, names)
    np.testing.assert_array_equal(desired, inp.arrays["desired"])
    assert not delta.any() and info["finger_target_max_deg"] == 0
    recipe["phases"] = [dict(start_s=.5, rise_s=.2, end_s=1.8, fall_s=.2,
                             finger_deg={"f1": 1.5}, translation_m=[.001, 0, 0])]
    desired, delta, info = edit_targets(inp, recipe, names)
    np.testing.assert_array_equal(desired[:61], inp.arrays["desired"][:61])
    np.testing.assert_array_equal(delta[:61], 0)
    assert np.isclose(delta[:, 1].max(), np.deg2rad(1.5))
    assert not delta[:, [0, *range(2, 22)]].any()
    assert np.isclose(info["wrist_translation_max_mm"], 1)
    assert not delta[-1].any() and not desired[-1].any()


def test_rotation_is_geometric_and_bounded():
    inp = make_input()
    inp.arrays["desired"][:, 3:6] = [.2, .3, .4]
    names = tuple(f"j{i}" for i in range(28))
    recipe = dict(schema="manorl.local-contact-repair.v1", row_id="example", phases=[
        dict(start_s=.3, rotation_deg=[0, 0, 2])])
    desired, _, info = edit_targets(inp, recipe, names)
    expected = Rotation.from_rotvec([0, 0, np.deg2rad(2)]) * Rotation.from_euler("XYZ", [.2, .3, .4])
    assert (expected.inv() * Rotation.from_euler("XYZ", desired[-1, 3:6])).magnitude() < 1e-12
    assert np.isclose(info["wrist_rotation_max_deg"], 2)
    recipe["phases"][0]["rotation_deg"] = [0, 0, 11]
    with pytest.raises(ValueError, match="envelope"):
        edit_targets(inp, recipe, names)


@pytest.mark.parametrize("phase", [
    {"start_s": -.1}, {"start_s": .3, "rise_s": 0},
    {"start_s": .3, "end_s": .4}, {"start_s": .3, "unknown": 1},
])
def test_bad_time_windows_fail_explicitly(phase):
    with pytest.raises(ValueError):
        smooth_envelope(np.arange(241) / 120, phase)


def test_finger_correction_survives_absolute_preload_once():
    model = SimpleNamespace(jnt_range=np.tile([-2., 2.], (28, 1)),
                            actuator_biasprm=np.zeros((28, 10)),
                            actuator_gainprm=np.ones((28, 10)),
                            dof_frictionloss=np.zeros(28))
    model.actuator_biasprm[:, 2] = -.1
    for weight, expected in ((0., .4), (.5, .6), (1., .8)):
        ctrl = finger_control(model, np.zeros(28), np.full(28, .2), np.ones(22),
                              1., weight, np.full(22, .7), np.full(22, .1), True)
        np.testing.assert_allclose(ctrl[6:], expected)
        np.testing.assert_allclose(ctrl[:6], .2)


def test_recipe_identity_rejects_another_capture():
    inp = make_input()
    inp.manifest = {"source": {"uuid": "capture-a"}}
    inp.provenance = {"baseline_trace_sha256": "trace-a"}
    names = tuple(f"j{i}" for i in range(28))
    recipe = dict(schema="manorl.local-contact-repair.v1", row_id="example", phases=[],
                  source_uuid="capture-b", baseline_trace_sha256="trace-a")
    with pytest.raises(ValueError, match="UUID"):
        edit_targets(inp, recipe, names)
    recipe.update(source_uuid="capture-a", baseline_trace_sha256="trace-b")
    with pytest.raises(ValueError, match="trace hash"):
        edit_targets(inp, recipe, names)
    recipe["baseline_trace_sha256"] = "trace-a"
    edit_targets(inp, recipe, names)


def test_catalog_is_complete_and_clock_bound(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "comparison.json").write_text(json.dumps({"rows": [{"id": "a"}, {"id": "b"}]}))
    recipe = tmp_path / "a.json"
    recipe.write_text("{}")
    catalog = dict(schema="manorl.local-contact-repair-catalog.v1", hand="cheyingtong",
                   control_hz=120, physics_hz=480, baseline_comparison_sha256=sha(baseline / "comparison.json"),
                   rows=[dict(row_id="a", recipe="a.json"), dict(row_id="b", recipe=None)])
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    assert load_catalog(path, baseline, 120) == {"a": recipe, "b": None}
    with pytest.raises(ValueError, match="clock"):
        load_catalog(path, baseline, 100)
    catalog["rows"].pop()
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="exactly"):
        load_catalog(path, baseline, 120)
    catalog["rows"].append(dict(row_id="a", recipe=None))
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="duplicate"):
        load_catalog(path, baseline, 120)


def test_finger_offset_must_not_name_wrist():
    inp = make_input()
    names = tuple(f"j{i}" for i in range(28))
    recipe = dict(schema="manorl.local-contact-repair.v1", row_id="example", phases=[
        dict(start_s=.3, finger_deg={"j1": 1})])
    with pytest.raises(ValueError, match="finger"):
        edit_targets(inp, recipe, names)
