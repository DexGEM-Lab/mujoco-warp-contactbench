import numpy as np
import pytest

from tools.run_uniform_direct_parent_canary import target_stream


def baseline(n=5):
    qpos = np.arange(n * 35, dtype=np.float64).reshape(n, 35) / 100.0
    ctrl = np.arange(n * 28, dtype=np.float64).reshape(n, 28) / 10.0
    sub = np.arange((n - 1) * 4 * 28, dtype=np.float64).reshape(n - 1, 4, 28) / 1000.0
    return {"qpos": qpos, "ctrl": ctrl, "ctrl_substeps": sub}


@pytest.mark.parametrize("kind,index", [("first", 0), ("last", 3)])
def test_substep_endpoint_targets(kind, index):
    source = baseline()
    out = target_stream(source, kind)
    np.testing.assert_array_equal(out[0], source["ctrl"][0])
    np.testing.assert_array_equal(out[1:], source["ctrl_substeps"][:, index])


def test_mean_substep_target():
    source = baseline()
    out = target_stream(source, "mean")
    np.testing.assert_array_equal(out[1:], source["ctrl_substeps"].mean(axis=1))


def test_teacher_targets_and_one_frame_lead():
    source = baseline()
    teacher = source["qpos"][:, :28]
    direct = target_stream(source, "teacher")
    lead = target_stream(source, "teacher-lead1")
    np.testing.assert_array_equal(direct[1:], teacher[1:])
    np.testing.assert_array_equal(lead[1:-1], teacher[2:])
    np.testing.assert_array_equal(lead[-1], teacher[-1])


def test_rejects_bad_substep_shape():
    source = baseline()
    source["ctrl_substeps"] = source["ctrl_substeps"][:-1]
    with pytest.raises(ValueError, match="unexpected substep controls"):
        target_stream(source, "mean")
