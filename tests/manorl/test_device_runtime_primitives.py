from __future__ import annotations

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from sim.manorl.device_runtime import (
    jax_to_torch_cuda,
    reduce_warp_contacts,
    torch_to_jax_cuda,
)
from sim.manorl.environment import _decode_contact_forces


def _fixture() -> dict[str, object]:
    rng = np.random.default_rng(124)
    batch, capacity, ngeom = 4, 17, 24
    geom = rng.integers(0, ngeom, size=(capacity, 2), dtype=np.int32)
    # Explicitly exercise both hand-object orders and padded rows.
    geom[:4] = np.asarray(((0, 20), (20, 1), (2, 20), (20, 3)), dtype=np.int32)
    return {
        "nacon": np.asarray([capacity - 1], dtype=np.int32),
        "nefc": np.full(batch, 31, dtype=np.int32),
        "geom": geom,
        "world": rng.integers(0, batch, size=capacity, dtype=np.int32),
        "dimension": np.full(capacity, 3, dtype=np.int32),
        "addresses": rng.integers(0, 31, size=(capacity, 4), dtype=np.int32),
        "friction": rng.normal(size=(capacity, 5)).astype(np.float32),
        "frame": rng.normal(size=(capacity, 3, 3)).astype(np.float32),
        "constraint_force": rng.normal(size=(batch, 64)).astype(np.float32),
        "ngeom": ngeom,
        "keypoint_geom_ids": tuple(range(16)),
        "object_geom_ids": (20,),
    }


def _jitted_reducer(fixture: dict[str, object]):
    static = {
        key: fixture[key]
        for key in ("ngeom", "keypoint_geom_ids", "object_geom_ids")
    }
    dynamic = {key: value for key, value in fixture.items() if key not in static}
    fn = jax.jit(lambda **values: reduce_warp_contacts(**values, **static))
    return fn(**dynamic)


def test_jitted_contact_reduction_matches_numpy_decoder_and_masks_capacity() -> None:
    fixture = _fixture()
    actual = _jitted_reducer(fixture)
    count = int(np.asarray(fixture["nacon"])[0])
    expected = _decode_contact_forces(
        count=count,
        geom=np.asarray(fixture["geom"]),
        world=np.asarray(fixture["world"]),
        dimension=np.asarray(fixture["dimension"]),
        addresses=np.asarray(fixture["addresses"]),
        nefc=np.asarray(fixture["nefc"]),
        friction=np.asarray(fixture["friction"]),
        frame=np.asarray(fixture["frame"]),
        constraint_force=np.asarray(fixture["constraint_force"]),
        ngeom=int(fixture["ngeom"]),
        keypoint_geom_ids=fixture["keypoint_geom_ids"],
        object_geom_ids=set(fixture["object_geom_ids"]),
    )
    np.testing.assert_allclose(
        np.asarray(actual.keypoint_forces), expected[0][:, :16], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(actual.hand_object_forces), expected[1], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(actual.per_world_count), expected[2])
    assert bool(actual.valid) is True


@pytest.mark.parametrize(
    ("field", "index", "value"),
    (("dimension", 0, 4), ("world", 0, -1), ("geom", (0, 0), 99), ("addresses", (0, 0), 31)),
)
def test_contact_reduction_reports_malformed_live_rows_without_unsafe_gathers(
    field: str, index: object, value: int
) -> None:
    fixture = _fixture()
    changed = np.asarray(fixture[field]).copy()
    changed[index] = value  # type: ignore[index]
    fixture[field] = changed
    actual = _jitted_reducer(fixture)
    assert bool(actual.valid) is False
    assert np.isfinite(np.asarray(actual.keypoint_forces)).all()
    assert np.isfinite(np.asarray(actual.hand_object_forces)).all()


def test_contact_reduction_rejects_capacity_saturation() -> None:
    fixture = _fixture()
    fixture["nacon"] = np.asarray([len(np.asarray(fixture["world"]))], dtype=np.int32)
    actual = _jitted_reducer(fixture)
    assert bool(actual.valid) is False


@pytest.mark.skipif(
    os.environ.get("MANORL_RUN_CUDA_INTEROP") != "1",
    reason="CUDA DLPack boundary is opt-in; focused CPU validation must not submit GPU work",
)
def test_cuda_dlpack_round_trip_retains_device_and_storage() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    jax_array = torch_to_jax_cuda(source)
    round_trip = jax_to_torch_cuda(jax_array)
    assert jax_array.device.platform == "cuda"
    assert round_trip.is_cuda
    np.testing.assert_array_equal(round_trip.detach().cpu().numpy(), source.detach().cpu().numpy())
