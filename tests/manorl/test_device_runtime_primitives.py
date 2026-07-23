from __future__ import annotations

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from sim.manorl.device_runtime import (
    extract_mjx_physical_features,
    jax_to_torch_cuda,
    reduce_warp_contacts,
    torch_to_jax_cuda,
)
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment, _decode_contact_forces
from sim.manorl.trajectory import load_reference_trajectory


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


def _jitted_reducer(fixture: dict[str, object], *, compute_dtype: str = "float32"):
    static = {
        key: fixture[key]
        for key in ("ngeom", "keypoint_geom_ids", "object_geom_ids")
    }
    static["compute_dtype"] = compute_dtype
    dynamic = {key: value for key, value in fixture.items() if key not in static}
    fn = jax.jit(lambda **values: reduce_warp_contacts(**values, **static))
    return fn(**dynamic)


def test_jitted_physical_feature_gather_normalizes_xyzw_and_keeps_only_required_fields() -> None:
    rng = np.random.default_rng(52)
    batch, nq, nv, nbody = 3, 40, 38, 26
    qpos = rng.normal(size=(batch, nq)).astype(np.float32)
    qvel = rng.normal(size=(batch, nv)).astype(np.float32)
    xpos = rng.normal(size=(batch, nbody, 3)).astype(np.float32)
    xquat = rng.normal(size=(batch, nbody, 4)).astype(np.float32)
    keypoint_ids = tuple(range(3, 19))
    tips = (15, 3, 6, 9, 12)
    offsets = rng.normal(size=(5, 3)).astype(np.float32)
    fn = jax.jit(lambda q, v, p, r: extract_mjx_physical_features(
        qpos=q, qvel=v, xpos=p, xquat=r,
        hand_qpos_start=2, hand_dof=28, object_body_id=22,
        object_qvel_address=30, keypoint_body_ids=keypoint_ids,
        fingertip_keypoint_ids=tips, fingertip_local_offsets=offsets,
    ))
    actual = fn(qpos, qvel, xpos, xquat)
    expected_xyzw = xquat[:, keypoint_ids][:, :, (1, 2, 3, 0)]
    expected_xyzw /= np.linalg.norm(expected_xyzw, axis=-1, keepdims=True)
    np.testing.assert_allclose(np.asarray(actual.mano_dof_pos), qpos[:, 2:30], atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual.hand_keypoint_positions), xpos[:, keypoint_ids], atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual.hand_keypoint_orientations_xyzw), expected_xyzw, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(actual.object_orientation_xyzw), axis=1), 1.0, atol=2e-6)
    assert bool(actual.valid)


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


def test_scalar_nefc_broadcast_matches_host_decoder() -> None:
    fixture = _fixture()
    fixture["nefc"] = np.asarray([31], dtype=np.int32)
    actual = _jitted_reducer(fixture)
    count = int(np.asarray(fixture["nacon"])[0])
    expected = _decode_contact_forces(
        count=count,
        geom=np.asarray(fixture["geom"]),
        world=np.asarray(fixture["world"]),
        dimension=np.asarray(fixture["dimension"]),
        addresses=np.asarray(fixture["addresses"]),
        nefc=np.full(4, 31, dtype=np.int64),
        friction=np.asarray(fixture["friction"]),
        frame=np.asarray(fixture["frame"]),
        constraint_force=np.asarray(fixture["constraint_force"]),
        ngeom=int(fixture["ngeom"]),
        keypoint_geom_ids=fixture["keypoint_geom_ids"],
        object_geom_ids=set(fixture["object_geom_ids"]),
    )
    np.testing.assert_allclose(np.asarray(actual.keypoint_forces), expected[0][:, :16], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual.hand_object_forces), expected[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual.per_world_count), expected[2])
    assert bool(actual.valid) is True


def test_nonfinite_live_floor_contact_fails_closed_like_host_decoder() -> None:
    fixture = _fixture()
    # Geom 21 is neither a hand keypoint nor the object geom. The row remains
    # live and therefore must fail before reduction hides it from outputs.
    fixture["geom"] = np.asarray(fixture["geom"]).copy()
    fixture["geom"][0] = (21, 22)
    fixture["constraint_force"] = np.asarray(fixture["constraint_force"]).copy()
    addresses = np.asarray(fixture["addresses"])
    fixture["constraint_force"][np.asarray(fixture["world"])[0], addresses[0, 0]] = np.nan
    actual = _jitted_reducer(fixture)
    assert bool(actual.valid) is False
    with pytest.raises(RuntimeError, match="non-finite world force"):
        _decode_contact_forces(
            count=int(np.asarray(fixture["nacon"])[0]),
            geom=np.asarray(fixture["geom"]),
            world=np.asarray(fixture["world"]),
            dimension=np.asarray(fixture["dimension"]),
            addresses=addresses,
            nefc=np.asarray(fixture["nefc"]),
            friction=np.asarray(fixture["friction"]),
            frame=np.asarray(fixture["frame"]),
            constraint_force=np.asarray(fixture["constraint_force"]),
            ngeom=int(fixture["ngeom"]),
            keypoint_geom_ids=fixture["keypoint_geom_ids"],
            object_geom_ids=set(fixture["object_geom_ids"]),
        )


def test_real_mjx_warp_private_buffers_match_host_decoder() -> None:
    """Pin the actual DataWarp field names/shapes and reduction parity."""

    if jax.config.x64_enabled:
        pytest.skip("MJX-Warp 3.10 FFI pins its model buffers to float32")
    environment = MujocoManoEnvironment(
        load_reference_trajectory(),
        EnvironmentConfig(num_envs=2, device="cpu", max_deviation_distance=1_000_000.0),
    )
    buffers = environment.producer.materialize_contact_buffers(environment.data, batch=2)
    host_geometry, host_hand_object, host_counts = environment.producer.decode_contact_buffers(buffers)
    keypoint, hand_object, counts = environment.producer._device_decode_contact_buffers(
        environment.data, batch=2
    )
    assert buffers.count < buffers.capacity
    np.testing.assert_allclose(
        keypoint,
        host_geometry[:, environment.producer.keypoint_geom_ids],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(hand_object, host_hand_object, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(counts, host_counts)


def test_device_contact_decode_config_fails_closed_for_cpu_and_debug_snapshots() -> None:
    with pytest.raises(ValueError, match="requires device='gpu'"):
        EnvironmentConfig(device="cpu", device_contact_decode=True, capture_transition_diagnostics=False)
    with pytest.raises(ValueError, match="capture_transition_diagnostics=False"):
        EnvironmentConfig(device="gpu", device_contact_decode=True)


def test_float64_reducer_matches_float64_host_decoder() -> None:
    if not jax.config.x64_enabled:
        pytest.skip("JAX x64 is disabled")
    fixture = _fixture()
    for key in ("friction", "frame", "constraint_force"):
        fixture[key] = np.asarray(fixture[key], dtype=np.float64)
    actual = _jitted_reducer(fixture, compute_dtype="float64")
    expected = _decode_contact_forces(
        count=int(np.asarray(fixture["nacon"])[0]),
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
    np.testing.assert_allclose(np.asarray(actual.keypoint_forces), expected[0][:, :16], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(actual.hand_object_forces), expected[1], rtol=1e-12, atol=1e-12)
    assert bool(actual.valid) is True


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
