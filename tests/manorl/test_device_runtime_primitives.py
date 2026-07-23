from __future__ import annotations

import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from sim.manorl.device_runtime import (
    DevicePhysicalFeatures,
    build_device_observation_28,
    extract_mjx_physical_features,
    jax_to_torch_cuda,
    reduce_warp_contacts,
    torch_to_jax_cuda,
)
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment, _decode_contact_forces
from sim.manorl.observations import (
    CURRENT_SOURCE_COMPATIBILITY,
    SOURCE_ALIGNED_COMPATIBILITY,
    ObservationState,
    PointCloudTemplate,
    build_observation,
)
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


def test_jitted_28dof_observation_matches_numpy_at_contact_thresholds_and_partial_rows() -> None:
    rng = np.random.default_rng(98)
    batch = 4
    mano = rng.uniform(-0.7, 0.7, size=(batch, 28)).astype(np.float32)
    lower, upper = np.full(28, -1.0, np.float32), np.full(28, 1.0, np.float32)
    object_quat = rng.normal(size=(batch, 4)).astype(np.float32)
    object_quat /= np.linalg.norm(object_quat, axis=1, keepdims=True)
    hand_quat = rng.normal(size=(batch, 4)).astype(np.float32)
    hand_quat /= np.linalg.norm(hand_quat, axis=1, keepdims=True)
    keypoint_quat = np.broadcast_to(hand_quat[:, None], (batch, 16, 4)).copy()
    object_pos = rng.normal(size=(batch, 3)).astype(np.float32)
    hand_pos = rng.normal(size=(batch, 3)).astype(np.float32)
    keypoints = rng.normal(size=(batch, 16, 3)).astype(np.float32)
    tips = rng.normal(size=(batch, 5, 3)).astype(np.float32)
    physical = DevicePhysicalFeatures(
        mano, hand_pos, hand_quat, keypoint_quat, object_pos, object_quat,
        rng.normal(size=(batch, 3)).astype(np.float32), keypoints, tips, jax.numpy.asarray(True),
    )
    forces = np.zeros((batch, 16, 3), dtype=np.float32)
    forces[:, 3, 0] = np.asarray([0.199999, 0.2, 0.200001, 10.0], dtype=np.float32)
    # Dynamic reset templates are per-environment; this pins that device
    # broadcasting does not accidentally reuse environment zero's cloud.
    points = rng.normal(size=(batch, 64, 3)).astype(np.float32)
    scale = rng.uniform(0.2, 0.4, size=(batch, 3)).astype(np.float32)
    support = rng.normal(size=(13, 3)).astype(np.float32)
    target_pos = rng.normal(size=(batch, 3)).astype(np.float32)
    target_quat = rng.normal(size=(batch, 4)).astype(np.float32)
    target_quat *= 2.5  # Device path must normalize target XYZW exactly as NumPy does for support points.
    expected = np.zeros((batch, 16), dtype=np.float32)
    expected[:, (3, 15)] = 1.0
    kwargs = dict(
        physical=physical, hand_keypoint_contact_forces=forces,
        target_object_position=target_pos, target_object_orientation_xyzw=target_quat,
        target_object_pos_next_5=rng.normal(size=(batch, 3)).astype(np.float32),
        cumulative_offset=rng.normal(size=(batch, 3)).astype(np.float32),
        cumulative_joint_offset=rng.normal(size=(batch, 22)).astype(np.float32),
        point_cloud_local=points, point_cloud_scale=scale,
        object_geometry=rng.normal(size=(batch, 12)).astype(np.float32),
        expected_contact_mask=expected, action_ids=np.asarray([1, 2, 49, 50], dtype=np.int32),
        object_support_points=support, table_surface_height=-0.001,
        mano_dof_lower=lower, mano_dof_upper=upper,
    )
    raw, valid = jax.jit(lambda: build_device_observation_28(**kwargs))()
    numpy_state = ObservationState(
        mano_dof_pos=mano, mano_dof_lower=lower, mano_dof_upper=upper,
        hand_position=hand_pos, hand_orientation_xyzw=hand_quat,
        object_position=object_pos, object_orientation_xyzw=object_quat,
        target_object_position=target_pos,
        target_object_orientation_xyzw=target_quat,
        target_object_pos_next_5=kwargs["target_object_pos_next_5"],
        cumulative_offset=kwargs["cumulative_offset"],
        cumulative_joint_offset=kwargs["cumulative_joint_offset"],
        point_cloud=PointCloudTemplate(points, mode="dynamic_reset", normalized=True, scale=scale),
        object_geometry=kwargs["object_geometry"], hand_keypoint_positions=keypoints,
        fingertip_positions=tips, hand_keypoint_contact_forces=forces,
        expected_contact_mask=expected, action_ids=kwargs["action_ids"],
        object_support_points=support, table_surface_height=-0.001,
    )
    expected_observation = build_observation(numpy_state, compatibility=SOURCE_ALIGNED_COMPATIBILITY)
    np.testing.assert_allclose(np.asarray(raw), expected_observation.raw, rtol=0, atol=2e-6)
    np.testing.assert_allclose(np.clip(np.asarray(raw), -5.0, 5.0), expected_observation.policy_input, rtol=0, atol=2e-6)
    assert bool(valid)
    # Match NumPy's strict 0.2N gate after its float32-to-float64 promotion.
    # 28D layout shifts the fixed 26D direction slice by four finger channels.
    assert np.linalg.norm(np.asarray(raw)[0, 416 + 3 * 3 : 416 + 3 * 4]) == 0.0
    assert np.linalg.norm(np.asarray(raw)[1, 416 + 3 * 3 : 416 + 3 * 4]) > 0.9


def test_builder_rejects_unordered_limits_and_nonfinite_table_height() -> None:
    """Mirror the host encoder's fail-closed limit/table validation."""
    rng = np.random.default_rng(11)
    batch = 2
    physical = DevicePhysicalFeatures(
        rng.normal(size=(batch, 28)).astype(np.float32), rng.normal(size=(batch, 3)).astype(np.float32),
        np.tile(np.asarray((0, 0, 0, 1), np.float32), (batch, 1)),
        np.tile(np.asarray((0, 0, 0, 1), np.float32), (batch, 16, 1)),
        rng.normal(size=(batch, 3)).astype(np.float32), np.tile(np.asarray((0, 0, 0, 1), np.float32), (batch, 1)),
        rng.normal(size=(batch, 3)).astype(np.float32), rng.normal(size=(batch, 16, 3)).astype(np.float32),
        rng.normal(size=(batch, 5, 3)).astype(np.float32), jax.numpy.asarray(True),
    )
    kwargs = dict(
        physical=physical, hand_keypoint_contact_forces=np.zeros((batch, 16, 3), np.float32),
        target_object_position=np.zeros((batch, 3), np.float32), target_object_orientation_xyzw=np.tile(np.asarray((0, 0, 0, 1), np.float32), (batch, 1)),
        target_object_pos_next_5=np.zeros((batch, 3), np.float32), cumulative_offset=np.zeros((batch, 3), np.float32),
        cumulative_joint_offset=np.zeros((batch, 22), np.float32), point_cloud_local=np.zeros((64, 3), np.float32),
        point_cloud_scale=np.ones(3, np.float32), object_geometry=np.zeros((batch, 12), np.float32),
        expected_contact_mask=np.zeros((batch, 16), np.float32), action_ids=np.ones(batch, np.int32),
        object_support_points=np.zeros((1, 3), np.float32), mano_dof_lower=np.zeros(28, np.float32),
        mano_dof_upper=np.ones(28, np.float32), table_surface_height=0.0,
    )
    invalid_limits = dict(kwargs, mano_dof_upper=np.zeros(28, np.float32))
    assert not bool(jax.jit(lambda: build_device_observation_28(**invalid_limits))()[1])
    invalid_table = dict(kwargs, table_surface_height=np.nan)
    assert not bool(jax.jit(lambda: build_device_observation_28(**invalid_table))()[1])


def test_physical_feature_offsets_preserve_left_handedness_flip() -> None:
    """The caller owns the left-side X flip exactly as materialize_state does."""
    qpos = np.zeros((1, 32), np.float32)
    qvel = np.zeros((1, 32), np.float32)
    xpos = np.zeros((1, 20, 3), np.float32)
    xquat = np.zeros((1, 20, 4), np.float32)
    xquat[..., 0] = 1.0  # identity in source wxyz order
    offsets = np.asarray(((0.1, 0.2, 0.3),) * 5, np.float32)
    common = dict(qpos=qpos, qvel=qvel, xpos=xpos, xquat=xquat, hand_qpos_start=0, hand_dof=28,
                  object_body_id=19, object_qvel_address=0, keypoint_body_ids=tuple(range(16)),
                  fingertip_keypoint_ids=(0, 1, 2, 3, 4))
    right = extract_mjx_physical_features(**common, fingertip_local_offsets=offsets)
    left_offsets = offsets.copy(); left_offsets[:, 0] *= -1
    left = extract_mjx_physical_features(**common, fingertip_local_offsets=left_offsets)
    np.testing.assert_allclose(np.asarray(right.fingertip_positions)[..., 1:], np.asarray(left.fingertip_positions)[..., 1:])
    np.testing.assert_allclose(np.asarray(right.fingertip_positions)[..., 0], -np.asarray(left.fingertip_positions)[..., 0])


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
