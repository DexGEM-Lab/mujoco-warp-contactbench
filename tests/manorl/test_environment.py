from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from sim.manorl.abi import ENVIRONMENT_CONTRACT_ID, check_termination
from sim.manorl.assets import OBJECT_MESH, compile_model
from sim.manorl.contracts import JOINT_DOF, KEYPOINT_NAMES
from sim.manorl.environment import (
    EnvironmentConfig,
    MjxWarpPhysicalProducer,
    MujocoManoEnvironment,
    PhysicalSnapshot,
    _model_hand_side_order,
    _aggregate_geometry_contact_forces,
    _decode_contact_forces,
    _scatter_routed_value,
)
from sim.manorl.mjx_sim import MujocoCpuReplay
from sim.manorl.observations import (
    CURRENT_SOURCE_COMPATIBILITY,
    SOURCE_ALIGNED_COMPATIBILITY,
    quat_rotate_xyzw,
)
from sim.manorl.rewards import PPO_REWARD_CONTRACT_ID, REWARD_CONTRACT_ID, REWARD_HAND_OBJECT_THRESHOLD_N, compute_rewards
from sim.manorl.trajectory import (
    TrajectoryBatch,
    _initial_support_shift,
    load_reference_trajectory,
)


@pytest.fixture(scope="module")
def trajectory():
    return load_reference_trajectory()


def _environment(trajectory, *, num_envs: int = 1, residual_enabled: bool = False) -> MujocoManoEnvironment:
    return MujocoManoEnvironment(
        trajectory,
        EnvironmentConfig(
            num_envs=num_envs,
            residual_enabled=residual_enabled,
            # Physical contact/state extraction is under test; this avoids
            # making an action-ABI fixture depend on a trajectory deviation.
            max_deviation_distance=1_000_000.0,
        ),
    )

def _physical_snapshot(*, batch: int, ngeom: int, marker: float) -> PhysicalSnapshot:
    return PhysicalSnapshot(
        mano_dof_pos=np.full((batch, 2), marker),
        hand_position=np.full((batch, 3), marker),
        hand_orientation_xyzw=np.full((batch, 4), marker),
        hand_keypoint_orientations_xyzw=np.full((batch, 2, 4), marker),
        object_position=np.full((batch, 3), marker),
        object_orientation_xyzw=np.full((batch, 4), marker),
        object_linear_velocity=np.full((batch, 3), marker),
        hand_keypoint_positions=np.full((batch, 2, 3), marker),
        fingertip_positions=np.full((batch, 1, 3), marker),
        hand_keypoint_contact_forces=np.full((batch, 2, 3), marker),
        object_contact_force=np.full((batch, ngeom, 3), marker),
        geom_contact_force_world_N=np.full((batch, ngeom, 3), marker),
        hand_object_force_on_object_world_N=np.full((batch, 2, 3), marker),
        contact_count=np.full(batch, int(marker), dtype=np.int64),
    )


def test_scatter_routed_physical_snapshots_drops_only_ragged_geometry_diagnostics() -> None:
    scattered = _scatter_routed_value(
        [
            (np.asarray([0], dtype=np.int64), _physical_snapshot(batch=1, ngeom=1, marker=1.0)),
            (np.asarray([1], dtype=np.int64), _physical_snapshot(batch=1, ngeom=3, marker=2.0)),
        ],
        total=2,
    )

    assert isinstance(scattered, PhysicalSnapshot)
    np.testing.assert_array_equal(scattered.hand_position[:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(scattered.contact_count, [1, 2])
    assert scattered.object_contact_force is None
    assert scattered.geom_contact_force_world_N is None


def test_scatter_routed_physical_snapshot_rejects_unrelated_ragged_arrays() -> None:
    with pytest.raises(RuntimeError, match="incompatible array shapes"):
        _scatter_routed_value(
            [
                (np.asarray([0], dtype=np.int64), _physical_snapshot(batch=1, ngeom=1, marker=1.0)),
                (
                    np.asarray([1], dtype=np.int64),
                    replace(
                        _physical_snapshot(batch=1, ngeom=1, marker=2.0),
                        hand_position=np.zeros((1, 4), dtype=np.float64),
                    ),
                ),
            ],
            total=2,
        )


def test_environment_config_accepts_unified_batch_with_explicit_ccd() -> None:
    config = EnvironmentConfig(
        unified_object_batch=True,
        warp_ccd_iterations=8,
        warp_ccd_contacts_per_world=12,
    )

    assert config.unified_object_batch is True
    assert config.warp_ccd_explicit is True


def test_persistent_ccd_workspace_config_requires_unified_gpu_explicit_capacity() -> None:
    with pytest.raises(ValueError, match="device='gpu'"):
        EnvironmentConfig(
            unified_object_batch=True,
            warp_ccd_contacts_per_world=12,
            warp_persistent_ccd_workspace=True,
        )
    with pytest.raises(ValueError, match="warp_ccd_contacts_per_world"):
        EnvironmentConfig(
            device="gpu",
            unified_object_batch=True,
            warp_persistent_ccd_workspace=True,
        )
    with pytest.raises(ValueError, match="unified_object_batch=True"):
        EnvironmentConfig(
            device="gpu",
            warp_ccd_contacts_per_world=12,
            warp_persistent_ccd_workspace=True,
        )

    config = EnvironmentConfig(
        device="gpu",
        unified_object_batch=True,
        warp_ccd_contacts_per_world=12,
        warp_persistent_ccd_workspace=True,
    )
    assert config.warp_persistent_ccd_workspace is True


def test_bimanual_reference_tables_use_compiled_right_left_order() -> None:
    """Metadata lookup may be left/right, but model qpos/ctrl slots are right/left."""

    assert _model_hand_side_order(("left", "right")) == ("right", "left")
    q_ref = {
        "left": np.full((1, 28), -1.0),
        "right": np.full((1, 28), 1.0),
    }
    model_reference = np.concatenate(
        [q_ref[side] for side in _model_hand_side_order(q_ref)], axis=-1
    )
    np.testing.assert_array_equal(model_reference[:, :28], 1.0)
    np.testing.assert_array_equal(model_reference[:, 28:], -1.0)


def test_visual_model_physical_producer_selects_only_collision_geometry() -> None:
    mujoco, model = compile_model(visual_meshes=True)
    producer = MjxWarpPhysicalProducer(mujoco, model)

    collision_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "cube1_collision"
    )
    visual_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "cube1_visual"
    )
    assert producer.object_geom_ids == {collision_id}
    assert visual_id not in producer.object_geom_ids


def test_native_contact_frame_and_geom_sign_match_body_external_force(trajectory) -> None:
    """Controlled native fixture validates the world-frame/sign convention."""

    replay = MujocoCpuReplay(trajectory)
    replay.step(np.zeros(26, dtype=np.float64))
    replay.mujoco.mj_rnePostConstraint(replay.model, replay.data)
    for contact_id, contact in enumerate(replay.data.contact[: replay.data.ncon]):
        wrench = np.zeros(6, dtype=np.float64)
        replay.mujoco.mj_contactForce(replay.model, replay.data, contact_id, wrench)
        if np.linalg.norm(wrench[:3]) <= 1e-8:
            continue
        first_geom, second_geom = map(int, contact.geom)
        first_body = int(replay.model.geom_bodyid[first_geom])
        second_body = int(replay.model.geom_bodyid[second_geom])
        if first_body == 0 or second_body == 0:
            world_force = wrench[:3] @ contact.frame.reshape(3, 3)
            moving_body = second_body if first_body == 0 else first_body
            expected = world_force if first_body == 0 else -world_force
            np.testing.assert_allclose(
                replay.data.cfrc_ext[moving_body, 3:], expected, rtol=1e-10, atol=1e-10
            )
            return
    pytest.fail("expected a solved floor-hand contact in the real accepted replay state")


def test_native_multi_contact_geom_aggregation_matches_external_force() -> None:
    """A tilted box has four contact rows, which must sum in world coordinates."""

    import mujoco

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.002" gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="floor" type="plane" size="2 2 .1"/>
            <body name="box" pos="0 0 .04" euler=".2 .1 .3">
              <freejoint/>
              <geom name="box" type="box" size=".1 .1 .05" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    mujoco.mj_rnePostConstraint(model, data)

    geom_forces = np.zeros((model.ngeom, 3), dtype=np.float64)
    for contact_id, contact in enumerate(data.contact[: data.ncon]):
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        world_force = wrench[:3] @ contact.frame.reshape(3, 3)
        first_geom, second_geom = map(int, contact.geom)
        geom_forces[first_geom] -= world_force
        geom_forces[second_geom] += world_force

    assert data.ncon == 4
    np.testing.assert_allclose(geom_forces[0], -geom_forces[1], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(geom_forces[1], data.cfrc_ext[1, 3:], rtol=1e-10, atol=1e-10)
    assert not np.allclose(geom_forces[1], np.zeros(3))


def test_geometry_contact_aggregation_routes_worlds_and_rejects_unsolved_addresses() -> None:
    """Synthetic solved rows expose sign, frame, sum, world, and nefc errors."""

    count = 3
    geom = np.asarray(((0, 1), (0, 1), (2, 3)), dtype=np.int64)
    world = np.asarray((0, 0, 1), dtype=np.int64)
    dimension = np.full(count, 3, dtype=np.int64)
    addresses = np.asarray(((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 2, 3)), dtype=np.int64)
    nefc = np.asarray((8, 4), dtype=np.int64)
    friction = np.ones((count, 5), dtype=np.float64)
    frame = np.asarray(
        (
            np.eye(3),
            ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            np.eye(3),
        ),
        dtype=np.float64,
    )
    constraint_force = np.asarray(
        (
            (1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 1.0),
            (4.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )

    forces, counts = _aggregate_geometry_contact_forces(
        count=count,
        geom=geom,
        world=world,
        dimension=dimension,
        addresses=addresses,
        nefc=nefc,
        friction=friction,
        frame=frame,
        constraint_force=constraint_force,
        ngeom=4,
    )
    first_world_force = np.asarray((10.0, -1.0, -1.0))
    second_world_force = np.asarray((2.0, -4.0, 0.0))
    np.testing.assert_allclose(forces[0, 0], -(first_world_force + second_world_force))
    np.testing.assert_allclose(forces[0, 1], first_world_force + second_world_force)
    np.testing.assert_allclose(forces[1, 2], (-5.0, -4.0, -1.0))
    np.testing.assert_allclose(forces[1, 3], (5.0, 4.0, 1.0))
    np.testing.assert_array_equal(counts, (2, 1))

    invalid_addresses = addresses.copy()
    invalid_addresses[2, 3] = nefc[1]
    with pytest.raises(RuntimeError, match="outside world 1's solved range"):
        _aggregate_geometry_contact_forces(
            count=count,
            geom=geom,
            world=world,
            dimension=dimension,
            addresses=invalid_addresses,
            nefc=nefc,
            friction=friction,
            frame=frame,
            constraint_force=constraint_force,
            ngeom=4,
        )


def test_pair_filtered_contact_decoder_preserves_geometry_aggregation_and_force_direction() -> None:
    """Only source-mapped hand/object rows contribute force exerted on the object."""

    keypoint_geom_ids = tuple(range(10, 10 + len(KEYPOINT_NAMES)))
    object_geom_id = 99
    count = 7
    geom = np.asarray(
        (
            (keypoint_geom_ids[0], object_geom_id),  # hand -> object
            (keypoint_geom_ids[0], object_geom_id),  # same pair, vector sum
            (object_geom_id, keypoint_geom_ids[1]),  # object -> hand
            (keypoint_geom_ids[2], 0),  # hand -> floor
            (object_geom_id, 0),  # object -> floor
            (keypoint_geom_ids[3], keypoint_geom_ids[4]),  # hand -> hand
            (100, object_geom_id),  # other -> object
        ),
        dtype=np.int64,
    )
    world = np.zeros(count, dtype=np.int64)
    dimension = np.full(count, 3, dtype=np.int64)
    addresses = np.arange(count * 4, dtype=np.int64).reshape(count, 4)
    nefc = np.asarray((count * 4,), dtype=np.int64)
    friction = np.ones((count, 5), dtype=np.float64)
    frame = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    world_forces = np.asarray(
        ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0), (1.0, 2.0, 3.0),
         (4.0, 5.0, 6.0), (7.0, 8.0, 9.0), (10.0, 11.0, 12.0)),
        dtype=np.float64,
    )
    constraint_force = np.zeros((1, count * 4), dtype=np.float64)
    for contact_id, (normal, tangent_x, tangent_y) in enumerate(world_forces):
        constraint_force[0, addresses[contact_id]] = (
            (normal + tangent_x) / 2.0,
            (normal - tangent_x) / 2.0,
            tangent_y / 2.0,
            -tangent_y / 2.0,
        )

    legacy_geometry, legacy_counts = _aggregate_geometry_contact_forces(
        count=count,
        geom=geom,
        world=world,
        dimension=dimension,
        addresses=addresses,
        nefc=nefc,
        friction=friction,
        frame=frame,
        constraint_force=constraint_force,
        ngeom=101,
    )
    geometry, hand_object, counts = _decode_contact_forces(
        count=count,
        geom=geom,
        world=world,
        dimension=dimension,
        addresses=addresses,
        nefc=nefc,
        friction=friction,
        frame=frame,
        constraint_force=constraint_force,
        ngeom=101,
        keypoint_geom_ids=keypoint_geom_ids,
        object_geom_ids={object_geom_id},
    )

    assert hand_object.dtype == np.float64
    assert hand_object.shape == (1, len(KEYPOINT_NAMES), 3)
    np.testing.assert_allclose(hand_object[0, 0], (2.0, 3.0, 0.0))
    np.testing.assert_allclose(hand_object[0, 1], (0.0, 0.0, -4.0))
    np.testing.assert_allclose(hand_object[0, 2:], 0.0)
    np.testing.assert_allclose(geometry, legacy_geometry, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(counts, legacy_counts)
    np.testing.assert_allclose(geometry.sum(axis=1), 0.0, rtol=0, atol=1e-12)


def test_reward_state_filters_broad_contacts_before_reward() -> None:
    env = object.__new__(MujocoManoEnvironment)
    env.config = EnvironmentConfig(num_envs=1)
    env.trajectory_lengths = np.asarray([10], dtype=np.int64)
    env.trajectory_steps = np.asarray([4], dtype=np.int64)
    env.reference_object_pos = np.zeros((1, 10, 3), dtype=np.float64)
    env.reference_object_quat_xyzw = np.tile(
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (1, 10, 1)
    )
    env.cumulative_offset = np.zeros((1, 3), dtype=np.float64)
    env.cumulative_joint_offset = np.zeros((1, 20), dtype=np.float64)
    env.active_joint_mask = np.zeros((1, 20), dtype=bool)
    env.expected_contact_mask = np.zeros((1, 16), dtype=np.float64)
    env.expected_contact_mask[:, 3] = 1.0
    env.expected_contact_weights = env.expected_contact_mask.copy()
    env.contact_start_frames = np.asarray([3], dtype=np.int64)
    env.contact_end_frames = np.asarray([5], dtype=np.int64)
    all_contact_forces = np.full((1, 16, 3), 100.0, dtype=np.float64)
    physical = PhysicalSnapshot(
        mano_dof_pos=np.zeros((1, 26), dtype=np.float64),
        hand_position=np.zeros((1, 3), dtype=np.float64),
        hand_orientation_xyzw=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        hand_keypoint_orientations_xyzw=np.tile(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (1, 16, 1)
        ),
        object_position=np.zeros((1, 3), dtype=np.float64),
        object_orientation_xyzw=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        object_linear_velocity=np.zeros((1, 3), dtype=np.float64),
        hand_keypoint_positions=np.zeros((1, 16, 3), dtype=np.float64),
        fingertip_positions=np.zeros((1, 5, 3), dtype=np.float64),
        hand_keypoint_contact_forces=all_contact_forces,
        object_contact_force=np.zeros((1, 3), dtype=np.float64),
        geom_contact_force_world_N=np.zeros((1, 1, 3), dtype=np.float64),
        hand_object_force_on_object_world_N=np.zeros((1, 16, 3), dtype=np.float64),
        contact_count=np.zeros(1, dtype=np.int64),
    )
    termination = check_termination(
        object_position=physical.object_position,
        target_position=physical.object_position,
        progress=np.asarray([4], dtype=np.int64),
        trajectory_lengths=np.asarray([10], dtype=np.int64),
        early_mask=np.asarray([False]),
    )

    broad_only = compute_rewards(
        env._reward_state(physical), compatibility=CURRENT_SOURCE_COMPATIBILITY, termination=termination
    )
    filtered_forces = np.zeros((1, 16, 3), dtype=np.float64)
    filtered_forces[:, 3] = [2.1, 0.0, 0.0]
    filtered = compute_rewards(
        env._reward_state(replace(physical, hand_object_force_on_object_world_N=filtered_forces)),
        compatibility=CURRENT_SOURCE_COMPATIBILITY,
        termination=termination,
    )

    np.testing.assert_allclose(broad_only.contact, [0.0])
    np.testing.assert_allclose(filtered.contact, [0.4])


def test_producer_keypoint_order_fingertips_and_static_template(trajectory) -> None:
    env = _environment(trajectory)
    assert env.last_physical is not None
    snapshot = env.last_physical
    xpos = np.asarray(env.data.xpos, dtype=np.float64)[0]
    np.testing.assert_allclose(
        snapshot.hand_keypoint_positions[0], xpos[np.asarray(env.producer.keypoint_body_ids)]
    )
    source_orientations = np.asarray(env.data.xquat, dtype=np.float64)[
        0, np.asarray(env.producer.keypoint_body_ids)
    ]
    source_orientations = source_orientations[:, (1, 2, 3, 0)]
    source_orientations /= np.linalg.norm(source_orientations, axis=1, keepdims=True)
    np.testing.assert_allclose(snapshot.hand_keypoint_orientations_xyzw[0], source_orientations)
    source_quat = snapshot.hand_orientation_xyzw[0]
    assert np.isclose(np.linalg.norm(source_quat), 1.0)
    fingertip_keypoint_ids = [15, 3, 6, 9, 12]
    quaternions = np.asarray(env.data.xquat, dtype=np.float64)[0, np.asarray(env.producer.keypoint_body_ids)[fingertip_keypoint_ids]]
    quaternions_xyzw = quaternions[:, (1, 2, 3, 0)]
    offsets = np.asarray(
        (
            (-0.028633, -0.004191, 0.023667),
            (-0.027384, -0.000215, -0.000966),
            (-0.027549, 0.000501, -0.004523),
            (-0.026165, -0.000075, -0.007781),
            (-0.018526, -0.001581, -0.011018),
        )
    )
    expected_tip = snapshot.hand_keypoint_positions[0, fingertip_keypoint_ids] + quat_rotate_xyzw(quaternions_xyzw, offsets)
    np.testing.assert_allclose(snapshot.fingertip_positions[0], expected_tip, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(np.flatnonzero(env.expected_contact_mask[0]), [3, 15])
    np.testing.assert_array_equal(env.active_joint_mask[0], [True] * 10 + [False] * 12)

    trimesh = pytest.importorskip("trimesh")
    source_mesh = trimesh.load(OBJECT_MESH, force="mesh")
    source_mesh.apply_scale(0.001)
    source_points, _ = trimesh.sample.sample_surface(source_mesh, 64, seed=42)
    lower, upper = source_points.min(axis=0), source_points.max(axis=0)
    expected_template = (source_points - (upper + lower) / 2.0) / np.maximum((upper - lower) / 2.0, 1e-6)
    np.testing.assert_allclose(env._static_template.local_points, expected_template, rtol=0, atol=1e-12)


def test_mjx_contact_producer_feeds_source_order_observation_and_reward(trajectory) -> None:
    env = _environment(trajectory)
    observation, reward, reset, _ = env.step(
        np.zeros((1, env.action_dim), dtype=np.float64)
    )
    assert env.last_physical is not None
    assert env.last_observation is not None
    assert env.last_reward is not None
    snapshot = env.last_physical
    assert snapshot.contact_count[0] > 0
    assert np.linalg.norm(snapshot.hand_keypoint_contact_forces[0, 3]) > 0.0
    np.testing.assert_allclose(
        env.last_observation.contacts.force_xyz, snapshot.hand_keypoint_contact_forces
    )
    np.testing.assert_allclose(
        env.last_observation.raw[
            :, env.observation_layout.slices["expected_contact_mask"]
        ],
        env.expected_contact_mask,
    )
    assert observation["obs"].shape == (1, env.observation_dim) == (1, 480)
    assert reward.shape == reset.shape == (1,)
    assert np.all(np.isfinite(reward))


def test_mjx_geometry_force_snapshot_matches_private_rows_and_legacy_aggregates(trajectory) -> None:
    env = _environment(trajectory, num_envs=2)
    env.step(np.zeros((2, env.action_dim), dtype=np.float64))
    assert env.last_physical is not None
    physical = env.last_physical
    impl = env.data._impl
    count = int(np.asarray(impl.nacon, dtype=np.int64).reshape(-1)[0])
    nefc_values = np.asarray(impl.nefc, dtype=np.int64).reshape(-1)
    nefc = np.full(2, nefc_values[0], dtype=np.int64) if nefc_values.shape == (1,) else nefc_values
    expected = np.zeros_like(physical.geom_contact_force_world_N)
    expected_hand_object = np.zeros((2, len(KEYPOINT_NAMES), 3), dtype=np.float64)
    keypoint_indices = {geom_id: index for index, geom_id in enumerate(env.producer.keypoint_geom_ids)}
    geom = np.asarray(impl.contact__geom, dtype=np.int64)
    world = np.asarray(impl.contact__worldid, dtype=np.int64)
    addresses = np.asarray(impl.contact__efc_address, dtype=np.int64)
    friction = np.asarray(impl.contact__friction, dtype=np.float64)
    frame = np.asarray(impl.contact__frame, dtype=np.float64)
    constraint_force = np.asarray(impl.efc__force, dtype=np.float64)
    for contact_id in range(count):
        world_id = int(world[contact_id])
        address = addresses[contact_id]
        assert np.all(address >= 0) and np.all(address < nefc[world_id])
        pyramid = constraint_force[world_id, address]
        local_force = np.asarray(
            (pyramid.sum(), (pyramid[0] - pyramid[1]) * friction[contact_id, 0], (pyramid[2] - pyramid[3]) * friction[contact_id, 1])
        )
        world_force = local_force @ frame[contact_id]
        first_geom, second_geom = map(int, geom[contact_id])
        expected[world_id, first_geom] -= world_force
        expected[world_id, second_geom] += world_force
        if first_geom in keypoint_indices and second_geom in env.producer.object_geom_ids:
            expected_hand_object[world_id, keypoint_indices[first_geom]] += world_force
        elif second_geom in keypoint_indices and first_geom in env.producer.object_geom_ids:
            expected_hand_object[world_id, keypoint_indices[second_geom]] -= world_force
    np.testing.assert_allclose(physical.geom_contact_force_world_N, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        physical.hand_keypoint_contact_forces,
        physical.geom_contact_force_world_N[:, env.producer.keypoint_geom_ids],
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        physical.object_contact_force,
        physical.geom_contact_force_world_N[:, sorted(env.producer.object_geom_ids)].sum(axis=1),
        rtol=0,
        atol=1e-10,
    )
    assert physical.hand_object_force_on_object_world_N.dtype == np.float64
    np.testing.assert_allclose(
        physical.hand_object_force_on_object_world_N,
        expected_hand_object,
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(env.object_gravity_world_force, (0.0, 0.0, -1.22625), rtol=0, atol=1e-12)


def test_source_counter_schedule_terminal_observation_and_delayed_reset(trajectory) -> None:
    env = _environment(trajectory)
    zero = np.zeros((1, env.action_dim), dtype=np.float64)
    clipped_reference = np.clip(
        env.reference_q_model,
        env.joint_lower,
        env.joint_upper,
    )[0]
    env.step(zero)
    np.testing.assert_allclose(env.last_controller_targets[0], clipped_reference[0])
    np.testing.assert_array_equal(env.progress, [1])
    np.testing.assert_array_equal(env.trajectory_steps, [0])
    env.step(zero)
    np.testing.assert_allclose(env.last_controller_targets[0], clipped_reference[0])
    np.testing.assert_array_equal(env.trajectory_steps, [1])
    np.testing.assert_allclose(
        env.last_observation.raw[
            0, env.observation_layout.slices["target_object_position"]
        ],
        trajectory.object_pos[1],
    )
    env.step(zero)
    np.testing.assert_allclose(env.last_controller_targets[0], clipped_reference[1])

    # Source state immediately before the final control call: target 789 is
    # commanded, post-step target 790 is observed, then reset is flagged.
    env.progress[:] = 790
    env.trajectory_steps[:] = 789
    terminal_obs, _, done, terminal_extras = env.step(zero)
    np.testing.assert_array_equal(done, [True])
    np.testing.assert_array_equal(terminal_extras["time_outs"], [False])
    assert env.last_transition is not None
    np.testing.assert_array_equal(env.last_transition.command_reference_indices, [789])
    np.testing.assert_array_equal(env.last_transition.target_indices, [790])
    np.testing.assert_array_equal(env.last_transition.reset_applied, [False])
    np.testing.assert_array_equal(env.last_transition.termination.reset, [True])
    np.testing.assert_array_equal(env.progress, [791])
    np.testing.assert_array_equal(env.trajectory_steps, [790])
    np.testing.assert_allclose(
        terminal_obs["obs"][
            0, env.observation_layout.slices["target_object_position"]
        ],
        trajectory.object_pos[790],
    )
    _, _, next_done, _ = env.step(zero)
    np.testing.assert_array_equal(next_done, [False])
    assert env.last_transition is not None
    np.testing.assert_array_equal(env.last_transition.reset_applied, [True])
    np.testing.assert_array_equal(env.last_transition.target_indices, [0])
    np.testing.assert_array_equal(env.progress, [0])
    np.testing.assert_array_equal(env.trajectory_steps, [0])
    assert env.last_physical is not None
    np.testing.assert_allclose(env.last_physical.object_position[0], trajectory.object_pos[0], atol=1e-7)


def test_compiled_floor_uses_checkerboard_material(trajectory) -> None:
    import mujoco

    env = _environment(trajectory)
    floor_id = env.model.geom("floor").id
    material_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_MATERIAL, "floor_checker"
    )
    texture_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_TEXTURE, "floor_checker"
    )

    assert material_id >= 0 and texture_id >= 0
    assert env.model.geom_matid[floor_id] == material_id
    assert texture_id in env.model.mat_texid[material_id]
    np.testing.assert_allclose(
        env.model.mat_texrepeat[material_id], (8.0, 8.0), rtol=0.0, atol=0.0
    )
    assert env.model.mat_texuniform[material_id]


def test_object_point_cloud_world_uses_metric_template_and_object_pose(trajectory) -> None:
    env = _environment(trajectory)
    env.step(np.zeros((1, env.action_dim), dtype=np.float64))
    assert env.last_physical is not None
    world_cloud = env.object_point_cloud_world()
    template = env._point_template()
    local = np.asarray(template.local_points, dtype=np.float64)
    if template.normalized:
        local = local * np.asarray(template.scale, dtype=np.float64)
    expected = quat_rotate_xyzw(
        np.broadcast_to(env.last_physical.object_orientation_xyzw[:, None, :], (1, 64, 4)),
        np.broadcast_to(local, (1, 64, 3)),
    ) + env.last_physical.object_position[:, None, :]
    np.testing.assert_allclose(world_cloud, expected, rtol=0.0, atol=1e-12)
    assert env.last_observation is not None
    normalized_hand_relative = env.last_observation.raw[
        :, env.observation_layout.slices["object_point_cloud_raw"]
    ].reshape(1, 64, 3)
    recovered_world = normalized_hand_relative + env.last_physical.hand_position[:, None, :]
    if template.normalized:
        assert not np.allclose(world_cloud, recovered_world)
    else:
        np.testing.assert_allclose(world_cloud, recovered_world, rtol=0.0, atol=1e-12)


def test_transition_snapshot_preserves_action_reference_and_partial_rerun_close(trajectory, tmp_path) -> None:
    from sim.manorl.rerun_recorder import ManoRerunRecorder

    env = _environment(trajectory)
    action = np.zeros((1, env.action_dim), dtype=np.float64)
    action[0, 1] = 0.25
    env.step(action)
    snapshot = env.last_transition
    assert snapshot is not None
    assert snapshot.control_call == 0
    np.testing.assert_allclose(snapshot.raw_actions, env.normalize_actions(action))
    np.testing.assert_array_equal(snapshot.command_reference_indices, [0])
    np.testing.assert_allclose(snapshot.command_targets, env.reference_q_model[:, [0]][:, 0])
    np.testing.assert_allclose(snapshot.controller_targets, env.last_controller_targets)
    recorder = ManoRerunRecorder(env, tmp_path / "env0.rrd")
    recorder.record_transition()
    assert recorder.close() is None
    assert not recorder.output.exists()
    assert not recorder.active_path.exists()


def test_rerun_blueprint_and_transition_context_default_to_step(trajectory) -> None:
    import rerun as rr

    from sim.manorl.contracts import CONTROL_TIMESTEP
    from sim.manorl.rerun_recorder import ManoRerunRecorder

    class RecordingStream:
        def __init__(self) -> None:
            self.time_context: dict[str, dict[str, float | int]] = {}
            self.logs: list[tuple[str, dict[str, dict[str, float | int]]]] = []

        def set_time(self, timeline: str, **kwargs: float | int) -> None:
            self.time_context[timeline] = kwargs

        def log(self, entity_path: str, *args, **kwargs) -> None:
            self.logs.append((entity_path, self.time_context.copy()))

    env = _environment(trajectory)
    env.step(np.zeros((1, env.action_dim), dtype=np.float64))
    env.step(np.zeros((1, env.action_dim), dtype=np.float64))
    snapshot = env.last_transition
    assert snapshot is not None

    recorder = object.__new__(ManoRerunRecorder)
    recorder.rr = rr
    recorder.environment = env
    recorder.env_id = 0
    recorder.episode_id = 0
    recorder.geometry_table = recorder._geometry_table()
    recorder.geometry_series_names = tuple(row["series_name"] for row in recorder.geometry_table)
    blueprint = recorder._default_blueprint()
    assert isinstance(blueprint.time_panel, rr.blueprint.TimePanel)
    assert blueprint.time_panel.timeline == "step"
    hand_object_view = blueprint.root_container.contents[3]
    assert hand_object_view.contents == [
        "contact/hand_object_force/on_object/magnitude_N",
        "contact/object/gravity/world/magnitude_N",
    ]

    recorder.recording = RecordingStream()
    recorder._record(snapshot)
    expected_context = {
        "step": {"sequence": snapshot.control_call},
        "simulation": {"duration": snapshot.control_call * CONTROL_TIMESTEP},
    }
    assert recorder.recording.logs
    assert all(context == expected_context for _, context in recorder.recording.logs)


def test_rerun_logs_urdf_resolved_mano_meshes_and_dynamic_link_transforms(trajectory) -> None:
    import rerun as rr

    from sim.manorl.rerun_recorder import ManoRerunRecorder

    class RecordingStream:
        def __init__(self) -> None:
            self.logs: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

        def set_time(self, timeline: str, **kwargs: float | int) -> None:
            del timeline, kwargs

        def log(self, entity_path: str, *args: object, **kwargs: object) -> None:
            self.logs.append((entity_path, args, kwargs))

    env = _environment(trajectory)
    env.step(np.zeros((1, env.action_dim), dtype=np.float64))
    assert env.last_transition is not None
    recorder = object.__new__(ManoRerunRecorder)
    recorder.rr = rr
    recorder.environment = env
    recorder.env_id = 0
    recorder.episode_id = 0
    recorder.geometry_table = recorder._geometry_table()
    recorder.geometry_series_names = tuple(row["series_name"] for row in recorder.geometry_table)
    recorder.hand_object_force_series_table = recorder._hand_object_force_series_table()
    recorder.hand_object_force_series_names = tuple(
        row["series_name"] for row in recorder.hand_object_force_series_table
    )
    recorder.hand_meshes = recorder._hand_meshes()
    assert len(recorder.hand_meshes) == len(KEYPOINT_NAMES) == 16
    assert all(mesh["vertex_count"] > 0 and mesh["triangle_count"] > 0 for mesh in recorder.hand_meshes)
    assert env.last_physical is not None
    geom_positions = np.asarray(env.data.geom_xpos, dtype=np.float64)[0]
    geom_rotations = np.asarray(env.data.geom_xmat, dtype=np.float64)[0]
    for keypoint_id, mesh in enumerate(recorder.hand_meshes):
        geom_id = int(mesh["geom_id"])
        mesh_id = int(mesh["mesh_id"])
        first_vertex = int(env.model.mesh_vertadr[mesh_id])
        source_vertex = np.asarray(env.model.mesh_vert[first_vertex], dtype=np.float64)
        expected_world_vertex = geom_rotations[geom_id] @ source_vertex + geom_positions[geom_id]
        body_orientation = env.last_physical.hand_keypoint_orientations_xyzw[0, keypoint_id]
        body_position = env.last_physical.hand_keypoint_positions[0, keypoint_id]
        recorded_world_vertex = quat_rotate_xyzw(
            body_orientation[None], np.asarray(mesh["vertices"], dtype=np.float64)[:1]
        )[0] + body_position
        np.testing.assert_allclose(recorded_world_vertex, expected_world_vertex, rtol=0.0, atol=1e-7)

    recorder.recording = RecordingStream()
    recorder._log_static_metadata()
    static_mesh_logs = [
        (path, args[0])
        for path, args, kwargs in recorder.recording.logs
        if kwargs.get("static") is True and args and isinstance(args[0], rr.Mesh3D)
    ]
    assert len(static_mesh_logs) == 16
    assert all(mesh.vertex_positions.as_arrow_array().to_pylist() for _, mesh in static_mesh_logs)
    assert all(mesh.triangle_indices.as_arrow_array().to_pylist() for _, mesh in static_mesh_logs)
    static_mesh_paths = {path for path, _ in static_mesh_logs}
    assert static_mesh_paths == {f"world/mano_hand/{name}" for name in KEYPOINT_NAMES}

    recorder._record(env.last_transition)
    transform_logs = [
        path
        for path, args, kwargs in recorder.recording.logs
        if path.startswith("world/mano_hand/") and args and isinstance(args[0], rr.Transform3D)
    ]
    assert len(transform_logs) == 16
    assert set(transform_logs) == static_mesh_paths


def test_rerun_geometry_force_series_metadata_and_continuity(trajectory) -> None:
    import rerun as rr

    from sim.manorl.rerun_recorder import ManoRerunRecorder

    class RecordingStream:
        def __init__(self) -> None:
            self.time_context: dict[str, dict[str, float | int]] = {}
            self.logs: list[tuple[str, tuple[object, ...], dict[str, object], dict[str, dict[str, float | int]]]] = []

        def set_time(self, timeline: str, **kwargs: float | int) -> None:
            self.time_context[timeline] = kwargs

        def log(self, entity_path: str, *args: object, **kwargs: object) -> None:
            self.logs.append((entity_path, args, kwargs, self.time_context.copy()))

    env = _environment(trajectory)
    recorder = object.__new__(ManoRerunRecorder)
    recorder.rr = rr
    recorder.environment = env
    recorder.env_id = 0
    recorder.episode_id = 0
    recorder.geometry_table = recorder._geometry_table()
    recorder.geometry_series_names = tuple(row["series_name"] for row in recorder.geometry_table)
    recorder.hand_object_force_series_table = recorder._hand_object_force_series_table()
    recorder.hand_object_force_series_names = tuple(
        row["series_name"] for row in recorder.hand_object_force_series_table
    )
    assert len(recorder.geometry_table) == env.model.ngeom
    assert [row["keypoint_name"] for row in recorder.hand_object_force_series_table] == list(KEYPOINT_NAMES)
    assert [row["geom_id"] for row in recorder.hand_object_force_series_table] == env.producer.keypoint_geom_ids
    assert recorder.geometry_table[0] == {
        "id": 0,
        "label": "floor",
        "series_name": "000:floor",
        "geom_type": "mjGEOM_PLANE",
        "body_id": 0,
        "body_name": "world",
        "mesh_name": None,
    }
    recorder.recording = RecordingStream()
    recorder._log_static_metadata()
    blueprint = recorder._default_blueprint()
    root = blueprint.root_container
    assert root.contents[0].contents == ["world/**", "target/**"]
    tabs = root.contents[2]
    assert isinstance(tabs, rr.blueprint.Tabs)
    assert tabs.name == "Collision geometry contact forces"
    assert [view.name for view in tabs.contents] == [
        "Net magnitude", "World F_x", "World F_y", "World F_z", "Object gravity"
    ]
    assert root.contents[1].contents[1].contents == [
        "contact/count", "contact/object_force_magnitude", "contact/keypoint_force_total", "action/**", "episode/**"
    ]
    hand_object_path = "contact/hand_object_force/on_object/magnitude_N"
    gravity_magnitude_path = "contact/object/gravity/world/magnitude_N"
    hand_object_view = root.contents[3]
    assert hand_object_view.name == "ManoHand-object contact forces"
    assert hand_object_view.contents == [hand_object_path, gravity_magnitude_path]
    gravity_series = hand_object_view.visualizer_overrides[gravity_magnitude_path]
    assert gravity_series.names.as_arrow_array().to_pylist() == ["Object gravity"]
    assert len(recorder.hand_object_force_series_names) + len(
        gravity_series.names.as_arrow_array().to_pylist()
    ) == len(KEYPOINT_NAMES) + 1 == 17
    assert tabs.contents[-1].contents == ["contact/object/gravity/world/**"]
    assert all(hand_object_path not in view.contents for view in tabs.contents)
    assert gravity_magnitude_path not in root.contents[1].contents[1].contents
    assert hand_object_path not in root.contents[1].contents[1].contents

    expected_hand_object_magnitudes = []
    expected_episode_returns = []
    for _ in range(2):
        env.step(np.zeros((1, env.action_dim), dtype=np.float64))
        assert env.last_transition is not None
        assert env.last_physical is not None
        expected_hand_object_magnitudes.append(
            np.linalg.norm(env.last_physical.hand_object_force_on_object_world_N[0], axis=1)
        )
        expected_episode_returns.append(float(env.last_transition.episode_return[0]))
        recorder._record(env.last_transition)

    geometry_paths = [f"contact/geometry_force/world/{component}" for component, _ in (
        ("magnitude_N", ""), ("x_N", ""), ("y_N", ""), ("z_N", "")
    )]
    static_logs = {
        logged_path: args[0]
        for logged_path, args, kwargs, _ in recorder.recording.logs
        if kwargs.get("static")
    }
    assert set(geometry_paths).issubset(static_logs)
    for path in geometry_paths:
        assert static_logs[path].names.as_arrow_array().to_pylist() == list(recorder.geometry_series_names)
    assert static_logs[hand_object_path].names.as_arrow_array().to_pylist() == list(
        recorder.hand_object_force_series_names
    )
    assert len(recorder.hand_object_force_series_names) == len(KEYPOINT_NAMES)
    metadata_document = next(
        args[0]
        for path, args, kwargs, _ in recorder.recording.logs
        if path == "run/metadata" and kwargs.get("static")
    )
    metadata = json.loads(metadata_document.text.as_arrow_array().to_pylist()[0])
    assert metadata["hand_object_force_on_object_world_N"] == {
        "direction": "net world-frame force exerted on the object by each hand collision geom",
        "filter": "only contact rows with exactly one source-mapped hand collision geom and one object collision geom",
        "series": recorder.hand_object_force_series_table,
    }
    assert metadata["reward_contract"] == REWARD_CONTRACT_ID
    assert metadata["ppo_reward_contract"] == PPO_REWARD_CONTRACT_ID
    assert metadata["ppo_reward_scale"] == 0.5
    assert metadata["environment_contract"] == ENVIRONMENT_CONTRACT_ID
    assert metadata["residual_action"]["position_scale"] == [0.002, 0.002, 0.002]
    assert metadata["residual_action"]["max_position_offset"] == [0.02, 0.02, 0.02]
    assert (
        metadata["thresholds"]["observation_contact_threshold_N"]
        == metadata["thresholds"]["reward_hand_object_threshold_N"]
        == env.config.reward_config.contact_force_threshold
        == REWARD_HAND_OBJECT_THRESHOLD_N
        == 0.2
    )
    assert "contact_force_threshold" not in metadata["thresholds"]
    episode_return_samples = [
        args[0].scalars.as_arrow_array().to_numpy()[0]
        for logged_path, args, kwargs, _ in recorder.recording.logs
        if logged_path == "episode/return" and not kwargs.get("static")
    ]
    np.testing.assert_allclose(episode_return_samples, expected_episode_returns, rtol=0, atol=1e-12)

    samples = {
        path: [args[0] for logged_path, args, kwargs, _ in recorder.recording.logs if logged_path == path and not kwargs.get("static")]
        for path in geometry_paths
    }
    assert all(len(values) == 2 for values in samples.values())
    hand_object_samples = [
        args[0]
        for logged_path, args, kwargs, _ in recorder.recording.logs
        if logged_path == hand_object_path and not kwargs.get("static")
    ]
    assert len(hand_object_samples) == 2
    for value, expected_magnitudes in zip(hand_object_samples, expected_hand_object_magnitudes, strict=True):
        actual = value.scalars.as_arrow_array().to_numpy()
        assert actual.shape == (len(KEYPOINT_NAMES),)
        np.testing.assert_allclose(actual, expected_magnitudes, rtol=0, atol=1e-12)
    assert np.any(np.stack(expected_hand_object_magnitudes) == 0.0)
    for values in samples.values():
        assert all(value.scalars.as_arrow_array().to_numpy().shape == (env.model.ngeom,) for value in values)
    zero_geom_ids = np.flatnonzero(
        np.all(
            np.stack([value.scalars.as_arrow_array().to_numpy() for value in samples["contact/geometry_force/world/magnitude_N"]]),
            axis=0,
        )
        == 0.0
    )
    assert len(zero_geom_ids) > 0
    for component in geometry_paths:
        for values in samples[component]:
            assert np.all(values.scalars.as_arrow_array().to_numpy()[zero_geom_ids] == 0.0)

    gravity_logs = {
        path: [args[0] for logged_path, args, kwargs, _ in recorder.recording.logs if logged_path == path and not kwargs.get("static")]
        for path in (
            "contact/object/gravity/world/x_N",
            "contact/object/gravity/world/y_N",
            "contact/object/gravity/world/z_N",
            gravity_magnitude_path,
        )
    }
    assert all(len(samples) == 2 for samples in gravity_logs.values())
    np.testing.assert_allclose(
        gravity_logs["contact/object/gravity/world/x_N"][-1].scalars.as_arrow_array().to_numpy(), (0.0,)
    )
    np.testing.assert_allclose(
        gravity_logs["contact/object/gravity/world/y_N"][-1].scalars.as_arrow_array().to_numpy(), (0.0,)
    )
    np.testing.assert_allclose(
        gravity_logs["contact/object/gravity/world/z_N"][-1].scalars.as_arrow_array().to_numpy(), (-1.22625,)
    )
    np.testing.assert_allclose(
        gravity_logs[gravity_magnitude_path][-1].scalars.as_arrow_array().to_numpy(), (1.22625,)
    )
    assert gravity_magnitude_path not in static_logs


def test_rerun_partial_close_preserves_existing_stable_artifact(trajectory, tmp_path) -> None:
    from sim.manorl.rerun_recorder import ManoRerunRecorder

    output = tmp_path / "episodes.rrd"
    sentinel = b"stable episode must survive partial close"
    output.write_bytes(sentinel)
    env = _environment(trajectory)
    recorder = ManoRerunRecorder(env, output)
    env.step(np.zeros((1, env.action_dim), dtype=np.float64))
    recorder.record_transition()

    assert recorder.close() is None
    assert output.read_bytes() == sentinel
    assert not recorder.active_path.exists()


def test_rerun_existing_stable_artifact_is_replaced_after_complete_episode(trajectory, tmp_path) -> None:
    from sim.manorl.rerun_recorder import ManoRerunRecorder

    output = tmp_path / "episodes.rrd"
    output.write_bytes(b"previous recording")
    env = _environment(trajectory)
    recorder = ManoRerunRecorder(env, output)
    env.progress[:] = len(trajectory.q_ref) - 2
    env.trajectory_steps[:] = len(trajectory.q_ref) - 3

    env.step(np.zeros((1, env.action_dim), dtype=np.float64))
    recorder.record_transition()
    env.step(np.zeros((1, env.action_dim), dtype=np.float64))
    recorder.record_transition()

    assert output.read_bytes() != b"previous recording"
    assert output.read_bytes()
    recorder.close()


def test_rerun_stale_active_stream_is_removed_before_new_recording(trajectory, tmp_path) -> None:
    from sim.manorl.rerun_recorder import ManoRerunRecorder

    output = tmp_path / "episodes.rrd"
    active = tmp_path / ".episodes.active.rrd"
    active.write_bytes(b"stale active stream")
    recorder = ManoRerunRecorder(_environment(trajectory), output)

    assert recorder.active_path == active
    assert active.exists()
    assert active.read_bytes() != b"stale active stream"
    recorder.close()


def test_rerun_close_disconnects_fresh_active_stream_once(tmp_path) -> None:
    from sim.manorl.rerun_recorder import ManoRerunRecorder

    class RecordingStream:
        def __init__(self) -> None:
            self.flushes = 0
            self.disconnects = 0

        def flush(self) -> None:
            self.flushes += 1

        def disconnect(self) -> None:
            self.disconnects += 1

    recorder = object.__new__(ManoRerunRecorder)
    recorder.output = tmp_path / "episodes.rrd"
    recorder.active_path = tmp_path / ".episodes.active.rrd"
    recorder.active_path.write_bytes(b"fresh active episode")
    recorder.recording = RecordingStream()
    recorder._published = True
    recorder._recording_open = True
    recorder._closed = False
    recorder._close_result = None

    assert recorder.close() == recorder.output
    assert recorder.close() == recorder.output
    assert recorder.recording.flushes == 1
    assert recorder.recording.disconnects == 1
    assert not recorder.active_path.exists()


def test_rerun_finalizes_terminal_episode_without_reset_tick(trajectory, tmp_path) -> None:
    from sim.manorl.rerun_recorder import ManoRerunRecorder

    env = _environment(trajectory)
    recorder = ManoRerunRecorder(env, tmp_path / "episodes.rrd")
    env.progress[:] = len(trajectory.q_ref) - 2
    env.trajectory_steps[:] = len(trajectory.q_ref) - 3
    env.step(np.zeros((1, env.action_dim), dtype=np.float64))
    assert env.last_transition is not None
    assert bool(env.last_transition.termination.reset[0])
    recorder.record_transition()
    assert recorder.output.name == "episodes.rrd"
    stable_bytes = recorder.output.read_bytes()
    assert stable_bytes
    assert recorder.active_path.exists()

    assert recorder.close() == recorder.output
    assert recorder.close() == recorder.output
    assert recorder.output.read_bytes() == stable_bytes
    assert not recorder.active_path.exists()


def test_residual_core_masks_inactive_fingers_in_live_environment(trajectory) -> None:
    env = _environment(trajectory, residual_enabled=True)
    env.progress[:] = 101
    env.trajectory_steps[:] = 100
    env.step(np.ones((1, JOINT_DOF), dtype=np.float64))
    assert np.all(np.abs(env.cumulative_joint_offset[0, :10]) > 0.0)
    np.testing.assert_allclose(env.cumulative_joint_offset[0, 10:], 0.0)
    assert np.all(
        np.abs(
            env.last_controller_targets[0, :3]
            - env.reference_q_by_side["right"][0, 100, :3]
        )
        > 0.0
    )


def test_two_world_cpu_vector_smoke_has_independent_equal_worlds(trajectory) -> None:
    env = _environment(trajectory, num_envs=2)
    zero = np.zeros((2, env.action_dim), dtype=np.float64)
    for _ in range(3):
        observation, reward, reset, extras = env.step(zero)
        assert observation["obs"].shape == (2, env.observation_dim) == (2, 480)
        assert reward.shape == reset.shape == extras["time_outs"].shape == (2,)
        point_slice = env.observation_layout.slices["object_point_cloud_raw"]
        np.testing.assert_allclose(
            observation["obs"][0, : point_slice.start],
            observation["obs"][1, : point_slice.start],
            rtol=0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            observation["obs"][0, point_slice.stop :],
            observation["obs"][1, point_slice.stop :],
            rtol=0,
            atol=1e-10,
        )
        assert not np.array_equal(observation["obs"][0, point_slice], observation["obs"][1, point_slice])
        np.testing.assert_allclose(reward[0], reward[1], rtol=0, atol=1e-10)
    assert env.last_physical is not None
    assert np.all(env.last_physical.contact_count > 0)
    np.testing.assert_allclose(
        env.last_physical.hand_keypoint_contact_forces[0], env.last_physical.hand_keypoint_contact_forces[1], rtol=0, atol=1e-10
    )


def test_heterogeneous_object_router_preserves_global_order_and_indexed_reset(trajectory) -> None:
    cube2_shift = _initial_support_shift(
        trajectory.object_pos_raw[0], trajectory.object_quat_xyzw[0], "cube2"
    )
    cube2_position = trajectory.object_pos_raw.copy()
    cube2_position[:, 2] += cube2_shift
    cube2 = replace(
        trajectory,
        identity=replace(trajectory.identity, identity="cube2_01_003"),
        object_pos=cube2_position,
        object_z_shift=cube2_shift,
    )
    env = MujocoManoEnvironment(
        TrajectoryBatch((trajectory, cube2, trajectory)),
        EnvironmentConfig(
            num_envs=3,
            residual_enabled=False,
            max_deviation_distance=1_000_000.0,
        ),
    )

    assert env.is_heterogeneous
    assert env.object_types == ("cube1", "cube2", "cube1")
    assert tuple(env._object_routes) == ("cube1", "cube2")
    np.testing.assert_array_equal(env._object_routes["cube1"][0], (0, 2))
    np.testing.assert_array_equal(env._object_routes["cube2"][0], (1,))
    assert env.object_geometry.shape == (3, 12)
    assert not np.array_equal(env.object_geometry[0], env.object_geometry[1])

    observation, reward, reset, extras = env.step(
        np.zeros((3, env.action_dim), dtype=np.float64)
    )
    assert observation["obs"].shape == (3, env.observation_dim) == (3, 480)
    assert reward.shape == reset.shape == extras["time_outs"].shape == (3,)
    np.testing.assert_array_equal(env.progress, (1, 1, 1))
    point_slice = env.observation_layout.slices["object_point_cloud_raw"]
    np.testing.assert_allclose(
        observation["obs"][0, : point_slice.start],
        observation["obs"][2, : point_slice.start],
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        observation["obs"][0, point_slice.stop :],
        observation["obs"][2, point_slice.stop :],
        rtol=0,
        atol=1e-10,
    )
    assert not np.array_equal(
        observation["obs"][0, point_slice], observation["obs"][2, point_slice]
    )

    reset_observation = env.reset(np.asarray([1], dtype=np.int64))
    assert reset_observation["obs"].shape == (3, env.observation_dim) == (3, 480)
    np.testing.assert_array_equal(env.progress, (1, 0, 1))
    assert env.last_physical is not None
    assert env.last_physical.object_position.shape == (3, 3)


def test_dynamic_template_variant_preserves_raw_surface_coordinates(trajectory) -> None:
    env = MujocoManoEnvironment(
        trajectory,
        EnvironmentConfig(
            compatibility=SOURCE_ALIGNED_COMPATIBILITY,
            residual_enabled=False,
            max_deviation_distance=1_000_000.0,
        ),
    )
    assert env._dynamic_templates is not None
    first = env._dynamic_templates.copy()
    assert np.all(np.abs(first).max(axis=(1, 2)) < 0.1)
    env.reset(np.asarray([0], dtype=np.int64))
    assert env._dynamic_templates is not None
    assert not np.array_equal(first, env._dynamic_templates)
