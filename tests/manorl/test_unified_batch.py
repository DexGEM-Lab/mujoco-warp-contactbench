from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sim.manorl.assets import (
    build_unified_scene_xml,
    compile_unified_model,
    object_runtime,
)
from sim.manorl.contracts import JOINT_DOF, TRAJECTORY_IDENTITY
from sim.manorl.environment import EnvironmentConfig, MujocoManoEnvironment
from sim.manorl.observations import CURRENT_SOURCE_COMPATIBILITY
from sim.manorl.trajectory import ReferenceTrajectory, TrajectoryBatch
from sim.manorl.unified_batch import UnifiedBatchConfig


def _require_materialized_objects() -> None:
    for object_type in ("cube1", "cube2"):
        try:
            runtime = object_runtime(object_type)
        except (FileNotFoundError, ValueError) as exc:
            pytest.skip(f"authoritative {object_type} assets are unavailable: {exc}")
        if not runtime.urdf_path.is_file() or not runtime.collision_mesh_path.is_file():
            pytest.skip(f"authoritative {object_type} assets are unavailable")


def _trajectory(object_type: str) -> ReferenceTrajectory:
    identity = replace(TRAJECTORY_IDENTITY, identity=f"{object_type}_01_000", source_start=0, source_stop=2, movement_start_raw=0, movement_end_raw=1)
    q_ref = np.zeros((2, 26), dtype=np.float64)
    object_pos = np.asarray(((0.0, 0.0, 0.5), (0.0, 0.0, 0.5)), dtype=np.float64)
    return ReferenceTrajectory(
        identity=identity,
        dataset_version=identity.dataset_version,
        source_indices=np.asarray((0, 1), dtype=np.int64),
        timestamps=np.asarray((0.0, 0.01), dtype=np.float64),
        q_ref=q_ref,
        object_pos_raw=object_pos,
        object_pos=object_pos,
        object_quat_xyzw=np.asarray(((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)), dtype=np.float64),
        object_z_shift=0.0,
    )


@pytest.mark.parametrize(("alias", "expected"), [("r", "right"), ("lhand", "left"), ("bimanual", "both")])
def test_unified_batch_config_canonicalizes_hand_side_aliases(
    alias: str, expected: str
) -> None:
    config = UnifiedBatchConfig(
        object_types=("cube1",),
        active_object_indices=(0,),
        hand_side=alias,
    )
    assert config.hand_side == expected


def test_unified_scene_preserves_real_meshes_and_fixed_hand_contract() -> None:
    _require_materialized_objects()
    xml = build_unified_scene_xml(object_types=("cube1", "cube2"))
    assert 'model="manorl_unified"' in xml
    assert xml.count('name="cube1_collision"') == 1
    assert xml.count('name="cube2_collision"') == 1
    mujoco, model = compile_unified_model(object_types=("cube1", "cube2"))
    # Two free object joints add 14 qpos / 12 qvel coordinates to the revised
    # 28-DoF hand model.
    assert (model.nq, model.nv, model.nu) == (42, 40, JOINT_DOF)
    for object_type in ("cube1", "cube2"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_type)
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{object_type}_collision")
        assert body >= 0 and geom >= 0
        assert model.body_gravcomp[body] == 0
        assert model.geom_contype[geom] == 2
        assert model.geom_conaffinity[geom] == 5


@pytest.mark.mjx_warp
def test_unified_mixed_environment_matches_route_and_indexed_reset() -> None:
    _require_materialized_objects()
    batch = TrajectoryBatch((_trajectory("cube1"), _trajectory("cube2")))
    common = dict(
        num_envs=2,
        device="cpu",
        residual_enabled=False,
        compatibility=CURRENT_SOURCE_COMPATIBILITY,
        max_deviation_distance=1_000_000.0,
        contact_capacity=256,
        constraint_capacity=1024,
    )
    routed = MujocoManoEnvironment(batch, EnvironmentConfig(**common))
    unified = MujocoManoEnvironment(
        batch, EnvironmentConfig(**common, unified_object_batch=True)
    )
    assert unified._object_routes == {}
    assert unified.model.nq == 42
    assert unified.data.qpos.shape == (2, 42)

    actions = np.zeros((2, JOINT_DOF), dtype=np.float64)
    routed_output = routed.step(actions)
    unified_output = unified.step(actions)
    np.testing.assert_allclose(routed_output[0]["obs"], unified_output[0]["obs"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(routed_output[1], unified_output[1], rtol=0, atol=1e-8)
    np.testing.assert_array_equal(routed_output[2], unified_output[2])
    assert routed.last_physical is not None and unified.last_physical is not None
    np.testing.assert_allclose(
        routed.last_physical.object_position,
        unified.last_physical.object_position,
        rtol=0,
        atol=1e-8,
    )

    reset_output = unified.reset(np.asarray([1], dtype=np.int64))
    assert reset_output["obs"].shape == (2, 480)
    np.testing.assert_array_equal(unified.progress, (1, 0))


def test_scene_model_enables_object_object_contact_only_when_requested():
    _require_materialized_objects()
    mj, model = compile_unified_model(object_types=("cube1", "cube2"), object_collisions=True)
    data = mj.MjData(model)
    data.qpos[:3] = [0., 0., 5.]  # Move hand out of contact probe.
    bodies = []
    for name in ("cube1", "cube2"):
        runtime = object_runtime(name)
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, runtime.body_name)
        bodies.append(body)
        joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, runtime.free_joint_name)
        address = model.jnt_qposadr[joint]
        data.qpos[address:address+3] = [.0, .0, .5]
        data.qpos[address+3:address+7] = [1., 0., 0., 0.]
    mj.mj_forward(model, data)
    assert any(set(model.geom_bodyid[c.geom]) == set(bodies) for c in data.contact)


def test_scene_reset_places_present_bodies_and_parks_only_absent_types():
    from types import SimpleNamespace
    from sim.manorl.trajectory import xyzw_to_wxyz
    active = _trajectory("cube2")
    scene = replace(
        active,
        identity=replace(active.identity, object_index=1),
        scene_object_types=("cube1", "cube2"),
        scene_object_initial_pos=np.array([[.1,.2,.3], [.0,.0,.5]]),
        scene_object_initial_quat_xyzw=np.tile([0.,0.,0.,1.], (2,1)),
    )
    env = object.__new__(MujocoManoEnvironment)
    env.config = SimpleNamespace(num_envs=2, object_init_xy_offset_range_m=0., object_init_xy_offsets_m=None)
    env.model = SimpleNamespace(nq=49)
    env.model_hand_sides = ("right",)
    env.hand_dof = 28
    env.reference_q_by_side = {"right": np.zeros((2, 2, 28))}
    env.reference_object_pos = np.stack([scene.object_pos, active.object_pos])
    env.reference_object_quat_xyzw = np.stack([scene.object_quat_xyzw, active.object_quat_xyzw])
    env.trajectories = (scene, active)
    env.object_types = ("cube2", "cube2")
    env._unified_object_batch = True
    env._unified_object_types = ("cube1", "cube2", "bowl")
    env._unified_qpos_addresses = np.array([28,35,42])
    qpos = env._initial_qpos()
    np.testing.assert_allclose(qpos[0, 28:31], [.1,.2,.3])
    np.testing.assert_allclose(qpos[:, 35:38], [[0.,0.,.5], [0.,0.,.5]])
    np.testing.assert_allclose(qpos[0,31:35], xyzw_to_wxyz(scene.scene_object_initial_quat_xyzw[0]))
    assert qpos[1,28] >= 1000  # no passive cube1 in world 1
    assert np.all(qpos[:,42] >= 1000)  # bowl absent from both worlds
