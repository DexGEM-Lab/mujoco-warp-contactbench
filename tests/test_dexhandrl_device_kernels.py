from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from sim.dexhandrl.contact import collect_contact_summary
from sim.dexhandrl.device_contacts import reduce_dexhand_contacts
from sim.dexhandrl.device_task import (
    apply_device_control,
    build_device_observation,
    compute_device_reward_done,
    gather_trajectory,
    route_active_to_full,
)
from sim.dexhandrl.hand_mapping import active_to_full_dof
from sim.dexhandrl.parity import build_minimal_observation
from sim.dexhandrl.skrl_env import apply_active_target_post_filters


def _contact_inputs(batch: int = 1):
    return dict(
        nacon=jnp.asarray(1, jnp.int32), nefc=jnp.full((batch,), 4, jnp.int32),
        contact_geom=jnp.asarray([[0, 1], [0, 0]], jnp.int32),
        contact_worldid=jnp.asarray([0, 0], jnp.int32), contact_dim=jnp.asarray([3, 3], jnp.int32),
        contact_efc_address=jnp.asarray([[0, 1, 2, 3], [0, 1, 2, 3]], jnp.int32),
        contact_friction=jnp.asarray([[.5, .25, 0, 0, 0], [.5, .25, 0, 0, 0]], jnp.float32),
        contact_frame=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (2, 3, 3)),
        efc_force=jnp.tile(jnp.asarray([[2., 1., 4., 2., 0., 0., 0., 0.]], jnp.float32), (batch, 1)),
        ngeom=2, geom_to_contact_body=np.asarray([6, -1], np.int32), object_geom_flags=np.asarray([False, True]),
    )


def test_device_contact_matches_scalar_summary_and_jit() -> None:
    kw = _contact_inputs()
    dynamic = {name: value for name, value in kw.items() if name not in {"ngeom", "geom_to_contact_body", "object_geom_flags"}}
    result = jax.jit(lambda **values: reduce_dexhand_contacts(
        **values, ngeom=2, geom_to_contact_body=np.asarray([6, -1], np.int32), object_geom_flags=np.asarray([False, True])
    ))(**dynamic)
    scalar = collect_contact_summary(
        active_contacts=1, contact_geom=np.asarray([[0, 1]]), contact_frame=np.eye(3, dtype=np.float32)[None],
        contact_friction=np.asarray([[.5, .25, 0, 0, 0]], np.float32), contact_dim=np.asarray([3]),
        contact_efc_address=np.asarray([[0, 1, 2, 3]], np.int32), efc_force=np.asarray([2., 1., 4., 2.]),
        pyramidal=True, geom_body_names={0: "RH1_3", 1: "cube1"}, geom_names={},
        contact_body_names=("RFH1", "RH0_0", "RH0_2", "RH0_3", "RH1_0", "RH1_2", "RH1_3", "RH2_0", "RH2_2", "RH2_3", "RH3_0", "RH3_2", "RH3_3", "RH4_0", "RH4_2", "RH4_3"), object_name="cube1",
    )
    np.testing.assert_allclose(np.asarray(result.hand)[0], scalar["contact_force_vectors"])
    np.testing.assert_allclose(np.asarray(result.object)[0], scalar["object_force_vector"])
    assert bool(result.valid) and int(result.count[0]) == 1


@pytest.mark.parametrize("nacon,address,valid", [(0, [0, 1, 2, 3], True), (1, [-1, 1, 2, 3], False), (2, [0, 1, 2, 3], False)])
def test_device_contact_fails_closed_for_empty_bad_or_capacity(nacon, address, valid) -> None:
    kw = _contact_inputs()
    kw["nacon"] = jnp.asarray(nacon, jnp.int32)
    kw["contact_efc_address"] = kw["contact_efc_address"].at[0].set(jnp.asarray(address))
    result = reduce_dexhand_contacts(**kw)
    assert bool(result.valid) is valid
    if nacon == 0:
        np.testing.assert_array_equal(np.asarray(result.hand), 0)


def test_device_contact_multiverse_scatter() -> None:
    kw = _contact_inputs(7)
    kw["contact_worldid"] = jnp.asarray([6, 0], jnp.int32)
    result = reduce_dexhand_contacts(**kw)
    assert bool(result.valid)
    np.testing.assert_allclose(np.asarray(result.hand)[6, 6], [-9, -.5, -.5])
    np.testing.assert_array_equal(np.asarray(result.hand)[:6], 0)


def test_device_route_and_filters_match_numpy_helpers() -> None:
    rng = np.random.default_rng(4)
    active = rng.normal(size=(7, 22)).astype(np.float32)
    current = rng.uniform(0, .1, size=(7, 26)).astype(np.float32)
    upper = np.full((4, 2), 1.5, np.float32)
    actual = route_active_to_full(active=jnp.asarray(active), current_full=jnp.asarray(current), route_upper=jnp.asarray(upper))
    expected = np.stack([active_to_full_dof(a, c, {6:(1.5,1.5),9:(1.5,1.5),12:(1.5,1.5),15:(1.5,1.5)}) for a,c in zip(active,current)])
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-6, atol=2e-6)
    previous = np.zeros((7,22),np.float32); corrected=np.ones((7,22),np.float32)
    ctrl = apply_device_control(action=jnp.zeros((7,22)), reference_active=jnp.asarray(corrected), current_full=jnp.asarray(current), previous_active=jnp.asarray(previous), base_offset=jnp.zeros((7,3)), joint_offset=jnp.zeros((7,16)), active_joint_mask=jnp.ones((7,16)), frame=jnp.arange(7), start_frame=jnp.zeros(7,jnp.int32), last_contact_frame=jnp.full(7,100), use_residual=jnp.zeros(7,bool), early_phase_frames=130, base_decay=jnp.ones(3), post_contact_base_decay=jnp.ones(3), joint_decay=1., post_contact_joint_decay=1., base_scales=jnp.ones(3), base_max_offset=.04, active_max_deltas=jnp.full(22,.02), active_lower=jnp.full(22,-10.), active_upper=jnp.full(22,10.), route_upper=jnp.asarray(upper), actuator_ids=jnp.arange(26), nu=26)
    expected_filtered = apply_active_target_post_filters(previous[0], corrected[0], np.full(22,.02,np.float32), np.full(22,-10,np.float32), np.full(22,10,np.float32))
    np.testing.assert_allclose(np.asarray(ctrl.applied_active)[0], expected_filtered)
    assert bool(ctrl.valid)


def test_device_observation_matches_scalar_layout_and_gather_bounds() -> None:
    n=1; full=jnp.zeros((n,26)); kwargs=dict(action_onehot=jnp.eye(50,dtype=jnp.float32)[:1],geometry=jnp.zeros((n,12)),current_full=full,active_finger_lower=jnp.full(16,-1.),active_finger_upper=jnp.ones(16),hand_position=jnp.zeros((n,3)),hand_quat_xyzw=jnp.asarray([[0,0,0,1.]],jnp.float32),fingertip_positions=jnp.zeros((n,5,3)),contact_body_positions=jnp.zeros((n,16,3)),episode_time=jnp.zeros(n),object_position=jnp.zeros((n,3)),object_quat_xyzw=jnp.asarray([[0,0,0,1.]],jnp.float32),object_linear_velocity=jnp.zeros((n,3)),target_position=jnp.zeros((n,3)),target_quat_xyzw=jnp.asarray([[0,0,0,1.]],jnp.float32),target_next_5=jnp.zeros((n,3)),expected_contact_mask=jnp.zeros((n,16)),hand_contact_force=jnp.zeros((n,16,3)),object_contact_force=jnp.zeros((n,3)),base_offset=jnp.zeros((n,3)),joint_offset=jnp.zeros((n,16)),point_cloud_local=jnp.zeros((64,3)))
    obs, valid = jax.jit(lambda **kw: build_device_observation(**kw))(**kwargs)
    scalar = build_minimal_observation(action="01", object_name="cube1", geometry={"type":"box","size":[0,0,0]}, active_finger_dof_pos=np.zeros(16), hand_pose=np.asarray([0,0,0,0,0,0,1]), fingertip_positions_world=np.zeros(15), body16_positions_world=np.zeros(48), episode_time=0., object_position=np.zeros(3), object_orientation=np.asarray([0,0,0,1]), object_linear_velocity=np.zeros(3), target_object_position=np.zeros(3), target_object_orientation=np.asarray([0,0,0,1]), target_object_pos_next_5=np.zeros(3), expected_contact_mask=np.zeros(16), contact_force_magnitude=np.zeros(16), contact_force_directions=np.zeros(48), object_contact_force_magnitude=0.)
    np.testing.assert_allclose(np.asarray(obs)[0], scalar); assert bool(valid)
    gathered=gather_trajectory(values=jnp.arange(7*3*2).reshape(7,3,2),frame=jnp.full(7,99),lengths=jnp.asarray([1,2,3,1,2,3,1])); np.testing.assert_array_equal(np.asarray(gathered),np.asarray(jnp.arange(7*3*2).reshape(7,3,2))[np.arange(7),[0,1,2,0,1,2,0]])


def test_device_reward_done_is_finite_and_early_gated() -> None:
    n=7
    reward=compute_device_reward_done(frame=jnp.arange(n),start_frame=jnp.zeros(n),last_contact_frame=jnp.full(n,3),lengths=jnp.full(n,5),object_position=jnp.zeros((n,3)),target_position=jnp.zeros((n,3)),object_quat_xyzw=jnp.tile(jnp.asarray([[0,0,0,1.]]),(n,1)),target_quat_xyzw=jnp.tile(jnp.asarray([[0,0,0,1.]]),(n,1)),object_linear_velocity=jnp.zeros((n,3)),hand_contact_force=jnp.ones((n,16,3))*6,expected_contact_mask=jnp.ones((n,16)),base_offset=jnp.zeros((n,3)),joint_offset=jnp.zeros((n,16)),active_joint_mask=jnp.ones((n,16)),fingertip_inside_penalty=jnp.zeros(n),rotation_disabled=jnp.zeros(n,bool),early_phase_frames=2,contact_force_threshold=5.,max_contact_reward=.4,max_stability_reward=.4,stability_scale=.1,object_terminal_threshold=.3,reward_weights=jnp.ones(10))
    assert bool(reward.valid); np.testing.assert_array_equal(np.asarray(reward.raw)[:2,1:7],0); assert np.isfinite(np.asarray(reward.total)).all()
