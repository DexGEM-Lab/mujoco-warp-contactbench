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
    fingertip_inside_object_penalty,
    gather_trajectory,
    route_active_to_full,
    route_full_to_active,
)
from sim.dexhandrl.hand_mapping import active_to_full_dof, full_to_active_dof
from sim.dexhandrl.parity import (
    add_termination_reward,
    apply_reward_weights,
    build_minimal_observation,
    object_tracking_rewards,
    rotation_tracking_reward,
)
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
    np.testing.assert_allclose(np.asarray(route_full_to_active(full=actual)), np.stack([full_to_active_dof(f) for f in expected]), rtol=2e-6, atol=2e-6)
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


def test_device_fingertip_cube_penalty_matches_depth_ramp() -> None:
    penalty = fingertip_inside_object_penalty(
        fingertips_world=jnp.asarray([[[.046,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0]]]),
        object_position=jnp.zeros((1,3)), object_quat_xyzw=jnp.asarray([[0,0,0,1.]]),
        primitive_type=jnp.asarray([0]), primitive_params=jnp.asarray([[.1,.1,.1]]),
        active_fingertip_mask=jnp.asarray([[1,0,0,0,0.]]), min_depth=.001, max_depth=.003,
        max_penalty=.4,
    )
    np.testing.assert_allclose(np.asarray(penalty), [-.4])


def test_device_control_matches_scalar_raw_reference_correction_residual_filters_and_ctrl() -> None:
    rng, n = np.random.default_rng(17), 7
    route_upper = np.asarray([[1.1, .8], [1.0, .9], [.9, .7], [1.2, .6]], np.float32)
    route_limits = {index: tuple(pair) for index, pair in zip((6, 9, 12, 15), route_upper)}
    reference = rng.uniform(-.3, 2.7, (n, 22)).astype(np.float32)
    current = rng.uniform(.15, .95, (n, 26)).astype(np.float32)
    action = rng.uniform(-1., 1., (n, 22)).astype(np.float32)
    previous = rng.uniform(-.2, .2, (n, 22)).astype(np.float32)
    base_offset = rng.uniform(-.02, .02, (n, 3)).astype(np.float32)
    joint_offset = rng.uniform(-.08, .08, (n, 16)).astype(np.float32)
    mask = (rng.uniform(size=(n, 16)) > .3).astype(np.float32)
    frame = np.asarray([0, 1, 2, 4, 5, 6, 7], np.int32)
    last = np.asarray([3, 3, 1, 3, 4, 5, 1], np.int32)
    max_delta = np.linspace(.01, .04, 22, dtype=np.float32)
    lower, upper = np.full(22, -.5, np.float32), np.full(22, 2.0, np.float32)
    result = apply_device_control(action=jnp.asarray(action), reference_active=jnp.asarray(reference), current_full=jnp.asarray(current), previous_active=jnp.asarray(previous), base_offset=jnp.asarray(base_offset), joint_offset=jnp.asarray(joint_offset), active_joint_mask=jnp.asarray(mask), frame=jnp.asarray(frame), start_frame=jnp.zeros(n, jnp.int32), last_contact_frame=jnp.asarray(last), use_residual=jnp.ones(n, bool), early_phase_frames=3, base_decay=jnp.asarray([.8, .7, .6]), post_contact_base_decay=jnp.asarray([.4, .3, .2]), joint_decay=.75, post_contact_joint_decay=.25, base_scales=jnp.asarray([.01, .02, .03]), base_max_offset=.04, active_max_deltas=jnp.asarray(max_delta), active_lower=jnp.asarray(lower), active_upper=jnp.asarray(upper), route_upper=jnp.asarray(route_upper), actuator_ids=jnp.asarray(np.arange(25, -1, -1)), nu=26)
    expected_reference, expected_corrected, expected_applied, expected_full, expected_ctrl = [], [], [], [], []
    for row in range(n):
        corrected = full_to_active_dof(active_to_full_dof(reference[row], current[row], route_limits))
        early, after = frame[row] < 3, frame[row] > last[row]
        if early:
            next_base, next_joint = np.zeros(3, np.float32), np.zeros(16, np.float32)
        else:
            bdecay = np.asarray([.4, .3, .2] if after else [.8,.7,.6], np.float32)
            next_base = np.clip(bdecay * base_offset[row] + np.clip(action[row,:3], -1, 1) * np.asarray([.01,.02,.03]), -.04, .04)
            jdecay = .25 if after else .75
            next_joint = jdecay * joint_offset[row] + np.clip(action[row,6:], -1, 1) * mask[row] * (np.asarray([.01,.035,.02,.02,.01,.03,.02,.01,.03,.02,.01,.03,.02,.01,.03,.02], np.float32) * 3)
            next_joint = np.clip(next_joint, -(np.asarray([.1,.35,.2,.2,.1,.4,.2,.1,.4,.2,.1,.4,.2,.1,.4,.2], np.float32) * 2), np.asarray([.1,.35,.2,.2,.1,.4,.2,.1,.4,.2,.1,.4,.2,.1,.4,.2], np.float32) * 2) * mask[row]
        corrected = corrected.copy(); corrected[:3] += next_base; corrected[6:] += next_joint
        applied = apply_active_target_post_filters(previous[row], corrected, max_delta, lower, upper)
        full = active_to_full_dof(applied, current[row], route_limits)
        ctrl = np.zeros(26, np.float32); ctrl[np.arange(25, -1, -1)] = full
        expected_reference.append(reference[row]); expected_corrected.append(corrected); expected_applied.append(applied); expected_full.append(full); expected_ctrl.append(ctrl)
    np.testing.assert_allclose(np.asarray(result.reference_active), expected_reference, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(result.corrected_active), expected_corrected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(result.applied_active), expected_applied, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(result.full_target), expected_full, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(result.ctrl), expected_ctrl, rtol=2e-6, atol=2e-6)
    assert np.max(np.abs(np.asarray(result.corrected_active) - reference)) > .1


def test_device_reward_components_and_termination_match_scalar_helpers() -> None:
    n = 7
    frame = np.asarray([0, 1, 2, 4, 5, 6, 7], np.int32)
    lengths = np.asarray([10, 10, 10, 10, 10, 6, 8], np.int32)
    # Row five is both terminal and truncated, proving failure takes priority
    # over timeout exactly as add_termination_reward does in the scalar env.
    obj = np.asarray([[0,0,0], [.1,0,0], [0,.1,0], [.1,.1,.1], [.5,0,0], [.5,0,.1], [0,0,0]], np.float32)
    target = np.zeros((n,3), np.float32)
    quat = np.tile(np.asarray([0,0,0,1.], np.float32), (n,1)); quat[4] = np.asarray([0,0,np.sin(np.pi/8),np.cos(np.pi/8)], np.float32)
    force = np.zeros((n,16,3), np.float32); force[:, :2, 0] = np.asarray([[6,4],[7,8],[2,9],[6,6],[8,2],[9,9],[0,0]], np.float32)
    expected = np.zeros((n,16), np.float32); expected[:, :2] = 1
    base = np.linspace(-.02,.02,n*3,dtype=np.float32).reshape(n,3); joint = np.linspace(-.1,.1,n*16,dtype=np.float32).reshape(n,16); mask = np.tile(np.asarray([1,0]*8, np.float32), (n,1))
    weights = np.asarray([.001,.1,.2,.3,.4,.5,.6,.7,.8,.9], np.float32)
    result = compute_device_reward_done(frame=jnp.asarray(frame), start_frame=jnp.zeros(n,jnp.int32), last_contact_frame=jnp.full(n,3), lengths=jnp.asarray(lengths), object_position=jnp.asarray(obj), target_position=jnp.asarray(target), object_quat_xyzw=jnp.asarray(quat), target_quat_xyzw=jnp.tile(jnp.asarray([[0,0,0,1.]],jnp.float32),(n,1)), object_linear_velocity=jnp.asarray(np.arange(n*3,dtype=np.float32).reshape(n,3)/20), hand_contact_force=jnp.asarray(force), expected_contact_mask=jnp.asarray(expected), base_offset=jnp.asarray(base), joint_offset=jnp.asarray(joint), active_joint_mask=jnp.asarray(mask), fingertip_inside_penalty=jnp.linspace(-.3,0,n), rotation_disabled=jnp.asarray([0,1,0,0,0,0,0],bool), early_phase_frames=2, contact_force_threshold=5., max_contact_reward=.4, max_stability_reward=.4, stability_scale=.1, object_terminal_threshold=.3, reward_weights=jnp.asarray(weights), termination_failure_reward=-123., termination_timeout_reward=4.5)
    names = ("alive", "object_pos_tracking_x", "object_pos_tracking_y", "object_pos_tracking_z", "contact_quality", "object_stability_velocity", "object_rot_tracking", "fingertip_inside_object_penalty", "action_penalty_position", "action_penalty_joint")
    scalar_raw, scalar_weighted, scalar_total = [], [], []
    for row in range(n):
        after, early = frame[row] > 3, frame[row] < 2
        quality = ((np.linalg.norm(force[row],axis=-1) > 5) * (expected[row] > .5)).sum() / max(expected[row].sum(), 1) * .4
        quality = 0. if after else quality
        tracking = object_tracking_rewards(obj[row], target[row])
        angle = 2 * np.arccos(np.clip(abs(np.dot(quat[row], [0,0,0,1.])), 0, 1)) * 180 / np.pi
        raw = dict(zip(names, [1., quality*tracking["object_pos_tracking_x"], quality*tracking["object_pos_tracking_y"], quality*tracking["object_pos_tracking_z"], quality, .4*np.exp(-(np.linalg.norm(np.arange(row*3,row*3+3)/20)/.1)**2) if after else 0., 0. if row == 1 else rotation_tracking_reward(angle), float(np.linspace(-.3,0,n)[row]), -np.abs(base[row]*100).sum()*(.5 if after else 1.), -np.abs(joint[row]*10).sum()/max(mask[row].sum(),1)*8*(.5 if after else 1.)]))
        if early:
            for name in names[1:7]: raw[name] = 0.
        components = apply_reward_weights(raw, dict(zip(names, weights)))
        terminated, truncated = (not early and np.linalg.norm(obj[row]) > .3), frame[row] >= lengths[row] - 1
        if terminated: add_termination_reward(components, "failure", {"failure": -123.})
        elif truncated: add_termination_reward(components, "timeout", {"timeout": 4.5})
        scalar_raw.append([raw[name] for name in names]); scalar_weighted.append([raw[name]*weights[index] for index,name in enumerate(names)]); scalar_total.append(components["total"])
    np.testing.assert_allclose(np.asarray(result.raw), scalar_raw, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(result.weighted), scalar_weighted, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(result.total), scalar_total, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(np.asarray(result.terminated), [False,False,False,False,True,True,False])
    np.testing.assert_array_equal(np.asarray(result.truncated), [False,False,False,False,False,True,True])
    assert bool(result.valid)
