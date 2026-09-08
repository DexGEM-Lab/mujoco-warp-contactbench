from __future__ import annotations
import numpy as np
from types import SimpleNamespace
from sim.manorl.autonomy import align_reference_trajectory, dense_autonomous_reward, shared_support_alignment, surface_intent
from sim.manorl.autonomy_contracts import ACTION_DIM, OBSERVATION_DIM, OBSERVATION_CONTRACT, rate_limited_command
from sim.manorl.contracts import FLOOR_TOP_Z, simulation_clock

def test_m2_clock_is_canonical_120hz_and_rate_units_per_second():
    clock = simulation_clock(120)
    assert clock.physics_fps == 480 and clock.physics_substeps_per_control == 4
    previous = np.zeros(ACTION_DIM); action = np.full(ACTION_DIM, .5); limits = np.full(ACTION_DIM, 1.)
    command = rate_limited_command(previous, action, -limits, limits, np.full(ACTION_DIM, 12.), control_timestep=clock.control_timestep)
    np.testing.assert_allclose(command, .05)

def test_reference_independent_command_map_preserves_load_error_envelope():
    previous = np.zeros(ACTION_DIM); action = np.ones(ACTION_DIM); limits = np.full(ACTION_DIM, 1.)
    command = rate_limited_command(previous, action, -limits, limits, np.full(ACTION_DIM, 120.), measured_qpos=np.full(ACTION_DIM, -.4), control_timestep=1/120, max_tracking_error=np.full(ACTION_DIM, .2))
    np.testing.assert_allclose(command, -.2)

def test_surface_intent_uses_collision_surface_and_confidence():
    surface=np.asarray([[-.05,0,0],[.05,0,0],[0,.05,0],[0,0,.05]])
    points=np.concatenate((surface, np.repeat(surface[:1],12,axis=0)))
    proximity,anchors,confidence=surface_intent(points,np.zeros(3),np.array([0,0,0,1.]),surface)
    np.testing.assert_allclose(anchors,points); np.testing.assert_allclose(proximity,1.0); np.testing.assert_allclose(confidence,1.0)

def test_alignment_is_single_shared_translation_to_floor_and_reward_requires_measured_contact():
    raw_obj=np.array([0.,0.,.1]); hand=np.array([0.,0.,.2]); vertices=np.array([[0,0,-.05],[0,0,.05]])
    aligned_hand,aligned_obj,shift=shared_support_alignment(hand,raw_obj,vertices)
    np.testing.assert_allclose(shift,[0,0,FLOOR_TOP_Z-.05]); np.testing.assert_allclose(aligned_hand-aligned_obj,hand-raw_obj)
    assert np.isclose(np.min(vertices[:,2]+aligned_obj[2]), FLOOR_TOP_Z)
    zeros=np.zeros((16,3)); terms=dense_autonomous_reward(aligned_obj,aligned_obj,np.zeros(3),aligned_hand-aligned_obj,aligned_hand-aligned_obj,np.ones(16),np.zeros(16),np.zeros(28),phase=0.,measured_hand_object_force=zeros,supporting_object_force=np.zeros(3))
    assert terms["measured_contact"] == 0.0 and terms["contact_anchor_correspondence"] > .99

def test_alignment_preserves_source_relative_geometry_and_arrays():
    q=np.zeros((2,28)); q[:,2]=.2; raw=np.array([[0.,0.,.1],[0.,0.,.11]]); quat=np.tile([0.,0.,0.,1.],(2,1)); traj=SimpleNamespace(q_ref=q,object_pos_raw=raw,object_quat_xyzw=quat)
    q_aligned,obj_aligned,shift=align_reference_trajectory(traj,np.array([[0.,0.,-.05],[0.,0.,.05]]))
    np.testing.assert_array_equal(q[:, :3], np.array([[0.,0.,.2],[0.,0.,.2]])); np.testing.assert_array_equal(raw,np.array([[0.,0.,.1],[0.,0.,.11]])); np.testing.assert_allclose(q_aligned[:,2]-obj_aligned[:,2], q[:,2]-raw[:,2])

def test_observation_contract_contains_reference_intent_and_forces():
    assert OBSERVATION_DIM == OBSERVATION_CONTRACT.dimension
    names={name for name,_ in OBSERVATION_CONTRACT.fields}
    assert {"reference_surface_proximity","reference_surface_anchor_local","measured_hand_object_force","supporting_object_net_force","relative_contact_motion"} <= names
