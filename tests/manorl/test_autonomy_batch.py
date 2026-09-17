"""Public v4 runtime boundary tests.

The prior file exercised removed v3 witness/reward symbols and made collection
fail after the intentionally versioned v4 transition replacement.  Numerical
contact/cache behavior lives in test_autonomy_v4; these checks keep the public
runtime module importable and its declared batch ABI explicit.
"""
from __future__ import annotations
import pytest
from sim.manorl.autonomy_batch import (
    AutonomyTransitionState, BatchedAutonomyRuntime, V3_OBSERVATION_DIM,
    V4_OBSERVATION_DIM,
)
from sim.manorl.environment import recommended_warp_contact_capacity
from sim.manorl.autonomy_contracts import ACTION_DIM, OBSERVATION_DIM


def test_v4_public_runtime_dimensions_and_transition_state_fields():
    assert V4_OBSERVATION_DIM == OBSERVATION_DIM == 957
    # Compatibility alias is deliberately v4 raw width, not legacy 538.
    assert V3_OBSERVATION_DIM == V4_OBSERVATION_DIM
    assert AutonomyTransitionState._fields == ("data", "indices", "pending_reset", "previous_command")


def test_runtime_constructor_rejects_invalid_batch_and_workspace_contracts():
    # Validate before asset/cache construction: these are stable public errors.
    with pytest.raises(ValueError, match="positive"):
        BatchedAutonomyRuntime(None, num_envs=0)
    with pytest.raises(ValueError, match="requires device"):
        BatchedAutonomyRuntime(None, num_envs=1, device="cpu", persistent_ccd_workspace=True)
    with pytest.raises(TypeError, match="full_horizon_diagnostic"):
        BatchedAutonomyRuntime(None, full_horizon_diagnostic=1)
    assert ACTION_DIM == 28


def test_explicit_ccd_scratch_expands_global_contact_arena():
    """The Warp naccdmax <= naconmax precondition holds at B4096."""
    batch, contacts_per_world = 4096, 121
    recommended = recommended_warp_contact_capacity(batch, ("right",))
    contact_arena, ccd_capacity = BatchedAutonomyRuntime._capacity_contract(
        batch, contacts_per_world, recommended
    )
    assert recommended == 262208
    assert ccd_capacity == 495616
    assert contact_arena == ccd_capacity
    assert ccd_capacity <= contact_arena


def test_default_contact_arena_remains_recommended():
    batch = 4096
    recommended = recommended_warp_contact_capacity(batch, ("right",))
    contact_arena, ccd_capacity = BatchedAutonomyRuntime._capacity_contract(
        batch, None, recommended
    )
    assert ccd_capacity is None
    assert contact_arena == recommended == 262208
