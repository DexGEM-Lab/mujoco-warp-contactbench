# Staged Heterogeneous Steps

## Goal

Increase mixed-object GPU occupancy without changing environment or PPO
semantics. Submit the physics work for every homogeneous object route before
materializing route state and contacts on the host.

## Scope

- `sim/manorl/environment.py`
- Focused heterogeneous environment tests in `tests/manorl/test_environment.py`

## Invariants

- Homogeneous `step()` behavior and outputs remain unchanged.
- Mixed-object global environment order remains unchanged.
- Every route completes action processing, physics dispatch, delayed reset
  dispatch, and progress updates before any route starts physical/contact host
  materialization.
- Reward, termination, observation, diagnostics, and reset semantics remain in
  their existing order within each route.

## Validation

- Focused environment tests pass.
- A call-order regression test proves all route prepares precede completes.
- The server2 2048-environment all-pair benchmark improves over the 3,595
  transitions/s baseline before the change is used for a formal run.
