# Parallel Heterogeneous Routes

## Goal

Test whether persistent host threads allow JAX/Warp to dispatch independent
static object-model steps concurrently and overlap route-local host decoding.

## Scope

- `sim/manorl/environment.py`
- Focused heterogeneous environment concurrency test

## Constraints

- Enable parallel route execution only for GPU heterogeneous environments.
- Preserve deterministic global output ordering and existing route-local step
  semantics.
- Shut down worker threads when the heterogeneous environment is collected.
- Do not change trajectory assignment, PPO, reward, observation, or W&B logic.

## Acceptance

- Focused tests pass, including a barrier proving two route calls can enter
  concurrently.
- The server2 2,048-environment all-pair preflight materially exceeds the
  3,737.5 transitions/s same-host baseline and improves sampled GPU occupancy.
- Remove the feature if it remains within run-to-run variation.
