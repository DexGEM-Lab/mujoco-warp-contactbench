# Full-Pair Grasp Mapping

## Objective

Make every object/action pair selected by `--all-pairs` resolve a validated
expected-contact grasp mapping before launching the server2 full training run.

## Scope

- `sim/manorl/assets.py`
- `sim/manorl/environment.py`
- `tests/manorl/test_trajectory.py`

## Acceptance

- Cube1 action 01 remains numerically unchanged.
- Every one of the 77 resolved s02 pairs parses through the runtime grasp mapper.
- Focused asset, trajectory, and environment tests pass.
