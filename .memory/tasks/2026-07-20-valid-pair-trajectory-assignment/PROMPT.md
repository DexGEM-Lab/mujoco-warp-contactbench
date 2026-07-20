# Valid Pair Trajectory Assignment

## Objective

Assign only fully decoded, valid Lance trajectories for `--all-pairs` while
keeping full-run host memory bounded.

## Scope

- `sim/manorl/trajectory.py`
- `tests/manorl/test_trajectory.py`

## Acceptance

- Invalid full rows, including non-monotonic timestamps, are skipped.
- Pair and trajectory assignment remains deterministic round-robin.
- Lance full-row decoding uses bounded chunks rather than one large PyList.
- Real 77-pair tests and focused trajectory tests pass.
