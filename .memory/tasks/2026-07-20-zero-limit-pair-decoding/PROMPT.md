# Zero-Limit Pair Decoding

## Objective

Allow bounded evaluation batches to select fewer environments than the number
of resolved Lance object/action pairs.

## Scope

- `sim/manorl/trajectory.py`
- `tests/manorl/test_trajectory.py`

## Acceptance

- Resolved pairs with zero assigned environment slots are not fully decoded.
- Pairs with assigned slots still require at least one valid full trajectory.
- `--all-pairs --evaluation-num-envs 1` can build its trajectory batch.
- Focused trajectory and trainer tests pass.
