# Configurable Training Dataset Path

## Objective

Allow the production ManoRL trainer to use the restored server2 Lance mount
without changing the repository's default dataset contract.

## Scope

- `tools/train_manorl_cube1.py`
- focused trainer CLI tests
- training protocol documentation

## Acceptance

- `--dataset-path` defaults to the existing pinned Lance path.
- The resolved path is used by training and evaluation trajectory assignment.
- The path remains visible in serialized runtime/W&B configuration.
- Focused CPU tests pass before server2 GPU3 preflight.
