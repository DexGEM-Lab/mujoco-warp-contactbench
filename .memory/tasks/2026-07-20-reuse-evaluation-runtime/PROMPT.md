# Reuse Evaluation Runtime

## Goal

Prevent the full-pair trained evaluation from constructing a second 77-pair
MJX-Warp evaluator after training. Reuse the bounded evaluator that already ran
the zero and untrained baselines, load the final native checkpoint into it, and
then run the trained evaluation.

## Scope

- `tools/train_manorl_cube1.py`
- Focused lifecycle tests in `tests/manorl/test_train_budget.py`
- `docs/manorl_cube1_training_protocol.md`

## Validation

- The initial evaluator remains alive through training.
- The 2,048-world training runtime is released before trained evaluation.
- The final checkpoint is loaded into the original evaluator.
- Focused trainer tests and Python compilation pass.
