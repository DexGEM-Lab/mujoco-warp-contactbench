# Gym-Aligned W&B Telemetry

## Objective

Make heterogeneous ManoRL full training observable with the same aggregation
levels as the Isaac Gym implementation while retaining one shared-policy W&B
run.

## Scope

- `sim/manorl/environment.py`
- `sim/manorl/skrl_runtime.py`
- `tools/train_manorl_cube1.py`
- focused tests under `tests/manorl/`
- training protocol documentation when the public logging contract changes

## Acceptance

- Keep global training and episode metrics.
- Log reward components and outcomes per object and per object/action pair with
  deterministic Gym-compatible labels such as `cube1_01` and `object_cube1`.
- Never average completed episodes from different pairs into the only reported
  value.
- Full-pair evaluation covers every resolved pair, rather than only the first
  sorted pair.
- Preserve one W&B run for the one shared policy and record its aggregation
  schema in config.
- Focused CPU tests, a real 77-pair trajectory check, and a server2 GPU preflight
  pass before restarting the full run on physical GPU 3.

## Current user requirement

At clean `313e322`, add Gym/W&B-equivalent TensorBoard telemetry and exact checkpoint resume for the live Server2-compatible DexHandRL all-action runtime. Do not touch Server2 or any live process. The implementation must preserve PPO and physics, export only measurements produced after the new run begins, aggregate device-resident per-world raw/shaped returns and weighted reward components before host materialization at update boundaries, and resume global updates from an explicit sidecar-validated checkpoint.

## Outputs

Preserve all prior training outputs. New preflight and full-run paths must use
unique prefixes under `outputs/manorl/`.
