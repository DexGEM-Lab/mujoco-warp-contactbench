## Phenomenon

W&B currently exposes environment transitions as the primary logging step, but the desired checkpoint and dashboard semantics identify completed PPO updates as the primary axis.

## Live mechanism

The trainer has separate completed-update and environment-transition counters. W&B log calls can use the update counter as their explicit step while retaining transitions as a logged secondary metric. Evaluation requires the same distinction, including the initial zero-update evaluation and final acceptance evaluation.

## Current claim

The bounded implementation should change only W&B step arguments, aliases, and serializable axis metadata. PPO, reward, runtime, and throughput calculations should remain unchanged.

## Evidence

The focused W&B tests now show update logging steps `[1, 2]`, evaluation steps `[0, 2]`, and update transitions `[3072, 6144]`; evaluation transitions remain explicit as `[0, 6144]`. The configured-runtime test command passed 12 tests, and `py_compile` plus `git diff --check` passed. Provenance is recorded in `OPS.md`.
