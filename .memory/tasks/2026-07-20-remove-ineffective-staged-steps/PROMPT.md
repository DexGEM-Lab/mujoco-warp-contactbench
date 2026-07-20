# Remove Ineffective Staged Steps

## Goal

Remove the staged heterogeneous route step refactor after the server2 A/B
benchmark showed no meaningful throughput or GPU-utilization improvement.

## Scope

- Revert the staged step changes in `sim/manorl/environment.py`.
- Remove the staged call-order test instrumentation.
- Preserve the original performance task memory and add measured results.

## Validation

- Focused environment and device-resident tests remain green on the restored
  implementation.
- The measured 3,740.5 transitions/s staged result is documented against the
  3,737.5 transitions/s same-host baseline.
