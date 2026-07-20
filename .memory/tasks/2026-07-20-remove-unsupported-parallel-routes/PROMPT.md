# Remove Unsupported Parallel Routes

## Goal

Remove concurrent heterogeneous route execution after the production CUDA
preflight aborted in XLA stream-capture synchronization.

## Scope

- Restore serial heterogeneous route execution.
- Remove the thread-pool concurrency test instrumentation.
- Preserve and document the failed server2 experiment.

## Validation

- Focused environment and device-resident tests pass on the restored code.
- Remote source hashes match the restored `dev` implementation.
