# Current Epistemic Model

## Starting hypothesis

The current environment is not end-to-end device resident. Each rollout step is
serialized by cross-framework host boundaries: skrl/Torch actions enter a NumPy
Gymnasium adapter, residual action processing is host-side, MJX-Warp advances on
the JAX device, `MjxWarpPhysicalProducer.extract` copies state and private contact
buffers to NumPy, and observation/reward/termination are then computed on the host
before skrl reconstructs Torch tensors.

## Existing evidence

- A narrow combined mode (device-side controller/reset writes plus disabled
  transition snapshots) completed one matched 2,048-env, 12-update run at
  11,783.9 transitions/s versus 10,221.1/s legacy (+15.29% throughput, 13.26%
  wall-time reduction).
- This was one fixed-order legacy-then-optimized pair, so it does not isolate the
  controller change from disabled diagnostics or quantify order/cache variance.
- Zero-reset updates improved more than reset-heavy updates; reset-heavy update 7
  did not improve. This points toward remaining extraction/forward/reset and host
  semantic work, but is not a phase profile.
- The existing narrow test covers CPU, one environment, one action, and terminal
  plus delayed reset; it does not establish GPU/multi-world equivalence.

## Belief discipline

Treat the state/contact extraction plus downstream NumPy semantics as the leading
hypothesis, not a proven percentage. Do not claim that `TransitionSnapshot` copies
the full physical/contact state: `producer.extract()` already performs the device
transfer, while the snapshot mostly adds copies of controller/action/progress arrays
and retains references to physical/observation/reward objects. Update this file after
phase timing and ablations identify the actual dominant costs.

## Attribution repair state

The local profiler now distinguishes raw action NumPy materialization, full
runtime `env.step`, skrl action and response conversions, controller/physics/reset
writes and forward, state materialization, contact-buffer materialization, Python
contact decode, reward, observation, transition recording, trainer finite checks,
host telemetry, and post-interaction. PPO optimizer timing is counted only when
skrl's rollout and `learning_starts` condition predicts that `post_interaction`
will call `update`. Contact metadata records per-step `nacon` and array
allocation/materialization sizes; these values are not PCIe transfer measurements.

The benchmark harness is now v3 and fail-closed: every mode/repeat runs in an
independent child process, each child publishes a schema-checked JSON artifact
and logs, and the aggregate is written only after all children succeed. Source
and mirror file hashes plus the dirty-worktree diff hash are required in the
provenance artifact. The isolated mirror received a transfer before this
expanded attribution diff, but its final hash check failed because the remote
shell had no `python` alias; no remote result is attributable to the current
expanded diff yet.
