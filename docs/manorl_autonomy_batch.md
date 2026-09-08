# Batched contact-conditioned autonomy

`sim.manorl.autonomy_batch.BatchedAutonomyRuntime` is the opt-in homogeneous
cube2 device path. It compiles one pinned model, replicates one validated
reference across rows, executes four MJX-Warp physics substeps per 120 Hz
transition, and keeps physics state, fixed reference tables, command state,
observation and reward reductions in JAX arrays.

The active command map is the v2 measured-state rate map: actor actions are
28-dimensional and all DOFs are policy-owned from the first step. Reference
arrays never enter command generation. CUDA policy boundaries use the existing
same-ordinal Torch/JAX DLPack helpers; only compact done/validity telemetry is
intended for host bookkeeping. `step` is one init-cached JAX transition graph:
its reset branch, measured-state command, four physics substeps, physical and
contact extraction, observation, reward, termination, and counters execute as
one dispatch. After a terminal row, call `prepare_action()` before asking the
actor for its next action; this returns the reset observation while preserving
the terminal observation returned by the preceding `step`.

`reference_witness_tables` is the one-time native-MuJoCo operation. It stores
both endpoint witnesses in hand-segment-body and object-body local frames,
signed distance, `exp(-abs(distance)/0.01)` confidence, source-order geometry
IDs, body IDs, and a digest. Runtime transforms those exact endpoints with
actual device body poses. It does not run `mj_geomDistance` or a nearest-point
search in `step`.

Use the operator benchmark after the immutable package is available:

```bash
python tools/benchmark_manorl_autonomy_batch.py \
  --package outputs/manorl/contact_conditioned_autonomy/cube2_02_v295_f120_pre180_post180 \
  --identity cube2_02_2833 --device gpu --num-envs 64 --steps 256
```

For large batches, the existing Warp scratch workspaces can be reused without
changing capacities or solver/CCD settings. Enable them explicitly with the
same configured contacts-per-world value used by the baseline, for example:

```bash
... --persistent-ccd-workspace --ccd-contacts-per-world 128
```

The command reports compile/warmup time, steady environment transitions/sec,
reset and validity counts, and declared host-transfer scope. It intentionally
leaves peak memory as operator-side telemetry and makes no speed claim before
execution. Start with B=1/64/256 correctness and memory checks, then measure
B=4096/8192 only on the assigned server without displacing existing jobs.
