# Epistemic model

## Accepted clock mechanism
The public runtime couples source/reference, policy inference, and control at the selected mode:

- 100 Hz: 10 ms policy interval, 400 Hz physics, four equal 2.5 ms substeps.
- 120 Hz: 8.333333 ms policy interval, 480 Hz physics, four equal 2.083333 ms substeps.

This removes the impossible `400/120 = 10/3` integer ratio and rejects 3/3/4 timing jitter. `reference_fps` and `control_fps` remain separate checkpoint fields but must match in public modes. Historical v6/v7 checkpoints decode internally as 200 Hz policy, 400 Hz physics, and two substeps.

Real v295 CPU/MJX transitions confirm the mechanism: model timesteps are exactly 1/400 and 1/480, the environment executes four substeps, and observed one-step simulation-time deltas are approximately 1/100 and 1/120 respectively (OPS.md 2026-08-03T21:32:42).

## Discrete-time semantics
The user selected source-aligned transitions rather than preserving former 200 Hz physical durations. `early_phase_steps=30`, PPO rollout 48, residual recurrence/scales, PPO gamma/lambda, and per-transition reward remain unchanged. Early phase lasts 0.30 s at 100 Hz or 0.25 s at 120 Hz; a rollout spans 0.48 s or 0.40 s. One policy decision corresponds to one source-clock slot.

## Exact padding and pair coverage
The first implementation exposed a contradiction: clipping modern captures to available source margins reduced pre180 eligibility from 75 to 73 pairs and produced as few as 14 post steps (OPS.md 2026-08-03T21:31:00). The desired state requires both exact padding duration and all 75 pairs, while two pairs have no capture with 180 real pre-frames and 23 pairs have no capture with 250 real post-frames.

The only mechanism satisfying both is stationary edge extension. Modern captures now hold their first/last pose for missing margin slots and repeat the corresponding `source_indices`; captured motion is never extrapolated. Real v295 audits at both public rates resolve all 75 pairs with movement start exactly 180 and exactly 250 steps after movement end. Sixty-two of 75 selected identities require at least one edge hold, so this behavior is central to the contract, not an exceptional fallback (OPS.md 2026-08-03T21:32:00).

## Checkpoint migration
The durable source is server1 `checkpoint-005900.pt`, SHA256 `4b85bfa58b8b9191e86c7f9975ec3bf283c279c045f86481be229804de8cf4da`: 5900 local updates plus upstream 1200 = 7100 conceptual updates. Its v6/pre100/200 Hz MDP cannot strictly resume. Warm start transfers policy, value, observation normalizer, and value normalizer, while optimizer, scheduler, rollout memory, and progress remain fresh.

This has been tested against the actual payload, not only mocks. Every transferred tensor matched, target optimizer state stayed empty, strict resume failed closed, and the migrated 120/480×4/pre180 runtime produced a finite 28D action (OPS.md 2026-08-03T21:35:18). Running 12,900 new updates yields a declared conceptual learned-update total of 20,000; this is lineage, not a claim of equal simulated seconds.

## Persisted outputs
New checkpoint ABI binds reference/control/physics clocks, both timesteps, substeps, pre/post-padding, hand/action layout, residual action, reward, model, and normalizers. Synthetic Lance writing advances to clock-aware v2.3: schema, rows, timestamps, and provenance carry dynamic 100/120 or legacy 200 Hz clocks. Validator read support for fixed-200-Hz v2.2 remains.

## Operational launch gate
Training starts only after integration into `dev`, focused/broad validation, and a server1 GPU smoke. Unison must be active and server1 `dev` HEAD must equal the locally integrated commit; observing only a Unison process is insufficient. The server1 source checkpoint remains at its original durable path and hash.

## Current justified claim
The clock, exact padding, 75-pair coverage, real checkpoint warm start, and focused affected suite with fully materialized assets are supported by direct evidence. Remaining uncertainty is operational: whether the feature merges cleanly under GitGuard and whether a 120 Hz all-pair GPU smoke fits server1 memory and produces acceptable throughput.

## Highest-value next experiment
Integrate the accepted commit, verify Unison commit equality on server1, then run a bounded 120 Hz GPU startup/update smoke with the real warm-start checkpoint. That directly tests the remaining launch risk before the 12,900-update run.
