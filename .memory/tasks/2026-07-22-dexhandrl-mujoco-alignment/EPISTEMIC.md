# Epistemic Record

## Current training-launch claim

The requested Server2 production run cannot be launched correctly through the current training entry point. Its dataset path filters for exactly one action, while the request requires a balanced heterogeneous batch over actions `01`, `02`, `03`, `04`, `09`, `10`, and `18`. Its `SyncVectorEnv` path also constructs independent single-world runtimes; raising the environment count to 1,024 would multiply Python/runtime overhead without providing true batched, device-resident physics. A direct 1,024-environment launch would therefore violate both the data-assignment and execution contracts.

The target is one action-conditioned policy for `cube1`, trained with 1,024 environments for 3,000 PPO updates on Server2 GPU 1. The implementation must first provide deterministic balanced all-action assignment (all seven actions represented, counts differing by at most one) and a genuinely batched device-resident physics path. The discriminating preflight is one finite PPO update at the full 1,024-environment scale: it must complete on GPU 1 with adequate headroom while PIDs `1872925` and `1881324` remain alive. Only then is the 3,000-update run eligible to launch in a uniquely named tmux session from an isolated Server2 mirror.

Server2 is ready in dependencies, data, and source inputs, but it does not yet contain this feature mirror or a `sim/dexhandrl` checkout. Existing Unison configuration covers only the primary `dev` root and is stopped, so it cannot be treated as feature synchronization. GPU 1 currently has the two existing processes, approximately 21.5 GiB free, and 0% utilization in the sampled observation. Free memory is encouraging but does not establish training headroom; the finite full-scale preflight is the required evidence.

## Batch allocation claim

N7 and N256 batch runs succeeded. At N256, the global contact high water was 1,377 against a 131,136-row actual Warp contact buffer; per-world constraints peaked at 56 against 8,192 rows, and native initial-state `nefc` measured 146. The old N1024 request therefore did not reveal a physical capacity limit: it requested 524,352 contacts and 8,192 constraints per world, reached 24,080 MiB during Warp graph creation, and OOMed before the first transition.

The runtime-only allocation is now `128 * N + 64` global contact rows and 512 per-world constraint rows. At N1024 this makes the contact request 131,136 rows, while construction still rejects a constraint capacity below measured initial `nefc + 64` (210 at the N256 measurement) and rollout still fails at 80% utilization of actual buffers. No physics, solver, substep, geometry, reward, or task parameter changed. The causal prediction is that the 4x smaller contact allocation and 16x smaller per-world constraint allocation reduce Warp graph memory enough for the preflight. If it still OOMs, the next discriminating test changes contact versus constraint capacity one at a time; do not reduce physics. See OPS capacity entries dated 2026-07-24.

Production acceptance additionally requires observing the first training progress from the 3,000-update run and recording the concrete progress and checkpoint output paths. Existing GPU jobs must not be killed to make the launch fit.

## Accepted viewer claim

The explicit CPU-only `mujoco-native` backend is accepted for the interactive, single-environment DexHandRL viewer. It makes the viewer responsive while executing the production environment rather than replaying or mirroring an injected trace. It uses the same generated scene, immutable `physics_contract`, controller, reset, ordered filters, frame/reference schedule, state-dependent target routing, observations, rewards, and termination logic as `mjx-warp`. Viewer telemetry names the active backend.

This is an interactive-visualization acceptance, not a numerical-backend-equivalence claim. MJX-Warp remains unchanged as the default and authority for training, batched execution, and exact reference traces. Native MuJoCo is selected explicitly outside the viewer and cannot silently replace or fall back from MJX-Warp.

## Accepted latency mechanism

The primary interactive latency was the single-world MJX execution path. The hand collision model is large—22 convex hulls, 9,850 vertices, and 19,612 faces—but disabling contact changed MJX CPU step time only about four percent. In the fixed final measurement, steady full-`env.step` median latency was about `0.696 ms` for native MuJoCo and `295 ms` for MJX-Warp on CPU, approximately `424x` apart. A visible 300-step native run at `speed=0.25` advanced responsively and exited 0.

Native stepping removes the dominant single-world MJX overhead without changing the backend used where batching amortizes that overhead or where frozen MJX behavior is authoritative. This confirms the performance prediction preregistered before implementation.

## Shared semantics and corrected exactness model

The fixed cube1/action02/sequence0, residual-off, reset-plus-50-step v2 comparison validates one identity, physics contract, frame schedule, and zero-action input across both backends. Frame indices and `reference_active_target` are bit-exact. The preregistered prediction that corrected and applied active targets would also be bit-exact was too strong: routing first reads each backend's live `current_full` state, so physical divergence can enter these arrays before stepping the next row. Their observed maximum difference is only `4.66e-10`, within the retained `1e-7` applied-target bound. `full_target` and `ctrl` likewise depend on each backend's live distal qpos and are reported rather than required exact.

Each backend reset already advances its own configured `settle_frames * substeps` before the reset observation is captured. Reset-row divergence therefore reflects independently settled solver states, not a missing settle step. The evidence separates reset from post-step rows instead of folding this difference into a false common initial-state claim.

After 50 post-reset control steps, qpos RMSE/max are `3.20e-4`/`3.39e-3`, object-position maximum difference is `3.31e-6 m`, and processed-observation RMSE/max are approximately `0.0287`/`0.932`. Observation error is dominated by contact magnitude/direction slices; kinematic and object-position differences are much smaller. Exact reward in this early reference-only phase is near-trivial and provides no backend-equivalence evidence.

Canonical evidence is `outputs/dexhandrl_backend_comparison/cube1_action02_sequence0_50step_v2_20260723.json` plus its NPZ, whose SHA256 is `84ded0067481278dfdfdbd4634aabe2d8b14f8747ac88abffc7d2889c248a549`. The JSON carries authority, exactness, reset, target-routing, timing, dtype, and per-slice interpretation; the NPZ preserves paired raw arrays.

## Parameter provenance and invariants

`DexHandPhysicsConfig` remains the sole production physical source. Its frozen `DexHandServoConfig` and `DexHandMaterialConfig` own the accepted DexHand values: control/physics periods `0.01/0.001 s`, Euler/Newton with 80 iterations, base-position `5000/500/500`, base-rotation `10000/125/100`, finger `15/0/50`, hand/object friction `0.5/0.01/0.001`, and ground friction `0.5/0.5/0`. Backend choice does not create a second set of physics constants.

Scene construction and compiled validation continue to enforce source asset assumptions, timestep, integrator/solver settings, dimensions and ordering, gains and limits, friction/contact settings, reset-critical names, and capacities. Compiled model capacities (`njmax=2048`, `nconmax=8192`) remain distinct from MJX-Warp runtime allocation (`njmax=8192`, `naconmax=8192`). Native execution reuses the compiled model contract; it does not inherit MJX runtime allocation as physical provenance.

## Remaining migration boundary

The native viewer does not establish Isaac Gym checkpoint parity or stable native closed-loop grasp-lift-place. The prior 461D checkpoint result remains: the network mapping reproduces recorded policy outputs, but simulator-specific physical feedback and state-dependent 22-to-26 routed targets diverge. Exact full routed Gym controls can drive the corrected MuJoCo plant, while the same 22D action alone is under-specified across different live distal states. Resolving that limitation requires a routing-state adapter or policy adaptation/retraining; the viewer backend does not change it.

The accepted scope is therefore precise: native MuJoCo is the interactive visualization backend; MJX-Warp remains the production numerical authority; backend divergence is measured and preserved rather than hidden.
