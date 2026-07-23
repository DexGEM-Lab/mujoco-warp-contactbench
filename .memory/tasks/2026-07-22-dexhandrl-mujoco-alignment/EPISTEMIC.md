# Epistemic Record

## Current claim

The completed ManoRL-style structural refactor remains behavior-neutral at its accepted candidate: its generated scene, compiled model signature, and frozen reset-plus-50-step arrays were bit-exact to the clean pre-refactor `legacy_port_v1` baseline. The newly requested native interactive backend is **preregistered and not yet implemented or accepted**.

MJX-Warp remains the unchanged default and authoritative execution path for training, batches, and exact reference traces. Native MuJoCo is scoped to an explicit CPU-only backend for the interactive `env=1` viewer. Backend selection must be visible and fail closed; native execution cannot become a default, automatic fallback, or source of an “exact trace” claim.

## Live performance theory

Single-world MJX execution overhead dominates the interactive viewer workload. At one control frame, measured MJX CPU time is about `290 ms` (`3.45 Hz`). Disabling contacts changes it only to `279 ms`, about a four-percent improvement, despite the hand collision model containing 22 convex geoms with 9,850 vertices and 19,612 faces. Reducing to one substep changes the cost to `50.5 ms`; Newton with 10 iterations costs `139.6 ms`; a fused 10-substep path costs `170 ms` and is exact at the sampled state. MJX on GPU still costs `93 ms` per control frame. Native MuJoCo advances the same 10 substeps in `0.551 ms` on CPU.

These observations reject collision complexity as the principal single-environment bottleneck. They support dispatch, compilation/runtime, and single-world MJX execution cost as the dominant mechanism. Native stepping is therefore the principled interactive backend: it removes the dominant MJX single-world path while preserving MJX where batching amortizes that path and where existing exact traces are authoritative.

This theory predicts that native MuJoCo will make the one-environment viewer responsive without changing MJX training or trace behavior. It does not predict numerical state equivalence. Even with the same model and control inputs, native MuJoCo and MJX-Warp may diverge through backend solver, contact, and floating-point execution. The comparison must expose that divergence rather than hide it behind shared configuration.

## Shared semantic contract

Both backends must consume one generated scene and the same immutable serialized `physics_contract`. They must share controller parameters and update order, reset state and settle semantics, active-to-full target routing, velocity clamp then position clamp, frame schedule, reference targets, residual-off behavior, termination/reward/task logic, and observation construction.

The interactive native path advances physics with native MuJoCo CPU stepping. The viewer may render that native state directly, but viewer code remains outside controller and task ownership. Telemetry must name the backend so a user can distinguish `native_mujoco` from `mjx_warp` at runtime.

## Acceptance preregistration

Acceptance requires all of the following:

1. The existing MJX-Warp default is unchanged and its frozen default trace remains bit-exact. Training, batched execution, and exact-trace production continue to select MJX-Warp unless the caller explicitly requests otherwise.
2. Native MuJoCo is an explicit CPU-only, `env=1`, interactive viewer backend. Unsupported batch, accelerator, or exact-trace use fails explicitly; no silent fallback is allowed.
3. The native and MJX paths resolve the same scene and `physics_contract` and execute the same controller, reset, ordered filters, frame/target schedule, routing, observations, rewards, and termination semantics.
4. A fixed cube1/action02/sequence0, residual-off comparison runs reset plus 50 control steps. Its paired records prove exact frame indices, reference targets, corrected/applied active targets, full controller outputs, and controls before backend stepping.
5. The comparison reports per-field state and observation differences, including first divergence and magnitude. State or observation equality is recorded only if observed; shared inputs must never be presented as silent backend equivalence.
6. Viewer telemetry identifies the active backend, and the required `DISPLAY=:1`, non-headless, `env=1` native run visibly advances through native MuJoCo stepping.
7. Existing observation/action/checkpoint ABI, provenance, diagnostic isolation, and compatibility entry points remain intact.

## Parameter provenance and invariants

`DexHandPhysicsConfig` remains the sole production physical source. Its frozen `DexHandServoConfig` and `DexHandMaterialConfig` own the accepted DexHand values: control/physics periods `0.01/0.001 s`, Euler/Newton with 80 iterations, base-position `5000/500/500`, base-rotation `10000/125/100`, finger `15/0/50`, hand/object friction `0.5/0.01/0.001`, and ground friction `0.5/0.5/0`. Backend choice does not create a second set of physics constants.

Scene construction and compiled validation continue to enforce source asset assumptions, timestep, integrator/solver settings, dimensions and ordering, gains and limits, friction/contact settings, reset-critical names, and capacities. Compiled model capacities (`njmax=2048`, `nconmax=8192`) remain distinct from MJX-Warp runtime allocation (`njmax=8192`, `naconmax=8192`). Native execution reuses the compiled model contract; it does not inherit MJX runtime allocation as physical provenance.

## Scope boundary

This change targets interactive single-environment stepping. It does not replace the MJX-Warp training or batch architecture, redefine exact trace authority, tune physics, transfer ManoRL numeric parameters, or claim native checkpoint Gym parity. The prior native 461D checkpoint limitation also remains: numerically faithful network mapping does not make MuJoCo and Gym physical feedback or state-dependent routed targets equivalent.
