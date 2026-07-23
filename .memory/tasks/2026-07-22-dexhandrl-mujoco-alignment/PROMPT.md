# ManoRL-Style DexHandRL Refactor

## Objective

Inspect the current `dev` ManoRL implementation and refactor DexHandRL so its structure, coding style, and parameter provenance align with ManoRL wherever the two tasks share semantics.

Treat this as a T3 cross-module contract change. Preserve DexHandRL-specific observation, action, and checkpoint contracts. Use ManoRL as the architectural and provenance reference for shared mechanisms, not as a source of physics constants to copy wholesale into a different hand/task.

## Parameter contracts

Make every relevant parameter's ownership explicit:

- **Common parameters** represent genuinely shared simulator or runtime semantics. They should follow the same organization, naming, default-resolution pattern, and provenance documentation as ManoRL where applicable.
- **Task-specific parameters** encode DexHand hardware, trajectory, observation/action layout, checkpoint compatibility, actuator/contact behavior, or other DexHand-only semantics. Keep them in DexHand-owned configuration with an explicit source or rationale.
- A numeric value appearing in ManoRL is not evidence that it applies to DexHandRL. Adopt a ManoRL physics value only when the underlying semantics and model assumptions are shared and evidence supports the transfer.
- Diagnostic and experiment-only overrides must remain visibly separate from production defaults and must not alter default behavior merely by being present or importable.

## Required work

- Inspect the current `dev` ManoRL control flow, configuration boundaries, parameter provenance, environment construction, rollout path, and tests before choosing the DexHandRL refactor boundary.
- Align DexHandRL module structure, naming, style, configuration flow, and parameter-resolution patterns with ManoRL for shared semantics.
- Preserve DexHandRL observation layout and normalization, 22-dimensional action and state-dependent full-target routing semantics, native checkpoint loading/mapping, and trace/checkpoint compatibility unless an explicitly approved contract change is supported by evidence.
- Provide a native, visible DexHandRL rollout path with residual correction disabled (`residual=false`). For the interactive `env=1` viewer, physics must advance through native MuJoCo stepping on CPU. This backend must use the same generated scene, immutable `physics_contract`, controller, reset, ordered filters, and task/frame/target semantics as the MJX-Warp path; it must not be a diagnostic replay or injected Gym trace. It must obey the existing visible-run contract: `env=1`, `DISPLAY=:1`, headless disabled, and viewer enabled when supported.
- Keep MJX-Warp unchanged as the default and authoritative backend for training, batched execution, and exact reference traces. Native MuJoCo must be an explicit CPU-only viewer backend and must never be selected silently by a default or fallback.
- Add a fixed cube1/action02/sequence0, residual-off, 50-control-step native-MuJoCo-versus-MJX comparison. Drive both backends with exactly the same frames, reference targets, controller outputs, and ordered-filter outputs. Report state and observation divergence explicitly; do not claim or silently assume numerical equivalence.
- Keep diagnostic flags fail-closed and behavior-neutral by default. Enabling, importing, or defining diagnostic facilities must not change production defaults; any diagnostic intervention must require explicit opt-in and record its provenance.
- Add focused regression tests for shared-vs-task-specific parameter ownership, default behavior, the native `residual=false` path, preserved observation/action/checkpoint contracts, and diagnostic isolation.
- Compare refactored behavior against the existing recorded DexHandRL baselines, including native checkpoint behavior and the established exact-control/source-conditioned reference. Report whether differences arise from structural refactoring, intentional parameter provenance changes, or pre-existing simulator-specific dynamics.

## Acceptance target

The refactor is accepted when:

1. shared ManoRL/DexHandRL mechanisms have aligned structure and configuration style without duplicated or conflicting sources of truth;
2. common and DexHand-specific parameters have explicit, testable ownership and provenance contracts;
3. DexHand-specific observation, action, routed-target, and checkpoint contracts remain compatible;
4. the explicit CPU-only native MuJoCo backend powers the interactive `env=1` visible `residual=false` rollout, while MJX-Warp remains the unchanged default and authority for training, batches, and exact traces;
5. native and MJX paths consume the same scene, `physics_contract`, controller, reset, filters, frames, targets, and task semantics, with backend identity visible in viewer telemetry;
6. a fixed cube1/action02/sequence0 residual-off 50-step comparison proves exact frame/target/control inputs and reports state/observation divergence without an equivalence claim;
7. diagnostic flags and tooling do not affect production defaults unless explicitly enabled;
8. focused regression tests cover the changed contracts; and
9. behavior is compared with existing baselines, with observed changes attributed to a concrete mechanism rather than assumed parity.

## Constraints

Work only in this linked feature worktree. Preserve current ManoRL behavior and unrelated files. Do not wholesale-copy ManoRL physics numbers into DexHandRL. Do not substitute reward tuning, retraining, or Gym-trace injection for the native rollout contract. Use explicit-file staging, inspect staged diffs, and commit coherent checkpoints. Do not merge this feature into `dev`.

All visual tests and final rollouts must use `env=1`, export `DISPLAY=:1`, and run explicitly non-headless with the viewer enabled when the runtime supports one. Viewer telemetry must name the active backend. Numerical trace collection may remain headless. The native backend is CPU-only; requests that violate its single-environment interactive contract must fail explicitly rather than fall back or change the MJX-Warp default.
