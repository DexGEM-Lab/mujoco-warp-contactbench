# Epistemic Record

## Refactor theory

The next milestone is a behavior-neutral structural refactor of DexHandRL. ManoRL is the reference for architecture and evidence discipline: immutable configuration contracts, a single scene source of truth, structured compilation, validation of the compiled model, explicit runtime capacities, one production rollout/viewer path, compatibility shims at legacy entry points, and baseline artifacts that make unintended changes mechanically detectable. ManoRL's numeric physics parameters are not transferable evidence for DexHandRL because the hands, actuators, contacts, observations, and source simulator semantics differ.

DexHandRL's accepted behavior is frozen as the `legacy_port_v1` profile. It owns the existing control period (`0.01 s`), physics period (`0.001 s`, ten substeps), Euler integrator, Newton solver with 80 iterations, servo gains and effort limits, hand/object/ground friction, contact configuration, reset sequence, observation/action/checkpoint ABI, active-to-full target routing, and default CLI/API behavior. These values are preserved because they define the existing port and its recorded baselines, not because they are presumed physically optimal. Structural Stage 1 performs no parameter tuning.

## Target contract

A central immutable DexHand contract is the sole production source for physics, servo, material/contact, reset, and capacity settings. Every field carries one provenance class:

- `source_contract`: dictated by an upstream asset, dataset, or DexHand/Isaac interface;
- `legacy_port_v1`: inherited MuJoCo-port behavior frozen for compatibility;
- `runtime_capacity`: allocation-only limits that must not alter scene semantics;
- `diagnostic_override`: explicit opt-in experiment state, excluded from production defaults.

The contract names the integrator, solver, solver iterations, and both time scales explicitly. Scene generation consumes the contract rather than performing string replacement or rediscovering defaults. It emits a structured MuJoCo document and fails closed when required source assets, names, dimensions, or provenance are absent. Compilation is a separate boundary and validates the resulting model's timestep, integrator, solver, iterations, joint/actuator order and dimensions, servo arrays and limits, materials/friction/contact masks, reset-critical names, and model capacities.

Model compiler capacities and MJX runtime capacities remain distinct contracts. The frozen scene/model values are `njmax=2048` and `nconmax=8192`. MJX/Warp allocation remains `njmax=8192` and `naconmax=8192`. A refactor must never serialize the runtime constraint allocation into the model's `njmax`, nor conflate MuJoCo `nconmax` with MJX/Warp `naconmax`.

## Runtime and compatibility model

There is one production viewer/rollout implementation. With `residual=false` and no checkpoint it supplies a zero 22-dimensional action through the native production environment; it does not inject a diagnostic trace or bypass control construction. Rendering uses the same passive mirror primitive as other production visualization: simulation advances only in the environment stepper, while the viewer copies the selected environment state into a native MuJoCo viewer model under the viewer lock and synchronizes without stepping physics.

Existing command names and import surfaces remain compatibility shims that translate legacy arguments into the central contract and delegate to the production implementation. CLI and Python API defaults, observation width/layout and normalization, 22-dimensional action semantics, state-dependent 26-DoF full-target routing, checkpoint mapping, termination/reset behavior, and output conventions remain unchanged.

## Stage 1 preregistered acceptance

The baseline frozen at clean HEAD `ac2cc9c645da5b47bf417ae6ab488aeeeeb1dcbf` is the Stage 1 oracle. Under the same CPU backend, dependency versions, asset/dataset bytes, configuration, seed, and machine execution environment, all of the following must hold:

1. The default generated XML is byte-identical and has the same SHA-256.
2. The compiled model signature is bit-exact, including dimensions, names/order, option fields, capacities, actuator arrays, joint arrays, and relevant geom/material/contact arrays.
3. Every array in the reset-plus-50-step zero-action NPZ is bit-exact in dtype, shape, and C-order bytes. Numeric tolerance, dtype coercion, or selective-array comparison is a failure.
4. Existing CLI/API defaults and compatibility entry points are unchanged.
5. Focused contract, scene/compiled-validation, environment, ABI/checkpoint, zero-action, viewer/passive-mirror, and compatibility-shim tests pass.
6. A real graphical run succeeds with `DISPLAY=:1`, `env=1`, action `02`, no checkpoint, `residual=false`, and the production viewer. The observed policy input is zero action and the viewer mirrors state without stepping it.

Any XML, model-signature, or NPZ-array difference blocks Stage 1 and must be localized before further work. No mismatch may be explained away as an expected consequence of refactoring because the proposed mechanism is behavior-neutral.

## Evidence boundary

The earlier PhysX-versus-MuJoCo parity investigation established why ManoRL values must not be copied numerically: direct transfers of Gym finger and wrist gains degraded response, while DexHand-specific COM preservation and contact evidence identified task-local mechanisms. Those experiments remain historical evidence about simulator transfer. They do not authorize Stage 1 tuning and are outside its acceptance surface.

The current `dev` ManoRL implementation supports the architectural theory: it separates immutable contracts, structured scene/model construction, compiled validation, runtime MJX capacity allocation, environment stepping, and a passive viewer mirror. DexHandRL currently spreads the corresponding ownership across scene generation, environment configuration, replay/viewer scripts, and CLI surfaces. The refactor should remove that ownership ambiguity while preserving `legacy_port_v1` bytes and behavior.

Evidence provenance and frozen artifact hashes: `OPS.md`.
