# Position Action Scale and Deviation Reset Contract

## User request

Change the default normalized action-to-physical residual mapping and training reset threshold:

- Position XYZ single-step scale: `0.003 m` per normalized action unit.
- Position gamma: unchanged at `0.9`.
- Position cumulative limit: `+/-0.03 m`.
- Rotation effective scale: unchanged at `0.00025`, instantaneous.
- Thumb/finger scales, gamma, and limits: unchanged.
- Object/reference-object deviation reset threshold: change from `0.10 m` to `0.15 m`.
- Normalized 26D Gymnasium/PPO action space remains `[-1, 1]`.

## Required behavior

- Update the single source of truth defaults in residual action and environment config.
- All terminal-enabled entry points (training runtime, bounded evaluation, viewer, Rerun recorder tool, local visual wrapper through trainer) must resolve `0.15 m`; terminal-disabled diagnostics remain effectively unbounded.
- Boundary semantics remain strict `distance > threshold`; exactly `0.15 m` does not reset, above it does, and early-phase suppression remains unchanged.
- Position recurrence must be `offset[t] = 0.9 * offset[t-1] + 0.003 * action[t]`, clipped to `+/-0.03 m`; verify steady state and both signs.
- Rotation and all joint scaling values must be regression-tested unchanged.
- Introduce an explicit environment/control contract identifier serialized in native checkpoint sidecars, runtime metadata, metrics/W&B/Rerun metadata. Strict resume must reject old checkpoints missing/mismatching this contract; inference-only visualization remains allowed.
- Update durable docs and tests; do not rewrite historical Isaac source ABI inventory claims as though source defaults changed. Clearly distinguish new target training defaults from source MANOHand values.

## Constraints

- Work only in `feat/action-scale-reset-threshold`.
- Preserve reward equations, raw 1.0x PPO reward scale, 26D normalized action bounds, early phase 100, gamma values, rotation and joint mapping.
- No Server2/GPU/process operations or training launch.
- Do not touch `PI_HANDOFF.md` or generated outputs.

## Acceptance

- ABI/environment/viewer/train/checkpoint/W&B/Rerun focused tests pass.
- Tests demonstrate exact 0.003/0.03 and strict 0.15 boundary through production configuration paths.
- Strict checkpoint resume rejects prior control contract; inference loader remains compatible.
- Independent review finds no stale 0.10 terminal-enabled path or accidental rotation/joint/action-space change.
