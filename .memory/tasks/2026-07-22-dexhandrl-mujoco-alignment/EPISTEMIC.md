# Epistemic Record

## Current model

The migration branch is a historical fork from `8ab80fc`, whereas current `dev`
contains the complete MANO reference migration and 171 subsequent commits. Its
first DexHandRL commit adds an independent `sim/dexhandrl` implementation,
assets, scripts, tests, and optional runtime dependencies. Applying its changes
onto current `dev` must preserve both surfaces; conflicts from historical files
are integration conflicts, not evidence that current MANO behavior should be
removed.

Direct rl-games checkpoint reuse is the milestone's causal target. A successful
`cube1`/`01` result requires a common trajectory selection and the same ordered
boundary transformations: reference state -> raw policy observation -> running
normalization -> model action -> simulator control target -> rollout state.
Passing imports can establish that code is wired; it cannot establish parity.
The focused inventory will determine whether the dataset, IsaacGym source
checkout, checkpoint, and compatible Python runtime are available to execute
that chain.

## Open discriminating question

Which first boundary diverges for `cube1`/`01` once the exact reference dataset
and checkpoint are available: trajectory selection, raw observation, normalizer,
policy action, control mapping, or physics rollout?

## Established availability

The local vendored DexHand assets, default Lance archive, and an IsaacGym source
checkout with candidate `.pth` checkpoints are accessible. Execution is blocked
before any imported-code boundary because this worktree lacks `.venv/bin/python`.
The executable baseline therefore needs an environment satisfying this repo's
Python 3.11 lock before it can identify the first parity divergence.
