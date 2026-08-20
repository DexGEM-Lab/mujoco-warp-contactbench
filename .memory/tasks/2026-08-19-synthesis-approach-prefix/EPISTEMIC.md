# Current model

## Phenomenon
Successful grasp/placement trajectories are sensitive to their approach initial condition. The desired artifact expands only the free-space approach distribution while preserving the solved pre60/grasp/lift/place suffix.

## Mechanism
One accepted row from the prior scalable synthetic Lance fixes source identity, exact checkpoint SHA256, reference clock, and successful object XY offset. A seeded start is sampled in a bounded positive-Z spherical sector around the source XY approach direction. A quintic wrist translation/orientation prefix with exact discrete C1 splice ends at the original pre60 frame0 state; fingers remain in the source frame0 pose. During the prefix policy forward is not called and effective/cumulative residual stays zero. At the splice, deterministic checkpoint policy residual and normal deviation termination resume.

## Contracts
- Base reference is pre60/post250 at 120 Hz.
- Old checkpoint environment ABI is not the reference truth; it is a policy-transfer source.
- Exactly one env is active; each isolated synthesis reset/attempt rebuilds a fresh immutable prefix from attempt_seed + accepted source identity.
- Compact v2_contact remains the row schema; augmentation is reproducible from row seed/identity plus companion manifest contract/config.
- Ordinary training trajectories remain early30 and strict checkpoint inference remains unchanged.

## Ruled out
- Batched viewer worlds advanced in the background and caused apparent last-contact jumps; final reset semantics use one active env and rebuild from frame0.
- Strict raw-acceleration matching was falsified by a 3429-row audit: capture differentiation noise created non-human spikes. Exact discrete C1 velocity matching is retained instead.
- Dynamic mutation of references inside the general training reset path: unnecessary and would broaden training ABI.
- PPO-memory omission as a blocker: synthesis does not call record_transition.
- Linear interpolation: violates human-like smoothness and splice dynamics.

## Open checks
- End-to-end production validation must confirm compact movement metadata shifts by L, prefix policy forward is skipped, exact suffix persists within serialization tolerance, and target replay reads the augmented movement window.
