## Phenomenon
The delivered artifact is a corrected v2 synthetic Lance generated from complete deterministic-mean ManoRL checkpoint episodes for cube2:02. Gym's historical v1 remains structurally useful but carries a 100/200 Hz mismatch, a wrist fallback mislabeled as joint-frame force, a 0.5 force multiplier, and an object-force sign change.

## Supported mechanism
MJX-Warp exposes raw contact positions, geom pairs, contact frames, and solved pyramidal force rows. The exporter reconstructs only the normal component as `sum(pyramid) * frame[0]`, orients it hand-to-object from geom ordering, and applies scale 1.0. Actual live hand-link, wrist, and object body transforms produce every local position/force field.

The exporter records the reset state plus exactly T-1 policy transitions, stopping each vector row at its first trajectory-complete terminal. This yields T state/contact frames and T-1 auditable rollout records without concatenating post-reset episodes. It refuses duplicate assigned identities and early deviation failures by default. Predecoded inputs are SHA256 checked and preserve the existing workaround for unstable nested source-row reads.

NumPy and Torch/CUDA use a recorded seed (42 for the delivered artifact). This fixes the stochastic point-cloud observation source. It does not make MJX-Warp GPU contacts bitwise deterministic: repeated seed-42 smoke runs had identical point-cloud observations but <=7.45e-7 initial contact-feature differences, which the closed policy/physics loop amplified. The artifact is one complete, validated realization of the recorded contract.

## Validated claim
`synthetic_mano_28d_checkpoint_rollout_v2` records:
- data_fps=200 and dt=0.005;
- normal-only hand-to-object forces at scale 1.0 in world/wrist/actual-link/object frames;
- 28D physical/controller targets, 21 keypoints and object pose;
- reference state/frame indices, observation and next observation, deterministic mean and processed actions, cumulative residuals, reward and termination;
- checkpoint path/SHA/update/runtime sidecar, v5 action contract, software commit, seed and source identity.

The final Server2 dataset contains every 49 valid cube2:02 identity exactly once: 21,558 states, 21,509 transitions, 16,021 contact frames and 52,111 contact points. All rows end with one success code. Position and force-frame invariants hold below 1.1e-6 in their respective units. The policy produced 602,252 nonzero action values, so the artifact is checkpoint-generated rather than a reference copy.

## Remaining boundary
Server2 Lance 0.24.1 can nondeterministically segfault while importing/decoding large nested rows. This is a reader-runtime failure, not a malformed-row signal: isolated per-row decoding with bounded retries validated all rows, with one of 49 requiring a second attempt. The validator exposes retry counts instead of hiding this evidence. Lance 0.24 output compatibility with Lance 7.0 was separately established on the same explicit nested schema.
