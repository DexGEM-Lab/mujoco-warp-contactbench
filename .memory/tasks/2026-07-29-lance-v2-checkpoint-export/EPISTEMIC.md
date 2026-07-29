## Phenomenon
The requested artifact is a corrected v2 synthetic Lance generated from complete deterministic ManoRL checkpoint episodes for cube2:02. Gym's historical v1 is structurally useful but carries a 100/200 Hz mismatch, a wrist fallback mislabeled as joint-frame force, a 0.5 force multiplier, and an object-force sign change.

## Supported mechanism
MJX-Warp exposes raw contact positions, geom pairs, contact frames, and solved pyramidal force rows. The exporter reconstructs only the normal component as `sum(pyramid) * frame[0]`, orients it hand-to-object from geom ordering, and applies no scale. Actual live hand-link, wrist, and object body transforms produce every local position/force field. A real cube2:02 MJX-Warp transition confirmed `contact__pos` availability and the added processed-action diagnostic boundary.

The exporter records the reset state plus exactly T-1 policy transitions, stopping each vector row at its first trajectory-complete terminal. This yields T state/contact frames and T-1 auditable rollout records without concatenating post-reset episodes. It refuses duplicate assigned identities and early deviation failures by default.

## Current claim
The implementation now provides:
- explicit Arrow schema `synthetic_mano_28d_checkpoint_rollout_v2` with data_fps=200 and dt=0.005;
- normal-only hand-to-object force in every frame at scale 1.0;
- actual link-frame `pos_joint`/`total_force_joint` and consistent object-frame force;
- 28D physical/controller targets, 21 keypoints, reference state, observations, mean/processed actions, residuals, rewards and termination;
- checkpoint SHA/path/update/runtime sidecar and source identity provenance;
- Lance writer output readable by both local Lance 0.24 and Lance 7.0 in a compatibility probe.

Focused schema/contact/MANO/writer/action/episode/shell tests pass. The decisive remaining evidence is a real Server2 GPU run with checkpoint-000500.pt and subsequent nested-row invariant validation.
