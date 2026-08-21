# All-banana prefix-retreat augmentation

## Objective
Produce a new, separate banana augmented dataset with prefix and v4 retreat for every source that satisfies the strict accepted-parent v3 anchor contract. Preserve the historical raw dataset unchanged.

## Contract
- Historical `/mnt/nas-222-projects/sunjieqiang/mujoco_synthetic/gg_f120_xy02cm_8obj_contact_20260811.lance` is immutable input and must not be overwritten.
- New output is a separate `banana_augmented_prefix_retreat` path.
- Strict eligible source set is the 263 banana identities with an accepted source row satisfying movement-end contact, movement-end+15 anchor, and nonzero horizontal retreat direction. The 56 ineligible identities are recorded in an unavailable report and excluded from the strict formal Lance.
- Each eligible source targets five augmented episodes (`1:5`), with at most twelve attempts per source. Actual accepted yield is delivered if the target is not reached.
- One accepted-parent descriptor is bound to each source identity. A multi-environment attempt batch must use each environment's descriptor for its own frame-0 raw hand pose, object XY offset, movement-end+15 anchor, augmentation identity, and provenance.
- Prefix mode is far; every accepted row includes the v4 retreat suffix. Prefix uses the established 4 cm endpoint-smooth arc. No policy/residual is used in prefix; v4 retreat stops new policy/action accumulation and smoothly unloads entry residual.
- Clock is 120 Hz control/reference, 480 Hz physics, four substeps. Output is compact `synthetic_mano_target_replay_visual_v2_contact`.
- New seed block and episode-index namespace must not collide with old rows. Production uses episode indices 5..9 and deterministic action-separated seed bases.
- Use all available stable parallelism on Server1, but do not exceed the measured banana convex-collision capacity. Action-homogeneous batches target at most the number of eligible identities in that action (<=50) unless a pilot proves a larger stable batch.

## Owned implementation
- `tools/export_manorl_synthetic_lance.py`: per-identity accepted-parent mapping, descriptor manifest, episode-index offset.
- `tools/run_manorl_all_banana_prefix_retreat.py`: action-sharded multi-GPU orchestrator.
- Focused tests and task memory.

## Acceptance
- 263 strict source identities represented in the input manifest; 56 unavailable identities explicitly listed.
- Each attempted identity has counters with attempts <=12 and saved <=5.
- Every accepted row has prefix and v4 retreat metadata, unique UUID, unique seed/episode pair, checkpoint SHA, source identity, contact/reference/command mapping, and 120/480x4 contract.
- Actual output may be below 1315 rows; no retries outside the 12-attempt budget and no parameter changes to improve yield.
- Validate compact schema before local staging and `cp -r` NAS publication.

## Constraints
- No subagents.
- Work only in this feature worktree until merged through GitGuard.
- Do not touch primary worktree user modifications or old Lance datasets.
