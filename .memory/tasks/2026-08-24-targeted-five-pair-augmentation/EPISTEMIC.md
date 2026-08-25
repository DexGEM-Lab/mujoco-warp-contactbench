# Current model

## Parent qualification and selection
A production parent must pass both the current persisted-float32 atomic gate and accepted-parent late-contact/anchor eligibility. Historical descriptor existence alone is insufficient; `banana_02_1256` was the GPU counterexample (24/24 prefix candidates complete/contact-qualified but 46.03–56.27 deg final rotation).

Parent discovery is now deliberately broad and simple: batch-run candidates, apply hard gates, then choose robust and spatially diverse parents. The final selector uses every hard-qualified candidate; pre60 wrist/object distance is a soft preference. Quality is the weakest normalized margin among final rotation to 35 deg, contact frames above 100, and last-contact lateness through movement_end+15. The first parent scores 0.70 quality + 0.30 distance; subsequent parents score 0.50 quality + 0.35 spatial separation + 0.15 distance.

This removed fragile distance-only choices. Existing-pair selected minimum quality margins are banana:02 0.630, banana:18 0.190, bowl:04 0.892.

## Cylinder6 pool
Clean v530 pre60/pre180 bundles contain all 100 source trajectories. Explicit policy transfer runs both actions to reason 1. Screening rounds seed42–48 used zero offset once and bounded XY02 thereafter. XY02 is screening-only and every artifact is `publish=false`.

Bulk seeds46–48 added enough late-contact parents. Merging all descriptor sets yields:
- cylinder6:03 — 8 distinct eligible source identities.
- cylinder6:09 — 10 distinct eligible source identities.
- 31 physical parent variants across 18 sources.

Different no-prefix bootstrap datasets can reuse the same generated base UUID because legacy UUID identity omits seed/offset. Parent variants are therefore identified by accepted-parent descriptor SHA, which includes parent dataset/version/row, seed, fixed offset, checkpoint, source, and anchors. The final plan binds this SHA and fails closed on replacement. Final prefix rows remain unique through v4 augmentation identity.

## Formal XY contract
Formal Near/Far collection draws no new random object XY offset: `formal_random_object_xy_offset_range_m = 0.0`. Each selected accepted parent retains its fixed object offset as part of the parent ABI. Screening randomness is frozen into the selected descriptor and is not resampled during production.

## Final plan
The final selection has five parents for each of banana:02, banana:18, bowl:04, cylinder6:03, cylinder6:09. Plan:
- path: `/mnt/user-home/jay/data/manorl_targeted_five_pair_augmentation_20260824/final_five_pair_coverage_plan_v1.json`
- digest: `6d320ada8c6bf3de6ed057a02a3bd26bc99ef867fcd50d51ee633a2275434d9d`
- 25 parents, 50 parent/mode tasks.
- 250 Far and 150 Near target slots per pair; 2,000 total.
- 24,000 unique same-cell fallback seeds.
- Far: 5 radius x 5 azimuth x 2 height per parent.
- Near: 5 empirical distance x 3 azimuth x 2 height per parent.

The runner preserves the complete pre60 source tail, uses no retreat, disables formal random XY, reuses the parent fixed offset, applies atomic acceptance plus prefix collision gate before append, and journals pending writes for deterministic recovery.

## Live question
A formal-runner smoke is running on the highest-quality selected cylinder6:03 and cylinder6:09 parents for Far/Near slot0. The decisive evidence is at least one accepted row plus independent validation of v4 identity, random XY=0 manifest, parent fixed offset, complete tail, and atomic gate. On success, start the resumable formal 2,000-slot run.
