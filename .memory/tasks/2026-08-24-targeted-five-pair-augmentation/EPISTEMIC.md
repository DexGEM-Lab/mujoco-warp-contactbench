# Current model

## Available sources and parents
banana:02, banana:18, and bowl:04 have 32, 29, and 48 accepted-parent descriptors. Five parents per pair have been selected deterministically from the farther half of pre60 frame-0 right-wrist/object distances, then diversified by relative XYZ/azimuth coverage.

Cylinder6 is absent from v295/all75 but clean v530 contains 50 trajectories for each of cylinder6:03 and cylinder6:09. All 100 pass authoritative right-hand decoding, pre60/pre180 compilation, finite-value checks, and movement_end+15 validity. Runtime assets pass. Canonical path-bound packages are:
- pre60 package `a5656f99d6788ed0cbb91f8e4a9a4879b3b095f7d42c3ffe3d8ddffbd0e991aa`, catalog `2a94792e73645f3ffe2acbefd1a751d212bafd9b79e3815cc58013c50d803a3a`.
- pre180 package `9ca04e74f5bfd0d599213a497f4dbf492c50a53b63d46c40b5a27162c17a16cc`, catalog `2df50d15daf4e65bbb2b125af9adc7e1d31d87781204094ede757d76316f33e3`.
Cylinder6 still needs GPU base-parent bootstrap and five-parent selection per pair.

## Production mechanism
Base-parent bootstrap uses pre180 plus explicit policy transfer. Final augmentation uses pre60, parent-bound object offset/raw start/anchor, 4 cm prefix, complete original tail, and no retreat.

Coverage is explicit rather than post-hoc random. Every selected parent has 50 Far slots (5 radius x 5 azimuth x 2 height) and 30 Near slots (5 empirical distance x 3 azimuth x 2 height). Each slot has twelve fallback seeds selected near that cell center. A failure only advances within the same cell, so bounded shortfall is visible and accepted positions do not silently collapse into easy regions. Near and Far are independent.

The runner reuses one MJX/checkpoint runtime per parent/mode, but every candidate is independently reset and evaluated by the same production three-rule gate plus prefix collision gate. Only accepted rows reach Lance. Status and manifests support interruption/resume and bind accepted rows to slot/seed/attempt diagnostics.

## Current plan
The existing three pairs have a frozen plan at `/home/jay/data/manorl_targeted_five_pair_augmentation_20260824/existing_pairs_coverage_plan_v1.json`: 15 parents, 30 tasks, 1,200 target rows, 14,400 unique fallback seeds, digest `0d96bed5018bc17b6462d4c46ab61aad089aee6cb6e6998f001ecc4447572d67`.

## Live uncertainty
All Server1 GPUs are currently occupied by other users, so no production has started. The next discriminating evidence is one Far and one Near coverage-slot smoke on an available GPU, followed by cylinder6 pre180 base smoke. If cylinder6 fails, the blocker is policy generalization rather than source/asset availability.
