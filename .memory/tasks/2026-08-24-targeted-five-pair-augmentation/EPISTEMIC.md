# Current model

## Available sources
banana:02, banana:18, and bowl:04 already have 32, 29, and 48 accepted-parent descriptors respectively under the v295/pre60 lineage.

Cylinder6 is absent from v295/all75 but clean v530 contains 50 trajectories for each of cylinder6:03 and cylinder6:09. All 100 pass the authoritative 120 Hz right-hand decoder, pre60/post250 and pre180/post250 compilation, finite-value checks, and movement_end+15 validity. Runtime assets and digest checks pass.

## Cylinder6 packages
- pre60 MTP: package `06fc8b999d3864196dd2ccc0118242cb0572abe558799f870afc195caea07c12`, 100 trajectories; final augmentation reference.
- pre180 MTP: package `743852137e7cdc83eda140e723030abb33aa44ccf6c7b6bed74e68306b53121a`, 100 trajectories; base-parent bootstrap.
- Matching predecoded bundles exist locally. Both are source v530 and are intentionally distinct from the checkpoint's v295 package signature.

## Mechanism
Base-parent production uses pre180 and explicit policy transfer. The transfer boundary ignores only environment signature differences while still validating reward/environment families, model architecture, tensor finiteness, and exact state-dict shapes. Every base candidate still passes the atomic quality gate. Successful base rows can then be converted into accepted-parent v3 descriptors. Final Near/Far uses pre60, parent-bound offsets, 4 cm prefix, complete original tail, and no retreat.

## Selection
The requested 150 Near and 250 Far are pair-level totals across five selected parents. Selection must prefer greater pre60 frame-0 right-wrist/object distance while maximizing relative XYZ/azimuth coverage; it cannot be simple top-five distance ranking.

## Live uncertainty
The all75 checkpoint has never been evaluated on cylinder6. Server1 GPU policy-transfer smoke is the discriminating test. If no cylinder6 base row passes the quality gate, augmentation is blocked by policy generalization rather than source or asset availability.
