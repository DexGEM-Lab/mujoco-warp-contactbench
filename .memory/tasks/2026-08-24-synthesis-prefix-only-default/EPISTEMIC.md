# Current model

## Phenomenon
The approach transform already preserves the original reference after prepending Far/Near frames. Near's movement-end+15 dependency is only an endpoint-distribution input. Previous production runners nevertheless requested retreat suffix replacement, and the three-predicate gate excluded prefix trajectories.

## Implemented mechanism
The default production contract is `manorl_pre60_far_near_approach_prefix_only_complete_original_tail_v1`:
- canonical pre60 source;
- seeded Far or Near 4 cm approach prefix;
- Near uses movement-end+15 only to sample its start;
- complete original reference tail copied unchanged;
- no retreat suffix and no tail policy/residual manipulation.

Every no-retreat candidate passes the three-predicate quality gate, conjuncted with the prefix collision gate. Prefix-only rows use `manorl_synthesis_prefix_only_augmentation_identity_v4`, which is persisted in both full and compact Lance schemas. The validator binds pre60/4cm/Far-Near/no-retreat/v4 identity and independently recomputes quality acceptance.

## Production entrypoints
`synthesize.sh` defaults to accepted-parent Far, requires a predecoded pre60 manifest, uses 5 targets/12 attempts, preserves bounded partial yield, and rejects retreat. `tools/run_manorl_all_objects_near_far.py` runs five Far plus five Near per verified source, never passes retreat, propagates `--replace`, inherits the process environment, runs one sequential object queue per GPU, and surfaces worker setup exceptions in its summary.

## Evidence and claim
Local focused tests cover unchanged tails, gate composition, identity persistence, append, shell defaults, runner commands/failures, and historical identity separation.

Server1 GPU0, canonical checkpoint and pre60, source `banana_01_052`:
- Far: 189-prefix + 566 original frames, 5.2888° final mean rotation error, 300 contact frames, one accepted row.
- Near: 54-prefix + 566 original frames, 5.0819°, 308 contact frames, one accepted row.
- Both manifests record `retreat_suffix=null`; both Lance rows carry distinct v4 identities.
- Independent validator passes both.
- Every persisted post-prefix reference value equals the original pre60 bundle at its float32 storage representation; source indices are exactly equal.

Two earlier smoke failures were implementation defects caught before integration: duplicate single/mapping parent parameters, then a numpy boolean in JSON diagnostics. Both received direct regression tests. The final smoke supersedes them and is diagnostic-only.
