## Objective
Modify the ManoRL synthetic Lance export path so compact target-replay/visualization data does not repeat large checkpoint runtime metadata or persist unused audit/training intermediates in every row. Preserve the existing v2.2 full/audit contract as an explicit compatibility mode and do not mutate already-published datasets.

## Workbench
- Inspect current v2.2 schema, row builder, exporter, validator, and replay consumers.
- Define a compact replay+visual schema with explicit version and metadata contracts.
- Implement opt-in/default CLI export behavior without silently dropping fields from the existing full contract.
- Add focused schema projection and round-trip tests; run writer/readback and relevant tests.
- Commit one coherent feature change and report exact validation evidence.

## Context
- Worker worktree: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-manorl-compact-lance`.
- Primary dev remains authoritative; do not edit it directly.
- Current v2.2 dataset contract: `synthetic_mano_28d_checkpoint_rollout_v2_2`.
- Current direct target replay reads `index`, `trajectory_metadata`, `timestamp`, `hands`, `objects`, and `provenance`; physical control uses `hands[right].urdf_dof_target`, object pose is used for initialization/comparison, and CCD uses only iterations/contacts settings from provenance.
- Size evidence from the current 5425-row NAS dataset: `provenance.checkpoint_metadata_json` is 2,826,639 identical bytes in every row (~15.33 GB total); rollout is ~13.68 GB, including duplicated observation_t/next_observation (~11.59 GB).
- Existing published NAS and local staging artifacts must remain untouched.

## Task specifications
- Keep full v2.2 row/schema generation available as an explicit audit/full mode.
- Compact mode must externalize checkpoint runtime metadata at dataset level or a sidecar and retain only small replay-relevant provenance scalars, including checkpoint identity and Warp CCD settings.
- Compact replay+visual mode should retain index/lineage, minimal trajectory metadata, target DOF, optional recorded DOF, object pose, and MANO visual fields when requested.
- Do not claim offline-training compatibility unless the required observations/actions/rewards are deliberately retained in a separately named schema.
- New schemas require explicit constants, metadata identifiers, and tests that reject accidental mixing with v2.2.
- Validate with focused tests, py_compile, a small Lance writer/readback smoke, and `git diff --check`.

## Constraints
- Work only in this linked feature worktree and assigned source/test/docs files.
- Do not use subagents from this worker session.
- Do not modify, delete, or republish any existing NAS/local generated Lance dataset.
- Do not change physics, action semantics, checkpoint provenance meaning, or direct replay behavior for existing v2.2 data.
- Preserve user-unrelated modifications and untracked files.
- Commit only explicit feature files after reviewing the staged diff.
