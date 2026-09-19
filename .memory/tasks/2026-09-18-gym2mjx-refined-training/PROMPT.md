## Objective
Make the refined 12,200-row RL-episode Lance dataset usable by ManoRL, measure physical zero-residual replay success on all 600 `mayonnaisebottle` rows using the fixed `cheyingtong` physical hand at 120 Hz, start a server-side training run from an immutable Lance-free package with both joint residual scale and cap multipliers set to 1.0, and annotate every source Lance row with the reference-object movement onset. Done means the replay denominator and outcome are explicit, the trainer has crossed a meaningful startup/contact/reset boundary, movement annotations cover all 12,200 UUIDs in a new immutable Lance dataset with confidence diagnostics, and checkpoints/logs/config are durable and inspectable.

## Workbench
1. Add an explicit refined-RL-episode import contract on `feat/gym2mjx` without weakening historical/generated-reference contracts.
2. Compile and validate the six mayonnaise action groups, then run vectorized physical replay to completion and report success by action.
3. Compile the intended training catalog, stage it on reliable Server1, and launch fresh training with joint multipliers 1.0.
4. Preserve run identifiers, package/checkpoint digests, logs, and the exact server/GPU configuration.
5. Derive sustained reference-object motion onset for all 12,200 rows, publish a separate annotated Lance dataset from Server1, and preserve per-row confidence/anomaly evidence.

## Context
Repository worker: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-gym2mjx`.
Source dataset: `/mnt/nas-222-project/sunjieqiang/new_vla/mano_rl_refined_100_per_action.lance`, version 1.
Source selection manifest: `/mnt/nas-222-project/sunjieqiang/new_vla/mano_rl_refined_100_per_action.manifest.json`.
Reliable training host alias: `manorl-server1`; deployment root historically `/home/jay/dexrobot/FromSSH/manoRL_mujoco_dexstream`.
The dataset contains 12,200 rows / 122 object-action groups. Mayonnaise has actions 01,02,03,04,05,08 with 100 rows each and mixed 100/120/200 Hz source clocks.

## Task specifications
- Keep source Lance immutable; never overwrite or compact it in place.
- The current project `pylance==0.24.1` can read the manifest/schema but panics on data pages because the file uses `lance.encodings21.PageLayout`; pylance 7.0.0 reads rows successfully. Establish a reproducible compatible decoder boundary.
- Refined rows are generated full RL episodes with one active object, `index.scene`, `index.action_code`, one active hand, full timestamps/28-DoF hand targets/object poses, and `provenance.source_rl_*`; they omit raw-capture `trajectory_info.object_move` and canonical synthetic-reference checkpoint provenance.
- Import must preserve row UUID, source provenance, action identity, source timing, and full-episode boundaries. All rows, including 100/200 Hz sources, must be normalized through the existing timestamp-based trajectory resampler to 120 Hz for replay, packaging, and training.
- Ignore source-operator/source-hand-shape identity for physical model selection. Every row must execute with the repository's canonical `cheyingtong` hand asset profile; bind that asset manifest into MTP/checkpoint provenance rather than silently switching per row.
- Do not route refined rows through the canonical generated-reference contract; that contract means saved 120/480 Hz ManoRL physical episodes and has stricter provenance/asset requirements.
- Replay success means the MJX physical environment reaches each reference horizon under zero residual action before the normal position-deviation failure threshold. Report aggregate and per-action numerators/denominators; source-recorded object motion alone is not replay evidence.
- Use `mayonnaisebottle` as the replay object and include all 600 rows unless a deterministic invalidity is recorded with row identity and reason.
- Set both `--joint-scale-multiplier 1.0` and `--joint-max-offset-multiplier 1.0` for training. Do not silently change unrelated generic library defaults.
- Production training must consume a verified local MTP package, not map Lance/PyArrow in the long-lived process.
- Movement annotation is reference-only: use the stored object position/orientation track, not policy replay or contact. Distinguish initial gravity/settling transients from post-settle task movement.
- Annotate every row. High-confidence rows use a stable-baseline departure; rows without a clean stable window or without post-settle motion receive an explicit low-confidence/fallback status rather than silent omission.
- Preserve the source Lance byte/logical content. Publish a new dataset containing standard `trajectory_info.object_move` plus a per-row reference-motion annotation, diagnostics JSONL, and manifest; never overwrite the source.
- Prefer Server1 because prior Server2 MTP training reproduced a host-level SIGSEGV after checkpoint 25.

## Constraints
Do not modify or commit from protected `dev`.
Stay on `feat/gym2mjx` for implementation.
Do not use subagents without explicit user authorization.
Do not delete source Lance, packages, checkpoints, logs, or other users' processes.
Use focused tests before broad validation.
Use tmux or another inspectable persistent session for long-lived server jobs.
