## Objective
Make the refined RL-episode corpus train from its recorded actuator target track rather than replaying measured state as a command. Build a new immutable 120 Hz Lance dataset that preserves both `urdf_dof` measured state and `urdf_dof_target` actuator target from each provenance-pinned source RL row, then extend the ManoRL/MTP contract so initial conditions and evaluation use state while controller base commands use target. Recompile the four all-object shards and, after local replay plus 4096-env preflight passes, replace the current state-as-target baseline with fresh target-based 10,000-update training.

Done means: all 12,200 rows retain order/UUID/source provenance; state, target and object pose are timestamp-resampled together to 120 Hz; motion onset is valid on the new clock; MTP/checkpoint metadata binds state-vs-target semantics; random multi-object replay demonstrates target-driven actual state against recorded state; four MTPs cover 20 supported non-scissor objects/117 pairs/11,700 trajectories; and four production shards start fresh only after preflight.

## Workbench
1. Add an explicit refined-RL state/target import contract without weakening historical/generated-reference contracts.
2. Generate and validate a new target120 Lance dataset; never overwrite prior source or annotated Lance datasets.
3. Preserve measured state and actuator target as distinct arrays in ReferenceTrajectory and MTP.
4. Use measured state for initial qpos/ground-truth diagnostics and actuator target for controller commands plus RL residual.
5. Recompile, replay, preflight, then switch formal training atomically after evidence supports the new contract.

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
- Set both `--joint-scale-multiplier 2.0` and `--joint-max-offset-multiplier 2.0` for the new campaign. Do not silently change unrelated generic library defaults.
- New training uses `--pre-padding 60 --early-phase-steps 120 --max-deviation-distance 0.15`; all are resolved control-step/checkpoint contracts.
- Train from scratch; the stopped 1×/pre0/early30/0.10m checkpoint is evidence, not a resume source.
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
