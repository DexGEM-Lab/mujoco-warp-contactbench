## Objective
Make ManoRL checkpoint synthesis default to five accepted episodes per raw identity with at most ten completed attempts, and penalize any late hand-object contact after a ten-frame grace period.

## Reward contract
- Existing positive expected-contact reward remains active on the inclusive source window `[contact_start, contact_end]`.
- Frames `contact_end+1..contact_end+10` are neutral for contact reward.
- For `step > contact_end+10`, any of the 16 hand keypoint contact-force magnitudes above `0.2 N` produces an environment contact term of `-3 * max_contact_reward = -1.2`.
- This late penalty is based on any hand-object contact, not the expected-contact mask.
- Preserve distance gating and object stability unless evidence shows a contradiction.
- Host NumPy and device JAX reward paths must be equivalent; bump reward/PPO contract IDs and checkpoint metadata.

## Synthesis contract
- Defaults: `target_episodes_per_identity=5`, `max_attempts_per_identity=10`.
- One attempt is one completed rollout candidate for one raw identity.
- Save only accepted complete episodes; assign `episode_index=0..4`, record generation attempt and seed, and derive unique generated UUIDs.
- If any identity has fewer than five accepted episodes after ten attempts, do not publish a complete final dataset; preserve exact counters/failure evidence.
- Publish a new Lance schema version because provenance/multiplicity semantics change.

## Inputs and delivery
- Source: Guangguan v295 right-hand cube2:02, 49 identities.
- Checkpoint: cube2 v5 checkpoint-000500.pt, SHA256 `dafa2135a4de46e52dfec8c0ee264ca82fcd1af18db6f402024f1499d6e7c0ee`.
- Existing checkpoint behavior will not change from reward relabeling; a newly trained policy is required to learn release behavior. Existing checkpoint is used only to validate the synthesis machinery and persisted reward values.
- The validated 245-row v2.2 run is diagnostic only; the user paused NAS publication on 2026-07-30. Keep the authoritative v2.1 NAS artifact unchanged.

## Training extension
- Train a new policy from scratch under the contact-v2 reward on every valid right-hand `cube2` identity.
- Coverage is the 213 predecoded identities across `cube2:01,02,03,04,10,11`, assigned once before deterministic padding to N4096.
- Use v6 action defaults, 5,000 requested updates, checkpoints every 100 updates, GPU1, and W&B group `cube2-all-contact-v2`.
- Do not load the old reward-v1 checkpoint as a strict resume. It remains the behavioral baseline.
- Keep the feature worktree intact while the remote process imports from it; source edits to runtime/reward modules are frozen for the duration of the run.
