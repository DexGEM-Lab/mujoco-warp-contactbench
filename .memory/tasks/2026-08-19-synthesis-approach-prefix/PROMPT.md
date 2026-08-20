## Objective
Implement an opt-in synthesis-only human-like approach prefix before the unchanged pre60 reference. Each synthesis reset/attempt samples a new seeded start. Finish when generated rows retain the original suffix exactly, use positive-Z 30–50 cm starts within XY ±30°, gate RL residual and 0.10 m deviation termination through the prefix, remain replay-compatible, and are documented/tested.

## Workbench
- Implement and validate immutable prefix generation.
- Wire per-trajectory control/termination gates into synthesis attempts.
- Use old checkpoints as explicit policy-transfer inference when padding/assets differ.
- Preserve compact v2_contact and record augmentation in companion manifest.

## Context
- Repository/worktree: /home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-synthesis-approach-prefix
- Base reference: pre60/post250 at 120 Hz; original pre60 and all later frames must remain byte-identical.
- Existing successful checkpoints may be pre180 and old geometry; they provide policy weights only.
- Synthesis attempts already run in isolated fresh processes with attempt_seed incremented per retry.

## Task specifications
- Sample hand start at 0.30–0.50 m from object, azimuth within ±30° of object→source-hand XY direction, elevation strictly positive (default +15° to +30°) with a 4 cm endpoint-smooth vertical approach arc.
- Generate adaptive 100–300 total pre-padding using a non-linear human-like profile with an exact discrete C1 velocity splice; prefix length is total minus base60.
- Select bounded source-conditioned orientation templates; keep finger pose fixed in v1.
- Prefix is the synthetic early phase: processed/cumulative residual is zero while step < prefix length and opens at original pre60 frame 0.
- The >0.10 m object deviation terminal is disabled only while step < prefix length and resumes at original pre60 frame 0.
- Each reset/attempt resamples deterministically from attempt seed + source identity.
- Synthesis inference must not write PPO memory.
- Keep compact v2_contact row schema; companion manifest records algorithm contract/config/per-row sample, while row seed+identity make sampling reproducible.

## Constraints
- Work only in feat/synthesis-approach-prefix.
- Preserve the primary dev worktree and user-modified test.sh.
- Do not change ordinary training early30 semantics.
- Do not treat a padding/asset-mismatched checkpoint as strict inference; use an explicit policy-transfer boundary.
- No subagents.
