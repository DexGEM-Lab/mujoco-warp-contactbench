## Objective
Make the default direct hand-object contact contribution exactly 3x its current value without changing the contact-quality basis, distance gate, or distance rewards. Version the changed reward/checkpoint contracts, prove the behavior with focused CPU tests, and commit the accepted implementation on `feat/contact-reward-3x`.

## Workbench
- Trace `sim/manorl/rewards.py`, its diagnostics consumers, exact contract assertions, and current training protocol text.
- Implement one explicit direct-contact multiplier with default `3.0` while retaining a configurable `1.0` comparison path.
- Add focused 1x-versus-3x regression coverage and run scoped validation.
- Inspect and commit only the accepted feature files.

## Context
- Worktree: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-contact-reward-3x`
- Branch: `feat/contact-reward-3x`
- Parent integration branch at assignment: `dev` at `bb8e935`.
- This is an assigned feature worker worktree; do not merge into `dev`.

## Task specifications
- The default direct hand-object contact reward is exactly 3x the prior contribution.
- `max_contact_reward` remains `0.4` and continues to define the unscaled contact-quality basis and distance gate.
- The distance reward gate and `distance_x`, `distance_y`, and `distance_z` remain exactly unchanged; they are not multiplied by 3.
- `diagnostics.raw_contact` is the unscaled pre-window contact-quality value.
- `diagnostics.contact` is the actual scaled, windowed direct contribution summed into total reward.
- `REWARD_CONTRACT_ID` and `PPO_REWARD_CONTRACT_ID` are versioned coherently because checkpoint reward semantics changed.
- Focused tests compare multiplier `1.0` and default `3.0`, proving direct contact and the corresponding total delta change while gate and distance components are identical.
- Update exact contract assertions and current protocol documentation only where the changed semantics apply.
- Acceptance requires focused CPU tests, relevant `py_compile`, and `git diff --check` to pass.

## Constraints
- Do not change PPO hyperparameters or `PPO_REWARD_SCALE`.
- Do not touch W&B schema or device-environment worktrees.
- Do not operate on remote Server2.
- Do not modify unrelated files.
- Preserve unrelated user changes and generated outputs.
- Commit the accepted implementation on `feat/contact-reward-3x`, then stop for coordinator review.
