## Objective
Implement a production synthesis acceptance gate so a candidate Lance row is saved only when all three user-defined quality predicates pass.

## Workbench
1. Implement one reusable acceptance evaluator.
2. Apply it before generic exporter row construction and preserve diagnostics/failure causes.
3. Add focused boundary tests and run a no-prefix/no-retreat smoke.
4. Commit the feature; do not merge into `dev` until Server1 mirror validation succeeds.

## Context
- Worker worktree: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-synthesis-acceptance-gate`.
- Server1 test mirror: `/mnt/user-home/jay/dexrobot/FromSSH/manoRL_mujoco_feat_all_banana`.
- Existing historical Lance datasets are immutable.
- Generic exporter: `tools/export_manorl_synthetic_lance.py`.

## Task specifications
A candidate must satisfy all three predicates:
1. It terminates by reaching the final reference state (`termination_reason_code == 1`), rather than an early deviation failure.
2. At the final state, convert simulated and reference object quaternions independently to intrinsic `XYZ` Euler degrees. For each coordinate, take the absolute shortest wrapped angular difference in `[-180, 180]`; the mean of the three absolute differences must be <=35 degrees. A value strictly greater than 35 fails.
3. Count distinct state/control frames whose materialized solved right-hand-to-target-object contact contains at least one normal-force vector with Euclidean magnitude strictly >0.2 N. The count must be strictly >100, i.e. at least 101 frames.

The predicates form an atomic AND gate for synthesis with both prefix and retreat disabled. Rejected candidates are not written to Lance. Record distinct failure reasons and per-attempt diagnostics in manifests. Do not change the separately versioned augmentation acceptance contract. Reward is not an acceptance criterion.

## Constraints
- No subagents.
- Do not edit or commit from protected `dev`.
- Preserve primary-worktree user changes and all existing `.lance` outputs.
- Do not silently weaken `>`/`<=` boundary semantics.
- Use focused tests before any GPU smoke.
