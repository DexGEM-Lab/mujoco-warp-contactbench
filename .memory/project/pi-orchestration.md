# Pi Task Orchestration

## Operating Model

One coordinator Pi in the primary `dev` worktree owns orchestration for two
protected product lines:

- `dev` — ManoRL integration.
- `dexhand` — DexHandRL integration.

The user normally talks only to this coordinator. Workers use isolated linked
worktrees and never share the primary working directory.

```text
user
  -> coordinator Pi (primary dev: orchestration, review, integration)
       -> feat/* or case/*/* worktree       -> ManoRL task
       -> dexfeat/* or dexcase/*/* worktree -> DexHandRL task
       -> critic                            -> optional authorized review
```

The branch, product line, linked worktree, Pi display/intercom name, output
directory, and ownership boundary together identify one task. A Pi name alone
does not create a parent-child relationship. Subagents remain opt-in and require
explicit user authorization for the current task.

## Proportional Task Classification

- **T0**: Read-only Q&A, status, inspection, or diagnosis stays in the
  coordinator session without a task, worktree, or subagent.
- **T1**: A localized reversible change uses one isolated task worktree and
  focused validation. Use at most one bounded executor when delegation is
  explicitly authorized.
- **T2/T3**: Cross-module contracts, persistence, migration, architecture, or
  other high-risk work uses explicit task ownership, focused review where the
  contract can fail, and broad final validation when warranted.

Classification changes coordination depth, not GitGuard, worktree isolation,
ownership, or branch lifecycle requirements.

## Product Branch Mapping

| Product | Protected integration | Feature family | Case family | Pi prefix |
|---|---|---|---|---|
| ManoRL | `dev` | `feat/*` | `case/*/*` | `manorl-` |
| DexHandRL | `dexhand` | `dexfeat/*` | `dexcase/*/*` | `dexhand-` |

`main` remains the release branch and accepts only `dev` according to the
current release policy. Neither `dev` nor `dexhand` accepts direct commits.

## Coordinator Responsibilities

- Translate the objective into the correct product line and bounded task.
- Define success criteria, owned files, output paths, parent feature where
  applicable, and integration order.
- Create every task from the primary worktree with `scripts/start_pi_task.sh`.
- Use `feat`/`case` for ManoRL and `dexfeat`/`dexcase` for DexHandRL.
- Delegate only when the user has explicitly authorized subagents for the task.
- Review commits and evidence, require the feature to sync its current protected
  integration branch, and integrate accepted work into that same product line.
- Keep the primary `dev` worktree free of feature edits and direct commits.

After one executor budget failure, preserve the partial worktree and report the
exact failure. Resume only for a concrete correction or explicit user decision.

## Worker Responsibilities

- Work only in the assigned linked worktree and branch.
- Stay within the assigned product line, files, outputs, and parent-task boundary.
- Preserve unrelated user changes and generated artifacts.
- Run scoped validation, commit coherent results, and report evidence and
  residual risks.
- Do not merge into a protected integration branch or delegate additional
  workers unless explicitly assigned that responsibility.

## Task Lifecycle

1. The coordinator identifies ManoRL or DexHandRL ownership.
2. Choose the matching feature or case family.
3. Create the task from the primary worktree, for example:
   - `scripts/start_pi_task.sh feat <topic> --no-launch`
   - `scripts/start_pi_task.sh dexfeat <topic> --no-launch`
4. Run the worker in the resulting linked worktree.
5. The worker commits and reports; independent review is added only when
   warranted and authorized.
6. ManoRL delivery is `case/*/* -> feat/* -> dev`; DexHandRL delivery is
   `dexcase/*/* -> dexfeat/* -> dexhand`. Before delivery, the feature merges
   its current protected integration branch.
7. After integration, verify cleanliness, preserve any required ignored outputs
   or audit artifacts, detach or remove the completed worktree as appropriate,
   and delete the completed task branch. Do not retain a delivered task ref at
   the integration-branch SHA.
8. Start new work as a new task; never reuse the primary shared cwd for feature
   implementation.

GitGuard enforces the local topology. Canonical implementation details are in
`.git-guard/contribution.md`, `scripts/start_pi_task.sh`, and the `Pi Task
Worktrees` section of `README.md`.
