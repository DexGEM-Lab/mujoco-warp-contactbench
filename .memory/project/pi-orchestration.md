# Pi Task Orchestration

## Operating Model

The default development interface is one coordinator Pi in the primary `dev`
worktree. The user normally talks only to this coordinator. The coordinator
decomposes work and assigns bounded tasks to worker Pi instances; workers do not
share the primary working directory.

```text
user
  -> coordinator Pi (primary dev: orchestration, review, integration)
       -> feat worktree -> executor or interactive Pi
       -> case worktree -> debugging or experiment Pi
       -> critic Pi -> independent read-only review
```

The branch, linked worktree, Pi display/intercom name, output directory, and file
ownership boundary together identify one task. A Pi name alone does not create a
parent-child relationship; delegation and intercom establish coordination.

## Coordinator Responsibilities

- Translate the user's objective into features or bounded cases.
- Define success criteria, owned files, output paths, parent feature where
  applicable, and integration order.
- Create task branches/worktrees from `dev` with `scripts/start_pi_task.sh`.
- Use async subagents for bounded implementation, validation, review, and
  operations; use an interactive named Pi for GUI, GPU, tmux, or evolving work.
- Monitor workers through subagent status or intercom without duplicating their
  investigation.
- Review commits and evidence, require the feature to sync current `dev`, and
  integrate accepted work into `dev`.
- Keep the primary `dev` worktree free of feature edits and direct commits.

## Worker Responsibilities

- Work only in the assigned linked worktree and branch.
- Respect the assigned file and output ownership boundary.
- Complete only the assigned task; do not silently expand scope or start another
  task.
- Preserve unrelated user changes and outputs.
- Run scoped validation, commit coherent results on the task branch, and report
  the commit, evidence, and residual risks to the coordinator.
- Do not merge into `dev` or delegate additional workers unless the coordinator
  explicitly assigned an orchestrator role.

## Task Lifecycle

1. The user gives the coordinator an objective.
2. The coordinator chooses `feat/<topic>` or `case/<context>/<topic>` and defines
   ownership and acceptance criteria.
3. Create the task without launching an interactive Pi when using a subagent:
   `scripts/start_pi_task.sh feat <topic> --no-launch`.
4. Run the worker with that worktree as its cwd, or omit `--no-launch` to start
   the matching interactive Pi session.
5. The worker commits and reports; an independent critic reviews when warranted.
6. A `case/*/*` task merges into its owning `feat/*`; a `feat/*` task first
   merges current `dev`, then is the only task type the coordinator integrates
   into `dev`. The coordinator verifies the combined state.
7. Start subsequent work as a new task; never reuse the primary shared cwd for
   concurrent implementation.

GitGuard enforces the local branch topology and catches accidental workflow
violations. Native Git worktrees provide file isolation. Local hooks are a
governance boundary, not protection against a user intentionally bypassing them.

Canonical implementation details are in `.git-guard/contribution.md`,
`scripts/start_pi_task.sh`, and the `Pi Task Worktrees` section of `README.md`.
