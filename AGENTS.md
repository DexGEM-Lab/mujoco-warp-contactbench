# Pi Startup

Before substantive work, every Pi session reads
`.memory/project/pi-orchestration.md`.

- In the primary `dev` worktree, act only as the coordinator for both product
  lines: launch, assign, review, and integrate tasks. Do not make feature edits
  or direct commits.
- `dev` is the protected ManoRL integration line; `dexhand` is the protected
  DexHandRL integration line. Neither accepts direct commits.
- In an assigned linked `feat/*`, `case/*/*`, `dexfeat/*`, or `dexcase/*/*`
  worktree, act as that task's worker. Read its task memory before work, and stay
  within its assigned files, outputs, product line, and parent-task boundary.
- Create tasks only from the primary worktree with `scripts/start_pi_task.sh`;
  use `feat`/`case` for ManoRL and `dexfeat`/`dexcase` for DexHandRL. Do not
  create worktree branches directly.
- Follow GitGuard without bypasses. Ownership, branch topology, lifecycle, and
  cleanup rules are in project memory and `.git-guard/contribution.md`.

## Proportional Task Handling

- Classify the request before creating a task. T0 read-only Q&A, status,
  inspection, or diagnosis stays in the coordinator session: do not create a
  task, worktree, or subagent.
- For T1 localized, reversible changes, preserve worktree isolation and use at
  most one bounded executor. Do not add a critic or run the full test suite by
  default; use focused validation instead.
- T2/T3 cross-module, persistent, migration, or high-risk changes retain the
  full coordinator, worker, review, and integration workflow.
- After one executor budget failure, preserve the partial worktree and report
  the failure. Do not resume automatically; resume only for a concrete missing
  step or an explicit user decision.

See `.memory/project/pi-orchestration.md`, `scripts/start_pi_task.sh`, and the
`Pi Task Worktrees` section of `README.md` for the operating details.
