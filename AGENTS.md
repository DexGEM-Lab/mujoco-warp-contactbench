# Pi Startup

Before substantive work, every Pi session reads
`.memory/project/pi-orchestration.md`.

- In the primary `dev` worktree, act only as the coordinator: launch, assign,
  review, and integrate tasks. Do not make feature edits or direct commits.
- In an assigned linked `feat/*` or `case/*/*` worktree, act as that task's
  worker. Read its task memory before work, and stay within its assigned files,
  outputs, and parent-task boundary.
- Create tasks only from the primary worktree with
  `scripts/start_pi_task.sh`; do not create worktree branches directly.
- Follow GitGuard without bypasses. Ownership, branch topology, lifecycle, and
  cleanup rules are in project memory and `.git-guard/contribution.md`.

See `.memory/project/pi-orchestration.md`, `scripts/start_pi_task.sh`, and the
`Pi Task Worktrees` section of `README.md` for the operating details.
