# ManoRL Pi Task Flow

```mermaid
gitGraph TB:
    commit id:"init"

    checkout main
    branch dev order: 1
    checkout main
    commit id:"main after dev fork"
    branch "feature/manorl-mujoco-migration" order: 2

    checkout "feature/manorl-mujoco-migration"
    commit id:"migration work"

    checkout dev
    merge "feature/manorl-mujoco-migration" id:"migration to dev"

    checkout dev
    commit id:"dev branch history"
    branch "feat/*" order: 3
    checkout dev
    branch "case/*/*" order: 4

    checkout "case/*/*"
    commit id:"case work"

    checkout "feat/*"
    merge "case/*/*" id:"case/*/* to feat/*"
    commit id:"feature work"

    checkout dev
    commit id:"dev branch sync point"
    checkout "feat/*"
    merge dev id:"dev to feat/* sync"

    checkout dev
    merge "feat/*" id:"feat/* to dev"

    checkout main
    merge dev id:"dev to main"
```

## Rules

- `main` is the literal release branch. It accepts only merges from `dev` and never direct commits.
- `dev` is the literal protected integration branch. It branches from `main`, accepts only `feat/*` merges and the temporary `feature/manorl-mujoco-migration` merge, and never direct commits.
- `feat/*` branches from `dev`, may receive `case/*/*` merges, and may merge only into `dev` after merging the current `dev` into the feature branch.
- `case/*/*` means `case/<context>/<topic>`. It branches from `dev`, may merge only into `feat/*`, and never directly into `dev` or `main`.
- Direct commits are permitted only on `feat/*`, `case/*/*`, and the temporary `feature/manorl-mujoco-migration` branch.
- `feature/manorl-mujoco-migration` branches from `main` and may merge only into `dev`. Remove it from this policy after migration adoption.
- One task has one branch, one linked worktree, and one Pi session.
- Pi names are `manorl-feat-<topic>` for features and `manorl-case-<context>--<topic>` for cases; case worktree keys are `case-<context>--<topic>`. The double hyphen separates independently hyphenated components. A Pi name is a display/intercom name, not a Git ref.
- GitGuard is a local accidental-workflow guard, not a security boundary.
