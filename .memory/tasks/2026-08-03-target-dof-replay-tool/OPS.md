# Operations

Append-only evidence trail for the target-DOF replay tool task.

- 2026-08-03: Task initialized in linked worktree `feat/target-dof-replay-tool` from `dev` at `eebcb74`. No implementation changes yet.
- 2026-08-03: Prior diagnostic evidence: Server1 row 2239 target replay passed over 520 frames; recorded maximum lift `0.211849879 m`, replay maximum lift approximately `0.211850148 m`, max object position error approximately `3.84e-3 m`; Server2 40-frame headless prefix passed and GUI window was confirmed on X display `:1`.
- 2026-08-03: Formal package loader/CLI implementation added in the feature worktree. Seven focused tests passed; `py_compile`, CLI help, and Black checks passed.
- 2026-08-03: Formal CLI CPU override smoke passed 8 transitions after temporarily linking the worktree to the materialized authoritative assets. Report recorded `CPU replay omitted package Warp CCD allocation`; max object-position error was `8.33e-9 m`.
- 2026-08-03: Formal CLI GPU smoke passed the full 519 transitions on the local RTX 4060 Ti with package CCD settings 16/16. Report: max object-position error `3.8365e-3 m`, max rotation error `0.0214028 rad`, replay max lift `0.211850207 m`, status `pass`.
- 2026-08-03: Initial CPU smoke without temporary asset links failed before environment creation because linked worktree submodules are empty; this is an environment materialization issue, not a replay failure. Asset links were restored immediately after validation.
