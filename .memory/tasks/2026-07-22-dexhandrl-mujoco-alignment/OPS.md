# Operations Record

- 2026-07-22T00:00:00Z — Task initialized in `feat/dexhandrl-mujoco-alignment` from current `dev` (`ea920d0`). Target migration commits are `f3cb5fc`, `934e48f`, `c06ad8b`, rooted at `8ab80fc`. Initial worktree was clean.
- 2026-07-22T00:00:00Z — Predicted conflict surface from a three-way precheck: `README.md` and possibly workflow/dependency files; resolution invariant is retain current MANO material and add DexHandRL material, never replace MANO with the historical branch snapshot.
- 2026-07-22T00:05:00Z — Cherry-picked the three migration commits in required order onto `ea920d0`, creating `d908596`, `a688f84`, and then the current third commit. Semantic resolutions retained current MANO dependency pins/workflow and added the named DexHandRL extra, while retaining both MANO and DexHandRL README documentation. The historical lock resolution was replaced with current-dev's compatible lock and `uv lock --offline` was run before staging.
