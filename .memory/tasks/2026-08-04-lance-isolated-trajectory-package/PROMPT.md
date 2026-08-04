## Objective
Implement and validate a Lance-isolated ManoRL trajectory-package pipeline so long-lived MuJoCo/MJX/PPO trainers never import or open Lance/PyArrow. Completion means a content-addressed package compiled from pinned v295/all-75/right/f120/pre180/post250 data is consumed by an N8192 one-update trainer on server2 with finite metrics and a native checkpoint.

## Workbench
1. Define and test `mtp.v1` package schema and isolated compiler workers.
2. Split trajectory catalog loading from deterministic environment assignment.
3. Add `--trajectory-package` to training and bind package digest to checkpoint ABI.
4. Compile and equivalence-check the canonical v295 f120 package on a healthy host.
5. Sync/stage package on server2 and run N8192 one-update acceptance.

## Context
- Primary coordinator worktree: `/home/jay/dexrobot/FromSSH/manoRL_mujoco`.
- Assigned worker worktree: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-lance-isolated-trajectory-package` on `feat/lance-isolated-trajectory-package`.
- Dataset: `/mnt/nas-222-projects/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance`, pinned version 295.
- server2: `ubuntu@192.168.10.22`; GPU0 reserved for ManoRL; GPU1 belongs to mocap2dexhand.
- Current integrated source: `dev@9360be8d650bcdf478d740a6c0a3698ee71e8b95`.
- Direct-Lance N4096 trained through update 108; direct-Lance N8192 crashed before GPU with random-address `libpython` SIGSEGV/exit139.

## Task specifications
- Package schema `manorl.trajectory_package.v1` is a directory containing canonical `manifest.json`, `READY`, `offsets.npy`, and concatenated allow-pickle-free NPY arrays for source indices, timestamps, q_ref, raw/shifted object position, and object quaternion.
- The package is a complete trajectory catalog independent of `num_envs`; N4096/N8192 assignments reuse the same catalog.
- Manifest binds dataset logical path/version/schema/discovery digest, ordered row identities, all transformation/clock/padding/hand-side contract IDs, per-pair counts, array dtype/shape/hash, aggregate catalog digest, and package digest.
- Compiler coordinator must not import Lance/PyArrow. Lance discovery/decode occurs only in short-lived subprocess workers with explicit outputs. Exit139 may be retried at most three times; semantic validation failures are fatal and rows may not be silently omitted.
- Final package publication is atomic. Failed temporary/shard artifacts remain inspectable; no READY marker appears before all hashes and semantic validation pass.
- Trainer production path accepts `--trajectory-package`, loads via JSON+NumPy only, validates hashes before GPU initialization, and never silently falls back to direct Lance.
- Refactor assignment so selector, pair assignment cycle, and num_envs operate deterministically over a loaded catalog.
- Strict checkpoint resume binds package schema/manifest/catalog digest; warm-start package changes remain explicit lineage rather than strict continuation.
- Preserve public 100/120 coupled clocks, fixed four physics substeps, pre180/post250, early30/rollout48/reward/action semantics, all 75 pairs, and current direct-Lance behavior for non-production compatibility unless an explicit package argument is supplied.
- Compile canonical package on a healthy host and compare every serialized trajectory identity/array against direct decode. Package loader must use `allow_pickle=False`; production package loading must not leave `lance` or `pyarrow` imported.
- N8192 one-update acceptance requires finite rollout/PPO metrics, expected 120/480x4 pre180/post250 sidecar, populated optimizer state, no direct Lance import, and no GPU OOM/Xid.

## Constraints
- Do not edit or commit in primary `dev`.
- Do not touch unrelated untracked files or generated outputs.
- Do not use subagents; user did not authorize delegation.
- Keep one logical change per commit and stage explicit files only.
- Do not silently retry semantic errors or silently drop trajectories.
- Do not silently fall back from package loading to Lance.
- Do not delete generated Lance datasets, failed compiler evidence, or existing training runs.
- Do not use GPU1 or signal mocap2dexhand processes.
