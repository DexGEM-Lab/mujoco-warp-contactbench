## Objective
Archive pre-DexStream ManoRL physical assets with an auditable, reversible compatibility boundary, then train the complete daily Lance catalog using the pinned DexStream assets, a Lance-free trajectory package, 100 Hz, pre-padding 180, and post-padding 180. Done means every valid trajectory is represented in the assignment, both Server1 and Server2 pass the same bounded training smoke, and the production run has durable checkpoints and W&B telemetry.

## Workbench
1. Preserve and verify the legacy-asset archive manifest and compatibility symlinks.
2. Preserve the compiled daily-v12 MTP and its source-equivalence evidence.
3. Monitor the Server1 production run through scheduled checkpoints and completion/failure.
4. Interpret training metrics by object/action rather than treating process survival as success.

## Context
Repository: `/home/jay/dexrobot/FromSSH/manoRL_mujoco`.
Source Lance: `/mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_remake_v3/human_p1_daily_20260902_clean.lance`, version 12.
Local MTP root: `/home/jay/data/manorl_trajectory_packages`.
Server1 deployment: `/home/jay/dexrobot/FromSSH/manoRL_mujoco_dexstream`.
Server2 deployment: `/home/ubuntu/dexrobot/FromSSH/manoRL_mujoco_dexstream`.
DexStream commit: `f98da997f316c8a6b4bc2931cabed19e831ef163`.

## Task specifications
The complete source catalog contains 696 right-hand trajectories, five objects, and fourteen pairs. Compile all rows with `reference_fps=control_fps=100`, `pre_padding=180`, and `post_padding=180`; deterministic invalid rows must be recorded rather than silently dropped.

Use a verified `manorl.trajectory_package.v1` package for every long-lived trainer. The trainer must not map Lance or PyArrow. Package digest, catalog digest, DexStream repository/commit, asset-manifest SHA, and padding/clock settings must appear in checkpoint sidecars.

Balanced pair assignment needs 99 slots per pair to include the longest 99-trajectory pair, so production uses `14 * 99 = 1386` environments. A 696-environment balanced assignment is insufficient because it covers only 646 unique trajectories.

Run fresh training; do not strict-resume a checkpoint from the old physical assets. Require a five-update smoke on both Server1 and Server2 before production. Production uses Server1 GPU2, 5000 updates, checkpoint interval 25, W&B online, and a 72-hour timeout.

## Constraints
Local simulation tests use exactly one environment.
Do not run a long-lived trainer directly from Lance/PyArrow.
Do not kill or reconfigure other users' processes.
Keep legacy compatibility paths valid while external jobs may reset.
Do not delete source Lance, MTP, checkpoints, W&B data, or archive manifests.
