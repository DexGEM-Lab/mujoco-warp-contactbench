## Objective
Align the two canonical real-world raw-capture sources to one future ManoRL input contract: every row uses the same Cheyingtong right MANO physical hand, complete trajectories are converted to a duration-preserving 120 Hz reference, and verified Lance-free packages can be consumed together by one runtime. Delivery is input-path code, tests, and runbook; full-corpus compilation and policy training are outside the current request.

## Workbench
1. Finish focused validation and commit the raw-transfer input feature.
2. Preserve real-sample package/scene evidence and exact source accounting.
3. Report the future compile commands and known source rejections.

## Context
- Worker: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-raw-capture-transfer-120hz`, branch `feat/raw-capture-transfer-120hz`.
- Managed integration: `/home/jay/dexrobot/FromSSH/manoRL_mujoco_dev`, `dev` at `e23d7f3`.
- Remake source: `/mnt/nas-222-projects/mocap_v2/lance_datasets/human_p1_remake_v3/human_p1_remake_clean.lance`, v978.
- Guangguan source: `/mnt/nas-222-projects/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance`, v530.
- Verified Cheyingtong/f98 manifest: `outputs/raw_capture_transfer/cheyingtong_asset_manifest.json` in the worker output; SHA256 `d9fa818613cfc899041d4d195d14a8d5cfd8d45af8a00234b55e75c0b59aeb42`.

## Task specifications
- Use canonical clean datasets only; do not mix daily, dirty, anomaly, generated, synthetic, or IsaacGym-refined rows.
- Select raw right-hand q. Rows without a right hand are explicit source rejections.
- `index.operator` and source MANO betas remain provenance only. They never select or modify the physical hand. The physical hand is always the pinned Cheyingtong right MANO asset.
- Raw `urdf_dof` is the one kinematic actuator-target reference; do not invent `q_state_ref`.
- Preserve complete approach, manipulation, release, and withdrawal. Movement annotations define phases but do not crop the capture.
- Resample elapsed timestamps to a coupled120Hz grid: linear hand/object XYZ, unwrapped-linear hand angles, quaternion SLERP for objects, final-pose preservation with at most one grid-edge hold.
- Initialize every scene body once from frame0 after one common support shift. All objects evolve as free MJX-Warp bodies thereafter.
- Preserve compound bottle+cap scenes. Use explicit `bottle:18` target selection while retaining cap as a free colliding body.
- Keep Lance/PyArrow inside isolated compiler workers. Packages bind source path/version/schema/catalog, source operator/betas, fixed physical manifest, clock, and interpolation contract.
- Permit several independently verified source packages to feed one policy runtime without pretending they came from one Lance dataset.

## Constraints
- Do not run additional PPO training or frozen policy evaluation for this task.
- Do not compile the full15,704-row corpus unless the user asks.
- Do not alter either source Lance.
- Do not use IsaacGym refined state-target semantics.
- Do not commit directly to protected `dev`.
- Do not touch unrelated GPU processes.
