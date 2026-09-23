## Objective
Build and validate one package-driven ManoRL training path that physically repairs two real-world raw capture corpora with a fixed Cheyingtong right MANO hand at 120 Hz control / 480 Hz physics. Completion requires source-complete accounting, duration-preserving 120 Hz references, small learned residuals, and physical grasp/transport/place/release evaluation.

## Workbench
1. Implement the raw-transfer source/package contract and focused tests on `feat/raw-capture-transfer-120hz`.
2. Compile verified per-source packages and a combined catalog.
3. Run zero-residual baselines before any PPO smoke test.
4. Choose long-training scale only from measured baseline failure mechanisms.

## Context
- Runtime source snapshot: `/home/jay/dexrobot/FromSSH/manoRL_mujoco` (deployed snapshot, no usable top-level Git history).
- Managed protected integration: `/home/jay/dexrobot/FromSSH/manoRL_mujoco_dev`, `dev` at `e23d7f3`.
- Deployed source claims lost Git object `edcd2a3cda16f15c9901af3c0231c6c72ba99b40`.
- Remake source: `/mnt/nas-222-projects/mocap_v2/lance_datasets/human_p1_remake_v3/human_p1_remake_clean.lance`, v978.
- Guangguan source: `/mnt/nas-222-projects/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance`, v530.
- Fixed hand manifest: `/home/jay/dexrobot/FromSSH/manoRL_mujoco/outputs/cheyingtong120_start_augmentation_20260917/seeds/asset_manifest.json`.

## Task specifications
- Use the complete canonical clean datasets only; do not mix daily, dirty, anomaly, generated, IsaacGym-refined, or synthetic rows.
- Select right-hand motion. Explicitly reject rows without right-hand data.
- Source operator and source MANO betas remain provenance. The physical model is always the pinned Cheyingtong right-hand asset.
- Raw `urdf_dof` is a kinematic actuator-target reference. Raw capture has no separate `q_state_ref`; do not invent one.
- Preserve complete capture approach, manipulation, release, and withdrawal. Movement annotations define reward/evaluation phases but do not crop the source.
- Remake is a physical 100 Hz source. Resample by elapsed timestamp to a 120 Hz grid with duration/final-pose preservation.
- Guangguan is nominally/mean 120 Hz despite millisecond-quantized median intervals near 8 ms. Normalize by elapsed timestamps to the same 120 Hz grid.
- Hand XYZ/object XYZ: linear interpolation. Hand angular coordinates: unwrap then linear interpolation. Object orientation: quaternion SLERP.
- Initialize every scene object once from frame 0 plus one common support shift; all objects then evolve as free MJX-Warp bodies. Never write reference object poses after reset.
- Execute absolute target `q_raw_120 + residual`; no reference action is injected beyond the base position target.
- Separate action-specific expected-contact reward from the residual-active finger mask. The raw repair profile may adjust all 22 finger joints, with explicit residual-size regularization.
- Preserve the compound `bottle,cap` action18 scene and account for its 55 rows through an explicit `bottle:18` target override; do not split labels heuristically or silently drop the cap.
- Training consumes Lance-free verified package data. Lance/PyArrow stay in isolated compile workers.
- Combined training must preserve both source paths/versions and each row UUID; it may not pretend two sources are one Lance dataset.

## Constraints
- Do not use the 12,200-row IsaacGym refined Lance for this task.
- Do not use `q_state_ref` semantics for raw capture.
- Do not alter either source Lance.
- Do not start a long PPO run before zero-residual physical baselines identify the dominant failure modes.
- Do not commit feature changes directly to protected `dev`.
- Do not touch unrelated GPU processes.
