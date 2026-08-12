# OPS — append-only evidence trail

## 2026-08-12T13:56:14+08:00 — task initialization

- User fixed the scope: data already exists; make the DexHand simulation and residual RL work. The intended causal chain is MANO/mocap task data from the ManoRL domain → DexHand021Pro retarget reference → physical MuJoCo/MJX trajectory via residual RL.
- Created `dexfeat/simulation-rl-alignment` from protected `dexhand@72b96f5` with `scripts/start_pi_task.sh dexfeat simulation-rl-alignment --no-launch`.
- Worktree: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/dexfeat-simulation-rl-alignment`.
- Data is read-only and out of implementation scope:
  `/mnt/nas-222-project/mocap/for_retargeting/dexhand021pro/results_all_mano/all/lance_new_all_generated_mano/lance_result_archive_full/dexhand021pro_lance_new_all_generated_mano_retarget_result_archive.lance`.
- Direct dataset observation with Lance 0.24.1: version 30, 7,539 rows, columns `index`, `trajectory_metadata`, `hands`, `objects`; row 0 reports `retarget_schema=dexhand_retarget_minimal_v1`, `hand_model_type=dexhand021pro`, 22D `dexhand021pro_controls_world_v1`, retarget UUID and seed UUID. Dataset schema metadata is empty and no upstream dataset path/version column is present.
- Historical memory establishes that N1024 device PPO ran and a production run reached at least update 1452 before exit 130, then resumed from checkpoint 1400 through at least update 1404. Scalar reward participated in PPO; component telemetry was later repaired.
- Reward comparison observation:
  - shared structure already exists: object xyz tracking, contact fraction, rotation, post-contact stability, survival/alive, residual penalties, deviation termination;
  - dev current equations additionally define `(0.5,0.5,2.0)` tracking scales, late-contact penalty after 10 frames, strict 0.2 N threshold, raw environment reward plus PPO scale 0.5, and current action penalty scale 0;
  - DexHand current contract uses 5 N threshold, YAML weights, active base/joint residual penalties, separate fingertip penetration, no late-contact penalty, and raw PPO scale 1.
- Prediction before the first simulation experiment: if current residual-off reference is physically viable, all selected runs stay finite and object tracking/contact failures cluster by difficult identities rather than a common phase or shared mechanism. A common ejection, penetration, loss-of-contact, reset, or target-routing pattern across actions will identify a simulation/control defect that must be fixed before PPO.

Cross-reference: `EPISTEMIC.md`.
