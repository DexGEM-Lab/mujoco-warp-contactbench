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

## 2026-08-12T17:15:37+08:00 — latest decomposed assets, corrected baseline interpretation, and N8 evidence

- Pinned runtime assets in the feature worktree to `all_assets@7228b5cfce8d9a072ed4bded7a489cf73d521b68` and nested `Assets/sim/mano_assets@cde03ef94816b589f574ca6f358695005d3d1a3f`.
- Migrated DexHand hand/object defaults to the repository asset root while retaining the separate task YAML at `assets/isaac_source_root/dexhand_env/cfg/task/Dexhand021proReconstruction.yaml`.
- Object source of truth is now `Assets/sim/decomposed/<object>/<object>.urdf`: its source visual mesh supplies rendering/point-cloud geometry and every collision mesh supplies a distinct physical geom. Native compilation observed cube1 as 1 collision + 1 non-colliding visual and bowl as 28 collisions + 1 visual.
- Focused asset/reward/runtime/evaluator contract tests passed: 54 passed, 1 skipped for the asset/parser set; then 75 passed for reward/runtime/viewer/evaluator; phase-window naming and threshold record changes passed 42 focused tests.
- Prediction before the latest-asset GPU N8: if the old object collision path was the primary cause of absent grasp/lift, the authoritative CoACD geometry should materially increase timely multi-finger contact and lift. If not, the stable failure remains the residual policy's task gap or a nominal retarget/control limitation.
- Full latest-asset residual-off N8 artifact: `outputs/dexhandrl_sim_rl_alignment/n8_reference_decomposed_full/reference_summary.json` plus `reference_trace.npz`.
  - All seven primary worlds completed full horizons with finite/valid device transitions and no measured fingertip penetration.
  - Maximum utilization: contact 5.33%, constraints 16.80%, CCD 43.75%; no capacity mechanism explains failure.
  - All seven worlds crossed the counterfactual 0.10 m deviation threshold; mean position error was about 0.0854 m and mean maximum error about 0.2557 m.
  - Maximum actual lift across primary worlds stayed below 8 mm while target lift reached about 0.32 m. The authoritative CoACD migration therefore did not make nominal reference control grasp/lift reliably.
  - Long-horizon world 0/7 duplicate divergence persisted: object position difference up to 0.109 m and contact-force difference up to 351 N despite unsaturated capacity.
- Contact-threshold evidence over active primary horizons: 406 positive strict hand↔object force slots; minimum positive 0.565 N; every positive slot exceeded 0.2 N; 88.18% exceeded 5 N. Selected 0.2 N as the calibrated latest-asset contract because it preserves all observed physical contacts while 5 N rejects about 12%.
- User corrected the acceptance premise: reference-only grasp failure is expected because residual RL exists to learn the correction. Revised commitment: reference-only must establish a finite, full-horizon, asset/control-valid, capacity-safe measured baseline; task failure itself does not block PPO.
- Fixed-seed deterministic untrained mean-policy artifact: `outputs/dexhandrl_sim_rl_alignment/n8_untrained_decomposed_full/untrained_summary.json` plus `untrained_trace.npz`.
  - All primary worlds finite/full-horizon; capacity maxima 8.46% contact, 16.41% constraints, 43.75% CCD.
  - Raw return mean 350.83 versus 271.84 reference-only; mean position error 0.0817 m; all seven still crossed 0.10 m.
  - This is the direct short-PPO baseline; its higher return is initialization bias, not learning evidence.

Cross-reference: `EPISTEMIC.md`.
