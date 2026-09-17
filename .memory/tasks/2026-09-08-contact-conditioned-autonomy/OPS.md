# Operational evidence

## 2026-09-08T08:00Z — Authorization and baseline
User authorized major ManoRL algorithm refactoring, an isolated new branch, server deployment, parallel multi-GPU settings, timely reports/reviews, then explicitly authorized subagents. No nested delegation was authorized.
Primary `dev` head is `c38f787`; `test.sh` is already modified and multiple historical scratch/task files are untracked. Those are not this task's changes. The coordinator made no feature edits or commits in primary.
Ran `scripts/start_pi_task.sh feat contact-conditioned-autonomy --no-launch` from primary. Created `feat/contact-conditioned-autonomy` at the protected `dev` base; feature worktree was clean. GitGuard reported automatic installed-runtime hook synchronization during creation; no bypass was used.
The configured `~/.pi/agent/PROMPT.template.md` path was absent. Located and read the actual template in the Pi configuration root, preserving its Objective/Workbench/Context/Task specifications/Constraints structure for this task.

## 2026-09-08T08:00Z — Direct evidence collected during design discussion
Inspected `trajectory.py`, `environment.py`, `device_runtime.py`, `abi.py`, `model.py`, `rewards.py`, `observations.py`, `assets.py`, `contracts.py`, the current task contact mapping, historical learning/synthesis records, and the daily-v12 package manifest/NumPy arrays.
A read-only comparison of each daily-v12 trajectory's `object_pos` and `object_pos_raw` found 696 trajectories; independent object-Z adjustment absolute counts >1/3/5/10 mm: 512/214/16/1. Signed min/median/max: -6.578292138 / -1.471677278 / 10.391250776 mm. Largest: `cylinder5_04_092`; per-trajectory variation of the shift was at floating-point noise level. Code retains original hand DOFs when applying this object shift. Confirmed geometric alteration; success-rate causality is untested.
The local daily-v12 package includes no cube2. The guangguan Lance directory exists; current pinned DexStream includes cube2 with declared 5 cm dimensions and mass 0.037 kg. Historical cube2 source/checkpoint counts are not current-source acceptance evidence.
Current action processing includes hard reference addition, early-phase reference-only behavior, inactive-finger masking, and non-accumulating wrist-angle residual scale 0.025 × 0.01 = 0.00025 rad/axis. Current contact reward thresholds aggregated expected-segment forces at 0.2 N and gates position reward during the object movement window.

## 2026-09-08T08:00Z — Initial commitments and predictions
Use the accurate demonstration as strong state/contact intent, not as a direct actuator command. Preserve the existing baseline while implementing a separate versioned autonomous route. Start with cube2; retain PD/MJX/PPO where useful.
Prediction: strong contact-region conditions plus near-contact physical learning should give more useful contact-support learning than replacing reference execution with randomly initialized unrestricted actions under the old reward. This remains an experimental claim.
Initial review boundary: autonomous command provenance, source alignment/contact semantics, physically simulated object motion, reference fidelity, and fair evaluation. Resource operator may inspect, but cannot deploy/start workloads in the initial inventory slice.
See EPISTEMIC.md for current model; future entries append here.

## 2026-09-08T08:24Z — Initial audit, failure preservation, unattended continuation
Initial authorized parallel run `32f7d690-ccab-4a0f-ace9-6f39b5a03103`: critic completed; resource operator aborted after 20 assistant turns (12+2 budget). No feature or server mutation was assigned/performed. Its raw output is preserved in the session's async artifacts. A race-time steer to stop discovery failed because the run had already stopped; no hidden retry occurred.
Critic confirmed the reference-action coupling, pre-grasp reward gap, missing reference-hand/velocity/contact-phase observations, and object-only alignment. Parent accepts separate autonomous control/observation/reward contracts, additive dense objectives, fixed demonstration timing initially and explicit hand+object alignment. Parent rejects modifying legacy contract constants globally, comparing equal wall time instead of equal transitions, or downgrading the objective to contact-only when the baseline is strong.
Resource raw evidence: Server1 GPU2 has 222 MiB total use and only a small context from another user's GPU3 job; GPUs0/1/3 host active foreign work and GPU1 also has a deliberate hold. Server1 Unison returned to zsh after Broken pipe/Lost connection; Server2 Unison was active. The first remote Git/Python inspection targeted a non-Git Unison mirror and an unset Python PATH. Exact corrective resume `c9b6adbf` is limited to the current DexStream deployment/explicit interpreter plus finalization, with no re-scan or workload launch. This is the only allowed resume after that budget failure.
User requested unattended continuation with delegation until the objective, a genuine user decision, or an irreversible high-risk action. Updated PROMPT accordingly. First runnable implementation remains an intermediate milestone; no new policy result exists yet.

## 2026-09-08T09:10Z — Autonomous milestone implementation
The standard executor previously failed transport before any tool/action; this alternate task-scoped executor then began work. The completed cube2:02 MTP package was already available and loaded directly: dataset version 295, 50 trajectories, 27,837 frames, 120/120/480 Hz, zero rejects, package digest `6ac29825d444069071f4803718bff45345fab56131b4d9bf4b6e7ef3d8178e90`, manifest SHA-256 `e826de23d4586611001230d340eaf0c68752b09988eed7928bf8884f056d8706`.

The feature worktree initially had an unmaterialized/stale DexStream submodule checkout; `git submodule update --init assets/dexstream_digital_assets` materialized the manifest-declared pin `f98da997f316c8a6b4bc2931cabed19e831ef163`. No source pin was changed. Real one-world MJX-Warp smoke then compiled cube2 and stepped one action successfully, producing a finite 359-D observation and 28-D command.

Implemented separate autonomous contracts, rate-bounded measured-state command map, geometry-derived surface intent, shared support alignment, dense additive reward, one-world MJX-Warp environment, train/evaluate PPO entrypoint, focused tests, and docs. Legacy residual modules were not edited. Execution-policy refinement requires operator/executor runs to use 20+ turns or scoped checkpoints; this run uses the existing 32+4 allowance.

## 2026-09-08T10:36:52Z — M2 correction evidence
The prior M2 diff and two full traces were preserved as negative evidence: zero hold had a 19.4 cm target lift but exactly zero producer hand-on-object force; contact_count=35 and per-hand-geom force reflected table contacts. Reference pursuit reached 0.6299 m wrist Z and displaced the cube about 20 cm XY, terminating at step 233. Mean prior reference proximity maxed at 0.0744, so its release threshold never activated.

Corrected v2 uses MuJoCo `mj_geomDistance` witnesses between every hand-segment collision geom and object collision geom, with object-local anchors and signed-distance confidence. It uses `MjxWarpPhysicalProducer.hand_object_force_on_object_world_N` for hand-object force and masks the object-local relative-motion slip proxy by those forces; table contact force is excluded. Physical rates are explicit (.5 m/s wrist XYZ, 2 rad/s wrist rotation, 4 rad/s fingers) with measured-state envelopes (.02 m, .25 rad, .35 rad). The pursuit diagnostic targets the bounded next command rather than integrating measured tracking error. Standalone PPO smoke was removed/refused pending canonical GAE/PPO.

Semantic tests pass (6). Post-correction one-world full-start diagnostics on identity cube2_02_2833 completed 538 steps for both zero and reference-pursuit; zero force stayed 0 with contact_count up to 35, pursuit hand-object force reached 2.07 N over 84 steps and slip proxy reached 0.281 m/s. Neither diagnostic is competence evidence; both report full source/contract/clock/config hashes. Package summary loads all 50 identities.

## 2026-09-08T10:44:00Z — Post-commit diagnostic evidence (e9b388d)
After commit e9b388d, the same identity cube2_02_2833 was run full-start for 538 control frames with both diagnostic actors. `m2-e9b388d-zero-hold.json`: return 2087.4894, failure_phase=complete, max contact_count 35, hand-object force exactly 0, object Z 0.02316–0.02467 m. `m2-e9b388d-reference-pursuit.json`: return 2130.2058, failure_phase=complete, max hand-object force 2.0664 N, force-positive 84 frames, slip proxy max 0.2814 m/s, object Z 0.02316–0.02601 m. Neither lifts the 19.4 cm target; these are diagnostic actor traces, not competence. Both traces include source package/catalog hashes, v2 contracts, 120/480 clock, config hash, actual/target quaternion, reference hand q, witness anchors, per-term rewards and failure phase. `m2-e9b388d-cube2-02-summary.json` loaded all 50 identities.

## 2026-09-08T19:30:00Z — M3 first formal learner
Near-term scope was narrowed to the shortest formal learner on one representative TRAIN identity; multi-reference rotation, held-out rollout and vectorization generality are later milestones. The deterministic split contract fixes inspected `cube2_02_2833`, `cube2_02_2835`, and `cube2_02_2837` in TRAIN, then seeded-permutes the remaining 47 into 37 TRAIN/5 validation/5 test. Seed 0 split digest is persisted by `identity_split`.

Added `sim/manorl/autonomy_training.py` with an honest N=1 Gymnasium boundary over `Cube2AutonomousMJX` and canonical `RlGamesPPO`/skrl `compute_gae`; no handwritten return loop. Added `formaltrain`/`evaluate` CLI. One local real MJX-Warp update on identity cube2_02_2833 completed with W&B offline run `0sto73qw`, reward mean 4.474207, 2 transitions, checkpoint `outputs/manorl/contact_conditioned_autonomy/m3-143e846-formalppo.pt`. Evaluation loaded the checkpoint and ran 4 deterministic learned-policy steps, return 17.900504, trace `m3-143e846-formalppo-eval.json`; first-step task_success=false and actual_peak_lift=0.0. Checkpoint stores model/optimizer/RNG/normalizer plus source/split/contract/config provenance.

Parent B1 decision: do not gate training on pursuit-vs-hold return or multiply contact terms; inspect physical progress and reward variance. Parent B2 decision: do not call max-lift+endpoint pose success; use sustained airborne genuine hand-object contact, path error and release as diagnostics. Existing M2 raw evidence remains unchanged: zero hold lift 0 cm; pursuit lift 0.135 cm against 19.38 cm target, path RMSE 6.93 cm, 59 net-force-positive frames, and control-q windup under .02 m.

## 2026-09-08T20:10:00Z — M3 launch-boundary correction
Parent inspection of installed skrl found two concrete blockers. The v2 adapter incorrectly advertised CPU while GPU training built a CUDA model, and NEXT_STEP vector autoreset could apply an action chosen from a terminal observation to the initial state. Corrected commit `63add62` changes the first learner to an ordinary N=1 Gymnasium environment with explicit reset-before-next-action and device-aware wrapper/model/memory placement. Finite task horizon, success, drop and divergence are all `terminated=True`; rollout cuts before task end remain nonterminal for canonical last-value bootstrap.

Focused boundary tests cover two terminal/reset transitions and horizon termination. GPU2 local N=1 formal train then completed one canonical RlGamesPPO update with 2 transitions, reward mean 4.476710, W&B offline run `v3u4zejj`, checkpoint `outputs/manorl/contact_conditioned_autonomy/m3-63add62-formalppo-gpu.pt`. GPU evaluation loaded that checkpoint and ran 4 deterministic learned steps, return 17.903394, trace `m3-63add62-formalppo-gpu-eval.json`; no competence claim. Evaluation now honors checkpoint setting, restores normalizer when present, and serializes raw trace metadata. No broad suite or server job was run.

## 2026-09-08T12:44:35Z — Telemetry/publication continuation request
Parent supplied the completed first learner artifacts at primary `outputs/manorl/contact_conditioned_autonomy/server1-first-86ef408/`: `learned-full-start.json` contains 538 frames, with `checkpoint-final.pt`, `first-rollout-metrics.json`, and `train.log`. Parent analysis reports 0 frames with hand-object force >0.02 N, 0 cm peak rise versus 19.379 cm target rise, path RMSE 0.068146 m, wrist-reference RMSE 0.1047 m, final phase `horizon_reached`, and `task_success=false`. This is preserved as honest no-grasp/no-downstream-success evidence; training PPO losses cannot be retroactively added.

The user requested reuse of mature PPO/logger names, transition-axis W&B history, physical metrics/video, and a bounded post-training publication command that appends only to existing FINISHED run `n5zkg0eg`. The separate GPU batching design remains read-only and out of scope. Implemented telemetry/publication source and focused tests are the scoped continuation; no new online training is launched.

## 2026-09-08T20:30:00Z — M3 continuity-loop correction
Parent review found formaltrain reset observations at every update even when the episode was nonterminal, preventing a 538-frame trajectory from continuing across 256-step rollout cuts. The loop now carries `observations` across updates, records the terminal next observation, and explicitly resets after every terminal transition including the final rollout timestep. A focused schedule regression (`horizon=5, rollouts=3, updates=3`) verifies phases `1,2,3,4,5,1,2,3,4`, exactly 9 transitions, and only initial plus post-step-5 reset.

The same patch excludes argparse callable `fn` from checkpoint config and includes actual/target object quaternions plus reference hand q in `env.step` info for learned rollout inspection. No physics was rerun; prior GPU N=1 evidence remains valid. Changes are committed in the continuity-fix commit.

## 2026-09-08T21:00:00Z — Corrective telemetry/publication resume
Parent audit found and corrected seven semantic issues in the telemetry/publication slice: update FPS uses the current window transition count; ongoing episode return/length persist across `clear()` and only completed episodes emit episode statistics; genuine airborne contact uses cached cube2 collision vertices rotated by actual quaternion and requires bottom clearance above `FLOOR_TOP_Z + 0.01 m` plus >0.02 N hand-object force; orientation metrics expose normalized quaternion geodesic radians/degrees; resumed W&B evaluation uses automatic SDK monotonic history steps while recording checkpoint transitions and logs `wandb.Video`/`wandb.Table`; 120 Hz traces are decimated four-to-one for 30 FPS labeled video; `full_task_success` is nullable rather than hardcoded false and reference-release-window contact-free frames are labeled as a statistic. No new training, batching, or physics was run. Real trace publication and middle-frame extraction passed.

## 2026-09-08T21:30:00Z — Batched implementation priority override
Parent override freezes reward/observation/BC/initiation redesign and prioritizes a runnable direct device port for throughput. The implementation must preserve existing v2 control map, reward weights/terms, 120 Hz policy / 480 Hz physics / four substeps, and compare geometry only when needed to remove host collision queries. 64/256 are short correctness/memory checks; operator owns 4096/8192 scale measurements. No local multi-row physics benchmark or remote workload was launched.

Added `sim/manorl/autonomy_batch.py`: homogeneous batched MJX-Warp runtime, JAX measured-state rate map, masked row reset with on-device forward, source-order offline fixed witness extraction, device witness transform, additive device reward term family, and a thin Torch adapter. Added dynamic actor input sizing, v3 checkpoint rejection metadata, focused kernel tests, and an operator benchmark CLI. One local N=1 synthetic cube2 runtime smoke completed with observation `(1,538)`, finite reward, valid reduction, and no host FK/geom-distance call in the step path. The real pinned package is unavailable in this linked worktree, so no package-identity training or scale benchmark was run.

## 2026-09-08T22:10:00Z — Batched correction resume
Parent identified five concrete blockers in cb426c6. Correction commit `44827f4` fixes global contact-only versus per-world constraint capacity (`naconmax` scales B; `njmax=CONSTRAINT_CAPACITY=512`), stores unique T-frame references/witnesses on device with active-index gathers, applies masked reset through a device conditional before the next actor action without unconditional full observe/forward, restores actual qvel/relative motion/reference FK relation/backward target velocity and M2 confidence/release/drop/divergence semantics, and adds benchmark modes physics/environment/sampling/both/all with aggregate boundary synchronization. Real pinned package N=1 GPU check passed: tables `(539,28)`/`(539,16,3)`, finite valid step, scalar command parity max error `1.39e-08`. Focused suite: 20 passed. No multi-row local physics or remote benchmark launched.

## 2026-09-08T23:00:00Z — Direct batch PPO boundary

User froze autonomy algorithm/geometry/reward/control/clock and prioritised integration of the measured 11dc250 direct batched runtime with existing RlGamesPPO. Implemented a direct JAX/Torch DLPack trainer: terminal next observations are stored for canonical GAE before `prepare_action()` resets completed rows; rewards and dones remain `(B,1)` tensors; update telemetry uses device reductions and compact scalar egress. `AutonomyActorCritic` explicitly dispatches Gaussian policy and deterministic value roles, correcting MRO dispatch that otherwise sent value calls through GaussianMixin. Local CUDA N=1 smoke against the primary pinned package completed 2 transitions and one PPO minibatch with persistent workspace enabled: reward mean 4.4840136, valid_rows=1, value loss 23.53766; v3 save/load evaluation ran two deterministic mean-action steps, return 8.96734. This is integration evidence, not a learning or throughput result. Server B=8192 rollout/reset/load checks remain parent-owned.

## 2026-09-09T00:00:00Z — Parent-owned full-horizon runtime soak

Independent parent execution on unchanged 11dc250 Server1 GPU2 passed the runtime capacity boundary: 8,192 environments × 1,100 measurement steps = 9,011,200 environment transitions; 16,384 row resets; all state/contact validity finite; 55,542.7 environment transitions/s; peak 8,424 MiB; exit 0. Authoritative artifacts are primary `outputs/manorl/contact_conditioned_autonomy/reset-soak-20260909/result.json`, `run.py`, and `log`. This clears full-horizon/reset capacity for the measured runtime. It is runtime evidence, not a PPO learning result; do not repeat the physics soak in this feature worker.

## 2026-09-09T00:20:00Z — PPO launch-boundary correction

Corrected the direct batch PPO bridge before server launch: runtime validity is a global device scalar and now reduces by `.bool().all()` without row reshape; update reward/rate denominators use `B*rollouts` rather than cumulative transitions; completed episode returns/lengths are reduced before only completed rows clear; current-transition device physics reductions are captured before `prepare_action`; CUDA synchronization brackets sampling and optimization timings; and update callback publishes rows as they occur. Checkpoints now carry explicit `policy_steps` and `environment_transitions` (legacy `global_policy_step` remains environment transitions), numbered periodic snapshots plus atomic latest/final, and provenance including source commit, asset pin, clock, package/split/witness. Frozen loader verifies supplied package/split/witness provenance before model load. Focused B=2/three-update real-loop test passed scalar global valid, metric invariance, callback ordering, periodic snapshots, and completion boundaries. Local GPU N=1 persistent-workspace PPO update passed 2 transitions with valid=1, then wrote latest and `.update000001` snapshots. No B=8192 soak was repeated; parent result remains authoritative.

## 2026-09-09T00:31:06Z — Archive deployment provenance correction
A server preflight reached `tools/train_manorl_autonomy.py` in an immutable source archive without `.git`; `_git_revision(_ROOT)` therefore failed before training despite the deployment process supplying `DEPLOYED_COMMIT=70ff46a...`. `_git_revision` now reads only a present `DEPLOYED_COMMIT` marker, requires one lowercase full 40-character SHA, and otherwise preserves the existing Git resolution failure. The same resolver continues to read the asset pin from its Git repository unless that root explicitly has a valid marker. Focused ManoRL Conda test module passed 7 tests, including valid non-Git marker, invalid-marker rejection, and missing-marker/non-Git failure. The default interpreter could not collect the module because `gymnasium` is absent; this was an environment mismatch, not a test failure.

## 2026-09-09T00:45:00Z — Phase-only v3.1 correction

Parent completed the 67,108,864-transition Server1 run and frozen full-start 538-step evaluation: zero contacts/lift, target distance 0.193793 m, path RMSE 0.068146 m, wrist RMSE 0.122904 m. The clipping/log-probability lead is ruled out: the unchanged deployed final policy under installed `GaussianMixin` had 78.8574% clipped rows and exact `exp(new_logprob-old_logprob)-1 == 0`; the mixin clips before evaluating the action log-probability. The confirmed bug is unique reference-table phase normalization using `reference_q.shape[1]-1` (28-1) rather than horizon T-1. Corrected only that denominator to the existing `horizon`, versioned the unchanged-width observation/checkpoint ABI as v3.1, and reject v3 phase-bug metadata unless an explicit legacy reader is used. Focused CPU tests passed (14): unique T=539 indices 0/134/269/538 yield 0/134/538/0.5/1 (with 134/538 exact), batched compatibility matches, and installed GaussianMixin saturated-action log-prob identity holds. Next experiment: from-scratch phase-fixed only, same 67M budget; do not alter PPO, BC, reward, controller, clock, geometry, or initialization.

## 2026-09-09T09:58:00Z — Bounded teacher-imitation initialization
Prediction: a diagnostic command-pursuit rollout passing through the unchanged measured-state rate map will give the existing MLP a physically valid approach/contact prior, reducing full-start neural wrist error without redefining task success. Local CPU N=1 used only pinned package `cube2_02_v295_f120_pre180_post180`, seed/split seed 0 and TRAIN identity `cube2_02_2833`. `pretrain` collected 538 pre-action observations/actions under real fused MJX-Warp physics, then fit the existing MLP mean for 2,000 Huber steps (batch 128, lr 1e-3). Artifacts: primary `outputs/manorl/contact_conditioned_autonomy/teacher-init-local-20260909/teacher_dataset.npz`, `imitation_init.pt`, and `fit_metrics.json`. Teacher: wrist RMSE 0.0524113 m, 59 force-positive (>0.02 N) frames, max hand-object force 2.06644 N. Fit teacher-action MSE: 0.104227 -> 0.000924422; MAE 0.0110136.

Frozen `batchevaluate` used neural mean actions for all 538 full-start steps: wrist RMSE 0.0979889 m, 242 force-positive frames, max hand-object force 8.41783 N. This is below the phase-fixed PPO 0.1798 m comparator but is not lift/success evidence. A real N=1 `batchtrain --init-checkpoint` smoke validated strict v3.1/package/split/witness loading and an explicitly fresh PPO optimizer: 2 transitions, valid=1, one optimizer minibatch, checkpoint `warmstart-smoke.pt`. Focused CPU tests: 19 passed (teacher clipping, TRAIN-only rejection, fit, loader/warmstart, existing batch contracts).

## 2026-09-09T11:10:00Z — Separate critic implementation boundary

Implemented optional `batchtrain --separate-critic` without changing PPO, reward, runtime, observation, action, learning rate, or loss weights. The policy remains `net`/`mean`/`log_std`; `role="value"` uses a registered independent 128×128 `value_net` plus existing `value` head only when selected. Shared remains the default.

New frozen and imitation checkpoints write exact actor-critic architecture metadata, and `batchevaluate` reads it before constructing the model. Strict frozen loading rejects shared/separate mismatches. The explicit `--init-checkpoint` weights-only path accepts old/shared BC state only when the sole missing target keys are `value_net.*`, then copies the loaded actor trunk into it; output lineage records that conversion. All other missing or unexpected keys fail.

Focused primary `.venv` CPU tests passed: 21 tests in `tests/manorl/test_autonomy_training.py` and `tests/manorl/test_autonomy_batch_training.py`; CLI help exposes `--separate-critic`; `py_compile` and `git diff --check` passed. The new direct tests show copied shared-BC policy/value outputs exactly match before optimization, a critic-only backward/Adam step makes exact-zero policy/actor changes, policy loss leaves `value_net` gradient-free, and separate PPO save/load preserves outputs and metadata. This establishes removal of the critic-to-actor gradient edge; it does not establish grasp success or eliminate the independently observed PPO actor-gradient drift.

## 2026-09-09T12:20:00Z — Raw Gaussian likelihood correction and finite-update guard

Parent direct probe of optimizer-resume checkpoint update 288 found policy means `[-9.77, 6.83]`, minimum standard deviation `0.31`, 83.8% clipped actions, and clipped actions up to 28.3 standard deviations from their Normal mean. This differs from the earlier falsified identity theory: with an unchanged policy, installed `GaussianMixin` returns exact old/new log-probability identity even when it clips sampled actions. The actual defect appears after a mean update: clipping turns a continuum of raw samples into boundary point mass, but PPO evaluates the Gaussian density at that boundary. At a deliberately saturated mean 4.01/std .01, an actual `GaussianMixin` clipped-boundary likelihood changes by >299 log units for a 0.01 mean change and `exp(log_ratio)` overflows; the same raw Normal sample has a finite ratio.

Implemented raw-normal PPO collection only in the batched builder: `AutonomyActorCritic` retains legacy `clip_actions=True` by default, `build_batched_runtime` selects false, `run_batched_ppo` records unmodified samples, and `BatchedAutonomyAdapter.step` clips a physical-execution copy. Fixed-RNG tests demonstrate clipped physical commands are bitwise identical to old sampling while memory equals raw actions. Frozen mean-action evaluation remains clipped. New checkpoints write `policy_sampling_contract`; legacy optimizer resumes restore unchanged model/Adam/RNG state and declare that future rollouts use the raw-likelihood correction.

After each PPO update, model parameters, recursive Adam state, and all exposed native PPO metrics are checked before callback/checkpoint writes. A non-finite update writes `<checkpoint>.nonfinite-updateNNNNNN.json`, raises, and leaves earlier numbered finite checkpoints intact. `latest_ppo_metrics` no longer drops non-finite values.

Focused primary `.venv` test suite: `22 passed` before the added legacy-resume regression, then local real CPU N=1 batch PPO: 1 update x 2 rollouts, 2 transitions, reward mean 4.4815607, valid=1, approximate KL=0, finite model state, optimizer state entries=9. Command required `--no-persistentworkspace` because the CLI default persistent CCD workspace is GPU-only. Output checkpoint was `/tmp/manorl-raw-contract-n1.pt` and includes raw-normal policy-sampling metadata.

## 2026-09-10T00:00:00Z — v4 Warp-only cube2 vertical slice

User stopped training and authorized replacement of the v3.1/538-D autonomous route. Implemented v4 source in `sim/manorl/autonomy_contracts.py`, `sim/manorl/autonomy_v4.py`, and `sim/manorl/autonomy_batch.py`; added focused independent tests and `docs/manorl_autonomy_v4.md`. The production cache compiler creates MJX `Data` through `mjx.make_data` and performs N=1 `mjx.forward(..., impl="warp")`; it uses static compiled collision geom metadata and exact cube2 box collision distance. It does not construct native `MjData` or call `mj_forward`, `mj_step`, or `mj_geomDistance`.

Prediction before runtime check: a forward-consistent N=1 CPU reset and action should produce finite `(1,957)` raw and `(1,829)` encoded observations, a finite reward, and a valid pinned contact reduction. Observed exactly that for package identity `cube2_02_2833`: reset `(1,957)`, `(1,829)`, all finite; one zero-action step `(1,829)`, reward `1.8973922729492188`, done false, valid true, all finite. The transition executes four Warp steps then one Warp forward before all extraction.

Focused `tests/manorl/test_autonomy_v4.py`: 7 passed. It independently covers seven-block boundaries, encoded width, quaternion double cover/6D, positive-gap reference anchor identity, rotating-anchor velocity, canonical pair order/object-all/table exclusion and lever torque, reward clipping/severe reason bits, and v3 checkpoint rejection. No training, server, GPU scaling, or W&B workload was launched.

## 2026-09-10T01:00:00Z — v4 cache/state correction

Corrected 09f9c13 cache/state semantics without starting training. `compile_model_metadata_only` now constructs and validates static MuJoCo model metadata but avoids legacy `validate_static_fk`, whose native MjData/mj_forward oracle would otherwise enter the v4 production call path indirectly. The v4 compiler validates the path through Warp FK only.

The cache now samples each of the 16 real compiled mesh collision geoms from its `mesh_vert`/`mesh_face` slices after applying static geom pose/quaternion once to body-local coordinates. Cube2 collision triangles from `object_collision_vertices` are already body-local and are not transformed by the object geom pose again. JAX triangle closest-point and convex-halfspace classification supplies signed gaps and projects interior points to mesh surfaces. Samples are deterministic area-spread 128/region and 64/object. Cached q raw and feasible q both receive the common support shift before Warp FK. COM/palm derivatives are 7-frame/poly2 filtered; angular velocity is quaternion-derived and nonzero on the pinned reference. The cache hash includes version, arrays, shape/dtypes, parameters and static mesh/geom values.

Physical state now retains raw SI q separately from normalized q; command clamps and finger reward consume raw q. Resolved MANOHandAutonomousV1 `dofRate` and `antiwindupError` arrays are the common source for qdot, servo error and envelope. Observation scales, future/reference frames, cached table-relative bottom pair and fallen threshold are corrected. Runtime emits raw 957; trainable PointNet/829 is deferred.

Focused test command passed 11 tests. Corrected real N=1 CPU pinned package `cube2_02_2833`: reset raw/runtime `(1,957)/(1,957)`, finite; one zero-action Warp step reward `2.2570745944976807`, done false, valid true, finite true; max cached reference angular speed `1.4964786768`. No training/server/GPU-scale/W&B process ran.

## 2026-09-10T02:00:00Z — v4 raw-model integration

Commit `531e51a` replaces the public stale v3 training CLI with stopped-training `inspect` and `smoke` commands. Raw 957 observations flow through `BatchedAutonomyAdapter` into a registered trainable Torch PointNet and both actor/value consume the derived 829 feature. Model forward/backward tests prove pointnet gradients and checkpoint state roundtrip. No optimizer/trainer was invoked. The pinned `inspect` and frozen untrained `smoke` commands passed on cube2_02_2833 CPU with finite valid one-step physics.

The fast reducer now rejects non-pyramidal explicit requests and skips unsolved `efc_address=-1` rows before NaN contact positions can enter a torque lever. It remains a pyramidal/condim3-only fast path; helper parity, actual pair-contact oracle, slip and full batch reset integration remain unresolved.

## 2026-09-10T03:00:00Z — v4 B4096 CCD/contact capacity contract correction
The B4096 persistent-CCD retry failed in `mjx.make_data` before workspace installation: compiled cube2 has 121 mesh×mesh convex pairs/world, so explicit `ccd_contacts_per_world=121` produced `naccdmax=121*4096=495616`; v4 allocated recommended `naconmax=262208`; pinned MJX-Warp requires `naccdmax <= naconmax`. The old fast path's successful B4096 capacity was 524352, establishing that global contact arena capacity must cover explicit global CCD scratch. Corrected only `BatchedAutonomyRuntime` capacity construction: `ccd_capacity=ccd_contacts_per_world*num_envs`; `warp_contact_capacity=max(recommended_warp_contact_capacity(num_envs,("right",)), ccd_capacity)` when explicitly supplied; default retains recommended allocation. `njmax=512`, CCD iterations, model, reward, and four physics steps are unchanged. Focused contract tests and py_compile are the pending discriminating validation; no native production call or training is authorized.

Focused validation: `PYTHONPATH=. /home/jay/dexrobot/FromSSH/manoRL_mujoco/.venv/bin/python -m pytest -q tests/manorl/test_autonomy_batch.py tests/manorl/test_autonomy_v4.py tests/manorl/test_autonomy_v4_integration.py` passed 22 tests in 19.27s. The pure runtime capacity contract proves B4096 explicit `121*4096=495616` yields `naconmax=495616` and `naccdmax=495616`, satisfying the pinned inequality; absent explicit scratch, B4096 retains the recommended `262208` arena. `py_compile sim/manorl/autonomy_batch.py` and `git diff --check` passed. No production/native runtime or training was run.

## 2026-09-09T17:20:52Z — v4 single-reference PPO restoration

Updated task scope to the bounded v4 public learner/frozen-evaluation objective. Replaced stale v3 batch-training provenance/import surface with v4 `RlGamesPPO` reuse; added a raw-957 builder with registered PointNet model. The loop records raw Normal samples/log-probabilities, gives only clipped copies to physics, records terminal next observation before `prepare_action()`, and resets from runtime `last_done`.

First N=1 CPU command reached the physics transition then failed before PPO update: `V4Contact` exposes `paired_force_on_object`, while newly added compact telemetry used stale `pair_force`. Corrected the field name before rerun; this was an adapter telemetry integration error, not a physics/PPO failure.

Reran the authorized local command with the primary interpreter and no W&B/network: `PYTHONPATH=. /home/jay/dexrobot/FromSSH/manoRL_mujoco/.venv/bin/python tools/train_manorl_autonomy.py train --device cpu --num-envs 1 --no-persistentworkspace --updates 2 --rollouts 8 --learning-epochs 1 --mini-batches 1 --no-wandb --checkpoint /tmp/manorl-v4-n1/ppo.pt`. It completed 16 physical control transitions. Update 1: reward mean 2.2136263847, valid=1, actor loss -2.4028e-07, critic loss 45.68338; update 2: reward mean 2.1885323524, valid=1, actor loss 1.2666e-07, critic loss 44.33468. Both updates completed one optimizer minibatch and checkpointed finite model/Adam state.

Frozen load/evaluate command: `PYTHONPATH=. /home/jay/dexrobot/FromSSH/manoRL_mujoco/.venv/bin/python tools/train_manorl_autonomy.py evaluate --device cpu --num-envs 1 --no-persistentworkspace --checkpoint /tmp/manorl-v4-n1/ppo.pt --steps 4 --trace /tmp/manorl-v4-n1/eval.json`. It loaded v4 PointNet/model/optimizer metadata, ran four deterministic controls from frame 0, returned 8.8244338036, and had no natural termination in the requested four controls. Checkpoint: policy_steps=16, optimizer-state entries=25, normalizer=null, cache hash `e82724f5...` recorded but not compared.

Focused v4 verification after restoration: `pytest -q tests/manorl/test_autonomy_v4_integration.py tests/manorl/test_autonomy_v4.py` passed 20 in 19.25s. It covers public CLI arguments, raw action versus physical clipping, terminal/reset transition boundary, PPO optimizer/PointNet update, frozen checkpoint deterministic mean/load, and existing geometry/contact/cache contracts. No server/GPU-scale/long training/W&B run occurred.

## 2026-09-10T04:00:00Z — v3 route fail-closed correction

Removed the obsolete `sim/manorl/autonomy_imitation.py` producer and all exported v3 contract sentinel aliases. The aliases previously resolved to `REJECTED_BY_V4`, which allowed the dead imitation module to emit artifacts carrying that string rather than making the removed route fail. Replaced stale v3 test imports with compact v4 regressions: raw-Normal pre-update likelihood identity, non-finite update diagnostics retaining the named last-finite checkpoint, terminal/reset coverage, absence of v3 aliases/module, and v4 imports. This does not alter v4 checkpoint validation: v3/538-D metadata remains rejected before model loading. No physics, observation, reward, action map, or PPO algorithm behavior changed.

## 2026-09-09T18:46:30Z — v4 PPO model-only teacher warm-start

Added optional public `train --warmstart` and a strict v4 model-state loader. The PPO builder constructs its registered PointNet actor/value and fresh full Adam first, then loads only the source model before reset/first rollout. Warm-start compatibility gates model architecture, package digest, manifest/catalog, split, identity, asset pin, contracts, and physical control/physics clock fields. The source diagnostic-continuation flag remains recorded but is not treated as a physical clock mismatch. Output config/provenance record the resolved source path and `mode=ppo_warmstart`; periodic/latest/final checkpoint publication remains atomic.

Prediction: loading the offline teacher artifact must reproduce its deterministic means exactly before rollout while the PPO Adam state is empty and covers every trainable actor/value/PointNet parameter. Focused tests observed exact-zero-tolerance mean equality, zero initial Adam state, full parameter identity coverage, successful PPO state creation after one tiny update, and rejection of v3, architecture-mismatched, and provenance-mismatched sources. The real offline artifact loaded with exact equality across all 25 model state entries, including 16 PointNet entries, and produced a finite 1×28 mean. The four focused v4 test modules passed 27 tests; the tighter integration module passed 6 tests after adding mismatch coverage. `py_compile` and `git diff --check` passed. No physics training, GPU, remote, W&B, reward, observation, controller, or clock operation ran.

## 2026-09-10T08:43:09Z — v4 learning-rate interface and fixed-data attribution

Parent reports completed B2048 W&B run `bd9dd2d7`, 33,554,432 transitions with no lift, and authorizes a future GPU1 B2048 follow-up. The fixed-data diagnostic `outputs/manorl/contact_conditioned_autonomy/first-update-attribution-20260910T082353Z/result.json` uses one real N=1 first-32-control prefix and one full-minibatch Adam step: injected-versus-canonical real RlGamesPPO maximum parameter delta is 0. Late-phase KL full=0.085064, actor-only=0.086622, value-only=3.739e-6; changing only LR to 3e-5 gives 0.000847. This localizes the measured drift to actor updates rather than critic dominance. It is not a replay of historical B2048 training; physical evaluation still fails/no lift. EPISTEMIC now reflects that scope and keeps the empirical autonomy objective.

Exposed optional keyword/CLI learning rate, default 3e-4, through train, run_batched_ppo, build_batched_runtime and the shared config factory. Positive finite validation precedes CLI data/physics and runtime construction. Adam construction and all other config keys are unchanged. Actual Adam LR is retained in checkpoint config; CLI resolved W&B metadata and existing live per-update telemetry receive the same rate. A model-only teacher checkpoint cannot restore its optimizer LR.

Focused command: `PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/jay/dexrobot/FromSSH/manoRL_mujoco/.venv/bin/python -m pytest -q tests/manorl/test_autonomy_v4_integration.py tests/manorl/test_autonomy_batch_training.py`. Result: 21 passed. Added default/LR-only config invariance, two-step fake-adapter real PPO with model-only warm-start, actual Adam/resolved/checkpoint LR, stubbed CLI/W&B propagation, and zero/negative/NaN/±Inf early-rejection coverage. No real physics, training job, GPU, server, or network execution. `git diff --check` passed.

## 2026-09-10T09:39:40Z — v4 persisted optimizer continuation

Implemented exact v4 model/full-Adam/RNG continuation with cumulative total update targets, strict architecture/config/physical provenance/finite-state validation before W&B, and explicit full-start physical/episode-telemetry restart. W&B stubs exercise explicit ID, resume=must, nonconflicting initialization, allow_val_change budget/lineage update and cumulative log steps. Original warm-start history and source checkpoint SHA256/commit/counts/config/provenance are retained. No PPO objective, per-update epoch/GAE math, physics, reward or control changes.

Prediction: at the same explicit reset boundary, saved full Adam and RNG should reproduce the next samples and optimizer updates exactly. Observed zero-tolerance equality for resumed versus uninterrupted Python/NumPy/Torch samples, raw actions, final model tensors and every Adam tensor in a two-update-to-four real RlGamesPPO fake-adapter test; first resumed act uses cumulative timestep4 of8 with existing Adam step2. Repeated continuation to5 also passes. Checkpoint suffixes and rows begin at update3, not1.

Focused command: PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 primary-venv/python -m pytest -q tests/manorl/test_autonomy_v4_resume.py tests/manorl/test_autonomy_v4_integration.py tests/manorl/test_autonomy_batch_training.py tests/manorl/test_autonomy_v4_telemetry.py. Result: 66 passed. py_compile and git diff --check passed. Tests cover partial22-versus29 teacher optimizer, missing Adam, model/Adam shapes/non-finites, ABI/sampling/architecture, null normalizer, counter/budget errors, config/physical provenance drift, CLI conflict, writer preflight and explicit W&B identity. Installed Config.update signature supports allow_val_change.

CPU-only real artifact evidence: parent-provided immutable old4bb update192 checkpoint passed inspect_v4_resume for cumulative4096 at B2048×32, 4 epochs/16 minibatches/LR3e-5/separate critic, with config device still gpu. All29 full Adam states loaded onto CPU; retained Adam step12288, policy_steps6144, environment_transitions12582912; one CUDA RNG entry. No B2048 runtime was allocated and no remote, GPU, physical training or network writer was used. The artifact validates old-format loader compatibility, not GPU execution or empirical learning. EPISTEMIC records the explicit physical restart limitation; parent retains responsibility for waiting for successful update512 pipeline completion before the single-writer handoff.

## 2026-09-13T02:48:32Z — Directional anti-windup intervention, prediction
Clean starting HEAD aaf611ac488eba1603303c99aed62048a70d9b51. Changed only the v4
command map and public helper to block outward delta at/beyond the resolved
margin and retain the command under push-back. Also repaired the helper's
pre-existing array-truth validation parentheses, which prevented valid calls.
Focused 17 directional/parity tests pass. Prediction: preserving load-bearing
servo error may turn contact into lift; confirmation requires >5 mm bottom
clearance with sustained loaded contact, not contact counts alone. Baseline
teacher: terminal228/reason2, preterminal33 contacts/loaded, zero airborne,
max clearance -2.765667159e-5 m; full538 has204 contacts/loaded, zero airborne,
max +5.883525591e-5 m. Fixed rollout preserves the original teacher formula and
all physical settings; adds saved per-frame traces and a reset natural run.

## 2026-09-13T02:52:29.048473+00:00 — Directional anti-windup validation and negative A/B

Primary-venv CPU tests: test_autonomy_v4*.py plus test_autonomy_batch.py passed
103 tests in24.65s; selected test_autonomy.py command_map/rate_units passed
2 tests (5 deselected). The 17 new tests exercise the actual jitted transition
command branch and NumPy parity, both signs, exact margin, push-back, reverse,
zero hold, inside/crossing margin, clipping, and nonexecution preservation.

Local CPU N1 teacher pursuit used the original formula and package identity,
full538 diagnostic followed by full-start natural-terminal reset. Fixed
preterminal:33 contact/loaded,0 airborne>5mm, max clearance -2.766971011e-5m;
terminal228/reason2. Full:200 contact/loaded,0 airborne>5mm, max clearance
0.000203637290m. Natural run:228 frames,33 contact/loaded,0 airborne,
max clearance -2.769019920e-5m, terminal228/reason2. Longest loaded-airborne
sequence=0. Preterminal maximum clearance differs between the diagnostic and
natural reset runs by less than0.03micrometre.

Raw rows show contact starts at183, but terminal228 has no paired contacts:
object z0.023539625 versus reference0.124215175m. The maximum diagnostic
clearance is only0.204mm at436 (loaded), not a lift. Thus preserving servo error
is not sufficient to recover this teacher lift, and is not confirmed as the
dominant reason-2 cause. No reward/control weakening, training, network, GPU
or server operation occurred. Artifacts: feature outputs/manorl/
contact_conditioned_autonomy/antiwindup-directional/{teacher-physics-fixed.json,
teacher-physics-fixed-rows.json,teacher-physics-fixed-natural-rows.json,
evaluate_teacher_physics_fixed.py,teacher.log,tests.log,legacy-focused-tests.log}.

## 2026-09-13T04:48:10.143364+00:00 — Optional online teacher-anchor PPO

Parent-approved evidence: chase plus +0.2rad gated finger squeeze completes538
with129 airborne frames and tolerates action noise0.05; squeeze-BC closed-loop
drifts despite low training loss, and its PPO continuation lost lift. These are
parent-reported observations, not reproduced by this implementation worker.
Prediction: supervising the mean on pre-step states visited by PPO will reduce
policy-teacher error after two updates relative to an identically seeded control.
Implemented training-only runtime labels and separate post-PPO same-Adam mean
steps with full random minibatch passes; no physical/reference execution change.
Explicit None gradients isolate value-only and log_std momentum. Metadata gates
resume while absent legacy fields normalize to disabled defaults.

Primary-venv CPU command with PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1:
python -m pytest -q tests/manorl/test_autonomy_teacher_anchor.py
tests/manorl/test_autonomy_v4*.py tests/manorl/test_autonomy_batch_training.py
tests/manorl/test_autonomy_batch.py:117 passed in24.99s. Initial subset:73 passed.
New tests independently exercise gate/last-frame/joint clipping/current command,
PointNet/trunk/mean gradients versus value-only Adam momentum, disabled exact
seeded trajectory/model/optimizer counts, pre-step label count, CLI propagation,
legacy resume and enabled deterministic resume. No broad unrelated suite run.

CPU N1 discriminator used one runtime reset between arms, seed0, two updates of16
rollouts, one PPO epoch/minibatch, LR3e-4, separate critic, squeeze-BC model-only
warmstart, beta0 versus beta1/two anchor passes. Both arms completed32 physical
transitions, model/Adam/metrics finite, valid=1 each update. Both frozen means
were measured on the same32 pre-step beta0 physical observations and labels:
beta0 MSE0.6698881984; beta1 MSE0.4651518166 (30.5628% reduction). Own-rollout
beta1 MSE0.4371310174 is separately labeled, not the matched comparison.
Beta1 recorded16 labels and2 anchor optimizer steps per update. First rollout
reward means differ by4.77e-7 after full-start reset (CPU numerical scale).
This32-frame approach prefix does not test the frame200 squeeze or grasp.
Artifacts: outputs/manorl/contact_conditioned_autonomy/teacher-anchor-ppo/
metrics.json, discriminate.py, physics.log, tests-final.log, beta0.pt/beta1.pt
and their final snapshots. No GPU/server/network/W&B or foreign process touched.

## 2026-09-14T01:57:35Z — Multi-reference v4 runtime bank

Started clean at3b9d1d9; asset pin f98da997. Added device bank time fields,
per-env gathers/length/initialization, same-reference subset resets and all40
TRAIN CLI selection with ordered assignment provenance. Reference/control,
reward, raw957, PointNet and checkpoint ABI remain unchanged.
Focused initial run36 passed/1 failed due to incomplete synthetic cache fixture;
updated fixture to real cache field dimensions. Broad v4/batch run107 passed/
16 failed because isolated command fixture omitted new env_ref=None; corrected
fixture without changing command semantics. Focused recheck42 passed in14.85s.
CPU N1 bank-one versus single2833 eight-step qpos/raw957/reward/done bitwise
parity passed. Two-reference smoke reached8 finite/valid steps; evidence script
first failed slicing global scalar contact.valid, then strict JIT-versus-eager
observation comparison differed by1.49e-6. Script now compares reference gathers
with identical eager arithmetic exactly and runtime JIT output at3e-6 tolerance.
No physical/controller/reward changes were made for these evidence corrections.
Artifacts: outputs/manorl/contact_conditioned_autonomy/multi-reference-runtime/
(m1-tests.log, m1-recheck.log, physics-parity.json, physics-bank*.log and scripts).

2026-09-14T01:57:47Z: Two-env2833/2835 CPU smoke passed: distinct initial hand and
object poses,8 finite/valid transitions, own-reference raw957 eager equality,
subset reset env1 restores its own frame0, env0 qpos/qvel/ctrl bitwise unchanged,
indices[8,0]. Physical clock120/480Hz and4substeps unchanged. See physics-bank.json.

## 2026-09-14T01:59:46Z — Intent-gated teacher intervention

M1 committed a78444c. Replaced next-frame>=200 squeeze activation with current
assigned-reference max(proximity*confidence*valid)>=0.5. Kept next-frame feasible
q, same+0.2rad joints/limits/rates and training-only label execution boundary.
Metadata records gate/threshold/comparison; old disabled recipes normalize to
disabled, while enabled frame-gated optimizer resumes reject recipe drift.
Prediction: teacher2833 must naturally complete all538 transitions with loaded
airborne contact; the threshold may activate at a different frame and disengage
later, so the old129 airborne count is not assumed. CPU N1 full natural-terminal
teacher job uses unchanged runtime/control/physics and stores per-frame intent,
loaded contacts, lowest-point clearance and reasons. No neural training launched.

## 2026-09-14T02:00:39Z — Intent-gated validation

CPU N1 2833 natural-terminal chase fromframe0 completed538/538, reason1,
all finite/valid. First active current frame162;230 loaded-contact frames;
130 airborne>5mm frames, all130 loaded; maximum bottom clearance0.168938637m.
No diagnostic continuation was used. Original frame200 recipe baseline129
is parent-reported; current130 is direct physical evidence. Per-frame rows and
recipe/clock saved in physics-teacher.json/physics-teacher-rows.json.
Focused suite command: primary-venv python -m pytest -q
 tests/manorl/test_autonomy_teacher_anchor.py tests/manorl/test_autonomy_v4*.py
 tests/manorl/test_autonomy_batch_training.py tests/manorl/test_autonomy_batch.py
with PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1:125 passed in34.36s.
Added telemetry-own-target/progress test afterward; bank suite7 passed. Tests
exercise exact0.5 boundary, no squeeze before, per-env intent differences,
disengagement, next-frame/limits/current-command arithmetic, disabled legacy
resume and enabled gate-drift rejection. No neural training/server/GPU/network
or W&B job; physical evidence is teacher-only. Exact launch recipe remains
parent-owned and model-only warmstarts the successful single2833 policy.

## 2026-09-17T18:08:40+08:00 — Finalized reference-speed-gated contact-priority reward

User explicitly approved updating current source to the discussed reward, then explicitly prohibited further subagent use after the delegated executor hit a provider usage limit before tool work. Main session implemented the source change directly; no training, server action, W&B run, checkpoint replacement, or cleanup occurred.

`sim/manorl/autonomy_v4.py::compute_reward` now gates object position, rotation, and velocity reward terms using reference-only effective speed `sqrt(||v_ref_COM||^2 + (r ||omega_ref||)^2)`, `r=0.04330042412758056m`, with smoothstep 1% at/below0.01m/s through100% at/above0.10m/s. It changes hand-relative0.5→0.125, geometry0.2→1.2, and severe-penalty−25→−75; all other reward, force, observation, action, done/reason, control, and physics behavior remains unchanged. Geometry remains the only contact-related reward (reference-intent-weighted hand--object anchor correspondence); no force magnitude/table/slip reward was introduced. Exact collision-vertex radius about COM is compiled/cache-hashed and required shared for reference banks. Training and frozen evaluation provenance/config now include the JSON-safe reward parameter set `manorl.autonomy.reward.v4.reference-speed-gated.contact-priority.v1`; it is deliberately not a model loading gate so old released inference checkpoints remain usable.

Predictions: code-level gate endpoints0.01/0.505/1.0 at effective speed0/0.055/0.10m/s; actual object speed must not influence object-term gate; per-environment multi-reference speed selection must differ correctly. Observations: `test_autonomy_v4.py` and `test_autonomy_v4_multi_reference.py` test those exact contracts plus new coefficients and−75; `JAX_PLATFORMS=cpu pytest -q tests/manorl/test_autonomy_v4*.py` passed108; `pytest -q tests/manorl/test_autonomy_batch.py tests/manorl/test_autonomy_batch_training.py tests/manorl/test_autonomy_teacher_anchor.py` passed20; py_compile and git diff --check passed. An unrestricted first test attempt failed because JAX tried CUDA while the local GPU was out of memory; same tests passed with CPU explicitly selected, so this is an environment scheduling fact, not source regression.

Frozen N1 CPU eval using unchanged bundled `checkpoints/manorl_autonomy_v4_cube2/policy.pt`, cube2_02_2833, and the pinned package wrote ignored evidence `outputs/manorl/contact_conditioned_autonomy/reward-v4-gated-validation/frozen-eval-cpu.{json,npz}`: 538 steps, natural reason1,359 loaded hand--object contact frames,126 loaded-airborne frames, maximum clearance0.168186753988266m, return302.1019148826599. Runtime source compiled radius exactly0.04330042412758056m. It is physically equivalent to the prior successful trajectory within repeat noise; return differs0.036780 from offline v3's302.0651347453153 because the independent CPU repeat diverged sub-millimetre/numerically. User explicitly accepted this difference as within error and requested continuation. No severe event occurred, so−75 was independently unit tested rather than exercised by this success trace.

## 2026-09-17T18:25:22+08:00 — Authorized 1000-update single-reference reward run

User explicitly authorized a new server training run, changed the budget to1000 updates, and then explicitly prohibited push, fetch, merge, and rebase; current local feature commits `c73397a`/`eeccca1` are the only source baseline. No Git network operation occurred. Server1's mandatory Unison preflight found the watcher had exited after Broken pipe into zsh; its profile syncs the dirty protected local dev root (which lacks autonomy_v4) rather than this clean feature worktree. Restarting it would risk propagating unrelated dev untracked files while still not delivering the intended code. Server1's live mirrored source is likewise an old non-Git tree without autonomy_v4. The observed existing valid run mechanism is a content-addressed immutable per-run `source/` snapshot, so the training source was deployed by a one-way `git archive` of tracked files at `eeccca195606d3c546b12a402d1ae7666bac4af3`, local/remote archive SHA256 `65af1f6b5f7ccfff11b9dd38e43ead5865ba12ac179ffdda7c37d8ba28ee76fe`, extracted only to the new run path and linked to the existing pinned DexStream assets. No protected source, old run, or remote Git branch was modified.

Server1 GPU2 UUID `GPU-c4f5fc2c-32d3-d6ac-ae31-81ab5ca14c50` was idle at15MiB; W&B authenticated as `sunjay45711`; the remote source printed the intended parameter set. New run: `/home/jay/dexrobot/FromSSH/manoRL_mujoco_autonomy_runs/v4-gpu2-b4096-reward-gated-u1000-20260917T102157Z`; tmux `manorl-v4-reward-gated-u1000`; W&B ID `rwg1000c733v1`. It uses exact successful comparator initialization SHA256 `18c29d4b7e983b4959cf8e2e2280929d6bd16713a8ef994988ad4734b53c2cb2`, cube2_02_2833 package manifest SHA `e826de23d4586611001230d340eaf0c68752b09988eed7928bf8884f056d8706`, B4096×32,1000 updates/131072000 transitions,4 epochs/16 minibatches/LR3e-5, separate critic, beta1/pass2 contact-intent teacher anchor, new Adam/RNG, checkpoint interval64. `--resume-checkpoint` is intentionally absent. The launch shell has started and created launcher/trainer PIDs; at the first 20-second sample Python was live in initial cache/JAX compilation and no update/GPU allocation had yet occurred. Await the first metric; do not interpret startup as a training result.

## 2026-09-17T19:45:16+08:00 — Asset-pin correction and stable 1000-update retry

The first authorized run `/home/jay/dexrobot/FromSSH/manoRL_mujoco_autonomy_runs/v4-gpu2-b4096-reward-gated-u1000-20260917T102157Z` stopped before update1: its private source symlink resolved to the machine-global DexStream deployment at commit `778614d09e917deffed0bff3f357aa237efa762d`, while the source manifest correctly required trained asset pin `f98da997f316c8a6b4bc2931cabed19e831ef163`. The failure is explicit in `phase=FAILED` and the preserved traceback; no metric row or training transition was produced. No global asset tree was changed.

Server1 had an exact materialized checkout at `/home/jay/dexrobot/FromSSH/manoRL_mujoco_dev/assets/dexstream_digital_assets`, Git HEAD `f98da997f316c8a6b4bc2931cabed19e831ef163`. Its many Git-status modifications are expected materialized LFS/mesh bytes and were not used as the acceptance proxy. A fresh run-local source snapshot instead linked privately to that checkout and passed the direct N1 `inspect` path: identity cube2_02_2833, raw957, control/physics clock1/120 and1/480 with4 substeps, CCD121, constraint capacity512, cache hash `20bbed72ba6d29535645bbf01ca39f472fb7e451f087151e9b1400be2f8e97a9`.

Active retry: `/home/jay/dexrobot/FromSSH/manoRL_mujoco_autonomy_runs/v4-gpu2-b4096-reward-gated-u1000-f98-20260917T111510Z`; tmux `manorl-v4-reward-gated-u1000-f98`; W&B `https://wandb.ai/sunjay45711-dexerto/mujoco-mano/runs/rwg1000c733f98v1`. Inputs preserve source archive SHA256 `65af1f6b5f7ccfff11b9dd38e43ead5865ba12ac179ffdda7c37d8ba28ee76fe`, `initial.pt` SHA256 `18c29d4b7e983b4959cf8e2e2280929d6bd16713a8ef994988ad4734b53c2cb2`, and package manifest SHA256 `e826de23d4586611001230d340eaf0c68752b09988eed7928bf8884f056d8706`. At the stable sample it had reached update205/1000 (26,869,760 transitions), `valid=1`, actor loss0.03681, critic loss13.0290, teacher-anchor loss0.75929, exact KL0.07206; GPU2 used about16.9GiB at100%. Consecutive rows remain finite and W&B is syncing. These are stability observations only; final frozen physical evaluation remains required after update1000.

## 2026-09-17T20:00:00+08:00 — Case-branch packaging against current dev

User authorized publishing as a new case branch without updating the existing remote feature. Created `case/contact-conditioned-autonomy/reward-gated-v1` through `scripts/start_pi_task.sh` from current protected `dev` commit `6d66c49`. Applied the complete autonomy delta from original feature fork base `c38f787` through source evidence tip `41ae360`; patch SHA256 `f40111ff5758ffcbddfd284c1d84e40cdab398ae4e79a4279fa73e22e1e214a8`. This avoids deleting unrelated files added to dev after the original fork.

One semantic document conflict in `.memory/project/manorl-clock-contract.md` was resolved by retaining both current dev's 100→120Hz repair invariants and the autonomy optimizer/teacher/multi-reference boundaries. `sim/manorl/assets.py` three-way-applied cleanly, then received the minimum schema compatibility needed for the exact trained f98 asset contract: optional object `mjcf`, `textures`, and `material` records retain full new-schema validation while f98 falls back to authoritative object rgba. The case explicitly changes the DexStream gitlink and manifest together from current dev `778614…`/32-object schema to trained `f98da997…`/28-object schema; leaving either half unmatched was experimentally shown to fail before environment construction. No asset content was committed.

Validation used a temporary local materialized f98 checkout while the branch records only the gitlink. Focused CPU suite: autonomy v4/control/integration/multi-reference/resume/telemetry, batch/batch-training, teacher-anchor, and assets; 148 tests passed after integration. Core reward source, public CLI, tests, docs, and bundled policy are byte-identical to feature source `41ae360`; the case additionally preserves current-dev changes outside the autonomy delta and the explicit dual-schema compatibility above. Remote push must be ordinary/non-force to a previously absent case ref; existing `feat/contact-conditioned-autonomy` remains untouched.
