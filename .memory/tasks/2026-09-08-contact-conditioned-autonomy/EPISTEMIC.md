# Current model

## Objective
Accurate demonstrations specify hand/object geometry and contact intent but not physically valid actuator commands. The autonomous actor must own all 28 commands and learn physical grasp/lift/transport/place/release under the pinned simulator.

## Supported mechanism
M1's standalone path was corrected in M2 rather than treated as competence. `simulation_clock(120)` and `compile_model(... physics_timestep=1/480)` enforce 120 Hz control and four 480 Hz physics substeps. Raw object pose plus initial quaternion and actual collision vertices define one support translation; that translation is applied to every reference hand XYZ/object frame and reset while source arrays remain immutable. Reference FK is evaluated per frame.

Reference and measured contact intent now use `mj_geomDistance` over actual hand-segment and object collision geoms. The nearest signed-distance witness endpoint is transformed to object-local coordinates and receives distance/penetration confidence. Solved hand-object forces come from `MjxWarpPhysicalProducer.hand_object_force_on_object_world_N`; table contacts cannot masquerade as hand-object force. Object-local relative keypoint motion is an explicitly named slip proxy masked by measured hand-object force.

The action map has explicit physical rates (.5 m/s wrist XYZ, 2 rad/s wrist rotation, 4 rad/s fingers) and measured-state envelopes (.02 m/.25 rad/.35 rad). The envelope bounds integrated target windup around measured state without clamping to a reference, preserving load-induced servo error. The reference-pursuit diagnostic actor computes a bounded next-command action; the command map sees only that output and measured state.

Reward compares object path and orientation, reference hand/object relation, demonstrated contact proximity/anchors, measured hand-object contact, finger configuration, velocity and masked slip. Net supporting force is trace-only; impact magnitude is not rewarded. Release is inferred from the demonstrated proximity window. Drop/path-divergence termination is distinct from horizon completion.

## Telemetry/publication boundary

Future formal PPO updates use `sim.manorl.autonomy_telemetry.TelemetryAccumulator` to reduce physical `info` fields and reward terms once per update, while `latest_ppo_metrics` reads only native `RlGamesPPO.tracking_data`. Existing aliases are retained; unavailable statistics (currently clip fraction) are omitted. W&B history is keyed by environment transitions, with canonical entity/project defaults and explicit overrides. `tools/publish_manorl_autonomy_evaluation.py` consumes a saved trace plus matching checkpoint, rejects format mismatches, computes actual-state metrics with denominators, and appends evaluation JSON/MP4 only to a verified FINISHED run. Max-lift/endpose remains diagnostic.

The completed 538-step learned first-run trace is stronger failure evidence than the earlier four-step smoke: hand-object force-positive frames = 0, peak lift = 0 cm against target 19.379 cm, path RMSE = 0.068146 m, wrist-reference RMSE = 0.1047 m, final phase horizon reached, success false. No historical PPO losses can be reconstructed.

## Evidence
Prior negative traces showed zero hold could earn high reward while target cube lift was 19.4 cm, actual hand-on-object force exactly zero, and contact_count 35/per-hand geometry force represented table contacts. Prior pursuit diverged at 233 after large XY displacement; mean reference proximity max 0.0744 made the old release threshold unreachable.

After correction, six semantic tests pass. Same identity `cube2_02_2833` full-start diagnostics (538 frames) complete for zero and bounded reference pursuit. Corrected zero still has zero hand-object force and no lift despite 35 contacts; corrected pursuit has hand-object force up to 2.07 N over 84 frames and slip proxy up to 0.281 m/s but no lift competence. Both traces record source package, contract IDs, clock and config hash. Full package summary loads all 50 identities.

## Boundary and next question
M2 establishes a discriminating runnable diagnostic, not learned competence. The next training interface must reuse canonical GAE/PPO with identity splits and evaluate full-start episodes; no standalone PPO smoke remains. The next causal question is whether a canonical policy can convert demonstrated collision witnesses and measured hand-object wrench into sustained support and target lift without table-contact leakage.

## M3 first learner
The shortest valid learning path now exists: one N=1 Gymnasium boundary over the v2 physical MJX-Warp environment feeds canonical `RlGamesPPO` and skrl GAE/bootstrap. A deterministic 40/5/5 identity split fixes the three inspected cube2 identities in TRAIN and persists a digest. The first two-transition update and checkpoint/load deterministic rollout succeeded on `cube2_02_2833`, but learned competence is absent: four-step rollout has zero peak lift and `task_success=false`. W&B is enabled by default; local smoke used explicit offline mode. Multi-reference, held-out learned rollout, and batched throughput remain later milestones.

## Launch-boundary evidence
The first formal learner's initial GPU incompatibility was caused by an adapter hardcoding CPU despite a CUDA runtime. A second semantic bug allowed NEXT_STEP to consume terminal observations against reset state. Commit `63add62` removes the vector wrapper for the N=1 first learner and requires explicit resets, while preserving canonical skrl GAE/bootstrap. GPU N=1 training and checkpoint evaluation now run; the four-step learned rollout remains physically unsuccessful and is reported as such. Near-term acceptance is runnable learning; final acceptance requires a frozen checkpoint to consume new held-out references without retraining, masks or special reward.

## Corrective telemetry model
Telemetry now distinguishes update-window throughput from cumulative transition totals and distinguishes ongoing episodes from completed episodes across PPO cuts. Airborne contact is a physical predicate: measured hand-object force plus actual rotated collision-mesh bottom clearance above the floor. Orientation is a normalized quaternion geodesic angle reported in radians/degrees. Evaluation publication is appended to a finished W&B run using automatic monotonic SDK history steps; checkpoint transition count is a field, not that history step. The current first-run trace remains no grasp/no lift; `full_task_success` is nullable and release-window contact-free counts are descriptive only.
