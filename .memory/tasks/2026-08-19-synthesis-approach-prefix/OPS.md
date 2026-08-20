# OPS

## 2026-08-19T00:00:00Z
- Created feature worktree from dev@4558504.
- Confirmed synthesis checkpoint stepper calls deterministic_actions + env.step only; it does not call record_transition/post_interaction and therefore does not write PPO memory.
- Confirmed ordinary early phase is fixed early30; process_residual_actions zeros processed/cumulative residual during that interval.
- Confirmed each repeated synthesis retry launches a fresh child with attempt_seed = base_seed + attempt_round - 1.
- Implemented initial immutable positive-Z prefix transform and verified against /home/jay/data/manorl_mtp_pre60: original suffix/source indices remain exact and sampled Z is positive.
- Decision: augmented trajectories use prefix length as per-row early phase; ordinary trajectories preserve early30.
- Decision: 0.10 m deviation terminal is disabled only in the generated prefix and resumes at original pre60 frame 0.

## 2026-08-19T12:00:00Z
- Full pre60 catalog audit: 3,429 trajectories; generated prefix L=41..187 (median 110); original suffix/source indices/object reference exact; discrete splice velocity error max 0.
- Raw C2 acceleration extrapolation was falsified: noisy capture acceleration produced max 3.37 m/s and 139 rad/s. Replaced with exact discrete C1 velocity splice. Final audit max linear speed 0.79 m/s and max angular speed 8.30 rad/s, consistent with source frame0 intervals.
- Real MJX-Warp iphone audit showed low-positive-Z straight paths could contact the floor. Raised minimum elevation to 15 degrees and added a 4 cm endpoint-smooth wrist arc; 12/12 sampled prefixes then had no controlled-right-hand solved floor contact above 0.2 N.
- Viewer batch-loop experiment was rejected because background worlds had already progressed. Final visible loop uses one active env; terminal seed49 reset to seed50 with progress=0 and distinct start, then continued seed51+ in one window.
- Selected accepted parent row ee54c011-8237-5cc4-9061-cd2ea8e5163c from the 10,605-row production Lance: source banana_01_052, checkpoint SHA daa1d2e73a6bff496b09296d9400a53a75cdf1a7765f443bf24ad40be62964ad, object XY offset (-0.0137602, -0.0176767) m.
- End-to-end GPU exporter produced one successful compact v2_contact row from this parent on seed49. Manifest bound parent and prefix; rollout completed at 655 steps with success reason 1.
- End-to-end audit found movement metadata remained at source 60 instead of shifted 150; build_v2_row was corrected to serialize trajectory movement_start/end.

## 2026-08-20T18:42:30+08:00
- User-approved visible far/near approach and retreat behavior was followed by a mechanism review rather than immediate merge.
- Added accepted-parent v2 fields: raw source frame0 right q_ref[3:28], parent movement-end, last solved contact diagnostic, state-aligned retreat source frame, and explicit movement-end+5 anchor contract.
- Contact-pair force magnitude >0.2 N is authoritative; selector rejects a parent whose last solved right-hand/object contact precedes movement-end.
- State/transition audit found command-array state mapping was off by two for banana. Fixed selector to use reference.source_frame_index[anchor_state] and verify command mapping. Banana movement-end435 +5 maps state440→source528→canonical pre60 index320.
- Rejected old cube1_02_1002 parent under final rule: last contact380 < movement-end389. Removed its stale local descriptor.
- Added far/near modes. Near seed is hash(episode, near-approach); retreat seed is episode seed. Both share one endpoint config but never the same sample.
- Added exact raw source-row frame0 q_ref[3:28] start. Wrist orientation and 22 finger joints use discrete-C1 quintic to pre60; duration includes translation, wrist rotation, and maximum finger displacement. Forty-six representative families had no raw/pre60 Euler branch crossing.
- Added prefix candidate rejection for solved right-hand/table and right-hand/object contact >0.2 N.
- UUID review found far/near/config collisions. Added semantic augmentation identity v2 SHA256; excluded machine paths while retaining parent/source/checkpoint/config/seed identity.
- Full-row audit found hard residual zero at retreat entry caused 3.0 cm XYZ and 0.402 rad finger target jumps. Replaced it with zero action/no new accumulation plus quintic cumulative-residual decay. Final banana row anchor target jump equals reference only (0.245 mm, 0.00282 rad), residual decays monotonically to exact zero at the final issued command, and terminal reason is 1.
- Final full artifact: /tmp/manorl_final_review_full.lance; UUID f20e1dc9-5d27-56dd-9581-b73c1e977536; 637 frames / 636 transitions.
- Final compact projection: /tmp/manorl_final_review_compact.lance. Upgraded validator accepts v2_contact fields and verifies reference/command mapping plus force-frame consistency.
- Final focused: 80 passed. Broad selected suite: 120 passed, 7 skipped, 19 setup errors; every error is the known absent external fixture /mnt/nas-222-project/mocap_v2/lance_datasets/human_p1_remake/npy_s02_v3.lance.
- Raw/pre60 reference values can exceed current URDF lower bounds. Actual final-row audit: peak initial joint rate 1.65 rad/s, acceleration 79.9 rad/s^2; no instability. Preserved exact raw source pose per user contract and documented this data↔URDF limitation.

## 2026-08-20T19:45:48+08:00
- User changed the production retreat anchor from parent movement-end+5 to movement-end+15. At 120 Hz this delays retreat by 10 additional frames / 83.3 ms relative to the tested +5 contract, for a total 125 ms post-movement delay.
- User explicitly judged a repeated GPU rollout unnecessary because the change moves the same retreat mechanism later. Validation was therefore limited to versioned contracts, state/source/command mappings, real predecoded trajectories, full-catalog geometry/tail margins, and focused tests. The +15 physical result is an inference from the +5 mechanism tests, not a separately observed rollout.
- Upgraded accepted-parent contract to manorl_accepted_synthetic_parent_v3, anchor contract to parent_movement_end_plus15_v2, approach contract to the near-world-Z v4 definition, retreat contract to movement-end+15 v4, and augmentation UUID identity to v3. Old v2/+5 descriptors fail closed.
- Rebuilt banana descriptor: movement-end435 -> anchor450, source frame538, last-contact461, anchor-to-final XY 0.104508 m.
- Rebuilt cube1_02_1006 descriptor: movement-end379 -> anchor394, source frame472, last-contact384, anchor-to-final XY 0.111082 m.
- Offline transforms on canonical pre60 references mapped banana base anchor330 -> augmented anchor401 and cube1 base anchor274 -> augmented anchor352. Both retain 235 geometrically replaced tail states.
- Full 3,429-reference +15 audit: 3,426 valid near geometries, minimum 236 tail states, no below-floor start (minimum world wrist Z 0.063602 m). Three rows have zero horizontal anchor-to-final displacement and are rejected: cylinder4_01_533, cylinder4_14_3047, cylinder7_10_3340.
- Near Z mechanism corrected before +15 migration: endpoint XY remains final-object-relative and yaw-mapped to initial object; Z preserves sampled retreat endpoint absolute world height. Translating Z by object height had mapped lifted cube2_04_2979 below the floor.
- Final cheap regression after +15 migration: 97 passed; compileall, bash -n, staged and unstaged diff checks passed. No GPU process was launched.
