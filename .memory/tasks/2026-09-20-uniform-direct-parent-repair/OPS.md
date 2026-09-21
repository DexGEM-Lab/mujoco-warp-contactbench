# Evidence

## 2026-09-20T14:25:09+08:00 task initialization
- User authorized repairing the 42 historical parents into one setting; no subagents authorized.
- Created branch feat/uniform-direct-parent-repair at dev@6d66c49 in isolated worktree /home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/feat-uniform-direct-parent-repair.
- Current immutable dataset: compact Lance v2, 1923 rows. Formal selection sidecar, final visual layout, 1389 lineage, and rebuild plan v2 published beside the dataset on NAS.
- Verified all 1389 original trace.npz files: 1389/1389 accepted, SHA-matching, required state/control fields present, ctrl_substeps shape [N-1,4,28]; 1.73GiB total.
- Original parent split: 30 unchanged120Hz baselines + 12 local repairs; actions 003=22,005=6,006=1,007=7,009=6.

## 2026-09-20T14:33:38+08:00 A_row035 compression canary
- Implemented `tools/run_uniform_direct_parent_canary.py`: no state forcing after frame0; one 120Hz direct target held across four 480Hz substeps; compares U1 pyramidal/impratio1 vs U2 elliptic/impratio100 and first/mean/last/teacher/lead targets.
- A_row035 U1: all five target initializations failed physical_pose; best max object error 0.2247m (teacher), mean error 0.6543m.
- A_row035 U2: mean-substep target passed every physical gate with 0.0089m max object error and 0.0125m max hand-root error; first-substep also passed with 0.0345m object error. U2 last missed final position; teacher state targets failed.
- Interpretation: hidden PID is not inherently required if its per-substep actuator stream is compressed appropriately, but U1 contact solver is insufficient for this 1.28kg pitcher. U2 retains self-contained direct-target semantics and is now the leading global candidate.

## 2026-09-20T14:37:00+08:00 final setting clarified by user
- User requires exact setting parity with the existing 534 rows. U1 (direct target, pyramidal, impratio1) is final; U2 is diagnostic only and cannot be published.
- Five hard-parent canaries: U1/mean failed all action families. U2/first or mean passed A_row035(006), B_row099(003), A_row020(005), B_row200(007); A_row050(009) became functional and missed only the 15deg final orientation gate by0.52deg under U2/mean.
- Next intervention: U1 iterative target fitting from successful actual-state teacher. If bounded automatic fitting fails, user explicitly authorized local Astra plus interactive simulator repair.

## 2026-09-20T14:59:29.075906 bounded native-U1 placement checkpoint
# A_row035 bounded U1 placement checkpoint

No accepted trajectory. All three new attempts ran native U1 from frame0; no model or physical parameters changed. Target bytes through frame1030 and after the recipe endpoint remain unchanged. No U2 simulation or target conversion was used.

## Mechanism and chronology
Iter6 fails grasp/lift; iter7–9 progressively improve lift but topple. Iter10 reaches14.28cm lift with opposed contact, is low during pouring (frame780: support contact and58.25deg tilt versus teacher82.92deg airborne), then recovers upright supported placement at1030 (0.68deg). The teacher is state evidence, not an alternate simulator. Frame1030 is a recovered pre-withdrawal checkpoint, not a globally teacher-matching checkpoint. Original withdrawal rotates while contacting the handle: tilt15.61deg at1110, followed by recontact and toppling.

Holding wrist orientation during release eliminates toppling in release01. Its nominal open posture retains palm/pinky/thumb contact; withdrawal drags pitcher backward. Zeroing fingers in release02 does not open the thumb away from the handle: frame1170 still contacts palm and thumb_cmc/mcp/ip. Lateral extraction in release03 leaves only thumb_ip contact at1110/1170 and topples the pitcher. Persistent thumb hooking, not wrist trajectory error alone, is the remaining mechanism.

| Attempt | Edit window | Lift cm | Final source-reference error cm | Tilt deg | Permanently clear frame | Failed gates |
|---|---|---:|---:|---:|---|---|
| iter10 existing | global fit |14.28|25.61|86.22|not measured here|upright, position|
| release01 |1030–1300|14.27|12.41|0.45|1186|position|
| release02 |1030–1380|14.27|29.31|85.71|never|release, upright, position|
| release03 |1030–1380|14.28|20.15|87.06|1237|upright, position|

Best target/trace/contact/provenance: release01/. Its final teacher-object error is8.36cm, distinct from the original gate's12.41cm source-reference error. Maximum teacher-object error8.77cm. All failed attempts retained; no frozen-target verification performed because no candidate passes.

Unchanged-prefix GPU replays differ from archived iter10 (max generalized-coordinate difference .0104–.0191; checkpoint object differences millimetric). Exact target prefix is guaranteed; exact state prefix is not. Two passing complete frozen-target replays remain mandatory.

## Concrete next experiment (not executed)
Retain release01 prefix and stationary release window. Change thumb CMC abduction target (DoF6) by +0.3rad and CMC flexion (DoF7) by -0.3rad over1031–1090, with release01's other finger targets and fixed wrist orientation; hold through1110. Inspect whether thumb_ip/palm contacts actually clear before any translation. Those directions are an untested opening hypothesis, not verified clearance. If still hooked, inspect joint-axis rendered poses before another edit. Do not repeat +y closed-grasp translation: +6cm desired-y moved actual wrist only~2cm and pitcher backward. After a contact-clear release exists, correct the placement-y deficit in the still-airborne return phase, not by dragging a supported pitcher.

Focused tests:18 passed. Rendered release01 frames1030/1090/1170; imagery retained under outputs/u1_placement_v1/.

## 2026-09-20 live workspace correction
- Added native Warp U1 interactive target editor, following egg viewer/session/inbox pattern. No new offline trial. Uses release01 selectable target, full frame0 replay to1030, every native device array checkpointed plus target cursor and edits. Contacts explicitly CPU geometry witnesses; teacher relative transform displayed as evidence only.
- GUI launched DISPLAY=:1 in feat-uniform-direct-parent-repair-gui managed tmux pane %59; output outputs/u1_live_A_row035; log /tmp/u1-live-A-row035.log. Focused parser/ramp/checkpoint and prior placement tests:12 passed.
- Final persistent launch: feat-uniform-direct-parent-repair-gui:live, DISPLAY=:1, outputs/u1_live_A_row035_ready, /tmp/u1-live-A-row035-ready.log. Frame0→1030 completed. Both X windows confirmed. Saved live checkpoint via command inbox.
- All155 device arrays restore exactly. Repeated12-frame GPU continuation has max qpos difference1.430511474609375e-6 and qvel6.154179573059082e-5; fails chosen1e-6 pose bound, passes1e-4 velocity bound. Keep failure visible, do not claim deterministic continuation verified. No repaired trajectory. Code commit d5a9db6,12 focused tests passed.

## 2026-09-20T15:30:00+08:00 Sept15 row82 high-precision pitcher source
- User identified `/mnt/nas-222-project/mocap_v2/lance_datasets/dexgem_vla_demo_cma2lance_20260915_030735.lance` as higher precision, especially pitcher.
- Dataset v4 has three action006 rows; row82 UUID f32f8c68-84f1-4c67-8179-3b35d9f93e52 is the cleanest: 120Hz,2196 frames,max position jump4.74mm,max joint rotation jump3.26deg,no wrist freeze. The other two include119Hz and one has9.27cm/26.9deg jumps.
- Row82 source pitcher lifts16.51cm and returns within5.15mm translation /5.29deg rotation. Contact-geometry reconstruction at recorded poses shows the desired topology: index/middle/ring enter at frame252, pinky by261, thumb contact only begins318-323; during release thumb leaves1172, four fingers withdraw through1241, then all contact clears.
- This source should replace historical A_row035 as the pitcher teacher if native U1 follow-hand replay passes. Old A_row035 interactive workspace remains diagnostic negative evidence.
- Initial raw replay launch failed before physics because the legacy tool regenerated a manifest against its worktree-default asset root. A bounded executor is adding explicit pinned asset-root/manifest binding and running row82 U1 replay; no silent fallback allowed.

## 2026-09-20T15:31:32.223783+08:00 explicit profile / raw row82 checkpoint
- Scoped request: require explicit pinned asset root/manifest in replay_capture_no_policy; replay only Sept15 v4 raw row82, no render or repair.
- Removed manifest regeneration and deferred physical imports until explicit SHA/operator/commit validation and binding. Nine focused profile/CLI tests pass.
- outputs/sept15_row82_u1_replay_v2 contains manifest, summary, trace and REPORT. Confirmed pyramidal/impratio1, 120/480Hz; 2196 frames. Source/actual max lift 0.165055/0.022814m; longest free grip 0.241667s; source-peak error 0.198432m; final error 0.387811m; wrist RMSE 0.058476m. Raw target does not achieve intended pitcher lift. No NAS writes.

## 2026-09-20T07:42:33.960948+00:00 raw row82 workspace launch
# Row82 interactive U1 handoff

Paused at frame230 after physical replay from prestep frame0; no hand contact in live or source geometry. No target edits, no repaired-row claim. Source v4 row82 UUID f32f8c68-84f1-4c67-8179-3b35d9f93e52; 2196 frames, common grounding shift 0.002811035589636718 m.

- DISPLAY=:1; ordinary tmux `feat-uniform-direct-parent-repair-row82-live:live`; PID 518945.
- Attach: `tmux attach -t feat-uniform-direct-parent-repair-row82-live`
- Capture: `tmux capture-pane -t feat-uniform-direct-parent-repair-row82-live:live -p`
- Stop only this workspace: `tmux kill-session -t feat-uniform-direct-parent-repair-row82-live`
- Exact launch command: `outputs/u1_row82_launch/launch.sh`; log: `outputs/u1_row82_launch/workspace.log`.
- Workspace/state: `outputs/u1_live_row82/state.json`; identity: `source.json`.
- Checkpoint: `outputs/u1_live_row82/checkpoints/20260920T074116093396Z/` (`state.npz`, `workspace.json`).
- Validation: `restore_validation.json`: all155 arrays restored exactly; two12-frame continuations max qpos delta 5.80592995902407e-09, qvel 6.764894351363182e-07; bounds1e-6 and1e-4 passed. Cursor returned to230. This tests230–242 only.
- Both row82 Tk and MuJoCo windows confirmed on X display. Transform/contact telemetry includes aligned source frame230 as display-only evidence. CPU contact geometry is not a force measurement.
- Next: human/Astra local wrist/finger offsets around230–270; four non-thumb fingers through handle, thumb outside. No automated search performed.
- Legacy raw replay indexes state0 after an interval; interactive frame0 remains prestep and target[t] advances state[t-1] to state[t]. Legacy trace metrics are not exact workspace prefix comparisons.

Controls/commands: `docs/u1_raw_workspace.md`. Restore uses latest in-memory saved checkpoint; disk state is preserved but cross-process loading is not provided.

Focused adapter/profile/session validation:21 tests passed. NAS and Lance unchanged.

## 2026-09-20T15:42:00+08:00 row82 source replacement and interactive evidence
- Sept15 source v4 row82 UUID f32f8c68... selected as clean pitcher teacher:120Hz,2196 frames, source lift16.51cm, return5.2mm/5.3deg, no wrist freeze, max position jump4.74mm.
- Recorded geometry has correct topology: index/middle/ring enter frame252, pinky261, thumb only318-323; release removes thumb1172 then four fingers through1241.
- Explicit-pinned native U1 raw follow replay (commit e1bf28e) lifts only2.28cm; raw source qpos is a state backbone, not an adequate servo target.
- Global wrist lead4/8/12/16 worsened grasp (max lift2.27/0.58/0.33/0.21cm). Whole-trajectory inverse-dynamics projection also failed (max lift0.58cm) because object/contact generalized forces cannot be represented by hand-only target projection.
- Raw row82 interactive workspace commit d2129f8 launched DISPLAY:1 at frame230; all155 buffers restore exactly; qpos/qvel continuation deltas5.81e-9/6.76e-7. Live/source aligned arrival indexing.
- Manual overlapping target offsets are unstable and can accumulate. Source contacts and near-view imagery show live fingers remain outside handle while source inserts them. Needed mechanism is a workspace recorded-follow controller: at each120Hz frame, use source qpos plus bounded correction from current source-live error, record the resulting absolute target, and permit local contact-phase overrides. Exported target is static/self-contained and must replay under U1 with no runtime feedback.

## 2026-09-20T08:38:45.898750+00:00 terminal-first god-mode clarification
- User clarified god mode means full simulator observability and checkpoint rollback/retry at a failed frame, NOT a requirement to overwrite object states throughout an accepted trajectory. Terminal output should replace repeated GUI/file switching.
- Added observation-only tools/watch_u1_repair.py attached to the existing u1_live_row82_best process. No restart, physics, or target edit performed by observer. It labels geometry-only contacts (not force), source-relative pose discrepancy, servo ctrl-q difference, object motion/tilt, and command errors/acknowledgments.
- Current frame320 has all four nonthumb groups plus thumb, yet pitcher tilt14.45deg and relative source discrepancy22.72mm/17.89deg. Contact count is not evidence of stable grasp; previous descriptions of this state as successful were overstated.
- Hypothesis not yet tested: fixed-state hold trials conflate wrist load sag with relative slip. 1.28kg at100N/m implies125.6mm static wrist target lead if fully load-bearing; previous 40-60mm compensation caps could exclude required support. Do not interpret those tests as proof that U1 cannot grasp.

## 2026-09-20T17:48:54.355208+08:00 prior-iteration evidence and current correction
- Corrected local repair CCD35 to formal CCD16 and single-world capacity1024/256/4096 (dad6c5f). Old CCD35 checkpoints/diagnostics are not fully matched-contract evidence. Native-force/FP-tip telemetry remains uncommitted in this worker. Native mass1.28kg, wrist-kp100.
- Static source500 load comparison: analytical weight target compensation suppresses wrist sag; this is planted-state diagnosis, not acquisition acceptance. See outputs/row82_load_separation_ccd16_v1.
- From-frame0 damping+weight target trial lifted14.33cm but failed full return (outputs/row82_damping_load_ccd16/damping_load). Three-centimetre whole-hand insertion+delayed thumb failed: early geometry blocked and moved object.
- Geometry proposal gives four tip y=-20..-26mm, max hand-object penetration0.741mm. Straight interpolation of pose/approach gets blocked. Holding the insertion shape outside the aperture and translating along its normal improves acquisition.
- Corridor lift_v2 peaks6.62cm then drops. lift_v3_pinch peaks7.12cm in planned8cm test, maintains upward hand force~12.5N but slips/rotates relative to hand. No full accepted trajectory. See outputs/row82_insertion_geometry_v1.
- User corrected task framing: intended pouring overturns the pitcher. World tilt alone must not be a failure gate. User requests thumb/index pad contact, not mere proximity.
- Direct CPU geometry audit on saved native v3 qpos: actual480 thumb_ip/index_dip surface gap18.621mm, source500 gap3.803mm. No thumb/index native geometry contact at these inspected actual frames. Actual720 relative object-to-hand rotation differs74.75deg from480 while wrist Euler changes~1deg. Target-finger geometry still has9.706mm surface gap at480 even if target were perfectly reached.
- Prediction: a closure aimed at a signed surface gap nearzero on the far side of the handle (local y<-20mm), rather than FK-tip distance across the handle bar, should close the escape path. Compare from the same complete420 checkpoint with unchanged U1 and support compensation; native thumb/index contact force and relative pose drift are deciding evidence.

## 2026-09-20T18:50:14.041468+08:00 closed-grip reference-wrist candidate passes two fresh U1 replays
- User confirmed target geometry image and clarified: wrist lag is acceptable; once grasped, use existing wrist reference for pouring.
- Surface-closure geometry search reached0.22mm gap near the source grasp. Native synthetic approach closed fingertips in empty space, not around the handle;8N target preload produced~6N tip-pair force but no lift. Correcting its wrist tracking error did not repair blocked insertion. These failed artifacts remain in outputs/row82_surface_closure_v1.
- Returned to the stronger source-damping/load replay; at400 real thumb/index surface gap~1.96mm and hand already bears~11N. Holding acquired finger actuator targets plus an8N Jacobian-derived closing target vector preserved grasp during the original wrist motion. Latch-only and latch+loop both lift15cm and return upright; loop-preload adds actual distal contact. Before task registration the spout misses bowl and landing error is11cm.
- Applied smooth recorded XYZ task registration: pour(-35,+100,0)mm, landing(+95,+45,0)mm with explicit damping lead; wrist rotations/timing and all native physics parameters unchanged. No live object writes.
- aligned01 and fresh independent01 use identical target-array SHA 81faf6e05c4173de79e5007c61a310506ad8c389be4458ebc2242f072a580e39. Maximum lift0.152960/0.152924m; final position errors0.004713/0.004053m; final tilt0.453deg; source-relative full orientation errors13.431/13.726deg. All established physical gates and extra tip-contact/rim-proxy/settling/control-only gates pass. All-frame pour proxy maximum radius56.5/56.1mm within77.1mm conservative bowl-radius proxy; positive native pair force in sampled pour frames.
- Both runs retain native numerical variability. Object orientation differs0.044deg at1104,0.124deg at1200,1.346deg at1241, then27.06deg finally; final positions differ7.76mm, each within4.71mm of source. Divergence is localized to withdrawal/settling. Do not claim deterministic state replay or broad robustness.
- Final evidence: outputs/row82_reference_grip_hold_v1/validation.json, aligned01/, independent01/, independent_storyboard.png, candidate_manifest.json. New candidate UUID c78563eb-0dda-5845-a3da-50bfb1d53a19. No Lance/NAS publication or changes.
- Committed native-force telemetry05f2deb; committed deterministic recipe/builder/docs184feb6. Builder replayed recipe to exact target SHA;3 focused builder tests pass. Other41 parents remain unmodified.

## 2026-09-20T18:52:48.505661+08:00 accepted-target live workspace replacement
- New native CCD16 viewer in manorl-uniform-direct-repair:accepted-live, DISPLAY:1, outputs/u1_live_row82_accepted, PID1250422; accepted target aligned01/target.npy. Paused at0, no edits. Observer in accepted-telemetry.
- Full155-array restore at0 passes12-frame continuation: qpos5.9604645e-8, qvel3.7774444e-6. State publication and process liveness verified.
- Retired only verified obsolete CCD35 GUI685719 after replacement was ready; its workspace/checkpoints were preserved. No existing dataset changes.

## 2026-09-20T19:21:37.289332+08:00 four-family single-trajectory phase
- User asks to do the other four actions, first one trajectory per action, all in the same setting. Scope interpreted from current historical five-family contract:003/005/007/009 plus already completed006. No augmentation now.
- Working tree clean except local task memory; existing water-pitcher candidate preserved. Raw Sept15 v4 catalog contains29 action003,10 action005,2 action007 (rows64/68),1 action009 (row65); all listed operatorcheyingtong. Audit base vs generated rows and motion quality before choosing sources. Disk free5.8GiB.

## 2026-09-20T19:48:41.295473+08:00 four-family first passes and load-phase correction
- Selected rawSept15 rows71/126/64/65; all physics/options/hand-actuator/passive/mass fingerprints equal water U1: c6db552f105d4de493880a52518772dbacad9496b105d2f4e43e87a907e692cd. Raw initial passes fail grasp. At source lift start, bowl71/64 only index geometry contacts and bowl65 has none; raw joint tracks are not a physical grasp.
- Existing physical teacher poses/finger targets were tried as reference-only data for B_row099/A_row020/B_row200/A_row050, with all NEW simulation still U1 and no PID. No model fields changed; finger targets are constant within each old four-substep interval. Bottle lifts13.3cm but loses grip; other cases incomplete.
- Anomaly resolved: movement-from-initial used physical settling to trigger payload compensation (B099 frame7, B200 frame5) rather than real capture lift(frame236/129). Full35mm empty-hand upward offset prevented approach contact. Replaced onset with first captured reference upward lift>5mm, no preload before lift; all other settings unchanged. Predict restoration of acquisition/contact and larger lift. Results pending in lift_timing_* logs.

## 2026-09-20T21:13:33.628979+08:00 visible bowl workspace after user objection
- Previous new trials were real native U1 headless runs, but the visible window was not synchronized. Old water process1250422 is now absent; its state.json is stale at1485/running=true. Exit cause was not established; do not call that file a live observation.
- Current bowl source is existing formal U1 row5 bowl_03_052, with verified same setting hashc6db552f.... Its base and held-grip attempts still slip/drop near300. Prior raw/teacher attempts remain incomplete for all four new actions.
- User-directed thumb-only6mm-toward-middle target atframe216 was fit by FK IK and replayed. Wrist and other finger targets are byte-identical to base; fitted motionerror5.75micrometres is only kinematic evidence. Real replay lift8.97cm but still slips; NOT accepted. Formal thumb IP actual state exceeds soft joint limit by0.00081rad; only IK starting guess was bounded. Untouched float32-origin control values were preserved exactly rather than reclipped globally.
- Started native live bowl GUI PID1686600 in manorl-uniform-direct-repair:bowl-live, DISPLAY:1; launcher outputs/four_action_single_v1/launch_bowl_live.py; workspace outputs/four_action_single_v1/live_bowl_thumb_middle. Full initial155-array checkpoint test passes(qpos2.98e-8,qvel2.38e-6).
- Actual GUI advanced0→192, saved complete checkpoint192, then visibly played192→640 with current thumb-middle target; command acknowledgments and telemetry confirm progression. Restored to192 and verified current process/state freshness. Observer is :bowl-telemetry.
- At live192, native object forces by link: ring(-11.46,-2.55,+7.54)N; index(-4.43,-2.88,+4.54)N; middle(-0.71,-0.27,+1.21)N; thumb(+16.64,+5.44,-9.83)N. Opposing forces sum to carrying weight; contacts are not inferred from shape alone.
- Counterfactual geometry inspection of thumb-middle trial: middle actual gap grows0.49mm at264 to7.13mm at300, whereas keeping the bowl rigidly in the216 hand frame gives penetration~2.12/1.72mm. Middle is closing, not opening away; object slip causes the lost contact.
- Proposed inverse-static grasp generator and derived009 loader exist as scratch, but were NOT executed. Current priority is visible local thumb→middle pinch, not another hidden batch. Water passing target/recipe and all original Lances preserved.

## 2026-09-21T00:20:47.036062+08:00 correction: reject extra-motion bowl hybrids; local-only003 passes
- User correctly rejected row607 whole-grasp-prefix+120-frame handoff trajectories: they insert an extra pickup/pause before the original task. All such003/007/009 candidates and the three-action viewer are do-not-publish; prior local physical pass did not satisfy task semantics.
- Correct contract: original timeline and wrist target; only a bounded grasp-window wrist micro-adjust and finger grip correction. After the window wrist is byte-identical to its original reference.
- First local-only attempt copied old-hand finger angles and failed. Contact evidence showed original formal003 already has thumb/index/middle/ring; replacing middle/pinky removed middle.
- Final corrected003 v4 changes wrist six targets only frames86-192 and thumb targets86-318; index/middle/ring/pinky targets are byte-identical for all frames; wrist is byte-identical after192. Wrist peak deltaXYZ[-8.36,-5.26,-9.80]mm and Euler[-0.24,+1.91,+0.70]deg; thumb delta[-.0339,+.0315,+.0559,-.0337,+.0325,+.0112]rad.
- Visible v4 replay ends upright/released/world-supported with5.33cm final-position error. Independent frame0 fixed-target replay passes every established gate: lift10.786cm vs ref15.562cm, airborne multifinger1.0, final error2.529cm, tilt0.388deg, longest contact-lift0.967s. Setting hashc6db552f..., no object forcing.
- Applying the same world-space wrist correction to007/009 failed; those initial wrist/object poses differ, so each needs its own local contact-space wrist correction. Do not display or publish prior whole-prefix007/009 successes.

## 2026-09-21T00:41:42.628869+08:00 semantics-correct local bowl repairs003/007/009
- User rejected earlier row607 whole-prefix hybrids because they inserted an extra pickup/pause before original task motion. Those candidates/viewer were marked rejected and closed.
- Correct mechanism: preserve original frame count/time axis; local grasp-window wrist micro-adjust+finger grip; wrist target byte-identical after rejoin. Use row607 only as contact-topology evidence, never its later wrist/object motion.
- Correct003 v4 edits wrist86–192 and thumb86–318 only; index/middle/ring/pinky targets byte-identical. Independent U1 replay passes: lift0.10786m, airborne multifinger1.0, world support/release, final2.529cm, tilt0.388deg. Targetfbffed86....
- Correct007 v4: action-specific wrist solve, totalZ-10mm; all fingers follow successful003 grip, four fingers before thumb; wrist byte-identical after201. Visible and independent U1 pass: lift0.11434m, airborne multifinger1.0, final cuboid1 support/release,4.950cm, tilt0.254deg. Target808a2248.... User watched and approved.
- Correct009 v2 grasp passed lift/hold but independent full orientation16.40deg. +1deg axis probes switched contact basin and worsened20–22deg; -1deg axis probes all passed. Chose smallest best-margin wrist-Z -1deg local correction. Independent target cd20ce72... passes: lift0.29246m, airborne multifinger1.0, final held4rays/no support,3.686cm, full orientation14.336deg, tilt10.885deg.
- Canonical local manifest outputs/four_action_single_v1/correct_bowl_single_candidates.json; no Lance/NAS writes, no augmentation/H200 robustness claim.

## 2026-09-21T02:09:41.561942+08:00 unattended final scope and current evidence
- User approved unattended continuation/delegation. Priority: faithful strong-pinch005, then006 release audit/repair, then160/action initial hand pose augmentation for003/005/006/007/009 under one U1.
- Semantics-correct bowl parents are locally passing and user-reviewed:003 target fbffed86...;007 target808a2248...;009 targetcd20ce72.... All preserve original task timeline, edit only grasp window, restore original wrist tail, and passed fresh frame0 U1 gates.
- 005 deep-inversion remains unresolved. A020/A030 baselines lift but slip around70–100deg. Tried row847 five-finger grasp compressed into approach, 50/100/200% measured servo-deflection preload, moving teacher finger correction, middle/pinky recontact, 1cm dynamic-independent palm approach, six-point palm+five-finger static enclosure, action-specific five-point contact solve, four alternate row847-style grasps, five alternate A-row references, and one task-space ILC. All deep-inversion attempts fall/side-rest; static force increases do not preserve contact past inversion.
- Row847 shallow pick/place was mapped to current Cheyingtong/U1 and independently passed: lift18.85cm, airborne multifinger0.9815, final error0.54cm, tilt1.13deg, frozen control exact. User requires original deep-inversion005 semantics, so this remains diagnostic/alternate negative evidence. GUI closed to free GPU.
- Current hypothesis before delegated review: deep inversion needs geometry/dynamic object-frame registration or interlock, not another scalar force increase. Do not rerun equivalent static-preload trials without new mechanism.
- Dispatched authorized fresh-context parallel read-only agents: theorist for005 causal model/next experiment; operator for006 release fresh audit; critic for5x160 augmentation contract. Async id db3e78be-d3eb-463a-99a4-63163b2f90bf.

## 2026-09-21T00:00:59.462542+00:00 bounded 5x160 adapter checkpoint
- Implemented immutable bundle registry, seeded balanced plan/reserves, persistent native reset/runtime, digest-bound ledgers and independent-process replay path. New user scope labels005 row847 alternate pick/place explicitly. No production/Lance writes.
- Focused campaign and Session tests:13 passed. Ten tmux pilots stored in outputs/u1_5x160_pilot_v1.003/005/007/009 zero/extreme pass;006 both fail final source orientation<15deg only. Direct canonical fresh02 release-v6 inspection:59.730deg; zero59.281deg/extreme58.758deg. Release-v6 is upright but terminal yaw violates retained orientation gate. No rerolls.
- Outstanding: second-process physical qualification,480Hz/CCD capacity instrumentation; no exporter. Source docs preserve precise boundary.

## 2026-09-21T10:12:17.053276+08:00 final006 and exact800 distributed production
- Fresh audit disproved old row82 release robustness: accepted target toppled2/2 fresh runs. Iterative fixes localized the mechanism to supported yaw correction plus collision-safe withdrawal.
- Final release-v13 preserves prefix<1130, registers yaw while supported, moves thumb clear first(frame1569 last contact), then holds thumb open and reverses same-U1 acquisition deltas for nonthumb/wrist extraction(nonthumb last1726-1743). Two fresh runs pass max release tilt6.15/5.87deg, final tilt0.45deg, full orientation8.39/10.03deg, final position7.09/6.86mm, settled/clear. Target SHA f16c7c23..., UUID010689af....
- Final parent registryv3/planv3 passes zero+extreme pilots for all five actions under every480Hz substep guards(max42 contacts/196 constraints) and independent new-process frozen-control repeats.
- First production attempt selected77 action003 slots but was aborted and isolated because native contact basis matrices were not persisted; exact force export would be impossible. Production runner commitc3775ac adds native contact frame/friction/dim/EFC/world capture; structural smoke passes.
- Restarted exact campaign NAS staging `.manorl_u1_5x160_20260921_staging`; existing1923 unchanged. Sequential scalar run was user-flagged slow. Historical evidence: local32-world1344 runs≈14min; formal8-world batches; Server1 H200 batch32≈1449episodes/hour on a different prefix workload.
- Accelerated safely at whole-action boundary: local003/009; Server1 GPU0/2/3 owns005/006/007. Server1 mandatory Unison restored and healthy; exact edcd2a3 source deployed at canonical absolute path; registry/runtime load passed. Process-private unshare bind maps plural NAS to singular canonical path. Cross-host plan regeneration differs1e-18 due CPU floating normalization, so SHA-bound wrapper72a3d8bc uses exact signed plan JSON and fixes `__file__` ledger identity to canonical/home; identities verified equal, no physics/candidate change.
- Exporter commits edcd2a3/0c87aaa/884fc45: exact800 fail-closed quotas/hashes/PIDs, second-pass states/controls, native basis force conversion, full trace extension/readback, one active movement interval/row, and visualization layout. Event-driven coordinator/finalizer waits five quotas, exports/validates/layouts, then atomically publishes a new NAS directory.

## 2026-09-21T11:36:00+08:00 — live semantic audit invalidates row847 action005
- User watched selected003/005/006/007/009 examples. Initial qpos viewers were replaced by genuine live native-U1 viewers: fixed absolute target, prestep frame0, four480Hz Warp substeps per120Hz frame. Live setting hash was `c6db552f...`.
- User correctly rejected action005 row847: it performs strong-pinch pickup/place but omits the required move-over-bowl deep pour/inversion and return. This is a category error, not a weak metric. Existing160 row847 children are quarantined; `BLOCKED_INVALID_ACTION005.json` written to campaign staging; exact800 finalizer/completion publication processes stopped. Other action evidence preserved.
- Correct A030 historical state reference reaches139.44deg orientation change near frame581. It is historical PID/U2 physical evidence; its qpos movie was explicitly labeled semantic reference only. Genuine live U1 replay of the A030 full target lifts/moves but slips/falls during tilt, matching prior diagnostics.
- Historical lineage audit: old155 action005 rows derive from only six parents: A01932, A0206, A02232, A02432, A02922, A03031. Because start offsets taper to zero before contact, these are six grasp topologies, not155. All six have U1 reports; A030 is strongest (airborne multifinger0.687, longest contact-lift2.49s) but fails upright/final placement. Additional donor rows767/847/866/946/1026 were also screened.
- Updated commitment: search U1-stable strong pinch at critical A030 tilt poses first; only then fit the complete A030 timeline, require two frozen-control frame0 replays, and regenerate005×160. Release-v13 remains the authoritative006 mechanism.

## 2026-09-21T11:58:00+08:00 — action005-only scope and wrench-search contract
- User narrowed active work to005 only. Other actions, augmentation, export and publication are paused; current evidence/artifacts remain preserved.
- Fresh theorist audit corrected the failure boundary: clean slow-inversion v2 first reaches5mm relative drift near51deg actual bottle tilt, while hand vertical force remains2.916N versus2.943N bottle weight. By67deg it reaches20.6deg relative rotation and index/middle/ring friction-pyramid utilisation is1.0. Total vertical support remains adequate while rotational retention fails.
- The prior shoulder planted diagnostic begins with only middle contact and cannot refute a realized five-finger shoulder/neck grasp. Do not cite it as U1 infeasibility evidence.
- Committed experiment contract: at exact A030 orientations near50/70/90/110/122.5deg, test at most32 deterministic row847/A030-derived geometries under unchanged U1. Measure every480Hz native contact wrench/basis, COM net force/torque, friction utilisation, actuator saturation, support and hand-relative drift. Require five force-bearing fingers and strict final-second drift/rate/wrench reserve gates. Static planting is diagnostic only; any winner must still be acquired from frame0 and complete the full A030 action twice with frozen controls.
- Authorized bounded executor run `6874d5ce-1aa2-44c1-8849-d1ba4f7a09ff` to implement/run this search only; no unrelated work.

## 2026-09-21T12:55:00+08:00 — visible U1 web-grip search and gravity-separated diagnosis
- Implemented v1 bounded critical-orientation search in commit45197ee;6 focused tests pass.32 deterministic geometries×five exact A030 orientations recorded native contact/wrench/reserve evidence. No candidate passed. Candidate29 most consistently established five fingers; candidate23 had longer survival but invalid17mm initial finger penetration.
- User required visible simulator. `DISPLAY=:1` now continuously re-integrates the current diagnostic target under native U1; no qpos movie. Event-driven finalizers automatically replace the window with each search's ranked best candidate.
- Geometry decision: model has no dedicated web geom. Web witness is palm_collision0..1mm plus index_mcp0..3mm; thumb_cmc is not forced near the bottle. Candidate29 has palm0mm/index-MCP1.615mm but deep planted thumb/index intersections.
- Deterministic least-squares fit at exact A030122.546deg solved all five distal signed distances to≈-0.400mm, palm0mm,index-MCP2.551mm,max hand-bottle penetration0.4001mm. Thus shallow web+five-contact geometry is kinematically feasible.
- User proposed removing bottle gravity while pinching, then restoring it. Diagnostic v2 applies an upward external force equal to bottle weight for60frames while closing, ramps it to zero over30frames, then scores240 normal-gravity frames. External force is diagnostic only and forbidden in the final frame0 replay.
- V2 result: all five fingers carry load at the end of countergravity unload. One frame after full gravity returns, pinky contact disappears; axial span collapses≈86→62mm; ring later disappears and relative rotation exceeds20deg. Best candidate15 survives0.192s normal gravity. This separates contact establishment from load retention and localizes missing torque to lower pinky/ring support.
- Started v3 sixteen-target search: base thumb/index/middle closure0.12rad; ring0.12/0.18rad; pinky0.15/0.21/0.27/0.33rad; pinky abduction-0.04/-0.08rad. tmux `manorl-uniform-direct-repair:005-pinky-v3`; output `outputs/action005_pinky_grip_v3`; completion auto-displays best candidate.

## 2026-09-21T13:07:26.458938+08:00 — user-authorized half-mass full005 replay
- User requested bottle half current mass and full action playback. Intervention: private runtime body_mass0.30→0.15kg; body_inertia and every other native field unchanged; A030 alternate_screen1321-frame absolute target unchanged. Normal gravity from original frame0, no applied external forces, no object state writes during episodes.
- Prior static searches are paused; v6 already finished, its display process is being replaced. Prediction: reduced payload force may postpone grip loss; unchanged full target may still rotate/slip or over-lift because its original load lead remains fixed. Full native integration and saved episode determine the outcome.

## 2026-09-21T13:10:46.376370+08:00 — half-mass full replay observed
- Live process334776 on DISPLAY=:1 completed two1321-frame full A030 loops, from original frame0, mass0.15kg. Source target unchanged, maximum applied-target error1.191e-7; external force arrays zero throughout. All native model fields except bottle body_mass asserted unchanged, including rotational inertia.
- Both loops failed final upright/place gates: lift12.674/12.671cm, airborne multifinger fraction0.8445, continuous contact-lift3.50s, final tilt89.653/89.646deg, final position error14.724/14.764cm. Nominal archived baseline:10.464cm lift,0.6866 airborne fraction,2.492s contact-lift,29.918cm final position error. Reduced mass improves retention but does not complete005.
- Artifacts: outputs/action005_half_mass_full_v1/{manifest.json,diagnostic_model.mjb,loop_000,loop_001,latest_result.json,state.json,display_snapshot.png}; runtime display keeps looping, only first two traces retained. X window verified; no source asset/campaign mutation. ShellGate broker socket became unavailable after detached launch; live diagnostic child remained healthy and was checked via normal local shell.

## 2026-09-21T13:21:35.544717+08:00 — pause005 and start four-action large-pose augmentation
- Paused verified005 half-mass viewer PID334776 with SIGSTOP; observed stateT, last frame744/loop39 on DISPLAY=:1.005 repair and static searches are not continued.
- User redefined augmentation:003/006/007/009 only,160 each, initial hand translations5/10/15cm and wrist30deg; fixed nominal U1. Prior small-perturbation children remain evidence only.
- Read actual history: start_position_campaign.md/start_augmentation.py/run_start_augmentation.py use0.1s precontact guard bounded by source movement timing; approach_prefix.py::_discrete_c1_prefix instead prepends a quintic/4cm-arc approach, sets last prefix sample2*x0-x1, then appends all original reference frames. Duration uses distance/speed and orientation/speed. +15 anchors near/retreat, not the approach splice. Historical doc prose mislabels SciPy Euler conventions; use numerical rotation composition/tests, not that label.
- Four current parent first scene contacts:003=86,006=237,007=84,009=196 at120Hz. Old merge indices74/225/72/184 were tuned to small perturbations, not evidence for15cm/30deg convergence. Commitment: use distance/angle-sized added approach, no splice-state reset; test real settling and action gates before production.

## 2026-09-21T14:15:32.812347+08:00 — 003 large-pose pilot passes twice
- Main-agent pilot outputs/pilot_003_largepose.py: prepend90-frame quintic offset prefix (5cm +++ octant, +30deg roll) to accepted003 parent, byte-exact suffix, actual qpos0[:6] offset, objects/fingers/qvel unchanged, no state reset at splice.
- First pass accepted: lift10.70cm, final error2.43cm, tilt0.39deg; controls_canonical, initial_pose/velocity, executed_suffix_exact all true.
- Second independent-process frozen-control replay accepted: lift10.71cm, final error2.56cm, tilt0.39deg; frozen_exact true. PID differs from first pass.
- Confirms historical prepend approach transfers to the new 15cm/30deg scale without contact-minus15 forcing.

## 2026-09-21T07:07:21.229967+00:00 — reopened005 reference-contact fit; failed focused replay
- Added tools/fit_action005_reference_contacts.py and3 focused tests;9 tests pass with existing critical-grip tests. Exact commands are in outputs/action005_reference_contacts_v1/REPORT.md. No unrelated largepose edits staged.
- Prediction: maintaining lower ring/pinky contacts with>=70mm span through reference50–90deg should arrest relative rotation. One offline fixed-surface-patch fit retains original1321-frame A030, bounds wrist15mm/8deg and finger pose corrections1rad plus measured donor servo deflection<=0.3rad. Original wrist lead retained; target prefix<225 and release tail>=1050 exact. Target is C1-interpolated knot corrections, but total finger target maximum step0.119rad: continuity does not establish physical smoothness.
- Direct donor-relative wrist mapping was diagnostic math only, never applied:26.3deg correction needed at reference76.5deg, >50deg during early approach. It is incompatible with micro-registration.
- Fixed donor patches cannot be matched by this fit: atframe425 pinky18.75mm and ring12.94mm residual with bounds active. This is failed local fit evidence, not global kinematic infeasibility.
- Native nominal-U1 replay from original frame0 to470 completed: setting c6db552f105d4de493880a52518772dbacad9496b105d2f4e43e87a907e692cd; four480Hz substeps; native absolute controls; no applied forces or post-frame0 state assignments. All finite, max applied-target difference1.186e-7. Every-substep maxima25contacts/55broadphase pairs/128constraints; broadphase count guard256 conservatively bounds CCD work, not an exact CCD occupancy measurement.
- Atframe405 reference42.41deg/actual51.86deg, ring0.747N and pinky0.794N, span92.64mm but relative rotation8.60deg sinceframe380. At410 reference44.99deg/actual55.12deg, only thumb1.566N remains and span=0; relative rotation10.80deg. Atfirst reference>50deg frame418 lower contacts absent, relative rotation18.41deg. Peak interval relative rotation72.79deg. Focused mechanism FAILED; no full episode, second replay or video generated.
- New discriminating geometry audit discards fixed hand-surface patches and allows closest-surface sliding with bottle-local ordering/axial bands. At20mm/15deg bounds: reference50deg fits five gaps~-0.4mm with16.41mm/11.48deg correction;90deg fits five gaps~-0.4mm, max penetration0.900mm with16.64mm/9.65deg correction.70deg single-start local solve stalls with pinky3.879mm gap and13.774mm axial error. This falsifies fixed-patch residuals as a topology infeasibility certificate. Surface fits have not been dynamically tested or connected continuously; normals/opposition under load unverified. No blind dynamic sweep.
- Preserved artifacts: outputs/action005_reference_contacts_v1/{target.npy,fit.npz,fit.json,contract.json,trace.npz,telemetry.jsonl.gz,result.json,surface_feasibility.json,surface_feasibility_qpos.npy,REPORT.md}; outputs/audit_action005_surface_feasibility.py; fit/native/surface log siblings.

## 2026-09-21T07:08:38.003077+00:00 —70deg neighbor-seed discriminator
- Explicit user continuation requested a bounded surface-topology audit. Re-solving70deg from the feasible50deg delta improves pinky gap3.879→0.460mm and axial error13.774→9.896mm; ring axial error5.140mm. Thumb/index/middle/ring/pinky signed gaps[-0.323,-0.619,-0.428,-0.571,+0.460]mm, max penetration0.619mm. Wrist17.420mm/11.175deg remains within bounds; largest normalized wrist component0.986, largest finger correction0.921rad.
- The required pinky contact/5mm axial-band condition still fails. Two local70deg fits are negative evidence, not proof that no continuous branch exists. No new target/physical sweep follows an unverified geometry. Remaining exact question: can a continuous closest-surface branch close the0.460mm pinky gap and recover4.896mm excess axial error while preserving opposed normals and the same reference bounds?
- Preserved outputs/action005_reference_contacts_v1/surface_feasibility_70_neighbor{.json,_qpos.npy}, outputs/audit_action005_surface_feasibility_70_neighbor.py and outputs/action005_reference_contacts_v1_surface70.log. Worker turn budget reached; stopping with localized unresolved geometry rather than asserting infeasibility.

## 2026-09-21T07:20:45.997390+00:00 — topology correction and single native discriminator
- Restored exact five005-owned files in new commit7ad67f9; did not rewrite ca9823a/37e3096 or touch unrelated largepose paths. Current disk task memory preserved.
- Removed donor axial/azimuth penalties.70deg nearest-seed correction closes every gap to~-0.400mm; span93.743mm, wrist17.195mm/10.993deg, opposed normals0.957–0.981. PCHIP continuation across4-frame reference knots; target remains1321frames, prefix<225/tail>=1050 exact.50–90deg span>=86.819mm; max penetration1.057mm.24 all-path geometry audit failures remain (including1.307mm gap near726); no claim full-path feasibility.
- Predicted ring/pinky bearing persists, span>=80mm and rotation acceleration ceases. Ran one frame0 native-U1 diagnostic in managed tmux feat-uniform-direct-parent-repair-005-surface; artifacts outputs/action005_reference_contacts_v2 and sibling fit/native/audit logs. Commands in docs/action005_reference_contact_diagnostic.md. No second physical trial.
- Native stopped418, first reference50deg sample: ring/pinky1.667/2.157N, span80.826mm, opposed normals0.927–0.980 survive. Middle first zero after380 at402, remainszero418 with actual gap+0.017mm; thumb9.499N and penetration1.656mm. Relative drift4.090deg, rate20.964deg/s, acceleration87.444deg/s². Friction utilizationmax0.704; static wrench reserve1.253. Hence maintained lower leverage improves but does not establish five-contact retention.
- Acquisition sequence also not demonstrated: recorded>=265 window thumb bearing270 precedes pinky301/ring311/middle333. Four-finger target ramp ordering did not enforce physical order.
- Runtime aggregator incorrectly zeroed span/opposition when middle missing. Inspected raw contact points; corrected function preserves available contacts, explicit missing fingers, unknown complete axial order. Raw result untouched; offline audit.json supersedes derived fields. Regression test added. No resimulation.
- All finite, fixed hashc6db552f..., CCD16, four480Hz substeps/native absolute targets; max applied error1.184e-7, no applied force/post-frame0 episode state write. Capacity maxima25/58/128. Forces sampled lastsubstep120Hz, capacity480Hz; broadphase bound not exact CCD occupancy.13 focused tests pass. No complete episode/repeat/video; localized blocker is realized acquisition/load distribution (early thumb, unloaded middle), not pinky axial position.

## 2026-09-21T07:22:47.133876+00:00 — authorized single balance follow-up



A single follow-up was explicitly authorized after v2: use actual frame418
geometry to solve only thumb/middle joint offsets toward−0.9/−0.4mm gaps,
with minimum-displacement regularization and±0.05rad limits. Apply offsets to
frozen controls with a smooth350–380 onset and existing grip-window taper.
Wrist/index/ring/pinky targets, prefix<350 and tail>=1050 are byte-identical
to v2. No sweep was performed. `balance.json` records the exact joint vector.

The corrected geometry predicted thumb/middle gaps
-0.900000/-0.400000mm, but native replay again
stopped418: middle0N, thumb9.212N, ring1.585N, pinky2.078N, span80.888mm.
Thumb penetration remained1.617mm; rotation acceleration81.232deg/s² versus
87.444 previously (only7.1% lower), drift4.059deg. Middle first zero after380
is frame410. A kinematic displacement applied as
a servo-target displacement mostly redistributes spring load instead of
realizing that displacement under the coupled grasp. The predicted middle
bearing was not established. This is the stopping boundary; no further trial.

Artifacts: `outputs/action005_reference_contacts_v3/` contains `balance.json`,
`target.npy`, `trace.npz`, `telemetry.jsonl.gz`, `result.json`, `audit.json`,
`contract.json`; its `fit.json` is inherited v2 geometry, not a measured v3 fit.
Run the same CLI with `--output outputs/action005_reference_contacts_v3
--balance-from outputs/action005_reference_contacts_v2` to construct/replay
(in a new output directory). Finite/canonical, no applied forces/state writes;
capacity25/58/128, control error1.184e−7. **Two complete passes: no; zero full
episodes.** Localized blocker remains middle unloading under thumb-dominated
load, not absent lower leverage.

Initial launch into prior tmux session failed because that session no longer existed; new managed session feat-uniform-direct-parent-repair-005-balance ran the sole v3 physics episode.

## 2026-09-21T15:39:50+08:00 — v4 reconstruction contract and prediction
- Historical v3 contains qpos/qvel/ctrl, not full native buffers. Supervisor approved one exact-target prefix reconstruction at current HEAD9890d25; unrelated descendant left intact. Predetermined tolerances: qpos1e-6,qvel1e-4,ctrl exact; every archived force within0.02N+1% and same0.2N bearing mask. Any mismatch stops before probes.
- Planned compact basis: thumb/middle normal and axial signed-distance gradients, central +/-0.004 and +/-0.002rad;8-frame continuation and last4-frame mean force/gap/wrench, quadratic relative-rotation acceleration. Predicted thumb unloading plus middle loading if smooth compliance dominates; scale/symmetry discrepancy>30% or condition>1000 rejects inversion. Raw contacts and exact native buffers preserved under v4.

## 2026-09-21T15:43:46+08:00 — reconstruction mismatch; stopped before probes
- One frame0 frozen-v3 prefix completed409. Controls bitwise; qpos max0.0008723437786102295 at409 coordinate33, qvel0.1016545295715332 at291 coordinate33. Both first exceed prescribed tolerance at29. Force masks differ266/267/269/273/274/288;40 force samples fail0.02N+1%.380 index3.47321859 versus3.40321302N.400/409 force errors0.0100861/0.00998116N pass.
- Full buffers380/400/409 saved but not qualified. Zero probes/inverse/candidate replay/full passes/repeats. No applied forces/post-frame0 writes during reconstruction. Numerical source not isolated.
- v4 artifacts and sibling_native.log preserved; tool renamed reconstruct_action005_native after failure, unexecuted probe code removed.20 focused tests pass. Docs include reproduction command. No unrelated changes.

## 2026-09-21T15:51:47+08:00 — fresh-baseline sensitivity intervention
- One NEW v3-control frame0 realization through418; checkpoints380/400/409 retained in process. Two unit joint-space signed-gap gradients (thumb and middle) are the minimum basis addressing the two load/depth objectives. ±0.004/±0.002rad,8-frame horizon, last4-frame mean forces/gaps/net-wrench and quadratic relative-rotation acceleration. Restore every native buffer before every branch, verify assignment exactly, and run paired zero continuations. Fixed normalized scale/symmetry error limit30%, condition1000. Expected positive opening reduces force/increases gap; unilateral middle loss can refute local smooth invertibility.
- No historical state matching; compiled model hash must match saved v4 and initial states/U1/target matchv3. Supervisor explicitly clarified baseline and acquisition gates in PROMPT.

## 2026-09-21T15:52:59+08:00 — operational interruption and authorized restart
- Managed foreground timeout then ShellGate off removed process near280 before checkpoints. Supervisor approved exactly one identical operational restart into v5/run_001; preserved partial manifest/model/log under interrupted_launch with no scientific conclusion.

## 2026-09-21T15:57:34+08:00 — fresh sensitivity rejected
- run_001 exit0, PID1013981;155 buffers restored,24 signed probes+6 zero branches. Baseline reproduces middle0 at418 with thumb9.208N/1.616mm, span80.865mm, drift4.038deg/alpha80.963deg/s². Restore max qpos1.79e-7/qvel3.20e-5.
- All sensitivity gates fail despite correct signs/full rank:380 angular75.7%,400 task-force53.9% with contact loss408 for+.004 not+.002,409 force77.5%/gap51.4% with mode switches. Raw contacts inspected; repeat noise much smaller. No inverse/target/transfer/full pass/repeat. Reference wrench not computed.27 scoped005 tests pass.

## 2026-09-21T16:39:48+08:00 — direct120 source audit prediction
- Read-only audit of all ten v4 captures under the pinned source adapter; compile right-only native U1 scenes and measure every-frame signed surface distances, object-local points/normals, geometric contact order and hand-object relative transform. Source qpos is evidence, not a physics rollout. Prediction: semantically strongest rows128/130 may still have incompatible recorded fingertip geometry, which would disqualify direct topology transfer before any replay. Full-pose relative motion localizes row131 identity/tracking concerns. No target edits or integration yet.

## 2026-09-21T16:46:44+08:00 — selection frozen before MJX baseline
- Completed all10 source and independent CPU collision audits. Zero airborne plausible five-contact frames in every row; zero five-contact deep interval fraction. CPU source geometry cannot certify bearing forces or physical acquisition order. mj_geomDistance zero with inconsistent witnesses produced false earlier touch estimates; collision/summary.json supersedes those initial fields.
- User relaxed geometry eligibility mid-run. Frozen best kinematic row130/version4 in selection.json; no donor or changed reference. Prediction: missing middle and shallow acquisition lead may prevent pickup despite correct kinematic inversion. One full MJX-Warp cuda:0 frame0 baseline using existing recorded-qpos velocity/friction/gravity-lead build. Native capacities480Hz and contacts/wrenches120Hz. No physical correction yet.

## 2026-09-21T16:48:34.245201 — row130 baseline fails acquisition
{"row": 130, "version": 4, "uuid": "95c72d14-d854-47af-8239-849cd9757465", "backend": "MJX-Warp cuda:0", "setting_sha256": "c6db552f105d4de493880a52518772dbacad9496b105d2f4e43e87a907e692cd", "physical": {"row_id": "sept15_v4_row130", "physical_pose_pass": false, "functional_pass": false, "gates": {"lift_tracks_reference_scale": false, "airborne_opposing_contact": false, "final_supported": true, "final_released": true, "final_object_upright": true, "final_object_position_under_8cm": true}, "failed_gates": ["lift_tracks_reference_scale", "airborne_opposing_contact"], "category": "005-mayonnaisebottle-bowl-\u62ff\u8d77\u756a\u8304\u9171\u74f6", "hold": false, "max_lift_m": 0.0, "reference_max_lift_m": 0.1780657842755318, "airborne_multi_finger_fraction": 0.0, "final_support": ["world"], "final_finger_rays": 0, "object_position_final_cm": 1.0267472717348625, "object_orientation_final_deg": 8.070967792871894, "final_world_tilt_deg": 1.0559451640673476, "final_opening_axis_reference_error_deg": 0.8890912596611124, "longest_contact_lift_s": 0.0, "max_hand_penetration_mm": 2.036311588602899, "actual_wrist_position_p95_mm": 31.253695800843598, "actual_wrist_angle_p95_deg": 0.7862985251230541, "contact_semantics": "native geometry reconstructed from measured qpos; not force closure or GPU force telemetry", "terminal_semantics": "support and release", "outlet_alignment": "unmeasured; no fluid simulation"}, "first_bearing_frames": {"thumb": 206, "index": 207, "middle": null, "ring": null, "pinky": null}, "first_lift_failure_frame": 229, "first_lift_failure_bearing": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}, "first_contact_frame": 206, "capacity": {"nacon": 22, "ncollision": 54, "nefc": 116}, "ctrl_target_error": 1.1916267128597724e-07, "peak_actual_tilt": 1.8973780953047885, "complete_pass_count": 0, "correction": "not run: worker turn budget exhausted", "next_step": "one own-row source-state nonthumb-first/thumb-delayed acquisition correction, then focused MJX replay"}
- Three focused audit tests pass. No repair/full accepted replay/repeat/video. Unrelated largepose paths changed concurrently and were not touched or staged. Raw outputs preserved.

## 2026-09-21T16:56:21.981205+08:00 — direct120 v2 bounded acquisition failure

## Direct120 v2: first acquisition boundary fails

One fixed-wrist solve at supported frame200 found five distal gaps of−0.400mm
(index→pinky axial span59.94mm). No wrist correction was used. Nonthumb
closure ramps140–190; earlier source thumb pose is held before the205–223
thumb closure; correction fades300–340. Wrist targets are byte-identical to
baseline throughout and every target is baseline-exact from340 onward.

The single completed focused MJX-Warp replay stopped at **frame193**:
thumb1.0412N, index/middle/ring/pinky0N. No nonthumb had previously crossed
0.2N. Maximum lift0cm; peak tilt1.217deg. The first physical-order gate fails
before reference lift229; no full episode or repeat was permitted. Finite
states/controls, maximum target error5.96e−8; capacity maxima18contacts,
50broadphase pairs,106constraints (480Hz). Nominal U1 unchanged.

The source-frame160 thumb target does not keep the thumb physically clear
while the reference wrist approaches: high loading begins at193 despite the
delayed205 target-close phase. The shallow frame200 static fit therefore did
not establish an acquired four-finger grasp. This is the exact failure boundary;
no second target or solve was attempted.

An initial launch crashed on missing telemetry local-contact coordinates at
first bearing contact. Its log/manifest/partial telemetry and INCOMPLETE marker
are retained in `interrupted/`. One explicitly authorized instrumentation-only
restart reused the identical target SHA
`3cd2a2b929c44eaca0e2b34093e7f3c26451e550eed2cfa896cd631fb2154137`.
The local-coordinate first-bearing regression test passes. Setting hashes match;
the archived baseline model-file hash is provenance, not a separately captured
interrupted runtime model binary. No claim of deterministic state reproduction.

Artifacts: `outputs/action005_direct120_v2/{fit.json,target.npy,fit_qpos.npy,
compact_result.json,focused/,interrupted/}`. Reproduce the construction and single
focused replay with `python -m tools.repair_action005_direct120 --output NEW`.
**Complete passes:0.**
