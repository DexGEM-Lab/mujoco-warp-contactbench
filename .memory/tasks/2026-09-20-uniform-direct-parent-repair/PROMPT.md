# Uniform direct-target parent repair

## User objective
Repair the 42 strict Cheyingtong 120Hz historical parent trajectories into one self-contained direct-target dynamics setting, then use those parents to rebuild the historical 1261 XYZ-start and 128 pitcher XYZ+rotation children. The final 1923-row dataset must use one setting; mixed direct-target and hidden PID/controller settings are not acceptable.

## Scientific decision
This is Astra-guided physical target repair, not new policy training by default. The 42 parents derive from historical 100Hz source trajectories replayed/repaired onto Cheyingtong at 120/480Hz: 30 unchanged physical baselines and 12 prior local repairs. Old successful actual hand/object states are teacher evidence only. New outputs must be re-integrated from frame0 under one frozen direct-target contract, with no object forcing, attachment, teleportation, or post-frame0 state writing.

## Frozen setting contract
Cheyingtong right 28DoF; 120Hz control; 480Hz physics; 4 substeps; frame0 prestep; one arrival-indexed absolute 28D target held unchanged for all four substeps; native position actuators only; no hidden wrist PID, finger feedforward, preload, or correction state; pyramidal cone; impratio=1; asset commit 778614d; manifest e686d9. This is U1 and must exactly match the existing 534 direct rows. U2 elliptic/impratio100 is diagnostic-only and cannot enter final outputs.

## Work plan
1. Five canaries: one parent each from actions 003/005/006/007/009; U2 may be used only to diagnose target compression, while all accepted repairs must pass U1.
2. Build direct-target initialization from successful actual qpos teacher plus servo lead/inverse-position compensation.
3. Replay under U1/U2, localize first divergence, optimize bounded target windows around approach/grasp/lift/release.
4. Keep U1 fixed; all five canaries must pass the original action-specific physical gates and independently replay.
5. Repair all 42 parents, assign new UUIDs with full source lineage, then rebuild 1261+128 children.

## Acceptance
- One identical setting hash across all output parents/children.
- Every parent and child physically re-integrated from frame0 and passes its action-specific gates.
- No lucky-repeat selection; retain all attempts and require an independent replay of accepted parents.
- Deliver new Lance; preserve current 1923-row dataset immutable.

## Current interactive follow-up
Use raw Sept15 v4 row82 as the water-pitcher hand reference. Preserve the reference-driven hand backbone and fixed U1 setting; apply local target corrections based on contact and motion evidence. User god mode means complete state observability and save/restore to revisit the first bad frame. Maintain continuous terminal telemetry instead of repeatedly switching to the GUI or reading whole JSON files. Diagnostic state placement/rollback is separate from final full-frame0 acceptance; row82 now has two passing fresh U1 replays. Keep existing captures and NAS deliveries unchanged.

## Current grasp criterion clarified by user
Four non-thumb fingers enter the handle before thumb closure. Thumb and index distal pad surfaces should actually meet to close the loop around the handle, while the fingers support/clamp the handle. Keypoint distance alone is not this criterion. The complete action includes intentional overturning/pouring, then return upright, placement, thumb-first release and withdrawal. Diagnose unwanted object motion relative to the hand, not large world tilt during intended pouring. User requested continued iteration on this grasp.

## Wrist-follow clarification
User accepts wrist lag and asks to follow the existing reference wrist position and rotation after grasp, rather than synthesizing a new transport. Do not make wrist tracking error an acceptance gate. The current local row82 candidate has passed two fresh frozen-target replays; next work must preserve it and its lineage.

## Current user priority: four more single trajectories
Complete exactly one trajectory first for each remaining historical action family003 (bowl off stove to table),005 (bottle handling/pouring),007 (bowl onto stove),009 (lift bowl and hold). Keep the successful006 pitcher untouched. All use the same frozen U1 setting and native assets; no augmentation or41-parent batch yet. Each single trajectory needs action-appropriate physical acceptance and a fresh frozen-target replay from frame0. Keep all old Lances immutable.

## Current interaction requirement
User explicitly challenged the unchanged simulator window. During local repair, the visible live native simulator must be the current object and target, with terminal telemetry and full-checkpoint rollback. Headless physical trials may verify a candidate but must not silently replace the visible interactive process. Verify process liveness/state freshness, not merely an old state.json. The user specifically proposes thumb toward the middle fingertip for the bowl pinch.


## Unattended completion scope added by user
Proceed without yielding while reversible forward work remains. Final near-term objective:
1. Repair action005 under the same U1 so it strongly pinches the bottle and follows the original deep-inversion reference; the shallow row847 pick/place is diagnostic only, not completion.
2. Re-audit and repair action006 pitcher release if fresh fixed-target replay topples after release.
3. Once003/005/006/007/009 parents pass, physically generate exactly160 initial-hand-position/orientation variants per action (800 total), all under the identical U1 setting, with new lineage and a new Lance; preserve the existing1923-row dataset.
Subagent use is explicitly authorized for this scope. Verification, not visual appearance or contact count, determines completion.

## 2026-09-21 semantic correction and active-scope narrowing after live review
The user rejected the row847-derived005 parent after watching it: action005 must use the strongest feasible five-finger pinch while reproducing the full A030 reference semantics—pickup, move over the bowl, deep pour/inversion, return upright, placement and release. Pick/place-only children are invalid regardless of physical success. Quarantine all160 existing row847 campaign children and block exact800 publication. Rebuild005 from a U1-stable high-tilt grasp and require two fresh frozen-control frame0 replays. The user then narrowed active work to005 only; pause006, other actions, augmentation, export and publication until005 is repaired.

## Current user-authorized half-mass diagnostic
The user requests a complete005 A030 replay with the bottle at half current mass. Set body_mass0.30→0.15kg in a private runtime model only; keep rotational inertia, geometry, contact/controller settings, initial states and full target stream unchanged. Use normal gravity and no countergravity/applied external support. Display live Warp integration on DISPLAY=:1. This diagnostic is not nominal-U1 acceptance, and must not mutate assets or existing datasets. Pause further static searches while this complete replay is inspected.

## Current objective supersedes005 repair and old small-pose campaign
User requests: pause current replay; stop005 repair for now; newly augment only003/006/007/009,160 accepted new trajectories each. Translation perturbation radii are5,10,15cm and wrist perturbation is30deg. All native U1 physical properties remain unchanged. Do not reuse the old5–30mm/0.5deg children for the new640 quota.005 stays excluded and its paused half-mass display must not resume automatically.

Historical method verified: approach_prefix.py prepends a smooth path, chooses duration from translation/angular distance, joins the original reference at frame0 with discrete C1 position/velocity continuity, and preserves the complete original reference suffix. Contact-minus15 is not a universal approach contract; movement_end+15/contact_end+15 belong to retreat-related logic. New campaign adopts a physically integrated prepend approach with no state reset at the splice. Objects and initial velocities remain unchanged; extra precontact settling is real physics and must be tested, not corrected by injecting the parent state.

New deterministic sampling:3 radii ×8 octants ×6 rotations(±30deg one of roll/pitch/yaw)=144 base cells, plus16 additional spatial draws distributed over these cells. Use54/53/53 slots by radius,20 per octant, and27/27/27/27/26/26 by rotation.16 fixed reserve directions per slot at identical radius/octant/orientation; explicit rejection/exhaustion, never shrink. Every selected child must pass its real task gates and a separate-process frozen-control frame0 replay. Run pilots before production. Preserve native contact basis/wrench and new UUID/source lineage. Final deliverable is new640-row Lance after full readback; old datasets immutable.

## Reopened action005 objective (supersedes paused005 / four-action augmentation)
The user explicitly authorizes one writer to repair A030005 under nominal U1. Pause augmentation/other actions. Preserve original A030 object and wrist timeline: pickup, move over bowl, deep inversion/pour, return upright, placement, thumb-first release. Borrow candidate04 opposing contact topology only, never its root transport. Four fingers close before thumb. Fit continuous bounded offline contact corrections; permitted wrist micro-registration <=20mm/15deg (prefer smaller). Runtime is frozen absolute targets only; no post-frame0 state writes, feedback, forces, attachment or physics changes. First verify native50–90deg contact/force/span/drift/capacity evidence from frame0; run full and second fresh-process replay only if mechanism holds. Preserve all artifacts and unrelated largepose files. Source/tests/task memory are the only commit scope.

## Authorized continuation: topology metric correction
Restore only 005-owned paths in a new commit; never rewrite ca9823a/37e3096 or touch largepose files. Donor axial coordinates are not invariants: use opposite sides/normals, axial order, >=80mm span, shallow contacts, <=20mm/15deg wrist correction and continuous reference-conditioned branch. Close the70deg +0.460mm gap minimally; four fingers must physically acquire before thumb. One native frame0 focused diagnostic; stop on failure, no sweep. Only a positive focused result permits complete A030 and fresh-process frozen-control repeat. Update005 docs/memory honestly.

## Mid-run authorized single load-balance correction
After v2, exactly one thumb-opening/middle-closing correction is permitted, geometry-derived, smooth before380, taper with grip, wrist/index/ring/pinky unchanged. Rerun focused once; stop if false. No sweep.

## Authorized v4 native target-to-wrench identification
Only005: reconstruct frozen-v3 nominal-U1 prefix, identify compact thumb/middle central sensitivities at380/400/409, then at most one inverse correction/focused replay if controllable. Preserve wrist/index/ring/pinky/reference timing. Stop on discontinuity/rank deficiency/wrong sign; full A030 and independent repeat only after focused invariants pass. No sweeps, other actions or augmentation. Preserve v2/v3 and unrelated largepose paths.
Supervisor approved HEAD9890d25 (unrelated descendant of7de8fa9) and reconstruction because historical full-native buffers do not exist. Compare every archived qpos/qvel/ctrl frame through409 at1e-6/1e-4/0 tolerances and bearing forces at0.02N+1%; fail closed on mismatch. Accepted reconstructed checkpoints are diagnostic only, never called historical original states.

## Authorized v5 fresh-realization continuation
From c3e26bd, do not reconstruct historical v3 identity. Capture one NEW frozen-v3 nominal-U1 baseline and complete native checkpoints380/400/409 in the same process; qualify middle unloading with retained opposed index/ring/pinky contact, then compact central normal-direction probes ±0.004/±0.002rad for8frames. Reject >30% scale/symmetry inconsistency, inadequate rank/signs or condition>1000. Only if qualified solve exactly one regularized inverse; freeze v5 before independent frame0 transfer. Stop at first failed validation; complete A030 and second fresh repeat only after focused pass. Preserve all old outputs and unrelated largepose paths.
Supervisor clarification: archived checkpoint spans are78.989/79.259/79.734mm, so baseline gate is same basin/no collapse, not >=80mm. V5 focus must retain ring/pinky and axial order, all five bearing, >=78mm and no>5% drop from380 through first>50deg; report original80mm criterion separately. Physical acquisition requires all four nonthumb>0.2N before thumb>1N; light thumb touch permitted. Any single correction must begin early enough to affect this order, not only350.

## Authorized direct Cheyingtong nominal120Hz action005 pivot
Supersedes the A030-only reference constraint. Audit immutable Sept15 Lance v4
non-generated005 rows62,69,124–131 before selecting any reference or replaying.
Use the original right-hand/object/wrist/finger trajectories, pinned Cheyingtong
assets and exact nominal U1 c6db552f...,120/480Hz, four substeps, CCD16.
Require pickup, bowl alignment, deep inversion, upright return, placement and
release; reject shallow/easy references. Audit every row's timestamp/identity,
discontinuities, hand-object relative motion and source-state surface topology.
Geometric contacts must not be represented as measured bearing forces. Freeze
selection before physics; if no deep-pour reference has suitable topology, stop.
Otherwise one native frame0 compensated baseline, then one bounded source-derived
offline correction and focused high-tilt test. No sweep, donor transport, physics
changes, runtime feedback, forces, attachment or post-frame0 state writes. Stop
on first falsified repair mechanism. Full action and fresh frozen-control repeat
only after focused success. Video only after two physical passes. Output root:
outputs/action005_direct120_v1. Other actions/augmentation and unrelated largepose
paths remain untouched. No delegation.

## Mid-run clarification: approximate source geometry; MJX adjudication
Source contact geometry is diagnostic, not an eligibility veto. Select the best
complete direct120 kinematic reference (row130); imperfect captured qpos need
not be an equilibrium. Run nominal MJX-Warp U1 before donor consideration.
A bounded same-dataset donor would require demonstrably better MJX acquisition,
not CPU contacts alone. If budget cannot support qualified donor comparison,
stop after baseline with its exact failure and a concrete next experiment.

## Authorized fast direct120 v2 continuation
Starting committed9dc54cb and v1 artifacts, one fixed-wrist local shallow acquisition solve before206, four nonthumb before delayed thumb. Preserve row130 wrist/object and full task timeline. Exactly one focused nominal MJX-Warp frame0 replay; stop first failed boundary, no variants. Only focused success permits full and fresh-process repeat. Unique outputs/action005_direct120_v2; no changes to unrelated largepose files.

## Current resumed objective: finish and publish strict-C1 exact640
Action005 work remains paused and excluded. Resume the completed v3 strict-C1
production for actions003/006/007/009: exactly160 selected/action and640 total.
Merge disjoint003/006 shards without changing artifacts; verify every canonical
artifact against its ledger SHA; require two accepted frozen-control replays with
distinct PIDs, zero solved-force prefix contacts, exact strict-C1 identities and
byte-exact parent suffixes. Export a new self-contained640-row Lance, perform
full source-bound row readback, build a per-UUID visualization layout with zero
unresolved background clearances, and atomically publish to a new NAS directory.
The existing1,923-row dataset and all v1/v2/005 evidence remain immutable. No
subagents for this resumed publication work.

## Current action005 minimal-replay follow-up
The user supplied a ten-row u3300 action005 Lance and then restored the passive
bowl track. Preserve that supplied file. Produce a sibling minimal replay Lance
containing only the shared ten core compact columns: hand spatial state and
28D targets, all scene-object poses, compact contacts, timestamps, reference
mapping, identity and provenance. Do not fabricate the exact640 audit extensions
or fill unavailable evidence with zeros. Repair the duplicate generated UUID
deterministically while leaving every other decoded value unchanged. Extend the
direct target-DOF viewer to reconstruct all declared scene objects, validate a
real two-object GPU reset/replay, and leave parent126/128 alternating visibly on
local DISPLAY=:1. This source inspection does not qualify005 for nominal-U1 or
add it to the published exact640.
