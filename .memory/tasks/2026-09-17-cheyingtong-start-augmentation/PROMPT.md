# Cheyingtong120Hz initial-hand-position augmentation

## User objective
User requests unattended execution with bounded delegation:1) finish preliminary repair;2) augment successful120Hz Cheyingtong trajectories by perturbing the hand initial position, analogous to the historical49-to1000+ campaign.

Preliminary success set is frozen to42 strict passes in the published repair bundle. B035 remains functional-only and is excluded, without changing its gate or discarding its separate unresolved issue. Do not make augmentation wait for more blind B035 offset searches.

## Ownership
ManoRL feat/cheyingtong-start-augmentation, created by primary scripts/start_pi_task.sh --no-launch from dev fa975eb. Repair source ba84ce9 was integrated into dev; its former worktree is detached with all evidence preserved. Primary is coordinator-only; no direct source edits/commits there. One executor is the only writer of this linked worktree. No nested delegation. Atomic explicit-file commits after diff review. Generated data only under new outputs; no overwrite of published inputs or historical artifacts.

Own new augmentation source under sim/manorl/, CLI under tools/, focused tests/docs, and task/output evidence. Existing repair engine/data are the source of truth; modify them only if a concrete compatibility defect is demonstrated, and report it first.

## Immutable inputs
Bundle: primary outputs/cheyingtong120_repair43_20260917. accepted_full_pose_ids.txt supplies exactly42 IDs. Use inputs100 plus recipes/catalog.json through load_input(...,120) and edit_targets; assert reconstructed desired/finger_target_delta equal the delivered measured-run input arrays. Source hand profile sunke denotes the predecessor physical replay model, not the capture operator (parent investigation finds literal raw index.operator=huangmeiai); physical replay hand is Cheyingtong/right28DOF. Manifest e686d91931979c444c923d4f962ddd135ce8204e0de9df6fd7368bfdb855e29b, asset commit778614d09e917deffed0bff3f357aa237efa762d. Physical files come from primary materialized DexStream root. Bind selected manifest before asset imports and validate native model arrays.

120Hz command sampling, four480Hz physics substeps, frame0 is PRESTEP, N-1 intervals. Preserve source time/duration. No changed mass, friction, gains, gravity, solver profile, object initialization, or object forces/poses after reset. Keep repaired wrist rotations/finger curves, absolute donor preload, envelopes and controller law.

## Intervention
Reuse historical minimum-jerk positional residual only:
 h_aug[k,:3] = h_repaired[k,:3] + delta_xyz * (1-10u^3+15u^4-6u^5), u=clip(k/(C-1),0,1).
Set residual exactly zero at C-1 and afterward. Preserve command suffix byte-exact and central-difference velocity at C. NO new orientation/finger/retreat/scene transforms.

IMPORTANT: current replay initializes from inp.initial['qpos'], not desired[0]. Also shift initial['qpos'][:3] by delta_xyz; all other initial qpos and all qvel stay identical. Never reset physical/controller state at C. Do not replay measured hand qpos or reuse old applied controls as the command recipe.

## Pilot timing decision
The old0.4s precontact guard leaves only0.0917s residual duration for B043, forcing10cm perturbations to~2m/s and69m/s². Do not silently shrink meaningful augmentation to millimetres merely to preserve that operational guard.

Test a0.10s unchanged-command guard before the parent's first measured active-object contact, using the old movement-start constraint where applicable. With first contact0.5s, B043 has~0.39s taper, so10cm adds~0.48m/s and3.8m/s². These are predictions, not hardware limits. Guard must be configurable/recorded for comparison; do not alter it silently. Proposed planning caps: residual speed0.5m/s and acceleration5m/s², chosen to avoid violent prefix motion. Production range will be fixed from physical pilot evidence; initial pilot probes3–10cm, with longer-window parents potentially extending to15cm.

## Deliverable for first writer slice
Implement the position-only augmentation layer and executable scalar pilot atop existing local_contact_dynamics.replay. Do NOT build a full production/batching framework before demonstrating the mechanism.

1. Deterministic plan/provenance,42-seed checks, exact target reconstruction.
2. Initial-state feasibility check: reject hand/world or hand/scene-object penetration; do not count existing hand self-contact as a scene collision. Record exclusion reasons. No object state adjustments.
3. Full continuous physical replay with new hand start; save augmented desired, original parent desired, actual qpos/qvel/objects, controller/substep targets, clocks, merge metadata and source/recipe/model hashes. Unique variant identities. Fresh output directories.
4. Existing action-specific physical+pose gates plus no unintended premerge hand/environment contact and no>=25ms unsupported/no-hand interval above2cm. Contact geometry is not force closure. Preserve failures; no repeated unchanged runs until one passes.
5. Six-run pilot: A_row035 (pitcher) and B_row043 (short approach/hold), each zero offset and two opposite collision-free positional offsets up to10cm. Choose a tangential horizontal direction from the parent approach if axis-aligned directions collide initially; record it. Static infeasibility is a recorded outcome, not a zero-offset fallback. Inspect merge-state error/contact chronology, not just final counts.
6. Focused tests for physical initialization, unchanged other states, C2 envelope, suffix/derivative identity, clocks and failure serialization. Commit coherent code and report physical results.

Stop after this pilot and report. No1344-row production yet, no scene/rotation/retreat augmentation, no source/Lance overwrite, no unrelated training. If first zero run contradicts the parent significantly, inspect input/control/model evidence before proceeding. No broad suite by default. Budget:20+3 turns,45soft/70hard tool calls; return precise remaining question at soft cap. GPU jobs use task-derived inspectable tmux, PID/log/exit files, one simulation stream. Do not babysit/poll long external jobs; return job identifiers if necessary.

## Intended subsequent scale (parent decision)
Approximately42 parents×32 distinct positions=1344 physical candidates plus parents, enough to target1000+ accepted data if yield supports it. Balanced directional octants/radial strata, deterministic seed2026091701. Sample physically admissible initial states with recorded bounded preflight rejection; never silently change strata/zero perturbations. Batch same-parent worlds may be implemented after pilot correctness to avoid recompiling1344 single-world processes. Acceptance count is measured, not guaranteed or inherited.

## Historical evidence
Fresh read-only reports: primary .pi-subagents/artifacts/502bbaf9-ecc2-435b-95db-5a96c9606198_{critic_0,theorist_1}_output.md.
Old mechanism: sibling case-reference-faithful-contact-repair--augmentation/tools/prepare_start_perturbation.py, run_augmentation_full.py, augmentation_1500.py and .memory/tasks/2026-09-11-start-perturbation. Historical lineage is43→241→300→1475;840 rows were transformed recordings, not independently simulated starts. No orientation-waiver inheritance; no old frame0-integrated adapter reuse.
