# Current model

## Objective and central mechanism
The user has accurate geometric human demonstrations and wants an autonomous policy that realizes their contact structure and object motion under simulation physics. Demonstrated joint states need not equal the loaded actuator commands that sustain them. Preserve strong hand-object state/contact imitation while allowing the policy to choose all actual commands.

## Supported observations
- Reference addition, a very small non-accumulating wrist-angle correction, finger masks, and reference-only warm-up constrain current control authority.
- Current contact intent is a static object/action segment mask; force aggregation and global point pooling do not expose the demonstrated contact surface correspondence.
- Contact rewards begin with object movement, although grip must usually be established before lifting. Translation reward is gated by expected-segment contact during that interval.
- Independent object support-plane alignment alters original hand-object geometry: 214/696 daily-v12 trajectories shift object Z by over 3 mm, maximum 10.391 mm. It is not yet established that this explains a material fraction of failures.
- The local daily-v12 package lacks cube2; a different pinned source selection is required.
Provenance: OPS.md, direct-evidence entry.

## Live causal hypotheses
1. Better contact-region/segment intent plus true control authority lets the policy learn the pressure and coordination needed to realize high-quality demonstrated grasps.
2. Some failures arise from import/asset alignment before policy learning; matching source-relative geometry may change learnability independently of model design.
3. Learning fails if a full autonomous action space is introduced without an effective initialization distribution and dense physical/contact objectives.

## Commitments
Build an actual cube2 autonomous-control training/evaluation path, retain a frozen residual baseline, preserve original-identity splits, and judge success together with object/contact fidelity. Contacts inferred from geometry remain intent with confidence, never invented force labels. Use initial near-contact tasks to expose support dynamics, then require full-start autonomous evaluation.
A single fixed object is sufficient for initial control/contact validation but insufficient for unseen-shape claims.

## Accepted implementation boundary
A separate versioned autonomous path must own all 28 DOFs from the first step and expose actual/reference contact relationships, velocities, future demonstration conditions and command history. Use a rate-bounded, reference-independent servo map, shared hand/object support alignment and additive dense physical objectives. Preserve demonstration timing initially. Reuse the existing producer/PD/PPO where practical, with legacy residual behavior unchanged. Independent audit supports this boundary; it does not establish policy competence (OPS.md, initial audit entry).

## Operational evidence and remaining questions
Server1 GPU2 is the only currently unoccupied main-workload card in the collected snapshot; the others have foreign jobs. Its Unison watcher disconnected, so preflight must be repaired before remote project launches. The first inventory operator failed its turn budget after collecting useful facts; one bounded corrective resume targets the known correct deployment/interpreter only. No repeated broad inventory is warranted.
A local compiler run for pinned guangguan cube2:02 should establish actual source identities, clocks and package data before geometry/learning tests. New algorithm implementation and training remain outstanding. Continue unattended toward results, not merely setup completion.

## Milestone-1 evidence update
The ready MTP package is executable through the new path. `load_assigned_trajectory_package` resolves cube2:02 identity `cube2_02_2833` from version 295 and preserves package digest/manifest provenance. The first physics smoke compiled the pinned cube2 asset and selected MJX-Warp; reset returned a finite 359-D observation and one zero-action step returned a finite dense reward (2.4008) and 28-D command. Focused contract tests pass. This establishes runnable physical plumbing, not learned task competence.

The autonomous command map accepts measured previous command, policy action, physical limits and rate only; reference perturbation cannot alter its output. Surface anchors are nearest points on the real collision mesh with distance-derived confidence. Remaining uncertainty is whether the compact PPO network and reward weighting can learn full-start grasp/lift/transport/place; no claim is made until rollout traces provide physical success and per-identity fidelity.
