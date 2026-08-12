# EPISTEMIC

## Phenomenon

DexHandRL has a complete DexHand021Pro retarget reference dataset and a functioning MJX-Warp PPO runtime. The reference actions alone do not reliably establish a grasp or follow the object's large lift. This is the residual policy's intended optimization gap, provided the physical baseline is correctly specified, finite, capacity-safe, and measured under the same assets/reward/evaluation contract as training.

## Causal model

The retarget trajectory defines a nominal hand-control reference and the object trajectory defines the task target. MuJoCo/MJX adds embodiment-specific dynamics, contacts, actuator response, collision geometry, and state-dependent 22→26 target routing. The residual policy modifies the nominal reference to create the contacts and forces needed for object tracking. Reference-only grasp/lift success is therefore not a prerequisite for RL; reference-only execution is a prerequisite measurement that makes the residual gap attributable.

The reward factors into:

1. a shared task-space functional over object/reference motion and phase;
2. a DexHand observation adapter supplying strict hand↔object contact slots and residual state;
3. an explicitly separate fingertip-penetration regularizer;
4. a separately recorded 0.5× PPO reward transform.

The retarget rows contain object-motion boundaries but no measured contact interval. Those boundaries remain an explicit reward-phase proxy and are never described as observed contact timing.

## Supported observations

- The read-only Lance dataset is version 30 with 7,539 rows, 22D `dexhand021pro_controls_world_v1` references, retarget UUID, and `seed_uuid`. It does not expose an upstream dataset path/version.
- Runtime assets are pinned to `all_assets@7228b5cfce8d9a072ed4bded7a489cf73d521b68` and nested `mano_assets@cde03ef94816b589f574ca6f358695005d3d1a3f`.
- Every object resolves from `Assets/sim/decomposed/<object>/<object>.urdf`. Visual geometry is used for rendering/point-cloud observation; every CoACD collision piece participates in physics. Native compilation confirms cube1 has 1 collision piece plus 1 non-colliding visual and bowl has all 28 collision pieces plus 1 visual.
- The latest-asset residual-off cube1 N8 completes all seven primary horizons with finite state, no measured fingertip penetration, and maximum utilization of 5.3% contact rows, 16.8% constraints, and 43.8% CCD candidates.
- All seven primary reference worlds exceed the counterfactual 0.10 m deviation threshold and actual maximum lift is below 8 mm while target lift reaches about 0.32 m. This is the residual-learning baseline, not a simulator-contract failure.
- In active residual-off N8 horizons, 406 positive strict hand↔object force slots were observed. Every positive slot exceeded 0.2 N; 88.2% exceeded 5 N. Thus 0.2 N preserves the observed physical-contact population while 5 N discards about 12% and is not silently retained.
- A fixed-seed untrained mean policy also completes all horizons with finite state and safe capacity. Its primary-world raw return mean is about 350.8 versus 271.8 for reference-only, while all seven worlds still exceed 0.10 m. This is initialization bias and the direct short-PPO comparison baseline.
- Exact duplicate worlds diverge after tiny early differences under long contact dynamics despite unsaturated capacity. This is evidence of batch contact sensitivity; it is reported as evaluation variance rather than treated as an absolute blocker for bounded PPO.
- Historical DexHand PPO reached production-scale execution (N1024 and >1400 updates), proving runtime closure and scalar reward flow.

## Ruled out

- Missing reward: historical PPO consumed scalar reward, and the current device transition exposes a fixed additive 10-component vector.
- Missing data or required data regeneration: the supplied Lance is the intended immutable input.
- Stale object path or first-piece-only physics: the authoritative decomposed URDF and complete collision set now compile and run in MJX-Warp.
- GPU memory/capacity as the cause of the N8 gap: measured buffers remain far below saturation.
- Friction alone or missing 22→26 routing as a sufficient explanation for failed reference-only lift: bounded probes did not repair the behavior.
- Reference-only task success as an RL admission criterion: the user explicitly clarified that grasp failure is the problem residual RL is meant to solve.
- Direct whole-branch synchronization: `dev` and `dexhand` are independent product histories; only bounded shared semantics are ported.

## Current commitment

Freeze the latest-asset reference and untrained evaluations as baselines, save the calibrated reward/assets/evidence contracts in reproducible commits, then run a one-update PPO runtime smoke followed by a bounded short PPO with enough transitions to complete trajectories. Evaluate the trained mean policy on the identical N8 identity set. Improvement is judged against untrained per-action shared reward and physical tracking/contact/lift metrics; short training is not expected to solve the full task.

## Live uncertainties

- Does the current short-PPO budget produce a measurable change beyond seed/contact variance on any action?
- Are training-side completed-episode and component metrics correct when rollouts span mixed trajectory lengths and autoresets?
- Can exact duplicate sensitivity be reduced or bounded enough for reliable later seed comparisons?
- Does scaled training need action-penalty regularization after the unregularized first shared-reward experiment?

## Highest-value next experiment

Run the one-update N8 PPO smoke to validate checkpoint, sidecar, telemetry, finite gradients/parameters, reward shaping, and capacity. If it passes, run a bounded short training budget that covers completed episodes and compare its deterministic N8 result with the fixed-seed untrained baseline.

Cross-reference: `OPS.md`.
