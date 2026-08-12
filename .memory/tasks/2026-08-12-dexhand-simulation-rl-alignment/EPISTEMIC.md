# EPISTEMIC

## Phenomenon

DexHandRL has a complete 021Pro retarget reference dataset and a functioning MJX-Warp PPO runtime, but the simulation baseline, reward semantics, and evidence schema are not yet one coherent contract. A policy can receive scalar reward and train while reference infeasibility, embodiment-specific contact scaling, or telemetry naming prevents us from knowing whether it learned a physically consistent correction to the MANO-derived task.

## Causal model

The retarget trajectory defines desired task motion in the DexHand configuration space; the object reference remains the shared task target inherited from the MANO/mocap trajectory. Physical execution adds embodiment-specific dynamics, contacts, actuator response, collision geometry, and state-dependent 22→26 target routing. Residual RL should correct this execution gap. It cannot distinguish a correctable residual from a broken simulator contract unless residual-off reference behavior is measured first.

The reward should therefore factor into:

1. a shared task-space functional over object/reference motion and phase;
2. a DexHand observation adapter that supplies contact slots and residual state in the proper units;
3. explicitly separate embodiment regularizers such as fingertip penetration;
4. a separately recorded PPO reward transform.

This makes ManoRL and DexHandRL rewards comparable without claiming that their force distributions or actuator coordinates are identical.

## Supported observations

- Current retarget Lance exists locally at version 30 with 7,539 rows.
- Rows identify `dexhand_retarget_minimal_v1`, `dexhand021pro`, a 22D world-control format, movement windows, retarget UUID and `seed_uuid`.
- The dataset does not expose an upstream dataset path/version field, so exact source-row lineage cannot currently be asserted beyond the retained seed UUID.
- Historical DexHand PPO reached production-scale execution (N1024 and >1400 updates), proving runtime closure and scalar reward flow.
- Current DexHand and dev reward structures already share object tracking, contact quality, rotation, stability, survival/action penalties, and deviation termination in broad form.
- Their equations are not identical: dev uses `(0.5,0.5,2.0)` tracking scales, late-contact penalty, strict 0.2 N contact, PPO scale 0.5, and currently disabled action penalty; DexHand uses task-YAML weights, 5 N contact, no late-contact penalty, direct raw PPO reward, active residual penalties, and a fingertip penetration term.
- Current recorders differ materially: dev has completed-episode v2 records, per-pair/group metrics, deterministic evaluation modes, W&B artifacts, and richer checkpoint binding; DexHand has Gym-style TensorBoard and progress/checkpoint records but no equivalent per-identity evaluation artifact.

## Ruled out

- Missing reward: historical DexHand PPO consumed scalar reward; the earlier defect was loss of component telemetry across the JAX→Torch boundary.
- Missing data: the user confirms the retarget dataset is the intended input and no regeneration is required.
- Direct whole-branch synchronization: `dev` and `dexhand` are independent product histories; shared semantics must be ported as a bounded contract, not merged wholesale.

## Current commitment

First establish a residual-off physical reference baseline. Then port/version the current dev task reward equations behind a DexHand adapter and align recording/evaluation. Only after fixed-tensor host/device parity and reference/untrained baselines will short residual PPO be accepted as evidence.

## Live uncertainties

- Which of the seven cube1 actions are physically feasible under current MJX scene/control without residual correction?
- Does the 0.2 N dev threshold separate live DexHand contacts from numerical/background forces, or does DexHand require a versioned calibrated threshold while retaining the same contact-fraction equation?
- Should DexHand residual penalties be enabled in the first shared-reward version, and at what scale relative to dev's currently disabled action penalty? The answer must come from zero/reference and action-scale probes, not legacy YAML inertia.
- What is the final historical 3000-update run state and learning curve? It is useful baseline evidence but not required to define the new contract.

## Highest-value next experiment

Run deterministic residual-off reference evaluation for the seven cube1 actions and collect force distributions plus physical tracking/penetration metrics. This single experiment distinguishes simulator/control defects, reference feasibility, and the force threshold needed by the shared contact equation before reward or PPO changes are judged.

Cross-reference: `OPS.md`.
