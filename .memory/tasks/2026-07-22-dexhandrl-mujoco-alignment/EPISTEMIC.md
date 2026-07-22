# Epistemic Record

## Current model

Exact checkpoint alignment is a sequence of observable transformations rather than a single rollout metric. Trace schema v2 fixes the Lance identity to row 22 / `cube1` / `01` / sequence 1 / UUID `c21d4fa5-185a-5ea7-aa22-e1d34beae42f`; every row separates the action boundary `decision_frame` from its post-step `state_frame`. It compares processed/RMS 461D observations by every canonical slice, deterministic action, residual offsets/targets, and only then post-physics state. This separates shared-input divergence from simulator physics divergence. MuJoCo's diagnostic captures raw Lance active reference and the actual pre-coupling corrected active target inside `_apply_residual_action`; its applied target equals corrected with explicit `post_filters=[]`.

The MuJoCo producer has run against the supplied environment, Lance archive, and epoch-20000 checkpoint for three exact steps and generated a contract-valid trace (OPS 2026-07-22T21:34:00Z). It records MuJoCo's lack of Gym ActionRules post-filters explicitly: `filtered_active_target` is the unfiltered corrected target, never presented as a matched Gym filter result. The existing task-level distal routing remains untouched.

Gym imports and help are now operational through `/home/jay/anaconda3/envs/dexhand_rl/bin/python`; the obsolete claim that no Python runtime exists is false. The live source-read-only wrapper verifies its expected task/action hook surface after importing Isaac Gym before Torch and snapshots source Git state. It stops because the sibling checkout lacks a stable public factory that creates the exact one-env player without reimplementing Hydra/player internals. Required live capture boundary: `DexHandBase.pre_physics_step/post_physics_step`, `task.obs_dict`, and `action_processor` targets. This is the present discriminating runtime boundary, not an inferred parity failure.

## Open discriminating question

Can a public, read-only Gym one-environment player construction path be supplied so the wrapper can attach its runtime-only hooks and emit the matching trace? Once available, the comparator will identify the first strict-tolerance stage before the physics boundary.
