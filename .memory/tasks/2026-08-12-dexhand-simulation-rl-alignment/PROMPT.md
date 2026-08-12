## Objective

Make the existing DexHand021Pro retarget dataset executable as a physically consistent MuJoCo/MJX reference and train residual PPO against that reference using the same task-space reward semantics and training evidence contract as ManoRL `dev`.

Completion requires both layers to work:

1. `use_residual=false` follows representative and all selected retarget references without numerical failure, pathological penetration/ejection, or a systematic simulator/control defect that RL would have to hide.
2. Residual PPO consumes a versioned ManoRL-aligned task reward, records comparable update/episode/evaluation evidence, and demonstrates improvement over reference-only and untrained-policy baselines on physical task metrics.

## Workbench

1. Freeze and inspect the current residual-off DexHand reference baseline.
2. Correct simulation/control mechanisms exposed by that baseline.
3. Implement a versioned ManoRL-aligned reward contract with a DexHand embodiment adapter.
4. Align progress, episode, checkpoint, evaluation, and W&B/TensorBoard evidence schemas.
5. Run reference-only, untrained, short-PPO, then scaled learning validation.

## Context

- Product integration branch: `dexhand` (protected; no direct commits).
- Task branch: `dexfeat/simulation-rl-alignment`.
- Worktree: `/home/jay/dexrobot/FromSSH/manoRL_mujoco-worktrees/dexfeat-simulation-rl-alignment`.
- Base: `dexhand@72b96f5`.
- Dataset (read-only): `/mnt/nas-222-project/mocap/for_retargeting/dexhand021pro/results_all_mano/all/lance_new_all_generated_mano/lance_result_archive_full/dexhand021pro_lance_new_all_generated_mano_retarget_result_archive.lance`.
- Observed dataset state: Lance version 30, 7,539 rows, `retarget_schema=dexhand_retarget_minimal_v1`, `hand_model_type=dexhand021pro`, 22D `dexhand021pro_controls_world_v1` references.
- Intended data chain: MANO/mocap task trajectory from the ManoRL source domain → DexHand021Pro kinematic retarget reference → MuJoCo/MJX physical execution → residual RL correction.
- Existing DexHand ABI: 461D observation, 22D residual action, state-dependent routing to 26D full target.
- Existing historical production evidence: N1024 device PPO passed one-update preflight and later reached at least update 1452/1404 across initial/resumed runs; reward existed, but its contract and recording differ from current `dev`.

## Task specifications

### Data boundary

- Treat the supplied retarget Lance as complete task input; do not regenerate or mutate it.
- Preserve retarget trajectory UUID, row ID, object/action identity, movement window, dataset path, and Lance version in every run/checkpoint/evaluation contract.
- Preserve `seed_uuid` as available lineage evidence. Missing upstream dataset path/version is reported, not inferred or fabricated.

### Simulation acceptance

- The residual-off baseline is a prerequisite for RL, not an optional diagnostic.
- Evaluate at least all selected `cube1` actions `01,02,03,04,09,10,18`, with deterministic identity selection and one full movement/release horizon per action before scaled training.
- Record reference tracking error, object translation/orientation error, contact coverage, object lift/placement, fingertip penetration, object speed/angular speed, termination reason, and finite-state validity.
- Separate reference feasibility from policy quality. A repeated reference-only failure caused by scene assets, inertial origin, contact decoding, target routing, actuator response, reset, timing, or capacity must be corrected before PPO is used as evidence.
- Keep `mjx-warp` authoritative for batch training. Native MuJoCo may be used as a discriminating simulator/control diagnostic, with backend identity explicit.

### Shared reward contract

Use the current ManoRL `dev` task-space equations as the semantic source:

- per-axis object position tracking `scale * exp(-40 * abs(error))` with scales `(0.5, 0.5, 2.0)`;
- object rotation piecewise reward with the dev 20°/90° segments;
- expected-contact fraction with maximum `0.4`, active only in the reference contact window;
- position tracking gated by contact within the window and by `0.4` after the window;
- late hand-object contact after the grace interval penalized by `3 * 0.4`;
- post-contact object stability `0.4 * exp(-(speed/0.1)^2)`;
- survival `0.001` outside the early pure-reference phase;
- deviation termination at `0.10 m` with an explicit one-shot penalty;
- PPO reward scale recorded separately from the raw environment reward.

DexHand-specific adaptation is limited to embodiment observations:

- map 16 DexHand contact bodies to the same 16 semantic contact slots;
- use DexHand residual base/joint offsets and active-joint mask in the shared action-regularization interface;
- keep fingertip penetration as a separately named embodiment penalty, never folded invisibly into a shared term;
- measure contact in Newtons and version any selected threshold. Do not silently retain 5 N under a metric named as if it were the dev 0.2 N contract.

The raw term vector, weighted term vector, total, early-phase behavior, and termination contribution must be host/device equivalent on fixed tensors.

### Recording contract

- Use one versioned cross-embodiment metric vocabulary. At minimum record raw and PPO-scaled reward, every reward component, episode return and component totals, episode length, termination reason, reference identity, object/action groups, physical tracking metrics, performance, capacity high-water marks, and PPO diagnostics.
- Distinguish instantaneous rollout means from completed-episode sums in names and schemas.
- Record grouped metrics for object and object/action; include UUID/row-level evaluation records.
- Checkpoint sidecars bind dataset path/version, ordered assignments, reward contract ID/config, PPO reward scale, physics contract, observation/action ABI, model config, progress, and resume reset semantics.
- Establish deterministic `reference`, `untrained`, and `trained` evaluation modes on the same identity set.

### Learning acceptance

- Run in order: residual-off reference baseline → residual-enabled zero/untrained policy → bounded short PPO smoke → longer/scaled PPO only when prior evidence is valid.
- PPO/loss changes alone are insufficient. Acceptance requires improvement over the untrained baseline in shared task reward and physical metrics without worsening penetration/ejection/termination behavior.
- Report aggregate and per-action results so easy actions cannot hide regressions.

## Constraints

- Work only in `dexfeat/simulation-rl-alignment`.
- Do not modify or republish the Lance dataset.
- Do not commit directly to `dexhand`, `dev`, or `main`.
- Do not merge `dev` wholesale into the DexHand product line.
- Do not use subagents; the user has not authorized delegation.
- Do not launch a long production run before the reference and short-PPO gates pass.
- Do not use RL to mask a repeatable reference-only simulator defect.
- Preserve DexHand021Pro observation/action/routing ABI unless evidence requires an explicitly versioned change.
- Stage explicit files only and keep commits atomic.
