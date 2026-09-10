## Objective
Build and empirically validate a demonstration-conditioned autonomous ManoRL policy. Human demonstrations are geometrically accurate but not physically consistent under the simulator. Exploit their hand/object motion, contact locations, contacting finger segments, and action identity strongly, while all executed hand controls come from the policy rather than an additive reference action. First establish cube2 physics/control competence, then full-trajectory and multi-action behavior. Completion requires actual simulated grasp/lift/transport/place/release evidence, reference/contact fidelity, held-out demonstration evaluation, and independent review—not only implementation or passing tests.

## Workbench
1. Implement the accepted first vertical slice: separate versioned autonomous control, reference/contact-aware observations, dense physical task reward, runnable PPO and real MJX-Warp evaluation. Retain demonstration timing for this first slice.
2. Prepare a pinned cube2:02 source/package and isolated Server1 deployment; the initial snapshot found GPU2 available and a disconnected Server1 Unison watcher.
3. Continue unattended toward actual empirical results; pause for user input only for a genuine unresolved user decision or an irreversible high-risk action.
4. Review its physical semantics and contracts, deploy an immutable candidate, and run bounded multi-GPU discriminating settings.
5. Inspect actual failures and iterate; report at runnable-code, first-results, and mechanism-changing milestones.

## Context
Product: ManoRL. Feature: `feat/contact-conditioned-autonomy`, parent integration: `dev`, base: `c38f787`.
Task memory: `.memory/tasks/2026-09-08-contact-conditioned-autonomy/` in this feature worktree.
The user explicitly permits major algorithm refactoring, a new branch, server deployment, multiple GPUs/settings, timely reports, and independent review. Explicit subagent authorization was granted for this task; children may not delegate recursively. The user subsequently requested unattended continuation: keep executing/delegating until the objective is achieved or a genuine user decision/irreversible high-risk step requires input. Receiving a child result, finishing setup, or completing a code-only milestone is not final completion.
Primary integration tree contains unrelated tracked/untracked changes and must remain untouched by implementation. Local machine paths, host routes, resource snapshots, and run artifacts belong in ignored `.memory/local/` or `outputs/`; never commit credentials or machine-specific deployment state.
Authoritative host routing and W&B/source-sync preflight policy are in the primary worktree's `.memory/local/servers.md` and `.memory/local/training-wandb-policy.md`.
The current daily-v12 package has 696 trajectories / 14 pairs but no cube2. The guangguan Lance location exists and historical cube2:02 data/rollouts exist; pin and verify the actual dataset before experiments. Do not assume old-asset checkpoints are physics-equivalent to the pinned DexStream assets.

## Task specifications
### Scientific formulation
- Preserve accurate demonstration hand-object geometry and motion/contact intent. Learn the actuator commands and contact loads needed to realize them under simulated dynamics.
- A demonstration state is not necessarily its required position-servo command under load. Strong state/contact imitation is compatible with autonomous control; hard-wired `q_ref + residual` execution is not.
- References may enter observations, training supervision, rewards, and explicit task initialization. They must not enter the active control-map after the actor output, through warm-up reference playback, forced inactive-finger behavior, or hidden target-following fallbacks.
- The object remains physically simulated; no target-pose overwrite, attachment, or nonphysical support may create task success.
- Contact intent should distinguish finger/hand surface region, object surface region, phase and confidence. Static object/action contact aliases are useful priors, not complete contact labels or joint-disable commands. Inferred contacts are not measured forces.
- Policy inputs should connect actual and reference hand/object relationships, velocities and control history, present/future demonstration conditions, action identity, and global/local object geometry. Preserve metric scale, gravity/table context, and known relevant physical asset properties.
- Reward physical object motion and sustained contact/low slip while retaining demonstration grasp and trajectory fidelity. Avoid gating all translation learning behind a binary contact threshold. Use contact-stage information rather than only `object_move` timestamps. Begin with demonstration timing; justify any bounded phase freedom through evidence.

### Initial evidence and live uncertainty
- Existing control in `sim/manorl/abi.py` adds reference targets, constrains wrist XYZ offsets, applies non-accumulating default wrist-angle corrections of 0.00025 rad/axis, and masks finger actions by expected contact.
- Existing observation/model use global 64-point PointNet pooling, action-type one-hot, static expected contacts, and aggregated per-segment forces; source hand future/contact regions and velocities are not fully represented.
- Existing contact reward counts expected segment force norms above 0.2 N within the object movement window; this does not establish correct surface location or sufficient supporting wrench.
- Source decode in `sim/manorl/trajectory.py` independently shifts object Z to its initial support plane. A read-only daily-v12 package measurement found 214/696 absolute shifts >3 mm and a maximum 10.391250776 mm, without corresponding hand-reference translation. This is confirmed geometry alteration, not a proven dominant cause of failures. Investigate/declare alignment before reusing derived contact labels; preserve the legacy baseline unchanged.

### Accepted first implementation boundary
- Retain legacy residual behavior and checkpoints without changing their existing contract constants. Introduce a separate autonomous route and explicit contracts. Reuse the physical producer, asset/clock/package components and PPO where practical; do not copy the entire legacy environment merely to change control.
- All 28 DOFs are policy-owned from the first physics step. Use a reference-independent, rate-bounded servo-command mapping with observed previous command and measured state; guard command wind-up using physical limits rather than a reference-relative offset cap.
- Provide current/multi-horizon human and object references, actual velocities, previous command, and demonstrated per-segment surface proximity/anchor intent. Include a minimal spatially explicit contact/geometry encoder; a larger geometry model is an ablation, not a prerequisite for first physics learning.
- Apply declared support alignment consistently to hand and object in the autonomous input path; leave original arrays/baseline unchanged. Derive contact intent only after validating source-to-simulation geometry. Do not inherit the silent missing-asset alignment fallback.
- Use additive object motion, reference hand-object relationship, proximity/contact and stability/release objectives. Preserve the source clock and fixed progression initially. Full-start evaluation is mandatory; near-contact initialization is an explicit training/diagnostic option.
- Review recommendation to match wall clock is not accepted as the causal budget: match environment transitions and report wall clock separately. A strong residual baseline does not change the autonomy objective into a contact-only objective.

### Delivery and experiment acceptance
- First vertical slice must be runnable on real MJX-Warp, with explicit autonomous action, observation, reward, clock, data, asset, and checkpoint contracts. Keep the existing residual path available as a frozen comparator.
- A fixed physical state and fixed policy action must produce identical commands regardless of reference changes. Network outputs may respond to reference changes; the control map must not add them.
- Extract and inspect actual demonstration-conditioned surface/contact features on cube2 before calling them correct. Validate geometry/coordinate transforms against examples, not only tensor dimensions.
- Use original-identity train/validation/test splits, shared by all settings; source siblings/augmentations must not leak across splits. Test demonstrations remain known conditioning inputs but are not training examples.
- Match environment transitions, data splits and evaluation scenarios across comparisons. Record wall-clock efficiency separately. Begin with a small set of mechanistically different settings; use seed replication after a useful signal rather than a blind hyperparameter grid.
- Near-contact initialization and short physical subskills may accelerate learning, but final acceptance must use full-start autonomous episodes. Never report subskill success as full-task competence.
- Report per-identity/pair success with explicit denominator, object path/pose error, contact-location/segment fidelity, slip and failure phase, plus representative real rollout artifacts.
- Single fixed-object results do not establish unseen-shape generalization. Later test held-out geometry/size before making that claim.

## Constraints
Follow `.memory/project/pi-orchestration.md` and GitGuard; no direct commits on protected integration branches.
Only the assigned executor writes feature source; read-only reviewers and operators do not modify it.
Preserve all unrelated working-tree changes and historical source/checkpoint/data artifacts.
Use separate versioned new contracts rather than silently reinterpreting old checkpoints or packages.
Use Source → Compile → Run; long-lived trainers must not import Lance/PyArrow.
Local physics tests use exactly one environment; server scale tests are separately assigned.
Check Unison/preflight policy before any remote project launch; feature deployments must be isolated from mutable integration sync and verified against the exact feature commit.
Use W&B for new training unless the user explicitly opts out.
Never displace another user's GPU job, hold process, or tmux session.
Do not run Server2 production until its native host/VM crash issue is explicitly cleared.
Run persistent commands in inspectable tmux/background processes, record identifiers, verify startup, and return; no LLM sleep/poll loops.
Stage explicit files, inspect staged diffs, and commit coherent atomic checkpoints in the feature worktree.
Escalate incompatible source conventions, invalid physical contact labels, or unapproved semantic changes; preserve the evidence rather than substituting a fallback.

Execution-policy refinement (2026-09-08): operator/executor runs for this task use at least 20 turns or an explicitly scoped checkpoint contract; the default 6+1 bounded recipe is not appropriate for this T3 continuation.

## M3 near-term versus final acceptance (2026-09-08)

Near-term milestone is the shortest runnable formal PPO/GAE learner on one or a few representative cube2:02 references. N=1 is an honest first learner when batching blocks; preserve real 28-DOF actor control, canonical clock/contact/PPO, immutable source, and W&B provenance. The inspected identities cube2_02_2833, cube2_02_2835, and cube2_02_2837 remain TRAIN; multi-reference rotation, held-out learned evaluation, vectorized throughput, BC and ablations follow after the first learned rollout.

Final acceptance is stronger: one frozen neural checkpoint must accept a new held-out hand-object reference from full start without per-reference retraining/fine-tuning, manual contact masks, or special reward. New action, geometry and multi-object generalization are separate later evaluations. Max lift plus endpoint pose is not full-task success; report sustained airborne genuine hand-object contact, matched path, slip and release diagnostics.

## Phase-only correction checkpoint (2026-09-09)

Parent's 67M run/eval is a failed learning result (0 contact, 0 lift), not an authorization to alter PPO, BC, reward, controller, clock, geometry, or initialization. The prior clipping/log-probability theory is explicitly ruled out by deployed installed-GaussianMixin identity evidence. The sole confirmed implementation correction is unique-table phase encoding: divide index by T-1, not 28-1. v3.1 is the unchanged-width corrected observation ABI and old v3 phase-bug checkpoints/normalizers require an explicit legacy reader. The next discriminating run is phase-fixed-from-scratch under the same 67M budget.

## Teacher-imitation initialization slice (2026-09-09)
The phase-fixed 67M PPO result remains a failure: no contact/lift and full-start wrist RMSE 0.1798 m. This bounded slice does not alter reward, physics, clock, PPO semantics, observation width, or the neural actor architecture. It tests the narrower hypothesis that the existing MLP needs a physically valid approach/contact action prior: collect exactly one full-start real MJX-Warp rollout from TRAIN identity `cube2_02_2833`; target each pre-action observation with `clip((aligned_ref_q[t+1]-previous_command)/(rate_per_second*dt), -1, 1)`; fit only the policy mean offline; then warm-start PPO from weights only after v3.1/package/split/witness validation. Teacher controls never enter PPO or frozen neural evaluation.

## Fixed-config optimizer continuation (2026-09-09)

The shared v3.1 67M run is extended for learning time only: preserve its shared
model, Adam state, fixed physics/control/reward/observation/PPO configuration,
and all established package/split/witness/clock contracts. Continue from update
256 through cumulative update 2048 using an additional 1792 updates at
8192×32 transitions, with the old checkpoint retained unchanged. The optional
separate critic remains held out. The continuation must explicitly record that
optimizer/model state continues but the MJX runtime begins full-start episodes;
old checkpoint CUDA RNG absence cannot be represented as bit-exact continuation.

## Raw action likelihood repair (2026-09-09)

Fixed-config optimizer continuation retains the shared critic OFF, reward, controller, clock, PPO objective, learning rate, rollout shape and long-window intent. The only numerical correction is the batched PPO action contract: retain the raw Normal sample and its raw Gaussian log-probability in PPO memory; pass a clipped copy only to physical execution. This preserves commands for identical samples while removing the invalid boundary-density likelihood. Optimizer resume from the update-288 legacy checkpoint restores the same weights/Adam/RNG at the update boundary and records the future raw-likelihood correction in checkpoint metadata. Every update must detect non-finite model/optimizer/metric state before successful checkpointing, preserve named finite snapshots, write a diagnostic, and raise.

## v4 implementation boundary (2026-09-10)

The active implementation contract is the pinned cube2 `manorl.autonomy.*.v4` seven-block 957/829 Warp-only vertical slice. v3.1/538-D checkpoint and cache contracts are rejected.

## v4 single-reference PPO restoration (latest scope)

Restore the smallest usable public v4 `train`/frozen `evaluate` route for TRAIN identity `cube2_02_2833`. Reuse mature RlGamesPPO GAE/clip/Normal/Adam logic rather than reimplementing returns. PPO memory must retain raw 957 observations and raw Normal actions; PointNet remains registered/trainable and checkpointed. Reset is driven by `runtime.last_done`; terminal observation/reward is stored before reset and finite-horizon bootstrap remains correct. Frozen evaluation begins at reference frame 0 and reports natural termination; its optional full-horizon diagnostic labels continuation after that boundary.

Authorized runtime evidence is exactly one local CPU N=1 two-update integration test and a short frozen load/evaluate pass. The previous long-training stop remains: no server, remote, long GPU, W&B/network training, or reward/physics/clock/geometry tuning follows from this implementation. Real CLI training defaults W&B enabled; the local test may explicitly disable it. The B4096 GPU public surface exposes num-envs, persistent workspace, CCD121 and current njmax512 allocation contract. Cache float hashes are recorded separately and do not block compatible package/asset/source/ABI evaluation.

## v4 learning-rate follow-up (latest bounded scope)

Expose optional `train --learning-rate` (finite, positive; default `3e-4`) through
the canonical PPO builder and metadata, with model-only warm-start preserving
the selected fresh-Adam rate. No algorithm, reward, physics, normalizer, model,
or other logging changes. Validate with CPU fake adapters only. The completed
B2048 run has no lift; fixed-data N=1 attribution identifies actor-update drift,
not critic dominance, and motivates `3e-5`. The user authorizes a future GPU1
B2048 run; launching it remains parent-owned and outside this implementation.
The overall empirical autonomy objective remains unchanged.
