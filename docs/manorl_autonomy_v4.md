# Contact-conditioned autonomy v4 (cube2 vertical slice)

This document describes the runnable single-reference v4 PPO path and its
frozen ABI. It is an implementation/one-small-update validation surface, not a
claim of learned grasp competence.

## ABI and provenance

`manorl.autonomy.observation.v4` has seven raw blocks:

| block | raw slice | width |
|---|---:|---:|
| actual | `0:119` | 119 |
| reference | `119:228` | 109 |
| future | `228:351` | 123 |
| geometry | `351:703` | 352 |
| object-local collision point cloud | `703:895` | 192 |
| action type | `895:945` | 50 |
| object geometry | `945:957` | 12 |

A fixed-weight PointNet replaces only the `64×3` raw cloud with 64 features,
yielding the actor/critic width 829. v4 checkpoint metadata must name the v4
observation, action, reward and checkpoint contracts. v2/v3/v3.1 normalizers
and checkpoints are rejected rather than reinterpreted.

The cache is T-only and hashes its cache inputs. It contains feasible/raw q,
body-origin poses, COM-filtered object velocity, palm-origin velocity, 16
material anchors, `delta_ref`, reference signed gap/proximity/confidence/valid,
future tables, static collision cloud, action ID, support shift, normalized
object geometry, and the maximum collision-vertex radius about the object COM.
Identity is never embedded in policy input.

## Geometry and contact semantics

The reference compiler uses `mjx.make_data`, `mjx.forward(..., impl="warp")`,
compiled collision geom metadata, and the exact cube2 collision box. It neither
creates `MjData` nor calls `mj_forward`, `mj_step`, or `mj_geomDistance`.
Reference cache gap/proximity/confidence/valid are immutable demonstration
features, not live nearest distances. For each region,
`e = delta_actual^O - delta_ref^O`; exact replay is therefore zero even for a
positive reference gap or reference penetration.

The actual block’s last three values are **the all-contact resultant on the
object**, including table support. The geometry block’s force triplet is the
separate, canonical **hand-region → object paired force**; table/self contacts
cannot enter it. The pinned reducer accepts only pyramidal `condim=3` buffers,
asserts contacts/constraints remain below capacity, and consumes same-forward
`geom/world/dim/address/friction/frame/pos/efc.force` buffers. Other cone or
dimension combinations fail closed pending bundled Warp-helper parity.

Object geometry uses existing `geometry_encoding` semantics: four normalized
category slots `[box xyz, cylinder diameter-diameter-height, sphere diameter³,
irregular xyz] / 0.2`, clipped to `[0,1]`. Pinned cube2 populates box `[0:3]`
from its collision extents; no mystery zero vector is used.

## State/reward timing

Each action maps only `(actual q, previous command, clipped actor action)` to
its servo command. Directional anti-windup integrates
`delta = clip(action, -1, 1) * dofRate * control_dt` from the previous command.
With `error = previous_command - actual_q`, it zeros delta only where
`abs(error) >= antiwindupError` and `error * delta > 0`. At or beyond the
margin, outward integration stops while reverse motion remains available.
The target is never re-anchored to measured q; existing load-bearing servo
error is preserved even if contact pushes q farther away. A step starting
inside the margin can cross it; only final joint limits clip the target.
The resolved rate/margin arrays and 120 Hz control / 480 Hz physics clock are
unchanged. Existing v4 checkpoint formats remain loadable; their subsequent
execution uses this corrected map, so old physical trajectories need not replay.
It takes four `mjx.step` calls and then exactly one
`mjx.forward` before extracting q, origins, COMs, velocities and contacts.
The object COM/origin and anchor-point formulas use `xipos`, `subtree_com` and
`cvel`; anchor velocity includes the required co-rotating `-omega_O×delta` term.

Reward evaluates `s(t+1)` against `ref(t+1)`. The raw object terms are the
axis-position total `1.2`, shortest-angle piecewise orientation, and world
COM/angular-velocity match. All three are multiplied by the same **reference
motion gate**, never by actual-object speed:

\[
v_{\mathrm{eff}}=\sqrt{\lVert v_{\mathrm{ref,COM}}\rVert^2+
(r\lVert\omega_{\mathrm{ref}}\rVert)^2},\qquad r=0.0433004241\;\mathrm{m}
\]

\[
u=\operatorname{clip}((v_{\mathrm{eff}}-0.01)/0.09,0,1),\quad
w_{\mathrm{obj}}=0.01+0.99u^2(3-2u).
\]

Thus a static reference retains 1% object tracking, the gate reaches 50.5% at
`0.055 m/s`, and it is 100% at/above `0.10 m/s`. One deliberate exception:
while the reference is static (`v_eff <= 0.01`), an object rotation error
against the reference above `45 deg` escalates only the rotation term's gate
from `0.01` to `0.75`; position and velocity keep `w_obj`, and a moving or
transitioning reference keeps `w_obj` for all three. At exactly 45 deg the
unweighted rotation term is `-0.1` (its linear segment's floor onset), so the
override turns a persistent gross twist during a static hold from `-0.001` to
`-0.075` per control. Within the gate, object world position carries coefficient `0.5` (native
maximum `1.2 -> 0.6` at full gate) and object world COM/angular velocity
carries coefficient `4.0` (native maximum `0.1 -> 0.4`); the rotation term
keeps unit coefficient.
Hand--object relative pose
has coefficient `0.125`; feasible fingers retain `0.2`; reference-contact
anchor correspondence has coefficient `1.2`; action penalty remains
`-0.002 mean(clipped_action²)`; survival remains `0.001`; and a deviation,
fall, or non-finite physical/contact state adds severe `-75`. The severe
component preserves the existing termination semantics and is added once to
the other terms. It has no force-magnitude, table-contact, slip, or hidden
contact reward. Reasons are complete=1, deviation=2, fallen=4, nonfinite=8.

The exact parameter set is emitted as
`manorl.autonomy.reward.v4.reference-speed-gated.contact-priority.static-rotation-override.half-object-position.quadruple-object-velocity.v1`
in new training provenance/config and frozen-evaluation provenance. It is deliberately
separate from the model ABI: an old frozen checkpoint can load, but an
evaluation performed with this runtime reports the new reward and must not be
compared numerically to its historical return.

## Validation boundary

`tests/manorl/test_autonomy_v4.py` independently checks slices/dimensions,
xyzw double cover and rotation 6D, exact-reference anchors, analytic rotating
anchor velocity, contact order/sign/table exclusion/all-object force/torque,
reward boundaries (including the reference-speed gate, contact-priority
coefficients, and `-75` severe component), multi-reference own-speed gathers,
and checkpoint rejection. The N=1 CPU smoke loads
`cube2_02_2833`, resets to finite raw `(1,957)` / encoded `(1,829)`, then
performs one finite Warp transition. No trainer, server, GPU scale job, or W&B
run is launched by this implementation.

## v4.1 cache/state correction

The initial v4 cache was corrected to use every compiled hand collision mesh
(`mesh_vert`/`mesh_face`) with its `geom_pos` and `geom_quat` transformed once
into the owning body frame. Cube2's source collision triangles are already
body-local and deliberately do not receive the object geom transform again.
JAX closest-triangle plus oriented convex-halfspace logic supplies signed gaps;
interior points project onto a mesh surface. Surface templates are deterministic
area-spread 128-per-region and 64-on-object samples, not AABB corners.

V4 compilation now calls `compile_model_metadata_only`, which retains static
MjModel/XML validation but does not execute the legacy `MjData`/`mj_forward`
static-FK oracle. Warp FK validates the v4 path. Reference raw and feasible q
both receive the common support shift before FK, and cached velocities use
7-frame/poly2 filtering with quaternion-derived angular velocity. Runtime
observation stays raw 957; trainable PointNet/829 is a later model slice.

Raw SI q, not normalized q, is used in commands and reward. The shared v4
configuration resolves source `dofRate` and `antiwindupError`; qdot, servo
error and command envelope use their respective source quantities. Bottom
fields use collision vertices and the cached table height.

## v4 integration boundary

The environment and PPO memory ABI are raw 957. `AutonomyActorCritic` owns a
registered trainable `PointNetEncoder` and replaces only raw cloud `703:895`
with its 64-D embedding before both policy and value trunks consume all 829
features.

## PPO, checkpoint, and public commands

`tools/train_manorl_autonomy.py` has three public commands: `inspect`, `train`,
and `evaluate`. `train` uses the existing `RlGamesPPO` GAE/clip/Normal/Adam
implementation. PPO memory stores raw 957-D observations and the unmodified
Normal sample; `BatchedAutonomyAdapter.step` alone clips a separate copy at the
physical boundary. The same raw action used to choose the command is recorded
with its original Normal log-probability.

The loop records `terminal_next` before `prepare_action()`. The runtime resets
from `last_done`, not the unused `pending_reset` field. Thus a physical terminal
transition has no bootstrap, while a rollout cut has the normal finite-horizon
value bootstrap. `PointNetEncoder` is a registered model parameter and is in
both the Adam optimizer and model state dict.

A checkpoint is `manorl.autonomy.ppo.v4` and requires the v4 action,
observation, reward, model-architecture, raw-sampling, model, optimizer and
RNG metadata; v3/538-D inputs fail validation. It records source and asset
commit, package/catalog/manifest/split, clock, identity and v4 ABI. Cache hash
is recorded independently but is deliberately not compared at load: equivalent
supported cache builds can differ in float bytes, whereas source/asset/package
and ABI/clock are the compatibility contract. New training and evaluation
artifacts also record the exact reward parameter set; it is provenance rather
than a model-loading gate so the released inference checkpoint stays usable.
There is no observation normalizer in this minimal route (`normalizer: null`).

Real GPU train commands default to W&B enabled and expose the B4096 controls:
`--num-envs`, `--persistentworkspace`, `--ccd-contacts-per-world 121`; runtime
keeps `njmax=512` per world. CPU integration commands explicitly use
`--no-wandb --no-persistentworkspace`.

`train --warmstart PATH` validates the v4 model architecture plus pinned
asset, package, catalog, manifest, split, identity, ABI, and physical clock
provenance. With the default shared critic it loads an exact same-architecture
model. `--separate-critic` selects the existing independent value trunk. A
shared-trunk teacher may initialize that target through the one supported
cross-architecture transfer: PointNet, policy `net`, `mean`, and `log_std` are
copied exactly, while `value_net` and `value` retain their fresh target
initialization. All other architecture direction or field mismatches fail.

PPO always constructs a fresh Adam over every target actor/value parameter; the
source checkpoint's optimizer, progress, and RNG are ignored. Outputs record
`separate_critic`, `mode=ppo_warmstart`, the resolved `warmstart_checkpoint`,
and `warmstart_transfer_mode` in both config and provenance. Frozen evaluation
reads the checkpoint's exact shared/separate architecture before strict model
loading.

```bash
TEACHER=outputs/manorl/contact_conditioned_autonomy/single-trajectory-cpu-20260909/teacher-bc-v4.pt

# Bounded CPU wiring smoke with the offline teacher-initialized v4 model.
python tools/train_manorl_autonomy.py train --device cpu --num-envs 1 \
  --no-persistentworkspace --updates 2 --rollouts 8 --learning-epochs 1 \
  --mini-batches 1 --no-wandb --warmstart "$TEACHER" \
  --checkpoint /tmp/cube2-v4-warmstart-smoke.pt

# Future GPU launch candidate; run only in an approved free GPU window.
python tools/train_manorl_autonomy.py train --device gpu --num-envs 4096 \
  --persistentworkspace --ccd-contacts-per-world 121 --separate-critic \
  --warmstart "$TEACHER" \
  --checkpoint outputs/manorl/contact_conditioned_autonomy/cube2_02_v4_separate_critic.pt

python tools/train_manorl_autonomy.py evaluate --device cpu --num-envs 1 \
  --no-persistentworkspace --checkpoint /tmp/cube2-v4-warmstart-smoke.pt --steps 4
```

`train --learning-rate` selects a finite positive Adam step size; the default
remains `3e-4`. Model-only warm-start preserves the requested fresh-optimizer
rate. Checkpoint config, W&B resolved config, and per-update `config/learning_rate`
record it. The fixed N=1 first-update attribution motivates this lower-rate
B2048 follow-up; it does not establish lift success:

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train_manorl_autonomy.py train \
  --device gpu --num-envs 2048 --persistentworkspace --ccd-contacts-per-world 121 \
  --updates 512 --rollouts 32 --learning-epochs 4 --mini-batches 16 \
  --total-transitions 33554432 --checkpoint-interval 64 --learning-rate 3e-5 \
  --warmstart "$TEACHER" --separate-critic --seed 0 --split-seed 0 --wandb \
  --checkpoint outputs/manorl/contact_conditioned_autonomy/cube2_02_v4_lr3e-5.pt
```

Frozen evaluation always resets at reference frame 0, follows deterministic
clipped means, and reports the natural first termination. Its optional
`--full-horizon-diagnostic` continues after that boundary only while preserving
`natural_first_termination` and an explicit diagnostic-boundary label.

## v4 telemetry and frozen physical artifacts

`train` reduces cached **post-transition** `last_reward`, `last_physical`, and
`last_contact` before `prepare_action()` can reset completed rows. The
`autonomy_v4_telemetry` accumulator keeps reward-component returns, episode
lengths, maximum bottom clearance, and loaded-contact frame counts per
environment across PPO rollout cuts; it emits episode return statistics only
when at least one physical episode completed. The named reward stream is
`reward/object_position`, `object_rotation`, `object_velocity`,
`hand_relative`, `fingers`, `geometry`, `action`, `survival`, `severe`, and
`total`. `total` is the finite v4 reward sum, including `severe` exactly once.

Physical keys distinguish the object-wide resultant (which can contain table
support) from the paired hand-region force and COM torque. Loaded contact means
paired force over 0.02 N; airborne means the actual lowest collision vertex is
more than 5 mm above the table. Active tangential-slip means use only regions
with a paired contact. Object clearance is actual bottom minus table; origin
lift is explicitly an origin delta and is not a clearance claim. Completion
reason bits are independently counted (`reference_complete=1`, `deviation=2`,
`fallen=4`, `nonfinite=8`); a pure bit-1 horizon is recorded as
`termination/horizon_only` and never classified as grasp success.

Raw-policy action magnitude, physical-clipping fraction, executed norm, and
measured command-envelope/antiwindup fractions are logged separately. Raw
magnitude uses the sum of absolute action elements divided by its explicit
28-elements-per-world denominator; `action/raw_abs_max` is the true largest
element, not a maximum of environment means. The contact reducer emits
per-region tangential velocity as `[B,16,3]`; telemetry takes its L2 norm once
to form the `[B,16]` active-slip scalar. Invalid reward rows are excluded with
`where(valid, residual, 0)`, so a `NaN * 0` cannot contaminate reconciliation.
Native
RlGamesPPO losses/KL/std/lr remain named under their existing keys. Stats that
the PPO implementation does not expose directly (sample clip fraction,
gradient norm before clipping, advantage/return/explained variance) are absent
rather than inferred. On CUDA, the per-transition reductions and episode
accounting remain device tensors; completed episodes use fixed-size masked
count/sum/min/max aggregates rather than boolean gathers, and raw Torch actions
never take a NumPy conversion while constructing telemetry. Only one compact
scalar row leaves the device per update. `train` appends that row to `<checkpoint>.metrics.jsonl` regardless
of W&B and defines W&B's common `transitions` axis before logging. W&B config
also records resolved PPO values rather than only CLI flags: the PPO builder
uses one installed-default-plus-v4-override factory, while each update records
the instantiated agent's effective grad clip, Adam, GAE/discount,
clipping/loss, observation/value preprocessor status, unconditional GAE
advantage standardization, mixed-precision, rollout, epoch, and minibatch
values.

A separate critic still shares PointNet with the actor. Its loss values alone
therefore do not identify which gradient path dominates; telemetry preserves
those losses as observations and does not change the PPO algorithm in response.

`evaluate` defaults to one environment and writes its JSON summary plus a
compressed `.npz` beside `--trace` (or `--artifact`). The NPZ contains actual
and reference object/palm positions, actual/raw/feasible qpos, all nine reward
terms with their `reward_term_names` labels, paired and object-all forces,
bottom clearance, reason codes, and the natural-prefix length. Each JSON trace
row records measured `finite` and runtime `valid` status. Evaluation rejects
anything other than one environment and positive explicit step counts. It
preserves a natural terminal prefix separately from any diagnostic continuation.

The fast contact reducer now asserts the static pyramidal cone, requires
ownership-disjoint geoms in real models, rejects unsupported cones, and skips
live contacts with inactive `efc_address=-1` without allowing masked NaN
positions into a lever arm. General cone/helper parity and real slip diagnostics
are still separate work.

## Continuing a v4 PPO run

`train --resume-checkpoint CHECKPOINT --updates TOTAL` restores exact model,
full Adam and Python/NumPy/Torch/CUDA sampling RNG at a completed rollout
boundary. `TOTAL` is cumulative, not additional: resuming update512 to4096
runs updates513–4096 (3584 additional; 268,435,456 total B2048×32 transitions).
Use the same package path, identity/split seed, device, environment/rollout
counts, PPO epochs/minibatches/LR, critic architecture and runtime settings.
Defaults do not inherit from the checkpoint: any fixed CLI drift is rejected.
`--warmstart` and `--resume-checkpoint` are mutually exclusive. Teacher Adam,
missing/partial optimizer state, non-finite tensors, architecture conversion,
non-null normalizers and old ABI contracts cannot enter resume.

After the previous writer has exited successfully, an operator can extend its
run using the original fixed arguments, replacing the initialization/budget:

```bash
# Keep the original --package, --identity and all other fixed arguments.
python tools/train_manorl_autonomy.py train \
  --package "$ORIGINAL_PACKAGE" --identity cube2_02_2833 \
  --device gpu --num-envs 2048 --rollouts 32 \
  --learning-epochs 4 --mini-batches 16 --learning-rate 3e-5 --separate-critic \
  --persistentworkspace --ccd-contacts-per-world 121 --seed 0 --split-seed 0 \
  --resume-checkpoint "$COMPLETED_CHECKPOINT" --updates 4096 \
  --total-transitions 268435456 --checkpoint "$NEW_OUTPUT" \
  --wandb-run-id "$EXISTING_RUN_ID"
```

W&B requires an explicit `--wandb-run-id` or `WANDB_RUN_ID` and online
`resume="must"`; it cannot silently create a new resumed run. The previous ID
is equality-checked when saved in the checkpoint. Older checkpoints need the
operator-supplied ID. Previous configuration and provenance remain in resume
lineage; the new budget is updated with `allow_val_change=True`. The operator
must ensure there is only one writer. CPU tests may explicitly use `--no-wandb`.

Physics state is not checkpointed. Every continuation declares
`physics_restart="full_start_new_episodes"`; episode returns, lengths and
telemetry accumulators reset. RNG is restored after runtime reset, just before
the first action. This preserves optimizer/sampling progress but does not
continue an unfinished physical trajectory bit-exactly. Rows, W&B history,
checkpoint suffixes and agent timesteps use cumulative counters; only new
updates are returned. Resume lineage includes source SHA256/commit, previous
counts and warm-start history. The summary distinguishes `start_update`,
`completed_updates` and `additional_updates`.

`inspect_v4_resume` loads tensors on CPU and validates model/Adam/config and
physical provenance without constructing physics or rollout storage. This
supports inspecting a GPU checkpoint on CPU; actual training retains the fixed
device, and CUDA resume requires the saved logical visible-device RNG count.
Only trusted local Torch checkpoint files are supported.

## Object-type autonomy boundary

The same v4 route accepts one shared object type selected by the package.
Reference identity names provide the object/action pair; the runtime compiles
that object's pinned collision geometry and does not hard-code cube2. Current
capacity/persistent CCD settings remain the cube2-validated `121` contacts per
world; new object types must be smoke-checked before B4096 production because
mesh pair counts can differ. `--identity` remains the provenance witness, and
all-package training still assigns every reference round-robin.

## Optional online teacher-action anchor (training supervision)

`train --teacher-anchor-beta 1 --teacher-anchor-passes 2` adds mean-action
supervision **after** each canonical PPO update, using the same Adam. The default
beta is zero: no teacher queries, label collection, random permutations or extra
optimizer steps occur. Frozen `evaluate` remains policy-only.

Labels are computed at each rollout's pre-action observation, from that world's
current previous servo command. With `next = min(index + 1, last_frame)`, the
joint-limited target is `q_feasible[next]` plus 0.2 rad on flexion joints
`[7,9,10,13,14,15,17,18,19,21,22,23,25,26,27]` when the current frame's
`max_regions(proximity * confidence * valid) >= 0.5`. The gate is evaluated
independently for each assigned reference and turns off when intent falls below
threshold; there is no frame-number gate or latch. The label is
`clip((target - previous_command) / (rate * control_dt), -1, 1)`. This label
never enters the physics step; only the policy's sampled action executes.

Each anchor pass randomly partitions all `rollouts * num_envs` stored pairs into
`mini_batches` batches, retaining remainder samples. It minimizes
`beta * mean((policy_mean - teacher_action)^2)` with independent zero-grad,
backward, gradient-norm clipping at 0.5, and Adam step. PointNet, policy trunk and
mean head receive gradients. Value-only parameters and `log_std` have `None`
gradients, preventing their existing Adam momentum from updating them during the
anchor. Shared features may still change value predictions. PPO's raw Normal
likelihood, rollout memory, GAE, rewards and control map are unchanged.

Checkpoint config/provenance and resolved telemetry store beta, passes and the
complete squeeze recipe under `teacher_anchor`. Update telemetry exposes
`config/teacher_anchor_beta`, `config/teacher_anchor_passes`, squeeze constants,
`teacher_anchor/samples`, `teacher_anchor/rollout_steps`, and optimizer-step
counts. `teacher_anchor/mse` is the sample-weighted **pre-minibatch-update** MSE
across all anchor passes, not a frozen post-update evaluation score.
Old checkpoints without anchor fields resume with the disabled defaults;
disabled explicit old frame-gate recipes also resume disabled. Enabled
frame-gated checkpoints require model-only transfer to the new intent recipe;
optimizer resume requires an unchanged enabled anchor configuration. To introduce an
anchor to an existing policy, use model-only `--warmstart`, not fixed-config
`--resume-checkpoint`.

This is an opt-in training intervention for closed-loop drift. An approach-prefix
MSE improvement alone does not demonstrate physical grasp or contact fidelity;
acceptance of a learned policy still requires full-start policy-only evaluation.

## Multi-reference training modes

`train --all-train-references --split-seed 0` assigns environments round-robin
in `identity_split(...)["train_indices"]` order over the 40 TRAIN identities.
This preserves the original TRAIN-only multi-reference experiment.

`train --all-references` assigns environments round-robin over every trajectory
in the supplied package. The pinned cube2:02 package contains 50 references.
Provenance records `reference_assignment.mode=all_package_references`, the
ordered identities, and the reference count. Use B4096 for training when
sampling all 50; reference count and environment count are independent.

Both modes require shared cube2 object geometry; the all-package mode permits
multiple cube2 action IDs and stores action identity per reference. The single-
identity CLI remains the default, and frozen evaluation still selects one identity.
The v4 checkpoint, raw957/PointNet/PPO, reward and physical clock are unchanged.
Model-only warm-start retains the existing identity witness; optimizer resume
requires an unchanged ordered reference assignment and mode.

The bank stores time-varying cache fields as device `[R,T,...]` arrays, padded
with the final row only for storage. Each environment clamps gathers and ends
at its own reference length, including future-validity and progress features.
Joint limits are stacked per reference; geometry/action constants must agree.
References are conditions, never additions to executed policy commands.

Each environment starts at its assigned reference frame0 and retains that
identity throughout training. Subset resets restore only qpos/qvel/ctrl for
finished worlds, then forward the global Warp contact arena. Identity rotation
and heterogeneous geometry are outside this contract. Reference bank gathering
and telemetry stay on-device; the CUDA adapter retains DLPack transfer.

CPU teacher check on `cube2_02_2833`: intent gating first activates at frame162,
and natural execution completes538/538 transitions (reason1), with130 frames
above5mm bottom clearance, all130 carrying loaded hand-object contact. This
validates the analytical supervision recipe, not learned multi-reference grasp.
