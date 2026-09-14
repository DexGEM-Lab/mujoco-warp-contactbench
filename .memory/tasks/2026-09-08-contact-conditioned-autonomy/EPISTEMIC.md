# Multi-reference runtime milestone

Parent evidence: anchored trained2833 completes with359 loaded-contact and126
airborne frames; all five held-out identities fail with0 airborne despite up to99
contacts. The current hypothesis is that single-identity/frame200 supervision
teaches identity-specific phase behavior. Fixed round-robin all40 TRAIN reference
conditioning and current-frame contact-intent gating are the next intervention,
not evidence of learned held-out success.

The new bank preserves per-reference lengths, frame0 hand/object state, own
reference future/contact observations and same-identity reset. Global contact
storage remains global. Synthetic per-reference comparisons and CPU N1 bank-one
versus single physical rollouts have bitwise-equal qpos/raw957/reward/done over8
steps. CPU two-reference8-step/reset validation is recorded in OPS. Checkpoint
ABI is unchanged; ordered assignment is added to provenance and equality-gated
on optimizer resume. CUDA-scale startup memory/throughput and all40 physical
training are unmeasured. Canonical contract: docs/manorl_autonomy_v4.md.

## Intent gate physical evidence

Current-frame max regional proximity*confidence*valid>=0.5 activates first at
frame162 on2833. The unchanged chase/+0.2rad recipe naturally completes538/538
with reason1,230 loaded-contact frames and130 airborne>5mm frames, all130 loaded;
maximum lowest-point clearance0.168939m, all states valid. Thus contact-intent
activation preserves the physical teacher's load-bearing mechanism despite
removing its identity-specific frame200 gate. Old reported baseline129 airborne
is parent evidence; this worker's130 is from saved natural-terminal rows.
This is analytical teacher feasibility, not learned multi-reference competence.
Full-start frozen policy evaluation on the five held-out references remains the
next decisive learned test after training. Evidence: OPS intent-gated validation;
physics-teacher.json and physics-teacher-rows.json.

Enabled old frame-gated optimizer checkpoints are intentionally rejected under
intent gating; use model-only warmstart for this changed supervision recipe.
Missing/explicit disabled old recipes resume disabled. A125-test v4/teacher/batch
suite and7-test bank/telemetry follow-up pass. No GPU/server/W&B work occurred.

## Earlier anchor evidence


The active bounded intervention is optional online teacher-action anchoring after
PPO. Parent evidence establishes that feasible-q chase plus +0.2rad finger squeeze
from frame200 can sustain the object (538 completed,129 airborne), while BC and
unanchored PPO lose closed-loop fidelity. This localizes a useful intervention to
supervision on states the learner actually visits; it does not establish that
this intervention solves the whole trajectory.

The new CPU N1 discriminator supports that narrow mechanism: after two16-step
updates from the squeeze-BC warmstart at seed0/LR3e-4/separate critic, beta1/two
passes reduces mean-teacher MSE on identical control-rollout physical observations
from0.6698881984 to0.4651518166 (30.56%). Both arms are finite and valid. Only
32 approach frames were sampled, so neither gated squeeze nor grasp was tested.
Full-start frozen policy-only contact/lift/transport remains the decisive next
measurement for any trained candidate. Evidence: OPS “Optional online
teacher-anchor PPO”; artifact metrics.json in feature teacher-anchor-ppo outputs.

Teacher labels use the current previous command and next-frame feasible-q,
joint-limit clipping and physical rate/control_dt; they never execute. Anchor
passes update PointNet/trunk/mean through the same Adam after PPO, and None
gradients prevent value-only/log_std momentum updates. Raw Normal PPO collection,
reward, observation, architecture and physical execution remain unchanged.
Beta0 makes no label calls or extra steps. Old missing anchor metadata means
disabled defaults on resume; enabled continuation preserves the recipe and RNG.
117 focused tests include actual label arithmetic, full sample coverage, critic
isolation, metadata and old/enabled continuation. Docs are canonical at
`docs/manorl_autonomy_v4.md` → “Optional online teacher-action anchor”.

## Earlier physical and persistence evidence

The v4 command map now uses directional anti-windup: preserve previous servo
command under push-back, block only outward delta at/beyond the margin, allow
reverse delta, then clip joint limits. The measured-state hard re-anchoring
mechanism was real, but fixing it alone did not produce lift in the identical
cube2_02_2833 teacher pursuit. The 538-frame CPU N1 rollout still naturally
terminates at228/reason2 with33 loaded-contact frames and0 airborne>5mm frames;
full diagnostic has200 loaded frames and peak bottom clearance only0.204mm.
An independently reset natural run confirms228/reason2. At228 the object is
at z0.02354m while its target is0.12422m and paired contact has disappeared.
The remaining question is why feasible-q pursuit loses load-bearing contact
before the desired rise; this A/B does not establish exact-reference
infeasibility or neural learned competence. Evidence: OPS “Directional
anti-windup validation and negative A/B”. Existing v4 checkpoint formats remain
unchanged, but their future physical execution uses the corrected command map;
old physical traces need not reproduce. No new training is launched by this
bounded implementation.

The overarching objective remains empirical autonomous grasp/lift/transport
replication. The B2048 W&B run `bd9dd2d7` completed 33,554,432 transitions
without lift. The user requests longer RL training; the parent targets cumulative
4096 PPO updates at fixed B2048×32, LR 3e-5, four epochs and 16 minibatches.
The active update512 lower-LR job and its handoff are parent-owned. This worker
implements continuation and tests it on CPU fake adapters without launching
physics, GPU, remote or W&B jobs.

The v4 resume contract preserves exact model/full Adam, sampling RNG and
cumulative policy-step/transition counts. Fixed config, full architecture,
physical/package/ABI provenance and finite complete optimizer state are checked
before creating a W&B writer. Physical Warp state is absent from checkpoints:
resume starts full-start new episodes and resets episode/telemetry accumulators.
RNG restoration occurs after that reset, immediately before the first action.
An explicit reset-boundary fixture reproduces the uninterrupted next samples,
model and Adam exactly; this does not establish bit-exact ongoing physical
trajectory continuation. CPU inspection of the actual old update192 artifact
validates all 29 Adam states (step 12288), policy_steps 6144 and 12,582,912
transitions for target4096. W&B resumes only an explicit existing ID with
`resume=must`; the parent must prevent overlapping writers. Evidence: OPS entry
“v4 persisted optimizer continuation”.

v4 raw state is 957-D. The registered PointNet converts only the raw cloud to
the 829-D actor/value feature inside `AutonomyActorCritic`; PPO memory retains
raw 957 observations. All 28 commands remain policy-owned. PPO records raw
Normal samples and their raw Gaussian likelihoods; clipping occurs only at the
physical adapter boundary. The direct CPU two-update run produced finite losses,
gradients/Adam state and a v4 checkpoint. This validates wiring, not learning.

The PPO loop preserves the mature RlGamesPPO GAE/clip/Normal/Adam path. It
records post-step terminal observations/rewards before `prepare_action`; reset
is driven solely by `runtime.last_done`. Physical terminal transitions do not
bootstrap; rollout cuts do. `pending_reset` is retained only as a runtime
structural field and cannot control PPO resets.

Frozen load rejects old v3/538-D contracts and validates model architecture plus
package/catalog/manifest/split provenance. v4 checkpoints contain raw-action
sampling metadata, PointNet state, optimizer state, config, source/asset and
package provenance, and available RNG. Cache float hashes may differ across
supported builds; they are recorded but not equality-gated. Compatibility gates
source/asset/package/ABI/clock contracts instead.

The current v4 cache/state model remains: compiled hand mesh surfaces are
Warp-FK-derived, object collision geometry is body-local, raw SI q drives
control/reward, and cache geometry/velocities are physically named. The B4096
allocation contract remains `--ccd-contacts-per-world 121`, global CCD scratch
495616 and `naconmax >= naccdmax`, with `njmax=512` per world.

N1 result: two updates × eight control transitions on CPU for frame-0
`cube2_02_2833` were finite/valid with rewards 2.213626 and 2.188532;
hand-object force was zero, as expected at 16 untrained transitions. A frozen
four-control evaluation loaded the produced v4 model+PointNet checkpoint and
returned 8.824434 with no natural termination in those four controls. This is
not a reward-tuning signal and does not change physics/reward/clock/geometry.

The offline v4 teacher checkpoint for `cube2_02_2833` is the current strongest
initialization signal: frozen neural means produced 175/538 paired-contact
frames and 9.64 mm peak height delta, but naturally terminated at index 228
with object-deviation reason 2. The production PPO route now accepts this as a
strict model-only warm-start after validating architecture, package,
manifest/catalog, split, identity, asset, ABI, and physical clock provenance.
The teacher optimizer, progress, and RNG cannot enter PPO; a new full
actor/value Adam is constructed before the weights load. The teacher's
`full_horizon_diagnostic=true` is not a physical clock field and is therefore
recorded but excluded from the equality gate; control/physics timestep and
substeps remain equality-gated.

The fixed-data N=1 first-32-control diagnostic reproduces the canonical real
RlGamesPPO update exactly (injected-versus-canonical maximum parameter delta
zero). Late-phase KL is 0.085064 for the full update, 0.086622 actor-only, and
3.739e-6 value-only. Actor updates, rather than critic dominance, explain this
measured drift. Reducing only the step size to `3e-5` lowers late-phase KL to
0.000847 on the same update. This is a one-update sensitivity result, not a
historical B2048 replay or proof of grasp: physical evaluation still fails
without lift. The next intervention is the otherwise fixed B2048 lower-LR
follow-up from the same model-only teacher initialization. Actual trajectory
competence requires frozen physical contact/lift/transport evidence. Provenance:
OPS entry “v4 learning-rate interface and fixed-data attribution”.
