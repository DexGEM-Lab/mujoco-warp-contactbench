# Current model

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
