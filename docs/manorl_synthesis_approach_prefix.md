# ManoRL synthesis approach-prefix augmentation

## Purpose

The approach-prefix feature augments one row that has already succeeded in the
prior scalable synthetic Lance publication. The accepted parent fixes the
source identity, checkpoint SHA256, reference clock, and successful object XY
initial offset. Only a fresh hand approach is added; the canonical pre60
reference and all later content remain unchanged. It is synthesis-only: PPO
training, strict checkpoint resume, and ordinary viewers retain their existing
early30 contract unless this feature is explicitly enabled by the exporter.

Enable the default contract with:

```bash
python tools/select_manorl_synthetic_parent.py \
  --input /path/prior-scalable-successes.lance \
  --row-uuid <accepted-generated-row-uuid> \
  --output /path/accepted-parent.json

MANORL_SYNTH_APPROACH_PREFIX=true \
MANORL_SYNTH_ACCEPTED_PARENT=/path/accepted-parent.json \
CHECKPOINT=/path/to/the-parent-checkpoint.pt \
MANORL_PREDECODED_MANIFEST=/path/to/pre60-bundle/manifest.json \
./synthesize.sh <object> <action> 1 <gpu>
```

The corresponding direct exporter option is `--approach-prefix`.

## Reference structure

For a sampled prefix of length `L`:

```text
[0, L)          generated approach prefix
[L, L + 60)     original pre60 reference, unchanged
[L + 60, ...]   original movement/grasp/lift/place/post250, unchanged
```

The source pre60 trajectory is the reference truth. Its hand targets, object
targets, source indices, movement window, command mapping, and all later frames
are copied exactly after index `L`. The feature refuses a source whose base
movement start is not exactly 60 frames.

The generated prefix repeats the first source frame index and holds the initial
object reference pose. The output movement start/end indices are shifted by
`L`, so replay sees the same movement content at its new absolute position.

## Per-reset sampling

Accepted-parent synthesis always owns one active environment. Repeated synthesis
uses one fresh isolated process per attempt/reset. Attempt `k` uses
`attempt_seed = base_seed + k`; the bound source identity derives a stable RNG
stream from `(attempt_seed, source_identity)`. Therefore:

- the same seed and identity reproduce the same prefix exactly;
- a new synthesis reset/attempt produces another random start;
- failed candidates do not alter later source identities' random streams.

The companion manifest records the algorithm contract, complete configuration,
and accepted per-row samples. The row already records `seed` and
`source_identity`, making every start reproducible.

## Start distribution

The start is sampled relative to the object's initial centre with two
independent axes:

```text
XY radius (horizontal): 0.30–0.70 m
Z offset (above centre): 0.08–0.30 m
XY azimuth offset:       ±30° around the source object→hand XY direction
```

If `ψ0` is the source object→hand XY azimuth, sampled azimuth `ψ`, XY radius
`r_xy`, Z offset `z`, and object centre `o` define:

```text
ψ = ψ0 + Uniform(-30°, +30°)
r_xy = Uniform(0.30, 0.70) m
z = Uniform(0.08, 0.30) m
p_start = o + [r_xy cosψ, r_xy sinψ, z]
```

XY radius and Z offset are independent, so a tall/near start and a low/far
start are both reachable. The positive Z lower bound is 8 cm, rather than
merely greater than zero, because real MJX-Warp inspection showed low
approaches could develop right-hand floor contact before reaching the source
reference.

## Human-like motion

Prefix duration is derived from translation and orientation distance, then
jittered by ±10% and clamped through total pre-padding 100–300 frames:

```text
L = effective_pre_padding - 60
40 <= L <= 240
```

The defaults use nominal translation speed 0.30 m/s and nominal angular speed
90°/s. A quintic approach provides zero start velocity/acceleration. A 4 cm
endpoint-smooth vertical arc gives the wrist a human-like raised path and avoids
premature table descent.

The splice is exact in the discrete 120 Hz sequence: the last prefix interval
is equal to the original source frame0→frame1 interval. This produces zero
sampled velocity error at the boundary. Raw source acceleration is not forced
as an endpoint constraint; full-catalog validation showed that backward
extrapolation of capture acceleration noise produces non-human velocity and
angular-velocity spikes.

The wrist orientation template is selected from the source grasp family:

- `top_oblique`;
- `object_facing`;
- `source_neutral`.

Templates apply only a bounded source-relative perturbation (default norm at
most 12°) and return exactly to the source orientation. Finger targets remain
fixed at the original pre60 frame0 pose throughout the generated prefix.

## Action and terminal gates

The generated prefix is the synthesis early phase:

```text
trajectory_step < L:
    policy forward is not called
    processed residual action = 0
    cumulative position residual = 0
    cumulative joint residual = 0
    0.10 m object-deviation terminal disabled

trajectory_step >= L:
    deterministic checkpoint policy residual enabled
    cumulative residual enabled
    normal 0.10 m object-deviation terminal enabled
```

Thus the first original pre60 command uses the checkpoint policy and ordinary
termination semantics. Non-augmented environments preserve the existing fixed
early30 behavior, including its established exit-frame handling.

Synthesis does not train PPO. Its checkpoint stepper calls deterministic mean
policy inference and `env.step` only; it never calls `record_transition`,
`post_interaction`, or an optimizer. No PPO samples are written to memory.

## Checkpoint boundary

The accepted parent binds the exact checkpoint SHA256; selecting a different
checkpoint is a hard error. The checkpoint is a policy-transfer source, not the
reference ABI. A successful
pre180 or old-geometry checkpoint may drive a pre60/new-asset synthesis rollout
only through the explicit policy-transfer inference loader. That loader retains
fail-closed checks for checkpoint format, known reward/environment families,
model architecture, tensor shapes, and finite state, while deliberately not
claiming padding/asset signature equality. It transfers policy, value, and
normalizer modules without optimizer or PPO-memory state.

Ordinary viewer inference and strict training resume continue to require full
environment-signature equality.

## Candidate acceptance

The exporter rejects an augmented attempt if the controlled right hand develops
an actual solved floor contact above the production 0.2 N threshold during the
prefix. Broadphase candidates and contacts on a compiled reference-following
left hand do not count. Candidate success still requires source completion
without deviation failure after the prefix gate opens.

A positive wrist Z alone is not treated as proof of clearance; this solved-force
check is the physical acceptance criterion.

## Validation expectations

Before publishing a new dataset:

1. verify start radius, XY offset, and positive elevation bounds;
2. verify the exact source suffix, source indices, and object reference suffix;
3. verify zero discrete splice velocity error;
4. audit peak translation/angular speed over the selected catalog;
5. verify processed/cumulative residual is zero only in the prefix;
6. verify the 0.10 m terminal is disabled before `L` and live at `L`;
7. inspect at least one generated prefix in the native viewer;
8. validate compact v2_contact structure and target replay;
9. preserve the sibling manifest containing accepted per-row samples.

The generated row UUID is salted with parent row UUID, attempt seed, and
attempt number, so multiple augmentations of the same parent cannot collide.

## Interactive viewer

`tools/view_manorl_approach_prefix.py` opens one MuJoCo window and loops over
fresh approach-prefix episodes of the bound accepted parent. One MJX-Warp
environment and one checkpoint policy runtime are reused across every reset:

```bash
python tools/view_manorl_approach_prefix.py \
  --checkpoint /path/checkpoint-001000.pt \
  --accepted-parent /path/accepted-parent.json \
  --predecode-dir /path/pre60-bundle \
  --seed 49 \
  --speed 1.0 \
  --print-every 20
```

Each terminal advances `--seed`, reinstalls the next seeded
approach-prefixed reference, and replays from frame 0 in the same window.
Machine-readable JSON lines on stdout report `RESET` (sampled start, XY
radius, Z offset, prefix length), `FRAME` (progress and policy gate), and
`TERMINAL` (reason code). `--max-episodes N` stops after N episodes for
bounded smoke runs.

The production compact row contract remains
`synthetic_mano_target_replay_visual_v2_contact`; augmentation semantics live in
the sibling manifest while row seed/identity/reference arrays retain complete
replay and reproducibility evidence.
