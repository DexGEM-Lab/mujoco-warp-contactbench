# ManoRL accepted-parent approach and retreat augmentation

## Purpose

This synthesis-only feature augments one row that already succeeded in a prior
compact v2_contact publication. The accepted parent binds the exact source row,
checkpoint SHA256, 120 Hz reference clock, successful object XY offset, source
row frame-0 right-hand pose, and a validated retreat anchor. Training, strict
resume, and ordinary viewers keep their existing contracts.

The output reference has three control regions:

```text
approach prefix       pure reference; policy/residual/deviation terminal disabled
original task body    deterministic parent checkpoint policy; normal deviation terminal
retreat tail          policy/action disabled; prior residual smoothly discharged; deviation enabled
```

The base trajectory must use pre60/post250 at 120 Hz. Approach frames are
prepended; the retreat deforms wrist XYZ after a fixed parent-derived anchor
without changing trajectory length.

## Accepted-parent v3

Create the descriptor on a Lance-healthy host:

```bash
python tools/select_manorl_synthetic_parent.py \
  --input /path/prior-successes.lance \
  --row-uuid <accepted-row-uuid> \
  --output /path/accepted-parent.json
```

`manorl_accepted_synthetic_parent_v3` records:

- parent dataset version, row UUID/index, and row contract;
- exact source dataset version/row/identity;
- checkpoint SHA256/update and parent episode provenance;
- successful object XY offset;
- raw source-row frame-0 right-hand `q_ref[3:28]`;
- parent movement-end and last solved right-hand/object contact diagnostics;
- retreat anchor contract, source frame, and anchor→final horizontal distance.

A parent is rejected when its last solved contact occurs before its movement-end,
when movement-end+15 does not exist, or when no nonzero horizontal retreat
direction remains after that anchor.
The retreat anchor is deterministic:

```text
parent anchor state = parent movement-end state + 15 frames
```

The selector maps that state through
`reference.source_frame_index[anchor_state]`. It also verifies the persisted
command/reference/source mapping. Contact does not choose the anchor; it only
qualifies that the parent remained in contact through its declared task motion.

## Approach modes

Select the mode with `--approach-mode far|near` or
`MANORL_SYNTH_APPROACH_MODE=far|near`. The default is `far`.

### Far close

The start is sampled around the initial object:

```text
horizontal radius: 0.30–0.70 m
world-up Z offset: 0.08–0.30 m
azimuth: initial object→pre60 hand direction ±30°
```

The episode seed is the far-approach seed, preserving the original far sampling
sequence.

### Near close

Near and retreat use the same endpoint-distribution family but independent
seeds. Near does not copy the actual retreat endpoint.

```text
near seed    = hash(episode seed, "near-approach")
retreat seed = episode seed
```

A near endpoint is sampled from the source retreat geometry:

```text
horizontal distance = original anchor→final XY distance + Uniform(0.03, 0.15) m
azimuth               = original retreat direction + Uniform(-30°, +30°)
height                 = original final wrist Z + Uniform(0.04, 0.10) m
```

The endpoint XY is represented relative to the final object, rotated by the
initial-vs-final object yaw, and translated to the initial object XY. Z keeps
the sampled retreat endpoint's absolute world height. This avoids translating
a hand below the table when the final object has been lifted above its initial
pose. Near and retreat remain independently sampled and normally differ.

## Approach hand pose and motion

Only start XYZ is sampled. Start `q_ref[3:28]` is copied exactly from the raw
source row frame 0:

```text
q[0:3]   sampled wrist XYZ
q[3:6]   raw source-row frame-0 wrist Euler XYZ
q[6:28]  raw source-row frame-0 finger joints
```

Wrist translation, wrist orientation, and all 22 finger joints use a quintic
trajectory to the pre60 frame 0 pose. The last prefix interval exactly matches
the pre60 frame0→frame1 interval, giving zero discrete splice-velocity error.
The source is rejected if raw frame-0 and pre60 wrist Euler coordinates differ
by more than π on any axis rather than silently changing the stored raw pose.

Prefix duration is the maximum required by:

- translation at nominal 0.30 m/s;
- wrist rotation at nominal 90°/s;
- maximum individual finger displacement at nominal 90°/s.

Duration receives ±10% jitter and is clamped by total pre-padding 100–360 frames
(base pre-padding is 60). A 4 cm endpoint-smooth wrist Z arc keeps the approach
clear of the table.

During the prefix:

```text
policy forward                              disabled
processed residual                          0
cumulative wrist/finger residual            0
0.10 m object-deviation terminal            disabled
right-hand/table solved contact >0.2 N      reject candidate
right-hand/object solved contact >0.2 N     reject candidate
```

At original pre60 frame 0, policy, residual, and deviation semantics resume.

## Retreat tail

The parent descriptor's movement-end+15 source frame is mapped into the augmented
reference. Frames through that anchor remain bit-identical. After it:

- the original per-frame wrist retreat is preserved;
- a discrete-C2 smooth endpoint displacement is added;
- wrist orientation and finger sequence remain the original tail;
- source indices, timestamps, object reference, movement window, and total
  trajectory length remain unchanged.

The retreat endpoint distribution is the one described under Near close. The
smooth deformation has zero added discrete velocity and acceleration at both
anchor and final boundaries. The policy-free tail window begins at the
movement-end+15 command itself:

```text
policy forward                    disabled
processed residual action         0
new residual accumulation         disabled
entry cumulative residual         quintic decay to 0 over the tail
0.10 m deviation terminal         enabled
```

Retaining the entry residual at the first tail command prevents a controller
jump; the quintic decay has zero endpoint slope and reaches exact zero on the
last command issued by ManoRL's delayed-terminal counter. The actual new rollout
may retain solved contact after the parent-derived anchor because randomized
approach changes closed-loop physics. The contract is therefore explicitly
**parent movement-end+15**, not the new rollout's own final contact. Computing an
augmentation from each candidate's future final contact would require a
two-pass simulation.

## Reproducibility and UUIDs

The sibling manifest records accepted parent v3, far/near config, near endpoint
config, retreat config, seed streams, and every accepted sample. Generated UUIDs
use `manorl_synthesis_augmentation_identity_v3`, a canonical SHA256 over semantic
parent identity, mode/config, episode/attempt, and resolved approach/retreat
seeds. Far and near outputs, or outputs with different bounds, cannot collide.
Machine-specific dataset paths are excluded from this content identity; pinned
versions/row identity/checkpoint digest remain included.

## Synthesis

```bash
MANORL_SYNTH_APPROACH_PREFIX=true \
MANORL_SYNTH_APPROACH_MODE=near \
MANORL_SYNTH_RETREAT_SUFFIX=true \
MANORL_SYNTH_ACCEPTED_PARENT=/path/accepted-parent.json \
MANORL_PREDECODED_MANIFEST=/path/pre60-bundle/manifest.json \
CHECKPOINT=/path/exact-parent-checkpoint.pt \
./synthesize.sh <object> <action> 1 <gpu>
```

The long-lived process consumes the predecoded bundle, not source Lance. The
checkpoint is loaded through explicit policy-transfer inference when its saved
padding/assets differ; strict training resume remains fail-closed.

## Viewer

```bash
python tools/view_manorl_approach_prefix.py \
  --checkpoint /path/exact-parent-checkpoint.pt \
  --accepted-parent /path/accepted-parent.json \
  --predecode-dir /path/pre60-bundle \
  --approach-mode near \
  --retreat-suffix \
  --seed 49 \
  --speed 1.0
```

The viewer uses one active MJX-Warp environment and reuses one policy runtime.
Each terminal increments the episode seed and reinstalls the next immutable
reference from frame 0. RESET telemetry reports mode, independent approach and
retreat seeds, approach geometry, prefix length, retreat anchor, and endpoint.

## Publication validation

Production format is
`synthetic_mano_target_replay_visual_v2_contact`. Before publication:

1. validate accepted-parent v3 and exact checkpoint SHA;
2. verify far/near bounds and independent seed streams;
3. verify raw frame-0 `q[3:28]` at approach start;
4. verify zero prefix and retreat splice-velocity errors;
5. verify prefix policy/processed/cumulative residual are zero; verify retreat
   policy/processed action are zero and cumulative residual decays monotonically
   from its entry value to zero without a controller-target jump;
6. verify prefix has no solved table or object contact above 0.2 N;
7. verify retreat anchor equals augmented movement-end+15;
8. verify success termination and normal deviation semantics in the tail;
9. validate compact contact/reference/command mapping and force-frame
   consistency with `tools/validate_manorl_synthetic_lance.py`;
10. preserve the sibling manifest and checkpoint metadata catalog.
