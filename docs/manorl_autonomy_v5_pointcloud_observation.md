# ManoRL Autonomy v5 observation spec (point-cloud state)

Status: design release, 2026-09-18. Not yet implemented in code/ABI.

This spec supersedes the 352-dim hand-crafted contact-intent block of v4 with
hand and object point clouds, and removes the explicit slip-velocity channel.
Reward and teacher supervision keep using cache geometry directly and are
unaffected by the observation change.

## Design principles

- Geometric correspondence is learned by the point-cloud encoder, not
  hand-written per-region anchor features.
- Physical feedback that no cloud can provide is kept as explicit numbers:
  reference intent gating and per-region load transfer.
- References only condition observations, rewards, and training supervision.
  They never enter the executed action path. Frozen evaluation executes the
  clipped policy mean only.
- Teacher squeeze gating and the geometry reward use cache anchors directly,
  independent of this observation spec.

## Blocks (base version, no future clouds)

### 1. Actual state — 119 (unchanged from v4)

| Group | Dims | Scale |
|---|---:|---|
| `q_normalized` (28 joint positions) | 28 | joint limits |
| `qdot` | 28 | raw |
| `command_error = previous_command − q` | 28 | raw |
| object position / 6D rotation | 3 + 6 | 1.0 m |
| object COM linear / angular velocity (object frame) | 3 + 3 | 1.0 / 3.0 |
| palm relative position / 6D orientation (object frame) | 3 + 6 | 0.1 m |
| palm angular velocity (object frame) | 3 | 3.0 |
| bottom clearance: actual + same-frame reference | 1 + 1 | 0.1 m |
| gravity direction in object frame | 3 | — |
| total object contact force incl. table (tanh) | 3 | 5 N |

### 2. Reference current state — 109 (unchanged from v4)

- Reference fingers q, palm pose/velocities, object pose/velocities.
- Live actual-vs-reference errors (position/orientation/fingers/velocities/relative pose).
- Phase progress (`index/t`, `1 − index/t`) per assigned reference: 2.

### 3. Future numeric — 12 (shrunk from 123)

| Content | Dims |
|---|---:|
| per-horizon validity flag (6/12/24) | 3 |
| per-horizon object position delta (object frame) | 9 |

All other future geometry moves to the optional future point clouds.

### 4. Contact intent (kept physics feedback) — 80

| Content | Dims | Why kept |
|---|---:|---|
| `confidence` (reference contact cleanliness, soft gate) | 16 | cache intent |
| `valid` (reference contact usability, hard gate) | 16 | cache intent |
| per-region paired hand→object force (tanh) | 48 | only load-transfer evidence; no cloud gives force |

Removed from the previous 128-dim retained block: anchor relative velocity /
slip (48). Tracking rewards already penalize the consequences of slip, and
future in-hand-adjustment work requires intentional slip, so a hard-coded
"no slip" feedback channel would encode the wrong prior.

### 5. Point clouds

| Cloud | Points | Raw dims | Encoded dims |
|---|---:|---:|---:|
| Object surface cloud (existing) | 64×3 | 192 | 64 (existing PointNet) |
| Hand surface cloud (16 regions × 16 points) | 256×3 | 768 | 64 (new PointNet) |

First version uses one global PointNet for the hand cloud, mirroring the
object encoder; no per-region tokenization yet.

### 6. Action identity — 50 (unchanged)

Per-env action one-hot, gathered from the reference bank.

### 7. Object geometry — 12 (unchanged)

## Removed v4 geometry features

| Removed content | Old dims | Replacement |
|---|---:|---|
| `signed_gap`, `proximity` | 32 | point-cloud encoder learns approach |
| `region_anchor_hand` / `region_anchor_object` local anchors | 96 | hand + object point clouds |
| `delta_ref` (reference anchor offset) | 48 | reference cloud geometry |
| `error` (actual vs reference anchor offset) | 48 | current cloud vs reference cloud |
| anchor relative velocity / slip | 48 | removed by design decision |

## Optional future reference clouds

| Per horizon | Raw dims | Encoded dims |
|---|---:|---:|
| object reference cloud | 192 | 64 |
| hand reference cloud | 768 | 64 |

With 3 horizons (6/12/24): +2880 raw, +384 encoded.
Recommended first step: a single horizon (12) to validate benefit.

## Totals

| Version | Raw dims | Encoded dims |
|---|---:|---:|
| v4 (current) | 957 | 829 |
| v5 base (this spec) | 1342 | 510 |
| v5 + 3-horizon future clouds | 4270 | 942 |

## Boundaries

- This is a new observation ABI and a new model ABI. Existing v4 checkpoints
  are not loadable; a new checkpoint contract and fresh training are required.
- Teacher squeeze gating and the geometry reward continue to use cache
  proximity/confidence/valid and anchor errors directly; they do not read the
  observation, so the spec change cannot silently weaken training signals.
- Frozen evaluation remains: reset at reference frame 0, deterministic clipped
  policy mean, natural first termination, no diagnostic continuation.
