# ManoRL autonomy v4 reward — complete reference

Status: current implementation, 2026-09-21. Source of truth is
`sim/manorl/autonomy_v4.py` (`compute_reward`, `log_fraction`,
`motion_gate_weight`, the `REWARD_*` constants, `reward_parameters`). This
document describes **every** term, scale, gate and termination condition; if it
disagrees with the code, the code wins.

Contract ids:

- structural reward contract `manorl.autonomy.reward.v4` (term list/order, unchanged).
- parameter provenance `manorl.autonomy.reward.v4.reference-speed-gated.contact-priority.static-rotation-override.half-object-position.quadruple-object-velocity.log-curves.v1`,
  emitted into training and frozen-evaluation provenance by
  `reward_parameters()`.

## 1. Where the reward sits in the loop

- Control runs at **120 Hz**; physics runs at **480 Hz** with four `mjx.step`
  calls and exactly one post-step `mjx.forward` per control step.
- The reward for control step `t` is evaluated on the **post-action** state
  `s(t+1)` against the reference cache at the **post-action index** `ref(t+1)`.
- The policy emits a raw Normal action; the executed command and the reward
  both use `clip(action, -1, 1)`. The executed action enters the reward only
  through the action-penalty term.
- One scalar reward per environment per control step. PPO stores the raw
  observation and the raw action; the reward is used as-is (no normalisation,
  no clipping, no discounting inside the term).
- All quantities are SI: metres, radians, seconds. Joint positions come from
  raw `qpos`; the observation-side normalisations (`/0.1`, `/3.0`, …) are
  **not** used inside the reward.

Each step produces nine reward components plus `total`, `done`, `reason` and
`valid`; `total` is their sum, unless the step is non-finite (then it is the
severe value alone). The component order is fixed and is also the column order
of the telemetry/artifact `reward_terms` array: `object_position`,
`object_rotation`, `object_velocity`, `hand_relative`, `fingers`, `geometry`,
`action`, `survival`, `severe`.

```
total = pos + rot + vel + hand + fingers + geometry + action + survival + severe
```

## 2. The reference motion gate

The three object-tracking terms are multiplied by one gate driven **only by the
reference motion**, never by the actual object's speed, so the policy cannot
reduce its own tracking requirement by moving differently.

```
v_eff = sqrt( ||v_ref,COM||² + (r · ||ω_ref||)² )          r = 0.04330042412758056 m
u     = clip((v_eff - 0.01) / 0.09, 0, 1)
w_obj = 0.01 + 0.99 · u²(3 - 2u)                           smoothstep
```

`r` is the cube2 maximum collision-vertex distance from the object COM; it is
compiled from the exact collision mesh, cache-hashed, and required to be shared
across a reference bank.

| reference `v_eff` | 0 | 0.01 | 0.055 | 0.10 | ≥0.10 m/s |
|---|---:|---:|---:|---:|---:|
| `w_obj` | 0.010 | 0.010 | **0.505** | 1.000 | 1.000 |

A static reference therefore keeps 1 % of object tracking; the ramp is smooth
and monotone in between.

### 2.1 Static rotation override

A separate multiplier applies to the rotation term only:

```
w_rot = 0.75                    if v_eff <= 0.01 and rotation error > 45°
w_rot = w_obj                   otherwise
```

Position and velocity always use `w_obj`. The purpose is to stop a gross
orientation error from hiding behind the 1 % static floor. It is a step: at
44.9° the static multiplier is 0.01, at 45.1° it is 0.75.

## 3. The log ramp used by the three object tracking terms

```
L(e) = clip( log(1 + e/s) / log(1 + D/s), 0, 1 )        s = e50² / (D − 2·e50)
```

- `L(0) = 0`, `L(e50) = 0.5` exactly, `L(e ≥ D) = 1`.
- `e50` is the error that keeps half the reward; `D` is the error where the
  term reaches its floor.
- The ramp is monotone, has no flat top (the slope is largest at zero error)
  and no dead tail (the slope decays algebraically, not exponentially).
- `log_fraction` raises `ValueError` unless `0 < e50 < D/2`; all three
  parameter pairs satisfy this.

## 4. Object tracking terms

### 4.1 `object_position` — object world position, coefficient 0.5

Per-axis absolute position error against the reference object origin, each axis
run through its own log ramp, then combined with fixed axis weights:

```
err_axis = |p_actual − p_ref|            per axis, metres
pos = 0.5 · [ 0.2·(1 − L(err_x, 0.015, 0.10))
            + 0.2·(1 − L(err_y, 0.015, 0.10))
            + 0.8·(1 − L(err_z, 0.015, 0.10)) ] · w_obj
```

- `e50 = 1.5 cm` per axis, `D = 10 cm` per axis.
- `D` equals the deviation termination threshold, so a **single-axis** error of
  10 cm drives that axis to zero exactly where the episode would end.
- Axis weights `0.2 / 0.2 / 0.8` mean height (z) carries four times the weight
  of either lateral axis. The weights sum to 1.2, which is the full-gate
  maximum of the raw term; with coefficient 0.5 the maximum is **0.6**.
- The term is never negative.

| per-axis error | 0 | 0.5 cm | 1 cm | 1.5 cm | 2 cm | 5 cm | 10 cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| `L` | 0.000 | 0.271 | 0.408 | **0.500** | 0.570 | 0.809 | 1.000 |
| z contribution (max 0.4) | 0.400 | 0.292 | 0.237 | 0.200 | 0.172 | 0.076 | 0.000 |
| x/y contribution (max 0.1) | 0.100 | 0.073 | 0.059 | 0.050 | 0.043 | 0.019 | 0.000 |

### 4.2 `object_rotation` — object world orientation, coefficient 1.0

```
deg = shortest_angle(q_actual, q_ref) · 180/π        degrees, 0…180
rot = −0.5 + 0.8 · (1 − L(deg, 35, 90)) · w_rot
```

- `e50 = 35°`, `D = 90°`; value **+0.3 at 0°**, **−0.5 at ≥90°**.
- The only term that can be negative; it is a bonus for alignment and a penalty
  for misalignment.
- Multiplied by `w_rot` (§2.1), so the static override can raise a >45° error
  in a static hold from ×0.01 to ×0.75.
- `shortest_angle` uses the absolute quaternion inner product, so `q` and `−q`
  are equivalent and the error is always in `[0, 180°]`.

| orientation error | 0° | 5° | 10° | 20° | 25° | 35° | 45° | 60° | ≥90° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rot` (full gate) | +0.300 | +0.231 | +0.166 | +0.050 | −0.003 | **−0.100** | −0.188 | −0.304 | −0.500 |
| static reference (`×0.01`) | +0.0030 | +0.0023 | +0.0017 | +0.0005 | −0.0000 | −0.0010 | −0.0019 | −0.0030 | −0.0050 |
| static + >45° (`×0.75`) | — | — | — | — | — | — | −0.141 | −0.228 | −0.375 |

The zero crossing is at **24.7°**.

### 4.3 `object_velocity` — object world COM/angular velocity, coefficient 4.0

One combined normalised error, then a log ramp:

```
u = sqrt( Σ_axes ((v_actual − v_ref)/0.25)² + Σ_axes ((ω_actual − ω_ref)/2.0)² )
vel = 4.0 · 0.1 · (1 − L(u, 0.5, 2.5)) · w_obj
```

- Normalisation is unchanged from earlier versions: linear `0.25 m/s`, angular
  `2.0 rad/s`. `u` is dimensionless and mixes both; a mismatch in either grows
  the same error.
- `e50 = 0.5` (≈0.125 m/s linear or 1 rad/s angular alone), `D = 2.5`
  (≈0.625 m/s or 5 rad/s). Maximum **0.4**, floor 0.
- `u` is recovered from the stored trace by inverting the old Gaussian when
  replaying historical rollouts.

| `u` | 0 | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| equivalent Δv (m/s) | 0 | 0.063 | 0.125 | 0.250 | 0.375 | 0.500 | 0.625 |
| equivalent Δω (rad/s) | 0 | 0.50 | 1.00 | 2.00 | 3.00 | 4.00 | 5.00 |
| `vel` (full gate) | 0.400 | 0.268 | **0.200** | 0.119 | 0.068 | 0.030 | 0.000 |

## 5. Grasp and contact terms

### 5.1 `hand_relative` — palm pose relative to the object, coefficient 0.125

Both the actual and the reference relationship are expressed **in their own
object frame**, so the term is invariant to where the object is in the world;
it measures how the object sits in the hand (grip geometry), not the trajectory.

```
rel   = R_obj⁻¹ (p_palm − p_obj)                 actual, object frame
relq  = q_obj⁻¹ ⊗ q_palm                         actual relative orientation
rrel, rrelq                                      same quantities for the reference

hand = 0.125 · exp( − ||rel − rrel||²/0.04² − (shortest_angle(relq, rrelq)/0.35)² )
```

- Position scale **4 cm**, orientation scale **0.35 rad (20°)**.
- Gaussian (exponential of a squared error): flat at the top, maximum 0.125,
  never negative, **not** gated by reference motion.
- Frames are picked per environment from the reference bank
  (`env_ref`), so multi-reference training uses each environment's own cache.

| position error | 0 cm | 1 cm | 2 cm | 4 cm | 8 cm |
|---|---:|---:|---:|---:|---:|
| with 0° orientation error | 0.1250 | 0.1174 | 0.0974 | 0.0460 | 0.0023 |
| with 20° orientation error | 0.0460 | 0.0432 | 0.0358 | 0.0169 | 0.0008 |

### 5.2 `fingers` — finger joint configuration, coefficient 0.2

```
fingers = 0.2 · exp( − mean_{22 joints}( (q_raw[t, 6:28] − q_feasible_ref[t, 6:28])² ) / 0.35² )
```

- The **22 finger joints** (dofs 6…27) only; the six floating-base dofs
  (3 translation + 3 rotation) are deliberately excluded — wrist placement is
  covered by `hand_relative` and the object terms.
- Compared against the reference **feasible** q (the demonstration clipped to
  joint limits), not the raw reference, so unreachable demonstrated angles
  cannot create an unattainable target.
- Scale `0.35 rad` on the RMS error; maximum 0.2, floor 0, never negative,
  never gated.

| RMS joint error | 0° | 5.7° | 11.5° | 20.1° | 40.1° |
|---|---:|---:|---:|---:|---:|
| `fingers` | 0.2000 | 0.1843 | 0.1443 | 0.0736 | 0.0037 |

### 5.3 `geometry` — contact anchor correspondence, coefficient 1.2

The only contact-related term. For each of the 16 hand regions it compares the
actual hand-anchor-to-object-anchor offset against the reference offset, in the
object frame, weighted by reference contact intent:

```
delta      = region hand anchor − region object anchor, in the object frame
e          = delta − delta_ref
weight     = proximity · confidence · valid            per region, from the cache
geometry   = 1.2 · Σ_regions ( weight · exp(−||e||²/0.01²) ) / max(Σ weight, 1)
```

- Anchor scale **1 cm**; coefficient 1.2 → maximum 1.2 per step when all intent
  weights are present and every anchor matches.
- `proximity = exp(−(max(gap,0)/0.01)²)`, `confidence =
  exp(−(max(depth−0.001,0)/0.003)²)·(1−fraction)`, `valid = depth ≤ 0.01` are
  reference-only penetration-quality measures computed at cache compile time.
  Regions without intent contribute zero weight and are divided out, so the
  term is a *relative* correspondence quality, not an absolute count.
- Rewards location and correspondence, **not** force magnitude, table contact,
  slip or weight support.

| anchor error | 0 mm | 2 mm | 5 mm | 10 mm | 20 mm |
|---|---:|---:|---:|---:|---:|
| `geometry` at full intent | 1.2000 | 1.1529 | 0.9346 | 0.4415 | 0.0220 |

## 6. Regularisers and failure terms

### 6.1 `action` — command magnitude penalty, coefficient 0.002

```
action = −0.002 · mean( clip(executed_action, −1, 1)² )        ∈ [−0.002, 0]
```

Encourages smaller commands; applied to the **clipped** action, i.e. the same
signal that drives the servo. Magnitude is small by design: on the reference
trace it totals −0.032 over 538 steps.

### 6.2 `survival` — per-step constant

```
survival = +0.001        every finite step
```

Totals +0.538 over a 538-step episode. It guarantees a strictly positive
per-step floor and distinguishes "alive" from terminal steps.

### 6.3 `severe` — failure penalty, coefficient 75

```
fallen    = object_bottom < table_height − 0.05 m
deviation = ||p_actual − p_ref|| > 0.10 m          (3D norm, not per axis)
finite    = all reward terms finite AND physical.valid AND contact.valid

severe = −75   if fallen or deviation or not finite
severe = 0     otherwise
```

- Applied **once per step**, not accumulated; it is the last step of the
  episode in practice because any of the conditions also terminates it.
- If the state is non-finite, `total` is set to `−75` outright (the other terms
  are then meaningless).
- Magnitude context: −75 equals ~188 control steps of the best achievable
  tracking rate (0.4/step), i.e. roughly a third of a 538-step successful
  episode.

## 7. Termination and reason codes

```
reason = 1·(index ≥ reference_length − 1)      reference complete
       | 2·deviation                           > 0.10 m from the reference object
       | 4·fallen                              bottom below table − 0.05 m
       | 8·(not finite)                        non-finite physics or contact
done   = reason ≠ 0
```

- `reason` is a **bitmask**; a step can set several bits.
- `1` is a natural completion, not a failure: the reference has been played to
  its end. `2/4/8` are failures and all carry the severe penalty.
- Terminal steps are stored with their terminal observation before reset; PPO
  does not bootstrap across a physical terminal.
- The deviation test uses the 3D norm of the object position error, while
  `object_position` applies its ramp per axis with lateral weights of 0.2 — the
  two are deliberately independent, so the reward and the terminator do not
  agree on the cost of a purely lateral error.

## 8. Deliberately absent

The reward contains **no** term for force magnitude, table/ground contact,
slip or tangential velocity, release timing or grip aperture, hand world pose,
object in-hand pose beyond `hand_relative`, energy or torque cost, joint-limit
proximity, or reference-speed-dependent bonuses. Constraint-violation
quantities (`object_all_force` including table contact, `paired_force_on_object`,
`paired_torque_com`, `tangential_slip`, `paired_count`) exist in the runtime
and are exposed through telemetry and observations, but they are not rewarded.

Contact evidence convention used everywhere (reward, telemetry, evaluation):
`max_region ||paired hand→object force|| > 0.02 N` counts as an effective
hand–object contact frame; "airborne" means the lowest object collision point is
more than 5 mm above the table; "loaded airborne" is the intersection. Contact
is evidence of touch, **not** proof of weight support.

## 9. Reference-trace calibration

All values below are the current reward applied offline to one immutable
538-step successful rollout (`cube2_02_2833`, released v4 checkpoint, natural
terminal `reason=1`, 359 loaded-contact frames, 126 loaded-airborne frames,
maximum clearance 0.168 m). No optimizer, policy or physics step was re-run.

Per-term totals:

| term | total | note |
|---|---:|---|
| `fingers` | 93.302 | 86.7 % of its 107.6 ceiling |
| `geometry` | 68.655 | 26 % of its intent-weighted 265.2 ceiling |
| `object_position` | 43.948 | 47.9 % of its motion-gated 91.8 ceiling |
| `object_velocity` | 25.765 | 42.1 % of its 61.2 ceiling |
| `hand_relative` | 22.803 | 33.9 % of its 67.2 ceiling |
| `object_rotation` | −40.284 | below zero, driven by the static override |
| `survival` | 0.538 | |
| `action` | −0.032 | |
| `severe` | 0.000 | no failure in this episode |
| **total** | **214.695** | 0.399/step |

Per phase (phase boundaries come from the reference clearance, not the policy):

| phase | controls | frames | total | per step | mean `w_obj` |
|---|---|---:|---:|---:|---:|
| approach / contact build-up | 1–204 | 204 | 84.768 | 0.4155 | 0.090 |
| lift to reference peak | 205–270 | 66 | 64.168 | 0.9722 | 0.878 |
| lower back to the table | 271–339 | 69 | 73.401 | 1.0638 | 0.982 |
| rest and hand withdrawal | 340–538 | 199 | −7.641 | −0.0384 | 0.045 |

The rest phase is negative because the released object sits at a median 90°
from the reference orientation, and 136 of those frames exceed the 45° static
override threshold.

## 10. Provenance, ABI and compatibility

`reward_parameters()` emits the JSON-safe parameter set recorded in every
training run and frozen evaluation: the gate (source, radius, endpoints,
static/moving weights), the static rotation override (condition, threshold,
weight), the coefficient map, and a `tracking_curves` block with each term's
error variable, `e50` and horizon.

- The parameter id changes whenever a coefficient or curve changes; the values
  above correspond to the id listed at the top of this document.
- The **structural** contract id `manorl.autonomy.reward.v4` and the nine-term
  order are unchanged, and the reward is deliberately **not** a model-loading
  gate: an old frozen checkpoint still loads, but its return must not be
  compared numerically with a return produced by a different parameter set.
- The cache content hash covers the reference geometry, intent weights and the
  object radius. It is recorded in provenance but not equality-gated, because
  supported build paths can produce different float bytes for identical
  physics.

## 11. Parameter table

| constant | value | meaning |
|---|---:|---|
| `REWARD_MOTION_GATE_LOW` | 0.01 m/s | reference `v_eff` at/below which the gate floor applies |
| `REWARD_MOTION_GATE_HIGH` | 0.10 m/s | reference `v_eff` at/above which the gate is 1 |
| `REWARD_MOTION_STATIC_WEIGHT` | 0.01 | static floor of the object gate |
| `REWARD_MOTION_RADIUS` | 0.04330042412758056 m | cube2 collision radius used in `v_eff` |
| `REWARD_OBJECT_POSITION_COEF` | 0.5 | object position coefficient |
| `REWARD_OBJECT_VELOCITY_COEF` | 4.0 | object velocity coefficient |
| `REWARD_HAND_RELATIVE_COEF` | 0.125 | palm–object relative pose coefficient |
| `REWARD_GEOMETRY_COEF` | 1.2 | contact anchor correspondence coefficient |
| `REWARD_SEVERE_PENALTY` | 75.0 | one-shot failure penalty |
| `REWARD_STATIC_ROT_ERROR_DEG` | 45.0° | rotation error that escapes the static gate |
| `REWARD_STATIC_ROT_WEIGHT` | 0.75 | rotation gate weight above that threshold |
| `REWARD_POSITION_HALF_M` | 0.015 m | position `e50` |
| `REWARD_POSITION_HORIZON_M` | 0.10 m | position `D` |
| `REWARD_VELOCITY_HALF` | 0.5 | velocity `e50` (normalised) |
| `REWARD_VELOCITY_HORIZON` | 2.5 | velocity `D` |
| `REWARD_ROTATION_HALF_DEG` | 35.0° | rotation `e50` |
| `REWARD_ROTATION_HORIZON_DEG` | 90.0° | rotation `D` |
| `REWARD_ROTATION_MAX` | 0.3 | rotation value at perfect alignment |
| `REWARD_ROTATION_MIN` | −0.5 | rotation floor |

Fixed values not exposed as constants: axis weights `0.2/0.2/0.8`, velocity
normalisation `0.25 m/s` and `2.0 rad/s`, hand-relative scales `0.04 m` and
`0.35 rad`, finger scale `0.35 rad`, geometry scale `0.01 m`, action
coefficient `0.002`, survival `0.001`, deviation threshold `0.10 m`, fall
threshold `0.05 m`, contact threshold `0.02 N`, airborne threshold `0.005 m`.

## 12. Tests and interactive inspection

- `tests/manorl/test_autonomy_v4.py` checks, among others: the gate endpoints
  `0.01 / 0.01 / 0.505 / 1.0`, that the gate follows the reference rather than
  the actual speed, the log ramp's `L(0)=0`, `L(e50)=0.5`, `L(≥D)=1`,
  monotonicity and non-flat top, rotation endpoints and half point, the
  static-override thresholds (30°/44.9° stay at ×0.01, 45.1°/60° escalate to
  ×0.75) and its position/velocity independence, the action-penalty boundary,
  the severe conditions and the reason bits, and multi-reference gathers.
- `tests/manorl/test_autonomy_v4_multi_reference.py` checks that each
  environment uses its own reference speed and its own action one-hot.
- `outputs/manorl/contact_conditioned_autonomy/reward-playground/` (ignored,
  local) renders this trace as an interactive page: reward weights, motion-gate
  parameters, the static override, curve-family selection, per-frame scrubbing
  with the rollout video, per-phase totals and cumulative curves. It verifies
  two in-browser anchors: `302.065135` for the historic v3 configuration and
  `214.695139` for the current one.
