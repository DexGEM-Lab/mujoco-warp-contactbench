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
future tables, static collision cloud, action ID, support shift and normalized
object geometry. Identity is never embedded in policy input.

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
its servo command. It takes four `mjx.step` calls and then exactly one
`mjx.forward` before extracting q, origins, COMs, velocities and contacts.
The object COM/origin and anchor-point formulas use `xipos`, `subtree_com` and
`cvel`; anchor velocity includes the required co-rotating `-omega_O×delta` term.

Reward evaluates `s(t+1)` against `ref(t+1)`: axis position total 1.2,
shortest-angle piecewise orientation, world COM/angular velocity, hand-relative
pose, feasible fingers, reference-weighted anchor error, clipped-executed
action penalty, survival `0.001`, and severe `-25`. It has no force magnitude
reward. Reasons are complete=1, deviation=2, fallen=4, nonfinite=8.

## Validation boundary

`tests/manorl/test_autonomy_v4.py` independently checks slices/dimensions,
xyzw double cover and rotation 6D, exact-reference anchors, analytic rotating
anchor velocity, contact order/sign/table exclusion/all-object force/torque,
reward boundaries, and checkpoint rejection. The N=1 CPU smoke loads
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
and ABI/clock are the compatibility contract. There is no observation
normalizer in this minimal route (`normalizer: null`).

Real GPU train commands default to W&B enabled and expose the B4096 controls:
`--num-envs`, `--persistentworkspace`, `--ccd-contacts-per-world 121`; runtime
keeps `njmax=512` per world. CPU integration commands explicitly use
`--no-wandb --no-persistentworkspace`.

```bash
# Real training launch candidate; do not start it without the launch decision.
python tools/train_manorl_autonomy.py train --device gpu --num-envs 4096 \
  --persistentworkspace --ccd-contacts-per-world 121 --checkpoint outputs/manorl/contact_conditioned_autonomy/cube2_02_v4_ppo.pt

# Authorized bounded local validation.
python tools/train_manorl_autonomy.py train --device cpu --num-envs 1 \
  --no-persistentworkspace --updates 2 --rollouts 8 --learning-epochs 1 \
  --mini-batches 1 --no-wandb --checkpoint /tmp/cube2-v4-n1.pt
python tools/train_manorl_autonomy.py evaluate --device cpu --num-envs 1 \
  --no-persistentworkspace --checkpoint /tmp/cube2-v4-n1.pt --steps 4
```

Frozen evaluation always resets at reference frame 0, follows deterministic
clipped means, and reports the natural first termination. Its optional
`--full-horizon-diagnostic` continues after that boundary only while preserving
`natural_first_termination` and an explicit diagnostic-boundary label.

The fast contact reducer now asserts the static pyramidal cone, requires
ownership-disjoint geoms in real models, rejects unsupported cones, and skips
live contacts with inactive `efc_address=-1` without allowing masked NaN
positions into a lever arm. General cone/helper parity and real slip diagnostics
are still separate work.
