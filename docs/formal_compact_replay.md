# Formal multi-object compact control replay

`tools/replay_formal_compact.py` tests the28-row formal Cheyingtong no-augmentation
bundle by executing saved actuator targets, without invoking a policy. The input
is `synthetic_mano_target_replay_visual_v2_contact`,120Hz control/480Hz physics.
It requires the matching asset manifest and the bundle's hashed row metadata and
recording audits to establish the recorded control and initialization contract.

## Why a separate entry point

The formal compact rows preserve all scene objects and include names containing
underscores (e.g. `egg_ellipsoid`). The older single-object target replay parser
assumes one object and outgoing-frame target indexing. These records instead save
arrival-frame targets: `urdf_dof_target[t]` generated state `t` with four identical
physics-substep targets. Replay therefore initializes from state0 and executes
targets1..N-1. The source audit verifies that equality explicitly for each row.
No source state is imposed after reset.

The physical hand/profile and source environment's CCD settings are pinned. The
runner reuses the unified MJX-Warp environment with all named bodies and original
native gains, friction, mass and solver defaults. It does not add the historical
local-repair wrist PID or elliptic-friction recipe to an unrelated RL export.
Initial velocities are zero, independently verified against the recording audit.

## Execute

```bash
PYTHONPATH=. python -m tools.replay_formal_compact \
  --dataset /path/to/formal_bundle/compact.lance \
  --asset-root /path/to/dexstream_digital_assets \
  --output outputs/formal_target_replay --prepare

PYTHONPATH=. python -m tools.replay_formal_compact \
  --dataset /path/to/formal_bundle/compact.lance \
  --asset-root /path/to/dexstream_digital_assets \
  --output outputs/formal_target_replay --group 03-bowl
```

Other groups are `01-egg`, `02-stir`, `04-pour`. Each group runs in a fresh process
with independent worlds, and existing outputs are never overwritten. All accepted
source rows are tested once; no tuning or success hunting. GPU work should run in
an inspectable tmux session. The group summary contains every row, including failed
outcomes. Source bundle/data are read-only. No training or archive packaging occurs.

## Interpret results

Two questions are measured separately:

1. **Task quality:** non-egg rows use the formal source gate—complete the recorded
   horizon without crossing the original0.1m object/reference deviation boundary
   after the early30steps, wrapped intrinsicXYZ mean final error<=35deg, and>100
   native hand/target contact frames above0.2N. The reconstructed monitored target
   index comes from the original recording audit (state k uses reference k-1 in
   this bundle). A forced full-horizon rollout alone is not marked complete.
2. **Recorded-motion parity:** actual new hand/object motion is compared with the
   saved compact states; final/max/p95 position and rotational differences are
   reported independently from acceptance. Small contact-dependent variation is
   possible even with unchanged controls.

Egg rows use the original semantic exception: a released egg crosses the actual
bin rim downward, inside0.8 of the rim's bounding ellipse and above its bottom.
They do not require ordinary trajectory-terminal code1, matching postrelease
orientation, or extra settling. Replay does not append task time.

New native contacts are solved by MJX-Warp and recorded at the last480Hz forward
boundary per120Hz state. Their derived object poses match the source's contact
boundary; qpos is postintegration, which can differ by one physics substep. Both
qpos-derived and solver-derived object arrays are saved so visualization and
source parity each use the appropriate fields. No force reconstruction is claimed
as native telemetry. Container outcomes do not measure liquid delivery.
