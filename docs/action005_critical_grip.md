# Action005 critical-orientation grip diagnostic

Run from the assigned feature worktree with the existing pinned U1 environment:

```sh
python -m tools.search_action005_critical_grip \
  --source-root outputs/four_action_single_v1 \
  --output outputs/action005_critical_grip_v1
```

The output directory must be new. The runner reads the realized row847
`independent01` frame300 grip and accepted target, and maps its full hand/object
relative transform to A030's nearest ascending-inversion frames at bottle tilts
50,70,90,110,122.5 degrees. The manifest records exact teacher quaternions and
frames; these are not synthetic pure-axis rotations. All other scene coordinates
come from that teacher frame. State placement happens only during initialization;
the bottle remains free thereafter. This experiment does not execute acquisition,
pouring, or release and cannot accept action005.

## Search and actuation

There are exactly32 deterministic candidates: the unmodified relative grip plus
31 low-discrepancy wrist/finger geometries. Wrist displacement and rotation-vector
norms are bounded by20mm and15 degrees in the bottle frame; the first seven
nonbaseline candidates emphasize axial shifts/rocking. All22 finger target
coordinates vary independently by at most0.3rad, clipped to native limits. The
realized finger pose is retained at initialization; geometry changes establish
through native position springs. This is not a scalar squeeze sweep.

A constant wrist target includes analytical payload gravity compensation using
the wrist Jacobian evaluated at the initial bottle COM. The wrist remains
compliant, with native gains, gravity, friction, mass, and force limits unchanged.
There is no wrist PID or object feedback. Each absolute target remains unchanged
across four480Hz substeps. U1's full setting hash must equal `c6db552f...`; Session
also checks native parameters, CCD16 and1024/256/4096 capacities.

Each geometry is tried at every orientation for up to2s. A trial stops at >20mm
or >20deg hand-relative bottle drift, native nonhand support, or12 consecutive
force-free control frames. Structural/runtime errors stop the entire search and
write `ERROR.json`; no fallback or reroll is used.

## Measurement and decision boundaries

**Telemetry is120Hz, at the last native480Hz substep of each control interval.**
This explicitly does not measure intervening force peaks, brief loss of support,
or actuator saturation. Capacity guards run at all480Hz substeps. Native contact
positions, bases, distances, friction, dimensions, EFC/world IDs and local wrenches
are retained. Contact forces and native COM refer to the solver's pre-integration
kinematic epoch; relative poses and rates use post-integration state. Rates are
one-control-interval finite differences. Wrist sag is recorded separately from
hand-relative translation/rotation.

Per-link wrenches, net force including gravity, net torque about native bottle COM,
actuator force/saturation, support, penetration, applied-control error, axial
contact span and thumb/nonthumb opposition are retained. Five-bearing means each
finger's summed native contact normal force exceeds0.2N; palm force cannot replace
a missing finger. Pyramid utilisation uses the L1 sum of friction-normalized
active tangential/torsional/rolling components, divided by normal force.

Wrench reserve is an instantaneous linear program over native pyramidal contact
generators, with each finger contact's **measured normal force as its ceiling**.
It maximizes the gravity-load multiplier subject to zero net force/torque. Reserve
is that multiplier minus1. It measures conservative static reserve with current
normal loads, not extra squeeze available from actuators. Net torque alone is a
realized imbalance, not a proof of the maximum feasible torque.

A screen pass requires a full2s run at all five orientations; during the final1s,
five-bearing occupancy is at least95%, drift stays below3mm/3deg, every sampled
bearing contact has positive friction reserve and every sample has positive
wrench reserve. Terminal rates must be below1mm/s and1deg/s. No sampled actuator
saturation or support is allowed anywhere. There is no relaxed short-hold pass.
A passing screen remains a **sampled static diagnostic only**, never an accepted
005 trajectory or evidence of continuous480Hz threshold satisfaction.

`ranking.json` is updated after each candidate, ordered by all-orientation pass,
worst final-second five-bearing fraction, worst whole-trial establishment
fraction, worst survival duration, then worst terminal translation drift.
Per-orientation results identify failures and terminal net torque/reserve.
Every attempted orientation retains `target.npy`, `trace.npz`, streamed
`telemetry.jsonl.gz` and `result.json`. Generated files remain ignored.

If no candidate establishes and retains five bearing fingers, use the ranked
force/axial/opposition evidence to synthesize a contact-space grasp with axial
separation and thumb opposition; do not scale the same preload again. If a
candidate passes, the next intervention is to fit that relative grasp into the
full A030 acquisition/inversion/return/release timeline and test fresh U1 replays.
