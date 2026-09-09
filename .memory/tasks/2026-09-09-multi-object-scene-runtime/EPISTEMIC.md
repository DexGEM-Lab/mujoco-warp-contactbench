# Current model

## Mechanism and intervention

One manipulation target does not imply one physical object. The previous
adapter discarded cuboid1, a near-static support below bowl. Per-object ground
alignment also moved the bowl off its captured support height.

The corrected decoder preserves all initial object poses at the selected
padded frame. The lowest collision surface determines one shared vertical
translation, also applied to the hand references. Non-target bodies then evolve
as unactuated free bodies under gravity/contact rather than being overwritten
by recorded poses. This is sufficient for physical initialization; full passive
pose time series are not needed by the runtime.

Composite worlds use the existing unified topology and place all present
bodies. Only absent types are parked. Object-object collisions are enabled for
these worlds only; single-object defaults remain unchanged. Producer routing
continues to select one target body for observations and rewards.

## Evidence and justified boundary

The corrected bowl:03 row-0 initial state has four bowl–cuboid1 contacts, with
approximately 2.06 mm initial penetration inherited from capture/collision-mesh
alignment. Its shared grounding translation is +0.321 mm. DISPLAY=:1 GPU replay
completed the source horizon and reset. A screenshot of the actual GLFW window
shows bowl above cuboid1. All 59 rows retain both physical initial states.
See OPS.md for validation provenance.

Trajectory package v2 carries these initial states and makes old readers fail
rather than discard supports. New readers retain v1 compatibility. Composite
scenes reject device-transition/contact-decode optimizations because their
current reducers include non-target contacts. Baseline host-decoded physics is
supported; large-batch training performance and grasp convergence have not been
measured. Completion of a no-policy reference replay is not evidence of a
successful learned grasp.
