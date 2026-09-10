# Full container receiver templates

This directory prepares 18 UUID-bound receiver repairs from immutable row-19 bottle and row-31 pitcher donors. It contains no measured replay result. Parent GPU execution uses `RUNNERS.md`.

- **Bottles 20–30:** retain the receiver object trajectory and source prefix, retime the donor acquisition/level/release anchors by receiver movement duration, and choose the object-frame grasp orientation whose world-side best matches the observed receiver approach.
- **Pitchers 32, 33, 34, 35, 37, 38, 39:** retain a 40-command receiver prefix, then add the donor frozen-command minus donor-reference residual to a monotonic phase-resampled receiver reference. The held tail retains the donor thumb-clear then sideways release residual against the receiver endpoint.

## Supported failure modes

1. The pitcher donor required 10–14 cm load compensation under its 1.28 kg gravity load; receiver geometry/timing can change the needed residual enough to lose or sag the handle grasp.
2. Bottle receivers differ in initial yaw. The recorded approach-side comparison selects an orientation mapping, but an asymmetric bottle can still collide or miss contact under physics.
3. Phase resampling preserves receiver source/object references, not contact dynamics; GPU replay is the required discriminator for lift, tilt, placement, and release.
