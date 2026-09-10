# v3 late-placement release repairs

This bundle treats `patches/full_containers_v2/` as immutable input. Each of the six replay candidates is byte-identical to its v2 command track through the `prefix_end_exclusive` step recorded in its patch, then uses measured pass-2 wrist/object geometry to finish placement, remove wrist load, open, and retreat. It contains no simulation stepping, force, object pose, friction, gain, training, or NAS changes.

| Row | Diagnosis from physical trace | v3 intervention |
|---:|---|---|
| 26 | Upright world-supported bottle at frames 618–644 toppled under continued late wrist load. | De-load at 610, open after placement, radial source-like retreat. |
| 29 | Bottle returned at 29° tilt and never established upright support. | Carry its measured grasp frame to the source goal while leveling before release. |
| 30 | Upright world-supported bottle at frames 633–683 toppled under late wrist load/rotation. | De-load at 625, open after placement, radial source-like retreat. |
| 35 | Pitcher stayed hand-supported at a tilted placement and fell after delayed release. | Level its measured grasp frame into the goal, thumb-first release, handle-side retreat. |
| 37 | Pitcher was upright/supported/released until the late retreat reintroduced palm contact. | De-load at 1040, thumb-first release, handle-local-Y retreat without attitude change. |
| 38 | Pitcher contacted the bowl during late tilted descent and toppled before release. | Preserve prefix through 429; add 7cm clearance, level, thumb-first release, handle-side retreat. |
| 33 | Physical outcome is upright, released, supported, near goal; only `tilt_over_bowl` failed. | `row33_tilt_over_bowl_metric_review.json`: center was 5.6cm from bowl while 10.2cm above it and tilted 54.1°. No fake replay candidate. |

`receiver_inventory.json` contains patch paths, UUIDs, command hashes, and prefix boundaries. `RUNNERS.md` lists exactly the six replay commands. CPU `mj_forward` recreated every intervention anchor with the recorded scene order `['bowl', active]`, active qpos address 35, finite state, and zero active-body position reconstruction error; it does not simulate a candidate.
