# Epistemic model

## Phenomenon
The source of truth for target-DOF replay is now one explicit synthetic Lance row. NPZ was an operational workaround for Server2, but the requested product contract is direct `dataset/version/row` decoding on a healthy Lance host.

## Implemented mechanism
The replay CLI projects only `index`, `trajectory_metadata`, `timestamp`, `hands`, `objects`, and `provenance`, validates the corrected v2.2 contract, resolves the canonical right slot, and extracts recorded 28D qpos, post-controller 28D target qpos, object pose, timing, CCD settings, checkpoint identity, generated-row identity, and original source lineage. It then uses the unchanged MJX-Warp target adapter: frame-0 reset state, target `t` written directly to `data.ctrl`, and exactly two physics substeps per 5 ms control transition. GUI rendering mirrors this same state into the native viewer without native physics.

## Evidence
Nine direct-row/fake-Lance tests pass, including the two-slot hand contract, projected columns, lineage separation, timing/shape checks, malformed CCD rejection, CPU CCD boundary, and finite CLI numeric validation. The existing viewer regression remains green. A real one-row Lance fixture projected from merged row 2239 decoded locally and completed all 519 GPU transitions with max object-position error `3.8365e-3 m`, max rotation error `0.021403 rad`, and replay max lift `0.211850177 m`. On local physical X11 display `:1` with one DP-2 monitor, the MuJoCo window was observed alive and looping; the process was then stopped cleanly.

## Boundaries
The current Server2 native Lance/PyArrow path remains nondeterministically unsafe and is no longer a supported replay host for the direct-Lance tool. Direct replay is for healthy Lance hosts. The row must be a complete reset-origin corrected v2.2 synthetic episode; replay initializes velocity to zero. Historical external NPZ artifacts remain preserved as audit evidence but current code neither creates nor reads them.
