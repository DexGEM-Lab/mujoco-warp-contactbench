# Current model

## Phenomenon
Real captures are kinematically coherent but can fail under free-body MJX-Warp because recorded MANO geometry, time grids, contact state, and servo dynamics differ from the fixed Cheyingtong model. The policy should learn only the correction around the raw right-hand motion.

## Supported mechanisms
- Physical hand identity is now explicit and independent of every source row: all decoded references bind the same Cheyingtong/f98 asset manifest. Source operator and ten source MANO betas remain per-row provenance only.
- Remake is physically 100 Hz and needs elapsed-timestamp interpolation. A real 12.621 s row became 12.625 s at 120 Hz, retaining its final pose with a 4 ms terminal hold.
- Guangguan is physically 120 Hz by elapsed duration despite median adjacent intervals near 8 ms. A real 4.809 s row became 4.816667 s, retaining its final pose with one 7.667 ms edge hold.
- Full-capture selection preserves approach, annotated movement, release, and withdrawal; movement annotations only define reward/evaluation phases.
- Compound bottle+cap rows are representable without relabeling: bottle is the explicit active target and cap remains a second free scene body with object-object collisions.

## Dataset accounting
- Remake v978: 10,584 rows, all right-hand, valid increasing timestamps, 103 pairs.
- Guangguan v530: 5,164 rows. Exactly 35 lack right hand and 9 have non-increasing timestamps. The 55 compound bottle/cap rows now compile under explicit `bottle:18` targeting.
- Expected combined accepted count remains 15,704 rows across 139 pairs; the full compile has not yet been run.

## Ruled out
- Selecting a physical MANO model from `index.operator`.
- Requiring raw betas to equal Cheyingtong betas.
- Reusing IsaacGym `q_state_ref` semantics; raw capture has only one kinematic q reference.
- Treating Remake frames as already120Hz; that speeds motion by20%.
- Interpreting Guangguan's median dt as a125Hz episode clock.
- Dropping the cap or forcing object poses after reset.

## Anomalies
- Nine Guangguan rows have non-increasing timestamps and must appear as explicit compiler rejections.
- Thirty-five Guangguan rows lack a right hand and must appear as explicit compiler rejections.
- A broad reward profile still needs implementation: all-five-fingertip contact intent conflicts with labeled pinch/lateral/adduction grasps, and the current default residual penalty is zero.
- The trainer currently accepts one package; simultaneous two-source training needs verified multi-package catalog composition.

## Current justified claim
The corrected source→fixed-Cheyingtong→120Hz→free-object package path works for real 100Hz, real nominal120Hz, and compound-scene examples. The next intervention is the small-residual training profile plus multi-package assignment, followed by zero-residual failure-mode measurement.

## Highest-value next question
With corrected references and one fixed hand, does zero-residual failure arise mainly at contact acquisition, load-bearing transport, or release? That measurement determines whether the conservative 1× residual envelope is sufficient.
