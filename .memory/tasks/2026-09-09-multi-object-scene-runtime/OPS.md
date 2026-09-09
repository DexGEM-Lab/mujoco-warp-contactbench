# Operations and evidence

## 2026-09-09T21:35:00+08:00 — user-visible anomaly

Observation: DISPLAY=:1 policy-free replay of `bowl,cuboid1` showed only bowl. The prior adapter used the passive name only to select the active object's `objects` slot; `ReferenceTrajectory` retained one object pose and the environment compiled a one-object model.

Prediction: cuboid1 is physically relevant support rather than incidental background. Across 39 bowl/cuboid1 rows, cuboid1 moves only 0.7–4.8 mm over the capture. Typical initial centers are bowl z≈0.05 m and cuboid1 z≈0.015 m with 3.6–3.7 cm separation, consistent with bowl resting on cuboid1. Mayonnaise/pitcher passive bowls are 0.48–0.70 m away.

Existing unified models compile several real bodies but park all except one per world. Object collision geoms use contype=2/conaffinity=5, which disables object-object collision. Correct physical replay therefore requires multi-body placement, a shared scene ground shift, and object-object collision; visual-only rendering cannot validate the task.

## 2026-09-09T22:01:32+08:00 — physical replay and compatibility evidence

The implementation stores scene initial poses, not full passive reference
sequences: after reset, passive bodies evolve freely under physics. It keeps
single-object defaults and enables object-object masks only for composite
models. Composite package schema v2 retains scene states; v1 remains readable.

Native initialization of bowl:03 row 0 produced four bowl–cuboid1 contacts with
penetration about 2.058 mm. All objects and the hand received the same +0.321 mm
Z translation. Across 95 balanced slots, all 59 distinct source rows decoded
and retained both scene states; every active initial pose matched its reference.

The first broker-launched GPU viewer completed a horizon, but disconnecting the
managed backend also ended its child process. Relaunched in persistent tmux
`manorl-composite-replay:two-objects`. The GLFW window on DISPLAY=:1 shows both
bowl and cuboid1 and the log records repeated reason-1 horizon completions.
Actual window screenshot: `outputs/replays/two_objects_display1.png`.
These outputs remain local and untracked.

Focused validation across trajectory, package, viewer, unified-model, and asset
tests: 96 passed, 7 data-dependent skips, 1 MJX test deselected. Tests cover
second-slot target selection, shared hand/object translation, invalid passive
poses, package round-trip, real object-object contact, present-body placement,
and parking only absent bodies. GPU evidence comes from the actual user viewer.
