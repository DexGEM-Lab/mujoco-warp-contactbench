# U1 reference-driven grasp repair

Canonical recipe and evidence: `docs/u1_raw_workspace.md`, section
“Reproducible row82 closed-grip candidate”; `recipes/u1_pitcher_row82.json`.

- U1 needs CCD16 as well as pyramidal/impratio1 and120/480Hz. Early CCD35 repair
  trials/checkpoints are diagnostic history, not fully matched-contract evidence.
- A position-actuated wrist supporting1.28kg at100N/m needs about125.6mm static
  target deflection. Small target-error caps can exclude necessary load support.
  Encode compensation in saved targets; never change native gains or introduce a
  hidden replay controller. Partial/table support must not be counted as grasp.
- Frame400 acquired finger targets plus measured thumb/index closing preload let
  row82 follow its wrist-reference pouring motion. Continuing capture finger
  motion changed the grip. This latch is specific to this pitcher; moving finger
  references were necessary for the earlier egg028 repair.
- FK fingertip proximity is not surface contact. Surface contact outside an
  ungrasped handle also does not prove grasp. The accepted local candidate has
  actual native tip-pair force while freely carrying the object.
- Intentional pouring rotation is not instability. Track unwanted object motion
  in the hand frame; wrist tracking error alone is not an acceptance criterion.
- A modified physical grasp can offset the spout and landing point even while
  following the reference wrist. Register those task-space positions smoothly;
  evaluate bowl alignment and placement, not merely lift/tilt/contact counts.
- Row82 has two successful fresh local frozen-target replays and exact recipe
  reconstruction. Other parents, augmentation, H200 parity and Lance publication
  remain separate work; existing1923-row data are unchanged.

Interactive repair visibility: keep the live GUI synchronized with the current
object/target and demonstrate edits from a complete checkpoint. User explicitly
rejected leaving the visible simulator stale while conducting headless trials.
Headless native verification is separate, not a substitute for that interaction.
Check PID liveness and state freshness before calling a workspace live.
