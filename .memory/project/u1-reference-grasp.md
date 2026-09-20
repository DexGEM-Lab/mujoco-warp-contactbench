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

## Bowl grasp transfer

Do not prepend a successful grasp trajectory to another task. That inserts an
extra pickup/pause and changes task semantics even when physical gates pass. Keep
the original frame count and wrist reference; edit only the acquisition window,
then rejoin the original wrist target byte-for-byte.

The transferable row607 mechanism is contact topology, not old-hand joint angles:
four non-thumb fingers seat on the rim before the thumb pinch. Cheyingtong needs
an action-specific local wrist correction because003/007/009 approach the bowl
from different poses. Reusing one world-space wrist delta across actions fails.

Locally validated single trajectories under the same U1 setting hash
`c6db552f...`:

-003: wrist frames86–192 plus thumb frames86–318; all other finger targets remain
 byte-identical. Independent lift10.79cm, final world support/release,2.53cm error.
-007: action-specific wrist (total Z−1cm), successful003 four-finger target then
 thumb; wrist exact after201. Independent lift11.43cm, cuboid1 support/release,
4.95cm error,0.25deg tilt.
-009: action-specific wrist (total Z−1cm), same ordered finger closure, wrist exact
after328. Contact dynamics made +1deg orientation probes switch to a worse basin;
all −1deg probes passed. The chosen local wrist-Z −1deg correction gives29.25cm
lift, final four-ray unsupported hold and14.34deg full orientation error.

Evidence and exact target hashes live in ignored local artifact
`outputs/four_action_single_v1/correct_bowl_single_candidates.json`. Existing
Lances remain unchanged; these are local single-trajectory results, not H200 or
augmentation robustness claims.
