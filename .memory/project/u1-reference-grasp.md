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

## Final pitcher release v13

The earlier row82 target was not robust: two later fresh replays toppled after
release. Final v13 preserves the physical grasp/pour/return prefix through1130,
then uses supported yaw registration and a strict release sequence. The thumb is
moved radially clear first; with the thumb held open, the four fingers and wrist
reverse the actual same-U1 acquisition deltas so the fingers exit along the
handle-entry corridor. This satisfies the user-required topology without forcing
the object.

Two fresh frame0 replays pass. Thumb last-contact frame1569 precedes nonthumb
last contacts1726–1743; max release tilt is6.15/5.87deg, final tilt0.45deg, full
orientation8.39/10.03deg, final position0.71/0.69cm, and the pitcher ends clear
and settled. Canonical manifest: `recipes/u1_pitcher_row82_release_v13.json`.
Target SHA: `f16c7c23e6eff801b877603ea13d0e3179ac6597e81c1c82da2b516b6562a93a`.

## Exact800 campaign

The final five parents are bundled in signed registry v3 under
`outputs/u1_5x160_registry_v3`; the current production registry is copied to the
hidden NAS campaign staging. Plan:5 radii x8 XYZ octants x4 nonzero roll/pitch
orientations=160 slots/action,16 deterministic candidate directions/slot. Every
selected child must pass a native U1 first replay and a separate-process replay
of the same frozen actual controls. Every480Hz substep checks contact/constraint
capacity. Native contact capture includes frame basis matrices, so compact force
export never substitutes CPU normals.

Old augmentation speed came from32-world batches (1344 runs in about14minutes)
and formal8-world batches. The exact campaign initially used one scalar process;
it was accelerated to five concurrent whole-action owners. Local owns003/009;
Server1 GPUs0/2/3 own005/006/007 through an exact-SHA source deployment and a
process-private NAS path alias. Ledger identity remains the canonical `/home`
path. The cross-host wrapper accepts only the exact signed registry/plan SHA;
it bypasses only 1e-18 cross-CPU re-normalization differences, never candidate,
UUID, physics or validation checks.
