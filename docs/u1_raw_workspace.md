# Raw-row native U1 workspace

`tools/view_u1_repair.py` accepts either historical `--bundle/--target` or raw
`--dataset/--version/--row/--asset-manifest`, with explicit `--asset-root` and
`--output`. Raw mode validates the pinned Cheyingtong profile, selects the
recorded right-hand slot, and adds only the common scene grounding shift.
It never writes Lance or asset files.

```bash
DISPLAY=:1 PYTHONPATH=. python tools/view_u1_repair.py \
  --dataset /path/to/capture.lance --version 4 --row 82 \
  --asset-root /path/to/pinned/assets --asset-manifest /path/to/asset_manifest.json \
  --checkpoint 230 --output outputs/u1_live_row82
```

Run in a persistent ordinary tmux shell. The output directory must not exist.
The workspace physically advances from frame0, validates checkpoint restore,
then saves and pauses at the checkpoint. `state.json` exposes the live cursor,
contacts, source contacts, live/source hand-object transforms, and edit history;
`source.json` records dataset/version/positional row/UUID, asset identity and
exact grounding shift. Teacher states are used only for display after frame0.
Contacts are CPU geometric witnesses at live/source poses, not GPU force measurements.

Frame0 is **prestep**. Target0 initializes control without advancing physics;
target[t] produces state[t] from state[t-1] through four 480Hz substeps. Teacher
frame[t] is compared with live state[t]. The legacy `replay_capture_no_policy`
trace instead records its frame0 after one interval; its trace is not an exact
state-prefix comparator for this workspace.

Save/Restore includes every native device array, cursor, target and edits.
Disk checkpoints contain `state.npz` and `workspace.json`; Restore uses the
latest in-memory saved checkpoint in this process. `restore_validation.json`
reports exact buffer restoration and two 12-frame continuations, with absolute
qpos tolerance 1e-6 and qvel tolerance 1e-4. A failed bound is surfaced, not hidden.

Use Play/pause, Step1/Step12, Save, Restore, Reset/clear, Export or the JSON
command inbox (`commands/*.json`). Example commands:

```json
{"action":"step","frames":1}
{"action":"restore"}
{"action":"offset","joint":1,"value":0.001,"start":230,"end":270,"ramp":12}
```

Offsets alter future absolute targets with smooth zero-ended windows and joint
limit checks; they never force object or hand states. Restore before comparing
branches. For row82, inspect frames230–270 and adjust only bounded wrist/finger
approach offsets: four non-thumb fingers through the handle, thumb outside.
Exported candidates carry source identity and are explicitly unaccepted until
full frame0 U1 validation and independent replay. Workspace readiness does not
establish repaired grasp or lift.
